"""Orchestrates mesh and texture generation pipelines."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import trimesh
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.services.imagen_client import ImagePayload
from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FloaterRemover,
    DegenerateFaceRemover,
    FaceReducer,
)
from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors
from hy3dgen.texgen import Hunyuan3DPaintPipeline


logger = logging.getLogger(__name__)


@dataclass
class ShapeGenerationSettings:
    num_inference_steps: int = 4
    guidance_scale: float = 3.0
    box_v: float = 0.9
    octree_resolution: int = 192
    num_chunks: int = 10000
    mc_algo: Optional[str] = None
    seed: Optional[int] = None


@dataclass
class TextureGenerationSettings:
    enabled: bool = True
    face_count: int = 20000
    delight_steps: int = 20
    multiview_steps: int = 12
    reuse_delighting: bool = False
    delight_cache_size: int = 8
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        self.delight_steps = max(1, int(self.delight_steps))
        self.multiview_steps = max(1, int(self.multiview_steps))
        self.delight_cache_size = max(0, int(self.delight_cache_size))
        if isinstance(self.reuse_delighting, str):
            self.reuse_delighting = self.reuse_delighting.lower() in {"1", "true", "yes", "on"}
        if self.seed is not None and self.seed != "":
            self.seed = int(self.seed)
        else:
            self.seed = None
class GenerationService:
    """Wraps pipelines while reusing decoded imagery."""

    @staticmethod
    def _tag_mesh_metadata(mesh: trimesh.Trimesh, *, face_target: Optional[int] = None) -> trimesh.Trimesh:
        metadata = getattr(mesh, 'metadata', None)
        if metadata is None:
            mesh.metadata = {}
        hunyuan_meta = mesh.metadata.setdefault('hunyuan3d', {})
        hunyuan_meta['standardized'] = True
        if face_target is not None:
            hunyuan_meta['face_reduced_to'] = face_target
        return mesh

    def __init__(
        self,
        *,
        model_path: str = 'tencent/Hunyuan3D-2mini',
        tex_model_path: Optional[str] = 'tencent/Hunyuan3D-2',
        device: str = 'cuda',
        enable_texture: bool = True,
        shape_model_subfolder: str = 'hunyuan3d-dit-v2-mini-turbo',
        tex_model_subfolder: str = 'hunyuan3d-paint-v2-0-turbo',
        output_dir: Optional[str] = None,
    ) -> None:
        self.device = device
        self.rembg = BackgroundRemover()

        try:
            torch.set_float32_matmul_precision("high")
            if torch.backends.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:  # pragma: no cover - defensive; these flags are best-effort
            logger.debug('Torch backend optimisations could not be applied.', exc_info=True)

        sage_pref = os.environ.get('USE_SAGEATTN')
        if sage_pref != '0':
            if sage_pref is None:
                os.environ['USE_SAGEATTN'] = '1'
            try:  # pragmatically enable SageAttention if the package is available
                import sageattention  # type: ignore # noqa: F401
            except ImportError:
                logger.warning('Sage attention requested but package "sageattention" is not installed; disabling USE_SAGEATTN.')
                os.environ['USE_SAGEATTN'] = '0'

        default_output = Path.cwd() / 'generated_meshes'
        configured_output = Path(output_dir) if output_dir else Path(
            os.environ.get('HY3DGEN_OUTPUT_DIR', default_output)
        )
        self.output_root = configured_output.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            device=device,
            enable_flashvdm=True,
            use_safetensors=False,
            subfolder=shape_model_subfolder,
        )
        try:
            self.shape_pipeline.enable_flashvdm(topk_mode='merge')
        except Exception:  # pragma: no cover - fallback to defaults if enhanced mode unavailable
            logger.debug('Failed to set flashVDM topk_mode="merge"; continuing with pipeline defaults.', exc_info=True)
        # Configure default surface extractor once to avoid repeated deprecation warnings during inference.
        try:
            self.shape_pipeline.vae.surface_extractor = SurfaceExtractors['mc']()
        except Exception:  # pragma: no cover - fallback to diffusers default if configuration fails
            logger.warning('Unable to configure default surface extractor; continuing with pipeline defaults.', exc_info=True)

        self.float_remover = FloaterRemover()
        self.degenerate_face_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()

        self.texture_enabled = enable_texture and tex_model_path is not None
        self.texture_pipeline: Optional[Hunyuan3DPaintPipeline] = None
        if self.texture_enabled:
            try:
                self.texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                    tex_model_path,
                    subfolder=tex_model_subfolder,
                )
            except Exception as exc:
                logger.warning(
                    "Texture pipeline unavailable (%s); continuing with texture generation disabled.",
                    exc,
                )
                self.texture_enabled = False

    def prepare_image(self, payload: ImagePayload) -> Image.Image:
        image = payload.as_pil_rgba()
        image = self.rembg(image)
        return image

    def generate_mesh(
        self,
        image: Image.Image,
        settings: ShapeGenerationSettings,
    ) -> trimesh.Trimesh:
        generator = None
        if settings.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(settings.seed)

        meshes = self.shape_pipeline(
            image=image,
            num_inference_steps=settings.num_inference_steps,
            guidance_scale=settings.guidance_scale,
            box_v=settings.box_v,
            octree_resolution=settings.octree_resolution,
            num_chunks=settings.num_chunks,
            mc_algo=settings.mc_algo,
            generator=generator,
            output_type='trimesh',
        )
        return meshes[0]

    def standardize_mesh(
        self,
        mesh: trimesh.Trimesh,
        *,
        face_count: Optional[int] = None,
    ) -> trimesh.Trimesh:
        face_target = face_count or TextureGenerationSettings().face_count
        mesh = self.float_remover(mesh)
        mesh = self.degenerate_face_remover(mesh)
        if face_target:
            mesh = self.face_reducer(mesh, max_facenum=face_target)
        return self._tag_mesh_metadata(mesh, face_target=face_target)

    def generate_textured_mesh(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        settings: TextureGenerationSettings,
    ) -> trimesh.Trimesh:
        if not self.texture_enabled or not settings.enabled:
            raise RuntimeError("Texture generation is disabled or unavailable (check custom_rasterizer installation)")
        metadata = getattr(mesh, 'metadata', {}) or {}
        hunyuan_meta = metadata.get('hunyuan3d') if isinstance(metadata, dict) else None
        previously_reduced = None
        if isinstance(hunyuan_meta, dict):
            previously_reduced = hunyuan_meta.get('face_reduced_to')

        stage_start = time.perf_counter()
        if not (isinstance(hunyuan_meta, dict) and hunyuan_meta.get('standardized')):
            mesh = self.float_remover(mesh)
            mesh = self.degenerate_face_remover(mesh)
            mesh = self._tag_mesh_metadata(mesh)
            logger.debug('Texture preprocessing sanitized mesh in %.3fs', time.perf_counter() - stage_start)

        face_target = settings.face_count
        if face_target and (previously_reduced is None or previously_reduced > face_target):
            mesh = self.face_reducer(mesh, max_facenum=face_target)
            mesh = self._tag_mesh_metadata(mesh, face_target=face_target)
        else:
            mesh = self._tag_mesh_metadata(mesh, face_target=previously_reduced)
        if self.texture_pipeline is None:
            raise RuntimeError("Texture pipeline is not initialized")
        runtime_seed = settings.seed if settings.seed is not None else None
        stage_start = time.perf_counter()
        textured_mesh = self.texture_pipeline(
            mesh,
            image,
            delight_steps=settings.delight_steps,
            multiview_steps=settings.multiview_steps,
            reuse_delighting=settings.reuse_delighting,
            delight_cache_size=settings.delight_cache_size,
            seed=runtime_seed,
        )
        logger.debug('Texture pipeline completed in %.3fs', time.perf_counter() - stage_start)
        return textured_mesh

    def export_mesh(
        self,
        mesh: trimesh.Trimesh,
        job_id: str,
        *,
        textured: bool = False,
        file_type: str = 'glb',
    ) -> str:
        sanitized = re.sub(r"[^0-9a-zA-Z_.-]", "_", job_id) or "job"
        job_dir = self.output_root / sanitized
        job_dir.mkdir(parents=True, exist_ok=True)

        filename = 'textured_mesh' if textured else 'mesh'
        path = job_dir / f"{filename}.{file_type}"

        export_kwargs = {}
        if file_type.lower() in {'glb', 'gltf'}:
            export_kwargs['include_normals'] = textured

        mesh.export(str(path), file_type=file_type, **export_kwargs)
        logger.info('Saved %s mesh to %s', 'textured' if textured else 'generated', path)
        return str(path)
