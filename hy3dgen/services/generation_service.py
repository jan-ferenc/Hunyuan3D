"""Orchestrates mesh and texture generation pipelines."""

from __future__ import annotations

import logging
import concurrent.futures
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
import trimesh
from PIL import Image
import numpy as np

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
    box_v: float = 1.01
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
    adaptive_face_count: bool = True
    adaptive_face_ratio: float = 0.05
    min_face_count: int = 10000
    max_face_count: int = 20000
    high_detail_surface_area: float = 5.0
    high_detail_density: float = 0.43
    high_detail_original_ratio: float = 1.5

    def __post_init__(self) -> None:
        self.delight_steps = max(1, int(self.delight_steps))
        self.multiview_steps = max(1, int(self.multiview_steps))
        self.delight_cache_size = max(0, int(self.delight_cache_size))
        self.adaptive_face_ratio = float(self.adaptive_face_ratio) if self.adaptive_face_ratio else 0.1
        self.min_face_count = max(1, int(self.min_face_count))
        self.max_face_count = max(self.min_face_count, int(self.max_face_count))
        self.high_detail_surface_area = float(self.high_detail_surface_area)
        self.high_detail_density = float(self.high_detail_density)
        self.high_detail_original_ratio = max(1.0, float(self.high_detail_original_ratio))
        if isinstance(self.reuse_delighting, str):
            self.reuse_delighting = self.reuse_delighting.lower() in {"1", "true", "yes", "on"}
        if self.seed is not None and self.seed != "":
            self.seed = int(self.seed)
        else:
            self.seed = None
        if self.face_count is not None:
            self.face_count = int(self.face_count)
            if self.face_count < self.min_face_count:
                self.face_count = self.min_face_count
            if self.face_count > self.max_face_count:
                self.face_count = self.max_face_count
class GenerationService:
    """Wraps pipelines while reusing decoded imagery."""

    @staticmethod
    def sanitize_job_id(job_id: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_.-]", "_", job_id) or "job"

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
            if torch.cuda.is_available():
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
        self._texture_compiled = False
        if self.texture_enabled:
            try:
                self.texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                    tex_model_path,
                    subfolder=tex_model_subfolder,
                )
                self._maybe_compile_texture_models()
            except Exception as exc:
                logger.warning(
                    "Texture pipeline unavailable (%s); continuing with texture generation disabled.",
                    exc,
                )
                self.texture_enabled = False

        self._warmup_pipelines()

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
        texture_settings: Optional[TextureGenerationSettings] = None,
        face_count: Optional[int] = None,
    ) -> trimesh.Trimesh:
        if texture_settings is None:
            texture_settings = TextureGenerationSettings()
        original_faces = len(mesh.faces)
        metadata = getattr(mesh, 'metadata', {}) or {}
        hunyuan_meta = metadata.setdefault('hunyuan3d', {})
        hunyuan_meta.setdefault('original_face_count', original_faces)

        adaptive_target = self._select_face_target(mesh, texture_settings, fallback=face_count)
        face_target = adaptive_target
        mesh = self.float_remover(mesh)
        mesh = self.degenerate_face_remover(mesh)
        if face_target:
            mesh = self.face_reducer(mesh, max_facenum=face_target)
        hunyuan_meta['face_reduced_to'] = face_target
        mesh.metadata = metadata
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

        face_target = self._select_face_target(mesh, settings, fallback=settings.face_count)
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

    def _select_face_target(
        self,
        mesh: trimesh.Trimesh,
        settings: TextureGenerationSettings,
        *,
        fallback: Optional[int] = None,
    ) -> Optional[int]:
        candidate = fallback or settings.face_count
        if not settings.adaptive_face_count:
            return candidate
        original_faces = max(1, len(mesh.faces))
        extent = mesh.bounds[1] - mesh.bounds[0] if hasattr(mesh, 'bounds') else np.ones(3)
        bbox_area = float(2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[0] * extent[2]))
        bbox_area = max(bbox_area, 1e-6)
        surface_area = float(getattr(mesh, 'area', bbox_area)) or bbox_area
        density = surface_area / bbox_area
        density = float(np.clip(density, 0.4, 2.0))
        base_target = int(original_faces * settings.adaptive_face_ratio * density)
        min_faces = settings.min_face_count
        max_faces = settings.max_face_count
        if candidate is not None:
            max_faces = min(max_faces, int(candidate))
        target = int(np.clip(base_target, min_faces, max_faces))
        candidate_faces = int(candidate) if candidate is not None else None
        if (
            candidate_faces is not None
            and target <= min_faces
            and candidate_faces <= max_faces
            and original_faces >= candidate_faces * settings.high_detail_original_ratio
            and surface_area >= settings.high_detail_surface_area
            and density >= settings.high_detail_density
        ):
            target = candidate_faces
        logger.debug(
            'Adaptive face target: original=%s, surface=%.2f, bbox_area=%.2f, density=%.3f, base=%s, target=%s',
            original_faces,
            surface_area,
            bbox_area,
            density,
            base_target,
            target,
        )
        return target

    def _maybe_compile_texture_models(self) -> None:
        if self._texture_compiled or not hasattr(torch, "compile") or not self.texture_pipeline:
            return
        if os.environ.get('HY3DGEN_TEXTURE_COMPILE', '0') not in {'1', 'true', 'TRUE', 'True'}:
            return
        compiled = []
        try:
            delight = self.texture_pipeline.models.get('delight_model') if hasattr(self.texture_pipeline, 'models') else None
            pipe = getattr(delight, 'pipeline', None)
            if pipe is not None and hasattr(pipe, 'unet'):
                pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead')
                compiled.append('delight_unet')
        except Exception:
            logger.debug('torch.compile failed for delight_model', exc_info=True)
        try:
            multiview = self.texture_pipeline.models.get('multiview_model') if hasattr(self.texture_pipeline, 'models') else None
            pipe = getattr(multiview, 'pipeline', None)
            if pipe is not None and hasattr(pipe, 'unet'):
                pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead')
                compiled.append('multiview_unet')
        except Exception:
            logger.debug('torch.compile failed for multiview_model', exc_info=True)
        if compiled:
            self._texture_compiled = True
            logger.info('Enabled torch.compile for texture modules: %s', ', '.join(compiled))

    def _warmup_pipelines(self) -> None:
        start_time = time.perf_counter()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                futures.append(executor.submit(self._warmup_shape))
                if self.texture_enabled and self.texture_pipeline is not None:
                    futures.append(executor.submit(self._warmup_texture))
                for future in futures:
                    try:
                        future.result()
                    except Exception:  # pragma: no cover - warmup best effort
                        logger.debug('Warmup task failed.', exc_info=True)
        except Exception:  # pragma: no cover
            logger.debug('Pipeline warmup failed.', exc_info=True)
        else:
            logger.info('Pipeline warmup completed in %.2fs', time.perf_counter() - start_time)

    def _warmup_shape(self) -> None:
        if self.shape_pipeline is None:
            return
        try:
            dummy = Image.new('RGB', (512, 512), (128, 128, 128))
            self.shape_pipeline(
                image=dummy,
                num_inference_steps=1,
                guidance_scale=0.0,
                box_v=0.5,
                octree_resolution=64,
                num_chunks=512,
                enable_pbar=False,
                output_type='trimesh',
            )
        except Exception:
            logger.debug('Shape pipeline warmup failed.', exc_info=True)

    def _warmup_texture(self) -> None:
        if self.texture_pipeline is None:
            return
        try:
            mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
            mesh.metadata = {}
            dummy_image = Image.new('RGBA', (512, 512), (255, 255, 255, 255))
            warm_settings = TextureGenerationSettings(
                enabled=True,
                face_count=8000,
                delight_steps=1,
                multiview_steps=1,
                reuse_delighting=False,
                delight_cache_size=0,
                seed=0,
                adaptive_face_count=False,
            )
            mesh = self.standardize_mesh(mesh, texture_settings=warm_settings)
            self.texture_pipeline(
                mesh,
                dummy_image,
                delight_steps=warm_settings.delight_steps,
                multiview_steps=warm_settings.multiview_steps,
                reuse_delighting=warm_settings.reuse_delighting,
                delight_cache_size=warm_settings.delight_cache_size,
                seed=warm_settings.seed,
            )
        except Exception:
            logger.debug('Texture pipeline warmup failed.', exc_info=True)

    def export_mesh(
        self,
        mesh: trimesh.Trimesh,
        job_id: str,
        *,
        textured: bool = False,
        file_type: str = 'glb',
    ) -> str:
        path = self.resolve_export_path(job_id, textured=textured, file_type=file_type)

        export_kwargs = {}
        if file_type.lower() in {'glb', 'gltf'}:
            export_kwargs['include_normals'] = textured

        mesh.export(str(path), file_type=file_type, **export_kwargs)
        logger.info('Saved %s mesh to %s', 'textured' if textured else 'generated', path)
        return str(path)

    def resolve_export_path(self, job_id: str, *, textured: bool = False, file_type: str = 'glb') -> Path:
        sanitized = self.sanitize_job_id(job_id)
        job_dir = self.output_root / sanitized
        job_dir.mkdir(parents=True, exist_ok=True)
        filename = 'textured_mesh' if textured else 'mesh'
        return job_dir / f"{filename}.{file_type}"

    def mesh_to_glb_bytes(
        self,
        mesh: trimesh.Trimesh,
        *,
        textured: bool = False,
        file_type: str = 'glb',
    ) -> bytes:
        buffer = BytesIO()
        export_kwargs = {}
        if file_type.lower() in {'glb', 'gltf'}:
            export_kwargs['include_normals'] = True
        mesh.export(buffer, file_type=file_type, **export_kwargs)
        return buffer.getvalue()
