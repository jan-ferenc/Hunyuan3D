"""Orchestrates mesh and texture generation pipelines."""

from __future__ import annotations

import logging
import os
import re
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
from hy3dgen.texgen import Hunyuan3DPaintPipeline


logger = logging.getLogger(__name__)


@dataclass
class ShapeGenerationSettings:
    num_inference_steps: int = 4
    guidance_scale: float = 3.0
    box_v: float = 0.9
    octree_resolution: int = 192
    num_chunks: int = 20000
    mc_algo: Optional[str] = 'mc'
    seed: Optional[int] = None


@dataclass
class TextureGenerationSettings:
    enabled: bool = True
    face_count: int = 40000
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
            subfolder=shape_model_subfolder,
        )

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
        mesh = self.float_remover(mesh)
        mesh = self.degenerate_face_remover(mesh)

        metadata = getattr(mesh, 'metadata', {}) or {}
        hunyuan_meta = metadata.get('hunyuan3d') if isinstance(metadata, dict) else None
        previously_reduced = None
        if isinstance(hunyuan_meta, dict):
            previously_reduced = hunyuan_meta.get('face_reduced_to')

        face_target = settings.face_count
        if face_target and (previously_reduced is None or previously_reduced > face_target):
            mesh = self.face_reducer(mesh, max_facenum=face_target)
            mesh = self._tag_mesh_metadata(mesh, face_target=face_target)
        else:
            mesh = self._tag_mesh_metadata(mesh, face_target=previously_reduced)
        if self.texture_pipeline is None:
            raise RuntimeError("Texture pipeline is not initialized")
        return self.texture_pipeline(mesh, image)

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
