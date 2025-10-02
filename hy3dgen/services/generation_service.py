"""Orchestrates mesh and texture generation pipelines."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
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


def mesh_to_base64(mesh: trimesh.Trimesh, file_type: str = 'glb') -> str:
    buffer = BytesIO()
    mesh.export(buffer, file_type=file_type)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class GenerationService:
    """Wraps pipelines while reusing decoded imagery."""

    def __init__(
        self,
        *,
        model_path: str = 'tencent/Hunyuan3D-2mini',
        tex_model_path: Optional[str] = 'tencent/Hunyuan3D-2',
        device: str = 'cuda',
        enable_texture: bool = True,
        shape_model_subfolder: str = 'hunyuan3d-dit-v2-mini-turbo',
        tex_model_subfolder: str = 'hunyuan3d-paint-v2-0-turbo',
    ) -> None:
        self.device = device
        self.rembg = BackgroundRemover()

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
        self.texture_pipeline: Optional[Hunyuan3DPaintPipeline]
        if self.texture_enabled:
            self.texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                tex_model_path,
                subfolder=tex_model_subfolder,
            )
        else:
            self.texture_pipeline = None

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

    def generate_textured_mesh(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        settings: TextureGenerationSettings,
    ) -> trimesh.Trimesh:
        if not self.texture_enabled or not settings.enabled:
            raise RuntimeError("Texture generation is disabled")
        mesh = self.float_remover(mesh)
        mesh = self.degenerate_face_remover(mesh)
        mesh = self.face_reducer(mesh, max_facenum=settings.face_count)
        if self.texture_pipeline is None:
            raise RuntimeError("Texture pipeline is not initialized")
        return self.texture_pipeline(mesh, image)

    @staticmethod
    def to_base64(mesh: trimesh.Trimesh) -> str:
        return mesh_to_base64(mesh)
