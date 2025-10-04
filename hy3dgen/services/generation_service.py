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
from typing import Optional, Tuple

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


def _resolve_cuda_device_index(device: str) -> Optional[int]:
    if not torch.cuda.is_available() or not device.startswith('cuda'):
        return None
    idx = torch.cuda.current_device()
    if device != 'cuda':
        try:
            idx = int(device.split(':', 1)[1])
        except (IndexError, ValueError):
            logger.debug('Unable to parse CUDA device index from %s; using current device.', device)
    return idx


def _device_supports_bf16(device: str) -> bool:
    idx = _resolve_cuda_device_index(device)
    if idx is None:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(idx)
        if major < 9:  # Limit automatic enablement to Hopper (SM90) or newer
            return False
    except Exception:
        logger.debug('Unable to resolve CUDA capability; assuming bf16 unsupported.', exc_info=True)
        return False
    if hasattr(torch.cuda, 'is_bf16_supported'):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            logger.debug('torch.cuda.is_bf16_supported probe failed; assuming unsupported.', exc_info=True)
            return False
    return True


def _select_default_dtype(device: str) -> torch.dtype:
    """Resolve the working dtype for diffusion modules with a safe default."""
    prefer_setting = os.environ.get('HY3DGEN_USE_BF16', '').strip().lower()
    if prefer_setting in {'1', 'true', 'yes', 'on'}:
        if _device_supports_bf16(device):
            return torch.bfloat16
        logger.warning('HY3DGEN_USE_BF16 requested but device lacks bf16 support; using float16 instead.')
        return torch.float16
    if prefer_setting in {'auto'}:
        if _device_supports_bf16(device):
            return torch.bfloat16
        logger.warning('HY3DGEN_USE_BF16=auto but device lacks bf16 support; using float16.')
        return torch.float16
    return torch.float16


def _device_supports_torch_compile(device: str) -> bool:
    if not hasattr(torch, 'compile'):
        return False
    idx = _resolve_cuda_device_index(device)
    if idx is None:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(idx)
        return major >= 9
    except Exception:
        logger.debug('torch.compile support probe failed; defaulting to disabled.', exc_info=True)
        return False


def _texture_compile_requested() -> str:
    return os.environ.get('HY3DGEN_TEXTURE_COMPILE', '').strip().lower()


def _try_compile_module(module: Optional[torch.nn.Module], *, mode: str = 'reduce-overhead') -> Tuple[Optional[torch.nn.Module], bool]:
    if module is None or not hasattr(torch, "compile"):
        return module, False
    try:
        compiled = torch.compile(module, mode=mode)
    except Exception:
        logger.debug('torch.compile failed for %s', getattr(module, '__class__', type(module)), exc_info=True)
        return module, False
    if not isinstance(compiled, torch.nn.Module):
        logger.debug('torch.compile returned non-module for %s; skipping optimisation.', getattr(module, '__class__', type(module)))
        return module, False
    # Preserve common metadata used downstream (e.g., diffusers expects `.config`).
    for attr in ("config",):
        if hasattr(module, attr) and not hasattr(compiled, attr):
            setattr(compiled, attr, getattr(module, attr))
    return compiled, True


def _ensure_inductor_cudagraphs_disabled() -> None:
    """Disable CUDA graph capture in torch.compile runtime to avoid unsupported ops."""
    if getattr(_ensure_inductor_cudagraphs_disabled, '_applied', False):
        return
    try:
        from torch._inductor import config as inductor_config  # type: ignore
    except Exception:
        logger.debug('torch._inductor.config unavailable; cannot tweak cudagraph settings.', exc_info=True)
        _ensure_inductor_cudagraphs_disabled._applied = True  # type: ignore[attr-defined]
        return
    try:
        current = getattr(getattr(inductor_config, 'triton', None), 'cudagraphs', None)
        if current is False:
            _ensure_inductor_cudagraphs_disabled._applied = True  # type: ignore[attr-defined]
            return
        if current is not None:
            inductor_config.triton.cudagraphs = False
            logger.info('Disabled torch._inductor.triton.cudagraphs for texture torch.compile path.')
    except Exception:
        logger.debug('Failed to disable torch._inductor.triton.cudagraphs; continuing with defaults.', exc_info=True)
    finally:
        _ensure_inductor_cudagraphs_disabled._applied = True  # type: ignore[attr-defined]


def _module_to_channels_last(module: Optional[torch.nn.Module]) -> bool:
    if module is None:
        return False
    try:
        module.to(memory_format=torch.channels_last)
    except Exception:
        logger.debug('Unable to convert %s to channels_last layout', type(module), exc_info=True)
        return False
    return True


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
    face_count: int = 15000
    delight_steps: int = 20
    multiview_steps: int = 8
    reuse_delighting: bool = False
    delight_cache_size: int = 8
    seed: Optional[int] = 0
    adaptive_face_count: bool = True
    adaptive_face_ratio: float = 0.035
    min_face_count: int = 10000
    max_face_count: int = 15000
    high_detail_surface_area: float = 5.0
    high_detail_density: float = 0.43
    high_detail_original_ratio: float = 1.5
    multiview_preset: Optional[str] = None

    def __post_init__(self) -> None:
        self.delight_steps = max(1, int(self.delight_steps))
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
        preset = (self.multiview_preset or '').strip().lower()
        if preset:
            preset_map = {
                'fast': 6,
                'balanced': 8,
                'quality': 12,
                'ultra': 16,
            }
            steps = preset_map.get(preset)
            if steps is not None:
                self.multiview_steps = steps
            else:
                logger.warning('Unknown multiview_preset "%s"; using explicit multiview_steps=%s', preset, self.multiview_steps)
        else:
            self.multiview_preset = None
        self.multiview_steps = max(1, int(self.multiview_steps))
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
        model_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.device = device
        self.dtype: torch.dtype = model_dtype or _select_default_dtype(device)
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
            dtype=self.dtype,
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
        self._compile_shape_pipeline()
        self._apply_shape_channels_last()

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
                    torch_dtype=self.dtype,
                )
                self._maybe_compile_texture_models()
                self._apply_texture_channels_last()
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

        if previously_reduced is not None:
            try:
                face_target = int(previously_reduced)
            except (TypeError, ValueError):
                face_target = None
            else:
                logger.debug('Texture stage reusing face target from mesh stage: %s', face_target)
        else:
            face_target = self._select_face_target(mesh, settings, fallback=settings.face_count)

        if face_target:
            if len(mesh.faces) > face_target:
                mesh = self.face_reducer(mesh, max_facenum=face_target)
            elif len(mesh.faces) < face_target:
                logger.debug('Texture stage retaining mesh with %s faces below target %s', len(mesh.faces), face_target)
            else:
                logger.debug('Texture stage preserving exact face count: %s', face_target)

        mesh = self._tag_mesh_metadata(mesh, face_target=face_target)
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
        current_faces = max(1, len(mesh.faces))
        metadata = getattr(mesh, 'metadata', None)
        stored_original = None
        if isinstance(metadata, dict):
            hunyuan_meta = metadata.get('hunyuan3d')
            if isinstance(hunyuan_meta, dict):
                stored_original = hunyuan_meta.get('original_face_count')
                if stored_original is not None:
                    try:
                        stored_original = int(stored_original)
                    except (TypeError, ValueError):
                        stored_original = None
        original_faces = stored_original if stored_original and stored_original > 0 else current_faces
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
            'Adaptive face target: original=%s, current=%s, surface=%.2f, bbox_area=%.2f, density=%.3f, base=%s, target=%s',
            original_faces,
            current_faces,
            surface_area,
            bbox_area,
            density,
            base_target,
            target,
        )
        return target

    def _maybe_compile_texture_models(self) -> None:
        if self._texture_compiled or not self.texture_pipeline:
            return
        env_pref = _texture_compile_requested()
        if env_pref in {'', '0', 'false', 'no', 'off'}:
            return
        if not hasattr(torch, 'compile'):
            return
        if env_pref not in {'1', 'true', 'yes', 'on'}:
            if env_pref != 'auto' or not _device_supports_torch_compile(self.device):
                return
        _ensure_inductor_cudagraphs_disabled()
        compiled = []
        try:
            delight = self.texture_pipeline.models.get('delight_model') if hasattr(self.texture_pipeline, 'models') else None
            pipe = getattr(delight, 'pipeline', None)
            if pipe is not None and hasattr(pipe, 'unet'):
                compiled_unet, ok = _try_compile_module(pipe.unet)
                if ok:
                    pipe.unet = compiled_unet
                    compiled.append('delight_unet')
        except Exception:
            logger.debug('torch.compile failed for delight_model', exc_info=True)
        try:
            multiview = self.texture_pipeline.models.get('multiview_model') if hasattr(self.texture_pipeline, 'models') else None
            pipe = getattr(multiview, 'pipeline', None)
            if pipe is not None and hasattr(pipe, 'unet'):
                compiled_unet, ok = _try_compile_module(pipe.unet)
                if ok:
                    pipe.unet = compiled_unet
                    compiled.append('multiview_unet')
        except Exception:
            logger.debug('torch.compile failed for multiview_model', exc_info=True)
        if compiled:
            self._texture_compiled = True
            logger.info('Enabled torch.compile for texture modules: %s', ', '.join(compiled))

    def _apply_shape_channels_last(self) -> None:
        device_str = str(self.device)
        if not (torch.cuda.is_available() and device_str.startswith('cuda')):
            return
        converted: list[str] = []
        if _module_to_channels_last(getattr(self.shape_pipeline, 'model', None)):
            converted.append('shape_unet')
        if _module_to_channels_last(getattr(self.shape_pipeline, 'vae', None)):
            converted.append('shape_vae')
        if _module_to_channels_last(getattr(self.shape_pipeline, 'conditioner', None)):
            converted.append('shape_conditioner')
        if converted:
            logger.info('Applied channels_last layout to %s', ', '.join(converted))

    def _apply_texture_channels_last(self) -> None:
        device_str = str(self.device)
        if not (self.texture_pipeline and torch.cuda.is_available() and device_str.startswith('cuda')):
            return
        converted: list[str] = []
        models = getattr(self.texture_pipeline, 'models', {})
        delight = models.get('delight_model') if isinstance(models, dict) else None
        multiview = models.get('multiview_model') if isinstance(models, dict) else None

        delight_pipeline = getattr(delight, 'pipeline', None)
        if delight_pipeline is not None:
            if _module_to_channels_last(getattr(delight_pipeline, 'unet', None)):
                converted.append('delight_unet')
            if _module_to_channels_last(getattr(delight_pipeline, 'vae', None)):
                converted.append('delight_vae')

        multiview_pipeline = getattr(multiview, 'pipeline', None)
        if multiview_pipeline is not None:
            if _module_to_channels_last(getattr(multiview_pipeline, 'unet', None)):
                converted.append('multiview_unet')
            if _module_to_channels_last(getattr(multiview_pipeline, 'vae', None)):
                converted.append('multiview_vae')

        if converted:
            logger.info('Applied channels_last layout to %s', ', '.join(converted))

    def _compile_shape_pipeline(self) -> None:
        if not hasattr(torch, "compile"):
            return
        compile_fn = getattr(self.shape_pipeline, 'compile', None)
        if compile_fn is None:
            return
        try:
            compile_fn()
            logger.info('Enabled torch.compile for shape pipeline modules.')
        except Exception:
            logger.debug('torch.compile failed for shape pipeline', exc_info=True)

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
        mesh: Optional[trimesh.Trimesh],
        job_id: str,
        *,
        textured: bool = False,
        mesh_bytes: Optional[bytes] = None,
        file_type: str = 'glb',
    ) -> str:
        path = self.resolve_export_path(job_id, textured=textured, file_type=file_type)

        if mesh_bytes is not None:
            path.write_bytes(mesh_bytes)
            logger.info('Saved %s mesh to %s', 'textured' if textured else 'generated', path)
            return str(path)

        if mesh is None:
            raise ValueError('mesh or mesh_bytes must be provided when exporting a mesh')

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
