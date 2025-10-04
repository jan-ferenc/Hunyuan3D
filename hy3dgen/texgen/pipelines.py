# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import concurrent.futures
import logging
import numpy as np
import os
import torch
from PIL import Image
from typing import List, Union, Optional
import time

try:
    import safetensors.torch
except ImportError:  # pragma: no cover - optional dependency but expected to be present
    safetensors = None
else:
    safetensors = safetensors.torch


from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.imagesuper_utils import Image_Super_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, subfolder_name, torch_dtype=torch.float16):
        self.device = 'cuda'
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path
        self.torch_dtype = torch_dtype

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 512
        self.texture_size = 512
        self.bake_exp = 4
        self.merge_method = 'fast'

        self.pipe_dict = {'hunyuan3d-paint-v2-0': 'hunyuanpaint', 'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo'}
        self.pipe_name = self.pipe_dict[subfolder_name]


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path, subfolder='hunyuan3d-paint-v2-0-turbo', torch_dtype=torch.float16):
        original_model_path = model_path
        if not os.path.exists(model_path):
            # try local path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)

            if not os.path.exists(delight_model_path) or not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    # download from huggingface
                    allow_patterns = [
                        "hunyuan3d-delight-v2-0/*",
                        "hunyuan3d-delight-v2-0/**",
                        f"{subfolder}/*",
                        f"{subfolder}/**",
                        f"{subfolder}/**/*",
                    ]
                    local_cache_dir = os.path.expanduser(os.path.join(base_dir, original_model_path))
                    os.makedirs(local_cache_dir, exist_ok=True)
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path,
                        allow_patterns=allow_patterns,
                        local_dir=local_cache_dir,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                        force_download=False,
                        revision='main',
                    )
                    cls._ensure_safetensors(model_path, subfolder)
                    delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    multiview_model_path = os.path.join(model_path, subfolder)
                    return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder, torch_dtype=torch_dtype))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Something wrong while loading {model_path}")
            else:
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder, torch_dtype=torch_dtype))
        else:
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)
            cls._ensure_safetensors(model_path, subfolder)
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder, torch_dtype=torch_dtype))

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)

        self.load_models()

    def load_models(self):
        # empty cude cache
        torch.cuda.empty_cache()
        # Load model
        self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)
        # self.models['super_model'] = Image_Super_Net(self.config)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
    @staticmethod
    def _ensure_safetensors(root_path: str, subfolder: str) -> None:
        if safetensors is None:
            logger.warning("safetensors not available; skipping .bin to .safetensors conversion")
            return

        targets = [
            os.path.join(root_path, subfolder, 'vae'),
            os.path.join(root_path, subfolder, 'unet'),
        ]
        for path in targets:
            if not os.path.isdir(path):
                continue
            bin_path = os.path.join(path, 'diffusion_pytorch_model.bin')
            safetensor_path = os.path.join(path, 'diffusion_pytorch_model.safetensors')
            if os.path.exists(safetensor_path) or not os.path.exists(bin_path):
                continue
            try:
                logger.info("Converting %s to %s", bin_path, safetensor_path)
                weights = torch.load(bin_path, map_location='cpu')
                safetensors.save_file(weights, safetensor_path)
            except Exception:  # pragma: no cover - safeguard
                logger.exception("Failed converting %s to safetensors", bin_path)


    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='pl')
            position_maps.append(position_map)

        return position_maps

    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB':
            return image
        elif image.mode == 'L':
            image = image.convert('RGB')
            return image

        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")

        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)

        cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))

        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        new_width = width + 2 * border_width
        new_height = height + 2 * border_height

        square_size = max(new_width, new_height)

        new_image = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))

        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height

        new_image.paste(cropped_image, (paste_x, paste_y))
        return new_image

    @torch.no_grad()
    def __call__(
        self,
        mesh,
        image,
        *,
        delight_steps: int = 50,
        multiview_steps: int = 30,
        reuse_delighting: bool = False,
        delight_cache_size: int = 8,
        seed: Optional[int] = None,
    ):

        if not isinstance(image, List):
            image = [image]

        images_prompt = []
        for i in range(len(image)):
            if isinstance(image[i], str):
                image_prompt = Image.open(image[i])
            else:
                image_prompt = image[i]
            images_prompt.append(image_prompt)

        images_prompt = [self.recenter_image(image_prompt) for image_prompt in images_prompt]

        selected_camera_elevs = self.config.candidate_camera_elevs
        selected_camera_azims = self.config.candidate_camera_azims
        selected_view_weights = self.config.candidate_view_weights

        def _prepare_mesh_assets(mesh_input):
            unwrap_start = time.perf_counter()
            wrapped_mesh = mesh_uv_wrap(mesh_input)
            unwrap_elapsed = time.perf_counter() - unwrap_start
            load_start = time.perf_counter()
            self.render.load_mesh(wrapped_mesh)
            load_elapsed = time.perf_counter() - load_start
            normals = self.render_normal_multiview(
                selected_camera_elevs,
                selected_camera_azims,
                use_abs_coor=True,
            )
            positions = self.render_position_multiview(
                selected_camera_elevs,
                selected_camera_azims,
            )
            return wrapped_mesh, unwrap_elapsed, load_elapsed, normals, positions

        prep_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        prep_future: Optional[concurrent.futures.Future] = None
        try:
            prep_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            prep_future = prep_executor.submit(_prepare_mesh_assets, mesh)
        except Exception:
            logger.debug('Unable to schedule background mesh preparation; will prepare synchronously.', exc_info=True)
            if prep_executor is not None:
                prep_executor.shutdown(wait=False)
            prep_executor = None
            prep_future = None

        delighted_images = []
        for image_prompt in images_prompt:
            delight_start = time.perf_counter()
            delighted = self.models['delight_model'](
                image_prompt,
                num_inference_steps=delight_steps,
                use_cache=reuse_delighting,
                cache_size=delight_cache_size,
                seed=seed,
            )
            delight_elapsed = time.perf_counter() - delight_start
            logger.debug('Texture delight stage completed in %.3fs (steps=%s)', delight_elapsed, delight_steps)
            delighted_images.append(delighted)
        images_prompt = delighted_images

        if prep_future is not None:
            try:
                mesh, uv_elapsed, load_elapsed, normal_maps, position_maps = prep_future.result()
                logger.debug('Texture UV unwrap completed in %.3fs (faces=%s)', uv_elapsed, len(mesh.faces))
                logger.debug('Texture load_mesh completed in %.3fs', load_elapsed)
            except Exception:
                logger.exception('Background mesh preparation failed; retrying synchronously')
                sync_start = time.perf_counter()
                mesh = mesh_uv_wrap(mesh)
                uv_elapsed = time.perf_counter() - sync_start
                logger.debug('Texture UV unwrap completed in %.3fs (faces=%s)', uv_elapsed, len(mesh.faces))
                load_start = time.perf_counter()
                self.render.load_mesh(mesh)
                load_elapsed = time.perf_counter() - load_start
                logger.debug('Texture load_mesh completed in %.3fs', load_elapsed)
                normal_maps = self.render_normal_multiview(
                    selected_camera_elevs,
                    selected_camera_azims,
                    use_abs_coor=True,
                )
                position_maps = self.render_position_multiview(
                    selected_camera_elevs,
                    selected_camera_azims,
                )
            finally:
                if prep_executor is not None:
                    prep_executor.shutdown(wait=False)
        else:
            sync_start = time.perf_counter()
            mesh = mesh_uv_wrap(mesh)
            uv_elapsed = time.perf_counter() - sync_start
            logger.debug('Texture UV unwrap completed in %.3fs (faces=%s)', uv_elapsed, len(mesh.faces))
            load_start = time.perf_counter()
            self.render.load_mesh(mesh)
            load_elapsed = time.perf_counter() - load_start
            logger.debug('Texture load_mesh completed in %.3fs', load_elapsed)
            normal_maps = self.render_normal_multiview(
                selected_camera_elevs,
                selected_camera_azims,
                use_abs_coor=True,
            )
            position_maps = self.render_position_multiview(
                selected_camera_elevs,
                selected_camera_azims,
            )

        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        multiview_start = time.perf_counter()
        multiviews = self.models['multiview_model'](
            images_prompt,
            normal_maps + position_maps,
            camera_info,
            num_inference_steps=multiview_steps,
            seed=seed,
        )
        logger.debug('Texture multiview diffusion completed in %.3fs (steps=%s)', time.perf_counter() - multiview_start, multiview_steps)

        for i in range(len(multiviews)):
            # multiviews[i] = self.models['super_model'](multiviews[i])
            multiviews[i] = multiviews[i].resize(
                (self.config.render_size, self.config.render_size))

        bake_start = time.perf_counter()
        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)
        logger.debug('Texture baking completed in %.3fs', time.perf_counter() - bake_start)

        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        inpaint_start = time.perf_counter()
        texture = self.texture_inpaint(texture, mask_np)
        logger.debug('Texture inpaint completed in %.3fs', time.perf_counter() - inpaint_start)

        self.render.set_texture(texture)
        save_start = time.perf_counter()
        textured_mesh = self.render.save_mesh()
        logger.debug('Texture save_mesh completed in %.3fs', time.perf_counter() - save_start)

        return textured_mesh
