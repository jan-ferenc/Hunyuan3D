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

import cv2
import logging
import numpy as np
import os
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import hashlib
from collections import OrderedDict
import threading
from typing import Optional


logger = logging.getLogger(__name__)


def _ensure_inductor_cudagraphs_disabled() -> None:
    if getattr(_ensure_inductor_cudagraphs_disabled, '_applied', False):
        return
    try:
        from torch._inductor import config as inductor_config  # type: ignore
    except Exception:
        logger.debug('torch._inductor.config unavailable for cudagraph toggle.', exc_info=True)
        _ensure_inductor_cudagraphs_disabled._applied = True  # type: ignore[attr-defined]
        return
    try:
        current = getattr(getattr(inductor_config, 'triton', None), 'cudagraphs', None)
        if current is not None and current is not False:
            inductor_config.triton.cudagraphs = False
            logger.info('Disabled torch._inductor.triton.cudagraphs for delight torch.compile path.')
    except Exception:
        logger.debug('Unable to disable torch._inductor.triton.cudagraphs in delight helper.', exc_info=True)
    finally:
        _ensure_inductor_cudagraphs_disabled._applied = True  # type: ignore[attr-defined]


class Light_Shadow_Remover():
    def __init__(self, config):
        self.device = config.device
        self.cfg_image = 1.5
        self.cfg_text = 1.0
        self._cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.default_cache_size = 8
        self.default_seed = 42

        torch_dtype = getattr(config, 'torch_dtype', torch.float16)

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            config.light_remover_ckpt_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline.set_progress_bar_config(disable=True)
        try:  # Optional acceleration when xformers is available
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:  # pragma: no cover - best effort, do not fail on missing dependency
            pass

        self.pipeline = pipeline.to(self.device, torch_dtype)
        compile_flag = os.environ.get('HY3DGEN_TEXTURE_COMPILE', '0').lower() in {'1', 'true', 'yes', 'on'}
        if compile_flag and hasattr(torch, "compile"):
            _ensure_inductor_cudagraphs_disabled()
            try:
                compiled_unet = torch.compile(self.pipeline.unet, mode='reduce-overhead')
            except Exception:
                logger.debug('torch.compile failed for Light_Shadow_Remover UNet', exc_info=True)
            else:
                if isinstance(compiled_unet, torch.nn.Module):
                    if hasattr(self.pipeline.unet, 'config') and not hasattr(compiled_unet, 'config'):
                        compiled_unet.config = self.pipeline.unet.config
                    self.pipeline.unet = compiled_unet
                    logger.info('Enabled torch.compile for Light_Shadow_Remover UNet.')
                else:
                    logger.debug('torch.compile returned non-module for Light_Shadow_Remover UNet; skipping optimisation.')

    def _make_cache_key(self, image: Image.Image, steps: int, seed: int) -> str:
        # Hash resized RGBA bytes to avoid redundant pix2pix invocations.
        data = image.tobytes()
        digest = hashlib.sha1(data).hexdigest()
        return f"{image.size[0]}x{image.size[1]}:{steps}:{seed}:{digest}"

    def _get_cached(self, key: str) -> Optional[Image.Image]:
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            # Preserve LRU order.
            self._cache.move_to_end(key)
            return cached.copy()

    def _store_cached(self, key: str, image: Image.Image, *, cache_size: int) -> None:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = image.copy()
            while len(self._cache) > cache_size:
                self._cache.popitem(last=False)
    
    def recorrect_rgb(self, src_image, target_image, alpha_channel, scale=0.95):
        
        def flat_and_mask(bgr, a):
            mask = torch.where(a > 0.5, True, False)
            bgr_flat = bgr.reshape(-1, bgr.shape[-1])
            mask_flat = mask.reshape(-1)
            bgr_flat_masked = bgr_flat[mask_flat, :]
            return bgr_flat_masked
        
        src_flat = flat_and_mask(src_image, alpha_channel)
        target_flat = flat_and_mask(target_image, alpha_channel)
        corrected_bgr = torch.zeros_like(src_image)

        for i in range(3): 
            src_mean, src_stddev = torch.mean(src_flat[:, i]), torch.std(src_flat[:, i])
            target_mean, target_stddev = torch.mean(target_flat[:, i]), torch.std(target_flat[:, i])
            corrected_bgr[:, :, i] = torch.clamp(
                (src_image[:, :, i] - scale * src_mean) * 
                (target_stddev / src_stddev) + scale * target_mean, 
                0, 1)

        src_mse = torch.mean((src_image - target_image) ** 2)
        modify_mse = torch.mean((corrected_bgr - target_image) ** 2)
        if src_mse < modify_mse:
            corrected_bgr = torch.cat([src_image, alpha_channel], dim=-1)
        else: 
            corrected_bgr = torch.cat([corrected_bgr, alpha_channel], dim=-1)

        return corrected_bgr

    @torch.no_grad()
    def __call__(
        self,
        image,
        *,
        num_inference_steps: Optional[int] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
        cache_size: Optional[int] = None,
        seed: Optional[int] = None,
    ):

        image = image.resize((512, 512))
        steps = num_inference_steps or 50
        runtime_seed = seed if seed is not None else self.default_seed
        resolved_cache_size = self.default_cache_size if cache_size is None else max(0, int(cache_size))
        key = None
        if use_cache and resolved_cache_size > 0:
            key = cache_key or self._make_cache_key(image, steps, runtime_seed)
            cached = self._get_cached(key)
            if cached is not None:
                return cached

        if image.mode == 'RGBA':
            image_array = np.array(image)
            alpha_channel = image_array[:, :, 3]
            erosion_size = 3
            kernel = np.ones((erosion_size, erosion_size), np.uint8)
            alpha_channel = cv2.erode(alpha_channel, kernel, iterations=1)
            image_array[alpha_channel == 0, :3] = 255
            image_array[:, :, 3] = alpha_channel
            image = Image.fromarray(image_array)

            image_tensor = torch.tensor(np.array(image) / 255.0).to(self.device)
            alpha = image_tensor[:, :, 3:]
            rgb_target = image_tensor[:, :, :3]
        else:
            image_tensor = torch.tensor(np.array(image) / 255.0).to(self.device)
            alpha = torch.ones_like(image_tensor)[:, :, :1]
            rgb_target = image_tensor[:, :, :3]

        image = image.convert('RGB')

        generator = torch.Generator(device=self.device)
        generator.manual_seed(runtime_seed)

        with self._inference_lock:
            image = self.pipeline(
                prompt="",
                image=image,
                generator=generator,
                height=512,
                width=512,
                num_inference_steps=steps,
                image_guidance_scale=self.cfg_image,
                guidance_scale=self.cfg_text,
            ).images[0]

        image_tensor = torch.tensor(np.array(image)/255.0).to(self.device)
        rgb_src = image_tensor[:,:,:3]
        image = self.recorrect_rgb(rgb_src, rgb_target, alpha)
        image = image[:,:,:3]*image[:,:,3:] + torch.ones_like(image[:,:,:3])*(1.0-image[:,:,3:])
        image = Image.fromarray((image.cpu().numpy()*255).astype(np.uint8))

        if use_cache and key is not None and resolved_cache_size > 0:
            self._store_cached(key, image, cache_size=resolved_cache_size)

        return image
