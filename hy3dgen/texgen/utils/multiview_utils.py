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

import os
import logging
import random

import numpy as np
import torch
from typing import List
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, LCMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline as BasePipeline

import importlib

logger = logging.getLogger(__name__)


class Multiview_Diffusion_Net():
    def __init__(self, config) -> None:
        self.device = config.device
        self.view_size = 512
        multiview_ckpt_path = config.multiview_ckpt_path

        current_file_path = os.path.abspath(__file__)
        custom_pipeline_path = os.path.join(os.path.dirname(current_file_path), '..', 'hunyuanpaint')

        pipeline = DiffusionPipeline.from_pretrained(
            multiview_ckpt_path,
            custom_pipeline=custom_pipeline_path,
            torch_dtype=torch.float16,
        )

        # Ensure UNet class identity matches the custom pipeline's runtime module
        try:
            unet_mod = importlib.import_module('diffusers_modules.local.unet.modules')
            expected_unet_cls = getattr(unet_mod, 'UNet2p5DConditionModel')
        except Exception:
            unet_mod = None
            expected_unet_cls = None

        if expected_unet_cls is not None and not isinstance(pipeline.unet, expected_unet_cls):
            current_cls = pipeline.unet.__class__
            if current_cls.__name__ == expected_unet_cls.__name__:
                logger.info(
                    'Aligning UNet class identity with %s',
                    unet_mod.__name__ if unet_mod else 'expected module',
                )
                pipeline.unet.__class__ = expected_unet_cls
            else:
                unet_path = os.path.join(multiview_ckpt_path, 'unet')
                dtype = getattr(pipeline.unet, 'dtype', torch.float16)
                device = pipeline.device
                logger.info('Reloading UNet weights into expected class from %s', unet_mod.__name__)
                try:
                    pipeline.unet = expected_unet_cls.from_pretrained(
                        unet_path,
                        torch_dtype=dtype,
                    ).to(device)
                except Exception:  # pragma: no cover - safety net
                    logger.exception("Failed to load expected UNet2p5DConditionModel from %s", unet_path)
                    raise RuntimeError(
                        "Texture pipeline requires UNet2p5DConditionModel; conversion failed."
                    )

        if config.pipe_name in ['hunyuanpaint']:
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config,
                                                                             timestep_spacing='trailing')
        elif config.pipe_name in ['hunyuanpaint-turbo']:
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config,
                                                        timestep_spacing='trailing')
            pipeline.set_turbo(True)
            # pipeline.prepare() 

        pipeline.set_progress_bar_config(disable=True)
        self.pipeline = pipeline.to(self.device)

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    def __call__(self, input_images, control_images, camera_info):

        self.seed_everything(0)

        if not isinstance(input_images, List):
            input_images = [input_images]

        input_images = [input_image.resize((self.view_size, self.view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((self.view_size, self.view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')

        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        camera_info_gen = [camera_info]
        camera_info_ref = [[0]]
        kwargs['width'] = self.view_size
        kwargs['height'] = self.view_size
        kwargs['num_in_batch'] = num_view
        kwargs['camera_info_gen'] = camera_info_gen
        kwargs['camera_info_ref'] = camera_info_ref
        kwargs["normal_imgs"] = normal_image
        kwargs["position_imgs"] = position_image

        mvd_image = self.pipeline(input_images, num_inference_steps=30, **kwargs).images

        return mvd_image
