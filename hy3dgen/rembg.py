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

import logging

from PIL import Image
from rembg import remove, new_session


class BackgroundRemover():
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        providers = None
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = new_session(model_name="u2net", providers=providers)
        except Exception:
            self.logger.warning(
                "Falling back to rembg default providers; unable to initialize CUDAExecutionProvider.",
                exc_info=True,
            )
            # Fall back to default CPU session if CUDA provider is unavailable.
            self.session = new_session()
        finally:
            # rembg>=2.0.60 exposes `providers` attribute (list[str]).
            # Older versions expose the getter on the ONNX session instead.
            session_providers = getattr(self.session, "providers", None)
            if session_providers is None:
                session_providers = getattr(getattr(self.session, "session", None), "get_providers", lambda: None)()
            if session_providers:
                self.logger.info("rembg session providers: %s", session_providers)

    def __call__(self, image: Image.Image):
        output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
        return output
