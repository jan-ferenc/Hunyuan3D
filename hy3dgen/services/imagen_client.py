"""Google Imagen v4 fast client with minimal-copy image accessors."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

from PIL import Image

try:  # pragma: no cover - optional dependency is runtime provided
    from google import genai
    from google.genai import types
except Exception as exc:  # pragma: no cover - defer import errors until used
    genai = None  # type: ignore
    types = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

DEFAULT_PROMPT_TEMPLATE = (
    "Generate an image of {USER_QUERY}. "
    "A photorealistic product-studio photograph showing exactly one {USER_QUERY}, "
    "centered and filling ~90% of the frame, fully visible with no cropping. "
    "Seamless pure white background (#FFFFFF) that is featureless; no props, no other objects, "
    "no text, no logos, no people, no hands, no stands, no reflections, no shadows (or only a very soft, "
    "faint contact shadow directly beneath the object if needed for realism). "
    "Neutral, even softbox lighting, sharp focus, true-to-life materials and colors, clean edges, professional catalog look."
)


@dataclass
class ImagePayload:
    """Caches different views of a base64 image to avoid redundant copies."""

    base64_data: str
    _binary: Optional[bytes] = field(default=None, init=False, repr=False)
    _pil: Optional[Image.Image] = field(default=None, init=False, repr=False)
    _pil_rgba: Optional[Image.Image] = field(default=None, init=False, repr=False)

    def as_base64(self) -> str:
        return self.base64_data

    def as_bytes(self) -> bytes:
        if self._binary is None:
            self._binary = base64.b64decode(self.base64_data)
        return self._binary

    def as_bytes_io(self) -> BytesIO:
        return BytesIO(self.as_bytes())

    def as_pil(self) -> Image.Image:
        if self._pil is None:
            self._pil = Image.open(self.as_bytes_io())
        return self._pil

    def as_pil_rgba(self) -> Image.Image:
        if self._pil_rgba is None:
            self._pil_rgba = self.as_pil().convert("RGBA")
        return self._pil_rgba


class GoogleImagenClient:
    """Thin wrapper around Google's Imagen 4.0 fast model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model_name: str = "imagen-4.0-fast-generate-001",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("A Google Imagen API key is required. Set GOOGLE_API_KEY or pass api_key explicitly.")

        if genai is None or types is None:  # pragma: no cover - triggered when dependency missing
            raise RuntimeError(
                "google-genai dependency is not available",  # type: ignore[possibly-undefined]
            ) from _import_error

        self.api_key = api_key
        self.model_name = model_name
        self.prompt_template = prompt_template
        self._client = genai.Client(api_key=api_key)

    def build_prompt(self, user_query: str) -> str:
        return self.prompt_template.format(USER_QUERY=user_query)

    def generate(self, user_query: str) -> ImagePayload:
        prompt = self.build_prompt(user_query)
        response = self._client.models.generate_images(
            model=self.model_name,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )

        if not getattr(response, "generated_images", None):
            raise RuntimeError("Imagen response did not contain any images")

        image_obj = response.generated_images[0]
        base64_data = self._normalize_to_base64(getattr(image_obj, "image", None))
        if not base64_data:
            raise RuntimeError("Unable to extract image payload from Imagen response")
        return ImagePayload(base64_data=base64_data)

    @staticmethod
    def _normalize_to_base64(image_field) -> str:
        if image_field is None:
            return ""

        if isinstance(image_field, str):
            data = image_field.strip()
            if data.startswith("data:"):
                _, data = data.split(",", 1)
            return data

        if isinstance(image_field, bytes):
            return base64.b64encode(image_field).decode("utf-8")

        read = getattr(image_field, "read", None)
        if callable(read):
            return base64.b64encode(read()).decode("utf-8")

        raw_bytes = getattr(image_field, "image_bytes", None)
        if isinstance(raw_bytes, (bytes, bytearray)):
            return base64.b64encode(bytes(raw_bytes)).decode("utf-8")

        alt_base64 = getattr(image_field, "image_base64", None) or getattr(image_field, "image_b64", None)
        if isinstance(alt_base64, str):
            return alt_base64.strip()

        raise TypeError(f"Unsupported image payload type: {type(image_field)!r}")
