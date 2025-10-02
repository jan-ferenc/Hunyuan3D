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

"""FastAPI server that now sources conditioning images from Google Imagen."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import sys
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from hy3dgen.services.generation_service import (
    GenerationService,
    ShapeGenerationSettings,
    TextureGenerationSettings,
)
from hy3dgen.services.imagen_client import GoogleImagenClient

LOGDIR = '.'

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """Fake file-like stream object that redirects writes to a logger instance."""

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

logger = build_logger("controller", f"{SAVE_DIR}/controller.log")

model_semaphore: Optional[asyncio.Semaphore] = None
generation_service: Optional[GenerationService] = None
imagen_client: Optional[GoogleImagenClient] = None


def _merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for data in dicts:
        for key, value in data.items():
            if value is not None:
                merged[key] = value
    return merged


def parse_shape_settings(payload: Dict[str, Any]) -> ShapeGenerationSettings:
    shape_overrides = payload.get('shape') or {}
    if not isinstance(shape_overrides, dict):
        raise HTTPException(status_code=400, detail="'shape' must be an object")

    top_level: Dict[str, Any] = {}
    for key in ("num_inference_steps", "guidance_scale", "box_v", "octree_resolution", "num_chunks", "mc_algo", "seed"):
        if key in payload:
            top_level[key] = payload[key]

    combined = _merge_dicts(ShapeGenerationSettings().__dict__, shape_overrides, top_level)
    try:
        return ShapeGenerationSettings(**combined)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid shape settings: {exc}") from exc


def parse_texture_settings(payload: Dict[str, Any]) -> TextureGenerationSettings:
    texture_overrides = payload.get('texture_settings') or {}
    if not isinstance(texture_overrides, dict):
        raise HTTPException(status_code=400, detail="'texture_settings' must be an object")

    enabled = payload.get('texture')
    top_level: Dict[str, Any] = {}
    if enabled is not None:
        top_level['enabled'] = bool(enabled)
    if 'face_count' in payload:
        top_level['face_count'] = payload['face_count']

    combined = _merge_dicts(TextureGenerationSettings().__dict__, texture_overrides, top_level)
    try:
        return TextureGenerationSettings(**combined)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid texture settings: {exc}") from exc


def parse_user_query(payload: Dict[str, Any]) -> str:
    user_query = payload.get('user_query') or payload.get('prompt')
    if not user_query or not isinstance(user_query, str):
        raise HTTPException(status_code=400, detail="Field 'user_query' (or 'prompt') is required and must be a string.")
    return user_query


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def stream_generation(
    job_id: str,
    user_query: str,
    shape_settings: ShapeGenerationSettings,
    texture_settings: TextureGenerationSettings,
) -> AsyncGenerator[str, None]:
    if generation_service is None or imagen_client is None or model_semaphore is None:
        raise RuntimeError("Generation service is not initialized")

    loop = asyncio.get_running_loop()

    async with model_semaphore:
        try:
            image_payload = await loop.run_in_executor(None, imagen_client.generate, user_query)
            conditioning_image = await loop.run_in_executor(None, generation_service.prepare_image, image_payload)

            mesh = await loop.run_in_executor(
                None,
                generation_service.generate_mesh,
                conditioning_image,
                shape_settings,
            )
            mesh_base64 = await loop.run_in_executor(None, generation_service.to_base64, mesh)
            yield json.dumps({
                "event": "mesh",
                "id": job_id,
                "mesh_base64": mesh_base64,
            }) + "\n"

            if texture_settings.enabled:
                if generation_service.texture_enabled:
                    textured_mesh = await loop.run_in_executor(
                        None,
                        generation_service.generate_textured_mesh,
                        mesh,
                        conditioning_image,
                        texture_settings,
                    )
                    textured_base64 = await loop.run_in_executor(
                        None,
                        generation_service.to_base64,
                        textured_mesh,
                    )
                    yield json.dumps({
                        "event": "textured_mesh",
                        "id": job_id,
                        "mesh_base64": textured_base64,
                    }) + "\n"
                else:
                    yield json.dumps({
                        "event": "texture_skipped",
                        "id": job_id,
                        "reason": "Texture pipeline disabled",
                    }) + "\n"

            yield json.dumps({
                "event": "complete",
                "id": job_id,
            }) + "\n"
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.exception("Generation failed: %s", exc)
            yield json.dumps({
                "event": "error",
                "id": job_id,
                "message": str(exc),
            }) + "\n"


@app.post("/generate")
async def generate(request: Request):
    payload = await request.json()
    user_query = parse_user_query(payload)
    shape_settings = parse_shape_settings(payload)
    texture_settings = parse_texture_settings(payload)

    job_id = payload.get('job_id') or str(uuid.uuid4())
    logger.info("Starting generation job %s", job_id)

    stream = stream_generation(job_id, user_query, shape_settings, texture_settings)
    return StreamingResponse(stream, media_type="application/x-ndjson")


@app.get("/healthz")
async def healthcheck():
    if generation_service is None or imagen_client is None:
        return JSONResponse({"status": "initializing"}, status_code=503)
    return JSONResponse({"status": "ok"}, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--tex_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--disable-texture", action='store_true')
    parser.add_argument("--google-api-key", type=str, default=None)
    args = parser.parse_args()
    logger.info("args: %s", args)

    api_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Google API key not provided. Set GOOGLE_API_KEY or pass --google-api-key.")
        sys.exit(1)

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    imagen_client = GoogleImagenClient(api_key=api_key)
    generation_service = GenerationService(
        model_path=args.model_path,
        tex_model_path=args.tex_model_path,
        device=args.device,
        enable_texture=not args.disable_texture,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
