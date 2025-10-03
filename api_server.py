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
from collections import OrderedDict
from functools import partial
import json
import logging
import logging.handlers
import os
import sys
import uuid
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from hy3dgen.services.generation_service import (
    GenerationService,
    ShapeGenerationSettings,
    TextureGenerationSettings,
)
from hy3dgen.services.imagen_client import GoogleImagenClient

LOGDIR = '.'

handler = None


def build_logger(logger_name, logger_filename, level: str = "INFO"):
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

    resolved_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(resolved_level)

    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)
                item.setLevel(resolved_level)

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

_api_log_level = os.environ.get("API_LOG_LEVEL", "INFO")
logger = build_logger("controller", f"{SAVE_DIR}/controller.log", level=_api_log_level)

if _api_log_level.upper() == "DEBUG":
    logging.getLogger("hy3dgen").setLevel(logging.DEBUG)
    logging.getLogger("hy3dgen.texgen").setLevel(logging.DEBUG)
    logging.getLogger("hy3dgen.services").setLevel(logging.DEBUG)

model_semaphore: Optional[asyncio.Semaphore] = None
generation_service: Optional[GenerationService] = None
imagen_client: Optional[GoogleImagenClient] = None

VIEWER_HTML_PATH = Path(__file__).resolve().parent / "assets" / "viewer.html"

MESH_CACHE_LIMIT = int(os.environ.get("HY3DGEN_MESH_CACHE_LIMIT", "24"))
mesh_cache: "OrderedDict[str, Dict[str, bytes]]" = OrderedDict()
mesh_cache_lock = asyncio.Lock()
_background_exports: "set[asyncio.Task]" = set()


def _merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for data in dicts:
        for key, value in data.items():
            if value is not None:
                merged[key] = value
    return merged


def _load_viewer_html() -> str:
    if VIEWER_HTML_PATH.is_file():
        try:
            return VIEWER_HTML_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read viewer HTML: %s", exc)
    return "<html><body><p>Viewer UI not found. Ensure assets/viewer.html exists.</p></body></html>"


async def _cache_mesh_bytes(cache_key: str, variant: str, data: bytes) -> None:
    async with mesh_cache_lock:
        entry = mesh_cache.get(cache_key)
        if entry is None:
            entry = {}
        entry[variant] = data
        mesh_cache[cache_key] = entry
        mesh_cache.move_to_end(cache_key)
        while len(mesh_cache) > MESH_CACHE_LIMIT:
            mesh_cache.popitem(last=False)


async def _get_cached_mesh_bytes(cache_key: str, variant: str) -> Optional[bytes]:
    async with mesh_cache_lock:
        entry = mesh_cache.get(cache_key)
        if entry is None:
            return None
        mesh_cache.move_to_end(cache_key)
        return entry.get(variant)


def _schedule_background_task(coro: Awaitable[None]) -> None:
    task = asyncio.create_task(coro)
    _background_exports.add(task)
    task.add_done_callback(lambda finished: _background_exports.discard(finished))


async def _export_mesh_background(mesh, run_id: str, *, textured: bool) -> None:
    if generation_service is None:
        return
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None,
            partial(
                generation_service.export_mesh,
                mesh,
                run_id,
                textured=textured,
            ),
        )
    except Exception:  # pragma: no cover - best effort persistence
        logger.exception(
            "Background export failed for job %s (textured=%s)",
            run_id,
            textured,
        )


def _build_mesh_payload(*, mesh_path: Path, cache_key: str, variant: str) -> Dict[str, str]:
    payload: Dict[str, str] = {"mesh_path": str(mesh_path)}
    payload["mesh_url"] = f"/jobs/{cache_key}/{variant}.glb"
    if generation_service is None:
        return payload
    try:
        relative_path = mesh_path.resolve().relative_to(generation_service.output_root)
    except Exception:
        relative_path = mesh_path.name
    payload["mesh_relative_path"] = str(relative_path)
    return payload


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
    for key in (
        'delight_steps',
        'multiview_steps',
        'reuse_delighting',
        'delight_cache_size',
    ):
        if key in payload:
            top_level[key] = payload[key]
    if 'texture_seed' in payload:
        top_level['seed'] = payload['texture_seed']

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


def parse_benchmark_runs(payload: Dict[str, Any]) -> int:
    raw = payload.get('benchmark_runs')
    if raw is None:
        return 1
    try:
        runs = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="'benchmark_runs' must be an integer") from exc
    if runs < 1:
        raise HTTPException(status_code=400, detail="'benchmark_runs' must be at least 1")
    if runs > 50:
        raise HTTPException(status_code=400, detail="'benchmark_runs' must be <= 50")
    return runs


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_load_viewer_html())


@app.get("/viewer", response_class=HTMLResponse)
async def viewer():
    return HTMLResponse(_load_viewer_html())


@app.get("/jobs/{cache_key}/{variant}.glb")
async def serve_job_asset(cache_key: str, variant: str):
    normalized_variant = "textured" if variant.lower().startswith("textured") else "mesh"
    cached = await _get_cached_mesh_bytes(cache_key, normalized_variant)
    if cached is not None:
        return Response(content=cached, media_type="model/gltf-binary")

    if generation_service is not None:
        try:
            path = generation_service.resolve_export_path(
                cache_key,
                textured=(normalized_variant == "textured"),
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Failed to resolve export path for %s: %s", cache_key, exc)
        else:
            if path.exists():
                return FileResponse(path, media_type="model/gltf-binary")

    raise HTTPException(status_code=404, detail="Mesh asset not found")


async def stream_generation(
    job_id: str,
    user_query: str,
    shape_settings: ShapeGenerationSettings,
    texture_settings: TextureGenerationSettings,
    benchmark_runs: int,
) -> AsyncGenerator[str, None]:
    if generation_service is None or imagen_client is None or model_semaphore is None:
        raise RuntimeError("Generation service is not initialized")

    loop = asyncio.get_running_loop()
    timing_records = []

    async with model_semaphore:
        try:
            for run_idx in range(benchmark_runs):
                run_id = job_id if benchmark_runs == 1 else f"{job_id}-run{run_idx + 1}"
                run_start = time.perf_counter()

                image_payload = await loop.run_in_executor(None, imagen_client.generate, user_query)
                image_ready = time.perf_counter()

                image_seconds = image_ready - run_start
                try:
                    pil_image = image_payload.as_pil()
                    fmt = (pil_image.format or "PNG") if pil_image is not None else "PNG"
                    mime_type = Image.MIME.get(fmt.upper(), "image/png")
                except Exception:
                    mime_type = "image/png"

                image_event: Dict[str, Any] = {
                    "event": "image",
                    "id": run_id,
                    "image_base64": image_payload.as_base64(),
                    "mime_type": mime_type,
                    "timings": {
                        "image_seconds": image_seconds,
                    },
                }
                if benchmark_runs > 1:
                    image_event["run"] = run_idx + 1
                    image_event["runs_total"] = benchmark_runs
                    image_event["parent_job"] = job_id
                yield json.dumps(image_event) + "\n"

                conditioning_image = await loop.run_in_executor(None, generation_service.prepare_image, image_payload)

                mesh = await loop.run_in_executor(
                    None,
                    generation_service.generate_mesh,
                    conditioning_image,
                    shape_settings,
                )
                mesh = await loop.run_in_executor(
                    None,
                    partial(
                        generation_service.standardize_mesh,
                        mesh,
                        texture_settings=texture_settings,
                    ),
                )
                cache_key = GenerationService.sanitize_job_id(run_id)
                mesh_path = generation_service.resolve_export_path(run_id, textured=False)
                mesh_bytes = await loop.run_in_executor(
                    None,
                    partial(
                        generation_service.mesh_to_glb_bytes,
                        mesh,
                        textured=False,
                    ),
                )
                await _cache_mesh_bytes(cache_key, "mesh", mesh_bytes)
                mesh_ready = time.perf_counter()

                mesh_stage_seconds = max(mesh_ready - image_ready, 0.0)
                run_timing: Dict[str, Any] = {
                    "run": run_idx + 1,
                    "image_seconds": image_seconds,
                    "mesh_seconds": mesh_stage_seconds,
                    "textured_seconds": None,
                    "total_seconds": mesh_ready - run_start,
                }

                mesh_payload = _build_mesh_payload(
                    mesh_path=mesh_path,
                    cache_key=cache_key,
                    variant="mesh",
                )
                mesh_event = {
                    "event": "mesh",
                    "id": run_id,
                    **mesh_payload,
                    "timings": {
                        "image_seconds": run_timing["image_seconds"],
                        "mesh_seconds": run_timing["mesh_seconds"],
                        "total_seconds": run_timing["total_seconds"],
                    },
                }
                if benchmark_runs > 1:
                    mesh_event["run"] = run_idx + 1
                    mesh_event["runs_total"] = benchmark_runs
                    mesh_event["parent_job"] = job_id
                yield json.dumps(mesh_event) + "\n"

                mesh_copy_for_disk = mesh.copy()
                _schedule_background_task(
                    _export_mesh_background(
                        mesh_copy_for_disk,
                        run_id,
                        textured=False,
                    )
                )

                if texture_settings.enabled:
                    if generation_service.texture_enabled:
                        textured_mesh = await loop.run_in_executor(
                            None,
                            generation_service.generate_textured_mesh,
                            mesh,
                            conditioning_image,
                            texture_settings,
                        )
                        textured_path = generation_service.resolve_export_path(run_id, textured=True)
                        textured_bytes = await loop.run_in_executor(
                            None,
                            partial(
                                generation_service.mesh_to_glb_bytes,
                                textured_mesh,
                                textured=True,
                            ),
                        )
                        await _cache_mesh_bytes(cache_key, "textured", textured_bytes)
                        textured_ready = time.perf_counter()
                        run_timing["textured_seconds"] = max(textured_ready - mesh_ready, 0.0)
                        run_timing["total_seconds"] = textured_ready - run_start

                        textured_payload = _build_mesh_payload(
                            mesh_path=textured_path,
                            cache_key=cache_key,
                            variant="textured",
                        )
                        textured_event = {
                            "event": "textured_mesh",
                            "id": run_id,
                            **textured_payload,
                            "timings": {
                                "image_seconds": run_timing["image_seconds"],
                                "mesh_seconds": run_timing["mesh_seconds"],
                                "textured_seconds": run_timing["textured_seconds"],
                                "total_seconds": run_timing["total_seconds"],
                            },
                        }
                        if benchmark_runs > 1:
                            textured_event["run"] = run_idx + 1
                            textured_event["runs_total"] = benchmark_runs
                            textured_event["parent_job"] = job_id
                        yield json.dumps(textured_event) + "\n"

                        textured_copy_for_disk = textured_mesh.copy()
                        _schedule_background_task(
                            _export_mesh_background(
                                textured_copy_for_disk,
                                run_id,
                                textured=True,
                            )
                        )
                    else:
                        skipped_event = {
                            "event": "texture_skipped",
                            "id": run_id,
                            "reason": "Texture pipeline disabled",
                        }
                        if benchmark_runs > 1:
                            skipped_event["run"] = run_idx + 1
                            skipped_event["runs_total"] = benchmark_runs
                            skipped_event["parent_job"] = job_id
                        yield json.dumps(skipped_event) + "\n"

                timing_records.append(run_timing.copy())

                if benchmark_runs > 1:
                    yield json.dumps({
                        "event": "run_complete",
                        "id": run_id,
                        "run": run_idx + 1,
                        "runs_total": benchmark_runs,
                        "parent_job": job_id,
                        "timings": {
                            "image_seconds": run_timing["image_seconds"],
                            "mesh_seconds": run_timing["mesh_seconds"],
                            "textured_seconds": run_timing["textured_seconds"],
                            "total_seconds": run_timing["total_seconds"],
                        },
                    }) + "\n"

            complete_event: Dict[str, Any] = {
                "event": "complete",
                "id": job_id,
            }
            if timing_records:
                complete_event["timings"] = timing_records[-1]
            if benchmark_runs > 1:
                averages: Dict[str, float] = {}
                for key in ("image_seconds", "mesh_seconds", "textured_seconds", "total_seconds"):
                    values = [t[key] for t in timing_records if t.get(key) is not None]
                    if values:
                        averages[key] = sum(values) / len(values)
                complete_event["benchmark"] = {
                    "runs": benchmark_runs,
                    "averages": averages,
                    "per_run": timing_records,
                }

            yield json.dumps(complete_event) + "\n"
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
    benchmark_runs = parse_benchmark_runs(payload)

    job_id = payload.get('job_id') or str(uuid.uuid4())
    logger.info("Starting generation job %s (benchmark_runs=%s)", job_id, benchmark_runs)

    stream = stream_generation(
        job_id,
        user_query,
        shape_settings,
        texture_settings,
        benchmark_runs,
    )
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

    try:
        app.mount(
            "/generated",
            StaticFiles(directory=str(generation_service.output_root)),
            name="generated",
        )
    except Exception as exc:
        logger.warning("Unable to mount generated assets directory: %s", exc)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
