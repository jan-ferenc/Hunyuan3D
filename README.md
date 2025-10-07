# Hunyuan3D Local API Server

This repository packages a thin HTTP API around the Hunyuan3D generators so you can serve text-to-3D, image-to-3D, and texture workflows from your own machine. The focus of this guide is to help you stand up the development server quickly.

## Requirements

- Python 3.10+
- A virtual environment located at `.venv` with the project dependencies installed (see `requirements.txt`).
- GPU drivers and CUDA/cuDNN that match the versions required by the bundled models (optional for CPU-only experimentation but heavily recommended).

If you still need to install the Python dependencies, activate the virtual environment and run:

```bash
pip install -r requirements.txt
```

The requirements pin `rembg[gpu]` with `onnxruntime-gpu`, so background removal uses CUDA by default on deployment servers with NVIDIA GPUs.

## Launching the API Server

Follow these steps from your terminal in order:

1. `cd Hunyuan3D`
2. `source .venv/bin/activate`
3. `python api_server.py --host 0.0.0.0 --port 8081`

The server binds to all interfaces on port `8081`. When the startup log shows that Uvicorn is running, you can send HTTP requests to `http://<your-host>:8081`.

## Basic Health Check

With the server running, probe the root endpoint in a new terminal tab:

```bash
curl http://localhost:8081/health
```

You should receive a JSON response confirming the service is ready.

## Next Steps

- Review `api_server.py` to explore the available routes and expected payloads.
- Consult `hy3dgen/services/imagen_client.py` for examples of how requests are assembled inside the project.
- Integrate the API with your own client or automation once the health check passes.
