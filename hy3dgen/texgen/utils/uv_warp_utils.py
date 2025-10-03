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

import hashlib
import os
from collections import OrderedDict

import numpy as np
import trimesh
import xatlas

UV_CACHE: "OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
UV_CACHE_SIZE = int(os.environ.get('HY3DGEN_UV_CACHE_SIZE', 8))


def _hash_mesh(mesh: trimesh.Trimesh) -> str:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    digest = hashlib.sha1(vertices.tobytes())
    digest.update(faces.tobytes())
    return digest.hexdigest()


def mesh_uv_wrap(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if len(mesh.faces) > 500000000:
        raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")

    metadata = getattr(mesh, 'metadata', {}) or {}
    hunyuan_meta = metadata.get('hunyuan3d') if isinstance(metadata, dict) else None
    existing_uv = getattr(getattr(mesh, 'visual', None), 'uv', None)
    if existing_uv is not None and isinstance(hunyuan_meta, dict) and hunyuan_meta.get('uv_wrapped'):
        return mesh

    cache_key = None
    if isinstance(hunyuan_meta, dict):
        cache_key = hunyuan_meta.get('uv_cache_key')

    if cache_key is None:
        try:
            cache_key = _hash_mesh(mesh)
        except Exception:
            cache_key = None

    if cache_key is not None:
        cached = UV_CACHE.get(cache_key)
        if cached is not None:
            cached_vertices, cached_faces, cached_uvs = cached
            mesh.vertices = cached_vertices.copy()
            mesh.faces = cached_faces.copy()
            mesh.visual.uv = cached_uvs.copy()
            hunyuan_meta = metadata.setdefault('hunyuan3d', {})
            hunyuan_meta['uv_wrapped'] = True
            hunyuan_meta['uv_cache_key'] = cache_key
            UV_CACHE.move_to_end(cache_key)
            mesh.metadata = metadata
            return mesh

    kwargs = {}
    chart_options_ctor = getattr(xatlas, 'ChartOptions', None)
    if callable(chart_options_ctor):
        try:
            chart_options = chart_options_ctor()
            for attr, value in (
                ('use_spatial_hash', True),
                ('max_iterations', 1),
                ('max_chart_area', 0.0),
            ):
                if hasattr(chart_options, attr):
                    setattr(chart_options, attr, value)
            kwargs['chart_options'] = chart_options
        except Exception:
            kwargs = {}

    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces, **kwargs)

    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    if not isinstance(metadata, dict):
        metadata = {}
    hunyuan_meta = metadata.setdefault('hunyuan3d', {})
    hunyuan_meta['uv_wrapped'] = True
    if cache_key is not None and UV_CACHE_SIZE > 0:
        UV_CACHE[cache_key] = (mesh.vertices.copy(), mesh.faces.copy(), mesh.visual.uv.copy())
        while len(UV_CACHE) > UV_CACHE_SIZE:
            UV_CACHE.popitem(last=False)
        hunyuan_meta['uv_cache_key'] = cache_key
    mesh.metadata = metadata

    return mesh
