# Copyright (c) 2025.
# GPU-accelerated mesh post-processing utilities for decimation and component filtering.
#
# Design goals:
# - Run heavy geometry ops on the GPU using PyTorch (CUDA).
# - Keep I/O compatibility with existing CPU pipeline (trimesh / PyMeshLab) where possible.
# - Provide fast, robust "voxel grid" decimation and degenerate/floaters cleanup.
# - Optional accelerators: RAPIDS cuGraph (for connected components), Kaolin (future).
#
# This module does not require Kaolin. If Kaolin or cuGraph are available, we use them.
#
# Usage (drop-in example):
#   from gpu_postprocessors import GPUMeshProcessor
#   proc = GPUMeshProcessor(device='cuda')   # falls back to 'cpu' if CUDA unavailable
#   tm = trimesh.load('input.obj')
#   tm_out = proc.reduce_face(tm, max_facenum=200_000)
#   tm_out.export('output.obj')
#
# Notes:
# - We intentionally import heavy deps lazily inside functions to avoid import-time failures.
# - We expect torch >= 1.13.0. For best performance and GPU connected-component labeling,
#   torch >= 2.2.0 (scatter_reduce_) or cuGraph is recommended.
# - Degenerate removal (zero/near-zero area) and decimation run fully on the GPU.
# - Connected components (floaters removal) runs on:
#     1) cuGraph (fastest, GPU), if available
#     2) torch scatter_reduce (GPU), if available
#     3) CPU numpy BFS (fallback) if neither available.
#
# Limitations:
# - Voxel-grid decimation is very fast and robust but not as high-fidelity as QEM edge collapse.
#   It is recommended for "cleanup" and large-ratio reductions. For highest fidelity, integrate
#   a GPU QEM decimator (e.g., via Kaolin once supported) as a future upgrade.
#
# Author: (Your team)
# License: MIT (for this file)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Iterable

import math

# Minimal torch typing hints without importing at module import time
_TORCH_AVAILABLE = True
try:
    import torch
except Exception as _e:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

_TRIMESH_AVAILABLE = True
try:
    import trimesh
except Exception as _e:
    _TRIMESH_AVAILABLE = False
    trimesh = None  # type: ignore

# Optional GPU graph library for connected components
_CUGRAPH_AVAILABLE = True
try:
    import cudf
    import cugraph
except Exception as _e:
    _CUGRAPH_AVAILABLE = False
    cudf = None   # type: ignore
    cugraph = None  # type: ignore


@dataclass
class GPUMesh:
    """Lightweight mesh container with GPU tensors."""
    verts: 'torch.Tensor'  # (V, 3), float32
    faces: 'torch.Tensor'  # (F, 3), int64

    def to(self, device: Union[str, torch.device]) -> 'GPUMesh':
        return GPUMesh(self.verts.to(device), self.faces.to(device))

    @property
    def device(self) -> 'torch.device':
        return self.verts.device

    def clone(self) -> 'GPUMesh':
        return GPUMesh(self.verts.clone(), self.faces.clone())

    def num_verts(self) -> int:
        return int(self.verts.shape[0])

    def num_faces(self) -> int:
        return int(self.faces.shape[0])


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for gpu_postprocessors but is not importable.")


def _as_mesh(obj: Union['trimesh.Trimesh', GPUMesh, Tuple, dict]) -> GPUMesh:
    """Convert various inputs into a GPUMesh (on current default device)."""
    _require_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(obj, GPUMesh):
        return obj.to(device)
    if _TRIMESH_AVAILABLE and isinstance(obj, trimesh.Trimesh):
        v = torch.tensor(obj.vertices, dtype=torch.float32, device=device)
        f = torch.tensor(obj.faces, dtype=torch.int64, device=device) if obj.faces is not None else \
            torch.empty((0, 3), dtype=torch.int64, device=device)
        return GPUMesh(v, f)
    # Tuple / dict of (verts, faces)
    if isinstance(obj, tuple) and len(obj) == 2:
        v, f = obj
        v = torch.as_tensor(v, dtype=torch.float32, device=device)
        f = torch.as_tensor(f, dtype=torch.int64, device=device)
        return GPUMesh(v, f)
    if isinstance(obj, dict) and 'verts' in obj and 'faces' in obj:
        v = torch.as_tensor(obj['verts'], dtype=torch.float32, device=device)
        f = torch.as_tensor(obj['faces'], dtype=torch.int64, device=device)
        return GPUMesh(v, f)
    raise TypeError("Unsupported mesh input type. Pass trimesh.Trimesh, GPUMesh, or (verts, faces).")


def _to_trimesh(mesh: GPUMesh) -> 'trimesh.Trimesh':
    if not _TRIMESH_AVAILABLE:
        raise RuntimeError("trimesh is required to export to trimesh.Trimesh.")
    v = mesh.verts.detach().cpu().numpy()
    f = mesh.faces.detach().cpu().numpy().astype('int64') if mesh.faces.numel() > 0 else None
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def _triangle_areas(verts: 'torch.Tensor', faces: 'torch.Tensor', eps: float = 0.0) -> 'torch.Tensor':
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    area = 0.5 * torch.linalg.vector_norm(torch.cross(v1 - v0, v2 - v0, dim=1), ord=2, dim=1)
    if eps > 0:
        area = torch.clamp_min(area, eps)
    return area


def _remove_degenerate_faces(mesh: GPUMesh, eps: float = 1e-12) -> GPUMesh:
    if mesh.num_faces() == 0:
        return mesh
    areas = _triangle_areas(mesh.verts, mesh.faces)
    keep = areas > eps
    if not torch.all(keep):
        faces = mesh.faces[keep]
        return GPUMesh(mesh.verts, faces)
    return mesh


def _voxel_grid_quantize(verts: 'torch.Tensor', voxel_size: float, bb_min: 'torch.Tensor') -> 'torch.Tensor':
    """Return integer voxel indices for each vertex."""
    q = torch.floor((verts - bb_min) / voxel_size).to(torch.int64)
    return q


def _linearize_indices(q: 'torch.Tensor', res: 'torch.Tensor') -> 'torch.Tensor':
    """Map 3D voxel coords to a single linear index."""
    return (q[:, 0] * res[1] + q[:, 1]) * res[2] + q[:, 2]


def _remap_and_dedup_faces(faces: 'torch.Tensor', vmap: 'torch.Tensor') -> 'torch.Tensor':
    """Remap faces using new vertex ids; drop degenerate/duplicate faces."""
    f = vmap[faces]  # (F,3)
    # Remove faces with duplicate vertices
    good = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
    f = f[good]
    # Canonicalize winding for dedup (sort vertex indices within each face)
    f_sorted, _ = torch.sort(f, dim=1)
    # Unique rows (requires torch>=1.13)
    if f_sorted.numel() == 0:
        return f_sorted
    f_unique, inv = torch.unique(f_sorted, dim=0, return_inverse=True)
    # Recover original winding by mapping unique back to first occurrence (we keep sorted to avoid duplicates).
    return f_unique


def _estimate_voxel_size_to_hit_faces(mesh: GPUMesh, target_faces: int) -> float:
    """Heuristic: derive voxel size from target face count.

    For a manifold-ish triangular mesh, F ≈ ~2V. We target V' ~ target_faces / 2,
    and voxel size from cubic grid resolution to produce ~V' occupied cells.
    """
    V = max(1, mesh.num_verts())
    F = mesh.num_faces()
    verts = mesh.verts
    bb_min = verts.min(dim=0).values
    bb_max = verts.max(dim=0).values
    extent = (bb_max - bb_min).max().item()  # longest dimension
    # Target vertex clusters
    target_V = max(10, int(target_faces / 2))
    # Current grid cells ~ V (each vertex unique). Scale side length proportional to (V / target_V)^(1/3)
    scale = (V / float(target_V)) ** (1.0 / 3.0)
    # Avoid zero or inf
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    # Base cell = mean edge length proxy
    # Compute a rough average edge length proxy from bbox
    base_cell = extent / (V ** (1.0 / 3.0) + 1e-6)
    voxel_size = max(1e-9, base_cell * scale)
    return float(voxel_size)


def _voxel_decimate(mesh: GPUMesh, target_faces: int, max_iters: int = 6) -> GPUMesh:
    """Fast voxel-grid decimation on GPU to approximately reach target face count."""
    if mesh.num_faces() == 0 or mesh.num_faces() <= target_faces:
        return mesh

    verts = mesh.verts
    faces = mesh.faces
    device = verts.device

    # Compute bbox
    bb_min = verts.min(dim=0).values
    bb_max = verts.max(dim=0).values
    extent = bb_max - bb_min
    longest = torch.max(extent).item()

    voxel_size = _estimate_voxel_size_to_hit_faces(mesh, target_faces)

    # Iteratively adjust voxel size to approach target_faces
    for it in range(max_iters):
        q = _voxel_grid_quantize(verts, voxel_size, bb_min)  # (V,3) int64
        # Map each voxel cell to a unique id
        # Build resolution estimate to linearize; we do not need exact res, so we use hashing approach
        # Use a stable hash via unique on rows
        q_unique, inv = torch.unique(q, dim=0, return_inverse=True)
        # Compute cluster centers
        counts = torch.bincount(inv, minlength=q_unique.shape[0]).to(verts.dtype)
        sums = torch.zeros((q_unique.shape[0], 3), dtype=verts.dtype, device=device)
        sums.index_add_(0, inv, verts)
        centers = sums / counts.unsqueeze(1)

        # Remap faces
        f_remap = _remap_and_dedup_faces(faces, inv)
        approx_F = int(f_remap.shape[0])

        # Adjust voxel size adaptively
        if approx_F <= target_faces or it == max_iters - 1:
            # Commit
            verts2 = centers
            faces2 = f_remap
            mesh2 = GPUMesh(verts2, faces2)
            mesh2 = _remove_degenerate_faces(mesh2)
            return mesh2
        else:
            # Not enough reduction; increase voxel size (merge more)
            # Multiply by factor proportional to (approx_F / target_faces)^(1/3)
            factor = (approx_F / float(target_faces)) ** (1.0 / 3.0)
            factor = max(1.1, min(4.0, factor))
            voxel_size *= factor

    # Fallback return
    return _remove_degenerate_faces(GPUMesh(centers, f_remap))


def _build_edge_index_from_faces(faces: 'torch.Tensor') -> 'torch.Tensor':
    """Return undirected edges (E,2) as int64 from faces (F,3)."""
    i0 = faces[:, [0, 1]]
    i1 = faces[:, [1, 2]]
    i2 = faces[:, [2, 0]]
    e = torch.cat([i0, i1, i2], dim=0)
    # Sort each edge's endpoints to canonicalize undirected edges
    e_sorted, _ = torch.sort(e, dim=1)
    # Unique edges
    e_unique = torch.unique(e_sorted, dim=0)
    return e_unique


def _connected_components_cugraph(num_verts: int, edges: 'torch.Tensor') -> 'torch.Tensor':
    """Connected components using cuGraph if available; returns per-vertex labels on GPU."""
    if not _CUGRAPH_AVAILABLE:
        raise RuntimeError("cuGraph not available")
    # Build cudf edge dataframe (src, dst)
    # Ensure GPU tensors
    src = edges[:, 0].detach().cpu().numpy()
    dst = edges[:, 1].detach().cpu().numpy()
    df = cudf.DataFrame({'src': src, 'dst': dst})
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst', renumber=False)
    comps = cugraph.connected_components(G)
    # comps columns: ['vertex', 'component']
    # Reorder to contiguous labels per vertex id
    comps = comps.sort_values('vertex')
    labels = torch.as_tensor(comps['component'].to_numpy(), dtype=torch.int64, device='cuda')
    # If num_verts > labels.size(0) (isolated vertices), pad
    if labels.numel() < num_verts:
        out = torch.arange(num_verts, device='cuda', dtype=torch.int64)
        out[:labels.numel()] = labels
        labels = out
    return labels


def _connected_components_torch(num_verts: int, edges: 'torch.Tensor') -> Optional['torch.Tensor']:
    """Connected components via label propagation using torch.scatter_reduce (GPU).

    Returns labels tensor if supported by current torch; otherwise None.
    """
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        return None
    device = edges.device
    # Ensure both directions
    e = torch.cat([edges, edges[:, [1, 0]]], dim=0)
    src = e[:, 0].contiguous()
    dst = e[:, 1].contiguous()

    labels = torch.arange(num_verts, device=device, dtype=torch.int64)
    changed = True
    iters = 0
    max_iters = 100
    while changed and iters < max_iters:
        iters += 1
        # For each dst, compute amin of src labels
        tmp = torch.full((num_verts,), torch.iinfo(torch.int64).max, device=device, dtype=torch.int64)
        tmp.scatter_reduce_(0, dst, labels[src], reduce='amin', include_self=False)
        new_labels = torch.minimum(labels, tmp)
        changed = bool(torch.any(new_labels != labels))
        labels = new_labels
    return labels


def _connected_components_cpu(num_verts: int, edges: 'torch.Tensor') -> 'torch.Tensor':
    """CPU union-find for connected components. Returns torch.LongTensor on CPU."""
    import numpy as np

    e = edges.detach().cpu().numpy()
    parent = np.arange(num_verts, dtype=np.int64)
    rank = np.zeros(num_verts, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in e:
        union(int(a), int(b))

    # Path compress all
    for i in range(num_verts):
        parent[i] = find(i)

    labels = torch.as_tensor(parent, dtype=torch.int64, device='cpu')
    return labels


def _remove_small_components(mesh: GPUMesh, min_face_ratio: float = 0.005) -> GPUMesh:
    """Remove small disconnected components by face count ratio (similar to PyMeshLab's behavior)."""
    V = mesh.num_verts()
    F = mesh.num_faces()
    if F == 0:
        return mesh
    device = mesh.device
    edges = _build_edge_index_from_faces(mesh.faces)

    # 1) Try cuGraph
    labels = None
    if device.type == 'cuda' and _CUGRAPH_AVAILABLE:
        try:
            labels = _connected_components_cugraph(V, edges)
        except Exception:
            labels = None

    # 2) Try torch scatter_reduce
    if labels is None and device.type == 'cuda':
        labels = _connected_components_torch(V, edges)

    # 3) CPU fallback
    if labels is None:
        labels_cpu = _connected_components_cpu(V, edges)
        # Map faces using CPU labels then move back to device
        f_labels = labels_cpu[mesh.faces.cpu()]
        comp_per_face_cpu = torch.min(f_labels, dim=1).values
        comp_per_face = comp_per_face_cpu.to(device)
    else:
        # Face belongs to the min label among its 3 vertices
        comp_per_face = torch.min(labels[mesh.faces], dim=1).values

    # Count faces per component
    unique_c, counts = torch.unique(comp_per_face, return_counts=True)
    # Keep components with sufficient size
    thr = max(1, int(F * min_face_ratio))
    big_components = unique_c[counts >= thr]
    keep = torch.isin(comp_per_face, big_components)
    faces2 = mesh.faces[keep]
    return GPUMesh(mesh.verts, faces2)


class GPUMeshProcessor:
    """Mesh cleanup and decimation on CUDA (or CPU fallback)."""

    def __init__(
        self,
        device: Optional[Union[str, 'torch.device']] = None,
        prefer: str = 'voxel',         # 'voxel' | 'qem' (future)
        remove_degenerate_eps: float = 1e-12,
        min_component_ratio: float = 0.005,
    ) -> None:
        _require_torch()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.prefer = prefer
        self.remove_degenerate_eps = float(remove_degenerate_eps)
        self.min_component_ratio = float(min_component_ratio)

    # ---------- Public API ----------

    def reduce_face(self, mesh_in: Union['trimesh.Trimesh', GPUMesh, Tuple, dict], max_facenum: int = 200_000) -> 'trimesh.Trimesh':
        """GPU decimation and cleanup to target face count (~)."""
        mesh = _as_mesh(mesh_in).to(self.device)

        # 1) Degenerate cleanup
        mesh = _remove_degenerate_faces(mesh, eps=self.remove_degenerate_eps)

        # 2) Floater removal (small components)
        mesh = _remove_small_components(mesh, min_face_ratio=self.min_component_ratio)

        # 3) Decimation
        if self.prefer == 'voxel':
            mesh = _voxel_decimate(mesh, target_faces=max_facenum)
        else:
            # Placeholder for future GPU QEM
            mesh = _voxel_decimate(mesh, target_faces=max_facenum)

        # 4) Post-checks (degenerate once again after decimation)
        mesh = _remove_degenerate_faces(mesh, eps=self.remove_degenerate_eps)

        # Return as trimesh for compatibility
        return _to_trimesh(mesh)

    def remove_floater(self, mesh_in: Union['trimesh.Trimesh', GPUMesh, Tuple, dict]) -> 'trimesh.Trimesh':
        mesh = _as_mesh(mesh_in).to(self.device)
        mesh = _remove_small_components(mesh, min_face_ratio=self.min_component_ratio)
        return _to_trimesh(mesh)

    # Convenience helpers
    def to_trimesh(self, mesh_in: Union['trimesh.Trimesh', GPUMesh, Tuple, dict]) -> 'trimesh.Trimesh':
        return _to_trimesh(_as_mesh(mesh_in).to(self.device))

    def to_gpu(self, mesh_in: Union['trimesh.Trimesh', GPUMesh, Tuple, dict]) -> GPUMesh:
        return _as_mesh(mesh_in).to(self.device)
