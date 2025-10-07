# GPU Port Plan & Integration Notes

This document explains how to port the existing CPU mesh cleanup (PyMeshLab) pipeline to a CUDA GPU-backed
implementation using **PyTorch** (and optional **RAPIDS cuGraph**) for maximum performance. We also include a drop-in
module `gpu_postprocessors.py` that provides a replacement for the current `reduce_face` and `remove_floater`
functions.

---

## 1) Goals

- Replace PyMeshLab CPU filters with GPU-accelerated logic:
  - **Decimation** → voxel-grid clustering decimator (fully GPU), with a future hook for a QEM edge-collapse path.
  - **Floater removal** (small disconnected components) → GPU connected components (cuGraph or torch), with CPU fallback.
  - **Degenerate faces** → GPU zero-area filtering.
- Keep the I/O API compatible with the current code (support `trimesh.Trimesh` and simple `(verts, faces)` tuples).
- Make the change minimal and reversible: if CUDA is not available, we gracefully fall back to CPU.

> Why voxel-grid decimation first? It is massively parallel, robust, and very fast on GPU. It preserves large-scale
> shape while removing detail. If you need the highest fidelity, we recommend adding a GPU QEM path later (Kaolin / custom).

---

## 2) What changes in the codebase

### A) Add the new module
Place `gpu_postprocessors.py` (provided) next to your existing `postprocessors.py` (or under `hy3dgen/shapegen/`).

### B) Patch the current functions to use GPU when available

In `postprocessors.py`, update **only** the two public functions that call PyMeshLab filters:

```python
# --- top of file ---
try:
    from .gpu_postprocessors import GPUMeshProcessor
    _GPU_PROC = GPUMeshProcessor(device='cuda')
    _HAS_GPU = True
except Exception:
    _GPU_PROC, _HAS_GPU = None, False

# ... keep the rest of your imports ...
```

Then modify the call sites:

```python
def reduce_face(mesh, max_facenum: int = 200_000):
    # If GPU is ready, use it
    if _HAS_GPU:
        # mesh can be a pymeshlab MeshSet or a trimesh.Trimesh; we convert internally
        tm = pymeshlab2trimesh(mesh) if isinstance(mesh, pymeshlab.MeshSet) else mesh
        tm_out = _GPU_PROC.reduce_face(tm, max_facenum=max_facenum)
        # Optionally convert back to PyMeshLab MeshSet if the rest of the code expects it:
        return trimesh2pymeshlab(tm_out) if isinstance(mesh, pymeshlab.MeshSet) else tm_out
    # --- CPU fallback (existing behavior) ---
    # (Your current PyMeshLab filter call remains here)
    # mesh.apply_filter("meshing_decimation_quadric_edge_collapse", ...)
    # return mesh
```

```python
def remove_floater(mesh):
    if _HAS_GPU:
        tm = pymeshlab2trimesh(mesh) if isinstance(mesh, pymeshlab.MeshSet) else mesh
        tm_out = _GPU_PROC.remove_floater(tm)
        return trimesh2pymeshlab(tm_out) if isinstance(mesh, pymeshlab.MeshSet) else tm_out
    # CPU fallback (existing PyMeshLab selection+remove filters)
```

> If your file structure differs, adjust the relative import accordingly. Putting the GPU module in the same package
> makes the relative import `from .gpu_postprocessors import GPUMeshProcessor` work everywhere in that package.

---

## 3) Dependency plan

**Required**:
- `torch` (>= 1.13; 2.x recommended). CUDA build preferred. 

**Optional (recommended for fastest floater removal on very large meshes)**:
- `cudf`, `cugraph` (RAPIDS) — provides very fast GPU connected components.
  - See RAPIDS install matrix for your CUDA/Python combo.
- `trimesh` — convenient for I/O. If your pipeline already uses it, nothing else to do.
- `kaolin` — future path for a high-fidelity QEM decimator (not required for this PR).

No runtime dependency on PyMeshLab for GPU mode.

---

## 4) How the GPU version works

**Degenerate face removal**: compute triangle areas on the GPU and drop faces below `eps` (`1e-12` by default).

**Floater removal**:
- Build the undirected edge set from faces.
- Compute connected components using (in order of preference):
  1. **cuGraph** (GPU; fastest).
  2. **Torch `scatter_reduce_` label-propagation** (GPU; requires torch >= 2.2).
  3. **CPU union-find** (fallback).
- Drop components smaller than `min_component_ratio * total_faces` (default 0.5%).

**Decimation (voxel grid)**:
- Quantize vertices into a 3D grid (`voxel_size` chosen from mesh size and target face count).
- Merge each cell to its centroid (parallelized with `index_add_` on GPU).
- Remap faces → drop duplicates / degenerates → final cleanup pass.

> The voxel size adapts iteratively to approximately hit `max_facenum` within a few iterations.

---

## 5) Performance & quality expectations

- **Speed**: On modern NVIDIA GPUs, voxel decimation can reduce tens of millions of faces in seconds-to-minutes
  depending on I/O and memory bandwidth. Connected components with cuGraph scales linearly with edge count.
- **Quality**: Voxel-grid preserves overall shape but not fine detail or sharp features. In most cleanup use-cases,
  the results are close to CPU QEM decimation with a large reduction ratio. If sharp-feature preservation is critical,
  prioritize a future QEM GPU path and use a smaller reduction ratio for voxel decimation.

---

## 6) Testing plan

1. **Functional**: Compare face counts and component counts before/after on a set of meshes; verify no degenerate faces remain.
2. **Equivalence (sanity)**: Run the original CPU pipeline and the GPU pipeline on a sample and check bounding box and
   surface area are within acceptable deviation (< 2–5% depending on reduction ratio).
3. **Stress**: Large meshes (≥ 10M faces) to check OOM behavior; ensure iterative voxel sizing degrades gracefully.
4. **Determinism**: Voxel pipeline is deterministic given identical inputs.
5. **Fallbacks**: When `torch.cuda.is_available()` is false, ensure CPU path still works unchanged.

---

## 7) Example usage

```python
from gpu_postprocessors import GPUMeshProcessor
import trimesh

proc = GPUMeshProcessor(device='cuda', prefer='voxel', min_component_ratio=0.005)

tm = trimesh.load('input.obj', process=False)
tm2 = proc.reduce_face(tm, max_facenum=200_000)  # decimate + cleanup
tm2.export('output.obj')

tm3 = proc.remove_floater(tm)  # only floater removal
tm3.export('nofloaters.obj')
```

---

## 8) Future: QEM edge-collapse on GPU

- Wire a second decimator: `prefer='qem'`.
- Implement edge prioritization by quadric error metric:
  - Precompute per-vertex quadrics in parallel (GPU).
  - Maintain a bucketed priority structure per edge (approximate with fixed rounds).
  - Collapse in parallel with conflict resolution (graph coloring / independent set selection).
- Alternatively, integrate an existing Kaolin decimator when available.

---

## 9) Rollout

- Land this as a new module and a behind-the-scenes switch. Keep CPU as fallback for safety.
- Try on a representative batch of meshes. If metrics and visual QA pass, make CUDA the default in production
  for hosts with compatible GPUs.
