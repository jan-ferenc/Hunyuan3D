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
import logging
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from .camera_utils import (
    transform_pos,
    get_mv_matrix,
    get_orthographic_projection_matrix,
    get_perspective_projection_matrix,
)
from .mesh_processor import meshVerticeInpaint
from .mesh_utils import load_mesh, save_mesh

# Add a module-level logger
logger = logging.getLogger(__name__)


def stride_from_shape(shape):
    stride = [1]
    for x in reversed(shape[1:]):
        stride.append(stride[-1] * x)
    return list(reversed(stride))


def scatter_add_nd_with_count(input, count, indices, values, weights=None):
    # input: [..., C], D dimension + C channel
    # count: [..., 1], D dimension
    # indices: [N, D], long
    # values: [N, C]

    D = indices.shape[-1]
    C = input.shape[-1]
    size = input.shape[:-1]
    stride = stride_from_shape(size)

    assert len(size) == D

    input = input.view(-1, C)  # [HW, C]
    count = count.view(-1, 1)

    flatten_indices = (indices * torch.tensor(stride,
                                              dtype=torch.long, device=indices.device)).sum(-1)  # [N]

    if weights is None:
        weights = torch.ones_like(values[..., :1])

    input.scatter_add_(0, flatten_indices.unsqueeze(1).repeat(1, C), values)
    count.scatter_add_(0, flatten_indices.unsqueeze(1), weights)

    return input.view(*size, C), count.view(*size, 1)


def linear_grid_put_2d(H, W, coords, values, return_count=False):
    # coords: [N, 2], float in [0, 1]
    # values: [N, C]

    C = values.shape[-1]

    indices = coords * torch.tensor(
        [H - 1, W - 1], dtype=torch.float32, device=coords.device
    )
    indices_00 = indices.floor().long()  # [N, 2]
    indices_00[:, 0].clamp_(0, H - 2)
    indices_00[:, 1].clamp_(0, W - 2)
    indices_01 = indices_00 + torch.tensor(
        [0, 1], dtype=torch.long, device=indices.device
    )
    indices_10 = indices_00 + torch.tensor(
        [1, 0], dtype=torch.long, device=indices.device
    )
    indices_11 = indices_00 + torch.tensor(
        [1, 1], dtype=torch.long, device=indices.device
    )

    h = indices[..., 0] - indices_00[..., 0].float()
    w = indices[..., 1] - indices_00[..., 1].float()
    w_00 = (1 - h) * (1 - w)
    w_01 = (1 - h) * w
    w_10 = h * (1 - w)
    w_11 = h * w

    result = torch.zeros(H, W, C, device=values.device,
                         dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device,
                        dtype=values.dtype)  # [H, W, 1]
    weights = torch.ones_like(values[..., :1])  # [N, 1]

    result, count = scatter_add_nd_with_count(
        result, count, indices_00, values * w_00.unsqueeze(1), weights * w_00.unsqueeze(1))
    result, count = scatter_add_nd_with_count(
        result, count, indices_01, values * w_01.unsqueeze(1), weights * w_01.unsqueeze(1))
    result, count = scatter_add_nd_with_count(
        result, count, indices_10, values * w_10.unsqueeze(1), weights * w_10.unsqueeze(1))
    result, count = scatter_add_nd_with_count(
        result, count, indices_11, values * w_11.unsqueeze(1), weights * w_11.unsqueeze(1))

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


class MeshRender():
    def __init__(
        self,
        camera_distance=1.45, camera_type='orth',
        default_resolution=1024, texture_size=1024,
        use_antialias=True, max_mip_level=None, filter_mode='linear',
        bake_mode='linear', raster_mode='cr', device='cuda'):

        self.device = self._resolve_device(device)

        self.set_default_render_resolution(default_resolution)
        self.set_default_texture_resolution(texture_size)

        self.camera_distance = camera_distance
        self.use_antialias = use_antialias
        self.max_mip_level = max_mip_level
        self.filter_mode = filter_mode

        self.bake_angle_thres = 75
        self.bake_unreliable_kernel_size = int(
            (2 / 512) * max(self.default_resolution[0], self.default_resolution[1]))
        self.bake_mode = bake_mode

        self.raster_mode = raster_mode
        if self.raster_mode == 'cr':
            try:
                import custom_rasterizer as cr
            except ImportError as exc:
                raise ImportError(
                    "custom_rasterizer extension not found. Install it via "
                    "'pip install ./hy3dgen/texgen/custom_rasterizer' or run with --disable-texture."
                ) from exc
            self.raster = cr
        else:
            raise f'No raster named {self.raster_mode}'

        if camera_type == 'orth':
            self.ortho_scale = 1.2
            self.camera_proj_mat = get_orthographic_projection_matrix(
                left=-self.ortho_scale * 0.5, right=self.ortho_scale * 0.5,
                bottom=-self.ortho_scale * 0.5, top=self.ortho_scale * 0.5,
                near=0.1, far=100
            )
        elif camera_type == 'perspective':
            self.camera_proj_mat = get_perspective_projection_matrix(
                49.13, self.default_resolution[1] / self.default_resolution[0],
                0.01, 100.0
            )
        else:
            raise f'No camera type {camera_type}'

        self._mesh_hash = None
        self._seam_cache = {}
        self._vertex_cache = {}

    def _resolve_device(self, device_spec):
        if device_spec is None:
            device_spec = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            resolved = torch.device(device_spec)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Unsupported device specification: {device_spec!r}") from exc
        if resolved.type == 'cuda' and not torch.cuda.is_available():
            logger.info("MeshRender requested CUDA device '%s' but CUDA is unavailable; switching to CPU.", device_spec)
            resolved = torch.device('cpu')
        return resolved

    def raster_rasterize(self, pos, tri, resolution, ranges=None, grad_db=True):

        if self.raster_mode == 'cr':
            rast_out_db = None
            if pos.dim() == 2:
                pos = pos.unsqueeze(0)
            findices, barycentric = self.raster.rasterize(pos, tri, resolution)
            rast_out = torch.cat((barycentric, findices.unsqueeze(-1)), dim=-1)
            rast_out = rast_out.unsqueeze(0)
        else:
            raise f'No raster named {self.raster_mode}'

        return rast_out, rast_out_db

    def raster_interpolate(self, uv, rast_out, uv_idx, rast_db=None, diff_attrs=None):

        if self.raster_mode == 'cr':
            textd = None
            barycentric = rast_out[0, ..., :-1]
            findices = rast_out[0, ..., -1]
            if uv.dim() == 2:
                uv = uv.unsqueeze(0)
            textc = self.raster.interpolate(uv, findices, barycentric, uv_idx)
        else:
            raise f'No raster named {self.raster_mode}'

        return textc, textd

    def raster_texture(self, tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode='auto',
                       boundary_mode='wrap', max_mip_level=None):

        if self.raster_mode == 'cr':
            raise f'Texture is not implemented in cr'
        else:
            raise f'No raster named {self.raster_mode}'

        return color

    def raster_antialias(self, color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):

        if self.raster_mode == 'cr':
            # Antialias has not been supported yet
            color = color
        else:
            raise f'No raster named {self.raster_mode}'

        return color

    def load_mesh(
        self,
        mesh,
        scale_factor=1.15,
        auto_center=True,
    ):
        vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data = load_mesh(mesh)
        self.mesh_copy = mesh
        self.set_mesh(vtx_pos, pos_idx,
                      vtx_uv=vtx_uv, uv_idx=uv_idx,
                      scale_factor=scale_factor, auto_center=auto_center
                      )
        if texture_data is not None:
            self.set_texture(texture_data)

    def save_mesh(self):
        texture_data = self.get_texture()
        texture_data = Image.fromarray((texture_data * 255).astype(np.uint8))
        return save_mesh(self.mesh_copy, texture_data)

    def set_mesh(
        self,
        vtx_pos, pos_idx,
        vtx_uv=None, uv_idx=None,
        scale_factor=1.15, auto_center=True
    ):

        self.vtx_pos = torch.from_numpy(vtx_pos).to(self.device).float()
        self.pos_idx = torch.from_numpy(pos_idx).to(self.device).to(torch.int)
        if (vtx_uv is not None) and (uv_idx is not None):
            self.vtx_uv = torch.from_numpy(vtx_uv).to(self.device).float()
            self.uv_idx = torch.from_numpy(uv_idx).to(self.device).to(torch.int)
        else:
            self.vtx_uv = None
            self.uv_idx = None

        self.vtx_pos[:, [0, 1]] = -self.vtx_pos[:, [0, 1]]
        self.vtx_pos[:, [1, 2]] = self.vtx_pos[:, [2, 1]]
        if (vtx_uv is not None) and (uv_idx is not None):
            self.vtx_uv[:, 1] = 1.0 - self.vtx_uv[:, 1]

        if auto_center:
            max_bb = (self.vtx_pos - 0).max(0)[0]
            min_bb = (self.vtx_pos - 0).min(0)[0]
            center = (max_bb + min_bb) / 2
            scale = torch.norm(self.vtx_pos - center, dim=1).max() * 2.0
            self.vtx_pos = (self.vtx_pos - center) * \
                           (scale_factor / float(scale))
            self.scale_factor = scale_factor
        self._mesh_hash = None
        self._seam_cache.clear()
        self._vertex_cache.clear()

    def set_texture(self, tex):
        """
        Store texture data as a float32 tensor on the renderer device.
        Accepts numpy arrays, PIL images, or torch tensors in either [0, 255] or [0, 1].
        """

        target_h, target_w = self.texture_size

        def _to_tensor(input_tex):
            if isinstance(input_tex, torch.Tensor):
                tensor = input_tex.detach()
                if tensor.dim() == 4:
                    if tensor.shape[0] != 1:
                        raise ValueError("set_texture expects tensor with shape [H,W,C] or [1,H,W,C]")
                    tensor = tensor[0]
                if tensor.dim() != 3:
                    raise ValueError("set_texture expects tensor with 3 channels in last dimension")
                tensor = tensor.to(device=self.device)
                if tensor.dtype in (torch.float16, torch.float32, torch.float64):
                    tensor = tensor.to(dtype=torch.float32)
                    if tensor.max().item() > 1.0005:
                        tensor = tensor / 255.0
                else:
                    tensor = tensor.to(dtype=torch.float32) / 255.0
                return tensor.clamp_(0.0, 1.0)
            if isinstance(input_tex, Image.Image):
                array = np.asarray(input_tex)
            else:
                array = np.asarray(input_tex)
            tensor = torch.from_numpy(array).to(device=self.device)
            if tensor.dtype in (torch.float16, torch.float32, torch.float64):
                tensor = tensor.to(dtype=torch.float32)
                if tensor.max().item() > 1.0005:
                    tensor = tensor / 255.0
            else:
                tensor = tensor.to(dtype=torch.float32) / 255.0
            return tensor.clamp_(0.0, 1.0)

        texture_tensor = _to_tensor(tex)
        if texture_tensor.dim() != 3 or texture_tensor.shape[-1] not in (3, 4):
            raise ValueError("set_texture expects an image tensor with shape [H, W, 3/4]")
        if texture_tensor.shape[-1] == 4:
            texture_tensor = texture_tensor[..., :3]

        if (texture_tensor.shape[0], texture_tensor.shape[1]) != (target_h, target_w):
            texture_tensor = texture_tensor.permute(2, 0, 1).unsqueeze(0)
            texture_tensor = F.interpolate(
                texture_tensor,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            )
            texture_tensor = texture_tensor.squeeze(0).permute(1, 2, 0).contiguous()
        else:
            texture_tensor = texture_tensor.contiguous()

        self.tex = texture_tensor

    def set_default_render_resolution(self, default_resolution):
        if isinstance(default_resolution, int):
            default_resolution = (default_resolution, default_resolution)
        self.default_resolution = default_resolution

    def set_default_texture_resolution(self, texture_size):
        if isinstance(texture_size, int):
            texture_size = (texture_size, texture_size)
        self.texture_size = texture_size

    def get_mesh(self):
        vtx_pos = self.vtx_pos.cpu().numpy()
        pos_idx = self.pos_idx.cpu().numpy()
        vtx_uv = self.vtx_uv.cpu().numpy()
        uv_idx = self.uv_idx.cpu().numpy()

        # 坐标变换的逆变换
        vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]]
        vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]

        vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]
        return vtx_pos, pos_idx, vtx_uv, uv_idx

    def _mesh_cache_key(self):
        if self._mesh_hash is None:
            hasher = hashlib.sha1()
            for tensor in (self.vtx_pos, self.vtx_uv, self.pos_idx, self.uv_idx):
                arr = tensor.detach().cpu().numpy()
                hasher.update(arr.tobytes())
            self._mesh_hash = hasher.hexdigest()
        return self._mesh_hash

    def _get_vertex_cache(self, tex_h, tex_w, device):
        if not hasattr(self, "_vertex_cache"):
            self._vertex_cache = {}
        mesh_key = (self._mesh_cache_key(), int(tex_h), int(tex_w))
        entry = self._vertex_cache.get(mesh_key)
        base_created = False
        if entry is None:
            uv_cpu = self.vtx_uv.detach().cpu()
            uv_idx_cpu = self.uv_idx.detach().cpu().view(-1).long()
            pos_idx_cpu = self.pos_idx.detach().cpu().view(-1).long()
            xs_cpu = torch.clamp((uv_cpu[:, 0] * (tex_w - 1)).round().long(), 0, tex_w - 1)
            ys_cpu = torch.clamp(((1.0 - uv_cpu[:, 1]) * (tex_h - 1)).round().long(), 0, tex_h - 1)
            entry = {
                "uv_x_cpu": xs_cpu,
                "uv_y_cpu": ys_cpu,
                "uv_idx_flat_cpu": uv_idx_cpu,
                "pos_idx_flat_cpu": pos_idx_cpu,
                "num_uv": int(uv_cpu.shape[0]),
                "num_vertices": int(self.vtx_pos.shape[0]),
                "device_cache": {},
            }
            self._vertex_cache[mesh_key] = entry
            base_created = True
        device_key = (device.type, device.index if device.index is not None else -1)
        cache = entry["device_cache"].get(device_key)
        device_created = False
        if cache is None:
            cache = {
                "uv_x": entry["uv_x_cpu"].to(device=device, non_blocking=True),
                "uv_y": entry["uv_y_cpu"].to(device=device, non_blocking=True),
                "uv_idx_flat": entry["uv_idx_flat_cpu"].to(device=device, non_blocking=True),
                "pos_idx_flat": entry["pos_idx_flat_cpu"].to(device=device, non_blocking=True),
            }
            entry["device_cache"][device_key] = cache
            device_created = True
        cache_info = {
            "vertex_cache_base_created": base_created,
            "vertex_cache_device_created": device_created,
        }
        return cache, entry, cache_info

    def _get_seam_lookup(self, tex_h, tex_w, device, mesh_cache=None):
        if not hasattr(self, "_seam_cache"):
            self._seam_cache = {}
        mesh_key = (self._mesh_cache_key(), int(tex_h), int(tex_w))
        entry = self._seam_cache.get(mesh_key)
        base_created = False
        if entry is None:
            if mesh_cache is None:
                mesh_cache = self.get_mesh()
            vtx_pos, pos_idx, vtx_uv, uv_idx = mesh_cache
            pos_to_uvs = defaultdict(set)
            tri_count = pos_idx.shape[0]
            for f in range(tri_count):
                for k in range(3):
                    pos_to_uvs[int(pos_idx[f, k])].add(int(uv_idx[f, k]))
            tex_h_m1 = tex_h - 1
            tex_w_m1 = tex_w - 1
            group_coords = []
            for uv_indices in pos_to_uvs.values():
                if len(uv_indices) <= 1:
                    continue
                coords = []
                for uidx in uv_indices:
                    x = int(np.clip(round(vtx_uv[uidx, 0] * tex_w_m1), 0, tex_w_m1))
                    y = int(np.clip(round(vtx_uv[uidx, 1] * tex_h_m1), 0, tex_h_m1))
                    coords.append((y, x))
                coords = list(dict.fromkeys(coords))
                if len(coords) > 1:
                    group_coords.append(coords)
            y_list, x_list, group_ids = [], [], []
            for gid, coords in enumerate(group_coords):
                for (y, x) in coords:
                    y_list.append(y)
                    x_list.append(x)
                    group_ids.append(gid)
            if len(group_ids) > 0:
                group_ids_cpu = torch.tensor(group_ids, dtype=torch.long)
                counts_cpu = torch.bincount(group_ids_cpu, minlength=len(group_coords)).to(torch.float32)
            else:
                group_ids_cpu = torch.zeros(0, dtype=torch.long)
                counts_cpu = torch.zeros(0, dtype=torch.float32)
            entry = {
                "y_cpu": torch.tensor(y_list, dtype=torch.long),
                "x_cpu": torch.tensor(x_list, dtype=torch.long),
                "group_ids_cpu": group_ids_cpu,
                "group_counts_cpu": counts_cpu,
                "num_groups": len(group_coords),
                "device_cache": {},
            }
            self._seam_cache[mesh_key] = entry
            base_created = True
        device_key = (device.type, device.index if device.index is not None else -1)
        cache = entry["device_cache"].get(device_key)
        device_created = False
        if cache is None:
            cache = {
                "y": entry["y_cpu"].to(device=device, non_blocking=True),
                "x": entry["x_cpu"].to(device=device, non_blocking=True),
                "group_ids": entry["group_ids_cpu"].to(device=device, non_blocking=True),
                "group_counts": entry["group_counts_cpu"].to(device=device, non_blocking=True),
                "kernel": torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32),
            }
            entry["device_cache"][device_key] = cache
            device_created = True
        cache_info = {
            "seam_cache_base_created": base_created,
            "seam_cache_device_created": device_created,
        }
        return cache, entry, cache_info

    def _gpu_vertex_inpaint(self, texture_tensor, mask_bool, device):
        """
        Perform lightweight vertex-based UV inpaint on the GPU to prefill seam texels.
        texture_tensor: float32 [H,W,C] on `device`
        mask_bool: bool [H,W] indicating known texels
        Returns updated texture tensor, mask bool, and stats dict.
        """
        tex_h, tex_w, channels = texture_tensor.shape
        cache, meta, cache_info = self._get_vertex_cache(tex_h, tex_w, device)
        uv_x = cache["uv_x"]
        uv_y = cache["uv_y"]
        uv_idx_flat = cache["uv_idx_flat"]
        pos_idx_flat = cache["pos_idx_flat"]
        num_uv = meta["num_uv"]
        num_vertices = meta["num_vertices"]

        start = time.perf_counter()
        known_uv = mask_bool[uv_y, uv_x]  # [num_uv]
        colors_uv = texture_tensor[uv_y, uv_x, :]  # [num_uv,C]
        flat_known = known_uv[uv_idx_flat]

        stats = {
            "vertex_gpu_updates": 0,
            "vertex_gpu_newfills": 0,
            "vertex_gpu_time_ms": 0.0,
        }
        stats.update(cache_info)

        if not flat_known.any():
            stats["vertex_gpu_time_ms"] = (time.perf_counter() - start) * 1000.0
            return texture_tensor, mask_bool, stats

        vertex_counts = torch.zeros(num_vertices, device=device, dtype=torch.float32)
        vertex_counts.index_add_(0, pos_idx_flat, flat_known.float())

        vertex_color_sum = torch.zeros((num_vertices, channels), device=device, dtype=torch.float32)
        weighted_colors = colors_uv[uv_idx_flat] * flat_known.float().unsqueeze(-1)
        vertex_color_sum.index_add_(0, pos_idx_flat, weighted_colors)

        vertex_has = vertex_counts > 0
        if not vertex_has.any():
            stats["vertex_gpu_time_ms"] = (time.perf_counter() - start) * 1000.0
            return texture_tensor, mask_bool, stats

        vertex_avg = torch.zeros_like(vertex_color_sum)
        vertex_avg[vertex_has] = vertex_color_sum[vertex_has] / torch.clamp(vertex_counts[vertex_has].unsqueeze(-1), min=1e-6)

        valid_vertex = vertex_has[pos_idx_flat]
        if not valid_vertex.any():
            stats["vertex_gpu_time_ms"] = (time.perf_counter() - start) * 1000.0
            return texture_tensor, mask_bool, stats

        uv_counts = torch.zeros(num_uv, device=device, dtype=torch.float32)
        uv_counts.index_add_(0, uv_idx_flat, valid_vertex.float())

        uv_color_sum = torch.zeros((num_uv, channels), device=device, dtype=torch.float32)
        uv_color_sum.index_add_(
            0,
            uv_idx_flat,
            vertex_avg[pos_idx_flat] * valid_vertex.float().unsqueeze(-1),
        )

        uv_has_color = uv_counts > 0
        if not uv_has_color.any():
            stats["vertex_gpu_time_ms"] = (time.perf_counter() - start) * 1000.0
            return texture_tensor, mask_bool, stats

        update_idx = torch.nonzero(uv_has_color, as_tuple=False).squeeze(1)
        if update_idx.numel() == 0:
            stats["vertex_gpu_time_ms"] = (time.perf_counter() - start) * 1000.0
            return texture_tensor, mask_bool, stats

        new_colors = uv_color_sum[update_idx] / torch.clamp(uv_counts[update_idx].unsqueeze(-1), min=1e-6)
        new_colors = new_colors.clamp_(0.0, 1.0)

        updated_texture = texture_tensor.clone()
        updated_mask = mask_bool.clone()
        ys = uv_y[update_idx]
        xs = uv_x[update_idx]
        updated_texture[ys, xs, :] = new_colors
        prev_known = known_uv[update_idx]
        updated_mask[ys, xs] = True

        stats["vertex_gpu_updates"] = int(update_idx.numel())
        stats["vertex_gpu_newfills"] = int((~prev_known).sum().item())
        stats["vertex_gpu_time_ms"] = (time.perf_counter() - start) * 1000.0
        return updated_texture, updated_mask, stats

    def get_texture(self):
        return self.tex.cpu().numpy()

    def to(self, device):
        self.device = self._resolve_device(device)

        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(self.device))

    def color_rgb_to_srgb(self, image):
        if isinstance(image, Image.Image):
            image_rgb = torch.tesnor(
                np.array(image) /
                255.0).float().to(
                self.device)
        elif isinstance(image, np.ndarray):
            image_rgb = torch.tensor(image).float()
        else:
            image_rgb = image.to(self.device)

        image_srgb = torch.where(
            image_rgb <= 0.0031308,
            12.92 * image_rgb,
            1.055 * torch.pow(image_rgb, 1 / 2.4) - 0.055
        )

        if isinstance(image, Image.Image):
            image_srgb = Image.fromarray(
                (image_srgb.cpu().numpy() *
                 255).astype(
                    np.uint8))
        elif isinstance(image, np.ndarray):
            image_srgb = image_srgb.cpu().numpy()
        else:
            image_srgb = image_srgb.to(image.device)

        return image_srgb

    def _render(
        self,
        glctx,
        mvp,
        pos,
        pos_idx,
        uv,
        uv_idx,
        tex,
        resolution,
        max_mip_level,
        keep_alpha,
        filter_mode
    ):
        pos_clip = transform_pos(mvp, pos)
        if isinstance(resolution, (int, float)):
            resolution = [resolution, resolution]
        rast_out, rast_out_db = self.raster_rasterize(
            glctx, pos_clip, pos_idx, resolution=resolution)

        tex = tex.contiguous()
        if filter_mode == 'linear-mipmap-linear':
            texc, texd = self.raster_interpolate(
                uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
            color = self.raster_texture(
                tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        else:
            texc, _ = self.raster_interpolate(uv[None, ...], rast_out, uv_idx)
            color = self.raster_texture(tex[None, ...], texc, filter_mode=filter_mode)

        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
        color = color * visible_mask  # Mask out background.
        if self.use_antialias:
            color = self.raster_antialias(color, rast_out, pos_clip, pos_idx)

        if keep_alpha:
            color = torch.cat([color, visible_mask], dim=-1)
        return color[0, ...]

    def render(
        self,
        elev,
        azim,
        camera_distance=None,
        center=None,
        resolution=None,
        tex=None,
        keep_alpha=True,
        bgcolor=None,
        filter_mode=None,
        return_type='th'
    ):

        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(
            elev=elev,
            azim=azim,
            camera_distance=self.camera_distance if camera_distance is None else camera_distance,
            center=center)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        if tex is not None:
            if isinstance(tex, Image.Image):
                tex = torch.tensor(np.array(tex) / 255.0)
            elif isinstance(tex, np.ndarray):
                tex = torch.tensor(tex)
            if tex.dim() == 2:
                tex = tex.unsqueeze(-1)
            tex = tex.float().to(self.device)
        image = self._render(r_mvp, self.vtx_pos, self.pos_idx, self.vtx_uv, self.uv_idx,
                             self.tex if tex is None else tex,
                             self.default_resolution if resolution is None else resolution,
                             self.max_mip_level, True, filter_mode if filter_mode else self.filter_mode)
        mask = (image[..., [-1]] == 1).float()
        if bgcolor is None:
            bgcolor = [0 for _ in range(image.shape[-1] - 1)]
        image = image * mask + (1 - mask) * \
                torch.tensor(bgcolor + [0]).to(self.device)
        if keep_alpha == False:
            image = image[..., :-1]
        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.squeeze(-1).cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        return image

    def render_normal(
        self,
        elev,
        azim,
        camera_distance=None,
        center=None,
        resolution=None,
        bg_color=[1, 1, 1],
        use_abs_coor=False,
        normalize_rgb=True,
        return_type='th'
    ):

        pos_camera, pos_clip = self.get_pos_from_mvp(elev, azim, camera_distance, center)
        if resolution is None:
            resolution = self.default_resolution
        if isinstance(resolution, (int, float)):
            resolution = [resolution, resolution]
        rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)

        if use_abs_coor:
            mesh_triangles = self.vtx_pos[self.pos_idx[:, :3], :]
        else:
            pos_camera = pos_camera[:, :3] / pos_camera[:, 3:4]
            mesh_triangles = pos_camera[self.pos_idx[:, :3], :]
        face_normals = F.normalize(
            torch.cross(mesh_triangles[:,
                        1,
                        :] - mesh_triangles[:,
                             0,
                             :],
                        mesh_triangles[:,
                        2,
                        :] - mesh_triangles[:,
                             0,
                             :],
                        dim=-1),
            dim=-1)

        faces = self.pos_idx[:, :3]
        vertex_normals = torch.zeros(
            (self.vtx_pos.shape[0], 3), device=self.device, dtype=face_normals.dtype
        )
        expanded_normals = face_normals.unsqueeze(1).expand(-1, faces.shape[1], -1)
        vertex_normals.index_add_(0, faces.reshape(-1), expanded_normals.reshape(-1, 3))
        vertex_normals = F.normalize(vertex_normals, dim=-1, eps=1e-8)

        # Interpolate normal values across the rasterized pixels
        normal, _ = self.raster_interpolate(
            vertex_normals[None, ...], rast_out, self.pos_idx)

        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
        normal = normal * visible_mask + \
                 torch.tensor(bg_color, dtype=torch.float32, device=self.device) * (1 -
                                                                                    visible_mask)  

        if normalize_rgb:
            normal = (normal + 1) * 0.5
        if self.use_antialias:
            normal = self.raster_antialias(normal, rast_out, pos_clip, self.pos_idx)

        image = normal[0, ...]
        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))

        return image

    def convert_normal_map(self, image):
        # blue is front, red is left, green is top
        if isinstance(image, Image.Image):
            image = np.array(image)
        mask = (image == [255, 255, 255]).all(axis=-1)

        image = (image / 255.0) * 2.0 - 1.0

        image[..., [1]] = -image[..., [1]]
        image[..., [1, 2]] = image[..., [2, 1]]
        image[..., [0]] = -image[..., [0]]

        image = (image + 1.0) * 0.5

        image = (image * 255).astype(np.uint8)
        image[mask] = [127, 127, 255]

        return Image.fromarray(image)

    def get_pos_from_mvp(self, elev, azim, camera_distance, center):
        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(
            elev=elev,
            azim=azim,
            camera_distance=self.camera_distance if camera_distance is None else camera_distance,
            center=center)

        pos_camera = transform_pos(r_mv, self.vtx_pos, keepdim=True)
        pos_clip = transform_pos(proj, pos_camera)

        return pos_camera, pos_clip

    def render_depth(
        self,
        elev,
        azim,
        camera_distance=None,
        center=None,
        resolution=None,
        return_type='th'
    ):
        pos_camera, pos_clip = self.get_pos_from_mvp(elev, azim, camera_distance, center)

        if resolution is None:
            resolution = self.default_resolution
        if isinstance(resolution, (int, float)):
            resolution = [resolution, resolution]
        rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)

        pos_camera = pos_camera[:, :3] / pos_camera[:, 3:4]
        tex_depth = pos_camera[:, 2].reshape(1, -1, 1).contiguous()

        # Interpolate depth values across the rasterized pixels
        depth, _ = self.raster_interpolate(tex_depth, rast_out, self.pos_idx)

        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
        depth_max, depth_min = depth[visible_mask >
                                     0].max(), depth[visible_mask > 0].min()
        depth = (depth - depth_min) / (depth_max - depth_min)

        depth = depth * visible_mask  # Mask out background.
        if self.use_antialias:
            depth = self.raster_antialias(depth, rast_out, pos_clip, self.pos_idx)

        image = depth[0, ...]
        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.squeeze(-1).cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        return image

    def render_position(self, elev, azim, camera_distance=None, center=None,
                        resolution=None, bg_color=[1, 1, 1], return_type='th'):
        pos_camera, pos_clip = self.get_pos_from_mvp(elev, azim, camera_distance, center)
        if resolution is None:
            resolution = self.default_resolution
        if isinstance(resolution, (int, float)):
            resolution = [resolution, resolution]
        rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)

        tex_position = 0.5 - self.vtx_pos[:, :3] / self.scale_factor
        tex_position = tex_position.contiguous()

        # Interpolate depth values across the rasterized pixels
        position, _ = self.raster_interpolate(
            tex_position[None, ...], rast_out, self.pos_idx)

        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)

        position = position * visible_mask + \
                   torch.tensor(bg_color, dtype=torch.float32, device=self.device) * (1 -
                                                                                      visible_mask) 
        if self.use_antialias:
            position = self.raster_antialias(position, rast_out, pos_clip, self.pos_idx)

        image = position[0, ...]

        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.squeeze(-1).cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        return image

    def render_uvpos(self, return_type='th'):
        image = self.uv_feature_map(self.vtx_pos * 0.5 + 0.5)
        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        return image

    def uv_feature_map(self, vert_feat, bg=None):
        vtx_uv = self.vtx_uv * 2 - 1.0
        vtx_uv = torch.cat(
            [vtx_uv, torch.zeros_like(self.vtx_uv)], dim=1).unsqueeze(0)
        vtx_uv[..., -1] = 1
        uv_idx = self.uv_idx
        rast_out, rast_out_db = self.raster_rasterize(
            vtx_uv, uv_idx, resolution=self.texture_size)
        feat_map, _ = self.raster_interpolate(vert_feat[None, ...], rast_out, uv_idx)
        feat_map = feat_map[0, ...]
        if bg is not None:
            visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ...]
            feat_map[visible_mask == 0] = bg
        return feat_map

    def render_sketch_from_geometry(self, normal_image, depth_image):
        normal_image_np = normal_image.cpu().numpy()
        depth_image_np = depth_image.cpu().numpy()

        normal_image_np = (normal_image_np * 255).astype(np.uint8)
        depth_image_np = (depth_image_np * 255).astype(np.uint8)
        normal_image_np = cv2.cvtColor(normal_image_np, cv2.COLOR_RGB2GRAY)

        normal_edges = cv2.Canny(normal_image_np, 80, 150)
        depth_edges = cv2.Canny(depth_image_np, 30, 80)

        combined_edges = np.maximum(normal_edges, depth_edges)

        sketch_image = torch.from_numpy(combined_edges).to(
            normal_image.device).float() / 255.0
        sketch_image = sketch_image.unsqueeze(-1)

        return sketch_image

    def render_sketch_from_depth(self, depth_image):
        depth_image_np = depth_image.cpu().numpy()
        depth_image_np = (depth_image_np * 255).astype(np.uint8)
        depth_edges = cv2.Canny(depth_image_np, 30, 80)
        combined_edges = depth_edges
        sketch_image = torch.from_numpy(combined_edges).to(
            depth_image.device).float() / 255.0
        sketch_image = sketch_image.unsqueeze(-1)
        return sketch_image

    def back_project(self, image, elev, azim,
                     camera_distance=None, center=None, method=None):
        if isinstance(image, Image.Image):
            image = torch.tensor(np.array(image) / 255.0)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image)
        if image.dim() == 2:
            image = image.unsqueeze(-1)
        image = image.float().to(self.device)
        resolution = image.shape[:2]
        channel = image.shape[-1]
        texture = torch.zeros(self.texture_size + (channel,)).to(self.device)
        cos_map = torch.zeros(self.texture_size + (1,)).to(self.device)

        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(
            elev=elev,
            azim=azim,
            camera_distance=self.camera_distance if camera_distance is None else camera_distance,
            center=center)
        pos_camera = transform_pos(r_mv, self.vtx_pos, keepdim=True)
        pos_clip = transform_pos(proj, pos_camera)
        pos_camera = pos_camera[:, :3] / pos_camera[:, 3:4]
        v0 = pos_camera[self.pos_idx[:, 0], :]
        v1 = pos_camera[self.pos_idx[:, 1], :]
        v2 = pos_camera[self.pos_idx[:, 2], :]
        face_normals = F.normalize(
            torch.cross(
                v1 - v0,
                v2 - v0,
                dim=-1),
            dim=-1)
        vertex_normals = trimesh.geometry.mean_vertex_normals(vertex_count=self.vtx_pos.shape[0],
                                                              faces=self.pos_idx.cpu(),
                                                              face_normals=face_normals.cpu(), )
        vertex_normals = torch.from_numpy(
            vertex_normals).float().to(self.device).contiguous()
        tex_depth = pos_camera[:, 2].reshape(1, -1, 1).contiguous()
        rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)
        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ...]

        normal, _ = self.raster_interpolate(
            vertex_normals[None, ...], rast_out, self.pos_idx)
        normal = normal[0, ...]
        uv, _ = self.raster_interpolate(self.vtx_uv[None, ...], rast_out, self.uv_idx)
        depth, _ = self.raster_interpolate(tex_depth, rast_out, self.pos_idx)
        depth = depth[0, ...]

        depth_max, depth_min = depth[visible_mask >
                                     0].max(), depth[visible_mask > 0].min()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        depth_image = depth_normalized * visible_mask  # Mask out background.

        sketch_image = self.render_sketch_from_depth(depth_image)

        lookat = torch.tensor([[0, 0, -1]], device=self.device)
        cos_image = torch.nn.functional.cosine_similarity(
            lookat, normal.view(-1, 3))
        cos_image = cos_image.view(normal.shape[0], normal.shape[1], 1)

        cos_thres = np.cos(self.bake_angle_thres / 180 * np.pi)
        cos_image[cos_image < cos_thres] = 0

        # shrink
        kernel_size = self.bake_unreliable_kernel_size * 2 + 1
        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size), dtype=torch.float32).to(
            sketch_image.device)

        visible_mask = visible_mask.permute(2, 0, 1).unsqueeze(0).float()
        visible_mask = F.conv2d(
            1.0 - visible_mask,
            kernel,
            padding=kernel_size // 2)
        visible_mask = 1.0 - (visible_mask > 0).float()  # 二值化
        visible_mask = visible_mask.squeeze(0).permute(1, 2, 0)

        sketch_image = sketch_image.permute(2, 0, 1).unsqueeze(0)
        sketch_image = F.conv2d(sketch_image, kernel, padding=kernel_size // 2)
        sketch_image = (sketch_image > 0).float()  # 二值化
        sketch_image = sketch_image.squeeze(0).permute(1, 2, 0)
        visible_mask = visible_mask * (sketch_image < 0.5)

        cos_image[visible_mask == 0] = 0

        method = self.bake_mode if method is None else method

        if method == 'linear':
            proj_mask = (visible_mask != 0).view(-1)
            uv = uv.squeeze(0).contiguous().view(-1, 2)[proj_mask]
            image = image.squeeze(0).contiguous().view(-1, channel)[proj_mask]
            cos_image = cos_image.contiguous().view(-1, 1)[proj_mask]
            sketch_image = sketch_image.contiguous().view(-1, 1)[proj_mask]

            texture = linear_grid_put_2d(
                self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], image)
            cos_map = linear_grid_put_2d(
                self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], cos_image)
            boundary_map = linear_grid_put_2d(
                self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], sketch_image)
        else:
            raise f'No bake mode {method}'

        return texture, cos_map, boundary_map

    def bake_texture(self, colors, elevs, azims,
                     camera_distance=None, center=None, exp=6, weights=None):
        for i in range(len(colors)):
            if isinstance(colors[i], Image.Image):
                colors[i] = torch.tensor(
                    np.array(
                        colors[i]) / 255.0,
                    device=self.device).float()
        if weights is None:
            weights = [1.0 for _ in range(colors)]
        textures = []
        cos_maps = []
        for color, elev, azim, weight in zip(colors, elevs, azims, weights):
            texture, cos_map, _ = self.back_project(
                color, elev, azim, camera_distance, center)
            cos_map = weight * (cos_map ** exp)
            textures.append(texture)
            cos_maps.append(cos_map)

        texture_merge, trust_map_merge = self.fast_bake_texture(
            textures, cos_maps)
        return texture_merge, trust_map_merge

    @torch.no_grad()
    def fast_bake_texture(self, textures, cos_maps):

        channel = textures[0].shape[-1]
        texture_merge = torch.zeros(
            self.texture_size + (channel,)).to(self.device)
        trust_map_merge = torch.zeros(self.texture_size + (1,)).to(self.device)
        for texture, cos_map in zip(textures, cos_maps):
            view_sum = (cos_map > 0).sum()
            painted_sum = ((cos_map > 0) * (trust_map_merge > 0)).sum()
            if painted_sum / view_sum > 0.99:
                continue
            texture_merge += texture * cos_map
            trust_map_merge += cos_map
        texture_merge = texture_merge / torch.clamp(trust_map_merge, min=1E-8)

        return texture_merge, trust_map_merge > 1E-8

    def uv_inpaint(self, texture, mask):
        """
        Inpaint a UV texture map using a GPU-first path (Poisson + diffusion with seam screening),
        falling back to OpenCV Telea on CPU. Returns numpy uint8 [H,W,C].
        `mask` is uint8 where 255 indicates known texels and 0 indicates holes.
        """
        env_gpu_flag = os.getenv("HY3DGEN_GPU_INPAINT", "1")
        method_choice = os.getenv("HY3DGEN_INPAINT_METHOD", "auto").lower()
        debug_logs_enabled = os.getenv("HY3DGEN_INPAINT_DEBUG", "0") == "1"
        log_debug = logger.info if debug_logs_enabled else logger.debug
        info = {
            "input_type": type(texture).__name__,
            "mask_input_type": type(mask).__name__,
            "cuda_available": torch.cuda.is_available(),
            "env_gpu": env_gpu_flag,
            "env_method": method_choice,
        }
        texture_is_tensor = isinstance(texture, torch.Tensor)
        if texture_is_tensor:
            target_device = texture.device
            target_dtype = texture.dtype
        else:
            render_device = getattr(self, "device", "cuda")
            target_device = torch.device(render_device)
            target_dtype = torch.float32
        if isinstance(target_device, torch.device):
            if target_device.type == "cuda" and not torch.cuda.is_available():
                target_device = torch.device("cpu")
        info["input_is_tensor"] = bool(texture_is_tensor)
        skip_vertex_env = os.getenv("HY3DGEN_SKIP_VERTEX_INPAINT", "0")
        skip_vertex = skip_vertex_env == "1"
        gpu_vertex_env = os.getenv("HY3DGEN_GPU_VERTEX_INPAINT", "auto").lower()
        if gpu_vertex_env not in {"0", "1", "auto", "true", "false"}:
            gpu_vertex_env = "auto"
        gpu_vertex_flag = (
            gpu_vertex_env in {"1", "true"}
            or (gpu_vertex_env == "auto" and torch.cuda.is_available())
        )
        ops_device = target_device if (isinstance(target_device, torch.device) and target_device.type == "cuda") else torch.device("cpu")
        info["vertex_skip"] = skip_vertex
        info["vertex_gpu_env"] = gpu_vertex_env
        info["ops_device"] = str(ops_device)
        self._last_uv_inpaint_info = info
        if debug_logs_enabled:
            logger.info(
                "uv_inpaint: start input_type=%s mask_type=%s env_gpu=%s method=%s cuda_available=%s",
                info["input_type"],
                info["mask_input_type"],
                env_gpu_flag,
                method_choice,
                info["cuda_available"],
            )

        # -------- helpers (nested to minimize file-wide churn) ----------------
        def _to_tensor_image(device, x):
            """Convert numpy/torch/PIL to torch float [1,C,H,W] in [0,1]."""
            if isinstance(x, torch.Tensor):
                t = x.detach()
                if t.dim() == 2:
                    t = t.unsqueeze(-1)
                if t.dim() == 3:
                    t = t.permute(2, 0, 1).unsqueeze(0)
                elif t.dim() == 4:
                    pass
                else:
                    raise ValueError("Unsupported tensor dim for image")
                t = t.to(device=device, dtype=torch.float32).clamp_(0, 1)
                return t
            elif isinstance(x, np.ndarray):
                arr = x.astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[..., None]
                if arr.max() > 1.0:
                    arr = arr / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                return t.to(device=device, dtype=torch.float32).clamp_(0, 1)
            elif isinstance(x, Image.Image):
                return _to_tensor_image(device, np.array(x))
            else:
                raise ValueError("Unsupported image type")

        def _from_tensor_image(x):
            """Convert [1,C,H,W] float in [0,1] to uint8 numpy [H,W,C]."""
            if not isinstance(x, torch.Tensor):
                raise ValueError("Expected torch.Tensor")
            t = x.detach().clamp_(0, 1)
            if t.dim() == 4:
                t = t[0]
            t = t.permute(1, 2, 0).contiguous()
            return (t.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)

        def _to_texture_tensor(x, device):
            """Convert inputs to float32 torch tensor [H,W,C] on `device` in [0,1]."""
            if isinstance(x, torch.Tensor):
                t = x.detach()
                if t.dim() == 4:
                    if t.shape[0] != 1:
                        raise ValueError("Unsupported batch dimension for texture tensor")
                    t = t[0]
                if t.dim() != 3:
                    raise ValueError("Unsupported tensor shape for texture tensor conversion")
                t = t.to(device=device)
                if t.dtype in (torch.float16, torch.float32, torch.float64):
                    t = t.to(dtype=torch.float32)
                else:
                    t = t.float()
                max_val = float(t.max().item()) if t.numel() > 0 else 0.0
                if max_val > 1.0005:
                    t = t / 255.0
                return t.clamp_(0.0, 1.0).contiguous()
            elif isinstance(x, np.ndarray):
                arr = x.astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[..., None]
                max_val = float(arr.max()) if arr.size > 0 else 0.0
                if max_val > 1.0:
                    arr = arr / 255.0
                t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
                return t.clamp_(0.0, 1.0).contiguous()
            elif isinstance(x, Image.Image):
                return _to_texture_tensor(np.array(x), device)
            else:
                raise ValueError("Unsupported texture type for tensor conversion")

        def _conv2d_same(x, weight):
            """Grouped conv2d with replicate padding. x:[1,C,H,W], w:[C,1,kh,kw]."""
            kh, kw = weight.shape[-2], weight.shape[-1]
            ph, pw = kh // 2, kw // 2
            xpad = F.pad(x, pad=(pw, pw, ph, ph), mode="replicate")
            return F.conv2d(xpad, weight, groups=x.shape[1])

        def _build_4nbr_kernel(channels, device, dtype):
            k = torch.zeros((channels, 1, 3, 3), device=device, dtype=dtype)
            k[:, 0, 1, 0] = 1.0
            k[:, 0, 0, 1] = 1.0
            k[:, 0, 1, 2] = 1.0
            k[:, 0, 2, 1] = 1.0
            return k

        def _build_laplacian_kernel(channels, device, dtype):
            k = torch.zeros((channels, 1, 3, 3), device=device, dtype=dtype)
            k[:, 0, 1, 0] = 1.0
            k[:, 0, 0, 1] = 1.0
            k[:, 0, 1, 2] = 1.0
            k[:, 0, 2, 1] = 1.0
            k[:, 0, 1, 1] = -4.0
            return k

        def _make_box_kernels(channels, radius, device, dtype):
            k = 2 * radius + 1
            kh = torch.ones((channels, 1, 1, k), device=device, dtype=dtype) / float(k)
            kv = torch.ones((channels, 1, k, 1), device=device, dtype=dtype) / float(k)
            return kh, kv

        def _gpu_inpaint_diffusion(U0, unknown_mask, iters=12, schedule=(2, 2, 2, 1, 1, 1, 1), use_amp=True):
            """Fast separable masked diffusion; updates only unknowns."""
            device = U0.device
            C = U0.shape[1]
            U = U0.clone()
            dtype = torch.float16 if (use_amp and device.type == "cuda") else torch.float32
            for t in range(iters):
                r = schedule[t] if t < len(schedule) else schedule[-1]
                kh, kv = _make_box_kernels(C, r, device, dtype)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device.type == "cuda" and use_amp)):
                    Uh = _conv2d_same(U, kh)
                    Ub = _conv2d_same(Uh, kv)
                Ub = Ub.to(torch.float32)
                U = torch.where(unknown_mask, Ub, U0)
            return U

        def _build_seam_constraint_maps(
            U0,
            vtx_pos,
            pos_idx,
            vtx_uv,
            uv_idx,
            H,
            W,
            lambda_seam=1.0,
            known_mask=None,
        ):
            """
            Build seam weight/target maps by tying UV duplicates of the same 3D vertex.
            Returns seam_w:[1,1,H,W], seam_t:[1,C,H,W] float32 (device of U0) and cache stats.
            """
            device = U0.device
            cache, entry, cache_info = self._get_seam_lookup(H, W, device, (vtx_pos, pos_idx, vtx_uv, uv_idx))
            ys = cache["y"]
            xs = cache["x"]
            group_ids = cache["group_ids"]
            group_counts = cache["group_counts"]
            num_groups = entry["num_groups"]
            C = U0.shape[1]

            seam_w = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
            seam_t = torch.zeros((1, C, H, W), device=device, dtype=torch.float32)
            if ys.numel() == 0 or num_groups == 0:
                return seam_w, seam_t, cache_info

            if known_mask is not None:
                if known_mask.dim() == 4:
                    known_hw = known_mask[0, 0]
                elif known_mask.dim() == 3:
                    known_hw = known_mask[0]
                    if known_hw.dim() == 3:
                        known_hw = known_hw[0]
                else:
                    known_hw = known_mask
                known_hw = known_hw.to(device=device)
                valid = known_hw[ys, xs]
            else:
                valid = torch.ones_like(ys, dtype=torch.bool, device=device)

            if not valid.any():
                cache_info["seam_valid_texels"] = 0
                return seam_w, seam_t, cache_info

            ys = ys[valid]
            xs = xs[valid]
            group_ids = group_ids[valid]
            cache_info["seam_valid_texels"] = int(ys.numel())

            seam_w[0, 0, ys, xs] = float(lambda_seam)

            unique_groups = torch.unique(group_ids)
            for gid in unique_groups.tolist():
                idx = group_ids == gid
                if not idx.any():
                    continue
                ys_sel = ys[idx]
                xs_sel = xs[idx]
                colors = U0[0, :, ys_sel, xs_sel]  # [C,K]
                if colors.shape[1] > 1:
                    group_stat = colors.median(dim=1, keepdim=True).values
                else:
                    group_stat = colors.mean(dim=1, keepdim=True)
                seam_t[0, :, ys_sel, xs_sel] = group_stat

            ones3_scalar = cache["kernel"]
            ones3_channels = torch.ones((C, 1, 3, 3), device=device, dtype=torch.float32)
            w_d = F.conv2d(F.pad(seam_w, (1, 1, 1, 1), mode="replicate"), ones3_scalar)
            w_d = torch.clamp(w_d, min=1e-6)
            seam_w_broadcast = seam_w.repeat(1, C, 1, 1)
            num = F.conv2d(
                F.pad(seam_w_broadcast * seam_t, (1, 1, 1, 1), mode="replicate"),
                ones3_channels,
                groups=C,
            )
            t_avg = num / w_d.repeat(1, C, 1, 1)
            seam_t = t_avg.clamp_(0.0, 1.0)
            seam_w = F.conv2d(F.pad(seam_w, (1, 1, 1, 1), mode="replicate"), ones3_scalar)
            return seam_w.clamp_max_(lambda_seam * 9.0), seam_t, cache_info

        def _build_boundary_ring_constraints(U0, known_mask, scale=1.5):
            """
            Robust, shape-stable boundary constraint builder.
            Guarantees intermediates are exactly [H,W].
            """
            device = U0.device
            _, C, H, W = U0.shape

            if known_mask is None:
                zero_w = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
                zero_t = torch.zeros((1, C, H, W), device=device, dtype=torch.float32)
                return zero_w, zero_t

            # Normalize known_mask to [1,1,H,W] bool on the correct device.
            mask = known_mask.to(device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                if mask.shape[0] == 1:
                    mask = mask.unsqueeze(0)
                else:
                    mask = mask.unsqueeze(1)
            elif mask.dim() != 4:
                raise ValueError(f"known_mask must be 2D-4D, got shape {tuple(mask.shape)}")

            if mask.shape[0] != 1:
                log_debug(
                    "boundary ring trimming extra batch dimension: mask_shape=%s target_hw=(%d,%d)",
                    tuple(int(x) for x in mask.shape),
                    H,
                    W,
                )
                mask = mask[:1]
            if mask.shape[1] != 1:
                log_debug(
                    "boundary ring trimming extra channel dimension: mask_shape=%s target_hw=(%d,%d)",
                    tuple(int(x) for x in mask.shape),
                    H,
                    W,
                )
                mask = mask[:, :1]

            mask_f = mask.float()
            if mask_f.shape[-2:] != (H, W):
                log_debug(
                    "boundary ring resizing mask from %s to (H=%d,W=%d)",
                    tuple(int(x) for x in mask_f.shape[-2:]),
                    H,
                    W,
                )
                mask_f = F.interpolate(mask_f, size=(H, W), mode="nearest")
            known_mask = mask_f.gt(0.5).contiguous()
            known_f = known_mask.float()  # [1,1,H,W]

            # 4-neighbor kernel
            k4_scalar = torch.tensor([[[[0.0, 1.0, 0.0],
                                        [1.0, 0.0, 1.0],
                                        [0.0, 1.0, 0.0]]]], device=device, dtype=torch.float32)
            cnt4 = F.conv2d(known_f, k4_scalar, padding=1)  # [1,1,H,W]
            ring = (~known_mask) & (cnt4 > 0)
            if ring.dim() == 2:
                ring = ring.unsqueeze(0).unsqueeze(0)
            elif ring.dim() == 3:
                ring = ring.unsqueeze(0)
            if ring.dim() != 4:
                raise RuntimeError(
                    f"boundary ring unexpected dim {ring.dim()} (expected 4) "
                    f"known_mask_shape={tuple(int(x) for x in known_mask.shape)} "
                    f"cnt4_shape={tuple(int(x) for x in cnt4.shape)}"
                )
            if ring.shape[-2:] != (H, W):
                log_debug(
                    "boundary ring resizing ring from %s to (H=%d,W=%d)",
                    tuple(int(x) for x in ring.shape[-2:]),
                    H,
                    W,
                )
                ring = F.interpolate(ring.float(), size=(H, W), mode="nearest") > 0.5
            if not ring.any():
                return torch.zeros_like(cnt4), torch.zeros_like(U0)

            U_known = U0 * known_f
            k4_channels = k4_scalar.repeat(C, 1, 3, 3)
            sum4 = F.conv2d(U_known, k4_channels, padding=1, groups=C)  # [1,C,H,W]
            denom = cnt4.clamp(min=1e-6)
            if denom.shape[-2:] != sum4.shape[-2:]:
                denom = F.interpolate(denom, size=sum4.shape[-2:], mode="nearest")
            t = sum4 / denom  # [1,C,H,W]
            ring_f = ring.to(dtype=torch.float32)
            if ring_f.shape[-2:] != t.shape[-2:]:
                log_debug(
                    "boundary ring resizing ring_f from %s to %s",
                    tuple(int(x) for x in ring_f.shape[-2:]),
                    tuple(int(x) for x in t.shape[-2:]),
                )
                ring_f = F.interpolate(ring_f, size=t.shape[-2:], mode="nearest")
                ring = ring_f > 0.5
            t = t * ring_f
            w = (denom / 4.0) * float(scale)
            w = w * ring_f
            return w.to(torch.float32), t.to(torch.float32)

        def _normalize_seam_constraint_maps(seam_w, seam_t, H, W, C, device, log_fn):
            """Coerce seam tensors to [1,1,H,W] / [1,C,H,W] float32 on `device`."""
            if seam_w is None or seam_t is None:
                return None, None

            def _coerce_weight(weight):
                if weight is None:
                    return None
                if not isinstance(weight, torch.Tensor):
                    raise TypeError("seam weight must be torch.Tensor or None")
                wt = weight.to(device=device, dtype=torch.float32)
                if wt.dim() == 2:
                    wt = wt.unsqueeze(0).unsqueeze(0)
                elif wt.dim() == 3:
                    wt = wt.unsqueeze(0) if wt.shape[0] == 1 else wt.unsqueeze(1)
                if wt.dim() != 4:
                    raise ValueError(f"seam weight must be 2D-4D, got shape {tuple(weight.shape)}")
                if wt.shape[0] != 1 or wt.shape[1] != 1:
                    log_fn(
                        "seam weight trimming to single batch/channel: shape=%s target_hw=(%d,%d)",
                        tuple(int(x) for x in wt.shape),
                        H,
                        W,
                    )
                    wt = wt[:1, :1]
                if wt.shape[-2:] != (H, W):
                    log_fn(
                        "seam weight resizing from %s to (H=%d,W=%d)",
                        tuple(int(x) for x in wt.shape[-2:]),
                        H,
                        W,
                    )
                    wt = F.interpolate(wt, size=(H, W), mode="nearest")
                return wt.contiguous()

            def _coerce_target(target):
                if target is None:
                    return None
                if not isinstance(target, torch.Tensor):
                    raise TypeError("seam target must be torch.Tensor or None")
                tt = target.to(device=device, dtype=torch.float32)
                if tt.dim() == 3:
                    tt = tt.unsqueeze(0)
                if tt.dim() != 4:
                    raise ValueError(f"seam target must be 3D/4D, got shape {tuple(target.shape)}")
                if tt.shape[0] != 1:
                    log_fn("seam target trimming batch dim from %s", tuple(int(x) for x in tt.shape))
                    tt = tt[:1]
                if tt.shape[1] != C:
                    if tt.shape[1] == 1 and C != 1:
                        log_fn(
                            "seam target expanding channels from 1 to %d using repeat.",
                            C,
                        )
                        tt = tt.repeat(1, C, 1, 1)
                    elif tt.shape[1] != C:
                        raise ValueError(
                            f"seam target channel mismatch; expected {C}, got {tt.shape[1]}"
                        )
                if tt.shape[-2:] != (H, W):
                    log_fn(
                        "seam target resizing from %s to (H=%d,W=%d)",
                        tuple(int(x) for x in tt.shape[-2:]),
                        H,
                        W,
                    )
                    tt = F.interpolate(tt, size=(H, W), mode="nearest")
                return tt.contiguous()

            seam_w_norm = _coerce_weight(seam_w)
            seam_t_norm = _coerce_target(seam_t)
            if seam_w_norm is None or seam_t_norm is None or not seam_w_norm.any():
                return None, None
            return seam_w_norm, seam_t_norm

        def _gpu_inpaint_poisson(
            U0,
            unknown_mask,
            iters=220,
            tol=1e-3,
            omega=0.66,
            seam_w=None,
            seam_t=None,
            use_amp=True,
            use_graph=True,
            graph_key=None,
        ):
            """Harmonic inpainting via masked weighted-Jacobi with optional seam screening."""
            device = U0.device
            C = U0.shape[1]
            dtype_amp = torch.float16 if (device.type == "cuda" and use_amp) else torch.float32
            k4 = _build_4nbr_kernel(C, device, dtype_amp)
            klap = _build_laplacian_kernel(C, device, dtype_amp)

            # Lazy cache for CUDA graphs
            if not hasattr(self, "_inpaint_graphs"):
                self._inpaint_graphs = {}

            U = U0.clone()

            def _iterate(U):
                with torch.autocast(device_type="cuda", dtype=dtype_amp, enabled=(device.type == "cuda" and use_amp)):
                    ns = _conv2d_same(U, k4)
                    denom = 4.0
                    if seam_w is not None and seam_t is not None:
                        ns = ns + seam_w * seam_t
                        denom = denom + seam_w
                    Unew = ns / denom
                Unew = Unew.to(torch.float32)
                Urelax = omega * Unew + (1.0 - omega) * U
                return torch.where(unknown_mask, Urelax, U0)

            if device.type == "cuda" and use_graph:
                key = graph_key or (
                    tuple(U0.shape),
                    None if seam_w is None else tuple(int(x) for x in seam_w.shape),
                )
                ctx = self._inpaint_graphs.get(key)
                if ctx is None:
                    static = {
                        "U": U.clone(),
                        "U0": U0.clone(),
                        "mask": unknown_mask.clone(),
                        "seam_w": torch.zeros_like(unknown_mask) if seam_w is None else seam_w.clone(),
                        "seam_t": torch.zeros_like(U0) if seam_t is None else seam_t.clone(),
                    }
                    g = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(g):
                        U_static = static["U"]
                        for _ in range(iters):
                            with torch.autocast(device_type="cuda", dtype=dtype_amp, enabled=(device.type == "cuda" and use_amp)):
                                ns = _conv2d_same(U_static, k4)
                                denom = 4.0 + static["seam_w"]
                                ns = ns + static["seam_w"] * static["seam_t"]
                                Unew = ns / denom
                            Unew = Unew.to(torch.float32)
                            Urelax = omega * Unew + (1.0 - omega) * U_static
                            U_static.copy_(torch.where(static["mask"], Urelax, static["U0"]))
                    ctx = {"g": g, "static": static}
                    self._inpaint_graphs[key] = ctx
                # Copy inputs and replay
                s = ctx["static"]
                s["U"].copy_(U)
                s["U0"].copy_(U0)
                s["mask"].copy_(unknown_mask)
                if seam_w is not None:
                    s["seam_w"].copy_(seam_w)
                    s["seam_t"].copy_(seam_t if seam_t is not None else torch.zeros_like(U0))
                else:
                    s["seam_w"].zero_()
                    s["seam_t"].zero_()
                ctx["g"].replay()
                U = ctx["static"]["U"]
            else:
                check_every = max(10, iters // 10)
                for i in range(iters):
                    U = _iterate(U)
                    if (i + 1) % check_every == 0:
                        L = _conv2d_same(U, klap).to(torch.float32)
                        if seam_w is not None and seam_t is not None:
                            L = L + seam_w * (U - seam_t)
                        r = torch.sqrt((L * L).mean())
                        if r.item() < tol:
                            break
            return U

        def _gpu_inpaint_auto(texture_np, mask_np, vtx_pos, vtx_uv, pos_idx, uv_idx):
            """Auto-select diffusion/poisson/hybrid and run on CUDA if available."""
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            start_all = time.perf_counter()

            U0 = _to_tensor_image(device, texture_np)  # [1,C,H,W], [0,1]
            H, W = U0.shape[-2], U0.shape[-1]
            # Robust mask construction: always coerce to HxW to avoid size mismatches
            mh, mw = mask_np.shape[:2]
            m = torch.from_numpy((mask_np == 0).astype(np.uint8)).to(device).view(1, 1, mh, mw).float()
            if (mh, mw) != (H, W):
                m = F.interpolate(m, size=(H, W), mode='nearest')
            unknown_mask = m.bool()  # True where we must inpaint
            mask_dilate = int(os.getenv("HY3DGEN_INPAINT_MASK_DILATE", "1"))
            if mask_dilate > 0:
                um = unknown_mask.float()
                for _ in range(mask_dilate):
                    um = (F.max_pool2d(um, kernel_size=3, stride=1, padding=1) > 0).float()
                unknown_mask = um.bool()

            use_amp = os.getenv("HY3DGEN_INPAINT_PRECISION", "fp16").lower() == "fp16"
            use_graphs = os.getenv("HY3DGEN_INPAINT_USE_GRAPHS", "1") == "1"
            lambda_seam = float(os.getenv("HY3DGEN_INPAINT_SEAM_LAMBDA", "1.0"))
            lambda_ring = float(os.getenv("HY3DGEN_INPAINT_RING_LAMBDA", "1.5"))
            area_frac = float(unknown_mask.float().mean().item())
            stats = {
                "device": str(device),
                "use_amp": use_amp,
                "use_graphs": use_graphs,
                "lambda_seam": lambda_seam,
                "lambda_ring": lambda_ring,
                "area_frac": area_frac,
                "method": method_choice,
                "prefill_ms": 0.0,
                "poisson_ms": 0.0,
                "prefill_runs": 0,
                "poisson_runs": 0,
            }
            stats["mask_dilate"] = mask_dilate

            known_mask = (~unknown_mask).contiguous()
            seam_w, seam_t = _build_boundary_ring_constraints(U0, known_mask, scale=lambda_ring)
            ring_has = bool(seam_w is not None and seam_w.gt(0).any().item())
            stats["seam_constraints"] = False
            if lambda_seam > 0.0:
                try:
                    seam_w_uv, seam_t_uv, seam_cache_info = _build_seam_constraint_maps(
                        U0,
                        vtx_pos,
                        pos_idx,
                        vtx_uv,
                        uv_idx,
                        H,
                        W,
                        lambda_seam=lambda_seam,
                        known_mask=known_mask,
                    )
                    if seam_w is None:
                        seam_w = seam_w_uv
                        seam_t = seam_t_uv
                    else:
                        w_sum = seam_w + seam_w_uv + 1e-6
                        seam_t = (seam_t * seam_w + seam_t_uv * seam_w_uv) / w_sum
                        seam_w = w_sum
                    stats["seam_constraints"] = bool(seam_w_uv is not None and seam_w_uv.gt(0).any().item())
                    stats.update(seam_cache_info)
                except Exception as e:
                    log_debug("UV seam group build failed; continuing without. %s", e, exc_info=True)
                    stats["seam_error"] = str(e)

            seam_w, seam_t = _normalize_seam_constraint_maps(
                seam_w,
                seam_t,
                H,
                W,
                U0.shape[1],
                device,
                log_debug,
            )
            if seam_w is None or seam_t is None:
                seam_w = None
                seam_t = None
                ring_has = False

            stats["ring_constraints"] = ring_has
            stats["ring_pixels"] = int(seam_w.gt(0).sum().item()) if seam_w is not None else 0
            stats["seam_w_shape"] = tuple(int(x) for x in seam_w.shape) if seam_w is not None else None
            stats["seam_t_shape"] = tuple(int(x) for x in seam_t.shape) if seam_t is not None else None
            if seam_w is not None:
                log_debug(
                    "uv_inpaint: seam tensors normalized to seam_w=%s seam_t=%s unknown_mask=%s",
                    tuple(int(x) for x in seam_w.shape),
                    tuple(int(x) for x in seam_t.shape),
                    tuple(int(x) for x in unknown_mask.shape),
                )

            def _run_poisson(U_init):
                t0 = time.perf_counter()
                U = _gpu_inpaint_poisson(
                    U_init,
                    unknown_mask,
                    iters=int(os.getenv("HY3DGEN_INPAINT_ITERS", "220")),
                    tol=float(os.getenv("HY3DGEN_INPAINT_TOL", "1e-3")),
                    omega=float(os.getenv("HY3DGEN_INPAINT_OMEGA", "0.66")),
                    seam_w=seam_w,
                    seam_t=seam_t,
                    use_amp=use_amp,
                    use_graph=use_graphs,
                    graph_key=(
                        "poisson",
                        tuple(U_init.shape),
                        None if seam_w is None else tuple(int(x) for x in seam_w.shape),
                    ),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000.0
                stats["poisson_ms"] += elapsed
                stats["poisson_runs"] += 1
                log_debug("GPU Poisson inpaint: %.3f ms", elapsed)
                return U

            def _run_diffusion(U_init, iters=12):
                t0 = time.perf_counter()
                U = _gpu_inpaint_diffusion(
                    U_init, unknown_mask, iters=iters, schedule=(2, 2, 2, 1, 1, 1, 1), use_amp=use_amp
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000.0
                stats["prefill_ms"] += elapsed
                stats["prefill_runs"] += 1
                log_debug("GPU diffusion prefill: %.3f ms", elapsed)
                return U

            if method_choice == "cpu":
                log_debug("HY3DGEN_INPAINT_METHOD=cpu, skipping GPU path.")
                stats["skip_reason"] = "method=cpu"
                return None, stats

            if method_choice == "diffusion" or (method_choice == "auto" and area_frac < 0.05):
                U_prefill = _run_diffusion(U0, iters=int(os.getenv("HY3DGEN_DIFF_ITERS", "12")))
                stats["strategy"] = "diffusion" if method_choice == "diffusion" else "auto:diffusion+poisson"
                U = _run_poisson(U_prefill) if method_choice == "auto" else U_prefill
            elif method_choice == "poisson" or method_choice == "auto":
                if method_choice == "auto" and area_frac < 0.20:
                    U_prefill = _run_diffusion(U0, iters=6)
                    U = _run_poisson(U_prefill)
                    stats["strategy"] = "auto:prefill+poisson"
                else:
                    U = _run_poisson(U0)
                    stats["strategy"] = "poisson"
            else:
                U = _run_poisson(U0)
                stats["strategy"] = "poisson-default"

            if texture_is_tensor:
                out_tensor = U[0].permute(1, 2, 0).contiguous()
                if out_tensor.dtype != target_dtype:
                    out_tensor = out_tensor.to(dtype=target_dtype)
                if out_tensor.device != target_device:
                    out_tensor = out_tensor.to(device=target_device, non_blocking=True)
                stats["output_format"] = "torch"
                stats["output_dtype"] = str(out_tensor.dtype)
                stats["output_device"] = str(out_tensor.device)
                stats["output_shape"] = list(out_tensor.shape)
                out = out_tensor.detach()
            else:
                out_np = _from_tensor_image(U)
                stats["output_format"] = "numpy"
                stats["output_dtype"] = str(out_np.dtype)
                stats["output_shape"] = list(out_np.shape)
                out = out_np
            total_ms = (time.perf_counter() - start_all) * 1000.0
            stats["total_ms"] = total_ms
            log_debug("GPU inpaint total: %.3f ms (area=%.2f%%)", total_ms, area_frac * 100.0)
            stats["path"] = "gpu"
            return out, stats

        # -------- end helpers -----------------------------------------------

        # Normalize incoming texture to torch tensor [H,W,C] in [0,1]
        texture_tensor = _to_texture_tensor(texture, ops_device)
        tex_h, tex_w, tex_c = texture_tensor.shape
        info["texture_shape"] = [tex_h, tex_w, tex_c]

        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.detach()
            if mask_tensor.dim() == 4:
                if mask_tensor.shape[0] != 1:
                    raise ValueError("mask tensor must have shape [H,W], [H,W,1], or [1,H,W,1]")
                mask_tensor = mask_tensor[0]
            if mask_tensor.dim() == 3 and mask_tensor.shape[-1] == 1:
                mask_tensor = mask_tensor[..., 0]
            if mask_tensor.dim() != 2:
                raise ValueError("mask tensor must be 2D after squeezing")
            mask_tensor = mask_tensor.to(device=ops_device)
            if mask_tensor.dtype == torch.bool:
                mask_bool = mask_tensor
            elif mask_tensor.dtype.is_floating_point:
                mask_bool = mask_tensor > 0.5
            else:
                mask_bool = mask_tensor != 0
            mask_u8 = (mask_bool.detach().cpu().numpy().astype(np.uint8)) * 255
        else:
            mask_array = np.asarray(mask)
            if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
                mask_array = mask_array[..., 0]
            if mask_array.ndim != 2:
                raise ValueError("mask array must be 2D")
            if mask_array.dtype == np.bool_:
                mask_bool_np = mask_array
            else:
                mask_bool_np = mask_array > 0
            mask_u8 = (mask_bool_np.astype(np.uint8)) * 255
            mask_bool = torch.from_numpy(mask_bool_np).to(device=ops_device)

        mask_u8 = np.ascontiguousarray(mask_u8)

        total_texels = mask_bool.numel()
        if total_texels:
            known_texels = mask_bool.sum().item()
            info["mask_hole_frac"] = float(total_texels - known_texels) / float(total_texels)
        else:
            info["mask_hole_frac"] = 0.0

        # Gather mesh info and apply vertex-space inpaint prep
        mesh_cache = None
        if skip_vertex:
            vertex_mode = "skip"
        elif gpu_vertex_flag:
            vertex_mode = "gpu"
        else:
            vertex_mode = "cpu"
        info["vertex_mode"] = vertex_mode
        info["vertex_gpu_active"] = vertex_mode == "gpu"
        if vertex_mode == "cpu":
            vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()
            mesh_cache = (vtx_pos, pos_idx, vtx_uv, uv_idx)
            texture_np = texture_tensor.cpu().numpy()
            texture_np, mask_np = meshVerticeInpaint(texture_np, mask_u8, vtx_pos, vtx_uv, pos_idx, uv_idx)
            texture_tensor = torch.from_numpy(texture_np).to(device=ops_device, dtype=torch.float32)
            mask_bool = torch.from_numpy(mask_np > 0).to(device=ops_device)
            mask_u8 = np.ascontiguousarray(mask_np)
        elif vertex_mode == "gpu":
            texture_tensor, mask_bool, vertex_stats = self._gpu_vertex_inpaint(texture_tensor, mask_bool, ops_device)
            info.update(vertex_stats)
            mask_u8 = (mask_bool.detach().cpu().numpy().astype(np.uint8)) * 255
            mask_u8 = np.ascontiguousarray(mask_u8)
        else:  # skip
            info["vertex_preprocess_skipped"] = True

        # Try GPU path first
        use_gpu = (env_gpu_flag == "1") and torch.cuda.is_available()
        info["gpu_requested"] = env_gpu_flag == "1"
        info["gpu_attempted"] = bool(use_gpu and method_choice != "cpu")
        if debug_logs_enabled and not info["gpu_attempted"]:
            logger.info(
                "uv_inpaint: skipping GPU path (use_gpu_flag=%s cuda_available=%s method=%s)",
                env_gpu_flag,
                torch.cuda.is_available(),
                method_choice,
            )
        if use_gpu and method_choice != "cpu":
            try:
                start_gpu = time.perf_counter()
                if mesh_cache is None:
                    mesh_cache = self.get_mesh()
                vtx_pos, pos_idx, vtx_uv, uv_idx = mesh_cache
                out_val, stats = _gpu_inpaint_auto(texture_tensor, mask_u8, vtx_pos, vtx_uv, pos_idx, uv_idx)
                info.update(stats)
                if out_val is None:
                    raise RuntimeError(f"GPU inpaint skipped: {stats.get('skip_reason', 'unknown reason')}")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gpu_elapsed = (time.perf_counter() - start_gpu) * 1000.0
                info["mode"] = "gpu"
                info.setdefault("total_ms", gpu_elapsed)
                log_debug("uv_inpaint: GPU path completed in %.3f ms", gpu_elapsed)
                self._last_uv_inpaint_info = info
                return out_val
            except Exception as e:
                info["fallback_reason"] = str(e)
                logger.exception("GPU inpaint failed; falling back to CPU Telea. Reason: %s", e)

        # CPU Telea fallback
        start_cpu = time.perf_counter()
        texture_np_cpu = texture_tensor.detach().cpu().numpy()
        out_np = cv2.inpaint((texture_np_cpu * 255).astype(np.uint8), 255 - mask_u8, 3, cv2.INPAINT_TELEA)
        cpu_elapsed = (time.perf_counter() - start_cpu) * 1000.0
        info["mode"] = "cpu"
        info["cpu_duration_ms"] = cpu_elapsed
        self._last_uv_inpaint_info = info
        log_debug("uv_inpaint: CPU Telea completed in %.3f ms", cpu_elapsed)
        if texture_is_tensor:
            out_np = out_np.astype(np.float32) / 255.0
            out_t = torch.from_numpy(out_np)
            if out_t.dtype != target_dtype:
                out_t = out_t.to(dtype=target_dtype)
            out_t = out_t.to(device=target_device, non_blocking=(target_device.type == "cuda"))
            info["output_format"] = "torch"
            info["output_dtype"] = str(out_t.dtype)
            info["output_device"] = str(out_t.device)
            info["output_shape"] = list(out_t.shape)
            return out_t.contiguous()
        info["output_format"] = "numpy"
        info["output_dtype"] = str(out_np.dtype)
        info["output_shape"] = list(out_np.shape)
        return out_np
