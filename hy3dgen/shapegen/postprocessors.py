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
import tempfile
from typing import Union

import numpy as np
import pymeshlab
import torch
import trimesh

from .models.autoencoders import Latent2MeshOutput
from .utils import synchronize_timer


def load_mesh(path):
    if path.endswith(".glb"):
        mesh = trimesh.load(path)
    else:
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(path)
    return mesh


def _use_gpu_postprocessor() -> bool:
    return _GPU_POSTPROCESSOR is not None and getattr(_GPU_POSTPROCESSOR, "device", None) is not None \
        and _GPU_POSTPROCESSOR.device.type == "cuda"


def _gpu_reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int):
    assert _GPU_POSTPROCESSOR is not None
    tm_in = pymeshlab2trimesh(mesh)
    tm_out = _GPU_POSTPROCESSOR.reduce_face(tm_in, max_facenum=max_facenum)
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    faces_arr_out = getattr(tm_out, "faces", None)
    faces_arr_in = getattr(tm_in, "faces", None)
    faces_out = int(faces_arr_out.shape[0]) if faces_arr_out is not None else 0
    faces_in = int(faces_arr_in.shape[0]) if faces_arr_in is not None else 0
    logger.debug(
        "GPU face reduction result: input_faces=%s target=%s output_faces=%s",
        faces_in, max_facenum, faces_out,
    )
=======
    faces_out = len(getattr(tm_out, "faces", []) or [])
    faces_in = len(getattr(tm_in, "faces", []) or [])
<<<<<<< ours
>>>>>>> theirs
    target_ratio = getattr(_GPU_POSTPROCESSOR, "target_face_ratio", 0.8)
    min_expected = max(100, int(max_facenum * max(0.5, target_ratio - 0.1)))
=======
    faces_out = len(getattr(tm_out, "faces", []) or [])
    faces_in = len(getattr(tm_in, "faces", []) or [])
    min_expected = max(100, int(max_facenum * 0.75))
>>>>>>> theirs
=======
    min_expected = max(100, int(max_facenum * 0.75))
>>>>>>> theirs
    if faces_in >= min_expected and faces_out < min_expected:
        logger.warning(
            "GPU face reduction produced only %s faces (target=%s, input=%s); falling back to CPU decimation.",
            faces_out, max_facenum, faces_in,
        )
        raise RuntimeError("GPU face reduction below expected threshold")
=======
>>>>>>> theirs
    return trimesh2pymeshlab(tm_out)


def _gpu_remove_floater(mesh: pymeshlab.MeshSet):
    assert _GPU_POSTPROCESSOR is not None
    tm_in = pymeshlab2trimesh(mesh)
    tm_out = _GPU_POSTPROCESSOR.remove_floater(tm_in)
    return trimesh2pymeshlab(tm_out)


def reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int = 200000):
    if max_facenum > mesh.current_mesh().face_number():
        return mesh

    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return mesh


def remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    current = mesh.current_mesh()
    vertices = current.vertex_matrix()
    faces = current.face_matrix()
    out_mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
    return out_mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    if isinstance(mesh, trimesh.scene.Scene):
        combined = trimesh.Trimesh()
        for geom in mesh.geometry.values():
            combined = trimesh.util.concatenate([combined, geom])
        mesh = combined

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces), "converted_mesh")
    return ms


def export_mesh(input, output):
    if isinstance(input, pymeshlab.MeshSet):
        mesh = output
    elif isinstance(input, Latent2MeshOutput):
        output = Latent2MeshOutput()
        output.mesh_v = output.current_mesh().vertex_matrix()
        output.mesh_f = output.current_mesh().face_matrix()
        mesh = output
    else:
        mesh = pymeshlab2trimesh(output)
    return mesh


def import_mesh(mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str]) -> pymeshlab.MeshSet:
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    elif isinstance(mesh, Latent2MeshOutput):
        mesh = pymeshlab.MeshSet()
        mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=mesh.mesh_v, face_matrix=mesh.mesh_f)
        mesh.add_mesh(mesh_pymeshlab, "converted_mesh")

    if isinstance(mesh, (trimesh.Trimesh, trimesh.scene.Scene)):
        mesh = trimesh2pymeshlab(mesh)

    return mesh


class FaceReducer:
    @synchronize_timer('FaceReducer')
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
        max_facenum: int = 40000
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        ms = import_mesh(mesh)
        ms = reduce_face(ms, max_facenum=max_facenum)
        mesh = export_mesh(mesh, ms)
        return mesh


class FloaterRemover:
    @synchronize_timer('FloaterRemover')
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)
        ms = remove_floater(ms)
        mesh = export_mesh(mesh, ms)
        return mesh


class DegenerateFaceRemover:
    @synchronize_timer('DegenerateFaceRemover')
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
            ms.save_current_mesh(temp_file.name)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_file.name)

        mesh = export_mesh(mesh, ms)
        return mesh


def mesh_normalize(mesh):
    """
    Normalize mesh vertices to sphere
    """
    scale_factor = 1.2
    vtx_pos = np.asarray(mesh.vertices)
    max_bb = (vtx_pos - 0).max(0)[0]
    min_bb = (vtx_pos - 0).min(0)[0]

    center = (max_bb + min_bb) / 2

    scale = torch.norm(torch.tensor(vtx_pos - center, dtype=torch.float32), dim=1).max() * 2.0

    vtx_pos = (vtx_pos - center) * (scale_factor / float(scale))
    mesh.vertices = vtx_pos

    return mesh


class MeshSimplifier:
    def __init__(self, executable: str = None):
        if executable is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            executable = os.path.join(CURRENT_DIR, "mesh_simplifier.bin")
        self.executable = executable

    @synchronize_timer('MeshSimplifier')
    def __call__(
        self,
        mesh: Union[trimesh.Trimesh],
    ) -> Union[trimesh.Trimesh]:
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_input:
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_output:
                mesh.export(temp_input.name)
                os.system(f'{self.executable} {temp_input.name} {temp_output.name}')
                ms = trimesh.load(temp_output.name, process=False)
                if isinstance(ms, trimesh.Scene):
                    combined_mesh = trimesh.Trimesh()
                    for geom in ms.geometry.values():
                        combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
                    ms = combined_mesh
                ms = mesh_normalize(ms)
                return ms
