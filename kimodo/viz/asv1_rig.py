# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ASV1 robot rig: GLB mesh loading and viser scene setup for ASV1 skeleton."""

import json
import os
import struct
from typing import Optional, Tuple

import numpy as np
import trimesh

import viser
import viser.transforms as tf
from kimodo.skeleton import ASV1Skeleton26

# Map from skeleton joint names to GLB mesh node indices (even-numbered nodes = mesh carriers)
# In the GLB, each joint has two nodes: odd = joint transform, even = mesh geometry
# The skin joint list maps to the odd nodes; the even children carry the meshes.
ASV1_MESH_JOINT_MAP = {
    "RobotOrigin": ["RobotOrigin"],
    "R_LEG_HIP_PITCH": ["R_LEG_HIP_PITCH"],
    "R_LEG_HIP_ROLL": ["R_LEG_HIP_ROLL"],
    "R_LEG_HIP_YAW": ["R_LEG_HIP_YAW"],
    "R_LEG_KNEE": ["R_LEG_KNEE"],
    "R_LEG_ANKLE_PITCH": ["R_LEG_ANKLE_PITCH"],
    "R_LEG_ANKLE_ROLL": ["R_LEG_ANKLE_ROLL"],
    "L_LEG_HIP_PITCH": ["L_LEG_HIP_PITCH"],
    "L_LEG_HIP_ROLL": ["L_LEG_HIP_ROLL"],
    "L_LEG_HIP_YAW": ["L_LEG_HIP_YAW"],
    "L_LEG_KNEE": ["L_LEG_KNEE"],
    "L_LEG_ANKLE_PITCH": ["L_LEG_ANKLE_PITCH"],
    "L_LEG_ANKLE_ROLL": ["L_LEG_ANKLE_ROLL"],
    "WAIST_YAW": ["WAIST_YAW"],
    "HEAD_YAW": ["HEAD_YAW"],
    "HEAD_PITCH": ["HEAD_PITCH"],
    "R_SHOULDER_PITCH": ["R_SHOULDER_PITCH"],
    "R_SHOULDER_ROLL": ["R_SHOULDER_ROLL"],
    "R_SHOULDER_YAW": ["R_SHOULDER_YAW"],
    "R_ELBOW": ["R_ELBOW"],
    "R_WRIST_YAW": ["R_WRIST_YAW"],
    "L_SHOULDER_PITCH": ["L_SHOULDER_PITCH"],
    "L_SHOULDER_ROLL": ["L_SHOULDER_ROLL"],
    "L_SHOULDER_YAW": ["L_SHOULDER_YAW"],
    "L_ELBOW": ["L_ELBOW"],
    "L_WRIST_YAW": ["L_WRIST_YAW"],
}

_ASV1_MESH_DATA_CACHE: Optional[list[dict]] = None


def _load_asv1_mesh_data(
    glb_path: str,
    skeleton: ASV1Skeleton26,
) -> list[dict]:
    """Load per-joint meshes from the ASV1 GLB file. Cached after first call."""
    global _ASV1_MESH_DATA_CACHE
    if _ASV1_MESH_DATA_CACHE is not None:
        return _ASV1_MESH_DATA_CACHE

    scene = trimesh.load(glb_path, process=False)
    if not isinstance(scene, trimesh.Scene):
        _ASV1_MESH_DATA_CACHE = []
        return _ASV1_MESH_DATA_CACHE

    # Build name -> geometry mapping from the scene graph
    # GLB nodes with meshes have geometry entries in the scene
    geom_by_name: dict[str, trimesh.Trimesh] = {}
    for geom_name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            # Strip suffixes like ".002", "_mesh", etc. to match joint names
            base = geom_name
            for suffix in [".002", ".001", "_restored_mesh", "_asv1_mesh", "_asv1_mesh.001",
                           "_hand_mesh", "_restored_from_backup_mesh",
                           "_TORSO_COMBINED_mesh"]:
                base = base.replace(suffix, "")
            geom_by_name[base] = geom
            geom_by_name[geom_name] = geom  # also store exact name

    data_list: list[dict] = []
    for joint_name, mesh_keys in ASV1_MESH_JOINT_MAP.items():
        if joint_name not in skeleton.bone_index:
            continue
        joint_idx = skeleton.bone_index[joint_name]

        for mesh_key in mesh_keys:
            # Try to find geometry matching this joint
            geom = None
            for gname, g in scene.geometry.items():
                # Match by joint name appearing in geometry name
                if mesh_key in gname and isinstance(g, trimesh.Trimesh):
                    geom = g
                    break

            if geom is None:
                continue

            # Get the world transform for this geometry from the scene graph
            node_name = None
            for nname in scene.graph.nodes_geometry:
                if mesh_key in nname:
                    node_name = nname
                    break

            if node_name is not None:
                transform = scene.graph.get(node_name)[0]
            else:
                transform = np.eye(4)

            # Extract vertices in world space, then make relative to joint
            vertices = np.array(geom.vertices, dtype=np.float64)
            faces = np.array(geom.faces, dtype=np.int32)

            # Apply the node's world transform to vertices
            verts_hom = np.c_[vertices, np.ones(len(vertices))]
            verts_world = (transform @ verts_hom.T).T[:, :3]

            data_list.append({
                "mesh_name": mesh_key,
                "vertices": verts_world,
                "faces": faces,
                "joint_idx": joint_idx,
                "geom_pos": np.zeros(3, dtype=np.float64),
                "geom_rot": np.eye(3, dtype=np.float64),
            })

    _ASV1_MESH_DATA_CACHE = data_list
    return _ASV1_MESH_DATA_CACHE


class ASV1MeshRig:
    """Rig for ASV1 GLB meshes. Each joint has its own mesh piece."""

    def __init__(
        self,
        name: str,
        server: viser.ViserServer | viser.ClientHandle,
        skeleton: ASV1Skeleton26,
        glb_path: str,
        color: Tuple[int, int, int],
    ):
        self.server = server
        self.skeleton = skeleton
        self.color = color
        self.mesh_handles: list[viser.SceneHandle] = []
        self.mesh_items: list[dict] = []

        data_list = _load_asv1_mesh_data(glb_path, skeleton)

        # Compute bind-pose joint positions to make mesh vertices joint-local
        neutral = skeleton.neutral_joints.cpu().numpy().astype(np.float64)

        for item in data_list:
            mesh_name = item["mesh_name"]
            vertices = item["vertices"].copy()
            faces = item["faces"]
            joint_idx = item["joint_idx"]

            # Make vertices relative to joint rest position
            joint_pos = neutral[joint_idx]
            local_verts = vertices - joint_pos

            handle = server.scene.add_mesh_simple(
                f"/{name}/asv1_mesh/{mesh_name}",
                vertices=local_verts.astype(np.float32),
                faces=faces,
                opacity=None,
                color=self.color,
                wireframe=False,
                visible=True,
            )
            self.mesh_handles.append(handle)
            self.mesh_items.append({
                "handle": handle,
                "joint_idx": joint_idx,
                "local_verts": local_verts,
            })

    def set_visibility(self, visible: bool) -> None:
        for handle in self.mesh_handles:
            handle.visible = visible

    def set_opacity(self, opacity: float) -> None:
        for handle in self.mesh_handles:
            handle.opacity = opacity

    def set_wireframe(self, wireframe: bool) -> None:
        for handle in self.mesh_handles:
            handle.wireframe = wireframe

    def set_color(self, color: Tuple[int, int, int]) -> None:
        self.color = color
        for handle in self.mesh_handles:
            handle.color = color

    def set_pose(self, joints_pos: np.ndarray, joints_rot: np.ndarray) -> None:
        for item in self.mesh_items:
            handle = item["handle"]
            joint_idx = item["joint_idx"]

            joint_pos = joints_pos[joint_idx]
            joint_rot = joints_rot[joint_idx]

            handle.position = joint_pos
            handle.wxyz = tf.SO3.from_matrix(joint_rot).wxyz

    def clear(self) -> None:
        for handle in self.mesh_handles:
            self.server.scene.remove_by_name(handle.name)
        self.mesh_handles = []
        self.mesh_items = []
