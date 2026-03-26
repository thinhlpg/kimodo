# SPDX-License-Identifier: Apache-2.0
"""Retarget SMPL-X 22-joint motions to ASV1 26-joint robot skeleton.

Strategy:
- Ball joints (hip, shoulder) → decompose 3×3 rotation into Euler XYZ → assign to hinge chain
- Missing joints (spine2, collars) → merge rotation into nearest neighbor
- Single-axis joints (knee, ankle, neck) → extract dominant axis angle
- Scale root positions by height ratio

SMPL-X uses ball joints (1 joint = full 3×3 rotation, 3 DoF).
ASV1 uses hinge chains (3 joints × 1 DoF each, chained pitch→roll→yaw).

Coordinate system: Kimodo Y-up, Z-forward, X-right.
  - Pitch = rotation around X (sagittal plane: flexion/extension)
  - Yaw   = rotation around Y (transverse plane: internal/external rotation)
  - Roll  = rotation around Z (frontal plane: abduction/adduction)
"""

import torch
import torch.nn.functional as F

from kimodo.skeleton import ASV1Skeleton26, SMPLXSkeleton22


def _rotmat_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """Decompose rotation matrix into XYZ intrinsic Euler angles (pitch, yaw, roll).

    R = Rx(pitch) @ Ry(yaw) @ Rz(roll)

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) tensor of [pitch, yaw, roll] in radians.
    """
    # For XYZ intrinsic order (same as the MuJoCo exporter uses):
    # pitch (X) = atan2(R[2,1], R[2,2])
    # yaw   (Y) = atan2(-R[2,0], sqrt(R[2,1]^2 + R[2,2]^2))
    # roll  (Z) = atan2(R[1,0], R[0,0])
    pitch = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    yaw = torch.atan2(-R[..., 2, 0], torch.sqrt(R[..., 2, 1] ** 2 + R[..., 2, 2] ** 2))
    roll = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return torch.stack([pitch, yaw, roll], dim=-1)


def _euler_xyz_to_rotmat(angles: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to rotation matrix.

    Args:
        angles: (..., 3) tensor of [pitch, yaw, roll] in radians.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    pitch = angles[..., 0]
    yaw = angles[..., 1]
    roll = angles[..., 2]

    zeros = torch.zeros_like(pitch)
    ones = torch.ones_like(pitch)

    # Rx(pitch)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    Rx = torch.stack([
        ones, zeros, zeros,
        zeros, cp, -sp,
        zeros, sp, cp,
    ], dim=-1).reshape(*pitch.shape, 3, 3)

    # Ry(yaw)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    Ry = torch.stack([
        cy, zeros, sy,
        zeros, ones, zeros,
        -sy, zeros, cy,
    ], dim=-1).reshape(*yaw.shape, 3, 3)

    # Rz(roll)
    cr, sr = torch.cos(roll), torch.sin(roll)
    Rz = torch.stack([
        cr, -sr, zeros,
        sr, cr, zeros,
        zeros, zeros, ones,
    ], dim=-1).reshape(*roll.shape, 3, 3)

    return Rx @ Ry @ Rz


def _single_axis_angle(R: torch.Tensor, axis: int) -> torch.Tensor:
    """Extract single-axis rotation angle from rotation matrix.

    Args:
        R: (..., 3, 3) rotation matrices.
        axis: 0=X (pitch), 1=Y (yaw), 2=Z (roll).

    Returns:
        (...,) angle in radians.
    """
    if axis == 0:  # X-axis (pitch)
        return torch.atan2(R[..., 2, 1], R[..., 2, 2])
    elif axis == 1:  # Y-axis (yaw)
        return torch.atan2(R[..., 0, 2], R[..., 0, 0])
    elif axis == 2:  # Z-axis (roll)
        return torch.atan2(R[..., 1, 0], R[..., 1, 1])
    else:
        raise ValueError(f"Invalid axis: {axis}")


def _angle_to_rotmat(angle: torch.Tensor, axis: int) -> torch.Tensor:
    """Create rotation matrix from single-axis angle.

    Args:
        angle: (...,) angles in radians.
        axis: 0=X, 1=Y, 2=Z.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    c = torch.cos(angle)
    s = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)

    if axis == 0:  # X
        R = torch.stack([ones, zeros, zeros, zeros, c, -s, zeros, s, c], dim=-1)
    elif axis == 1:  # Y
        R = torch.stack([c, zeros, s, zeros, ones, zeros, -s, zeros, c], dim=-1)
    elif axis == 2:  # Z
        R = torch.stack([c, -s, zeros, s, c, zeros, zeros, zeros, ones], dim=-1)
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return R.reshape(*angle.shape, 3, 3)


# SMPL-X joint indices (from SMPLXSkeleton22 bone_order_names_with_parents)
_SX = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3,
    "left_knee": 4, "right_knee": 5, "spine2": 6,
    "left_ankle": 7, "right_ankle": 8, "spine3": 9,
    "left_foot": 10, "right_foot": 11, "neck": 12,
    "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
}

# ASV1 joint indices (from ASV1Skeleton26 bone_order_names_with_parents)
_AV = {
    "RobotOrigin": 0,
    "R_LEG_HIP_PITCH": 1, "R_LEG_HIP_ROLL": 2, "R_LEG_HIP_YAW": 3,
    "R_LEG_KNEE": 4, "R_LEG_ANKLE_PITCH": 5, "R_LEG_ANKLE_ROLL": 6,
    "L_LEG_HIP_PITCH": 7, "L_LEG_HIP_ROLL": 8, "L_LEG_HIP_YAW": 9,
    "L_LEG_KNEE": 10, "L_LEG_ANKLE_PITCH": 11, "L_LEG_ANKLE_ROLL": 12,
    "WAIST_YAW": 13,
    "HEAD_YAW": 14, "HEAD_PITCH": 15,
    "R_SHOULDER_PITCH": 16, "R_SHOULDER_ROLL": 17, "R_SHOULDER_YAW": 18,
    "R_ELBOW": 19, "R_WRIST_YAW": 20,
    "L_SHOULDER_PITCH": 21, "L_SHOULDER_ROLL": 22, "L_SHOULDER_YAW": 23,
    "L_ELBOW": 24, "L_WRIST_YAW": 25,
}

# Height ratio for scaling root positions
_SMPLX_HEIGHT = 1.70  # approximate SMPL-X standing height (meters)
_ASV1_HEIGHT = 1.12   # ASV1 robot height (meters)
_SCALE = _ASV1_HEIGHT / _SMPLX_HEIGHT


class SMPLXToASV1Retargeter:
    """Retarget SMPL-X 22-joint motion output to ASV1 26-joint robot skeleton.

    Usage:
        retargeter = SMPLXToASV1Retargeter()
        asv1_output = retargeter(smplx_output)
        # asv1_output has the same keys as smplx_output but with ASV1 skeleton.
    """

    def __init__(self, scale: float = _SCALE):
        self.scale = scale
        self.smplx_skel = SMPLXSkeleton22()
        self.asv1_skel = ASV1Skeleton26()

    def retarget_local_rotations(self, smplx_local_rots: torch.Tensor) -> torch.Tensor:
        """Convert SMPL-X local rotation matrices to ASV1 local rotation matrices.

        Args:
            smplx_local_rots: (B, T, 22, 3, 3) or (T, 22, 3, 3) SMPL-X local rotations.

        Returns:
            (B, T, 26, 3, 3) or (T, 26, 3, 3) ASV1 local rotations.
        """
        squeeze_batch = False
        if smplx_local_rots.dim() == 4:
            smplx_local_rots = smplx_local_rots.unsqueeze(0)
            squeeze_batch = True

        B, T, _, _, _ = smplx_local_rots.shape
        device = smplx_local_rots.device
        dtype = smplx_local_rots.dtype

        # Initialize ASV1 rotations as identity
        asv1_rots = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 1, 3, 3).expand(B, T, 26, 3, 3).clone()

        # --- Root (pelvis → RobotOrigin) ---
        asv1_rots[:, :, _AV["RobotOrigin"]] = smplx_local_rots[:, :, _SX["pelvis"]]

        # --- Right hip (ball → 3 hinges: pitch/roll/yaw) ---
        r_hip_rot = smplx_local_rots[:, :, _SX["right_hip"]]
        r_hip_euler = _rotmat_to_euler_xyz(r_hip_rot)  # (B, T, 3) = [pitch, yaw, roll]
        asv1_rots[:, :, _AV["R_LEG_HIP_PITCH"]] = _angle_to_rotmat(r_hip_euler[..., 0], axis=0)
        asv1_rots[:, :, _AV["R_LEG_HIP_ROLL"]] = _angle_to_rotmat(r_hip_euler[..., 2], axis=2)
        asv1_rots[:, :, _AV["R_LEG_HIP_YAW"]] = _angle_to_rotmat(r_hip_euler[..., 1], axis=1)

        # --- Left hip (ball → 3 hinges) ---
        l_hip_rot = smplx_local_rots[:, :, _SX["left_hip"]]
        l_hip_euler = _rotmat_to_euler_xyz(l_hip_rot)
        asv1_rots[:, :, _AV["L_LEG_HIP_PITCH"]] = _angle_to_rotmat(l_hip_euler[..., 0], axis=0)
        asv1_rots[:, :, _AV["L_LEG_HIP_ROLL"]] = _angle_to_rotmat(l_hip_euler[..., 2], axis=2)
        asv1_rots[:, :, _AV["L_LEG_HIP_YAW"]] = _angle_to_rotmat(l_hip_euler[..., 1], axis=1)

        # --- Knees (single axis: pitch/X) ---
        asv1_rots[:, :, _AV["R_LEG_KNEE"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["right_knee"]], axis=0), axis=0
        )
        asv1_rots[:, :, _AV["L_LEG_KNEE"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["left_knee"]], axis=0), axis=0
        )

        # --- Ankles (pitch) ---
        asv1_rots[:, :, _AV["R_LEG_ANKLE_PITCH"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["right_ankle"]], axis=0), axis=0
        )
        asv1_rots[:, :, _AV["L_LEG_ANKLE_PITCH"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["left_ankle"]], axis=0), axis=0
        )

        # --- Feet → ankle roll ---
        asv1_rots[:, :, _AV["R_LEG_ANKLE_ROLL"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["right_foot"]], axis=2), axis=2
        )
        asv1_rots[:, :, _AV["L_LEG_ANKLE_ROLL"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["left_foot"]], axis=2), axis=2
        )

        # --- Spine: merge spine1 + spine2 + spine3 → WAIST_YAW ---
        # Combine the three spine rotations into one, then extract yaw (dominant)
        spine_combined = (
            smplx_local_rots[:, :, _SX["spine1"]]
            @ smplx_local_rots[:, :, _SX["spine2"]]
            @ smplx_local_rots[:, :, _SX["spine3"]]
        )
        asv1_rots[:, :, _AV["WAIST_YAW"]] = spine_combined

        # --- Neck → HEAD_YAW ---
        asv1_rots[:, :, _AV["HEAD_YAW"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["neck"]], axis=1), axis=1
        )

        # --- Head → HEAD_PITCH ---
        asv1_rots[:, :, _AV["HEAD_PITCH"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["head"]], axis=0), axis=0
        )

        # --- Right shoulder: merge collar + shoulder ball → 3 hinges ---
        r_shoulder_combined = (
            smplx_local_rots[:, :, _SX["right_collar"]]
            @ smplx_local_rots[:, :, _SX["right_shoulder"]]
        )
        r_sh_euler = _rotmat_to_euler_xyz(r_shoulder_combined)
        asv1_rots[:, :, _AV["R_SHOULDER_PITCH"]] = _angle_to_rotmat(r_sh_euler[..., 0], axis=0)
        asv1_rots[:, :, _AV["R_SHOULDER_ROLL"]] = _angle_to_rotmat(r_sh_euler[..., 2], axis=2)
        asv1_rots[:, :, _AV["R_SHOULDER_YAW"]] = _angle_to_rotmat(r_sh_euler[..., 1], axis=1)

        # --- Left shoulder: merge collar + shoulder ball → 3 hinges ---
        l_shoulder_combined = (
            smplx_local_rots[:, :, _SX["left_collar"]]
            @ smplx_local_rots[:, :, _SX["left_shoulder"]]
        )
        l_sh_euler = _rotmat_to_euler_xyz(l_shoulder_combined)
        asv1_rots[:, :, _AV["L_SHOULDER_PITCH"]] = _angle_to_rotmat(l_sh_euler[..., 0], axis=0)
        asv1_rots[:, :, _AV["L_SHOULDER_ROLL"]] = _angle_to_rotmat(l_sh_euler[..., 2], axis=2)
        asv1_rots[:, :, _AV["L_SHOULDER_YAW"]] = _angle_to_rotmat(l_sh_euler[..., 1], axis=1)

        # --- Elbows (single axis: pitch/X) ---
        asv1_rots[:, :, _AV["R_ELBOW"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["right_elbow"]], axis=0), axis=0
        )
        asv1_rots[:, :, _AV["L_ELBOW"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["left_elbow"]], axis=0), axis=0
        )

        # --- Wrists (yaw/Y) ---
        asv1_rots[:, :, _AV["R_WRIST_YAW"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["right_wrist"]], axis=1), axis=1
        )
        asv1_rots[:, :, _AV["L_WRIST_YAW"]] = _angle_to_rotmat(
            _single_axis_angle(smplx_local_rots[:, :, _SX["left_wrist"]], axis=1), axis=1
        )

        if squeeze_batch:
            asv1_rots = asv1_rots.squeeze(0)

        return asv1_rots

    def __call__(self, smplx_output: dict) -> dict:
        """Retarget a full SMPL-X motion output dict to ASV1.

        Args:
            smplx_output: Dict with keys from Kimodo model output:
                - local_rot_mats: (B, T, 22, 3, 3)
                - root_positions: (B, T, 3)
                - foot_contacts: (B, T, 4)
                - global_root_heading: (B, T, 2)
                - smooth_root_pos: (B, T, 3)

        Returns:
            Dict with same keys but ASV1 skeleton (26 joints):
                - local_rot_mats: (B, T, 26, 3, 3)
                - global_rot_mats: (B, T, 26, 3, 3)
                - posed_joints: (B, T, 26, 3)
                - root_positions: (B, T, 3) scaled
                - foot_contacts: (B, T, 4)
                - global_root_heading: (B, T, 2)
                - smooth_root_pos: (B, T, 3) scaled
        """
        local_rots = smplx_output["local_rot_mats"]
        root_pos = smplx_output["root_positions"]

        # Retarget rotations
        asv1_local_rots = self.retarget_local_rotations(local_rots)

        # Scale root positions
        asv1_root_pos = root_pos * self.scale

        # Run FK on ASV1 skeleton to get global rotations and joint positions
        asv1_skel = self.asv1_skel.to(asv1_local_rots.device)
        if asv1_local_rots.dim() == 4:
            # (T, 26, 3, 3) → run FK directly
            global_rots, posed_joints, _ = asv1_skel.fk(asv1_local_rots, asv1_root_pos)
        else:
            # (B, T, 26, 3, 3) → run FK per batch
            B = asv1_local_rots.shape[0]
            global_rots_list, posed_joints_list = [], []
            for b in range(B):
                gr, pj, _ = asv1_skel.fk(asv1_local_rots[b], asv1_root_pos[b])
                global_rots_list.append(gr)
                posed_joints_list.append(pj)
            global_rots = torch.stack(global_rots_list)
            posed_joints = torch.stack(posed_joints_list)

        result = {
            "local_rot_mats": asv1_local_rots,
            "global_rot_mats": global_rots,
            "posed_joints": posed_joints,
            "root_positions": asv1_root_pos,
            "foot_contacts": smplx_output.get("foot_contacts", None),
            "global_root_heading": smplx_output.get("global_root_heading", None),
            "smooth_root_pos": smplx_output.get("smooth_root_pos", root_pos) * self.scale,
        }
        return result
