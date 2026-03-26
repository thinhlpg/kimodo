"""Create ASV1 skeleton asset files for the Kimodo project.

ASV1 is a humanoid robot with 26 joints. This script computes world-space
rest-pose joint positions from the joint hierarchy and local translations
(extracted from the GLB file), then saves:
  - joints.p        : torch tensor (26, 3) float64, neutral rest-pose positions in meters (Y-up)
  - bvh_joints.p    : same as joints.p (identical for now)
  - rest_pose_local_rot.p : torch tensor (26, 3, 3) float32, identity rotation matrices
"""

import os
from pathlib import Path

import torch

# ASV1 joint hierarchy: (name, parent_index, local_translation_xyz)
# Parent index -1 means root (no parent).
joints_data = [
    ("RobotOrigin",       -1, [0.0, 0.0, 0.0]),
    ("R_LEG_HIP_PITCH",    0, [0.0059, -0.0485, 0.0680]),
    ("R_LEG_HIP_ROLL",     1, [0.0423, 0.0, 0.0400]),
    ("R_LEG_HIP_YAW",      2, [-0.0422, -0.0520, 0.0]),
    ("R_LEG_KNEE",          3, [0.0, -0.1956, 0.0]),
    ("R_LEG_ANKLE_PITCH",   4, [0.0, -0.2947, 0.0]),
    ("R_LEG_ANKLE_ROLL",    5, [-0.0015, -0.0100, 0.0]),
    ("L_LEG_HIP_PITCH",     0, [0.0059, -0.0485, -0.0670]),
    ("L_LEG_HIP_ROLL",      7, [0.0423, -0.0, -0.0419]),
    ("L_LEG_HIP_YAW",       8, [-0.0422, -0.0520, 0.0]),
    ("L_LEG_KNEE",           9, [0.0, -0.1956, 0.0]),
    ("L_LEG_ANKLE_PITCH",   10, [0.0, -0.2947, 0.0]),
    ("L_LEG_ANKLE_ROLL",    11, [0.0, -0.0100, 0.0]),
    ("WAIST_YAW",            0, [0.0059, 0.0703, 0.0005]),
    ("HEAD_YAW",            13, [-0.0166, 0.3762, 0.0]),
    ("HEAD_PITCH",          14, [-0.0275, 0.0774, -0.0001]),
    ("R_SHOULDER_PITCH",    13, [-0.0262, 0.2591, 0.0965]),
    ("R_SHOULDER_ROLL",     16, [0.0, -0.0001, 0.0654]),
    ("R_SHOULDER_YAW",      17, [0.0, -0.1276, 0.0188]),
    ("R_ELBOW",             18, [0.0, -0.0956, 0.0141]),
    ("R_WRIST_YAW",         19, [0.0, -0.1054, 0.0155]),
    ("L_SHOULDER_PITCH",    13, [-0.0262, 0.2591, -0.0984]),
    ("L_SHOULDER_ROLL",     21, [0.0, -0.0001, -0.0654]),
    ("L_SHOULDER_YAW",      22, [0.0, -0.1276, -0.0188]),
    ("L_ELBOW",             23, [0.0, -0.0956, -0.0141]),
    ("L_WRIST_YAW",         24, [0.0, -0.1054, -0.0155]),
]

NUM_JOINTS = 26


def compute_world_positions(joints_data):
    """Compute world-space joint positions by accumulating local translations
    along the parent chain. Since the GLB rest pose has identity rotations for
    all joints, world position = sum of local translations from root to joint."""
    world_pos = torch.zeros(NUM_JOINTS, 3, dtype=torch.float64)

    for idx, (name, parent_idx, local_t) in enumerate(joints_data):
        local_t_tensor = torch.tensor(local_t, dtype=torch.float64)
        if parent_idx == -1:
            # Root joint
            world_pos[idx] = local_t_tensor
        else:
            # World position = parent world position + local translation
            # (rotations are identity, so local translation == world offset)
            world_pos[idx] = world_pos[parent_idx] + local_t_tensor

    return world_pos


def main():
    assert len(joints_data) == NUM_JOINTS, (
        f"Expected {NUM_JOINTS} joints, got {len(joints_data)}"
    )

    # Output directory
    out_dir = Path(__file__).resolve().parent.parent / "kimodo" / "assets" / "skeletons" / "asv1skel26"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # 1. Compute world-space joint positions
    joints = compute_world_positions(joints_data)
    print(f"joints.p shape: {joints.shape}, dtype: {joints.dtype}")

    # Sanity checks
    assert joints.shape == (NUM_JOINTS, 3)
    # Root should be at origin
    assert torch.allclose(joints[0], torch.zeros(3, dtype=torch.float64)), "Root not at origin"

    # 2. Save joints.p
    joints_path = out_dir / "joints.p"
    torch.save(joints, joints_path)
    print(f"Saved: {joints_path}")

    # 3. Save bvh_joints.p (same as joints.p for now)
    bvh_joints_path = out_dir / "bvh_joints.p"
    torch.save(joints.clone(), bvh_joints_path)
    print(f"Saved: {bvh_joints_path}")

    # 4. Save rest_pose_local_rot.p - identity rotation matrices
    rest_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(NUM_JOINTS, -1, -1).contiguous()
    assert rest_pose.shape == (NUM_JOINTS, 3, 3)
    rest_pose_path = out_dir / "rest_pose_local_rot.p"
    torch.save(rest_pose, rest_pose_path)
    print(f"Saved: {rest_pose_path}")

    # Print joint names and positions for verification
    print("\nJoint positions (world space, Y-up, meters):")
    print(f"{'Idx':<4} {'Name':<24} {'X':>9} {'Y':>9} {'Z':>9}")
    print("-" * 60)
    for idx, (name, parent_idx, _) in enumerate(joints_data):
        pos = joints[idx]
        print(f"{idx:<4} {name:<24} {pos[0]:9.4f} {pos[1]:9.4f} {pos[2]:9.4f}")

    print(f"\nAll {NUM_JOINTS} joint assets saved to {out_dir}")


if __name__ == "__main__":
    main()
