import argparse
import os
import pickle
from pathlib import Path

import numpy as np


# SONIC reference CSVs store G1 joints in IsaacLab order. GMR's Unitree G1
# MuJoCo model expects qpos[7:] in MuJoCo joint order.
ISAACLAB_TO_MUJOCO_DOF = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)


def load_csv_array(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a SONIC G1 reference-motion directory to a GMR Unitree G1 pickle."
    )
    parser.add_argument("--motion_dir", required=True, help="Directory containing joint_pos/body_pos/body_quat CSVs.")
    parser.add_argument("--save_path", required=True, help="Output GMR Unitree G1 pickle path.")
    parser.add_argument("--fps", type=float, default=50.0, help="SONIC reference motions are normally 50 Hz.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    motion_dir = Path(args.motion_dir)

    joint_pos_isaaclab = load_csv_array(motion_dir / "joint_pos.csv")
    body_pos_flat = load_csv_array(motion_dir / "body_pos.csv")
    body_quat_flat = load_csv_array(motion_dir / "body_quat.csv")

    if joint_pos_isaaclab.ndim != 2 or joint_pos_isaaclab.shape[1] != 29:
        raise ValueError(f"joint_pos.csv must have shape [T, 29], got {joint_pos_isaaclab.shape}")
    if body_pos_flat.ndim != 2 or body_pos_flat.shape[1] < 3:
        raise ValueError(f"body_pos.csv must have at least 3 columns, got {body_pos_flat.shape}")
    if body_quat_flat.ndim != 2 or body_quat_flat.shape[1] < 4:
        raise ValueError(f"body_quat.csv must have at least 4 columns, got {body_quat_flat.shape}")
    if not (joint_pos_isaaclab.shape[0] == body_pos_flat.shape[0] == body_quat_flat.shape[0]):
        raise ValueError(
            "Frame count mismatch: "
            f"joint_pos={joint_pos_isaaclab.shape[0]}, "
            f"body_pos={body_pos_flat.shape[0]}, "
            f"body_quat={body_quat_flat.shape[0]}"
        )

    root_pos = body_pos_flat[:, :3]
    root_rot_wxyz = body_quat_flat[:, :4]
    root_rot_norm = np.linalg.norm(root_rot_wxyz, axis=1, keepdims=True)
    if np.any(root_rot_norm < 1e-8):
        raise ValueError("Encountered a near-zero root quaternion in body_quat.csv")
    root_rot_wxyz = root_rot_wxyz / root_rot_norm

    dof_pos_mujoco = joint_pos_isaaclab[:, ISAACLAB_TO_MUJOCO_DOF]
    qpos = np.concatenate([root_pos, root_rot_wxyz, dof_pos_mujoco], axis=1)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    motion_data = {
        "fps": float(args.fps),
        "root_pos": root_pos,
        "root_rot": root_rot_wxyz[:, [1, 2, 3, 0]],  # xyzw, matching GMR pickle convention
        "dof_pos": dof_pos_mujoco,
        "qpos": qpos,
        "local_body_pos": None,
        "link_body_list": None,
        "root_rot_order": "xyzw",
        "src_robot": "unitree_g1",
        "source_format": "sonic_reference_csv",
        "source_motion_dir": str(motion_dir),
        "source_joint_order": "isaaclab",
        "dof_pos_order": "mujoco_qpos",
    }

    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"Saved GMR Unitree G1 motion to {args.save_path}")
    print(f"Frames: {qpos.shape[0]}, fps: {args.fps:g}, qpos shape: {qpos.shape}")
