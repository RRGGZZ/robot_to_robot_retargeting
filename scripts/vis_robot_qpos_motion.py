import argparse
import pickle
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import RobotMotionViewer


def load_qpos_motion(path, root_rot_order="xyzw"):
    with open(path, "rb") as f:
        motion_data = pickle.load(f)

    fps = motion_data.get("fps", 30)
    if "qpos" in motion_data:
        qpos = np.asarray(motion_data["qpos"], dtype=np.float64)
    else:
        root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
        root_rot = np.asarray(motion_data["root_rot"], dtype=np.float64)
        if root_rot_order == "xyzw":
            root_rot = root_rot[:, [3, 0, 1, 2]]
        elif root_rot_order != "wxyz":
            raise ValueError(f"Unsupported root_rot_order: {root_rot_order}")
        dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float64)
        qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=-1)

    if qpos.ndim == 1:
        qpos = qpos[None, :]
    return motion_data, fps, qpos


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a saved robot qpos motion with the native robot mesh.")
    parser.add_argument("--motion_path", required=True)
    parser.add_argument("--robot_type", required=True)
    parser.add_argument("--root_rot_order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/robot_qpos_motion.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _, fps, qpos = load_qpos_motion(args.motion_path, args.root_rot_order)

    viewer = RobotMotionViewer(
        robot_type=args.robot_type,
        motion_fps=fps,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    frame_idx = 0
    while True:
        frame = qpos[frame_idx]
        viewer.step(
            root_pos=frame[:3],
            root_rot=frame[3:7],
            dof_pos=frame[7:],
            rate_limit=args.rate_limit,
            follow_camera=True,
        )

        frame_idx += 1
        if frame_idx >= len(qpos):
            if not args.loop:
                break
            frame_idx = 0

    viewer.close()
