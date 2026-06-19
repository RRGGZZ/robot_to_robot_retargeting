import argparse
import pickle
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from general_motion_retargeting import RobotMotionViewer, RobotToRobotRetargeting

from g1_to_adam_pro import _as_wxyz, load_source_qpos


def load_adam_motion(path):
    with open(path, "rb") as f:
        motion_data = pickle.load(f)

    fps = motion_data.get("fps", 30)
    if "qpos" in motion_data:
        qpos = np.asarray(motion_data["qpos"], dtype=float)
    else:
        root_pos = np.asarray(motion_data["root_pos"], dtype=float)
        root_rot = _as_wxyz(motion_data["root_rot"], "xyzw")
        dof_pos = np.asarray(motion_data["dof_pos"], dtype=float)
        qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=-1)
    return motion_data, fps, qpos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Adam Pro motion retargeted from Unitree G1, optionally with G1-derived IK targets."
    )
    parser.add_argument("--adam_motion_path", required=True, help="Output pickle from scripts/g1_to_adam_pro.py.")
    parser.add_argument(
        "--g1_motion_path",
        default=None,
        help="Optional source G1 pickle. Used only if ik_target_frames were not saved in the Adam pickle.",
    )
    parser.add_argument("--g1_root_rot_order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--show_source_body_names", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/g1_to_adam_pro_vis.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    motion_data, fps, adam_qpos = load_adam_motion(args.adam_motion_path)

    ik_target_frames = motion_data.get("ik_target_frames")
    if ik_target_frames is None and args.g1_motion_path is not None:
        _, _, g1_qpos = load_source_qpos(args.g1_motion_path, args.g1_root_rot_order)
        retargeter = RobotToRobotRetargeting(src_robot="unitree_g1", tgt_robot="pnd_adam_pro", verbose=False)
        ik_target_frames = []
        for qpos in g1_qpos[: len(adam_qpos)]:
            _, ik_targets = retargeter.retarget_qpos(qpos, return_ik_targets=True)
            ik_target_frames.append(ik_targets)

    viewer = RobotMotionViewer(
        robot_type="pnd_adam_pro",
        motion_fps=fps,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    frame_idx = 0
    while True:
        overlay = ik_target_frames[frame_idx] if ik_target_frames is not None else None
        qpos = adam_qpos[frame_idx]
        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=overlay,
            show_human_body_name=args.show_source_body_names,
            human_point_scale=0.08,
            rate_limit=args.rate_limit,
            follow_camera=True,
        )

        frame_idx += 1
        if frame_idx >= len(adam_qpos):
            if not args.loop:
                break
            frame_idx = 0

    viewer.close()
