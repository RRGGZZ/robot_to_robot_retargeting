import argparse
import os
import pathlib
import pickle
import sys

import numpy as np
from rich import print
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import RobotMotionViewer, RobotToRobotRetargeting


def _as_wxyz(root_rot, order):
    root_rot = np.asarray(root_rot, dtype=float)
    if order == "wxyz":
        return root_rot
    if order == "xyzw":
        return root_rot[..., [3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion order: {order}")


def load_source_qpos(path, root_rot_order="xyzw"):
    with open(path, "rb") as f:
        motion_data = pickle.load(f)

    fps = motion_data.get("fps", 30)
    if "qpos" in motion_data:
        qpos = np.asarray(motion_data["qpos"], dtype=float)
        if qpos.ndim == 1:
            qpos = qpos[None, :]
        return motion_data, fps, qpos

    root_pos = np.asarray(motion_data["root_pos"], dtype=float)
    root_rot = _as_wxyz(motion_data["root_rot"], root_rot_order)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=float)
    qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=-1)
    return motion_data, fps, qpos


def save_target_motion(
    path,
    target_qpos,
    fps,
    source_motion_path,
    ik_target_frames=None,
    src_robot="unitree_g1",
    tgt_robot="pnd_adam_pro",
):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    target_qpos = np.asarray(target_qpos)
    motion_data = {
        "fps": fps,
        "root_pos": target_qpos[:, :3],
        "root_rot": target_qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": target_qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
        "qpos": target_qpos,
        "root_rot_order": "xyzw",
        "src_robot": src_robot,
        "tgt_robot": tgt_robot,
        "source_motion_path": source_motion_path,
    }
    if ik_target_frames is not None:
        motion_data["ik_target_frames"] = ik_target_frames

    with open(path, "wb") as f:
        pickle.dump(motion_data, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retarget a saved Unitree G1 motion pickle to PND Adam Pro. "
            "The Adam Pro target uses assets/pnd_models/adam_pro/adam_pro.xml "
            "because that MuJoCo model is loadable and matches the supplied URDF."
        )
    )
    parser.add_argument("--g1_motion_path", required=True, help="Input Unitree G1 motion pickle.")
    parser.add_argument("--save_path", required=True, help="Output Adam Pro motion pickle.")
    parser.add_argument("--root_rot_order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--max_iter", type=int, default=15)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--damping", type=float, default=5e-1)
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/g1_to_adam_pro.mp4")
    parser.add_argument("--no_save_ik_targets", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    _, fps, source_qpos = load_source_qpos(args.g1_motion_path, args.root_rot_order)
    end_frame = args.end_frame if args.end_frame is not None else len(source_qpos)
    source_qpos = source_qpos[args.start_frame:end_frame]
    if len(source_qpos) == 0:
        raise ValueError("No source frames selected.")

    retargeter = RobotToRobotRetargeting(
        src_robot="unitree_g1",
        tgt_robot="pnd_adam_pro",
        solver=args.solver,
        damping=args.damping,
        verbose=False,
    )
    retargeter.retargeter.max_iter = args.max_iter

    target_qpos = []
    ik_target_frames = [] if not args.no_save_ik_targets else None
    viewer = None

    if args.visualize:
        viewer = RobotMotionViewer(
            robot_type="pnd_adam_pro",
            motion_fps=fps,
            record_video=args.record_video,
            video_path=args.video_path,
        )

    for qpos in tqdm(source_qpos, desc="Retargeting G1 to Adam Pro"):
        if ik_target_frames is None and viewer is None:
            adam_qpos = retargeter.retarget_qpos(qpos)
            ik_targets = None
        else:
            adam_qpos, ik_targets = retargeter.retarget_qpos(qpos, return_ik_targets=True)

        target_qpos.append(adam_qpos.copy())
        if ik_target_frames is not None:
            ik_target_frames.append(ik_targets)

        if viewer is not None:
            viewer.step(
                root_pos=adam_qpos[:3],
                root_rot=adam_qpos[3:7],
                dof_pos=adam_qpos[7:],
                human_motion_data=ik_targets,
                human_point_scale=0.08,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

    if viewer is not None:
        viewer.close()

    save_target_motion(
        args.save_path,
        np.asarray(target_qpos),
        fps,
        args.g1_motion_path,
        ik_target_frames=ik_target_frames,
    )
    print(f"Saved Adam Pro motion to {args.save_path}")
