import argparse
import copy
import pathlib
import pickle
import sys
import time
import xml.etree.ElementTree as ET

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import ROBOT_BASE_DICT, ROBOT_XML_DICT


REF_ATTRS = {
    "body",
    "body1",
    "body2",
    "camera",
    "class",
    "childclass",
    "joint",
    "material",
    "mesh",
    "objname",
    "site",
    "target",
    "texture",
}


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


def prefixed(value, prefix):
    if not value:
        return value
    return f"{prefix}{value}"


def mesh_dir_for(xml_path, xml_root):
    compiler = xml_root.find("compiler")
    meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    mesh_dir = pathlib.Path(meshdir)
    if not mesh_dir.is_absolute():
        mesh_dir = xml_path.parent / mesh_dir
    return mesh_dir.resolve()


def prefix_element(element, prefix, mesh_dir=None):
    element = copy.deepcopy(element)
    for node in element.iter():
        if "name" in node.attrib:
            node.attrib["name"] = prefixed(node.attrib["name"], prefix)

        for attr in REF_ATTRS:
            if attr in node.attrib:
                node.attrib[attr] = prefixed(node.attrib[attr], prefix)

        if mesh_dir is not None and node.tag == "mesh" and "file" in node.attrib:
            mesh_path = pathlib.Path(node.attrib["file"])
            if not mesh_path.is_absolute():
                mesh_path = mesh_dir / mesh_path
            node.attrib["file"] = str(mesh_path.resolve())
    return element


def merge_defaults(target_default, source_root, prefix):
    for default in source_root.findall("default"):
        for child in list(default):
            target_default.append(prefix_element(child, prefix))


def merge_assets(target_asset, source_root, prefix, mesh_dir):
    for asset in source_root.findall("asset"):
        for child in list(asset):
            if child.tag == "mesh":
                target_asset.append(prefix_element(child, prefix, mesh_dir=mesh_dir))
            elif child.tag in {"material", "texture"} and "name" in child.attrib:
                target_asset.append(prefix_element(child, prefix))


def append_robot_bodies(target_worldbody, source_root, prefix):
    robot_body_count = 0
    for worldbody in source_root.findall("worldbody"):
        for child in list(worldbody):
            if child.tag == "body":
                target_worldbody.append(prefix_element(child, prefix))
                robot_body_count += 1

    if robot_body_count == 0:
        raise ValueError("No top-level robot body found in MJCF.")


def build_dual_robot_xml(g1_xml_path, adam_xml_path, g1_y_offset, adam_y_offset):
    g1_xml_path = pathlib.Path(g1_xml_path).resolve()
    adam_xml_path = pathlib.Path(adam_xml_path).resolve()

    g1_root = ET.parse(g1_xml_path).getroot()
    adam_root = ET.parse(adam_xml_path).getroot()

    root = ET.Element("mujoco", {"model": "g1_adam_pro_comparison"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    ET.SubElement(root, "statistic", {"center": "0 0 1.0", "extent": "2.4"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.15 0.15 0.15"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "-90", "elevation": "-18", "offwidth": "1920", "offheight": "1080"})

    default = ET.SubElement(root, "default")
    merge_defaults(default, g1_root, "g1_")
    merge_defaults(default, adam_root, "adam_")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "comparison_ground",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.86 0.86 0.86",
            "rgb2": "0.68 0.72 0.76",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "comparison_ground",
            "texture": "comparison_ground",
            "texuniform": "true",
            "texrepeat": "8 8",
            "reflectance": "0.05",
        },
    )

    merge_assets(asset, g1_root, "g1_", mesh_dir_for(g1_xml_path, g1_root))
    merge_assets(asset, adam_root, "adam_", mesh_dir_for(adam_xml_path, adam_root))

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "comparison_floor",
            "type": "plane",
            "size": "0 0 0.01",
            "material": "comparison_ground",
            "friction": "0.8",
            "condim": "3",
        },
    )
    ET.SubElement(worldbody, "light", {"pos": "-3 -4 6", "dir": "3 4 -6", "diffuse": "0.7 0.7 0.7"})
    append_robot_bodies(worldbody, g1_root, "g1_")
    append_robot_bodies(worldbody, adam_root, "adam_")

    return ET.tostring(root, encoding="unicode")


def validate_qpos(qpos, model, robot_name):
    if qpos.shape[1] != model.nq:
        raise ValueError(f"{robot_name} qpos has width {qpos.shape[1]}, but {robot_name} model nq is {model.nq}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Unitree G1 source motion and Adam Pro retargeted motion together.")
    parser.add_argument("--g1_motion_path", required=True)
    parser.add_argument("--adam_motion_path", required=True)
    parser.add_argument("--g1_root_rot_order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--adam_root_rot_order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--g1_y_offset", type=float, default=0.75)
    parser.add_argument("--adam_y_offset", type=float, default=-0.75)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--check_only", action="store_true", help="Build the combined model and print dimensions without opening the viewer.")
    parser.add_argument("--dump_xml_path", default=None, help="Optional path to save the generated combined MJCF.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    _, g1_fps, g1_qpos = load_qpos_motion(args.g1_motion_path, args.g1_root_rot_order)
    _, adam_fps, adam_qpos = load_qpos_motion(args.adam_motion_path, args.adam_root_rot_order)
    motion_fps = float(g1_fps if g1_fps is not None else adam_fps)
    if abs(float(g1_fps) - float(adam_fps)) > 1e-6:
        print(f"Warning: G1 fps={g1_fps}, Adam fps={adam_fps}; using G1 fps={motion_fps}.")

    g1_xml_path = ROBOT_XML_DICT["unitree_g1"]
    adam_xml_path = ROBOT_XML_DICT["pnd_adam_pro"]
    g1_model = mj.MjModel.from_xml_path(str(g1_xml_path))
    adam_model = mj.MjModel.from_xml_path(str(adam_xml_path))
    validate_qpos(g1_qpos, g1_model, "G1")
    validate_qpos(adam_qpos, adam_model, "Adam Pro")

    combined_xml = build_dual_robot_xml(g1_xml_path, adam_xml_path, args.g1_y_offset, args.adam_y_offset)
    if args.dump_xml_path is not None:
        dump_path = pathlib.Path(args.dump_xml_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(combined_xml)

    model = mj.MjModel.from_xml_string(combined_xml)
    data = mj.MjData(model)
    print(
        f"Loaded comparison scene: G1 nq={g1_model.nq}, Adam nq={adam_model.nq}, "
        f"combined nq={model.nq}, frames={min(len(g1_qpos), len(adam_qpos))}, fps={motion_fps}."
    )
    if args.check_only:
        sys.exit(0)

    viewer = mjv.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False)
    viewer.cam.lookat[:] = np.array([0.0, (args.g1_y_offset + args.adam_y_offset) * 0.5, 1.0])
    viewer.cam.distance = 3.4
    viewer.cam.elevation = -12
    viewer.cam.azimuth = -90

    g1_base_body = model.body(f"g1_{ROBOT_BASE_DICT['unitree_g1']}").id
    adam_base_body = model.body(f"adam_{ROBOT_BASE_DICT['pnd_adam_pro']}").id
    rate_limiter = RateLimiter(frequency=motion_fps, warn=False)

    frame_idx = 0
    frame_count = min(len(g1_qpos), len(adam_qpos))
    while viewer.is_running():
        g1_frame = g1_qpos[frame_idx].copy()
        adam_frame = adam_qpos[frame_idx].copy()
        g1_frame[1] += args.g1_y_offset
        adam_frame[1] += args.adam_y_offset

        data.qpos[: g1_model.nq] = g1_frame
        data.qpos[g1_model.nq : g1_model.nq + adam_model.nq] = adam_frame
        mj.mj_forward(model, data)

        viewer.cam.lookat[:] = 0.5 * (data.xpos[g1_base_body] + data.xpos[adam_base_body])
        viewer.sync()

        frame_idx += 1
        if frame_idx >= frame_count:
            if not args.loop:
                break
            frame_idx = 0

        if args.rate_limit:
            rate_limiter.sleep()

    viewer.close()
    time.sleep(0.2)
