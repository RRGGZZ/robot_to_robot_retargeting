import os
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix


# =========================
# Config
# =========================
# Hard-coded project root path requested by user.
ROOT_DIR = r"c:\Users\cyber\Desktop\scene"
OUTPUT_XML = os.path.join(ROOT_DIR, "scene.xml")
OBJECTS_DIR = os.path.join(ROOT_DIR, "objects")
TEXTURES_DIR = os.path.join(ROOT_DIR, "textures")

# Treat meshes with many faces as "complex"
COLLISION_FACE_THRESHOLD = 5000
# Keep this ratio for decimated collision mesh
COLLISION_DECIMATE_RATIO = 0.5
# Lower bound to avoid over-decimation
COLLISION_MIN_FACES = 500

# Blender(-Y forward, Z up) -> MuJoCo(X forward, Z up)
# x_mj = -y_bl, y_mj = x_bl, z_mj = z_bl
BLENDER_TO_MUJOCO_ROT = Matrix(
    (
        (0.0, -1.0, 0.0),
        
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
)


def ensure_dirs():
    os.makedirs(OBJECTS_DIR, exist_ok=True)
    os.makedirs(TEXTURES_DIR, exist_ok=True)


def sanitize_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def to_mujoco_pos(blender_world_translation):
    return BLENDER_TO_MUJOCO_ROT @ blender_world_translation


def to_mujoco_quat(blender_world_quat):
    rot_bl = blender_world_quat.to_matrix()
    rot_mj = BLENDER_TO_MUJOCO_ROT @ rot_bl @ BLENDER_TO_MUJOCO_ROT.transposed()
    quat_mj = rot_mj.to_quaternion()  # w, x, y, z
    return quat_mj


def select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def export_obj_with_scale_only(src_obj, export_path):
    # Export mesh with world scale baked, but without world pos/rot baked.
    dup = src_obj.copy()
    dup.data = src_obj.data.copy()
    bpy.context.collection.objects.link(dup)

    world_scale = src_obj.matrix_world.to_scale()
    scale_mat = Matrix.Diagonal((world_scale.x, world_scale.y, world_scale.z, 1.0))
    dup.data.transform(scale_mat)
    dup.matrix_world = Matrix.Identity(4)
    dup.parent = None

    select_only(dup)

    bpy.ops.wm.obj_export(
        filepath=export_path,
        export_selected_objects=True,
        apply_modifiers=True,
        export_materials=False,
        export_uv=True,
        export_normals=True,
        forward_axis='Y',
        up_axis='Z',
    )

    bpy.data.objects.remove(dup, do_unlink=True)


def generate_collision_obj(src_obj, export_path):
    dup = src_obj.copy()
    dup.data = src_obj.data.copy()
    bpy.context.collection.objects.link(dup)

    world_scale = src_obj.matrix_world.to_scale()
    scale_mat = Matrix.Diagonal((world_scale.x, world_scale.y, world_scale.z, 1.0))
    dup.data.transform(scale_mat)
    dup.matrix_world = Matrix.Identity(4)
    dup.parent = None

    face_count = len(dup.data.polygons)
    if face_count > COLLISION_FACE_THRESHOLD:
        ratio = max(COLLISION_MIN_FACES / float(face_count), COLLISION_DECIMATE_RATIO)
        dec = dup.modifiers.new(name="MJCF_Decimate", type='DECIMATE')
        dec.ratio = min(ratio, 1.0)
        dec.use_collapse_triangulate = True

    select_only(dup)

    bpy.ops.wm.obj_export(
        filepath=export_path,
        export_selected_objects=True,
        apply_modifiers=True,
        export_materials=False,
        export_uv=False,
        export_normals=False,
        forward_axis='Y',
        up_axis='Z',
    )

    bpy.data.objects.remove(dup, do_unlink=True)


def save_texture_for_object(obj, tex_name):
    # Priority:
    # 1) First image texture node found in materials
    # 2) Fallback to a generated flat color texture from Principled base color
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                img = node.image
                out_path = os.path.join(TEXTURES_DIR, f"{tex_name}.png")
                img.save_render(filepath=out_path)
                return out_path

    # Fallback: generated 1x1 texture from material base color or white
    color = (1.0, 1.0, 1.0, 1.0)
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf:
            color = bsdf.inputs["Base Color"].default_value
            break

    img = bpy.data.images.new(name=f"{tex_name}_tmp", width=1, height=1, alpha=True, float_buffer=False)
    img.pixels = list(color)
    out_path = os.path.join(TEXTURES_DIR, f"{tex_name}.png")
    img.filepath_raw = out_path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)
    return out_path


def pretty_indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            pretty_indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def main():
    ensure_dirs()

    root = ET.Element("mujoco", {"model": "scene"})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local"})
    ET.SubElement(root, "option", {"gravity": "0 0 -9.81"})

    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.name.startswith("_"):
            continue

        base = sanitize_name(obj.name)
        mesh_name = base
        mat_name = f"{base}_mat"
        tex_name = f"{base}_tex"
        col_mesh_name = f"{base}_col"

        obj_path = os.path.join(OBJECTS_DIR, f"{mesh_name}.obj")
        col_obj_path = os.path.join(OBJECTS_DIR, f"{col_mesh_name}.obj")

        export_obj_with_scale_only(obj, obj_path)
        tex_path = save_texture_for_object(obj, tex_name)

        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": f"objects/{mesh_name}.obj"})
        ET.SubElement(asset, "texture", {"name": tex_name, "type": "2d", "file": f"textures/{os.path.basename(tex_path)}"})
        ET.SubElement(asset, "material", {"name": mat_name, "texture": tex_name})

        need_collision_mesh = len(obj.data.polygons) > COLLISION_FACE_THRESHOLD
        if need_collision_mesh:
            generate_collision_obj(obj, col_obj_path)
            ET.SubElement(asset, "mesh", {"name": col_mesh_name, "file": f"objects/{col_mesh_name}.obj"})

        t_bl = obj.matrix_world.to_translation()
        q_bl = obj.matrix_world.to_quaternion()
        t_mj = to_mujoco_pos(t_bl)
        q_mj = to_mujoco_quat(q_bl)

        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": base,
                "pos": f"{t_mj.x:.6f} {t_mj.y:.6f} {t_mj.z:.6f}",
                "quat": f"{q_mj.w:.8f} {q_mj.x:.8f} {q_mj.y:.8f} {q_mj.z:.8f}",
            },
        )

        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "mesh": mesh_name,
                "material": mat_name,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )

        if need_collision_mesh:
            ET.SubElement(
                body,
                "geom",
                {
                    "type": "mesh",
                    "mesh": col_mesh_name,
                    "material": mat_name,
                    "group": "0",
                },
            )
        else:
            ET.SubElement(
                body,
                "geom",
                {
                    "type": "mesh",
                    "mesh": mesh_name,
                    "material": mat_name,
                    "group": "0",
                },
            )

    pretty_indent(root)
    tree = ET.ElementTree(root)
    tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=False)
    print(f"[OK] scene exported: {OUTPUT_XML}")
    print(f"[OK] objects dir: {OBJECTS_DIR}")
    print(f"[OK] textures dir: {TEXTURES_DIR}")


if __name__ == "__main__":
    main()
