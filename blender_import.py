import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import bpy


def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Import level_viewer JSON+textures dump.zip")
    parser.add_argument("--zip", dest="zip_path", required=True, help="Path to dump.zip")
    parser.add_argument(
        "--output",
        dest="output_path",
        default="blender_import.blend",
        help="Output .blend path",
    )
    return parser.parse_args(argv)


def load_level(zf: zipfile.ZipFile) -> dict[str, Any]:
    return json.loads(zf.read("level/level.json").decode("utf-8"))


def load_images(zf: zipfile.ZipFile, temp_dir: Path) -> dict[str, bpy.types.Image]:
    images: dict[str, bpy.types.Image] = {}
    for member in zf.namelist():
        if not member.lower().endswith(".png"):
            continue
        if not (member.startswith("tex/") or member.startswith("lightmaps/")):
            continue

        key = member.split("/", 1)[1].removesuffix(".png")
        out_path = temp_dir / member
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(zf.read(member))

        img = bpy.data.images.load(filepath=str(out_path), check_existing=True)
        img.name = member
        images[key] = img
    return images


def rgba_u8_to_f32(v: dict[str, Any] | None) -> tuple[float, float, float, float]:
    if not isinstance(v, dict):
        return (1.0, 1.0, 1.0, 1.0)
    return (
        float(v.get("r", 255)) / 255.0,
        float(v.get("g", 255)) / 255.0,
        float(v.get("b", 255)) / 255.0,
        float(v.get("a", 255)) / 255.0,
    )


def tex_name_from_mat(mat: dict[str, Any], idx: int) -> str | None:
    maps = mat.get("maps")
    if not isinstance(maps, list) or idx >= len(maps):
        return None
    entry = maps[idx]
    if not isinstance(entry, dict):
        return None
    name = entry.get("texture")
    if isinstance(name, str) and name:
        return name
    return None


def extract_texcoords(vertex: dict[str, Any], field: str) -> tuple[float, float] | None:
    value = vertex.get(field)
    if value is None:
        return None

    if isinstance(value, list) and len(value) >= 2:
        return (float(value[0]), float(value[1]))

    if isinstance(value, dict):
        inner = value.get("0")
        if isinstance(inner, list) and len(inner) >= 2:
            return (float(inner[0]), float(inner[1]))

    return None


def build_material(
    material_name: str,
    mat: dict[str, Any],
    images: dict[str, bpy.types.Image],
    lightmap_name: str | None,
) -> bpy.types.Material:
    out_mat = bpy.data.materials.new(name=material_name)
    out_mat.use_nodes = True
    nodes = out_mat.node_tree.nodes
    links = out_mat.node_tree.links

    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (540, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (260, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = rgba_u8_to_f32(mat.get("diffuse"))
    bsdf.inputs["Emission Color"].default_value = rgba_u8_to_f32(mat.get("glow"))

    props = mat.get("mat_props")
    if isinstance(props, dict):
        out_mat.use_backface_culling = int(props.get("two_sided", 0)) == 0

    uv_tex = nodes.new("ShaderNodeUVMap")
    uv_tex.location = (-940, 120)
    uv_tex.uv_map = "tex"

    base_color_socket = None
    diffuse_name = tex_name_from_mat(mat, 0)
    if diffuse_name and diffuse_name in images:
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.location = (-700, 140)
        tex_node.image = images[diffuse_name]
        links.new(uv_tex.outputs["UV"], tex_node.inputs["Vector"])
        base_color_socket = tex_node.outputs["Color"]
        links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])

    bump_name = tex_name_from_mat(mat, 3)
    if bump_name and bump_name in images:
        bump_tex = nodes.new("ShaderNodeTexImage")
        bump_tex.location = (-700, -220)
        bump_tex.image = images[bump_name]
        bump_tex.image.colorspace_settings.name = "Non-Color"
        bump_node = nodes.new("ShaderNodeNormalMap")
        bump_node.location = (-460, -220)
        links.new(uv_tex.outputs["UV"], bump_tex.inputs["Vector"])
        links.new(bump_tex.outputs["Color"], bump_node.inputs["Color"])
        links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])

    if lightmap_name and lightmap_name in images:
        uv_lm = nodes.new("ShaderNodeUVMap")
        uv_lm.location = (-940, -40)
        uv_lm.uv_map = "lightmap"

        lm_tex = nodes.new("ShaderNodeTexImage")
        lm_tex.location = (-700, -40)
        lm_tex.image = images[lightmap_name]
        links.new(uv_lm.outputs["UV"], lm_tex.inputs["Vector"])

        mul = nodes.new("ShaderNodeMixRGB")
        mul.location = (-260, 20)
        mul.blend_type = "MULTIPLY"
        mul.inputs["Fac"].default_value = 1.0

        if base_color_socket is None:
            base_rgb = nodes.new("ShaderNodeRGB")
            base_rgb.location = (-520, 180)
            base_rgb.outputs["Color"].default_value = rgba_u8_to_f32(mat.get("diffuse"))
            links.new(base_rgb.outputs["Color"], mul.inputs[1])
        else:
            links.new(base_color_socket, mul.inputs[1])
        links.new(lm_tex.outputs["Color"], mul.inputs[2])
        base_color_socket = mul.outputs["Color"]

    if base_color_socket is not None:
        links.new(base_color_socket, bsdf.inputs["Base Color"])

    return out_mat


def get_tri_vertices(tri_data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("geometry_verts", "lightmap_verts"):
        lfvf = tri_data.get(key)
        if not isinstance(lfvf, dict):
            continue
        inner = lfvf.get("inner")
        if not isinstance(inner, dict):
            continue
        verts = inner.get("data")
        if isinstance(verts, list) and verts:
            return [v for v in verts if isinstance(v, dict)]
    return []


def map_by_key(entries: list[Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if (
            isinstance(entry, list)
            and len(entry) == 2
            and isinstance(entry[0], int)
            and isinstance(entry[1], dict)
        ):
            out[int(entry[0])] = entry[1]
    return out


def lightmaps_by_key(entries: list[Any]) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        data = entry.get("data")
        if isinstance(key, int) and isinstance(data, list) and len(data) >= 3:
            first = data[0]
            second = data[2]
            if isinstance(first, str) and isinstance(second, str):
                out[int(key)] = (first, second)
    return out


def import_level(level: dict[str, Any], images: dict[str, bpy.types.Image]) -> None:
    emi = level.get("emi")
    if not isinstance(emi, dict):
        return

    materials = map_by_key(emi.get("materials", []))
    lightmaps = lightmaps_by_key(emi.get("maps", []))
    tris = emi.get("tri", [])
    if not isinstance(tris, list):
        return

    material_cache: dict[tuple[int, str], bpy.types.Material] = {}

    for tri in tris:
        if not isinstance(tri, dict):
            continue
        tri_name = str(tri.get("name", "tri"))
        tri_data = tri.get("data")
        if not isinstance(tri_data, dict):
            continue

        verts = get_tri_vertices(tri_data)
        indices = tri_data.get("tris")
        if not isinstance(indices, list) or not verts:
            continue

        pos = []
        uv0 = []
        uv1 = []
        for vert in verts:
            xyz = vert.get("xyz")
            if not (isinstance(xyz, list) and len(xyz) >= 3):
                pos.append((0.0, 0.0, 0.0))
            else:
                pos.append((float(xyz[0]) / 5000.0, float(xyz[2]) / 5000.0, float(xyz[1]) / 5000.0))

            t0 = extract_texcoords(vert, "tex_0")
            t1 = extract_texcoords(vert, "tex_1")
            uv0.append(t0 or (0.0, 0.0))
            uv1.append(t1 or (0.0, 0.0))

        faces = []
        for face in indices:
            if isinstance(face, list) and len(face) == 3:
                faces.append((int(face[1]), int(face[0]), int(face[2])))
        if not faces:
            continue

        mesh = bpy.data.meshes.new(tri_name)
        mesh.from_pydata(pos, [], faces)

        layer_tex = mesh.uv_layers.new(name="tex")
        layer_lm = mesh.uv_layers.new(name="lightmap")
        for i, loop in enumerate(mesh.loops):
            vi = loop.vertex_index
            if vi < len(uv0):
                u, v = uv0[vi]
                layer_tex.data[i].uv = (float(u), float(v))
            if vi < len(uv1):
                u, v = uv1[vi]
                layer_lm.data[i].uv = (float(u), float(1.0 - v))

        obj = bpy.data.objects.new(tri_name, mesh)

        mat_key = tri_data.get("mat_key")
        mat_key_int = int(mat_key) if isinstance(mat_key, int) else None
        map_key = tri_data.get("map_key")
        mat_data = materials.get(mat_key_int) if mat_key_int is not None else None
        lm_name = None
        if isinstance(map_key, int) and map_key in lightmaps:
            lm_name = lightmaps[map_key][0]

        if mat_data is not None and mat_key_int is not None:
            cache_key = (mat_key_int, lm_name or "")
            if cache_key not in material_cache:
                material_cache[cache_key] = build_material(
                    f"mat_{mat_key}",
                    mat_data,
                    images,
                    lm_name,
                )
            obj.active_material = material_cache[cache_key]

        bpy.context.scene.collection.objects.link(obj)

    bpy.context.scene["scrap_level_path"] = str(level.get("path", ""))
    deps = level.get("dependencies")
    if isinstance(deps, dict):
        bpy.context.scene["scrap_dependency_count"] = int(len(deps))


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    bpy.ops.wm.read_factory_settings(use_empty=True)

    with tempfile.TemporaryDirectory(prefix="level_viewer_blender_") as tmp:
        temp_dir = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            level = load_level(zf)
            images = load_images(zf, temp_dir)
            import_level(level, images)

    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), compress=True)


if __name__ == "__main__":
    main()
