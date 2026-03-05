from __future__ import annotations

import gzip
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import bpy


bpy.ops.wm.read_factory_settings(use_empty=True)


def excepthook(exc_type, exc_value, exc_traceback):
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    exit(1)


sys.excepthook = excepthook


ZIP_PATH = Path(r"D:/devel/rust/bevy_test/dump.zip")
WORLD_SCALE = 5000.0


def read_gzip_json(zf: zipfile.ZipFile, name: str) -> Any:
    with zf.open(name, "r") as fh:
        return json.loads(gzip.decompress(fh.read()).decode("utf-8"))


def rgba8_to_float(color: dict[str, Any] | None) -> tuple[float, float, float, float]:
    if not color:
        return (1.0, 1.0, 1.0, 1.0)
    return (
        float(color.get("r", 255)) / 255.0,
        float(color.get("g", 255)) / 255.0,
        float(color.get("b", 255)) / 255.0,
        float(color.get("a", 255)) / 255.0,
    )


def pascal_string(value: Any, default: str = "") -> str:
    if isinstance(value, dict):
        return str(value.get("string", default))
    if value is None:
        return default
    return str(value)


def normalize_key(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def without_extension(path: str) -> str:
    p = PurePosixPath(path.replace("\\", "/"))
    if p.suffix:
        p = p.with_suffix("")
    return p.as_posix()


def shader_image_for_register(
    shader_info: dict[str, Any],
    register: str,
    role_images: dict[str, bpy.types.Image],
) -> bpy.types.Image | None:
    register = register.lower()
    semantic = ""
    for inp in shader_info.get("inputs", []):
        if str(inp.get("register", "")).lower() == register:
            semantic = str(inp.get("semantic", "")).lower()
            break

    role_hints = [
        ("diffuse", "diffuse"),
        ("env", "reflection"),
        ("reflection", "reflection"),
        ("glow", "glow"),
        ("bump", "bump"),
        ("normal", "bump"),
        ("lightmap", "lightmap"),
        ("noise", "noise"),
        ("scan", "scans"),
    ]
    for token, role in role_hints:
        if token in semantic and role in role_images:
            return role_images[role]

    if register.startswith("t"):
        try:
            idx = int(register[1:])
        except ValueError:
            idx = -1
        register_fallback = {
            0: "diffuse",
            1: "reflection",
            2: "glow",
            3: "lightmap",
        }
        role = register_fallback.get(idx)
        if role and role in role_images:
            return role_images[role]
    return None


def make_math_node(nodes: bpy.types.Nodes, op: str):
    node = nodes.new("ShaderNodeVectorMath")
    op = op.lower()
    if op == "add":
        node.operation = "ADD"
    elif op == "sub":
        node.operation = "SUBTRACT"
    elif op == "mul":
        node.operation = "MULTIPLY"
    elif op == "dp3":
        node.operation = "DOT_PRODUCT"
    else:
        node.operation = "ADD"
    return node


def build_expression_nodes(
    node_tree: bpy.types.NodeTree,
    shader_info: dict[str, Any],
    role_images: dict[str, bpy.types.Image],
):
    expr = shader_info.get("expression_tree")
    if not isinstance(expr, dict):
        return None

    node_defs = {int(v["id"]): v for v in expr.get("nodes", []) if "id" in v}
    output_id = expr.get("outputs", {}).get("r0")
    if output_id is None:
        return None

    nodes = node_tree.nodes
    links = node_tree.links
    built: dict[int, bpy.types.Node] = {}

    def resolve(node_id: int):
        if node_id in built:
            return built[node_id]
        node_def = node_defs[node_id]
        kind = str(node_def.get("kind", ""))

        if kind == "input":
            register = str(node_def.get("register", "")).lower()
            if register.startswith("t"):
                tex = nodes.new("ShaderNodeTexImage")
                image = shader_image_for_register(shader_info, register, role_images)
                if image is not None:
                    tex.image = image
                built[node_id] = tex
                return tex
            rgb = nodes.new("ShaderNodeRGB")
            rgb.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            built[node_id] = rgb
            return rgb

        if kind == "operation":
            args = [int(v) for v in node_def.get("args", [])]
            op = str(node_def.get("op", "add"))
            if op == "tex" and args:
                built[node_id] = resolve(args[0])
                return built[node_id]
            if op == "mad" and len(args) >= 3:
                mul_node = make_math_node(nodes, "mul")
                add_node = make_math_node(nodes, "add")
                a = resolve(args[0])
                b = resolve(args[1])
                c = resolve(args[2])
                links.new(a.outputs[0], mul_node.inputs[0])
                links.new(b.outputs[0], mul_node.inputs[1])
                links.new(mul_node.outputs[0], add_node.inputs[0])
                links.new(c.outputs[0], add_node.inputs[1])
                built[node_id] = add_node
                return add_node

            op_node = make_math_node(nodes, op)
            if args:
                links.new(resolve(args[0]).outputs[0], op_node.inputs[0])
            if len(args) > 1:
                links.new(resolve(args[1]).outputs[0], op_node.inputs[1])
            built[node_id] = op_node
            return op_node

        passthrough_source = node_def.get("input")
        if passthrough_source is None:
            passthrough_source = node_def.get("value", node_def.get("base"))
        if passthrough_source is not None:
            built[node_id] = resolve(int(passthrough_source))
            return built[node_id]

        rgb = nodes.new("ShaderNodeRGB")
        built[node_id] = rgb
        return rgb

    return resolve(int(output_id))


def get_vertex_data(tri_data: dict[str, Any]) -> list[dict[str, Any]]:
    geom = tri_data.get("geometry_verts") or {}
    geom_inner = geom.get("inner") if isinstance(geom, dict) else None
    if isinstance(geom_inner, dict) and isinstance(geom_inner.get("data"), list):
        return geom_inner["data"]
    lm = tri_data.get("lightmap_verts") or {}
    lm_inner = lm.get("inner") if isinstance(lm, dict) else None
    if isinstance(lm_inner, dict) and isinstance(lm_inner.get("data"), list):
        return lm_inner["data"]
    return []


def uv_from_vertex(vertex: dict[str, Any], field: str) -> tuple[float, float] | None:
    tex = vertex.get(field)
    if isinstance(tex, list) and tex and isinstance(tex[0], list) and len(tex[0]) >= 2:
        return (float(tex[0][0]), 1.0 - float(tex[0][1]))
    return None


def build_material(
    mat_key: int,
    mat_data: dict[str, Any],
    shader_info: dict[str, Any] | None,
    role_images: dict[str, bpy.types.Image],
) -> bpy.types.Material:
    mat_name = pascal_string(mat_data.get("name"), default=f"MAT:{mat_key}")
    material = bpy.data.materials.new(name=mat_name)
    material.use_nodes = True

    node_tree = material.node_tree
    assert node_tree is not None
    nodes = node_tree.nodes
    links = node_tree.links

    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return material

    bsdf.inputs["Base Color"].default_value = rgba8_to_float(mat_data.get("diffuse"))
    bsdf.inputs["Emission Color"].default_value = rgba8_to_float(mat_data.get("glow"))

    if "diffuse" in role_images:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = role_images["diffuse"]
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        if tex.outputs.get("Alpha") is not None:
            links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])

    if "glow" in role_images:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = role_images["glow"]
        links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])

    if "bump" in role_images:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = role_images["bump"]
        normal = nodes.new("ShaderNodeNormalMap")
        links.new(tex.outputs["Color"], normal.inputs["Color"])
        links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])

    if "reflection" in role_images:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = role_images["reflection"]
        links.new(tex.outputs["Color"], bsdf.inputs["Metallic"])

    if shader_info:
        shader_node = build_expression_nodes(node_tree, shader_info, role_images)
        if shader_node is not None:
            try:
                links.new(shader_node.outputs[0], bsdf.inputs["Base Color"])
                links.new(shader_node.outputs[0], bsdf.inputs["Emission Color"])
            except Exception as exc:
                print(f"Shader node wiring failed for '{mat_name}': {exc}")

    return material


def load_image_from_zip(zf: zipfile.ZipFile, zip_member: str) -> bpy.types.Image:
    data = zf.read(zip_member)
    image = bpy.data.images.new(zip_member, 0, 0)
    image.pack(data=data, data_len=len(data))
    image.name = zip_member
    image.source = "FILE"
    return image


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = set(zf.namelist())

        level = read_gzip_json(zf, "level.json.gz")
        textures_map: dict[str, Any] = read_gzip_json(zf, "textures.json.gz")
        shader_entries = read_gzip_json(zf, "shaders.json.gz") if "shaders.json.gz" in names else []

        shader_by_material_key: dict[int, dict[str, Any]] = {}
        for entry in shader_entries:
            if isinstance(entry, dict) and isinstance(entry.get("material_key"), int):
                shader_by_material_key[int(entry["material_key"])] = entry

        images_by_texture_key: dict[str, bpy.types.Image] = {}
        for texture_key in textures_map.keys():
            if not isinstance(texture_key, str):
                continue
            tex_rel = f"tex/{without_extension(texture_key)}.png"
            if tex_rel in names:
                images_by_texture_key[normalize_key(texture_key)] = load_image_from_zip(zf, tex_rel)

        for member in names:
            if not member.startswith("tex/") or not member.endswith(".png"):
                continue
            norm_rel = normalize_key(member[len("tex/") : -len(".png")])
            if norm_rel not in images_by_texture_key:
                images_by_texture_key[norm_rel] = load_image_from_zip(zf, member)

    materials_list = level["emi"]["materials"]
    materials_by_key: dict[int, dict[str, Any]] = {
        int(item[0]): item[1] for item in materials_list if isinstance(item, list) and len(item) == 2
    }

    lightmap_by_map_key: dict[int, tuple[str, str]] = {}
    for map_entry in level["emi"].get("maps", []):
        key = map_entry.get("key")
        data = map_entry.get("data")
        if not isinstance(key, int) or not isinstance(data, list) or len(data) != 3:
            continue
        lightmap_by_map_key[key] = (pascal_string(data[0]), pascal_string(data[2]))

    material_cache: dict[tuple[int, int], bpy.types.Material] = {}

    for tri in level["emi"].get("tri", []):
        tri_name = pascal_string(tri.get("name"), default="mesh")
        tri_data = tri.get("data", {})
        faces = tri_data.get("tris", [])
        vertices_data = get_vertex_data(tri_data)
        if not vertices_data or not faces:
            continue

        vertices = []
        for v in vertices_data:
            xyz = v.get("xyz")
            if not isinstance(xyz, list) or len(xyz) < 3:
                continue
            x = float(xyz[0]) / WORLD_SCALE
            y = float(xyz[2]) / WORLD_SCALE
            z = float(xyz[1]) / WORLD_SCALE
            vertices.append((x, y, z))
        if len(vertices) < 3:
            continue

        mesh = bpy.data.meshes.new(tri_name)
        mesh.from_pydata(vertices, [], [tuple(face) for face in faces])

        uv_tex = mesh.uv_layers.new(name="tex")
        uv_lm = mesh.uv_layers.new(name="lightmap")
        for i, loop in enumerate(mesh.loops):
            vtx = vertices_data[loop.vertex_index]
            uv0 = uv_from_vertex(vtx, "tex_0")
            if uv0 is not None:
                uv_tex.data[i].uv = uv0
            uv1 = uv_from_vertex(vtx, "tex_1")
            if uv1 is not None:
                uv_lm.data[i].uv = uv1

        obj = bpy.data.objects.new(tri_name, mesh)

        mat_key = int(tri_data.get("mat_key", 0))
        map_key = int(tri_data.get("map_key", 0))
        cache_key = (mat_key, map_key)
        if cache_key not in material_cache and mat_key in materials_by_key:
            mat_data = materials_by_key[mat_key]
            maps = mat_data.get("maps", [])

            role_images: dict[str, bpy.types.Image] = {}
            role_to_idx = {
                "diffuse": 0,
                "metallic": 1,
                "reflection": 2,
                "bump": 3,
                "glow": 4,
            }
            for role, idx in role_to_idx.items():
                if idx >= len(maps) or not isinstance(maps[idx], dict):
                    continue
                map_tex = maps[idx].get("texture")
                tex_name = pascal_string(map_tex)
                if not tex_name:
                    continue
                image = images_by_texture_key.get(normalize_key(tex_name))
                if image is not None:
                    role_images[role] = image

            if map_key in lightmap_by_map_key:
                lm1, lm2 = lightmap_by_map_key[map_key]
                lm_img = images_by_texture_key.get(normalize_key(lm1))
                if lm_img is None:
                    lm_img = images_by_texture_key.get(normalize_key(lm2))
                if lm_img is not None:
                    role_images["lightmap"] = lm_img

            shader_info = shader_by_material_key.get(mat_key)
            material_cache[cache_key] = build_material(mat_key, mat_data, shader_info, role_images)

        if cache_key in material_cache:
            obj.active_material = material_cache[cache_key]

        assert bpy.context.scene is not None
        bpy.context.scene.collection.objects.link(obj)

    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath="blender_import.blend", compress=True)


if __name__ == "__main__":
    main()
