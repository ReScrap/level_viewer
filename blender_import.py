from typing import Any
import bpy
import bmesh
import zipfile
import typing
import gzip
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass

import sys

bpy.ops.wm.read_factory_settings(use_empty=True)

def excepthook(exc_type, exc_value, exc_traceback):
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    exit(1)

sys.excepthook=excepthook

@dataclass
class MaterialInfo:
    depth_bias: str
    base_color: tuple
    emissive: tuple
    unlit: bool
    alpha_mode: str
 

attr_map = {
    "Vertex_Position": "pos",
    "Vertex_Normal": "norm",
    "Vertex_Color": "color",
    "Vertex_Uv": "uv:default",
    "Vertex_Uv_1": "uv:lightmap",
}

dtype_map: dict[str, tuple[tuple[int, int], Any]] = {
    "Float32x2": ((-1, 2), np.float32),
    "Float32x3": ((-1, 3), np.float32),
    "Float32x4": ((-1, 4), np.float32),
}
idx_map: dict[str, tuple[tuple[int, int], Any]] = {
    "IDX32": ((-1, 3), np.uint32),
    "IDX16": ((-1, 3), np.uint16),
}


class DataReader(object):
    path: zipfile.Path

    @classmethod
    def open(cls, path: Path) -> "DataReader":
        return cls(zipfile.Path(path))

    def __init__(self, path: zipfile.Path):
        self.path = path

    def __repr__(self) -> str:
        return f"DataReader({self.path})"

    def __getitem__(self, path: str) -> "DataReader":
        return DataReader(self.path / path)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.path, name)

    def read_json(self) -> Any:
        return json.loads(self.path.read_text())

    def read_json_gz(self) -> Any:
        data = gzip.decompress(self.path.read_bytes())
        return json.loads(data.decode("utf-8"))

    def iterdir(self, recursive=False) -> typing.Iterator["DataReader"]:
        for entry in self.path.iterdir():
            yield DataReader(entry)
            if recursive and entry.is_dir():
                yield from DataReader(entry).iterdir(recursive)
    
    @property
    def zip_path(self) -> str:
        return str(self.path.filename.relative_to(self.path.root.filename))

    def load_image(self) -> bpy.types.Image:
        img_data = (self.path).read_bytes()
        filename = self.zip_path
        print("Loading image:", filename)
        img = bpy.data.images.new(filename, 0, 0)
        img.pack(data=img_data, data_len=len(img_data))
        img.name = filename
        img.source = "FILE"
        return img

zip_path =  r"D:/devel/rust/bevy_test/dump.zip"
objects_info: dict[Any, Any] = {}
root = DataReader.open(Path(zip_path))
images = {}
materials = {}
shader_graphs = {}

def rgba(*,red: float, green: float, blue: float, alpha: float) -> tuple[float, float, float, float]:
    return (red,green,blue,alpha)

def get_socket(sockets, name, dtype):
    return list(filter(lambda i: (i.type, i.name) == (dtype, name), sockets))

def shader_image_for_register(shader_info: dict[str, Any], material_info: dict[str, Any], register: str):
    register = register.lower()
    semantic = None
    for input_info in shader_info.get("inputs", []):
        if input_info.get("register", "").lower() == register:
            semantic = input_info.get("semantic", "").lower()
            break

    key_by_semantic = {
        "diffuse": "tex:diffuse",
        "env": "tex:reflection",
        "glow": "tex:glow",
        "bump": "tex:bump",
        "lightmap": "lm1",
    }
    if semantic:
        for token, key in key_by_semantic.items():
            if token in semantic and key in material_info:
                return material_info.get(key)
    return None

def make_math_node(nodes, op: str):
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

def build_expression_nodes(node_tree, shader_info: dict[str, Any], material_info: dict[str, Any]):
    expr = shader_info.get("expression_tree")
    if not expr:
        return None

    nodes = node_tree.nodes
    links = node_tree.links
    built: dict[int, bpy.types.Node] = {}
    node_defs = {n["id"]: n for n in expr.get("nodes", [])}

    def resolve(node_id: int):
        if node_id in built:
            return built[node_id]

        node_def = node_defs[node_id]
        kind = node_def.get("kind")

        if kind == "input":
            register = node_def["register"].lower()
            if register.startswith("t"):
                tex = nodes.new("ShaderNodeTexImage")
                img = shader_image_for_register(shader_info, material_info, register)
                if img is not None:
                    tex.image = img
                built[node_id] = tex
                return tex
            rgb = nodes.new("ShaderNodeRGB")
            rgb.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            built[node_id] = rgb
            return rgb

        if kind == "operation":
            args = node_def.get("args", [])
            op = node_def.get("op", "add")
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
                a = resolve(args[0])
                links.new(a.outputs[0], op_node.inputs[0])
            if len(args) > 1:
                b = resolve(args[1])
                links.new(b.outputs[0], op_node.inputs[1])
            built[node_id] = op_node
            return op_node

        if kind in ("swizzle", "modifier", "negate", "merge"):
            source_id = node_def.get("input")
            if source_id is None:
                source_id = node_def.get("value", node_def.get("base"))
            built[node_id] = resolve(source_id)
            return built[node_id]

        rgb = nodes.new("ShaderNodeRGB")
        built[node_id] = rgb
        return rgb

    output_id = expr.get("outputs", {}).get("r0")
    if output_id is None:
        return None
    return resolve(output_id)

@typing.no_type_check
def build_material(root: DataReader, name: str, attrs: dict) -> bpy.types.Material:
    info = materials[name]
    get_image = lambda key: key and (images.get(key) or images.get(f"{key}.png"))
    for k,v in list(info.items()):
        if isinstance(v,str) and k.startswith("tex:"):
            info[k] = get_image(v)
            if info[k] is None:
                print("Warning: texture not found:", v)
    active_ligmap = None
    for k in ("lm1","lm2"):
        if lightmap := attrs.get(k):
            info[k] = get_image(lightmap)
            active_ligmap = (k,lightmap)
    print("Material info:", name, info, attrs)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba(**info["base_color"])
    bsdf.inputs["Emission Color"].default_value = rgba(**info["emissive"])
    if active_ligmap:
        img_node = nodes.new("ShaderNodeTexImage")
        img_node.name = active_ligmap[1]
        img_node.image = info[active_ligmap[0]]
    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.uv_map = "lightmap"

    shader_info = shader_graphs.get(name)
    if shader_info:
        out = nodes.get("Material Output")
        shader_node = build_expression_nodes(mat.node_tree, shader_info, info)
        if shader_node is not None:
            try:
                mat.node_tree.links.new(shader_node.outputs[0], bsdf.inputs["Base Color"])
                mat.node_tree.links.new(shader_node.outputs[0], bsdf.inputs["Emission Color"])
            except Exception as exc:
                print("Shader node wiring failed:", name, exc)
    return mat


for lightmap in root["lightmaps"].iterdir():
    if  not lightmap.is_file():
        continue
    key=lightmap.zip_path.replace("\\","/").split("/",1)[1].removesuffix(".png")
    images[key] = lightmap.load_image()

for tex in root["tex"].iterdir(True):
    if not tex.is_file():
        continue
    print("Tex:",tex.zip_path)
    key=tex.zip_path.replace("\\","/").split("/",1)[1].removesuffix(".png")
    images[key] = tex.load_image()

print(images.keys())

for mat in root["mat"].iterdir():
    if not mat.is_file():
        continue
    print("MAT:", mat.stem)
    mat_info = mat.read_json()
    materials[mat.stem.removesuffix(".png")] = mat_info

try:
    shader_entries = root["shaders.json.gz"].read_json_gz()
    shader_graphs = {
        entry.get("material_name"): entry
        for entry in shader_entries
        if entry.get("material_name")
    }
    print("Loaded shader graphs:", len(shader_graphs))
except Exception as exc:
    print("No shader graph payload found:", exc)

for obj in root["obj"].iterdir():
    # print("OBJ:", obj.name)
    objects_info[obj.name] = {}
    for attr in obj.iterdir():
        if idx := idx_map.get(attr.name):
            shape, dtype = idx
            data = np.frombuffer(attr.read_bytes(), dtype=dtype)
            data = data.reshape(shape)
            objects_info[obj.name]["idx"] = data
        elif "." in attr.name:
            name, dtype = attr.name.split(".", 1)
            name = attr_map[name]
            shape, dtype = dtype_map[dtype]
            data = np.frombuffer(attr.read_bytes(), dtype=dtype)
            data = data.reshape(shape)
            objects_info[obj.name][name] = data
        elif attr.name.startswith("_"):
            objects_info[obj.name][attr.name[1:]] = attr.read_text()

for name, attrs in objects_info.items():
    # print("Creating mesh:", name)
    # print("Lightmaps:", attrs.get("lm1"), attrs.get("lm2"))
    me = bpy.data.meshes.new(name)
    attrs["pos"] = attrs["pos"][:,[0,2,1]]
    me.from_pydata(attrs["pos"] / 5000.0, [], attrs["idx"])
    tex_uv = me.uv_layers.new(name="tex")
    lightmap_uv = me.uv_layers.new(name="lightmap")
    for i,loop in enumerate(me.loops):
        if "uv:default" in attrs:
            tex_uv.data[i].uv = attrs["uv:default"][loop.vertex_index]
        if "uv:lightmap" in attrs:
            u,v = attrs["uv:lightmap"][loop.vertex_index]
            lightmap_uv.data[i].uv = (u, 1.0-v)
    ob = bpy.data.objects.new(name, me)
    if mat := attrs.get("mat"):
        ob.active_material = build_material(root, mat, attrs)
    assert bpy.context.scene is not None
    bpy.context.scene.collection.objects.link(ob)

bpy.ops.file.pack_all()

bpy.ops.wm.save_as_mainfile(filepath="blender_import.blend", compress=True)
