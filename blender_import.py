from typing import Any
import bpy
import bmesh
import zipfile
import typing
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

def rgba(*,red: float, green: float, blue: float, alpha: float) -> tuple[float, float, float, float]:
    return (red,green,blue,alpha)

def get_socket(sockets, name, dtype):
    return list(filter(lambda i: (i.type, i.name) == (dtype, name), sockets))

#TODO: crate material nodes

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