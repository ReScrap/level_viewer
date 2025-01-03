use core::f32;
use std::{
    collections::{BTreeMap, HashMap},
    hash::Hash,
    path::PathBuf,
};

use anyhow::{anyhow, bail, Result};
use bevy::{
    app::AppExit,
    core_pipeline::{
        bloom::{Bloom, BloomCompositeMode, BloomPrefilter},
        dof::{DepthOfField, DepthOfFieldMode, DepthOfFieldSettings},
        tonemapping::Tonemapping,
    },
    diagnostic::{
        FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
    },
    ecs::reflect,
    image::{
        CompressedImageFormats, Image, ImageAddressMode, ImageFilterMode, ImageSampler,
        ImageSamplerDescriptor, ImageType,
    },
    log::tracing_subscriber::fmt::format,
    math::{Affine3, Affine3A},
    pbr::{
        wireframe::{Wireframe, WireframeColor, WireframeConfig, WireframePlugin},
        Lightmap,
    },
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
        render_resource::Face,
    },
    utils::warn,
    window::{PresentMode, PrimaryWindow, WindowMode},
};
use bevy_atmosphere::plugin::{AtmosphereCamera, AtmospherePlugin};
use bevy_egui::{
    egui::{self, RichText, ScrollArea, TextureId as EguiTextureId, TextureOptions},
    EguiContexts, EguiPlugin,
};
use bevy_inspector_egui::{quick::WorldInspectorPlugin, reflect_inspector};
use configparser::ini::Ini;
use parser::{multi_pack_fs::MultiPackFS, Data, Level, NodeData, Vertex, LFVF};
use petgraph::{graphmap::GraphMap, Directed};
use regex::Regex;
use smooth_bevy_cameras::{
    controllers::unreal::{UnrealCameraBundle, UnrealCameraController, UnrealCameraPlugin},
    LookTransformPlugin, Smoother,
};
use vfs::VfsPath;

use crate::{
    // materials::Hologram,
    parser::ParsedData,
};
mod find_scrap;
mod materials;
mod packed_vfs;
mod parser;

#[derive(Resource, Debug, Default, Deref, DerefMut)]
struct EguiTexHandles(HashMap<String, (Handle<Image>, EguiTextureId)>);

#[derive(Component, Deref, Debug, Reflect)]
struct MaterialName(String);

#[derive(Clone, Component, Deref, Debug, Reflect)]
struct MapNames(Vec<(Slot, String)>);

#[derive(Clone, Component, Deref, Debug, Reflect)]
struct MapTex(HashMap<Slot, Option<Handle<Image>>>);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
enum Slot {
    Diffuse,
    Metallic,
    Reflection,
    Bump,
    Glow,
}

struct AnimTexture {
    fps: f32,
    images: Vec<Handle<Image>>,
    slot: Slot,
    mat: Handle<StandardMaterial>,
}

type AnimMat = HashMap<u32, Vec<(Slot, f32, Vec<Handle<Image>>)>>;

#[derive(Resource)]
struct State {
    fs: MultiPackFS,
    data_path: Option<String>,
    data: Option<ParsedData>,
    picked_object: Option<Entity>,
    show_ui: bool,
    show_nodes: bool,
    lightmaps: bool,
    dummy_size: f32,
    lightmap_exposure: f32,
    edges: Vec<(Vec3, Vec3)>,
    textures: HashMap<String, Handle<Image>>,
    anim_textures: Vec<AnimTexture>,
}

fn transform_pos(p: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = p;
    [x / -1000.0, y / 1000.0, z / 1000.0]
}

fn animate_textures(
    state: Res<State>,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for tex in &state.anim_textures {
        let frame = (time.elapsed_secs() * tex.fps).floor() as usize;
        let img = tex.images[frame % tex.images.len()].clone();
        let Some(mat) = materials.get_mut(&tex.mat) else {
            continue;
        };
        match &tex.slot {
            Slot::Diffuse => {
                mat.base_color_texture = Some(img);
            }
            Slot::Metallic => {
                mat.metallic_roughness_texture = Some(img);
            }
            Slot::Bump => {
                mat.normal_map_texture = Some(img);
            }
            Slot::Glow => {
                mat.emissive_texture = Some(img);
            }
            Slot::Reflection => (),
        }
    }
}

// fn animate_textures(time: Res<Time>,
//     mut materials: ResMut<Assets<StandardMaterial>>,
//      tex: &[AnimTexture]) {

//     for mesh in &mesh {
//         if let Ok(handle) = mat.get(mesh) {
//             if let Some(mat) = materials.get_mut(handle) {
//                 if state.lightmaps {
//                     mat.lightmap_exposure = state.lightmap_exposure;
//                 } else {
//                     mat.lightmap_exposure = 0.0;
//                 }
//             }
//         }
//     }

//     for tex in tex {
//         match tex.slot {
//             Slot::Diffuse => tex.mat,
//             Slot::Metallic => todo!(),
//             Slot::Reflection => todo!(),
//             Slot::Bump => todo!(),
//             Slot::Glow => todo!(),
//         }
//     }
// }

// TODO: move outside of Bevy app
fn load_data() -> Result<MultiPackFS> {
    let data_regex = Regex::new(r"[Dd]ata\d*\.packed")?;
    let packed_files: Vec<PathBuf> = find_scrap::get_path()
        .and_then(parser::find_packed)
        .expect("Failed to find .packed files")
        .into_iter()
        .filter(|p| {
            let file_name = p
                .file_name()
                .unwrap_or_default()
                .to_str()
                .unwrap_or_default();
            data_regex.is_match(file_name)
        })
        .collect();
    parser::multi_pack_fs::MultiPackFS::new(packed_files)
}

fn main() -> Result<()> {
    better_panic::install();
    let state = State {
        fs: load_data()?,
        data_path: std::env::args().nth(1),
        picked_object: None,
        data: None,
        show_ui: true,
        lightmaps: true,
        show_nodes: true,
        textures: HashMap::default(),
        lightmap_exposure: 50000.0,
        anim_textures: Vec::default(),
        dummy_size: 0.1,
        edges: Vec::default(),
    };
    App::new()
        .insert_resource(AmbientLight {
            // color: Color::WHITE,
            // brightness: 0.0,
            ..default()
        })
        .insert_resource(WireframeConfig {
            global: true,
            default_color: Color::WHITE,
        })
        .insert_resource(EguiTexHandles::default())
        .insert_resource(state)
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        mode: WindowMode::BorderlessFullscreen(MonitorSelection::Current),
                        title: "Scrap Asset Viewer".to_owned(),
                        present_mode: PresentMode::AutoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    mode: AssetMode::Processed,
                    ..default()
                }),
            WireframePlugin,
            MeshPickingPlugin,
            EguiPlugin,
            WorldInspectorPlugin::new().run_if(|state: Res<State>| state.show_ui),
            SystemInformationDiagnosticsPlugin,
            FrameTimeDiagnosticsPlugin,
            LogDiagnosticsPlugin::default(),
            // AtmospherePlugin,
            LookTransformPlugin,
            UnrealCameraPlugin::new(false),
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                ui_input_toggle,
                browser,
                keyboard_handler,
                help_window,
                show_dummies,
                inspector,
                tree_overlay,
                animate_textures,
            ),
        )
        // .add_systems(FixedUpdate, autofocus)
        .init_gizmo_group::<DefaultGizmoConfigGroup>()
        .register_type::<Smoother>()
        .register_type::<UnrealCameraController>()
        .register_type::<Lightmap>()
        .register_type::<Slot>()
        .register_type::<MapNames>()
        .register_type::<MaterialName>()
        .register_type::<Image>()
        .run();
    Ok(())
}

fn node_color(node: &parser::Node) -> Color {
    if let Some(node_data) = node.content.as_ref() {
        match node_data {
            NodeData::Camera(_) => Color::linear_rgb(1., 1., 0.), // #ffff00
            NodeData::Dummy => Color::linear_rgb(1., 1., 1.),     // #ffffff
            NodeData::TriangleMesh(_) => Color::linear_rgb(0., 0., 1.), // #0000ff
            NodeData::D3DMesh(_) => Color::linear_rgb(0., 1., 1.), // #00ffff
            NodeData::Light(_) => Color::linear_rgb(1., 0., 1.),  // #ff00ff
            NodeData::Ground(_) => Color::linear_rgb(0.5, 1., 0.), // #7fff00
            NodeData::SistPart => Color::linear_rgb(1., 0.5, 0.), // #ff7f00
            NodeData::Graphic3D(_) => Color::linear_rgb(0., 1., 0.5), // #00ff7f
            NodeData::Flare => Color::linear_rgb(0., 0.5, 1.),    // #007fff
            NodeData::Portal(_) => Color::linear_rgb(1., 0.5, 1.), // #ff7fff
        }
    } else {
        Color::linear_rgb(1.0, 1.0, 1.0) // #ffffff
    }
}

fn show_text(
    contexts: &mut EguiContexts,
    text: &str,
    pos: Vec3,
    cam: &Camera,
    win: &Window,
    t_cam: &GlobalTransform,
) {
    let ctx = contexts.ctx_mut();
    let c_pos = win.cursor_position();
    if let Ok(screen_pos) = cam.world_to_viewport(t_cam, pos) {
        if screen_pos.x < 0.0 || screen_pos.x > win.width() {
            return;
        }
        if screen_pos.y < 0.0 || screen_pos.y > win.height() {
            return;
        }
        let d_cursor = c_pos
            .map(|p| (screen_pos.xy() - p).length())
            .unwrap_or(f32::INFINITY);
        let d_world = (pos - t_cam.translation()).length();
        if d_cursor < 100.0 && d_world < 5.0 {
            ctx.debug_painter().text(
                [screen_pos.x, screen_pos.y].into(),
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::monospace((20.0 / d_world).clamp(0.0, 20.0)),
                egui::Color32::WHITE,
            );
        }
    };
}

fn show_dummies(
    mut gizmos: Gizmos,
    state: Res<State>,
    cam: Query<(&Camera, &GlobalTransform)>,
    mut contexts: EguiContexts,
    win: Query<&Window, With<PrimaryWindow>>,
) {
    use parser::NodeData;
    if !state.show_nodes {
        return;
    }
    let win = win.single();
    let (cam, t_cam) = cam.single();
    let Some(ParsedData::Level(lvl)) = state.data.as_ref() else {
        return;
    };
    for dum in &lvl.dummies.dummies {
        let name = dum.name.string.as_str();
        let pos: Vec3 = transform_pos(dum.pos).into();
        show_text(&mut contexts, name, pos, cam, win, t_cam);
        gizmos.sphere(
            Isometry3d::new(
                pos,
                Quat::from_euler(EulerRot::XYZ, dum.rot[0], dum.rot[1], dum.rot[2]),
            ),
            state.dummy_size,
            LinearRgba::GREEN,
        );
    }
    for sm3 in &lvl.sm3 {
        let Some(sm3) = sm3.as_ref() else {
            continue;
        };
        for node in &sm3.scene.nodes {
            // LinearRgba::BLUE
            // let t = Affine3A::from_mat4(Mat4::from_cols_array_2d(&node.transform));
            // let t_inv = Affine3A::from_mat4(Mat4::from_cols_array_2d(&node.transform_inv));
            let label = format!("{}: {}", node_type(&node.content), node.name.string);
            show_text(
                &mut contexts,
                &label,
                transform_pos(node.pos_offset).into(),
                cam,
                win,
                t_cam,
            );
            gizmos.sphere(
                Isometry3d::new(
                    transform_pos(node.pos_offset),
                    Quat::from_mat4(&Mat4::from_cols_array_2d(&node.transform)),
                ),
                state.dummy_size,
                node_color(node),
            );
        }
    }
}

fn tex_coords(verts: &[parser::Vertex], id: usize) -> Result<Vec<[f32; 2]>> {
    let mut coords = Vec::with_capacity(verts.len());
    for vert in verts {
        let Some(ref tex) = (match id {
            0 => &vert.tex_0,
            1 => &vert.tex_1,
            2 => &vert.tex_2,
            3 => &vert.tex_3,
            4 => &vert.tex_4,
            5 => &vert.tex_5,
            6 => &vert.tex_6,
            7 => &vert.tex_7,
            i => bail!("Invalid index {i}!"),
        })
        .as_ref()
        .and_then(|t| t.0.clone().try_into().ok()) else {
            bail!("Coordinate set {id} unavailable!");
        };
        coords.push(*tex);
    }
    Ok(coords)
}

fn load_texture(path: &str, key: &str, fs: &MultiPackFS) -> Result<Image> {
    info!("Loading texture {path}");
    let Ok(mut fh) = fs.open_file(path) else {
        bail!("Failed to open file");
    };
    let ext = path.split(".").last().unwrap_or_default();
    let mut data = vec![];
    fh.read_to_end(&mut data).unwrap();
    let img = Image::from_buffer(
        #[cfg(debug_assertions)]
        key.to_owned(),
        &data,
        ImageType::Extension(ext),
        CompressedImageFormats::all(),
        true,
        ImageSampler::Descriptor(ImageSamplerDescriptor {
            label: Some(key.to_owned()),
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::Repeat,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            ..default()
        }),
        RenderAssetUsages::all(),
    )?;
    Ok(img)
}

fn load_level(
    state: &mut State,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    ass: ResMut<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    material_res: ResMut<Assets<StandardMaterial>>,
) -> [f32; 3] {
    let mut centroid = [0.0, 0.0, 0.0];
    let mut total_count: usize = 0;
    let Some(ParsedData::Level(level)) = state.data.as_ref() else {
        return centroid;
    };
    commands
        .spawn(Transform::default())
        .with_children(|parent| {
            let mut anim_tex = HashMap::new();
            let mut anim_mat: AnimMat = AnimMat::new();
            let mut textures = HashMap::new();
            let prop_re = Regex::new(r"\(\+(\w+)(?::(\w*))?\)").unwrap();
            for (key, path) in &level.dependencies {
                // let mut data = vec![];
                if let Some((name, _)) = path.rsplit_once(".") {
                    let name = name.replace("/dds/", "/");
                    let txa_name = format!("{name}.txa");
                    if let Ok(mut txa) = state.fs.open_file(&txa_name) {
                        let mut frames = vec![];
                        let mut txa_cont = vec![];
                        if let Ok(txa) = txa
                            .read_to_end(&mut txa_cont)
                            .map_err(|e| anyhow!(e))
                            .and_then(|_| std::str::from_utf8(&txa_cont).map_err(|e| anyhow!(e)))
                        {
                            for frame in 0.. {
                                let frame_num = format!(".{:03}.", frame);
                                let path = path.replace(".000.", &frame_num);
                                let key = key.replace(".000.", &frame_num);
                                if !state.fs.exists(&path).unwrap_or(false) {
                                    break;
                                }
                                match load_texture(&path, &key, &state.fs) {
                                    Ok(img) => {
                                        textures.insert(key.to_owned(), ass.add(img));
                                    }
                                    Err(err) => {
                                        warn!("Failed to load {key}: {err}");
                                        continue;
                                    }
                                }
                                frames.push(key);
                            }
                            let txa = Ini::new().read(txa.to_owned()).unwrap_or_default();
                            let fps = txa
                                .get("anim")
                                .and_then(|anim| anim.get("fps").and_then(|v| v.as_deref()))
                                .and_then(|fps| fps.parse::<f32>().ok())
                                .unwrap_or_default();
                            anim_tex.insert(key.as_str(), (frames, fps));
                        };
                    };
                }
                match load_texture(path, key, &state.fs) {
                    Ok(img) => {
                        textures.insert(key.to_owned(), ass.add(img));
                    }
                    Err(err) => {
                        warn!("Failed to load {key}: {err}");
                        continue;
                    }
                }
            }
            let mut materials = BTreeMap::new();
            for (key, mat) in &level.emi.materials {
                let mat_name = mat
                    .name
                    .as_ref()
                    .map(|p| p.string.clone())
                    .unwrap_or_else(|| format!("MAT:{key}"));
                info!("Loading material {mat_name}");
                let mat_props: HashMap<&str, &str> = prop_re
                    .captures_iter(&mat_name)
                    .filter_map(|g| {
                        let c: Vec<_> = g.iter().map(|c| c.map(|c| c.as_str())).collect();
                        let key: Option<&str> = c.get(1).and_then(|c| *c);
                        let value: &str = c.get(2).and_then(|c| *c).unwrap_or("");
                        key.map(|k| (k, value))
                    })
                    .collect();
                info!("Material properties: {mat_props:?}");
                let maps = vec![
                    (Slot::Diffuse, &mat.maps.diffuse),
                    (Slot::Metallic, &mat.maps.metallic),
                    (Slot::Reflection, &mat.maps.reflection),
                    (Slot::Bump, &mat.maps.bump),
                    (Slot::Glow, &mat.maps.glow),
                ];
                let mut map_names = vec![];
                for (slot, map) in maps {
                    let map: String = map
                        .as_ref()
                        .map(|m: &parser::MAP| m.texture.string.clone())
                        .unwrap_or_default();
                    if let Some((frames, fps)) = anim_tex.get(map.as_str()) {
                        let mut frame_handles = vec![];
                        for frame in frames {
                            let Some(img) = textures.get(frame) else {
                                break;
                            };
                            frame_handles.push(img.clone())
                        }
                        if frames.len() == frame_handles.len() {
                            anim_mat
                                .entry(*key)
                                .or_default()
                                .push((slot, *fps, frame_handles));
                        }
                    }
                    map_names.push((slot, map));
                }
                let has_alpha = mat
                    .maps
                    .diffuse
                    .as_ref()
                    .map(|m| m.texture.string.contains(".alpha."))
                    .unwrap_or(false);
                let base_color_texture = mat
                    .maps
                    .diffuse
                    .as_ref()
                    .and_then(|m| textures.get(m.texture.string.as_str()))
                    .cloned();
                let emissive_texture = mat
                    .maps
                    .glow
                    .as_ref()
                    .and_then(|m| textures.get(m.texture.string.as_str()))
                    .cloned();

                // B=metallic
                // G=roughness
                let metallic_roughness_texture = mat
                    .maps
                    .metallic
                    .as_ref()
                    .and_then(|m| textures.get(m.texture.string.as_str()))
                    .cloned();
                let normal_map_texture = mat
                    .maps
                    .bump
                    .as_ref()
                    .and_then(|m| textures.get(m.texture.string.as_str()))
                    .cloned();

                let reflection_map = mat
                    .maps
                    .reflection
                    .as_ref()
                    .and_then(|m| textures.get(m.texture.string.as_str()))
                    .cloned();

                let map_tex: HashMap<Slot, Option<Handle<Image>>> = [
                    (
                        Slot::Diffuse,
                        base_color_texture.as_ref().map(|h| h.clone_weak()),
                    ),
                    (
                        Slot::Metallic,
                        metallic_roughness_texture.as_ref().map(|h| h.clone_weak()),
                    ),
                    (
                        Slot::Reflection,
                        reflection_map.as_ref().map(|h| h.clone_weak()),
                    ),
                    (
                        Slot::Bump,
                        normal_map_texture.as_ref().map(|h| h.clone_weak()),
                    ),
                    (
                        Slot::Glow,
                        emissive_texture.as_ref().map(|h| h.clone_weak()),
                    ),
                ]
                .into_iter()
                .collect();

                let mut mat = StandardMaterial {
                    base_color: Color::WHITE.with_alpha(1.0),
                    base_color_texture,
                    emissive_texture,
                    // unlit: true,
                    perceptual_roughness: 0.5,
                    metallic: 0.0,
                    reflectance: 0.5,
                    metallic_roughness_texture,
                    normal_map_texture,
                    double_sided: true,
                    cull_mode: Some(Face::Back),
                    ..default()
                };

                mat.emissive = match mat.emissive_texture {
                    Some(_) => LinearRgba::new(10.0, 10.0, 10.0, 1.0),
                    None => Color::BLACK.to_linear(),
                };
                mat.alpha_mode = match mat_props.get("transp").copied() {
                    Some("filter") => AlphaMode::Multiply,
                    Some("premult") => {
                        mat.base_color.set_alpha(0.0);
                        AlphaMode::Premultiplied
                    }
                    _ if has_alpha => AlphaMode::Premultiplied,
                    _ => AlphaMode::Opaque,
                };

                if has_alpha {
                    mat.base_color.set_alpha(1.0);
                }

                if mat_props.contains_key("alpha_test") {
                    mat.alpha_mode = AlphaMode::Blend;
                }

                if mat_props.contains_key("zbias") {
                    if mat.alpha_mode == AlphaMode::Opaque {
                        mat.alpha_mode = AlphaMode::Blend;
                    }
                    mat.depth_bias = 100.0;
                };
                if mat.base_color_texture.is_none() && mat.alpha_mode == AlphaMode::Opaque {
                    mat.alpha_mode = AlphaMode::Multiply;
                    mat.base_color.set_alpha(0.0);
                };

                match mat_props.get("shader").copied().unwrap_or_default() {
                    "hologram" => {
                        mat.emissive = Color::WHITE.to_linear();
                        mat.alpha_mode = AlphaMode::Premultiplied;
                        mat.base_color.set_alpha(0.0);
                    }
                    "" => {}
                    other => {
                        println!("Shader {other} not implemented!");
                    }
                }
                let mat = ass.add(mat);
                if let Some(anim) = anim_mat.get(key) {
                    for (slot, fps, frames) in anim {
                        state.anim_textures.push(AnimTexture {
                            fps: *fps,
                            images: frames.clone(),
                            slot: *slot,
                            mat: mat.clone(),
                        });
                    }
                }
                materials.insert(key, (mat_name, map_names, map_tex, mat));
                // if let Some(mat) = material_res.get_mut(&mat) {
                //     for (id,img) in images.iter() {
                //         dbg!(id);
                //     }
                //     dbg!(images.len());
                //     if let Some(img) = images.get_mut(h) {
                //         dbg!(img.data.len());
                //         let hex = rhexdumps!(&img.data[..0x100]);
                //         info!("{hex}");
                //     }
                // }
                // let mat = match mat_props.get("shader").map(|&s| s).unwrap_or("") {
                //     // "hologram" => ass.add(ExtendedMaterial {
                //     //     base: mat,
                //     //     extension: Hologram {},
                //     // }),
                //     _ => ass.add(mat)
                // };
            }
            let mut lmaps = HashMap::new();
            for lm in level.emi.maps.iter() {
                if let Some((key_1, _, key_2)) = &lm.data {
                    lmaps.insert(
                        lm.key,
                        (
                            textures.get(key_1.string.as_str()).cloned(),
                            textures.get(key_2.string.as_str()).cloned(),
                        ),
                    );
                }
            }
            // for dum in &level.dummies.dummies {
            //     dum.
            // }
            for tri in &level.emi.tri {
                let name = &tri.name.string;
                let data = &tri.data;
                let (mat_name, map_names, map_tex, mat) = &materials[&data.mat_key];
                let mat_inst = material_res.get(mat);
                info!("Loading mesh {name} with material {mat_name}");
                // dbg!(&tri.sector_num);
                let mesh_props: HashMap<&str, &str> = prop_re
                    .captures_iter(name)
                    .filter_map(|g| {
                        let c: Vec<_> = g.iter().map(|c| c.map(|c| c.as_str())).collect();
                        let key: Option<&str> = c.get(1).and_then(|c| *c);
                        let value: &str = c.get(2).and_then(|c| *c).unwrap_or("");
                        key.map(|k| (k, value))
                    })
                    .collect();
                info!("Mesh properties: {mesh_props:?}");
                let faces = &data.tris;
                for verts in [&data.verts_1, &data.verts_2]
                    .iter()
                    .filter_map(|v| v.inner.as_ref())
                {
                    let mesh = mesh_from_m3d(faces, &verts.data);
                    let needs_normals = verts.data.iter().any(|v| v.normal.is_none())
                        || mat_inst
                            .as_ref()
                            .and_then(|m| m.normal_map_texture.as_ref())
                            .is_some();
                    if needs_normals {
                        // mesh.duplicate_vertices();
                        // mesh.compute_flat_normals();
                    }
                    // mesh.generate_tangents()
                    //     .expect("Failed to generate tangents");
                    // TODO: custom materials based on shaders
                    let h_mesh = meshes.add(mesh);
                    let mut pbr = parent.spawn((
                        Mesh3d(h_mesh),
                        MeshMaterial3d(mat.clone()),
                        Name::new(name.to_owned()),
                        MaterialName(mat_name.to_owned()),
                        MapNames(map_names.clone()),
                        MapTex(map_tex.clone()),
                    ));
                    pbr.observe(mesh_clicked);
                    if let Some((lmap_1, lmap_2)) = lmaps.get(&data.map_key) {
                        if let Some(lm_1) = lmap_1.as_ref() {
                            pbr.insert(Lightmap {
                                image: lm_1.clone(),
                                ..default()
                            });
                        }
                        if let Some(lm_2) = lmap_2.as_ref() {
                            pbr.insert(Lightmap {
                                image: lm_2.clone(),
                                ..default()
                            });
                        }
                    }
                }
            }
            dbg!(anim_mat);
        });
    centroid[0] /= total_count as f32;
    centroid[1] /= total_count as f32;
    centroid[2] /= total_count as f32;
    centroid
}

fn mesh_clicked(
    trigger: Trigger<Pointer<Click>>,
    mut contexts: EguiContexts,
    mut commands: Commands,
    mut state: ResMut<State>,
) {
    let egui_waints_input =
        contexts.ctx_mut().wants_keyboard_input() || contexts.ctx_mut().wants_pointer_input();
    if egui_waints_input {
        return;
    }
    if trigger.button != PointerButton::Primary {
        return;
    }
    let ent = trigger.entity();
    if let Some(old_ent) = state.picked_object.replace(ent) {
        if let Some(mut e) = commands.get_entity(old_ent) {
            e.remove::<WireframeColor>().remove::<Wireframe>();
        }
    }
    commands
        .entity(ent)
        .insert(Wireframe)
        .insert(WireframeColor {
            color: Color::srgb_u8(255, 0, 0),
        });
}

fn ui_input_toggle(mut contexts: EguiContexts, mut cam_ctrl: Query<&mut UnrealCameraController>) {
    let egui_ctx = contexts.ctx_mut();
    cam_ctrl.single_mut().enabled =
        !(egui_ctx.wants_keyboard_input() || egui_ctx.wants_pointer_input());
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    a * (1.0 - t) + b * t
}

fn tree_overlay(state: Res<State>, mut gizmos: Gizmos) {
    for (p1, p2) in &state.edges {
        gizmos.line(*p1, *p2, Color::WHITE);
    }
}

// fn autofocus(
//     mut raycast: Raycast,
//     mut cam: Query<(&Camera, &Transform, Option<&mut DepthOfField>)>,
//     cursor_ray: Res<CursorLocation>,
//     keyboard: Res<ButtonInput<KeyCode>>,
//     time: Res<Time>,
// ) {
//     let mut cam = cam.get_single_mut().unwrap();
//     let (_, tr, Some(ref mut dof)) = cam else {
//         return;
//     };

//     let pick_ray = if keyboard.pressed(KeyCode::KeyF) {
//         **cursor_ray
//     } else {
//         Some(Ray3d::new(tr.translation, *tr.forward()))
//     };
//     let Some((_, hit)) = pick_ray.and_then(|cursor_ray| {
//         raycast
//             .cast_ray(
//                 cursor_ray,
//                 &RaycastSettings {
//                     visibility: RaycastVisibility::MustBeVisible,
//                     ..default()
//                 },
//             )
//             .first()
//     }) else {
//         return;
//     };

//     let l: f32 = 0.1;
//     let dt = time.delta().as_secs_f32();
//     dof.focal_distance = lerp(dof.focal_distance, hit.distance(), dt / l);
// }

fn inspector(
    state: Res<State>,
    images: Res<Assets<Image>>,
    mut tex: ResMut<EguiTexHandles>,
    mut contexts: EguiContexts,
    name: Query<&Name>,
    mat: Query<(
        &MeshMaterial3d<StandardMaterial>,
        &MaterialName,
        &MapNames,
        &MapTex,
    )>,
) {
    if state.show_ui {
        if let Some((_, _, map_names, map_tex)) =
            state.picked_object.and_then(|ent| mat.get(ent).ok())
        {
            for (slot, tex_name) in map_names.iter() {
                let Some(map_tex) = map_tex.get(slot).and_then(|v| v.as_ref()) else {
                    continue;
                };
                tex.entry(tex_name.clone()).or_insert_with(|| {
                    (
                        map_tex.clone_weak(),
                        contexts.add_image(map_tex.clone_weak()),
                    )
                });
            }
        };

        egui::Window::new("Inspector").show(contexts.ctx_mut(), |ui| {
            let Some(ent) = state.picked_object else {
                return;
            };
            if let Ok(name) = name.get(ent) {
                ui.heading(format!("{name} ({ent:?})"));
            } else {
                ui.heading(format!("{ent:?}"));
            }
            let Ok((_, mat_name, map_names, _)) = mat.get(ent) else {
                return;
            };
            let mat_name = mat_name.as_str();
            // let mat = materials.get(mat);
            ui.label(format!("Material: {mat_name}"));
            // dbg!(&map_names);
            for (slot, tex_name) in map_names.iter() {
                let resp = ui.label(format!("{slot:?}: {tex_name}"));
                if resp.hovered() {
                    resp.show_tooltip_ui(|ui| {
                        let Some((img, tex_id)) = tex.get(tex_name) else {
                            return;
                        };
                        let Some(img) = images.get(img) else {
                            return;
                        };
                        ui.add(
                            egui::widgets::Image::new(egui::load::SizedTexture::new(
                                *tex_id,
                                [img.width() as f32, img.height() as f32],
                            ))
                            .max_size([256.0, 256.0].into())
                            .rounding(0.5)
                            .maintain_aspect_ratio(true),
                        );
                    });
                }
            }
        });
    }
}

// #[allow(clippy::too_many_arguments)]
// fn entity_picker(
//     mut commands: Commands,
//     mouse: Res<ButtonInput<MouseButton>>,
//     cursor_ray: Res<CursorRay>,
//     mut state: ResMut<State>,
//     mut contexts: EguiContexts,
//     cam: Query<(&Camera, &Transform, &GlobalTransform)>,
// ) {
//     let pick_ray = if mouse.pressed(MouseButton::Left) {
//         **cursor_ray
//     } else {
//         let (_, tr, _) = cam.get_single().unwrap();
//         Some(Ray3d::new(tr.translation, *tr.forward()))
//     };
//     let egui_waints_input =
//         contexts.ctx_mut().wants_keyboard_input() || contexts.ctx_mut().wants_pointer_input();
//     let picked_entity = pick_ray.and_then(|cursor_ray| {
//         raycast
//             .cast_ray(cursor_ray, &RaycastSettings::default())
//             .first()
//     });
//     if !egui_waints_input && mouse.just_pressed(MouseButton::Left) {
//         if let Some(ent) = state.picked_object.take() {
//             if let Some(mut e) = commands.get_entity(ent) {
//                 e.remove::<WireframeColor>().remove::<Wireframe>();
//             }
//         }
//         state.picked_object = picked_entity.map(|(ent, _)| {
//             commands
//                 .entity(*ent)
//                 .insert(Wireframe)
//                 .insert(WireframeColor {
//                     color: Color::srgb_u8(255, 0, 0),
//                 });
//             *ent
//         });
//     }
// }

fn keyboard_handler(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut wireframe: ResMut<WireframeConfig>,
    mut exit: EventWriter<AppExit>,
    mut state: ResMut<State>,
    mut cam: Query<(&Camera, &Transform, Option<&mut DepthOfField>)>,
) {
    let (_, _, mut dof) = cam.get_single_mut().unwrap();

    for key in keyboard.get_pressed() {
        match key {
            KeyCode::NumpadAdd => {
                state.lightmap_exposure += 1000.0;
                println!("LM: {}", state.lightmap_exposure);
            }
            KeyCode::NumpadSubtract => {
                state.lightmap_exposure -= 1000.0;
                println!("LM: {}", state.lightmap_exposure);
            }
            KeyCode::NumpadMultiply => {
                state.lightmap_exposure *= 1.1;
                println!("LM: {}", state.lightmap_exposure);
            }
            KeyCode::NumpadDivide => {
                state.lightmap_exposure /= 1.1;
                println!("LM: {}", state.lightmap_exposure);
            }
            KeyCode::Numpad1 => {
                if let Some(ref mut dof) = dof {
                    dof.aperture_f_stops /= 1.1;
                    println!("F-Stops: {}", dof.aperture_f_stops);
                }
            }
            KeyCode::Numpad2 => {
                if let Some(ref mut dof) = dof {
                    dof.aperture_f_stops *= 1.1;
                    println!("F-Stops: {}", dof.aperture_f_stops);
                }
            }
            _ => {}
        }
    }
    for key in keyboard.get_just_pressed() {
        match key {
            KeyCode::Escape => {
                exit.send(AppExit::Success);
            }
            KeyCode::F1 => {
                state.show_ui = !state.show_ui;
            }
            KeyCode::F2 => {
                state.lightmaps = !state.lightmaps;
            }
            KeyCode::F3 => {
                wireframe.global = !wireframe.global;
            }
            KeyCode::F4 => {
                state.show_nodes = !state.show_nodes;
            }
            _ => {}
        }
    }
}

fn help_window(
    mut contexts: EguiContexts,
    mut state: ResMut<State>,
    mut wireframe: ResMut<WireframeConfig>,
    mut tonemap: Query<&mut Tonemapping, With<Camera>>,
    type_registry: ResMut<AppTypeRegistry>,
) {
    egui::Window::new("Help").show(contexts.ctx_mut(), |ui| {
        let state = state.as_mut();
        let tonemapping = tonemap.single_mut();
        reflect_inspector::ui_for_value(tonemapping.into_inner(), ui, &type_registry.read());
        ui.toggle_value(&mut state.show_ui, "Show UI [F1]");
        ui.toggle_value(&mut state.lightmaps, "Enable Lightmaps [F2]");
        ui.toggle_value(&mut wireframe.global, "Enable Wireframes [F3]");
        ui.horizontal(|ui| {
            ui.toggle_value(&mut state.show_nodes, "Show Nodes [F4]");
            if state.show_nodes {
                ui.add(egui::DragValue::new(&mut state.dummy_size).speed(0.1));
            }
        });
        if state.show_nodes {
            ui.label("Node types:");
            let nodes_types = [
                (
                    "Map Dummy",
                    "#00ff00",
                    "Labeled position in 3D space with optional metadata attached",
                ),
                ("Camera", "#ffff00", "Self explanatory"),
                ("Dummy Node", "#ffffff", "Same as Map Dummy, just as a node"),
                (
                    "TriangleMesh",
                    "#0000ff",
                    "Mesh made of Trigangles, seems to be unused",
                ),
                (
                    "D3DMesh",
                    "#00ffff",
                    "DirectX8 Mesh using the Flexible Vertex Format",
                ),
                ("Light", "#ff00ff", "Light"),
                ("Ground", "#7fff00", "Ground Plane"),
                ("SistPart", "#ff7f00", "Particle System"),
                ("Graphic3D", "#00ff7f", "3D Graphic"),
                ("Flare", "#007fff", "Lens Flare"),
                ("Portal", "#ff7fff", "Map Portal"),
            ];
            for (name, color, tooltip) in nodes_types {
                ui.colored_label(egui::Color32::from_hex(color).unwrap(), name)
                    .on_hover_text(tooltip);
            }
        }
    });
}

fn load_amc(
    state: &mut State,
    mut commands: Commands,
    ass: &mut ResMut<AssetServer>,
    meshes: &mut ResMut<Assets<Mesh>>,
) {
    let Some(ParsedData::Data(Data::AMC(amc))) = state.data.as_ref() else {
        return;
    };
    todo!();
}

fn node_type(node: &Option<NodeData>) -> &'static str {
    node.as_ref()
        .map(|data| match data {
            NodeData::Camera(_) => "Camera",
            NodeData::Dummy => "Dummy",
            NodeData::TriangleMesh(_) => "TriangleMesh",
            NodeData::D3DMesh(_) => "D3DMesh",
            NodeData::Light(_) => "Light",
            NodeData::Ground(_) => "Ground",
            NodeData::SistPart => "Particle System",
            NodeData::Graphic3D(_) => "3D Graphic",
            NodeData::Flare => "Flare",
            NodeData::Portal(_) => "Portal",
        })
        .unwrap_or_else(|| "<Empty>")
}

fn mesh_from_m3d(faces: &[[u16; 3]], verts: &[Vertex]) -> Mesh {
    let pos: Vec<_> = verts.iter().map(|v| transform_pos(v.xyz)).collect();
    let normal: Vec<_> = verts.iter().map(|v| v.normal.unwrap_or_default()).collect();
    let color: Vec<[f32; 4]> = verts
        .iter()
        .map(|v| {
            let c_diffuse = v
                .diffuse
                .as_ref()
                .map(|c| Color::srgba_u8(c.r, c.g, c.b, c.a))
                .unwrap_or(Color::WHITE)
                .to_linear();
            let c_specular = v
                .specular
                .as_ref()
                .map(|c| Color::srgba_u8(c.r, c.g, c.b, c.a))
                .unwrap_or(Color::WHITE)
                .to_linear();
            let c = Color::WHITE;
            let diffuse = [
                c_diffuse.red,
                c_diffuse.green,
                c_diffuse.blue,
                c_diffuse.alpha,
            ];
            let specular = [
                c_specular.red,
                c_specular.green,
                c_specular.blue,
                c_specular.alpha,
            ];
            // specular
            // diffuse
            [1.0, 1.0, 1.0, 1.0]
        })
        .collect();

    let idx = faces
        .iter()
        .copied()
        .flat_map(|mut face| {
            face.swap(0, 1);
            face
        })
        .collect::<Vec<_>>();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normal)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, color)
    .with_inserted_indices(Indices::U16(idx));
    for (id, attr) in [(0, Mesh::ATTRIBUTE_UV_0), (1, Mesh::ATTRIBUTE_UV_1)] {
        let coords = match tex_coords(verts, id) {
            Ok(coords) => coords,
            Err(_) => {
                continue;
            }
        };
        mesh.insert_attribute(attr, coords);
    }
    mesh
}

fn node_to_ent(
    commands: &mut Commands,
    node: &parser::Node,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    meshes: &mut ResMut<Assets<Mesh>>,
) -> Entity {
    let mut transform = Transform::from_matrix(Mat4::from_cols_array_2d(&node.transform));
    transform.translation = transform.translation.div_euclid(Vec3::splat(10.0));
    let color = node_color(node);

    let Some(node_data) = &*node.content else {
        return commands
            .spawn_empty()
            .insert(transform)
            .insert(Visibility::default())
            .insert(Name::new(node.name.string.clone()))
            .id();
    };
    let mesh = match node_data {
        NodeData::D3DMesh(md3d) => {
            if let Some(verts) = md3d.verts.inner.as_ref().map(|v| &v.data) {
                mesh_from_m3d(&md3d.tris.tris, verts)
            } else {
                Sphere::new(0.1).mesh().build()
            }
        }
        _ => Sphere::new(0.1).mesh().build(),
    };
    commands
        .spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add({
                let mut m = StandardMaterial::from(color);
                m.unlit = true;
                m
            })),
            GlobalTransform::from(transform),
        ))
        .insert(Name::new(node.name.string.clone()))
        .id()
}

fn load_sm3(
    state: &mut State,
    mut commands: Commands,
    imgs: ResMut<Assets<Image>>,
    ass: ResMut<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(ParsedData::Data(Data::SM3(sm3))) = state.data.as_ref() else {
        return;
    };

    let mut nodes = HashMap::new();

    dbg!(&sm3.scene.mat.len());

    for node in &sm3.scene.nodes {
        let t = Affine3A::from_mat4(Mat4::from_cols_array_2d(&node.transform));
        let t_inv = Affine3A::from_mat4(Mat4::from_cols_array_2d(&node.transform_inv));
        dbg!(&node.name.string);
        dbg!(&t);
        dbg!(&t_inv);
        dbg!(t.inverse());
        dbg!(t_inv.inverse());
        eprintln!("========");
    }

    // std::process::exit(1);

    for node in &sm3.scene.nodes {
        nodes.insert(node.name.string.as_str(), node);
    }
    dbg!(sm3.scene.mat.len());
    for node in &sm3.scene.nodes {
        dbg!(&node.name.string);
    }
    let graph: GraphMap<&str, (), Directed> = GraphMap::from_edges(
        sm3.scene
            .nodes
            .iter()
            .map(|n| (n.parent.string.as_str(), n.name.string.as_str())),
    );
    for (src, dst, _) in graph.all_edges() {
        let node = nodes[dst];
        println!("{} -> {} [{}]", src, dst, node_type(&node.content));
        println!("Flags: {:?}, Attrs: {:?}", node.flags, &*node.info);
    }
    for node_name in graph.nodes() {
        let Some(node) = nodes.get(node_name) else {
            println!("Entity {node_name:?} not found!");
            continue;
        };
        let ent = node_to_ent(&mut commands, node, &mut materials, &mut meshes);
    }

    // for (node, ent) in &ents {
    //     if !nodes.contains_key(node) {
    //         println!("Entity {node:?} not found!");
    //         continue;
    //     }
    //     let mut ent = commands.entity(*ent);
    //     let nt = node_type(&nodes[node].content);
    //     ent.insert((
    //         PbrBundle {
    //             mesh: meshes.add(Sphere::new(0.1).mesh().build()),
    //             material: materials.add({
    //                 let mut m = StandardMaterial::from(color);
    //                 m.unlit = true;
    //                 m
    //             }),
    //             global_transform: transform.into(),
    //             ..Default::default()
    //         },
    //         WireframeColor { color },
    //         Name::new(format!("{node} [{nt}]")),
    //     ));
    // }
    for (n1, n2, _) in graph.all_edges() {
        let (Some(n1), Some(n2)) = (nodes.get(n1), nodes.get(n2)) else {
            continue;
        };
        let t1 = Transform::from_matrix(Mat4::from_cols_array_2d(&n1.transform));
        let t2 = Transform::from_matrix(Mat4::from_cols_array_2d(&n2.transform));
        let p1 = t1.translation.div_euclid(Vec3::splat(10.0));
        let p2 = t2.translation.div_euclid(Vec3::splat(10.0));
        state.edges.push((p1, p2));
    }
}

type WithMeshAndMaterial = (MeshMaterial3d<StandardMaterial>, Mesh3d);

// TODO: cleanup, integrate Clap
#[allow(clippy::too_many_arguments)]
fn browser(
    mut imgs: ResMut<Assets<Image>>,
    mut commands: Commands,
    mut contexts: EguiContexts,
    mut state: ResMut<State>,
    mut ass: ResMut<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cam: Query<(&mut Camera, &mut Transform, &GlobalTransform)>,
    mesh: Query<(Entity, &Mesh3d, &mut MeshMaterial3d<StandardMaterial>)>,
) {
    let state = state.as_mut();
    if let Some(data_path) = state.data_path.take() {
        match state.fs.parse_file(&data_path) {
            Ok(ParsedData::Level(level)) => {
                state.data = Some(ParsedData::Level(level));
                let c = load_level(state, commands, imgs, ass, meshes, materials);
                let mut t = cam.get_single_mut().unwrap().1;
                t.translation.x = c[0];
                t.translation.y = c[1];
                t.translation.z = c[2];
            }
            Ok(ParsedData::Data(Data::SM3(sm3))) => {
                state.data = Some(ParsedData::Data(Data::SM3(sm3)));
                load_sm3(state, commands, imgs, ass, meshes, materials);
            }
            Ok(ParsedData::Data(Data::AMC(amc))) => {
                state.data = Some(ParsedData::Data(Data::AMC(amc)));
                load_amc(state, commands, &mut ass, &mut meshes);
            }
            Ok(ParsedData::Data(_)) => {
                error!("Don't know what to do with {data_path}");
            }
            Err(e) => {
                error!("Error loading {data_path}: {e}");
            }
            _ => (),
        };
        return;
    }

    for (_, mesh, mat) in &mesh {
        if let Some(mat) = materials.get_mut(mat) {
            if state.lightmaps {
                mat.lightmap_exposure = state.lightmap_exposure;
            } else {
                mat.lightmap_exposure = 0.0;
            }
        }
    }

    if !state.show_ui {
        return;
    }
    let cam = cam.get_single_mut().unwrap();
    let title = state
        .fs
        .pwd()
        .map(|pwd| format!("Browser [{}]", pwd))
        .unwrap_or_else(|_| "Browser".to_owned());
    egui::Window::new(title).show(contexts.ctx_mut(), |ui| {
        let files = state.fs.ls();
        if let Ok(files) = files {
            ScrollArea::vertical().show(ui, |ui| {
                if ui.button("..").clicked() {
                    if let Err(e) = state.fs.cd("..") {
                        dbg!(e);
                    };
                }
                for entry in &files {
                    let path = entry.path.strip_prefix('/').unwrap_or(&entry.path);
                    if ui.button(path).clicked() {
                        if let Err(e) = state.fs.cd(&entry.path) {
                            dbg!(e);
                        };
                    }
                }
                if state.fs.is_level(None).unwrap_or(false) {
                    ui.separator();
                    if ui.button(RichText::new("Load").heading()).clicked() {
                        if let Ok(pwd) = state.fs.pwd() {
                            println!("Loading {pwd}");
                        }
                        let path = state.fs.pwd().expect("Failed to get current directory");
                        match state.fs.parse_file(&path) {
                            Ok(ParsedData::Level(level)) => {
                                state.data = Some(ParsedData::Level(level));
                                for (ent, _, _) in &mesh {
                                    commands.entity(ent).despawn_recursive();
                                }
                                let c = load_level(state, commands, imgs, ass, meshes, materials);
                                let mut t = cam.1;
                                t.translation.x = c[0];
                                t.translation.y = c[1];
                                t.translation.z = c[2];
                            }
                            Err(e) => {
                                print!("Error loading {path}: {e}");
                            }
                            _ => (),
                        };
                    }
                }
            });
        }
    });
}

fn setup(mut commands: Commands) {
    commands
        .spawn((
            Camera3d::default(),
            Camera {
                hdr: true,
                ..Default::default()
            },
            Tonemapping::None,
            Msaa::Sample8,
            Bloom {
                intensity: 0.75,
                low_frequency_boost: 0.0,
                low_frequency_boost_curvature: 0.0,
                high_pass_frequency: 1.0,
                prefilter: BloomPrefilter {
                    threshold: 2.0,
                    threshold_softness: 1.0,
                },
                composite_mode: BloomCompositeMode::Additive,
                ..Default::default()
            },
            // DepthOfFieldSettings {
            //     focal_distance: 1.0, // Raycast from camera/cursor
            //     mode: DepthOfFieldMode::Bokeh,
            //     ..default()
            // },
            // AtmosphereCamera::default(),
        ))
        .insert(UnrealCameraBundle::new(
            UnrealCameraController {
                keyboard_mvmt_wheel_sensitivity: 1.0,
                ..default()
            },
            Vec3::new(-2.0, 5.0, 5.0),
            Vec3::new(0., 1., 0.),
            Vec3::Y,
        ));
    // .with_children(|builder| {
    //     builder.spawn(SpotLightBundle {
    //         spot_light: SpotLight {
    //             shadows_enabled: true,
    //             outer_angle: std::f32::consts::PI / 4.,
    //             intensity: 100000.0,
    //             range: 500.0,
    //             radius: 0.0,
    //             ..default()
    //         },
    //         ..default()
    //     });
    // })
}
