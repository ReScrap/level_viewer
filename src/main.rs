use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    hash::Hash,
    io::{BufWriter, Cursor, Read, Write},
    ops::{Deref, DerefMut},
    path::PathBuf,
};

use asset_loader::PackedAssetRepositoryPlugin;
use bevy::{
    anti_alias::{
        contrast_adaptive_sharpening::ContrastAdaptiveSharpening, taa::TemporalAntiAliasing,
    },
    app::AppExit,
    asset::{ErasedAssetLoader, RenderAssetUsages, embedded_asset},
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    image::{
        CompressedImageFormats, Image, ImageAddressMode, ImageFilterMode, ImageSampler,
        ImageSamplerDescriptor, ImageType,
    },
    input::{
        gamepad::{GamepadAxisChangedEvent, GamepadButtonChangedEvent},
        mouse::AccumulatedMouseMotion,
    },
    log::LogPlugin,
    mesh::{Indices, PrimitiveTopology},
    pbr::{
        ExtendedMaterial, Lightmap,
        wireframe::{Wireframe, WireframeColor, WireframeConfig, WireframePlugin},
    },
    post_process::{
        auto_exposure::AutoExposure,
        bloom::{Bloom, BloomCompositeMode, BloomPrefilter},
        dof::{DepthOfField, DepthOfFieldMode},
        motion_blur::MotionBlur,
    },
    prelude::{Result as BevyResult, *},
    render::{
        render_resource::Face,
        view::{ColorGrading, ColorGradingGlobal, Hdr},
    },
    window::{CursorGrabMode, CursorOptions, PresentMode, PrimaryWindow, WindowMode},
};
use bevy_inspector_egui::{
    bevy_egui::{
        EguiContexts, EguiPlugin, EguiPrimaryContextPass,
        egui::{self, RichText, ScrollArea, TextureId as EguiTextureId},
    },
    quick::WorldInspectorPlugin,
    reflect_inspector,
};
use binrw::{BinReaderExt, BinWrite, binread};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use color_eyre::eyre::{Context, Result, anyhow, bail};
use configparser::ini::Ini;
use itertools::Itertools;
use num_traits::Float;
use packed_vfs::MultiPack;
use parser::{Data, NodeData, Vertex, multi_pack_fs::MultiPackFS};
use petgraph::{Directed, graphmap::GraphMap};
use pid::Pid;
use regex::Regex;
use rhexdump::{rhexdump, rhexdumps};
use serde::{Deserialize, Serialize};

use crate::{
    // materials::Hologram,
    asset_loader::TestAsset,
    materials::TestMaterial,
    parser::{AniTrackType, AnimTracks, CM3, LightType, ParsedData, SM3},
};
mod asset_loader;
mod export;
mod find_scrap;
mod materials;
mod packed_vfs;
mod parser;
mod pixel_shader;

type ScrapMaterial = ExtendedMaterial<StandardMaterial, TestMaterial>;

#[derive(Resource, Debug, Default, Deref, DerefMut)]
struct EguiTexHandles(BTreeMap<String, (Handle<Image>, EguiTextureId)>);

#[derive(Component, Deref, Debug, Reflect)]
struct MaterialName(String);

#[derive(Clone, Component, Deref, Debug)]
struct ScrapMat(parser::MAT);

#[derive(Clone, Component, Deref, Debug, Reflect)]
struct MapNames(Vec<(Slot, String)>);

#[derive(Clone, Component, Deref, Debug, Reflect)]
struct MapTex(BTreeMap<Slot, Option<Handle<Image>>>);

#[derive(Clone, Component, Debug, Reflect)]
struct LightmapNames(String, String);

#[derive(Clone, Component, Debug, Reflect)]
struct LightmapHandles(Option<Handle<Image>>, Option<Handle<Image>>);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
enum Slot {
    Diffuse,
    Metallic,
    Reflection,
    Bump,
    Glow,
}

impl std::fmt::Display for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Reflect)]
struct AnimTexture {
    fps: f32,
    images: Vec<Handle<Image>>,
    slot: Slot,
    mat: Handle<ScrapMaterial>,
}

type AnimMat = HashMap<u32, Vec<(Slot, f32, Vec<Handle<Image>>)>>;

#[derive(Resource, Debug)]
struct State {
    fs: MultiPackFS,
    browser_tree: Option<BrowserTreeNode>,
    browser_tree_error: Option<String>,
    af_decay: f32,
    data_path: Option<String>,
    data: Option<ParsedData>,
    picked_object: Option<Entity>,
    show_ui: bool,
    show_browser_panel: bool,
    show_nodes: bool,
    show_collision: bool,
    lightmaps: bool,
    cam_physics: bool,
    cam_auto_level: bool,
    node_size: f32,
    thrust_power: f32,
    lightmap_exposure: f32,
    edges: Vec<(Vec3, Vec3)>,
    anim_textures: Vec<AnimTexture>,
    export: bool,
}

fn transform_pos(p: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = p;
    [x / -1.0, y / 1.0, z / 1.0]
}

fn animate_textures(
    state: Res<State>,
    time: Res<Time>,
    mut materials: ResMut<Assets<ScrapMaterial>>,
) {
    for tex in &state.anim_textures {
        let frame = (time.elapsed_secs() * tex.fps).floor() as usize;
        let img = tex.images[frame % tex.images.len()].clone();
        let Some(mat) = materials.get_mut(&tex.mat) else {
            continue;
        };
        let mat = &mut mat.base;
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

fn get_packed_files() -> Result<Vec<PathBuf>> {
    let data_regex = Regex::new(r"[Dd]ata\d*\.packed")?;
    let scrap_path = match find_scrap::get_path() {
        Ok(path) => path,
        Err(err) => {
            let err_msg = format!("{err}");
            rfd::MessageDialog::new()
                .set_title(err_msg)
                .set_description("Please locate the Scrapland installation folder manually")
                .set_buttons(rfd::MessageButtons::Ok)
                .set_level(rfd::MessageLevel::Warning)
                .show();
            {
                let Some(folder) = rfd::FileDialog::new()
                    .set_title("Scrapland installation folder")
                    .pick_folder() else
                {
                    std::process::exit(1);
                };
                folder
            }
        }
    };
    let packed_files: Vec<PathBuf> = parser::find_packed(scrap_path)
        .context("Failed to find .packed files")?
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
    if packed_files.is_empty() {
        bail!("No .packed files found!");
    }
    Ok(packed_files)
}

fn dump_ani(fs: &MultiPackFS, sm3: &str, cm3: &str) -> Result<HashMap<String, AnimTracks>> {
    let mut track_map: HashMap<String, AnimTracks> = HashMap::new();
    let ParsedData::Data(Data::SM3(sm3)) = fs.parse_file(sm3)? else {
        bail!("Failed to parse model!")
    };
    let ParsedData::Data(Data::CM3(cm3)) = fs.parse_file(cm3)? else {
        bail!("Failed to parse animation!")
    };
    let Some(ani) = cm3.scene.ani.get() else {
        bail!("No animation data found in CM3!");
    };
    for node in &sm3.scene.nodes {
        if node.object_index >= 0 {
            let idx: usize = node.object_index.try_into()?;
            if let Some(track) =
                ani.track_map[idx].and_then(|v| ani.get_track(v as usize).ok().flatten())
            {
                let name = (*node.name).to_owned();
                let nam = &ani.tracks[ani.track_map[idx].unwrap() as usize];
                println!(
                    "   {}: {:?} ({}+{} frames)",
                    name, nam.cm3_flags, nam.start_frame, nam.frames
                );
                track_map.insert(name, track);
            }
        }
    }
    return Ok(track_map);
}

fn find_numeric_null_candidates(json: &str, limit: usize) -> Vec<String> {
    fn fmt_path(path: &[String]) -> String {
        let mut out = String::new();
        for part in path {
            if let Some(idx) = part.strip_prefix('#') {
                out.push('[');
                out.push_str(idx);
                out.push(']');
            } else if out.is_empty() {
                out.push_str(part);
            } else {
                out.push('.');
                out.push_str(part);
            }
        }
        out
    }

    fn walk(
        value: &serde_json::Value,
        path: &mut Vec<String>,
        out: &mut Vec<String>,
        limit: usize,
    ) {
        if out.len() >= limit {
            return;
        }
        match value {
            serde_json::Value::Array(arr) => {
                let has_null = arr.iter().any(serde_json::Value::is_null);
                let has_num = arr.iter().any(serde_json::Value::is_number);
                if has_null && has_num {
                    for (i, item) in arr.iter().enumerate() {
                        if out.len() >= limit {
                            return;
                        }
                        if item.is_null() {
                            path.push(format!("#{i}"));
                            out.push(fmt_path(path));
                            path.pop();
                        }
                    }
                }
                for (i, item) in arr.iter().enumerate() {
                    if out.len() >= limit {
                        return;
                    }
                    path.push(format!("#{i}"));
                    walk(item, path, out, limit);
                    path.pop();
                }
            }
            serde_json::Value::Object(obj) => {
                for (k, v) in obj {
                    if out.len() >= limit {
                        return;
                    }
                    path.push(k.clone());
                    walk(v, path, out, limit);
                    path.pop();
                }
            }
            _ => {}
        }
    }

    let Ok(value) = serde_json::from_str::<serde_json::Value>(json) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut path = Vec::new();
    walk(&value, &mut path, &mut out, limit);
    out
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let packed_files = get_packed_files()?;
    let fs = MultiPackFS::new(&packed_files)?;
    // for sm3_entry in &entries {
    //     if !sm3_entry.path.ends_with(".sm3") {
    //         continue;
    //     }
    //     let Some((base,_)) = sm3_entry.path.rsplit_once('/') else {
    //         continue;
    //     };
    //     println!("{}", sm3_entry.path);
    //     for cm3_entry in &entries {
    //         if !cm3_entry.path.ends_with("play.cm3") {
    //             continue;
    //         }
    //         if !cm3_entry.path.starts_with(base) {
    //             continue;
    //         }
    //         println!("+ {}", cm3_entry.path);
    //     }
    // return Ok(());
    // // }
    // {
    //     let mut total = 0;
    //     let mut failed = 0;
    //     let mut fail_match = 0;
    //     let mut max_diff = 0;
    //     'outer: for entry in fs.entries()? {
    //         if [".cm3", ".sm3", ".emi", ".dum", ".amc"]
    //             .iter()
    //             .any(|e| entry.path.ends_with(e))
    //         {
    //             total += 1;
    //             let data = match fs.parse_file(&entry.path) {
    //                 Ok(data) => data,
    //                 Err(err) => {
    //                     println!("Fail: {}: {:#}", entry.path, err);
    //                     failed += 1;
    //                     continue;
    //                 }
    //             };
    //             let ani = match data {
    //                 ParsedData::Data(Data::CM3(CM3 {
    //                     scene: parser::SCN { ref ani, .. },
    //                     ..
    //                 }))
    //                 | ParsedData::Data(Data::SM3(SM3 {
    //                     scene: parser::SCN { ref ani, .. },
    //                     ..
    //                 })) => ani.get(),
    //                 _ => {
    //                     continue;
    //                 }
    //             };
    //             if let Some(ani) = ani {
    //                 for track in ani.track_map.iter().filter_map(|v| *v) {
    //                     if ani.get_track(track as usize).is_err() {
    //                         println!("Fail: {}", entry.path);
    //                         failed += 1;
    //                         continue 'outer;
    //                     }
    //                 }
    //             };
    //             let ParsedData::Data(data) = data else {
    //                 continue;
    //             };
    //             let data = serde_json::to_string_pretty(&data)?;
    //             let jd = &mut serde_json::Deserializer::from_str(&data);
    //             let data: Data = match serde_path_to_error::deserialize(jd) {
    //                 Ok(data) => data,
    //                 Err(err) => {
    //                     let col = err.inner().column();
    //                     let line = err.inner().line();
    //                     eprintln!(
    //                         "{} JSON_ROUNDTRIP_FAIL [{} {}:{}]: {}",
    //                         entry.path,
    //                         err.path(),
    //                         line,
    //                         col,
    //                         err
    //                     );
    //                     if err.path().to_string() == "." {
    //                         let suspects = find_numeric_null_candidates(&data, 16);
    //                         if !suspects.is_empty() {
    //                             eprintln!(
    //                                 "{} JSON_ROUNDTRIP_SUSPECT_PATHS: {}",
    //                                 entry.path,
    //                                 suspects.join(", ")
    //                             );
    //                         }
    //                     }
    //                     std::fs::write("roundtrip_error.json", &data)?;
    //                     fail_match += 1;
    //                     continue;
    //                 }
    //             };
    //             let mut orig_bytes = Vec::new();
    //             let mut orig_file = fs.open_file(&entry.path).unwrap();
    //             orig_file.read_to_end(&mut orig_bytes).unwrap();
    //             let mut buffer = Cursor::new(Vec::new());
    //             data.write_le(&mut buffer).unwrap();
    //             let buffer = buffer.into_inner();
    //             if buffer.len() != orig_bytes.len() {
    //                 eprintln!(
    //                     "{} LEN_MISMATCH: {} vs {}",
    //                     entry.path,
    //                     buffer.len(),
    //                     orig_bytes.len()
    //                 );
    //                 fail_match += 1;
    //                 continue;
    //             }
    //             if buffer != orig_bytes {
    //                 let first_diff = buffer
    //                     .iter()
    //                     .zip(orig_bytes.iter())
    //                     .take_while(|(a, b)| a == b)
    //                     .count();
    //                 println!(
    //                     "{path} @ 0x{first_diff:x} (0x{buf_len:x} vs 0x{orig_len:x})",
    //                     path = entry.path,
    //                     buf_len = buffer.len(),
    //                     orig_len = orig_bytes.len()
    //                 );
    //                 let start = first_diff.saturating_sub(16);
    //                 rhexdump!(&buffer[start..first_diff + 16], start as u64);
    //                 println!("---");
    //                 rhexdump!(&orig_bytes[start..first_diff + 16], start as u64);
    //                 // std::process::exit(1);
    //                 fail_match += 1;
    //                 continue;
    //             };
    //         }
    //     }
    //     println!(
    //         "CM3/SM3/EMI/DUM/AMC Parser: {}/{} parsed OK, {} failed to reconstruct",
    //         total - failed,
    //         total,
    //         fail_match
    //     );
    //     return Ok(());
    // }
    // {
    //     let mut dump_map: HashMap<String, HashMap<String, AnimTracks>> = HashMap::default();
    //     for entry in fs.entries()? {
    //         let parts: Vec<&str> = entry.path.split('/').filter(|s| !s.is_empty()).collect();
    //         if parts.len() > 4
    //             && parts[0] == "levels"
    //             && parts[2] == "map"
    //             && parts
    //                 .last()
    //                 .map(|p| p.ends_with("play.cm3"))
    //                 .unwrap_or(false)
    //         {
    //             let anm_name = format!("{}anm", parts[3]);
    //             if anm_name == parts[4] {
    //                 let level = parts[1];
    //                 let anm = parts[3];
    //                 let scene_path = ["levels", level, "map", anm, &format!("{anm}.sm3")].join("/");
    //                 println!("{level}: {anm}");
    //                 let exists = fs.exists(&scene_path).unwrap_or(false);
    //                 println!("{entry} -> {scene_path}: {exists}", entry = &entry.path,);
    //                 dump_map.insert(
    //                     entry.path.to_owned(),
    //                     dump_ani(&fs, &scene_path, &entry.path)?,
    //                 );
    //             }
    //         }
    //     }
    //     let mut fh = BufWriter::new(fs_err::File::create("anim.json")?);
    //     serde_json::to_writer(&mut fh, &dump_map)?;
    //     return Ok(());
    // }
    // {
    //     let mut track_map: HashMap<&str, AnimTracks> = HashMap::new();
    //     let model = "/levels/fake/map/fake2/fake2.sm3";
    //     let anm = "/levels/fake/map/fake2/fake2anm/play.cm3";
    //     let ParsedData::Data(Data::SM3(sm3)) = fs.parse_file(model)? else {
    //         bail!("Failed to parse model!")
    //     };
    //     let ParsedData::Data(Data::CM3(cm3)) = fs.parse_file(anm)? else {
    //         bail!("Failed to parse animation!")
    //     };
    //     let Some(ani) = cm3.scene.ani.get() else {
    //         bail!("No animation data found in CM3!");
    //     };
    //     println!("Model: {model}");
    //     println!("Animation: {anm}");
    //     for node in &sm3.scene.nodes {
    //         if node.object_index >= 0 {
    //             let idx: usize = node.object_index.try_into()?;
    //             if let Some(track) =
    //                 ani.track_map[idx].and_then(|v| ani.get_track(v as usize).ok().flatten())
    //             {
    //                 track_map.insert(&*node.name, track);
    //                 let nam = &ani.tracks[ani.track_map[idx].unwrap() as usize];
    //                 println!("{}: {:?}", node.name, nam.cm3_flags);
    //             }
    //         }
    //     }
    //     // let mut fh = BufWriter::new(fs_err::File::create("dump.json")?);
    //     // serde_json::to_writer(&mut fh, &track_map)?;
    //     return Ok(());
    // }
    let state = State {
        fs,
        browser_tree: None,
        browser_tree_error: None,
        data_path: std::env::args().nth(1),
        picked_object: None,
        data: None,
        show_ui: true,
        show_browser_panel: true,
        cam_physics: true,
        cam_auto_level: true,
        lightmaps: true,
        show_nodes: false,
        show_collision: false,
        lightmap_exposure: 10000.0,
        anim_textures: Vec::default(),
        node_size: 1000.0,
        edges: Vec::default(),
        af_decay: 50.0,
        thrust_power: 60_000.0,
        export: false,
    };
    let mut app = App::new();
    app.insert_resource(GlobalAmbientLight { ..default() })
        .insert_resource(WireframeConfig {
            global: false,
            default_color: Color::WHITE,
        })
        .insert_resource(EguiTexHandles::default())
        .insert_resource(state)
        .add_plugins((
            PackedAssetRepositoryPlugin::new(MultiPack::load_all(&packed_files)?),
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        mode: WindowMode::BorderlessFullscreen(MonitorSelection::Current),
                        title: "Scrap Asset Viewer".to_owned(),
                        present_mode: PresentMode::AutoVsync,
                        ..default()
                    }),
                    primary_cursor_options: Some(CursorOptions {
                        visible: true,
                        grab_mode: CursorGrabMode::Confined,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    mode: AssetMode::Unprocessed,
                    ..default()
                })
                .set(LogPlugin {
                    fmt_layer: |_| {
                        Some(Box::new(
                            bevy::log::tracing_subscriber::fmt::Layer::default()
                                .with_ansi(true)
                                .with_writer(std::io::stderr),
                        ))
                    },
                    custom_layer: bevy_debug_log::log_capture_layer,
                    ..default()
                }),
            // bevy_debug_log::LogViewerPlugin::default(),
            // WireframePlugin::default(),
            MeshPickingPlugin,
            EguiPlugin::default(),
            WorldInspectorPlugin::new().run_if(|state: Res<State>| state.show_ui),
            InputManagerPlugin::<DroneAction>::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                keyboard_handler,
                anim_debug,
                tree_overlay,
                render_amc,
                export::do_export.run_if(|state: Res<State>| state.export),
            ),
        )
        .add_systems(
            EguiPrimaryContextPass,
            (
                show_dummies,
                browser,
                inspector,
                ui_input_toggle,
                help_window,
                post_settings,
                DroneCam::update,
            ),
        )
        .add_systems(FixedUpdate, (autofocus, animate_camera, animate_textures))
        .add_plugins(MaterialPlugin::<ScrapMaterial>::default())
        .init_gizmo_group::<DefaultGizmoConfigGroup>()
        .init_asset::<asset_loader::TestAsset>();
    // .init_asset_loader::<asset_loader::TestLoader>();
    embedded_asset!(app, "shaders/test.wgsl");
    app.run();
    Ok(())
}

fn node_color(node: &parser::Node) -> Color {
    if let Some(node_data) = node.content.get() {
        match node_data {
            NodeData::Camera(_) => Color::linear_rgb(1., 1., 0.), // #ffff00
            NodeData::Dummy => Color::linear_rgb(1., 1., 1.),     // #ffffff
            NodeData::TriangleMesh => Color::linear_rgb(0., 0., 1.), // #0000ff
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

#[derive(Debug, Default)]
struct AnimState {
    cache: Vec<f32>,
    track_idx: usize,
    f_speed: f32,
    speed: f32,
    eye: Vec3,
    t: f32,
}

// TODO: adapt into method on ANI struct, returns Vec<(Isometry3d, FOV, LinearRGBA, Int, Vis)

fn anim_debug(state: Res<State>, time: Res<Time>, mut gizmos: Gizmos, mut did_run: Local<bool>) {
    let Some(ParsedData::Data(Data::CM3(cm3))) = &state.data else {
        return;
    };
    let Some(ani) = cm3.scene.ani.get() else {
        return;
    };
    let subframe = time.elapsed_secs() % ani.fps;
    let current_frame = (ani.first_frame as usize)
        + ((time.elapsed_secs() * ani.fps) as usize)
            % ((ani.last_frame - ani.first_frame) as usize);
    // dbg!(&ani.data.len());
    // dbg!(&ani.nabk.data.len());
    let total_size = ani
        .tracks
        .iter()
        .flat_map(|nam| &nam.tracks)
        .map(|block| block.size)
        .sum::<usize>();
    assert_eq!(total_size, ani.data.len());
    // println!("== DATA ==");
    let mut buffer = std::io::Cursor::new(&ani.data);
    for (track_idx, nam) in ani.tracks.iter().enumerate() {
        // println!("{nam:?}");
        let mut isometry = Isometry3d::IDENTITY;
        let mut active = false;
        let mut color = Oklcha::sequential_dispersed(track_idx as u32);
        for (block_idx, block) in nam.tracks.iter().enumerate() {
            let mut data = vec![0u8; block.size];
            buffer.read_exact(&mut data).unwrap();
            let mut fh = std::io::Cursor::new(&data);
            let mut block_start = nam.start_frame;
            let mut block_frame_count = nam.frames;
            let mut elem_count = block.size / block.elem_size;
            if block.stream && block.optimized {
                elem_count = (block.size - 6) / block.elem_size;
                let size = fh.read_u16::<LittleEndian>().unwrap();
                block_start = fh.read_u16::<LittleEndian>().unwrap().into();
                block_frame_count = fh.read_u16::<LittleEndian>().unwrap().into();
                assert_eq!(size as usize, block.size);
            }
            if !*did_run {
                /*
                println!("{block:?}");
                println!(
                    "NAM Start Frame: {}, Frames: {}",
                    nam.data.start_frame, nam.data.frames
                );
                println!("Element count: {elem_count}");
                println!(
                    "Anim Info: [Size: {block_size}, Start frame: {block_start}, Num frames: {block_frame_count}]",
                    block_size = block.size
                );
                assert!((block_start + block_frame_count) <= nam.data.frames);
                println!("=======================");
                */
            }
            // if block.stream || block.optimized {
            //     continue;
            // }
            match block.track_type {
                AniTrackType::Position => {
                    // let mut pos = Vec::new();
                    let mut dst = [0f32; 3];
                    let mut n: usize = nam.start_frame as usize;
                    while fh.position() != (fh.get_ref().len() as u64) {
                        fh.read_f32_into::<LittleEndian>(&mut dst).unwrap();
                        if n == current_frame {
                            isometry.translation = Vec3::from_array(transform_pos(dst)).into();
                            active = true;
                        }
                        n += 1;
                    }
                    // println!("{pos:?}");
                } // Pos,
                AniTrackType::Rotation => {
                    let mut dst = [0f32; 4];
                    let mut n: usize = nam.start_frame as usize;
                    while fh.position() != (fh.get_ref().len() as u64) {
                        fh.read_f32_into::<LittleEndian>(&mut dst).unwrap();
                        if n == current_frame {
                            isometry.rotation = Quat::from_array(dst);
                            active = true;
                        }
                        n += 1;
                    }
                }
                AniTrackType::Color => {
                    let mut dst = [0u8; 4];
                    let mut n: usize = nam.start_frame as usize;
                    while fh.position() != (fh.get_ref().len() as u64) {
                        fh.read_exact(&mut dst).unwrap();
                        if n == current_frame {
                            color = LinearRgba::from_u8_array(dst).into();
                            active = true;
                        }
                        n += 1;
                    }
                }
                AniTrackType::Visibility => {
                    let mut n: usize = nam.start_frame as usize;
                    while fh.position() != (fh.get_ref().len() as u64) {
                        let val = fh.read_u8().unwrap();
                        if n == current_frame {
                            active = val == 1;
                        }
                        n += 1;
                    }
                } // Vis,
                other => {
                    warn!("Unknown animation block id: {other:?}")
                }
            }
        }
        if active {
            gizmos.sphere(isometry, 10.0, color);
            gizmos.axes(isometry, 10.0);
        }
    }
    assert_eq!(buffer.position(), ani.data.len() as u64);
    // std::process::exit(0);
    *did_run = true;
}

fn animate_camera(
    mut cam_transform: Query<&mut Transform, With<Camera>>,
    state: ResMut<State>,
    time: Res<Time>,
    mut l_state: Local<AnimState>,
    mut gizmos: Gizmos,
    keyboard: Res<ButtonInput<KeyCode>>,
) -> BevyResult {
    return Ok(());
    let Some(ParsedData::Level(lvl)) = &state.data else {
        return Ok(());
    };
    let AnimState {
        cache,
        track_idx,
        f_speed,
        speed,
        eye,
        t,
    } = l_state.deref_mut();
    let mut look_transform = cam_transform.single_mut()?;
    // let mut cam_ctrl = cam_ctrl.single_mut();
    for key in keyboard.get_just_pressed() {
        match key {
            KeyCode::KeyQ => {
                *track_idx = track_idx.saturating_sub(1);
                cache.clear();
            }
            KeyCode::KeyE => {
                *track_idx = track_idx.saturating_add(1);
                cache.clear();
            }
            _ => (),
        }
    }
    for key in keyboard.get_pressed() {
        match key {
            KeyCode::KeyW => {
                *speed += 100.0;
                dbg!(&speed);
            }
            KeyCode::KeyS => {
                *speed -= 100.0;
                dbg!(&speed);
            }
            KeyCode::KeyD => {
                *f_speed += 100.0;
                dbg!(&f_speed);
            }
            KeyCode::KeyA => {
                *f_speed -= 100.0;
                dbg!(&f_speed);
            }
            _ => (),
        }
    }
    *f_speed = f_speed.max(1.0);
    let mut track: Vec<_> = lvl
        .dummies
        .dummies
        .iter()
        .filter(|d| d.name.starts_with(&format!("DM_Track{}_", *track_idx)))
        .collect();
    if track.is_empty() {
        return Ok(());
    }
    track.sort_by_key(|d| d.name.split("_").last().unwrap().parse::<usize>().unwrap());
    let dt = time.delta_secs();
    *t += dt * *speed;
    let p: Vec<Vec3> = track.iter().map(|d| transform_pos(d.pos).into()).collect();
    let curve = CubicBSpline::new(p).to_curve_cyclic().unwrap();
    if cache.is_empty() {
        const N: usize = 1000;
        *cache = curve
            .iter_positions(N)
            .tuple_windows()
            .scan(0f32, |acc, (a, b)| {
                *acc += (a - b).length();
                Some(*acc)
            })
            .collect_vec();
    }
    let t = *t / cache.last().unwrap();
    let dist = t.fract() * cache.last().unwrap();
    let mut t_val = t.fract();
    for (i, (&a, &b)) in cache.iter().tuple_windows().enumerate() {
        if (a..b).contains(&dist) {
            let n = cache.len() as f32;
            let i = i as f32;
            t_val = dist.remap(a, b, i / (n - 1.0), (i + 1.0) / (n - 1.0));
        }
    }
    let pos = curve.position((curve.segments().len() as f32) * t_val);
    eye.smooth_nudge(&pos, f32::ln(*f_speed), dt);
    *look_transform = look_transform.looking_at(pos, Vec3::Y);
    look_transform.translation = *eye;
    gizmos.sphere(
        Isometry3d::from_translation(pos),
        state.node_size,
        LinearRgba::RED,
    );
    Ok(())
}

fn show_text(
    contexts: &mut EguiContexts,
    text: &str,
    pos: Vec3,
    cam: &Camera,
    win: &Window,
    t_cam: &GlobalTransform,
    size: f32,
) -> BevyResult {
    let size = 20.0 * size;
    let ctx = contexts.ctx_mut()?;
    if let Ok(screen_pos) = cam.world_to_viewport(t_cam, pos) {
        if screen_pos.x < 0.0 || screen_pos.x > win.width() {
            return Ok(());
        }
        if screen_pos.y < 0.0 || screen_pos.y > win.height() {
            return Ok(());
        }
        let d_world = (pos - t_cam.translation()).length();
        if d_world < 5000.0
            && let Some(layer_id) = ctx.top_layer_id()
        {
            ctx.layer_painter(layer_id).text(
                [screen_pos.x, screen_pos.y].into(),
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::monospace((size / d_world).clamp(0.0, size)),
                egui::Color32::WHITE,
            );
        }
    };
    return Ok(());
}

fn show_dummies(
    mut gizmos: Gizmos,
    state: Res<State>,
    cam: Query<(&Camera, &GlobalTransform)>,
    mut contexts: EguiContexts,
    win: Query<&Window, With<PrimaryWindow>>,
) -> BevyResult {
    if !state.show_nodes {
        return Ok(());
    }
    let win = win.single()?;
    let (cam, t_cam) = cam.single()?;
    let Some(ParsedData::Level(lvl)) = state.data.as_ref() else {
        return Ok(());
    };
    for dum in &lvl.dummies.dummies {
        let pos: Vec3 = transform_pos(dum.pos).into();
        show_text(
            &mut contexts,
            &dum.name,
            pos,
            cam,
            win,
            t_cam,
            state.node_size,
        )?;
        gizmos.sphere(
            Isometry3d::new(
                pos,
                Quat::from_euler(EulerRot::XYZ, dum.rot[0], dum.rot[1], dum.rot[2]),
            ),
            state.node_size / 10.0,
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
            let mut label = format!(
                "{name} [{ty} {flags:?}]",
                name = &*node.name,
                ty = node_type(node.content.get()),
                flags = node.flags
            );
            if let Some(info) = node.info.get() {
                label.push_str(&format!("\n{info}"));
            }
            show_text(
                &mut contexts,
                &label,
                transform_pos(node.pos_offset).into(),
                cam,
                win,
                t_cam,
                state.node_size,
            )?;
            gizmos.sphere(
                Isometry3d::new(
                    transform_pos(node.pos_offset),
                    Quat::from_mat4(&Mat4::from_cols_array_2d(&node.transform_world)),
                ),
                state.node_size / 10.0,
                node_color(node),
            );
        }
    }
    Ok(())
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

fn load_texture(path: &str, fs: &MultiPackFS) -> Result<Image> {
    info!("Loading texture {path}");
    let Ok(mut fh) = fs.open_file(path) else {
        bail!("Failed to open file");
    };
    let ext = path.split(".").last().unwrap_or_default();
    let mut data = vec![];
    fh.read_to_end(&mut data).unwrap();
    let img = Image::from_buffer(
        &data,
        ImageType::Extension(ext),
        CompressedImageFormats::all(),
        true,
        ImageSampler::Descriptor(ImageSamplerDescriptor {
            label: Some(path.to_owned()),
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

fn make_material(
    mat: &parser::MAT,
    textures: &HashMap<String, Handle<Image>>,
    mat_props: &HashMap<&str, &str>,
) -> Result<(StandardMaterial, HashMap<Slot, Option<Handle<Image>>>)> {
    let maps = vec![
        (Slot::Diffuse, &mat.maps[0]),
        (Slot::Metallic, &mat.maps[1]),
        (Slot::Reflection, &mat.maps[2]),
        (Slot::Bump, &mat.maps[3]),
        (Slot::Glow, &mat.maps[4]),
    ];
    for (slot, map) in maps {
        let Some(map) = map.get().map(|m: &parser::MAP| &*m.texture) else {
            continue;
        };
        println!("{slot:?}: {map}");
    }

    let has_alpha = mat.maps[0]
        .get()
        .map(|m| m.texture.contains(".alpha."))
        .unwrap_or(false);
    let base_color_texture = mat.maps[0]
        .get()
        .and_then(|m| textures.get(&*m.texture))
        .cloned();
    let emissive_texture = mat.maps[4]
        .get()
        .and_then(|m| textures.get(&*m.texture))
        .cloned();

    // B=metallic
    // G=roughness
    let metallic_roughness_texture = mat.maps[1]
        .get()
        .and_then(|m| textures.get(&*m.texture))
        .cloned();
    let normal_map_texture = mat.maps[3]
        .get()
        .and_then(|m| textures.get(&*m.texture))
        .cloned();

    let reflection_map = mat.maps[2]
        .get()
        .and_then(|m| textures.get(m.texture.string.as_str()))
        .cloned();

    let map_tex: HashMap<Slot, Option<Handle<Image>>> = [
        (Slot::Diffuse, base_color_texture.clone()),
        (Slot::Metallic, metallic_roughness_texture.clone()),
        (Slot::Reflection, reflection_map.clone()),
        (Slot::Bump, normal_map_texture.clone()),
        (Slot::Glow, emissive_texture.clone()),
    ]
    .into_iter()
    .collect();

    let mut mat = StandardMaterial {
        base_color: Color::WHITE.with_alpha(1.0),
        base_color_texture,
        emissive_texture,
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
        Some(_) => (LinearRgba::WHITE * 10.0).with_alpha(1.0),
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
            mat.emissive_texture = mat.base_color_texture.clone();
            mat.emissive = Color::WHITE.to_linear() * 5.0;
            mat.alpha_mode = AlphaMode::Premultiplied;
            mat.base_color = Color::LinearRgba(Color::WHITE.to_linear() * 5.0);
            mat.base_color.set_alpha(0.0);
            mat.unlit = true;
        }
        "" => {}
        other => {
            warn!("Shader {other} not implemented!");
        }
    }
    Ok((mat, map_tex))
}

fn load_level(
    state: &mut State,
    mut commands: Commands,
    mut fog: &mut DistanceFog,
    mut images: ResMut<Assets<Image>>,
    ass: ResMut<AssetServer>,
    mut amb: ResMut<GlobalAmbientLight>,
    mut meshes: ResMut<Assets<Mesh>>,
    material_res: ResMut<Assets<ScrapMaterial>>,
) -> [f32; 3] {
    let Some(ParsedData::Level(level)) = state.data.as_ref() else {
        return [0.0, 0.0, 0.0];
    };
    let texture_path = level
        .config
        .get("model")
        .and_then(|inner_map| inner_map.get("texturepath"))
        .and_then(|opt_str| opt_str.as_deref());

    for [m1, m2] in &level.collisions.sector_col {
        for m in &[m1, m2] {
            info!("Loading collision data for {}", m.zone_name);
            let pos = m
                .verts
                .data
                .iter()
                .copied()
                .map(|p| Vec3::from(transform_pos(p)))
                .collect_vec();
            let tris = m
                .tris
                .data
                .iter()
                .map(|t| {
                    let mut idx = t.idx.map(|v| v as u32);
                    idx.swap(0, 1);
                    idx
                })
                .collect_vec();
            if pos.is_empty() || tris.is_empty() {
                continue;
            }
            // let mesh = Mesh::new(PrimitiveTopology::TriangleList);
        }
    }

    let mut fog_falloff = FogFalloff::Exponential { density: 0.0 };
    let mut fog_color = Srgba::BLACK.with_alpha(0.0);

    for sm3 in level.sm3.iter().flatten() {
        dbg!(&sm3.scene.ambient);
        dbg!(&sm3.scene.background);
        dbg!(&sm3.scene.bbox);
        if let Some(props) = sm3.scene.node_props.get() {
            let props = props.data();
            if props
                .getboolcoerce("niebla", "activa")
                .unwrap()
                .unwrap_or(false)
            {
                if props
                    .getboolcoerce("niebla", "lineal")
                    .unwrap()
                    .unwrap_or(false)
                {
                    let start = props
                        .getfloat("niebla", "rango_ini")
                        .unwrap()
                        .unwrap_or(0.0) as f32;
                    let end = props
                        .getfloat("niebla", "rango_fin")
                        .unwrap()
                        .unwrap_or(f64::INFINITY) as f32;
                    fog_falloff = FogFalloff::Linear {
                        start,
                        end: end * 1_000_000.0,
                    }
                } else {
                    let density =
                        props.getfloat("niebla", "densidad").unwrap().unwrap_or(0.0) as f32;
                    fog_falloff = FogFalloff::Exponential { density };
                }
                if let Some(c) = props.get("niebla", "color") {
                    let col: Vec<&str> = c.split_ascii_whitespace().collect();
                    match col.as_slice() {
                        [r, g, b, a] => {
                            let r = r.parse::<u8>().unwrap_or_default();
                            let g = g.parse::<u8>().unwrap_or_default();
                            let b = b.parse::<u8>().unwrap_or_default();
                            fog_color = Srgba::rgb_u8(r, g, b)
                                .with_alpha(a.parse::<f32>().unwrap_or_default());
                        }
                        [r, g, b] => {
                            let r = r.parse::<u8>().unwrap_or_default();
                            let g = g.parse::<u8>().unwrap_or_default();
                            let b = b.parse::<u8>().unwrap_or_default();
                            fog_color = Srgba::rgb_u8(r, g, b).with_alpha(1.0);
                        }
                        _ => {}
                    }
                }
            };
            eprintln!("Node: {props:#?}", props = props.get_map_ref());
        }
        *amb = GlobalAmbientLight {
            color: {
                let col = &sm3.scene.ambient;
                Srgba::from_u8_array_no_alpha([col.color.r, col.color.g, col.color.b]).into()
            },
            brightness: sm3.scene.ambient.intensity * 120.0,
            affects_lightmapped_meshes: true,
        };
        if let Some(props) = sm3.scene.user_props.get() {
            let props = props.data();
            eprintln!("User: {props:#?}", props = props.get_map_ref());
        }
    }
    // std::process::exit(1);
    *fog = DistanceFog {
        falloff: fog_falloff,
        color: fog_color.into(),
        ..Default::default()
    };
    // std::process::exit(1);
    commands
        .spawn((Transform::default(), Visibility::default()))
        .with_children(|parent| {
            for sm3 in level.sm3.iter().flatten() {
                for node in &sm3.scene.nodes {
                    let Some(NodeData::Light(light)) = node.content.get() else {
                        continue;
                    };
                    continue;
                    let transform = Transform::from_isometry(Isometry3d::new(
                        transform_pos(node.pos_offset),
                        Quat::from_mat4(&Mat4::from_cols_array_2d(&node.transform_world)),
                    ));
                    println!("{name:?} {transform:?} {}", name = node.name);
                    let color = Color::LinearRgba(LinearRgba::from_u8_array([
                        light.color.r,
                        light.color.g,
                        light.color.b,
                        light.color.a,
                    ]));
                    let mut ent = match light.light_type {
                        parser::LightType::Point => parent.spawn((
                            transform,
                            PointLight {
                                color,
                                intensity: light.power * light.mult * 0.01,
                                radius: 0.0,
                                range: 1000.0,
                                affects_lightmapped_mesh_diffuse: true,
                                shadows_enabled: light.shadows == 1,
                                ..default()
                            },
                        )),
                        parser::LightType::Spot => parent.spawn((
                            transform,
                            SpotLight {
                                color,
                                intensity: light.power * light.mult * 0.01,
                                radius: 0.0,
                                range: 10_000.0,
                                affects_lightmapped_mesh_diffuse: true,
                                shadows_enabled: light.shadows == 1,
                                ..default()
                            },
                        )),
                        parser::LightType::Directional => parent.spawn((
                            transform,
                            DirectionalLight {
                                color,
                                illuminance: light.power * light.mult * 0.01,
                                affects_lightmapped_mesh_diffuse: true,
                                shadows_enabled: light.shadows == 1,
                                ..default()
                            },
                        )),
                    };
                    let name: String = node.name.string.clone();
                    ent.insert(Name::new(name));
                }
            }

            let mut anim_tex = HashMap::new();
            let mut anim_mat: AnimMat = AnimMat::new();
            let mut textures = HashMap::new();
            let prop_re = Regex::new(r"\(\+(\w+)(?::(\w*))?\)").unwrap(); // Matches (+key:value) and (+key)
            for (key, path) in &level.dependencies {
                // let mut data = vec![];
                if let Some((name, _)) = path.rsplit_once(".") {
                    let dds_name = name.replace("/dds/", "/");
                    let txa_name = format!("{dds_name}.txa");
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
                                match load_texture(&path, &state.fs) {
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
                let img = match load_texture(path, &state.fs) {
                    Ok(img) => img,
                    Err(err) => {
                        warn!("Failed to load {key}: {err}");
                        let Some(tex_path) = texture_path else {
                            continue;
                        };
                        let path = format!("{tex_path}/{path}");
                        match load_texture(&path, &state.fs) {
                            Ok(img) => img,
                            Err(err) => {
                                warn!("Failed to load {key}: {err}");
                                continue;
                            }
                        }
                    }
                };
                textures.insert(key.to_owned(), ass.add(img));
            }
            let mut materials = BTreeMap::new();
            for (key, scrap_mat) in &level.emi.materials {
                let mat_name = scrap_mat
                    .name
                    .as_ref()
                    .map(|p| p.string.clone())
                    .unwrap_or_else(|| format!("MAT:{key}"));
                let colors = scrap_mat.colors();
                info!("Loading material {mat_name}");
                let mut occ_slots = vec![false; scrap_mat.maps.len()];
                for (s, m) in occ_slots.iter_mut().zip(scrap_mat.maps.iter()) {
                    *s = m.get().is_some();
                }
                /*
                {'[true, true, true, false, false]': {'MaskEnvmap'},
                '[true, false, false, false, false]': {'Diffuse'},
                '[true, true, false, false, false]': {'Diffuse'},
                '[true, true, true, false, true]': {'GlowmapMaskEnvmap', 'MaskEnvmap'},
                '[true, false, false, false, true]': {'Glowmap'},
                '[true, true, false, false, true]': {'Glowmap'},
                '[false, false, false, false, false]': {'Diffuse'},
                '[false, false, true, false, false]': {'EnvMap'}}
                */
                let mat_props: HashMap<&str, &str> = prop_re
                    .captures_iter(&mat_name)
                    .filter_map(|g| {
                        let c: Vec<_> = g.iter().map(|c| c.map(|c| c.as_str())).collect();
                        let key: Option<&str> = c.get(1).and_then(|c| *c);
                        let value: &str = c.get(2).and_then(|c| *c).unwrap_or("");
                        key.map(|k| (k, value))
                    })
                    .collect();
                // info!("Material properties: {mat_props:?}");
                let maps = vec![
                    (Slot::Diffuse, &scrap_mat.maps[0]),
                    (Slot::Metallic, &scrap_mat.maps[1]),
                    (Slot::Reflection, &scrap_mat.maps[2]),
                    (Slot::Bump, &scrap_mat.maps[3]),
                    (Slot::Glow, &scrap_mat.maps[4]),
                ];
                let mut map_names = vec![];
                for (slot, map) in maps {
                    let map: String = map
                        .get()
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

                let has_alpha = scrap_mat.maps[0]
                    .get()
                    .map(|m| m.texture.contains(".alpha."))
                    .unwrap_or(false);
                let base_color_texture = scrap_mat.maps[0]
                    .get()
                    .and_then(|m| textures.get(&*m.texture))
                    .cloned();
                let emissive_texture = scrap_mat.maps[4]
                    .get()
                    .and_then(|m| textures.get(&*m.texture))
                    .cloned();

                // B=metallic
                // G=roughness
                let metallic_roughness_texture = scrap_mat.maps[1]
                    .get()
                    .and_then(|m| textures.get(&*m.texture))
                    .cloned();
                let normal_map_texture = scrap_mat.maps[3]
                    .get()
                    .and_then(|m| textures.get(&*m.texture))
                    .cloned();

                let reflection_map = scrap_mat.maps[2]
                    .get()
                    .and_then(|m| textures.get(m.texture.string.as_str()))
                    .cloned();
                let map_tex: BTreeMap<Slot, Option<Handle<Image>>> = [
                    (Slot::Diffuse, base_color_texture.clone()),
                    (Slot::Metallic, metallic_roughness_texture.clone()),
                    (Slot::Reflection, reflection_map.clone()),
                    (Slot::Bump, normal_map_texture.clone()),
                    (Slot::Glow, emissive_texture.clone()),
                ]
                .into_iter()
                .collect();
                // TODO: fix transparency issues with alpha_test
                let mut mat = StandardMaterial {
                    // base_color: Color::Srgba(colors.diffuse),
                    base_color: Color::WHITE,
                    base_color_texture,
                    emissive_texture,
                    metallic_roughness_texture,
                    // emissive: Color::Srgba(colors.emissive).to_linear(),
                    normal_map_texture,
                    perceptual_roughness: (2.0 / (2.0 + colors.power)).powf(0.25),
                    metallic: 0.0,
                    reflectance: 0.5,
                    // specular_tint: Color::Srgba(colors.specular),
                    double_sided: scrap_mat.mat_props.two_sided == 1,
                    cull_mode: Some(Face::Back),
                    ..default()
                };

                mat.emissive = match mat.emissive_texture {
                    Some(_) => (LinearRgba::WHITE * 10.0).with_alpha(1.0),
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
                        mat.emissive_texture = mat.base_color_texture.take();
                        // mat.emissive = Color::WHITE.to_linear() * 5.0;
                        // mat.alpha_mode = AlphaMode::Premultiplied;
                        // mat.base_color = Color::LinearRgba(Color::WHITE.to_linear() * 5.0);
                        // mat.base_color.set_alpha(0.0);
                        // mat.unlit = true;
                    }
                    "" => {}
                    other => {
                        warn!("Shader {other} not implemented!");
                    }
                }
                let mat = ass.add(ExtendedMaterial {
                    base: mat,
                    extension: TestMaterial::default(),
                });
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
                materials.insert(key, (mat_name, map_names, map_tex, mat, scrap_mat));
            }
            let mut lmaps = HashMap::new();
            let mut lmap_names = HashMap::new();
            for lm in level.emi.maps.iter() {
                if let Some((key_1, _, key_2)) = &lm.data {
                    lmap_names.insert(lm.key, (key_1.string.clone(), key_2.string.clone()));
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
                let (mat_name, map_names, map_tex, mat, scrap_mat) = &materials[&data.mat_key];
                let mat_inst = material_res.get(mat);
                info!("Loading mesh {name} with material {mat_name}");
                // dbg!(&tri.zone_index);
                let mesh_props: HashMap<&str, &str> = prop_re
                    .captures_iter(name)
                    .filter_map(|g| {
                        let c: Vec<_> = g.iter().map(|c| c.map(|c| c.as_str())).collect();
                        let key: Option<&str> = c.get(1).and_then(|c| *c);
                        let value: &str = c.get(2).and_then(|c| *c).unwrap_or("");
                        key.map(|k| (k, value))
                    })
                    .collect();
                if !mesh_props.is_empty() {
                    info!("Mesh properties: {mesh_props:?}");
                }
                let faces = &data.tris;
                for verts in [&data.geometry_verts, &data.lightmap_verts]
                    .iter()
                    .filter_map(|v| v.inner.as_ref())
                {
                    let mesh = mesh_from_m3d(faces, &verts.data);
                    let needs_normals = verts.data.iter().any(|v| v.normal.is_none())
                        || mat_inst
                            .as_ref()
                            .and_then(|m| m.base.normal_map_texture.as_ref())
                            .is_some();
                    if needs_normals {
                        // mesh.duplicate_vertices();
                        // mesh.compute_flat_normals();
                    }
                    // mesh.generate_tangents()
                    //     .expect("Failed to generate tangents");
                    // TODO: custom materials based on shaders
                    let scrap_mat = (**scrap_mat).clone();
                    let mut pbr = parent.spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(mat.clone()),
                        Name::new(name.to_owned()),
                        MaterialName(mat_name.to_owned()),
                        MapNames(map_names.clone()),
                        MapTex(map_tex.clone()),
                        ScrapMat(scrap_mat),
                    ));
                    pbr.observe(mesh_clicked);
                    if !mesh_props.contains_key("no_lightmap") {
                        if let Some((name_1, name_2)) = lmap_names.get(&data.map_key) {
                            pbr.insert(LightmapNames(name_1.clone(), name_2.clone()));
                        }
                        if let Some((lmap_1, lmap_2)) = lmaps.get(&data.map_key) {
                            pbr.insert(LightmapHandles(lmap_1.clone(), lmap_2.clone()));
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
            }
        });
    for dummy in &level.dummies.dummies {
        if dummy.name.starts_with("DM_Player_Spawn") {
            return transform_pos(dummy.pos);
        }
        if dummy.name.starts_with("DM_Ship_Spawn") {
            return transform_pos(dummy.pos);
        }
    }
    return [0.0, 0.0, 0.0];
}

fn mesh_clicked(
    trigger: On<Pointer<Click>>,
    mut contexts: EguiContexts,
    mut commands: Commands,
    mut state: ResMut<State>,
) -> BevyResult {
    let ctx = contexts.ctx_mut()?;
    let egui_waints_input = ctx.wants_keyboard_input() || ctx.wants_pointer_input();
    if egui_waints_input {
        return Ok(());
    }
    if trigger.button != PointerButton::Primary {
        return Ok(());
    }
    let ent = trigger.entity;
    if let Some(old_ent) = state.picked_object.replace(ent)
        && let Ok(mut e) = commands.get_entity(old_ent)
    {
        e.remove::<WireframeColor>().remove::<Wireframe>();
    }
    commands
        .entity(ent)
        .insert(Wireframe)
        .insert(WireframeColor {
            color: Oklcha::sequential_dispersed(ent.index().index()).into(),
        });
    Ok(())
}

fn ui_input_toggle(mut contexts: EguiContexts) -> BevyResult {
    let egui_ctx = contexts.ctx_mut()?;
    let _ = !(egui_ctx.wants_keyboard_input() || egui_ctx.wants_pointer_input());
    Ok(())
    //cam_ctrl.single_mut().enabled = enabled;
}

fn tree_overlay(state: Res<State>, mut gizmos: Gizmos) {
    for (p1, p2) in &state.edges {
        gizmos.line(*p1, *p2, Color::WHITE);
    }
}

fn autofocus(
    mut ray_cast: MeshRayCast,
    mut cam: Query<(&Camera, &Transform, Option<&mut DepthOfField>)>,
    time: Res<Time>,
    state: Res<State>,
) {
    let mut cam = cam.single_mut().unwrap();
    let (_, tr, Some(ref mut dof)) = cam else {
        return;
    };
    let Ok(dir) = Dir3::new(*tr.forward()) else {
        return;
    };
    let cam_ray = Ray3d::new(tr.translation, dir);
    let Some((_, hit)) = ray_cast
        .cast_ray(cam_ray, &MeshRayCastSettings { ..default() })
        .first()
    else {
        return;
    };
    dof.focal_distance.smooth_nudge(
        &hit.distance,
        f32::ln(state.af_decay),
        time.delta().as_secs_f32(),
    );
}

pub(crate) struct MatColors {
    pub diffuse: Srgba,
    pub ambient: Srgba,
    pub specular: Srgba,
    pub emissive: Srgba,
    pub power: f32,
}

impl parser::MAT {
    fn shader(&self) -> &'static str {
        todo!("port 0x6a9050");
    }
    fn colors(&self) -> MatColors {
        const SCALE: f32 = 1.0 / 255.0;
        // EngineVars::MatEmissive.value is 0.0 in your context
        const MAT_EMISSIVE: f32 = 0.0;

        // Helper to convert u8 RGBA to normalized f32 array
        let norm = |c: &parser::RGBA| [c.r, c.g, c.b, c.a].map(|v| v as f32 * SCALE);

        let glow = norm(&self.glow);
        let mod_color = norm(&self.diffuse_mod);
        let amb_override = norm(&self.ambient_override);

        // 1. Initialize Diffuse and Ambient
        // C++: Diffuse starts as self.diffuse, Alpha 1.0
        let mut diffuse = norm(&self.diffuse);
        diffuse[3] = 1.0;

        // C++: Ambient starts as White (1.0, 1.0, 1.0, 1.0)
        let mut ambient = [1.0, 1.0, 1.0, 1.0];

        // 2. Logic if no diffuse map (maps[0] == NULL)
        if self.maps[0].get().is_none() {
            // Apply Diffuse Mod
            diffuse[0] *= mod_color[0];
            diffuse[1] *= mod_color[1];
            diffuse[2] *= mod_color[2];

            // Set Ambient based on attrib flag
            // C++: if (attrib & 0x80) != 0 (UseAmbient)
            if self
                .mat_props
                .attrib
                .contains(&parser::MatPropAttrib::USE_AMBIENT)
            {
                ambient[0] = diffuse[0] * 0.8;
                ambient[1] = diffuse[1] * 0.8;
                ambient[2] = diffuse[2] * 0.8;
            } else {
                ambient[0] = amb_override[0];
                ambient[1] = amb_override[1];
                ambient[2] = amb_override[2];
            }
        }

        // 3. Calculate Specular
        // C++: (specular * spec_mult) / 255.0
        // Since norm() already divides by 255.0, we just multiply by spec_mult
        let spec = norm(&self.specular);
        let spec_mult = self.spec_mult;
        let specular = [
            spec[0] * spec_mult,
            spec[1] * spec_mult,
            spec[2] * spec_mult,
            1.0,
        ];

        // 4. Apply Glow/Emissive interactions
        // C++: Emissive is overwritten with (Diffuse * Glow) + (Diffuse * 0)
        let mut emissive = [
            diffuse[0] * glow[0],
            diffuse[1] * glow[1],
            diffuse[2] * glow[2],
            1.0,
        ];

        // C++: Add EngineVars contribution (0 here)
        emissive[0] += diffuse[0] * MAT_EMISSIVE;
        emissive[1] += diffuse[1] * MAT_EMISSIVE;
        emissive[2] += diffuse[2] * MAT_EMISSIVE;

        // C++: Diffuse *= (1.0 - Glow)
        diffuse[0] *= 1.0 - glow[0];
        diffuse[1] *= 1.0 - glow[1];
        diffuse[2] *= 1.0 - glow[2];

        // C++: Ambient *= (1.0 - Glow)
        ambient[0] *= 1.0 - glow[0];
        ambient[1] *= 1.0 - glow[1];
        ambient[2] *= 1.0 - glow[2];

        // 5. Finalize Alphas
        // C++: Diffuse.a set to mod_color.a (overwriting previous calcs)
        diffuse[3] = mod_color[3];
        // C++: Ambient.a set to 0.0
        ambient[3] = 0.0;

        MatColors {
            diffuse: Srgba::new(diffuse[0], diffuse[1], diffuse[2], diffuse[3]),
            ambient: Srgba::new(ambient[0], ambient[1], ambient[2], ambient[3]),
            specular: Srgba::new(specular[0], specular[1], specular[2], specular[3]),
            emissive: Srgba::new(emissive[0], emissive[1], emissive[2], emissive[3]),
            power: self.spec_power,
        }
    }
}

fn inspector(
    state: Res<State>,
    images: Res<Assets<Image>>,
    mut tex: ResMut<EguiTexHandles>,
    mut contexts: EguiContexts,
    name: Query<&Name>,
    mat: Query<(
        &MeshMaterial3d<ScrapMaterial>,
        &MaterialName,
        &MapNames,
        &MapTex,
        &ScrapMat,
    )>,
) -> BevyResult {
    if state.show_ui {
        // if let Some((_, _, map_names, map_tex, scrap_mat)) =
        //     state.picked_object.and_then(|ent| mat.get(ent).ok())
        // {
        //     for (slot, tex_name) in map_names.iter() {
        //         let Some(map_tex) = map_tex.get(slot).and_then(|v| v.as_ref()) else {
        //             continue;
        //         };
        //         tex.entry(tex_name.clone()).or_insert_with(|| {
        //             (
        //                 map_tex.clone(),
        //                 contexts.add_image(
        //                     bevy_inspector_egui::bevy_egui::EguiTextureHandle::Strong(
        //                         map_tex.clone(),
        //                     ),
        //                 ),
        //             )
        //         });
        //     }
        // };

        egui::Window::new("Inspector").show(contexts.ctx_mut()?, |ui| {
            let Some(ent) = state.picked_object else {
                return;
            };
            if let Ok(name) = name.get(ent) {
                ui.heading(format!("{name} ({ent:?})"));
            } else {
                ui.heading(format!("{ent:?}"));
            }
            let Ok((_, mat_name, map_names, _, scrap_mat)) = mat.get(ent) else {
                return;
            };
            let mat_name = mat_name.as_str();
            ui.label(format!("Material: {mat_name}"));
            ui.label(format!(
                "Orig Blend Mode: {:?}",
                scrap_mat.mat_props.src_blend
            ));
            ui.label(format!(
                "Dest Blend Mode: {:?}",
                scrap_mat.mat_props.dst_blend
            ));
            ui.label(format!("Two Sided: {}", scrap_mat.mat_props.two_sided != 0));
            ui.label(format!(
                "Dyn. Illum.: {}",
                scrap_mat.mat_props.dyn_illum != 0
            ));
            ui.label(format!(
                "Dif. Alpha: {}",
                scrap_mat.mat_props.diffuse_alpha != 0
            ));
            ui.label(format!("Env Map: {}", scrap_mat.mat_props.env_map != 0));
            ui.label(format!("Attrib: {:?}", scrap_mat.mat_props.attrib));
            ui.label(format!("Z-Write: {:?}", scrap_mat.mat_props.z_write != 0));
            ui.label(format!("Z-Func: {:?}", scrap_mat.mat_props.zfunc));
            ui.label("Raw:");
            for (label, col) in [
                ("Ambient Override", &scrap_mat.ambient_override),
                ("Diffuse", &scrap_mat.diffuse),
                ("Diffuse Mod", &scrap_mat.diffuse_mod),
                ("Specular", &scrap_mat.specular),
                ("Glow", &scrap_mat.glow),
            ] {
                let mut col = col.as_array();
                ui.horizontal(|ui| {
                    ui.label(label);
                    ui.color_edit_button_rgba_unmultiplied(&mut col);
                });
            }
            ui.label("Computed:");
            let colors = scrap_mat.colors();
            for (label, col) in [
                ("Diffuse", &colors.diffuse),
                ("Ambient", &colors.ambient),
                ("Emissive", &colors.emissive),
                ("Specular", &colors.specular),
            ] {
                let col = &mut [col.red, col.green, col.blue, col.alpha];
                ui.horizontal(|ui| {
                    ui.label(label);
                    ui.color_edit_button_rgba_unmultiplied(col);
                });
            }
            ui.label(format!("Specular Power: {}", colors.power));
            // ui.label(format!("Raw\n{:#?}", scrap_mat));
        });
    }
    Ok(())
}

fn keyboard_handler(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut wireframe: ResMut<WireframeConfig>,
    mut exit: MessageWriter<AppExit>,
    mut state: ResMut<State>,
    mut win: Query<&mut Window, With<PrimaryWindow>>,
    mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut commands: Commands,
    mut cam: Query<(&Camera, &Transform, Option<&mut DepthOfField>)>,
) -> BevyResult {
    let (_, _, mut dof) = cam.single_mut()?;
    for key in keyboard.get_pressed() {
        match key {
            KeyCode::NumpadAdd => {
                state.lightmap_exposure += 1000.0;
            }
            KeyCode::NumpadSubtract => {
                state.lightmap_exposure -= 1000.0;
            }
            KeyCode::NumpadMultiply => {
                state.lightmap_exposure *= 1.1;
            }
            KeyCode::NumpadDivide => {
                state.lightmap_exposure /= 1.1;
            }
            KeyCode::Numpad1 => {
                if let Some(ref mut dof) = dof {
                    dof.aperture_f_stops /= 1.1;
                }
            }
            KeyCode::Numpad2 => {
                if let Some(ref mut dof) = dof {
                    dof.aperture_f_stops *= 1.1;
                }
            }
            _ => {}
        }
    }
    for key in keyboard.get_just_pressed() {
        match key {
            KeyCode::Escape => {
                exit.write(AppExit::Success);
            }
            KeyCode::F1 => {
                let mut cursor = cursor.single_mut()?;
                state.show_ui = !state.show_ui;
                cursor.visible = state.show_ui;
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
            KeyCode::F5 => {
                state.show_browser_panel = !state.show_browser_panel;
            }
            KeyCode::KeyF => {
                state.cam_auto_level = !state.cam_auto_level;
            }
            KeyCode::Delete => {
                // if let Some(ent) = state.picked_object.take() {
                //     commands.entity(ent).despawn();
                // }
            }
            _ => {}
        }
    }
    Ok(())
}

type CameraQuery<'a> = (
    &'a Camera,
    &'a Transform,
    &'a mut Projection,
    &'a mut DepthOfField,
    &'a mut Tonemapping,
    &'a mut MotionBlur,
    &'a mut Bloom,
    &'a mut ColorGrading,
    &'a mut AutoExposure,
);

fn post_settings(
    mut contexts: EguiContexts,
    type_registry: ResMut<AppTypeRegistry>,
    mut state: ResMut<State>,
    mut cam: Query<CameraQuery>,
) -> BevyResult {
    if !state.show_ui {
        return Ok(());
    }
    let (_, _, proj, dof, tonemap, motion_blur, bloom, grade, auto_exposure) = cam.single_mut()?;
    egui::Window::new("Postprocessing").show(contexts.ctx_mut()?, |ui| {
        ui.collapsing("Tonemapping", |ui| {
            reflect_inspector::ui_for_value(tonemap.into_inner(), ui, &type_registry.read());
        });
        ui.collapsing("Color Grading", |ui| {
            reflect_inspector::ui_for_value(grade.into_inner(), ui, &type_registry.read());
        });
        ui.collapsing("Depth of Field", |ui| {
            reflect_inspector::ui_for_value(dof.into_inner(), ui, &type_registry.read());
        });
        ui.collapsing("Motion Blur", |ui| {
            reflect_inspector::ui_for_value(motion_blur.into_inner(), ui, &type_registry.read());
        });
        ui.collapsing("Bloom", |ui| {
            reflect_inspector::ui_for_value(bloom.into_inner(), ui, &type_registry.read());
        });
        ui.collapsing("Auto Exposure", |ui| {
            reflect_inspector::ui_for_value(auto_exposure.into_inner(), ui, &type_registry.read());
        });
        ui.collapsing("Camera", |ui| {
            reflect_inspector::ui_for_value(proj.into_inner(), ui, &type_registry.read());
            ui.add(egui::DragValue::new(&mut state.af_decay).prefix("Autofocus decay"));
        });
    });
    Ok(())
}

fn help_window(
    mut contexts: EguiContexts,
    mut state: ResMut<State>,
    mut wireframe: ResMut<WireframeConfig>,
) -> BevyResult {
    egui::Window::new("Help").show(contexts.ctx_mut()?, |ui| {
        let state = state.as_mut();
        ui.toggle_value(&mut state.cam_physics, "Enable Camera Physics");
        if state.cam_physics {
            ui.horizontal(|ui| {
                ui.label("Thrust");
                ui.add(egui::DragValue::new(&mut state.thrust_power).speed(0.1));
            });
        }
        ui.toggle_value(&mut state.show_ui, "Show UI [F1]");
        ui.toggle_value(&mut state.lightmaps, "Enable Lightmaps [F2]");
        ui.toggle_value(&mut wireframe.global, "Enable Wireframes [F3]");
        ui.toggle_value(&mut state.cam_auto_level, "Camera Auto-Level [F]");
        ui.toggle_value(&mut state.show_browser_panel, "Show Browser Panel [F5]");
        ui.horizontal(|ui| {
            ui.toggle_value(&mut state.show_nodes, "Show Nodes [F4]");
            if state.show_nodes {
                ui.add(egui::DragValue::new(&mut state.node_size).speed(0.1));
            }
        });
        ui.horizontal(|ui| {
            ui.toggle_value(&mut state.show_collision, "Show Collision");
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
                    "Mesh made of Triangles, seems to be unused",
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
    Ok(())
}

fn render_amc(
    state: Res<State>,
    mut gizmos: Gizmos,
    // mut commands: Commands,
    // ass: &mut ResMut<AssetServer>,
    // meshes: &mut ResMut<Assets<Mesh>>,
) {
    if !state.show_collision {
        return;
    }
    let amc = match state.data.as_ref() {
        Some(ParsedData::Data(Data::AMC(amc))) => amc,
        Some(ParsedData::Level(lvl)) => &lvl.collisions,
        _ => {
            return;
        }
    };
    for [p1, p2, p3] in &amc.cmsh[0].verts.data {
        let pos = Vec3::from(transform_pos([*p1, *p2, *p3]));
        gizmos.sphere(
            Isometry3d::from_translation(pos),
            10.00,
            Oklcha::sequential_dispersed(0),
        );
    }
    for [p1, p2, p3] in &amc.cmsh[1].verts.data {
        let pos = Vec3::from(transform_pos([*p1, *p2, *p3]));
        gizmos.sphere(
            Isometry3d::from_translation(pos),
            100.00,
            Oklcha::sequential_dispersed(1),
        );
    }
    for [m1, m2] in &amc.sector_col {
        let pos = &m1.verts.data;
        let tris = &m1.tris.data;
        let color = Oklcha::sequential_dispersed(2 + m1.sector as u32);
        for t in tris {
            let fp = t.idx.map(|p| Vec3::from(transform_pos(pos[p as usize])));
            gizmos.line(fp[0], fp[1], color);
            gizmos.line(fp[1], fp[2], color);
            gizmos.line(fp[2], fp[0], color);
        }
        let pos = &m2.verts.data;
        let tris = &m2.tris.data;
        let color = Oklcha::sequential_dispersed(2 + m2.sector as u32);
        for t in tris {
            let fp = t.idx.map(|p| Vec3::from(transform_pos(pos[p as usize])));
            gizmos.line(fp[0], fp[1], color);
            gizmos.line(fp[1], fp[2], color);
            gizmos.line(fp[2], fp[0], color);
        }
    }
}

fn node_type(node: Option<&NodeData>) -> &'static str {
    node.map(|data| match data {
        NodeData::Camera(_) => "Camera",
        NodeData::Dummy => "Dummy",
        NodeData::TriangleMesh => "TriangleMesh",
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
    materials: &mut ResMut<Assets<ScrapMaterial>>,
    meshes: &mut ResMut<Assets<Mesh>>,
) -> Vec<Entity> {
    let mut global_transform =
        Transform::from_matrix(Mat4::from_cols_array_2d(&node.transform_world));
    global_transform.translation = global_transform.translation.div_euclid(Vec3::splat(10.0));
    let mut local_transform =
        Transform::from_matrix(Mat4::from_cols_array_2d(&node.transform_local));
    local_transform.translation = local_transform.translation.div_euclid(Vec3::splat(10.0));
    let color = node_color(node);
    let Some(node_data) = node.content.get() else {
        return vec![
            commands
                .spawn_empty()
                .insert(local_transform)
                .insert(Name::new(node.name.string.clone()))
                .id(),
        ];
    };
    let mut res = vec![];
    match node_data {
        NodeData::D3DMesh(md3d) => {
            let mut md3d = Some(&**md3d);
            while let Some(md3d_data) = md3d {
                if let Some(verts) = md3d_data.verts.inner.as_ref().map(|v| &v.data) {
                    res.push(mesh_from_m3d(&md3d_data.tris.tris, verts));
                } else {
                    res.push(Sphere::new(0.1).mesh().build());
                }
                md3d = md3d_data.child.as_deref();
            }
        }
        _ => {
            res.push(Sphere::new(0.1).mesh().build());
        }
    };
    res.into_iter()
        .map(|mesh| {
            commands
                .spawn((
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(materials.add({
                        let mut m = StandardMaterial::from(color);
                        m.unlit = false;
                        ExtendedMaterial {
                            base: m,
                            extension: TestMaterial::default(),
                        }
                    })),
                    GlobalTransform::from(global_transform),
                    local_transform,
                ))
                .insert(Name::new(node.name.string.clone()))
                .id()
        })
        .collect()
}

fn load_sm3(
    state: &mut State,
    mut commands: Commands,
    imgs: ResMut<Assets<Image>>,
    ass: ResMut<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ScrapMaterial>>,
) {
    // TODO: generate materials, load mesh, apply material, process node tree
    let prop_re = Regex::new(r"\(\+(\w+)(?::(\w*))?\)").unwrap();
    let Some(ParsedData::Data(data)) = state.data.as_ref() else {
        return;
    };
    let mut imgs = HashMap::new();
    for dep in data.dependencies() {
        let mut segments: Vec<&str> = dep.split('/').collect();
        segments.insert(segments.len() - 1, "dds");
        if let Some((name, _)) = segments.join("/").rsplit_once(".") {
            let dds_name = name.to_owned() + ".dds";
            let dds_alpha_name = name.to_owned() + ".alpha.dds";
            let mut found = false;
            for path in [&dep, &dds_name, &dds_alpha_name] {
                if state.fs.exists(path).unwrap_or(false) {
                    if !imgs.contains_key(&dep) {
                        imgs.insert(dep.clone(), ass.add(load_texture(path, &state.fs).unwrap()));
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                warn!("Failed to load texture: {dep}");
            }
        }
    }
    let Data::SM3(sm3) = data else {
        return;
    };

    for node in &sm3.scene.nodes {
        println!(
            "{} <- {}: {}",
            node.name,
            node.parent,
            node_type(node.content.get())
        );
    }

    /* Outskirts map3d.sm3
    Ambient: 20 16 12 0 1.0 rgbai
    Backgnd: 115 115 115 0 1.0
    Extents: [-]
    */

    /* DTritus
    Ambient: 0 0 0 1
    Backgnd: 0 0 0 1
    Extents: [-34 -129 -20] [35 98 23]
    Dims: [69 227 43] 120.3
    Root: DC_Root
    ===
    Escena: Models/Chars/Dtritus/Dtritus.M3D
    _raiz_escena                                       -1   0  -1  c:(null)    f:00000001  a:0000
        DC_Root                                           0   1  -1  c:(null)    f:00010090  a:0000
        DC_Camera                                       1  93  -1  c:(null)    f:00420090  a:0000
        DC_Floor                                        2  94  -1  c:(null)    f:00420090  a:0000
        Bip Detritus MASTER                             4   2  -1  c:(null)    f:00200090  a:0000
            Bip Detritus                                  5   3  -1  c:(null)    f:00200190  a:0000

    */

    // dbg!(&sm3);

    // std::process::exit(1);

    let mut mesh_materials = HashMap::new();
    for node in &sm3.scene.nodes {
        eprintln!(
            "[{}] ({} <- {})",
            node_type(node.content.get()),
            node.name.string,
            node.parent.string
        );
        eprintln!(
            "\tFlags: {:?}, Attrs: {:?}",
            node.flags,
            node.info.get().map(|i| format!("{i}"))
        );
        if let Some(NodeData::D3DMesh(mesh)) = node.content.get() {
            let mut mesh = Some(&**mesh);
            while let Some(m) = mesh {
                let mut mesh_mat = StandardMaterial::from_color(node_color(node));
                if m.mat_index != -1 {
                    let mat = &sm3.scene.mat[m.mat_index as usize];
                    eprintln!("\t{:?}: {:?}", &m.name, &mat.name);
                    let mut mat_props: HashMap<&str, &str> = HashMap::default();
                    if let Some(mat_name) = mat.name.as_ref() {
                        mat_props = prop_re
                            .captures_iter(mat_name)
                            .filter_map(|g| {
                                let c: Vec<_> = g.iter().map(|c| c.map(|c| c.as_str())).collect();
                                let key: Option<&str> = c.get(1).and_then(|c| *c);
                                let value: &str = c.get(2).and_then(|c| *c).unwrap_or("");
                                key.map(|k| (k, value))
                            })
                            .collect();
                    };
                    match make_material(mat, &imgs, &mat_props) {
                        Ok((mat, _)) => {
                            mesh_mat = mat;
                        }
                        Err(e) => {
                            eprintln!("Failed to load material: {e}");
                        }
                    }
                    mesh_materials.insert(m.mat_index as usize, mesh_mat);
                };
                mesh = m.child.as_deref();
            }
        }
        eprintln!();
    }

    dbg!(&mesh_materials);

    let mut nodes = HashMap::new();

    // for node in &sm3.scene.nodes {
    //     let t = Affine3A::from_mat4(Mat4::from_cols_array_2d(&node.transform));
    //     let t_inv = Affine3A::from_mat4(Mat4::from_cols_array_2d(&node.transform_inv));
    //     dbg!(&node.name.string);
    //     dbg!(&t);
    //     dbg!(&t_inv);
    //     dbg!(t.inverse());
    //     dbg!(t_inv.inverse());
    //     eprintln!("========");
    // }

    // std::process::exit(1);

    for node in &sm3.scene.nodes {
        nodes.insert(node.name.string.as_str(), node);
    }
    dbg!(sm3.scene.mat.len());
    for node in &sm3.scene.nodes {
        dbg!(&node.name.string, node_type(node.content.get()));
    }
    let graph: GraphMap<&str, (), Directed> = GraphMap::from_edges(
        sm3.scene
            .nodes
            .iter()
            .map(|n| (n.parent.string.as_str(), n.name.string.as_str())),
    );
    for (src, dst, _) in graph.all_edges() {
        let node = nodes[dst];
        println!("{} -> {} [{}]", src, dst, node_type(node.content.get()));
        println!("Flags: {:?}, Attrs: {:?}", node.flags, node.info.get());
    }
    let mut ents = HashMap::new();
    for node_name in graph.nodes() {
        let Some(node) = nodes.get(node_name) else {
            println!("Entity {node_name:?} not found!");
            continue;
        };
        let ent = node_to_ent(&mut commands, node, &mut materials, &mut meshes);
        if let Some(NodeData::D3DMesh(mesh)) = node.content.get()
            && let Some(mat) = mesh_materials.get(&(mesh.mat_index as usize))
        {
            for e in &ent {
                let scrap_mat = ExtendedMaterial {
                    base: mat.clone(),
                    extension: TestMaterial::default(),
                };
                commands
                    .entity(*e)
                    .remove::<MeshMaterial3d<ScrapMaterial>>()
                    .insert(MeshMaterial3d(ass.add(scrap_mat)));
            }
        }
        ents.insert(node_name, (node, ent));
    }
    let mut q = Vec::new();
    q.push((0usize, ""));
    while let Some((depth, node_name)) = q.pop() {
        if let Some((node, ent)) = ents.get(node_name) {
            let indent = " ".repeat(depth.saturating_sub(1));
            println!(
                "{indent}{name} [{node_type}] {flags:?} {attrs:?} {ent:?}",
                name = node_name,
                node_type = node_type(node.content.get()),
                flags = node.flags,
                attrs = node.info.get()
            );
            let (scale, rot, tran) =
                Mat4::from_cols_array_2d(&node.transform_world).to_scale_rotation_translation();
            let (ax, ang) = rot.to_axis_angle();
            let ang = ang.to_degrees();
            let (rz, rx, ry) = rot.to_euler(EulerRot::ZXY);
            let (rz, rx, ry) = (rz.to_degrees(), rx.to_degrees(), ry.to_degrees());
            println!("{indent}| Scale: {scale}",);
            println!("{indent}| Pos: {tran}",);
            println!("{indent}| Rot: {ax} | {ang}",);
            println!("{indent}| r ({rz},{rx},{ry})",);
            println!("{indent}| {:?}", node.transform_world,);
        };
        for (_, dst, _) in graph.edges(node_name) {
            q.push((depth + 1, dst));
        }
    }
    // std::process::exit(1);

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
        let t1 = Transform::from_matrix(Mat4::from_cols_array_2d(&n1.transform_world));
        let t2 = Transform::from_matrix(Mat4::from_cols_array_2d(&n2.transform_world));
        let p1 = t1.translation.div_euclid(Vec3::splat(10.0));
        let p2 = t2.translation.div_euclid(Vec3::splat(10.0));
        state.edges.push((p1, p2));
    }
}

type WithMeshAndMaterial = (MeshMaterial3d<StandardMaterial>, Mesh3d);

#[repr(C, packed)]
#[derive(Debug)]
struct Vec3f(f32, f32, f32);

#[repr(C, packed)]
#[derive(Debug)]
struct Vec4f(f32, f32, f32, f32);

#[derive(Debug, Default)]
struct BrowserTreeNode {
    path: String,
    is_file: bool,
    is_level: bool,
    children: BTreeMap<String, BrowserTreeNode>,
}

impl BrowserTreeNode {
    fn insert(&mut self, full_path: &str, is_file: bool) {
        let mut current = self;
        let mut parts = full_path
            .split('/')
            .filter(|part| !part.is_empty())
            .peekable();
        while let Some(segment) = parts.next() {
            let is_leaf = parts.peek().is_none();
            let current_path = current.path.clone();
            let next_path = if current_path.is_empty() {
                segment.to_owned()
            } else {
                format!("{current_path}/{segment}")
            };

            let child = current
                .children
                .entry(segment.to_owned())
                .or_insert_with(|| BrowserTreeNode {
                    path: next_path,
                    is_file: false,
                    is_level: false,
                    children: BTreeMap::new(),
                });
            if is_leaf {
                child.is_file = is_file;
            }
            current = child;
        }
    }

    fn annotate_levels(&mut self, fs: &MultiPackFS) -> Result<()> {
        if !self.is_file && !self.path.is_empty() {
            self.is_level = fs.is_level(Some(self.path.clone()))?;
        }
        for child in self.children.values_mut() {
            child.annotate_levels(fs)?;
        }
        Ok(())
    }
}

fn is_supported_browser_file(path: &str) -> bool {
    let ext = path.rsplit('.').next().unwrap_or_default();
    ext.eq_ignore_ascii_case("sm3")
        || ext.eq_ignore_ascii_case("cm3")
        || ext.eq_ignore_ascii_case("amc")
}

fn build_browser_tree(fs: &MultiPackFS) -> Result<BrowserTreeNode> {
    let mut root = BrowserTreeNode::default();
    for entry in fs.entries()? {
        let path = entry.path.trim_matches('/');
        if path.is_empty() {
            continue;
        }
        root.insert(path, entry.is_file);
    }
    root.annotate_levels(fs)?;
    Ok(root)
}

fn browser_text_color() -> egui::Color32 {
    egui::Color32::from_rgb(188, 196, 206)
}

fn render_browser_tree_node(
    ui: &mut egui::Ui,
    name: &str,
    node: &BrowserTreeNode,
    selected_path: &mut Option<String>,
) {
    let text_color = browser_text_color();
    if node.is_file {
        if is_supported_browser_file(&node.path) {
            if ui
                .add(egui::Button::new(RichText::new(name).color(text_color)))
                .clicked()
            {
                *selected_path = Some(node.path.clone());
            }
        } else {
            ui.label(RichText::new(name).color(text_color));
        }
        return;
    }

    let tree = egui::CollapsingHeader::new(RichText::new(name).color(text_color))
        .id_salt(("browser_node", node.path.as_str()))
        .default_open(false)
        .show(ui, |ui| {
            for (child_name, child) in node.children.iter().filter(|(_, child)| !child.is_file) {
                render_browser_tree_node(ui, child_name, child, selected_path);
            }
            for (child_name, child) in node.children.iter().filter(|(_, child)| child.is_file) {
                render_browser_tree_node(ui, child_name, child, selected_path);
            }
        });

    if node.is_level {
        let should_load = tree.header_response.double_clicked();
        tree.header_response
            .on_hover_text("Double-click to load this level");
        if should_load {
            *selected_path = Some(node.path.clone());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn browser(
    mut imgs: ResMut<Assets<Image>>,
    mut commands: Commands,
    mut contexts: EguiContexts,
    mut state: ResMut<State>,
    mut ass: ResMut<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut amb: ResMut<GlobalAmbientLight>,
    mut materials: ResMut<Assets<ScrapMaterial>>,
    mut cam: Query<(
        &mut Camera,
        &mut DistanceFog,
        &mut Transform,
        &GlobalTransform,
    )>,
    mesh: Query<(Entity, &Mesh3d, &mut MeshMaterial3d<ScrapMaterial>)>,
) -> BevyResult {
    let state = state.as_mut();
    if let Some(data_path) = state.data_path.take() {
        match state.fs.parse_file(&data_path) {
            Ok(ParsedData::Level(level)) => {
                for (ent, _, _) in &mesh {
                    commands.entity(ent).despawn();
                }
                state.data = Some(ParsedData::Level(level));
                let (_, mut fog, mut t, _) = cam.single_mut()?;
                t.translation =
                    load_level(state, commands, &mut fog, imgs, ass, amb, meshes, materials).into();
            }
            Ok(ParsedData::Data(Data::SM3(sm3))) => {
                for (ent, _, _) in &mesh {
                    commands.entity(ent).despawn();
                }
                state.data = Some(ParsedData::Data(Data::SM3(sm3)));
                load_sm3(state, commands, imgs, ass, meshes, materials);
            }
            Ok(ParsedData::Data(Data::AMC(amc))) => {
                state.data = Some(ParsedData::Data(Data::AMC(amc)));
            }
            Ok(ParsedData::Data(Data::CM3(cm3))) => {
                state.data = Some(ParsedData::Data(Data::CM3(cm3)));
            }
            Ok(ParsedData::Data(_)) => {
                error!("Don't know what to do with {data_path}");
                std::process::exit(1);
            }
            Err(e) => {
                println!("Error loading {data_path}: {e:#}");
            }
        };
        return Ok(());
    }

    for (_, _, mat) in &mesh {
        if let Some(mat) = materials.get_mut(mat) {
            if state.lightmaps {
                mat.base.lightmap_exposure = state.lightmap_exposure;
            } else {
                mat.base.lightmap_exposure = 0.0;
            }
        }
    }

    if !state.show_ui {
        return Ok(());
    }

    if state.show_browser_panel
        && state.browser_tree.is_none()
        && state.browser_tree_error.is_none()
    {
        match build_browser_tree(&state.fs) {
            Ok(tree) => {
                state.browser_tree = Some(tree);
            }
            Err(err) => {
                state.browser_tree_error = Some(format!("{err:#}"));
            }
        }
    }

    let ctx = contexts.ctx_mut()?;
    let mut selected_path = None;
    let mut refresh_tree = false;
    if state.show_browser_panel {
        egui::SidePanel::left("file_browser_panel")
            .resizable(true)
            .default_width(320.0)
            .min_width(240.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("File Browser");
                    if ui.small_button("Refresh").clicked() {
                        refresh_tree = true;
                    }
                    if ui.small_button("Hide [F5]").clicked() {
                        state.show_browser_panel = false;
                    }
                });
                let text_color = browser_text_color();
                ui.label(
                    RichText::new("Double-click a level folder to load it.").color(text_color),
                );
                ui.label(RichText::new("Supported files: .sm3, .cm3, .amc").color(text_color));
                ui.separator();

                if let Some(root) = state.browser_tree.as_ref() {
                    ScrollArea::vertical().show(ui, |ui| {
                        for (child_name, child) in
                            root.children.iter().filter(|(_, child)| !child.is_file)
                        {
                            render_browser_tree_node(ui, child_name, child, &mut selected_path);
                        }
                        for (child_name, child) in
                            root.children.iter().filter(|(_, child)| child.is_file)
                        {
                            render_browser_tree_node(ui, child_name, child, &mut selected_path);
                        }
                    });
                } else if let Some(err) = state.browser_tree_error.as_ref() {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 90, 90),
                        format!("Failed to build tree: {err}"),
                    );
                } else {
                    ui.label(RichText::new("Building file tree...").color(text_color));
                }
            });
    }

    if refresh_tree {
        state.browser_tree = None;
        state.browser_tree_error = None;
        if state.show_browser_panel {
            match build_browser_tree(&state.fs) {
                Ok(tree) => {
                    state.browser_tree = Some(tree);
                }
                Err(err) => {
                    state.browser_tree_error = Some(format!("{err:#}"));
                }
            }
        }
    }

    if let Some(path) = selected_path {
        state.data_path = Some(path);
    }

    egui::Window::new("Export").show(ctx, |ui| {
        if ui.button("Export!").clicked() {
            state.export = true;
        }
    });
    Ok(())
}

#[derive(Debug, Reflect, Clone)]
struct PidCtl {
    p: f32,
    i: f32,
    d: f32,
    last_err: Option<f32>,
    last_inp: Option<f32>,
    p_val: f32,
    i_val: f32,
    d_val: f32,
}

impl PidCtl {
    fn new(p: f32, i: f32, d: f32) -> Self {
        Self {
            p,
            i,
            d,
            last_err: None,
            last_inp: None,
            p_val: 0.0,
            i_val: 0.0,
            d_val: 0.0,
        }
    }
    fn update(&mut self, inp: f32, dt: f32, target: f32) -> f32 {
        if dt == 0.0 {
            return self.p_val + self.i_val + self.d_val;
        }
        let err = target - inp;
        let d_err = self.last_err.unwrap_or(err) - err;
        // let d_err = self.last_inp.unwrap_or(inp) - inp;
        self.last_err = Some(err);
        self.last_inp = Some(inp);
        self.p_val = self.p * err;
        self.i_val += (self.p / self.i) * err * dt;
        self.d_val = (self.d * self.p) * d_err / dt;
        self.i_val = self.i_val.clamp(-100.0, 100.0);
        return self.p_val + self.i_val + self.d_val;
    }
    fn reset(&mut self) {
        self.last_err = None;
        self.last_inp = None;
        self.p_val = 0.0;
        self.i_val = 0.0;
        self.d_val = 0.0;
    }
}

#[derive(Debug, Reflect, Clone)]
struct Pid3 {
    x: PidCtl,
    y: PidCtl,
    z: PidCtl,
}

impl Pid3 {
    fn new(p: f32, i: f32, d: f32) -> Self {
        Self {
            x: PidCtl::new(p, i, d),
            y: PidCtl::new(p, i, d),
            z: PidCtl::new(p, i, d),
        }
    }
    fn update(&mut self, inp: Vec3, dt: f32, setpoint: Vec3) -> Vec3 {
        let x = self.x.update(inp.x, dt, setpoint.x);
        let y = self.y.update(inp.y, dt, setpoint.y);
        let z = self.z.update(inp.z, dt, setpoint.z);
        Vec3::new(x, y, z)
    }
    fn set(&mut self, p: f32, i: f32, d: f32) {
        self.reset();
        for c in [&mut self.x, &mut self.y, &mut self.z] {
            c.p = p;
            c.i = i;
            c.d = d;
        }
    }
    fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
        self.z.reset();
    }
}

#[derive(Debug, Component, Reflect, Clone)]
struct DroneCam {
    pos: Vec3,
    sensitivity: f32,
    velocity: Vec3,
    ang_vel: Vec3,
    boost: f32,
    pid_thrust: Pid3,
    pid_ang: Pid3,
    ax_state: HashMap<GamepadAxis, f32>,
    btn_state: HashMap<GamepadButton, f32>,
}

impl Default for DroneCam {
    fn default() -> Self {
        Self {
            pos: Vec3::new(-2.0, 5.0, 5.0),
            sensitivity: 0.25,
            velocity: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            pid_thrust: Pid3::new(1.0, 1.0, 1.0),
            pid_ang: Pid3::new(1.0, 1.0, 1.0),
            btn_state: HashMap::new(),
            ax_state: HashMap::new(),
            boost: 1.0,
        }
    }
}

use leafwing_input_manager::prelude::*;

impl DroneCam {
    fn update(
        mut query: Query<(
            &mut Transform,
            &mut DroneCam,
            &mut MotionBlur,
            &ActionState<DroneAction>,
        )>,
        time: Res<Time>,
        state: Res<State>,
        mut ray_cast: MeshRayCast,
        mut contexts: EguiContexts,
    ) -> BevyResult {
        let dt = time.delta_secs();
        /*
        #[actionlike(Axis)] Throttle,
        #[actionlike(Axis)] Yaw,
        #[actionlike(Axis)] Pitch,
        #[actionlike(Axis)] Roll,
        #[actionlike(Button)] Boost,
        #[actionlike(Button)] TurnBoost,
        */
        const RESTITUTION: f32 = 0.8;
        const COLLISION_MARGIN: f32 = 2.0;
        let mut turn_rate: f32 = 10.0;
        let global_drag: f32 = 2.0;
        let ang_drag_mult: f32 = 1.0;
        let (mut tran, mut drone, mut blur, input_state) = query.single_mut()?;
        let mut throttle = input_state.clamped_value(&DroneAction::Throttle);
        throttle = throttle.powf(2.0).copysign(throttle);
        let mut reverse = input_state.button_value(&DroneAction::Reverse);
        reverse = reverse.powf(2.0).copysign(reverse);
        let mut strafe = input_state.clamped_value(&DroneAction::Strafe);
        strafe = strafe.powf(2.0).copysign(strafe);
        let mut yaw = input_state.clamped_value(&DroneAction::Yaw);
        yaw = yaw.powf(2.0).copysign(yaw);
        let mut pitch = input_state.clamped_value(&DroneAction::Pitch);
        pitch = pitch.powf(2.0).copysign(pitch);
        let mut roll_in = input_state.clamped_value(&DroneAction::Roll);
        roll_in = roll_in.powf(2.0).copysign(roll_in);
        let turn_boost = input_state.button_value(&DroneAction::TurnBoost);
        let boost = input_state.button_value(&DroneAction::Boost);
        blur.shutter_angle = 60.0 / 24.0 * 0.2;
        let mut brake = 0.0;
        if throttle < 0.0 {
            brake = -throttle;
            throttle = 0.0;
        }
        let dt = time.delta_secs();
        turn_rate *= 1.0 + turn_boost * 2.0;
        if boost > 0.0 {
            let boost_mult = 1.0 + 2.0 * drone.boost;
            throttle *= boost_mult;
            strafe *= boost_mult;
            blur.shutter_angle = 60.0 / 24.0 * 0.5;
        }

        {
            // Rotation
            drone.ang_vel -= Vec3::new(yaw, -pitch, roll_in) * dt * turn_rate;
            let yaw = drone.ang_vel.x * dt;
            let pitch = drone.ang_vel.y * dt;
            let roll = drone.ang_vel.z * dt;
            let local_up = *tran.up();
            let local_right = *tran.right();
            let local_forward = *tran.forward();
            let delta_rot = Quat::from_axis_angle(local_up, yaw)
                * Quat::from_axis_angle(local_right, pitch)
                * Quat::from_axis_angle(local_forward, roll);
            tran.rotation = (delta_rot * tran.rotation).normalize();
            let ang_drag = drone.ang_vel * dt * (global_drag + brake * 5.0) * ang_drag_mult;
            drone.ang_vel -= ang_drag;

            // Auto-level only when the user is not actively rolling.
            // This keeps keyboard flight horizon-stable while still allowing full flips.
            if state.cam_auto_level && roll_in.abs() < 0.05 {
                let level_lerp = 1.0 - (-0.8 * dt).exp();
                let forward = *tran.forward();
                let level_up = Vec3::Y - forward * Vec3::Y.dot(forward);
                if level_up.length_squared() > f32::EPSILON {
                    let current_up = *tran.up();
                    let target_up = level_up.normalize();
                    let mut roll_error = current_up.angle_between(target_up);
                    let sign = forward.dot(current_up.cross(target_up)).signum();
                    roll_error *= sign;
                    let correction = Quat::from_axis_angle(forward, roll_error * level_lerp);
                    tran.rotation = correction * tran.rotation;
                }
                drone.ang_vel.z = drone.ang_vel.z.lerp(0.0, level_lerp);
            }
        }

        {
            // Translation
            let acc = (tran.forward() * (throttle - reverse) + tran.right() * strafe)
                * state.thrust_power
                * dt;
            drone.velocity += acc;
            let mv_dir = Dir3::new(drone.velocity);
            if mv_dir.is_ok() && state.cam_physics {
                let ray = Ray3d::new(tran.translation, mv_dir.unwrap());
                if let Some((_, hit)) = ray_cast
                    .cast_ray(
                        ray,
                        &MeshRayCastSettings {
                            visibility: RayCastVisibility::Visible,
                            ..default()
                        },
                    )
                    .first()
                    && (hit.distance - (drone.velocity.length() * dt)) < COLLISION_MARGIN
                {
                    drone.velocity = drone.velocity.reflect(hit.normal) * RESTITUTION;
                };
            }
            if !state.cam_physics {
                drone.velocity *= 100.0
            }
            tran.translation += drone.velocity * dt;
            let drag = drone.velocity * dt * (global_drag + brake * 5.0);
            drone.velocity -= drag;
            if !state.cam_physics {
                drone.velocity = Vec3::ZERO;
            }
        }

        tran.translation += drone.velocity * dt;

        egui::Window::new("Drone").show(contexts.ctx_mut()?, |ui| {
            ui.horizontal(|ui| {
                ui.label("Thrust");
                let (mut p, mut i, mut d) = (
                    drone.pid_thrust.x.p,
                    drone.pid_thrust.x.i,
                    drone.pid_thrust.x.d,
                );
                let p_changed = ui.add(egui::DragValue::new(&mut p).speed(0.1)).changed();
                let i_changed = ui.add(egui::DragValue::new(&mut i).speed(0.1)).changed();
                let d_changed = ui.add(egui::DragValue::new(&mut d).speed(0.1)).changed();
                if p_changed || i_changed || d_changed {
                    drone.pid_thrust.set(p, i, d);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Angle");
                let (mut p, mut i, mut d) =
                    (drone.pid_ang.x.p, drone.pid_ang.x.i, drone.pid_ang.x.d);
                ui.add(egui::DragValue::new(&mut p).speed(0.1));
                ui.add(egui::DragValue::new(&mut i).speed(0.1));
                ui.add(egui::DragValue::new(&mut d).speed(0.1));
                drone.pid_ang.set(p, i, d);
            });
            let (yaw, pitch, roll) = tran.rotation.to_euler(EulerRot::YXZ);
            ui.label(format!("Boost: {:.2}%", drone.boost * 100.0));
            ui.label(format!("Position: {}", tran.translation));
            ui.label(format!(
                "Velocity: {} ({})",
                drone.velocity.length(),
                drone.velocity
            ));
            ui.label(format!("Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}"));
            ui.label(format!("Angular Velocity: {}", drone.ang_vel));
            for (k, v) in input_state.all_action_data() {
                ui.label(format!("{k:?}: {v:?}"));
            }
        });
        Ok(())
    }
}

#[derive(Bundle)]
struct CameraBundle {
    camera: Camera3d,
    inputs: InputMap<DroneAction>,
    hdr: Hdr,
    tonemapping: Tonemapping,
    color_grading: ColorGrading,
    bloom: Bloom,
    motion_blur: MotionBlur,
    depth_of_field: DepthOfField,
    projection: Projection,
    contrast_adaptive_sharpening: ContrastAdaptiveSharpening,
    msaa: Msaa,
    exposure: AutoExposure,
    ctl: DroneCam,
    fog: DistanceFog,
}

#[derive(Actionlike, PartialEq, Eq, Hash, Clone, Copy, Debug, Reflect)]
enum DroneAction {
    #[actionlike(Axis)]
    Throttle,
    #[actionlike(Button)]
    Reverse,
    #[actionlike(Axis)]
    Strafe,
    #[actionlike(Axis)]
    Yaw,
    #[actionlike(Axis)]
    Pitch,
    #[actionlike(Axis)]
    Roll,
    #[actionlike(Button)]
    Boost,
    #[actionlike(Button)]
    TurnBoost,
}

/* TODO:
X Left: Yaw
Y Left: Throttle
X Right: Roll
Y Right: Pitch
*/

// TODO: Desired normal and angular velocity -> PID controller -> Thrust

impl CameraBundle {
    fn new() -> Self {
        Self {
            camera: Camera3d::default(),
            inputs: InputMap::default()
                .with_axis(
                    DroneAction::Throttle,
                    VirtualAxis::new(KeyCode::KeyS, KeyCode::KeyW),
                )
                .with(DroneAction::Reverse, KeyCode::KeyS)
                .with_axis(
                    DroneAction::Strafe,
                    VirtualAxis::new(KeyCode::KeyA, KeyCode::KeyD),
                )
                .with_axis(
                    DroneAction::Yaw,
                    VirtualAxis::new(KeyCode::ArrowLeft, KeyCode::ArrowRight),
                )
                .with_axis(
                    DroneAction::Pitch,
                    VirtualAxis::new(KeyCode::ArrowDown, KeyCode::ArrowUp),
                )
                .with_axis(
                    DroneAction::Roll,
                    VirtualAxis::new(KeyCode::KeyQ, KeyCode::KeyE).inverted(),
                )
                .with_axis(
                    DroneAction::Throttle,
                    VirtualAxis::new(GamepadButton::LeftTrigger2, GamepadButton::RightTrigger2), // GamepadStick::LEFT.inverted_y().with_circle_deadzone(0.01).y,
                )
                .with_axis(
                    DroneAction::Strafe,
                    GamepadStick::LEFT.with_circle_deadzone(0.01).x,
                )
                .with_axis(
                    DroneAction::Pitch,
                    GamepadStick::RIGHT.with_circle_deadzone(0.01).y,
                )
                .with_axis(
                    DroneAction::Yaw,
                    GamepadStick::RIGHT.with_circle_deadzone(0.01).x,
                )
                .with_axis(
                    DroneAction::Pitch,
                    MouseMoveAxis::Y.sensitivity(0.5).inverted(),
                )
                .with_axis(DroneAction::Yaw, MouseMoveAxis::X.sensitivity(0.5))
                .with(DroneAction::Boost, GamepadButton::RightTrigger)
                .with(DroneAction::Boost, MouseButton::Right)
                .with(DroneAction::TurnBoost, GamepadButton::LeftTrigger)
                .with(DroneAction::TurnBoost, KeyCode::ShiftLeft)
                .with(DroneAction::TurnBoost, KeyCode::ShiftRight),
            hdr: Hdr,
            tonemapping: Tonemapping::AcesFitted,
            color_grading: ColorGrading {
                global: ColorGradingGlobal {
                    exposure: 1.0,
                    ..default()
                },
                ..default()
            },
            bloom: Bloom {
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
            motion_blur: MotionBlur {
                samples: 32,
                shutter_angle: 60.0 / 24.0 * 0.2,
            },
            depth_of_field: DepthOfField {
                mode: DepthOfFieldMode::Bokeh,
                sensor_height: 18.66,
                aperture_f_stops: 1.0,
                ..default()
            },
            projection: Projection::Perspective(PerspectiveProjection {
                fov: std::f32::consts::FRAC_PI_2, // 90°
                near: 0.01,
                ..default()
            }),
            exposure: AutoExposure { ..default() },
            contrast_adaptive_sharpening: ContrastAdaptiveSharpening::default(),
            msaa: Msaa::Sample8,
            ctl: DroneCam::default(),
            fog: DistanceFog {
                falloff: FogFalloff::Exponential { density: 0.0 },
                color: LinearRgba::WHITE.with_alpha(0.0).into(),
                ..Default::default()
            },
        }
    }
}

fn setup(
    mut commands: Commands,
    assets: Res<AssetServer>,
    mut test_ass: ResMut<Assets<TestAsset>>,
) {
    // let meta = asset_loader::TestLoader.default_meta();
    // let serialized_meta = meta.serialize();
    // println!("{}",unsafe {std::str::from_utf8_unchecked(&serialized_meta)});
    // let ass = dbg!(assets.load::<asset_loader::TestAsset>("packed://Levels/Outskirts/Map/Map3d.emi"));
    // let test = test_ass.get(&ass);
    // dbg!(test);
    // std::process::exit(0);
    commands.spawn(CameraBundle::new()).with_children(|parent| {
        // parent.spawn(PointLight { ..default() });
    });
}
