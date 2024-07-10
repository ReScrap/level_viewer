#![feature(adt_const_params)]
use std::{
    borrow::BorrowMut,
    collections::{BTreeMap, HashMap},
    ops::Deref,
    path::PathBuf,
};

use anyhow::{anyhow, bail, Result};
use bevy::{
    app::AppExit,
    core_pipeline::{
        auto_exposure::{AutoExposurePlugin, AutoExposureSettings},
        bloom::{BloomPrefilterSettings, BloomSettings},
        dof::{DepthOfFieldMode, DepthOfFieldSettings},
        tonemapping::{DebandDither, Tonemapping},
    },
    diagnostic::{
        EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin,
        SystemInformationDiagnosticsPlugin,
    },
    input::{keyboard::KeyboardInput, mouse::MouseButtonInput},
    math::{vec2, Affine3A},
    pbr::{ExtendedMaterial, Lightmap, LightmapPlugin, ScreenSpaceAmbientOcclusionBundle},
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
        render_resource::Face,
        texture::{
            CompressedImageFormats, Image, ImageAddressMode, ImageFilterMode, ImageSampler,
            ImageSamplerDescriptor, ImageType,
        },
    },
    window::{PresentMode, PrimaryWindow, WindowMode},
};
use bevy_atmosphere::plugin::{AtmosphereCamera, AtmospherePlugin};
use bevy_editor_cam::{prelude::EditorCam, DefaultEditorCamPlugins};
use bevy_egui::{
    egui::{self, RichText, ScrollArea},
    EguiContexts, EguiPlugin,
};
use bevy_inspector_egui::quick::{FilterQueryInspectorPlugin, WorldInspectorPlugin};
use bevy_mod_raycast::{
    cursor::CursorRay,
    immediate::{Raycast, RaycastSettings},
    prelude::*,
};
use iyes_perf_ui::{entries::PerfUiCompleteBundle, PerfUiPlugin};
use parser::{multi_pack_fs::MultiPackFS, Level};
use regex::Regex;
use smooth_bevy_cameras::{
    controllers::unreal::{UnrealCameraBundle, UnrealCameraController, UnrealCameraPlugin},
    LookTransformPlugin,
};

use crate::{
    materials::Hologram,
    parser::{ParsedData, TexCoords},
};
mod find_scrap;
mod materials;
mod packed_vfs;
mod parser;

#[derive(Component, Deref)]
struct MaterialName(String);

#[derive(Component, Deref)]
struct MapNames(Vec<(String, String)>);

#[derive(Resource)]
struct State {
    fs: MultiPackFS,
    level: Option<String>,
    level_data: Option<Level>,
    picked_object: Option<Entity>,
    show_ui: bool,
    lightmaps: bool,
    lightmap_exposure: f32,
    textures: HashMap<String, Handle<Image>>,
}

// TODO: move outside of Bevy app
fn load_data() -> Result<MultiPackFS> {
    let data_regex = Regex::new(r"[Dd]ata\d*\.packed")?;
    let packed_files: Vec<PathBuf> = find_scrap::get_path()
        .ok_or_else(|| anyhow!("Scrapland path not found!"))
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
        level: std::env::args().nth(1),
        picked_object: None,
        level_data: None,
        show_ui: true,
        lightmaps: true,
        textures: default(),
        lightmap_exposure: 50000.0,
    };
    App::new()
        .insert_resource(Msaa::Sample8)
        .insert_resource(AmbientLight {
            color: Color::BLACK,
            brightness: 0.0,
        })
        .insert_resource(state)
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        mode: WindowMode::BorderlessFullscreen,
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
            // AutoExposurePlugin,
            EguiPlugin,
            WorldInspectorPlugin::new().run_if(|state: Res<State>| state.show_ui),
            SystemInformationDiagnosticsPlugin,
            // EntityCountDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin,
            LogDiagnosticsPlugin::default(),
            PerfUiPlugin,
            AtmospherePlugin,
            LookTransformPlugin,
            CursorRayPlugin,
            UnrealCameraPlugin::new(false),
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                ui_input_toggle,
                browser,
                autofocus,
                entity_picker,
                keyboard_handler,
            ),
        )
        .init_gizmo_group::<DefaultGizmoConfigGroup>()
        .run();
    Ok(())
}

fn load_level(
    state: &mut State,
    mut commands: Commands,
    images: &Assets<Image>,
    ass: &mut AssetServer,
    meshes: &mut Assets<Mesh>,
    material_res: ResMut<Assets<StandardMaterial>>,
) -> [f32; 3] {
    let mut centroid = [0.0, 0.0, 0.0];
    let mut total_count: usize = 0;
    let Some(level) = state.level_data.as_ref() else {
        return centroid;
    };
    commands
        .spawn(SpatialBundle::default())
        .with_children(|parent| {
            let prop_re = Regex::new(r"\(\+(\w+)(?::(\w*))?\)").unwrap();
            let mut textures = HashMap::new();
            for (key, path) in &level.dependencies {
                let mut data = vec![];
                info!("Loading texture {path}");
                state
                    .fs
                    .open_file(path)
                    .expect("Failed to open file")
                    .read_to_end(&mut data)
                    .expect("Failed to read file");
                let ext = path.split(".").last().unwrap_or_default();
                let img = Image::from_buffer(
                    #[cfg(debug_assertions)]
                    key.clone(),
                    &data,
                    ImageType::Extension(ext),
                    CompressedImageFormats::all(),
                    true,
                    ImageSampler::Descriptor(ImageSamplerDescriptor {
                        label: Some(key.clone()),
                        address_mode_u: ImageAddressMode::Repeat,
                        address_mode_v: ImageAddressMode::Repeat,
                        address_mode_w: ImageAddressMode::Repeat,
                        mag_filter: ImageFilterMode::Linear,
                        min_filter: ImageFilterMode::Linear,
                        mipmap_filter: ImageFilterMode::Linear,
                        ..default()
                    }),
                    RenderAssetUsages::all(),
                )
                .expect("Failed to load iamge");
                // rhexdump::rhexdump!(&img.data[..0x100]);
                // let n_dim = match img.texture_descriptor.dimension {
                //     bevy::render::render_resource::TextureDimension::D1 => 1,
                //     bevy::render::render_resource::TextureDimension::D2 => 2,
                //     bevy::render::render_resource::TextureDimension::D3 => 3,
                // };
                // dbg!(img.texture_descriptor.size);
                // dbg!(n_dim);
                // dbg!(img.texture_descriptor.format.components());
                textures.insert(key.clone(), ass.add(img));
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
                    ("Base", &mat.maps.diffuse),
                    ("Metallic", &mat.maps.metallic),
                    ("Environment", &mat.maps.reflection),
                    ("Bump", &mat.maps.bump),
                    ("Glow", &mat.maps.glow),
                ];
                let mut map_names = vec![];
                for (name, map) in maps {
                    let map: String = map
                        .as_ref()
                        .map(|m: &parser::MAP| m.texture.string.clone())
                        .unwrap_or_default();
                    map_names.push((name.to_owned(), map));
                }
                let base_color_texture = mat
                    .maps
                    .diffuse
                    .as_ref()
                    .and_then(|m| textures.get(&m.texture.string))
                    .cloned();
                let emissive_texture = mat
                    .maps
                    .glow
                    .as_ref()
                    .and_then(|m| textures.get(&m.texture.string))
                    .cloned();

                // B=metallic
                // G=roughness
                let metallic_roughness_texture = mat
                    .maps
                    .metallic
                    .as_ref()
                    .and_then(|m| textures.get(&m.texture.string))
                    .cloned();
                let normal_map_texture = mat
                    .maps
                    .bump
                    .as_ref()
                    .and_then(|m| textures.get(&m.texture.string))
                    .cloned();

                let mut mat = StandardMaterial {
                    base_color: Color::WHITE,
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
                    Some(_) => LinearRgba::rgb(100.0, 100.0, 100.0),
                    None => Color::BLACK.to_linear(),
                };
                mat.alpha_mode = match mat_props.get("transp").copied() {
                    Some("premult" | "filter") => AlphaMode::Multiply,
                    _ => AlphaMode::Opaque,
                };

                if mat_props.contains_key("zbias") {
                    if mat.alpha_mode == AlphaMode::Opaque {
                        mat.alpha_mode = AlphaMode::Blend;
                    }
                    mat.depth_bias = 10.0;
                };
                if mat.base_color_texture.is_none() && mat.alpha_mode == AlphaMode::Opaque {
                    mat.alpha_mode = AlphaMode::Multiply
                };
                // let mat = match mat_props.get("shader").map(|&s| s).unwrap_or("") {
                //     // "hologram" => ass.add(ExtendedMaterial {
                //     //     base: mat,
                //     //     extension: Hologram {},
                //     // }),
                //     _ => ass.add(mat)
                // };
                materials.insert(key, (mat_name, map_names, ass.add(mat)));
            }
            let mut lmaps = HashMap::new();
            for lm in level.emi.maps.iter() {
                if let Some((key_1, _, key_2)) = &lm.data {
                    lmaps.insert(
                        lm.key,
                        (
                            textures.get(&key_1.string).cloned(),
                            textures.get(&key_2.string).cloned(),
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
                let (mat_name, map_names, mat) = &materials[&data.mat_key];
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
                    let verts = &verts.data;
                    let pos: Vec<_> = verts
                        .iter()
                        .map(|v| {
                            let [x, y, z] = v.xyz;
                            [x / -1000.0, y / 1000.0, z / 1000.0]
                        })
                        .inspect(|[x, y, z]| {
                            centroid[0] += x;
                            centroid[1] += y;
                            centroid[2] += z;
                            total_count += 1;
                        })
                        .collect();
                    let needs_normals = verts.iter().any(|v| v.normal.is_none())
                        || mat_inst
                            .as_ref()
                            .and_then(|m| m.normal_map_texture.as_ref())
                            .is_some();
                    let normal: Vec<_> =
                        verts.iter().map(|v| v.normal.unwrap_or_default()).collect();
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
                            [1.0, 1.0, 1.0, 0.0]
                        })
                        .collect();

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
                    let idx = faces
                        .iter()
                        .copied()
                        .flat_map(|mut face| {
                            face.swap(0, 1);
                            face
                        })
                        .collect::<Vec<_>>();
                    // for face in idx.chunks_mut(3) {
                    //     face.swap(0,1);
                    //     // let face = face.iter().map(|&v| pos.get_mut(v as usize).unwrap()).collect::<Vec<_>>();

                    // }
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
                    if needs_normals {
                        // mesh.duplicate_vertices();
                        // mesh.compute_flat_normals();
                    }
                    // mesh.generate_tangents()
                    //     .expect("Failed to generate tangents");
                    // TODO: custom materials based on shaders
                    let mut pbr = parent.spawn((
                        PbrBundle {
                            mesh: meshes.add(mesh),
                            material: mat.clone(),
                            ..Default::default()
                        },
                        Name::new(name.to_owned()),
                        MaterialName(mat_name.to_owned()),
                        MapNames(map_names.clone()),
                    ));
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
        });
    centroid[0] /= total_count as f32;
    centroid[1] /= total_count as f32;
    centroid[2] /= total_count as f32;
    centroid
}

fn ui_input_toggle(mut contexts: EguiContexts, mut cam_ctrl: Query<&mut UnrealCameraController>) {
    let egui_ctx = contexts.ctx_mut();
    cam_ctrl.single_mut().enabled =
        !(egui_ctx.wants_keyboard_input() || egui_ctx.wants_pointer_input());
}

fn autofocus(
    mut raycast: Raycast,
    mut cam: Query<(&Camera, &Transform, &mut DepthOfFieldSettings)>,
    cursor_ray: Res<CursorRay>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    let cam = cam.get_single_mut().unwrap();
    let (_, tr, mut dof) = cam;

    let pick_ray = if keyboard.pressed(KeyCode::KeyF) {
        **cursor_ray
    } else {
        Some(Ray3d::new(tr.translation, *tr.forward()))
    };
    let Some((_, hit)) = pick_ray.and_then(|cursor_ray| {
        raycast
            .cast_ray(
                cursor_ray,
                &RaycastSettings {
                    visibility: RaycastVisibility::Ignore,
                    ..default()
                },
            )
            .first()
    }) else {
        return;
    };
    dof.focal_distance = hit.distance();
}

#[allow(clippy::too_many_arguments)]
fn entity_picker(
    mouse: Res<ButtonInput<MouseButton>>,
    cursor_ray: Res<CursorRay>,
    images: Res<Assets<Image>>,
    mut gizmos: Gizmos,
    mut state: ResMut<State>,
    mut raycast: Raycast,
    mut contexts: EguiContexts,
    cam: Query<(&Camera, &Transform, &GlobalTransform)>,
    ent_info_query: Query<(Entity, &Name, &MaterialName, &MapNames)>,
) {
    let pick_ray = if mouse.pressed(MouseButton::Left) {
        **cursor_ray
    } else {
        let (_, tr, _) = cam.get_single().unwrap();
        Some(Ray3d::new(tr.translation, *tr.forward()))
    };
    let egui_waints_input =
        contexts.ctx_mut().wants_keyboard_input() || contexts.ctx_mut().wants_pointer_input();
    let picked_entity = pick_ray.and_then(|cursor_ray| {
        raycast
            .cast_ray(cursor_ray, &RaycastSettings::default())
            .first()
    });
    if let Some((_, hit)) = picked_entity {
        gizmos.sphere(hit.position(), Quat::IDENTITY, 0.1, LinearRgba::RED);
    }
    if !egui_waints_input && mouse.just_pressed(MouseButton::Left) {
        state.picked_object = picked_entity.map(|(e, _)| *e);
    }
    if state.show_ui {
        egui::Window::new("Inspector").show(contexts.ctx_mut(), |ui| {
            let Some(ent) = state.picked_object else {
                return;
            };
            let Ok((_, name, mat_name, map_names)) = ent_info_query.get(ent) else {
                return;
            };
            let name = name.as_str();
            ui.heading(format!("{name} ({ent:?})"));
            let mat_name = mat_name.as_str();
            ui.label(format!("Material: {mat_name}"));
            for (kind, tex_name) in map_names.iter() {
                // if let Some(tex_handle) = tex_handle {
                //     ui.collapsing(format!("{kind}: {tex_name}"),|ui| {
                //         let img = contexts.add_image(tex_handle.clone_weak());
                //         let tex = images.get(tex_handle).unwrap();
                //         let width = tex.texture_descriptor.size.width;
                //         let height = tex.texture_descriptor.size.height;
                //         ui.add();
                //     });
                // } else {
                ui.label(format!("{kind}: {tex_name}"));
                // };
            }
        });
    }
}

fn keyboard_handler(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>,
    mut state: ResMut<State>,
) {
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
            _ => {}
        }
    }
}

type WithMeshAndMaterial = (With<Handle<Mesh>>, With<Handle<StandardMaterial>>);

// TODO: cleanup, integrate Clap
#[allow(clippy::too_many_arguments)]
fn browser(
    imgs: Res<Assets<Image>>,
    mut commands: Commands,
    mut contexts: EguiContexts,
    mut state: ResMut<State>,
    mut ass: ResMut<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cam: Query<(&mut Camera, &mut Transform, &GlobalTransform)>,
    mesh: Query<Entity, WithMeshAndMaterial>,
    mut mat: Query<&mut Handle<StandardMaterial>>,
) {
    let state = state.as_mut();
    if let Some(level_path) = state.level.take() {
        match state.fs.parse_file(&level_path) {
            Ok(ParsedData::Level(level)) => {
                state.level_data = Some(level);
                let c = load_level(state, commands, &imgs, &mut ass, &mut meshes, materials);
                let mut t = cam.get_single_mut().unwrap().1;
                t.translation.x = c[0];
                t.translation.y = c[1];
                t.translation.z = c[2];
            }
            Err(e) => {
                print!("Error loading {level_path}: {e}");
            }
            _ => (),
        };
        return;
    }

    for mesh in &mesh {
        if let Ok(handle) = mat.get(mesh) {
            if let Some(mat) = materials.get_mut(handle) {
                if state.lightmaps {
                    mat.lightmap_exposure = state.lightmap_exposure;
                } else {
                    mat.lightmap_exposure = 0.0;
                }
            }
        }
    }

    if !state.show_ui {
        return;
    }
    let cam = cam.get_single_mut().unwrap();
    egui::Window::new("Browser").show(contexts.ctx_mut(), |ui| {
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
                                state.level_data = Some(level);
                                for item in &mesh {
                                    commands.entity(item).despawn_recursive();
                                }
                                let c = load_level(
                                    state,
                                    commands,
                                    &imgs,
                                    &mut ass,
                                    &mut meshes,
                                    materials,
                                );
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
    commands.spawn(PerfUiCompleteBundle::default());
    commands
        .spawn((
            Camera3dBundle {
                camera: Camera {
                    hdr: true,
                    ..default()
                },
                tonemapping: Tonemapping::AgX,
                ..default()
            },
            // AutoExposureSettings { ..default() },
            BloomSettings {
                prefilter_settings: BloomPrefilterSettings {
                    threshold: 0.0,
                    threshold_softness: 0.0,
                },
                ..BloomSettings::NATURAL
            },
            DepthOfFieldSettings {
                focal_distance: 1.0, // Raycast from camera/cursor
                mode: DepthOfFieldMode::Bokeh,
                ..default()
            },
            AtmosphereCamera::default(),
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
