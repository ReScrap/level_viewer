use std::{
    collections::{BTreeMap, HashMap, HashSet},
    default,
    io::{BufWriter, Write},
};

use color_eyre::eyre::{Context, Result, anyhow, bail};
use bevy::{
    log::{error, info, tracing_subscriber::registry, warn},
    mesh::{MeshVertexAttribute, MeshVertexAttributeId, PrimitiveTopology, VertexFormat},
    pbr::StandardMaterial,
    prelude::{Result as BevyResult, *},
    reflect::{TypeRegistration, TypeRegistry, serde::ReflectSerializer},
};
use image::{GenericImageView, Rgba};
use rhexdump::rhexdump;
use zip::unstable::write;

use crate::{
    LightmapHandles, LightmapNames, MapNames, MapTex, MaterialName, ScrapMat, ScrapMaterial, State, parser::{NodeData, ParsedData}
};

fn compress_image(img: &Image, quantize: bool, optimize: bool) -> BevyResult<Vec<u8>> {
    use bevy::render::render_resource::TextureFormat;
    let mut liq = imagequant::new();
    liq.set_speed(1)?;
    liq.set_quality(20, 80)?;
    liq.set_log_callback(|_, msg| {
        info!(target:"imagequant","{msg}");
    });
    let Some(data) = img.data.as_ref() else {
        return Err("No texture data!".to_string().into());
    };
    let width = img.width();
    let height = img.height();
    let img = if img.texture_descriptor.format.is_bcn() {
        let format = match img.texture_descriptor.format {
            TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => texpresso::Format::Bc1,
            TextureFormat::Bc2RgbaUnorm | TextureFormat::Bc2RgbaUnormSrgb => texpresso::Format::Bc2,
            TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb => texpresso::Format::Bc3,
            TextureFormat::Bc4RUnorm | TextureFormat::Bc4RSnorm => texpresso::Format::Bc4,
            TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm => texpresso::Format::Bc5,
            other => {
                warn!("Unsupported texture format: {other:?}");
                return Ok(vec![]);
            }
        };
        let w = width as usize;
        let h = height as usize;
        let mut buffer = vec![0u8; w * h * 4];
        format.decompress(data, w, h, &mut buffer);
        buffer
    } else if let Some(img) = img.convert(TextureFormat::Rgba8UnormSrgb) {
        img.data.unwrap()
    } else {
        warn!("Invalid format: {:?}", img.texture_descriptor);
        return Ok(vec![]);
    };
    assert_eq!(img.len(), (width * height * 4) as usize);
    let mut buffer = std::io::Cursor::new(Vec::new());
    if quantize {
        let pixels = img
            .chunks(4)
            .map(|w| imagequant::RGBA::from([w[0], w[1], w[2], w[3]]))
            .collect::<Vec<_>>();
        let mut img = liq.new_image(pixels, width as usize, height as usize, 0.0)?;
        let mut res = liq.quantize(&mut img)?;
        let (palette, pixels) = res.remapped(&mut img)?;
        let mut quantized_img = image::ImageBuffer::new(width, height);
        for (x, y, pixel) in quantized_img.enumerate_pixels_mut() {
            let idx = (y * width + x) as usize;
            let p = &palette[pixels[idx] as usize];
            *pixel = image::Rgba([p.r, p.g, p.b, p.a]);
        }
        quantized_img.write_to(&mut buffer, image::ImageFormat::Png)?;
    } else {
        let img = image::ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, img).unwrap();
        img.write_to(&mut buffer, image::ImageFormat::Png)?;
    }
    if !optimize {
        return Ok(buffer.into_inner());
    }
    let png_bytes = oxipng::optimize_from_memory(
        &buffer.into_inner(),
        &oxipng::Options {
            deflater: oxipng::Deflater::Zopfli(oxipng::ZopfliOptions {
                iteration_count: 10.try_into().unwrap(),
                ..Default::default()
            }),
            optimize_alpha: true,
            ..oxipng::Options::default()
        },
    )
    .map_err(|e| e.to_string())?;
    return Ok(png_bytes);
}

fn dump_mesh(mesh: &Mesh) -> BTreeMap<String, Vec<u8>> {
    let mut ret = BTreeMap::new();
    if let Some(idx) = mesh.indices() {
        let (key, data) = match idx {
            bevy::mesh::Indices::U16(items) => (
                "IDX16",
                items
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            bevy::mesh::Indices::U32(items) => (
                "IDX32",
                items
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
        };
        ret.insert(key.to_owned(), data);
    }
    for (attr, data) in mesh.attributes() {
        let data = data.get_bytes();
        let key = format!("{}.{:?}", attr.name, attr.format);
        ret.insert(key.to_owned(), data.to_owned());
    }
    return ret;
}

fn dump_mat_props(mat: &StandardMaterial) -> BTreeMap<String, serde_json::Value> {
    let mut mat_props: BTreeMap<String, serde_json::Value> = BTreeMap::default();
    mat_props.insert("base_color".to_string(), {
        serde_json::to_value(mat.base_color.to_linear()).unwrap()
    });
    mat_props.insert("emissive".to_string(), {
        serde_json::to_value(mat.emissive).unwrap()
    });
    mat_props.insert("depth_bias".to_string(), {
        serde_json::to_value(mat.depth_bias).unwrap()
    });
    mat_props.insert("unlit".to_string(), {
        serde_json::to_value(mat.unlit).unwrap()
    });
    mat_props.insert("alpha_mode".to_string(), {
        serde_json::to_value(format!("{:?}", mat.alpha_mode)).unwrap()
    });
    mat_props
}

// TODO: export material properties
pub(crate) fn do_export(
    mut state: ResMut<State>,
    images: Res<Assets<Image>>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<ScrapMaterial>>,
    query: Query<(
        Entity,
        &Mesh3d,
        &Name,
        &MaterialName,
        &MapNames,
        &MapTex,
        &ScrapMat,
        Option<&LightmapHandles>,
        Option<&LightmapNames>,
    )>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) -> BevyResult {
    state.export = false;
    let Some(ParsedData::Level(lvl)) = state.data.as_ref() else {
        state.export = true;
        return Ok(());
    };
    // for sm3 in lvl.sm3.iter().flatten() {
    //     for node in &sm3.scene.nodes {
    //         if let Some(NodeData::Light(luz)) = node.content.as_ref() {
    //             dbg!(&luz);
    //         }
    //     };
    // }
    // std::process::exit(1);
    // return Ok(());
    println!("Exporting!");
    let fh = BufWriter::new(std::fs::File::create("dump.zip")?);
    let mut zf = zip::ZipWriter::new(fh);
    let opts = zip::write::SimpleFileOptions::default();
    let mut name_idx: BTreeMap<String, u32> = BTreeMap::new();
    let mut mat_name_count: BTreeMap<String, u32> = BTreeMap::new();
    let mut used_mats: HashSet<Handle<StandardMaterial>> = HashSet::new();
    let mut textures_to_export: HashMap<String, (&Handle<Image>, bool)> = HashMap::new();
    for (
        _ent,
        mesh,
        name,
        MaterialName(mat_name),
        MapNames(map_names),
        MapTex(map_tex),
        ScrapMat(scrap_map),
        lm_handles,
        lm_names,
    ) in query.iter()
    {
        let n: &mut u32 = name_idx.entry(name.to_string()).or_default();
        *n += 1;

        // Get unique material name
        let unique_mat_name = {
            let count = mat_name_count.entry(mat_name.clone()).or_default();
            *count += 1;
            if *count > 1 {
                format!("{}_{}", mat_name, count)
            } else {
                mat_name.clone()
            }
        };
        let map_names = map_names
            .iter()
            .map(|(s, n)| (*s, n.clone()))
            .collect::<HashMap<_, _>>();
        for (slot, handle) in map_tex.iter() {
            if let Some(handle) = handle
                && images.get(handle).is_some()
            {
                let key = map_names[slot].as_str();
                textures_to_export
                    .entry(key.to_owned())
                    .or_insert_with(|| (handle, false));
            }
        }
        if let (Some(LightmapHandles(h1, h2)), Some(LightmapNames(name1, name2))) =
            (lm_handles, lm_names)
        {
            if let Some(h) = h1.as_ref() {
                textures_to_export
                    .entry(name1.clone())
                    .or_insert_with(|| (h, true));
                zf.start_file_from_path(format!("obj/{name}_{n}/_lm1", name = name.as_str()), opts)
                    .unwrap();
                zf.write_all(name1.as_bytes()).unwrap();
            }
            if let Some(h) = h2.as_ref() {
                textures_to_export
                    .entry(name2.clone())
                    .or_insert_with(|| (h, true));
                zf.start_file_from_path(format!("obj/{name}_{n}/_lm2", name = name.as_str()), opts)
                    .unwrap();
                zf.write_all(name2.as_bytes()).unwrap();
            }
        }
        let mesh = meshes.get(mesh).unwrap();
        zf.start_file_from_path(format!("obj/{name}_{n}/_mat", name = name.as_str()), opts)
            .unwrap();
        zf.write_all(unique_mat_name.as_bytes()).unwrap();
        for (k, v) in dump_mesh(mesh) {
            zf.start_file_from_path(format!("obj/{name}_{n}/{k}", name = name.as_str()), opts)
                .unwrap();
            zf.write_all(&v).unwrap();
        }
        zf.start_file_from_path(format!("mat/{name}.json", name = unique_mat_name), opts)
            .unwrap();
        let mat_json = serde_json::to_string_pretty(scrap_map).unwrap();
        zf.write_all(mat_json.as_bytes()).unwrap();
    }

    // Finally, export all collected textures
    for (path, (handle, is_lightmap)) in textures_to_export {
        if let Some(img) = images.get(handle)
            && let Ok(data) = compress_image(img, !is_lightmap, false)
            && !data.is_empty()
        {
            let file_path = if is_lightmap {
                format!("lightmaps/{path}.png")
            } else {
                format!("tex/{path}.png")
            };
            zf.start_file_from_path(&file_path, opts).unwrap();
            zf.write_all(&data).unwrap();
        }
    }
    return Ok(());
}
