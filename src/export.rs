use std::{
    collections::HashSet,
    io::{BufWriter, Read, Write},
};

use bevy::{
    log::{info, warn},
    prelude::{ResMut, Result as BevyResult},
};
use color_eyre::eyre::{eyre, Result};

use crate::{
    parser::{multi_pack_fs::MultiPackFS, ParsedData},
    State,
};

struct DecodedImage {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

fn compress_image(img: &DecodedImage, quantize: bool, optimize: bool) -> BevyResult<Vec<u8>> {
    let mut liq = imagequant::new();
    liq.set_speed(1)?;
    liq.set_quality(20, 80)?;
    liq.set_log_callback(|_, msg| {
        info!(target:"imagequant","{msg}");
    });

    assert_eq!(img.rgba.len(), (img.width * img.height * 4) as usize);
    let mut buffer = std::io::Cursor::new(Vec::new());

    if quantize {
        let pixels = img
            .rgba
            .chunks(4)
            .map(|w| imagequant::RGBA::from([w[0], w[1], w[2], w[3]]))
            .collect::<Vec<_>>();
        let mut img_data = liq.new_image(pixels, img.width as usize, img.height as usize, 0.0)?;
        let mut res = liq.quantize(&mut img_data)?;
        let (palette, pixels) = res.remapped(&mut img_data)?;
        let mut quantized = image::ImageBuffer::new(img.width, img.height);
        for (x, y, pixel) in quantized.enumerate_pixels_mut() {
            let idx = (y * img.width + x) as usize;
            let p = &palette[pixels[idx] as usize];
            *pixel = image::Rgba([p.r, p.g, p.b, p.a]);
        }
        quantized.write_to(&mut buffer, image::ImageFormat::Png)?;
    } else {
        let rgba = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
            img.width,
            img.height,
            img.rgba.clone(),
        )
        .ok_or_else(|| "Invalid RGBA image buffer".to_string())?;
        rgba.write_to(&mut buffer, image::ImageFormat::Png)?;
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
    Ok(png_bytes)
}

fn decode_standard_image(data: &[u8]) -> Result<DecodedImage> {
    let img = image::load_from_memory(data)?;
    let rgba = img.to_rgba8();
    Ok(DecodedImage {
        width: rgba.width(),
        height: rgba.height(),
        rgba: rgba.into_raw(),
    })
}

fn parse_dds_header(data: &[u8]) -> Option<(u32, u32, texpresso::Format, usize)> {
    if data.len() < 128 || &data[0..4] != b"DDS " {
        return None;
    }

    let read_u32 = |off: usize| {
        data.get(off..off + 4)
            .and_then(|v| v.try_into().ok())
            .map(u32::from_le_bytes)
    };

    let height = read_u32(12)?;
    let width = read_u32(16)?;
    let fourcc = read_u32(84)?;

    let format = match fourcc {
        0x3154_5844 => texpresso::Format::Bc1,
        0x3354_5844 => texpresso::Format::Bc2,
        0x3554_5844 => texpresso::Format::Bc3,
        0x3149_5441 | 0x5534_4342 | 0x5334_4342 => texpresso::Format::Bc4,
        0x3249_5441 | 0x5535_4342 | 0x5335_4342 => texpresso::Format::Bc5,
        _ => return None,
    };

    Some((width, height, format, 128))
}

fn decode_dds_with_texpresso(data: &[u8]) -> Result<DecodedImage> {
    let (width, height, format, offset) =
        parse_dds_header(data).ok_or_else(|| eyre!("Unsupported DDS format"))?;

    let blocks_w = width.div_ceil(4) as usize;
    let blocks_h = height.div_ceil(4) as usize;
    let block_size = match format {
        texpresso::Format::Bc1 | texpresso::Format::Bc4 => 8,
        texpresso::Format::Bc2 | texpresso::Format::Bc3 | texpresso::Format::Bc5 => 16,
    };
    let needed = blocks_w * blocks_h * block_size;
    let end = offset + needed;
    if end > data.len() {
        return Err(eyre!("DDS data is truncated"));
    }

    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    format.decompress(
        &data[offset..end],
        width as usize,
        height as usize,
        &mut rgba,
    );

    Ok(DecodedImage {
        width,
        height,
        rgba,
    })
}

fn load_texture(path: &str, fs: &MultiPackFS) -> Result<DecodedImage> {
    let mut fh = fs.open_file(path)?;
    let mut data = vec![];
    fh.read_to_end(&mut data)?;

    if path.to_ascii_lowercase().ends_with(".dds") {
        return decode_dds_with_texpresso(&data).or_else(|_| decode_standard_image(&data));
    }

    decode_standard_image(&data)
}

fn is_texture_path(path: &str) -> bool {
    let ext = path.rsplit('.').next().unwrap_or_default();
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "dds" | "png" | "bmp" | "tga" | "jpg" | "jpeg"
    )
}

pub(crate) fn do_export(mut state: ResMut<State>) -> BevyResult {
    state.export = false;
    let Some(ParsedData::Level(level)) = state.data.as_ref() else {
        state.export = true;
        return Ok(());
    };

    println!("Exporting textures + JSON only");
    let fh = BufWriter::new(std::fs::File::create("dump.zip")?);
    let mut zf = zip::ZipWriter::new(fh);
    let opts = zip::write::SimpleFileOptions::default();

    zf.start_file_from_path("level/level.json", opts)?;
    let level_json = serde_json::to_string_pretty(&level)?;
    zf.write_all(level_json.as_bytes())?;

    let lightmap_names: HashSet<String> = level
        .emi
        .maps
        .iter()
        .filter_map(|entry| entry.data.as_ref())
        .flat_map(|(a, _, b)| [a.string.clone(), b.string.clone()])
        .collect();

    let mut exported: HashSet<String> = HashSet::new();
    for (logical_name, resolved_path) in &level.dependencies {
        if exported.contains(logical_name) || !is_texture_path(resolved_path) {
            continue;
        }

        let image = match load_texture(resolved_path, &state.fs) {
            Ok(img) => img,
            Err(err) => {
                warn!("Failed to load texture {resolved_path}: {err}");
                continue;
            }
        };

        let is_lightmap = lightmap_names.contains(logical_name);
        let png = match compress_image(&image, !is_lightmap, false) {
            Ok(bytes) if !bytes.is_empty() => bytes,
            Ok(_) => continue,
            Err(err) => {
                warn!("Failed to compress texture {resolved_path}: {err}");
                continue;
            }
        };

        let name = logical_name.replace('\\', "/");
        let zip_path = if is_lightmap {
            format!("lightmaps/{name}.png")
        } else {
            format!("tex/{name}.png")
        };

        zf.start_file_from_path(zip_path, opts)?;
        zf.write_all(&png)?;
        exported.insert(logical_name.clone());
    }

    zf.finish()?;
    Ok(())
}
