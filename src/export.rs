use std::{
    collections::HashMap,
    io::{BufWriter, Cursor, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use bevy::{
    log::tracing::{error, info, warn},
    prelude::ResMut,
};
use color_eyre::eyre::{Result, bail};
use crossbeam_channel::{Receiver, Sender};
use dds::{
    Format,
    header::{Header, ParseOptions},
};
use flate2::{Compression, Status};
use image::{DynamicImage, GenericImage, GenericImageView, ImageFormat, Rgba, RgbaImage};
use scrap_parser::parser::{Level, ParsedData, multi_pack_fs::MultiPackFS, resolve_dep};
use zip::write::SimpleFileOptions;

use crate::State;

fn compress_image_from_bytes(
    data: &[u8],
    ext: &str,
    quantize: bool,
    optimize: bool,
) -> Result<Vec<u8>> {
    if ext.to_lowercase() == "dds" {
        return convert_dds_to_png(data, quantize, optimize);
    }

    let format = match ext.to_lowercase().as_str() {
        "jpg" | "jpeg" => ImageFormat::Jpeg,
        "png" => ImageFormat::Png,
        "bmp" => ImageFormat::Bmp,
        "tga" => ImageFormat::Tga,
        _ => ImageFormat::Png,
    };

    let img = match image::load_from_memory_with_format(data, format) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to decode image {}: {}", ext, e);
            return Ok(vec![]);
        }
    };
    compress_rgba_image(&img, quantize, optimize)
}

fn convert_dds_to_png(data: &[u8], quantize: bool, optimize: bool) -> Result<Vec<u8>> {
    let data = Cursor::new(data);
    let mut decoder = dds::Decoder::new(data).unwrap();
    let size = decoder.main_size();
    let bpp = decoder.native_color().bytes_per_pixel() as usize;
    let mut data = vec![0_u8; size.pixels() as usize * bpp];
    let view = dds::ImageViewMut::new(&mut data, size, dds::ColorFormat::RGBA_U8).unwrap();
    decoder.read_surface(view)?;
    let img = RgbaImage::from_raw(size.width, size.height, data).unwrap();
    let img = DynamicImage::ImageRgba8(img);
    compress_rgba_image(&img, quantize, optimize)
}

fn compress_rgba_image(img: &DynamicImage, quantize: bool, optimize: bool) -> Result<Vec<u8>> {
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();

    let mut buffer = std::io::Cursor::new(Vec::new());

    if quantize {
        let mut liq = imagequant::new();
        liq.set_speed(1)?;
        liq.set_quality(0, 100)?;
        liq.set_log_callback(|_, msg| {
            info!("imagequant: {}", msg);
        });

        let pixels: Vec<_> = rgba
            .as_raw()
            .chunks(4)
            .map(|w| imagequant::RGBA::from([w[0], w[1], w[2], w[3]]))
            .collect();

        let mut img_q = liq.new_image(pixels, width as usize, height as usize, 0.0)?;
        let mut res = liq.quantize(&mut img_q)?;
        let (palette, pixels) = res.remapped(&mut img_q)?;

        let mut quantized_img = RgbaImage::new(width, height);
        for (x, y, pixel) in quantized_img.enumerate_pixels_mut() {
            let idx = (y * width + x) as usize;
            let p = &palette[pixels[idx] as usize];
            *pixel = Rgba([p.r, p.g, p.b, p.a]);
        }

        quantized_img.write_to(&mut buffer, ImageFormat::Png)?;
    } else {
        rgba.write_to(&mut buffer, ImageFormat::Png)?;
    }
    if !optimize {
        return Ok(buffer.into_inner());
    }
    let png_bytes = match oxipng::optimize_from_memory(
        buffer.get_ref(),
        &oxipng::Options {
            deflater: oxipng::Deflater::Zopfli(oxipng::ZopfliOptions {
                iteration_count: 10.try_into().unwrap(),
                ..Default::default()
            }),
            optimize_alpha: true,
            ..oxipng::Options::default()
        },
    ) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("oxipng optimization failed: {}", e);
            return Ok(buffer.into_inner());
        }
    };
    return Ok(png_bytes);
}

fn export_worker(
    fs: MultiPackFS,
    rx: Receiver<(String, String, bool)>, // (src_fs_path, dst_zip_path, is_lightmap)
    tx: Sender<(String, Vec<u8>)>,        // (dst_zip_path, png_data)
) -> Result<()> {
    let mut buffer = Vec::with_capacity(4 * 1024 * 1024); // reuse across jobs
    for (src_path, dst_zip_path, is_lightmap) in rx {
        info!("Processing {} -> {}", src_path, dst_zip_path);
        let (_, src_ext) = src_path.rsplit_once('.').unwrap_or((&src_path, "png"));
        let Ok(mut fh) = fs.open_file(&src_path) else {
            warn!("Failed to open texture: {}", src_path);
            continue;
        };

        buffer.clear();
        if fh.read_to_end(&mut buffer).is_err() {
            warn!("Failed to rad {}", src_path);
            continue;
        }

        let Ok(png_data) = compress_image_from_bytes(&buffer, src_ext, true, true) else {
            warn!("Failed to compress {}", src_path);
            continue;
        };

        if !png_data.is_empty() {
            let (dst_path, _) = dst_zip_path
                .rsplit_once('.')
                .unwrap_or((&dst_zip_path, "png"));
            let dst_path = format!("tex/{dst_path}.png");
            let _ = tx.send((dst_path, png_data));
        }
    }
    Ok(())
}

pub fn export_level(fs: &MultiPackFS, lvl: &Level, output_path: &Path) -> Result<()> {
    info!("Exporting level to {}", output_path.display());
    let ncpus = std::thread::available_parallelism()?.get();
    info!("Compressing textures with {ncpus} workers");
    let (tx, rx, handles) = {
        let fs = fs.clone();
        let (job_tx, job_rx) = crossbeam_channel::unbounded();
        let (res_tx, res_rx) = crossbeam_channel::bounded(ncpus);
        let handles: Vec<_> = (0..ncpus)
            .map(move |_| {
                let (fs, rx, tx) = (fs.clone(), job_rx.clone(), res_tx.clone());
                std::thread::spawn(move || export_worker(fs, rx, tx))
            })
            .collect();
        (job_tx, res_rx, handles)
    };

    let fh = BufWriter::new(std::fs::File::create(output_path)?);
    let mut zf = zip::ZipWriter::new(fh);
    let opts = SimpleFileOptions::default();

    let mut textures_to_export: HashMap<String, (String, bool)> = HashMap::new();
    for mat in &lvl.emi.materials {
        for map_opt in &mat.1.maps {
            if let Some(map) = map_opt.get()
                && !map.texture.string.is_empty()
                && let Some(rpath) = lvl.dependencies.get(&map.texture.string)
            {
                textures_to_export
                    .entry(map.texture.string.clone())
                    .or_insert((rpath.clone(), false));
            }
        }
    }

    for lm in &lvl.emi.maps {
        if let Some((key_1, _, key_2)) = &lm.data {
            for k in &[key_1, key_2] {
                if let Some(rpath) = lvl.dependencies.get(&k.string) {
                    textures_to_export
                        .entry(k.string.clone())
                        .or_insert((rpath.clone(), false));
                }
            }
        }
    }

    zf.start_file("textures.json.gz", opts)?;
    let mut comp = flate2::write::GzEncoder::new(Vec::new(), Compression::best());
    facet_json::to_writer_std(&mut comp, &textures_to_export)?;
    let data = comp.finish()?;
    zf.write_all(&data)?;

    info!("Exporting: {} textures", textures_to_export.len());
    for (dst_path, (src_path, is_lightmap)) in textures_to_export {
        tx.send((src_path, dst_path, is_lightmap))?;
    }
    drop(tx);
    zf.start_file("level.json.gz", opts)?;
    let mut comp = flate2::write::GzEncoder::new(Vec::new(), Compression::best());
    facet_json::to_writer_std(&mut comp, lvl)?;
    let data = comp.finish()?;
    zf.write_all(&data)?;

    for (dst_path, png_data) in rx.iter() {
        if png_data.is_empty() {
            continue;
        }
        zf.start_file(&dst_path, opts)?;
        zf.write_all(&png_data)?;
        info!("wrote {}", dst_path);
    }

    for h in handles {
        h.join().unwrap().unwrap();
    }
    zf.finish()?;
    println!("Export complete!");
    Ok(())
}

pub(crate) fn do_export(mut state: ResMut<State>) -> bevy::prelude::Result {
    state.export = false;

    let fs = &state.fs;
    let lvl: &Level = match state.data.as_ref() {
        Some(ParsedData::Level(l)) => &l,
        _ => {
            return Ok(());
        }
    };

    let out_path = PathBuf::from("dump.zip");

    if let Err(e) = export_level(fs, lvl, &out_path) {
        eprintln!("Export failed: {}", e);
    }
    Ok(())
}
