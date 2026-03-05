use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
};

use color_eyre::eyre::Result;
use serde::Serialize;

use crate::parser::{multi_pack_fs::MultiPackFS, BlendMode, IniData, MatPropAttrib, MAT};

#[derive(Debug, Clone, Serialize)]
pub(crate) struct EngineVarsSnapshot {
    pub pshaders: bool,
    pub psmask: u32,
    pub env_blend: bool,
    pub env_map_view_dep: i32,
    pub env_map_scale: f32,
    pub env_map_offset: f32,
    pub env_bump_scale: f32,
    pub env_bump_bias: bool,
    pub cloud_vel1x: f32,
    pub cloud_vel1y: f32,
    pub cloud_vel2x: f32,
    pub cloud_vel2y: f32,
    pub cloud_scale1: f32,
    pub cloud_scale2: f32,
    pub cloud_emi: f32,
    pub cloud_r: f32,
    pub cloud_g: f32,
    pub cloud_b: f32,
    pub cloud_a: f32,
    pub glow_flick_tile: f32,
    pub glow_flick_rot: f32,
    pub glow_flick_bump: f32,
    pub glow_flick_brot: f32,
    pub glow_flick_mod: f32,
}

impl Default for EngineVarsSnapshot {
    fn default() -> Self {
        Self {
            pshaders: true,
            psmask: 0,
            env_blend: true,
            env_map_view_dep: 2,
            env_map_scale: 1.0,
            env_map_offset: 0.0,
            env_bump_scale: 1.0,
            env_bump_bias: false,
            cloud_vel1x: 0.0,
            cloud_vel1y: 0.0,
            cloud_vel2x: 0.0,
            cloud_vel2y: 0.0,
            cloud_scale1: 1.0,
            cloud_scale2: 1.0,
            cloud_emi: 1.0,
            cloud_r: 1.0,
            cloud_g: 1.0,
            cloud_b: 1.0,
            cloud_a: 1.0,
            glow_flick_tile: 1.0,
            glow_flick_rot: 0.0,
            glow_flick_bump: 0.4,
            glow_flick_brot: 0.0,
            glow_flick_mod: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RenderUvLayer {
    pub slot: u8,
    pub velocity_u: f32,
    pub velocity_v: f32,
    pub scale: f32,
    pub rotation: f32,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ShaderRenderLogic {
    pub shader_kind: String,
    pub uses_env_map: bool,
    pub uses_env_bump: bool,
    pub uses_env_blend: bool,
    pub env_stage: Option<u8>,
    pub env_map_view_dep: i32,
    pub env_map_scale: f32,
    pub env_map_offset: f32,
    pub env_bump_scale: f32,
    pub env_bump_bias: bool,
    pub env_bump_extra_offset: f32,
    pub pshader_enabled: bool,
    pub psmask_block_bit: Option<u32>,
    pub uv_layers: Vec<RenderUvLayer>,
    pub cloud_tint: Option<[f32; 4]>,
    pub glow_flick: Option<[f32; 3]>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct MaterialShaderInfo {
    pub material_key: u32,
    pub material_name: String,
    pub shader_name: String,
    pub shader_file: Option<String>,
    pub assigned_via: String,
    pub inputs: Vec<ShaderInput>,
    pub assembly: Option<String>,
    pub expression_tree: Option<ExpressionTree>,
    pub engine_vars: EngineVarsSnapshot,
    pub render_logic: ShaderRenderLogic,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ShaderInput {
    pub register: String,
    pub semantic: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ParsedInstruction {
    pub line: usize,
    pub op: String,
    pub op_modifiers: Vec<String>,
    pub coissue: bool,
    pub dst: Option<Operand>,
    pub src: Vec<Operand>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct Operand {
    pub register: String,
    pub swizzle: Option<String>,
    pub modifiers: Vec<String>,
    pub negate: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ExpressionTree {
    pub instructions: Vec<ParsedInstruction>,
    pub nodes: Vec<ExpressionNode>,
    pub outputs: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ExpressionNode {
    pub id: usize,
    #[serde(flatten)]
    pub kind: ExpressionNodeKind,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ExpressionNodeKind {
    Input {
        register: String,
    },
    Operation {
        op: String,
        args: Vec<usize>,
        modifiers: Vec<String>,
        coissue: bool,
        line: usize,
    },
    Swizzle {
        input: usize,
        mask: String,
    },
    Modifier {
        input: usize,
        modifier: String,
    },
    Negate {
        input: usize,
    },
    Merge {
        base: usize,
        value: usize,
        mask: String,
    },
}

pub(crate) fn analyze_level_material_shaders_from_fs(
    materials: &[(u32, MAT)],
    fs: &MultiPackFS,
    config: &IniData,
) -> Result<Vec<MaterialShaderInfo>> {
    let engine_vars = engine_vars_from_config(config);
    let mut out = Vec::with_capacity(materials.len());
    for (key, mat) in materials {
        out.push(analyze_material_shader_from_fs(
            *key,
            mat,
            fs,
            &engine_vars,
        )?);
    }
    Ok(out)
}

fn analyze_material_shader_from_fs(
    material_key: u32,
    mat: &MAT,
    fs: &MultiPackFS,
    engine_vars: &EngineVarsSnapshot,
) -> Result<MaterialShaderInfo> {
    let material_name = mat
        .name
        .as_ref()
        .map(|name| name.string.clone())
        .unwrap_or_else(|| format!("MAT:{material_key}"));

    let auto_shader = auto_assign_shader(mat);
    let override_shader = parse_shader_override(&material_name);
    let (shader_name, assigned_via) = match override_shader {
        Some(name) if !name.is_empty() => (name, "material_override".to_owned()),
        _ => (auto_shader, "material_flags".to_owned()),
    };

    let mut warnings = Vec::new();
    let mut inputs = Vec::new();
    let mut assembly = None;
    let mut expression_tree = None;
    let mut shader_file = None;

    if let Some((path, src)) = load_shader_from_multipack(fs, &shader_name)? {
        let (parsed_inputs, tree) = parse_shader_source(&src);
        inputs = parsed_inputs;
        assembly = Some(src);
        expression_tree = Some(tree);
        shader_file = Some(path);
    } else {
        warnings.push(format!(
            "No .psh file found in MultiPackFS for shader '{shader_name}'"
        ));
    }

    let render_logic = infer_render_logic(&shader_name, engine_vars);

    Ok(MaterialShaderInfo {
        material_key,
        material_name,
        shader_name,
        shader_file,
        assigned_via,
        inputs,
        assembly,
        expression_tree,
        engine_vars: engine_vars.clone(),
        render_logic,
        warnings,
    })
}

fn infer_render_logic(shader_name: &str, vars: &EngineVarsSnapshot) -> ShaderRenderLogic {
    let lower = shader_name.to_ascii_lowercase();
    let shader_kind = shader_kind_from_name(&lower).to_owned();
    let uses_env_map = lower.contains("envmap") || lower.contains("envbump");
    let uses_env_bump = lower.contains("envbump");
    let uses_env_blend = lower.contains("maskenv") || lower.contains("envmap");
    let env_stage = if uses_env_bump {
        Some(2)
    } else if uses_env_map {
        Some(1)
    } else {
        None
    };
    let psmask_block_bit = shader_psmask_block_bit(shader_name);
    let mut uv_layers = Vec::new();
    let mut cloud_tint = None;
    let mut glow_flick = None;

    match shader_kind.as_str() {
        "clouds" => {
            uv_layers.push(RenderUvLayer {
                slot: 0,
                velocity_u: vars.cloud_vel1x,
                velocity_v: vars.cloud_vel1y,
                scale: vars.cloud_scale1,
                rotation: 0.0,
            });
            uv_layers.push(RenderUvLayer {
                slot: 1,
                velocity_u: vars.cloud_vel2x,
                velocity_v: vars.cloud_vel2y,
                scale: vars.cloud_scale2,
                rotation: 0.0,
            });
            cloud_tint = Some([
                vars.cloud_r * vars.cloud_emi,
                vars.cloud_g * vars.cloud_emi,
                vars.cloud_b * vars.cloud_emi,
                vars.cloud_a * vars.cloud_emi,
            ]);
        }
        "glowflick" => {
            uv_layers.push(RenderUvLayer {
                slot: 1,
                velocity_u: 0.01,
                velocity_v: 0.1,
                scale: vars.glow_flick_tile,
                rotation: vars.glow_flick_rot,
            });
            glow_flick = Some([
                vars.glow_flick_bump,
                vars.glow_flick_brot,
                vars.glow_flick_mod,
            ]);
        }
        "electric" => {
            uv_layers.push(RenderUvLayer {
                slot: 0,
                velocity_u: -0.1,
                velocity_v: 0.26,
                scale: 1.2,
                rotation: 0.0,
            });
            uv_layers.push(RenderUvLayer {
                slot: 1,
                velocity_u: 0.08,
                velocity_v: -0.06,
                scale: 5.4,
                rotation: 0.0,
            });
        }
        "fire" => {
            uv_layers.push(RenderUvLayer {
                slot: 0,
                velocity_u: -0.5,
                velocity_v: 0.8,
                scale: 4.0,
                rotation: 0.0,
            });
            uv_layers.push(RenderUvLayer {
                slot: 1,
                velocity_u: 0.02,
                velocity_v: -0.03,
                scale: 6.0,
                rotation: 0.0,
            });
        }
        _ => {}
    }

    ShaderRenderLogic {
        shader_kind,
        uses_env_map,
        uses_env_bump,
        uses_env_blend,
        env_stage,
        env_map_view_dep: vars.env_map_view_dep,
        env_map_scale: vars.env_map_scale,
        env_map_offset: vars.env_map_offset,
        env_bump_scale: vars.env_bump_scale,
        env_bump_bias: vars.env_bump_bias,
        env_bump_extra_offset: if vars.env_bump_bias {
            vars.env_bump_scale * 0.5
        } else {
            0.0
        },
        pshader_enabled: vars.pshaders,
        psmask_block_bit,
        uv_layers,
        cloud_tint,
        glow_flick,
    }
}

fn shader_kind_from_name(lower: &str) -> &'static str {
    match lower {
        "clouds" => "clouds",
        "glowflick" => "glowflick",
        "electric" => "electric",
        "fire" => "fire",
        "glass" => "glass",
        "waves" => "waves",
        "scroll" => "scroll",
        "bloomfilter" | "bloomblur" | "bloomtarget" => "postprocess_bloom",
        "motionbluradd" | "motionblurtarget" => "postprocess_motion_blur",
        "blurtarget" => "postprocess_blur",
        "dudvfilter" | "dudvtarget" => "postprocess_dudv",
        "radialblurtarget" => "postprocess_radial_blur",
        _ => "material",
    }
}

fn shader_psmask_block_bit(shader_name: &str) -> Option<u32> {
    let lower = shader_name.to_ascii_lowercase();
    if lower == "envmap" {
        return Some(0x400);
    }
    if lower == "envmaplightmap" {
        return Some(0x800);
    }
    if lower == "maskenvmap" {
        return Some(0x10);
    }
    if lower == "maskenvmaplightmap" {
        return Some(0x100);
    }
    if lower == "maskenvbump" {
        return Some(0x8);
    }
    if lower == "maskenvbumplightmap" {
        return Some(0x80);
    }
    if lower == "glowmapmaskenvmap" {
        return Some(0x4000);
    }
    if lower == "glowmapmaskenvmaplightmap" {
        return Some(0x20000);
    }
    if lower == "glowmapmaskenvbump" {
        return Some(0x10000);
    }
    if lower == "glowmapmaskenvbumplightmap" {
        return Some(0x100000);
    }
    if lower == "clouds" {
        return Some(0x200);
    }
    if lower == "glowmap" {
        return Some(0x1000);
    }
    if lower == "glowflick" {
        return Some(0x8000);
    }
    if lower == "electric" {
        return Some(0x10000);
    }
    if lower == "fire" {
        return Some(0x20000);
    }
    if lower == "glass" {
        return Some(0x40000);
    }
    if lower == "waves" {
        return Some(0x80000);
    }
    if lower == "bloomfilter" {
        return Some(0x100000);
    }
    if lower == "bloomblur" {
        return Some(0x200000);
    }
    if lower == "bloomtarget" {
        return Some(0x400000);
    }
    if lower == "motionbluradd" {
        return Some(0x800000);
    }
    if lower == "motionblurtarget" {
        return Some(0x1000000);
    }
    if lower == "blurtarget" {
        return Some(0x2000000);
    }
    if lower == "dudvfilter" {
        return Some(0x4000000);
    }
    if lower == "dudvtarget" {
        return Some(0x8000000);
    }
    if lower == "radialblurtarget" {
        return Some(0x10000000);
    }
    if lower == "diffusenolit" {
        return Some(0x20000000);
    }
    None
}

fn engine_vars_from_config(config: &IniData) -> EngineVarsSnapshot {
    let mut flat = HashMap::<String, String>::new();
    for section in config.values() {
        for (key, value) in section {
            if let Some(value) = value {
                flat.insert(key.to_ascii_lowercase(), value.trim().to_owned());
            }
        }
    }

    let get = |name: &str| flat.get(&name.to_ascii_lowercase()).map(|s| s.as_str());
    let mut vars = EngineVarsSnapshot::default();

    if let Some(v) = get("r_pshaders") {
        vars.pshaders = parse_bool(v, vars.pshaders);
    }
    if let Some(v) = get("r_psmask") {
        vars.psmask = parse_u32(v, vars.psmask);
    }
    if let Some(v) = get("r_envblend") {
        vars.env_blend = parse_bool(v, vars.env_blend);
    }
    if let Some(v) = get("r_envmapviewdep") {
        vars.env_map_view_dep = parse_i32(v, vars.env_map_view_dep);
    }
    if let Some(v) = get("r_envmapscale") {
        vars.env_map_scale = parse_f32(v, vars.env_map_scale);
    }
    if let Some(v) = get("r_envmapoffset") {
        vars.env_map_offset = parse_f32(v, vars.env_map_offset);
    }
    if let Some(v) = get("r_envbumpscale") {
        vars.env_bump_scale = parse_f32(v, vars.env_bump_scale);
    }
    if let Some(v) = get("r_envbumpbias") {
        vars.env_bump_bias = parse_bool(v, vars.env_bump_bias);
    }
    if let Some(v) = get("r_cloudvel1x") {
        vars.cloud_vel1x = parse_f32(v, vars.cloud_vel1x);
    }
    if let Some(v) = get("r_cloudvel1y") {
        vars.cloud_vel1y = parse_f32(v, vars.cloud_vel1y);
    }
    if let Some(v) = get("r_cloudvel2x") {
        vars.cloud_vel2x = parse_f32(v, vars.cloud_vel2x);
    }
    if let Some(v) = get("r_cloudvel2y") {
        vars.cloud_vel2y = parse_f32(v, vars.cloud_vel2y);
    }
    if let Some(v) = get("r_cloudscale1") {
        vars.cloud_scale1 = parse_f32(v, vars.cloud_scale1);
    }
    if let Some(v) = get("r_cloudscale2") {
        vars.cloud_scale2 = parse_f32(v, vars.cloud_scale2);
    }
    if let Some(v) = get("r_cloudemi") {
        vars.cloud_emi = parse_f32(v, vars.cloud_emi);
    }
    if let Some(v) = get("r_cloudr") {
        vars.cloud_r = parse_f32(v, vars.cloud_r);
    }
    if let Some(v) = get("r_cloudg") {
        vars.cloud_g = parse_f32(v, vars.cloud_g);
    }
    if let Some(v) = get("r_cloudb") {
        vars.cloud_b = parse_f32(v, vars.cloud_b);
    }
    if let Some(v) = get("r_clouda") {
        vars.cloud_a = parse_f32(v, vars.cloud_a);
    }
    if let Some(v) = get("r_glowflicktile") {
        vars.glow_flick_tile = parse_f32(v, vars.glow_flick_tile);
    }
    if let Some(v) = get("r_glowflickrot") {
        vars.glow_flick_rot = parse_f32(v, vars.glow_flick_rot);
    }
    if let Some(v) = get("r_glowflickbump") {
        vars.glow_flick_bump = parse_f32(v, vars.glow_flick_bump);
    }
    if let Some(v) = get("r_glowflickbrot") {
        vars.glow_flick_brot = parse_f32(v, vars.glow_flick_brot);
    }
    if let Some(v) = get("r_glowflickmod") {
        vars.glow_flick_mod = parse_f32(v, vars.glow_flick_mod);
    }

    vars
}

fn parse_bool(raw: &str, default: bool) -> bool {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => default,
    }
}

fn parse_u32(raw: &str, default: u32) -> u32 {
    let s = raw.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        return u32::from_str_radix(hex, 16).unwrap_or(default);
    }
    s.parse::<u32>().unwrap_or(default)
}

fn parse_i32(raw: &str, default: i32) -> i32 {
    raw.trim().parse::<i32>().unwrap_or(default)
}

fn parse_f32(raw: &str, default: f32) -> f32 {
    raw.trim().parse::<f32>().unwrap_or(default)
}

fn auto_assign_shader(mat: &MAT) -> String {
    let has_bump = mat.maps[3].get().is_some();
    let has_glow = mat.maps[4].get().is_some();
    let env_flag = mat.mat_props.env_map != 0;
    let has_env_map = env_flag
        && mat.maps[0]
            .get()
            .map(|map| map.is_env != 0)
            .unwrap_or(false);

    let mut shader = if has_glow {
        if has_bump {
            if has_env_map {
                "GlowmapMaskEnvBump"
            } else {
                "Glowmap"
            }
        } else if has_env_map {
            "GlowmapMaskEnvmap"
        } else {
            "Glowmap"
        }
    } else if has_bump {
        if has_env_map {
            "MaskEnvBump"
        } else if env_flag {
            "EnvMap"
        } else {
            "Diffuse"
        }
    } else if has_env_map {
        "MaskEnvmap"
    } else if env_flag {
        "EnvMap"
    } else {
        "Diffuse"
    };

    let diffuse_no_lit = mat.mat_props.diffuse_alpha != 0
        && mat.mat_props.src_blend == BlendMode::Zero
        && mat.mat_props.dst_blend == BlendMode::SrcColor
        && !mat.mat_props.attrib.contains(&MatPropAttrib::ZBIAS);
    if diffuse_no_lit {
        shader = "DiffuseNoLit";
    }

    shader.to_owned()
}

fn parse_shader_override(material_name: &str) -> Option<String> {
    let lower = material_name.to_ascii_lowercase();
    let start = lower.find("(+shader:")?;
    let rest = &material_name[start + "(+shader:".len()..];
    let end = rest.find(')')?;
    Some(rest[..end].trim().to_owned())
}

fn shader_file_candidates(shader_name: &str) -> Vec<String> {
    let mut candidates = vec![format!("bmp/{shader_name}.psh")];

    if shader_name.eq_ignore_ascii_case("Clouds") {
        candidates.push("bmp/CloudTest.psh".to_owned());
    }
    if shader_name.eq_ignore_ascii_case("GlowFlick") {
        candidates.push("bmp/GlowFlickLightmap.psh".to_owned());
    }

    candidates.push(format!("bmp/{}.psh", shader_name.to_ascii_lowercase()));

    let mut dedup = BTreeMap::<String, ()>::new();
    for candidate in candidates {
        dedup.entry(candidate).or_insert(());
    }
    dedup.into_keys().collect()
}

fn load_shader_from_multipack(
    fs: &MultiPackFS,
    shader_name: &str,
) -> Result<Option<(String, String)>> {
    for candidate in shader_file_candidates(shader_name) {
        let Ok(mut fh) = fs.open_file(&candidate) else {
            continue;
        };
        let mut data = Vec::new();
        fh.read_to_end(&mut data)?;
        let source = String::from_utf8_lossy(&data).to_string();
        return Ok(Some((candidate, source)));
    }
    Ok(None)
}

fn parse_shader_source(src: &str) -> (Vec<ShaderInput>, ExpressionTree) {
    let mut declared_inputs = BTreeMap::<String, String>::new();
    let mut instructions = Vec::<ParsedInstruction>::new();
    let mut used_registers = BTreeMap::<String, ()>::new();

    for (idx, raw_line) in src.lines().enumerate() {
        let line_num = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix("//") {
            if let Some((register, semantic)) = parse_input_comment(rest.trim()) {
                declared_inputs.insert(register, semantic);
            }
            continue;
        }

        if line.starts_with("ps.") || line.starts_with("def ") {
            continue;
        }

        if let Some(parsed) = parse_instruction(line, line_num) {
            if let Some(dst) = parsed.dst.as_ref() {
                used_registers.insert(dst.register.clone(), ());
            }
            for src in &parsed.src {
                used_registers.insert(src.register.clone(), ());
            }
            instructions.push(parsed);
        }
    }

    for register in used_registers.keys() {
        declared_inputs
            .entry(register.clone())
            .or_insert_with(|| infer_semantic(register));
    }

    let inputs = declared_inputs
        .into_iter()
        .map(|(register, semantic)| ShaderInput { register, semantic })
        .collect();

    let tree = build_expression_tree(&instructions);
    (inputs, tree)
}

fn parse_input_comment(comment: &str) -> Option<(String, String)> {
    let (left, right) = comment.split_once('=')?;
    let register = left.trim().to_ascii_lowercase();
    if !is_register_name(&register) {
        return None;
    }
    Some((register, right.trim().to_owned()))
}

fn parse_instruction(line: &str, line_num: usize) -> Option<ParsedInstruction> {
    let mut text = line.trim();
    let coissue = text.starts_with('+');
    if coissue {
        text = text.trim_start_matches('+').trim_start();
    }

    let mut split = text.splitn(2, char::is_whitespace);
    let head = split.next()?.trim();
    let args = split.next().unwrap_or_default().trim();
    if head.is_empty() {
        return None;
    }

    let mut head_parts = head.split('_');
    let op = head_parts.next()?.to_ascii_lowercase();
    let op_modifiers = head_parts
        .map(|m| m.to_ascii_lowercase())
        .collect::<Vec<_>>();

    let parsed_args = args
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .filter_map(parse_operand)
        .collect::<Vec<_>>();

    let (dst, src) = if op == "tex" {
        if parsed_args.is_empty() {
            (None, Vec::new())
        } else {
            (Some(parsed_args[0].clone()), vec![parsed_args[0].clone()])
        }
    } else {
        let mut iter = parsed_args.into_iter();
        let dst = iter.next();
        let src = iter.collect();
        (dst, src)
    };

    Some(ParsedInstruction {
        line: line_num,
        op,
        op_modifiers,
        coissue,
        dst,
        src,
    })
}

fn parse_operand(text: &str) -> Option<Operand> {
    let mut s = text.trim();
    if s.is_empty() {
        return None;
    }

    let negate = s.starts_with('-');
    if negate {
        s = s.trim_start_matches('-').trim();
    }

    let (base, swizzle) = match s.split_once('.') {
        Some((lhs, rhs)) => (lhs, Some(rhs.to_ascii_lowercase())),
        None => (s, None),
    };

    let mut parts = base.split('_');
    let register = parts.next()?.to_ascii_lowercase();
    if !is_register_name(&register) {
        return None;
    }

    let modifiers = parts.map(|m| m.to_ascii_lowercase()).collect::<Vec<_>>();
    Some(Operand {
        register,
        swizzle,
        modifiers,
        negate,
    })
}

fn is_register_name(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !matches!(first, 'r' | 't' | 'v' | 'c') {
        return false;
    }
    chars.all(|c| c.is_ascii_digit())
}

fn infer_semantic(register: &str) -> String {
    let mut chars = register.chars();
    let Some(prefix) = chars.next() else {
        return "unknown".to_owned();
    };
    match prefix {
        't' => "texture".to_owned(),
        'v' => "vertex_color".to_owned(),
        'c' => "constant".to_owned(),
        'r' => "temp_register".to_owned(),
        _ => "unknown".to_owned(),
    }
}

fn build_expression_tree(instructions: &[ParsedInstruction]) -> ExpressionTree {
    let mut nodes = Vec::<ExpressionNode>::new();
    let mut register_nodes = HashMap::<String, usize>::new();

    let ensure_input = |register: &str,
                        nodes: &mut Vec<ExpressionNode>,
                        register_nodes: &mut HashMap<String, usize>| {
        if let Some(id) = register_nodes.get(register) {
            return *id;
        }
        let id = nodes.len();
        nodes.push(ExpressionNode {
            id,
            kind: ExpressionNodeKind::Input {
                register: register.to_owned(),
            },
        });
        register_nodes.insert(register.to_owned(), id);
        id
    };

    for inst in instructions {
        let mut args = Vec::with_capacity(inst.src.len());
        for src in &inst.src {
            let mut cur = register_nodes
                .get(&src.register)
                .copied()
                .unwrap_or_else(|| ensure_input(&src.register, &mut nodes, &mut register_nodes));

            if let Some(swizzle) = &src.swizzle {
                let id = nodes.len();
                nodes.push(ExpressionNode {
                    id,
                    kind: ExpressionNodeKind::Swizzle {
                        input: cur,
                        mask: swizzle.clone(),
                    },
                });
                cur = id;
            }

            for modifier in &src.modifiers {
                let id = nodes.len();
                nodes.push(ExpressionNode {
                    id,
                    kind: ExpressionNodeKind::Modifier {
                        input: cur,
                        modifier: modifier.clone(),
                    },
                });
                cur = id;
            }

            if src.negate {
                let id = nodes.len();
                nodes.push(ExpressionNode {
                    id,
                    kind: ExpressionNodeKind::Negate { input: cur },
                });
                cur = id;
            }

            args.push(cur);
        }

        let op_id = nodes.len();
        nodes.push(ExpressionNode {
            id: op_id,
            kind: ExpressionNodeKind::Operation {
                op: inst.op.clone(),
                args,
                modifiers: inst.op_modifiers.clone(),
                coissue: inst.coissue,
                line: inst.line,
            },
        });

        if let Some(dst) = &inst.dst {
            let base = register_nodes
                .get(&dst.register)
                .copied()
                .unwrap_or_else(|| ensure_input(&dst.register, &mut nodes, &mut register_nodes));
            let mask = dst.swizzle.clone().unwrap_or_else(|| "xyzw".to_owned());
            let final_node = if mask == "xyzw" || mask == "rgba" {
                op_id
            } else {
                let id = nodes.len();
                nodes.push(ExpressionNode {
                    id,
                    kind: ExpressionNodeKind::Merge {
                        base,
                        value: op_id,
                        mask,
                    },
                });
                id
            };
            register_nodes.insert(dst.register.clone(), final_node);
        }
    }

    let mut outputs = BTreeMap::new();
    for (register, id) in register_nodes {
        if register.starts_with('r') {
            outputs.insert(register, id);
        }
    }

    ExpressionTree {
        instructions: instructions.to_vec(),
        nodes,
        outputs,
    }
}
