use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
};

use color_eyre::eyre::Result;
use serde::Serialize;

use crate::parser::{multi_pack_fs::MultiPackFS, BlendMode, MatPropAttrib, MAT};

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
) -> Result<Vec<MaterialShaderInfo>> {
    let mut out = Vec::with_capacity(materials.len());
    for (key, mat) in materials {
        out.push(analyze_material_shader_from_fs(*key, mat, fs)?);
    }
    Ok(out)
}

fn analyze_material_shader_from_fs(
    material_key: u32,
    mat: &MAT,
    fs: &MultiPackFS,
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

    Ok(MaterialShaderInfo {
        material_key,
        material_name,
        shader_name,
        shader_file,
        assigned_via,
        inputs,
        assembly,
        expression_tree,
        warnings,
    })
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
