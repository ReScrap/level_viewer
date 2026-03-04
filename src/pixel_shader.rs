#![allow(dead_code)]
// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx9-graphics-reference-asm-ps-1-x
// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx9-graphics-reference-asm-ps-registers-modifiers-source
// http://archive.gamedev.net/archive/columns/hardcore/dxshader3/page4.html
// r_bias -> r-0.5
// r_x2 -> r*2.0
// r_bx2 -> (r-0.5)*2.0
// [inst]_x(v) -> res*=v
// [inst]_d(v) -> res/=v
// [inst]_sat -> res=clamp(res,0,1)

// TODO: write proper parser

use std::{collections::HashMap, convert::Infallible, str::FromStr};

use color_eyre::eyre::{Context, Result, anyhow, bail};
use fs_err as fs;
use itertools::Itertools;

#[derive(Debug, Clone)]
struct Arg {
    register: String,
    modifiers: Vec<String>,
    field: Vec<usize>,
}

impl FromStr for Arg {
    type Err = color_eyre::eyre::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // t1_bias.b
        let (base_idx_mods, field) = {
            let mut s = s.split('.');
            let base_idx_mods = s.next().unwrap();
            let field = s.next().map(|v| v.to_owned());
            (base_idx_mods, field)
        };
        let (register, modifiers) = {
            let mut s = base_idx_mods.split('_');
            let register = s.next().unwrap();
            let modifiers = s.map(|s| s.to_owned()).collect::<Vec<String>>();
            (register.to_owned(), modifiers)
        };
        let field = field_to_idx(field.as_deref().unwrap_or_default())?;
        Ok(Self {
            register,
            modifiers,
            field,
        })
    }
}

impl std::fmt::Display for Arg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut inner = self.register.to_owned();
        for m in &self.modifiers {
            inner = format!("{m}({inner})");
        }
        if !self.field.is_empty() {
            write!(f, "{inner}{field:?}", field = self.field)
        } else {
            write!(f, "{inner}")
        }
    }
}

#[derive(Debug, Clone)]
struct Cmd {
    name: String,
    modifiers: Vec<String>,
    dest: Option<Arg>,
    args: Vec<Arg>,
}

// impl Cmd {
//     fn apply_dest(&mut self) -> Option<&str> {
//         match self.name.as_str() {
//             "tex" =>
//         }
//     }
// }

impl std::fmt::Display for Cmd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let args = self.args.iter().skip(1).map(|a| format!("{a}")).join(", ");
        let mut inner = format!("{name}({args})", name = self.name);
        for m in &self.modifiers {
            inner = format!("{m}({inner})");
        }
        if let Some(dest) = self.args.first() {
            write!(f, "{dest}={inner}")
        } else {
            write!(f, "{inner}")
        }
    }
}

impl FromStr for Cmd {
    type Err = color_eyre::eyre::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let line: Vec<&str> = s
            .split(|c: char| c.is_ascii_whitespace() || c == ',')
            .map(|c| c.trim_end_matches(',').trim_start_matches('+'))
            .filter(|w| !w.is_empty())
            .collect();
        let (cmd, args) = match line.split_first() {
            Some((&cmd, args)) => (cmd, args),
            None => unreachable!(),
        };
        let cmd = cmd.split('_').collect::<Vec<&str>>();
        let (&cmd, cmd_mods) = cmd.split_first().unwrap();
        let mut args = args
            .iter()
            .map(|a| a.parse::<Arg>())
            .collect::<Result<Vec<Arg>>>()?;
        if cmd == "tex" {
            args = vec![args[0].clone(), args[0].clone()];
        }
        Ok(Cmd {
            name: cmd.to_owned(),
            modifiers: cmd_mods.iter().map(|&s| s.to_owned()).collect(),
            args,
            dest: None,
        })
    }
}

fn field_to_idx(field: &str) -> Result<Vec<usize>> {
    field
        .chars()
        .map(|c| {
            Ok(match c {
                'x' | 'r' => 0,
                'y' | 'g' => 1,
                'z' | 'b' => 2,
                'w' | 'a' => 3,
                fld => bail!("Invalid index: {fld}"),
            })
        })
        .collect::<Result<Vec<usize>>>()
}

#[derive(Debug)]
enum Node {
    Args {
        register: String,
        modifiers: Vec<String>,
        field: Option<String>,
    },
    Cmd {
        name: String,
        modifiers: Vec<String>,
        args: Vec<Self>,
    },
}

fn parse(path: &str) -> Result<()> {
    let mut cmds = vec![];
    let data = fs::read_to_string(path)?;
    for line in data.lines() {
        let mut line = line.trim().split("//");
        let line = line.next().unwrap_or_default();
        if line.is_empty() || line.starts_with("ps.") {
            continue;
        }
        let cmd = line.parse::<Cmd>().unwrap();
        println!("{line} -> {cmd}");
        cmds.push(cmd);
    }
    // TODO: write own expression tree builder
    todo!()
}

// #[cfg(test)]
// mod test {
//     #[test]
//     fn test() {
//         let var = std::env::var("SHADER_FILE").unwrap();
//         super::parse(&var).unwrap();
//         // super::parse(r"E:\Games\Steam\steamapps\common\Scrapland\ext\bmp\hologram.psh").unwrap();
//     }
// }
