//! Trace a Mog program and emit its observed I/O behaviour as JSONL.
//!
//! First step toward "learn from a program by observing it execute" — the
//! same pattern that downstream binary-corpus pretraining will use, with the
//! Mog runtime standing in for an ARM64 emulator until that's wired up.
//!
//! Each input file produces one JSONL line:
//!
//!     {"file":"path", "fn_name":"...", "n_args":N, "examples":[ [[i,j], k], ... ]}
//!
//! The output is directly compatible with downstream synthesis: wrap each
//! `[inputs, output]` pair into an `Example { inputs: Vec<Value::Int>, expected }`.
//!
//! Usage:
//!     cargo run --release --bin program_to_examples -- \
//!         --in path1.mog [--in path2.mog ...]          \
//!         [--out traces.jsonl]                         \
//!         [--n-eval N]                                 \
//!         [--seed N]                                   \
//!         [--fn-name NAME]
//!
//! `--fn-name` overrides automatic detection (read from the `fn` declaration).

use std::fs::{read_to_string, File};
use std::io::{BufWriter, Write};

use mog_synth::program_trace::{trace_function, InputSampler};

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Config {
    inputs: Vec<String>,
    out_path: Option<String>,
    n_eval: usize,
    seed: u64,
    fn_name_override: Option<String>,
}

impl Config {
    fn from_args(args: &[String]) -> Self {
        let inputs: Vec<String> = collect_repeated(args, "--in");
        Self {
            inputs,
            out_path: arg_value(args, "--out"),
            n_eval: arg_value(args, "--n-eval")
                .and_then(|v| v.parse().ok())
                .unwrap_or(16),
            seed: arg_value(args, "--seed")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0xc0ffee),
            fn_name_override: arg_value(args, "--fn-name"),
        }
    }
}

fn collect_repeated(args: &[String], flag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if arg == flag {
            if let Some(val) = iter.next() {
                out.push(val.clone());
            }
        }
    }
    out
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

// ─── Program signature parsing ───────────────────────────────────────────────

/// Parse the first `fn name(arg1: i64, arg2: i64, ...)` from Mog source.
/// Returns `(fn_name, n_args)`.
fn detect_signature(code: &str) -> Option<(String, usize)> {
    let after_fn = code.split_once("fn ")?.1;
    let (name, rest) = after_fn.split_once('(')?;
    let (params, _) = rest.split_once(')')?;
    let n_args = params.split(',').filter(|p| !p.trim().is_empty()).count();
    Some((name.trim().to_string(), n_args))
}

// ─── JSON emission (no external dependency) ──────────────────────────────────

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn examples_json(examples: &[(Vec<i64>, i64)]) -> String {
    let mut s = String::from("[");
    for (i, (inputs, out)) in examples.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("[[");
        for (j, v) in inputs.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&v.to_string());
        }
        s.push_str("],");
        s.push_str(&out.to_string());
        s.push(']');
    }
    s.push(']');
    s
}

fn row_json(file: &str, fn_name: &str, n_args: usize, examples: &[(Vec<i64>, i64)]) -> String {
    format!(
        r#"{{"file":"{}","fn_name":"{}","n_args":{},"examples":{}}}"#,
        json_escape(file),
        json_escape(fn_name),
        n_args,
        examples_json(examples),
    )
}

// ─── Per-file processing ─────────────────────────────────────────────────────

fn process_file(path: &str, cfg: &Config, sampler: &mut InputSampler) -> Option<String> {
    let code = match read_to_string(path) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("[trace] cannot read {path}: {err}");
            return None;
        }
    };
    let (detected_name, n_args) = match detect_signature(&code) {
        Some(sig) => sig,
        None => {
            eprintln!("[trace] {path}: no `fn name(...)` signature found");
            return None;
        }
    };
    let fn_name = cfg.fn_name_override.as_deref().unwrap_or(&detected_name);
    let traces = trace_function(&code, fn_name, n_args, cfg.n_eval, sampler);
    if traces.is_empty() {
        eprintln!(
            "[trace] {path}: no successful traces from {} attempts",
            cfg.n_eval
        );
        return None;
    }
    Some(row_json(path, fn_name, n_args, &traces))
}

// ─── Entry ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = Config::from_args(&args);

    if cfg.inputs.is_empty() {
        eprintln!("[trace] no --in files given; nothing to do");
        std::process::exit(1);
    }

    let mut writer: Option<BufWriter<File>> = match &cfg.out_path {
        Some(path) => match File::create(path) {
            Ok(f) => Some(BufWriter::new(f)),
            Err(err) => {
                eprintln!("[trace] cannot open {path}: {err}");
                std::process::exit(1);
            }
        },
        None => None,
    };

    let mut sampler = InputSampler {
        min: -10,
        max: 20,
        seed: cfg.seed,
    };

    let mut succeeded = 0usize;
    for path in &cfg.inputs {
        if let Some(line) = process_file(path, &cfg, &mut sampler) {
            succeeded += 1;
            if let Some(w) = writer.as_mut() {
                let _ = w.write_all(line.as_bytes());
                let _ = w.write_all(b"\n");
            } else {
                println!("{line}");
            }
        }
    }

    if let Some(w) = writer.as_mut() {
        let _ = w.flush();
    }
    eprintln!("[trace] traced {}/{} files", succeeded, cfg.inputs.len());
}
