//! Standalone NPCoT library runtime binary (NV4).
//!
//! Loads an `ArrayProgramLibrary` JSON, consults it on a query, and emits
//! the result. Runs with zero Python dependencies — pure Rust + optional
//! Metal GPU shader. Intended as the kernel for embedded / edge / WASM
//! inference paths.
//!
//! Usage:
//!     npcot_run <library.json> --hidden 1.0,0.0,0.0 --array 1.0,2.0,3.0
//!     npcot_run lib.json --hidden-file hidden.txt --array-file arr.txt
//!     npcot_run lib.json --hidden 1,0,0 --array 1,2,3 --length 3
//!     npcot_run lib.json --benchmark --iters 10000
//!
//! This binary is intentionally dependency-free beyond `ncpu_metal` itself.
//! It does not pull in clap, serde, or any other crates — CLI parsing is
//! hand-rolled so the resulting binary is tiny (~2 MB stripped).

use ncpu_metal::npcot_exec::{
    consult_library_native, execute_cpu_one, load_library_from_json_bytes,
    DiscreteProgram, NativeLibraryIndex, NpcotGpu,
};
use std::env;
use std::fs;
use std::process;
use std::time::Instant;

struct Args {
    library_path: String,
    hidden: Option<Vec<f32>>,
    array: Option<Vec<f32>>,
    length: Option<u32>,
    hidden_file: Option<String>,
    array_file: Option<String>,
    benchmark: bool,
    iters: usize,
    metal: bool,
    verbose: bool,
}

fn print_usage() {
    eprintln!(
        "usage: npcot_run <library.json> [options]\n\
         \n\
         options:\n\
           --hidden v1,v2,v3       query hidden-state vector (comma separated)\n\
           --array v1,v2,...       input array (comma separated)\n\
           --length N              effective array length (default = array len)\n\
           --hidden-file PATH      read hidden vector from file\n\
           --array-file PATH       read array from file\n\
           --benchmark             time a hot loop of `--iters` consults\n\
           --iters N               iterations for --benchmark (default 10000)\n\
           --metal                 prefer Metal GPU path on benchmark (if available)\n\
           --verbose               extra diagnostics\n"
    );
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    if raw.len() < 2 {
        return Err("library path required".into());
    }
    let mut args = Args {
        library_path: raw[1].clone(),
        hidden: None,
        array: None,
        length: None,
        hidden_file: None,
        array_file: None,
        benchmark: false,
        iters: 10_000,
        metal: false,
        verbose: false,
    };
    let mut i = 2;
    while i < raw.len() {
        match raw[i].as_str() {
            "--hidden" => {
                args.hidden = Some(parse_csv_floats(&raw[i + 1])?);
                i += 2;
            }
            "--array" => {
                args.array = Some(parse_csv_floats(&raw[i + 1])?);
                i += 2;
            }
            "--length" => {
                args.length = Some(raw[i + 1].parse().map_err(|e| format!("{e}"))?);
                i += 2;
            }
            "--hidden-file" => {
                args.hidden_file = Some(raw[i + 1].clone());
                i += 2;
            }
            "--array-file" => {
                args.array_file = Some(raw[i + 1].clone());
                i += 2;
            }
            "--benchmark" => {
                args.benchmark = true;
                i += 1;
            }
            "--iters" => {
                args.iters = raw[i + 1].parse().map_err(|e| format!("{e}"))?;
                i += 2;
            }
            "--metal" => {
                args.metal = true;
                i += 1;
            }
            "--verbose" => {
                args.verbose = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(args)
}

fn parse_csv_floats(s: &str) -> Result<Vec<f32>, String> {
    s.split(',')
        .map(|x| x.trim().parse::<f32>().map_err(|e| format!("{e}")))
        .collect()
}

fn load_vector_from_file(path: &str) -> Result<Vec<f32>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut out = Vec::new();
    for line in text.split(|c: char| c.is_whitespace() || c == ',') {
        if line.is_empty() {
            continue;
        }
        out.push(line.parse::<f32>().map_err(|e| format!("{e}"))?);
    }
    Ok(out)
}

fn find_program(
    index: &NativeLibraryIndex,
    hidden: &[f32],
) -> Option<DiscreteProgram> {
    let norm: f32 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return None;
    }
    let normalized: Vec<f32> = hidden.iter().map(|v| v / norm).collect();
    index.lookup(&normalized).map(|entry| entry.program)
}

fn run_benchmark(
    args: &Args,
    index: &NativeLibraryIndex,
    hidden: &[f32],
    array: &[f32],
    length: u32,
) -> Result<(), String> {
    if let Some(program) = find_program(index, hidden) {
        println!(
            "benchmark: consult+execute for {} iters, library {} entries",
            args.iters,
            index.len()
        );
        // CPU timing
        let start = Instant::now();
        let mut accumulator = 0.0f32;
        for _ in 0..args.iters {
            accumulator += execute_cpu_one(program, array, length);
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "cpu   : {:.3} s total, {:.3} us/call  (acc={})",
            elapsed,
            (elapsed / args.iters as f64) * 1e6,
            accumulator
        );

        if args.metal {
            match NpcotGpu::new() {
                Ok(gpu) => {
                    let start = Instant::now();
                    for _ in 0..args.iters {
                        let _ = gpu
                            .execute(&[program], array, &[length], array.len())
                            .map_err(|e| format!("metal error: {e:?}"))?;
                    }
                    let elapsed = start.elapsed().as_secs_f64();
                    println!(
                        "metal : {:.3} s total, {:.3} us/call",
                        elapsed,
                        (elapsed / args.iters as f64) * 1e6
                    );
                }
                Err(e) => println!("metal: unavailable ({e:?})"),
            }
        }
    } else {
        return Err("benchmark hidden didn't hit any library entry".into());
    }
    Ok(())
}

fn main_inner() -> Result<(), String> {
    let raw: Vec<String> = env::args().collect();
    if raw.len() < 2 {
        print_usage();
        return Err("no arguments supplied".into());
    }
    let args = parse_args(&raw)?;

    let payload = fs::read(&args.library_path)
        .map_err(|e| format!("read library {}: {e}", args.library_path))?;
    let (threshold, index) = load_library_from_json_bytes(&payload)?;
    if args.verbose {
        eprintln!(
            "loaded library: {} entries, similarity_threshold={:.3}",
            index.len(),
            threshold
        );
    }

    let hidden = match (&args.hidden, &args.hidden_file) {
        (Some(h), _) => h.clone(),
        (None, Some(path)) => load_vector_from_file(path)?,
        (None, None) => return Err("--hidden or --hidden-file required".into()),
    };
    let array = match (&args.array, &args.array_file) {
        (Some(a), _) => a.clone(),
        (None, Some(path)) => load_vector_from_file(path)?,
        (None, None) => return Err("--array or --array-file required".into()),
    };
    let length = args.length.unwrap_or(array.len() as u32);

    if args.benchmark {
        return run_benchmark(&args, &index, &hidden, &array, length);
    }

    match consult_library_native(&index, &hidden, &array, length) {
        Some(result) => {
            println!("{result}");
            Ok(())
        }
        None => {
            eprintln!("miss: no library entry matched at threshold");
            process::exit(1);
        }
    }
}

fn main() {
    if let Err(e) = main_inner() {
        eprintln!("error: {e}");
        process::exit(2);
    }
}
