//! High-level code generation CLI: "give me a function from examples".
//!
//! Accepts I/O examples as JSON, runs the full solver, transpiles the
//! synthesized Mog program to the requested target language, prints the
//! result. This is the single entry point a user or downstream tool
//! should call when they want a working function.
//!
//! Input schema (JSON on stdin OR via `--examples JSON`):
//!   {
//!     "name": "double",
//!     "signature": "fn double(a: i64) -> i64",
//!     "examples": [
//!       {"inputs": [0], "expected": 0},
//!       {"inputs": [3], "expected": 6},
//!       ...
//!     ]
//!   }
//!
//! Output: transpiled code on stdout (or `--out PATH`). Supported targets:
//!   python, rust, typescript, mog (source Mog program untouched)
//!
//! Usage:
//!     # pipe JSON via stdin
//!     echo '{"name":"double",...}' | \
//!         cargo run --release --bin nsynth_codegen -- --lang python
//!
//!     # explicit args
//!     cargo run --release --bin nsynth_codegen -- \
//!         --examples '{"name":"triple","signature":"fn triple(a: i64) -> i64","examples":[{"inputs":[1],"expected":3}]}' \
//!         --lang rust
//!
//! Exit codes:
//!   0 - success, code printed
//!   1 - parse / synthesis failure
//!   2 - bad CLI args

use std::io::{Read, Write};

use serde::Deserialize;

use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::mog_transpile::{to_python, to_rust, to_typescript};
use mog_synth::solver::solve_problem;

#[derive(Deserialize)]
struct InputExample {
    inputs: Vec<serde_json::Value>,
    expected: i64,
}

#[derive(Deserialize)]
struct InputProblem {
    name: String,
    signature: String,
    examples: Vec<InputExample>,
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn value_from_json(v: &serde_json::Value) -> Option<Value> {
    if let Some(i) = v.as_i64() {
        return Some(Value::Int(i));
    }
    if let Some(arr) = v.as_array() {
        let ints: Option<Vec<i64>> = arr.iter().map(|x| x.as_i64()).collect();
        if let Some(ints) = ints {
            return Some(Value::int_array(&ints));
        }
    }
    if let Some(obj) = v.as_object() {
        if let Some(n) = obj.get("Int").and_then(|x| x.as_i64()) {
            return Some(Value::Int(n));
        }
    }
    None
}

fn to_problem(input: InputProblem) -> Result<Problem, String> {
    let mut examples = Vec::with_capacity(input.examples.len());
    for (i, ex) in input.examples.into_iter().enumerate() {
        let mut ins = Vec::with_capacity(ex.inputs.len());
        for v in ex.inputs {
            let val = value_from_json(&v).ok_or_else(|| {
                format!("example {i}: input {v} not recognised (expected Int or [Int])")
            })?;
            ins.push(val);
        }
        examples.push(Example {
            inputs: ins,
            expected: Value::Int(ex.expected),
        });
    }
    if examples.is_empty() {
        return Err("no examples provided".to_string());
    }
    // Leak the signature string — the Problem struct holds `&'static str`
    // because the baked-in benchmark is all literals. Leaking once per
    // invocation is fine.
    let signature: &'static str = Box::leak(input.signature.into_boxed_str());
    Ok(Problem {
        name: input.name,
        category: "codegen",
        description: "",
        signature,
        examples,
        holdouts: vec![],
        reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        functions: vec![],
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let lang = arg_value(&args, "--lang").unwrap_or_else(|| "python".to_string());
    let out_path = arg_value(&args, "--out");
    let verbose = has_flag(&args, "--verbose");

    // Read problem JSON from --examples arg or stdin.
    let raw = match arg_value(&args, "--examples") {
        Some(s) => s,
        None => {
            let mut buf = String::new();
            if std::io::stdin().read_to_string(&mut buf).is_err() || buf.trim().is_empty() {
                eprintln!("[nsynth_codegen] no JSON on stdin and no --examples arg");
                std::process::exit(2);
            }
            buf
        }
    };

    let parsed: InputProblem = match serde_json::from_str(&raw) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("[nsynth_codegen] JSON parse error: {err}");
            std::process::exit(1);
        }
    };

    let problem = match to_problem(parsed) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("[nsynth_codegen] invalid problem: {err}");
            std::process::exit(1);
        }
    };

    if verbose {
        eprintln!(
            "[nsynth_codegen] solving {}: {} examples, lang={}",
            problem.name,
            problem.examples.len(),
            lang
        );
    }

    let t0 = std::time::Instant::now();
    let result = solve_problem(&problem);
    if !result.success {
        eprintln!(
            "[nsynth_codegen] synthesis failed for {} after {:.1}s",
            problem.name,
            t0.elapsed().as_secs_f32()
        );
        if let Some(err) = &result.error {
            eprintln!("[nsynth_codegen] synthesizer error: {err}");
        }
        std::process::exit(1);
    }
    if verbose {
        eprintln!(
            "[nsynth_codegen] ✓ solved via {} in {:.2}s",
            result.method,
            t0.elapsed().as_secs_f32()
        );
    }

    // Transpile. `mog` target passes through the raw synthesizer output.
    let output = match lang.as_str() {
        "python" | "py" => to_python(&result.code),
        "rust" | "rs" => to_rust(&result.code),
        "typescript" | "ts" => to_typescript(&result.code),
        "mog" => result.code.clone(),
        other => {
            eprintln!("[nsynth_codegen] unknown --lang {other:?}; use python|rust|typescript|mog");
            std::process::exit(2);
        }
    };

    match out_path {
        Some(path) => {
            let mut f = std::fs::File::create(&path).unwrap_or_else(|err| {
                eprintln!("[nsynth_codegen] cannot open {path}: {err}");
                std::process::exit(1);
            });
            let _ = f.write_all(output.as_bytes());
            if verbose {
                eprintln!("[nsynth_codegen] wrote {}", path);
            }
        }
        None => print!("{}", output),
    }
}
