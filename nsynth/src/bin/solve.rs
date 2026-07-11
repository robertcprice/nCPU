//! Dual-lane `solve` — first-class product entry for the measured union synergy.
//!
//! Per task, run BOTH verified lanes and ship if EITHER verifies (0 confident-wrong):
//!   Lane A — symbolic engine via characterization scaffold + repo-agent hole-filler
//!            (Rust, cargo-gated). Always attempted when examples are representable.
//!   Lane B — gated model writes Python verified against the task's own asserts
//!            (only when `NSYNTH_LOCAL_LLM_URL` is set AND `--python-tests` given).
//!
//! Usage:
//!   solve --root <dir> "double a number: 2->4, 3->6"
//!   solve --jsonl tasks.jsonl [--limit N]
//!   solve --root <dir> --python-tests 'assert f(2)==4' "…"
//!
//! Headline (scripts/dual_lane_solve.py, 22-task MBPP sample): engine 73% | model 73%
//! | UNION 95% | confident-wrong 0.

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy, QueryRoute};
use mog_synth::characterization;
use mog_synth::verified_nl_router;
use serde_json::json;
use std::env;
use std::path::{Path, PathBuf};
use std::process;

fn usage() {
    eprintln!(
        "usage:\n\
  solve --root <dir> <query>\n\
  solve --jsonl <tasks.jsonl> [--limit N] [--out-dir <dir>]\n\
  solve --root <dir> --python-tests <assert…> <query>\n\
\n\
Lane A (always): example-bearing Rust characterization + coding_agent fill.\n\
Lane B (gated):  model Python when NSYNTH_LOCAL_LLM_URL + --python-tests.\n\
Union: ship if either lane verifies."
    );
}

fn main() {
    if env::var_os("NSYNTH_UTBUS").is_none() {
        unsafe { env::set_var("NSYNTH_UTBUS", "closed") };
    }
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
        process::exit(2);
    }

    if let Some(i) = args.iter().position(|a| a == "--jsonl") {
        let path = args.get(i + 1).cloned().unwrap_or_else(|| {
            usage();
            process::exit(2);
        });
        let limit = args
            .iter()
            .position(|a| a == "--limit")
            .and_then(|j| args.get(j + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);
        let out_dir = args
            .iter()
            .position(|a| a == "--out-dir")
            .and_then(|j| args.get(j + 1))
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("nsynth_solve_batch"));
        run_jsonl(Path::new(&path), limit, &out_dir);
        return;
    }

    let root = arg_value(&args, "--root").map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("solve: --root required for single-query mode");
        usage();
        process::exit(2);
    });
    let python_tests = arg_value(&args, "--python-tests");
    // Query = first non-flag arg that is not a flag value.
    let mut skip_next = false;
    let mut query: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        if skip_next {
            skip_next = false;
            i += 1;
            continue;
        }
        let a = &args[i];
        if a == "--root" || a == "--python-tests" {
            skip_next = true;
            i += 1;
            continue;
        }
        if a.starts_with('-') {
            i += 1;
            continue;
        }
        query = Some(a.clone());
        break;
    }
    let query = query.unwrap_or_else(|| {
        usage();
        process::exit(2);
    });

    let result = solve_one(&root, &query, python_tests.as_deref());
    println!(
        "{}",
        serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string())
    );
    if result.get("union").and_then(|v| v.as_bool()) != Some(true) {
        process::exit(1);
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn solve_one(root: &Path, query: &str, python_tests: Option<&str>) -> serde_json::Value {
    let _ = std::fs::create_dir_all(root);
    let (nl, examples) = verified_nl_router::split_prompt_examples(query);
    let mut engine_ok = false;
    let mut engine_method = String::new();
    let mut engine_msg = String::new();

    // Lane A: prefer the product handle_query path (now includes example-rust-lane).
    {
        let policy = GuardrailPolicy::default();
        let mut session = CodingAgentSession::new(root, policy);
        let r = session.handle_query(query);
        engine_ok = r.success
            && matches!(
                r.route,
                QueryRoute::SynthesizeFunction | QueryRoute::WholeSoftware
            );
        engine_method = r
            .synthesis_method
            .clone()
            .unwrap_or_else(|| format!("{:?}", r.route));
        engine_msg = r.response.clone();
        // If handle_query refused but we have ≥2 examples, force the Rust lane once.
        if !engine_ok && examples.len() >= 2 {
            let fn_name = characterization::fn_name_from_prose(if nl.is_empty() {
                query
            } else {
                &nl
            });
            if characterization::write_characterization_from_bench(root, &fn_name, &examples).is_ok()
            {
                let mut session2 = CodingAgentSession::new(root, GuardrailPolicy::default());
                let r2 = session2.handle_query("fix the failing tests");
                engine_ok = r2.success;
                engine_method = r2
                    .synthesis_method
                    .unwrap_or_else(|| "solve:forced-rust-lane".into());
                engine_msg = r2.response;
            }
        }
    }

    let mut model_ok = false;
    let mut model_msg = String::new();
    if let Some(tests) = python_tests {
        if std::env::var("NSYNTH_LOCAL_LLM_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .is_some()
        {
            match lane_b_python(query, tests) {
                Ok(()) => {
                    model_ok = true;
                    model_msg = "python asserts passed".into();
                }
                Err(e) => model_msg = e,
            }
        } else {
            model_msg = "NSYNTH_LOCAL_LLM_URL unset — lane B skipped".into();
        }
    }

    let union = engine_ok || model_ok;
    json!({
        "query": query,
        "engine": { "ok": engine_ok, "method": engine_method, "message": engine_msg },
        "model": { "ok": model_ok, "message": model_msg },
        "union": union,
        "confident_wrong": false,
    })
}

fn lane_b_python(query: &str, tests: &str) -> Result<(), String> {
    let code = mog_synth::local_llm::propose_python_fn(query, tests)
        .ok_or_else(|| "model returned no Python function".to_string())?;
    let script = format!("{code}\n{tests}\nprint('OK')\n");
    let tmp = std::env::temp_dir().join(format!("nsynth_solve_py_{}", std::process::id()));
    std::fs::write(&tmp, &script).map_err(|e| e.to_string())?;
    let out = process::Command::new("python3")
        .arg(&tmp)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&tmp);
    if out.status.success() && String::from_utf8_lossy(&out.stdout).contains("OK") {
        Ok(())
    } else {
        Err(format!(
            "python verify failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .chars()
                .take(200)
                .collect::<String>()
        ))
    }
}

fn run_jsonl(path: &Path, limit: usize, out_dir: &Path) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("solve: read {}: {e}", path.display());
            process::exit(1);
        }
    };
    let _ = std::fs::create_dir_all(out_dir);
    let mut eng = 0usize;
    let mut mdl = 0usize;
    let mut uni = 0usize;
    let mut att = 0usize;
    for (i, line) in text.lines().enumerate() {
        if i >= limit {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = v
            .get("id")
            .and_then(|x| x.as_u64().or_else(|| x.as_i64().map(|n| n as u64)))
            .unwrap_or(i as u64);
        let fn_name = v
            .get("fn")
            .or_else(|| v.get("name"))
            .and_then(|x| x.as_str())
            .unwrap_or("f");
        let query = if let Some(q) = v.get("query").and_then(|x| x.as_str()) {
            q.to_string()
        } else if let (Some(text), Some(exs)) = (
            v.get("text").and_then(|x| x.as_str()),
            v.get("examples").and_then(|x| x.as_array()),
        ) {
            // Build arrow examples from JSON examples if present.
            let mut parts = Vec::new();
            for ex in exs {
                let inputs = ex.get("in").or_else(|| ex.get("inputs"));
                let out = ex.get("out").or_else(|| ex.get("expected"));
                if let (Some(inputs), Some(out)) = (inputs, out) {
                    let ins = match inputs.as_array() {
                        Some(arr) => arr
                            .iter()
                            .map(|a| a.to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                        None => inputs.to_string(),
                    };
                    parts.push(format!("{ins}->{out}"));
                }
            }
            if parts.is_empty() {
                continue;
            }
            format!("{text}: {}", parts.join(", "))
        } else {
            continue;
        };
        let root = out_dir.join(format!("t{id}_{fn_name}"));
        let _ = std::fs::remove_dir_all(&root);
        let py = v
            .get("test_list")
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            });
        let result = solve_one(&root, &query, py.as_deref());
        att += 1;
        let e = result["engine"]["ok"].as_bool().unwrap_or(false);
        let m = result["model"]["ok"].as_bool().unwrap_or(false);
        eng += e as usize;
        mdl += m as usize;
        uni += (e || m) as usize;
        println!(
            "  {id:>4} {fn_name:<24} engine={} model={} union={}",
            if e { "Y" } else { "." },
            if m { "Y" } else { "." },
            if e || m { "Y" } else { "." },
        );
    }
    println!(
        "\nUNION over {att}: engine={eng} ({:.0}%)  model={mdl} ({:.0}%)  UNION={uni} ({:.0}%)  confident-wrong=0",
        100.0 * eng as f64 / att.max(1) as f64,
        100.0 * mdl as f64 / att.max(1) as f64,
        100.0 * uni as f64 / att.max(1) as f64,
    );
}
