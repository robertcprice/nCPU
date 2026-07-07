//! CLI: universal coding agent — any NL query, full secure tools, session resume.

use mog_synth::agent::{
    AgentQueryResult, CodingAgentSession, GuardrailPolicy, QueryRoute, ToolCall,
};
use std::env;
use std::path::PathBuf;
use std::process;

fn usage() {
    eprintln!(
        "usage: coding_agent --root <repo_path> [options] [query]\n\
\n\
Commands (default: query):\n\
  query <text>              Run NL query through registry workflow router\n\
  --tools                   List allowed secure tool capabilities\n\
  --capabilities            Emit runtime engine capability introspection (JSON)\n\
  --tool <name> <action>    Direct tool invoke (e.g. --tool fs read path=src/lib.rs)\n\
  --clarify <answer>        Answer pending clarification for --session\n\
\n\
Options:\n\
  --root <path>             Repository / sandbox root (required)\n\
  --session <id>            Session name for persist/resume (default: main)\n\
  --json                    Emit JSON result\n\
  --emit <lang>             Also transpile the solved function to a device-native\n\
                            language (python|rust|typescript|go|java) — the\n\
                            toolchain-free export path (no rustc; runs on Pi/phone)\n\
  --allow-http <host>       Add HTTP host allowlist entry\n\
\n\
Examples:\n\
  coding_agent --root . query \"add two numbers\"\n\
  coding_agent --root . --emit python query \"double a number: 2->4, 3->6\"\n\
  coding_agent --root . --tools\n\
  coding_agent --root . --tool fs list path=src\n\
  coding_agent --root . --session dev query \"fix failing tests\"\n\
  coding_agent --root . --session dev --clarify \"multiply two integers\""
    );
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
        process::exit(2);
    }

    let root = arg_value(&args, "--root").map(PathBuf::from);
    let session_id = arg_value(&args, "--session").unwrap_or_else(|| "main".to_string());
    let json_out = args.iter().any(|a| a == "--json");
    let emit_lang = arg_value(&args, "--emit");
    let list_tools = args.iter().any(|a| a == "--tools");
    let capabilities = args.iter().any(|a| a == "--capabilities");
    let clarify = arg_value(&args, "--clarify");

    let Some(root) = root else {
        usage();
        process::exit(2);
    };
    if !root.is_dir() {
        eprintln!("error: --root is not a directory: {}", root.display());
        process::exit(2);
    }

    let mut policy = GuardrailPolicy::default();
    for (i, arg) in args.iter().enumerate() {
        if arg == "--allow-http" {
            if let Some(host) = args.get(i + 1) {
                policy = policy.with_allowed_http_host(host);
            }
        }
    }

    let mut session =
        CodingAgentSession::load(&root, policy.clone(), &session_id).unwrap_or_else(|e| {
            eprintln!("error: load session: {e}");
            process::exit(2);
        });

    if list_tools {
        let caps = session.allowed_tool_capabilities();
        if json_out {
            println!(
                "{}",
                serde_json::to_string_pretty(&caps).unwrap_or_default()
            );
        } else {
            for cap in caps {
                println!("{cap}");
            }
        }
        return;
    }

    if capabilities {
        let doc = mog_synth::agent_introspect::engine_capabilities_json(&root, policy);
        if json_out {
            println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
        } else {
            println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
        }
        return;
    }

    if let Some(tool_idx) = args.iter().position(|a| a == "--tool") {
        let tool = args.get(tool_idx + 1).cloned().unwrap_or_default();
        let action = args.get(tool_idx + 2).cloned().unwrap_or_default();
        if tool.is_empty() || action.is_empty() {
            eprintln!("error: --tool requires <name> <action> [key=value ...]");
            process::exit(2);
        }
        let mut call = ToolCall::new(action);
        for arg in args.iter().skip(tool_idx + 3) {
            if let Some((k, v)) = arg.split_once('=') {
                call = call.arg(k, v);
            }
        }
        match session.invoke_tool(&tool, &call) {
            Ok(out) => {
                if json_out {
                    let body = serde_json::json!({
                        "tool": tool,
                        "content": out.content,
                        "metadata": out.metadata,
                    });
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&body).unwrap_or_default()
                    );
                } else {
                    print!("{}", out.content);
                    if !out.content.ends_with('\n') {
                        println!();
                    }
                }
            }
            Err(error) => {
                eprintln!("tool error: {error}");
                process::exit(1);
            }
        }
        return;
    }

    if let Some(answer) = clarify {
        if !session.has_pending_clarification() {
            eprintln!("error: no pending clarification for session '{session_id}'");
            process::exit(2);
        }
        let result = session.clarify_and_continue(&answer).unwrap_or_else(|e| {
            eprintln!("error: clarify: {e}");
            process::exit(1);
        });
        emit_result(&result, json_out);
        if !result.success && result.route != QueryRoute::Clarification {
            process::exit(1);
        }
        return;
    }

    // Indices that are the VALUE of a known value-flag (e.g. `python` after
    // `--emit`, the path after `--root`) so they are never mistaken for query
    // tokens in the bare-query form.
    let value_flags = ["--root", "--session", "--allow-http", "--clarify", "--emit"];
    let skip_idx: std::collections::HashSet<usize> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| value_flags.contains(&a.as_str()))
        .map(|(i, _)| i + 1)
        .collect();

    let query = if let Some(pos) = args.iter().position(|a| a == "query") {
        args.iter()
            .enumerate()
            .skip(pos + 1)
            .filter(|(i, a)| !skip_idx.contains(i) && !a.starts_with("--"))
            .map(|(_, s)| s.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        args.iter()
            .enumerate()
            .skip(1)
            .filter(|(i, a)| !skip_idx.contains(i) && !a.starts_with("--"))
            .filter(|(_, a)| {
                !matches!(
                    a.as_str(),
                    "query" | "read" | "write" | "list" | "get" | "post" | "run"
                )
            })
            .map(|(_, s)| s.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };

    if query.trim().is_empty() {
        usage();
        process::exit(2);
    }

    // PURE-EXAMPLE FAST PATH: a spec that is just `name: a->b, c->d` (no natural
    // language) is the natural on-device input — solve it directly through the
    // verified synthesizer instead of dead-ending at the NL classifier's
    // "what kind of task is this?" clarification. Falls through to the NL front
    // door on a refusal.
    if let Some((name, examples)) = parse_example_spec(query.trim()) {
        if let Some(code) = solve_example_spec(&name, &examples) {
            if json_out {
                let value = serde_json::json!({
                    "route": "SynthesizeFunction",
                    "success": true,
                    "workflow": "pure_example_pbe",
                    "response": code,
                });
                println!("{}", serde_json::to_string_pretty(&value).unwrap_or_default());
            } else {
                println!("route: SynthesizeFunction");
                println!("workflow: pure_example_pbe");
                println!("success: true");
                println!("---");
                println!("{code}");
            }
            if let Some(lang) = &emit_lang {
                emit_source_from_code(&code, lang);
            }
            return;
        }
    }

    let result = session.handle_query(query.trim());
    emit_result(&result, json_out);
    if let Some(lang) = &emit_lang {
        emit_target_source(&result, lang);
    }
    if !result.success && result.route != QueryRoute::Clarification {
        process::exit(1);
    }
}

/// Transpile the solved function to a device-native language — the toolchain-free
/// export path for edge deployment. The core solve+verify runs on the device with
/// no compiler (Mog is interpreter-verified); this hands back runnable source in a
/// language the Pi/phone already has (Python/JS/Go/Java), so the verified logic can
/// drop straight into the device's own stack without shipping a Rust toolchain.
fn emit_target_source(result: &AgentQueryResult, lang: &str) {
    if !result.success {
        eprintln!("--emit: nothing to emit (query did not solve)");
        return;
    }
    // The verified Mog function is embedded in the response; pull it out with the
    // brace-matched extractor (handles the `fn NAME(..) -> RET { .. }` block).
    match mog_synth::doc_ingest::extract_rust_fn_sources(&result.response)
        .into_iter()
        .next()
    {
        Some((_name, src)) => emit_source_from_code(&src, lang),
        None => eprintln!("--emit: no function found in the result to transpile"),
    }
}

/// Transpile a verified Mog function to a device-native language and print it.
/// Shared by `--emit` (from an NL result) and the pure-example fast path.
fn emit_source_from_code(code: &str, lang: &str) {
    use mog_synth::mog_transpile as tp;
    let out = match lang.to_ascii_lowercase().as_str() {
        "python" | "py" => tp::to_python(code),
        "rust" | "rs" => tp::to_rust(code),
        "typescript" | "ts" => tp::to_typescript(code),
        "go" => tp::to_go(code),
        "java" => tp::to_java(code),
        other => {
            eprintln!("--emit: unknown language '{other}' (python|rust|typescript|go|java)");
            return;
        }
    };
    println!("--- emit:{} ---", lang.to_ascii_lowercase());
    println!("{out}");
}

/// Parse a PURE example-list spec — `f: 0->7, 1->10, 2->17`, `add: 2,3->5, 10,4->14`,
/// or bare `2->4, 3->6, 5->10` — into (name, examples). Returns None the moment the
/// query carries natural language (anything beyond an optional leading `name:` and
/// numeric pairs), so real NL still goes to the front door. Integers before an arrow
/// are that pair's inputs; the first integer after it is the output; any remaining
/// integers belong to the next pair.
fn parse_example_spec(query: &str) -> Option<(String, Vec<mog_synth::benchmark::Example>)> {
    use mog_synth::benchmark::{Example, Value};
    let q = query.trim();
    if !q.contains("->") {
        return None;
    }
    let is_ident = |s: &str| -> bool {
        let mut chars = s.chars();
        matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
            && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    let (name, body) = match q.split_once(':') {
        Some((head, rest)) if is_ident(head.trim()) => (head.trim().to_string(), rest.trim()),
        _ => ("f".to_string(), q),
    };
    // A pure spec contains only digits, separators, and the arrow — no NL.
    if !body
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '-' | '>' | ',' | ' ' | '\t'))
    {
        return None;
    }
    let parse_ints = |s: &str| -> Vec<i64> {
        s.split(',').filter_map(|t| t.trim().parse::<i64>().ok()).collect()
    };
    let parts: Vec<&str> = body.split("->").collect();
    if parts.len() < 3 {
        return None; // need >= 2 pairs (arrows), not a single point
    }
    let mut examples = Vec::new();
    let mut pending = parse_ints(parts[0]);
    for mid in &parts[1..parts.len() - 1] {
        let ints = parse_ints(mid);
        if ints.is_empty() || pending.is_empty() {
            return None;
        }
        examples.push(Example {
            inputs: pending.iter().map(|&v| Value::Int(v)).collect(),
            expected: Value::Int(ints[0]),
        });
        pending = ints[1..].to_vec();
    }
    let last = parse_ints(parts[parts.len() - 1]);
    if last.len() != 1 || pending.is_empty() {
        return None;
    }
    examples.push(Example {
        inputs: pending.iter().map(|&v| Value::Int(v)).collect(),
        expected: Value::Int(last[0]),
    });
    let arity = examples[0].inputs.len();
    if examples.iter().any(|e| e.inputs.len() != arity) {
        return None;
    }
    Some((name, examples))
}

/// Solve a pure-example spec directly through the verified synthesizer, bypassing
/// the NL classifier. Strict-verifies before returning — never emits an unproven
/// program. Returns the verified Mog code, or None on refusal.
fn solve_example_spec(name: &str, examples: &[mog_synth::benchmark::Example]) -> Option<String> {
    let sig: &'static str = Box::leak(
        mog_synth::linguigenesis_bridge::infer_signature(name, examples).into_boxed_str(),
    );
    let problem = mog_synth::benchmark::Problem {
        name: name.to_string(),
        category: "cli-pbe",
        description: "",
        signature: sig,
        examples: examples.to_vec(),
        ..Default::default()
    };
    let res = mog_synth::solver::solve_problem(&problem);
    if !res.success {
        return None;
    }
    if mog_synth::runtime::verify_problem_code_strict(&problem, &res.code).is_err() {
        return None;
    }
    // Same memorization guard as the teach path: refuse a fit whose magic constants
    // rival the data they explain (the examples underdetermine the op). Keeps the
    // CLI fast path from emitting an uncorroborated memorized closed form.
    if mog_synth::synth_confidence::is_memorization_overfit(&res.code, examples) {
        return None;
    }
    Some(res.code)
}

fn emit_result(result: &AgentQueryResult, json_out: bool) {
    if json_out {
        let value = serde_json::json!({
            "route": format!("{:?}", result.route),
            "success": result.success,
            "workflow": result.workflow,
            "response": result.response,
            "clarification_questions": result.clarification_questions,
            "synthesis_method": result.synthesis_method,
            "tool_trace": result.tool_trace,
            "repo_result": result.repo_result,
            "explanation": explain_solution(result),
            "security": security_report(result),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value).unwrap_or_default()
        );
        return;
    }
    println!("route: {:?}", result.route);
    println!("workflow: {}", result.workflow);
    println!("success: {}", result.success);
    if let Some(method) = &result.synthesis_method {
        println!("synthesis: {}", method);
    }
    if !result.clarification_questions.is_empty() {
        println!("clarifications:");
        for q in &result.clarification_questions {
            println!("  - {q}");
        }
    }
    println!("---");
    println!("{}", result.response);
    if let Some(explanation) = explain_solution(result) {
        println!("---");
        println!("explanation: {explanation}");
    }
    if let Some(sec) = security_report(result) {
        println!("---");
        println!("security: {sec}");
    }
}

/// Vulnerability scan of the synthesized solution — wires the security/ scanner.
/// Reports ONLY when findings exist (no noise on clean code). Best-effort.
fn security_report(result: &AgentQueryResult) -> Option<String> {
    if !result.success || result.synthesis_method.is_none() {
        return None;
    }
    let scan = mog_synth::security::scan_vulnerabilities(&result.response, "synthesized");
    if scan.findings.is_empty() {
        return None;
    }
    let crit = if scan.has_critical() {
        " (CRITICAL present)"
    } else {
        ""
    };
    Some(format!("{} potential issue(s){crit}", scan.findings.len()))
}

/// Natural-language explanation of a synthesized solution — wires the
/// bidirectional code→NL pipeline (parse → semantics → NL). Best-effort: only for
/// a successful synthesis whose response IS the code; skipped if it can't parse.
fn explain_solution(result: &AgentQueryResult) -> Option<String> {
    if !result.success || result.synthesis_method.is_none() {
        return None;
    }
    let nl = mog_synth::bidirectional::code_to_nl(&result.response).ok()?;
    let nl = nl.trim();
    if nl.is_empty() {
        None
    } else {
        Some(nl.to_string())
    }
}
