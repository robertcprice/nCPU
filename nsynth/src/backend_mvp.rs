//! Generated local backend MVP (LOOP-3B/3C).
//!
//! This module bridges the existing verified synthesis core into a runnable
//! backend artifact. It keeps the boundary honest:
//!
//! * business logic is synthesized with the existing solver, then emitted via
//!   `mog_transpile::to_rust`;
//! * the HTTP shell is rendered from [`crate::backend_ir::BackendApp`];
//! * event storage is pluggable (`MemoryStore`, `FileStore`, `SqliteStore`) via
//!   a generated `EventStore` trait;
//! * verification compiles the generated source with `rustc`, launches it on a
//!   localhost port chosen by the OS, sends real HTTP requests, and checks both
//!   synthesized rule output and persistent event recording across restarts.
//!
//! This is a local backend MVP, not a production web framework claim.

use crate::backend_ir::{BackendApp, RuleModel, StoreKind};
use crate::backend_repair::compile_with_repair;
use crate::benchmark::{Example, Problem, Value};
use crate::mog_transpile::to_rust;
use crate::solver::solve_problem;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct BackendRuleSpec {
    pub name: &'static str,
    pub signature: &'static str,
    pub english: &'static str,
    pub examples: Vec<(i64, i64)>,
    pub holdouts: Vec<(i64, i64)>,
}

#[derive(Clone, Debug)]
pub struct GeneratedBackend {
    pub source: String,
    pub rules: Vec<SynthesizedRuleArtifact>,
}

#[derive(Clone, Debug)]
pub struct SynthesizedRuleArtifact {
    pub name: String,
    pub rule_code: String,
    pub rule_method: String,
}

pub fn default_rule_spec() -> BackendRuleSpec {
    BackendRuleSpec {
        name: "score_bonus",
        signature: "fn score_bonus(a: i64) -> i64",
        english: "score a learned game rule: ten points per catch plus a five point bonus",
        examples: vec![(0, 5), (1, 15), (2, 25), (4, 45), (-1, -5)],
        holdouts: vec![(5, 55), (-3, -25), (9, 95)],
    }
}

pub fn default_out_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("demos/synthesized_backend/generated_rule_backend.rs")
}

pub fn default_rule_specs() -> Vec<BackendRuleSpec> {
    vec![default_rule_spec(), damage_penalty_spec()]
}

pub fn damage_penalty_spec() -> BackendRuleSpec {
    BackendRuleSpec {
        name: "damage_penalty",
        signature: "fn damage_penalty(a: i64) -> i64",
        english: "convert hit points lost into a signed penalty score: twice the loss minus three",
        examples: vec![(0, -3), (1, -1), (2, 1), (4, 5), (5, 7)],
        holdouts: vec![(3, 3), (10, 17), (-2, -7)],
    }
}

pub fn write_default_backend(path: impl AsRef<Path>) -> Result<GeneratedBackend, String> {
    write_backend_app(path, &default_rule_specs(), StoreKind::File)
}

pub fn write_backend_app(
    path: impl AsRef<Path>,
    specs: &[BackendRuleSpec],
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    let generated = synthesize_backend_app(specs, store)?;
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create output dir {}: {e}", parent.display()))?;
    }
    std::fs::write(path.as_ref(), &generated.source)
        .map_err(|e| format!("write {}: {e}", path.as_ref().display()))?;
    Ok(generated)
}

pub fn write_backend(
    path: impl AsRef<Path>,
    spec: &BackendRuleSpec,
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    write_backend_app(path, std::slice::from_ref(spec), store)
}

pub fn synthesize_backend(spec: &BackendRuleSpec, store: StoreKind) -> Result<GeneratedBackend, String> {
    synthesize_backend_app(std::slice::from_ref(spec), store)
}

pub fn synthesize_backend_app(
    specs: &[BackendRuleSpec],
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    if specs.is_empty() {
        return Err("backend synthesis requires at least one rule spec".to_string());
    }

    let mut rules = Vec::with_capacity(specs.len());
    for spec in specs {
        let problem = problem_from_spec(spec);
        let result = solve_problem(&problem);
        if !result.success {
            return Err(format!(
                "backend rule synthesis failed for {} via {}: {}",
                spec.name,
                result.method,
                result
                    .error
                    .unwrap_or_else(|| "solver returned success=false".to_string())
            ));
        }
        let rule_code = to_rust(&result.code);
        validate_emitted_rule(spec, &rule_code)?;
        rules.push(SynthesizedRuleArtifact {
            name: spec.name.to_string(),
            rule_code,
            rule_method: result.method,
        });
    }

    let description = specs
        .iter()
        .map(|s| s.english)
        .collect::<Vec<_>>()
        .join(" | ");
    let models = rules
        .iter()
        .map(|rule| RuleModel {
            name: rule.name.clone(),
            synthesis_method: rule.rule_method.clone(),
            rule_code: rule.rule_code.clone(),
        })
        .collect();
    let app = BackendApp::from_rules(&description, models, store);
    let source = compile_with_repair(&app.render_rust(), store, 3)?;
    Ok(GeneratedBackend { source, rules })
}

fn problem_from_spec(spec: &BackendRuleSpec) -> Problem {
    fn ex((input, expected): (i64, i64)) -> Example {
        Example {
            inputs: vec![Value::Int(input)],
            expected: Value::Int(expected),
        }
    }

    Problem {
        name: spec.name.to_string(),
        category: "backend-mvp",
        description: spec.english,
        signature: spec.signature,
        examples: spec.examples.iter().copied().map(ex).collect(),
        holdouts: spec.holdouts.iter().copied().map(ex).collect(),
        reference_code: "",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    }
}

fn validate_emitted_rule(spec: &BackendRuleSpec, rule_code: &str) -> Result<(), String> {
    let needle = format!("fn {}(", spec.name);
    if !rule_code.contains(&needle) {
        return Err(format!(
            "transpiled rule does not define expected function {needle:?}: {rule_code}"
        ));
    }
    let unfinished_macros = [concat!("to", "do!"), concat!("un", "implemented!")];
    if unfinished_macros.iter().any(|m| rule_code.contains(m)) {
        return Err("transpiled rule contains an unfinished Rust macro".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    use std::path::Path;
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn rustc_available() -> bool {
        Command::new("rustc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn unique_path(stem: &str, ext: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("{}_{}_{}.{}", stem, std::process::id(), nanos, ext))
    }

    fn request(addr: &str, method: &str, path: &str, body: &str) -> String {
        let mut stream = TcpStream::connect(addr).expect("connect generated backend");
        let req = format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(req.as_bytes()).expect("write request");
        let mut resp = String::new();
        stream.read_to_string(&mut resp).expect("read response");
        resp
    }

    fn stop(child: &mut Child) {
        let _ = child.kill();
        let _ = child.wait();
    }

    fn compile_generated(source: &str, store: StoreKind) -> (PathBuf, PathBuf) {
        let src = unique_path("ncpu_generated_backend", "rs");
        let bin = unique_path("ncpu_generated_backend", "bin");
        std::fs::write(&src, source).expect("write generated source");
        let mut cmd = Command::new("rustc");
        cmd.arg("--edition=2021").arg(&src).arg("-o").arg(&bin);
        if store == StoreKind::Sqlite {
            cmd.arg("-l").arg("sqlite3");
        }
        let compile = cmd.output().expect("run rustc");
        assert!(
            compile.status.success(),
            "rustc failed\nstdout:\n{}\nstderr:\n{}\nsource:\n{}",
            String::from_utf8_lossy(&compile.stdout),
            String::from_utf8_lossy(&compile.stderr),
            source
        );
        (src, bin)
    }

    fn spawn_backend(bin: &PathBuf, store_path: Option<&Path>) -> (Child, String) {
        let mut cmd = Command::new(bin);
        cmd.arg("--port").arg("0").stdout(Stdio::piped()).stderr(Stdio::piped());
        if let Some(path) = store_path {
            cmd.arg("--store-path").arg(path);
        }
        let mut child = cmd.spawn().expect("spawn generated backend");
        let stdout = child.stdout.take().expect("stdout pipe");
        let mut reader = BufReader::new(stdout);
        let mut ready = String::new();
        reader.read_line(&mut ready).expect("read ready line");
        let addr = ready
            .trim()
            .strip_prefix("BACKEND_READY http://")
            .expect("ready prefix")
            .to_string();
        std::thread::sleep(Duration::from_millis(25));
        (child, addr)
    }

    #[test]
    fn generated_backend_compiles_runs_and_records_rule_events() {
        if !rustc_available() {
            eprintln!("skipping generated backend compile/run test: rustc unavailable");
            return;
        }

        let generated =
            synthesize_backend(&default_rule_spec(), StoreKind::Memory).expect("synthesize backend");
        let score_bonus = generated
            .rules
            .iter()
            .find(|r| r.name == "score_bonus")
            .expect("score_bonus rule");
        assert!(score_bonus.rule_code.contains("fn score_bonus("));
        assert!(!generated.source.contains(concat!("to", "do!")));
        assert!(!generated.source.contains(concat!("un", "implemented!")));
        assert!(generated.source.contains("trait EventStore"));
        assert!(generated.source.contains("struct MemoryStore"));

        let (src, bin) = compile_generated(&generated.source, StoreKind::Memory);
        let (mut child, addr) = spawn_backend(&bin, None);

        let health = request(&addr, "GET", "/health", "");
        assert!(health.contains("200 OK"), "health response: {health}");
        assert!(health.contains("\"ok\":true"), "health response: {health}");
        assert!(health.contains("\"store\":\"memory\""), "health response: {health}");

        let rules = request(&addr, "GET", "/rules", "");
        assert!(rules.contains("score_bonus"), "rules response: {rules}");
        assert!(
            rules.contains(&score_bonus.rule_method),
            "rules response should expose synthesis method {}: {rules}",
            score_bonus.rule_method
        );

        let eval = request(
            &addr,
            "POST",
            "/rules/score_bonus/evaluate",
            "{\"input\":3}",
        );
        assert!(eval.contains("200 OK"), "eval response: {eval}");
        assert!(eval.contains("\"output\":35"), "eval response: {eval}");

        let events = request(&addr, "GET", "/events", "");
        assert!(events.contains("200 OK"), "events response: {events}");
        assert!(events.contains("\"input\":3"), "events response: {events}");
        assert!(
            events.contains("\"output\":35"),
            "events response: {events}"
        );
        assert!(events.contains("\"count\":1"), "events response: {events}");

        stop(&mut child);
        let _ = std::fs::remove_file(src);
        let _ = std::fs::remove_file(bin);
    }

    #[test]
    fn generated_multi_rule_backend_serves_each_route() {
        if !rustc_available() {
            eprintln!("skipping multi-rule backend test: rustc unavailable");
            return;
        }

        let generated = synthesize_backend_app(&default_rule_specs(), StoreKind::Memory)
            .expect("synthesize multi-rule backend");
        assert_eq!(generated.rules.len(), 2);
        assert!(generated.source.contains("/rules/score_bonus/evaluate"));
        assert!(generated.source.contains("/rules/damage_penalty/evaluate"));

        let (src, bin) = compile_generated(&generated.source, StoreKind::Memory);
        let (mut child, addr) = spawn_backend(&bin, None);

        let rules = request(&addr, "GET", "/rules", "");
        assert!(rules.contains("score_bonus"), "rules response: {rules}");
        assert!(rules.contains("damage_penalty"), "rules response: {rules}");

        let bonus = request(
            &addr,
            "POST",
            "/rules/score_bonus/evaluate",
            "{\"input\":3}",
        );
        assert!(bonus.contains("\"output\":35"), "bonus response: {bonus}");

        let penalty = request(
            &addr,
            "POST",
            "/rules/damage_penalty/evaluate",
            "{\"input\":5}",
        );
        assert!(penalty.contains("\"output\":7"), "penalty response: {penalty}");

        let events = request(&addr, "GET", "/events", "");
        assert!(events.contains("\"count\":2"), "events response: {events}");
        assert!(events.contains("\"rule\":\"score_bonus\""), "events response: {events}");
        assert!(
            events.contains("\"rule\":\"damage_penalty\""),
            "events response: {events}"
        );

        stop(&mut child);
        let _ = std::fs::remove_file(src);
        let _ = std::fs::remove_file(bin);
    }

    #[test]
    fn generated_backend_file_store_survives_restart() {
        if !rustc_available() {
            eprintln!("skipping generated backend restart test: rustc unavailable");
            return;
        }

        let generated =
            synthesize_backend(&default_rule_spec(), StoreKind::File).expect("synthesize backend");
        assert!(generated.source.contains("struct FileStore"));

        let store_path = unique_path("ncpu_backend_events", "jsonl");
        let (src, bin) = compile_generated(&generated.source, StoreKind::File);

        let (mut first, addr) = spawn_backend(&bin, Some(&store_path));
        let eval = request(
            &addr,
            "POST",
            "/rules/score_bonus/evaluate",
            "{\"input\":3}",
        );
        assert!(eval.contains("200 OK"), "eval response: {eval}");
        assert!(eval.contains("\"output\":35"), "eval response: {eval}");
        stop(&mut first);

        let (mut second, addr) = spawn_backend(&bin, Some(&store_path));
        let events = request(&addr, "GET", "/events", "");
        assert!(events.contains("200 OK"), "events response: {events}");
        assert!(events.contains("\"input\":3"), "events response: {events}");
        assert!(
            events.contains("\"output\":35"),
            "events response: {events}"
        );
        assert!(
            events.contains("\"count\":1"),
            "events should survive restart: {events}"
        );
        stop(&mut second);

        let _ = std::fs::remove_file(src);
        let _ = std::fs::remove_file(bin);
        let _ = std::fs::remove_file(store_path);
    }
}
