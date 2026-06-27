//! NL-driven backend spec intake (LOOP-4).
//!
//! Mirrors the proven `build_game_nl` door: free-text English with named
//! function clauses and inline `name(x)=y` examples flows through the real
//! `LinguigenesisBridge::synthesize_project` path, then verified i64 rules are
//! emitted into the BackendIR HTTP artifact.

use crate::backend_http::{cleanup_temp_artifacts, compile_to_temp_bin, verify_backend_http, HttpRuleCheck};
use crate::backend_ir::{BackendApp, RuleModel, StoreKind};
use crate::benchmark::Value as BValue;
use crate::backend_mvp::{GeneratedBackend, SynthesizedRuleArtifact};
use crate::backend_repair::compile_with_repair;
use crate::linguigenesis_bridge::LinguigenesisBridge;
use crate::mog_transpile::to_rust;
use crate::runtime::{execute_function, Value as RValue};
use crate::solver::SolveResult;
use linguigenesis_core::coding_requirements::LiteralValue;
use linguigenesis_core::inline_examples::parse_inline_examples;
use std::path::Path;

/// Default English backend contract: two i64 business rules with inline examples.
/// No rule bodies appear here — bodies are synthesized through the NL door.
pub const DEFAULT_BACKEND_ENGLISH: &str = "\
A function score_bonus that scores ten points per catch plus a five point bonus, \
score_bonus(0)=5 and score_bonus(1)=15 and score_bonus(2)=25 and score_bonus(4)=45 and score_bonus(-1)=-5. \
A function damage_penalty that converts hit points lost into a signed penalty score twice the loss minus three, \
damage_penalty(0)=-3 and damage_penalty(1)=-1 and damage_penalty(2)=1 and damage_penalty(4)=5 and damage_penalty(5)=7.";

pub fn default_required_rule_names() -> &'static [&'static str] {
    &["score_bonus", "damage_penalty"]
}

pub fn write_backend_from_english(
    path: impl AsRef<Path>,
    english: &str,
    required: &[&str],
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    let generated = build_backend_from_english(english, required, store)?;
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create output dir {}: {e}", parent.display()))?;
    }
    std::fs::write(path.as_ref(), &generated.source)
        .map_err(|e| format!("write {}: {e}", path.as_ref().display()))?;
    Ok(generated)
}

pub fn build_backend_from_english(
    english: &str,
    required: &[&str],
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    if required.is_empty() {
        return Err("NL backend build requires at least one required rule name".to_string());
    }

    let bridge = LinguigenesisBridge::new();
    if let Some(err) = bridge.registry_load_error() {
        return Err(format!("NL registry failed to load: {err}"));
    }

    let (solved, skipped) = bridge
        .synthesize_project(english)
        .map_err(|e| format!("synthesize_project failed: {e}"))?;

    let by_name: std::collections::HashMap<&str, &SolveResult> =
        solved.iter().map(|(n, r)| (n.as_str(), r)).collect();

    let mut rules = Vec::with_capacity(required.len());
    for name in required {
        let res = by_name.get(name).copied().ok_or_else(|| {
            format!(
                "rule '{name}' was NOT synthesized (missing or skipped). skipped={skipped:?}"
            )
        })?;
        if !res.success {
            return Err(format!(
                "rule '{name}' did not synthesize: {}",
                res.error.clone().unwrap_or_else(|| "no solution".into())
            ));
        }
        if !is_i64_scalar_rule(&res.code) {
            return Err(format!(
                "rule '{name}' is not a scalar i64 rule — backend artifact only ships i64 Rust handlers.\n  mog: {}",
                res.code.lines().next().unwrap_or("").trim()
            ));
        }

        let examples = examples_for_rule_in_text(english, name);
        if examples.is_empty() {
            return Err(format!(
                "rule '{name}' has no inline i64 examples in the English contract"
            ));
        }
        verify_mog_i64_examples(&res.code, name, &examples)?;

        let rule_code = to_rust(&res.code);
        if !rule_code.contains(&format!("fn {name}(")) {
            return Err(format!(
                "transpiled Rust for '{name}' does not define fn {name}(...): {rule_code}"
            ));
        }

        rules.push(SynthesizedRuleArtifact {
            name: (*name).to_string(),
            rule_code,
            rule_method: res.method.clone(),
        });
    }

    let description = english
        .split('.')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .take(3)
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
    verify_built_backend_http(english, &source, &rules, store)?;
    Ok(GeneratedBackend { source, rules })
}

fn verify_built_backend_http(
    english: &str,
    source: &str,
    rules: &[SynthesizedRuleArtifact],
    store: StoreKind,
) -> Result<(), String> {
    if !rustc_available() {
        return Ok(());
    }
    let (src, bin) = compile_to_temp_bin(source, store == StoreKind::Sqlite)?;
    let checks: Vec<HttpRuleCheck> = rules
        .iter()
        .filter_map(|rule| {
            let examples = examples_for_rule_in_text(english, &rule.name);
            examples.first().map(|(input, output)| HttpRuleCheck {
                rule: rule.name.clone(),
                input: *input,
                output: *output,
            })
        })
        .collect();
    if checks.is_empty() {
        cleanup_temp_artifacts(&src, &bin);
        return Ok(());
    }
    let result = verify_backend_http(&bin, &checks, 2);
    cleanup_temp_artifacts(&src, &bin);
    result
}

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn split_function_clauses(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let marker = "a function ";
    let mut starts = Vec::new();
    for (idx, _) in lower.match_indices(marker) {
        starts.push(idx);
    }
    if starts.is_empty() {
        return vec![text.trim().to_string()];
    }
    let mut clauses = Vec::new();
    for (i, &start) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(text.len());
        clauses.push(text[start..end].trim().to_string());
    }
    clauses
}

pub fn examples_for_rule_in_text(text: &str, rule_name: &str) -> Vec<(i64, i64)> {
    let mut out = examples_from_clauses(text, rule_name);
    if out.len() < 3 {
        out = examples_from_call_sites(text, rule_name);
    }
    dedup_examples(&mut out);
    out
}

fn examples_from_clauses(text: &str, rule_name: &str) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for clause in split_function_clauses(text) {
        let mentions_rule = clause.to_lowercase().contains(&format!("function {rule_name}"))
            || clause.contains(&format!("{rule_name}("));
        if !mentions_rule {
            continue;
        }
        out.extend(parse_i64_examples(&clause));
    }
    out
}

fn examples_from_call_sites(text: &str, rule_name: &str) -> Vec<(i64, i64)> {
    let needle = format!("{rule_name}(");
    let mut out = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find(&needle) {
        let start = search_from + rel;
        let end = (start + needle.len() + 48).min(text.len());
        out.extend(parse_i64_examples(&text[start..end]));
        search_from = start + needle.len();
    }
    out
}

fn parse_i64_examples(segment: &str) -> Vec<(i64, i64)> {
    let set = parse_inline_examples(segment);
    let mut out = Vec::new();
    for ex in set.examples {
        if ex.inputs.len() != 1 {
            continue;
        }
        let (LiteralValue::Int(x), LiteralValue::Int(y)) = (&ex.inputs[0], &ex.expected) else {
            continue;
        };
        out.push((*x, *y));
    }
    out
}

fn dedup_examples(out: &mut Vec<(i64, i64)>) {
    out.sort_unstable();
    out.dedup();
}

fn is_i64_scalar_rule(mog: &str) -> bool {
    let header = mog.lines().next().unwrap_or("");
    let lower = header.to_lowercase();
    let banned = ["f64", "f32", "float", "string", "str", "&str", "char", "bool", "[", "vec<"];
    if banned.iter().any(|b| lower.contains(b)) {
        return false;
    }
    lower.contains("-> i64")
}

fn verify_mog_i64_examples(mog: &str, name: &str, examples: &[(i64, i64)]) -> Result<(), String> {
    for &(x, want) in examples {
        let got = execute_function(mog, name, &[BValue::Int(x)], name)
            .map_err(|e| format!("rule '{name}': Mog errored on example input {x}: {e}"))?;
        match got {
            RValue::Int(n) if n == want => {}
            RValue::Int(n) => {
                return Err(format!(
                    "rule '{name}': Mog disagrees with inline example {name}({x}) = {n}, expected {want}"
                ));
            }
            other => {
                return Err(format!(
                    "rule '{name}': non-integer return for {name}({x}): {other:?}"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn english_example_parser_finds_rule_io_pairs() {
        let bonus = examples_for_rule_in_text(DEFAULT_BACKEND_ENGLISH, "score_bonus");
        assert!(bonus.contains(&(1, 15)));
        assert!(bonus.iter().any(|(x, y)| *x == 0 && *y == 5));

        let penalty = examples_for_rule_in_text(DEFAULT_BACKEND_ENGLISH, "damage_penalty");
        assert!(penalty.len() >= 4, "penalty examples: {penalty:?}");
        assert!(penalty.contains(&(0, -3)));
        assert!(
            penalty.contains(&(5, 7)) || penalty.contains(&(4, 5)),
            "penalty examples: {penalty:?}"
        );
    }

    fn rustc_available() -> bool {
        Command::new("rustc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn unique_path(stem: &str, ext: &str) -> std::path::PathBuf {
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

    #[test]
    fn nl_backend_from_default_english_compiles_and_serves_both_rules() {
        if !rustc_available() {
            eprintln!("skipping NL backend integration test: rustc unavailable");
            return;
        }

        let generated = build_backend_from_english(
            DEFAULT_BACKEND_ENGLISH,
            default_required_rule_names(),
            StoreKind::Memory,
        )
        .expect("build backend from English");

        assert_eq!(generated.rules.len(), 2);
        assert!(generated.source.contains("/rules/score_bonus/evaluate"));
        assert!(generated.source.contains("/rules/damage_penalty/evaluate"));

        let src = unique_path("ncpu_nl_backend", "rs");
        let bin = unique_path("ncpu_nl_backend", "bin");
        std::fs::write(&src, &generated.source).expect("write generated source");
        let compile = Command::new("rustc")
            .arg("--edition=2021")
            .arg(&src)
            .arg("-o")
            .arg(&bin)
            .output()
            .expect("run rustc");
        assert!(
            compile.status.success(),
            "rustc failed\nstderr:\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );

        let mut child = Command::new(&bin)
            .arg("--port")
            .arg("0")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn generated backend");
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

        stop(&mut child);
        let _ = std::fs::remove_file(src);
        let _ = std::fs::remove_file(bin);
    }
}
