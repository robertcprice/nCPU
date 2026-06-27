//! P2C backend intake — English prose without inline examples (LOOP-6).
//!
//! Uses the proven P2C compositional door: `classify_compositional` →
//! `emit_scalar_reference` → `problem_from_reference` (auto-manufactures
//! examples) → solve → strict-verify. No `name(x)=y` literals required.

use crate::backend_http::{
    cleanup_temp_artifacts, compile_to_temp_bin, verify_backend_http, HttpRuleCheck,
};
use crate::backend_ir::{BackendApp, RuleModel, StoreKind};
use crate::backend_mvp::{GeneratedBackend, SynthesizedRuleArtifact};
use crate::backend_nl::split_function_clauses;
use crate::backend_repair::compile_with_repair;
use crate::linguigenesis_bridge::LinguigenesisBridge;
use crate::mog_transpile::to_rust;
use std::path::Path;

/// Default P2C backend contract: compositional prose only, no inline examples.
/// Uses registry-resolvable `then` chains (same P2C door as `reference_nl` e2e tests).
pub const DEFAULT_BACKEND_P2C_ENGLISH: &str = "\
A function score_bonus that negates a number then triples it then increments it. \
A function damage_penalty that takes the absolute value of a number then increments it.";

pub fn default_p2c_http_checks() -> Vec<HttpRuleCheck> {
    vec![
        HttpRuleCheck {
            rule: "score_bonus".to_string(),
            input: 1,
            output: -2,
        },
        HttpRuleCheck {
            rule: "damage_penalty".to_string(),
            input: -5,
            output: 6,
        },
    ]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct P2cRuleClause {
    pub name: String,
    pub description: String,
}

pub fn parse_p2c_rule_clauses(text: &str) -> Vec<P2cRuleClause> {
    split_function_clauses(text)
        .iter()
        .filter_map(|clause| parse_named_function_clause(clause))
        .collect()
}

fn parse_named_function_clause(clause: &str) -> Option<P2cRuleClause> {
    let lower = clause.to_lowercase();
    let prefix = "a function ";
    let start = lower.find(prefix)? + prefix.len();
    let that = " that ";
    let that_rel = lower[start..].find(that)?;
    let name = clause[start..start + that_rel].trim().to_string();
    let desc_start = start + that_rel + that.len();
    let description = clause[desc_start..]
        .trim()
        .trim_end_matches('.')
        .to_string();
    if name.is_empty() || description.is_empty() {
        return None;
    }
    Some(P2cRuleClause { name, description })
}

pub fn write_backend_from_p2c_prose(
    path: impl AsRef<Path>,
    english: &str,
    required: &[&str],
    http_checks: &[HttpRuleCheck],
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    let generated = build_backend_from_p2c_prose(english, required, http_checks, store)?;
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create output dir {}: {e}", parent.display()))?;
    }
    std::fs::write(path.as_ref(), &generated.source)
        .map_err(|e| format!("write {}: {e}", path.as_ref().display()))?;
    Ok(generated)
}

pub fn build_backend_from_p2c_prose(
    english: &str,
    required: &[&str],
    http_checks: &[HttpRuleCheck],
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    if required.is_empty() {
        return Err("P2C backend build requires at least one required rule name".to_string());
    }

    let clauses = parse_p2c_rule_clauses(english);
    let by_name: std::collections::HashMap<&str, &P2cRuleClause> =
        clauses.iter().map(|c| (c.name.as_str(), c)).collect();

    let bridge = LinguigenesisBridge::new();
    if let Some(err) = bridge.registry_load_error() {
        return Err(format!("NL registry failed to load: {err}"));
    }

    let mut rules = Vec::with_capacity(required.len());
    for name in required {
        let clause = by_name.get(name).ok_or_else(|| {
            format!(
                "P2C rule '{name}' not found in English contract (parsed {} clause(s))",
                clauses.len()
            )
        })?;
        let res = bridge
            .synthesize_p2c_scalar_named(name, &clause.description)
            .map_err(|e| format!("P2C synthesis failed for '{name}': {e}"))?;

        if !is_i64_scalar_rule(&res.code) {
            return Err(format!(
                "P2C rule '{name}' is not scalar i64 after synthesis.\n  mog: {}",
                res.code.lines().next().unwrap_or("").trim()
            ));
        }

        let rule_code = to_rust(&res.code);
        if !rule_code.contains(&format!("fn {name}(")) {
            return Err(format!(
                "transpiled Rust for '{name}' does not define fn {name}(...): {rule_code}"
            ));
        }

        rules.push(SynthesizedRuleArtifact {
            name: (*name).to_string(),
            rule_code,
            rule_method: format!("p2c:{}", res.method),
        });
    }

    let description = clauses
        .iter()
        .map(|c| format!("{}: {}", c.name, c.description))
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
    verify_built_backend_http(&source, http_checks, store)?;
    Ok(GeneratedBackend { source, rules })
}

fn verify_built_backend_http(
    source: &str,
    checks: &[HttpRuleCheck],
    store: StoreKind,
) -> Result<(), String> {
    if checks.is_empty() || !rustc_available() {
        return Ok(());
    }
    let (src, bin) = compile_to_temp_bin(source, store == StoreKind::Sqlite)?;
    let result = verify_backend_http(&bin, checks, 2);
    cleanup_temp_artifacts(&src, &bin);
    result
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

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_nl::default_required_rule_names;

    #[test]
    fn p2c_parser_extracts_named_descriptions_without_examples() {
        let clauses = parse_p2c_rule_clauses(DEFAULT_BACKEND_P2C_ENGLISH);
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0].name, "score_bonus");
        assert!(clauses[0].description.contains("negate"));
        assert_eq!(clauses[1].name, "damage_penalty");
        assert!(clauses[1].description.contains("absolute"));
        assert!(examples_absent(DEFAULT_BACKEND_P2C_ENGLISH));
    }

    fn examples_absent(text: &str) -> bool {
        linguigenesis_core::inline_examples::parse_inline_examples(text).is_empty()
    }

    #[test]
    fn p2c_backend_from_prose_compiles_and_serves_both_rules() {
        if !rustc_available() {
            eprintln!("skipping P2C backend integration test: rustc unavailable");
            return;
        }

        let generated = build_backend_from_p2c_prose(
            DEFAULT_BACKEND_P2C_ENGLISH,
            default_required_rule_names(),
            &default_p2c_http_checks(),
            StoreKind::Memory,
        )
        .expect("build P2C backend");

        assert_eq!(generated.rules.len(), 2);
        assert!(generated.rules[0].rule_method.starts_with("p2c:"));
        assert!(generated.source.contains("/rules/score_bonus/evaluate"));
        assert!(generated.source.contains("/rules/damage_penalty/evaluate"));
    }
}
