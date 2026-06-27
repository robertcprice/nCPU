//! Unified backend prose intake (LOOP-7) — multi-door router + auto HTTP checks.
//!
//! Eliminates ceilings by trying, per rule:
//! 1. inline examples (when present in the contract text),
//! 2. compositional P2C `then`-chains,
//! 3. single registry unary op,
//! 4. NL comprehend + strict-verify.

use crate::backend_http::HttpRuleCheck;
use crate::backend_ir::{BackendApp, RuleModel, StoreKind};
use crate::backend_mvp::{GeneratedBackend, SynthesizedRuleArtifact};
use crate::backend_nl::{examples_for_rule_in_text, split_function_clauses};
use crate::backend_p2c::parse_p2c_rule_clauses;
use crate::backend_repair::build_with_compile_and_http_repair;
use crate::benchmark::Value as BValue;
use crate::linguigenesis_bridge::LinguigenesisBridge;
use crate::mog_transpile::to_rust;
use crate::runtime::{execute_function, Value as RValue};
use crate::solver::SolveResult;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProseSynthesisDoor {
    InlineExamples,
    Compositional,
    SingleOp,
    NlComprehend,
}

impl ProseSynthesisDoor {
    pub fn method_prefix(self) -> &'static str {
        match self {
            Self::InlineExamples => "inline",
            Self::Compositional => "prose:p2c",
            Self::SingleOp => "prose:single-op",
            Self::NlComprehend => "prose:nl-desc",
        }
    }
}

pub fn build_backend_unified(
    english: &str,
    required: &[&str],
    http_checks: Option<&[HttpRuleCheck]>,
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    if required.is_empty() {
        return Err("unified backend build requires at least one required rule name".to_string());
    }

    let bridge = LinguigenesisBridge::new();
    if let Some(err) = bridge.registry_load_error() {
        return Err(format!("NL registry failed to load: {err}"));
    }

    let clauses = parse_p2c_rule_clauses(english);
    let by_name: HashMap<&str, &str> = clauses
        .iter()
        .map(|c| (c.name.as_str(), c.description.as_str()))
        .collect();

    let mut rules = Vec::with_capacity(required.len());
    let mut mog_by_name: HashMap<String, String> = HashMap::new();

    for name in required {
        let description = by_name.get(name).copied().ok_or_else(|| {
            format!(
                "rule '{name}' not found in English contract (parsed {} clause(s))",
                clauses.len()
            )
        })?;
        let (res, door) = synthesize_rule_for_prose(&bridge, english, name, description)?;
        if !is_i64_scalar_rule(&res.code) {
            return Err(format!(
                "rule '{name}' is not scalar i64 after synthesis via {:?}.\n  mog: {}",
                door,
                res.code.lines().next().unwrap_or("").trim()
            ));
        }

        let rule_code = to_rust(&res.code);
        if !rule_code.contains(&format!("fn {name}(")) {
            return Err(format!(
                "transpiled Rust for '{name}' does not define fn {name}(...): {rule_code}"
            ));
        }

        mog_by_name.insert((*name).to_string(), res.code.clone());
        rules.push(SynthesizedRuleArtifact {
            name: (*name).to_string(),
            rule_code,
            rule_method: format!("{}:{}", door.method_prefix(), res.method),
        });
    }

    let checks = match http_checks {
        Some(c) if !c.is_empty() => c.to_vec(),
        _ => derive_http_checks(english, &rules, &mog_by_name),
    };

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
    let source = build_with_compile_and_http_repair(&app, &checks, store, 3)?;
    Ok(GeneratedBackend { source, rules })
}

pub fn write_backend_unified(
    path: impl AsRef<std::path::Path>,
    english: &str,
    required: &[&str],
    http_checks: Option<&[HttpRuleCheck]>,
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    let generated = build_backend_unified(english, required, http_checks, store)?;
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create output dir {}: {e}", parent.display()))?;
    }
    std::fs::write(path.as_ref(), &generated.source)
        .map_err(|e| format!("write {}: {e}", path.as_ref().display()))?;
    Ok(generated)
}

pub fn synthesize_rule_for_prose(
    bridge: &LinguigenesisBridge,
    english: &str,
    name: &str,
    description: &str,
) -> Result<(SolveResult, ProseSynthesisDoor), String> {
    let examples = examples_for_rule_in_text(english, name);
    if examples.len() >= 2 {
        if let Ok(res) = synthesize_from_inline_clause(bridge, english, name, description) {
            return Ok((res, ProseSynthesisDoor::InlineExamples));
        }
    }

    let (res, door_tag) = bridge
        .synthesize_prose_scalar_named(name, description)
        .map_err(|e| format!("prose synthesis failed for '{name}': {e}"))?;
    let door = match door_tag {
        "prose:p2c" => ProseSynthesisDoor::Compositional,
        "prose:single-op" => ProseSynthesisDoor::SingleOp,
        "prose:nl-desc" => ProseSynthesisDoor::NlComprehend,
        other => {
            return Err(format!("unexpected prose door tag for '{name}': {other}"));
        }
    };
    Ok((res, door))
}

fn synthesize_from_inline_clause(
    bridge: &LinguigenesisBridge,
    english: &str,
    name: &str,
    description: &str,
) -> Result<SolveResult, String> {
    let clause = find_clause_for_rule(english, name).ok_or_else(|| {
        format!("inline path: no clause found for rule '{name}' in English contract")
    })?;
    let mini = if clause.to_lowercase().contains(&format!("function {name}")) {
        clause
    } else {
        format!("A function {name} that {description}. {}", inline_example_literals(name, english))
    };
    let (solved, skipped) = bridge
        .synthesize_project(&mini)
        .map_err(|e| format!("inline synthesize_project for '{name}': {e}"))?;
    if !skipped.is_empty() {
        return Err(format!(
            "inline synthesize_project for '{name}' skipped: {skipped:?}"
        ));
    }
    solved
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, r)| r)
        .ok_or_else(|| format!("inline synthesize_project did not return rule '{name}'"))
}

fn find_clause_for_rule(english: &str, name: &str) -> Option<String> {
    split_function_clauses(english).into_iter().find(|clause| {
        clause.to_lowercase().contains(&format!("function {name}"))
            || clause.contains(&format!("{name}("))
    })
}

fn inline_example_literals(name: &str, english: &str) -> String {
    examples_for_rule_in_text(english, name)
        .into_iter()
        .map(|(x, y)| format!("{name}({x})={y}"))
        .collect::<Vec<_>>()
        .join(" and ")
}

pub fn derive_http_checks(
    english: &str,
    rules: &[SynthesizedRuleArtifact],
    mog_by_name: &HashMap<String, String>,
) -> Vec<HttpRuleCheck> {
    rules
        .iter()
        .filter_map(|rule| {
            let examples = examples_for_rule_in_text(english, &rule.name);
            if let Some((input, output)) = examples.first().copied() {
                return Some(HttpRuleCheck {
                    rule: rule.name.clone(),
                    input,
                    output,
                });
            }
            mog_by_name
                .get(&rule.name)
                .and_then(|mog| probe_rule_io(mog, &rule.name))
                .map(|(input, output)| HttpRuleCheck {
                    rule: rule.name.clone(),
                    input,
                    output,
                })
        })
        .collect()
}

pub fn probe_rule_io(mog: &str, name: &str) -> Option<(i64, i64)> {
    for x in [1_i64, 0, -1, 3, 5, -5] {
        if let Ok(RValue::Int(y)) = execute_function(mog, name, &[BValue::Int(x)], name) {
            return Some((x, y));
        }
    }
    None
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
    use crate::backend_p2c::{default_p2c_http_checks, DEFAULT_BACKEND_P2C_ENGLISH};

    #[test]
    fn prose_router_single_op_increments_without_then_chain() {
        let bridge = LinguigenesisBridge::new();
        let (res, door) = bridge
            .synthesize_prose_scalar_named("bump", "increments a number")
            .expect("single-op increment");
        assert_eq!(door, "prose:single-op");
        assert!(res.success);
        assert!(res.code.contains("fn bump"));
    }

    #[test]
    fn unified_build_uses_compositional_p2c_default() {
        if !rustc_available() {
            eprintln!("skipping unified P2C integration test: rustc unavailable");
            return;
        }

        let generated = build_backend_unified(
            DEFAULT_BACKEND_P2C_ENGLISH,
            default_required_rule_names(),
            Some(&default_p2c_http_checks()),
            StoreKind::Memory,
        )
        .expect("unified P2C build");

        assert_eq!(generated.rules.len(), 2);
        assert!(
            generated.rules[0].rule_method.starts_with("prose:p2c:")
                || generated.rules[0].rule_method.starts_with("prose:single-op:")
        );
        assert!(generated.source.contains("/rules/score_bonus/evaluate"));
    }

    #[test]
    fn unified_build_auto_routes_inline_examples_when_present() {
        if !rustc_available() {
            eprintln!("skipping unified inline auto-route test: rustc unavailable");
            return;
        }

        let english = "\
A function score_bonus that scores ten points per catch plus a five point bonus, \
score_bonus(0)=5 and score_bonus(1)=15 and score_bonus(2)=25. \
A function damage_penalty that converts hit points lost into a signed penalty score twice the loss minus three, \
damage_penalty(0)=-3 and damage_penalty(1)=-1 and damage_penalty(2)=1.";

        let generated = build_backend_unified(
            english,
            default_required_rule_names(),
            None,
            StoreKind::Memory,
        )
        .expect("unified inline build");

        assert!(generated
            .rules
            .iter()
            .any(|r| r.rule_method.starts_with("inline:")));
    }

    #[test]
    fn derive_http_checks_from_mog_probe_when_no_inline_examples() {
        let bridge = LinguigenesisBridge::new();
        let (res, _) = bridge
            .synthesize_prose_scalar_named("bump", "increments a number")
            .expect("synthesize bump");
        let mog = res.code.clone();
        let rules = vec![SynthesizedRuleArtifact {
            name: "bump".to_string(),
            rule_code: to_rust(&mog),
            rule_method: "prose:single-op:test".to_string(),
        }];
        let mut mog_map = HashMap::new();
        mog_map.insert("bump".to_string(), mog);
        let checks = derive_http_checks("", &rules, &mog_map);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].rule, "bump");
        assert_eq!(checks[0].output, checks[0].input + 1);
    }
}
