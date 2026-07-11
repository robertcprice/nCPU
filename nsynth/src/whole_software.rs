//! Whole-software product path: prose → scaffold oracle → fill → cargo gate.
//!
//! Closes the gap where Phases 2–3 existed only as standalone bins. The product
//! `handle_query` route calls [`try_scaffold`] then the caller runs the repo-agent
//! ladder against the emitted crate.
//!
//! - **Tier A (model-free):** schema-decidable prose → stubs + per-method tests.
//! - **Tier A′ (model-free):** characterization from ≥2 inline examples.
//! - **Tier B (gated):** `propose_spec` when `NSYNTH_LOCAL_LLM_URL` is set.
//! - **Tier C (gated):** project decompose via `NSYNTH_LOCAL_LLM_PROJECT` / URL.
//! - Otherwise: `None` so the session falls through to honest refuse/clarify.

use crate::characterization;
use crate::local_llm;
use crate::schema_component::{self, WrittenSchemaCrate};
use std::path::{Path, PathBuf};

/// How the scaffold oracle was manufactured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaffoldKind {
    Schema,
    Spec,
    Characterization,
    Decompose,
}

/// A checkable crate written under `root`, ready for the hole-filler ladder.
#[derive(Debug, Clone)]
pub struct ScaffoldedCrate {
    pub root: PathBuf,
    pub kind: ScaffoldKind,
    pub method: &'static str,
    pub summary: String,
    pub n_tests: usize,
    /// Stable promote name: schema collection, characterization fn, etc.
    pub component_name: String,
}

impl From<WrittenSchemaCrate> for ScaffoldedCrate {
    fn from(w: WrittenSchemaCrate) -> Self {
        let summary = format!(
            "schema {} {{ items: Vec<{}> }} ({} fields, {} tests)",
            w.collection, w.record, w.n_fields, w.n_tests
        );
        Self {
            root: w.root,
            kind: ScaffoldKind::Schema,
            method: w.method,
            summary,
            n_tests: w.n_tests,
            component_name: w.collection,
        }
    }
}

/// Try to manufacture a checkable scaffold for `prose` under `out_dir`.
///
/// Order: schema → characterization (≥2 inline examples) → gated `propose_spec`.
/// Returns `None` when none apply (caller must refuse/clarify).
pub fn try_scaffold(out_dir: &Path, prose: &str) -> Option<ScaffoldedCrate> {
    if let Some(written) = schema_component::try_write_schema_crate(out_dir, prose) {
        return Some(written.into());
    }
    if let Some(char_sc) = try_scaffold_characterization(out_dir, prose) {
        return Some(char_sc);
    }
    try_scaffold_from_spec(out_dir, prose)
}

/// WP2: bootstrap a characterization oracle from inline examples in prose.
pub fn try_scaffold_characterization(out_dir: &Path, prose: &str) -> Option<ScaffoldedCrate> {
    let examples = characterization::parse_inline_char_examples(prose);
    if examples.len() < 2 {
        return None;
    }
    let fn_name = characterization::infer_fn_name(prose, None);
    let written =
        characterization::write_characterization_crate(out_dir, &fn_name, &examples).ok()?;
    Some(ScaffoldedCrate {
        root: out_dir.to_path_buf(),
        kind: ScaffoldKind::Characterization,
        method: written.method,
        summary: format!(
            "characterization fn {} ({} tests)",
            written.fn_name, written.n_tests
        ),
        n_tests: written.n_tests,
        component_name: written.fn_name,
    })
}

/// Phase-3 path: model writes SPEC (signatures + tests), never trusted code.
pub fn try_scaffold_from_spec(out_dir: &Path, prose: &str) -> Option<ScaffoldedCrate> {
    let lib = local_llm::propose_spec(prose)?;
    let n_tests = lib.matches("#[test]").count();
    if n_tests == 0 {
        return None;
    }
    schema_component::write_lib_crate(out_dir, "spec_crate", &lib).ok()?;
    let reqs: Vec<&str> = lib
        .lines()
        .filter(|l| l.trim_start().starts_with("//!"))
        .collect();
    let req_note = if reqs.is_empty() {
        String::new()
    } else {
        format!(
            "; requirements: {}",
            reqs.iter()
                .map(|r| r.trim_start().trim_start_matches("//!").trim())
                .filter(|s| !s.is_empty())
                .take(6)
                .collect::<Vec<_>>()
                .join(" | ")
        )
    };
    Some(ScaffoldedCrate {
        root: out_dir.to_path_buf(),
        kind: ScaffoldKind::Spec,
        method: "whole-software:spec",
        summary: format!("model-written spec ({n_tests} tests){req_note}"),
        n_tests,
        component_name: "spec_crate".into(),
    })
}

/// WP3: gated multi-component decompose → verified project write.
///
/// Requires `NSYNTH_LOCAL_LLM_PROJECT` or `NSYNTH_LOCAL_LLM_URL`. On success with
/// ≥1 verified component, writes via [`write_verified_project`] and returns a
/// [`ScaffoldKind::Decompose`] crate.
pub fn try_decompose_project(root: &Path, prose: &str) -> Option<ScaffoldedCrate> {
    let project_set = std::env::var("NSYNTH_LOCAL_LLM_PROJECT")
        .ok()
        .filter(|s| !s.is_empty())
        .is_some();
    let url_set = std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .is_some();
    if !project_set && !url_set {
        return None;
    }
    // synthesize_project_with_contracts itself gates on NSYNTH_LOCAL_LLM_PROJECT;
    // when only URL is set, temporarily enable the project flag for this call.
    let _project_guard = if !project_set && url_set {
        std::env::set_var("NSYNTH_LOCAL_LLM_PROJECT", "1");
        Some(EnvRestore {
            key: "NSYNTH_LOCAL_LLM_PROJECT",
            prev: None,
        })
    } else {
        None
    };

    let bridge = crate::linguigenesis_bridge::LinguigenesisBridge::new();
    let (verified, _failed) = bridge.synthesize_project_with_contracts(prose)?;
    if verified.is_empty() {
        return None;
    }
    let components: Vec<(String, String, Vec<crate::benchmark::Example>)> = verified
        .into_iter()
        .map(|v| (v.name, v.result.code, v.examples))
        .collect();
    let n = components.len();
    let _outcome =
        crate::agent::repo::write_verified_project(root, "decompose_crate", &components).ok()?;
    Some(ScaffoldedCrate {
        root: root.to_path_buf(),
        kind: ScaffoldKind::Decompose,
        method: "whole-software:decompose",
        summary: format!("decomposed project ({n} verified components)"),
        n_tests: n,
        component_name: "decompose_crate".into(),
    })
}

struct EnvRestore {
    key: &'static str,
    prev: Option<String>,
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

/// True when prose should attempt the whole-software door before honest refuse.
/// Schema-shaped prose always qualifies; otherwise a construction verb is required
/// so we don't hijack informational queries into a model-spec call.
pub fn should_attempt_whole_software(prose: &str) -> bool {
    schema_component::is_schema_prose(prose)
        || has_construction_verb(prose)
        || characterization::parse_inline_char_examples(prose).len() >= 2
}

fn has_construction_verb(query: &str) -> bool {
    const BUILD_VERBS: [&str; 13] = [
        "build", "make", "create", "implement", "write", "generate", "develop", "design", "add",
        "code", "program", "produce", "scaffold",
    ];
    let lower = query.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|tok| BUILD_VERBS.contains(&tok))
}

/// Write a Phase-3 spec crate from an already-obtained lib.rs body (tests / mocks).
pub fn write_spec_crate(out_dir: &Path, lib_rs: &str) -> Result<ScaffoldedCrate, String> {
    let n_tests = lib_rs.matches("#[test]").count();
    if n_tests == 0 {
        return Err("spec has no #[test]".into());
    }
    schema_component::write_lib_crate(out_dir, "spec_crate", lib_rs)?;
    Ok(ScaffoldedCrate {
        root: out_dir.to_path_buf(),
        kind: ScaffoldKind::Spec,
        method: "whole-software:spec",
        summary: format!("spec crate ({n_tests} tests)"),
        n_tests,
        component_name: "spec_crate".into(),
    })
}

/// WP7 telemetry: bounded scaffold → fill → observe loop phases.
#[derive(Debug, Clone, Default)]
pub struct BuildPlan {
    pub phases: Vec<String>,
}

impl BuildPlan {
    pub fn new() -> Self {
        Self { phases: Vec::new() }
    }

    pub fn record(&mut self, phase: impl Into<String>) {
        self.phases.push(phase.into());
    }
}

/// Record scaffold → fill → observe phases for WP7 telemetry.
pub fn run_bounded_loop(plan: &mut BuildPlan, scaffold_label: &str) {
    plan.record(format!("scaffold:{scaffold_label}"));
    plan.record("fill");
    plan.record("observe");
}

/// WP4 product-facing helper: parse `property:` / `satisfies:` Mog predicate requests.
///
/// Expected shape (flexible whitespace):
/// ```text
/// property: fn inc(x: i64) -> i64 satisfies: fn gt(x: i64, out: i64) -> i64 { … }
/// ```
pub fn parse_property_request(
    prose: &str,
) -> Option<(String, String, String, String, String)> {
    let lower = prose.to_lowercase();
    if !lower.contains("property:") && !lower.contains("satisfies:") {
        return None;
    }
    let sat_idx = lower.find("satisfies:")?;
    let after_sat = prose[sat_idx + "satisfies:".len()..].trim_start();
    let (pred_name, pred_sig, pred_code) = parse_fn_with_body(after_sat)?;

    let cand_src = if let Some(p) = lower.find("property:") {
        let start = p + "property:".len();
        prose[start..sat_idx].trim()
    } else {
        prose[..sat_idx].trim()
    };
    let (cand_name, cand_sig) = parse_fn_sig(cand_src)?;
    Some((cand_name, cand_sig, pred_name, pred_sig, pred_code))
}

fn parse_fn_sig(src: &str) -> Option<(String, String)> {
    let s = src.trim();
    let body_cut = s.find('{').map(|i| &s[..i]).unwrap_or(s).trim();
    let after_fn = if let Some(i) = body_cut.find("fn ") {
        body_cut[i + 3..].trim_start()
    } else {
        body_cut
    };
    let name_end = after_fn
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(after_fn.len());
    let name = after_fn[..name_end].to_string();
    if name.is_empty() {
        return None;
    }
    let rest = after_fn[name_end..].trim_start();
    if !rest.starts_with('(') {
        return None;
    }
    Some((name.clone(), format!("fn {name}{rest}").trim().to_string()))
}

fn parse_fn_with_body(src: &str) -> Option<(String, String, String)> {
    let (name, sig) = parse_fn_sig(src)?;
    let brace = src.find('{')?;
    let body_src = &src[brace..];
    let mut depth = 0i32;
    let mut end = None;
    for (i, c) in body_src.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end?;
    let body = body_src[..end].trim();
    let code = format!("{sig} {body}");
    Some((name, sig, code))
}

/// WP4: verify candidate code against a Mog property predicate.
pub fn try_property_verify(
    candidate_code: &str,
    candidate_name: &str,
    candidate_signature: &str,
    predicate_name: &str,
    predicate_signature: &str,
    predicate_code: &str,
) -> Result<(), String> {
    use crate::agent::coding_intent::Spec;
    let spec = Spec::Property {
        candidate_name: candidate_name.to_string(),
        candidate_signature: candidate_signature.to_string(),
        predicate_name: predicate_name.to_string(),
        predicate_signature: predicate_signature.to_string(),
        predicate_code: predicate_code.to_string(),
    };
    spec.verify(candidate_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_scaffold_via_product_helper() {
        let root = std::env::temp_dir().join(format!(
            "nsynth_ws_schema_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let s = try_scaffold(
            &root,
            "a cart where each item has a name and a price number",
        )
        .expect("schema scaffold");
        assert_eq!(s.kind, ScaffoldKind::Schema);
        assert!(s.n_tests >= 8);
        assert!(root.join("src/lib.rs").is_file());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn characterization_scaffold_from_inline_examples() {
        let root = std::env::temp_dir().join(format!(
            "nsynth_ws_char_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let s = try_scaffold(&root, "double(2)=4, double(3)=6")
            .expect("characterization scaffold");
        assert_eq!(s.kind, ScaffoldKind::Characterization);
        assert_eq!(s.n_tests, 2);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn snake_game_is_not_schema_but_is_construction() {
        assert!(!schema_component::is_schema_prose(
            "build a snake game with keyboard controls"
        ));
        assert!(should_attempt_whole_software(
            "build a snake game with keyboard controls"
        ));
        // Without a model URL, try_scaffold must decline (honest fallthrough).
        if std::env::var("NSYNTH_LOCAL_LLM_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .is_none()
        {
            let root = std::env::temp_dir().join(format!(
                "nsynth_ws_snake_{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            assert!(try_scaffold(&root, "build a snake game with keyboard controls").is_none());
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn write_spec_crate_from_hand_spec() {
        let root = std::env::temp_dir().join(format!(
            "nsynth_ws_spec_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let lib = r#"
pub struct Account { pub balance: i64 }
impl Account {
    pub fn new() -> Self { Account { balance: 0 } }
    pub fn deposit(&mut self, amount: i64) {}
    pub fn withdraw(&mut self, amount: i64) {}
    pub fn balance(&self) -> i64 {}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn t_deposit() {
        let mut a = Account::new();
        a.deposit(10);
        assert_eq!(a.balance(), 10);
    }
    #[test]
    fn t_withdraw() {
        let mut a = Account::new();
        a.deposit(10);
        a.withdraw(3);
        assert_eq!(a.balance(), 7);
    }
}
"#;
        let s = write_spec_crate(&root, lib).expect("write");
        assert_eq!(s.kind, ScaffoldKind::Spec);
        assert_eq!(s.n_tests, 2);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn build_plan_records_bounded_loop_phases() {
        let mut plan = BuildPlan::new();
        run_bounded_loop(&mut plan, "schema");
        assert_eq!(plan.phases, ["scaffold:schema", "fill", "observe"]);
    }

    #[test]
    fn parse_property_request_extracts_parts() {
        let prose = "property: fn inc(x: i64) -> i64 satisfies: fn gt(x: i64, out: i64) -> i64 { if out > x { return 1; } return 0; }";
        let (n, s, pn, ps, pc) = parse_property_request(prose).expect("parse");
        assert_eq!(n, "inc");
        assert!(s.contains("inc"));
        assert_eq!(pn, "gt");
        assert!(ps.contains("gt"));
        assert!(pc.contains("out > x"));
    }
}
