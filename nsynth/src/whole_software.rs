//! Whole-software product path (WP1): prose → scaffold oracle → fill → cargo gate.
//!
//! Closes the gap where Phases 2–3 existed only as standalone bins. The product
//! `handle_query` route calls [`try_scaffold`] then the caller runs the repo-agent
//! ladder against the emitted crate.
//!
//! - **Tier A (model-free):** schema-decidable prose → stubs + per-method tests.
//! - **Tier B (gated):** `propose_spec` when `NSYNTH_LOCAL_LLM_URL` is set.
//! - Otherwise: `None` so the session falls through to honest refuse/clarify.

use crate::local_llm;
use crate::schema_component::{self, WrittenSchemaCrate};
use std::path::{Path, PathBuf};

/// How the scaffold oracle was manufactured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaffoldKind {
    Schema,
    Spec,
}

/// A checkable crate written under `root`, ready for the hole-filler ladder.
#[derive(Debug, Clone)]
pub struct ScaffoldedCrate {
    pub root: PathBuf,
    pub kind: ScaffoldKind,
    pub method: &'static str,
    pub summary: String,
    pub n_tests: usize,
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
        }
    }
}

/// Try to manufacture a checkable scaffold for `prose` under `out_dir`.
///
/// Order: schema (decidable, model-free) → gated `propose_spec`. Returns `None`
/// when neither applies (caller must refuse/clarify — never invent unverified code).
pub fn try_scaffold(out_dir: &Path, prose: &str) -> Option<ScaffoldedCrate> {
    if let Some(written) = schema_component::try_write_schema_crate(out_dir, prose) {
        return Some(written.into());
    }
    try_scaffold_from_spec(out_dir, prose)
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
    })
}

/// True when prose should attempt the whole-software door before honest refuse.
/// Schema-shaped prose always qualifies; otherwise a construction verb is required
/// so we don't hijack informational queries into a model-spec call.
pub fn should_attempt_whole_software(prose: &str) -> bool {
    schema_component::is_schema_prose(prose) || has_construction_verb(prose)
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
    })
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
}
