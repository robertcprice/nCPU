//! Experience → routing bridge (Phase 4.1 learning loop).
//!
//! The agentic solve path records every run into an [`ExperienceDB`]
//! (`~/.nsynth_experience.json`). Until now that data was write-only. This
//! module reads it back at solve time: `PreferMethod` lessons whose pattern
//! matches the incoming problem become *synthetic win boosts* for the
//! corresponding route, nudging the method router toward strategies that
//! historically solved similar problems.
//!
//! Design constraints:
//! - **No-op by default under test.** [`db_path`] returns `None` when
//!   `cfg!(test)` and no env override is set, so the on-disk DB is never touched
//!   and the existing routing tests see exactly zero behavioral change.
//! - **Best-effort.** A missing/corrupt DB yields empty boosts, never an error.
//! - **Cheap.** The DB is loaded once per process and cached; per-problem work
//!   is an in-memory lesson filter.

use crate::benchmark::Problem;
use crate::learning::experience::{ExperienceDB, LessonAction};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Synthetic wins contributed per unit of `effectiveness * confidence`. Kept
/// small so experience nudges the ranking without steamrolling live win/miss
/// statistics from the method router.
const BOOST_SCALE: f64 = 5.0;

/// Canonical path to the experience DB. `NSYNTH_EXPERIENCE_DB` overrides;
/// otherwise `~/.nsynth_experience.json`. Returns `None` under test (unless
/// overridden) so unit tests never read or write the shared DB.
pub(crate) fn db_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_EXPERIENCE_DB") {
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".nsynth_experience.json"))
}

/// Process-global, lazily-loaded experience DB. `None` when there is no DB to
/// load (e.g. under test). Loaded once; subsequent solves reuse the in-memory
/// copy.
fn cached_db() -> Option<&'static ExperienceDB> {
    static DB: OnceLock<Option<ExperienceDB>> = OnceLock::new();
    DB.get_or_init(|| {
        let path = db_path()?;
        ExperienceDB::new(path).ok()
    })
    .as_ref()
}

/// Pure core: derive per-method win boosts from the lessons in `db` that match
/// `problem`. Keyed by the raw lesson method string (e.g. `"search_scalar_expr"`);
/// the caller normalizes those to route constants. Exposed for direct testing
/// without touching global state.
pub(crate) fn boosts_from_db(db: &ExperienceDB, problem: &Problem) -> HashMap<String, u32> {
    let mut boosts: HashMap<String, u32> = HashMap::new();
    for lesson in db.get_effective_actions(problem) {
        if let LessonAction::PreferMethod { method } = &lesson.action {
            let weight = lesson.effectiveness * lesson.confidence * BOOST_SCALE;
            let boost = weight.round() as i64;
            if boost > 0 {
                *boosts.entry(method.clone()).or_insert(0) += boost as u32;
            }
        }
    }
    boosts
}

/// Experience-derived win boosts for `problem`, or empty when no DB is loaded.
/// This is the function the router consults at solve time.
pub(crate) fn route_boosts(problem: &Problem) -> HashMap<String, u32> {
    match cached_db() {
        Some(db) => boosts_from_db(db, problem),
        None => HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};
    use crate::solver::SolveResult;

    fn add_problem() -> Problem {
        Problem {
            name: "add_two".to_string(),
            category: "arithmetic",
            description: "",
            signature: "fn add_two(a: i64, b: i64) -> i64",
            examples: vec![
                Example { inputs: vec![Value::Int(1), Value::Int(2)], expected: Value::Int(3) },
                Example { inputs: vec![Value::Int(4), Value::Int(5)], expected: Value::Int(9) },
            ],
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    fn success(method: &str) -> SolveResult {
        SolveResult {
            success: true,
            code: "fn add_two(a: i64, b: i64) -> i64 { return (a + b); }".to_string(),
            method: method.to_string(),
            error: None,
            metadata: Default::default(),
        }
    }

    #[test]
    fn empty_db_yields_no_boosts() {
        let dir = std::env::temp_dir().join(format!("nsynth_advisor_empty_{}", std::process::id()));
        let path = dir.join("exp.json");
        let db = ExperienceDB::new(path).unwrap();
        assert!(boosts_from_db(&db, &add_problem()).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn recorded_success_boosts_its_method() {
        let dir = std::env::temp_dir().join(format!("nsynth_advisor_win_{}", std::process::id()));
        let path = dir.join("exp.json");
        let mut db = ExperienceDB::new(path).unwrap();
        let problem = add_problem();
        // Record the same successful method several times to build effectiveness.
        for _ in 0..3 {
            db.record_experience(&problem, &success("search_scalar_expr"), 5)
                .unwrap();
        }
        let boosts = boosts_from_db(&db, &problem);
        assert!(
            boosts.get("search_scalar_expr").copied().unwrap_or(0) > 0,
            "expected a positive boost for the historically-winning method, got {boosts:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn route_boosts_is_noop_without_db() {
        // Under cfg!(test) with no env override, db_path() is None.
        assert!(db_path().is_none() || std::env::var("NSYNTH_EXPERIENCE_DB").is_ok());
    }
}
