//! Phase 5.1 — Bounded Recursive Self-Improvement.
//!
//! A safe, offline hill-climb that improves the solver by tuning **only** the
//! 32-dimensional [`MetaWeights`] ranker vector. It mutates no source, generates
//! no code, and accepts a candidate **iff** the set of benchmark problems it
//! solves is a *strict superset* of the incumbent's (zero per-problem
//! regression AND at least one net new solve). Everything else stays fixed.
//!
//! ## Why this is safe (design rails, all enforced here or by unmodified code)
//! - **Data-only mutation.** The loop touches only `MetaWeights.w` through
//!   [`crate::meta_learner`]; it has no codegen/eval/exec path.
//! - **Bounded values.** Candidates are built with `apply_weight_gradient`,
//!   which clamps every dimension to `[0.01, 100.0]`; `MetaWeights::load`
//!   re-clamps on next start. A pathological delta cannot poison the vector.
//! - **Side-effect-free fitness.** Evaluation runs under
//!   [`crate::learning_freeze`], so measuring a candidate cannot write the
//!   solved cache or mutate the weights mid-evaluation — the gate is
//!   deterministic and the surface stays singular.
//! - **Scratch isolation.** During the search the weights path is repointed to a
//!   scratch file; production weights are written only at an explicit commit.
//! - **Snapshot + immediate restore.** The incumbent is snapshotted to disk
//!   before iteration 0, and re-applied on every reject.
//! - **Three independent budget ceilings.** `max_iters`, `max_no_improve`
//!   (patience), and a wall-clock `deadline` — the loop cannot run unbounded.
//! - **Non-fabrication preserved.** `solve_problem` only counts a problem solved
//!   after `verify_problem_code_strict`; weights merely reorder which cached
//!   teacher is tried first, never whether unverified code is accepted.

use crate::benchmark::Problem;
use crate::meta_learner::{apply_weight_gradient, MetaWeights, FEATURE_DIM};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

/// Configuration for one RSI run.
#[derive(Clone, Debug)]
pub struct Config {
    pub max_iters: usize,
    pub max_no_improve: usize,
    pub deadline: Duration,
    pub sigma: f64,
    pub lr: f64,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_iters: 50,
            max_no_improve: 15,
            deadline: Duration::from_secs(1800),
            sigma: 0.25,
            lr: 0.1,
            seed: 1,
        }
    }
}

/// The fitness of a weight vector: which benchmark problems it solves.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fitness {
    pub solved: usize,
    pub total: usize,
    pub pass_set: BTreeSet<String>,
}

/// Outcome of a hill-climb (independent of any commit decision).
#[derive(Clone, Debug)]
pub struct Outcome {
    pub baseline_solved: usize,
    pub final_solved: usize,
    pub total: usize,
    pub accepted: usize,
    pub iterations: usize,
    pub winner: MetaWeights,
    pub improved: bool,
}

/// Strict-superset acceptance gate: accept iff the candidate solves a strict
/// superset of the incumbent's problems (zero regression + ≥1 net new solve).
/// An equal set, or a swap (equal count, different set), is rejected.
pub fn gate(baseline: &Fitness, candidate: &Fitness) -> bool {
    candidate.solved > baseline.solved && baseline.pass_set.is_subset(&candidate.pass_set)
}

/// Deterministic xorshift64* PRNG — seeded, no external dependency, so a run is
/// fully reproducible from `Config::seed`.
struct XorShift(u64);
impl XorShift {
    fn new(seed: u64) -> Self {
        // Avoid the zero fixed point.
        XorShift(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1.0, 1.0).
    fn unit(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
        u * 2.0 - 1.0
    }
}

/// Propose a candidate by perturbing the incumbent with a seeded random delta.
/// `apply_weight_gradient` clamps every dimension to `[0.01, 100.0]`.
fn propose(incumbent: &MetaWeights, rng: &mut XorShift, sigma: f64, lr: f64) -> MetaWeights {
    let mut candidate = incumbent.clone();
    let mut delta = [0.0_f64; FEATURE_DIM];
    for d in delta.iter_mut() {
        *d = rng.unit() * sigma;
    }
    apply_weight_gradient(&mut candidate, &delta, lr);
    candidate
}

/// Measure the CURRENT in-memory weights against `problems`, under a learning
/// freeze so the measurement has no side effects (no cache writes, no weight
/// self-updates). A problem counts as solved only via the verifier-gated
/// `solve_problem`, so the gate cannot be gamed by fabricated solutions.
pub fn evaluate_current(problems: &[Problem]) -> Fitness {
    let _frozen = crate::learning_freeze::freeze();
    let mut pass_set = BTreeSet::new();
    for p in problems {
        // Isolate per-problem panics: some synthesizers can panic on adversarial
        // inputs, and one panicking problem must not crash the whole evaluation
        // (it simply counts as unsolved). Mirrors the benchmark runner's
        // per-problem isolation.
        let solved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::solver::solve_problem(p).success
        }))
        .unwrap_or(false);
        if solved {
            pass_set.insert(p.name.clone());
        }
    }
    Fitness {
        solved: pass_set.len(),
        total: problems.len(),
        pass_set,
    }
}

/// Pure hill-climb core, independent of file/env orchestration so it is unit
/// testable with injected `eval`/`restore` closures.
///
/// - `eval(&candidate)` applies the candidate weights and returns its fitness.
/// - `restore(&incumbent)` re-applies the incumbent weights (called on reject).
/// - `audit(event)` records a structured event (no-op in tests).
fn hill_climb<E, R, A>(
    cfg: &Config,
    mut incumbent: MetaWeights,
    mut baseline: Fitness,
    started: Instant,
    mut eval: E,
    mut restore: R,
    mut audit: A,
) -> Outcome
where
    E: FnMut(&MetaWeights) -> Fitness,
    R: FnMut(&MetaWeights),
    A: FnMut(String),
{
    let baseline_solved = baseline.solved;
    let total = baseline.total;
    let mut rng = XorShift::new(cfg.seed);
    let mut accepted = 0usize;
    let mut no_improve = 0usize;
    let mut iter = 0usize;

    while iter < cfg.max_iters
        && no_improve < cfg.max_no_improve
        && started.elapsed() < cfg.deadline
    {
        iter += 1;
        let candidate = propose(&incumbent, &mut rng, cfg.sigma, cfg.lr);
        audit(format!("PROPOSE iter={iter} delta_seeded"));
        let cand_fit = eval(&candidate);
        if gate(&baseline, &cand_fit) {
            let newly: Vec<&String> = cand_fit.pass_set.difference(&baseline.pass_set).collect();
            audit(format!(
                "ACCEPT iter={iter} {}->{} newly={:?}",
                baseline.solved, cand_fit.solved, newly
            ));
            incumbent = candidate;
            baseline = cand_fit;
            accepted += 1;
            no_improve = 0;
        } else {
            let reason = if cand_fit.solved <= baseline.solved {
                "no_improvement"
            } else {
                "regressed"
            };
            audit(format!("REJECT iter={iter} reason={reason}"));
            // Immediate restore so the rejected vector never persists.
            restore(&incumbent);
            no_improve += 1;
        }
    }

    Outcome {
        baseline_solved,
        final_solved: baseline.solved,
        total,
        accepted,
        iterations: iter,
        winner: incumbent,
        improved: baseline.solved > baseline_solved,
    }
}

// ----------------------------------------------------------------------------
// Orchestration: scratch isolation, snapshot, lockfile, commit, audit.
// ----------------------------------------------------------------------------

use std::path::{Path, PathBuf};

const WEIGHTS_ENV: &str = "NSYNTH_META_WEIGHTS_PATH";

/// Filesystem locations the run uses. Injectable so tests can point everything
/// at a temp dir (and avoid touching the real `~/.nsynth_*`).
#[derive(Clone, Debug)]
pub struct Paths {
    /// Production weights file — written only on commit.
    pub production: PathBuf,
    /// Scratch weights file — all search writes go here.
    pub scratch: PathBuf,
    /// Directory for pre-run incumbent snapshots.
    pub snapshot_dir: PathBuf,
    /// Single-writer lockfile.
    pub lock: PathBuf,
}

impl Paths {
    /// Default locations under `$HOME`.
    pub fn defaults() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let h = PathBuf::from(home);
        let pid = std::process::id();
        Paths {
            production: std::env::var(WEIGHTS_ENV)
                .ok()
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| h.join(".nsynth_meta_weights.tsv")),
            scratch: h.join(format!(".nsynth_meta_rsi_scratch.{pid}.tsv")),
            snapshot_dir: h.join(".nsynth_meta_rsi_snapshots"),
            lock: h.join(".nsynth_rsi.lock"),
        }
    }
}

/// Single-writer lock: creates the lockfile exclusively, removes it on drop.
struct LockFile(PathBuf);
impl LockFile {
    fn acquire(path: &Path) -> Result<Self, String> {
        use std::fs::OpenOptions;
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(_) => Ok(LockFile(path.to_path_buf())),
            Err(e) => Err(format!(
                "RSI lock {} held or unwritable ({e}); another run in progress?",
                path.display()
            )),
        }
    }
}
impl Drop for LockFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Append-only JSONL audit sink (no-op when no path is given).
struct Audit {
    path: Option<PathBuf>,
}
impl Audit {
    fn log(&mut self, event: &str) {
        if let Some(p) = &self.path {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
                let _ = writeln!(f, "{event}");
            }
        }
    }
}

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Full report of a run.
#[derive(Clone, Debug)]
pub struct RunReport {
    pub outcome: Outcome,
    pub committed: bool,
    pub snapshot: PathBuf,
}

/// Run bounded RSI over the default benchmark suite and default paths.
pub fn run(cfg: &Config, commit: bool, audit_log: Option<&Path>) -> Result<RunReport, String> {
    let problems = crate::benchmark::get_benchmark(1);
    run_with(cfg, commit, audit_log, &Paths::defaults(), &problems)
}

/// Run bounded RSI with explicit paths and problem set (testable seam).
pub fn run_with(
    cfg: &Config,
    commit: bool,
    audit_log: Option<&Path>,
    paths: &Paths,
    problems: &[Problem],
) -> Result<RunReport, String> {
    let _lock = LockFile::acquire(&paths.lock)?;
    let mut audit = Audit { path: audit_log.map(|p| p.to_path_buf()) };

    // Capture the incumbent (loaded from the current/production weights path).
    let incumbent = crate::meta_learner::current_weights();

    // Pre-run snapshot — an out-of-process rollback artifact.
    std::fs::create_dir_all(&paths.snapshot_dir)
        .map_err(|e| format!("snapshot dir: {e}"))?;
    let snapshot = paths.snapshot_dir.join(format!("meta_weights_{}.tsv", unix_secs()));
    write_weights_tsv(&snapshot, &incumbent)?;

    // Repoint to scratch so nothing touches production until commit.
    std::env::set_var(WEIGHTS_ENV, &paths.scratch);
    crate::meta_learner::set_weights(incumbent.clone())?;

    let started = Instant::now();
    let baseline = evaluate_current(problems);
    audit.log(&format!(
        "RUN_START seed={} problems={} baseline_solved={}",
        cfg.seed, baseline.total, baseline.solved
    ));

    let outcome = hill_climb(
        cfg,
        incumbent.clone(),
        baseline,
        started,
        |w| {
            let _ = crate::meta_learner::set_weights(w.clone());
            evaluate_current(problems)
        },
        |w| {
            let _ = crate::meta_learner::set_weights(w.clone());
        },
        |e| audit.log(&e),
    );

    // Commit only on explicit request AND a genuine improvement.
    let committed = if commit && outcome.improved {
        std::env::set_var(WEIGHTS_ENV, &paths.production);
        crate::meta_learner::set_weights(outcome.winner.clone())?;
        audit.log(&format!(
            "COMMIT from_solved={} to_solved={} accepted={}",
            outcome.baseline_solved, outcome.final_solved, outcome.accepted
        ));
        true
    } else {
        audit.log(&format!(
            "NO_CHANGE improved={} committed=false (dry_run_or_no_gain)",
            outcome.improved
        ));
        false
    };

    // Restore the live singleton + env to production so the process continues
    // with production weights, and clean up scratch.
    std::env::set_var(WEIGHTS_ENV, &paths.production);
    let _ = crate::meta_learner::set_weights(crate::meta_learner::MetaWeights::load());
    let _ = std::fs::remove_file(&paths.scratch);

    Ok(RunReport { outcome, committed, snapshot })
}

/// Persist a weight vector to `path` in the same `%.6` TSV format `save()` uses,
/// atomically (temp + rename).
fn write_weights_tsv(path: &Path, w: &MetaWeights) -> Result<(), String> {
    let mut s = String::new();
    for (i, v) in w.w.iter().enumerate() {
        if i > 0 {
            s.push('\t');
        }
        s.push_str(&format!("{v:.6}"));
    }
    s.push('\n');
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::write(&tmp, &s).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename snapshot: {e}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit(names: &[&str], total: usize) -> Fitness {
        let pass_set: BTreeSet<String> = names.iter().map(|s| s.to_string()).collect();
        Fitness { solved: pass_set.len(), total, pass_set }
    }

    #[test]
    fn gate_accepts_strict_superset_only() {
        let base = fit(&["a", "b"], 3);
        assert!(gate(&base, &fit(&["a", "b", "c"], 3)), "superset accepts");
        assert!(!gate(&base, &fit(&["a", "c"], 3)), "swap rejects (regression)");
        assert!(!gate(&base, &fit(&["a", "b"], 3)), "equal rejects");
        assert!(!gate(&base, &fit(&["a"], 3)), "subset rejects");
        // Equal count but extra-and-missing => not a superset => reject.
        assert!(!gate(&fit(&["a", "b"], 3), &fit(&["b", "c"], 3)));
    }

    #[test]
    fn propose_stays_in_bounds_and_is_deterministic() {
        let incumbent = MetaWeights { w: [1.0; FEATURE_DIM] };
        let mut r1 = XorShift::new(42);
        let mut r2 = XorShift::new(42);
        let a = propose(&incumbent, &mut r1, 0.5, 0.2);
        let b = propose(&incumbent, &mut r2, 0.5, 0.2);
        assert_eq!(a.w, b.w, "same seed => same proposal");
        for v in a.w {
            assert!((0.01..=100.0).contains(&v), "clamped to bounds: {v}");
        }
    }

    #[test]
    fn hill_climb_accepts_improvement_and_restores_on_reject() {
        // Stub evaluator: the FIRST proposed candidate improves (adds "c"),
        // every later candidate regresses (drops "a"). Verifies one accept then
        // restores; counts restore calls on rejects.
        let cfg = Config { max_iters: 4, max_no_improve: 10, ..Config::default() };
        let incumbent = MetaWeights { w: [1.0; FEATURE_DIM] };
        let baseline = fit(&["a", "b"], 3);
        let mut calls = 0;
        let mut restores = 0;
        let outcome = hill_climb(
            &cfg,
            incumbent,
            baseline,
            Instant::now(),
            |_w| {
                calls += 1;
                if calls == 1 {
                    fit(&["a", "b", "c"], 3) // improvement
                } else {
                    fit(&["b", "c"], 3) // dropped "a" => regression
                }
            },
            |_w| restores += 1,
            |_e| {},
        );
        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.final_solved, 3);
        assert!(outcome.improved);
        // iters 2..=4 reject and restore (3 restores).
        assert_eq!(restores, 3);
        assert_eq!(outcome.iterations, 4);
    }

    use crate::benchmark::{Example, Value};

    fn scalar_problem(name: &str, sig: &'static str, ex: Vec<(i64, i64)>) -> Problem {
        Problem {
            name: name.to_string(),
            category: "arithmetic",
            description: "",
            signature: sig,
            examples: ex
                .into_iter()
                .map(|(i, o)| Example { inputs: vec![Value::Int(i)], expected: Value::Int(o) })
                .collect(),
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

    #[test]
    fn run_with_dry_run_leaves_production_untouched_and_releases_lock() {
        crate::meta_learner::with_test_lock(|| {
            crate::solved_cache::with_test_lock(|| {
                let dir = std::env::temp_dir().join(format!(
                    "nsynth_rsi_test_{}_{:?}",
                    std::process::id(),
                    std::thread::current().id()
                ));
                let _ = std::fs::create_dir_all(&dir);
                let prod = dir.join("prod.tsv");
                let paths = Paths {
                    production: prod.clone(),
                    scratch: dir.join("scratch.tsv"),
                    snapshot_dir: dir.join("snaps"),
                    lock: dir.join("lock"),
                };

                // Point the weights API at the temp production file and seed it.
                std::env::set_var(WEIGHTS_ENV, &prod);
                crate::meta_learner::reset_for_tests();
                crate::meta_learner::set_weights(MetaWeights { w: [1.0; FEATURE_DIM] }).unwrap();
                let prod_before = std::fs::read_to_string(&prod).unwrap();

                let problems = vec![
                    scalar_problem("dbl", "fn dbl(n: i64) -> i64", vec![(1, 2), (2, 4), (3, 6)]),
                    scalar_problem("inc", "fn inc(n: i64) -> i64", vec![(1, 2), (2, 3), (5, 6)]),
                ];
                let cfg = Config {
                    max_iters: 3,
                    max_no_improve: 5,
                    deadline: Duration::from_secs(60),
                    ..Config::default()
                };

                let report = run_with(&cfg, /*commit=*/ false, None, &paths, &problems).unwrap();

                assert!(!report.committed, "dry run must not commit");
                assert_eq!(report.outcome.total, 2);
                // Production weights unchanged by a dry run (scratch isolation).
                let prod_after = std::fs::read_to_string(&prod).unwrap();
                assert_eq!(prod_before, prod_after, "production weights must be untouched");
                // Lock released; a snapshot was written.
                assert!(!paths.lock.exists(), "lock released on drop");
                assert!(report.snapshot.exists(), "pre-run snapshot written");

                std::env::remove_var(WEIGHTS_ENV);
                crate::meta_learner::reset_for_tests();
                let _ = std::fs::remove_dir_all(&dir);
            });
        });
    }

    #[test]
    fn run_with_refuses_concurrent_lock() {
        crate::meta_learner::with_test_lock(|| {
            crate::solved_cache::with_test_lock(|| {
                let dir = std::env::temp_dir().join(format!(
                    "nsynth_rsi_lock_{}_{:?}",
                    std::process::id(),
                    std::thread::current().id()
                ));
                let _ = std::fs::create_dir_all(&dir);
                let lock = dir.join("lock");
                std::fs::write(&lock, "held").unwrap();
                let paths = Paths {
                    production: dir.join("prod.tsv"),
                    scratch: dir.join("scratch.tsv"),
                    snapshot_dir: dir.join("snaps"),
                    lock: lock.clone(),
                };
                let err = run_with(&Config::default(), false, None, &paths, &[]);
                assert!(err.is_err(), "must refuse to start when lock is held");
                // Pre-existing lock not removed by the refused run.
                assert!(lock.exists());
                let _ = std::fs::remove_dir_all(&dir);
            });
        });
    }

    #[test]
    fn hill_climb_respects_patience_budget() {
        let cfg = Config { max_iters: 100, max_no_improve: 2, ..Config::default() };
        let incumbent = MetaWeights { w: [1.0; FEATURE_DIM] };
        let baseline = fit(&["a"], 2);
        let mut iters = 0;
        let outcome = hill_climb(
            &cfg,
            incumbent,
            baseline,
            Instant::now(),
            |_w| {
                iters += 1;
                fit(&["a"], 2) // never improves
            },
            |_w| {},
            |_e| {},
        );
        // Stops after max_no_improve consecutive rejects.
        assert_eq!(outcome.iterations, 2);
        assert_eq!(outcome.accepted, 0);
        assert!(!outcome.improved);
    }
}
