//! Plugin registry for differentiable synthesis strategies.
//!
//! Each differentiable-program family or synthesis technique the solver knows about
//! (gradient-only, array gradient, expression-only, register machine, template
//! catalog, etc.) implements [`SynthesisStrategy`]. The default solver consults
//! them in priority order via [`default_strategies`] and returns the first
//! [`SolveResult`] whose `success` flag is true.
//!
//! Consumers who want to extend the solver — e.g. add a new program family without
//! modifying [`crate::solver`] — can build their own `Vec<Box<dyn SynthesisStrategy>>`
//! and iterate over it with [`run_strategies`].

use crate::benchmark::{Problem, Value};
use crate::solver::SolveResult;
use crate::synthesis;

/// A differentiable-program family or synthesis technique the solver can try.
///
/// Implementations should be zero-sized marker types when possible so the
/// dispatch table costs nothing beyond a vtable pointer per strategy.
pub trait SynthesisStrategy: Send + Sync {
    /// Stable identifier for logging and metrics. Must match the `method`
    /// string that the underlying synthesizer writes into its [`SolveResult`]
    /// so external tools can correlate them.
    fn name(&self) -> &'static str;

    /// Cheap input-shape check. Returning `false` skips this strategy entirely
    /// so the solver doesn't pay setup cost for obviously-inapplicable problems.
    fn applicable(&self, problem: &Problem) -> bool;

    /// Attempt synthesis. Return `None` on miss; the solver advances to the next.
    fn try_solve(&self, problem: &Problem) -> Option<SolveResult>;
}

// ─── Input-shape helpers shared by strategy gates ────────────────────────────

fn is_scalar_only(problem: &Problem) -> bool {
    problem
        .examples
        .iter()
        .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_))))
}

fn has_array_input(problem: &Problem) -> bool {
    problem
        .examples
        .first()
        .map(|ex| ex.inputs.iter().any(|v| matches!(v, Value::Array(_))))
        .unwrap_or(false)
}

fn n_args(problem: &Problem) -> usize {
    problem
        .examples
        .first()
        .map(|e| e.inputs.len())
        .unwrap_or(0)
}

fn external_multi_arg(problem: &Problem) -> bool {
    problem.category == "external" && n_args(problem) > 3
}

// ─── Concrete strategies ──────────────────────────────────────────────────────

/// Native scalar gradient stack. First non-enumerative attempt in the default
/// pipeline. Works for any problem whose inputs are all `Value::Int`.
pub struct GradientOnly;
impl SynthesisStrategy for GradientOnly {
    fn name(&self) -> &'static str {
        "gradient_only"
    }
    fn applicable(&self, p: &Problem) -> bool {
        is_scalar_only(p) && !external_multi_arg(p)
    }
    fn try_solve(&self, p: &Problem) -> Option<SolveResult> {
        synthesis::synthesize_gradient_only(p)
    }
}

/// Array-input gradient stack: enumerative fold, expression-only, the native
/// array ensemble, and the universal-array fallback.
pub struct ArrayGradient;
impl SynthesisStrategy for ArrayGradient {
    fn name(&self) -> &'static str {
        "array_gradient"
    }
    fn applicable(&self, p: &Problem) -> bool {
        has_array_input(p)
    }
    fn try_solve(&self, p: &Problem) -> Option<SolveResult> {
        synthesis::synthesize_array(p)
    }
}

/// Expression-only scalar synthesis. Faster than the full gradient stack — used
/// as a second pass for problems the first stage missed.
pub struct ScalarExprOnly;
impl SynthesisStrategy for ScalarExprOnly {
    fn name(&self) -> &'static str {
        "expr_only"
    }
    fn applicable(&self, p: &Problem) -> bool {
        is_scalar_only(p)
    }
    fn try_solve(&self, p: &Problem) -> Option<SolveResult> {
        synthesis::synthesize_scalar_expr_only(p)
    }
}

/// Universal register machine. Can discover any scalar program with N_RM_STEPS
/// or fewer instructions; tried after the specialised program types.
pub struct RegisterMachine;
impl SynthesisStrategy for RegisterMachine {
    fn name(&self) -> &'static str {
        "register_machine"
    }
    fn applicable(&self, p: &Problem) -> bool {
        is_scalar_only(p)
    }
    fn try_solve(&self, p: &Problem) -> Option<SolveResult> {
        synthesis::synthesize_register_machine(p)
    }
}

/// Expression + loop templates fallback (tries templates without running
/// gradient descent).
pub struct ScalarExprTemplates;
impl SynthesisStrategy for ScalarExprTemplates {
    fn name(&self) -> &'static str {
        "expr_templates"
    }
    fn applicable(&self, _p: &Problem) -> bool {
        true
    }
    fn try_solve(&self, p: &Problem) -> Option<SolveResult> {
        synthesis::synthesize_scalar_expr_templates_only(p)
    }
}

/// Hand-written scalar template catalog. Last native-stack stage before search.
pub struct ScalarTemplates;
impl SynthesisStrategy for ScalarTemplates {
    fn name(&self) -> &'static str {
        "scalar_templates"
    }
    fn applicable(&self, p: &Problem) -> bool {
        is_scalar_only(p) && !external_multi_arg(p)
    }
    fn try_solve(&self, p: &Problem) -> Option<SolveResult> {
        synthesis::synthesize_scalar_templates_only(p)
    }
}

/// Fully emergent synthesis — no hand-designed program skeletons, no templates,
/// no reference code, no search. Routes through the two subsystems that are
/// genuinely learned end-to-end:
///   1. [`synthesis::synthesize_register_machine`] — discovers op, src, dst,
///      and gating wiring via gradient descent with no pre-defined structure.
///   2. [`synthesis::synthesize_universal_and_collect`] — the emergent-architecture
///      universal program; init/loop/post role of each slot emerges from training.
///
/// Use this to measure what the system solves with zero hand-engineered priors —
/// a moving coverage bar that should grow as the learned components improve.
pub struct PureEmergent;
impl SynthesisStrategy for PureEmergent {
    fn name(&self) -> &'static str {
        "pure_emergent"
    }
    fn applicable(&self, p: &Problem) -> bool {
        // Both emergent synthesizers are scalar-only today.
        is_scalar_only(p)
    }
    fn try_solve(&self, p: &Problem) -> Option<SolveResult> {
        // RM is cheaper (6 steps × 5 restarts); try it first.
        if let Some(result) = synthesis::synthesize_register_machine(p) {
            if result.success {
                return Some(result);
            }
        }
        // Cold-start universal program: ignore params, return only the result.
        if let Some((result, _params)) = synthesis::synthesize_universal_and_collect(p, 400) {
            if result.success {
                return Some(result);
            }
        }
        None
    }
}

/// Cross-problem knowledge transfer via the persistent solved cache.
///
/// Iterates previously-solved programs in [`crate::solved_cache`] and tries
/// each as a teacher for the current problem via
/// [`synthesis::synthesize_scalar_from_teacher`]. The teacher's code is run on
/// the current problem's inputs to produce extra training signal; gradient
/// descent then discovers a program that fits both the original and the
/// teacher-derived examples.
///
/// [`crate::meta_learner::rank_teachers`] sorts the snapshot by learned
/// weighted-L2 distance first, and the top K entries are attempted in order.
/// K defaults to [`DEFAULT_TEACHER_TOPK`] and can be overridden with the
/// `NSYNTH_TEACHER_TOPK` environment variable — setting it to `0` disables
/// the cap entirely (revert to exhaustive iteration). This keeps per-problem
/// cost bounded as the cache grows: N teachers × O(gradient descent) is
/// replaced by K × O(gradient descent).
///
/// No hand-tuned retrieval, no template matching — the ranker *learns* which
/// features predict successful transfer from the cross-run solve log. Every
/// successful solve becomes a building block for the next one.
pub struct CachedTeachers;

/// Number of top-ranked teachers tried per problem when `NSYNTH_TEACHER_TOPK`
/// is unset. The measured Pareto-optimal value on the benchmark cache:
/// `tools/diversity_pareto.sh` showed K=48 dominates K=0 (same 80% win rate,
/// 30% lower mean wall-clock). See `artifacts/diversity_pareto.md` for the
/// sweep data.
///
/// Re-measure the sweep after any ranker change (distance formula, feature
/// set, training pass) and bump this constant when the optimum moves. The
/// autotune script (`tools/autotune_topk.sh`) reads the Pareto CSV and
/// writes the winner to `tools/config/nsynth_autotune.tsv`, which
/// `teacher_topk()` also consults — so production uses the latest measured
/// best without editing this constant.
pub const DEFAULT_TEACHER_TOPK: usize = 48;

fn teacher_topk() -> usize {
    // Resolution order, first hit wins:
    //   1. NSYNTH_TEACHER_TOPK env var (explicit override)
    //   2. tools/config/nsynth_autotune.tsv "topk" entry (measured winner)
    //   3. DEFAULT_TEACHER_TOPK constant
    //
    // Environment always wins so humans and CI can force a value; the
    // config file expresses the last Pareto measurement; the constant is
    // the hard-coded fallback.
    if let Ok(raw) = std::env::var("NSYNTH_TEACHER_TOPK") {
        if let Ok(v) = raw.parse::<usize>() {
            return v;
        }
    }
    if let Some(v) = teacher_topk_from_config() {
        return v;
    }
    DEFAULT_TEACHER_TOPK
}

/// Read the `topk` value from the autotune config file (`tsv`, lines of
/// `key\tvalue`). Override the path with `NSYNTH_AUTOTUNE_CONFIG`. Returns
/// `None` on any failure — missing file, unparseable value, etc. —
/// because this lookup is advisory and the env var / constant fall
/// through safely.
fn teacher_topk_from_config() -> Option<usize> {
    let path = std::env::var("NSYNTH_AUTOTUNE_CONFIG")
        .unwrap_or_else(|_| "tools/config/nsynth_autotune.tsv".to_string());
    let raw = std::fs::read_to_string(&path).ok()?;
    for line in raw.lines() {
        let mut parts = line.splitn(2, '\t');
        let key = parts.next()?.trim();
        let val = parts.next()?.trim();
        if key == "topk" {
            return val.parse::<usize>().ok();
        }
    }
    None
}

/// Default wall-clock cap for a single `CachedTeachers::try_solve` call.
/// Keeps trivial problems cheap: even when the cache has hundreds of
/// entries, a pipeline call-site can't be stuck in teacher-distillation for
/// longer than this budget.
pub const DEFAULT_TEACHER_BUDGET_SEC: f32 = 15.0;

fn teacher_budget_sec() -> f32 {
    match std::env::var("NSYNTH_TEACHER_BUDGET_SEC") {
        Ok(raw) => raw.parse::<f32>().unwrap_or(DEFAULT_TEACHER_BUDGET_SEC),
        Err(_) => DEFAULT_TEACHER_BUDGET_SEC,
    }
}

impl SynthesisStrategy for CachedTeachers {
    fn name(&self) -> &'static str {
        "cached_teachers"
    }
    fn applicable(&self, p: &Problem) -> bool {
        // synthesize_scalar_from_teacher requires scalar inputs.
        is_scalar_only(p)
    }
    fn try_solve(&self, p: &Problem) -> Option<SolveResult> {
        let snapshot = crate::solved_cache::snapshot_solutions_with_meta();
        if snapshot.is_empty() {
            return None;
        }
        // Pass the top-K through the diversity pass so a cache dominated by
        // one program family doesn't starve out rare-but-relevant teachers
        // in the rank head.
        let k = teacher_topk();
        let ranked = crate::meta_learner::rank_teachers_with_meta_topk(p, snapshot, k);
        // K = 0 disables the cap (exhaustive iteration, useful for debugging
        // and for the emergent_coverage benchmark where we want to measure
        // full transfer capacity without budget limits).
        let budget = teacher_budget_sec();
        let t0 = std::time::Instant::now();
        let effective_iter: Vec<(f64, String, String, u32, u64)> = if k == 0 {
            ranked
        } else {
            ranked.into_iter().take(k).collect()
        };

        // Capture the teachers we're about to try so we can attribute a
        // whole-stage miss back to them in artifacts/transfer_failures.jsonl
        // (opt-in). Cloning up to K small (method, code) pairs is cheap
        // compared to the gradient work about to happen.
        let mut attempted: Vec<(String, String)> = Vec::new();

        for (_dist, method, code, _success_count, _last_used) in effective_iter {
            // Wall-clock gate: check *before* starting the next teacher round,
            // not during, so a single over-budget distillation can't cut off
            // mid-gradient (the caller expects a clean `Option<SolveResult>`).
            if budget > 0.0 && t0.elapsed().as_secs_f32() >= budget {
                log_teacher_miss(p, &attempted, "budget_exceeded");
                return None;
            }
            attempted.push((method.clone(), code.clone()));
            if let Some(mut result) = synthesis::synthesize_scalar_from_teacher(p, &code) {
                if result.success {
                    // Online update #1: reinforce the feature weights that
                    // pointed the ranker at the winning teacher.
                    crate::meta_learner::record_transfer_success(p, &code);
                    // Online update #2: credit the cache entry itself so the
                    // success_count reward in rank_teachers_with_meta nudges
                    // this teacher toward the top next time.
                    crate::solved_cache::note_transfer_success(&method, &code);
                    // Tag the method with a `cached_teachers:` prefix so
                    // downstream telemetry (transfer_curve, method_router)
                    // can distinguish Stage 0.5 transfer wins from wins
                    // produced by any other synthesis stage. The underlying
                    // method name is preserved after the colon.
                    result.method = format!("cached_teachers:{}", result.method);
                    return Some(result);
                }
            }
        }
        log_teacher_miss(p, &attempted, "all_teachers_missed");
        None
    }
}

/// Opt-in miss attribution. When `NSYNTH_LOG_TEACHER_FAILURES=1` and the
/// strategy exhausts its top-K without a win, append one JSONL row to
/// `artifacts/transfer_failures.jsonl` so downstream analysis can cluster
/// the misses and answer "what program family is the system
/// nearly-but-not-quite able to solve?" — a direct prioritization signal
/// for solver development.
///
/// Quiet when the env var is unset (default): logging every miss on a
/// real-world bench would swamp the artifacts dir.
fn log_teacher_miss(problem: &Problem, attempted: &[(String, String)], reason: &str) {
    if std::env::var("NSYNTH_LOG_TEACHER_FAILURES").as_deref() != Ok("1") {
        return;
    }
    let path = std::env::var("NSYNTH_TEACHER_FAILURES_PATH")
        .unwrap_or_else(|_| "artifacts/transfer_failures.jsonl".to_string());
    if let Some(parent) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Ok(mut file) = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)
    else {
        return;
    };
    use std::io::Write;
    let mut preview = String::new();
    preview.push('[');
    for (i, (method, code)) in attempted.iter().take(3).enumerate() {
        if i > 0 {
            preview.push(',');
        }
        let first_line: String = code
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .chars()
            .take(80)
            .collect();
        preview.push_str(&format!(
            r#"{{"method":"{}","preview":"{}"}}"#,
            json_escape_simple(method),
            json_escape_simple(&first_line),
        ));
    }
    preview.push(']');
    let n_args = problem
        .examples
        .first()
        .map(|ex| ex.inputs.len())
        .unwrap_or(0);
    let row = format!(
        r#"{{"problem":"{}","n_args":{},"n_examples":{},"n_attempted":{},"reason":"{}","attempted":{}}}"#,
        json_escape_simple(&problem.name),
        n_args,
        problem.examples.len(),
        attempted.len(),
        reason,
        preview,
    );
    let _ = writeln!(file, "{}", row);
}

fn json_escape_simple(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Strategy list for the fully-emergent evaluation mode (no hand-designed
/// skeletons or templates). Use this with [`run_strategies`] to measure what
/// the learned subsystems achieve on their own.
///
/// Includes [`CachedTeachers`] so prior solves act as warm starts via gradient
/// distillation — the cross-run learning loop the user mandated. The cache is
/// the system's "memory"; gradient descent is the "transfer mechanism".
pub fn emergent_only_strategies() -> Vec<Box<dyn SynthesisStrategy>> {
    vec![Box::new(CachedTeachers), Box::new(PureEmergent)]
}

/// Ordered list of strategies the default solver consults for its native
/// differentiable stack (between enumerative and search). Exposed so callers
/// can build custom pipelines by prepending or appending their own strategies.
///
/// ```ignore
/// use mog_synth::strategy::{default_strategies, run_strategies};
/// let result = run_strategies(&default_strategies(), &problem);
/// ```
pub fn default_strategies() -> Vec<Box<dyn SynthesisStrategy>> {
    vec![
        Box::new(GradientOnly),
        Box::new(ArrayGradient),
        Box::new(ScalarExprOnly),
        Box::new(RegisterMachine),
        Box::new(ScalarExprTemplates),
        Box::new(ScalarTemplates),
    ]
}

/// Iterate a strategy list and return the first successful [`SolveResult`].
/// Inapplicable strategies are skipped; applicable ones that return `None` or
/// an unsuccessful result advance to the next entry.
pub fn run_strategies(
    strategies: &[Box<dyn SynthesisStrategy>],
    problem: &Problem,
) -> Option<SolveResult> {
    for strat in strategies {
        if !strat.applicable(problem) {
            continue;
        }
        if let Some(result) = strat.try_solve(problem) {
            if result.success {
                return Some(result);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value};

    /// Smoke test: default pipeline solves a trivial two-arg add problem.
    #[test]
    fn default_strategies_solve_add_two() {
        let problem = Problem {
            name: "add_two_v0".to_string(),
            category: "test",
            description: "sum two ints",
            signature: "fn add_two(a: i64, b: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(1), Value::Int(2)],
                    expected: 3,
                },
                Example {
                    inputs: vec![Value::Int(10), Value::Int(-4)],
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Int(-3), Value::Int(-2)],
                    expected: -5,
                },
                Example {
                    inputs: vec![Value::Int(0), Value::Int(0)],
                    expected: 0,
                },
            ],
            holdouts: vec![],
            reference_code: "",
        };
        let strats = default_strategies();
        let result = run_strategies(&strats, &problem).expect("should solve add_two");
        assert!(result.success, "expected success, got {result:?}");
        // First applicable strategy for a pure-scalar problem is gradient_only.
        assert!(
            !result.method.is_empty(),
            "method should be populated by the winning synthesizer"
        );
    }

    /// Verify applicable() gates route array problems away from scalar strategies.
    #[test]
    fn array_problem_skips_scalar_strategies() {
        let problem = Problem {
            name: "array_sum_v0".to_string(),
            category: "test",
            description: "sum an array",
            signature: "fn array_sum(xs: Vec<i64>) -> i64",
            examples: vec![Example {
                inputs: vec![Value::Array(vec![1, 2, 3])],
                expected: 6,
            }],
            holdouts: vec![],
            reference_code: "",
        };
        assert!(!GradientOnly.applicable(&problem));
        assert!(!ScalarExprOnly.applicable(&problem));
        assert!(!RegisterMachine.applicable(&problem));
        assert!(!ScalarTemplates.applicable(&problem));
        assert!(ArrayGradient.applicable(&problem));
    }

    /// CachedTeachers: wiring smoke test. Verifies applicability gating and
    /// that the strategy is included in the emergent-only pipeline. Actual
    /// transfer behaviour (cache → teacher → gradient distill) is exercised by
    /// integration tests on the cache module itself.
    #[test]
    fn cached_teachers_wiring() {
        let scalar = Problem {
            name: "scalar_v0".to_string(),
            category: "test",
            description: "scalar",
            signature: "fn f(a: i64) -> i64",
            examples: vec![Example {
                inputs: vec![Value::Int(1)],
                expected: 1,
            }],
            holdouts: vec![],
            reference_code: "",
        };
        assert!(CachedTeachers.applicable(&scalar));
        assert_eq!(CachedTeachers.name(), "cached_teachers");

        let strats = emergent_only_strategies();
        assert_eq!(strats.len(), 2);
        assert_eq!(strats[0].name(), "cached_teachers");
        assert_eq!(strats[1].name(), "pure_emergent");
    }

    /// PureEmergent: wiring smoke test. Verifies the strategy passes the
    /// applicable() gate for scalars, rejects array-shaped inputs, and that
    /// `emergent_only_strategies()` returns a non-empty list with PureEmergent.
    /// Full synthesis coverage is measured by the corpus_harvest binary, not
    /// unit tests — gradient descent is too slow to embed here.
    #[test]
    fn pure_emergent_wiring() {
        let scalar = Problem {
            name: "scalar_v0".to_string(),
            category: "test",
            description: "scalar",
            signature: "fn f(a: i64) -> i64",
            examples: vec![Example {
                inputs: vec![Value::Int(1)],
                expected: 1,
            }],
            holdouts: vec![],
            reference_code: "",
        };
        assert!(PureEmergent.applicable(&scalar));
        assert_eq!(PureEmergent.name(), "pure_emergent");

        let array = Problem {
            name: "arr_v0".to_string(),
            category: "test",
            description: "arr",
            signature: "fn f(xs: Vec<i64>) -> i64",
            examples: vec![Example {
                inputs: vec![Value::Array(vec![1, 2])],
                expected: 3,
            }],
            holdouts: vec![],
            reference_code: "",
        };
        assert!(!PureEmergent.applicable(&array));

        let strats = emergent_only_strategies();
        assert!(strats.iter().any(|s| s.name() == "pure_emergent"));
    }

    /// The top-K cap is configurable via `NSYNTH_TEACHER_TOPK` and defaults
    /// to [`DEFAULT_TEACHER_TOPK`]. Setting it to `0` means "no cap" so the
    /// emergent-coverage benchmark can still measure full transfer capacity.
    #[test]
    fn teacher_topk_reads_env_override() {
        // Save + clear env before each assertion to avoid test-order flakiness.
        let prev = std::env::var("NSYNTH_TEACHER_TOPK").ok();

        // SAFETY: set_var is only unsafe under concurrent reads from other
        // threads; this test is single-threaded within its own scope.
        unsafe { std::env::remove_var("NSYNTH_TEACHER_TOPK") };
        assert_eq!(teacher_topk(), DEFAULT_TEACHER_TOPK);

        unsafe { std::env::set_var("NSYNTH_TEACHER_TOPK", "3") };
        assert_eq!(teacher_topk(), 3);

        unsafe { std::env::set_var("NSYNTH_TEACHER_TOPK", "0") };
        assert_eq!(teacher_topk(), 0);

        // Invalid parse → fall back to default rather than panic. Keeps a
        // typo in an env var from bringing down the whole solver.
        unsafe { std::env::set_var("NSYNTH_TEACHER_TOPK", "not_a_number") };
        assert_eq!(teacher_topk(), DEFAULT_TEACHER_TOPK);

        // Restore.
        match prev {
            Some(v) => unsafe { std::env::set_var("NSYNTH_TEACHER_TOPK", v) },
            None => unsafe { std::env::remove_var("NSYNTH_TEACHER_TOPK") },
        }
    }

    /// Verify external multi-arg problems skip the scalar gradient stack.
    #[test]
    fn external_multi_arg_skips_scalar_gradient() {
        let inputs: Vec<Value> = (0..5).map(Value::Int).collect();
        let problem = Problem {
            name: "big_scalar".to_string(),
            category: "external",
            description: "external 5-arg scalar",
            signature: "fn big_scalar(a: i64, b: i64, c: i64, d: i64, e: i64) -> i64",
            examples: vec![Example {
                inputs,
                expected: 0,
            }],
            holdouts: vec![],
            reference_code: "",
        };
        assert!(!GradientOnly.applicable(&problem));
        assert!(!ScalarTemplates.applicable(&problem));
        // Expression-only + register machine do not have the external guard.
        assert!(ScalarExprOnly.applicable(&problem));
        assert!(RegisterMachine.applicable(&problem));
    }
}
