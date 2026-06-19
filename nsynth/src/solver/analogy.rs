//! Analogy-driven synthesis — Phase 3.2.
//!
//! `analogy_solve` makes the A:B :: C:D transfer explicit: donor problem *A* was
//! solved by program *B*; a new problem *C* seeks program *D*. We find the
//! nearest donors to *C* in the [`CodeKnowledgeGraph`], adapt each donor *B* to
//! *C*'s context, re-fit a native program, and **emit only what the verifier
//! accepts**. Nothing here can fabricate code:
//!
//! * The transfer mechanic is [`crate::synthesis::synthesize_scalar_from_teacher`],
//!   which re-synthesizes a *native* program for the query and returns `Some`
//!   only after an internal [`crate::runtime::verify_problem_code_strict`]
//!   against the **original** problem (not the teacher-augmented example set).
//! * `analogy_solve` re-asserts `verify_problem_code_strict` defensively before
//!   emitting, and only on `result.success` (mirroring `CachedTeachers`).
//! * If every donor fails, it returns `None` — no code is emitted.
//!
//! The genuinely new capability over `CachedTeachers` is twofold: (1) it is an
//! explicit analogy operator over a knowledge-graph donor index, and (2) it
//! performs **cross-name transfer** — a donor that solved a differently-*named*
//! function can still seed the query, because we rename the donor's function to
//! the query's name so its behavior is callable for sampling. `CachedTeachers`
//! silently skips name-mismatched donors (the teacher executes under the query's
//! function name and a name mismatch makes the donor un-callable).

use super::SolveResult;
use crate::benchmark::{Example, Problem, Value};
use std::cell::Cell;
use std::time::Instant;

/// Hard wall-clock budget for a single universal re-fit re-solve. A re-fit feeds
/// teacher-augmented examples (possibly contradictory, for a wrong donor) back
/// through `solve_problem`; several post-enumerative routes are individually
/// expensive (e.g. `register_machine` can run ~100s, gradient stages are
/// effectively unbounded on non-converging data). Summed, a re-fit could stall
/// for minutes. The route dispatch checks this budget at each route entry and
/// bails, so a re-fit can overshoot by at most one already-started route. A
/// genuine transfer almost always lands via the cheap routes (direct rename,
/// enumerative, early teachers) well inside this window.
const REFIT_BUDGET_SECS: f32 = 15.0;

thread_local! {
    /// Set while the universal re-fitter is re-entering `solve_problem` on a
    /// teacher-augmented problem, so the pipeline's analogy stage skips itself
    /// (prevents unbounded recursion: analogy → re-fit → solve → analogy → …).
    static IN_ANALOGY_REFIT: Cell<bool> = const { Cell::new(false) };
    /// Start instant of the in-flight re-fit re-solve, used to enforce
    /// [`REFIT_BUDGET_SECS`]. `None` outside a re-fit.
    static REFIT_START: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// True while a universal re-fit re-solve is in progress. The pipeline gates the
/// analogy stage on `!in_refit()`.
pub(crate) fn in_refit() -> bool {
    IN_ANALOGY_REFIT.with(|c| c.get())
}

/// True when an in-flight re-fit re-solve has exceeded [`REFIT_BUDGET_SECS`].
/// Always false outside a re-fit, so top-level solves are never time-capped here.
pub(crate) fn refit_budget_exhausted() -> bool {
    REFIT_START.with(|c| c.get()).is_some_and(|t| t.elapsed().as_secs_f32() > REFIT_BUDGET_SECS)
}

/// Rename the donor program's entry function (and all of its references, e.g.
/// recursion or call sites) to `query_fn`, so the donor is callable under the
/// name the teacher-sampling path expects. Returns the donor unchanged when its
/// entry is already named `query_fn`, or `None` if no `fn <name>(` declaration
/// is found.
///
/// Robustness (hardened after adversarial review):
/// - The scan ignores `fn` tokens inside `//`/`/* */` comments and `"…"` string
///   literals (a comment like `// helper fn foo` must not be mistaken for a
///   declaration), via [`code_mask`].
/// - For a multi-function donor (helpers + entry), the **entry is taken as the
///   last top-level `fn` declaration** — generated programs define helpers first
///   and the entry last. Renaming only the entry keeps helper calls valid.
/// - Replacement is whole-identifier and code-only (string-literal contents are
///   never rewritten), with ASCII identifier boundaries consistent with
///   [`is_ident`].
///
/// This is a narrow *identifier* rename, not a semantic edit: the renamed donor
/// is only ever used to *sample behavior*; the emitted program is the freshly
/// re-fitted native solution, which the verifier still gates. So the rename can
/// never introduce unverified code — at worst it perturbs sampling and the
/// candidate is rejected by the verifier (a missed transfer, never a fabrication).
fn remap_donor_to_query(donor_code: &str, query_fn: &str) -> Option<String> {
    let mask = code_mask(donor_code);
    let names = fn_decl_names(donor_code, &mask);
    let donor_fn = names.last()?; // entry = last declared top-level fn
    if !is_ident(donor_fn) {
        return None;
    }
    if donor_fn == query_fn {
        return Some(donor_code.to_string());
    }
    Some(replace_ident_masked(donor_code, &mask, donor_fn, query_fn))
}

/// True when `s` is a non-empty ASCII identifier (letter/`_` first, then
/// letters/digits/`_`). ASCII-only so it agrees with [`is_ident_byte`] on
/// boundaries — generated Rust uses only ASCII identifiers.
fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Per-byte mask: `true` = source code, `false` = inside a `//` or `/* */`
/// comment or a `"…"` string literal. Char literals / lifetimes are treated as
/// code (generated donor programs contain neither). Used so `fn` detection and
/// identifier renaming ignore comments and string contents.
fn code_mask(src: &str) -> Vec<bool> {
    let b = src.as_bytes();
    let n = b.len();
    let mut mask = vec![true; n];
    #[derive(Clone, Copy)]
    enum S {
        Code,
        Line,
        Block,
        Str,
    }
    let mut s = S::Code;
    let mut i = 0;
    while i < n {
        match s {
            S::Code => {
                if b[i] == b'/' && i + 1 < n && b[i + 1] == b'/' {
                    mask[i] = false;
                    mask[i + 1] = false;
                    s = S::Line;
                    i += 2;
                } else if b[i] == b'/' && i + 1 < n && b[i + 1] == b'*' {
                    mask[i] = false;
                    mask[i + 1] = false;
                    s = S::Block;
                    i += 2;
                } else if b[i] == b'"' {
                    mask[i] = false;
                    s = S::Str;
                    i += 1;
                } else {
                    i += 1;
                }
            }
            S::Line => {
                mask[i] = false;
                if b[i] == b'\n' {
                    s = S::Code;
                }
                i += 1;
            }
            S::Block => {
                mask[i] = false;
                if b[i] == b'*' && i + 1 < n && b[i + 1] == b'/' {
                    mask[i + 1] = false;
                    s = S::Code;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            S::Str => {
                mask[i] = false;
                if b[i] == b'\\' && i + 1 < n {
                    mask[i + 1] = false;
                    i += 2;
                } else {
                    if b[i] == b'"' {
                        s = S::Code;
                    }
                    i += 1;
                }
            }
        }
    }
    mask
}

/// Collect the names of top-level `fn <ident>(` declarations at code positions
/// (skipping comments/strings via `mask`), in source order.
fn fn_decl_names(src: &str, mask: &[bool]) -> Vec<String> {
    let b = src.as_bytes();
    let n = b.len();
    let mut names = Vec::new();
    let mut i = 0;
    while i + 2 <= n {
        let kw = b[i] == b'f'
            && b[i + 1] == b'n'
            && mask[i]
            && (i == 0 || !is_ident_byte(b[i - 1]))
            && i + 2 < n
            && (b[i + 2] == b' ' || b[i + 2] == b'\t');
        if kw {
            let mut j = i + 2;
            while j < n && (b[j] == b' ' || b[j] == b'\t') {
                j += 1;
            }
            let start = j;
            while j < n && is_ident_byte(b[j]) {
                j += 1;
            }
            if j > start {
                names.push(src[start..j].to_string());
            }
            i = j;
        } else {
            i += 1;
        }
    }
    names
}

/// Replace whole-identifier occurrences of `from` with `to`, only at code
/// positions (per `mask`) and only when the entire match span is code. Matches
/// flanked by identifier bytes are left alone (so `sum` inside `summer` and
/// `"sum"` inside a string literal are untouched).
fn replace_ident_masked(src: &str, mask: &[bool], from: &str, to: &str) -> String {
    let bytes = src.as_bytes();
    let n = src.len();
    let mut out = String::with_capacity(n);
    let mut i = 0;
    while i < n {
        if mask[i] && src[i..].starts_with(from) {
            let end = i + from.len();
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_ok = end >= n || !is_ident_byte(bytes[end]);
            let span_code = (i..end).all(|k| mask[k]);
            if before_ok && after_ok && span_code {
                out.push_str(to);
                i = end;
                continue;
            }
        }
        let ch = src[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Type-specific "re-fit from teacher" synthesizers, tried as a fallback when a
/// donor is *close* to the query but needs constant/structure adaptation. Each
/// self-guards (returns `None` when inapplicable to the problem's type) and
/// verifies internally; `analogy_solve` re-asserts the verifier regardless.
/// Add a new entry here to extend re-fit transfer to a new problem family.
/// Cheap, native (non-pipeline-re-entrant) type-specific re-fitters. Each
/// self-guards by type and verifies internally. Tried for every donor.
const CHEAP_REFITTERS: &[fn(&Problem, &str) -> Option<SolveResult>] = &[
    crate::synthesis::synthesize_scalar_from_teacher,
    crate::synthesis::synthesize_array_from_teacher,
];

/// Cap on universal re-fit attempts per `analogy_solve` call. The universal
/// re-fitter re-enters the full solver (a whole-portfolio solve), so it is tried
/// only for the most-similar donors that direct + cheap re-fit couldn't
/// transfer, and additionally gated by the wall-clock budget at the call site.
const MAX_UNIVERSAL_REFITS: usize = 2;

/// Type-aware perturbations of a single value — used to generate fresh inputs on
/// which to sample the donor, giving the re-solve new labeled data beyond the
/// query's own examples. Returns variants of the SAME `Value` shape (so arity
/// and types are preserved when substituted into an input row).
fn perturb_value(v: &Value) -> Vec<Value> {
    match v {
        Value::Int(n) => vec![
            Value::Int(n.wrapping_add(1)),
            Value::Int(n.wrapping_sub(1)),
            Value::Int(n.wrapping_mul(2)),
            Value::Int(n.wrapping_add(7)),
            Value::Int(0i64.wrapping_sub(*n)),
        ],
        Value::Float(bits) => {
            let f = f64::from_bits(*bits);
            vec![Value::Float((f + 1.0).to_bits()), Value::Float((f * 2.0).to_bits())]
        }
        Value::Bool(b) => vec![Value::Bool(!b)],
        Value::Str(s) => {
            let mut doubled = s.clone();
            doubled.push_str(s);
            vec![Value::Str(doubled), Value::Str(String::new()), Value::Str(s.to_uppercase())]
        }
        Value::Array(a) => {
            let mut rev = a.clone();
            rev.reverse();
            let mut grown = a.clone();
            grown.push(a.first().copied().unwrap_or(0));
            let shrunk: Vec<i64> = if a.len() > 1 { a[..a.len() - 1].to_vec() } else { a.clone() };
            vec![Value::Array(rev), Value::Array(grown), Value::Array(shrunk)]
        }
        // Pair/Quad/Tree and any other shape: no synthetic perturbation. (Pair
        // inputs are only callable via Point/Rectangle-typed signatures, so a
        // generic perturbed Pair can't be sampled; rather than emit rows that
        // always fail to execute, produce none and rely on the donor sampled on
        // the query's own example inputs.)
        _ => vec![],
    }
}

/// Build fresh input rows by perturbing one position at a time across the
/// query's example inputs, capped to keep sampling cheap.
fn perturbed_input_rows(problem: &Problem) -> Vec<Vec<Value>> {
    const MAX_ROWS: usize = 12;
    let mut rows: Vec<Vec<Value>> = Vec::new();
    for example in &problem.examples {
        for pos in 0..example.inputs.len() {
            for variant in perturb_value(&example.inputs[pos]) {
                let mut row = example.inputs.clone();
                row[pos] = variant;
                if !rows.contains(&row) && row != example.inputs {
                    rows.push(row);
                    if rows.len() >= MAX_ROWS {
                        return rows;
                    }
                }
            }
        }
    }
    rows
}

/// Universal teacher re-fit for ANY program type. Samples the (already renamed)
/// donor on fresh perturbed inputs to harvest extra labeled examples, augments
/// the query, and re-solves with the full solver (analogy disabled via a
/// recursion guard). Emits only what `verify_problem_code_strict` accepts
/// against the ORIGINAL problem, so it cannot fabricate.
fn synthesize_universal_from_teacher(problem: &Problem, teacher_code: &str) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    let mut teacher_examples: Vec<Example> = Vec::new();
    for inputs in perturbed_input_rows(problem) {
        if let Ok(output) =
            crate::runtime::execute_function_for_problem(teacher_code, fn_name, &inputs, problem)
        {
            // The executor returns a `runtime::Value`; an `Example` needs a
            // `benchmark::Value`. Unsupported shapes (struct/enum/result) skip
            // the sample rather than fail the whole donor.
            if let Some(expected) = runtime_to_bench(&output) {
                teacher_examples.push(Example { inputs, expected });
            }
        }
    }
    if teacher_examples.is_empty() {
        return None;
    }

    let mut augmented = problem.clone();
    augmented.examples.extend(teacher_examples);

    // Re-solve with the analogy stage disabled so this cannot recurse. The RAII
    // guard restores the flag even if the re-solve panics. Contradictory
    // teacher-augmented examples (e.g. a wrong donor) can trip a panic deep in a
    // synthesizer, so the re-solve is isolated with `catch_unwind`: a panic
    // becomes a skipped donor, never a crashed top-level solve.
    let result = {
        let _guard = RefitGuard::enter();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::solver::solve_problem(&augmented)
        }))
        .ok()?
    };

    if result.success && crate::runtime::verify_problem_code_strict(problem, &result.code).is_ok() {
        Some(result)
    } else {
        None
    }
}

/// Convert an executor output (`runtime::Value`) into a benchmark `Value` for
/// use as an `Example::expected`. Returns `None` for shapes the benchmark value
/// model doesn't represent (struct/enum/result/non-int array elements).
fn runtime_to_bench(v: &crate::runtime::Value) -> Option<Value> {
    use crate::runtime::Value as R;
    match v {
        R::Int(i) => Some(Value::Int(*i)),
        // Reject non-finite floats: a NaN label never matches itself under the
        // verifier (NaN != NaN), so an augmented Example carrying NaN would make
        // the re-solve unsatisfiable. Skip the sample instead.
        R::Float(f) if f.is_finite() => Some(Value::Float(f.to_bits())),
        R::Float(_) => None,
        R::Bool(b) => Some(Value::Bool(*b)),
        R::Str(s) => Some(Value::Str(s.clone())),
        R::Pair(a, b) => Some(Value::Pair(*a, *b)),
        R::Quad(a, b, c, d) => Some(Value::Quad(*a, *b, *c, *d)),
        R::Array(items) => {
            let ints: Option<Vec<i64>> = items
                .iter()
                .map(|e| match e {
                    R::Int(i) => Some(*i),
                    _ => None,
                })
                .collect();
            ints.map(Value::Array)
        }
        _ => None,
    }
}

/// RAII guard: sets the analogy-refit flag on `enter`, restores its PRIOR value
/// on drop (panic-safe, and nesting-safe regardless of caller).
struct RefitGuard {
    prev: bool,
    prev_start: Option<Instant>,
}
impl RefitGuard {
    fn enter() -> Self {
        let prev = IN_ANALOGY_REFIT.with(|c| c.replace(true));
        let prev_start = REFIT_START.with(|c| c.replace(Some(Instant::now())));
        RefitGuard { prev, prev_start }
    }
}
impl Drop for RefitGuard {
    fn drop(&mut self) {
        IN_ANALOGY_REFIT.with(|c| c.set(self.prev));
        REFIT_START.with(|c| c.set(self.prev_start));
    }
}

/// Record transfer credit for a winning donor — but only when the query has
/// holdouts. With empty holdouts the strict verifier degenerates to
/// example-matching and cannot detect an overfit, so crediting the ranker/cache
/// on that weak evidence could poison future routing.
fn credit_transfer(problem: &Problem, donor: &crate::knowledge::DonorNode) {
    if !problem.holdouts.is_empty() {
        crate::meta_learner::record_transfer_success(problem, &donor.code);
        crate::solved_cache::note_transfer_success(&donor.method, &donor.code);
    }
}

fn make_result(code: String, method: String) -> SolveResult {
    SolveResult {
        success: true,
        code,
        method,
        error: None,
        metadata: Default::default(),
    }
}

/// Attempt analogy-driven transfer for `problem` — for **any** problem/donor
/// type. Returns a verified `SolveResult` (method tagged `analogy:<kind>:<inner>`)
/// on the first donor that transfers, or `None` if no donor transfers.
///
/// Two transfer mechanisms, tried per donor in order of cost:
///
/// 1. **Universal direct transfer.** Rename the donor's entry function to the
///    query's name and verify the renamed donor *as-is* against the query. This
///    is fully type-agnostic — `verify_problem_code_strict` handles
///    Int/Array/Tree/Str/Float/Pair/Quad — so a donor that solves a
///    structurally-identical but differently-*named* problem of ANY type
///    transfers here. Incompatible donors (wrong arity/types) simply fail
///    verification and are skipped (a missed transfer, never a fabrication).
/// 2. **Type-specific re-fit fallback** ([`REFITTERS`]) for donors that are
///    *close* but need adaptation (constant re-fitting via teacher distillation).
///
/// Non-fabrication: every emit path is gated by `verify_problem_code_strict`
/// against the ORIGINAL problem; nothing unverified is ever returned.
pub fn analogy_solve(problem: &Problem) -> Option<SolveResult> {
    let kg = crate::knowledge::CodeKnowledgeGraph::build_from_cache();
    if kg.is_empty() {
        return None;
    }

    let k = crate::strategy::teacher_topk();
    let donors = kg.nearest_donors(problem, k);
    let budget = crate::strategy::teacher_budget_sec();
    let t0 = std::time::Instant::now();
    let query_fn = problem.function_name().to_string();
    let mut universal_attempts = 0usize;

    for donor in donors {
        // Wall-clock gate, checked before each donor (a single over-budget
        // distillation can't cut off mid-gradient).
        if budget > 0.0 && t0.elapsed().as_secs_f32() >= budget {
            break;
        }

        // Cross-name transfer: make the donor callable under the query's name.
        let Some(adapted) = remap_donor_to_query(&donor.code, &query_fn) else {
            continue;
        };

        // (1) Universal direct transfer: the renamed donor itself, verified
        // against the query. Type-agnostic; the cheapest possible transfer.
        if crate::runtime::verify_problem_code_strict(problem, &adapted).is_ok() {
            credit_transfer(problem, &donor);
            return Some(make_result(
                adapted,
                format!("analogy:direct:{}", donor.method),
            ));
        }

        // (2) Cheap type-specific re-fit. Each self-guards by type and verifies
        // internally; we re-assert against the ORIGINAL problem (real examples +
        // holdouts), never the teacher-augmented set, and only accept a
        // `success` result (mirroring the trusted CachedTeachers gate).
        for refit in CHEAP_REFITTERS {
            if let Some(mut result) = refit(problem, &adapted) {
                if result.success
                    && crate::runtime::verify_problem_code_strict(problem, &result.code).is_ok()
                {
                    credit_transfer(problem, &donor);
                    result.method = format!("analogy:refit:{}", result.method);
                    return Some(result);
                }
            }
        }

        // (3) Universal re-fit (covers every remaining type), capped because it
        // re-enters the full solver. Re-check the wall-clock budget immediately
        // before attempting (the re-solve runs the whole portfolio and the
        // top-of-loop gate cannot interrupt one already in flight).
        let budget_ok = budget <= 0.0 || t0.elapsed().as_secs_f32() < budget;
        if universal_attempts < MAX_UNIVERSAL_REFITS && budget_ok {
            universal_attempts += 1;
            if let Some(mut result) = synthesize_universal_from_teacher(problem, &adapted) {
                if result.success
                    && crate::runtime::verify_problem_code_strict(problem, &result.code).is_ok()
                {
                    credit_transfer(problem, &donor);
                    result.method = format!("analogy:refit:universal:{}", result.method);
                    return Some(result);
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_renames_declaration_and_recursion() {
        let donor = "fn fact(n: i64) -> i64 { if (n <= 1) { return 1; } return (n * fact((n - 1))); }";
        let out = remap_donor_to_query(donor, "factorial").unwrap();
        assert!(out.contains("fn factorial("));
        assert!(out.contains("factorial((n - 1))"), "recursive call renamed: {out}");
        assert!(!out.contains("fact("), "no donor name left: {out}");
    }

    #[test]
    fn remap_leaves_substrings_alone() {
        // `sum` must not corrupt `summer`.
        let donor = "fn sum(a: i64) -> i64 { let summer = a; return summer; }";
        let out = remap_donor_to_query(donor, "total").unwrap();
        assert!(out.contains("fn total("));
        assert!(out.contains("summer"), "substring preserved: {out}");
        assert!(!out.contains(" sum("), "decl renamed: {out}");
    }

    #[test]
    fn remap_noop_when_names_match() {
        let donor = "fn f(n: i64) -> i64 { return n; }";
        assert_eq!(remap_donor_to_query(donor, "f").unwrap(), donor);
    }

    #[test]
    fn remap_none_without_fn() {
        assert!(remap_donor_to_query("let x = 1;", "f").is_none());
    }

    #[test]
    fn remap_multi_function_renames_entry_not_helper() {
        // helpers first, entry last (generated convention).
        let donor = "fn helper(n: i64) -> i64 { return (n + 1); }\nfn entry(n: i64) -> i64 { return helper((n * 2)); }";
        let out = remap_donor_to_query(donor, "target").unwrap();
        assert!(out.contains("fn target("), "entry renamed: {out}");
        assert!(out.contains("fn helper("), "helper preserved: {out}");
        assert!(out.contains("target(n: i64)"), "{out}");
        // helper call inside entry stays valid.
        assert!(out.contains("helper((n * 2))"), "{out}");
    }

    #[test]
    fn remap_ignores_fn_in_comment() {
        let donor = "// this helper fn add does stuff\nfn real(n: i64) -> i64 { return n; }";
        let out = remap_donor_to_query(donor, "query").unwrap();
        assert!(out.contains("fn query("), "real decl renamed: {out}");
        assert!(out.contains("// this helper fn add does stuff"), "comment untouched: {out}");
    }

    #[test]
    fn remap_preserves_string_literals() {
        let donor = "fn sum(n: i64) -> i64 { let _label = \"sum total\"; return n; }";
        let out = remap_donor_to_query(donor, "total").unwrap();
        assert!(out.contains("fn total("), "{out}");
        assert!(out.contains("\"sum total\""), "string literal preserved: {out}");
    }

    // ---- integration: real transfer through the verifier ----
    use crate::benchmark::{Example, Value};

    fn ex(i: i64, o: i64) -> Example {
        Example { inputs: vec![Value::Int(i)], expected: Value::Int(o) }
    }

    fn scalar_problem(
        name: &str,
        signature: &'static str,
        examples: Vec<Example>,
        holdouts: Vec<Example>,
    ) -> Problem {
        Problem {
            name: name.to_string(),
            category: "arithmetic",
            description: "",
            signature,
            examples,
            holdouts,
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    /// Run `f` with a scratch on-disk solved-cache (off under test by default).
    fn with_scratch_cache<R>(f: impl FnOnce() -> R) -> R {
        crate::solved_cache::with_test_lock(|| {
            let cache = std::env::temp_dir().join(format!(
                "nsynth_analogy_test_{}_{:?}.json",
                std::process::id(),
                std::thread::current().id(),
            ));
            std::env::set_var("NSYNTH_CACHE_PATH", &cache);
            crate::solved_cache::reset_for_tests();
            let _ = std::fs::remove_file(&cache);
            let result = f();
            std::env::remove_var("NSYNTH_CACHE_PATH");
            crate::solved_cache::reset_for_tests();
            let _ = std::fs::remove_file(&cache);
            result
        })
    }

    #[test]
    fn cross_name_transfer_emits_verified_solution() {
        with_scratch_cache(|| {
            // Donor solves doubling under the name `dbl`.
            let donor_problem =
                scalar_problem("donor", "fn dbl(n: i64) -> i64", vec![ex(1, 2), ex(2, 4)], vec![]);
            crate::solved_cache::record(
                &donor_problem,
                "search_scalar_expr",
                "fn dbl(n: i64) -> i64 { return (n * 2); }",
            );

            // Query is the SAME function, differently named, WITH holdouts.
            let query = scalar_problem(
                "scale2",
                "fn scale2(n: i64) -> i64",
                vec![ex(1, 2), ex(3, 6), ex(5, 10)],
                vec![ex(8, 16), ex(11, 22)],
            );

            let result = analogy_solve(&query).expect("cross-name transfer should succeed");
            assert!(result.success);
            assert!(
                result.method.starts_with("analogy:"),
                "method tagged analogy: got {}",
                result.method
            );
            // Non-fabrication: whatever is emitted passes the strict verifier
            // against the ORIGINAL query (examples + holdouts).
            assert!(
                crate::runtime::verify_problem_code_strict(&query, &result.code).is_ok(),
                "emitted code must verify against the query"
            );
        });
    }

    #[test]
    fn incompatible_donor_returns_none() {
        // A scalar donor cannot transfer to an array query (rename yields a
        // function whose body can't consume an array → verify fails; the scalar
        // re-fitter rejects array input; the array re-fitter can't sample a
        // scalar donor). Result: None, no fabrication.
        with_scratch_cache(|| {
            crate::solved_cache::record(
                &scalar_problem("donor", "fn dbl(n: i64) -> i64", vec![ex(1, 2)], vec![]),
                "m",
                "fn dbl(n: i64) -> i64 { return (n * 2); }",
            );
            let array_problem = Problem {
                examples: vec![Example {
                    inputs: vec![Value::Array(vec![1, 2, 3])],
                    expected: Value::Int(6),
                }],
                ..scalar_problem("arr_sum", "fn arr_sum(arr: [i64]) -> i64", vec![], vec![])
            };
            assert!(analogy_solve(&array_problem).is_none());
        });
    }

    fn arr_ex(xs: &[i64], o: i64) -> Example {
        Example { inputs: vec![Value::Array(xs.to_vec())], expected: Value::Int(o) }
    }

    #[test]
    fn array_donor_transfers_directly() {
        // Universal transfer: an ARRAY donor (different name) transfers to an
        // array query by rename+verify — no scalar gate, no re-synthesis needed.
        with_scratch_cache(|| {
            let donor = scalar_problem(
                "arr_total",
                "fn arr_total(arr: [i64]) -> i64",
                vec![arr_ex(&[1, 2, 3], 6)],
                vec![],
            );
            crate::solved_cache::record(
                &donor,
                "search_array",
                "fn arr_total(arr: [i64]) -> i64 { acc: i64 = 0; for x in arr { acc = acc + x; } return acc; }",
            );
            let query = scalar_problem(
                "bag_sum",
                "fn bag_sum(arr: [i64]) -> i64",
                vec![arr_ex(&[2, 4], 6), arr_ex(&[1, 1, 1], 3)],
                vec![arr_ex(&[5, 5, 5, 5], 20), arr_ex(&[10], 10)],
            );
            let result = analogy_solve(&query).expect("array donor should transfer");
            assert!(result.success);
            assert!(result.method.starts_with("analogy:"), "method: {}", result.method);
            assert!(
                crate::runtime::verify_problem_code_strict(&query, &result.code).is_ok(),
                "emitted array code must verify"
            );
        });
    }

    fn str_ex(s: &str, o: i64) -> Example {
        Example { inputs: vec![Value::Str(s.to_string())], expected: Value::Int(o) }
    }

    #[test]
    fn perturb_value_is_type_aware() {
        assert!(perturb_value(&Value::Int(5)).contains(&Value::Int(6)));
        assert_eq!(perturb_value(&Value::Bool(true)), vec![Value::Bool(false)]);
        let arr = perturb_value(&Value::Array(vec![1, 2, 3]));
        assert!(arr.contains(&Value::Array(vec![3, 2, 1])), "reversed variant");
        assert!(perturb_value(&Value::Str("ab".into())).contains(&Value::Str("abab".into())));
        // Unsupported shapes yield no perturbations (graceful).
        assert!(perturb_value(&Value::Tree(vec![])).is_empty());
    }

    #[test]
    fn runtime_to_bench_converts_common_shapes() {
        use crate::runtime::Value as R;
        assert_eq!(runtime_to_bench(&R::Int(7)), Some(Value::Int(7)));
        assert_eq!(runtime_to_bench(&R::Bool(true)), Some(Value::Bool(true)));
        assert_eq!(
            runtime_to_bench(&R::Array(vec![R::Int(1), R::Int(2)])),
            Some(Value::Array(vec![1, 2]))
        );
        // Heterogeneous / unsupported → None (skips the sample, never panics).
        assert_eq!(runtime_to_bench(&R::Array(vec![R::Bool(true)])), None);
    }

    #[test]
    fn universal_refit_solves_from_correct_teacher() {
        // The universal re-fitter samples the teacher on perturbed inputs and
        // re-solves; with a correct doubling teacher it produces a verified
        // doubling program. Exercises sampling + conversion + recursion-guarded
        // re-solve end-to-end.
        let query = scalar_problem(
            "scale2",
            "fn scale2(n: i64) -> i64",
            vec![ex(1, 2), ex(3, 6), ex(5, 10)],
            vec![ex(8, 16)],
        );
        let teacher = "fn scale2(n: i64) -> i64 { return (n * 2); }";
        let result = synthesize_universal_from_teacher(&query, teacher)
            .expect("universal re-fit should solve from a correct teacher");
        assert!(result.success);
        assert!(crate::runtime::verify_problem_code_strict(&query, &result.code).is_ok());
    }

    #[test]
    fn universal_refit_rejects_wrong_teacher() {
        // A teacher that computes the WRONG function (increment, not double) must
        // not yield an emitted program: its mislabeled samples either prevent a
        // solve or produce code that fails the ORIGINAL-problem verifier. Locks
        // the non-fabrication invariant (verify is against `problem`, never the
        // teacher-augmented set).
        let query = scalar_problem(
            "scale2",
            "fn scale2(n: i64) -> i64",
            vec![ex(1, 2), ex(3, 6), ex(5, 10)],
            vec![ex(8, 16)],
        );
        let wrong = "fn scale2(n: i64) -> i64 { return (n + 1); }";
        assert!(synthesize_universal_from_teacher(&query, wrong).is_none());
    }

    #[test]
    fn string_donor_transfers_directly() {
        // Universal transfer across a STRING-input donor/query.
        with_scratch_cache(|| {
            let donor = scalar_problem(
                "has_cat",
                "fn has_cat(s: string) -> i64",
                vec![str_ex("cat", 1)],
                vec![],
            );
            crate::solved_cache::record(
                &donor,
                "search_text",
                "fn has_cat(s: string) -> i64 { if s.contains(\"cat\") { return 1; } return 0; }",
            );
            let query = scalar_problem(
                "mentions_cat",
                "fn mentions_cat(s: string) -> i64",
                vec![str_ex("a cat sat", 1), str_ex("dog", 0)],
                vec![str_ex("scatter", 1), str_ex("zzz", 0)],
            );
            let result = analogy_solve(&query).expect("string donor should transfer");
            assert!(result.method.starts_with("analogy:"), "method: {}", result.method);
            assert!(
                crate::runtime::verify_problem_code_strict(&query, &result.code).is_ok(),
                "emitted string code must verify"
            );
        });
    }

    #[test]
    fn empty_cache_returns_none() {
        with_scratch_cache(|| {
            let query = scalar_problem("scale2", "fn scale2(n: i64) -> i64", vec![ex(1, 2)], vec![]);
            assert!(analogy_solve(&query).is_none());
        });
    }

    #[test]
    fn empty_holdouts_emit_but_do_not_credit_transfer() {
        with_scratch_cache(|| {
            let donor_problem =
                scalar_problem("donor", "fn dbl(n: i64) -> i64", vec![ex(1, 2), ex(2, 4)], vec![]);
            crate::solved_cache::record(
                &donor_problem,
                "search_scalar_expr",
                "fn dbl(n: i64) -> i64 { return (n * 2); }",
            );
            // Query with NO holdouts: verifier degenerates to example-matching,
            // so transfer credit must NOT be recorded (overfit-poison guard).
            let query = scalar_problem(
                "scale2",
                "fn scale2(n: i64) -> i64",
                vec![ex(1, 2), ex(3, 6), ex(5, 10)],
                vec![],
            );
            let before = crate::solved_cache::snapshot_solutions_with_meta();
            let donor_sc_before: u32 = before
                .iter()
                .find(|(_, c, _, _)| c.contains("n * 2"))
                .map(|(_, _, sc, _)| *sc)
                .unwrap_or(0);

            let result = analogy_solve(&query);
            // It still emits a verified solution (same standard as other stages)…
            if let Some(r) = &result {
                assert!(crate::runtime::verify_problem_code_strict(&query, &r.code).is_ok());
            }
            // …but the donor's success_count is unchanged (no credit on weak evidence).
            let after = crate::solved_cache::snapshot_solutions_with_meta();
            let donor_sc_after: u32 = after
                .iter()
                .find(|(_, c, _, _)| c.contains("n * 2"))
                .map(|(_, _, sc, _)| *sc)
                .unwrap_or(0);
            assert_eq!(
                donor_sc_before, donor_sc_after,
                "empty-holdout transfer must not credit the donor"
            );
        });
    }
}
