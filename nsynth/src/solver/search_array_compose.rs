//! Exact array-feature composition synthesis.
//!
//! This is the *array* analogue of `search_composed_features` in
//! `search_affine.rs`. Where that solver recovers `f(x) = c0 + Σ c_k·g_k(x)`
//! over derived scalar features (squares, cross-terms, modulo), this one
//! recovers
//!
//!     f(arr [, scalars]) = c0 + Σ c_k·R_k(arr) (+ Σ d_j·s_j)
//!
//! where each `R_k` is an integer *reduction* over the array (sum, max, length,
//! count-of-positive, sum-of-squares, …) and each scalar argument enters as its
//! own raw feature. It owns ONLY the genuinely compositional case: an affine mix
//! of two-or-more reductions, or one reduction wrapped in a non-trivial affine
//! term (`5 + 2*len + sum`, `sum_sq - 3*sum`, `base + 10*count_positive`). The
//! single-reduction restatements (`return sum`, `return max`, `return arr.len`)
//! are owned by the dedicated reduction solvers (`search_array_sum`,
//! `search_array_max`, `search_array_range`, …) which run EARLIER in
//! `SEARCH_CANDIDATES`, and this family refuses them (§ the trivial-restatement
//! guard) so it never duplicates their work or method keys.
//!
//! Like every solver in this crate it is exact and refusal-first: it builds one
//! regression column per candidate feature, runs the shared integer linear solve
//! (`solve_linear_features`), demands strong over-determination, predicts every
//! example in i128 before rendering, and finally re-verifies the emitted Mog
//! through the real runtime. A wrong-on-unseen fit dies at one of those gates
//! rather than being certified.
//!
//! NOT wired into `solve_multi_arg_affine` (the scalar fast-path in `search.rs`):
//! that early dispatch is gated on `extract_scalar_examples`, which only accepts
//! all-int signatures and returns `None` the moment an input is an array. Array
//! compositional rules therefore flow through the normal `solve_by_search`
//! portfolio, which is the correct lane — do not "fix" this by wiring it in.

use super::helpers::array_value;
use super::search_affine::solve_linear_features;
use super::search_codegen::verified_result;
use super::signature::{parse_param_types, ParamType};
use super::*;

/// Sparse-first search depth: at most a 3-reduction affine mix.
const KMAX: usize = 3;
/// Over-determination surplus: a k-feature fit (`k+1` unknowns) is attempted
/// only when there are at least `k + 1 + MARGIN` examples. MARGIN=3 is one
/// stricter than `search_composed_features` (MARGIN=2): an array reduction
/// collapses a whole array to a single scalar, so each row carries less
/// information than a scalar feature and we buy back rigor with more surplus.
const MARGIN: usize = 3;

/// One candidate feature: a reduction over the array (or a raw scalar argument),
/// together with its already-computed value on every example (`col`, for the fast
/// in-Rust exact-fit check), a complexity `rank` for the simplest-first ordering,
/// and a `kind` that drives both dedup and Mog code generation.
struct ArrFeature {
    kind: FeatureKind,
    col: Vec<i64>,
    rank: u32,
}

/// The reduction (or scalar arg) a feature stands for. Each variant renders to a
/// known-good Mog shape copied verbatim from an existing array codegen.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FeatureKind {
    Len,
    First,
    Last,
    Sum,
    Min,
    Max,
    CountPositive,
    CountZero,
    CountEven,
    Range,
    SumOfAbs,
    SumOfSquares,
    /// The `j`-th scalar argument (0-based over the trailing scalars), rendered
    /// directly as its parameter name.
    Scalar(usize),
}

impl FeatureKind {
    /// Fixed complexity rank for the simplest-first / Occam ordering and dedup
    /// tie-break: `len < first < last < sum < min < max < count_* < range <
    /// sum_of_abs < sum_of_squares`; scalar args rank ABOVE every reduction so a
    /// reduction is preferred when a column is shared (a reduction generalizes;
    /// a raw scalar that happens to equal one is a coincidence of the samples).
    fn rank(self) -> u32 {
        match self {
            FeatureKind::Len => 0,
            FeatureKind::First => 1,
            FeatureKind::Last => 2,
            FeatureKind::Sum => 3,
            FeatureKind::Min => 4,
            FeatureKind::Max => 5,
            FeatureKind::CountPositive => 6,
            FeatureKind::CountZero => 7,
            FeatureKind::CountEven => 8,
            FeatureKind::Range => 9,
            FeatureKind::SumOfAbs => 10,
            FeatureKind::SumOfSquares => 11,
            // Scalars rank below all reductions (higher number = simpler-first
            // ordering visits them last), matching the spec's "scalars rank below
            // all reductions" so a reduction wins any tie.
            FeatureKind::Scalar(_) => 100,
        }
    }

    /// True for the array reductions (everything but a raw scalar argument). Used
    /// by the trivial-restatement guard.
    fn is_reduction(self) -> bool {
        !matches!(self, FeatureKind::Scalar(_))
    }
}

/// Evaluate one reduction on a single array, in `i128` to defer overflow, or
/// `None` when the reduction is undefined here (empty array for the
/// position-bearing reductions) or overflows i64. A `None` on ANY example drops
/// the whole feature — we never emit a program that can trap or that we cannot
/// fully fit.
fn reduce(kind: FeatureKind, arr: &[i64]) -> Option<i128> {
    let v: i128 = match kind {
        FeatureKind::Len => arr.len() as i128,
        FeatureKind::First => *arr.first()? as i128,
        FeatureKind::Last => *arr.last()? as i128,
        FeatureKind::Sum => arr.iter().map(|&x| x as i128).sum(),
        FeatureKind::Min => *arr.iter().min()? as i128,
        FeatureKind::Max => *arr.iter().max()? as i128,
        FeatureKind::CountPositive => arr.iter().filter(|&&x| x > 0).count() as i128,
        FeatureKind::CountZero => arr.iter().filter(|&&x| x == 0).count() as i128,
        FeatureKind::CountEven => arr.iter().filter(|&&x| x.rem_euclid(2) == 0).count() as i128,
        FeatureKind::Range => (*arr.iter().max()? as i128) - (*arr.iter().min()? as i128),
        FeatureKind::SumOfAbs => arr.iter().map(|&x| (x as i128).abs()).sum(),
        FeatureKind::SumOfSquares => arr.iter().map(|&x| (x as i128) * (x as i128)).sum(),
        FeatureKind::Scalar(_) => unreachable!("scalar features are filled directly, not reduced"),
    };
    Some(v)
}

/// The 12-reduction library, in rank order. The genuinely universal array
/// reductions — like `+`/`*` for scalars — so no constant mining is needed.
const REDUCTIONS: [FeatureKind; 12] = [
    FeatureKind::Len,
    FeatureKind::First,
    FeatureKind::Last,
    FeatureKind::Sum,
    FeatureKind::Min,
    FeatureKind::Max,
    FeatureKind::CountPositive,
    FeatureKind::CountZero,
    FeatureKind::CountEven,
    FeatureKind::Range,
    FeatureKind::SumOfAbs,
    FeatureKind::SumOfSquares,
];

/// Per-example extracted inputs: the array (first param) and the trailing scalar
/// args, plus the target.
struct ArrExample {
    arr: Vec<i64>,
    scalars: Vec<i64>,
    target: i64,
}

/// Parse the signature and extract examples, accepting ONLY `[arr]` or
/// `[arr, i64{1,2}]` with the array FIRST. Any other shape (array not first, more
/// than one array, a non-int/non-array param, wrong input count, failed
/// extraction) → `None`, so the solver refuses everything outside its domain.
fn extract(problem: &Problem) -> Option<(Vec<ArrExample>, usize)> {
    let param_types = parse_param_types(problem.signature);
    if param_types.first()? != &ParamType::ArrayI64 {
        return None; // array must be the first parameter
    }
    let n_scalars = param_types.len() - 1;
    if n_scalars > 2 {
        return None; // at most two trailing scalars
    }
    if param_types[1..].iter().any(|t| t != &ParamType::I64) {
        return None; // trailing params must all be plain i64
    }
    let arity = param_types.len();
    let mut examples = Vec::with_capacity(problem.examples.len());
    for ex in &problem.examples {
        if ex.inputs.len() != arity {
            return None;
        }
        let arr = array_value(&ex.inputs[0])?.to_vec();
        let mut scalars = Vec::with_capacity(n_scalars);
        for input in &ex.inputs[1..] {
            scalars.push(int_value(input)?);
        }
        examples.push(ArrExample {
            arr,
            scalars,
            target: ex.expected_int(),
        });
    }
    if examples.is_empty() {
        return None;
    }
    Some((examples, n_scalars))
}

/// Build the feature pool: every reduction whose column is fully defined on all
/// examples, plus one column per scalar argument. Then drop constant columns
/// (indistinguishable from the intercept) and deduplicate bit-identical columns
/// keeping the simpler (lower-rank) representative. Returns the surviving features
/// in simplest-first (rank) order so the subset search is Occam-ordered and
/// deterministic.
fn build_features(examples: &[ArrExample], n_scalars: usize) -> Vec<ArrFeature> {
    let n = examples.len();
    let mut feats: Vec<ArrFeature> = Vec::new();

    // Reductions: keep one only if every example yields a defined, i64-fitting
    // value (a single abstain drops the whole feature).
    'red: for &kind in &REDUCTIONS {
        let mut col = Vec::with_capacity(n);
        for ex in examples {
            match reduce(kind, &ex.arr) {
                Some(v) if i64::try_from(v).is_ok() => col.push(v as i64),
                _ => continue 'red,
            }
        }
        feats.push(ArrFeature {
            kind,
            col,
            rank: kind.rank(),
        });
    }

    // Scalar args: each is its own raw feature column.
    for j in 0..n_scalars {
        let kind = FeatureKind::Scalar(j);
        let col: Vec<i64> = examples.iter().map(|ex| ex.scalars[j]).collect();
        feats.push(ArrFeature {
            kind,
            col,
            rank: kind.rank(),
        });
    }

    // Drop CONSTANT columns: a feature that takes one value across all examples
    // is absorbed into c0 and only inflates the basis (e.g. every array has the
    // same length → `len` is constant and dropped).
    feats.retain(|f| f.col.iter().any(|&v| v != f.col[0]));

    // Simplest-first order so dedup keeps the simpler representative and the
    // subset search is Occam-ordered.
    feats.sort_by_key(|f| f.rank);

    // Deduplicate bit-identical columns, keeping the FIRST (lowest-rank, already
    // sorted) — the array analogue of `insert_expr_candidate`'s simplest-rep
    // dedup. Prevents two coincidentally-equal reductions (min==first on sorted
    // samples) from both entering and producing a non-deterministic pick.
    let mut deduped: Vec<ArrFeature> = Vec::with_capacity(feats.len());
    for f in feats {
        if deduped.iter().any(|kept| kept.col == f.col) {
            continue;
        }
        deduped.push(f);
    }
    deduped
}

/// In-Rust exact gate: `c0 + Σ w_k·col_k[i] == target_i` for EVERY example
/// (i128 accumulation). The Gaussian solve only guarantees the `m` pivot rows;
/// this demands all `n`, rejecting any under-determined / f64-rounded fit before
/// we render. Copied in spirit from `composed_predicts`.
fn predicts_all(c0: i64, picks: &[(&ArrFeature, i64)], targets: &[i64]) -> bool {
    for (i, &t) in targets.iter().enumerate() {
        let mut acc = c0 as i128;
        for &(feat, w) in picks {
            acc += w as i128 * feat.col[i] as i128;
        }
        if acc != t as i128 {
            return false;
        }
    }
    true
}

/// The Mog reduction block for `kind` into local `r{slot}` (and helper locals
/// `lo{slot}`/`hi{slot}` for `range`), with a per-block loop variable `x{slot}`.
/// Every shape is copied verbatim from a known-good codegen so it is guaranteed
/// parseable (no `let`; typed locals and `:=` only). Scalars have no block (they
/// render their param name inline), so this is only called for reductions.
fn reduction_block(kind: FeatureKind, slot: usize) -> String {
    let r = format!("r{slot}");
    let x = format!("x{slot}");
    match kind {
        // `len(arr)` (the runtime builtin), NOT `arr.len` — the field form is
        // valid Mog but the Python transpiler renders it verbatim as `arr.len`,
        // which is not valid Python; the builtin transpiles cleanly to `len(arr)`
        // so the emitted program is correct on BOTH the Mog and Python paths.
        FeatureKind::Len => format!("    {r}: i64 = len(arr);\n"),
        FeatureKind::First => format!("    {r}: i64 = arr[0];\n"),
        FeatureKind::Last => format!("    {r}: i64 = arr[len(arr) - 1];\n"),
        FeatureKind::Sum => format!(
            "    {r}: i64 = 0;\n    for {x} in arr {{\n        {r} = {r} + {x};\n    }}\n"
        ),
        FeatureKind::Min => format!(
            "    {r}: i64 = arr[0];\n    for {x} in arr {{\n        if {x} < {r} {{\n            {r} = {x};\n        }}\n    }}\n"
        ),
        FeatureKind::Max => format!(
            "    {r}: i64 = arr[0];\n    for {x} in arr {{\n        if {x} > {r} {{\n            {r} = {x};\n        }}\n    }}\n"
        ),
        FeatureKind::CountPositive => format!(
            "    {r}: i64 = 0;\n    for {x} in arr {{\n        if {x} > 0 {{\n            {r} = {r} + 1;\n        }}\n    }}\n"
        ),
        FeatureKind::CountZero => format!(
            "    {r}: i64 = 0;\n    for {x} in arr {{\n        if {x} == 0 {{\n            {r} = {r} + 1;\n        }}\n    }}\n"
        ),
        FeatureKind::CountEven => format!(
            "    {r}: i64 = 0;\n    for {x} in arr {{\n        if ({x} % 2) == 0 {{\n            {r} = {r} + 1;\n        }}\n    }}\n"
        ),
        FeatureKind::Range => {
            let lo = format!("lo{slot}");
            let hi = format!("hi{slot}");
            format!(
                "    {lo}: i64 = arr[0];\n    {hi}: i64 = arr[0];\n    for {x} in arr {{\n        if {x} < {lo} {{\n            {lo} = {x};\n        }}\n        if {x} > {hi} {{\n            {hi} = {x};\n        }}\n    }}\n    {r}: i64 = {hi} - {lo};\n"
            )
        }
        FeatureKind::SumOfAbs => format!(
            "    {r}: i64 = 0;\n    for {x} in arr {{\n        if {x} < 0 {{\n            {r} = {r} + (0 - {x});\n        }} else {{\n            {r} = {r} + {x};\n        }}\n    }}\n"
        ),
        FeatureKind::SumOfSquares => format!(
            "    {r}: i64 = 0;\n    for {x} in arr {{\n        {r} = {r} + {x} * {x};\n    }}\n"
        ),
        FeatureKind::Scalar(_) => unreachable!("scalar features have no reduction block"),
    }
}

/// The reference expression for a feature in the final `return`: a reduction is
/// its local `r{slot}`, a scalar arg is its parameter name (`scalar_params[j]`).
fn feature_ref(kind: FeatureKind, slot: usize, scalar_params: &[String]) -> String {
    match kind {
        FeatureKind::Scalar(j) => scalar_params[j].clone(),
        _ => format!("r{slot}"),
    }
}

/// Render the affine `return` over the chosen feature references: drop zero
/// coeffs, render `1*ref` as `ref`, `-1*ref` as `(0 - ref)` and `w*ref` as
/// `(w) * ref` (parenthesised so a negative `w` does not parse as a binary minus),
/// append `c0` iff it is non-zero or there is no other term. No unary minus is
/// assumed anywhere.
fn affine_over_terms(c0: i64, terms: &[(String, i64)]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (refr, w) in terms {
        match *w {
            0 => continue,
            1 => parts.push(refr.clone()),
            -1 => parts.push(format!("(0 - {refr})")),
            w => parts.push(format!("({w}) * {refr}")),
        }
    }
    if c0 != 0 || parts.is_empty() {
        parts.push(c0.to_string());
    }
    parts.join(" + ")
}

/// Exact array-feature composition: recover `f(arr [, scalars]) = c0 + Σ c_k·R_k`
/// over the 12-reduction library plus the scalar args, sparse-first and fully
/// verified.
///
/// HONESTY (the contract — a 14-column basis can fit noise on a sparse set, so
/// the guards are the whole point):
///   * OVER-DETERMINED: a k-feature fit needs ≥ `k + 1 + MARGIN` examples
///     (MARGIN=3), so the system is never close to square.
///   * SPARSE / SIMPLEST-FIRST: subsets are tried size 1, then 2, then 3, each in
///     the fixed rank order; the first exact-and-verified fit wins.
///   * NOT A SINGLE-REDUCTION RESTATEMENT: a size-1 bare reduction with
///     `c0==0, c1==+1` (`return sum`) is refused — those belong to the dedicated
///     reduction solvers that run earlier. A coeff of -1 (`-sum`/`-max`, owned by
///     no dedicated solver) is KEPT. A size-1 scalar-only subset is also refused
///     (pure scalar affine is `search_affine`'s).
///   * EXACT INTEGER + ROUND-GATE + FULL VERIFY: `solve_linear_features` rejects
///     non-integral coefficients; `predicts_all` requires the integer fit to
///     reproduce every example; `verified_result` re-runs the emitted Mog.
pub(super) fn search_array_affine_features(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    let (examples, n_scalars) = extract(problem)?;
    let n = examples.len();
    let targets: Vec<i64> = examples.iter().map(|ex| ex.target).collect();

    // DEGENERATE-DATA REFUSAL: a constant output is not this family's job (no
    // affine-over-reductions structure to recover).
    if targets.iter().all(|&t| t == targets[0]) {
        return None;
    }

    let feats = build_features(&examples, n_scalars);
    if feats.is_empty() {
        return None; // every reduction was constant — nothing to compose
    }

    // Scalar parameter names. The array is always `arr`; trailing scalars take
    // single-letter names matching the existing mixed-signature codegen idiom.
    let scalar_param_letters = ["k", "t"];
    let scalar_params: Vec<String> = (0..n_scalars)
        .map(|j| scalar_param_letters[j].to_string())
        .collect();
    let mut signature = String::from("arr: [i64]");
    for name in &scalar_params {
        signature.push_str(&format!(", {name}: i64"));
    }

    // Solve for [c0, w_1, …, w_k] over a chosen feature subset and, if it is an
    // exact integer fit reproducing every example, render + verify it.
    let try_subset = |idxs: &[usize]| -> Option<SolveResult> {
        let k = idxs.len();
        // OVER-DETERMINATION gate.
        if n < k + 1 + MARGIN {
            return None;
        }
        let m = k + 1;
        let feature_rows: Vec<Vec<i64>> = (0..n)
            .map(|i| {
                let mut phi = Vec::with_capacity(m);
                phi.push(1);
                for &fi in idxs {
                    phi.push(feats[fi].col[i]);
                }
                phi
            })
            .collect();
        let w = solve_linear_features(&feature_rows, &targets, m)?;
        let c0 = w[0];
        let picks: Vec<(&ArrFeature, i64)> = idxs
            .iter()
            .enumerate()
            .map(|(slot, &fi)| (&feats[fi], w[slot + 1]))
            .collect();
        // OWNERSHIP / trivial-restatement guard, applied to the EFFECTIVE program
        // (the features that survive with a non-zero coefficient) — a larger
        // subset can still collapse to a single bare reduction when the other
        // coefficients solve to zero. The effective program is the bare
        // restatement when c0==0 and exactly one feature has a non-zero coeff of
        // magnitude 1 on a bare REDUCTION (`return sum`/`return max`/`return len`):
        // those are owned by the dedicated reduction solvers that run earlier, so
        // we refuse them. A non-trivial affine wrapper (c0!=0 or |coeff|!=1) or a
        // genuine multi-reduction mix is kept. (A lone scalar feature is also
        // refused — pure scalar affine is search_affine's job.)
        let active: Vec<&(&ArrFeature, i64)> = picks.iter().filter(|(_, w)| *w != 0).collect();
        if active.len() <= 1 {
            match active.first() {
                None => return None, // collapsed to a pure constant — not ours
                Some((feat, w1)) => {
                    if !feat.kind.is_reduction() {
                        return None; // pure scalar affine — search_affine's job
                    }
                    if c0 == 0 && *w1 == 1 {
                        // `return <reduction>` (coeff +1) is the bare restatement
                        // owned by the dedicated reduction solvers that run earlier.
                        // A coeff of -1 (`return 0 - <reduction>`, e.g. -sum / -max)
                        // is a GENUINE sign-flip transform owned by NO dedicated
                        // solver, so it is kept here (still gated by verified_result).
                        return None;
                    }
                }
            }
        }
        if !predicts_all(c0, &picks, &targets) {
            return None;
        }

        // Render: each chosen reduction into its own local block (in pick order),
        // then a single affine return over the locals + scalar refs.
        let mut body = String::new();
        let mut terms: Vec<(String, i64)> = Vec::with_capacity(k);
        for (slot, &(feat, w_k)) in picks.iter().enumerate() {
            if feat.kind.is_reduction() {
                body.push_str(&reduction_block(feat.kind, slot));
            }
            terms.push((feature_ref(feat.kind, slot, &scalar_params), w_k));
        }
        let ret = affine_over_terms(c0, &terms);
        let code = format!("fn {fn_name}({signature}) -> i64 {{\n{body}    return {ret};\n}}\n");
        verified_result(problem, code, "array_affine_features")
    };

    // SUBSET SEARCH. Scalar arguments are declared parameters of the function, so
    // they are ALWAYS candidate terms (an unused one simply solves to coefficient
    // zero and drops out of the render). The sparse 1-2-3 budget therefore applies
    // to the REDUCTIONS only — otherwise a perfectly ordinary rule like
    // `base + 2*len + sum` over a `(arr, base)` signature (3 reductions + 1 scalar
    // = 4 terms) could never be expressed and would time out in the gradient
    // fallback. Reductions are tried sparse-first (1, then 2, then 3) in the fixed
    // simplest-first order; the first exact-and-verified fit wins.
    let fc = feats.len();
    let reductions: Vec<usize> = (0..fc).filter(|&i| feats[i].kind.is_reduction()).collect();
    let scalars: Vec<usize> = (0..fc).filter(|&i| !feats[i].kind.is_reduction()).collect();
    let with_scalars = |reds: &[usize]| -> Vec<usize> {
        let mut idxs = scalars.clone();
        idxs.extend_from_slice(reds);
        idxs
    };
    let rc = reductions.len();
    for ai in 0..rc {
        if let Some(r) = try_subset(&with_scalars(&[reductions[ai]])) {
            return Some(r);
        }
    }
    if KMAX >= 2 {
        for ai in 0..rc {
            for bi in (ai + 1)..rc {
                if let Some(r) = try_subset(&with_scalars(&[reductions[ai], reductions[bi]])) {
                    return Some(r);
                }
            }
        }
    }
    if KMAX >= 3 {
        for ai in 0..rc {
            for bi in (ai + 1)..rc {
                for ci in (bi + 1)..rc {
                    if let Some(r) = try_subset(&with_scalars(&[
                        reductions[ai],
                        reductions[bi],
                        reductions[ci],
                    ])) {
                        return Some(r);
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value};

    /// Problem builder for `fn f(arr: [i64]) -> i64`.
    fn pa(rows: &[(&[i64], i64)]) -> Problem {
        Problem {
            name: "f".to_string(),
            category: "external",
            description: "",
            signature: "fn f(arr: [i64]) -> i64",
            examples: rows
                .iter()
                .map(|(arr, y)| Example {
                    inputs: vec![Value::int_array(arr)],
                    expected: Value::Int(*y),
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",

            synthetic_args: Vec::new(),

            synthetic_values: Vec::new(),

            recursive_allowed: false,

            tree_input: false,

            explicit_stack: false,

            functions: vec![],
        }
    }

    /// Problem builder for `fn f(arr: [i64], k: i64) -> i64`.
    fn pas(rows: &[(&[i64], i64, i64)]) -> Problem {
        Problem {
            name: "f".to_string(),
            category: "external",
            description: "",
            signature: "fn f(arr: [i64], k: i64) -> i64",
            examples: rows
                .iter()
                .map(|(arr, k, y)| Example {
                    inputs: vec![Value::int_array(arr), Value::Int(*k)],
                    expected: Value::Int(*y),
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",

            synthetic_args: Vec::new(),

            synthetic_values: Vec::new(),

            recursive_allowed: false,

            tree_input: false,

            explicit_stack: false,

            functions: vec![],
        }
    }

    // T1 — recovers a two-reduction affine `5 + 2*len + sum`, generalizing to
    // unseen arrays.
    #[test]
    fn recovers_len_plus_sum_affine() {
        let f = |a: &[i64]| 5 + 2 * (a.len() as i64) + a.iter().sum::<i64>();
        let train: [&[i64]; 8] = [
            &[],
            &[1],
            &[1, 2],
            &[3, 3, 3],
            &[-1, 4],
            &[10],
            &[2, 2, 2, 2],
            &[7, -3, 1],
        ];
        let rows: Vec<(&[i64], i64)> = train.iter().map(|&a| (a, f(a))).collect();
        let p = pa(&rows);
        let r = search_array_affine_features(&p, "f").expect("must recover 5 + 2*len + sum");
        let unseen: [&[i64]; 4] = [&[100], &[5, 5, 5, 5, 5], &[-9, -9], &[0, 1, 2, 3, 4, 5]];
        let check: Vec<(&[i64], i64)> = unseen.iter().map(|&a| (a, f(a))).collect();
        crate::runtime::verify_problem_code_strict(&pa(&check), &r.code)
            .expect("must be exact on unseen arrays");
    }

    // T2 — recovers a reduction-difference with a non-unit/negative coefficient
    // `sum_of_squares - 3*sum`, proving sum_of_squares is the chosen feature.
    #[test]
    fn recovers_sum_squares_minus_3_sum() {
        let f = |a: &[i64]| a.iter().map(|x| x * x).sum::<i64>() - 3 * a.iter().sum::<i64>();
        let train: [&[i64]; 9] = [
            &[1],
            &[1, 2],
            &[-1, 2, 3],
            &[4, 4],
            &[0, 5, -5],
            &[2, 2, 2, 2],
            &[7],
            &[-3, -3, -3],
            &[1, 2, 3, 4, 5],
        ];
        let rows: Vec<(&[i64], i64)> = train.iter().map(|&a| (a, f(a))).collect();
        let p = pa(&rows);
        let r = search_array_affine_features(&p, "f").expect("must recover sum_of_squares - 3*sum");
        let unseen: [&[i64]; 4] = [&[6, 6], &[-4, 2, 2], &[11], &[1, 1, 1, 1, 1, 1]];
        let check: Vec<(&[i64], i64)> = unseen.iter().map(|&a| (a, f(a))).collect();
        crate::runtime::verify_problem_code_strict(&pa(&check), &r.code)
            .expect("must be exact on unseen arrays");
    }

    // T3 — recovers an array+scalar mix `base + 2*len + min`, the scalar argument
    // entering additively as its own column and the mixed signature round-tripping.
    // (The basis is affine over reductions + raw scalars; a `scalar × reduction`
    // product is deliberately outside its span, so the scalar term is additive.)
    #[test]
    fn recovers_base_plus_len_plus_min() {
        let f = |a: &[i64], base: i64| base + 2 * (a.len() as i64) + *a.iter().min().unwrap();
        let train: [(&[i64], i64); 8] = [
            (&[1], 2),
            (&[1, 2], 3),
            (&[-1, 4], 1),
            (&[3, 3, 3], 5),
            (&[10, -2], 0),
            (&[7], 4),
            (&[2, 8, 2, 2], 2),
            (&[-5, -1, -3], 6),
        ];
        let rows: Vec<(&[i64], i64, i64)> = train.iter().map(|&(a, b)| (a, b, f(a, b))).collect();
        let p = pas(&rows);
        let r = search_array_affine_features(&p, "f").expect("must recover base + 2*len + min");
        let unseen: [(&[i64], i64); 4] = [
            (&[9, 9], 7),
            (&[-4, 0, 8], 3),
            (&[6], 11),
            (&[1, 2, 3, 4], 1),
        ];
        let check: Vec<(&[i64], i64, i64)> = unseen.iter().map(|&(a, b)| (a, b, f(a, b))).collect();
        crate::runtime::verify_problem_code_strict(&pas(&check), &r.code)
            .expect("must be exact on unseen (arr, base)");
    }

    // T4 (mandated refusal) — an order-sensitive target `Σ i*arr[i]` is NOT in
    // the span of the 12 order-free reductions, so the solver must refuse rather
    // than overfit.
    #[test]
    fn refuses_index_weighted_sum() {
        let f = |a: &[i64]| {
            a.iter()
                .enumerate()
                .map(|(i, &v)| i as i64 * v)
                .sum::<i64>()
        };
        let train: [&[i64]; 9] = [
            &[1, 2],
            &[3, 1, 4],
            &[1, 5, 9, 2],
            &[2, 7],
            &[8, 8, 8],
            &[1, 0, 0, 1],
            &[5, 4, 3, 2, 1],
            &[-1, 2, -3, 4],
            &[10, 20, 30],
        ];
        let rows: Vec<(&[i64], i64)> = train.iter().map(|&a| (a, f(a))).collect();
        let p = pa(&rows);
        assert!(
            search_array_affine_features(&p, "f").is_none(),
            "must refuse an order-sensitive (non-reduction) target"
        );
    }

    // T5 (ownership boundary) — a bare reduction `sum(arr)` (c0=0, c1=1) is left
    // to the dedicated solver: the trivial-restatement guard must refuse it.
    #[test]
    fn refuses_bare_sum() {
        let f = |a: &[i64]| a.iter().sum::<i64>();
        let train: [&[i64]; 8] = [
            &[1],
            &[1, 2],
            &[3, 3],
            &[-1, 4],
            &[10],
            &[2, 2, 2, 2],
            &[7, -3, 1],
            &[0, 0, 5],
        ];
        let rows: Vec<(&[i64], i64)> = train.iter().map(|&a| (a, f(a))).collect();
        let p = pa(&rows);
        assert!(
            search_array_affine_features(&p, "f").is_none(),
            "bare sum is owned by search_array_sum — must refuse"
        );
    }

    // T6 (coverage hole closed) — `-sum(arr)` (c0=0, coeff=-1) is a genuine
    // sign-flip transform owned by NO dedicated reduction solver. The OLD
    // `|coeff|==1` guard refused it (alongside the real +sum restatement), leaving
    // it unsolvable; the relaxed guard (refuse only +1) now KEEPS it. Still gated
    // by verified_result, and strict-verified here on UNSEEN arrays. T5 proves the
    // +sum boundary is preserved (still refused), so this did not over-open.
    #[test]
    fn recovers_negated_sum() {
        let f = |a: &[i64]| -a.iter().sum::<i64>();
        let train: [&[i64]; 8] = [
            &[1],
            &[1, 2],
            &[3, 3],
            &[-1, 4],
            &[10],
            &[2, 2, 2, 2],
            &[7, -3, 1],
            &[0, 0, 5],
        ];
        let rows: Vec<(&[i64], i64)> = train.iter().map(|&a| (a, f(a))).collect();
        let p = pa(&rows);
        let r = search_array_affine_features(&p, "f")
            .expect("must recover -sum (coeff -1 is not the +1 bare restatement)");
        let unseen: [&[i64]; 4] = [&[100], &[5, 5, 5, 5, 5], &[-9, -9], &[0, 1, 2, 3, 4, 5]];
        let check: Vec<(&[i64], i64)> = unseen.iter().map(|&a| (a, f(a))).collect();
        crate::runtime::verify_problem_code_strict(&pa(&check), &r.code)
            .expect("must be exact -sum on unseen arrays");
    }
}
