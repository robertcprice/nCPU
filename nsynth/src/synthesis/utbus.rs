//! UTBUS — Unified Typed Bottom-Up Structural Synthesizer (foundation).
//!
//! This module is the first slice of a generic, type-directed, bottom-up
//! enumeration core for program synthesis. The long-term goal (see
//! `docs/SYNTHESIS_NEXT_STEPS.md`, Part 2) is to express every search family
//! (scalar, array, string, tree, ...) on top of a *single* enumeration engine
//! rather than one hand-written template solver per family.
//!
//! What this slice proves is the abstraction itself: the array transform set
//! that the legacy [`super::native_array`] path covers by hand is here
//! re-derived from generic primitives — a typed value representation, an
//! observational-equivalence (OE) table that dedupes candidates by what they
//! *compute* on the examples, a size-bounded bottom-up enumerator, and
//! acceptance through the same proof-carrying oracle
//! ([`verify_problem_code_strict`]). Reaching parity here without touching the
//! legacy path is the evidence that the shared core is real.
//!
//! The whole module is gated behind the `NSYNTH_UTBUS=1` environment variable.
//! When that variable is unset (the default), [`synthesize_utbus`] returns
//! `None` immediately and the legacy behaviour is byte-for-byte unchanged.
//!
//! Scope of this slice (intentionally minimal but *real*):
//!   * Output types ScalarInt / ArrayInt / Str / Bool, inferred from the
//!     problem signature ([`Utype`]).
//!   * An OE / example-match filter over **scalar** outputs of array programs
//!     (transform layers + reduce).
//!   * A bottom-up enumerator for the `ArrayInt` intermediate type that builds
//!     the same class the legacy array path covers: identity, an element-wise
//!     affine / abs / square map, sort, reverse, prefix-sum, and a predicate
//!     filter (incl. threshold preds vs optional scalar `k`) — each then
//!     reduced via sum / max / min / count and emitted as Mog source.
//!   * Acceptance via [`verify_problem_code_strict`]; the first verified
//!     candidate wins (all example-matching programs are tried cheapest-first
//!     so Sum/Count collisions on examples can still pass holdouts).
//!
//! Out of scope for now (next phase — tracked in the docs): higher-order
//! combinators whose lambdas are themselves synthesized, scalar/string/tree
//! output families, and a unified cost model across all of them.

use super::*;

/// Whether the UTBUS path is enabled. Reads `NSYNTH_UTBUS`; only the exact
/// value `"1"` turns it on, so the default (unset / anything else) leaves the
/// legacy path entirely untouched.
fn utbus_enabled() -> bool {
    std::env::var("NSYNTH_UTBUS")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// True when the Mog signature is `[i64] → i64` or `[i64], k: i64 → i64`
/// (array param name may be `arr`, `xs`, `a`, …).
fn signature_is_array_to_scalar(signature: &str) -> bool {
    signature_array_scalar_arity(signature).is_some()
}

/// `Some(0)` = single `[i64]` arg; `Some(1)` = `[i64]` + one `i64` scalar (`k`).
fn signature_array_scalar_arity(signature: &str) -> Option<usize> {
    let sig = signature.trim();
    let Some(ret) = sig.rsplit("->").next() else {
        return None;
    };
    let ret = ret.split_whitespace().next().unwrap_or(ret).trim();
    if !ret.starts_with("i64") {
        return None;
    }
    let Some(open) = sig.find('(') else {
        return None;
    };
    let Some(close) = sig[open..].find(')') else {
        return None;
    };
    let params = sig[open + 1..open + close].trim();
    if params.is_empty() {
        return None;
    }
    let parts: Vec<&str> = params
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() || parts.len() > 2 {
        return None;
    }
    let (_name0, ty0) = parts[0].split_once(':')?;
    let ty0 = ty0.trim();
    if !(ty0.starts_with("[i64]") || ty0.starts_with("[i64")) {
        return None;
    }
    if parts.len() == 1 {
        return Some(0);
    }
    let (_name1, ty1) = parts[1].split_once(':')?;
    let ty1 = ty1.trim();
    if ty1.starts_with("i64") {
        Some(1)
    } else {
        None
    }
}

/// The output types the typed core understands. Derived from a problem
/// signature; this slice only *enumerates* over `ArrayInt`, but the full type
/// lattice is modelled here so later phases can dispatch on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Utype {
    ScalarInt,
    ArrayInt,
    Str,
    Bool,
}

impl Utype {
    /// Best-effort inference of the *return* type from a Mog signature such as
    /// `fn name(arr: [i64]) -> i64`.
    fn from_return_signature(signature: &str) -> Option<Utype> {
        let ret = signature.rsplit("->").next()?.trim();
        // Strip a trailing block / body if it leaked into the slice.
        let ret = ret.split_whitespace().next().unwrap_or(ret).trim();
        if ret.starts_with("[i64]") || ret.starts_with("[i64") {
            Some(Utype::ArrayInt)
        } else if ret.starts_with("i64") {
            Some(Utype::ScalarInt)
        } else if ret.starts_with("bool") {
            Some(Utype::Bool)
        } else if ret.starts_with("str") || ret.starts_with("String") {
            Some(Utype::Str)
        } else {
            None
        }
    }
}

/// An element-wise map applied to every array element. Kept as a small closed
/// set so the enumerator stays size-bounded; each variant knows how to evaluate
/// itself (for the OE table) and how to emit its Mog expression over a bound
/// loop variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ElemMap {
    /// `item` (identity).
    Identity,
    /// `mul * item + add` (affine; covers double, negate, shift, ...).
    Affine { mul: i64, add: i64 },
    /// `abs(item)`.
    Abs,
    /// `item * item`.
    Square,
}

impl ElemMap {
    fn apply(self, item: i64) -> i64 {
        match self {
            ElemMap::Identity => item,
            ElemMap::Affine { mul, add } => mul.saturating_mul(item).saturating_add(add),
            ElemMap::Abs => item.abs(),
            ElemMap::Square => item.saturating_mul(item),
        }
    }

    /// Mog expression computing this map over the bound variable `var`.
    fn emit(self, var: &str) -> String {
        match self {
            ElemMap::Identity => var.to_string(),
            ElemMap::Affine { mul, add } => {
                let mul_part = if mul == 1 {
                    var.to_string()
                } else {
                    format!("({mul} * {var})")
                };
                if add == 0 {
                    mul_part
                } else if add > 0 {
                    format!("{mul_part} + {add}")
                } else {
                    format!("{mul_part} - {}", -add)
                }
            }
            ElemMap::Abs => {
                // Emitted as a guarded statement, not an inline expression, so
                // callers requesting `Abs` go through `emit_map_loop`.
                format!("{var}")
            }
            ElemMap::Square => format!("({var} * {var})"),
        }
    }

    /// Whether this map needs the statement-level (guarded) emission path.
    fn needs_guard(self) -> bool {
        matches!(self, ElemMap::Abs)
    }
}

/// A predicate selecting which elements survive a filter step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ElemPred {
    None,
    Positive,
    Negative,
    Even,
    Odd,
    NonZero,
    /// `item > k` (needs the optional scalar arg).
    GtK,
    /// `item < k`.
    LtK,
    /// `item == k`.
    EqK,
    /// `item != k`.
    NeK,
    /// `item >= k`.
    GeK,
    /// `item <= k`.
    LeK,
}

impl ElemPred {
    fn uses_k(self) -> bool {
        matches!(
            self,
            ElemPred::GtK
                | ElemPred::LtK
                | ElemPred::EqK
                | ElemPred::NeK
                | ElemPred::GeK
                | ElemPred::LeK
        )
    }

    fn keep(self, item: i64, k: Option<i64>) -> bool {
        match self {
            ElemPred::None => true,
            ElemPred::Positive => item > 0,
            ElemPred::Negative => item < 0,
            ElemPred::Even => item % 2 == 0,
            ElemPred::Odd => item % 2 != 0,
            ElemPred::NonZero => item != 0,
            ElemPred::GtK => k.map(|k| item > k).unwrap_or(false),
            ElemPred::LtK => k.map(|k| item < k).unwrap_or(false),
            ElemPred::EqK => k.map(|k| item == k).unwrap_or(false),
            ElemPred::NeK => k.map(|k| item != k).unwrap_or(false),
            ElemPred::GeK => k.map(|k| item >= k).unwrap_or(false),
            ElemPred::LeK => k.map(|k| item <= k).unwrap_or(false),
        }
    }

    /// Mog boolean condition over the bound variable `var`, or `None` for the
    /// no-op predicate (caller skips the guard entirely).
    fn emit(self, var: &str) -> Option<String> {
        match self {
            ElemPred::None => None,
            ElemPred::Positive => Some(format!("{var} > 0")),
            ElemPred::Negative => Some(format!("{var} < 0")),
            ElemPred::Even => Some(format!("{var} % 2 == 0")),
            ElemPred::Odd => Some(format!("{var} % 2 != 0")),
            ElemPred::NonZero => Some(format!("{var} != 0")),
            ElemPred::GtK => Some(format!("{var} > k")),
            ElemPred::LtK => Some(format!("{var} < k")),
            ElemPred::EqK => Some(format!("{var} == k")),
            ElemPred::NeK => Some(format!("{var} != k")),
            ElemPred::GeK => Some(format!("{var} >= k")),
            ElemPred::LeK => Some(format!("{var} <= k")),
        }
    }
}

/// Reorderings of the whole array. Ordering matters, so these go through
/// explicit emitted loops (or `.sort()`) rather than `.map`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ordering {
    None,
    Sort,
    Reverse,
}

/// Scalar reduction over the transformed array (Phase A parity with native_array).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reduce {
    Sum,
    Max,
    Min,
    Count,
}

impl Reduce {
    fn label(self) -> &'static str {
        match self {
            Reduce::Sum => "sum",
            Reduce::Max => "max",
            Reduce::Min => "min",
            Reduce::Count => "count",
        }
    }

    fn apply(self, arr: &[i64]) -> i64 {
        match self {
            Reduce::Sum => arr.iter().copied().fold(0i64, i64::saturating_add),
            Reduce::Max => arr.iter().copied().max().unwrap_or(0),
            Reduce::Min => arr.iter().copied().min().unwrap_or(0),
            Reduce::Count => arr.len() as i64,
        }
    }
}

/// One fully-formed array program, built bottom-up by stacking layers:
///   1. `pred`  — a predicate filter
///   2. `map`   — an element-wise map
///   3. `order` — a reordering (sort / reverse / none)
///   4. `prefix` — an optional running prefix-sum scan
///   5. `reduce` — scalar fold (sum / max / min / count)
///
/// Treating `order` and `prefix` as *separate* stacked layers (rather than one
/// fused reshape) is what lets the order-invariance of a plain sum be broken:
/// `sort` and `reverse` are only observable once a prefix-sum scan sits on top
/// of them, and the enumerator discovers that composition for free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArrayProgram {
    pred: ElemPred,
    map: ElemMap,
    order: Ordering,
    prefix: bool,
    reduce: Reduce,
}

impl ArrayProgram {
    /// Evaluate this array program on a concrete input array. Used to build the
    /// observational-equivalence signature *before* we pay for emission +
    /// verification. `k` is the optional scalar threshold arg.
    fn eval(&self, input: &[i64], k: Option<i64>) -> Vec<i64> {
        let mut out: Vec<i64> = input
            .iter()
            .copied()
            .filter(|&x| self.pred.keep(x, k))
            .map(|x| self.map.apply(x))
            .collect();
        match self.order {
            Ordering::None => {}
            Ordering::Sort => out.sort_unstable(),
            Ordering::Reverse => out.reverse(),
        }
        if self.prefix {
            let mut acc: i64 = 0;
            for v in out.iter_mut() {
                acc = acc.saturating_add(*v);
                *v = acc;
            }
        }
        out
    }

    /// Scalar result after transform + reduce (OE signature for Phase A).
    fn eval_scalar(&self, input: &[i64], k: Option<i64>) -> i64 {
        self.reduce.apply(&self.eval(input, k))
    }

    /// A rough size / cost used to keep the cheapest representative in the OE
    /// table and to bias enumeration toward simpler programs.
    fn cost(&self) -> usize {
        let pred_cost = if matches!(self.pred, ElemPred::None) {
            0
        } else {
            1
        };
        let map_cost = match self.map {
            ElemMap::Identity => 0,
            ElemMap::Affine { mul, add } => (mul != 1) as usize + (add != 0) as usize,
            ElemMap::Abs | ElemMap::Square => 1,
        };
        let order_cost = if matches!(self.order, Ordering::None) {
            0
        } else {
            1
        };
        let prefix_cost = self.prefix as usize;
        let reduce_cost = match self.reduce {
            Reduce::Sum => 0, // default / cheapest
            Reduce::Count => 1,
            Reduce::Max | Reduce::Min => 1,
        };
        pred_cost + map_cost + order_cost + prefix_cost + reduce_cost
    }

    /// A short label naming the layers this program stacked, used as the method
    /// suffix on the [`SolveResult`] (e.g. `map`, `sort_prefix_sum`,
    /// `filter_map`). Lets tests assert *which* structural layer was used.
    fn label(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if !matches!(self.pred, ElemPred::None) {
            parts.push("filter");
        }
        match self.order {
            Ordering::None => {}
            Ordering::Sort => parts.push("sort"),
            Ordering::Reverse => parts.push("reverse"),
        }
        if self.prefix {
            parts.push("prefix_sum");
        }
        if !matches!(self.reduce, Reduce::Sum) {
            parts.push(self.reduce.label());
        }
        if parts.is_empty() {
            // Pure element-wise map (or identity) reduced to a sum.
            "map".to_string()
        } else {
            parts.join("_")
        }
    }

    /// Emit Mog source for `fn {fn_name}(arr: [i64][, k: i64]) -> i64` that
    /// builds the transformed array and returns its reduced scalar.
    fn emit(&self, fn_name: &str, with_k: bool) -> String {
        let mut body = String::new();
        if with_k || self.pred.uses_k() {
            body.push_str(&format!("fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n"));
        } else {
            body.push_str(&format!("fn {fn_name}(arr: [i64]) -> i64 {{\n"));
        }

        // 1. Build the working array `a` by filtering + mapping the input.
        body.push_str("    a: [i64] = [];\n");
        body.push_str("    for item in arr {\n");
        let indent = if self.pred.emit("item").is_some() {
            "            "
        } else {
            "        "
        };
        if let Some(cond) = self.pred.emit("item") {
            body.push_str(&format!("        if {cond} {{\n"));
        }
        if self.map.needs_guard() {
            // abs: branch on sign.
            body.push_str(&format!("{indent}v: i64 = item;\n"));
            body.push_str(&format!("{indent}if v < 0 {{\n"));
            body.push_str(&format!("{indent}    v = 0 - v;\n"));
            body.push_str(&format!("{indent}}}\n"));
            body.push_str(&format!("{indent}a.push(v);\n"));
        } else {
            body.push_str(&format!("{indent}a.push({});\n", self.map.emit("item")));
        }
        if self.pred.emit("item").is_some() {
            body.push_str("        }\n");
        }
        body.push_str("    }\n");

        // 2. Apply the reordering layer.
        match self.order {
            Ordering::None => {}
            Ordering::Sort => {
                body.push_str("    a.sort();\n");
            }
            Ordering::Reverse => {
                body.push_str("    r: [i64] = [];\n");
                body.push_str("    i: i64 = a.len - 1;\n");
                body.push_str("    while i >= 0 {\n");
                body.push_str("        r.push(a[i]);\n");
                body.push_str("        i = i - 1;\n");
                body.push_str("    }\n");
                body.push_str("    a = r;\n");
            }
        }

        // 3. Apply the optional prefix-sum scan layer.
        if self.prefix {
            body.push_str("    p: [i64] = [];\n");
            body.push_str("    running: i64 = 0;\n");
            body.push_str("    for item in a {\n");
            body.push_str("        running = running + item;\n");
            body.push_str("        p.push(running);\n");
            body.push_str("    }\n");
            body.push_str("    a = p;\n");
        }

        // 4. Reduce the working array to the scalar output.
        match self.reduce {
            Reduce::Sum => {
                body.push_str("    total: i64 = 0;\n");
                body.push_str("    for item in a {\n");
                body.push_str("        total = total + item;\n");
                body.push_str("    }\n");
                body.push_str("    return total;\n");
            }
            Reduce::Count => {
                body.push_str("    return a.len;\n");
            }
            Reduce::Max => {
                body.push_str("    if a.len == 0 {\n");
                body.push_str("        return 0;\n");
                body.push_str("    }\n");
                body.push_str("    best: i64 = a[0];\n");
                body.push_str("    i: i64 = 1;\n");
                body.push_str("    while i < a.len {\n");
                body.push_str("        if a[i] > best {\n");
                body.push_str("            best = a[i];\n");
                body.push_str("        }\n");
                body.push_str("        i = i + 1;\n");
                body.push_str("    }\n");
                body.push_str("    return best;\n");
            }
            Reduce::Min => {
                body.push_str("    if a.len == 0 {\n");
                body.push_str("        return 0;\n");
                body.push_str("    }\n");
                body.push_str("    best: i64 = a[0];\n");
                body.push_str("    i: i64 = 1;\n");
                body.push_str("    while i < a.len {\n");
                body.push_str("        if a[i] < best {\n");
                body.push_str("            best = a[i];\n");
                body.push_str("        }\n");
                body.push_str("        i = i + 1;\n");
                body.push_str("    }\n");
                body.push_str("    return best;\n");
            }
        }
        body.push_str("}\n");
        body
    }
}

/// The bottom-up grammar: every leaf/unary layer the enumerator can stack.
/// Returned in (roughly) increasing-cost order so the first verified candidate
/// is also among the cheapest.
fn enumerate_array_programs(include_k_preds: bool) -> Vec<ArrayProgram> {
    let mut maps = vec![ElemMap::Identity];
    // Affine layer: a curated but generic set of multipliers/offsets that
    // covers double, negate, shift-by-constant, scale-and-shift.
    for &mul in &[1i64, 2, 3, -1, -2] {
        for &add in &[0i64, 1, -1, 2, -2] {
            if mul == 1 && add == 0 {
                continue; // identity already present
            }
            maps.push(ElemMap::Affine { mul, add });
        }
    }
    maps.push(ElemMap::Abs);
    maps.push(ElemMap::Square);

    let preds_no_k = [
        ElemPred::None,
        ElemPred::Positive,
        ElemPred::Negative,
        ElemPred::Even,
        ElemPred::Odd,
        ElemPred::NonZero,
    ];
    let preds_k = [
        ElemPred::GtK,
        ElemPred::LtK,
        ElemPred::EqK,
        ElemPred::NeK,
        ElemPred::GeK,
        ElemPred::LeK,
    ];
    let mut preds: Vec<ElemPred> = preds_no_k.to_vec();
    if include_k_preds {
        preds.extend_from_slice(&preds_k);
    }
    let orders = [Ordering::None, Ordering::Sort, Ordering::Reverse];
    let reduces = [Reduce::Sum, Reduce::Count, Reduce::Max, Reduce::Min];

    let mut programs = Vec::new();
    for &pred in &preds {
        for &map in &maps {
            for &order in &orders {
                for &prefix in &[false, true] {
                    for &reduce in &reduces {
                        programs.push(ArrayProgram {
                            pred,
                            map,
                            order,
                            prefix,
                            reduce,
                        });
                    }
                }
            }
        }
    }
    // Cheapest first: stable sort keeps the curated within-cost order.
    programs.sort_by_key(|p| p.cost());
    programs
}

/// Pull observable `[i64]` inputs (+ optional scalar `k`) out of the examples.
fn array_examples(problem: &Problem) -> Option<(Vec<Vec<i64>>, Vec<Option<i64>>)> {
    let mut inputs = Vec::new();
    let mut ks = Vec::new();
    for example in &problem.examples {
        let arr = match example.inputs.first()?.as_i64_slice() {
            Some(values) => values,
            None => return None,
        };
        let mut k = None;
        for (i, value) in example.inputs[1..].iter().enumerate() {
            match value {
                Value::Int(v) if i == 0 => k = Some(*v),
                Value::Int(_) if i > 0 => return None, // only one scalar supported
                _ => return None,
            }
        }
        inputs.push(arr);
        ks.push(k);
    }
    if inputs.is_empty() {
        None
    } else {
        Some((inputs, ks))
    }
}

/// Public entry point. Returns `None` unless `NSYNTH_UTBUS=1`. When enabled,
/// runs the typed bottom-up enumerator over the array transform set and returns
/// the first candidate that passes [`verify_problem_code_strict`] (examples +
/// holdouts).
pub(super) fn synthesize_utbus(problem: &Problem) -> Option<SolveResult> {
    if !utbus_enabled() {
        return None;
    }

    // This slice only reaches parity for scalar-output array problems.
    if Utype::from_return_signature(problem.signature) != Some(Utype::ScalarInt) {
        return None;
    }
    let fn_name = problem.function_name();
    if fn_name.is_empty() {
        return None;
    }
    // Emit uses `arr` as the parameter name; accept `[i64]→i64` or `[i64],k→i64`.
    let n_scalar = signature_array_scalar_arity(problem.signature)?;
    let with_k = n_scalar >= 1;

    let (inputs, ks) = array_examples(problem)?;
    // Signature arity must agree with example shapes.
    if with_k {
        if ks.iter().any(|k| k.is_none()) {
            return None;
        }
    } else if ks.iter().any(|k| k.is_some()) {
        // Examples carry a scalar the signature doesn't declare — out of scope.
        return None;
    }

    let mut expected: Vec<i64> = Vec::with_capacity(problem.examples.len());
    for example in &problem.examples {
        match &example.expected {
            Value::Int(v) => expected.push(*v),
            _ => return None,
        }
    }

    // Keep EVERY program whose scalar outputs match the examples (cheapest
    // first). Do not OE-collapse to a single cheapest program: Sum vs Count/Max
    // can agree on the visible examples and diverge on holdouts — strict verify
    // must be allowed to try each match.
    let mut matching: Vec<ArrayProgram> = enumerate_array_programs(with_k)
        .into_iter()
        .filter(|program| {
            if program.pred.uses_k() && !with_k {
                return false;
            }
            inputs
                .iter()
                .zip(ks.iter())
                .zip(expected.iter())
                .all(|((arr, k), &y)| program.eval_scalar(arr, *k) == y)
        })
        .collect();
    matching.sort_by_key(|p| p.cost());

    for program in matching {
        let code = program.emit(fn_name, with_k);
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: format!("utbus_array_{}", program.label()),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Value;

    /// Build a self-contained array→sum problem from a closure that maps an
    /// input array to its scalar (summed-transform) output. Examples and
    /// holdouts both get distinct inputs so the strict oracle exercises both.
    fn array_problem(
        name: &'static str,
        signature: &'static str,
        example_inputs: &[&[i64]],
        holdout_inputs: &[&[i64]],
        oracle: impl Fn(&[i64]) -> i64,
    ) -> Problem {
        let mk = |arr: &&[i64]| Example {
            inputs: vec![Value::int_array(arr)],
            expected: Value::Int(oracle(arr)),
        };
        Problem {
            name: name.to_string(),
            category: "arrays",
            description: "utbus parity test",
            signature,
            examples: example_inputs.iter().map(mk).collect(),
            holdouts: holdout_inputs.iter().map(mk).collect(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    /// Shared lock guarding the process-global `NSYNTH_UTBUS` env var. Every
    /// test that reads or writes the gate goes through this so they cannot race
    /// each other's env mutations under the default parallel test runner.
    fn gate_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::Mutex;
        static GATE_LOCK: Mutex<()> = Mutex::new(());
        GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Run the core with the env gate forced on for the duration of the call.
    fn solve_with_gate(problem: &Problem) -> Option<SolveResult> {
        let _guard = gate_lock();
        std::env::set_var("NSYNTH_UTBUS", "1");
        let result = synthesize_utbus(problem);
        std::env::remove_var("NSYNTH_UTBUS");
        result
    }

    fn assert_solves(problem: &Problem, expect_reshape: &str) {
        let result = solve_with_gate(problem)
            .unwrap_or_else(|| panic!("utbus failed to solve {}", problem.name));
        assert!(result.success, "{} should report success", problem.name);
        assert!(
            result.method.starts_with("utbus_"),
            "{} method should be utbus_*, got {}",
            problem.name,
            result.method
        );
        assert!(
            result.method.contains(expect_reshape),
            "{} method {} should mention reshape {}",
            problem.name,
            result.method,
            expect_reshape
        );
        // The result must be proof-carrying: re-verify independently.
        assert!(
            crate::runtime::verify_problem_code_strict(problem, &result.code).is_ok(),
            "{} emitted code must pass the strict oracle:\n{}",
            problem.name,
            result.code
        );
    }

    #[test]
    fn utbus_disabled_by_default() {
        // Without the env gate the core must be a no-op so the legacy path is
        // byte-for-byte unchanged. Hold the shared gate lock so a concurrent
        // `solve_with_gate` can't transiently set the var underneath us.
        let _guard = gate_lock();
        std::env::remove_var("NSYNTH_UTBUS");
        let problem = array_problem(
            "double_sum_gateoff",
            "fn double_sum_gateoff(arr: [i64]) -> i64",
            &[&[1, 2, 3], &[4, 5]],
            &[&[0, 7], &[-1, 2, 3]],
            |arr| arr.iter().map(|x| x * 2).sum(),
        );
        assert!(
            synthesize_utbus(&problem).is_none(),
            "utbus must be a no-op when NSYNTH_UTBUS is unset"
        );
    }

    #[test]
    fn utbus_solves_double() {
        // double: elementwise affine map (mul=2), summed.
        let problem = array_problem(
            "double_sum",
            "fn double_sum(arr: [i64]) -> i64",
            &[&[1, 2, 3], &[5, 1], &[4, 4, 4], &[2, 7, 1, 0]],
            &[&[10, -5, 2], &[1, 2, 3, 4]],
            |arr| arr.iter().map(|x| x * 2).sum(),
        );
        assert_solves(&problem, "map");
    }

    #[test]
    fn utbus_solves_abs() {
        // abs: elementwise abs map, summed.
        let problem = array_problem(
            "abs_sum",
            "fn abs_sum(arr: [i64]) -> i64",
            &[&[-1, 2, -3], &[5, -1], &[-4, 4], &[2, -7, 1, 0]],
            &[&[-10, -5, 2], &[1, -2, 3, -4]],
            |arr| arr.iter().map(|x| x.abs()).sum(),
        );
        assert_solves(&problem, "map");
    }

    #[test]
    fn utbus_solves_sort() {
        // A plain sum is order-invariant, so `sort` is only *observable* once a
        // prefix-sum scan sits on top of it. This problem — the sum of the
        // prefix sums of the SORTED array — therefore genuinely requires the
        // bottom-up core to stack the sort layer under the prefix-sum layer.
        let problem = array_problem(
            "sorted_prefix_sum",
            "fn sorted_prefix_sum(arr: [i64]) -> i64",
            &[&[3, 1, 2], &[5, 1], &[4, 2, 4], &[2, 7, 1, 0]],
            &[&[10, -5, 2], &[1, 4, 3, 2]],
            |arr| {
                let mut v = arr.to_vec();
                v.sort_unstable();
                let mut acc = 0i64;
                let mut total = 0i64;
                for x in v {
                    acc += x;
                    total += acc;
                }
                total
            },
        );
        assert_solves(&problem, "sort");
    }

    #[test]
    fn utbus_solves_reverse() {
        // reverse then prefix-sum is order-sensitive, so this genuinely
        // requires the reverse reshape stacked under the prefix-sum scan.
        let problem = array_problem(
            "reverse_prefix_sum",
            "fn reverse_prefix_sum(arr: [i64]) -> i64",
            &[&[1, 2, 3], &[5, 1], &[4, 2, 1], &[2, 7, 1, 0]],
            &[&[10, 5, 2], &[1, 2, 3, 4]],
            |arr| {
                let mut v = arr.to_vec();
                v.reverse();
                let mut acc = 0i64;
                let mut total = 0i64;
                for x in v {
                    acc += x;
                    total += acc;
                }
                total
            },
        );
        assert_solves(&problem, "reverse");
    }

    #[test]
    fn utbus_solves_prefix_sum() {
        // prefix-sum then total: sum of running prefix sums.
        let problem = array_problem(
            "prefix_sum_total",
            "fn prefix_sum_total(arr: [i64]) -> i64",
            &[&[1, 2, 3], &[5, 1], &[4, 2, 1], &[2, 7, 1, 0]],
            &[&[10, 5, 2], &[1, 2, 3, 4]],
            |arr| {
                let mut acc = 0i64;
                let mut total = 0i64;
                for &x in arr {
                    acc += x;
                    total += acc;
                }
                total
            },
        );
        assert_solves(&problem, "prefix_sum");
    }

    #[test]
    fn utbus_solves_filter_even() {
        // filter-even then sum: predicate filter in the array core.
        let problem = array_problem(
            "filter_even_sum",
            "fn filter_even_sum(arr: [i64]) -> i64",
            &[&[1, 2, 3, 4], &[5, 6], &[2, 4, 6], &[1, 3, 5, 8]],
            &[&[10, 5, 2], &[1, 2, 3, 4]],
            |arr| arr.iter().filter(|x| *x % 2 == 0).sum(),
        );
        assert_solves(&problem, "filter");
    }

    #[test]
    fn utbus_type_inference_from_signature() {
        assert_eq!(
            Utype::from_return_signature("fn f(arr: [i64]) -> i64"),
            Some(Utype::ScalarInt)
        );
        assert_eq!(
            Utype::from_return_signature("fn f(arr: [i64]) -> [i64]"),
            Some(Utype::ArrayInt)
        );
        assert_eq!(
            Utype::from_return_signature("fn f(s: str) -> bool"),
            Some(Utype::Bool)
        );
    }

    #[test]
    fn signature_accepts_any_array_param_name() {
        assert!(signature_is_array_to_scalar("fn f(arr: [i64]) -> i64"));
        assert!(signature_is_array_to_scalar("fn f(xs: [i64]) -> i64"));
        assert!(signature_is_array_to_scalar("fn sum(a: [i64]) -> i64"));
        assert!(signature_is_array_to_scalar("fn f(arr: [i64], k: i64) -> i64"));
        assert_eq!(signature_array_scalar_arity("fn f(arr: [i64], k: i64) -> i64"), Some(1));
        assert_eq!(signature_array_scalar_arity("fn f(arr: [i64]) -> i64"), Some(0));
        assert!(!signature_is_array_to_scalar("fn f(arr: [i64]) -> [i64]"));
        assert!(!signature_is_array_to_scalar("fn f(x: i64) -> i64"));
        assert!(!signature_is_array_to_scalar(
            "fn f(a: [i64], k: i64, m: i64) -> i64"
        ));
    }

    #[test]
    fn utbus_solves_with_xs_param_name() {
        let problem = array_problem(
            "double_sum_xs",
            "fn double_sum_xs(xs: [i64]) -> i64",
            &[&[1, 2, 3], &[4], &[0, -1]],
            &[&[5, 5], &[10]],
            |arr| arr.iter().map(|x| x * 2).sum(),
        );
        assert_solves(&problem, "map");
    }

    #[test]
    fn utbus_solves_array_max() {
        let problem = array_problem(
            "array_max",
            "fn array_max(arr: [i64]) -> i64",
            &[&[1, 5, 3], &[-2, -9, 0], &[7], &[4, 4, 1]],
            &[&[10, -5, 2], &[1, 2, 3, 4]],
            |arr| arr.iter().copied().max().unwrap_or(0),
        );
        assert_solves(&problem, "max");
    }

    #[test]
    fn utbus_solves_array_min() {
        let problem = array_problem(
            "array_min",
            "fn array_min(arr: [i64]) -> i64",
            &[&[1, 5, 3], &[-2, -9, 0], &[7], &[4, 4, 1]],
            &[&[10, -5, 2], &[1, 2, 3, 4]],
            |arr| arr.iter().copied().min().unwrap_or(0),
        );
        assert_solves(&problem, "min");
    }

    #[test]
    fn utbus_solves_count_positives() {
        let problem = array_problem(
            "count_positives",
            "fn count_positives(arr: [i64]) -> i64",
            &[&[-1, 2, -3, 4], &[-5, -1], &[1, 2, 3], &[0, 1, -1]],
            &[&[10, -5, 2], &[-1, -2, -3]],
            |arr| arr.iter().filter(|&&x| x > 0).count() as i64,
        );
        assert_solves(&problem, "count");
    }

    #[test]
    fn eval_scalar_covers_all_reduces() {
        let base = ArrayProgram {
            pred: ElemPred::None,
            map: ElemMap::Identity,
            order: Ordering::None,
            prefix: false,
            reduce: Reduce::Sum,
        };
        let xs = [3i64, -1, 5, 0];
        assert_eq!(
            ArrayProgram {
                reduce: Reduce::Sum,
                ..base
            }
            .eval_scalar(&xs, None),
            7
        );
        assert_eq!(
            ArrayProgram {
                reduce: Reduce::Max,
                ..base
            }
            .eval_scalar(&xs, None),
            5
        );
        assert_eq!(
            ArrayProgram {
                reduce: Reduce::Min,
                ..base
            }
            .eval_scalar(&xs, None),
            -1
        );
        assert_eq!(
            ArrayProgram {
                reduce: Reduce::Count,
                ..base
            }
            .eval_scalar(&xs, None),
            4
        );
        assert_eq!(
            ArrayProgram {
                pred: ElemPred::Positive,
                reduce: Reduce::Count,
                ..base
            }
            .eval_scalar(&xs, None),
            2
        );
        assert_eq!(
            ArrayProgram {
                pred: ElemPred::GtK,
                reduce: Reduce::Count,
                ..base
            }
            .eval_scalar(&xs, Some(0)),
            2
        );
    }

    #[test]
    fn enumerate_includes_each_reduce() {
        let programs = enumerate_array_programs(false);
        assert!(programs.iter().any(|p| p.reduce == Reduce::Sum));
        assert!(programs.iter().any(|p| p.reduce == Reduce::Max));
        assert!(programs.iter().any(|p| p.reduce == Reduce::Min));
        assert!(programs.iter().any(|p| p.reduce == Reduce::Count));
        assert!(programs.iter().all(|p| !p.pred.uses_k()));
        let with_k = enumerate_array_programs(true);
        assert!(with_k.iter().any(|p| p.pred == ElemPred::GtK));
        // Plain max must be cheaper than filter+max of the same inputs' max.
        let plain_max = programs
            .iter()
            .find(|p| {
                matches!(p.reduce, Reduce::Max)
                    && matches!(p.pred, ElemPred::None)
                    && matches!(p.map, ElemMap::Identity)
                    && matches!(p.order, Ordering::None)
                    && !p.prefix
            })
            .expect("plain max");
        assert_eq!(plain_max.cost(), 1);
        assert_eq!(plain_max.label(), "max");
    }

    /// Array + scalar-k problem helper for threshold predicates.
    fn array_k_problem(
        name: &'static str,
        signature: &'static str,
        examples: &[(&[i64], i64, i64)],
        holdouts: &[(&[i64], i64, i64)],
    ) -> Problem {
        let mk = |(arr, k, y): &(&[i64], i64, i64)| Example {
            inputs: vec![Value::int_array(arr), Value::Int(*k)],
            expected: Value::Int(*y),
        };
        Problem {
            name: name.to_string(),
            category: "arrays",
            description: "utbus k-parity test",
            signature,
            examples: examples.iter().map(mk).collect(),
            holdouts: holdouts.iter().map(mk).collect(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    #[test]
    fn utbus_solves_count_greater_than_k() {
        let problem = array_k_problem(
            "count_gt",
            "fn count_gt(arr: [i64], k: i64) -> i64",
            &[
                (&[1, 5, 3, 0], 2, 2),
                (&[-1, 0, 1], 0, 1),
                (&[4, 4, 4], 4, 0),
                (&[10, 2, 8], 5, 2),
            ],
            &[(&[1, 2, 3, 4], 2, 2), (&[9, 1], 5, 1)],
        );
        assert_solves(&problem, "count");
    }

    #[test]
    fn utbus_solves_count_equal_k() {
        let problem = array_k_problem(
            "count_eq",
            "fn count_eq(arr: [i64], k: i64) -> i64",
            &[
                (&[1, 2, 1, 3, 1], 1, 3),
                (&[5, 5, 0], 5, 2),
                (&[0, 0, 0], 1, 0),
            ],
            &[(&[2, 2, 3], 2, 2), (&[7], 7, 1)],
        );
        assert_solves(&problem, "count");
    }

    #[test]
    fn eval_gt_k_filter() {
        let prog = ArrayProgram {
            pred: ElemPred::GtK,
            map: ElemMap::Identity,
            order: Ordering::None,
            prefix: false,
            reduce: Reduce::Count,
        };
        assert_eq!(prog.eval_scalar(&[1, 5, 3, 0], Some(2)), 2);
        assert_eq!(prog.emit("count_gt", true).contains("item > k"), true);
    }
}
