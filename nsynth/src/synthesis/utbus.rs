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
//!   * An OE table over the example outputs of array-valued candidates.
//!   * A bottom-up enumerator for the `ArrayInt` intermediate type that builds
//!     the same class the legacy array path covers: identity, an element-wise
//!     affine / abs / square map, sort, reverse, prefix-sum, and a predicate
//!     filter — each then reduced to the scalar output and emitted as Mog
//!     source.
//!   * Acceptance via [`verify_problem_code_strict`]; the first verified
//!     candidate wins.
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
}

impl ElemPred {
    fn keep(self, item: i64) -> bool {
        match self {
            ElemPred::None => true,
            ElemPred::Positive => item > 0,
            ElemPred::Negative => item < 0,
            ElemPred::Even => item % 2 == 0,
            ElemPred::Odd => item % 2 != 0,
            ElemPred::NonZero => item != 0,
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

/// One fully-formed array program, built bottom-up by stacking layers:
///   1. `pred`  — a predicate filter
///   2. `map`   — an element-wise map
///   3. `order` — a reordering (sort / reverse / none)
///   4. `prefix` — an optional running prefix-sum scan
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
}

impl ArrayProgram {
    /// Evaluate this array program on a concrete input array. Used to build the
    /// observational-equivalence signature *before* we pay for emission +
    /// verification.
    fn eval(&self, input: &[i64]) -> Vec<i64> {
        let mut out: Vec<i64> = input
            .iter()
            .copied()
            .filter(|&x| self.pred.keep(x))
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
        pred_cost + map_cost + order_cost + prefix_cost
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
        if parts.is_empty() {
            // Pure element-wise map (or identity) reduced to a sum.
            "map".to_string()
        } else {
            parts.join("_")
        }
    }

    /// Emit Mog source for `fn {fn_name}(arr: [i64]) -> i64` that builds the
    /// transformed array and returns its sum. The slice's "array transform set"
    /// is observed through this scalar reduction (the only output type the
    /// benchmark array problems expose).
    fn emit(&self, fn_name: &str) -> String {
        let mut body = String::new();
        body.push_str(&format!("fn {fn_name}(arr: [i64]) -> i64 {{\n"));

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

        // 4. Reduce the working array to the scalar output by summing.
        body.push_str("    total: i64 = 0;\n");
        body.push_str("    for item in a {\n");
        body.push_str("        total = total + item;\n");
        body.push_str("    }\n");
        body.push_str("    return total;\n");
        body.push_str("}\n");
        body
    }
}

/// The bottom-up grammar: every leaf/unary layer the enumerator can stack.
/// Returned in (roughly) increasing-cost order so the first verified candidate
/// is also among the cheapest.
fn enumerate_array_programs() -> Vec<ArrayProgram> {
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

    let preds = [
        ElemPred::None,
        ElemPred::Positive,
        ElemPred::Negative,
        ElemPred::Even,
        ElemPred::Odd,
        ElemPred::NonZero,
    ];
    let orders = [Ordering::None, Ordering::Sort, Ordering::Reverse];

    let mut programs = Vec::new();
    for &pred in &preds {
        for &map in &maps {
            for &order in &orders {
                for &prefix in &[false, true] {
                    programs.push(ArrayProgram {
                        pred,
                        map,
                        order,
                        prefix,
                    });
                }
            }
        }
    }
    // Cheapest first: stable sort keeps the curated within-cost order.
    programs.sort_by_key(|p| p.cost());
    programs
}

/// Pull the observable `[i64]` input arrays out of the examples so the OE table
/// can be built. Returns `None` unless the first argument is an array and every
/// other argument is scalar (the class this slice covers).
fn array_inputs(problem: &Problem) -> Option<Vec<Vec<i64>>> {
    let mut inputs = Vec::new();
    for example in &problem.examples {
        let arr = match example.inputs.first()? {
            Value::Array(values) => values.clone(),
            _ => return None,
        };
        // Extra scalar args are tolerated (the reducer ignores them) but any
        // non-scalar extra argument falls outside this slice.
        for value in &example.inputs[1..] {
            if !matches!(value, Value::Int(_)) {
                return None;
            }
        }
        inputs.push(arr);
    }
    if inputs.is_empty() {
        None
    } else {
        Some(inputs)
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
    // The reducer signature we emit is `fn name(arr: [i64]) -> i64`; only take
    // problems whose declared signature matches so the emitted wrapper type-checks.
    if !problem.signature.contains("arr: [i64]") {
        return None;
    }

    let inputs = array_inputs(problem)?;

    // Observational-equivalence table: signature (the per-example output
    // vectors) -> cheapest array program producing it. Dedup happens here so we
    // only emit + verify one representative per behaviour.
    let mut oe: std::collections::HashMap<Vec<Vec<i64>>, ArrayProgram> =
        std::collections::HashMap::new();

    for program in enumerate_array_programs() {
        let signature: Vec<Vec<i64>> = inputs.iter().map(|arr| program.eval(arr)).collect();
        match oe.get(&signature) {
            Some(existing) if existing.cost() <= program.cost() => continue,
            _ => {
                oe.insert(signature, program);
            }
        }
    }

    // Order the deduped representatives cheapest-first, then emit + verify.
    let mut representatives: Vec<ArrayProgram> = oe.into_values().collect();
    representatives.sort_by_key(|p| p.cost());

    for program in representatives {
        let code = program.emit(fn_name);
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
            inputs: vec![Value::Array(arr.to_vec())],
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
}
