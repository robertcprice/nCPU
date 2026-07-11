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
//! The module is gated behind `NSYNTH_UTBUS`:
//! - `NSYNTH_UTBUS=1` — full Phase A (closed families + filter→map→reduce enum)
//! - `NSYNTH_UTBUS=closed` — closed families only (cheap A3/A4/A5/k parity)
//! - unset — [`synthesize_utbus`] returns `None` (legacy path unchanged)
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
//!     reduced via sum / max / min / count / product and emitted as Mog source.
//!   * Closed dual-accum (range, second_max) and pairwise-scan (max abs diff,
//!     count adjacent diffs) families for native A3/A4 parity.
//!   * Acceptance via [`verify_problem_code_strict`]; the first verified
//!     candidate wins (all example-matching programs are tried cheapest-first
//!     so Sum/Count collisions on examples can still pass holdouts).
//!
//! Out of scope for now (next phase — tracked in the docs): higher-order
//! combinators whose lambdas are themselves synthesized, scalar/string/tree
//! output families, and a unified cost model across all of them.

use super::*;

/// Whether the UTBUS path is enabled. Reads `NSYNTH_UTBUS`:
/// - `"1"` — full Phase A (closed families + filter→map→reduce enum)
/// - `"closed"` — closed families only (dual/pairwise/index/k)
/// - `"0"` / unset — off (legacy path unchanged; product bins default to `closed`)
fn utbus_mode() -> Option<&'static str> {
    match std::env::var("NSYNTH_UTBUS").ok().as_deref() {
        Some("1") => Some("full"),
        Some("closed") => Some("closed"),
        Some("0") | None => None,
        // Treat unknown values as off (never-wrong: don't surprise).
        Some(_) => None,
    }
}

fn utbus_enabled() -> bool {
    utbus_mode().is_some()
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
    /// `item == 0`.
    Zero,
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
            ElemPred::Zero => item == 0,
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
            ElemPred::Zero => Some(format!("{var} == 0")),
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
    /// Product of elements (empty → 1).
    Product,
    /// Bitwise XOR fold (empty → 0).
    Xor,
    /// Bitwise OR fold (empty → 0).
    BitOr,
    /// Bitwise AND fold (empty → -1).
    BitAnd,
}

impl Reduce {
    fn label(self) -> &'static str {
        match self {
            Reduce::Sum => "sum",
            Reduce::Max => "max",
            Reduce::Min => "min",
            Reduce::Count => "count",
            Reduce::Product => "product",
            Reduce::Xor => "xor",
            Reduce::BitOr => "bitor",
            Reduce::BitAnd => "bitand",
        }
    }

    fn apply(self, arr: &[i64]) -> i64 {
        match self {
            Reduce::Sum => arr.iter().copied().fold(0i64, i64::saturating_add),
            Reduce::Max => arr.iter().copied().max().unwrap_or(0),
            Reduce::Min => arr.iter().copied().min().unwrap_or(0),
            Reduce::Count => arr.len() as i64,
            Reduce::Product => arr.iter().copied().fold(1i64, i64::saturating_mul),
            Reduce::Xor => arr.iter().copied().fold(0i64, |a, b| a ^ b),
            Reduce::BitOr => arr.iter().copied().fold(0i64, |a, b| a | b),
            Reduce::BitAnd => arr.iter().copied().fold(-1i64, |a, b| a & b),
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
            Reduce::Product => 1,
            Reduce::Xor | Reduce::BitOr | Reduce::BitAnd => 1,
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
            Reduce::Product => {
                body.push_str("    total: i64 = 1;\n");
                body.push_str("    for item in a {\n");
                body.push_str("        total = total * item;\n");
                body.push_str("    }\n");
                body.push_str("    return total;\n");
            }
            Reduce::Xor => {
                body.push_str("    total: i64 = 0;\n");
                body.push_str("    for item in a {\n");
                body.push_str("        total = total ^ item;\n");
                body.push_str("    }\n");
                body.push_str("    return total;\n");
            }
            Reduce::BitOr => {
                body.push_str("    total: i64 = 0;\n");
                body.push_str("    for item in a {\n");
                body.push_str("        total = total | item;\n");
                body.push_str("    }\n");
                body.push_str("    return total;\n");
            }
            Reduce::BitAnd => {
                body.push_str("    total: i64 = 0 - 1;\n");
                body.push_str("    for item in a {\n");
                body.push_str("        total = total & item;\n");
                body.push_str("    }\n");
                body.push_str("    return total;\n");
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
        ElemPred::Zero,
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
    let reduces = [
        Reduce::Sum,
        Reduce::Count,
        Reduce::Max,
        Reduce::Min,
        Reduce::Product,
        Reduce::Xor,
        Reduce::BitOr,
        Reduce::BitAnd,
    ];

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

/// Euclidean GCD on non-negative ints (Mog `%` matches rem toward zero for ≥0).
fn i64_gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn i64_lcm(a: i64, b: i64) -> Option<i64> {
    let g = i64_gcd(a, b);
    if g == 0 {
        return Some(0);
    }
    let aa = a.abs();
    let bb = b.abs();
    aa.checked_div(g)?.checked_mul(bb)
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

/// Dual-accumulator programs (native A3 parity): two coupled running state
/// variables reduced to one scalar. Closed set — not stacked on [`ArrayProgram`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DualAccum {
    /// `max(arr) - min(arr)`.
    Range,
    /// Second-largest element (teacher cascade; ties keep first).
    SecondMax,
    /// Second-smallest element (teacher cascade; ties keep first).
    SecondMin,
    /// One-buy one-sell max profit (0 if none).
    StockProfit,
    /// Sum of running maxima: for each prefix, add max so far.
    PrefixMaxSum,
    /// Sum of running minima.
    PrefixMinSum,
    /// Maximum contiguous subarray sum (Kadane).
    MaxSubarraySum,
    /// Minimum contiguous subarray sum (Kadane dual).
    MinSubarraySum,
    /// Sort ascending, return `sorted[len/2]` (upper-middle for even length).
    Median,
    /// GCD of all elements (abs values; empty → None).
    GcdAll,
    /// LCM of all elements (abs; empty → None; overflow → None).
    LcmAll,
    /// Truncating mean: `sum / len` (empty → None).
    MeanTrunc,
    /// Sum of squares.
    SumSquares,
    /// Sum of absolute values.
    AbsSum,
    /// Maximum absolute value.
    MaxAbs,
    /// Minimum absolute value (empty → 0).
    MinAbs,
    /// Minimum strictly-positive element (none → 0).
    MinPositive,
    /// Count of negative elements.
    CountNegatives,
    /// Count of even elements.
    CountEvens,
    /// Sum of positive elements.
    SumPositives,
    /// Sum of negative elements.
    SumNegatives,
    /// Count of odd elements.
    CountOdds,
    /// Length of the array.
    Len,
    /// Whether the array is empty (1/0).
    IsEmpty,
    /// 1 if all elements are equal (empty → 1), else 0.
    AllEqual,
    /// 1 if any element is positive, else 0.
    AnyPositive,
    /// 1 if any element is negative, else 0.
    AnyNegative,
    /// 1 if any element is zero, else 0.
    AnyZero,
    /// 1 if every element is positive (empty → 1), else 0.
    AllPositive,
    /// 1 if every element is negative (empty → 1), else 0.
    AllNegative,
    /// Count of zero elements.
    CountZeros,
    /// 1 if any value appears more than once, else 0.
    HasDuplicate,
    /// Maximum among strictly-negative elements (none → 0).
    MaxNegative,
    /// Sum of even-valued elements (value parity, not index).
    SumEvenValues,
    /// Sum of odd-valued elements (value parity, not index).
    SumOddValues,
    /// 1 if every element is ≥ 0 (empty → 1), else 0.
    AllNonNegative,
    /// Count of non-zero elements.
    CountNonZeros,
    /// Alternating sum `a0 - a1 + a2 - a3 + …`.
    AlternatingSum,
    /// Product of positive elements (none → 1).
    ProductPositives,
    /// Truncating mean of absolute values: `abs_sum / len`.
    MeanAbsTrunc,
    /// First strictly-positive element (none → 0).
    FirstPositive,
    /// Last strictly-positive element (none → 0).
    LastPositive,
    /// First strictly-negative element (none → 0).
    FirstNegative,
    /// Last strictly-negative element (none → 0).
    LastNegative,
    /// Maximum among strictly-positive elements (none → 0).
    MaxPositive,
    /// Minimum among strictly-negative elements (none → 0).
    MinNegative,
    /// Product of negative elements (none → 1).
    ProductNegatives,
    /// Sum of cubes.
    SumCubes,
    /// Count of elements strictly greater than truncating mean.
    CountGtMean,
    /// Count of elements strictly less than truncating mean.
    CountLtMean,
    /// 1 if arr is a palindrome (empty → 1), else 0.
    IsPalindrome,
    /// Product of even-valued elements (none → 1).
    ProductEvens,
    /// Product of odd-valued elements (none → 1).
    ProductOdds,
    /// 1 if every element is ≤ 0 (empty → 1), else 0.
    AllNonPositive,
    /// Weighted index sum `Σ i * arr[i]` (0-based).
    DotIndex,
    /// Sum of squared diffs from truncating mean (SSE).
    SumSqDiffMean,
    /// 1 if any element is non-zero, else 0.
    AnyNonZero,
    /// XOR of all elements (empty → 0).
    XorAll,
    /// Product of nonzero elements (none → 1).
    ProductNonZeros,
    /// Bitwise OR of all elements (empty → 0).
    OrAll,
    /// Bitwise AND of all elements (empty → -1).
    AndAll,
    /// Count of elements equal to truncating mean.
    CountEqMean,
    /// Product of absolute values (empty → 1).
    ProductAbs,
    /// Count of non-negative elements (`x >= 0`).
    CountNonNegatives,
    /// Count of strictly positive elements (`x > 0`).
    CountPositives,
    /// Max among even-valued elements (none → 0).
    MaxEvenValue,
    /// Max among odd-valued elements (none → 0).
    MaxOddValue,
    /// Min among even-valued elements (none → 0).
    MinEvenValue,
    /// Min among odd-valued elements (none → 0).
    MinOddValue,
    /// `max(|x|) - min(|x|)` (empty → 0).
    AbsRange,
    /// Product of non-negative elements (none → 1).
    ProductNonNegatives,
    /// Sum of non-negative elements (none → 0).
    SumNonNegatives,
    /// Sum of non-positive elements (`x <= 0`, none → 0).
    SumNonPositives,
    /// Count of non-positive elements (`x <= 0`).
    CountNonPositives,
    /// Product of non-positive elements (none → 1).
    ProductNonPositives,
    /// Sum of |x| for even-valued elements.
    SumAbsEvens,
    /// Sum of |x| for odd-valued elements.
    SumAbsOdds,
    /// Product of |x| for even-valued elements (none → 1).
    ProductAbsEvens,
    /// Product of |x| for odd-valued elements (none → 1).
    ProductAbsOdds,
    /// XOR of absolute values (empty → 0).
    XorAbsAll,
    /// Bitwise AND of absolute values (empty → -1).
    AndAbsAll,
    /// Bitwise OR of absolute values (empty → 0).
    OrAbsAll,
    /// XOR of |x| for even-valued elements (none → 0).
    XorAbsEvens,
    /// XOR of |x| for odd-valued elements (none → 0).
    XorAbsOdds,
    /// AND of |x| for even-valued elements (none → -1).
    AndAbsEvens,
    /// AND of |x| for odd-valued elements (none → -1).
    AndAbsOdds,
    /// OR of |x| for even-valued elements (none → 0).
    OrAbsEvens,
    /// OR of |x| for odd-valued elements (none → 0).
    OrAbsOdds,
    /// Sum of x*x for even-valued elements.
    SumSquaresEvens,
    /// Sum of x*x for odd-valued elements.
    SumSquaresOdds,
    /// Sum of x^3 for even-valued elements.
    SumCubesEvens,
    /// Sum of x^3 for odd-valued elements.
    SumCubesOdds,
    /// Product of x*x for even-valued elements (none → 1).
    ProductSquaresEvens,
    /// Product of x*x for odd-valued elements (none → 1).
    ProductSquaresOdds,
    /// Max of |x| among even-valued elements (none → 0).
    MaxAbsEvens,
    /// Max of |x| among odd-valued elements (none → 0).
    MaxAbsOdds,
    /// Min of |x| among even-valued elements (none → 0).
    MinAbsEvens,
    /// Min of |x| among odd-valued elements (none → 0).
    MinAbsOdds,
    /// Count of nonzero even-valued elements.
    CountNonZeroEvens,
    /// Count of nonzero odd-valued elements.
    CountNonZeroOdds,
    /// Sum of nonzero even-valued elements.
    SumNonZeroEvens,
    /// Sum of nonzero odd-valued elements.
    SumNonZeroOdds,
    /// Product of nonzero even-valued elements (none → 1).
    ProductNonZeroEvens,
    /// Product of nonzero odd-valued elements (none → 1).
    ProductNonZeroOdds,
    /// Truncating mean of |x| for even-valued elements (none → 0).
    MeanAbsEvensTrunc,
    /// Truncating mean of |x| for odd-valued elements (none → 0).
    MeanAbsOddsTrunc,
    /// GCD of |x| for even-valued elements (none → 0).
    GcdAbsEvens,
    /// GCD of |x| for odd-valued elements (none → 0).
    GcdAbsOdds,
    /// LCM of |x| for even-valued elements (none → 0).
    LcmAbsEvens,
    /// LCM of |x| for odd-valued elements (none → 0).
    LcmAbsOdds,
    /// Product of cubes of even-valued elements (empty → 1).
    ProductCubesEvens,
    /// Product of cubes of odd-valued elements (empty → 1).
    ProductCubesOdds,
    /// Sum of |x|^3 for even-valued elements (empty → 0).
    SumAbsCubesEvens,
    /// Sum of |x|^3 for odd-valued elements (empty → 0).
    SumAbsCubesOdds,
    /// Product of |x|^3 for even-valued elements (empty → 1).
    ProductAbsCubesEvens,
    /// Product of |x|^3 for odd-valued elements (empty → 1).
    ProductAbsCubesOdds,
    /// Sum of |x|^2 for even-valued elements (empty → 0).
    SumAbsSquaresEvens,
    /// Sum of |x|^2 for odd-valued elements (empty → 0).
    SumAbsSquaresOdds,
    /// Product of |x|^2 for even-valued elements (empty → 1).
    ProductAbsSquaresEvens,
    /// Product of |x|^2 for odd-valued elements (empty → 1).
    ProductAbsSquaresOdds,
    /// Truncating mean of |x|^2 for even-valued elements (none → 0).
    MeanAbsSquaresEvensTrunc,
    /// Truncating mean of |x|^2 for odd-valued elements (none → 0).
    MeanAbsSquaresOddsTrunc,
    /// Count of positive even-valued elements.
    CountPositiveEvens,
    /// Count of positive odd-valued elements.
    CountPositiveOdds,
    /// Count of negative even-valued elements.
    CountNegativeEvens,
    /// Count of negative odd-valued elements.
    CountNegativeOdds,
    /// Sum of positive even-valued elements.
    SumPositiveEvens,
    /// Sum of positive odd-valued elements.
    SumPositiveOdds,
    /// Sum of negative even-valued elements.
    SumNegativeEvens,
    /// Sum of negative odd-valued elements.
    SumNegativeOdds,
    /// Product of positive even-valued elements (empty → 1).
    ProductPositiveEvens,
    /// Product of positive odd-valued elements (empty → 1).
    ProductPositiveOdds,
    /// Product of negative even-valued elements (empty → 1).
    ProductNegativeEvens,
    /// Product of negative odd-valued elements (empty → 1).
    ProductNegativeOdds,
    /// Max among positive even-valued elements (none → 0).
    MaxPositiveEvens,
    /// Max among positive odd-valued elements (none → 0).
    MaxPositiveOdds,
    /// Min among positive even-valued elements (none → 0).
    MinPositiveEvens,
    /// Min among positive odd-valued elements (none → 0).
    MinPositiveOdds,
    /// Max among negative even-valued elements (none → 0).
    MaxNegativeEvens,
    /// Max among negative odd-valued elements (none → 0).
    MaxNegativeOdds,
    /// Min among negative even-valued elements (none → 0).
    MinNegativeEvens,
    /// Min among negative odd-valued elements (none → 0).
    MinNegativeOdds,
    /// Truncating mean of positive even-valued elements (none → 0).
    MeanPositiveEvensTrunc,
    /// Truncating mean of positive odd-valued elements (none → 0).
    MeanPositiveOddsTrunc,
    /// Truncating mean of negative even-valued elements (none → 0).
    MeanNegativeEvensTrunc,
    /// Truncating mean of negative odd-valued elements (none → 0).
    MeanNegativeOddsTrunc,
    /// 1 if every even-valued element is positive (none → 1).
    AllEvenPositive,
    /// 1 if every odd-valued element is positive (none → 1).
    AllOddPositive,
    /// 1 if every even-valued element is negative (none → 1).
    AllEvenNegative,
    /// 1 if every odd-valued element is negative (none → 1).
    AllOddNegative,
    /// 1 if any even-valued element is positive (none → 0).
    AnyEvenPositive,
    /// 1 if any odd-valued element is positive (none → 0).
    AnyOddPositive,
    /// 1 if any even-valued element is negative (none → 0).
    AnyEvenNegative,
    /// 1 if any odd-valued element is negative (none → 0).
    AnyOddNegative,
    /// 1 if any even-valued element is non-zero (none → 0).
    AnyEvenNonZero,
    /// 1 if any odd-valued element is non-zero (none → 0).
    AnyOddNonZero,
    /// 1 if every even-valued element is non-zero (none → 1).
    AllEvenNonZero,
    /// 1 if every odd-valued element is non-zero (none → 1).
    AllOddNonZero,
    /// 1 if every even-valued element is non-negative (none → 1).
    AllEvenNonNegative,
    /// 1 if every odd-valued element is non-negative (none → 1).
    AllOddNonNegative,
    /// 1 if every even-valued element is non-positive (none → 1).
    AllEvenNonPositive,
    /// 1 if every odd-valued element is non-positive (none → 1).
    AllOddNonPositive,
    /// 1 if any even-valued element is non-negative (none → 0).
    AnyEvenNonNegative,
    /// 1 if any odd-valued element is non-negative (none → 0).
    AnyOddNonNegative,
    /// 1 if any even-valued element is non-positive (none → 0).
    AnyEvenNonPositive,
    /// 1 if any odd-valued element is non-positive (none → 0).
    AnyOddNonPositive,
    /// Max among even-valued non-zero elements (none → 0).
    MaxEvenNonZero,
    /// Max among odd-valued non-zero elements (none → 0).
    MaxOddNonZero,
    /// Min among even-valued non-zero elements (none → 0).
    MinEvenNonZero,
    /// Min among odd-valued non-zero elements (none → 0).
    MinOddNonZero,
    /// Truncating mean of even-valued non-zero elements (none → 0).
    MeanEvenNonZeroTrunc,
    /// Truncating mean of odd-valued non-zero elements (none → 0).
    MeanOddNonZeroTrunc,
    /// XOR of even-valued non-zero elements (none → 0).
    XorEvenNonZero,
    /// XOR of odd-valued non-zero elements (none → 0).
    XorOddNonZero,
    /// Bitwise OR of even-valued non-zero elements (none → 0).
    OrEvenNonZero,
    /// Bitwise OR of odd-valued non-zero elements (none → 0).
    OrOddNonZero,
    /// Bitwise AND of even-valued non-zero elements (none → -1).
    AndEvenNonZero,
    /// Bitwise AND of odd-valued non-zero elements (none → -1).
    AndOddNonZero,
    /// Sum of abs of even-valued non-zero elements.
    SumAbsEvenNonZero,
    /// Sum of abs of odd-valued non-zero elements.
    SumAbsOddNonZero,
    /// Product of abs of even-valued non-zero elements (none → 1).
    ProductAbsEvenNonZero,
    /// Product of abs of odd-valued non-zero elements (none → 1).
    ProductAbsOddNonZero,
    /// GCD of abs of even-valued non-zero elements (none → 0).
    GcdAbsEvenNonZero,
    /// GCD of abs of odd-valued non-zero elements (none → 0).
    GcdAbsOddNonZero,
    /// LCM of abs of even-valued non-zero elements (none → 1).
    LcmAbsEvenNonZero,
    /// LCM of abs of odd-valued non-zero elements (none → 1).
    LcmAbsOddNonZero,
    /// Truncating mean of abs of even-valued non-zero elements (none → 0).
    MeanAbsEvenNonZeroTrunc,
    /// Truncating mean of abs of odd-valued non-zero elements (none → 0).
    MeanAbsOddNonZeroTrunc,
    /// Max abs among even-valued non-zero elements (none → 0).
    MaxAbsEvenNonZero,
    /// Max abs among odd-valued non-zero elements (none → 0).
    MaxAbsOddNonZero,
    /// Min abs among even-valued non-zero elements (none → 0).
    MinAbsEvenNonZero,
    /// Min abs among odd-valued non-zero elements (none → 0).
    MinAbsOddNonZero,
    /// Sum of squares of even-valued non-zero elements.
    SumSquaresEvenNonZero,
    /// Sum of squares of odd-valued non-zero elements.
    SumSquaresOddNonZero,
    /// Product of squares of even-valued non-zero elements (none → 1).
    ProductSquaresEvenNonZero,
    /// Product of squares of odd-valued non-zero elements (none → 1).
    ProductSquaresOddNonZero,
    /// Sum of cubes of even-valued non-zero elements.
    SumCubesEvenNonZero,
    /// Sum of cubes of odd-valued non-zero elements.
    SumCubesOddNonZero,
    /// Product of cubes of even-valued non-zero elements (none → 1).
    ProductCubesEvenNonZero,
    /// Product of cubes of odd-valued non-zero elements (none → 1).
    ProductCubesOddNonZero,
    /// Sum of fourth powers of even-valued non-zero elements.
    SumFourthPowersEvenNonZero,
    /// Sum of fourth powers of odd-valued non-zero elements.
    SumFourthPowersOddNonZero,
    /// Product of fourth powers of even-valued non-zero elements (none → 1).
    ProductFourthPowersEvenNonZero,
    /// Product of fourth powers of odd-valued non-zero elements (none → 1).
    ProductFourthPowersOddNonZero,
    /// Truncating mean of fourth powers of even-valued non-zero elements (none → 0).
    MeanFourthPowersEvenNonZeroTrunc,
    /// Truncating mean of fourth powers of odd-valued non-zero elements (none → 0).
    MeanFourthPowersOddNonZeroTrunc,
    /// Sum of fifth powers of even-valued non-zero elements.
    SumFifthPowersEvenNonZero,
    /// Sum of fifth powers of odd-valued non-zero elements.
    SumFifthPowersOddNonZero,
    /// Product of fifth powers of even-valued non-zero elements (none → 1).
    ProductFifthPowersEvenNonZero,
    /// Product of fifth powers of odd-valued non-zero elements (none → 1).
    ProductFifthPowersOddNonZero,
    /// Truncating mean of fifth powers of even-valued non-zero elements (none → 0).
    MeanFifthPowersEvenNonZeroTrunc,
    /// Truncating mean of fifth powers of odd-valued non-zero elements (none → 0).
    MeanFifthPowersOddNonZeroTrunc,
    /// Sum of sixth powers of even-valued non-zero elements.
    SumSixthPowersEvenNonZero,
    /// Sum of sixth powers of odd-valued non-zero elements.
    SumSixthPowersOddNonZero,
    /// Product of sixth powers of even-valued non-zero elements (none → 1).
    ProductSixthPowersEvenNonZero,
    /// Product of sixth powers of odd-valued non-zero elements (none → 1).
    ProductSixthPowersOddNonZero,
    /// Truncating mean of sixth powers of even-valued non-zero elements (none → 0).
    MeanSixthPowersEvenNonZeroTrunc,
    /// Truncating mean of sixth powers of odd-valued non-zero elements (none → 0).
    MeanSixthPowersOddNonZeroTrunc,
    /// Sum of seventh powers of even-valued non-zero elements.
    SumSeventhPowersEvenNonZero,
    /// Sum of seventh powers of odd-valued non-zero elements.
    SumSeventhPowersOddNonZero,
}

impl DualAccum {
    fn label(self) -> &'static str {
        match self {
            DualAccum::Range => "range",
            DualAccum::SecondMax => "second_max",
            DualAccum::SecondMin => "second_min",
            DualAccum::StockProfit => "stock_profit",
            DualAccum::PrefixMaxSum => "prefix_max_sum",
            DualAccum::PrefixMinSum => "prefix_min_sum",
            DualAccum::MaxSubarraySum => "max_subarray_sum",
            DualAccum::MinSubarraySum => "min_subarray_sum",
            DualAccum::Median => "median",
            DualAccum::GcdAll => "gcd_all",
            DualAccum::LcmAll => "lcm_all",
            DualAccum::MeanTrunc => "mean_trunc",
            DualAccum::SumSquares => "sum_squares",
            DualAccum::AbsSum => "abs_sum",
            DualAccum::MaxAbs => "max_abs",
            DualAccum::MinAbs => "min_abs",
            DualAccum::MinPositive => "min_positive",
            DualAccum::CountNegatives => "count_negatives",
            DualAccum::CountEvens => "count_evens",
            DualAccum::SumPositives => "sum_positives",
            DualAccum::SumNegatives => "sum_negatives",
            DualAccum::CountOdds => "count_odds",
            DualAccum::Len => "len",
            DualAccum::IsEmpty => "is_empty",
            DualAccum::AllEqual => "all_equal",
            DualAccum::AnyPositive => "any_positive",
            DualAccum::AnyNegative => "any_negative",
            DualAccum::AnyZero => "any_zero",
            DualAccum::AllPositive => "all_positive",
            DualAccum::AllNegative => "all_negative",
            DualAccum::CountZeros => "count_zeros",
            DualAccum::HasDuplicate => "has_duplicate",
            DualAccum::MaxNegative => "max_negative",
            DualAccum::SumEvenValues => "sum_even_values",
            DualAccum::SumOddValues => "sum_odd_values",
            DualAccum::AllNonNegative => "all_non_negative",
            DualAccum::CountNonZeros => "count_non_zeros",
            DualAccum::AlternatingSum => "alternating_sum",
            DualAccum::ProductPositives => "product_positives",
            DualAccum::MeanAbsTrunc => "mean_abs_trunc",
            DualAccum::FirstPositive => "first_positive",
            DualAccum::LastPositive => "last_positive",
            DualAccum::FirstNegative => "first_negative",
            DualAccum::LastNegative => "last_negative",
            DualAccum::MaxPositive => "max_positive",
            DualAccum::MinNegative => "min_negative",
            DualAccum::ProductNegatives => "product_negatives",
            DualAccum::SumCubes => "sum_cubes",
            DualAccum::CountGtMean => "count_gt_mean",
            DualAccum::CountLtMean => "count_lt_mean",
            DualAccum::IsPalindrome => "is_palindrome",
            DualAccum::ProductEvens => "product_evens",
            DualAccum::ProductOdds => "product_odds",
            DualAccum::AllNonPositive => "all_non_positive",
            DualAccum::DotIndex => "dot_index",
            DualAccum::SumSqDiffMean => "sum_sq_diff_mean",
            DualAccum::AnyNonZero => "any_non_zero",
            DualAccum::XorAll => "xor_all",
            DualAccum::ProductNonZeros => "product_non_zeros",
            DualAccum::OrAll => "or_all",
            DualAccum::AndAll => "and_all",
            DualAccum::CountEqMean => "count_eq_mean",
            DualAccum::ProductAbs => "product_abs",
            DualAccum::CountNonNegatives => "count_non_negatives",
            DualAccum::CountPositives => "count_positives",
            DualAccum::MaxEvenValue => "max_even_value",
            DualAccum::MaxOddValue => "max_odd_value",
            DualAccum::MinEvenValue => "min_even_value",
            DualAccum::MinOddValue => "min_odd_value",
            DualAccum::AbsRange => "abs_range",
            DualAccum::ProductNonNegatives => "product_non_negatives",
            DualAccum::SumNonNegatives => "sum_non_negatives",
            DualAccum::SumNonPositives => "sum_non_positives",
            DualAccum::CountNonPositives => "count_non_positives",
            DualAccum::ProductNonPositives => "product_non_positives",
            DualAccum::SumAbsEvens => "sum_abs_evens",
            DualAccum::SumAbsOdds => "sum_abs_odds",
            DualAccum::ProductAbsEvens => "product_abs_evens",
            DualAccum::ProductAbsOdds => "product_abs_odds",
            DualAccum::XorAbsAll => "xor_abs_all",
            DualAccum::AndAbsAll => "and_abs_all",
            DualAccum::OrAbsAll => "or_abs_all",
            DualAccum::XorAbsEvens => "xor_abs_evens",
            DualAccum::XorAbsOdds => "xor_abs_odds",
            DualAccum::AndAbsEvens => "and_abs_evens",
            DualAccum::AndAbsOdds => "and_abs_odds",
            DualAccum::OrAbsEvens => "or_abs_evens",
            DualAccum::OrAbsOdds => "or_abs_odds",
            DualAccum::SumSquaresEvens => "sum_squares_evens",
            DualAccum::SumSquaresOdds => "sum_squares_odds",
            DualAccum::SumCubesEvens => "sum_cubes_evens",
            DualAccum::SumCubesOdds => "sum_cubes_odds",
            DualAccum::ProductSquaresEvens => "product_squares_evens",
            DualAccum::ProductSquaresOdds => "product_squares_odds",
            DualAccum::MaxAbsEvens => "max_abs_evens",
            DualAccum::MaxAbsOdds => "max_abs_odds",
            DualAccum::MinAbsEvens => "min_abs_evens",
            DualAccum::MinAbsOdds => "min_abs_odds",
            DualAccum::CountNonZeroEvens => "count_nonzero_evens",
            DualAccum::CountNonZeroOdds => "count_nonzero_odds",
            DualAccum::SumNonZeroEvens => "sum_nonzero_evens",
            DualAccum::SumNonZeroOdds => "sum_nonzero_odds",
            DualAccum::ProductNonZeroEvens => "product_nonzero_evens",
            DualAccum::ProductNonZeroOdds => "product_nonzero_odds",
            DualAccum::MeanAbsEvensTrunc => "mean_abs_evens_trunc",
            DualAccum::MeanAbsOddsTrunc => "mean_abs_odds_trunc",
            DualAccum::GcdAbsEvens => "gcd_abs_evens",
            DualAccum::GcdAbsOdds => "gcd_abs_odds",
            DualAccum::LcmAbsEvens => "lcm_abs_evens",
            DualAccum::LcmAbsOdds => "lcm_abs_odds",
            DualAccum::ProductCubesEvens => "product_cubes_evens",
            DualAccum::ProductCubesOdds => "product_cubes_odds",
            DualAccum::SumAbsCubesEvens => "sum_abs_cubes_evens",
            DualAccum::SumAbsCubesOdds => "sum_abs_cubes_odds",
            DualAccum::ProductAbsCubesEvens => "product_abs_cubes_evens",
            DualAccum::ProductAbsCubesOdds => "product_abs_cubes_odds",
            DualAccum::SumAbsSquaresEvens => "sum_abs_squares_evens",
            DualAccum::SumAbsSquaresOdds => "sum_abs_squares_odds",
            DualAccum::ProductAbsSquaresEvens => "product_abs_squares_evens",
            DualAccum::ProductAbsSquaresOdds => "product_abs_squares_odds",
            DualAccum::MeanAbsSquaresEvensTrunc => "mean_abs_squares_evens_trunc",
            DualAccum::MeanAbsSquaresOddsTrunc => "mean_abs_squares_odds_trunc",
            DualAccum::CountPositiveEvens => "count_positive_evens",
            DualAccum::CountPositiveOdds => "count_positive_odds",
            DualAccum::CountNegativeEvens => "count_negative_evens",
            DualAccum::CountNegativeOdds => "count_negative_odds",
            DualAccum::SumPositiveEvens => "sum_positive_evens",
            DualAccum::SumPositiveOdds => "sum_positive_odds",
            DualAccum::SumNegativeEvens => "sum_negative_evens",
            DualAccum::SumNegativeOdds => "sum_negative_odds",
            DualAccum::ProductPositiveEvens => "product_positive_evens",
            DualAccum::ProductPositiveOdds => "product_positive_odds",
            DualAccum::ProductNegativeEvens => "product_negative_evens",
            DualAccum::ProductNegativeOdds => "product_negative_odds",
            DualAccum::MaxPositiveEvens => "max_positive_evens",
            DualAccum::MaxPositiveOdds => "max_positive_odds",
            DualAccum::MinPositiveEvens => "min_positive_evens",
            DualAccum::MinPositiveOdds => "min_positive_odds",
            DualAccum::MaxNegativeEvens => "max_negative_evens",
            DualAccum::MaxNegativeOdds => "max_negative_odds",
            DualAccum::MinNegativeEvens => "min_negative_evens",
            DualAccum::MinNegativeOdds => "min_negative_odds",
            DualAccum::MeanPositiveEvensTrunc => "mean_positive_evens_trunc",
            DualAccum::MeanPositiveOddsTrunc => "mean_positive_odds_trunc",
            DualAccum::MeanNegativeEvensTrunc => "mean_negative_evens_trunc",
            DualAccum::MeanNegativeOddsTrunc => "mean_negative_odds_trunc",
            DualAccum::AllEvenPositive => "all_even_positive",
            DualAccum::AllOddPositive => "all_odd_positive",
            DualAccum::AllEvenNegative => "all_even_negative",
            DualAccum::AllOddNegative => "all_odd_negative",
            DualAccum::AnyEvenPositive => "any_even_positive",
            DualAccum::AnyOddPositive => "any_odd_positive",
            DualAccum::AnyEvenNegative => "any_even_negative",
            DualAccum::AnyOddNegative => "any_odd_negative",
            DualAccum::AnyEvenNonZero => "any_even_non_zero",
            DualAccum::AnyOddNonZero => "any_odd_non_zero",
            DualAccum::AllEvenNonZero => "all_even_non_zero",
            DualAccum::AllOddNonZero => "all_odd_non_zero",
            DualAccum::AllEvenNonNegative => "all_even_non_negative",
            DualAccum::AllOddNonNegative => "all_odd_non_negative",
            DualAccum::AllEvenNonPositive => "all_even_non_positive",
            DualAccum::AllOddNonPositive => "all_odd_non_positive",
            DualAccum::AnyEvenNonNegative => "any_even_non_negative",
            DualAccum::AnyOddNonNegative => "any_odd_non_negative",
            DualAccum::AnyEvenNonPositive => "any_even_non_positive",
            DualAccum::AnyOddNonPositive => "any_odd_non_positive",
            DualAccum::MaxEvenNonZero => "max_even_non_zero",
            DualAccum::MaxOddNonZero => "max_odd_non_zero",
            DualAccum::MinEvenNonZero => "min_even_non_zero",
            DualAccum::MinOddNonZero => "min_odd_non_zero",
            DualAccum::MeanEvenNonZeroTrunc => "mean_even_non_zero_trunc",
            DualAccum::MeanOddNonZeroTrunc => "mean_odd_non_zero_trunc",
            DualAccum::XorEvenNonZero => "xor_even_non_zero",
            DualAccum::XorOddNonZero => "xor_odd_non_zero",
            DualAccum::OrEvenNonZero => "or_even_non_zero",
            DualAccum::OrOddNonZero => "or_odd_non_zero",
            DualAccum::AndEvenNonZero => "and_even_non_zero",
            DualAccum::AndOddNonZero => "and_odd_non_zero",
            DualAccum::SumAbsEvenNonZero => "sum_abs_even_non_zero",
            DualAccum::SumAbsOddNonZero => "sum_abs_odd_non_zero",
            DualAccum::ProductAbsEvenNonZero => "product_abs_even_non_zero",
            DualAccum::ProductAbsOddNonZero => "product_abs_odd_non_zero",
            DualAccum::GcdAbsEvenNonZero => "gcd_abs_even_non_zero",
            DualAccum::GcdAbsOddNonZero => "gcd_abs_odd_non_zero",
            DualAccum::LcmAbsEvenNonZero => "lcm_abs_even_non_zero",
            DualAccum::LcmAbsOddNonZero => "lcm_abs_odd_non_zero",
            DualAccum::MeanAbsEvenNonZeroTrunc => "mean_abs_even_non_zero_trunc",
            DualAccum::MeanAbsOddNonZeroTrunc => "mean_abs_odd_non_zero_trunc",
            DualAccum::MaxAbsEvenNonZero => "max_abs_even_non_zero",
            DualAccum::MaxAbsOddNonZero => "max_abs_odd_non_zero",
            DualAccum::MinAbsEvenNonZero => "min_abs_even_non_zero",
            DualAccum::MinAbsOddNonZero => "min_abs_odd_non_zero",
            DualAccum::SumSquaresEvenNonZero => "sum_squares_even_non_zero",
            DualAccum::SumSquaresOddNonZero => "sum_squares_odd_non_zero",
            DualAccum::ProductSquaresEvenNonZero => "product_squares_even_non_zero",
            DualAccum::ProductSquaresOddNonZero => "product_squares_odd_non_zero",
            DualAccum::SumCubesEvenNonZero => "sum_cubes_even_non_zero",
            DualAccum::SumCubesOddNonZero => "sum_cubes_odd_non_zero",
            DualAccum::ProductCubesEvenNonZero => "product_cubes_even_non_zero",
            DualAccum::ProductCubesOddNonZero => "product_cubes_odd_non_zero",
            DualAccum::SumFourthPowersEvenNonZero => "sum_fourth_powers_even_non_zero",
            DualAccum::SumFourthPowersOddNonZero => "sum_fourth_powers_odd_non_zero",
            DualAccum::ProductFourthPowersEvenNonZero => "product_fourth_powers_even_non_zero",
            DualAccum::ProductFourthPowersOddNonZero => "product_fourth_powers_odd_non_zero",
            DualAccum::MeanFourthPowersEvenNonZeroTrunc => "mean_fourth_powers_even_non_zero_trunc",
            DualAccum::MeanFourthPowersOddNonZeroTrunc => "mean_fourth_powers_odd_non_zero_trunc",
            DualAccum::SumFifthPowersEvenNonZero => "sum_fifth_powers_even_non_zero",
            DualAccum::SumFifthPowersOddNonZero => "sum_fifth_powers_odd_non_zero",
            DualAccum::ProductFifthPowersEvenNonZero => "product_fifth_powers_even_non_zero",
            DualAccum::ProductFifthPowersOddNonZero => "product_fifth_powers_odd_non_zero",
            DualAccum::MeanFifthPowersEvenNonZeroTrunc => "mean_fifth_powers_even_non_zero_trunc",
            DualAccum::MeanFifthPowersOddNonZeroTrunc => "mean_fifth_powers_odd_non_zero_trunc",
            DualAccum::SumSixthPowersEvenNonZero => "sum_sixth_powers_even_non_zero",
            DualAccum::SumSixthPowersOddNonZero => "sum_sixth_powers_odd_non_zero",
            DualAccum::ProductSixthPowersEvenNonZero => "product_sixth_powers_even_non_zero",
            DualAccum::ProductSixthPowersOddNonZero => "product_sixth_powers_odd_non_zero",
            DualAccum::MeanSixthPowersEvenNonZeroTrunc => "mean_sixth_powers_even_non_zero_trunc",
            DualAccum::MeanSixthPowersOddNonZeroTrunc => "mean_sixth_powers_odd_non_zero_trunc",
            DualAccum::SumSeventhPowersEvenNonZero => "sum_seventh_powers_even_non_zero",
            DualAccum::SumSeventhPowersOddNonZero => "sum_seventh_powers_odd_non_zero",
        }
    }

    fn eval(self, arr: &[i64]) -> Option<i64> {
        // Ops well-defined on [] — do not early-reject (IsEmpty/Len/Any*/All*/counts).
        if arr.is_empty() {
            return match self {
                DualAccum::Len => Some(0),
                DualAccum::IsEmpty => Some(1),
                DualAccum::AllEqual
                | DualAccum::AllPositive
                | DualAccum::AllNegative
                | DualAccum::AllNonNegative
                | DualAccum::AllNonPositive => Some(1),
                DualAccum::AnyPositive
                | DualAccum::AnyNegative
                | DualAccum::AnyZero
                | DualAccum::HasDuplicate
                | DualAccum::AnyNonZero
                | DualAccum::AnyEvenPositive
                | DualAccum::AnyOddPositive
                | DualAccum::AnyEvenNegative
                | DualAccum::AnyOddNegative
                | DualAccum::AnyEvenNonZero
                | DualAccum::AnyOddNonZero
                | DualAccum::AnyEvenNonNegative
                | DualAccum::AnyOddNonNegative
                | DualAccum::AnyEvenNonPositive
                | DualAccum::AnyOddNonPositive => Some(0),
                DualAccum::CountNegatives
                | DualAccum::CountEvens
                | DualAccum::CountOdds
                | DualAccum::CountZeros
                | DualAccum::CountNonZeros
                | DualAccum::CountNonNegatives
                | DualAccum::CountPositives
                | DualAccum::CountNonPositives
                | DualAccum::CountNonZeroEvens
                | DualAccum::CountNonZeroOdds
                | DualAccum::SumNonZeroEvens
                | DualAccum::SumNonZeroOdds
                | DualAccum::MaxEvenValue
                | DualAccum::MaxOddValue
                | DualAccum::MinEvenValue
                | DualAccum::MinOddValue
                | DualAccum::MaxAbsEvens
                | DualAccum::MaxAbsOdds
                | DualAccum::MinAbsEvens
                | DualAccum::MinAbsOdds
                | DualAccum::MeanAbsEvensTrunc
                | DualAccum::MeanAbsOddsTrunc
                | DualAccum::MeanAbsSquaresEvensTrunc
                | DualAccum::MeanAbsSquaresOddsTrunc
                | DualAccum::CountPositiveEvens
                | DualAccum::CountPositiveOdds
                | DualAccum::CountNegativeEvens
                | DualAccum::CountNegativeOdds
                | DualAccum::MaxEvenNonZero
                | DualAccum::MaxOddNonZero
                | DualAccum::MinEvenNonZero
                | DualAccum::MinOddNonZero
                | DualAccum::MeanEvenNonZeroTrunc
                | DualAccum::MeanOddNonZeroTrunc
                | DualAccum::XorEvenNonZero
                | DualAccum::XorOddNonZero
                | DualAccum::OrEvenNonZero
                | DualAccum::OrOddNonZero
                | DualAccum::SumAbsEvenNonZero
                | DualAccum::SumAbsOddNonZero
                | DualAccum::GcdAbsEvenNonZero
                | DualAccum::GcdAbsOddNonZero
                | DualAccum::MeanAbsEvenNonZeroTrunc
                | DualAccum::MeanAbsOddNonZeroTrunc
                | DualAccum::MaxAbsEvenNonZero
                | DualAccum::MaxAbsOddNonZero
                | DualAccum::MinAbsEvenNonZero
                | DualAccum::MinAbsOddNonZero
                | DualAccum::SumSquaresEvenNonZero
                | DualAccum::SumSquaresOddNonZero
                | DualAccum::SumCubesEvenNonZero
                | DualAccum::SumCubesOddNonZero
                | DualAccum::SumFourthPowersEvenNonZero
                | DualAccum::SumFourthPowersOddNonZero
                | DualAccum::MeanFourthPowersEvenNonZeroTrunc
                | DualAccum::MeanFourthPowersOddNonZeroTrunc
                | DualAccum::SumFifthPowersEvenNonZero
                | DualAccum::SumFifthPowersOddNonZero
                | DualAccum::MeanFifthPowersEvenNonZeroTrunc
                | DualAccum::MeanFifthPowersOddNonZeroTrunc
                | DualAccum::SumSixthPowersEvenNonZero
                | DualAccum::SumSixthPowersOddNonZero
                | DualAccum::MeanSixthPowersEvenNonZeroTrunc
                | DualAccum::MeanSixthPowersOddNonZeroTrunc
                | DualAccum::SumSeventhPowersEvenNonZero
                | DualAccum::SumSeventhPowersOddNonZero
                | DualAccum::SumPositiveEvens
                | DualAccum::SumPositiveOdds
                | DualAccum::SumNegativeEvens
                | DualAccum::SumNegativeOdds
                | DualAccum::MaxPositiveEvens
                | DualAccum::MaxPositiveOdds
                | DualAccum::MinPositiveEvens
                | DualAccum::MinPositiveOdds
                | DualAccum::MaxNegativeEvens
                | DualAccum::MaxNegativeOdds
                | DualAccum::MinNegativeEvens
                | DualAccum::MinNegativeOdds
                | DualAccum::MeanPositiveEvensTrunc
                | DualAccum::MeanPositiveOddsTrunc
                | DualAccum::MeanNegativeEvensTrunc
                | DualAccum::MeanNegativeOddsTrunc
                | DualAccum::GcdAbsEvens
                | DualAccum::GcdAbsOdds
                | DualAccum::LcmAbsEvens
                | DualAccum::LcmAbsOdds
                | DualAccum::AbsRange
                | DualAccum::SumPositives
                | DualAccum::SumNegatives
                | DualAccum::SumNonNegatives
                | DualAccum::SumNonPositives
                | DualAccum::SumAbsEvens
                | DualAccum::SumAbsOdds
                | DualAccum::XorAbsAll
                | DualAccum::OrAbsAll
                | DualAccum::XorAbsEvens
                | DualAccum::XorAbsOdds
                | DualAccum::OrAbsEvens
                | DualAccum::OrAbsOdds
                | DualAccum::SumSquaresEvens
                | DualAccum::SumSquaresOdds
                | DualAccum::SumCubesEvens
                | DualAccum::SumCubesOdds
                | DualAccum::SumAbsCubesEvens
                | DualAccum::SumAbsCubesOdds
                | DualAccum::SumAbsSquaresEvens
                | DualAccum::SumAbsSquaresOdds
                | DualAccum::SumSquares
                | DualAccum::AbsSum
                | DualAccum::MaxAbs
                | DualAccum::MinAbs
                | DualAccum::MinPositive
                | DualAccum::MaxNegative
                | DualAccum::MaxPositive
                | DualAccum::MinNegative
                | DualAccum::SumEvenValues
                | DualAccum::SumOddValues
                | DualAccum::AlternatingSum
                | DualAccum::FirstPositive
                | DualAccum::LastPositive
                | DualAccum::FirstNegative
                | DualAccum::LastNegative => Some(0),
                DualAccum::ProductPositives
                | DualAccum::ProductNegatives
                | DualAccum::ProductEvens
                | DualAccum::ProductOdds
                | DualAccum::ProductNonZeros
                | DualAccum::ProductAbs
                | DualAccum::ProductNonNegatives
                | DualAccum::ProductNonPositives
                | DualAccum::ProductAbsEvens
                | DualAccum::ProductAbsOdds
                | DualAccum::ProductSquaresEvens
                | DualAccum::ProductSquaresOdds
                | DualAccum::ProductCubesEvens
                | DualAccum::ProductCubesOdds
                | DualAccum::ProductAbsCubesEvens
                | DualAccum::ProductAbsCubesOdds
                | DualAccum::ProductAbsSquaresEvens
                | DualAccum::ProductAbsSquaresOdds
                | DualAccum::ProductPositiveEvens
                | DualAccum::ProductPositiveOdds
                | DualAccum::ProductNegativeEvens
                | DualAccum::ProductNegativeOdds
                | DualAccum::ProductNonZeroEvens
                | DualAccum::ProductNonZeroOdds
                | DualAccum::ProductAbsEvenNonZero
                | DualAccum::ProductAbsOddNonZero
                | DualAccum::LcmAbsEvenNonZero
                | DualAccum::LcmAbsOddNonZero
                | DualAccum::ProductSquaresEvenNonZero
                | DualAccum::ProductSquaresOddNonZero
                | DualAccum::ProductCubesEvenNonZero
                | DualAccum::ProductCubesOddNonZero
                | DualAccum::ProductFourthPowersEvenNonZero
                | DualAccum::ProductFourthPowersOddNonZero
                | DualAccum::ProductFifthPowersEvenNonZero
                | DualAccum::ProductFifthPowersOddNonZero
                | DualAccum::ProductSixthPowersEvenNonZero
                | DualAccum::ProductSixthPowersOddNonZero => Some(1),
                DualAccum::SumCubes
                | DualAccum::DotIndex
                | DualAccum::XorAll
                | DualAccum::OrAll => Some(0),
                DualAccum::AllEvenPositive
                | DualAccum::AllOddPositive
                | DualAccum::AllEvenNegative
                | DualAccum::AllOddNegative
                | DualAccum::AllEvenNonZero
                | DualAccum::AllOddNonZero
                | DualAccum::AllEvenNonNegative
                | DualAccum::AllOddNonNegative
                | DualAccum::AllEvenNonPositive
                | DualAccum::AllOddNonPositive => Some(1),
                DualAccum::AndAll
                | DualAccum::AndAbsAll
                | DualAccum::AndAbsEvens
                | DualAccum::AndAbsOdds
                | DualAccum::AndEvenNonZero
                | DualAccum::AndOddNonZero => Some(-1),
                DualAccum::IsPalindrome => Some(1),
                DualAccum::MeanAbsTrunc
                | DualAccum::MeanTrunc
                | DualAccum::CountGtMean
                | DualAccum::CountLtMean
                | DualAccum::CountEqMean
                | DualAccum::SumSqDiffMean => None,
                _ => None,
            };
        }
        match self {
            DualAccum::Range => {
                let mut lo = arr[0];
                let mut hi = arr[0];
                for &item in arr {
                    if item < lo {
                        lo = item;
                    }
                    if item > hi {
                        hi = item;
                    }
                }
                Some(hi.saturating_sub(lo))
            }
            DualAccum::SecondMax => {
                // Mirror search_codegen::code_second_max exactly.
                let mut first = arr[0];
                let mut second = arr[0];
                for &item in arr {
                    if item > first {
                        second = first;
                        first = item;
                    } else if item > second {
                        second = item;
                    }
                }
                Some(second)
            }
            DualAccum::SecondMin => {
                let mut first = arr[0];
                let mut second = arr[0];
                for &item in arr {
                    if item < first {
                        second = first;
                        first = item;
                    } else if item < second {
                        second = item;
                    }
                }
                Some(second)
            }
            DualAccum::StockProfit => {
                let mut min_price = arr[0];
                let mut best = 0i64;
                for &p in arr {
                    if p < min_price {
                        min_price = p;
                    }
                    let profit = p.saturating_sub(min_price);
                    if profit > best {
                        best = profit;
                    }
                }
                Some(best)
            }
            DualAccum::PrefixMaxSum => {
                let mut running_max = arr[0];
                let mut total = 0i64;
                for &x in arr {
                    if x > running_max {
                        running_max = x;
                    }
                    total = total.saturating_add(running_max);
                }
                Some(total)
            }
            DualAccum::PrefixMinSum => {
                let mut running_min = arr[0];
                let mut total = 0i64;
                for &x in arr {
                    if x < running_min {
                        running_min = x;
                    }
                    total = total.saturating_add(running_min);
                }
                Some(total)
            }
            DualAccum::MaxSubarraySum => {
                let mut current = 0i64;
                let mut best = arr[0];
                for &item in arr {
                    current = if current > 0 {
                        current.saturating_add(item)
                    } else {
                        item
                    };
                    if current > best {
                        best = current;
                    }
                }
                Some(best)
            }
            DualAccum::MinSubarraySum => {
                let mut current = 0i64;
                let mut best = arr[0];
                for &item in arr {
                    current = if current < 0 {
                        current.saturating_add(item)
                    } else {
                        item
                    };
                    if current < best {
                        best = current;
                    }
                }
                Some(best)
            }
            DualAccum::Median => {
                let mut sorted = arr.to_vec();
                sorted.sort_unstable();
                Some(sorted[sorted.len() / 2])
            }
            DualAccum::GcdAll => {
                let mut g = arr[0].abs();
                for &x in &arr[1..] {
                    g = i64_gcd(g, x);
                    if g == 1 {
                        break;
                    }
                }
                Some(g)
            }
            DualAccum::LcmAll => {
                let mut l = arr[0].abs();
                for &x in &arr[1..] {
                    l = i64_lcm(l, x)?;
                }
                Some(l)
            }
            DualAccum::MeanTrunc => {
                let sum = arr.iter().copied().fold(0i64, i64::saturating_add);
                Some(sum / (arr.len() as i64))
            }
            DualAccum::SumSquares => Some(
                arr.iter()
                    .copied()
                    .map(|x| x.saturating_mul(x))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::AbsSum => Some(
                arr.iter()
                    .copied()
                    .map(|x| x.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::MaxAbs => Some(arr.iter().copied().map(|x| x.abs()).max().unwrap_or(0)),
            DualAccum::MinAbs => Some(arr.iter().copied().map(|x| x.abs()).min().unwrap_or(0)),
            DualAccum::MinPositive => {
                // Mirror search_catalog: 0 when no positive element exists.
                let mut best = 0i64;
                let mut found = false;
                for &x in arr {
                    if x > 0 {
                        if !found || x < best {
                            best = x;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            DualAccum::CountNegatives => {
                Some(arr.iter().filter(|&&x| x < 0).count() as i64)
            }
            DualAccum::CountEvens => {
                Some(arr.iter().filter(|&&x| x % 2 == 0).count() as i64)
            }
            DualAccum::SumPositives => Some(
                arr.iter()
                    .filter(|&&x| x > 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumNegatives => Some(
                arr.iter()
                    .filter(|&&x| x < 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::CountOdds => {
                Some(arr.iter().filter(|&&x| x % 2 != 0).count() as i64)
            }
            DualAccum::Len => Some(arr.len() as i64),
            DualAccum::IsEmpty => Some(if arr.is_empty() { 1 } else { 0 }),
            DualAccum::AllEqual => {
                if arr.is_empty() {
                    return Some(1);
                }
                let first = arr[0];
                Some(if arr.iter().all(|&x| x == first) {
                    1
                } else {
                    0
                })
            }
            DualAccum::AnyPositive => {
                Some(if arr.iter().any(|&x| x > 0) { 1 } else { 0 })
            }
            DualAccum::AnyNegative => {
                Some(if arr.iter().any(|&x| x < 0) { 1 } else { 0 })
            }
            DualAccum::AnyZero => {
                Some(if arr.iter().any(|&x| x == 0) { 1 } else { 0 })
            }
            DualAccum::AllPositive => {
                Some(if arr.iter().all(|&x| x > 0) { 1 } else { 0 })
            }
            DualAccum::AllNegative => {
                Some(if arr.iter().all(|&x| x < 0) { 1 } else { 0 })
            }
            DualAccum::CountZeros => {
                Some(arr.iter().filter(|&&x| x == 0).count() as i64)
            }
            DualAccum::HasDuplicate => {
                let mut found = 0i64;
                for i in 0..arr.len() {
                    for j in 0..i {
                        if arr[j] == arr[i] {
                            found = 1;
                            break;
                        }
                    }
                    if found == 1 {
                        break;
                    }
                }
                Some(found)
            }
            DualAccum::MaxNegative => {
                let mut best = 0i64;
                let mut found = false;
                for &x in arr {
                    if x < 0 {
                        if !found || x > best {
                            best = x;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            DualAccum::SumEvenValues => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumOddValues => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::AllNonNegative => {
                Some(if arr.iter().all(|&x| x >= 0) { 1 } else { 0 })
            }
            DualAccum::CountNonZeros => {
                Some(arr.iter().filter(|&&x| x != 0).count() as i64)
            }
            DualAccum::AlternatingSum => {
                let mut total = 0i64;
                for (i, &x) in arr.iter().enumerate() {
                    if i % 2 == 0 {
                        total = total.saturating_add(x);
                    } else {
                        total = total.saturating_sub(x);
                    }
                }
                Some(total)
            }
            DualAccum::ProductPositives => Some(
                arr.iter()
                    .filter(|&&x| x > 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::MeanAbsTrunc => {
                let sum = arr
                    .iter()
                    .copied()
                    .map(|x| x.abs())
                    .fold(0i64, i64::saturating_add);
                Some(sum / (arr.len() as i64))
            }
            DualAccum::FirstPositive => {
                for &x in arr {
                    if x > 0 {
                        return Some(x);
                    }
                }
                Some(0)
            }
            DualAccum::LastPositive => {
                for &x in arr.iter().rev() {
                    if x > 0 {
                        return Some(x);
                    }
                }
                Some(0)
            }
            DualAccum::FirstNegative => {
                for &x in arr {
                    if x < 0 {
                        return Some(x);
                    }
                }
                Some(0)
            }
            DualAccum::LastNegative => {
                for &x in arr.iter().rev() {
                    if x < 0 {
                        return Some(x);
                    }
                }
                Some(0)
            }
            DualAccum::MaxPositive => {
                let mut best = 0i64;
                let mut found = false;
                for &x in arr {
                    if x > 0 {
                        if !found || x > best {
                            best = x;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            DualAccum::MinNegative => {
                let mut best = 0i64;
                let mut found = false;
                for &x in arr {
                    if x < 0 {
                        if !found || x < best {
                            best = x;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            DualAccum::ProductNegatives => Some(
                arr.iter()
                    .filter(|&&x| x < 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::SumCubes => Some(
                arr.iter()
                    .copied()
                    .map(|x| x.saturating_mul(x).saturating_mul(x))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::CountGtMean => {
                let sum = arr.iter().copied().fold(0i64, i64::saturating_add);
                let mean = sum / (arr.len() as i64);
                Some(arr.iter().filter(|&&x| x > mean).count() as i64)
            }
            DualAccum::CountLtMean => {
                let sum = arr.iter().copied().fold(0i64, i64::saturating_add);
                let mean = sum / (arr.len() as i64);
                Some(arr.iter().filter(|&&x| x < mean).count() as i64)
            }
            DualAccum::IsPalindrome => {
                let n = arr.len();
                Some(if (0..n / 2).all(|i| arr[i] == arr[n - 1 - i]) {
                    1
                } else {
                    0
                })
            }
            DualAccum::ProductEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::AllNonPositive => {
                Some(if arr.iter().all(|&x| x <= 0) { 1 } else { 0 })
            }
            DualAccum::DotIndex => Some(
                arr.iter()
                    .enumerate()
                    .map(|(i, &v)| (i as i64).saturating_mul(v))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumSqDiffMean => {
                let sum = arr.iter().copied().fold(0i64, i64::saturating_add);
                let mean = sum / (arr.len() as i64);
                Some(
                    arr.iter()
                        .map(|&x| {
                            let d = x.saturating_sub(mean);
                            d.saturating_mul(d)
                        })
                        .fold(0i64, i64::saturating_add),
                )
            }
            DualAccum::AnyNonZero => {
                Some(if arr.iter().any(|&x| x != 0) { 1 } else { 0 })
            }
            DualAccum::XorAll => Some(arr.iter().copied().fold(0i64, |a, b| a ^ b)),
            DualAccum::ProductNonZeros => Some(
                arr.iter()
                    .filter(|&&x| x != 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::OrAll => Some(arr.iter().copied().fold(0i64, |a, b| a | b)),
            DualAccum::AndAll => Some(arr.iter().copied().fold(-1i64, |a, b| a & b)),
            DualAccum::CountEqMean => {
                let sum = arr.iter().copied().fold(0i64, i64::saturating_add);
                let mean = sum / (arr.len() as i64);
                Some(arr.iter().filter(|&&x| x == mean).count() as i64)
            }
            DualAccum::ProductAbs => Some(
                arr.iter()
                    .map(|&x| x.abs())
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::CountNonNegatives => {
                Some(arr.iter().filter(|&&x| x >= 0).count() as i64)
            }
            DualAccum::CountPositives => {
                Some(arr.iter().filter(|&&x| x > 0).count() as i64)
            }
            DualAccum::MaxEvenValue => {
                let mut best = 0i64;
                let mut found = false;
                for &x in arr {
                    if x % 2 == 0 {
                        if !found || x > best {
                            best = x;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            DualAccum::MaxOddValue => {
                let mut best = 0i64;
                let mut found = false;
                for &x in arr {
                    if x % 2 != 0 {
                        if !found || x > best {
                            best = x;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            DualAccum::MinEvenValue => {
                let mut best = 0i64;
                let mut found = false;
                for &x in arr {
                    if x % 2 == 0 {
                        if !found || x < best {
                            best = x;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            DualAccum::MinOddValue => {
                let mut best = 0i64;
                let mut found = false;
                for &x in arr {
                    if x % 2 != 0 {
                        if !found || x < best {
                            best = x;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            DualAccum::AbsRange => {
                let mut lo = arr[0].abs();
                let mut hi = lo;
                for &x in arr.iter().skip(1) {
                    let a = x.abs();
                    if a < lo {
                        lo = a;
                    }
                    if a > hi {
                        hi = a;
                    }
                }
                Some(hi.saturating_sub(lo))
            }
            DualAccum::ProductNonNegatives => Some(
                arr.iter()
                    .filter(|&&x| x >= 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::SumNonNegatives => Some(
                arr.iter()
                    .filter(|&&x| x >= 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumNonPositives => Some(
                arr.iter()
                    .filter(|&&x| x <= 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::CountNonPositives => {
                Some(arr.iter().filter(|&&x| x <= 0).count() as i64)
            }
            DualAccum::ProductNonPositives => Some(
                arr.iter()
                    .filter(|&&x| x <= 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::SumAbsEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumAbsOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductAbsEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.abs())
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductAbsOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.abs())
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::XorAbsAll => Some(
                arr.iter().map(|&x| x.abs()).fold(0i64, |a, b| a ^ b),
            ),
            DualAccum::AndAbsAll => Some(
                arr.iter().map(|&x| x.abs()).fold(-1i64, |a, b| a & b),
            ),
            DualAccum::OrAbsAll => Some(
                arr.iter().map(|&x| x.abs()).fold(0i64, |a, b| a | b),
            ),
            DualAccum::XorAbsEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.abs())
                    .fold(0i64, |a, b| a ^ b),
            ),
            DualAccum::XorAbsOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.abs())
                    .fold(0i64, |a, b| a ^ b),
            ),
            DualAccum::AndAbsEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.abs())
                    .fold(-1i64, |a, b| a & b),
            ),
            DualAccum::AndAbsOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.abs())
                    .fold(-1i64, |a, b| a & b),
            ),
            DualAccum::OrAbsEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.abs())
                    .fold(0i64, |a, b| a | b),
            ),
            DualAccum::OrAbsOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.abs())
                    .fold(0i64, |a, b| a | b),
            ),
            DualAccum::SumSquaresEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.saturating_mul(x))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumSquaresOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.saturating_mul(x))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumCubesEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.saturating_mul(x).saturating_mul(x))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumCubesOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.saturating_mul(x).saturating_mul(x))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductSquaresEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.saturating_mul(x))
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductSquaresOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.saturating_mul(x))
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::MaxAbsEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.abs())
                    .max()
                    .unwrap_or(0),
            ),
            DualAccum::MaxAbsOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.abs())
                    .max()
                    .unwrap_or(0),
            ),
            DualAccum::MinAbsEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.abs())
                    .min()
                    .unwrap_or(0),
            ),
            DualAccum::MinAbsOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.abs())
                    .min()
                    .unwrap_or(0),
            ),
            DualAccum::CountNonZeroEvens => Some(
                arr.iter().filter(|&&x| x != 0 && x % 2 == 0).count() as i64,
            ),
            DualAccum::CountNonZeroOdds => Some(
                arr.iter().filter(|&&x| x != 0 && x % 2 != 0).count() as i64,
            ),
            DualAccum::SumNonZeroEvens => Some(
                arr.iter()
                    .filter(|&&x| x != 0 && x % 2 == 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumNonZeroOdds => Some(
                arr.iter()
                    .filter(|&&x| x != 0 && x % 2 != 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductNonZeroEvens => Some(
                arr.iter()
                    .filter(|&&x| x != 0 && x % 2 == 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductNonZeroOdds => Some(
                arr.iter()
                    .filter(|&&x| x != 0 && x % 2 != 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::MeanAbsEvensTrunc => {
                let vals: Vec<i64> = arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.abs()).collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    Some(vals.iter().sum::<i64>() / vals.len() as i64)
                }
            }
            DualAccum::MeanAbsOddsTrunc => {
                let vals: Vec<i64> = arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.abs()).collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    Some(vals.iter().sum::<i64>() / vals.len() as i64)
                }
            }
            DualAccum::GcdAbsEvens => {
                let mut g: Option<i64> = None;
                for &x in arr {
                    if x % 2 == 0 {
                        let a = x.abs();
                        g = Some(match g {
                            None => a,
                            Some(prev) => i64_gcd(prev, a),
                        });
                    }
                }
                Some(g.unwrap_or(0))
            }
            DualAccum::GcdAbsOdds => {
                let mut g: Option<i64> = None;
                for &x in arr {
                    if x % 2 != 0 {
                        let a = x.abs();
                        g = Some(match g {
                            None => a,
                            Some(prev) => i64_gcd(prev, a),
                        });
                    }
                }
                Some(g.unwrap_or(0))
            }
            DualAccum::LcmAbsEvens => {
                let mut l: Option<i64> = None;
                for &x in arr {
                    if x % 2 == 0 {
                        let a = x.abs();
                        l = Some(match l {
                            None => a,
                            Some(prev) => i64_lcm(prev, a).unwrap_or(0),
                        });
                    }
                }
                Some(l.unwrap_or(0))
            }
            DualAccum::LcmAbsOdds => {
                let mut l: Option<i64> = None;
                for &x in arr {
                    if x % 2 != 0 {
                        let a = x.abs();
                        l = Some(match l {
                            None => a,
                            Some(prev) => i64_lcm(prev, a).unwrap_or(0),
                        });
                    }
                }
                Some(l.unwrap_or(0))
            }
            DualAccum::ProductCubesEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| x.saturating_mul(x).saturating_mul(x))
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductCubesOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| x.saturating_mul(x).saturating_mul(x))
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::SumAbsCubesEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| {
                        let a = x.abs();
                        a.saturating_mul(a).saturating_mul(a)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumAbsCubesOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| {
                        let a = x.abs();
                        a.saturating_mul(a).saturating_mul(a)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductAbsCubesEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| {
                        let a = x.abs();
                        a.saturating_mul(a).saturating_mul(a)
                    })
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductAbsCubesOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| {
                        let a = x.abs();
                        a.saturating_mul(a).saturating_mul(a)
                    })
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::SumAbsSquaresEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| {
                        let a = x.abs();
                        a.saturating_mul(a)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumAbsSquaresOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| {
                        let a = x.abs();
                        a.saturating_mul(a)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductAbsSquaresEvens => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| {
                        let a = x.abs();
                        a.saturating_mul(a)
                    })
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductAbsSquaresOdds => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| {
                        let a = x.abs();
                        a.saturating_mul(a)
                    })
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::MeanAbsSquaresEvensTrunc => {
                let vals: Vec<i64> = arr
                    .iter()
                    .filter(|&&x| x % 2 == 0)
                    .map(|&x| {
                        let a = x.abs();
                        a.saturating_mul(a)
                    })
                    .collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    Some(vals.iter().sum::<i64>() / vals.len() as i64)
                }
            }
            DualAccum::MeanAbsSquaresOddsTrunc => {
                let vals: Vec<i64> = arr
                    .iter()
                    .filter(|&&x| x % 2 != 0)
                    .map(|&x| {
                        let a = x.abs();
                        a.saturating_mul(a)
                    })
                    .collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    Some(vals.iter().sum::<i64>() / vals.len() as i64)
                }
            }
            DualAccum::CountPositiveEvens => Some(
                arr.iter().filter(|&&x| x > 0 && x % 2 == 0).count() as i64,
            ),
            DualAccum::CountPositiveOdds => Some(
                arr.iter().filter(|&&x| x > 0 && x % 2 != 0).count() as i64,
            ),
            DualAccum::CountNegativeEvens => Some(
                arr.iter().filter(|&&x| x < 0 && x % 2 == 0).count() as i64,
            ),
            DualAccum::CountNegativeOdds => Some(
                arr.iter().filter(|&&x| x < 0 && x % 2 != 0).count() as i64,
            ),
            DualAccum::SumPositiveEvens => Some(
                arr.iter()
                    .filter(|&&x| x > 0 && x % 2 == 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumPositiveOdds => Some(
                arr.iter()
                    .filter(|&&x| x > 0 && x % 2 != 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumNegativeEvens => Some(
                arr.iter()
                    .filter(|&&x| x < 0 && x % 2 == 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumNegativeOdds => Some(
                arr.iter()
                    .filter(|&&x| x < 0 && x % 2 != 0)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductPositiveEvens => Some(
                arr.iter()
                    .filter(|&&x| x > 0 && x % 2 == 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductPositiveOdds => Some(
                arr.iter()
                    .filter(|&&x| x > 0 && x % 2 != 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductNegativeEvens => Some(
                arr.iter()
                    .filter(|&&x| x < 0 && x % 2 == 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductNegativeOdds => Some(
                arr.iter()
                    .filter(|&&x| x < 0 && x % 2 != 0)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::MaxPositiveEvens => Some(
                arr.iter()
                    .filter(|&&x| x > 0 && x % 2 == 0)
                    .copied()
                    .max()
                    .unwrap_or(0),
            ),
            DualAccum::MaxPositiveOdds => Some(
                arr.iter()
                    .filter(|&&x| x > 0 && x % 2 != 0)
                    .copied()
                    .max()
                    .unwrap_or(0),
            ),
            DualAccum::MinPositiveEvens => Some(
                arr.iter()
                    .filter(|&&x| x > 0 && x % 2 == 0)
                    .copied()
                    .min()
                    .unwrap_or(0),
            ),
            DualAccum::MinPositiveOdds => Some(
                arr.iter()
                    .filter(|&&x| x > 0 && x % 2 != 0)
                    .copied()
                    .min()
                    .unwrap_or(0),
            ),
            DualAccum::MaxNegativeEvens => Some(
                arr.iter()
                    .filter(|&&x| x < 0 && x % 2 == 0)
                    .copied()
                    .max()
                    .unwrap_or(0),
            ),
            DualAccum::MaxNegativeOdds => Some(
                arr.iter()
                    .filter(|&&x| x < 0 && x % 2 != 0)
                    .copied()
                    .max()
                    .unwrap_or(0),
            ),
            DualAccum::MinNegativeEvens => Some(
                arr.iter()
                    .filter(|&&x| x < 0 && x % 2 == 0)
                    .copied()
                    .min()
                    .unwrap_or(0),
            ),
            DualAccum::MinNegativeOdds => Some(
                arr.iter()
                    .filter(|&&x| x < 0 && x % 2 != 0)
                    .copied()
                    .min()
                    .unwrap_or(0),
            ),
            DualAccum::MeanPositiveEvensTrunc => {
                let vals: Vec<i64> = arr
                    .iter()
                    .filter(|&&x| x > 0 && x % 2 == 0)
                    .copied()
                    .collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    Some(vals.iter().sum::<i64>() / vals.len() as i64)
                }
            }
            DualAccum::MeanPositiveOddsTrunc => {
                let vals: Vec<i64> = arr
                    .iter()
                    .filter(|&&x| x > 0 && x % 2 != 0)
                    .copied()
                    .collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    Some(vals.iter().sum::<i64>() / vals.len() as i64)
                }
            }
            DualAccum::MeanNegativeEvensTrunc => {
                let vals: Vec<i64> = arr
                    .iter()
                    .filter(|&&x| x < 0 && x % 2 == 0)
                    .copied()
                    .collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    Some(vals.iter().sum::<i64>() / vals.len() as i64)
                }
            }
            DualAccum::MeanNegativeOddsTrunc => {
                let vals: Vec<i64> = arr
                    .iter()
                    .filter(|&&x| x < 0 && x % 2 != 0)
                    .copied()
                    .collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    Some(vals.iter().sum::<i64>() / vals.len() as i64)
                }
            }
            DualAccum::AllEvenPositive => Some(
                if arr.iter().filter(|&&x| x % 2 == 0).all(|&x| x > 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AllOddPositive => Some(
                if arr.iter().filter(|&&x| x % 2 != 0).all(|&x| x > 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AllEvenNegative => Some(
                if arr.iter().filter(|&&x| x % 2 == 0).all(|&x| x < 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AllOddNegative => Some(
                if arr.iter().filter(|&&x| x % 2 != 0).all(|&x| x < 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AnyEvenPositive => Some(
                if arr.iter().filter(|&&x| x % 2 == 0).any(|&x| x > 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AnyOddPositive => Some(
                if arr.iter().filter(|&&x| x % 2 != 0).any(|&x| x > 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AnyEvenNegative => Some(
                if arr.iter().filter(|&&x| x % 2 == 0).any(|&x| x < 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AnyOddNegative => Some(
                if arr.iter().filter(|&&x| x % 2 != 0).any(|&x| x < 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AnyEvenNonZero => Some(
                if arr.iter().filter(|&&x| x % 2 == 0).any(|&x| x != 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AnyOddNonZero => Some(
                if arr.iter().filter(|&&x| x % 2 != 0).any(|&x| x != 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AllEvenNonZero => Some(
                if arr.iter().filter(|&&x| x % 2 == 0).all(|&x| x != 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AllOddNonZero => Some(
                if arr.iter().filter(|&&x| x % 2 != 0).all(|&x| x != 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AllEvenNonNegative => Some(
                if arr.iter().filter(|&&x| x % 2 == 0).all(|&x| x >= 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AllOddNonNegative => Some(
                if arr.iter().filter(|&&x| x % 2 != 0).all(|&x| x >= 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AllEvenNonPositive => Some(
                if arr.iter().filter(|&&x| x % 2 == 0).all(|&x| x <= 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AllOddNonPositive => Some(
                if arr.iter().filter(|&&x| x % 2 != 0).all(|&x| x <= 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AnyEvenNonNegative => Some(
                if arr.iter().filter(|&&x| x % 2 == 0).any(|&x| x >= 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AnyOddNonNegative => Some(
                if arr.iter().filter(|&&x| x % 2 != 0).any(|&x| x >= 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AnyEvenNonPositive => Some(
                if arr.iter().filter(|&&x| x % 2 == 0).any(|&x| x <= 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::AnyOddNonPositive => Some(
                if arr.iter().filter(|&&x| x % 2 != 0).any(|&x| x <= 0) {
                    1
                } else {
                    0
                },
            ),
            DualAccum::MaxEvenNonZero => {
                let mut best: Option<i64> = None;
                for &x in arr {
                    if x % 2 == 0 && x != 0 {
                        best = Some(best.map_or(x, |b| b.max(x)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            DualAccum::MaxOddNonZero => {
                let mut best: Option<i64> = None;
                for &x in arr {
                    if x % 2 != 0 && x != 0 {
                        best = Some(best.map_or(x, |b| b.max(x)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            DualAccum::MinEvenNonZero => {
                let mut best: Option<i64> = None;
                for &x in arr {
                    if x % 2 == 0 && x != 0 {
                        best = Some(best.map_or(x, |b| b.min(x)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            DualAccum::MinOddNonZero => {
                let mut best: Option<i64> = None;
                for &x in arr {
                    if x % 2 != 0 && x != 0 {
                        best = Some(best.map_or(x, |b| b.min(x)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            DualAccum::MeanEvenNonZeroTrunc => {
                let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            DualAccum::MeanOddNonZeroTrunc => {
                let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            DualAccum::XorEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .fold(0i64, |a, &b| a ^ b),
            ),
            DualAccum::XorOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .fold(0i64, |a, &b| a ^ b),
            ),
            DualAccum::OrEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .fold(0i64, |a, &b| a | b),
            ),
            DualAccum::OrOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .fold(0i64, |a, &b| a | b),
            ),
            DualAccum::AndEvenNonZero => {
                let mut acc: Option<i64> = None;
                for &x in arr {
                    if x % 2 == 0 && x != 0 {
                        acc = Some(acc.map_or(x, |a| a & x));
                    }
                }
                Some(acc.unwrap_or(-1))
            }
            DualAccum::AndOddNonZero => {
                let mut acc: Option<i64> = None;
                for &x in arr {
                    if x % 2 != 0 && x != 0 {
                        acc = Some(acc.map_or(x, |a| a & x));
                    }
                }
                Some(acc.unwrap_or(-1))
            }
            DualAccum::SumAbsEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| x.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumAbsOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| x.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductAbsEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| x.abs())
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductAbsOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| x.abs())
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::GcdAbsEvenNonZero => {
                let mut g: Option<i64> = None;
                for &x in arr {
                    if x % 2 == 0 && x != 0 {
                        let a = x.abs();
                        g = Some(g.map_or(a, |g| i64_gcd(g, a)));
                    }
                }
                Some(g.unwrap_or(0))
            }
            DualAccum::GcdAbsOddNonZero => {
                let mut g: Option<i64> = None;
                for &x in arr {
                    if x % 2 != 0 && x != 0 {
                        let a = x.abs();
                        g = Some(g.map_or(a, |g| i64_gcd(g, a)));
                    }
                }
                Some(g.unwrap_or(0))
            }
            DualAccum::LcmAbsEvenNonZero => {
                let mut l: Option<i64> = None;
                for &x in arr {
                    if x % 2 == 0 && x != 0 {
                        let a = x.abs();
                        l = Some(l.map_or(a, |l| i64_lcm(l, a)));
                    }
                }
                Some(l.unwrap_or(1))
            }
            DualAccum::LcmAbsOddNonZero => {
                let mut l: Option<i64> = None;
                for &x in arr {
                    if x % 2 != 0 && x != 0 {
                        let a = x.abs();
                        l = Some(l.map_or(a, |l| i64_lcm(l, a)));
                    }
                }
                Some(l.unwrap_or(1))
            }
            DualAccum::MeanAbsEvenNonZeroTrunc => {
                let xs: Vec<i64> = arr
                    .iter()
                    .copied()
                    .filter(|&x| x % 2 == 0 && x != 0)
                    .map(|x| x.abs())
                    .collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            DualAccum::MeanAbsOddNonZeroTrunc => {
                let xs: Vec<i64> = arr
                    .iter()
                    .copied()
                    .filter(|&x| x % 2 != 0 && x != 0)
                    .map(|x| x.abs())
                    .collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            DualAccum::MaxAbsEvenNonZero => {
                let mut best: Option<i64> = None;
                for &x in arr {
                    if x % 2 == 0 && x != 0 {
                        let a = x.abs();
                        best = Some(best.map_or(a, |b| b.max(a)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            DualAccum::MaxAbsOddNonZero => {
                let mut best: Option<i64> = None;
                for &x in arr {
                    if x % 2 != 0 && x != 0 {
                        let a = x.abs();
                        best = Some(best.map_or(a, |b| b.max(a)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            DualAccum::MinAbsEvenNonZero => {
                let mut best: Option<i64> = None;
                for &x in arr {
                    if x % 2 == 0 && x != 0 {
                        let a = x.abs();
                        best = Some(best.map_or(a, |b| b.min(a)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            DualAccum::MinAbsOddNonZero => {
                let mut best: Option<i64> = None;
                for &x in arr {
                    if x % 2 != 0 && x != 0 {
                        let a = x.abs();
                        best = Some(best.map_or(a, |b| b.min(a)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            DualAccum::SumSquaresEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| x.saturating_mul(x))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumSquaresOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| x.saturating_mul(x))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductSquaresEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| x.saturating_mul(x))
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductSquaresOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| x.saturating_mul(x))
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::SumCubesEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| x.saturating_mul(x).saturating_mul(x))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumCubesOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| x.saturating_mul(x).saturating_mul(x))
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductCubesEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| x.saturating_mul(x).saturating_mul(x))
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductCubesOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| x.saturating_mul(x).saturating_mul(x))
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::SumFourthPowersEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumFourthPowersOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductFourthPowersEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s)
                    })
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductFourthPowersOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s)
                    })
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::MeanFourthPowersEvenNonZeroTrunc => {
                let xs: Vec<i64> = arr
                    .iter()
                    .copied()
                    .filter(|&x| x % 2 == 0 && x != 0)
                    .map(|x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s)
                    })
                    .collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            DualAccum::MeanFourthPowersOddNonZeroTrunc => {
                let xs: Vec<i64> = arr
                    .iter()
                    .copied()
                    .filter(|&x| x % 2 != 0 && x != 0)
                    .map(|x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s)
                    })
                    .collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            DualAccum::SumFifthPowersEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(x)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumFifthPowersOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(x)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductFifthPowersEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(x)
                    })
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductFifthPowersOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(x)
                    })
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::MeanFifthPowersEvenNonZeroTrunc => {
                let xs: Vec<i64> = arr
                    .iter()
                    .copied()
                    .filter(|&x| x % 2 == 0 && x != 0)
                    .map(|x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(x)
                    })
                    .collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            DualAccum::MeanFifthPowersOddNonZeroTrunc => {
                let xs: Vec<i64> = arr
                    .iter()
                    .copied()
                    .filter(|&x| x % 2 != 0 && x != 0)
                    .map(|x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(x)
                    })
                    .collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            DualAccum::SumSixthPowersEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(s)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumSixthPowersOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(s)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::ProductSixthPowersEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(s)
                    })
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::ProductSixthPowersOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(s)
                    })
                    .fold(1i64, i64::saturating_mul),
            ),
            DualAccum::MeanSixthPowersEvenNonZeroTrunc => {
                let xs: Vec<i64> = arr
                    .iter()
                    .copied()
                    .filter(|&x| x % 2 == 0 && x != 0)
                    .map(|x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(s)
                    })
                    .collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            DualAccum::MeanSixthPowersOddNonZeroTrunc => {
                let xs: Vec<i64> = arr
                    .iter()
                    .copied()
                    .filter(|&x| x % 2 != 0 && x != 0)
                    .map(|x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(s)
                    })
                    .collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            DualAccum::SumSeventhPowersEvenNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 == 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(s).saturating_mul(x)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
            DualAccum::SumSeventhPowersOddNonZero => Some(
                arr.iter()
                    .filter(|&&x| x % 2 != 0 && x != 0)
                    .map(|&x| {
                        let s = x.saturating_mul(x);
                        s.saturating_mul(s).saturating_mul(s).saturating_mul(x)
                    })
                    .fold(0i64, i64::saturating_add),
            ),
        }
    }

    fn emit(self, fn_name: &str) -> String {
        match self {
            DualAccum::Range => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    lo: i64 = arr[0];\n\
    hi: i64 = arr[0];\n\
    for item in arr {{\n\
        if item < lo {{\n\
            lo = item;\n\
        }}\n\
        if item > hi {{\n\
            hi = item;\n\
        }}\n\
    }}\n\
    return hi - lo;\n\
}}\n"
            ),
            DualAccum::SecondMax => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    first: i64 = arr[0];\n\
    second: i64 = arr[0];\n\
    for item in arr {{\n\
        if item > first {{\n\
            second = first;\n\
            first = item;\n\
        }} else {{\n\
            if item > second {{\n\
                second = item;\n\
            }}\n\
        }}\n\
    }}\n\
    return second;\n\
}}\n"
            ),
            DualAccum::SecondMin => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    first: i64 = arr[0];\n\
    second: i64 = arr[0];\n\
    for item in arr {{\n\
        if item < first {{\n\
            second = first;\n\
            first = item;\n\
        }} else {{\n\
            if item < second {{\n\
                second = item;\n\
            }}\n\
        }}\n\
    }}\n\
    return second;\n\
}}\n"
            ),
            DualAccum::StockProfit => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    min_price: i64 = arr[0];\n\
    best: i64 = 0;\n\
    for p in arr {{\n\
        if p < min_price {{ min_price = p; }}\n\
        profit: i64 = p - min_price;\n\
        if profit > best {{ best = profit; }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::PrefixMaxSum => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    running_max: i64 = arr[0];\n\
    total: i64 = 0;\n\
    for x in arr {{\n\
        if x > running_max {{ running_max = x; }}\n\
        total = total + running_max;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::PrefixMinSum => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    running_min: i64 = arr[0];\n\
    total: i64 = 0;\n\
    for x in arr {{\n\
        if x < running_min {{ running_min = x; }}\n\
        total = total + running_min;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::MaxSubarraySum => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    current: i64 = 0;\n\
    best: i64 = arr[0];\n\
    for item in arr {{\n\
        if current > 0 {{\n\
            current = current + item;\n\
        }} else {{\n\
            current = item;\n\
        }}\n\
        if current > best {{ best = current; }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinSubarraySum => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    current: i64 = 0;\n\
    best: i64 = arr[0];\n\
    for item in arr {{\n\
        if current < 0 {{\n\
            current = current + item;\n\
        }} else {{\n\
            current = item;\n\
        }}\n\
        if current < best {{ best = current; }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::Median => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    arr.sort();\n\
    return arr[arr.len / 2];\n\
}}\n"
            ),
            DualAccum::GcdAll => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    g: i64 = arr[0];\n\
    if g < 0 {{ g = 0 - g; }}\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = g;\n\
        b: i64 = arr[i];\n\
        if b < 0 {{ b = 0 - b; }}\n\
        while b != 0 {{\n\
            t: i64 = b;\n\
            b = a % b;\n\
            a = t;\n\
        }}\n\
        g = a;\n\
        i = i + 1;\n\
    }}\n\
    return g;\n\
}}\n"
            ),
            DualAccum::LcmAll => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    l: i64 = arr[0];\n\
    if l < 0 {{ l = 0 - l; }}\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = l;\n\
        b: i64 = arr[i];\n\
        if b < 0 {{ b = 0 - b; }}\n\
        x: i64 = a;\n\
        y: i64 = b;\n\
        while y != 0 {{\n\
            t: i64 = y;\n\
            y = x % y;\n\
            x = t;\n\
        }}\n\
        g: i64 = x;\n\
        l = (a / g) * b;\n\
        i = i + 1;\n\
    }}\n\
    return l;\n\
}}\n"
            ),
            DualAccum::MeanTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        total = total + item;\n\
    }}\n\
    return total / arr.len;\n\
}}\n"
            ),
            DualAccum::SumSquares => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        total = total + item * item;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::AbsSum => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        v: i64 = item;\n\
        if v < 0 {{ v = 0 - v; }}\n\
        total = total + v;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::MaxAbs => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len == 0 {{ return 0; }}\n\
    best: i64 = arr[0];\n\
    if best < 0 {{ best = 0 - best; }}\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        v: i64 = arr[i];\n\
        if v < 0 {{ v = 0 - v; }}\n\
        if v > best {{ best = v; }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
                        DualAccum::MinAbs => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len == 0 {{ return 0; }}\n\
    best: i64 = arr[0];\n\
    if best < 0 {{ best = 0 - best; }}\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        v: i64 = arr[i];\n\
        if v < 0 {{ v = 0 - v; }}\n\
        if v < best {{ best = v; }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
DualAccum::MinPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::CountNegatives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{ count = count + 1; }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::CountEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{ count = count + 1; }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::SumPositives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{ total = total + item; }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumNegatives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{ total = total + item; }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::CountOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{ count = count + 1; }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::Len => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    return arr.len;\n\
}}\n"
            ),
            DualAccum::IsEmpty => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len == 0 {{ return 1; }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::AllEqual => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len == 0 {{ return 1; }}\n\
    first: i64 = arr[0];\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] != first {{ return 0; }}\n\
        i = i + 1;\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::AnyPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item > 0 {{ return 1; }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::AnyNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item < 0 {{ return 1; }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::AnyZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item == 0 {{ return 1; }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::AllPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item <= 0 {{ return 0; }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::AllNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item >= 0 {{ return 0; }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::CountZeros => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item == 0 {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::HasDuplicate => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        j: i64 = 0;\n\
        while j < i {{\n\
            if arr[j] == arr[i] {{\n\
                return 1;\n\
            }}\n\
            j = j + 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::MaxNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::SumEvenValues => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumOddValues => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::AllNonNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item < 0 {{ return 0; }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::CountNonZeros => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::AlternatingSum => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if i % 2 == 0 {{\n\
            total = total + arr[i];\n\
        }} else {{\n\
            total = total - arr[i];\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductPositives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::MeanAbsTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            total = total - item;\n\
        }} else {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total / arr.len;\n\
}}\n"
            ),
            DualAccum::FirstPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item > 0 {{ return item; }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::LastPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        if arr[i] > 0 {{ return arr[i]; }}\n\
        i = i - 1;\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::FirstNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item < 0 {{ return item; }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::LastNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        if arr[i] < 0 {{ return arr[i]; }}\n\
        i = i - 1;\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::MaxPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::ProductNegatives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumCubes => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        total = total + item * item * item;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::CountGtMean => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    sum: i64 = 0;\n\
    for item in arr {{\n\
        sum = sum + item;\n\
    }}\n\
    mean: i64 = sum / arr.len;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item > mean {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::CountLtMean => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    sum: i64 = 0;\n\
    for item in arr {{\n\
        sum = sum + item;\n\
    }}\n\
    mean: i64 = sum / arr.len;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item < mean {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::IsPalindrome => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    i: i64 = 0;\n\
    j: i64 = arr.len - 1;\n\
    while i < j {{\n\
        if arr[i] != arr[j] {{ return 0; }}\n\
        i = i + 1;\n\
        j = j - 1;\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::ProductEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::AllNonPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item > 0 {{ return 0; }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::DotIndex => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        total = total + i * arr[i];\n\
        i = i + 1;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumSqDiffMean => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    sum: i64 = 0;\n\
    for item in arr {{\n\
        sum = sum + item;\n\
    }}\n\
    mean: i64 = sum / arr.len;\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        d: i64 = item - mean;\n\
        total = total + d * d;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::AnyNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item != 0 {{ return 1; }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::XorAll => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        total = total ^ item;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductNonZeros => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::OrAll => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        total = total | item;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::AndAll => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0 - 1;\n\
    for item in arr {{\n\
        total = total & item;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::CountEqMean => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    sum: i64 = 0;\n\
    for item in arr {{\n\
        sum = sum + item;\n\
    }}\n\
    mean: i64 = sum / arr.len;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item == mean {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::ProductAbs => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total * a;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::CountNonNegatives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item >= 0 {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::CountPositives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::MaxEvenValue => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MaxOddValue => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinEvenValue => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinOddValue => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::AbsRange => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    a0: i64 = arr[0];\n\
    if a0 < 0 {{ a0 = 0 - a0; }}\n\
    lo: i64 = a0;\n\
    hi: i64 = a0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < lo {{ lo = a; }}\n\
        if a > hi {{ hi = a; }}\n\
    }}\n\
    return hi - lo;\n\
}}\n"
            ),
            DualAccum::ProductNonNegatives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item >= 0 {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumNonNegatives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item >= 0 {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumNonPositives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item <= 0 {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::CountNonPositives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item <= 0 {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::ProductNonPositives => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item <= 0 {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumAbsEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumAbsOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductAbsEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total * a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductAbsOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total * a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::XorAbsAll => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total ^ a;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::AndAbsAll => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0 - 1;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total & a;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::OrAbsAll => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total | a;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
DualAccum::XorAbsEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total ^ a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
DualAccum::XorAbsOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total ^ a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
DualAccum::AndAbsEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0 - 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total & a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
DualAccum::AndAbsOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0 - 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total & a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
DualAccum::OrAbsEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total | a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
DualAccum::OrAbsOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total | a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
DualAccum::SumSquaresEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            total = total + (item * item);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
DualAccum::SumSquaresOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            total = total + (item * item);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumCubesEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            total = total + (item * item * item);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumCubesOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            total = total + (item * item * item);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductSquaresEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            total = total * (item * item);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductSquaresOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            total = total * (item * item);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::MaxAbsEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if found == 0 {{\n\
                best = a;\n\
                found = 1;\n\
            }} else {{\n\
                if a > best {{ best = a; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MaxAbsOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if found == 0 {{\n\
                best = a;\n\
                found = 1;\n\
            }} else {{\n\
                if a > best {{ best = a; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinAbsEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if found == 0 {{\n\
                best = a;\n\
                found = 1;\n\
            }} else {{\n\
                if a < best {{ best = a; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinAbsOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if found == 0 {{\n\
                best = a;\n\
                found = 1;\n\
            }} else {{\n\
                if a < best {{ best = a; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::CountNonZeroEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % 2 == 0 {{\n\
                count = count + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::CountNonZeroOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % 2 != 0 {{\n\
                count = count + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            DualAccum::SumNonZeroEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % 2 == 0 {{\n\
                total = total + item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumNonZeroOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % 2 != 0 {{\n\
                total = total + item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductNonZeroEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % 2 == 0 {{\n\
                total = total * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductNonZeroOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % 2 != 0 {{\n\
                total = total * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::MeanAbsEvensTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    if count == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            DualAccum::MeanAbsOddsTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    if count == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            DualAccum::GcdAbsEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    g: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if found == 0 {{\n\
                g = a;\n\
                found = 1;\n\
            }} else {{\n\
                aa: i64 = g;\n\
                bb: i64 = a;\n\
                while bb != 0 {{\n\
                    t: i64 = bb;\n\
                    bb = aa % bb;\n\
                    aa = t;\n\
                }}\n\
                g = aa;\n\
            }}\n\
        }}\n\
    }}\n\
    return g;\n\
}}\n"
            ),
            DualAccum::GcdAbsOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    g: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if found == 0 {{\n\
                g = a;\n\
                found = 1;\n\
            }} else {{\n\
                aa: i64 = g;\n\
                bb: i64 = a;\n\
                while bb != 0 {{\n\
                    t: i64 = bb;\n\
                    bb = aa % bb;\n\
                    aa = t;\n\
                }}\n\
                g = aa;\n\
            }}\n\
        }}\n\
    }}\n\
    return g;\n\
}}\n"
            ),
            DualAccum::LcmAbsEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    g: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if found == 0 {{\n\
                g = a;\n\
                found = 1;\n\
            }} else {{\n\
                aa: i64 = g;\n\
                bb: i64 = a;\n\
                while bb != 0 {{\n\
                    t: i64 = bb;\n\
                    bb = aa % bb;\n\
                    aa = t;\n\
                }}\n\
                gg: i64 = aa;\n\
                g = (g / gg) * a;\n\
            }}\n\
        }}\n\
    }}\n\
    return g;\n\
}}\n"
            ),
            DualAccum::LcmAbsOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    g: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if found == 0 {{\n\
                g = a;\n\
                found = 1;\n\
            }} else {{\n\
                aa: i64 = g;\n\
                bb: i64 = a;\n\
                while bb != 0 {{\n\
                    t: i64 = bb;\n\
                    bb = aa % bb;\n\
                    aa = t;\n\
                }}\n\
                gg: i64 = aa;\n\
                g = (g / gg) * a;\n\
            }}\n\
        }}\n\
    }}\n\
    return g;\n\
}}\n"
            ),
            DualAccum::ProductCubesEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            total = total * (item * item * item);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductCubesOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            total = total * (item * item * item);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumAbsCubesEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + (a * a * a);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumAbsCubesOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + (a * a * a);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductAbsCubesEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total * (a * a * a);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductAbsCubesOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total * (a * a * a);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumAbsSquaresEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + (a * a);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumAbsSquaresOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + (a * a);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductAbsSquaresEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total * (a * a);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductAbsSquaresOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total * (a * a);\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::MeanAbsSquaresEvensTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + (a * a);\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    if count == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            DualAccum::MeanAbsSquaresOddsTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + (a * a);\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    if count == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            DualAccum::CountPositiveEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 == 0 {{\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return n;\n\
}}\n"
            ),
            DualAccum::CountPositiveOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 != 0 {{\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return n;\n\
}}\n"
            ),
            DualAccum::CountNegativeEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 == 0 {{\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return n;\n\
}}\n"
            ),
            DualAccum::CountNegativeOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 != 0 {{\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return n;\n\
}}\n"
            ),
            DualAccum::SumPositiveEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 == 0 {{\n\
                total = total + item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumPositiveOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 != 0 {{\n\
                total = total + item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumNegativeEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 == 0 {{\n\
                total = total + item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumNegativeOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 != 0 {{\n\
                total = total + item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductPositiveEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 == 0 {{\n\
                total = total * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductPositiveOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 != 0 {{\n\
                total = total * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductNegativeEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 == 0 {{\n\
                total = total * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductNegativeOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 != 0 {{\n\
                total = total * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::MaxPositiveEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 == 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }} else {{\n\
                    if item > best {{ best = item; }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MaxPositiveOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 != 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }} else {{\n\
                    if item > best {{ best = item; }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinPositiveEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 == 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }} else {{\n\
                    if item < best {{ best = item; }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinPositiveOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 != 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }} else {{\n\
                    if item < best {{ best = item; }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MaxNegativeEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 == 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }} else {{\n\
                    if item > best {{ best = item; }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MaxNegativeOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 != 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }} else {{\n\
                    if item > best {{ best = item; }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinNegativeEvens => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 == 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }} else {{\n\
                    if item < best {{ best = item; }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinNegativeOdds => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 != 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }} else {{\n\
                    if item < best {{ best = item; }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MeanPositiveEvensTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 == 0 {{\n\
                total = total + item;\n\
                count = count + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if count == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            DualAccum::MeanPositiveOddsTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item > 0 {{\n\
            if item % 2 != 0 {{\n\
                total = total + item;\n\
                count = count + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if count == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            DualAccum::MeanNegativeEvensTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 == 0 {{\n\
                total = total + item;\n\
                count = count + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if count == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            DualAccum::MeanNegativeOddsTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item < 0 {{\n\
            if item % 2 != 0 {{\n\
                total = total + item;\n\
                count = count + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if count == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            DualAccum::AllEvenPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item <= 0 {{\n\
                return 0;\n\
            }}\n\
        }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::AllOddPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item <= 0 {{\n\
                return 0;\n\
            }}\n\
        }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::AllEvenNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item >= 0 {{\n\
                return 0;\n\
            }}\n\
        }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::AllOddNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item >= 0 {{\n\
                return 0;\n\
            }}\n\
        }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),


            DualAccum::AnyEvenPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item > 0 {{\n\
                return 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::AnyOddPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item > 0 {{\n\
                return 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),

            DualAccum::AnyEvenNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item < 0 {{\n\
                return 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::AnyOddNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item < 0 {{\n\
                return 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),

            DualAccum::AnyEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                return 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::AnyOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                return 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),

            DualAccum::AllEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item == 0 {{\n\
                return 0;\n\
            }}\n\
        }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::AllOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item == 0 {{\n\
                return 0;\n\
            }}\n\
        }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),

            DualAccum::AllEvenNonNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item < 0 {{\n\
                return 0;\n\
            }}\n\
        }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::AllOddNonNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item < 0 {{\n\
                return 0;\n\
            }}\n\
        }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),

            DualAccum::AllEvenNonPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item > 0 {{\n\
                return 0;\n\
            }}\n\
        }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            DualAccum::AllOddNonPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item > 0 {{\n\
                return 0;\n\
            }}\n\
        }}\n\
    }}\n\
    return 1;\n\
}}\n"
            ),

            DualAccum::AnyEvenNonNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item >= 0 {{\n\
                return 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::AnyOddNonNegative => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item >= 0 {{\n\
                return 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),

            DualAccum::AnyEvenNonPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item <= 0 {{\n\
                return 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            DualAccum::AnyOddNonPositive => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item <= 0 {{\n\
                return 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),

            DualAccum::MaxEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }}\n\
                if item > best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MaxOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }}\n\
                if item > best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),

            DualAccum::MinEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }}\n\
                if item < best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }}\n\
                if item < best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),

            DualAccum::MeanEvenNonZeroTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                total = total + item;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),
            DualAccum::MeanOddNonZeroTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                total = total + item;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),

            DualAccum::XorEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    x: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                x = x ^ item;\n\
            }}\n\
        }}\n\
    }}\n\
    return x;\n\
}}\n"
            ),
            DualAccum::XorOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    x: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                x = x ^ item;\n\
            }}\n\
        }}\n\
    }}\n\
    return x;\n\
}}\n"
            ),

            DualAccum::OrEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    x: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                x = x | item;\n\
            }}\n\
        }}\n\
    }}\n\
    return x;\n\
}}\n"
            ),
            DualAccum::OrOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    x: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                x = x | item;\n\
            }}\n\
        }}\n\
    }}\n\
    return x;\n\
}}\n"
            ),

            DualAccum::AndEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    x: i64 = 0 - 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                if found == 0 {{\n\
                    x = item;\n\
                    found = 1;\n\
                }} else {{\n\
                    x = x & item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return x;\n\
}}\n"
            ),
            DualAccum::AndOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    x: i64 = 0 - 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                if found == 0 {{\n\
                    x = item;\n\
                    found = 1;\n\
                }} else {{\n\
                    x = x & item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return x;\n\
}}\n"
            ),

            DualAccum::SumAbsEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                total = total + a;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumAbsOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                total = total + a;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),

            DualAccum::ProductAbsEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                prod = prod * a;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            DualAccum::ProductAbsOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                prod = prod * a;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),

            DualAccum::GcdAbsEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    g: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                if found == 0 {{\n\
                    g = a;\n\
                    found = 1;\n\
                }} else {{\n\
                    while a != 0 {{\n\
                        t: i64 = a;\n\
                        a = g % a;\n\
                        g = t;\n\
                    }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return g;\n\
}}\n"
            ),
            DualAccum::GcdAbsOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    g: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                if found == 0 {{\n\
                    g = a;\n\
                    found = 1;\n\
                }} else {{\n\
                    while a != 0 {{\n\
                        t: i64 = a;\n\
                        a = g % a;\n\
                        g = t;\n\
                    }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return g;\n\
}}\n"
            ),

            DualAccum::LcmAbsEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    l: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                if found == 0 {{\n\
                    l = a;\n\
                    found = 1;\n\
                }} else {{\n\
                    g: i64 = l;\n\
                    b: i64 = a;\n\
                    while b != 0 {{\n\
                        t: i64 = b;\n\
                        b = g % b;\n\
                        g = t;\n\
                    }}\n\
                    l = (l / g) * a;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return l;\n\
}}\n"
            ),
            DualAccum::LcmAbsOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    l: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                if found == 0 {{\n\
                    l = a;\n\
                    found = 1;\n\
                }} else {{\n\
                    g: i64 = l;\n\
                    b: i64 = a;\n\
                    while b != 0 {{\n\
                        t: i64 = b;\n\
                        b = g % b;\n\
                        g = t;\n\
                    }}\n\
                    l = (l / g) * a;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return l;\n\
}}\n"
            ),

            DualAccum::MeanAbsEvenNonZeroTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                total = total + a;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),
            DualAccum::MeanAbsOddNonZeroTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                total = total + a;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),

            DualAccum::MaxAbsEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                if found == 0 {{\n\
                    best = a;\n\
                    found = 1;\n\
                }}\n\
                if a > best {{\n\
                    best = a;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MaxAbsOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                if found == 0 {{\n\
                    best = a;\n\
                    found = 1;\n\
                }}\n\
                if a > best {{\n\
                    best = a;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),

            DualAccum::MinAbsEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                if found == 0 {{\n\
                    best = a;\n\
                    found = 1;\n\
                }}\n\
                if a < best {{\n\
                    best = a;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            DualAccum::MinAbsOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                if found == 0 {{\n\
                    best = a;\n\
                    found = 1;\n\
                }}\n\
                if a < best {{\n\
                    best = a;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),

            DualAccum::SumSquaresEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                total = total + item * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumSquaresOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                total = total + item * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),

            DualAccum::ProductSquaresEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                prod = prod * item * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            DualAccum::ProductSquaresOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                prod = prod * item * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),

            DualAccum::SumCubesEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                total = total + item * item * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumCubesOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                total = total + item * item * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),

            DualAccum::ProductCubesEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                prod = prod * item * item * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            DualAccum::ProductCubesOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                prod = prod * item * item * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            DualAccum::SumFourthPowersEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumFourthPowersOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductFourthPowersEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                prod = prod * sq * sq;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            DualAccum::ProductFourthPowersOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                prod = prod * sq * sq;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            DualAccum::MeanFourthPowersEvenNonZeroTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),
            DualAccum::MeanFourthPowersOddNonZeroTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),
            DualAccum::SumFifthPowersEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumFifthPowersOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductFifthPowersEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                prod = prod * sq * sq * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            DualAccum::ProductFifthPowersOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                prod = prod * sq * sq * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            DualAccum::MeanFifthPowersEvenNonZeroTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq * item;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),
            DualAccum::MeanFifthPowersOddNonZeroTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq * item;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),
            DualAccum::SumSixthPowersEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq * sq;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumSixthPowersOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq * sq;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::ProductSixthPowersEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                prod = prod * sq * sq * sq;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            DualAccum::ProductSixthPowersOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                prod = prod * sq * sq * sq;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            DualAccum::MeanSixthPowersEvenNonZeroTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq * sq;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),
            DualAccum::MeanSixthPowersOddNonZeroTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq * sq;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),
            DualAccum::SumSeventhPowersEvenNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 == 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq * sq * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            DualAccum::SumSeventhPowersOddNonZero => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % 2 != 0 {{\n\
            if item != 0 {{\n\
                sq: i64 = item * item;\n\
                total = total + sq * sq * sq * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),

        }
    }
}

/// Adjacent-pair scans (native A4 parity slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairwiseScan {
    /// `max |arr[i] - arr[i-1]|`.
    MaxAbsDiff,
    /// Count positions where `arr[i] != arr[i-1]`.
    CountAdjacentDiff,
    /// Count positions where `arr[i] > arr[i-1]`.
    CountIncreases,
    /// Count positions where `arr[i] < arr[i-1]`.
    CountDecreases,
    /// 1 if strictly increasing for all adjacent pairs, else 0.
    StrictlyIncreasing,
    /// 1 if strictly decreasing for all adjacent pairs, else 0.
    StrictlyDecreasing,
    /// 1 if non-decreasing (sorted ascending, ties ok), else 0.
    NonDecreasing,
    /// 1 if non-increasing (sorted descending, ties ok), else 0.
    NonIncreasing,
    /// Length of the longest run of equal consecutive elements.
    LongestPlateau,
    /// Sum of `|arr[i] - arr[i-1]|` over adjacent pairs.
    SumAbsDiff,
    /// Length of the longest strictly-increasing contiguous run.
    LongestIncreasingRun,
    /// Length of the longest strictly-decreasing contiguous run.
    LongestDecreasingRun,
    /// Count positions where `arr[i] == arr[i-1]`.
    CountAdjacentEq,
    /// Max positive adjacent rise `arr[i]-arr[i-1]` (0 if none).
    MaxIncrease,
    /// Max positive adjacent fall `arr[i-1]-arr[i]` (0 if none).
    MaxDecrease,
    /// Length of the longest non-decreasing contiguous run.
    LongestNonDecreasingRun,
    /// Length of the longest non-increasing contiguous run.
    LongestNonIncreasingRun,
    /// Sum of positive adjacent rises (ignore non-positive deltas).
    SumIncreases,
    /// Sum of positive adjacent falls (ignore non-positive deltas).
    SumDecreases,
    /// Number of equal-runs (plateaus) of length ≥ 1.
    CountPlateaus,
    /// 1 if adjacent deltas strictly alternate sign (len<3 → 1), else 0.
    IsZigZag,
    /// Min positive adjacent rise (0 if none).
    MinIncrease,
    /// Min positive adjacent fall (0 if none).
    MinDecrease,
    /// Truncating mean of adjacent absolute diffs (len<2 → 0).
    MeanAbsDiffTrunc,
    /// Count of adjacent pairs whose deltas have opposite signs (zeros ignored).
    CountSignChanges,
    /// Sum of squared adjacent diffs `(arr[i]-arr[i-1])^2`.
    SumSqDiff,
    /// Truncating mean of squared adjacent diffs (len<2 → 0).
    MeanSqDiffTrunc,
    /// First index i>0 where arr[i] > arr[i-1], else -1.
    FirstIncreaseIdx,
    /// First index i>0 where arr[i] < arr[i-1], else -1.
    FirstDecreaseIdx,
    /// Last index i>0 where arr[i] > arr[i-1], else -1.
    LastIncreaseIdx,
    /// Last index i>0 where arr[i] < arr[i-1], else -1.
    LastDecreaseIdx,
}

impl PairwiseScan {
    fn label(self) -> &'static str {
        match self {
            PairwiseScan::MaxAbsDiff => "max_abs_diff",
            PairwiseScan::CountAdjacentDiff => "count_adjacent_diff",
            PairwiseScan::CountIncreases => "count_increases",
            PairwiseScan::CountDecreases => "count_decreases",
            PairwiseScan::StrictlyIncreasing => "strictly_increasing",
            PairwiseScan::StrictlyDecreasing => "strictly_decreasing",
            PairwiseScan::NonDecreasing => "non_decreasing",
            PairwiseScan::NonIncreasing => "non_increasing",
            PairwiseScan::LongestPlateau => "longest_plateau",
            PairwiseScan::SumAbsDiff => "sum_abs_diff",
            PairwiseScan::LongestIncreasingRun => "longest_increasing_run",
            PairwiseScan::LongestDecreasingRun => "longest_decreasing_run",
            PairwiseScan::CountAdjacentEq => "count_adjacent_eq",
            PairwiseScan::MaxIncrease => "max_increase",
            PairwiseScan::MaxDecrease => "max_decrease",
            PairwiseScan::LongestNonDecreasingRun => "longest_non_decreasing_run",
            PairwiseScan::LongestNonIncreasingRun => "longest_non_increasing_run",
            PairwiseScan::SumIncreases => "sum_increases",
            PairwiseScan::SumDecreases => "sum_decreases",
            PairwiseScan::CountPlateaus => "count_plateaus",
            PairwiseScan::IsZigZag => "is_zigzag",
            PairwiseScan::MinIncrease => "min_increase",
            PairwiseScan::MinDecrease => "min_decrease",
            PairwiseScan::MeanAbsDiffTrunc => "mean_abs_diff_trunc",
            PairwiseScan::CountSignChanges => "count_sign_changes",
            PairwiseScan::SumSqDiff => "sum_sq_diff",
            PairwiseScan::MeanSqDiffTrunc => "mean_sq_diff_trunc",
            PairwiseScan::FirstIncreaseIdx => "first_increase_idx",
            PairwiseScan::FirstDecreaseIdx => "first_decrease_idx",
            PairwiseScan::LastIncreaseIdx => "last_increase_idx",
            PairwiseScan::LastDecreaseIdx => "last_decrease_idx",
        }
    }

    fn eval(self, arr: &[i64]) -> Option<i64> {
        if arr.is_empty() {
            return match self {
                PairwiseScan::LongestPlateau
                | PairwiseScan::LongestIncreasingRun
                | PairwiseScan::LongestDecreasingRun
                | PairwiseScan::LongestNonDecreasingRun
                | PairwiseScan::LongestNonIncreasingRun
                | PairwiseScan::CountPlateaus => None,
                PairwiseScan::StrictlyIncreasing
                | PairwiseScan::StrictlyDecreasing
                | PairwiseScan::NonDecreasing
                | PairwiseScan::NonIncreasing
                | PairwiseScan::IsZigZag => Some(1),
                PairwiseScan::FirstIncreaseIdx
                | PairwiseScan::FirstDecreaseIdx
                | PairwiseScan::LastIncreaseIdx
                | PairwiseScan::LastDecreaseIdx => Some(-1),
                _ => Some(0),
            };
        }
        if arr.len() < 2 {
            return Some(match self {
                PairwiseScan::StrictlyIncreasing
                | PairwiseScan::StrictlyDecreasing
                | PairwiseScan::NonDecreasing
                | PairwiseScan::NonIncreasing
                | PairwiseScan::IsZigZag => 1,
                PairwiseScan::LongestPlateau
                | PairwiseScan::LongestIncreasingRun
                | PairwiseScan::LongestDecreasingRun
                | PairwiseScan::LongestNonDecreasingRun
                | PairwiseScan::LongestNonIncreasingRun
                | PairwiseScan::CountPlateaus => 1,
                PairwiseScan::FirstIncreaseIdx
                | PairwiseScan::FirstDecreaseIdx
                | PairwiseScan::LastIncreaseIdx
                | PairwiseScan::LastDecreaseIdx => -1,
                _ => 0,
            });
        }
        match self {
            PairwiseScan::MaxAbsDiff => {
                let mut best = 0i64;
                for i in 1..arr.len() {
                    let mut diff = arr[i].saturating_sub(arr[i - 1]);
                    if diff < 0 {
                        diff = 0i64.saturating_sub(diff);
                    }
                    if diff > best {
                        best = diff;
                    }
                }
                Some(best)
            }
            PairwiseScan::CountAdjacentDiff => {
                let mut count = 0i64;
                for i in 1..arr.len() {
                    if arr[i] != arr[i - 1] {
                        count += 1;
                    }
                }
                Some(count)
            }
            PairwiseScan::CountIncreases => {
                let mut count = 0i64;
                for i in 1..arr.len() {
                    if arr[i] > arr[i - 1] {
                        count += 1;
                    }
                }
                Some(count)
            }
            PairwiseScan::CountDecreases => {
                let mut count = 0i64;
                for i in 1..arr.len() {
                    if arr[i] < arr[i - 1] {
                        count += 1;
                    }
                }
                Some(count)
            }
            PairwiseScan::StrictlyIncreasing => {
                for i in 1..arr.len() {
                    if arr[i] <= arr[i - 1] {
                        return Some(0);
                    }
                }
                Some(1)
            }
            PairwiseScan::StrictlyDecreasing => {
                for i in 1..arr.len() {
                    if arr[i] >= arr[i - 1] {
                        return Some(0);
                    }
                }
                Some(1)
            }
            PairwiseScan::NonDecreasing => {
                for i in 1..arr.len() {
                    if arr[i] < arr[i - 1] {
                        return Some(0);
                    }
                }
                Some(1)
            }
            PairwiseScan::NonIncreasing => {
                for i in 1..arr.len() {
                    if arr[i] > arr[i - 1] {
                        return Some(0);
                    }
                }
                Some(1)
            }
            PairwiseScan::LongestPlateau => {
                let mut best = 1i64;
                let mut cur = 1i64;
                for i in 1..arr.len() {
                    if arr[i] == arr[i - 1] {
                        cur += 1;
                        if cur > best {
                            best = cur;
                        }
                    } else {
                        cur = 1;
                    }
                }
                Some(best)
            }
            PairwiseScan::SumAbsDiff => {
                let mut total = 0i64;
                for i in 1..arr.len() {
                    let mut diff = arr[i].saturating_sub(arr[i - 1]);
                    if diff < 0 {
                        diff = 0i64.saturating_sub(diff);
                    }
                    total = total.saturating_add(diff);
                }
                Some(total)
            }
            PairwiseScan::LongestIncreasingRun => {
                let mut best = 1i64;
                let mut cur = 1i64;
                for i in 1..arr.len() {
                    if arr[i] > arr[i - 1] {
                        cur += 1;
                        if cur > best {
                            best = cur;
                        }
                    } else {
                        cur = 1;
                    }
                }
                Some(best)
            }
            PairwiseScan::LongestDecreasingRun => {
                let mut best = 1i64;
                let mut cur = 1i64;
                for i in 1..arr.len() {
                    if arr[i] < arr[i - 1] {
                        cur += 1;
                        if cur > best {
                            best = cur;
                        }
                    } else {
                        cur = 1;
                    }
                }
                Some(best)
            }
            PairwiseScan::CountAdjacentEq => {
                let mut count = 0i64;
                for i in 1..arr.len() {
                    if arr[i] == arr[i - 1] {
                        count += 1;
                    }
                }
                Some(count)
            }
            PairwiseScan::MaxIncrease => {
                let mut best = 0i64;
                for i in 1..arr.len() {
                    let rise = arr[i].saturating_sub(arr[i - 1]);
                    if rise > best {
                        best = rise;
                    }
                }
                Some(best)
            }
            PairwiseScan::MaxDecrease => {
                let mut best = 0i64;
                for i in 1..arr.len() {
                    let fall = arr[i - 1].saturating_sub(arr[i]);
                    if fall > best {
                        best = fall;
                    }
                }
                Some(best)
            }
            PairwiseScan::LongestNonDecreasingRun => {
                let mut best = 1i64;
                let mut cur = 1i64;
                for i in 1..arr.len() {
                    if arr[i] >= arr[i - 1] {
                        cur += 1;
                        if cur > best {
                            best = cur;
                        }
                    } else {
                        cur = 1;
                    }
                }
                Some(best)
            }
            PairwiseScan::LongestNonIncreasingRun => {
                let mut best = 1i64;
                let mut cur = 1i64;
                for i in 1..arr.len() {
                    if arr[i] <= arr[i - 1] {
                        cur += 1;
                        if cur > best {
                            best = cur;
                        }
                    } else {
                        cur = 1;
                    }
                }
                Some(best)
            }
            PairwiseScan::SumIncreases => {
                let mut total = 0i64;
                for i in 1..arr.len() {
                    let rise = arr[i].saturating_sub(arr[i - 1]);
                    if rise > 0 {
                        total = total.saturating_add(rise);
                    }
                }
                Some(total)
            }
            PairwiseScan::SumDecreases => {
                let mut total = 0i64;
                for i in 1..arr.len() {
                    let fall = arr[i - 1].saturating_sub(arr[i]);
                    if fall > 0 {
                        total = total.saturating_add(fall);
                    }
                }
                Some(total)
            }
            PairwiseScan::CountPlateaus => {
                let mut count = 1i64;
                for i in 1..arr.len() {
                    if arr[i] != arr[i - 1] {
                        count += 1;
                    }
                }
                Some(count)
            }
            PairwiseScan::IsZigZag => {
                if arr.len() < 3 {
                    return Some(1);
                }
                let mut ok = true;
                for i in 2..arr.len() {
                    let d0 = arr[i - 1].saturating_sub(arr[i - 2]);
                    let d1 = arr[i].saturating_sub(arr[i - 1]);
                    if d0 == 0 || d1 == 0 || (d0 > 0) == (d1 > 0) {
                        ok = false;
                        break;
                    }
                }
                Some(if ok { 1 } else { 0 })
            }
            PairwiseScan::MinIncrease => {
                let mut best = 0i64;
                let mut found = false;
                for i in 1..arr.len() {
                    let rise = arr[i].saturating_sub(arr[i - 1]);
                    if rise > 0 {
                        if !found || rise < best {
                            best = rise;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            PairwiseScan::MinDecrease => {
                let mut best = 0i64;
                let mut found = false;
                for i in 1..arr.len() {
                    let fall = arr[i - 1].saturating_sub(arr[i]);
                    if fall > 0 {
                        if !found || fall < best {
                            best = fall;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            PairwiseScan::MeanAbsDiffTrunc => {
                if arr.len() < 2 {
                    return Some(0);
                }
                let mut total = 0i64;
                for i in 1..arr.len() {
                    let mut d = arr[i].saturating_sub(arr[i - 1]);
                    if d < 0 {
                        d = 0i64.saturating_sub(d);
                    }
                    total = total.saturating_add(d);
                }
                Some(total / ((arr.len() - 1) as i64))
            }
            PairwiseScan::CountSignChanges => {
                let mut count = 0i64;
                let mut prev_sign = 0i64; // -1,0,1
                for i in 1..arr.len() {
                    let d = arr[i].saturating_sub(arr[i - 1]);
                    let sign = if d > 0 {
                        1
                    } else if d < 0 {
                        -1
                    } else {
                        0
                    };
                    if sign != 0 {
                        if prev_sign != 0 && sign != prev_sign {
                            count += 1;
                        }
                        prev_sign = sign;
                    }
                }
                Some(count)
            }
            PairwiseScan::SumSqDiff => {
                let mut total = 0i64;
                for i in 1..arr.len() {
                    let d = arr[i].saturating_sub(arr[i - 1]);
                    total = total.saturating_add(d.saturating_mul(d));
                }
                Some(total)
            }
            PairwiseScan::MeanSqDiffTrunc => {
                if arr.len() < 2 {
                    return Some(0);
                }
                let mut total = 0i64;
                for i in 1..arr.len() {
                    let d = arr[i].saturating_sub(arr[i - 1]);
                    total = total.saturating_add(d.saturating_mul(d));
                }
                Some(total / ((arr.len() - 1) as i64))
            }
            PairwiseScan::FirstIncreaseIdx => {
                for i in 1..arr.len() {
                    if arr[i] > arr[i - 1] {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            PairwiseScan::FirstDecreaseIdx => {
                for i in 1..arr.len() {
                    if arr[i] < arr[i - 1] {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            PairwiseScan::LastIncreaseIdx => {
                let mut best = -1i64;
                for i in 1..arr.len() {
                    if arr[i] > arr[i - 1] {
                        best = i as i64;
                    }
                }
                Some(best)
            }
            PairwiseScan::LastDecreaseIdx => {
                let mut best = -1i64;
                for i in 1..arr.len() {
                    if arr[i] < arr[i - 1] {
                        best = i as i64;
                    }
                }
                Some(best)
            }
        }
    }

    fn emit(self, fn_name: &str) -> String {
        match self {
            PairwiseScan::MaxAbsDiff => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        diff: i64 = arr[i] - arr[i - 1];\n\
        if diff < 0 {{ diff = 0 - diff; }}\n\
        if diff > best {{ best = diff; }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::CountAdjacentDiff => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] != arr[i - 1] {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            PairwiseScan::CountIncreases => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] > arr[i - 1] {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            PairwiseScan::CountDecreases => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] < arr[i - 1] {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            PairwiseScan::StrictlyIncreasing => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] <= arr[i - 1] {{\n\
            return 0;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            PairwiseScan::StrictlyDecreasing => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] >= arr[i - 1] {{\n\
            return 0;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            PairwiseScan::NonDecreasing => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] < arr[i - 1] {{\n\
            return 0;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            PairwiseScan::NonIncreasing => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] > arr[i - 1] {{\n\
            return 0;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            PairwiseScan::LongestPlateau => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 1;\n\
    cur: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] == arr[i - 1] {{\n\
            cur = cur + 1;\n\
            if cur > best {{ best = cur; }}\n\
        }} else {{\n\
            cur = 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::SumAbsDiff => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        diff: i64 = arr[i] - arr[i - 1];\n\
        if diff < 0 {{ diff = 0 - diff; }}\n\
        total = total + diff;\n\
        i = i + 1;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            PairwiseScan::LongestIncreasingRun => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 1;\n\
    cur: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] > arr[i - 1] {{\n\
            cur = cur + 1;\n\
            if cur > best {{ best = cur; }}\n\
        }} else {{\n\
            cur = 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::LongestDecreasingRun => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 1;\n\
    cur: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] < arr[i - 1] {{\n\
            cur = cur + 1;\n\
            if cur > best {{ best = cur; }}\n\
        }} else {{\n\
            cur = 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::CountAdjacentEq => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] == arr[i - 1] {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            PairwiseScan::MaxIncrease => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        rise: i64 = arr[i] - arr[i - 1];\n\
        if rise > best {{ best = rise; }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::MaxDecrease => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        fall: i64 = arr[i - 1] - arr[i];\n\
        if fall > best {{ best = fall; }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::LongestNonDecreasingRun => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 1;\n\
    cur: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] >= arr[i - 1] {{\n\
            cur = cur + 1;\n\
            if cur > best {{ best = cur; }}\n\
        }} else {{\n\
            cur = 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::LongestNonIncreasingRun => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 1;\n\
    cur: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] <= arr[i - 1] {{\n\
            cur = cur + 1;\n\
            if cur > best {{ best = cur; }}\n\
        }} else {{\n\
            cur = 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::SumIncreases => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        rise: i64 = arr[i] - arr[i - 1];\n\
        if rise > 0 {{ total = total + rise; }}\n\
        i = i + 1;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            PairwiseScan::SumDecreases => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        fall: i64 = arr[i - 1] - arr[i];\n\
        if fall > 0 {{ total = total + fall; }}\n\
        i = i + 1;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            PairwiseScan::CountPlateaus => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] != arr[i - 1] {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            PairwiseScan::IsZigZag => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len < 3 {{ return 1; }}\n\
    i: i64 = 2;\n\
    while i < arr.len {{\n\
        d0: i64 = arr[i - 1] - arr[i - 2];\n\
        d1: i64 = arr[i] - arr[i - 1];\n\
        if d0 == 0 {{ return 0; }}\n\
        if d1 == 0 {{ return 0; }}\n\
        if d0 > 0 {{\n\
            if d1 > 0 {{ return 0; }}\n\
        }} else {{\n\
            if d1 < 0 {{ return 0; }}\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 1;\n\
}}\n"
            ),
            PairwiseScan::MinIncrease => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        rise: i64 = arr[i] - arr[i - 1];\n\
        if rise > 0 {{\n\
            if found == 0 {{\n\
                best = rise;\n\
                found = 1;\n\
            }} else {{\n\
                if rise < best {{ best = rise; }}\n\
            }}\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::MinDecrease => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        fall: i64 = arr[i - 1] - arr[i];\n\
        if fall > 0 {{\n\
            if found == 0 {{\n\
                best = fall;\n\
                found = 1;\n\
            }} else {{\n\
                if fall < best {{ best = fall; }}\n\
            }}\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::MeanAbsDiffTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len < 2 {{ return 0; }}\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        d: i64 = arr[i] - arr[i - 1];\n\
        if d < 0 {{ d = 0 - d; }}\n\
        total = total + d;\n\
        i = i + 1;\n\
    }}\n\
    return total / (arr.len - 1);\n\
}}\n"
            ),
            PairwiseScan::CountSignChanges => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    prev: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        d: i64 = arr[i] - arr[i - 1];\n\
        sign: i64 = 0;\n\
        if d > 0 {{ sign = 1; }}\n\
        if d < 0 {{ sign = 0 - 1; }}\n\
        if sign != 0 {{\n\
            if prev != 0 {{\n\
                if sign != prev {{\n\
                    count = count + 1;\n\
                }}\n\
            }}\n\
            prev = sign;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            PairwiseScan::SumSqDiff => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        d: i64 = arr[i] - arr[i - 1];\n\
        total = total + d * d;\n\
        i = i + 1;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            PairwiseScan::MeanSqDiffTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len < 2 {{ return 0; }}\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        d: i64 = arr[i] - arr[i - 1];\n\
        total = total + d * d;\n\
        i = i + 1;\n\
    }}\n\
    return total / (arr.len - 1);\n\
}}\n"
            ),
            PairwiseScan::FirstIncreaseIdx => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] > arr[i - 1] {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            PairwiseScan::FirstDecreaseIdx => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] < arr[i - 1] {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            PairwiseScan::LastIncreaseIdx => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0 - 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] > arr[i - 1] {{\n\
            best = i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            PairwiseScan::LastDecreaseIdx => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0 - 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] < arr[i - 1] {{\n\
            best = i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
        }
    }
}

fn try_dual_and_pairwise(
    problem: &Problem,
    fn_name: &str,
    inputs: &[Vec<i64>],
    expected: &[i64],
) -> Option<SolveResult> {
    for dual in [
        DualAccum::Range,
        DualAccum::SecondMax,
        DualAccum::SecondMin,
        DualAccum::StockProfit,
        DualAccum::PrefixMaxSum,
        DualAccum::PrefixMinSum,
        DualAccum::MaxSubarraySum,
        DualAccum::MinSubarraySum,
        DualAccum::Median,
        DualAccum::GcdAll,
        DualAccum::LcmAll,
        DualAccum::MeanTrunc,
        DualAccum::SumSquares,
        DualAccum::AbsSum,
        DualAccum::MaxAbs,
        DualAccum::MinAbs,
        DualAccum::MinPositive,
        DualAccum::CountNegatives,
        DualAccum::CountEvens,
        DualAccum::SumPositives,
        DualAccum::SumNegatives,
        DualAccum::CountOdds,
        DualAccum::Len,
        DualAccum::IsEmpty,
        DualAccum::AllEqual,
        DualAccum::AnyPositive,
        DualAccum::AnyNegative,
        DualAccum::AnyZero,
        DualAccum::AllPositive,
        DualAccum::AllNegative,
        DualAccum::CountZeros,
        DualAccum::HasDuplicate,
        DualAccum::MaxNegative,
        DualAccum::SumEvenValues,
        DualAccum::SumOddValues,
        DualAccum::AllNonNegative,
        DualAccum::CountNonZeros,
        DualAccum::AlternatingSum,
        DualAccum::ProductPositives,
        DualAccum::MeanAbsTrunc,
        DualAccum::FirstPositive,
        DualAccum::LastPositive,
        DualAccum::FirstNegative,
        DualAccum::LastNegative,
        DualAccum::MaxPositive,
        DualAccum::MinNegative,
        DualAccum::ProductNegatives,
        DualAccum::SumCubes,
        DualAccum::CountGtMean,
        DualAccum::CountLtMean,
        DualAccum::IsPalindrome,
        DualAccum::ProductEvens,
        DualAccum::ProductOdds,
        DualAccum::AllNonPositive,
        DualAccum::DotIndex,
        DualAccum::SumSqDiffMean,
        DualAccum::AnyNonZero,
        DualAccum::XorAll,
        DualAccum::ProductNonZeros,
        DualAccum::OrAll,
        DualAccum::AndAll,
        DualAccum::CountEqMean,
        DualAccum::ProductAbs,
        DualAccum::CountNonNegatives,
        DualAccum::CountPositives,
        DualAccum::MaxEvenValue,
        DualAccum::MaxOddValue,
        DualAccum::MinEvenValue,
        DualAccum::MinOddValue,
        DualAccum::AbsRange,
        DualAccum::ProductNonNegatives,
        DualAccum::SumNonNegatives,
        DualAccum::SumNonPositives,
        DualAccum::CountNonPositives,
        DualAccum::ProductNonPositives,
        DualAccum::SumAbsEvens,
        DualAccum::SumAbsOdds,
        DualAccum::ProductAbsEvens,
        DualAccum::ProductAbsOdds,
        DualAccum::XorAbsAll,
        DualAccum::AndAbsAll,
        DualAccum::OrAbsAll,
        DualAccum::XorAbsEvens,
        DualAccum::XorAbsOdds,
        DualAccum::AndAbsEvens,
        DualAccum::AndAbsOdds,
        DualAccum::OrAbsEvens,
        DualAccum::OrAbsOdds,
        DualAccum::SumSquaresEvens,
        DualAccum::SumSquaresOdds,
        DualAccum::SumCubesEvens,
        DualAccum::SumCubesOdds,
        DualAccum::ProductSquaresEvens,
        DualAccum::ProductSquaresOdds,
        DualAccum::MaxAbsEvens,
        DualAccum::MaxAbsOdds,
        DualAccum::MinAbsEvens,
        DualAccum::MinAbsOdds,
        DualAccum::CountNonZeroEvens,
        DualAccum::CountNonZeroOdds,
        DualAccum::SumNonZeroEvens,
        DualAccum::SumNonZeroOdds,
        DualAccum::ProductNonZeroEvens,
        DualAccum::ProductNonZeroOdds,
        DualAccum::MeanAbsEvensTrunc,
        DualAccum::MeanAbsOddsTrunc,
        DualAccum::GcdAbsEvens,
        DualAccum::GcdAbsOdds,
        DualAccum::LcmAbsEvens,
        DualAccum::LcmAbsOdds,
        DualAccum::ProductCubesEvens,
        DualAccum::ProductCubesOdds,
        DualAccum::SumAbsCubesEvens,
        DualAccum::SumAbsCubesOdds,
        DualAccum::ProductAbsCubesEvens,
        DualAccum::ProductAbsCubesOdds,
        DualAccum::SumAbsSquaresEvens,
        DualAccum::SumAbsSquaresOdds,
        DualAccum::ProductAbsSquaresEvens,
        DualAccum::ProductAbsSquaresOdds,
        DualAccum::MeanAbsSquaresEvensTrunc,
        DualAccum::MeanAbsSquaresOddsTrunc,
        DualAccum::CountPositiveEvens,
        DualAccum::CountPositiveOdds,
        DualAccum::CountNegativeEvens,
        DualAccum::CountNegativeOdds,
        DualAccum::SumPositiveEvens,
        DualAccum::SumPositiveOdds,
        DualAccum::SumNegativeEvens,
        DualAccum::SumNegativeOdds,
        DualAccum::ProductPositiveEvens,
        DualAccum::ProductPositiveOdds,
        DualAccum::ProductNegativeEvens,
        DualAccum::ProductNegativeOdds,
        DualAccum::MaxPositiveEvens,
        DualAccum::MaxPositiveOdds,
        DualAccum::MinPositiveEvens,
        DualAccum::MinPositiveOdds,
        DualAccum::MaxNegativeEvens,
        DualAccum::MaxNegativeOdds,
        DualAccum::MinNegativeEvens,
        DualAccum::MinNegativeOdds,
        DualAccum::MeanPositiveEvensTrunc,
        DualAccum::MeanPositiveOddsTrunc,
        DualAccum::MeanNegativeEvensTrunc,
        DualAccum::MeanNegativeOddsTrunc,
        DualAccum::AllEvenPositive,
        DualAccum::AllOddPositive,
        DualAccum::AllEvenNegative,
        DualAccum::AllOddNegative,
        DualAccum::AnyEvenPositive,
        DualAccum::AnyOddPositive,
        DualAccum::AnyEvenNegative,
        DualAccum::AnyOddNegative,
        DualAccum::AnyEvenNonZero,
        DualAccum::AnyOddNonZero,
        DualAccum::AllEvenNonZero,
        DualAccum::AllOddNonZero,
        DualAccum::AllEvenNonNegative,
        DualAccum::AllOddNonNegative,
        DualAccum::AllEvenNonPositive,
        DualAccum::AllOddNonPositive,
        DualAccum::AnyEvenNonNegative,
        DualAccum::AnyOddNonNegative,
        DualAccum::AnyEvenNonPositive,
        DualAccum::AnyOddNonPositive,
        DualAccum::MaxEvenNonZero,
        DualAccum::MaxOddNonZero,
        DualAccum::MinEvenNonZero,
        DualAccum::MinOddNonZero,
        DualAccum::MeanEvenNonZeroTrunc,
        DualAccum::MeanOddNonZeroTrunc,
        DualAccum::XorEvenNonZero,
        DualAccum::XorOddNonZero,
        DualAccum::OrEvenNonZero,
        DualAccum::OrOddNonZero,
        DualAccum::AndEvenNonZero,
        DualAccum::AndOddNonZero,
        DualAccum::SumAbsEvenNonZero,
        DualAccum::SumAbsOddNonZero,
        DualAccum::ProductAbsEvenNonZero,
        DualAccum::ProductAbsOddNonZero,
        DualAccum::GcdAbsEvenNonZero,
        DualAccum::GcdAbsOddNonZero,
        DualAccum::LcmAbsEvenNonZero,
        DualAccum::LcmAbsOddNonZero,
        DualAccum::MeanAbsEvenNonZeroTrunc,
        DualAccum::MeanAbsOddNonZeroTrunc,
        DualAccum::MaxAbsEvenNonZero,
        DualAccum::MaxAbsOddNonZero,
        DualAccum::MinAbsEvenNonZero,
        DualAccum::MinAbsOddNonZero,
        DualAccum::SumSquaresEvenNonZero,
        DualAccum::SumSquaresOddNonZero,
        DualAccum::ProductSquaresEvenNonZero,
        DualAccum::ProductSquaresOddNonZero,
        DualAccum::SumCubesEvenNonZero,
        DualAccum::SumCubesOddNonZero,
        DualAccum::ProductCubesEvenNonZero,
        DualAccum::ProductCubesOddNonZero,
        DualAccum::SumFourthPowersEvenNonZero,
        DualAccum::SumFourthPowersOddNonZero,
        DualAccum::ProductFourthPowersEvenNonZero,
        DualAccum::ProductFourthPowersOddNonZero,
        DualAccum::MeanFourthPowersEvenNonZeroTrunc,
        DualAccum::MeanFourthPowersOddNonZeroTrunc,
        DualAccum::SumFifthPowersEvenNonZero,
        DualAccum::SumFifthPowersOddNonZero,
        DualAccum::ProductFifthPowersEvenNonZero,
        DualAccum::ProductFifthPowersOddNonZero,
        DualAccum::MeanFifthPowersEvenNonZeroTrunc,
        DualAccum::MeanFifthPowersOddNonZeroTrunc,
        DualAccum::SumSixthPowersEvenNonZero,
        DualAccum::SumSixthPowersOddNonZero,
        DualAccum::ProductSixthPowersEvenNonZero,
        DualAccum::ProductSixthPowersOddNonZero,
        DualAccum::MeanSixthPowersEvenNonZeroTrunc,
        DualAccum::MeanSixthPowersOddNonZeroTrunc,
        DualAccum::SumSeventhPowersEvenNonZero,
        DualAccum::SumSeventhPowersOddNonZero,
    ] {
        let ok = inputs
            .iter()
            .zip(expected.iter())
            .all(|(arr, &y)| dual.eval(arr) == Some(y));
        if !ok {
            continue;
        }
        let code = dual.emit(fn_name);
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: format!("utbus_dual_{}", dual.label()),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }
    for scan in [
        PairwiseScan::MaxAbsDiff,
        PairwiseScan::CountAdjacentDiff,
        PairwiseScan::CountIncreases,
        PairwiseScan::CountDecreases,
        PairwiseScan::StrictlyIncreasing,
        PairwiseScan::StrictlyDecreasing,
        PairwiseScan::NonDecreasing,
        PairwiseScan::NonIncreasing,
        PairwiseScan::LongestPlateau,
        PairwiseScan::SumAbsDiff,
        PairwiseScan::LongestIncreasingRun,
        PairwiseScan::LongestDecreasingRun,
        PairwiseScan::CountAdjacentEq,
        PairwiseScan::MaxIncrease,
        PairwiseScan::MaxDecrease,
        PairwiseScan::LongestNonDecreasingRun,
        PairwiseScan::LongestNonIncreasingRun,
        PairwiseScan::SumIncreases,
        PairwiseScan::SumDecreases,
        PairwiseScan::CountPlateaus,
        PairwiseScan::IsZigZag,
        PairwiseScan::MinIncrease,
        PairwiseScan::MinDecrease,
        PairwiseScan::MeanAbsDiffTrunc,
        PairwiseScan::CountSignChanges,
        PairwiseScan::SumSqDiff,
        PairwiseScan::MeanSqDiffTrunc,
        PairwiseScan::FirstIncreaseIdx,
        PairwiseScan::FirstDecreaseIdx,
        PairwiseScan::LastIncreaseIdx,
        PairwiseScan::LastDecreaseIdx,
    ] {
        let ok = inputs
            .iter()
            .zip(expected.iter())
            .all(|(arr, &y)| scan.eval(arr) == Some(y));
        if !ok {
            continue;
        }
        let code = scan.emit(fn_name);
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: format!("utbus_pairwise_{}", scan.label()),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }
    for idx in [
        IndexScan::SumEvenIndices,
        IndexScan::SumOddIndices,
        IndexScan::ProductEvenIndices,
        IndexScan::ProductOddIndices,
        IndexScan::CountPeaks,
        IndexScan::CountValleys,
        IndexScan::CountDistinct,
        IndexScan::ArgMax,
        IndexScan::ArgMin,
        IndexScan::First,
        IndexScan::Last,
        IndexScan::Mode,
        IndexScan::Middle,
        IndexScan::Second,
        IndexScan::SecondLast,
        IndexScan::MaxEvenIndices,
        IndexScan::MinEvenIndices,
        IndexScan::MaxOddIndices,
        IndexScan::MinOddIndices,
        IndexScan::ArgMaxAbs,
        IndexScan::ArgMinAbs,
        IndexScan::SumAbsEvenIndices,
        IndexScan::SumAbsOddIndices,
        IndexScan::CountEvenIndices,
        IndexScan::CountOddIndices,
        IndexScan::XorEvenIndices,
        IndexScan::XorOddIndices,
        IndexScan::OrEvenIndices,
        IndexScan::OrOddIndices,
        IndexScan::AndEvenIndices,
        IndexScan::AndOddIndices,
        IndexScan::ProductAbsEvenIndices,
        IndexScan::ProductAbsOddIndices,
        IndexScan::SumSquaresEvenIndices,
        IndexScan::SumSquaresOddIndices,
        IndexScan::MeanEvenTrunc,
        IndexScan::MeanOddTrunc,
        IndexScan::CountPositiveEvenIndices,
        IndexScan::CountPositiveOddIndices,
        IndexScan::CountNegativeEvenIndices,
        IndexScan::CountNegativeOddIndices,
        IndexScan::SumPositiveEvenIndices,
        IndexScan::SumPositiveOddIndices,
        IndexScan::SumNegativeEvenIndices,
        IndexScan::SumNegativeOddIndices,
        IndexScan::CountZeroEvenIndices,
        IndexScan::CountZeroOddIndices,
        IndexScan::MaxAbsEvenIndices,
        IndexScan::MaxAbsOddIndices,
        IndexScan::MinAbsEvenIndices,
        IndexScan::MinAbsOddIndices,
        IndexScan::MeanAbsEvenTrunc,
        IndexScan::MeanAbsOddTrunc,
        IndexScan::CountNonZeroEvenIndices,
        IndexScan::CountNonZeroOddIndices,
        IndexScan::ProductNonZeroEvenIndices,
        IndexScan::ProductNonZeroOddIndices,
        IndexScan::SumNonZeroEvenIndices,
        IndexScan::SumNonZeroOddIndices,
        IndexScan::MaxNonZeroEvenIndices,
        IndexScan::MaxNonZeroOddIndices,
        IndexScan::MinNonZeroEvenIndices,
        IndexScan::MinNonZeroOddIndices,
        IndexScan::CountEvenValueEvenIndices,
        IndexScan::CountEvenValueOddIndices,
        IndexScan::CountOddValueEvenIndices,
        IndexScan::CountOddValueOddIndices,
        IndexScan::SumEvenValueEvenIndices,
        IndexScan::SumEvenValueOddIndices,
        IndexScan::SumOddValueEvenIndices,
        IndexScan::SumOddValueOddIndices,
        IndexScan::ProductEvenValueEvenIndices,
        IndexScan::ProductEvenValueOddIndices,
        IndexScan::ProductOddValueEvenIndices,
        IndexScan::ProductOddValueOddIndices,
        IndexScan::SumAbsEvenValueEvenIndices,
        IndexScan::SumAbsEvenValueOddIndices,
        IndexScan::SumAbsOddValueEvenIndices,
        IndexScan::SumAbsOddValueOddIndices,
        IndexScan::OrAbsEvenIndices,
        IndexScan::OrAbsOddIndices,
        IndexScan::AndAbsEvenIndices,
        IndexScan::AndAbsOddIndices,
        IndexScan::XorAbsEvenIndices,
        IndexScan::XorAbsOddIndices,
    ] {
        let ok = inputs
            .iter()
            .zip(expected.iter())
            .all(|(arr, &y)| idx.eval(arr) == Some(y));
        if !ok {
            continue;
        }
        let code = idx.emit(fn_name);
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: format!("utbus_index_{}", idx.label()),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }
    None
}

/// Index-gated scans (native A5 parity slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndexScan {
    SumEvenIndices,
    SumOddIndices,
    /// Product of elements at even indices (empty → 1).
    ProductEvenIndices,
    /// Product of elements at odd indices (empty → 1).
    ProductOddIndices,
    /// Local peaks: `arr[i] > arr[i-1] && arr[i] > arr[i+1]` for interior i.
    CountPeaks,
    /// Local valleys: `arr[i] < arr[i-1] && arr[i] < arr[i+1]` for interior i.
    CountValleys,
    /// Number of unique values (O(n²) first-occurrence scan).
    CountDistinct,
    /// Index of the first maximum element (empty → None).
    ArgMax,
    /// Index of the first minimum element (empty → None).
    ArgMin,
    /// First element (empty → None).
    First,
    /// Last element (empty → None).
    Last,
    /// Most frequent value (first on a frequency tie; empty → None).
    Mode,
    /// Middle element `arr[len/2]` (empty → None).
    Middle,
    /// Second element `arr[1]` (len < 2 → None).
    Second,
    /// Second-to-last element (len < 2 → None).
    SecondLast,
    /// Max among even-index elements (empty → None).
    MaxEvenIndices,
    /// Min among even-index elements (empty → None).
    MinEvenIndices,
    /// Max among odd-index elements (no odd indices → None).
    MaxOddIndices,
    /// Min among odd-index elements (no odd indices → None).
    MinOddIndices,
    /// Index of the first maximum-absolute-value element (empty → None).
    ArgMaxAbs,
    /// Index of the first minimum-absolute-value element (empty → None).
    ArgMinAbs,
    /// Sum of absolute values at even indices.
    SumAbsEvenIndices,
    /// Sum of absolute values at odd indices.
    SumAbsOddIndices,
    /// Count of even indices (= ceil(len/2)).
    CountEvenIndices,
    /// Count of odd indices (= floor(len/2)).
    CountOddIndices,
    /// XOR of elements at even indices (empty → 0).
    XorEvenIndices,
    /// XOR of elements at odd indices (no odd → 0).
    XorOddIndices,
    /// Bitwise OR of elements at even indices (empty → 0).
    OrEvenIndices,
    /// Bitwise OR of elements at odd indices (no odd → 0).
    OrOddIndices,
    /// Bitwise AND of elements at even indices (empty → -1).
    AndEvenIndices,
    /// Bitwise AND of elements at odd indices (no odd → -1).
    AndOddIndices,
    /// Product of absolute values at even indices (empty → 1).
    ProductAbsEvenIndices,
    /// Product of absolute values at odd indices (no odd → 1).
    ProductAbsOddIndices,
    /// Sum of squares at even indices.
    SumSquaresEvenIndices,
    /// Sum of squares at odd indices.
    SumSquaresOddIndices,
    /// Truncating mean of even-index elements (empty → 0).
    MeanEvenTrunc,
    /// Truncating mean of odd-index elements (no odd → 0).
    MeanOddTrunc,
    /// Count of positive values at even indices.
    CountPositiveEvenIndices,
    /// Count of positive values at odd indices.
    CountPositiveOddIndices,
    /// Count of negative values at even indices.
    CountNegativeEvenIndices,
    /// Count of negative values at odd indices.
    CountNegativeOddIndices,
    /// Sum of positive values at even indices.
    SumPositiveEvenIndices,
    /// Sum of positive values at odd indices.
    SumPositiveOddIndices,
    /// Sum of negative values at even indices.
    SumNegativeEvenIndices,
    /// Sum of negative values at odd indices.
    SumNegativeOddIndices,
    /// Count of zeros at even indices.
    CountZeroEvenIndices,
    /// Count of zeros at odd indices.
    CountZeroOddIndices,
    /// Max absolute value at even indices (empty → 0).
    MaxAbsEvenIndices,
    /// Max absolute value at odd indices (no odd → 0).
    MaxAbsOddIndices,
    /// Min absolute value at even indices (empty → 0).
    MinAbsEvenIndices,
    /// Min absolute value at odd indices (no odd → 0).
    MinAbsOddIndices,
    /// Truncating mean of |v| at even indices (empty → 0).
    MeanAbsEvenTrunc,
    /// Truncating mean of |v| at odd indices (no odd → 0).
    MeanAbsOddTrunc,
    /// Count of nonzeros at even indices.
    CountNonZeroEvenIndices,
    /// Count of nonzeros at odd indices.
    CountNonZeroOddIndices,
    /// Product of nonzeros at even indices (none → 1).
    ProductNonZeroEvenIndices,
    /// Product of nonzeros at odd indices (none → 1).
    ProductNonZeroOddIndices,
    /// Sum of nonzeros at even indices.
    SumNonZeroEvenIndices,
    /// Sum of nonzeros at odd indices.
    SumNonZeroOddIndices,
    /// Max among nonzeros at even indices (none → 0).
    MaxNonZeroEvenIndices,
    /// Max among nonzeros at odd indices (none → 0).
    MaxNonZeroOddIndices,
    /// Min among nonzeros at even indices (none → 0).
    MinNonZeroEvenIndices,
    /// Min among nonzeros at odd indices (none → 0).
    MinNonZeroOddIndices,
    /// Count of even-valued elements at even indices.
    CountEvenValueEvenIndices,
    /// Count of even-valued elements at odd indices.
    CountEvenValueOddIndices,
    /// Count of odd-valued elements at even indices.
    CountOddValueEvenIndices,
    /// Count of odd-valued elements at odd indices.
    CountOddValueOddIndices,
    /// Sum of even-valued elements at even indices.
    SumEvenValueEvenIndices,
    /// Sum of even-valued elements at odd indices.
    SumEvenValueOddIndices,
    /// Sum of odd-valued elements at even indices.
    SumOddValueEvenIndices,
    /// Sum of odd-valued elements at odd indices.
    SumOddValueOddIndices,
    /// Product of even-valued elements at even indices (none → 1).
    ProductEvenValueEvenIndices,
    /// Product of even-valued elements at odd indices (none → 1).
    ProductEvenValueOddIndices,
    /// Product of odd-valued elements at even indices (none → 1).
    ProductOddValueEvenIndices,
    /// Product of odd-valued elements at odd indices (none → 1).
    ProductOddValueOddIndices,
    /// Sum of |v| for even-valued elements at even indices.
    SumAbsEvenValueEvenIndices,
    /// Sum of |v| for even-valued elements at odd indices.
    SumAbsEvenValueOddIndices,
    /// Sum of |v| for odd-valued elements at even indices.
    SumAbsOddValueEvenIndices,
    /// Sum of |v| for odd-valued elements at odd indices.
    SumAbsOddValueOddIndices,
    /// Bitwise OR of |v| at even indices (empty → 0).
    OrAbsEvenIndices,
    /// Bitwise OR of |v| at odd indices (no odd → 0).
    OrAbsOddIndices,
    /// Bitwise AND of |v| at even indices (empty → -1).
    AndAbsEvenIndices,
    /// Bitwise AND of |v| at odd indices (no odd → -1).
    AndAbsOddIndices,
    /// XOR of |v| at even indices (empty → 0).
    XorAbsEvenIndices,
    /// XOR of |v| at odd indices (no odd → 0).
    XorAbsOddIndices,
}

impl IndexScan {
    fn label(self) -> &'static str {
        match self {
            IndexScan::SumEvenIndices => "sum_even_indices",
            IndexScan::SumOddIndices => "sum_odd_indices",
            IndexScan::ProductEvenIndices => "product_even_indices",
            IndexScan::ProductOddIndices => "product_odd_indices",
            IndexScan::CountPeaks => "count_peaks",
            IndexScan::CountValleys => "count_valleys",
            IndexScan::CountDistinct => "count_distinct",
            IndexScan::ArgMax => "argmax",
            IndexScan::ArgMin => "argmin",
            IndexScan::First => "first",
            IndexScan::Last => "last",
            IndexScan::Mode => "mode",
            IndexScan::Middle => "middle",
            IndexScan::Second => "second",
            IndexScan::SecondLast => "second_last",
            IndexScan::MaxEvenIndices => "max_even_indices",
            IndexScan::MinEvenIndices => "min_even_indices",
            IndexScan::MaxOddIndices => "max_odd_indices",
            IndexScan::MinOddIndices => "min_odd_indices",
            IndexScan::ArgMaxAbs => "argmax_abs",
            IndexScan::ArgMinAbs => "argmin_abs",
            IndexScan::SumAbsEvenIndices => "sum_abs_even_indices",
            IndexScan::SumAbsOddIndices => "sum_abs_odd_indices",
            IndexScan::CountEvenIndices => "count_even_indices",
            IndexScan::CountOddIndices => "count_odd_indices",
            IndexScan::XorEvenIndices => "xor_even_indices",
            IndexScan::XorOddIndices => "xor_odd_indices",
            IndexScan::OrEvenIndices => "or_even_indices",
            IndexScan::OrOddIndices => "or_odd_indices",
            IndexScan::AndEvenIndices => "and_even_indices",
            IndexScan::AndOddIndices => "and_odd_indices",
            IndexScan::ProductAbsEvenIndices => "product_abs_even_indices",
            IndexScan::ProductAbsOddIndices => "product_abs_odd_indices",
            IndexScan::SumSquaresEvenIndices => "sum_squares_even_indices",
            IndexScan::SumSquaresOddIndices => "sum_squares_odd_indices",
            IndexScan::MeanEvenTrunc => "mean_even_trunc",
            IndexScan::MeanOddTrunc => "mean_odd_trunc",
            IndexScan::CountPositiveEvenIndices => "count_positive_even_indices",
            IndexScan::CountPositiveOddIndices => "count_positive_odd_indices",
            IndexScan::CountNegativeEvenIndices => "count_negative_even_indices",
            IndexScan::CountNegativeOddIndices => "count_negative_odd_indices",
            IndexScan::SumPositiveEvenIndices => "sum_positive_even_indices",
            IndexScan::SumPositiveOddIndices => "sum_positive_odd_indices",
            IndexScan::SumNegativeEvenIndices => "sum_negative_even_indices",
            IndexScan::SumNegativeOddIndices => "sum_negative_odd_indices",
            IndexScan::CountZeroEvenIndices => "count_zero_even_indices",
            IndexScan::CountZeroOddIndices => "count_zero_odd_indices",
            IndexScan::MaxAbsEvenIndices => "max_abs_even_indices",
            IndexScan::MaxAbsOddIndices => "max_abs_odd_indices",
            IndexScan::MinAbsEvenIndices => "min_abs_even_indices",
            IndexScan::MinAbsOddIndices => "min_abs_odd_indices",
            IndexScan::MeanAbsEvenTrunc => "mean_abs_even_trunc",
            IndexScan::MeanAbsOddTrunc => "mean_abs_odd_trunc",
            IndexScan::CountNonZeroEvenIndices => "count_nonzero_even_indices",
            IndexScan::CountNonZeroOddIndices => "count_nonzero_odd_indices",
            IndexScan::ProductNonZeroEvenIndices => "product_nonzero_even_indices",
            IndexScan::ProductNonZeroOddIndices => "product_nonzero_odd_indices",
            IndexScan::SumNonZeroEvenIndices => "sum_nonzero_even_indices",
            IndexScan::SumNonZeroOddIndices => "sum_nonzero_odd_indices",
            IndexScan::MaxNonZeroEvenIndices => "max_nonzero_even_indices",
            IndexScan::MaxNonZeroOddIndices => "max_nonzero_odd_indices",
            IndexScan::MinNonZeroEvenIndices => "min_nonzero_even_indices",
            IndexScan::MinNonZeroOddIndices => "min_nonzero_odd_indices",
            IndexScan::CountEvenValueEvenIndices => "count_even_value_even_indices",
            IndexScan::CountEvenValueOddIndices => "count_even_value_odd_indices",
            IndexScan::CountOddValueEvenIndices => "count_odd_value_even_indices",
            IndexScan::CountOddValueOddIndices => "count_odd_value_odd_indices",
            IndexScan::SumEvenValueEvenIndices => "sum_even_value_even_indices",
            IndexScan::SumEvenValueOddIndices => "sum_even_value_odd_indices",
            IndexScan::SumOddValueEvenIndices => "sum_odd_value_even_indices",
            IndexScan::SumOddValueOddIndices => "sum_odd_value_odd_indices",
            IndexScan::ProductEvenValueEvenIndices => "product_even_value_even_indices",
            IndexScan::ProductEvenValueOddIndices => "product_even_value_odd_indices",
            IndexScan::ProductOddValueEvenIndices => "product_odd_value_even_indices",
            IndexScan::ProductOddValueOddIndices => "product_odd_value_odd_indices",
            IndexScan::SumAbsEvenValueEvenIndices => "sum_abs_even_value_even_indices",
            IndexScan::SumAbsEvenValueOddIndices => "sum_abs_even_value_odd_indices",
            IndexScan::SumAbsOddValueEvenIndices => "sum_abs_odd_value_even_indices",
            IndexScan::SumAbsOddValueOddIndices => "sum_abs_odd_value_odd_indices",
            IndexScan::OrAbsEvenIndices => "or_abs_even_indices",
            IndexScan::OrAbsOddIndices => "or_abs_odd_indices",
            IndexScan::AndAbsEvenIndices => "and_abs_even_indices",
            IndexScan::AndAbsOddIndices => "and_abs_odd_indices",
            IndexScan::XorAbsEvenIndices => "xor_abs_even_indices",
            IndexScan::XorAbsOddIndices => "xor_abs_odd_indices",
        }
    }

    fn eval(self, arr: &[i64]) -> Option<i64> {
        match self {
            IndexScan::SumEvenIndices => {
                Some(
                    arr.iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 0)
                        .map(|(_, &v)| v)
                        .fold(0i64, i64::saturating_add),
                )
            }
            IndexScan::SumOddIndices => {
                Some(
                    arr.iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 1)
                        .map(|(_, &v)| v)
                        .fold(0i64, i64::saturating_add),
                )
            }
            IndexScan::ProductEvenIndices => {
                Some(
                    arr.iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 0)
                        .map(|(_, &v)| v)
                        .fold(1i64, i64::saturating_mul),
                )
            }
            IndexScan::ProductOddIndices => {
                Some(
                    arr.iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 1)
                        .map(|(_, &v)| v)
                        .fold(1i64, i64::saturating_mul),
                )
            }
            IndexScan::CountPeaks => {
                if arr.len() < 3 {
                    return Some(0);
                }
                let mut count = 0i64;
                for i in 1..arr.len() - 1 {
                    if arr[i] > arr[i - 1] && arr[i] > arr[i + 1] {
                        count += 1;
                    }
                }
                Some(count)
            }
            IndexScan::CountValleys => {
                if arr.len() < 3 {
                    return Some(0);
                }
                let mut count = 0i64;
                for i in 1..arr.len() - 1 {
                    if arr[i] < arr[i - 1] && arr[i] < arr[i + 1] {
                        count += 1;
                    }
                }
                Some(count)
            }
            IndexScan::CountDistinct => {
                let mut count = 0i64;
                for i in 0..arr.len() {
                    let mut seen = false;
                    for &v in &arr[..i] {
                        if v == arr[i] {
                            seen = true;
                            break;
                        }
                    }
                    if !seen {
                        count += 1;
                    }
                }
                Some(count)
            }
            IndexScan::ArgMax => {
                if arr.is_empty() {
                    return None;
                }
                let mut best_i = 0usize;
                for i in 1..arr.len() {
                    if arr[i] > arr[best_i] {
                        best_i = i;
                    }
                }
                Some(best_i as i64)
            }
            IndexScan::ArgMin => {
                if arr.is_empty() {
                    return None;
                }
                let mut best_i = 0usize;
                for i in 1..arr.len() {
                    if arr[i] < arr[best_i] {
                        best_i = i;
                    }
                }
                Some(best_i as i64)
            }
            IndexScan::First => arr.first().copied(),
            IndexScan::Last => arr.last().copied(),
            IndexScan::Mode => {
                if arr.is_empty() {
                    return None;
                }
                let mut best_val = arr[0];
                let mut best_count = 1i64;
                for i in 0..arr.len() {
                    let mut count = 0i64;
                    for &v in arr {
                        if v == arr[i] {
                            count += 1;
                        }
                    }
                    if count > best_count {
                        best_count = count;
                        best_val = arr[i];
                    }
                }
                Some(best_val)
            }
            IndexScan::Middle => {
                if arr.is_empty() {
                    None
                } else {
                    Some(arr[arr.len() / 2])
                }
            }
            IndexScan::Second => {
                if arr.len() < 2 {
                    None
                } else {
                    Some(arr[1])
                }
            }
            IndexScan::SecondLast => {
                if arr.len() < 2 {
                    None
                } else {
                    Some(arr[arr.len() - 2])
                }
            }
            IndexScan::MaxEvenIndices => {
                let mut best: Option<i64> = None;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 0 {
                        best = Some(match best {
                            Some(b) if b >= v => b,
                            _ => v,
                        });
                    }
                }
                best
            }
            IndexScan::MinEvenIndices => {
                let mut best: Option<i64> = None;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 0 {
                        best = Some(match best {
                            Some(b) if b <= v => b,
                            _ => v,
                        });
                    }
                }
                best
            }
            IndexScan::MaxOddIndices => {
                let mut best: Option<i64> = None;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 1 {
                        best = Some(match best {
                            Some(b) if b >= v => b,
                            _ => v,
                        });
                    }
                }
                best
            }
            IndexScan::MinOddIndices => {
                let mut best: Option<i64> = None;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 1 {
                        best = Some(match best {
                            Some(b) if b <= v => b,
                            _ => v,
                        });
                    }
                }
                best
            }
            IndexScan::ArgMaxAbs => {
                if arr.is_empty() {
                    return None;
                }
                let mut best_i = 0usize;
                let mut best_abs = arr[0].abs();
                for i in 1..arr.len() {
                    let a = arr[i].abs();
                    if a > best_abs {
                        best_abs = a;
                        best_i = i;
                    }
                }
                Some(best_i as i64)
            }
            IndexScan::ArgMinAbs => {
                if arr.is_empty() {
                    return None;
                }
                let mut best_i = 0usize;
                let mut best_abs = arr[0].abs();
                for i in 1..arr.len() {
                    let a = arr[i].abs();
                    if a < best_abs {
                        best_abs = a;
                        best_i = i;
                    }
                }
                Some(best_i as i64)
            }
            IndexScan::SumAbsEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .map(|(_, &v)| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumAbsOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .map(|(_, &v)| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::CountEvenIndices => Some(((arr.len() + 1) / 2) as i64),
            IndexScan::CountOddIndices => Some((arr.len() / 2) as i64),
            IndexScan::XorEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .fold(0i64, |acc, (_, &v)| acc ^ v),
            ),
            IndexScan::XorOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .fold(0i64, |acc, (_, &v)| acc ^ v),
            ),
            IndexScan::OrEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .fold(0i64, |acc, (_, &v)| acc | v),
            ),
            IndexScan::OrOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .fold(0i64, |acc, (_, &v)| acc | v),
            ),
            IndexScan::AndEvenIndices => {
                let mut it = arr.iter().enumerate().filter(|(i, _)| i % 2 == 0);
                match it.next() {
                    None => Some(-1),
                    Some((_, &v0)) => Some(it.fold(v0, |acc, (_, &v)| acc & v)),
                }
            }
            IndexScan::AndOddIndices => {
                let mut it = arr.iter().enumerate().filter(|(i, _)| i % 2 == 1);
                match it.next() {
                    None => Some(-1),
                    Some((_, &v0)) => Some(it.fold(v0, |acc, (_, &v)| acc & v)),
                }
            }
            IndexScan::ProductAbsEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .map(|(_, &v)| v.abs())
                    .fold(1i64, i64::saturating_mul),
            ),
            IndexScan::ProductAbsOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .map(|(_, &v)| v.abs())
                    .fold(1i64, i64::saturating_mul),
            ),
            IndexScan::SumSquaresEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .map(|(_, &v)| v.saturating_mul(v))
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumSquaresOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .map(|(_, &v)| v.saturating_mul(v))
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::MeanEvenTrunc => {
                let vals: Vec<i64> = arr
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .map(|(_, &v)| v)
                    .collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    let sum = vals.iter().copied().fold(0i64, i64::saturating_add);
                    Some(sum / (vals.len() as i64))
                }
            }
            IndexScan::MeanOddTrunc => {
                let vals: Vec<i64> = arr
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .map(|(_, &v)| v)
                    .collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    let sum = vals.iter().copied().fold(0i64, i64::saturating_add);
                    Some(sum / (vals.len() as i64))
                }
            }
            IndexScan::CountPositiveEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v > 0)
                    .count() as i64,
            ),
            IndexScan::CountPositiveOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v > 0)
                    .count() as i64,
            ),
            IndexScan::CountNegativeEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v < 0)
                    .count() as i64,
            ),
            IndexScan::CountNegativeOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v < 0)
                    .count() as i64,
            ),
            IndexScan::SumPositiveEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v > 0)
                    .map(|(_, &v)| v)
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumPositiveOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v > 0)
                    .map(|(_, &v)| v)
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumNegativeEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v < 0)
                    .map(|(_, &v)| v)
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumNegativeOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v < 0)
                    .map(|(_, &v)| v)
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::CountZeroEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v == 0)
                    .count() as i64,
            ),
            IndexScan::CountZeroOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v == 0)
                    .count() as i64,
            ),
            IndexScan::MaxAbsEvenIndices => {
                let mut best = 0i64;
                let mut found = false;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 0 {
                        let a = v.abs();
                        if !found || a > best {
                            best = a;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            IndexScan::MaxAbsOddIndices => {
                let mut best = 0i64;
                let mut found = false;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 1 {
                        let a = v.abs();
                        if !found || a > best {
                            best = a;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            IndexScan::MinAbsEvenIndices => {
                let mut best = 0i64;
                let mut found = false;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 0 {
                        let a = v.abs();
                        if !found || a < best {
                            best = a;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            IndexScan::MinAbsOddIndices => {
                let mut best = 0i64;
                let mut found = false;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 1 {
                        let a = v.abs();
                        if !found || a < best {
                            best = a;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            IndexScan::MeanAbsEvenTrunc => {
                let vals: Vec<i64> = arr
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .map(|(_, &v)| v.abs())
                    .collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    let sum = vals.iter().copied().fold(0i64, i64::saturating_add);
                    Some(sum / (vals.len() as i64))
                }
            }
            IndexScan::MeanAbsOddTrunc => {
                let vals: Vec<i64> = arr
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .map(|(_, &v)| v.abs())
                    .collect();
                if vals.is_empty() {
                    Some(0)
                } else {
                    let sum = vals.iter().copied().fold(0i64, i64::saturating_add);
                    Some(sum / (vals.len() as i64))
                }
            }
            IndexScan::CountNonZeroEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v != 0)
                    .count() as i64,
            ),
            IndexScan::CountNonZeroOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v != 0)
                    .count() as i64,
            ),
            IndexScan::ProductNonZeroEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v != 0)
                    .map(|(_, &v)| v)
                    .fold(1i64, i64::saturating_mul),
            ),
            IndexScan::ProductNonZeroOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v != 0)
                    .map(|(_, &v)| v)
                    .fold(1i64, i64::saturating_mul),
            ),
            IndexScan::SumNonZeroEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v != 0)
                    .map(|(_, &v)| v)
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumNonZeroOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v != 0)
                    .map(|(_, &v)| v)
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::MaxNonZeroEvenIndices => {
                let mut best = 0i64;
                let mut found = false;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 0 && v != 0 {
                        if !found || v > best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            IndexScan::MaxNonZeroOddIndices => {
                let mut best = 0i64;
                let mut found = false;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 1 && v != 0 {
                        if !found || v > best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            IndexScan::MinNonZeroEvenIndices => {
                let mut best = 0i64;
                let mut found = false;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 0 && v != 0 {
                        if !found || v < best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            IndexScan::MinNonZeroOddIndices => {
                let mut best = 0i64;
                let mut found = false;
                for (i, &v) in arr.iter().enumerate() {
                    if i % 2 == 1 && v != 0 {
                        if !found || v < best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            IndexScan::CountEvenValueEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v % 2 == 0)
                    .count() as i64,
            ),
            IndexScan::CountEvenValueOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v % 2 == 0)
                    .count() as i64,
            ),
            IndexScan::CountOddValueEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v % 2 != 0)
                    .count() as i64,
            ),
            IndexScan::CountOddValueOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v % 2 != 0)
                    .count() as i64,
            ),
            IndexScan::SumEvenValueEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v % 2 == 0)
                    .map(|(_, &v)| v)
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumEvenValueOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v % 2 == 0)
                    .map(|(_, &v)| v)
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumOddValueEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v % 2 != 0)
                    .map(|(_, &v)| v)
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumOddValueOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v % 2 != 0)
                    .map(|(_, &v)| v)
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::ProductEvenValueEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v % 2 == 0)
                    .map(|(_, &v)| v)
                    .fold(1i64, i64::saturating_mul),
            ),
            IndexScan::ProductEvenValueOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v % 2 == 0)
                    .map(|(_, &v)| v)
                    .fold(1i64, i64::saturating_mul),
            ),
            IndexScan::ProductOddValueEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v % 2 != 0)
                    .map(|(_, &v)| v)
                    .fold(1i64, i64::saturating_mul),
            ),
            IndexScan::ProductOddValueOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v % 2 != 0)
                    .map(|(_, &v)| v)
                    .fold(1i64, i64::saturating_mul),
            ),
            IndexScan::SumAbsEvenValueEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v % 2 == 0)
                    .map(|(_, &v)| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumAbsEvenValueOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v % 2 == 0)
                    .map(|(_, &v)| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumAbsOddValueEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 0 && v % 2 != 0)
                    .map(|(_, &v)| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::SumAbsOddValueOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, &v)| i % 2 == 1 && v % 2 != 0)
                    .map(|(_, &v)| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            IndexScan::OrAbsEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .map(|(_, &v)| v.abs())
                    .fold(0i64, |a, b| a | b),
            ),
            IndexScan::OrAbsOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .map(|(_, &v)| v.abs())
                    .fold(0i64, |a, b| a | b),
            ),
            IndexScan::AndAbsEvenIndices => {
                let vals: Vec<i64> = arr
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .map(|(_, &v)| v.abs())
                    .collect();
                if vals.is_empty() {
                    Some(-1)
                } else {
                    Some(vals.into_iter().fold(-1i64, |a, b| a & b))
                }
            }
            IndexScan::AndAbsOddIndices => {
                let vals: Vec<i64> = arr
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .map(|(_, &v)| v.abs())
                    .collect();
                if vals.is_empty() {
                    Some(-1)
                } else {
                    Some(vals.into_iter().fold(-1i64, |a, b| a & b))
                }
            }
            IndexScan::XorAbsEvenIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .map(|(_, &v)| v.abs())
                    .fold(0i64, |a, b| a ^ b),
            ),
            IndexScan::XorAbsOddIndices => Some(
                arr.iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .map(|(_, &v)| v.abs())
                    .fold(0i64, |a, b| a ^ b),
            ),
        }
    }

    fn emit(self, fn_name: &str) -> String {
        match self {
            IndexScan::SumEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        total = total + arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        total = total + arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::ProductEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        total = total * arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::ProductOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        total = total * arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::CountPeaks => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i + 1 < arr.len {{\n\
        if arr[i] > arr[i - 1] {{\n\
            if arr[i] > arr[i + 1] {{\n\
                count = count + 1;\n\
            }}\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::CountValleys => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i + 1 < arr.len {{\n\
        if arr[i] < arr[i - 1] {{\n\
            if arr[i] < arr[i + 1] {{\n\
                count = count + 1;\n\
            }}\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::CountDistinct => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        seen: i64 = 0;\n\
        j: i64 = 0;\n\
        while j < i {{\n\
            if arr[j] == arr[i] {{\n\
                seen = 1;\n\
            }}\n\
            j = j + 1;\n\
        }}\n\
        if seen == 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::ArgMax => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best_i: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] > arr[best_i] {{\n\
            best_i = i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best_i;\n\
}}\n"
            ),
            IndexScan::ArgMin => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best_i: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] < arr[best_i] {{\n\
            best_i = i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best_i;\n\
}}\n"
            ),
            IndexScan::First => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    return arr[0];\n\
}}\n"
            ),
            IndexScan::Last => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    return arr[arr.len - 1];\n\
}}\n"
            ),
            IndexScan::Mode => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best_val: i64 = arr[0];\n\
    best_count: i64 = 1;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        count: i64 = 0;\n\
        j: i64 = 0;\n\
        while j < arr.len {{\n\
            if arr[j] == arr[i] {{\n\
                count = count + 1;\n\
            }}\n\
            j = j + 1;\n\
        }}\n\
        if count > best_count {{\n\
            best_count = count;\n\
            best_val = arr[i];\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best_val;\n\
}}\n"
            ),
            IndexScan::Middle => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    return arr[arr.len / 2];\n\
}}\n"
            ),
            IndexScan::Second => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    return arr[1];\n\
}}\n"
            ),
            IndexScan::SecondLast => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    return arr[arr.len - 2];\n\
}}\n"
            ),
            IndexScan::MaxEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = arr[0];\n\
    i: i64 = 2;\n\
    while i < arr.len {{\n\
        if arr[i] > best {{ best = arr[i]; }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::MinEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = arr[0];\n\
    i: i64 = 2;\n\
    while i < arr.len {{\n\
        if arr[i] < best {{ best = arr[i]; }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::MaxOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = arr[1];\n\
    i: i64 = 3;\n\
    while i < arr.len {{\n\
        if arr[i] > best {{ best = arr[i]; }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::MinOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = arr[1];\n\
    i: i64 = 3;\n\
    while i < arr.len {{\n\
        if arr[i] < best {{ best = arr[i]; }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::ArgMaxAbs => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best_i: i64 = 0;\n\
    best_abs: i64 = arr[0];\n\
    if best_abs < 0 {{ best_abs = 0 - best_abs; }}\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > best_abs {{\n\
            best_abs = a;\n\
            best_i = i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best_i;\n\
}}\n"
            ),
            IndexScan::ArgMinAbs => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best_i: i64 = 0;\n\
    best_abs: i64 = arr[0];\n\
    if best_abs < 0 {{ best_abs = 0 - best_abs; }}\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < best_abs {{\n\
            best_abs = a;\n\
            best_i = i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return best_i;\n\
}}\n"
            ),
            IndexScan::SumAbsEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total + a;\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumAbsOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total + a;\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::CountEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    return (arr.len + 1) / 2;\n\
}}\n"
            ),
            IndexScan::CountOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    return arr.len / 2;\n\
}}\n"
            ),
            IndexScan::XorEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        total = total ^ arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::XorOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        total = total ^ arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::OrEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        total = total | arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::OrOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        total = total | arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::AndEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len == 0 {{ return 0 - 1; }}\n\
    total: i64 = arr[0];\n\
    i: i64 = 2;\n\
    while i < arr.len {{\n\
        total = total & arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::AndOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len < 2 {{ return 0 - 1; }}\n\
    total: i64 = arr[1];\n\
    i: i64 = 3;\n\
    while i < arr.len {{\n\
        total = total & arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::ProductAbsEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total * a;\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::ProductAbsOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total * a;\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumSquaresEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        total = total + arr[i] * arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumSquaresOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        total = total + arr[i] * arr[i];\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::MeanEvenTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len == 0 {{ return 0; }}\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        total = total + arr[i];\n\
        count = count + 1;\n\
        i = i + 2;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            IndexScan::MeanOddTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len < 2 {{ return 0; }}\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        total = total + arr[i];\n\
        count = count + 1;\n\
        i = i + 2;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            IndexScan::CountPositiveEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] > 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::CountPositiveOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] > 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::CountNegativeEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] < 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::CountNegativeOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] < 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::SumPositiveEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] > 0 {{\n\
            total = total + arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumPositiveOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] > 0 {{\n\
            total = total + arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumNegativeEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] < 0 {{\n\
            total = total + arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumNegativeOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] < 0 {{\n\
            total = total + arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::CountZeroEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] == 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::CountZeroOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] == 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::MaxAbsEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if found == 0 {{\n\
            best = a;\n\
            found = 1;\n\
        }} else {{\n\
            if a > best {{ best = a; }}\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::MaxAbsOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if found == 0 {{\n\
            best = a;\n\
            found = 1;\n\
        }} else {{\n\
            if a > best {{ best = a; }}\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::MinAbsEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if found == 0 {{\n\
            best = a;\n\
            found = 1;\n\
        }} else {{\n\
            if a < best {{ best = a; }}\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::MinAbsOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if found == 0 {{\n\
            best = a;\n\
            found = 1;\n\
        }} else {{\n\
            if a < best {{ best = a; }}\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::MeanAbsEvenTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len == 0 {{ return 0; }}\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total + a;\n\
        count = count + 1;\n\
        i = i + 2;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            IndexScan::MeanAbsOddTrunc => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len < 2 {{ return 0; }}\n\
    total: i64 = 0;\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total + a;\n\
        count = count + 1;\n\
        i = i + 2;\n\
    }}\n\
    return total / count;\n\
}}\n"
            ),
            IndexScan::CountNonZeroEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] != 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::CountNonZeroOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] != 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::ProductNonZeroEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] != 0 {{\n\
            total = total * arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::ProductNonZeroOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] != 0 {{\n\
            total = total * arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumNonZeroEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] != 0 {{\n\
            total = total + arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumNonZeroOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] != 0 {{\n\
            total = total + arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::MaxNonZeroEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] != 0 {{\n\
            if found == 0 {{\n\
                best = arr[i];\n\
                found = 1;\n\
            }} else {{\n\
                if arr[i] > best {{ best = arr[i]; }}\n\
            }}\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::MaxNonZeroOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] != 0 {{\n\
            if found == 0 {{\n\
                best = arr[i];\n\
                found = 1;\n\
            }} else {{\n\
                if arr[i] > best {{ best = arr[i]; }}\n\
            }}\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::MinNonZeroEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] != 0 {{\n\
            if found == 0 {{\n\
                best = arr[i];\n\
                found = 1;\n\
            }} else {{\n\
                if arr[i] < best {{ best = arr[i]; }}\n\
            }}\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::MinNonZeroOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] != 0 {{\n\
            if found == 0 {{\n\
                best = arr[i];\n\
                found = 1;\n\
            }} else {{\n\
                if arr[i] < best {{ best = arr[i]; }}\n\
            }}\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            IndexScan::CountEvenValueEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 == 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::CountEvenValueOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 == 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::CountOddValueEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 != 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::CountOddValueOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    count: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 != 0 {{\n\
            count = count + 1;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            IndexScan::SumEvenValueEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 == 0 {{\n\
            total = total + arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumEvenValueOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 == 0 {{\n\
            total = total + arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumOddValueEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 != 0 {{\n\
            total = total + arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumOddValueOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 != 0 {{\n\
            total = total + arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::ProductEvenValueEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 == 0 {{\n\
            total = total * arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::ProductEvenValueOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 == 0 {{\n\
            total = total * arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::ProductOddValueEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 != 0 {{\n\
            total = total * arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::ProductOddValueOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 != 0 {{\n\
            total = total * arr[i];\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumAbsEvenValueEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 == 0 {{\n\
            a: i64 = arr[i];\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumAbsEvenValueOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 == 0 {{\n\
            a: i64 = arr[i];\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumAbsOddValueEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 != 0 {{\n\
            a: i64 = arr[i];\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::SumAbsOddValueOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        if arr[i] % 2 != 0 {{\n\
            a: i64 = arr[i];\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::OrAbsEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total | a;\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::OrAbsOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total | a;\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::AndAbsEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len == 0 {{ return 0 - 1; }}\n\
    total: i64 = 0 - 1;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total & a;\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::AndAbsOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    if arr.len < 2 {{ return 0 - 1; }}\n\
    total: i64 = 0 - 1;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total & a;\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::XorAbsEvenIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total ^ a;\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            IndexScan::XorAbsOddIndices => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n\
    total: i64 = 0;\n\
    i: i64 = 1;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        total = total ^ a;\n\
        i = i + 2;\n\
    }}\n\
    return total;\n\
}}\n"
            ),

        }
    }
}

/// Closed (arr, k)→i64 families that are not filter→map→reduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KClosed {
    /// 1-indexed: sort ascending, return `arr[k-1]`.
    KthSmallest,
    /// 1-indexed: sort ascending, return `arr[len-k]`.
    KthLargest,
    /// First index where `arr[i] == k`, else -1.
    FirstIndexOf,
    /// Last index where `arr[i] == k`, else -1.
    LastIndexOf,
    /// 1-indexed from end: `arr[len - k]`.
    KthFromEnd,
    /// 0-indexed element at `k` (out of bounds → None).
    ElementAt,
    /// Count of elements equal to `k`.
    CountEqK,
    /// Sum of elements greater than `k`.
    SumGtK,
    /// Count of elements greater than `k`.
    CountGtK,
    /// Sum of elements less than `k`.
    SumLtK,
    /// Count of elements less than `k`.
    CountLtK,
    /// Sum of elements equal to `k`.
    SumEqK,
    /// Count of elements not equal to `k`.
    CountNeK,
    /// Max among elements strictly less than `k` (none → None).
    MaxLtK,
    /// Min among elements strictly greater than `k` (none → None).
    MinGtK,
    /// Sum of elements not equal to `k`.
    SumNeK,
    /// Max among elements strictly greater than `k` (none → None).
    MaxGtK,
    /// Min among elements strictly less than `k` (none → None).
    MinLtK,
    /// Count of elements ≥ `k`.
    CountGeK,
    /// Count of elements ≤ `k`.
    CountLeK,
    /// Sum of elements ≥ `k`.
    SumGeK,
    /// Sum of elements ≤ `k`.
    SumLeK,
    /// First index where `arr[i] >= k`, else -1.
    FirstGeK,
    /// First index where `arr[i] <= k`, else -1.
    FirstLeK,
    /// Last index where `arr[i] >= k`, else -1.
    LastGeK,
    /// Last index where `arr[i] <= k`, else -1.
    LastLeK,
    /// Sum of |v| for elements with v > k.
    SumAbsGtK,
    /// Sum of |v| for elements with v < k.
    SumAbsLtK,
    /// Count of elements with |v| > k.
    CountAbsGtK,
    /// Count of elements with |v| < k.
    CountAbsLtK,
    /// Sum of |v| for elements with v >= k.
    SumAbsGeK,
    /// Sum of |v| for elements with v <= k.
    SumAbsLeK,
    /// Count of elements with |v| == k.
    CountAbsEqK,
    /// Count of elements with |v| >= k.
    CountAbsGeK,
    /// Count of elements with |v| <= k.
    CountAbsLeK,
    /// First index where |arr[i]| >= k, else -1.
    FirstAbsGeK,
    /// Last index where |arr[i]| >= k, else -1.
    LastAbsGeK,
    /// First index where |arr[i]| == k, else -1.
    FirstAbsEqK,
    /// Last index where |arr[i]| == k, else -1.
    LastAbsEqK,
    /// First index where |arr[i]| <= k, else -1.
    FirstAbsLeK,
    /// Last index where |arr[i]| <= k, else -1.
    LastAbsLeK,
    /// Sum of |v| for elements with |v| == k.
    SumAbsEqK,
    /// First index where |arr[i]| > k, else -1.
    FirstAbsGtK,
    /// Last index where |arr[i]| > k, else -1.
    LastAbsGtK,
    /// Count of elements with |v| != k.
    CountAbsNeK,
    /// Sum of |v| for elements with |v| != k.
    SumAbsNeK,
    /// First index where |arr[i]| < k, else -1.
    FirstAbsLtK,
    /// Last index where |arr[i]| < k, else -1.
    LastAbsLtK,
    /// Max |v| among elements with |v| < k (none → 0).
    MaxAbsLtK,
    /// Min |v| among elements with |v| > k (none → 0).
    MinAbsGtK,
    /// Max |v| among elements with |v| > k (none → 0).
    MaxAbsGtK,
    /// Min |v| among elements with |v| < k (none → 0).
    MinAbsLtK,
    /// First index where |arr[i]| != k, else -1.
    FirstAbsNeK,
    /// Last index where |arr[i]| != k, else -1.
    LastAbsNeK,
    /// Sum of signed values where |v| == k.
    SumWhereAbsEqK,
    /// Product of signed values where |v| == k (none → 1).
    ProductWhereAbsEqK,
    /// Max signed value among elements with |v| == k (none → 0).
    MaxWhereAbsEqK,
    /// Min signed value among elements with |v| == k (none → 0).
    MinWhereAbsEqK,
    /// Sum of elements with |v| != k.
    SumWhereAbsNeK,
    /// Product of elements with |v| != k (none → 1).
    ProductWhereAbsNeK,
    /// Max among elements with |v| != k (none → 0).
    MaxWhereAbsNeK,
    /// Min among elements with |v| != k (none → 0).
    MinWhereAbsNeK,
    /// Sum of elements with |v| > k.
    SumWhereAbsGtK,
    /// Sum of elements with |v| < k.
    SumWhereAbsLtK,
    /// Product of elements with |v| > k (none → 1).
    ProductWhereAbsGtK,
    /// Product of elements with |v| < k (none → 1).
    ProductWhereAbsLtK,
    /// Max among elements with |v| > k (none → 0).
    MaxWhereAbsGtK,
    /// Min among elements with |v| > k (none → 0).
    MinWhereAbsGtK,
    /// Min among elements with |v| < k (none → 0).
    MinWhereAbsLtK,
    /// Max among elements with |v| >= k (none → 0).
    MaxWhereAbsGeK,
    /// Min among elements with |v| >= k (none → 0).
    MinWhereAbsGeK,
    /// Sum of elements with |v| >= k.
    SumWhereAbsGeK,
    /// Product of elements with |v| >= k (empty filter → 1).
    ProductWhereAbsGeK,
    /// Product of elements with |v| <= k (empty filter → 1).
    ProductWhereAbsLeK,
    /// Sum of elements with |v| <= k.
    SumWhereAbsLeK,
    /// Max among elements with |v| <= k (none → 0).
    MaxWhereAbsLeK,
    /// Min among elements with |v| <= k (none → 0).
    MinWhereAbsLeK,
    /// Count of elements with |v| >= k.
    CountWhereAbsGeK,
    /// Count of elements with |v| <= k.
    CountWhereAbsLeK,
    /// First element with |v| >= k (none → 0).
    FirstWhereAbsGeK,
    /// Last element with |v| >= k (none → 0).
    LastWhereAbsGeK,
    /// First element with |v| <= k (none → 0).
    FirstWhereAbsLeK,
    /// Last element with |v| <= k (none → 0).
    LastWhereAbsLeK,
    /// First element with |v| == k (none → 0).
    FirstWhereAbsEqK,
    /// Last element with |v| == k (none → 0).
    LastWhereAbsEqK,
    /// First element with |v| != k (none → 0).
    FirstWhereAbsNeK,
    /// Last element with |v| != k (none → 0).
    LastWhereAbsNeK,
    /// Count of elements with |v| != k.
    CountWhereAbsNeK,
    /// First index where |v| >= k (none → -1).
    FirstIndexWhereAbsGeK,
    /// Last index where |v| >= k (none → -1).
    LastIndexWhereAbsGeK,
    /// First index where |v| <= k (none → -1).
    FirstIndexWhereAbsLeK,
    /// Last index where |v| <= k (none → -1).
    LastIndexWhereAbsLeK,
    /// First index where |v| == k (none → -1).
    FirstIndexWhereAbsEqK,
    /// Last index where |v| == k (none → -1).
    LastIndexWhereAbsEqK,
    /// First index where |v| != k (none → -1).
    FirstIndexWhereAbsNeK,
    /// Last index where |v| != k (none → -1).
    LastIndexWhereAbsNeK,
    /// First index where |v| > k (none → -1).
    FirstIndexWhereAbsGtK,
    /// Last index where |v| > k (none → -1).
    LastIndexWhereAbsGtK,
    /// First index where |v| < k (none → -1).
    FirstIndexWhereAbsLtK,
    /// Last index where |v| < k (none → -1).
    LastIndexWhereAbsLtK,
    /// Count elements divisible by k (k == 0 → None).
    CountDivisibleByK,
    /// Sum of elements divisible by k (k == 0 → None).
    SumDivisibleByK,
    /// Product of elements divisible by k (k == 0 → None; empty product → 1).
    ProductDivisibleByK,
    /// First element divisible by k (none → 0; k == 0 → None).
    FirstDivisibleByK,
    /// Last element divisible by k (none → 0; k == 0 → None).
    LastDivisibleByK,
    /// Max among elements divisible by k (none → 0; k == 0 → None).
    MaxDivisibleByK,
    /// Min among elements divisible by k (none → 0; k == 0 → None).
    MinDivisibleByK,
    /// First index of element divisible by k (none → -1; k == 0 → None).
    FirstIndexDivisibleByK,
    /// Last index of element divisible by k (none → -1; k == 0 → None).
    LastIndexDivisibleByK,
    /// Sum of |v| for elements divisible by k (k == 0 → None).
    AbsSumDivisibleByK,
    /// Product of |v| for elements divisible by k (k == 0 → None; empty → 1).
    AbsProductDivisibleByK,
    /// Max |v| among elements divisible by k (none → 0; k == 0 → None).
    MaxAbsDivisibleByK,
    /// Min |v| among elements divisible by k (none → 0; k == 0 → None).
    MinAbsDivisibleByK,
    /// GCD of |v| for elements divisible by k (none → 0; k == 0 → None).
    GcdAbsDivisibleByK,
    /// LCM of |v| for elements divisible by k (none → 1; k == 0 → None).
    LcmAbsDivisibleByK,
    /// Truncating mean of |v| for elements divisible by k (none → 0; k == 0 → None).
    MeanAbsDivisibleByKTrunc,
    /// Count of non-zero elements divisible by k (k == 0 → None).
    CountNonZeroDivisibleByK,
    /// Sum of non-zero elements divisible by k (none → 0; k == 0 → None).
    SumNonZeroDivisibleByK,
    /// Product of non-zero elements divisible by k (none → 1; k == 0 → None).
    ProductNonZeroDivisibleByK,
    /// Max of non-zero elements divisible by k (none → 0; k == 0 → None).
    MaxNonZeroDivisibleByK,
    /// Min of non-zero elements divisible by k (none → 0; k == 0 → None).
    MinNonZeroDivisibleByK,
    /// First non-zero element divisible by k (none → 0; k == 0 → None).
    FirstNonZeroDivisibleByK,
    /// Last non-zero element divisible by k (none → 0; k == 0 → None).
    LastNonZeroDivisibleByK,
    /// Sum of |v| for non-zero elements divisible by k (none → 0; k == 0 → None).
    AbsSumNonZeroDivisibleByK,
    /// Product of |v| for non-zero elements divisible by k (none → 1; k == 0 → None).
    AbsProductNonZeroDivisibleByK,
    /// Truncating mean of non-zero elements divisible by k (none → 0; k == 0 → None).
    MeanNonZeroDivisibleByKTrunc,
    /// Max of |v| for non-zero elements divisible by k (none → 0; k == 0 → None).
    MaxAbsNonZeroDivisibleByK,
}

impl KClosed {
    fn label(self) -> &'static str {
        match self {
            KClosed::KthSmallest => "kth_smallest",
            KClosed::KthLargest => "kth_largest",
            KClosed::FirstIndexOf => "first_index_of",
            KClosed::LastIndexOf => "last_index_of",
            KClosed::KthFromEnd => "kth_from_end",
            KClosed::ElementAt => "element_at",
            KClosed::CountEqK => "count_eq_k",
            KClosed::SumGtK => "sum_gt_k",
            KClosed::CountGtK => "count_gt_k",
            KClosed::SumLtK => "sum_lt_k",
            KClosed::CountLtK => "count_lt_k",
            KClosed::SumEqK => "sum_eq_k",
            KClosed::CountNeK => "count_ne_k",
            KClosed::MaxLtK => "max_lt_k",
            KClosed::MinGtK => "min_gt_k",
            KClosed::SumNeK => "sum_ne_k",
            KClosed::MaxGtK => "max_gt_k",
            KClosed::MinLtK => "min_lt_k",
            KClosed::CountGeK => "count_ge_k",
            KClosed::CountLeK => "count_le_k",
            KClosed::SumGeK => "sum_ge_k",
            KClosed::SumLeK => "sum_le_k",
            KClosed::FirstGeK => "first_ge_k",
            KClosed::FirstLeK => "first_le_k",
            KClosed::LastGeK => "last_ge_k",
            KClosed::LastLeK => "last_le_k",
            KClosed::SumAbsGtK => "sum_abs_gt_k",
            KClosed::SumAbsLtK => "sum_abs_lt_k",
            KClosed::CountAbsGtK => "count_abs_gt_k",
            KClosed::CountAbsLtK => "count_abs_lt_k",
            KClosed::SumAbsGeK => "sum_abs_ge_k",
            KClosed::SumAbsLeK => "sum_abs_le_k",
            KClosed::CountAbsEqK => "count_abs_eq_k",
            KClosed::CountAbsGeK => "count_abs_ge_k",
            KClosed::CountAbsLeK => "count_abs_le_k",
            KClosed::FirstAbsGeK => "first_abs_ge_k",
            KClosed::LastAbsGeK => "last_abs_ge_k",
            KClosed::FirstAbsEqK => "first_abs_eq_k",
            KClosed::LastAbsEqK => "last_abs_eq_k",
            KClosed::FirstAbsLeK => "first_abs_le_k",
            KClosed::LastAbsLeK => "last_abs_le_k",
            KClosed::SumAbsEqK => "sum_abs_eq_k",
            KClosed::FirstAbsGtK => "first_abs_gt_k",
            KClosed::LastAbsGtK => "last_abs_gt_k",
            KClosed::CountAbsNeK => "count_abs_ne_k",
            KClosed::SumAbsNeK => "sum_abs_ne_k",
            KClosed::FirstAbsLtK => "first_abs_lt_k",
            KClosed::LastAbsLtK => "last_abs_lt_k",
            KClosed::MaxAbsLtK => "max_abs_lt_k",
            KClosed::MinAbsGtK => "min_abs_gt_k",
            KClosed::MaxAbsGtK => "max_abs_gt_k",
            KClosed::MinAbsLtK => "min_abs_lt_k",
            KClosed::FirstAbsNeK => "first_abs_ne_k",
            KClosed::LastAbsNeK => "last_abs_ne_k",
            KClosed::SumWhereAbsEqK => "sum_where_abs_eq_k",
            KClosed::ProductWhereAbsEqK => "product_where_abs_eq_k",
            KClosed::MaxWhereAbsEqK => "max_where_abs_eq_k",
            KClosed::MinWhereAbsEqK => "min_where_abs_eq_k",
            KClosed::SumWhereAbsNeK => "sum_where_abs_ne_k",
            KClosed::ProductWhereAbsNeK => "product_where_abs_ne_k",
            KClosed::MaxWhereAbsNeK => "max_where_abs_ne_k",
            KClosed::MinWhereAbsNeK => "min_where_abs_ne_k",
            KClosed::SumWhereAbsGtK => "sum_where_abs_gt_k",
            KClosed::SumWhereAbsLtK => "sum_where_abs_lt_k",
            KClosed::ProductWhereAbsGtK => "product_where_abs_gt_k",
            KClosed::ProductWhereAbsLtK => "product_where_abs_lt_k",
            KClosed::MaxWhereAbsGtK => "max_where_abs_gt_k",
            KClosed::MinWhereAbsGtK => "min_where_abs_gt_k",
            KClosed::MinWhereAbsLtK => "min_where_abs_lt_k",
            KClosed::MaxWhereAbsGeK => "max_where_abs_ge_k",
            KClosed::MinWhereAbsGeK => "min_where_abs_ge_k",
            KClosed::SumWhereAbsGeK => "sum_where_abs_ge_k",
            KClosed::ProductWhereAbsGeK => "product_where_abs_ge_k",
            KClosed::ProductWhereAbsLeK => "product_where_abs_le_k",
            KClosed::SumWhereAbsLeK => "sum_where_abs_le_k",
            KClosed::MaxWhereAbsLeK => "max_where_abs_le_k",
            KClosed::MinWhereAbsLeK => "min_where_abs_le_k",
            KClosed::CountWhereAbsGeK => "count_where_abs_ge_k",
            KClosed::CountWhereAbsLeK => "count_where_abs_le_k",
            KClosed::FirstWhereAbsGeK => "first_where_abs_ge_k",
            KClosed::LastWhereAbsGeK => "last_where_abs_ge_k",
            KClosed::FirstWhereAbsLeK => "first_where_abs_le_k",
            KClosed::LastWhereAbsLeK => "last_where_abs_le_k",
            KClosed::FirstWhereAbsEqK => "first_where_abs_eq_k",
            KClosed::LastWhereAbsEqK => "last_where_abs_eq_k",
            KClosed::FirstWhereAbsNeK => "first_where_abs_ne_k",
            KClosed::LastWhereAbsNeK => "last_where_abs_ne_k",
            KClosed::CountWhereAbsNeK => "count_where_abs_ne_k",
            KClosed::FirstIndexWhereAbsGeK => "first_index_where_abs_ge_k",
            KClosed::LastIndexWhereAbsGeK => "last_index_where_abs_ge_k",
            KClosed::FirstIndexWhereAbsLeK => "first_index_where_abs_le_k",
            KClosed::LastIndexWhereAbsLeK => "last_index_where_abs_le_k",
            KClosed::FirstIndexWhereAbsEqK => "first_index_where_abs_eq_k",
            KClosed::LastIndexWhereAbsEqK => "last_index_where_abs_eq_k",
            KClosed::FirstIndexWhereAbsNeK => "first_index_where_abs_ne_k",
            KClosed::LastIndexWhereAbsNeK => "last_index_where_abs_ne_k",
            KClosed::FirstIndexWhereAbsGtK => "first_index_where_abs_gt_k",
            KClosed::LastIndexWhereAbsGtK => "last_index_where_abs_gt_k",
            KClosed::FirstIndexWhereAbsLtK => "first_index_where_abs_lt_k",
            KClosed::LastIndexWhereAbsLtK => "last_index_where_abs_lt_k",
            KClosed::CountDivisibleByK => "count_divisible_by_k",
            KClosed::SumDivisibleByK => "sum_divisible_by_k",
            KClosed::ProductDivisibleByK => "product_divisible_by_k",
            KClosed::FirstDivisibleByK => "first_divisible_by_k",
            KClosed::LastDivisibleByK => "last_divisible_by_k",
            KClosed::MaxDivisibleByK => "max_divisible_by_k",
            KClosed::MinDivisibleByK => "min_divisible_by_k",
            KClosed::FirstIndexDivisibleByK => "first_index_divisible_by_k",
            KClosed::LastIndexDivisibleByK => "last_index_divisible_by_k",
            KClosed::AbsSumDivisibleByK => "abs_sum_divisible_by_k",
            KClosed::AbsProductDivisibleByK => "abs_product_divisible_by_k",
            KClosed::MaxAbsDivisibleByK => "max_abs_divisible_by_k",
            KClosed::MinAbsDivisibleByK => "min_abs_divisible_by_k",
            KClosed::GcdAbsDivisibleByK => "gcd_abs_divisible_by_k",
            KClosed::LcmAbsDivisibleByK => "lcm_abs_divisible_by_k",
            KClosed::MeanAbsDivisibleByKTrunc => "mean_abs_divisible_by_k_trunc",
            KClosed::CountNonZeroDivisibleByK => "count_non_zero_divisible_by_k",
            KClosed::SumNonZeroDivisibleByK => "sum_non_zero_divisible_by_k",
            KClosed::ProductNonZeroDivisibleByK => "product_non_zero_divisible_by_k",
            KClosed::MaxNonZeroDivisibleByK => "max_non_zero_divisible_by_k",
            KClosed::MinNonZeroDivisibleByK => "min_non_zero_divisible_by_k",
            KClosed::FirstNonZeroDivisibleByK => "first_non_zero_divisible_by_k",
            KClosed::LastNonZeroDivisibleByK => "last_non_zero_divisible_by_k",
            KClosed::AbsSumNonZeroDivisibleByK => "abs_sum_non_zero_divisible_by_k",
            KClosed::AbsProductNonZeroDivisibleByK => "abs_product_non_zero_divisible_by_k",
            KClosed::MeanNonZeroDivisibleByKTrunc => "mean_non_zero_divisible_by_k_trunc",
            KClosed::MaxAbsNonZeroDivisibleByK => "max_abs_non_zero_divisible_by_k",
        }
    }

    fn eval(self, arr: &[i64], k: i64) -> Option<i64> {
        match self {
            KClosed::KthSmallest => {
                if k < 1 || k as usize > arr.len() {
                    return None;
                }
                let mut sorted = arr.to_vec();
                sorted.sort_unstable();
                Some(sorted[(k as usize) - 1])
            }
            KClosed::KthLargest => {
                if k < 1 || k as usize > arr.len() {
                    return None;
                }
                let mut sorted = arr.to_vec();
                sorted.sort_unstable();
                Some(sorted[arr.len() - (k as usize)])
            }
            KClosed::FirstIndexOf => {
                for (i, &v) in arr.iter().enumerate() {
                    if v == k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastIndexOf => {
                for i in (0..arr.len()).rev() {
                    if arr[i] == k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::KthFromEnd => {
                if k < 1 || k as usize > arr.len() {
                    return None;
                }
                Some(arr[arr.len() - (k as usize)])
            }
            KClosed::ElementAt => {
                if k < 0 || k as usize >= arr.len() {
                    return None;
                }
                Some(arr[k as usize])
            }
            KClosed::CountEqK => Some(arr.iter().filter(|&&v| v == k).count() as i64),
            KClosed::SumGtK => Some(
                arr.iter()
                    .filter(|&&v| v > k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::CountGtK => Some(arr.iter().filter(|&&v| v > k).count() as i64),
            KClosed::SumLtK => Some(
                arr.iter()
                    .filter(|&&v| v < k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::CountLtK => Some(arr.iter().filter(|&&v| v < k).count() as i64),
            KClosed::SumEqK => Some(
                arr.iter()
                    .filter(|&&v| v == k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::CountNeK => Some(arr.iter().filter(|&&v| v != k).count() as i64),
            KClosed::MaxLtK => {
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v < k {
                        best = Some(match best {
                            Some(b) if b >= v => b,
                            _ => v,
                        });
                    }
                }
                best
            }
            KClosed::MinGtK => {
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v > k {
                        best = Some(match best {
                            Some(b) if b <= v => b,
                            _ => v,
                        });
                    }
                }
                best
            }
            KClosed::SumNeK => Some(
                arr.iter()
                    .filter(|&&v| v != k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::MaxGtK => {
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v > k {
                        best = Some(match best {
                            Some(b) if b >= v => b,
                            _ => v,
                        });
                    }
                }
                best
            }
            KClosed::MinLtK => {
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v < k {
                        best = Some(match best {
                            Some(b) if b <= v => b,
                            _ => v,
                        });
                    }
                }
                best
            }
            KClosed::CountGeK => Some(arr.iter().filter(|&&v| v >= k).count() as i64),
            KClosed::CountLeK => Some(arr.iter().filter(|&&v| v <= k).count() as i64),
            KClosed::SumGeK => Some(
                arr.iter()
                    .filter(|&&v| v >= k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::SumLeK => Some(
                arr.iter()
                    .filter(|&&v| v <= k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::FirstGeK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v >= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::FirstLeK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v <= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastGeK => {
                for i in (0..arr.len()).rev() {
                    if arr[i] >= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastLeK => {
                for i in (0..arr.len()).rev() {
                    if arr[i] <= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::SumAbsGtK => Some(
                arr.iter()
                    .filter(|&&v| v > k)
                    .map(|&v| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::SumAbsLtK => Some(
                arr.iter()
                    .filter(|&&v| v < k)
                    .map(|&v| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::CountAbsGtK => {
                Some(arr.iter().filter(|&&v| v.abs() > k).count() as i64)
            }
            KClosed::CountAbsLtK => {
                Some(arr.iter().filter(|&&v| v.abs() < k).count() as i64)
            }
            KClosed::SumAbsGeK => Some(
                arr.iter()
                    .filter(|&&v| v >= k)
                    .map(|&v| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::SumAbsLeK => Some(
                arr.iter()
                    .filter(|&&v| v <= k)
                    .map(|&v| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::CountAbsEqK => {
                Some(arr.iter().filter(|&&v| v.abs() == k).count() as i64)
            }
            KClosed::CountAbsGeK => {
                Some(arr.iter().filter(|&&v| v.abs() >= k).count() as i64)
            }
            KClosed::CountAbsLeK => {
                Some(arr.iter().filter(|&&v| v.abs() <= k).count() as i64)
            }
            KClosed::FirstAbsGeK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() >= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastAbsGeK => {
                for i in (0..arr.len()).rev() {
                    if arr[i].abs() >= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::FirstAbsEqK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() == k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastAbsEqK => {
                for i in (0..arr.len()).rev() {
                    if arr[i].abs() == k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::FirstAbsLeK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() <= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastAbsLeK => {
                for i in (0..arr.len()).rev() {
                    if arr[i].abs() <= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::SumAbsEqK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() == k)
                    .map(|&v| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::FirstAbsGtK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() > k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastAbsGtK => {
                for i in (0..arr.len()).rev() {
                    if arr[i].abs() > k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::CountAbsNeK => {
                Some(arr.iter().filter(|&&v| v.abs() != k).count() as i64)
            }
            KClosed::SumAbsNeK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() != k)
                    .map(|&v| v.abs())
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::FirstAbsLtK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() < k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastAbsLtK => {
                for i in (0..arr.len()).rev() {
                    if arr[i].abs() < k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::MaxAbsLtK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    let a = v.abs();
                    if a < k {
                        if !found || a > best {
                            best = a;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::MinAbsGtK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    let a = v.abs();
                    if a > k {
                        if !found || a < best {
                            best = a;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::MaxAbsGtK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    let a = v.abs();
                    if a > k {
                        if !found || a > best {
                            best = a;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::MinAbsLtK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    let a = v.abs();
                    if a < k {
                        if !found || a < best {
                            best = a;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::FirstAbsNeK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() != k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastAbsNeK => {
                for i in (0..arr.len()).rev() {
                    if arr[i].abs() != k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::SumWhereAbsEqK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() == k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::ProductWhereAbsEqK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() == k)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            KClosed::MaxWhereAbsEqK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() == k {
                        if !found || v > best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::MinWhereAbsEqK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() == k {
                        if !found || v < best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::SumWhereAbsNeK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() != k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::ProductWhereAbsNeK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() != k)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            KClosed::MaxWhereAbsNeK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() != k {
                        if !found || v > best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::MinWhereAbsNeK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() != k {
                        if !found || v < best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::SumWhereAbsGtK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() > k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::SumWhereAbsLtK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() < k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::ProductWhereAbsGtK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() > k)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            KClosed::ProductWhereAbsLtK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() < k)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            KClosed::MaxWhereAbsGtK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() > k {
                        if !found || v > best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::MinWhereAbsGtK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() > k {
                        if !found || v < best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::MinWhereAbsLtK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() < k {
                        if !found || v < best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::MaxWhereAbsGeK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() >= k {
                        if !found || v > best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::MinWhereAbsGeK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() >= k {
                        if !found || v < best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::SumWhereAbsGeK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() >= k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::ProductWhereAbsGeK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() >= k)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            KClosed::ProductWhereAbsLeK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() <= k)
                    .copied()
                    .fold(1i64, i64::saturating_mul),
            ),
            KClosed::SumWhereAbsLeK => Some(
                arr.iter()
                    .filter(|&&v| v.abs() <= k)
                    .copied()
                    .fold(0i64, i64::saturating_add),
            ),
            KClosed::MaxWhereAbsLeK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() <= k {
                        if !found || v > best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::MinWhereAbsLeK => {
                let mut best = 0i64;
                let mut found = false;
                for &v in arr {
                    if v.abs() <= k {
                        if !found || v < best {
                            best = v;
                            found = true;
                        }
                    }
                }
                Some(best)
            }
            KClosed::CountWhereAbsGeK => Some(
                arr.iter().filter(|&&v| v.abs() >= k).count() as i64,
            ),
            KClosed::CountWhereAbsLeK => Some(
                arr.iter().filter(|&&v| v.abs() <= k).count() as i64,
            ),
            KClosed::FirstWhereAbsGeK => Some(
                arr.iter()
                    .copied()
                    .find(|&v| v.abs() >= k)
                    .unwrap_or(0),
            ),
            KClosed::LastWhereAbsGeK => Some(
                arr.iter()
                    .copied()
                    .rev()
                    .find(|&v| v.abs() >= k)
                    .unwrap_or(0),
            ),
            KClosed::FirstWhereAbsLeK => Some(
                arr.iter()
                    .copied()
                    .find(|&v| v.abs() <= k)
                    .unwrap_or(0),
            ),
            KClosed::LastWhereAbsLeK => Some(
                arr.iter()
                    .copied()
                    .rev()
                    .find(|&v| v.abs() <= k)
                    .unwrap_or(0),
            ),
            KClosed::FirstWhereAbsEqK => Some(
                arr.iter()
                    .copied()
                    .find(|&v| v.abs() == k)
                    .unwrap_or(0),
            ),
            KClosed::LastWhereAbsEqK => Some(
                arr.iter()
                    .copied()
                    .rev()
                    .find(|&v| v.abs() == k)
                    .unwrap_or(0),
            ),
            KClosed::FirstWhereAbsNeK => Some(
                arr.iter()
                    .copied()
                    .find(|&v| v.abs() != k)
                    .unwrap_or(0),
            ),
            KClosed::LastWhereAbsNeK => Some(
                arr.iter()
                    .copied()
                    .rev()
                    .find(|&v| v.abs() != k)
                    .unwrap_or(0),
            ),
            KClosed::CountWhereAbsNeK => Some(
                arr.iter().filter(|&&v| v.abs() != k).count() as i64,
            ),
            KClosed::FirstIndexWhereAbsGeK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() >= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastIndexWhereAbsGeK => {
                for (i, &v) in arr.iter().enumerate().rev() {
                    if v.abs() >= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::FirstIndexWhereAbsLeK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() <= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastIndexWhereAbsLeK => {
                for (i, &v) in arr.iter().enumerate().rev() {
                    if v.abs() <= k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::FirstIndexWhereAbsEqK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() == k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastIndexWhereAbsEqK => {
                for (i, &v) in arr.iter().enumerate().rev() {
                    if v.abs() == k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::FirstIndexWhereAbsNeK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() != k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastIndexWhereAbsNeK => {
                for (i, &v) in arr.iter().enumerate().rev() {
                    if v.abs() != k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::FirstIndexWhereAbsGtK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() > k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastIndexWhereAbsGtK => {
                for (i, &v) in arr.iter().enumerate().rev() {
                    if v.abs() > k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::FirstIndexWhereAbsLtK => {
                for (i, &v) in arr.iter().enumerate() {
                    if v.abs() < k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastIndexWhereAbsLtK => {
                for (i, &v) in arr.iter().enumerate().rev() {
                    if v.abs() < k {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::CountDivisibleByK => {
                if k == 0 {
                    return None;
                }
                Some(arr.iter().filter(|&&v| v % k == 0).count() as i64)
            }
            KClosed::SumDivisibleByK => {
                if k == 0 {
                    return None;
                }
                Some(
                    arr.iter()
                        .filter(|&&v| v % k == 0)
                        .copied()
                        .fold(0i64, i64::saturating_add),
                )
            }
            KClosed::ProductDivisibleByK => {
                if k == 0 {
                    return None;
                }
                Some(
                    arr.iter()
                        .filter(|&&v| v % k == 0)
                        .copied()
                        .fold(1i64, i64::saturating_mul),
                )
            }
            KClosed::FirstDivisibleByK => {
                if k == 0 {
                    return None;
                }
                for &v in arr {
                    if v % k == 0 {
                        return Some(v);
                    }
                }
                Some(0)
            }
            KClosed::LastDivisibleByK => {
                if k == 0 {
                    return None;
                }
                for &v in arr.iter().rev() {
                    if v % k == 0 {
                        return Some(v);
                    }
                }
                Some(0)
            }
            KClosed::MaxDivisibleByK => {
                if k == 0 {
                    return None;
                }
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v % k == 0 {
                        best = Some(best.map_or(v, |b| b.max(v)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            KClosed::MinDivisibleByK => {
                if k == 0 {
                    return None;
                }
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v % k == 0 {
                        best = Some(best.map_or(v, |b| b.min(v)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            KClosed::FirstIndexDivisibleByK => {
                if k == 0 {
                    return None;
                }
                for (i, &v) in arr.iter().enumerate() {
                    if v % k == 0 {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::LastIndexDivisibleByK => {
                if k == 0 {
                    return None;
                }
                for (i, &v) in arr.iter().enumerate().rev() {
                    if v % k == 0 {
                        return Some(i as i64);
                    }
                }
                Some(-1)
            }
            KClosed::AbsSumDivisibleByK => {
                if k == 0 {
                    return None;
                }
                Some(
                    arr.iter()
                        .filter(|&&v| v % k == 0)
                        .map(|&v| v.abs())
                        .fold(0i64, i64::saturating_add),
                )
            }
            KClosed::AbsProductDivisibleByK => {
                if k == 0 {
                    return None;
                }
                Some(
                    arr.iter()
                        .filter(|&&v| v % k == 0)
                        .map(|&v| v.abs())
                        .fold(1i64, i64::saturating_mul),
                )
            }
            KClosed::MaxAbsDivisibleByK => {
                if k == 0 {
                    return None;
                }
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v % k == 0 {
                        let a = v.abs();
                        best = Some(best.map_or(a, |b| b.max(a)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            KClosed::MinAbsDivisibleByK => {
                if k == 0 {
                    return None;
                }
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v % k == 0 {
                        let a = v.abs();
                        best = Some(best.map_or(a, |b| b.min(a)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            KClosed::GcdAbsDivisibleByK => {
                if k == 0 {
                    return None;
                }
                let mut g: Option<i64> = None;
                for &v in arr {
                    if v % k == 0 {
                        let a = v.abs();
                        if a == 0 {
                            continue;
                        }
                        g = Some(g.map_or(a, |g| i64_gcd(g, a)));
                    }
                }
                Some(g.unwrap_or(0))
            }
            KClosed::LcmAbsDivisibleByK => {
                if k == 0 {
                    return None;
                }
                let mut l: Option<i64> = None;
                for &v in arr {
                    if v % k == 0 {
                        let a = v.abs();
                        if a == 0 {
                            continue;
                        }
                        l = Some(l.map_or(a, |l| i64_lcm(l, a)));
                    }
                }
                Some(l.unwrap_or(1))
            }
            KClosed::MeanAbsDivisibleByKTrunc => {
                if k == 0 {
                    return None;
                }
                let xs: Vec<i64> = arr
                    .iter()
                    .copied()
                    .filter(|&v| v % k == 0)
                    .map(|v| v.abs())
                    .collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            KClosed::CountNonZeroDivisibleByK => {
                if k == 0 {
                    return None;
                }
                Some(arr.iter().filter(|&&v| v != 0 && v % k == 0).count() as i64)
            }
            KClosed::SumNonZeroDivisibleByK => {
                if k == 0 {
                    return None;
                }
                Some(
                    arr.iter()
                        .filter(|&&v| v != 0 && v % k == 0)
                        .fold(0i64, |a, &b| a.saturating_add(b)),
                )
            }
            KClosed::ProductNonZeroDivisibleByK => {
                if k == 0 {
                    return None;
                }
                Some(
                    arr.iter()
                        .filter(|&&v| v != 0 && v % k == 0)
                        .fold(1i64, |a, &b| a.saturating_mul(b)),
                )
            }
            KClosed::MaxNonZeroDivisibleByK => {
                if k == 0 {
                    return None;
                }
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v != 0 && v % k == 0 {
                        best = Some(best.map_or(v, |b| b.max(v)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            KClosed::MinNonZeroDivisibleByK => {
                if k == 0 {
                    return None;
                }
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v != 0 && v % k == 0 {
                        best = Some(best.map_or(v, |b| b.min(v)));
                    }
                }
                Some(best.unwrap_or(0))
            }
            KClosed::FirstNonZeroDivisibleByK => {
                if k == 0 {
                    return None;
                }
                for &v in arr {
                    if v != 0 && v % k == 0 {
                        return Some(v);
                    }
                }
                Some(0)
            }
            KClosed::LastNonZeroDivisibleByK => {
                if k == 0 {
                    return None;
                }
                for &v in arr.iter().rev() {
                    if v != 0 && v % k == 0 {
                        return Some(v);
                    }
                }
                Some(0)
            }
            KClosed::AbsSumNonZeroDivisibleByK => {
                if k == 0 {
                    return None;
                }
                Some(
                    arr.iter()
                        .filter(|&&v| v != 0 && v % k == 0)
                        .map(|&v| v.abs())
                        .fold(0i64, i64::saturating_add),
                )
            }
            KClosed::AbsProductNonZeroDivisibleByK => {
                if k == 0 {
                    return None;
                }
                Some(
                    arr.iter()
                        .filter(|&&v| v != 0 && v % k == 0)
                        .map(|&v| v.abs())
                        .fold(1i64, |a, b| a.saturating_mul(b)),
                )
            }
            KClosed::MeanNonZeroDivisibleByKTrunc => {
                if k == 0 {
                    return None;
                }
                let xs: Vec<i64> = arr
                    .iter()
                    .copied()
                    .filter(|&v| v != 0 && v % k == 0)
                    .collect();
                if xs.is_empty() {
                    Some(0)
                } else {
                    Some(xs.iter().copied().fold(0i64, i64::saturating_add) / xs.len() as i64)
                }
            }
            KClosed::MaxAbsNonZeroDivisibleByK => {
                if k == 0 {
                    return None;
                }
                let mut best: Option<i64> = None;
                for &v in arr {
                    if v != 0 && v % k == 0 {
                        let a = v.abs();
                        best = Some(best.map_or(a, |b| b.max(a)));
                    }
                }
                Some(best.unwrap_or(0))
            }
        }
    }

    fn emit(self, fn_name: &str) -> String {
        match self {
            KClosed::KthSmallest => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    arr.sort();\n\
    return arr[k - 1];\n\
}}\n"
            ),
            KClosed::KthLargest => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    arr.sort();\n\
    return arr[arr.len - k];\n\
}}\n"
            ),
            KClosed::FirstIndexOf => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] == k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastIndexOf => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        if arr[i] == k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::KthFromEnd => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    return arr[arr.len - k];\n\
}}\n"
            ),
            KClosed::ElementAt => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    return arr[k];\n\
}}\n"
            ),
            KClosed::CountEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item == k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            KClosed::SumGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item > k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::CountGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item > k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            KClosed::SumLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item < k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::CountLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item < k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            KClosed::SumEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item == k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::CountNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item != k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            KClosed::MaxLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item < k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MinGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item > k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::SumNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item != k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::MaxGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item > k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MinLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        if item < k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::CountGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item >= k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            KClosed::CountLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        if item <= k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            KClosed::SumGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item >= k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::SumLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item <= k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::FirstGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] >= k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::FirstLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        if arr[i] <= k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        if arr[i] >= k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        if arr[i] <= k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::SumAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item > k {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::SumAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item < k {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::CountAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            KClosed::CountAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            KClosed::SumAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item >= k {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::SumAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item <= k {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::CountAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
                        KClosed::CountAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            KClosed::CountAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
KClosed::FirstAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::FirstAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::FirstAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::SumAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            total = total + a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::FirstAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::CountAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    count: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            count = count + 1;\n\
        }}\n\
    }}\n\
    return count;\n\
}}\n"
            ),
            KClosed::SumAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            total = total + a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::FirstAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::MaxAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < k {{\n\
            if found == 0 {{\n\
                best = a;\n\
                found = 1;\n\
            }} else {{\n\
                if a > best {{ best = a; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MinAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            if found == 0 {{\n\
                best = a;\n\
                found = 1;\n\
            }} else {{\n\
                if a < best {{ best = a; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MaxAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            if found == 0 {{\n\
                best = a;\n\
                found = 1;\n\
            }} else {{\n\
                if a > best {{ best = a; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MinAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < k {{\n\
            if found == 0 {{\n\
                best = a;\n\
                found = 1;\n\
            }} else {{\n\
                if a < best {{ best = a; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::FirstAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        a: i64 = arr[i];\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::SumWhereAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::ProductWhereAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::MaxWhereAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MinWhereAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
KClosed::SumWhereAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
KClosed::ProductWhereAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
KClosed::MaxWhereAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
KClosed::MinWhereAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::SumWhereAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::SumWhereAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::ProductWhereAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::ProductWhereAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < k {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::MaxWhereAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MinWhereAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MinWhereAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MaxWhereAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MinWhereAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::SumWhereAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::ProductWhereAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::ProductWhereAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 1;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            total = total * item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::SumWhereAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::MaxWhereAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item > best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MinWhereAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    best: i64 = 0;\n\
    found: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }} else {{\n\
                if item < best {{ best = item; }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::CountWhereAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            n = n + 1;\n\
        }}\n\
    }}\n\
    return n;\n\
}}\n"
            ),
            KClosed::CountWhereAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            n = n + 1;\n\
        }}\n\
    }}\n\
    return n;\n\
}}\n"
            ),
            KClosed::FirstWhereAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            return item;\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            KClosed::LastWhereAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            return item;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            KClosed::FirstWhereAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            return item;\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            KClosed::LastWhereAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            return item;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            KClosed::FirstWhereAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            return item;\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            KClosed::LastWhereAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            return item;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            KClosed::FirstWhereAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            return item;\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            KClosed::LastWhereAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            return item;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            KClosed::CountWhereAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            n = n + 1;\n\
        }}\n\
    }}\n\
    return n;\n\
}}\n"
            ),
            KClosed::FirstIndexWhereAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastIndexWhereAbsGeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a >= k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::FirstIndexWhereAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::LastIndexWhereAbsLeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a <= k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),
            KClosed::FirstIndexWhereAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),

            KClosed::LastIndexWhereAbsEqK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a == k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),

            KClosed::FirstIndexWhereAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),

            KClosed::LastIndexWhereAbsNeK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a != k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),

            KClosed::FirstIndexWhereAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),

            KClosed::LastIndexWhereAbsGtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a > k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),

            KClosed::FirstIndexWhereAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < k {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),

            KClosed::LastIndexWhereAbsLtK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        a: i64 = item;\n\
        if a < 0 {{ a = 0 - a; }}\n\
        if a < k {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),

            KClosed::CountDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            n = n + 1;\n\
        }}\n\
    }}\n\
    return n;\n\
}}\n"
            ),

            KClosed::SumDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            total = total + item;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),

            KClosed::ProductDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            prod = prod * item;\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),

            KClosed::FirstDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            return item;\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),

            KClosed::LastDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        if item % k == 0 {{\n\
            return item;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0;\n\
}}\n"
            ),

            KClosed::MaxDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }}\n\
            if item > best {{\n\
                best = item;\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),

            KClosed::MinDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            if found == 0 {{\n\
                best = item;\n\
                found = 1;\n\
            }}\n\
            if item < best {{\n\
                best = item;\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),

            KClosed::FirstIndexDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = 0;\n\
    while i < arr.len {{\n\
        item: i64 = arr[i];\n\
        if item % k == 0 {{\n\
            return i;\n\
        }}\n\
        i = i + 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),

            KClosed::LastIndexDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        if item % k == 0 {{\n\
            return i;\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0 - 1;\n\
}}\n"
            ),

            KClosed::AbsSumDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),

            KClosed::AbsProductDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            prod = prod * a;\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),

            KClosed::MaxAbsDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if found == 0 {{\n\
                best = a;\n\
                found = 1;\n\
            }}\n\
            if a > best {{\n\
                best = a;\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),

            KClosed::MinAbsDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if found == 0 {{\n\
                best = a;\n\
                found = 1;\n\
            }}\n\
            if a < best {{\n\
                best = a;\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),

            KClosed::GcdAbsDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    found: i64 = 0;\n\
    g: i64 = 0;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if a != 0 {{\n\
                if found == 0 {{\n\
                    g = a;\n\
                    found = 1;\n\
                }} else {{\n\
                    while a != 0 {{\n\
                        t: i64 = a;\n\
                        a = g % a;\n\
                        g = t;\n\
                    }}\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return g;\n\
}}\n"
            ),

            KClosed::LcmAbsDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    found: i64 = 0;\n\
    l: i64 = 1;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            if a != 0 {{\n\
                if found == 0 {{\n\
                    l = a;\n\
                    found = 1;\n\
                }} else {{\n\
                    g: i64 = l;\n\
                    b: i64 = a;\n\
                    while b != 0 {{\n\
                        t: i64 = b;\n\
                        b = g % b;\n\
                        g = t;\n\
                    }}\n\
                    l = (l / g) * a;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return l;\n\
}}\n"
            ),

            KClosed::MeanAbsDivisibleByKTrunc => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item % k == 0 {{\n\
            a: i64 = item;\n\
            if a < 0 {{ a = 0 - a; }}\n\
            total = total + a;\n\
            n = n + 1;\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),

            KClosed::CountNonZeroDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    return n;\n\
}}\n"
            ),
            KClosed::SumNonZeroDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                total = total + item;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::ProductNonZeroDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                prod = prod * item;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            KClosed::MaxNonZeroDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }}\n\
                if item > best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::MinNonZeroDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                if found == 0 {{\n\
                    best = item;\n\
                    found = 1;\n\
                }}\n\
                if item < best {{\n\
                    best = item;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),
            KClosed::FirstNonZeroDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                return item;\n\
            }}\n\
        }}\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            KClosed::LastNonZeroDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    i: i64 = arr.len - 1;\n\
    while i >= 0 {{\n\
        item: i64 = arr[i];\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                return item;\n\
            }}\n\
        }}\n\
        i = i - 1;\n\
    }}\n\
    return 0;\n\
}}\n"
            ),
            KClosed::AbsSumNonZeroDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                total = total + a;\n\
            }}\n\
        }}\n\
    }}\n\
    return total;\n\
}}\n"
            ),
            KClosed::AbsProductNonZeroDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    prod: i64 = 1;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                prod = prod * a;\n\
            }}\n\
        }}\n\
    }}\n\
    return prod;\n\
}}\n"
            ),
            KClosed::MeanNonZeroDivisibleByKTrunc => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    total: i64 = 0;\n\
    n: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                total = total + item;\n\
                n = n + 1;\n\
            }}\n\
        }}\n\
    }}\n\
    if n == 0 {{\n\
        return 0;\n\
    }}\n\
    return total / n;\n\
}}\n"
            ),
            KClosed::MaxAbsNonZeroDivisibleByK => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n\
    found: i64 = 0;\n\
    best: i64 = 0;\n\
    for item in arr {{\n\
        if item != 0 {{\n\
            if item % k == 0 {{\n\
                a: i64 = item;\n\
                if a < 0 {{ a = 0 - a; }}\n\
                if found == 0 {{\n\
                    best = a;\n\
                    found = 1;\n\
                }}\n\
                if a > best {{\n\
                    best = a;\n\
                }}\n\
            }}\n\
        }}\n\
    }}\n\
    return best;\n\
}}\n"
            ),

        }
    }
}

fn try_k_closed(
    problem: &Problem,
    fn_name: &str,
    inputs: &[Vec<i64>],
    ks: &[Option<i64>],
    expected: &[i64],
) -> Option<SolveResult> {
    for closed in [
        KClosed::KthSmallest,
        KClosed::KthLargest,
        KClosed::FirstIndexOf,
        KClosed::LastIndexOf,
        KClosed::KthFromEnd,
        KClosed::ElementAt,
        KClosed::CountEqK,
        KClosed::SumGtK,
        KClosed::CountGtK,
        KClosed::SumLtK,
        KClosed::CountLtK,
        KClosed::SumEqK,
        KClosed::CountNeK,
        KClosed::MaxLtK,
        KClosed::MinGtK,
        KClosed::SumNeK,
        KClosed::MaxGtK,
        KClosed::MinLtK,
        KClosed::CountGeK,
        KClosed::CountLeK,
        KClosed::SumGeK,
        KClosed::SumLeK,
        KClosed::FirstGeK,
        KClosed::FirstLeK,
        KClosed::LastGeK,
        KClosed::LastLeK,
        KClosed::SumAbsGtK,
        KClosed::SumAbsLtK,
        KClosed::CountAbsGtK,
        KClosed::CountAbsLtK,
        KClosed::SumAbsGeK,
        KClosed::SumAbsLeK,
        KClosed::CountAbsEqK,
        KClosed::CountAbsGeK,
        KClosed::CountAbsLeK,
        KClosed::FirstAbsGeK,
        KClosed::LastAbsGeK,
        KClosed::FirstAbsEqK,
        KClosed::LastAbsEqK,
        KClosed::FirstAbsLeK,
        KClosed::LastAbsLeK,
        KClosed::SumAbsEqK,
        KClosed::FirstAbsGtK,
        KClosed::LastAbsGtK,
        KClosed::CountAbsNeK,
        KClosed::SumAbsNeK,
        KClosed::FirstAbsLtK,
        KClosed::LastAbsLtK,
        KClosed::MaxAbsLtK,
        KClosed::MinAbsGtK,
        KClosed::MaxAbsGtK,
        KClosed::MinAbsLtK,
        KClosed::FirstAbsNeK,
        KClosed::LastAbsNeK,
        KClosed::SumWhereAbsEqK,
        KClosed::ProductWhereAbsEqK,
        KClosed::MaxWhereAbsEqK,
        KClosed::MinWhereAbsEqK,
        KClosed::SumWhereAbsNeK,
        KClosed::ProductWhereAbsNeK,
        KClosed::MaxWhereAbsNeK,
        KClosed::MinWhereAbsNeK,
        KClosed::SumWhereAbsGtK,
        KClosed::SumWhereAbsLtK,
        KClosed::ProductWhereAbsGtK,
        KClosed::ProductWhereAbsLtK,
        KClosed::MaxWhereAbsGtK,
        KClosed::MinWhereAbsGtK,
        KClosed::MinWhereAbsLtK,
        KClosed::MaxWhereAbsGeK,
        KClosed::MinWhereAbsGeK,
        KClosed::SumWhereAbsGeK,
        KClosed::ProductWhereAbsGeK,
        KClosed::ProductWhereAbsLeK,
        KClosed::SumWhereAbsLeK,
        KClosed::MaxWhereAbsLeK,
        KClosed::MinWhereAbsLeK,
        KClosed::CountWhereAbsGeK,
        KClosed::CountWhereAbsLeK,
        KClosed::FirstWhereAbsGeK,
        KClosed::LastWhereAbsGeK,
        KClosed::FirstWhereAbsLeK,
        KClosed::LastWhereAbsLeK,
        KClosed::FirstWhereAbsEqK,
        KClosed::LastWhereAbsEqK,
        KClosed::FirstWhereAbsNeK,
        KClosed::LastWhereAbsNeK,
        KClosed::CountWhereAbsNeK,
        KClosed::FirstIndexWhereAbsGeK,
        KClosed::LastIndexWhereAbsGeK,
        KClosed::FirstIndexWhereAbsLeK,
        KClosed::LastIndexWhereAbsLeK,
        KClosed::FirstIndexWhereAbsEqK,
        KClosed::LastIndexWhereAbsEqK,
        KClosed::FirstIndexWhereAbsNeK,
        KClosed::LastIndexWhereAbsNeK,
        KClosed::FirstIndexWhereAbsGtK,
        KClosed::LastIndexWhereAbsGtK,
        KClosed::FirstIndexWhereAbsLtK,
        KClosed::LastIndexWhereAbsLtK,
        KClosed::CountDivisibleByK,
        KClosed::SumDivisibleByK,
        KClosed::ProductDivisibleByK,
        KClosed::FirstDivisibleByK,
        KClosed::LastDivisibleByK,
        KClosed::MaxDivisibleByK,
        KClosed::MinDivisibleByK,
        KClosed::FirstIndexDivisibleByK,
        KClosed::LastIndexDivisibleByK,
        KClosed::AbsSumDivisibleByK,
        KClosed::AbsProductDivisibleByK,
        KClosed::MaxAbsDivisibleByK,
        KClosed::MinAbsDivisibleByK,
        KClosed::GcdAbsDivisibleByK,
        KClosed::LcmAbsDivisibleByK,
        KClosed::MeanAbsDivisibleByKTrunc,
        KClosed::CountNonZeroDivisibleByK,
        KClosed::SumNonZeroDivisibleByK,
        KClosed::ProductNonZeroDivisibleByK,
        KClosed::MaxNonZeroDivisibleByK,
        KClosed::MinNonZeroDivisibleByK,
        KClosed::FirstNonZeroDivisibleByK,
        KClosed::LastNonZeroDivisibleByK,
        KClosed::AbsSumNonZeroDivisibleByK,
        KClosed::AbsProductNonZeroDivisibleByK,
        KClosed::MeanNonZeroDivisibleByKTrunc,
        KClosed::MaxAbsNonZeroDivisibleByK,
    ] {
        let ok = inputs
            .iter()
            .zip(ks.iter())
            .zip(expected.iter())
            .all(|((arr, k), &y)| {
                k.and_then(|k| closed.eval(arr, k)) == Some(y)
            });
        if !ok {
            continue;
        }
        let code = closed.emit(fn_name);
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: format!("utbus_k_{}", closed.label()),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }
    None
}

/// Public entry point. Returns `None` unless `NSYNTH_UTBUS=1`. When enabled,
/// runs the typed bottom-up enumerator over the array transform set and returns
/// the first candidate that passes [`verify_problem_code_strict`] (examples +
/// holdouts).
pub(super) fn synthesize_utbus(problem: &Problem) -> Option<SolveResult> {
    let mode = utbus_mode()?;

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

    // Native A3/A4 closed families first (tiny candidate set) for single-array
    // signatures — dual-accum / pairwise cannot be expressed as filter→map→reduce.
    if !with_k {
        if let Some(hit) = try_dual_and_pairwise(problem, fn_name, &inputs, &expected) {
            return Some(hit);
        }
    } else if let Some(hit) = try_k_closed(problem, fn_name, &inputs, &ks, &expected) {
        return Some(hit);
    }

    // `NSYNTH_UTBUS=closed` stops here — no combinatorial enum.
    if mode == "closed" {
        return None;
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
        assert_eq!(
            ArrayProgram {
                reduce: Reduce::Product,
                ..base
            }
            .eval_scalar(&[2, 3, 4], None),
            24
        );
    }

    #[test]
    fn enumerate_includes_each_reduce() {
        let programs = enumerate_array_programs(false);
        assert!(programs.iter().any(|p| p.reduce == Reduce::Sum));
        assert!(programs.iter().any(|p| p.reduce == Reduce::Max));
        assert!(programs.iter().any(|p| p.reduce == Reduce::Min));
        assert!(programs.iter().any(|p| p.reduce == Reduce::Count));
        assert!(programs.iter().any(|p| p.reduce == Reduce::Product));
        assert!(programs.iter().any(|p| p.reduce == Reduce::Xor));
        assert!(programs.iter().any(|p| p.reduce == Reduce::BitOr));
        assert!(programs.iter().any(|p| p.reduce == Reduce::BitAnd));
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

    #[test]
    fn utbus_solves_array_product() {
        let problem = array_problem(
            "array_product",
            "fn array_product(arr: [i64]) -> i64",
            &[&[2, 3, 4], &[5, 5], &[1, 2, 3, 4], &[-2, 3]],
            &[&[2, 2, 2], &[7]],
            |arr| arr.iter().copied().fold(1i64, i64::saturating_mul),
        );
        assert_solves(&problem, "product");
    }

    #[test]
    fn utbus_solves_array_xor() {
        let problem = array_problem(
            "array_xor",
            "fn array_xor(arr: [i64]) -> i64",
            &[&[1, 2, 3], &[7, 1], &[4, 4, 1], &[0]],
            &[&[8, 1, 9], &[5, 5, 5]],
            |arr| arr.iter().copied().fold(0i64, |a, b| a ^ b),
        );
        assert_solves(&problem, "xor");
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

    #[test]
    fn dual_accum_eval_range_and_second_max() {
        assert_eq!(DualAccum::Range.eval(&[1, 5, 3]), Some(4));
        assert_eq!(DualAccum::Range.eval(&[-2, 10, 0]), Some(12));
        // Teacher cascade (matches search_codegen / benchmark seeds).
        assert_eq!(DualAccum::SecondMax.eval(&[3, 1, 4, 1, 5]), Some(4));
        assert_eq!(DualAccum::SecondMax.eval(&[2, 8, 3]), Some(3));
        assert_eq!(DualAccum::SecondMax.eval(&[5, 10, 8]), Some(8));
        assert!(DualAccum::Range
            .emit("array_range")
            .contains("return hi - lo"));
    }

    #[test]
    fn pairwise_eval_max_abs_and_count_diff() {
        assert_eq!(PairwiseScan::MaxAbsDiff.eval(&[10, 3, 8]), Some(7));
        assert_eq!(PairwiseScan::MaxAbsDiff.eval(&[5, 5]), Some(0));
        assert_eq!(
            PairwiseScan::CountAdjacentDiff.eval(&[1, 1, 2, 2, 3]),
            Some(2)
        );
        assert_eq!(PairwiseScan::CountAdjacentDiff.eval(&[7]), Some(0));
        assert_eq!(PairwiseScan::CountIncreases.eval(&[1, 3, 2, 5]), Some(2));
        assert_eq!(PairwiseScan::StrictlyIncreasing.eval(&[1, 2, 4]), Some(1));
        assert_eq!(PairwiseScan::StrictlyIncreasing.eval(&[1, 2, 2]), Some(0));
    }

    #[test]
    fn utbus_solves_array_range() {
        let problem = array_problem(
            "array_range",
            "fn array_range(arr: [i64]) -> i64",
            &[&[1, 5, 3], &[-2, 10, 0], &[7], &[4, 4, 1]],
            &[&[10, -5, 2], &[1, 2, 3, 4]],
            |arr| {
                let lo = arr.iter().copied().min().unwrap_or(0);
                let hi = arr.iter().copied().max().unwrap_or(0);
                hi - lo
            },
        );
        assert_solves(&problem, "range");
    }

    #[test]
    fn utbus_solves_second_max() {
        let problem = array_problem(
            "second_max",
            "fn second_max(arr: [i64]) -> i64",
            &[&[3, 1, 4, 1, 5], &[2, 8, 3], &[7, 7, 2, 9], &[1, 3]],
            &[&[5, 10, 8], &[4, 4, 4]],
            |arr| DualAccum::SecondMax.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "second_max");
    }

    #[test]
    fn utbus_solves_max_pair_diff() {
        let problem = array_problem(
            "max_pair_diff",
            "fn max_pair_diff(arr: [i64]) -> i64",
            &[&[10, 3, 8], &[5, 5], &[1, 2, 4, 0], &[-3, 1]],
            &[&[9, 1, 1], &[0, 0, 5]],
            |arr| PairwiseScan::MaxAbsDiff.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "max_abs_diff");
    }

    #[test]
    fn utbus_solves_count_increases() {
        let problem = array_problem(
            "count_increases",
            "fn count_increases(arr: [i64]) -> i64",
            &[&[1, 3, 2, 5], &[5, 4, 3], &[1, 2, 3, 4], &[7]],
            &[&[0, 0, 1], &[2, 1, 3, 3]],
            |arr| PairwiseScan::CountIncreases.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "count_increases");
    }

    #[test]
    fn utbus_solves_strictly_increasing() {
        let problem = array_problem(
            "strictly_increasing",
            "fn strictly_increasing(arr: [i64]) -> i64",
            &[&[1, 2, 4], &[1, 2, 2], &[5], &[3, 1], &[0, 1, 2, 3]],
            &[&[9, 10], &[4, 4]],
            |arr| PairwiseScan::StrictlyIncreasing.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "strictly_increasing");
    }

    #[test]
    fn utbus_solves_stock_profit() {
        let problem = array_problem(
            "max_stock_profit",
            "fn max_stock_profit(arr: [i64]) -> i64",
            &[&[7, 1, 5, 3, 6, 4], &[7, 6, 4, 3, 1], &[1, 2], &[2, 4, 1, 7]],
            &[&[3, 3, 3], &[1, 5, 2, 8]],
            |arr| DualAccum::StockProfit.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "stock_profit");
    }

    #[test]
    fn utbus_solves_longest_plateau() {
        let problem = array_problem(
            "longest_plateau",
            "fn longest_plateau(arr: [i64]) -> i64",
            &[&[1, 1, 2, 2, 2, 1], &[5, 5, 5, 5], &[1, 2, 3], &[3, 3, 1, 1, 1, 2]],
            &[&[7, 7, 3, 3], &[1]],
            |arr| PairwiseScan::LongestPlateau.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "longest_plateau");
    }

    #[test]
    fn utbus_solves_is_sorted_nondecreasing() {
        let problem = array_problem(
            "is_sorted",
            "fn is_sorted(arr: [i64]) -> i64",
            &[&[1, 2, 3], &[1, 2, 2], &[3, 1], &[5], &[4, 4, 4]],
            &[&[0, 1, 1], &[2, 1]],
            |arr| PairwiseScan::NonDecreasing.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "non_decreasing");
    }

    #[test]
    fn utbus_solves_prefix_max_sum() {
        let problem = array_problem(
            "prefix_max_sum",
            "fn prefix_max_sum(arr: [i64]) -> i64",
            &[&[1, 3, 2, 5], &[5, 4, 3], &[1, 1, 1], &[2, 5, 3, 8]],
            &[&[3, 1, 4, 2], &[7]],
            |arr| DualAccum::PrefixMaxSum.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "prefix_max_sum");
    }

    #[test]
    fn utbus_solves_count_zeros() {
        let problem = array_problem(
            "count_zeros",
            "fn count_zeros(arr: [i64]) -> i64",
            &[&[0, 1, 0, 2], &[1, 2, 3], &[0, 0, 0], &[5, 0]],
            &[&[0, 1, 0], &[7, 8]],
            |arr| arr.iter().filter(|&&x| x == 0).count() as i64,
        );
        assert_solves(&problem, "count");
    }

    #[test]
    fn utbus_solves_max_subarray_sum() {
        let problem = array_problem(
            "max_subarray_sum",
            "fn max_subarray_sum(arr: [i64]) -> i64",
            &[&[1, -2, 3, 4, -1], &[-2, -3, -1], &[5], &[2, -1, 2, -1, 3]],
            &[&[1, 2, 3], &[-5, 10, -3]],
            |arr| DualAccum::MaxSubarraySum.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "max_subarray_sum");
    }

    #[test]
    fn utbus_solves_sum_abs_diff() {
        let problem = array_problem(
            "sum_abs_diff",
            "fn sum_abs_diff(arr: [i64]) -> i64",
            &[&[1, 4, 2], &[5, 5], &[10, 3, 8], &[0]],
            &[&[1, 2, 4], &[9, 1]],
            |arr| PairwiseScan::SumAbsDiff.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "sum_abs_diff");
    }

    #[test]
    fn utbus_solves_sum_even_indices() {
        let problem = array_problem(
            "sum_even_indices",
            "fn sum_even_indices(arr: [i64]) -> i64",
            &[&[1, 2, 3, 4], &[5], &[10, 20, 30], &[1, 1, 1, 1, 1]],
            &[&[2, 4, 6, 8], &[7, 8]],
            |arr| IndexScan::SumEvenIndices.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "sum_even_indices");
    }

    #[test]
    fn utbus_solves_count_peaks() {
        let problem = array_problem(
            "count_peaks",
            "fn count_peaks(arr: [i64]) -> i64",
            &[&[1, 3, 2, 5, 1], &[1, 2, 3], &[5, 1, 5], &[1]],
            &[&[2, 5, 2, 5, 2], &[9, 1, 9]],
            |arr| IndexScan::CountPeaks.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "count_peaks");
    }

    #[test]
    fn utbus_solves_count_distinct() {
        let problem = array_problem(
            "count_distinct",
            "fn count_distinct(arr: [i64]) -> i64",
            &[&[1, 2, 1, 3], &[5, 5, 5], &[1, 2, 3, 4], &[]],
            &[&[7, 8, 7], &[0, 0, 1]],
            |arr| IndexScan::CountDistinct.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "count_distinct");
    }

    #[test]
    fn utbus_solves_min_subarray_sum() {
        let problem = array_problem(
            "min_subarray_sum",
            "fn min_subarray_sum(arr: [i64]) -> i64",
            &[&[1, -2, 3, -4], &[2, 3, 4], &[-1], &[5, -10, 3]],
            &[&[1, 2, -9], &[-2, -3]],
            |arr| DualAccum::MinSubarraySum.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "min_subarray_sum");
    }

    #[test]
    fn utbus_solves_count_decreases() {
        let problem = array_problem(
            "count_decreases",
            "fn count_decreases(arr: [i64]) -> i64",
            &[&[5, 3, 4, 1], &[1, 2, 3], &[9, 8, 7], &[4]],
            &[&[3, 1, 2], &[10, 10, 9]],
            |arr| PairwiseScan::CountDecreases.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "count_decreases");
    }

    #[test]
    fn utbus_solves_count_valleys() {
        let problem = array_problem(
            "count_valleys",
            "fn count_valleys(arr: [i64]) -> i64",
            &[&[3, 1, 4, 0, 2], &[1, 2, 3], &[5, 1, 5], &[1]],
            &[&[4, 1, 4, 1, 4], &[9, 2, 9]],
            |arr| IndexScan::CountValleys.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "count_valleys");
    }

    #[test]
    fn utbus_solves_argmax() {
        let problem = array_problem(
            "argmax",
            "fn argmax(arr: [i64]) -> i64",
            &[&[1, 5, 3], &[9], &[2, 2, 8, 2], &[-1, -5, -2]],
            &[&[0, 1, 0], &[7, 3, 7]],
            |arr| IndexScan::ArgMax.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "argmax");
    }

    #[test]
    fn utbus_solves_kth_smallest() {
        let problem = array_k_problem(
            "kth_smallest",
            "fn kth_smallest(arr: [i64], k: i64) -> i64",
            &[
                (&[3, 1, 4, 1, 5], 2, 1),
                (&[9, 2, 7], 1, 2),
                (&[5, 4, 3, 2, 1], 3, 3),
                (&[10], 1, 10),
            ],
            &[(&[8, 1, 6], 2, 6), (&[4, 4, 4], 1, 4)],
        );
        assert_solves(&problem, "kth_smallest");
    }

    #[test]
    fn utbus_solves_first_index_of() {
        let problem = array_k_problem(
            "first_index_of",
            "fn first_index_of(arr: [i64], k: i64) -> i64",
            &[
                (&[1, 5, 3, 5], 5, 1),
                (&[0, 0, 1], 1, 2),
                (&[4, 4, 4], 7, -1),
                (&[], 1, -1),
            ],
            &[(&[2, 3, 2], 2, 0), (&[9], 9, 0)],
        );
        assert_solves(&problem, "first_index_of");
    }

    #[test]
    fn utbus_solves_median() {
        let problem = array_problem(
            "median",
            "fn median(arr: [i64]) -> i64",
            &[&[1, 100, 2], &[10, 1, 5], &[0, 8, 4], &[7]],
            &[&[3, 1, 2], &[9, 0, 5, 1]],
            |arr| DualAccum::Median.eval(arr).unwrap_or(0),
        );
        assert_solves(&problem, "median");
    }

    #[test]
    fn index_scan_eval_helpers() {
        assert_eq!(IndexScan::SumEvenIndices.eval(&[1, 2, 3, 4]), Some(4));
        assert_eq!(IndexScan::SumOddIndices.eval(&[1, 2, 3, 4]), Some(6));
        assert_eq!(IndexScan::ProductEvenIndices.eval(&[2, 9, 3, 8]), Some(6));
        assert_eq!(IndexScan::ProductOddIndices.eval(&[2, 9, 3, 8]), Some(72));
        assert_eq!(IndexScan::CountDistinct.eval(&[1, 2, 1, 3]), Some(3));
        assert_eq!(IndexScan::CountDistinct.eval(&[]), Some(0));
        assert_eq!(IndexScan::CountValleys.eval(&[3, 1, 4, 0, 2]), Some(2));
        assert_eq!(IndexScan::ArgMax.eval(&[1, 5, 3]), Some(1));
        assert_eq!(IndexScan::ArgMin.eval(&[1, 5, 3]), Some(0));
        assert_eq!(IndexScan::First.eval(&[7, 8]), Some(7));
        assert_eq!(IndexScan::Last.eval(&[7, 8]), Some(8));
        assert_eq!(IndexScan::Mode.eval(&[1, 2, 2, 3, 2]), Some(2));
        assert_eq!(IndexScan::Mode.eval(&[5, 5, 1, 1]), Some(5));
        assert_eq!(DualAccum::MinSubarraySum.eval(&[1, -2, 3, -4]), Some(-4));
        assert_eq!(DualAccum::SecondMin.eval(&[3, 1, 4, 1, 5]), Some(1));
        assert_eq!(PairwiseScan::CountDecreases.eval(&[5, 3, 4, 1]), Some(2));
        assert_eq!(PairwiseScan::StrictlyDecreasing.eval(&[5, 3, 1]), Some(1));
        assert_eq!(PairwiseScan::NonIncreasing.eval(&[5, 5, 3]), Some(1));
        assert_eq!(
            PairwiseScan::LongestIncreasingRun.eval(&[1, 2, 0, 3, 4, 5, 1]),
            Some(4)
        );
        assert_eq!(
            PairwiseScan::LongestDecreasingRun.eval(&[5, 4, 3, 0, 2, 1]),
            Some(4)
        );
        assert_eq!(KClosed::KthSmallest.eval(&[3, 1, 4], 2), Some(3));
        assert_eq!(KClosed::KthLargest.eval(&[3, 1, 4], 1), Some(4));
        assert_eq!(KClosed::FirstIndexOf.eval(&[1, 5, 5], 5), Some(1));
        assert_eq!(KClosed::LastIndexOf.eval(&[1, 5, 5], 5), Some(2));
        assert_eq!(KClosed::FirstIndexOf.eval(&[1, 2], 9), Some(-1));
        assert_eq!(KClosed::KthFromEnd.eval(&[10, 20, 30, 40], 1), Some(40));
        assert_eq!(KClosed::KthFromEnd.eval(&[10, 20, 30, 40], 2), Some(30));
        assert_eq!(DualAccum::Median.eval(&[1, 100, 2]), Some(2));
        assert_eq!(DualAccum::Median.eval(&[10, 1, 5, 0]), Some(5));
        assert_eq!(DualAccum::GcdAll.eval(&[12, 18, 30]), Some(6));
        assert_eq!(DualAccum::GcdAll.eval(&[7, 14, 21]), Some(7));
        assert_eq!(DualAccum::LcmAll.eval(&[2, 3, 4]), Some(12));
        assert_eq!(DualAccum::LcmAll.eval(&[6, 9]), Some(18));
        assert_eq!(DualAccum::MeanTrunc.eval(&[1, 2, 3]), Some(2));
        assert_eq!(DualAccum::MeanTrunc.eval(&[10, 20, 30, 40]), Some(25));
        assert_eq!(DualAccum::SumSquares.eval(&[1, 2, 3]), Some(14));
        assert_eq!(DualAccum::SumSquares.eval(&[-2, 3]), Some(13));
        assert_eq!(DualAccum::AbsSum.eval(&[-1, 2, -3]), Some(6));
        assert_eq!(DualAccum::AbsSum.eval(&[5, 0, -2]), Some(7));
        assert_eq!(DualAccum::MaxAbs.eval(&[-1, 2, -5]), Some(5));
        assert_eq!(DualAccum::MaxAbs.eval(&[3, -2]), Some(3));
        assert_eq!(DualAccum::MinPositive.eval(&[-2, 5, 3, 0]), Some(3));
        assert_eq!(DualAccum::MinPositive.eval(&[-1, 0]), Some(0));
        assert_eq!(DualAccum::CountNegatives.eval(&[-1, 2, -3, 0]), Some(2));
        assert_eq!(DualAccum::CountEvens.eval(&[1, 2, 3, 4, 0]), Some(3));
        assert_eq!(DualAccum::SumPositives.eval(&[-1, 2, 3, -4]), Some(5));
        assert_eq!(DualAccum::SumNegatives.eval(&[-1, 2, 3, -4]), Some(-5));
        assert_eq!(DualAccum::CountOdds.eval(&[1, 2, 3, 4, 0]), Some(2));
        assert_eq!(DualAccum::AnyZero.eval(&[1, 0, -1]), Some(1));
        assert_eq!(DualAccum::AnyZero.eval(&[1, 2]), Some(0));
        assert_eq!(DualAccum::AllPositive.eval(&[1, 2, 3]), Some(1));
        assert_eq!(DualAccum::AllPositive.eval(&[1, 0, 3]), Some(0));
        assert_eq!(DualAccum::AllNegative.eval(&[-1, -2]), Some(1));
        assert_eq!(DualAccum::CountZeros.eval(&[0, 1, 0, 0]), Some(3));
        assert_eq!(DualAccum::HasDuplicate.eval(&[1, 2, 1]), Some(1));
        assert_eq!(DualAccum::HasDuplicate.eval(&[1, 2, 3]), Some(0));
        assert_eq!(DualAccum::MaxNegative.eval(&[-5, -1, 3]), Some(-1));
        assert_eq!(DualAccum::MaxNegative.eval(&[1, 2]), Some(0));
        assert_eq!(DualAccum::SumEvenValues.eval(&[1, 2, 3, 4]), Some(6));
        assert_eq!(DualAccum::SumOddValues.eval(&[1, 2, 3, 4]), Some(4));
        assert_eq!(DualAccum::Len.eval(&[]), Some(0));
        assert_eq!(DualAccum::IsEmpty.eval(&[]), Some(1));
        assert_eq!(DualAccum::IsEmpty.eval(&[1]), Some(0));
        assert_eq!(PairwiseScan::MaxIncrease.eval(&[1, 5, 2, 9]), Some(7));
        assert_eq!(PairwiseScan::MaxDecrease.eval(&[9, 2, 8, 1]), Some(7));
        assert_eq!(
            PairwiseScan::LongestNonDecreasingRun.eval(&[1, 2, 2, 0, 3, 4]),
            Some(3)
        );
        assert_eq!(IndexScan::Middle.eval(&[1, 2, 3]), Some(2));
        assert_eq!(IndexScan::Middle.eval(&[1, 2, 3, 4]), Some(3));
        assert_eq!(IndexScan::Second.eval(&[10, 20, 30]), Some(20));
        assert_eq!(IndexScan::SecondLast.eval(&[10, 20, 30]), Some(20));
        assert_eq!(KClosed::ElementAt.eval(&[10, 20, 30], 1), Some(20));
        assert_eq!(KClosed::ElementAt.eval(&[10, 20], 5), None);
        assert_eq!(DualAccum::AllNonNegative.eval(&[0, 1, 2]), Some(1));
        assert_eq!(DualAccum::AllNonNegative.eval(&[0, -1]), Some(0));
        assert_eq!(DualAccum::CountNonZeros.eval(&[0, 1, 0, 2]), Some(2));
        assert_eq!(DualAccum::AlternatingSum.eval(&[10, 3, 2, 1]), Some(8));
        assert_eq!(DualAccum::ProductPositives.eval(&[-1, 2, 3, 0]), Some(6));
        assert_eq!(DualAccum::ProductPositives.eval(&[-1, 0]), Some(1));
        assert_eq!(
            PairwiseScan::LongestNonIncreasingRun.eval(&[5, 4, 4, 1, 9, 8]),
            Some(4)
        );
        assert_eq!(IndexScan::MaxEvenIndices.eval(&[1, 9, 3, 8, 2]), Some(3));
        assert_eq!(IndexScan::MinEvenIndices.eval(&[5, 9, 1, 8]), Some(1));
        assert_eq!(IndexScan::MaxOddIndices.eval(&[1, 9, 3, 8]), Some(9));
        assert_eq!(IndexScan::MinOddIndices.eval(&[1, 9, 3, 2]), Some(2));
        assert_eq!(KClosed::CountEqK.eval(&[1, 5, 5, 2], 5), Some(2));
        assert_eq!(KClosed::SumGtK.eval(&[1, 5, 3, 2], 2), Some(8));
        assert_eq!(KClosed::CountGtK.eval(&[1, 5, 3, 2], 2), Some(2));
        assert_eq!(KClosed::SumLtK.eval(&[1, 5, 3, 2], 3), Some(3));
        assert_eq!(DualAccum::MeanAbsTrunc.eval(&[-2, 4, -6]), Some(4));
        assert_eq!(PairwiseScan::SumIncreases.eval(&[1, 5, 2, 9]), Some(11));
        assert_eq!(PairwiseScan::SumDecreases.eval(&[9, 2, 8, 1]), Some(14));
        assert_eq!(DualAccum::FirstPositive.eval(&[-2, 0, 5, 3]), Some(5));
        assert_eq!(DualAccum::LastPositive.eval(&[-2, 0, 5, 3]), Some(3));
        assert_eq!(DualAccum::FirstNegative.eval(&[2, -4, -1]), Some(-4));
        assert_eq!(DualAccum::FirstPositive.eval(&[-1, 0]), Some(0));
        assert_eq!(DualAccum::LastNegative.eval(&[2, -4, -1, 5]), Some(-1));
        assert_eq!(KClosed::CountLtK.eval(&[1, 5, 3, 2], 3), Some(2));
        assert_eq!(KClosed::SumEqK.eval(&[2, 5, 2, 2], 2), Some(6));
        assert_eq!(IndexScan::ArgMaxAbs.eval(&[1, -9, 3]), Some(1));
        assert_eq!(IndexScan::ArgMaxAbs.eval(&[-2, 5, -8]), Some(2));
        assert_eq!(DualAccum::MaxPositive.eval(&[-2, 5, 3, 0]), Some(5));
        assert_eq!(DualAccum::MaxPositive.eval(&[-1, 0]), Some(0));
        assert_eq!(DualAccum::MinNegative.eval(&[-2, -5, 3]), Some(-5));
        assert_eq!(DualAccum::MinNegative.eval(&[1, 0]), Some(0));
        assert_eq!(DualAccum::ProductNegatives.eval(&[-2, -3, 4]), Some(6));
        assert_eq!(DualAccum::SumCubes.eval(&[1, 2, -1]), Some(8));
        assert_eq!(DualAccum::CountGtMean.eval(&[1, 2, 3, 10]), Some(1));
        assert_eq!(PairwiseScan::CountPlateaus.eval(&[1, 1, 2, 2, 2, 3]), Some(3));
        assert_eq!(PairwiseScan::IsZigZag.eval(&[1, 3, 2, 5, 0]), Some(1));
        assert_eq!(PairwiseScan::IsZigZag.eval(&[1, 2, 3]), Some(0));
        assert_eq!(IndexScan::ArgMinAbs.eval(&[5, -1, 3]), Some(1));
        assert_eq!(KClosed::CountNeK.eval(&[1, 5, 5, 2], 5), Some(2));
        assert_eq!(KClosed::MaxLtK.eval(&[1, 5, 3, 2], 4), Some(3));
        assert_eq!(DualAccum::CountLtMean.eval(&[1, 2, 3, 10]), Some(3));
        assert_eq!(DualAccum::IsPalindrome.eval(&[1, 2, 1]), Some(1));
        assert_eq!(DualAccum::IsPalindrome.eval(&[1, 2, 3]), Some(0));
        assert_eq!(DualAccum::ProductEvens.eval(&[1, 2, 3, 4]), Some(8));
        assert_eq!(PairwiseScan::MinIncrease.eval(&[1, 5, 2, 9]), Some(4));
        assert_eq!(KClosed::MinGtK.eval(&[1, 5, 3, 2], 2), Some(3));
        assert_eq!(DualAccum::ProductOdds.eval(&[1, 2, 3, 4]), Some(3));
        assert_eq!(PairwiseScan::MinDecrease.eval(&[9, 2, 8, 1]), Some(7));
        assert_eq!(IndexScan::SumAbsEvenIndices.eval(&[-1, 2, -3, 4]), Some(4));
        assert_eq!(IndexScan::SumAbsOddIndices.eval(&[-1, 2, -3, 4]), Some(6));
        assert_eq!(KClosed::SumNeK.eval(&[1, 5, 5, 2], 5), Some(3));
        assert_eq!(DualAccum::AllNonPositive.eval(&[-1, 0, -2]), Some(1));
        assert_eq!(DualAccum::DotIndex.eval(&[10, 20, 30]), Some(80));
        assert_eq!(PairwiseScan::MeanAbsDiffTrunc.eval(&[1, 4, 2]), Some(2));
        assert_eq!(KClosed::MaxGtK.eval(&[1, 5, 3, 2], 2), Some(5));
        assert_eq!(DualAccum::SumSqDiffMean.eval(&[1, 2, 3]), Some(2));
        assert_eq!(IndexScan::CountEvenIndices.eval(&[1, 2, 3, 4, 5]), Some(3));
        assert_eq!(IndexScan::CountOddIndices.eval(&[1, 2, 3, 4, 5]), Some(2));
        assert_eq!(DualAccum::AnyNonZero.eval(&[0, 0, 1]), Some(1));
        assert_eq!(DualAccum::AnyNonZero.eval(&[0, 0]), Some(0));
        assert_eq!(PairwiseScan::CountSignChanges.eval(&[1, 3, 2, 5, 0]), Some(3));
        assert_eq!(KClosed::MinLtK.eval(&[1, 5, 3, 2], 4), Some(1));
        assert_eq!(DualAccum::XorAll.eval(&[1, 2, 3]), Some(0));
        assert_eq!(DualAccum::XorAll.eval(&[7, 1]), Some(6));
        assert_eq!(DualAccum::ProductNonZeros.eval(&[0, 2, 3, 0]), Some(6));
        assert_eq!(DualAccum::OrAll.eval(&[1, 2, 4]), Some(7));
        assert_eq!(PairwiseScan::SumSqDiff.eval(&[1, 4, 2]), Some(13));
        assert_eq!(IndexScan::XorEvenIndices.eval(&[1, 2, 4, 8]), Some(5));
        assert_eq!(IndexScan::XorOddIndices.eval(&[1, 2, 4, 8]), Some(10));
        assert_eq!(KClosed::CountGeK.eval(&[1, 5, 3, 2], 3), Some(2));
        assert_eq!(KClosed::CountLeK.eval(&[1, 5, 3, 2], 3), Some(3));
        assert_eq!(KClosed::SumGeK.eval(&[1, 5, 3, 2], 3), Some(8));
        assert_eq!(KClosed::SumLeK.eval(&[1, 5, 3, 2], 3), Some(6));
        assert_eq!(DualAccum::AndAll.eval(&[7, 3, 1]), Some(1));
        assert_eq!(DualAccum::AndAll.eval(&[]), Some(-1));
        assert_eq!(PairwiseScan::MeanSqDiffTrunc.eval(&[1, 4, 2]), Some(6));
        assert_eq!(KClosed::FirstGeK.eval(&[1, 5, 3, 2], 3), Some(1));
        assert_eq!(KClosed::FirstLeK.eval(&[5, 4, 1, 2], 2), Some(2));
        assert_eq!(DualAccum::CountEqMean.eval(&[1, 2, 3, 2]), Some(2));
        assert_eq!(PairwiseScan::FirstIncreaseIdx.eval(&[5, 4, 1, 3]), Some(3));
        assert_eq!(PairwiseScan::FirstIncreaseIdx.eval(&[5, 4, 1]), Some(-1));
        assert_eq!(KClosed::LastGeK.eval(&[1, 5, 3, 2], 3), Some(2));
        assert_eq!(KClosed::LastLeK.eval(&[5, 4, 1, 2], 2), Some(3));
        assert_eq!(DualAccum::ProductAbs.eval(&[-2, 3, -4]), Some(24));
        assert_eq!(PairwiseScan::FirstDecreaseIdx.eval(&[1, 3, 2, 5]), Some(2));
        assert_eq!(IndexScan::OrEvenIndices.eval(&[1, 2, 4, 8]), Some(5));
        assert_eq!(IndexScan::OrOddIndices.eval(&[1, 2, 4, 8]), Some(10));
        assert_eq!(KClosed::SumAbsGtK.eval(&[-5, 2, 4], 1), Some(6));
        assert_eq!(DualAccum::CountNonNegatives.eval(&[-1, 0, 2, -3]), Some(2));
        assert_eq!(PairwiseScan::LastIncreaseIdx.eval(&[1, 3, 2, 5]), Some(3));
        assert_eq!(IndexScan::AndEvenIndices.eval(&[7, 2, 3, 8]), Some(3));
        assert_eq!(IndexScan::AndOddIndices.eval(&[1, 7, 4, 3]), Some(3));
        assert_eq!(KClosed::SumAbsLtK.eval(&[-5, 2, 4], 3), Some(7));
        assert_eq!(DualAccum::CountPositives.eval(&[-1, 0, 2, 3]), Some(2));
        assert_eq!(PairwiseScan::LastDecreaseIdx.eval(&[1, 3, 2, 5, 0]), Some(4));
        assert_eq!(IndexScan::ProductAbsEvenIndices.eval(&[-2, 9, -3, 8]), Some(6));
        assert_eq!(IndexScan::ProductAbsOddIndices.eval(&[-2, 9, -3, 8]), Some(72));
        assert_eq!(KClosed::CountAbsGtK.eval(&[-5, 2, 4], 2), Some(2));
        assert_eq!(DualAccum::MaxEvenValue.eval(&[1, 8, 3, 4]), Some(8));
        assert_eq!(DualAccum::MaxOddValue.eval(&[1, 8, 3, 4]), Some(3));
        assert_eq!(IndexScan::SumSquaresEvenIndices.eval(&[2, 9, 3, 8]), Some(13));
        assert_eq!(IndexScan::SumSquaresOddIndices.eval(&[2, 9, 3, 8]), Some(145));
        assert_eq!(KClosed::CountAbsLtK.eval(&[-5, 2, 4], 3), Some(1));
        assert_eq!(DualAccum::MinEvenValue.eval(&[1, 8, 3, 4]), Some(4));
        assert_eq!(DualAccum::MinOddValue.eval(&[1, 8, 3, 4]), Some(1));
        assert_eq!(IndexScan::MeanEvenTrunc.eval(&[2, 9, 4, 8]), Some(3));
        assert_eq!(IndexScan::MeanOddTrunc.eval(&[2, 9, 4, 8]), Some(8));
        assert_eq!(KClosed::SumAbsGeK.eval(&[-5, 2, 4], 2), Some(6));
        assert_eq!(DualAccum::AbsRange.eval(&[-5, 2, -1]), Some(4));
        assert_eq!(IndexScan::CountPositiveEvenIndices.eval(&[-1, 2, 3, -4]), Some(1));
        assert_eq!(IndexScan::CountPositiveOddIndices.eval(&[-1, 2, 3, -4]), Some(1));
        assert_eq!(KClosed::SumAbsLeK.eval(&[-5, 2, 4], 2), Some(7));
        assert_eq!(IndexScan::CountNegativeEvenIndices.eval(&[-1, 2, 3, -4]), Some(1));
        assert_eq!(IndexScan::CountNegativeOddIndices.eval(&[-1, 2, 3, -4]), Some(1));
        assert_eq!(KClosed::CountAbsEqK.eval(&[-5, 2, 5, 4], 5), Some(2));
        assert_eq!(IndexScan::SumPositiveEvenIndices.eval(&[-1, 2, 3, -4]), Some(3));
        assert_eq!(IndexScan::SumPositiveOddIndices.eval(&[-1, 2, 3, -4]), Some(2));
        assert_eq!(KClosed::FirstAbsGeK.eval(&[1, -5, 2], 4), Some(1));
        assert_eq!(IndexScan::SumNegativeEvenIndices.eval(&[-1, 2, 3, -4]), Some(-1));
        assert_eq!(IndexScan::SumNegativeOddIndices.eval(&[-1, 2, 3, -4]), Some(-4));
        assert_eq!(KClosed::LastAbsGeK.eval(&[5, 1, -5, 2], 4), Some(2));
        assert_eq!(DualAccum::ProductNonNegatives.eval(&[-2, 3, 0, 4]), Some(0));
        assert_eq!(IndexScan::CountZeroEvenIndices.eval(&[0, 1, 2, 0]), Some(1));
        assert_eq!(IndexScan::CountZeroOddIndices.eval(&[0, 1, 2, 0]), Some(1));
        assert_eq!(KClosed::FirstAbsEqK.eval(&[1, -5, 5], 5), Some(1));
        assert_eq!(IndexScan::MaxAbsEvenIndices.eval(&[-3, 9, 2, 8]), Some(3));
        assert_eq!(IndexScan::MaxAbsOddIndices.eval(&[-3, 9, 2, 8]), Some(9));
        assert_eq!(KClosed::LastAbsEqK.eval(&[5, 1, -5, 2], 5), Some(2));
        assert_eq!(IndexScan::MinAbsEvenIndices.eval(&[-3, 9, 2, 8]), Some(2));
        assert_eq!(IndexScan::MinAbsOddIndices.eval(&[-3, 9, 2, 8]), Some(8));
        assert_eq!(KClosed::CountAbsGeK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(KClosed::CountAbsLeK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(IndexScan::MeanAbsEvenTrunc.eval(&[-4, 9, 2, 8]), Some(3));
        assert_eq!(IndexScan::MeanAbsOddTrunc.eval(&[-4, 9, 2, 8]), Some(8));
        assert_eq!(KClosed::FirstAbsLeK.eval(&[5, 1, -3, 2], 2), Some(1));
        assert_eq!(KClosed::LastAbsLeK.eval(&[5, 1, -3, 2], 2), Some(3));
        assert_eq!(DualAccum::SumNonNegatives.eval(&[-2, 3, 0, 4]), Some(7));
        assert_eq!(IndexScan::CountNonZeroEvenIndices.eval(&[0, 1, 2, 0]), Some(1));
        assert_eq!(IndexScan::CountNonZeroOddIndices.eval(&[0, 1, 2, 0]), Some(1));
        assert_eq!(KClosed::SumAbsEqK.eval(&[-5, 2, 5, 4], 5), Some(10));
        assert_eq!(DualAccum::MinAbs.eval(&[-3, 9, 2]), Some(2));
        assert_eq!(IndexScan::ProductNonZeroEvenIndices.eval(&[0, 9, -3, 8]), Some(-3));
        assert_eq!(IndexScan::ProductNonZeroOddIndices.eval(&[0, 9, -3, 8]), Some(72));
        assert_eq!(KClosed::FirstAbsGtK.eval(&[1, -5, 2], 2), Some(1));
        assert_eq!(KClosed::LastAbsGtK.eval(&[5, 1, -5, 2], 2), Some(2));
        assert_eq!(DualAccum::SumNonPositives.eval(&[-2, 3, 0, 4]), Some(-2));
        assert_eq!(IndexScan::SumNonZeroEvenIndices.eval(&[0, 9, -3, 8]), Some(-3));
        assert_eq!(IndexScan::SumNonZeroOddIndices.eval(&[0, 9, -3, 8]), Some(17));
        assert_eq!(KClosed::CountAbsNeK.eval(&[-5, 2, 5, 4], 5), Some(2));
        assert_eq!(DualAccum::CountNonPositives.eval(&[-2, 3, 0, 4]), Some(2));
        assert_eq!(IndexScan::MaxNonZeroEvenIndices.eval(&[0, 9, -3, 8]), Some(-3));
        assert_eq!(IndexScan::MaxNonZeroOddIndices.eval(&[0, 9, -3, 8]), Some(9));
        assert_eq!(KClosed::SumAbsNeK.eval(&[-5, 2, 5, 4], 5), Some(6));
        assert_eq!(DualAccum::ProductNonPositives.eval(&[-2, 3, 0, 4]), Some(0));
        assert_eq!(IndexScan::MinNonZeroEvenIndices.eval(&[5, 9, -3, 8]), Some(-3));
        assert_eq!(IndexScan::MinNonZeroOddIndices.eval(&[5, 9, -3, 8]), Some(8));
        assert_eq!(KClosed::FirstAbsLtK.eval(&[5, 1, -3, 2], 2), Some(1));
        assert_eq!(KClosed::LastAbsLtK.eval(&[5, 1, -3, 2], 2), Some(1));
        assert_eq!(IndexScan::CountEvenValueEvenIndices.eval(&[2, 9, 3, 8]), Some(1));
        assert_eq!(IndexScan::CountEvenValueOddIndices.eval(&[2, 9, 3, 8]), Some(1));
        assert_eq!(KClosed::MaxAbsLtK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(IndexScan::CountOddValueEvenIndices.eval(&[2, 9, 3, 8]), Some(1));
        assert_eq!(IndexScan::CountOddValueOddIndices.eval(&[2, 9, 3, 8]), Some(1));
        assert_eq!(KClosed::MinAbsGtK.eval(&[-5, 2, 4], 2), Some(4));
        assert_eq!(IndexScan::SumEvenValueEvenIndices.eval(&[2, 9, 4, 8]), Some(6));
        assert_eq!(IndexScan::SumEvenValueOddIndices.eval(&[2, 9, 4, 8]), Some(8));
        assert_eq!(KClosed::MaxAbsGtK.eval(&[-5, 2, 4], 2), Some(5));
        assert_eq!(IndexScan::SumOddValueEvenIndices.eval(&[2, 9, 3, 8]), Some(3));
        assert_eq!(IndexScan::SumOddValueOddIndices.eval(&[2, 9, 3, 8]), Some(9));
        assert_eq!(KClosed::MinAbsLtK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(IndexScan::ProductEvenValueEvenIndices.eval(&[2, 9, 4, 8]), Some(8));
        assert_eq!(IndexScan::ProductEvenValueOddIndices.eval(&[2, 9, 4, 8]), Some(8));
        assert_eq!(KClosed::FirstAbsNeK.eval(&[5, -5, 2], 5), Some(2));
        assert_eq!(IndexScan::ProductOddValueEvenIndices.eval(&[3, 8, 5, 2]), Some(15));
        assert_eq!(IndexScan::ProductOddValueOddIndices.eval(&[3, 9, 5, 7]), Some(63));
        assert_eq!(KClosed::LastAbsNeK.eval(&[5, 2, -5], 5), Some(1));
        assert_eq!(DualAccum::SumAbsEvens.eval(&[-4, 3, 2]), Some(6));
        assert_eq!(DualAccum::SumAbsOdds.eval(&[-4, 3, 2]), Some(3));
        assert_eq!(KClosed::SumWhereAbsEqK.eval(&[5, -5, 2], 5), Some(0));
        assert_eq!(DualAccum::ProductAbsEvens.eval(&[-4, 3, 2]), Some(8));
        assert_eq!(DualAccum::ProductAbsOdds.eval(&[-4, 3, 2]), Some(3));
        assert_eq!(KClosed::ProductWhereAbsEqK.eval(&[5, -5, 2], 5), Some(-25));
        assert_eq!(IndexScan::SumAbsEvenValueEvenIndices.eval(&[-4, 9, 2, 8]), Some(6));
        assert_eq!(IndexScan::SumAbsEvenValueOddIndices.eval(&[-4, 9, 2, 8]), Some(8));
        assert_eq!(KClosed::MaxWhereAbsEqK.eval(&[5, -5, 2], 5), Some(5));
        assert_eq!(IndexScan::SumAbsOddValueEvenIndices.eval(&[-3, 8, 5, 2]), Some(8));
        assert_eq!(IndexScan::SumAbsOddValueOddIndices.eval(&[-3, 9, 5, 7]), Some(16));
        assert_eq!(KClosed::MinWhereAbsEqK.eval(&[5, -5, 2], 5), Some(-5));
        assert_eq!(DualAccum::XorAbsAll.eval(&[-3, 5, 1]), Some(7));
        assert_eq!(IndexScan::OrAbsEvenIndices.eval(&[-1, 2, 4, 8]), Some(5));
        assert_eq!(IndexScan::OrAbsOddIndices.eval(&[-1, 2, 4, 8]), Some(10));
        assert_eq!(DualAccum::AndAbsAll.eval(&[-7, 3, 5]), Some(1));
        assert_eq!(IndexScan::AndAbsEvenIndices.eval(&[-7, 2, 3, 8]), Some(3));
        assert_eq!(IndexScan::AndAbsOddIndices.eval(&[-1, 7, 4, 3]), Some(3));
        assert_eq!(DualAccum::OrAbsAll.eval(&[-1, 2, 4]), Some(7));
        assert_eq!(IndexScan::XorAbsEvenIndices.eval(&[-1, 2, 4, 8]), Some(5));
        assert_eq!(IndexScan::XorAbsOddIndices.eval(&[-1, 2, 4, 8]), Some(10));
        assert_eq!(DualAccum::XorAbsEvens.eval(&[-4, 3, 2]), Some(6));
        assert_eq!(DualAccum::XorAbsOdds.eval(&[-4, 3, 2]), Some(3));
        assert_eq!(KClosed::SumWhereAbsNeK.eval(&[5, -5, 2], 5), Some(2));
        assert_eq!(DualAccum::AndAbsEvens.eval(&[-6, 3, 2]), Some(2));
        assert_eq!(DualAccum::AndAbsOdds.eval(&[-7, 3, 2]), Some(3));
        assert_eq!(KClosed::ProductWhereAbsNeK.eval(&[5, -5, 2], 5), Some(2));
        assert_eq!(DualAccum::OrAbsEvens.eval(&[-4, 3, 2]), Some(6));
        assert_eq!(DualAccum::OrAbsOdds.eval(&[-4, 3, 2]), Some(3));
        assert_eq!(KClosed::MaxWhereAbsNeK.eval(&[5, -5, 2], 5), Some(2));
        assert_eq!(DualAccum::SumSquaresEvens.eval(&[-4, 3, 2]), Some(20));
        assert_eq!(DualAccum::SumSquaresOdds.eval(&[-4, 3, 2]), Some(9));
        assert_eq!(KClosed::MinWhereAbsNeK.eval(&[5, -5, 2], 5), Some(2));
        assert_eq!(DualAccum::SumCubesEvens.eval(&[-4, 3, 2]), Some(-56));
        assert_eq!(DualAccum::SumCubesOdds.eval(&[-4, 3, 2]), Some(27));
        assert_eq!(KClosed::SumWhereAbsGtK.eval(&[-5, 2, 4], 2), Some(-1));
        assert_eq!(DualAccum::ProductSquaresEvens.eval(&[-4, 3, 2]), Some(64));
        assert_eq!(DualAccum::ProductSquaresOdds.eval(&[-4, 3, 2]), Some(9));
        assert_eq!(KClosed::SumWhereAbsLtK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::MaxAbsEvens.eval(&[-4, 3, 2]), Some(4));
        assert_eq!(DualAccum::MaxAbsOdds.eval(&[-4, 3, 2]), Some(3));
        assert_eq!(KClosed::ProductWhereAbsGtK.eval(&[-5, 2, 4], 2), Some(-20));
        assert_eq!(DualAccum::MinAbsEvens.eval(&[-4, 3, 2]), Some(2));
        assert_eq!(DualAccum::MinAbsOdds.eval(&[-4, 3, 2]), Some(3));
        assert_eq!(KClosed::ProductWhereAbsLtK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::CountNonZeroEvens.eval(&[-4, 0, 3, 2]), Some(2));
        assert_eq!(DualAccum::CountNonZeroOdds.eval(&[-4, 0, 3, 2]), Some(1));
        assert_eq!(KClosed::MaxWhereAbsGtK.eval(&[-5, 2, 4], 2), Some(4));
        assert_eq!(DualAccum::SumNonZeroEvens.eval(&[-4, 0, 3, 2]), Some(-2));
        assert_eq!(DualAccum::SumNonZeroOdds.eval(&[-4, 0, 3, 2]), Some(3));
        assert_eq!(KClosed::MinWhereAbsGtK.eval(&[-5, 2, 4], 2), Some(-5));
        assert_eq!(DualAccum::ProductNonZeroEvens.eval(&[-4, 0, 3, 2]), Some(-8));
        assert_eq!(DualAccum::ProductNonZeroOdds.eval(&[-4, 0, 3, 2]), Some(3));
        assert_eq!(KClosed::MinWhereAbsLtK.eval(&[-5, 2, 4, 1], 4), Some(1));
        assert_eq!(DualAccum::MeanAbsEvensTrunc.eval(&[-4, 3, 2]), Some(3));
        assert_eq!(DualAccum::MeanAbsOddsTrunc.eval(&[-4, 3, 2]), Some(3));
        assert_eq!(KClosed::MaxWhereAbsGeK.eval(&[-5, 2, 4], 4), Some(4));
        assert_eq!(DualAccum::GcdAbsEvens.eval(&[-4, 6, 3, 2]), Some(2));
        assert_eq!(DualAccum::GcdAbsOdds.eval(&[-4, 6, 3, 9]), Some(3));
        assert_eq!(KClosed::MinWhereAbsGeK.eval(&[-5, 2, 4], 4), Some(-5));
        assert_eq!(DualAccum::LcmAbsEvens.eval(&[-4, 6, 3, 2]), Some(12));
        assert_eq!(DualAccum::LcmAbsOdds.eval(&[-4, 6, 3, 9]), Some(9));
        assert_eq!(KClosed::SumWhereAbsGeK.eval(&[-5, 2, 4], 4), Some(-1));
        assert_eq!(DualAccum::ProductCubesEvens.eval(&[-2, 3, 2]), Some(-64));
        assert_eq!(DualAccum::ProductCubesOdds.eval(&[-2, 3, 1]), Some(27));
        assert_eq!(KClosed::ProductWhereAbsGeK.eval(&[-5, 2, 4], 4), Some(-20));
        assert_eq!(DualAccum::SumAbsCubesEvens.eval(&[-2, 3, 2]), Some(16));
        assert_eq!(DualAccum::SumAbsCubesOdds.eval(&[-2, 3, 1]), Some(28));
        assert_eq!(KClosed::ProductWhereAbsLeK.eval(&[-5, 2, 4], 4), Some(8));
        assert_eq!(DualAccum::ProductAbsCubesEvens.eval(&[-2, 3, 2]), Some(64));
        assert_eq!(DualAccum::ProductAbsCubesOdds.eval(&[-2, 3, 1]), Some(27));
        assert_eq!(KClosed::SumWhereAbsLeK.eval(&[-5, 2, 4], 4), Some(6));
        assert_eq!(DualAccum::SumAbsSquaresEvens.eval(&[-2, 3, 2]), Some(8));
        assert_eq!(DualAccum::SumAbsSquaresOdds.eval(&[-2, 3, 1]), Some(10));
        assert_eq!(KClosed::MaxWhereAbsLeK.eval(&[-5, 2, 4], 4), Some(4));
        assert_eq!(DualAccum::ProductAbsSquaresEvens.eval(&[-2, 3, 2]), Some(16));
        assert_eq!(DualAccum::ProductAbsSquaresOdds.eval(&[-2, 3, 1]), Some(9));
        assert_eq!(KClosed::MinWhereAbsLeK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::MeanAbsSquaresEvensTrunc.eval(&[-2, 3, 2]), Some(4));
        assert_eq!(DualAccum::MeanAbsSquaresOddsTrunc.eval(&[-2, 3, 1]), Some(5));
        assert_eq!(KClosed::CountWhereAbsGeK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::CountPositiveEvens.eval(&[-2, 3, 2, 4]), Some(2));
        assert_eq!(DualAccum::CountPositiveOdds.eval(&[-2, 3, 1]), Some(2));
        assert_eq!(KClosed::CountWhereAbsLeK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::CountNegativeEvens.eval(&[-2, 3, -4, 1]), Some(2));
        assert_eq!(DualAccum::CountNegativeOdds.eval(&[-2, 3, -1]), Some(1));
        assert_eq!(KClosed::FirstWhereAbsGeK.eval(&[-5, 2, 4], 4), Some(-5));
        assert_eq!(DualAccum::SumPositiveEvens.eval(&[-2, 3, 2, 4]), Some(6));
        assert_eq!(DualAccum::SumPositiveOdds.eval(&[-2, 3, 1]), Some(4));
        assert_eq!(KClosed::LastWhereAbsGeK.eval(&[-5, 2, 4], 4), Some(4));
        assert_eq!(DualAccum::SumNegativeEvens.eval(&[-2, 3, -4, 1]), Some(-6));
        assert_eq!(DualAccum::SumNegativeOdds.eval(&[-3, 2, -1]), Some(-4));
        assert_eq!(KClosed::FirstWhereAbsLeK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::ProductPositiveEvens.eval(&[-2, 3, 2, 4]), Some(8));
        assert_eq!(DualAccum::ProductPositiveOdds.eval(&[-2, 3, 1]), Some(3));
        assert_eq!(KClosed::LastWhereAbsLeK.eval(&[-5, 2, 4], 4), Some(4));
        assert_eq!(DualAccum::ProductNegativeEvens.eval(&[-2, 3, -4, 1]), Some(8));
        assert_eq!(DualAccum::ProductNegativeOdds.eval(&[-3, 2, -1]), Some(3));
        assert_eq!(KClosed::FirstWhereAbsEqK.eval(&[-5, 2, 4], 4), Some(4));
        assert_eq!(DualAccum::MaxPositiveEvens.eval(&[-2, 3, 2, 4]), Some(4));
        assert_eq!(DualAccum::MaxPositiveOdds.eval(&[-2, 3, 1, 5]), Some(5));
        assert_eq!(KClosed::LastWhereAbsEqK.eval(&[-5, 4, -4], 4), Some(-4));
        assert_eq!(DualAccum::MinPositiveEvens.eval(&[-2, 3, 2, 4]), Some(2));
        assert_eq!(DualAccum::MinPositiveOdds.eval(&[-2, 3, 1, 5]), Some(1));
        assert_eq!(KClosed::FirstWhereAbsNeK.eval(&[-4, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::MaxNegativeEvens.eval(&[-4, 3, -2, 1]), Some(-2));
        assert_eq!(DualAccum::MaxNegativeOdds.eval(&[-5, 2, -1, -3]), Some(-1));
        assert_eq!(KClosed::LastWhereAbsNeK.eval(&[-4, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::MinNegativeEvens.eval(&[-4, 3, -2, 1]), Some(-4));
        assert_eq!(DualAccum::MinNegativeOdds.eval(&[-5, 2, -1, -3]), Some(-5));
        assert_eq!(KClosed::CountWhereAbsNeK.eval(&[-4, 2, 4], 4), Some(1));
        assert_eq!(DualAccum::MeanPositiveEvensTrunc.eval(&[-2, 3, 2, 4]), Some(3));
        assert_eq!(DualAccum::MeanPositiveOddsTrunc.eval(&[-2, 3, 1, 5]), Some(3));
        assert_eq!(KClosed::FirstIndexWhereAbsGeK.eval(&[-1, 2, 5], 4), Some(2));
        assert_eq!(DualAccum::MeanNegativeEvensTrunc.eval(&[-4, 3, -2, 1]), Some(-3));
        assert_eq!(DualAccum::MeanNegativeOddsTrunc.eval(&[-5, 2, -1, -3]), Some(-3));
        assert_eq!(KClosed::LastIndexWhereAbsGeK.eval(&[-1, 5, 2, 6], 4), Some(3));
        assert_eq!(DualAccum::AllEvenPositive.eval(&[2, 3, 4]), Some(1));
        assert_eq!(DualAccum::AllOddPositive.eval(&[-2, 3, 5]), Some(1));
        assert_eq!(DualAccum::AllEvenPositive.eval(&[-2, 3, 4]), Some(0));
        assert_eq!(KClosed::FirstIndexWhereAbsLeK.eval(&[-5, 2, 4], 4), Some(1));
        assert_eq!(DualAccum::AllEvenNegative.eval(&[-4, 3, -2]), Some(1));
        assert_eq!(DualAccum::AllOddNegative.eval(&[-5, 2, -1]), Some(1));
        assert_eq!(DualAccum::AllEvenNegative.eval(&[-4, 3, 2]), Some(0));
        assert_eq!(KClosed::LastIndexWhereAbsLeK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::AnyEvenPositive.eval(&[2, -3, -4]), Some(1));
        assert_eq!(DualAccum::AnyOddPositive.eval(&[-2, 3, -5]), Some(1));
        assert_eq!(DualAccum::AnyEvenPositive.eval(&[-2, 3, -4]), Some(0));
        assert_eq!(KClosed::FirstIndexWhereAbsEqK.eval(&[-5, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::AnyEvenNegative.eval(&[-2, 3, 4]), Some(1));
        assert_eq!(DualAccum::AnyOddNegative.eval(&[2, -3, 5]), Some(1));
        assert_eq!(DualAccum::AnyEvenNegative.eval(&[2, 3, 4]), Some(0));
        assert_eq!(KClosed::LastIndexWhereAbsEqK.eval(&[-4, 2, 4], 4), Some(2));
        assert_eq!(DualAccum::AnyEvenNonZero.eval(&[0, 3, 2]), Some(1));
        assert_eq!(DualAccum::AnyOddNonZero.eval(&[0, 2, -1]), Some(1));
        assert_eq!(DualAccum::AnyEvenNonZero.eval(&[0, 3, 1]), Some(0));
        assert_eq!(KClosed::FirstIndexWhereAbsNeK.eval(&[-4, 2, 4], 4), Some(1));
        assert_eq!(DualAccum::AllEvenNonZero.eval(&[2, 3, 4]), Some(1));
        assert_eq!(DualAccum::AllOddNonZero.eval(&[2, 3, 5]), Some(1));
        assert_eq!(DualAccum::AllEvenNonZero.eval(&[0, 3, 2]), Some(0));
        assert_eq!(KClosed::LastIndexWhereAbsNeK.eval(&[-4, 2, 4], 4), Some(1));
        assert_eq!(DualAccum::AllEvenNonNegative.eval(&[2, -3, 4]), Some(1));
        assert_eq!(DualAccum::AllOddNonNegative.eval(&[2, 3, -4]), Some(1));
        assert_eq!(DualAccum::AllEvenNonNegative.eval(&[-2, 3, 4]), Some(0));
        assert_eq!(KClosed::FirstIndexWhereAbsGtK.eval(&[-1, 2, 5], 4), Some(2));
        assert_eq!(DualAccum::AllEvenNonPositive.eval(&[-2, 3, 0]), Some(1));
        assert_eq!(DualAccum::AllOddNonPositive.eval(&[2, -3, -5]), Some(1));
        assert_eq!(DualAccum::AllEvenNonPositive.eval(&[2, -3, 0]), Some(0));
        assert_eq!(KClosed::FirstIndexWhereAbsGtK.eval(&[-5, 2, 3], 4), Some(0));
        assert_eq!(DualAccum::AnyEvenNonNegative.eval(&[-2, 3, 4]), Some(1));
        assert_eq!(DualAccum::AnyOddNonNegative.eval(&[-2, -3, 5]), Some(1));
        assert_eq!(DualAccum::AnyEvenNonNegative.eval(&[-2, 3, -4]), Some(0));
        assert_eq!(KClosed::LastIndexWhereAbsGtK.eval(&[-5, 2, 6], 4), Some(2));
        assert_eq!(DualAccum::AnyEvenNonPositive.eval(&[-2, 3, 4]), Some(1));
        assert_eq!(DualAccum::AnyOddNonPositive.eval(&[2, -3, 5]), Some(1));
        assert_eq!(DualAccum::AnyEvenNonPositive.eval(&[2, 3, 4]), Some(0));
        assert_eq!(KClosed::FirstIndexWhereAbsLtK.eval(&[-5, 2, 4], 4), Some(1));
        assert_eq!(DualAccum::MaxEvenNonZero.eval(&[0, 2, 3, -4]), Some(2));
        assert_eq!(DualAccum::MaxOddNonZero.eval(&[0, 2, 3, -5]), Some(3));
        assert_eq!(DualAccum::MaxEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::LastIndexWhereAbsLtK.eval(&[-5, 2, 1], 4), Some(2));
        assert_eq!(DualAccum::MaxEvenNonZero.eval(&[-6, 0, 4, 3]), Some(4));
        assert_eq!(DualAccum::MaxOddNonZero.eval(&[-7, 0, 5, 2]), Some(5));
        assert_eq!(KClosed::CountDivisibleByK.eval(&[2, 3, 4, 6], 2), Some(3));
        assert_eq!(DualAccum::MinEvenNonZero.eval(&[0, 2, -4, 3]), Some(-4));
        assert_eq!(DualAccum::MinOddNonZero.eval(&[0, 5, -3, 2]), Some(-3));
        assert_eq!(DualAccum::MinEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::SumDivisibleByK.eval(&[2, 3, 4, 6], 2), Some(12));
        assert_eq!(DualAccum::MeanEvenNonZeroTrunc.eval(&[0, 2, 4, 3]), Some(3));
        assert_eq!(DualAccum::MeanOddNonZeroTrunc.eval(&[0, 3, 5, 2]), Some(4));
        assert_eq!(DualAccum::MeanEvenNonZeroTrunc.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::ProductDivisibleByK.eval(&[2, 3, 4], 2), Some(8));
        assert_eq!(DualAccum::XorEvenNonZero.eval(&[0, 2, 6, 3]), Some(4));
        assert_eq!(DualAccum::XorOddNonZero.eval(&[0, 1, 5, 2]), Some(4));
        assert_eq!(DualAccum::XorEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::FirstDivisibleByK.eval(&[3, 4, 6], 2), Some(4));
        assert_eq!(DualAccum::OrEvenNonZero.eval(&[0, 2, 4, 3]), Some(6));
        assert_eq!(DualAccum::OrOddNonZero.eval(&[0, 1, 4, 5]), Some(5));
        assert_eq!(DualAccum::OrEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::LastDivisibleByK.eval(&[3, 4, 6], 2), Some(6));
        assert_eq!(DualAccum::AndEvenNonZero.eval(&[0, 6, 4, 3]), Some(4));
        assert_eq!(DualAccum::AndOddNonZero.eval(&[0, 7, 5, 2]), Some(5));
        assert_eq!(DualAccum::AndEvenNonZero.eval(&[0, 1, 3]), Some(-1));
        assert_eq!(KClosed::MaxDivisibleByK.eval(&[3, 8, 4, 6], 2), Some(8));
        assert_eq!(DualAccum::SumAbsEvenNonZero.eval(&[0, -4, 2, 3]), Some(6));
        assert_eq!(DualAccum::SumAbsOddNonZero.eval(&[0, -5, 3, 2]), Some(8));
        assert_eq!(DualAccum::SumAbsEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::MinDivisibleByK.eval(&[3, -8, 4, 6], 2), Some(-8));
        assert_eq!(DualAccum::ProductAbsEvenNonZero.eval(&[0, -4, 2, 3]), Some(8));
        assert_eq!(DualAccum::ProductAbsOddNonZero.eval(&[0, -5, 3, 2]), Some(15));
        assert_eq!(DualAccum::ProductAbsEvenNonZero.eval(&[0, 1, 3]), Some(1));
        assert_eq!(KClosed::FirstIndexDivisibleByK.eval(&[3, 4, 6], 2), Some(1));
        assert_eq!(DualAccum::GcdAbsEvenNonZero.eval(&[0, 12, -18, 5]), Some(6));
        assert_eq!(DualAccum::GcdAbsOddNonZero.eval(&[0, 15, -25, 2]), Some(5));
        assert_eq!(DualAccum::GcdAbsEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::LastIndexDivisibleByK.eval(&[3, 4, 6], 2), Some(2));
        assert_eq!(DualAccum::LcmAbsEvenNonZero.eval(&[0, 4, 6, 3]), Some(12));
        assert_eq!(DualAccum::LcmAbsOddNonZero.eval(&[0, 3, 5, 2]), Some(15));
        assert_eq!(DualAccum::LcmAbsEvenNonZero.eval(&[0, 1, 3]), Some(1));
        assert_eq!(KClosed::AbsSumDivisibleByK.eval(&[-4, 3, 6], 2), Some(10));
        assert_eq!(DualAccum::MeanAbsEvenNonZeroTrunc.eval(&[0, -4, 2, 3]), Some(3));
        assert_eq!(DualAccum::MeanAbsOddNonZeroTrunc.eval(&[0, -5, 3, 2]), Some(4));
        assert_eq!(DualAccum::MeanAbsEvenNonZeroTrunc.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::AbsProductDivisibleByK.eval(&[-4, 3, 6], 2), Some(24));
        assert_eq!(DualAccum::MaxAbsEvenNonZero.eval(&[0, -8, 2, 3]), Some(8));
        assert_eq!(DualAccum::MaxAbsOddNonZero.eval(&[0, -7, 5, 2]), Some(7));
        assert_eq!(DualAccum::MaxAbsEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::MaxAbsDivisibleByK.eval(&[-8, 3, 4], 2), Some(8));
        assert_eq!(DualAccum::MinAbsEvenNonZero.eval(&[0, -8, 2, 3]), Some(2));
        assert_eq!(DualAccum::MinAbsOddNonZero.eval(&[0, -7, 5, 2]), Some(5));
        assert_eq!(DualAccum::MinAbsEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::MinAbsDivisibleByK.eval(&[-8, 3, 4], 2), Some(4));
        assert_eq!(DualAccum::SumSquaresEvenNonZero.eval(&[0, -4, 2, 3]), Some(20));
        assert_eq!(DualAccum::SumSquaresOddNonZero.eval(&[0, -3, 5, 2]), Some(34));
        assert_eq!(DualAccum::SumSquaresEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::GcdAbsDivisibleByK.eval(&[12, 18, 5], 2), Some(6));
        assert_eq!(DualAccum::ProductSquaresEvenNonZero.eval(&[0, -2, 4, 3]), Some(64));
        assert_eq!(DualAccum::ProductSquaresOddNonZero.eval(&[0, -3, 5, 2]), Some(225));
        assert_eq!(DualAccum::ProductSquaresEvenNonZero.eval(&[0, 1, 3]), Some(1));
        assert_eq!(KClosed::LcmAbsDivisibleByK.eval(&[4, 6, 5], 2), Some(12));
        assert_eq!(DualAccum::SumCubesEvenNonZero.eval(&[0, -2, 4, 3]), Some(56));
        assert_eq!(DualAccum::SumCubesOddNonZero.eval(&[0, -3, 5, 2]), Some(98));
        assert_eq!(DualAccum::SumCubesEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::MeanAbsDivisibleByKTrunc.eval(&[-4, 2, 6], 2), Some(4));
        assert_eq!(DualAccum::ProductCubesEvenNonZero.eval(&[0, -2, 2, 3]), Some(-64));
        assert_eq!(DualAccum::ProductCubesOddNonZero.eval(&[0, -3, 1, 2]), Some(-27));
        assert_eq!(DualAccum::ProductCubesEvenNonZero.eval(&[0, 1, 3]), Some(1));
        assert_eq!(KClosed::CountNonZeroDivisibleByK.eval(&[0, 4, 6, 3], 2), Some(2));
        assert_eq!(DualAccum::SumFourthPowersEvenNonZero.eval(&[0, -2, 2, 3]), Some(32));
        assert_eq!(DualAccum::SumFourthPowersOddNonZero.eval(&[0, -3, 1, 2]), Some(82));
        assert_eq!(DualAccum::SumFourthPowersEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::SumNonZeroDivisibleByK.eval(&[0, 4, 6, 3], 2), Some(10));
        assert_eq!(DualAccum::ProductFourthPowersEvenNonZero.eval(&[0, -2, 2, 3]), Some(256));
        assert_eq!(DualAccum::ProductFourthPowersOddNonZero.eval(&[0, -3, 1, 2]), Some(81));
        assert_eq!(DualAccum::ProductFourthPowersEvenNonZero.eval(&[0, 1, 3]), Some(1));
        assert_eq!(KClosed::ProductNonZeroDivisibleByK.eval(&[0, 4, 6, 3], 2), Some(24));
        assert_eq!(DualAccum::MeanFourthPowersEvenNonZeroTrunc.eval(&[0, -2, 2, 3]), Some(16));
        assert_eq!(DualAccum::MeanFourthPowersOddNonZeroTrunc.eval(&[0, -3, 1, 2]), Some(41));
        assert_eq!(DualAccum::MeanFourthPowersEvenNonZeroTrunc.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::MaxNonZeroDivisibleByK.eval(&[0, 4, 6, 3], 2), Some(6));
        assert_eq!(DualAccum::SumFifthPowersEvenNonZero.eval(&[0, -2, 2, 3]), Some(0));
        assert_eq!(DualAccum::SumFifthPowersOddNonZero.eval(&[0, -3, 1, 2]), Some(-242));
        assert_eq!(DualAccum::SumFifthPowersEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::MinNonZeroDivisibleByK.eval(&[-8, 0, 4, 6], 2), Some(-8));
        assert_eq!(DualAccum::ProductFifthPowersEvenNonZero.eval(&[0, -2, 2, 3]), Some(-1024));
        assert_eq!(DualAccum::ProductFifthPowersOddNonZero.eval(&[0, -3, 1, 2]), Some(-243));
        assert_eq!(DualAccum::ProductFifthPowersEvenNonZero.eval(&[0, 1, 3]), Some(1));
        assert_eq!(KClosed::FirstNonZeroDivisibleByK.eval(&[0, 4, 6, 3], 2), Some(4));
        assert_eq!(DualAccum::MeanFifthPowersEvenNonZeroTrunc.eval(&[0, -2, 2, 3]), Some(0));
        assert_eq!(DualAccum::MeanFifthPowersOddNonZeroTrunc.eval(&[0, -3, 1, 2]), Some(-121));
        assert_eq!(DualAccum::MeanFifthPowersEvenNonZeroTrunc.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::LastNonZeroDivisibleByK.eval(&[0, 4, 6, 3], 2), Some(6));
        assert_eq!(DualAccum::SumSixthPowersEvenNonZero.eval(&[0, -2, 2, 3]), Some(128));
        assert_eq!(DualAccum::SumSixthPowersOddNonZero.eval(&[0, -3, 1, 2]), Some(730));
        assert_eq!(DualAccum::SumSixthPowersEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::AbsSumNonZeroDivisibleByK.eval(&[0, -4, 6, 3], 2), Some(10));
        assert_eq!(DualAccum::ProductSixthPowersEvenNonZero.eval(&[0, -2, 2, 3]), Some(4096));
        assert_eq!(DualAccum::ProductSixthPowersOddNonZero.eval(&[0, -3, 1, 2]), Some(729));
        assert_eq!(DualAccum::ProductSixthPowersEvenNonZero.eval(&[0, 1, 3]), Some(1));
        assert_eq!(KClosed::AbsProductNonZeroDivisibleByK.eval(&[0, -4, 6, 3], 2), Some(24));
        assert_eq!(DualAccum::MeanSixthPowersEvenNonZeroTrunc.eval(&[0, -2, 2, 3]), Some(64));
        assert_eq!(DualAccum::MeanSixthPowersOddNonZeroTrunc.eval(&[0, -3, 1, 2]), Some(365));
        assert_eq!(DualAccum::MeanSixthPowersEvenNonZeroTrunc.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::MeanNonZeroDivisibleByKTrunc.eval(&[0, -4, 6, 3], 2), Some(1));
        assert_eq!(DualAccum::SumSeventhPowersEvenNonZero.eval(&[0, -2, 2, 3]), Some(0));
        assert_eq!(DualAccum::SumSeventhPowersOddNonZero.eval(&[0, -3, 1, 2]), Some(-2186));
        assert_eq!(DualAccum::SumSeventhPowersEvenNonZero.eval(&[0, 1, 3]), Some(0));
        assert_eq!(KClosed::MaxAbsNonZeroDivisibleByK.eval(&[0, -8, 4, 3], 2), Some(8));
    }
}
