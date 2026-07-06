//! General digit-decompose → filter → map → reduce search.
//!
//! The catalog already has one-off digit teachers (`search_sum_odd_digits_loop`,
//! `search_digit_product_loop`, …), but their COMBINATIONS ("product of the ODD
//! digits", "sum of SQUARES of even digits", "count of nonzero digits") each need
//! their own recognizer. This ONE search covers the whole family by enumerating
//! (predicate × element-map × reduction × empty-convention) over a scalar int and
//! validating each candidate against every example — so it is exact-by-construction
//! (never overfits: a combo is emitted only when it reproduces all examples, and
//! `verified_result` strict-re-verifies the emitted code). Bounded (~50 combos),
//! positive-integer digit convention (`while x > 0`, matching the existing digit
//! teachers), unary `i64 -> i64` only.

use super::*;
use super::search_codegen::verified_result;

#[derive(Clone, Copy)]
enum Pred {
    All,
    Odd,
    Even,
    Nonzero,
}
#[derive(Clone, Copy)]
enum Map {
    Id,
    Square,
    Cube,
}
#[derive(Clone, Copy)]
enum Reduce {
    Sum,
    Product,
    ProductEmptyZero,
    Count,
}

impl Pred {
    fn keep(self, d: i64) -> bool {
        match self {
            Pred::All => true,
            Pred::Odd => d % 2 == 1,
            Pred::Even => d % 2 == 0,
            Pred::Nonzero => d > 0,
        }
    }
    /// Mog `if` condition, or None when the body is unconditional (All).
    fn cond(self) -> Option<&'static str> {
        match self {
            Pred::All => None,
            Pred::Odd => Some("(d % 2) == 1"),
            Pred::Even => Some("(d % 2) == 0"),
            Pred::Nonzero => Some("d > 0"),
        }
    }
}
impl Map {
    fn apply(self, d: i64) -> i64 {
        match self {
            Map::Id => d,
            Map::Square => d * d,
            Map::Cube => d * d * d,
        }
    }
    fn expr(self) -> &'static str {
        match self {
            Map::Id => "d",
            Map::Square => "d * d",
            Map::Cube => "d * d * d",
        }
    }
}

/// Reference evaluation — MUST stay behaviorally identical to `emit` below, or
/// `verified_result`'s strict re-verify of the emitted code will (correctly) fail.
fn eval(n: i64, p: Pred, m: Map, r: Reduce) -> i64 {
    let mut x = n;
    let mut acc: i64 = match r {
        Reduce::Sum | Reduce::Count => 0,
        Reduce::Product | Reduce::ProductEmptyZero => 1,
    };
    let mut kept: i64 = 0;
    while x > 0 {
        let d = x % 10;
        if p.keep(d) {
            let v = m.apply(d);
            acc = match r {
                Reduce::Sum => acc + v,
                Reduce::Product | Reduce::ProductEmptyZero => acc * v,
                Reduce::Count => acc + 1,
            };
            kept += 1;
        }
        x /= 10;
    }
    if matches!(r, Reduce::ProductEmptyZero) && kept == 0 {
        acc = 0;
    }
    acc
}

fn emit(fn_name: &str, p: Pred, m: Map, r: Reduce) -> String {
    let init = match r {
        Reduce::Sum | Reduce::Count => 0,
        Reduce::Product | Reduce::ProductEmptyZero => 1,
    };
    let combine = match r {
        Reduce::Sum => format!("acc = acc + {};", m.expr()),
        Reduce::Product | Reduce::ProductEmptyZero => format!("acc = acc * {};", m.expr()),
        Reduce::Count => "acc = acc + 1;".to_string(),
    };
    // body: optional predicate gate around (v-binding? no — inline map expr) + combine + kept++
    let inner = format!("{combine}\n            kept = kept + 1;");
    let body = match p.cond() {
        Some(c) => format!("if {c} {{\n            {inner}\n        }}"),
        None => inner,
    };
    let empty = if matches!(r, Reduce::ProductEmptyZero) {
        "\n    if kept == 0 {\n        acc = 0;\n    }"
    } else {
        ""
    };
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    acc: i64 = {init};\n    kept: i64 = 0;\n    while x > 0 {{\n        d: i64 = x % 10;\n        {body}\n        x = x / 10;\n    }}{empty}\n    return acc;\n}}\n"
    )
}

/// Try every (pred × map × reduce) combo; emit the FIRST that reproduces all
/// examples. Exact-by-construction, so it preempts gradient distillation.
pub(super) fn search_digits_filter_map_reduce(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    if parse_param_types(problem.signature) != [ParamType::I64] {
        return None;
    }
    // Need >=2 examples with a multi-digit input so the family is not trivially
    // aliased by a constant/affine (those are handled by cheaper searches first).
    let has_multidigit = problem.examples.iter().any(|ex| {
        ex.inputs
            .first()
            .and_then(super::helpers::int_value)
            .map(|x| x.abs() >= 10)
            .unwrap_or(false)
    });
    if problem.examples.len() < 2 || !has_multidigit {
        return None;
    }
    let preds = [Pred::All, Pred::Odd, Pred::Even, Pred::Nonzero];
    let maps = [Map::Id, Map::Square, Map::Cube];
    let reduces = [
        Reduce::Sum,
        Reduce::Product,
        Reduce::ProductEmptyZero,
        Reduce::Count,
    ];
    for &r in &reduces {
        // Count ignores the map — only vary map for sum/product.
        let map_set: &[Map] = if matches!(r, Reduce::Count) {
            &[Map::Id]
        } else {
            &maps
        };
        for &p in &preds {
            for &m in map_set {
                if validate_unary_int(problem, |n| eval(n, p, m, r)) {
                    if let Some(res) =
                        verified_result(problem, emit(fn_name, p, m, r), "search_digits_filter_map_reduce")
                    {
                        return Some(res);
                    }
                }
            }
        }
    }
    None
}
