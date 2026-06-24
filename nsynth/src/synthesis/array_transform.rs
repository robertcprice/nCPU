//! Exact, fast synthesis for array-to-array transforms (`fn f(arr: [i64]) -> [i64]`).
//!
//! This is the structural analog of the string synthesizer: a bottom-up
//! enumeration over a small library of length-preserving and length-changing
//! array transforms (identity, elementwise affine/abs/square maps, sort,
//! reverse, prefix-sum scan, predicate filter). Every candidate is emitted as
//! Mog source and accepted ONLY when `verify_problem_code_strict` passes on all
//! examples and holdouts, so a returned `SolveResult` is proof-carrying.
//!
//! It is wired ahead of the gradient array core so the common cases resolve in
//! milliseconds instead of burning the full gradient budget (which previously
//! timed out on `[i64] -> [i64]` problems with no array-output path).

use super::*;
use crate::enumerative::{BinOp, CmpOp, Expr};

// ── Searched per-element body / predicate enumeration (U5ab) ───────────────
// REUSE of the PROVEN fold-body grammar (enumerative.rs:2536-2587 for the map
// body, enumerative.rs:2593-2630 for the predicate). The map body here is a
// per-element map (no acc / no ForFold wrapper); the predicate slot reuses the
// CmpOp-over-{item,i,consts,item%k} construction. We restrict body BinOps to
// {Add,Sub,Mul,Mod} — exactly the ops whose `Expr::to_mog` rendering is
// faithful to `Expr::eval` (Add/Sub/Mul/Mod render to `+ - * %`), so the
// `eval` pre-screen never disagrees with the emitted Mog the strict verifier
// runs. `Var(0)` = item, `Var(1)` = i. Enumeration is size-bounded (atoms, then
// one atom OP atom level — the same single-level shape as the fold enumerator).

/// Atoms for the element grammar: {item, i} ∪ a small constant set. Bounded.
fn element_atoms() -> Vec<Expr> {
    let mut atoms: Vec<Expr> = vec![Expr::Var(0), Expr::Var(1)];
    for &c in &[0i64, 1, -1, 2, 3] {
        atoms.push(Expr::Const(c));
    }
    atoms
}

/// Body operators whose `to_mog` rendering is exact w.r.t. `eval` (no bitwise
/// approximations leak in). Mirrors the fold body's {Add,Sub,Mul} plus Mod.
const ELEM_BODY_OPS: [BinOp; 4] = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Mod];

/// Enumerate per-element body expressions over {item, i, consts}: every atom,
/// plus every `atom OP atom` for OP in {Add,Sub,Mul,Mod}, and one extra level
/// `(atom OP atom) OP atom` so cubics like `item*item*item` are reachable.
/// Size-bounded: atoms (≈7) + one binop level (≈7·7·4) + a guarded third level.
fn element_bodies() -> Vec<Expr> {
    let atoms = element_atoms();
    let mut out: Vec<Expr> = atoms.clone();
    let mut level1: Vec<Expr> = Vec::new();
    for l in &atoms {
        for r in &atoms {
            for &op in &ELEM_BODY_OPS {
                level1.push(Expr::BinOp(op, Box::new(l.clone()), Box::new(r.clone())));
            }
        }
    }
    // Third level: (level1) OP atom — only Add/Sub/Mul (Mod-on-product is rare
    // and Mod is already covered at level 1 for the item%k case). This is what
    // reaches item*item*item without an unbounded grammar.
    for l in &level1 {
        for r in &atoms {
            for &op in &[BinOp::Add, BinOp::Sub, BinOp::Mul] {
                out.push(Expr::BinOp(op, Box::new(l.clone()), Box::new(r.clone())));
            }
        }
    }
    out.extend(level1);
    out
}

/// Enumerate boolean guard predicates `lhs CMP rhs` over {item, i, consts} plus
/// the `item % k` family — a direct reuse of the cond_pairs construction in
/// enumerative.rs:2593-2630. Returned as (rendered-cond-string, eval-closure-input).
/// We keep the structure (CmpOp, lhs:Expr, rhs:Expr) so it both renders and
/// pre-screens via `eval`.
fn element_predicates() -> Vec<(CmpOp, Expr, Expr)> {
    let mut preds: Vec<(CmpOp, Expr, Expr)> = Vec::new();
    let item = Expr::Var(0);
    let cmps = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
    ];
    // item CMP const
    for &cmp in &cmps {
        for &c in &[0i64, 1, -1, 2, 3] {
            preds.push((cmp, item.clone(), Expr::Const(c)));
        }
    }
    // item % k CMP r  (k ∈ {2,3,4,5}, r ∈ 0..k) — reaches item%3 == 1 etc.
    for &k in &[2i64, 3, 4, 5] {
        for r in 0..k {
            let modexpr = Expr::BinOp(BinOp::Mod, Box::new(item.clone()), Box::new(Expr::Const(k)));
            preds.push((CmpOp::Eq, modexpr.clone(), Expr::Const(r)));
            preds.push((CmpOp::Ne, modexpr, Expr::Const(r)));
        }
    }
    preds
}

/// Render a comparison operator to its Mog symbol.
fn cmp_symbol(cmp: CmpOp) -> &'static str {
    match cmp {
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Eq => "==",
        CmpOp::Ge => ">=",
        CmpOp::Gt => ">",
        CmpOp::Ne => "!=",
    }
}

/// Pre-screen a per-element body against the data with checked `eval` so we only
/// emit Mog for bodies that already reproduce every (input,output) pair. This is
/// an optimization; the binding accept gate remains `verify_problem_code_strict`.
fn body_fits(body: &Expr, rows: &[(Vec<i64>, Vec<i64>)]) -> bool {
    for (input, output) in rows {
        if input.len() != output.len() {
            return false;
        }
        for (i, (&item, &expected)) in input.iter().zip(output.iter()).enumerate() {
            match body.eval(&[item, i as i64]) {
                Some(v) if v == expected => {}
                _ => return false,
            }
        }
    }
    true
}

/// Pre-screen a per-element body COMPOSED WITH an order transform (sort/reverse):
/// `transform(map(input)) == output` must hold on every row. The map is applied
/// element-wise (via the same checked `eval`), then the mapped row is reordered.
/// This is the array-transform analogue of `body_fits` for the
/// map-then-reorder composite (one ScalarMap chain followed by ONE array
/// transform). Length-preserving by construction. The binding accept gate
/// remains `verify_problem_code_strict`.
fn body_then_order_fits(
    body: &Expr,
    rows: &[(Vec<i64>, Vec<i64>)],
    reorder: fn(&mut Vec<i64>),
) -> bool {
    for (input, output) in rows {
        if input.len() != output.len() {
            return false;
        }
        let mut mapped: Vec<i64> = Vec::with_capacity(input.len());
        for (i, &item) in input.iter().enumerate() {
            match body.eval(&[item, i as i64]) {
                Some(v) => mapped.push(v),
                None => return false,
            }
        }
        reorder(&mut mapped);
        if &mapped != output {
            return false;
        }
    }
    true
}

/// Pre-screen a guard predicate against the filter data: keeping exactly the
/// elements for which `lhs CMP rhs` is true must reproduce every output row.
fn predicate_fits(cmp: CmpOp, lhs: &Expr, rhs: &Expr, rows: &[(Vec<i64>, Vec<i64>)]) -> bool {
    for (input, output) in rows {
        let mut kept: Vec<i64> = Vec::new();
        for (i, &item) in input.iter().enumerate() {
            let l = match lhs.eval(&[item, i as i64]) {
                Some(v) => v,
                None => return false,
            };
            let r = match rhs.eval(&[item, i as i64]) {
                Some(v) => v,
                None => return false,
            };
            let keep = match cmp {
                CmpOp::Lt => l < r,
                CmpOp::Le => l <= r,
                CmpOp::Eq => l == r,
                CmpOp::Ge => l >= r,
                CmpOp::Gt => l > r,
                CmpOp::Ne => l != r,
            };
            if keep {
                kept.push(item);
            }
        }
        if &kept != output {
            return false;
        }
    }
    true
}

/// Single-array-input rows `(input, expected_output)` when the problem is a
/// `fn f(arr: [i64]) -> [i64]` shape. Returns None for any other signature so
/// the caller falls through to the existing array machinery untouched.
fn array_rows(problem: &Problem) -> Option<Vec<(Vec<i64>, Vec<i64>)>> {
    if problem.examples.is_empty() {
        return None;
    }
    problem
        .examples
        .iter()
        .map(|ex| match ex.inputs.as_slice() {
            [input @ Value::Array(_)] => {
                // Both the single array input and the array output must be all-int
                // for this numeric-transform path; a typed/nested array yields
                // `None` and the caller falls through to the array machinery.
                match (input.as_i64_slice(), ex.expected.as_i64_slice()) {
                    (Some(i), Some(o)) => Some((i, o)),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect()
}

/// Render `a * item + b` as a Mog expression over the loop variable `item`,
/// never emitting a bare negative literal (the lexer treats `-` as a binary
/// operator), so subtraction encodes every negative coefficient/offset.
fn affine_expr(a: i64, b: i64) -> String {
    let base = match a {
        0 => None,
        1 => Some("item".to_string()),
        -1 => Some("(0 - item)".to_string()),
        a if a > 0 => Some(format!("item * {a}")),
        a => Some(format!("(0 - item) * {}", -a)),
    };
    match base {
        None => const_expr(b),
        Some(base) if b == 0 => base,
        Some(base) if b > 0 => format!("{base} + {b}"),
        Some(base) => format!("{base} - {}", -b),
    }
}

/// Render an integer constant as a non-negative-literal Mog expression.
fn const_expr(c: i64) -> String {
    match c {
        c if c >= 0 => c.to_string(),
        c => format!("(0 - {})", -c),
    }
}

/// Wrap a per-element push body in the canonical map skeleton.
fn map_program(fn_name: &str, push_body: &str) -> String {
    format!(
        "fn {fn_name}(arr: [i64]) -> [i64] {{\n    result: [i64] = [];\n    for item in arr {{\n{push_body}    }}\n    return result;\n}}\n"
    )
}

/// Which single array transform a map-then-reorder composite applies after the
/// per-element map: sort ascending or reverse. (The only ArrayTransform ops in
/// the NL pipeline — `sort`/`reverse`, both `Vec<i64> -> Vec<i64>`.)
#[derive(Clone, Copy)]
enum ReorderKind {
    SortAsc,
    Reverse,
}

/// Emit a `fn f(arr) -> [i64]` that maps every element with `elem_expr` (over the
/// loop var `item`), then applies ONE array transform. The map result is built
/// into `mapped`, then sorted in place (`mapped.sort()`) or reversed into a new
/// array — reusing the exact DSL the dedicated sort/reverse candidates already
/// emit, so no new runtime construct is introduced.
fn map_then_reorder_program(fn_name: &str, elem_expr: &str, kind: ReorderKind) -> String {
    match kind {
        ReorderKind::SortAsc => format!(
            "fn {fn_name}(arr: [i64]) -> [i64] {{\n    mapped: [i64] = [];\n    for item in arr {{\n        mapped.push({elem_expr});\n    }}\n    mapped.sort();\n    return mapped;\n}}\n"
        ),
        ReorderKind::Reverse => format!(
            "fn {fn_name}(arr: [i64]) -> [i64] {{\n    mapped: [i64] = [];\n    for item in arr {{\n        mapped.push({elem_expr});\n    }}\n    result: [i64] = [];\n    i: i64 = mapped.len - 1;\n    while i >= 0 {{\n        result.push(mapped[i]);\n        i = i - 1;\n    }}\n    return result;\n}}\n"
        ),
    }
}

/// Solve an exact integer affine map `y = a*x + b` from observed element pairs.
/// Requires two distinct `x` to pin the slope; returns None when the data is not
/// consistent with a single integer-affine rule.
fn derive_affine(pairs: &[(i64, i64)]) -> Option<(i64, i64)> {
    let (x0, y0) = *pairs.first()?;
    let anchor = pairs.iter().find(|(x, _)| *x != x0);
    let (a, b) = match anchor {
        Some(&(x1, y1)) => {
            let dx = x1 - x0;
            let dy = y1 - y0;
            if dy % dx != 0 {
                return None;
            }
            let a = dy / dx;
            (a, y0 - a * x0)
        }
        // All inputs identical: cannot separate slope from offset; treat as the
        // constant map (a = 0) and let verification accept or reject it.
        None => (0, y0),
    };
    let consistent = pairs
        .iter()
        .all(|&(x, y)| a.checked_mul(x).and_then(|p| p.checked_add(b)) == Some(y));
    if consistent {
        Some((a, b))
    } else {
        None
    }
}

/// Solve an exact integer quadratic map `y = a*x^2 + b*x + c` from observed
/// element pairs. Requires at least three distinct `x` to pin the three
/// coefficients; returns None when the data is not consistent with a single
/// integer-quadratic rule, or when `a == 0` (that is the affine case, already
/// handled by `derive_affine` — do not duplicate it here). Pure/total: no
/// panics, no unwrap on user data; all arithmetic is overflow-checked.
fn derive_quadratic(pairs: &[(i64, i64)]) -> Option<(i64, i64, i64)> {
    // Collect three pairs with pairwise-distinct x values.
    let mut distinct: Vec<(i64, i64)> = Vec::new();
    for &(x, y) in pairs {
        if !distinct.iter().any(|&(dx, _)| dx == x) {
            distinct.push((x, y));
            if distinct.len() == 3 {
                break;
            }
        }
    }
    if distinct.len() < 3 {
        return None;
    }
    let (x0, y0) = distinct[0];
    let (x1, y1) = distinct[1];
    let (x2, y2) = distinct[2];

    // Solve the 3x3 Vandermonde system in i128 to avoid intermediate overflow.
    // | x0^2 x0 1 | |a|   |y0|
    // | x1^2 x1 1 | |b| = |y1|
    // | x2^2 x2 1 | |c|   |y2|
    let (x0, x1, x2) = (x0 as i128, x1 as i128, x2 as i128);
    let (y0, y1, y2) = (y0 as i128, y1 as i128, y2 as i128);

    let det3 = |m: [[i128; 3]; 3]| -> i128 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let base = [
        [x0 * x0, x0, 1],
        [x1 * x1, x1, 1],
        [x2 * x2, x2, 1],
    ];

    // Determinant of THIS matrix (column order x^2, x, 1) — computed directly so
    // Cramer's rule stays sign-consistent. Nonzero iff the three x are distinct.
    let det = det3(base);
    if det == 0 {
        return None;
    }

    // Cramer's rule. Replace each column with the RHS in turn.
    let mut ma = base;
    ma[0][0] = y0;
    ma[1][0] = y1;
    ma[2][0] = y2;
    let mut mb = base;
    mb[0][1] = y0;
    mb[1][1] = y1;
    mb[2][1] = y2;
    let mut mc = base;
    mc[0][2] = y0;
    mc[1][2] = y1;
    mc[2][2] = y2;

    let da = det3(ma);
    let db = det3(mb);
    let dc = det3(mc);

    // Require exact integer solutions.
    if da % det != 0 || db % det != 0 || dc % det != 0 {
        return None;
    }
    let a128 = da / det;
    let b128 = db / det;
    let c128 = dc / det;

    // a == 0 is the affine case; defer to derive_affine.
    if a128 == 0 {
        return None;
    }

    // Coefficients must fit in i64 to be emittable.
    let a = i64::try_from(a128).ok()?;
    let b = i64::try_from(b128).ok()?;
    let c = i64::try_from(c128).ok()?;

    // Verify the fitted (a,b,c) reproduces EVERY pair with checked arithmetic;
    // overflow or mismatch rejects (no false-fit).
    let reproduces = pairs.iter().all(|&(x, y)| {
        let sq = match x.checked_mul(x) {
            Some(v) => v,
            None => return false,
        };
        let ax2 = match a.checked_mul(sq) {
            Some(v) => v,
            None => return false,
        };
        let bx = match b.checked_mul(x) {
            Some(v) => v,
            None => return false,
        };
        match ax2.checked_add(bx).and_then(|p| p.checked_add(c)) {
            Some(v) => v == y,
            None => false,
        }
    });
    if reproduces {
        Some((a, b, c))
    } else {
        None
    }
}

/// Build the ordered candidate program list. Cheapest / most common transforms
/// first so verification short-circuits quickly.
fn candidates(problem: &Problem, rows: &[(Vec<i64>, Vec<i64>)]) -> Vec<(&'static str, String)> {
    let fn_name = problem.function_name();
    let mut out: Vec<(&'static str, String)> = Vec::new();

    let length_preserving = rows.iter().all(|(i, o)| i.len() == o.len());

    // Identity.
    out.push((
        "array_transform_identity",
        format!("fn {fn_name}(arr: [i64]) -> [i64] {{\n    return arr;\n}}\n"),
    ));

    if length_preserving {
        // EXACT named structural transforms FIRST. These reorder the array but do
        // not regress an element function, so a 1-example affine fit must never win
        // over them in the accept-first-pass loop: a request like "reverse a list"
        // with a single example can be spuriously matched by an affine that happens
        // to fit those few element pairs. Pushing the exact named transforms ahead
        // of the affine/quadratic regression candidates makes an exact structural
        // match win. (verify_problem_code_strict still gates each on fresh
        // holdouts, so a wrong structural guess is rejected, not blindly accepted.)

        // Sort ascending.
        out.push((
            "array_transform_sort",
            format!("fn {fn_name}(arr: [i64]) -> [i64] {{\n    arr.sort();\n    return arr;\n}}\n"),
        ));

        // Reverse.
        out.push((
            "array_transform_reverse",
            format!(
                "fn {fn_name}(arr: [i64]) -> [i64] {{\n    result: [i64] = [];\n    i: i64 = arr.len - 1;\n    while i >= 0 {{\n        result.push(arr[i]);\n        i = i - 1;\n    }}\n    return result;\n}}\n"
            ),
        ));

        // Sort descending (sort then reverse).
        out.push((
            "array_transform_sort_desc",
            format!(
                "fn {fn_name}(arr: [i64]) -> [i64] {{\n    arr.sort();\n    result: [i64] = [];\n    i: i64 = arr.len - 1;\n    while i >= 0 {{\n        result.push(arr[i]);\n        i = i - 1;\n    }}\n    return result;\n}}\n"
            ),
        ));

        // Elementwise affine map (covers double, +c, -c, negate, scale, const).
        let pairs: Vec<(i64, i64)> = rows
            .iter()
            .flat_map(|(i, o)| i.iter().copied().zip(o.iter().copied()))
            .collect();
        if let Some((a, b)) = derive_affine(&pairs) {
            let body = format!("        result.push({});\n", affine_expr(a, b));
            out.push(("array_transform_map_affine", map_program(fn_name, &body)));
        }

        // Absolute value.
        out.push((
            "array_transform_abs",
            map_program(
                fn_name,
                "        if item < 0 {\n            result.push(0 - item);\n        } else {\n            result.push(item);\n        }\n",
            ),
        ));

        // Square.
        out.push((
            "array_transform_square",
            map_program(fn_name, "        result.push(item * item);\n"),
        ));

        // Searched quadratic `y = a*x^2 + b*x + c` (genuinely new reach beyond
        // the fixed Identity/Affine/Abs/Square menu). Fitted in closed form from
        // the same element pairs; emitted AFTER the fixed templates so they
        // verify-and-win first, and only reached when they all fail. The
        // `a == 0` case is rejected inside `derive_quadratic` (it is affine),
        // so this never duplicates the affine path.
        if let Some((a, b, c)) = derive_quadratic(&pairs) {
            // `item * item * a` for the quadratic head; reuse affine_expr to
            // render the `b*item + c` tail with no bare negative literals.
            let sq_head = match a {
                1 => "item * item".to_string(),
                -1 => "(0 - item * item)".to_string(),
                a if a > 0 => format!("item * item * {a}"),
                a => format!("(0 - item * item) * {}", -a),
            };
            let tail = affine_expr(b, c);
            let body_expr = if tail == "0" {
                sq_head
            } else if b == 0 {
                // tail is the pure constant c; combine via +/- to avoid bare neg.
                match c {
                    c if c >= 0 => format!("{sq_head} + {c}"),
                    c => format!("{sq_head} - {}", -c),
                }
            } else {
                format!("{sq_head} + {tail}")
            };
            let body = format!("        result.push({body_expr});\n");
            out.push((
                "array_transform_map_searched_quadratic",
                map_program(fn_name, &body),
            ));
        }

        // Elementwise min/max against a derived constant.
        for c in derived_consts(rows) {
            out.push((
                "array_transform_map_min",
                map_program(
                    fn_name,
                    &format!(
                        "        if item < {c} {{\n            result.push(item);\n        }} else {{\n            result.push({c});\n        }}\n"
                    ),
                ),
            ));
            out.push((
                "array_transform_map_max",
                map_program(
                    fn_name,
                    &format!(
                        "        if item > {c} {{\n            result.push(item);\n        }} else {{\n            result.push({c});\n        }}\n"
                    ),
                ),
            ));
        }

        // Prefix-sum (running scan).
        out.push((
            "array_transform_prefix_sum",
            format!(
                "fn {fn_name}(arr: [i64]) -> [i64] {{\n    result: [i64] = [];\n    acc: i64 = 0;\n    for item in arr {{\n        acc = acc + item;\n        result.push(acc);\n    }}\n    return result;\n}}\n"
            ),
        ));

        // (U5a) SEARCHED per-element map. Enumerate bodies over {item, i, consts}
        // from the proven fold-body grammar, pre-screen with checked `eval`, and
        // emit each surviving body through the existing `to_mog` + `map_program`.
        // Appended AFTER the fixed Identity/Affine/Abs/Square/quadratic menu, so
        // those verify-and-win first in ms; this only reaches bodies they miss
        // (e.g. item*item*item, item%3). The strict verifier is the accept gate.
        for body in element_bodies() {
            if !body_fits(&body, rows) {
                continue;
            }
            let body_src = body.to_mog(&["item", "i"]);
            let push_body = format!("        result.push({body_src});\n");
            out.push((
                "array_transform_map_searched_body",
                map_program(fn_name, &push_body),
            ));
        }

        // (NL-COMPOSE-ARRTRANSFORM) MAP-then-REORDER composite: a per-element map
        // (the SAME searched element grammar) followed by ONE array transform —
        // sort ascending or reverse. This is the single array-transform stage of
        // the NL pipeline ("the sorted negated values" = sort(map(negate)),
        // "reverse the squared values" = reverse(map(square))). The map alone is
        // already covered above; this only reaches the genuinely-2-stage case
        // where the OUTPUT IS REORDERED, so the element-wise `body_fits` rejected
        // it. Pre-screened by APPLYING the map then the reorder (`body_then_order_fits`),
        // emitted simplest-map-first; the strict verifier remains the accept gate.
        // Bounded by the same `element_bodies()` grammar × 2 reorders.
        for body in element_bodies() {
            let body_src = body.to_mog(&["item", "i"]);
            // sort ascending
            if body_then_order_fits(&body, rows, |v| v.sort()) {
                out.push((
                    "array_transform_map_then_sort",
                    map_then_reorder_program(fn_name, &body_src, ReorderKind::SortAsc),
                ));
            }
            // reverse
            if body_then_order_fits(&body, rows, |v| v.reverse()) {
                out.push((
                    "array_transform_map_then_reverse",
                    map_then_reorder_program(fn_name, &body_src, ReorderKind::Reverse),
                ));
            }
        }
    }

    // Predicate filter (length may change). Thresholds derived from the data.
    let mut preds: Vec<(&'static str, String)> = vec![
        ("array_transform_filter_evens", "item % 2 == 0".to_string()),
        ("array_transform_filter_odds", "item % 2 != 0".to_string()),
        ("array_transform_filter_pos", "item > 0".to_string()),
        ("array_transform_filter_nonneg", "item >= 0".to_string()),
        ("array_transform_filter_neg", "item < 0".to_string()),
    ];
    for c in derived_consts(rows) {
        preds.push(("array_transform_filter_gt", format!("item > {c}")));
        preds.push(("array_transform_filter_ge", format!("item >= {c}")));
        preds.push(("array_transform_filter_lt", format!("item < {c}")));
        preds.push(("array_transform_filter_le", format!("item <= {c}")));
    }
    for (method, pred) in preds {
        out.push((
            method,
            map_program(
                fn_name,
                &format!("        if {pred} {{\n            result.push(item);\n        }}\n"),
            ),
        ));
    }

    // (U5b) SEARCHED filter predicate. Enumerate `lhs CMP rhs` guards over
    // {item, i, consts, item%k} from the proven cond_pairs grammar, pre-screen
    // with checked `eval`, and emit each surviving guard into the SAME guarded
    // push skeleton. Appended AFTER the cheap fixed predicates (which win first),
    // so this only reaches guards they miss (e.g. item % 3 == 1). The strict
    // verifier is the accept gate.
    for (cmp, lhs, rhs) in element_predicates() {
        if !predicate_fits(cmp, &lhs, &rhs, rows) {
            continue;
        }
        let ls = lhs.to_mog(&["item", "i"]);
        let rs = rhs.to_mog(&["item", "i"]);
        let pred = format!("{ls} {} {rs}", cmp_symbol(cmp));
        out.push((
            "array_transform_filter_searched",
            map_program(
                fn_name,
                &format!("        if {pred} {{\n            result.push(item);\n        }}\n"),
            ),
        ));
    }

    out
}

/// Small set of candidate integer constants observed in the data (input element
/// values plus 0), bounded to keep the verify loop fast.
fn derived_consts(rows: &[(Vec<i64>, Vec<i64>)]) -> Vec<i64> {
    let mut seen = std::collections::BTreeSet::new();
    seen.insert(0i64);
    for (input, output) in rows {
        for &v in input.iter().chain(output.iter()) {
            seen.insert(v);
        }
    }
    seen.into_iter().take(24).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value};

    /// Build a `fn f(arr: [i64]) -> [i64]` problem; the last two rows become
    /// holdouts so the strict verifier exercises generalization, not just fit.
    fn pa(rows: &[(&[i64], &[i64])]) -> Problem {
        let to_ex = |(input, output): &(&[i64], &[i64])| Example {
            inputs: vec![Value::int_array(input)],
            expected: Value::int_array(output),
        };
        let split = rows.len().saturating_sub(2);
        Problem {
            name: "f".to_string(),
            category: "external",
            description: "",
            signature: "fn f(arr: [i64]) -> [i64]",
            examples: rows[..split].iter().map(to_ex).collect(),
            holdouts: rows[split..].iter().map(to_ex).collect(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    fn solve_method(rows: &[(&[i64], &[i64])]) -> String {
        synthesize_array_transform(&pa(rows))
            .expect("expected a solution")
            .method
    }

    #[test]
    fn solves_identity() {
        assert_eq!(
            solve_method(&[
                (&[1, 2, 3], &[1, 2, 3]),
                (&[5], &[5]),
                (&[0, 9], &[0, 9]),
                (&[7, 8], &[7, 8])
            ]),
            "array_transform_identity"
        );
    }

    #[test]
    fn solves_elementwise_double() {
        assert_eq!(
            solve_method(&[
                (&[1, 2, 3], &[2, 4, 6]),
                (&[5], &[10]),
                (&[0, 1], &[0, 2]),
                (&[7], &[14])
            ]),
            "array_transform_map_affine"
        );
    }

    #[test]
    fn solves_increment() {
        assert_eq!(
            solve_method(&[
                (&[1, 2], &[2, 3]),
                (&[5], &[6]),
                (&[0, 9], &[1, 10]),
                (&[7], &[8])
            ]),
            "array_transform_map_affine"
        );
    }

    #[test]
    fn solves_abs() {
        assert_eq!(
            solve_method(&[
                (&[-1, 2, -3], &[1, 2, 3]),
                (&[-5], &[5]),
                (&[-9, 0], &[9, 0]),
                (&[-2], &[2])
            ]),
            "array_transform_abs"
        );
    }

    #[test]
    fn solves_square() {
        assert_eq!(
            solve_method(&[
                (&[1, 2, 3], &[1, 4, 9]),
                (&[5], &[25]),
                (&[0, 4], &[0, 16]),
                (&[6], &[36])
            ]),
            "array_transform_square"
        );
    }

    #[test]
    fn solves_sort() {
        assert_eq!(
            solve_method(&[
                (&[3, 1, 2], &[1, 2, 3]),
                (&[5, 4], &[4, 5]),
                (&[9, 0, 1], &[0, 1, 9]),
                (&[7, 3], &[3, 7])
            ]),
            "array_transform_sort"
        );
    }

    #[test]
    fn solves_reverse() {
        assert_eq!(
            solve_method(&[
                (&[1, 2, 3], &[3, 2, 1]),
                (&[5, 4], &[4, 5]),
                (&[9, 0, 1], &[1, 0, 9]),
                (&[8, 9], &[9, 8])
            ]),
            "array_transform_reverse"
        );
    }

    /// UN-GAMEABLE overfit guard (bug #2). A reverse whose ONLY example is
    /// `[1,2,3] -> [3,2,1]` is element-pair AMBIGUOUS with the affine `y = 4 - x`
    /// (pairs (1,3),(2,2),(3,1) all satisfy it). Before the reorder, the affine
    /// candidate was emitted ahead of the reverse candidate and won the
    /// accept-first-pass on examples-fit + a colluding holdout, producing the
    /// `push((0-item)+4)` overfit the CLI demo showed. This test pins that the
    /// EXACT structural reverse now wins, AND proves it is not coincidence by
    /// showing the affine `y = 4 - x` program is genuinely WRONG on a DIFFERENT
    /// holdout array — so the reorder is load-bearing, not luck.
    #[test]
    fn reverse_beats_affine_overfit_on_single_ambiguous_example() {
        // One example (affine-ambiguous: pairs of [1,2,3]->[3,2,1] all satisfy
        // y=4-x) + TWO differential holdout arrays the affine y=4-x does NOT
        // satisfy (4-5 = -1 != 8). `pa` reserves the last two rows as holdouts,
        // so examples = {[1,2,3]->[3,2,1]} only.
        let rows: &[(&[i64], &[i64])] = &[
            (&[1, 2, 3], &[3, 2, 1]),
            (&[5, 6, 7, 8], &[8, 7, 6, 5]),
            (&[9, 1], &[1, 9]),
        ];
        let problem = pa(rows);

        // FIX: exact reverse wins over the 1-example affine fit.
        assert_eq!(
            synthesize_array_transform(&problem)
                .expect("reverse must solve this")
                .method,
            "array_transform_reverse",
            "exact reverse must beat the 1-example affine fit"
        );

        // UN-GAMEABLE: the affine `y = 4 - x` candidate that previously won is
        // genuinely WRONG here — it must FAIL strict verification on the holdout.
        // (4 - 5 = -1, but the reverse of [5,6,7,8] is [8,7,6,5].) This proves the
        // reorder is what makes reverse win, not a coincidental first-pass.
        let affine_4_minus_x = map_program(
            problem.function_name(),
            &format!("        result.push({});\n", affine_expr(-1, 4)),
        );
        assert!(
            verify_problem_code_strict(&problem, &affine_4_minus_x).is_err(),
            "the prior affine y=4-x overfit MUST be rejected on the differential holdout"
        );

        // And the produced reverse is correct on BOTH distinct arrays (the CLI
        // anti-cheat: [1,2,3]->[3,2,1] AND [5,6,7,8]->[8,7,6,5]).
        let reverse_code = synthesize_array_transform(&problem).unwrap().code;
        assert!(
            verify_problem_code_strict(&problem, &reverse_code).is_ok(),
            "the synthesized reverse must verify on both distinct arrays"
        );
    }

    #[test]
    fn solves_prefix_sum() {
        assert_eq!(
            solve_method(&[
                (&[1, 2, 3], &[1, 3, 6]),
                (&[5], &[5]),
                (&[1, 1, 1, 1], &[1, 2, 3, 4]),
                (&[4, 4], &[4, 8])
            ]),
            "array_transform_prefix_sum"
        );
    }

    #[test]
    fn solves_filter_even() {
        assert_eq!(
            solve_method(&[
                (&[1, 2, 3, 4], &[2, 4]),
                (&[5, 6], &[6]),
                (&[1, 3, 5], &[]),
                (&[7, 8, 9, 10], &[8, 10])
            ]),
            "array_transform_filter_evens"
        );
    }

    #[test]
    fn solves_filter_positive() {
        assert_eq!(
            solve_method(&[
                (&[-1, 2, -3, 4], &[2, 4]),
                (&[5, -6], &[5]),
                (&[-1, -2], &[]),
                (&[-7, 8], &[8])
            ]),
            "array_transform_filter_pos"
        );
    }

    #[test]
    fn rejects_unlearnable_transform() {
        // No template explains a per-index reshuffle keyed to position; the
        // synthesizer must return None rather than a false positive.
        assert!(synthesize_array_transform(&pa(&[
            (&[1, 2, 3], &[2, 1, 3]),
            (&[4, 5, 6], &[6, 4, 5]),
            (&[7, 8, 9], &[8, 9, 7]),
            (&[1, 1, 1], &[1, 1, 1]),
        ]))
        .is_none());
    }

    #[test]
    fn ignores_scalar_output_problem() {
        // A `-> i64` problem is not this synthesizer's shape.
        let problem = Problem {
            signature: "fn f(arr: [i64]) -> i64",
            examples: vec![Example {
                inputs: vec![Value::int_array(&[1, 2, 3])],
                expected: Value::Int(6),
            }],
            ..pa(&[(&[1], &[1])])
        };
        assert!(synthesize_array_transform(&problem).is_none());
    }

    // ---- Searched-quadratic reach (transforms OUTSIDE the fixed menu) ----

    /// Run only the candidates that are NOT the searched-quadratic entry,
    /// mirroring `synthesize_array_transform` exactly (strict verify, first OK
    /// wins). Used to prove a quadratic is genuinely unsolvable by the fixed
    /// menu alone.
    fn synthesize_fixed_only(problem: &Problem) -> Option<SolveResult> {
        let rows = array_rows(problem)?;
        for (method, code) in candidates(problem, &rows) {
            // Exclude ALL searched entries (quadratic, the U5a element-map body,
            // AND the map-then-reorder composites — all enumerate the searched
            // element grammar): "fixed menu only" must mean no searched candidate.
            if method == "array_transform_map_searched_quadratic"
                || method == "array_transform_map_searched_body"
                || method == "array_transform_map_then_sort"
                || method == "array_transform_map_then_reverse"
            {
                continue;
            }
            if verify_problem_code_strict(problem, &code).is_ok() {
                return Some(SolveResult {
                    success: true,
                    code,
                    method: method.to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
        }
        None
    }

    #[test]
    fn solves_searched_quadratic() {
        // y = x*x + 1 — outside {Identity, Affine, Abs, Square}. Last two rows
        // are holdouts (per pa()), so success means the searched candidate was
        // accepted by verify_problem_code_strict on examples AND holdouts.
        assert_eq!(
            solve_method(&[
                (&[0, 1, 2], &[1, 2, 5]),
                (&[3], &[10]),
                (&[4, 5], &[17, 26]),
                (&[6], &[37]),
            ]),
            "array_transform_map_searched_quadratic"
        );
    }

    #[test]
    fn quadratic_not_solvable_by_fixed_menu() {
        // Same x*x+1 data: the fixed menu alone CANNOT solve it, but the full
        // synthesizer (with the searched candidate) CAN.
        let rows: &[(&[i64], &[i64])] = &[
            (&[0, 1, 2], &[1, 2, 5]),
            (&[3], &[10]),
            (&[4, 5], &[17, 26]),
            (&[6], &[37]),
        ];
        assert!(
            synthesize_fixed_only(&pa(rows)).is_none(),
            "fixed menu must NOT solve x*x+1"
        );
        assert!(
            synthesize_array_transform(&pa(rows)).is_some(),
            "full synthesizer must solve x*x+1"
        );
    }

    #[test]
    fn affine_still_wins_no_regression() {
        // x -> 2x and x -> x+1 must still resolve via the cheaper affine path,
        // not the searched-quadratic path (ordering preserved).
        assert_eq!(
            solve_method(&[
                (&[1, 2, 3], &[2, 4, 6]),
                (&[5], &[10]),
                (&[0, 1], &[0, 2]),
                (&[7], &[14])
            ]),
            "array_transform_map_affine"
        );
        assert_eq!(
            solve_method(&[
                (&[1, 2], &[2, 3]),
                (&[5], &[6]),
                (&[0, 9], &[1, 10]),
                (&[7], &[8])
            ]),
            "array_transform_map_affine"
        );
    }

    #[test]
    fn square_still_wins_no_regression() {
        // Pure x*x must resolve via the Square template (which precedes the
        // searched candidate), not the searched-quadratic path.
        assert_eq!(
            solve_method(&[
                (&[1, 2, 3], &[1, 4, 9]),
                (&[5], &[25]),
                (&[0, 4], &[0, 16]),
                (&[6], &[36])
            ]),
            "array_transform_square"
        );
    }

    #[test]
    fn rejects_unlearnable_via_holdout() {
        // Examples fit a quadratic (x*x+1) but a holdout breaks it (x*x+2):
        // verify_problem_code_strict must reject on the holdout, no false
        // examples-only accept.
        let problem = pa(&[
            // examples (all consistent with x*x+1)
            (&[0, 1, 2], &[1, 2, 5]),
            (&[3, 4], &[10, 17]),
            // holdouts (last two rows): first is x*x+1, second is x*x+2 -> breaks
            (&[5], &[26]),
            (&[6], &[38]), // 6*6+2 = 38, inconsistent with x*x+1
        ]);
        assert!(
            synthesize_array_transform(&problem).is_none(),
            "must reject when a holdout violates the fitted quadratic"
        );
    }

    // ---- (U5ab) Searched element-map + searched filter-predicate ----

    /// Run candidates EXCLUDING the searched element-map AND the searched
    /// quadratic (the only non-fixed map entries), mirroring
    /// `synthesize_array_transform` exactly. Proves a transform is genuinely
    /// unsolvable by the fixed map menu alone.
    fn synthesize_fixed_map_only(problem: &Problem) -> Option<SolveResult> {
        let rows = array_rows(problem)?;
        for (method, code) in candidates(problem, &rows) {
            if method == "array_transform_map_searched_body"
                || method == "array_transform_map_searched_quadratic"
                || method == "array_transform_map_then_sort"
                || method == "array_transform_map_then_reverse"
            {
                continue;
            }
            if verify_problem_code_strict(problem, &code).is_ok() {
                return Some(SolveResult {
                    success: true,
                    code,
                    method: method.to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
        }
        None
    }

    /// Run candidates EXCLUDING the searched filter predicate, mirroring
    /// `synthesize_array_transform`. Proves a predicate is genuinely unsolvable
    /// by the fixed predicate set alone.
    fn synthesize_fixed_filter_only(problem: &Problem) -> Option<SolveResult> {
        let rows = array_rows(problem)?;
        for (method, code) in candidates(problem, &rows) {
            if method == "array_transform_filter_searched" {
                continue;
            }
            if verify_problem_code_strict(problem, &code).is_ok() {
                return Some(SolveResult {
                    success: true,
                    code,
                    method: method.to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
        }
        None
    }

    /// Build the same `fn f(arr: [i64]) -> [i64]` problem as `pa()` but ship a
    /// NON-EMPTY, independent `reference_code` oracle. This flips the strict
    /// verifier off the hand-fallback holdouts onto GATE-0's *differential*
    /// holdouts: `generated_holdouts_with_source` samples FRESH inputs across the
    /// widened `[-64, 64]` range and labels them by RUNNING the reference, so the
    /// searched body must generalize, not merely fit the visible rows. The
    /// reference is an independent push-loop implementation of the transform — it
    /// is NOT derived from the candidate, so it is a sound oracle.
    fn pa_ref(rows: &[(&[i64], &[i64])], reference_code: &'static str) -> Problem {
        let mut problem = pa(rows);
        problem.reference_code = reference_code;
        problem
    }

    /// Assert the problem's strict-verify holdouts are GATE-0 *differential*
    /// (`Generated`): freshly sampled inputs labelled by running the reference,
    /// not the degraded hand-authored fallback. Proves the searched body was
    /// gated on generalization, not just example fit.
    fn assert_differential(problem: &Problem) {
        let (_holdouts, source) = crate::benchmark::generated_holdouts_with_source(problem);
        assert_eq!(
            source,
            crate::benchmark::HoldoutSource::Generated,
            "holdouts must be GATE-0 differential (Generated), not the hand fallback"
        );
    }

    #[test]
    fn solves_searched_cube_via_search() {
        // y = item*item*item — OUTSIDE {Identity, Affine, Abs, Square, quadratic}.
        // The reference runs under GATE-0 differential holdouts (fresh [-64,64]
        // inputs, incl. negatives), so success means the searched body passed
        // verify_problem_code_strict on the examples AND a generalization probe.
        let rows: &[(&[i64], &[i64])] = &[
            (&[0, 1, 2], &[0, 1, 8]),
            (&[3], &[27]),
            (&[4, 5], &[64, 125]),
            (&[6], &[216]),
        ];
        // Independent oracle: cube via a push loop (NOT derived from the candidate).
        let reference = "fn f(arr: [i64]) -> [i64] {\n    result: [i64] = [];\n    for item in arr {\n        result.push(item * item * item);\n    }\n    return result;\n}\n";
        let problem = pa_ref(rows, reference);
        // GATE-0 differential regime is exercised (fresh inputs, reference-labelled).
        assert_differential(&problem);
        // Winning method is the SEARCHED body, not a fixed template.
        assert_eq!(
            synthesize_array_transform(&problem)
                .expect("expected a solution")
                .method,
            "array_transform_map_searched_body"
        );
        // The fixed map menu ALONE cannot solve it (un-gameable: prior path None).
        assert!(
            synthesize_fixed_map_only(&problem).is_none(),
            "fixed map menu must NOT solve item*item*item"
        );
    }

    #[test]
    fn solves_searched_mod_map_via_search() {
        // y = item % 3 — OUTSIDE the fixed map menu (no closed-form fitter for it).
        let rows: &[(&[i64], &[i64])] = &[
            (&[0, 1, 2], &[0, 1, 2]),
            (&[3, 4, 5], &[0, 1, 2]),
            (&[6, 7], &[0, 1]),
            (&[8, 9], &[2, 0]),
        ];
        // Independent oracle: item % 3 via a push loop (matches the runtime's `%`
        // on negatives because reference AND candidate render the same `item % 3`).
        let reference = "fn f(arr: [i64]) -> [i64] {\n    result: [i64] = [];\n    for item in arr {\n        result.push(item % 3);\n    }\n    return result;\n}\n";
        let problem = pa_ref(rows, reference);
        assert_differential(&problem);
        assert_eq!(
            synthesize_array_transform(&problem)
                .expect("expected a solution")
                .method,
            "array_transform_map_searched_body"
        );
        assert!(
            synthesize_fixed_map_only(&problem).is_none(),
            "fixed map menu must NOT solve item % 3"
        );
    }

    #[test]
    fn solves_searched_mod_filter_via_search() {
        // filter(item % 3 == 1) — OUTSIDE the fixed predicate set (evens/odds/
        // sign/threshold). Must synthesize via the searched predicate, and the
        // fixed-predicate-only path must return None (un-gameable).
        let rows: &[(&[i64], &[i64])] = &[
            (&[0, 1, 2, 3, 4], &[1, 4]),
            (&[5, 6, 7], &[7]),
            (&[1, 2, 10], &[1, 10]),
            (&[3, 9, 13], &[13]),
        ];
        // Independent oracle: keep elements where item % 3 == 1, via a push loop.
        let reference = "fn f(arr: [i64]) -> [i64] {\n    result: [i64] = [];\n    for item in arr {\n        if item % 3 == 1 {\n            result.push(item);\n        }\n    }\n    return result;\n}\n";
        let problem = pa_ref(rows, reference);
        assert_differential(&problem);
        assert_eq!(
            synthesize_array_transform(&problem)
                .expect("expected a solution")
                .method,
            "array_transform_filter_searched"
        );
        assert!(
            synthesize_fixed_filter_only(&problem).is_none(),
            "fixed predicate set must NOT solve item % 3 == 1"
        );
    }

    #[test]
    fn solves_map_then_sort_via_composite() {
        // sort(negate(x)): map each element with negate, then sort ascending. The
        // output REORDERS the mapped array, so no element-wise map (`searched_body`)
        // and no bare sort/reverse can express it. Reference is an independent
        // map-then-sort push loop; differential holdouts (fresh [-64,64] inputs).
        let rows: &[(&[i64], &[i64])] = &[
            (&[3, 1, 2], &[-3, -2, -1]),
            (&[5, 2, 8], &[-8, -5, -2]),
            (&[4, 7, 3, 9], &[-9, -7, -4, -3]),
            (&[1, 6, 2], &[-6, -2, -1]),
        ];
        let reference = "fn f(arr: [i64]) -> [i64] {\n    mapped: [i64] = [];\n    for item in arr {\n        mapped.push(0 - item);\n    }\n    mapped.sort();\n    return mapped;\n}\n";
        let problem = pa_ref(rows, reference);
        assert_differential(&problem);
        assert_eq!(
            synthesize_array_transform(&problem)
                .expect("expected a solution")
                .method,
            "array_transform_map_then_sort"
        );
        // Un-gameable: the fixed map menu (no searched body, no composite) must NOT
        // solve a map-then-reorder — proving the composite is genuinely required.
        assert!(
            synthesize_fixed_map_only(&problem).is_none(),
            "fixed map menu must NOT solve sort(negate(x))"
        );
    }

    #[test]
    fn solves_map_then_reverse_via_composite() {
        // reverse(square(x)): square each element, then reverse the order. The
        // mapped array is reordered (reverse), so only the map-then-reverse
        // composite can express it. Independent map-then-reverse reference oracle.
        let rows: &[(&[i64], &[i64])] = &[
            (&[2, 3, 4], &[16, 9, 4]),
            (&[1, 5, 2], &[4, 25, 1]),
            (&[3, 6, 1, 2], &[4, 1, 36, 9]),
            (&[7, 1], &[1, 49]),
        ];
        let reference = "fn f(arr: [i64]) -> [i64] {\n    mapped: [i64] = [];\n    for item in arr {\n        mapped.push(item * item);\n    }\n    result: [i64] = [];\n    i: i64 = mapped.len - 1;\n    while i >= 0 {\n        result.push(mapped[i]);\n        i = i - 1;\n    }\n    return result;\n}\n";
        let problem = pa_ref(rows, reference);
        assert_differential(&problem);
        assert_eq!(
            synthesize_array_transform(&problem)
                .expect("expected a solution")
                .method,
            "array_transform_map_then_reverse"
        );
        assert!(
            synthesize_fixed_map_only(&problem).is_none(),
            "fixed map menu must NOT solve reverse(square(x))"
        );
    }

    #[test]
    fn searched_map_rejects_unlearnable_via_holdout() {
        // Examples fit item*item*item but a holdout breaks it: the searched body
        // path must reject on the holdout (no examples-only accept).
        let problem = pa(&[
            (&[0, 1, 2], &[0, 1, 8]),
            (&[3, 4], &[27, 64]),
            (&[5], &[125]),
            (&[6], &[999]), // 6^3 = 216, inconsistent
        ]);
        assert!(
            synthesize_array_transform(&problem).is_none(),
            "must reject when a holdout violates the searched cube"
        );
    }
}

/// Entry point: synthesize an exact `[i64] -> [i64]` transform, or None.
pub(super) fn synthesize_array_transform(problem: &Problem) -> Option<SolveResult> {
    let rows = array_rows(problem)?;
    let debug = std::env::var("NSYNTH_DEBUG_ARRAY_TRANSFORM").is_ok();
    for (method, code) in candidates(problem, &rows) {
        match verify_problem_code_strict(problem, &code) {
            Ok(()) => {
                if debug {
                    eprintln!("[array_transform] {method}: OK");
                }
                return Some(SolveResult {
                    success: true,
                    code,
                    method: method.to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
            Err(e) if debug => eprintln!("[array_transform] {method}: {e}"),
            Err(_) => {}
        }
    }
    None
}
