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

        // Prefix-sum (running scan).
        out.push((
            "array_transform_prefix_sum",
            format!(
                "fn {fn_name}(arr: [i64]) -> [i64] {{\n    result: [i64] = [];\n    acc: i64 = 0;\n    for item in arr {{\n        acc = acc + item;\n        result.push(acc);\n    }}\n    return result;\n}}\n"
            ),
        ));
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
            if method == "array_transform_map_searched_quadratic" {
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
