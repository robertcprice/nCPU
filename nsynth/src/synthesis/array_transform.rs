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
        .map(|ex| match (ex.inputs.as_slice(), &ex.expected) {
            ([Value::Array(input)], Value::Array(output)) => Some((input.clone(), output.clone())),
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
            inputs: vec![Value::Array(input.to_vec())],
            expected: Value::Array(output.to_vec()),
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
                inputs: vec![Value::Array(vec![1, 2, 3])],
                expected: Value::Int(6),
            }],
            ..pa(&[(&[1], &[1])])
        };
        assert!(synthesize_array_transform(&problem).is_none());
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
