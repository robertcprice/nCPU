//! Fixed-arity TUPLE-OUTPUT synthesis tier.
//!
//! A large slice of MBPP returns a small fixed-length composite — `(min, max)`,
//! `(quotient, remainder)`, `(sum, product)`, `(count, total)`. The runtime can
//! already REPRESENT such a value (a `Value::Array` of scalars, built by the Mog
//! array literal `[a, b]`), but no solver EMITTED one: the array machinery targets
//! variable-length transforms, and the scalar tiers each produce a single number.
//! So these tasks were representable-but-unsolvable — the engine could hold the
//! answer yet had no tier that attempted it.
//!
//! This tier closes that gap COMPOSITIONALLY. When every example output is a
//! length-K array (K constant, 2..=4) of scalars, it solves each output COLUMN as
//! an independent sub-problem — reusing the already-verified reference-op library,
//! the composition pipeline, and a few cheap primitives (constant / input
//! passthrough) — then assembles a multi-function Mog program:
//!
//! ```text
//! fn f(a0: i64, a1: i64) -> [i64] { return [col0(a0, a1), col1(a0, a1)]; }
//! fn col0(...) -> i64 { <verified body for column 0> }
//! fn col1(...) -> i64 { <verified body for column 1> }
//! ```
//!
//! The ENTRY function is emitted FIRST because the verifier
//! (`runtime::code_reproduces_examples`) takes the first `fn` as the entry point.
//! Calls are positional, so each `colN` keeps whatever parameter names its source
//! op used. Every column is example-verified on its own, and the assembled program
//! is re-verified end-to-end against the FULL example set before acceptance — so a
//! spurious per-column match cannot produce a false solve.

use crate::benchmark::{Example, Problem, Value};
use crate::solver::SolveResult;

/// Re-entrancy guard: the per-column solve calls back into `op_library` /
/// `op_pipeline` (cheap, bounded) but must never recurse into this tier again.
thread_local! {
    static IN_TUPLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Attempt a fixed-K tuple/pair output solve. Returns `None` (fast) for any
/// problem whose outputs are not all same-length scalar arrays with 2..=4 columns.
pub fn try_tuple(problem: &Problem) -> Option<SolveResult> {
    if IN_TUPLE.with(|f| f.get()) {
        return None;
    }
    let examples = &problem.examples;
    let first = examples.first()?;

    // Gate: every output is a Value::Array of the SAME length K in 2..=4, and every
    // element is a scalar Int/Bool (not a nested array — that is a transform).
    let k = match &first.expected {
        Value::Array(v) if (2..=4).contains(&v.len()) => v.len(),
        _ => return None,
    };
    for ex in examples {
        let Value::Array(v) = &ex.expected else { return None };
        if v.len() != k || !v.iter().all(is_scalar) {
            return None;
        }
    }
    // Inputs must be types we can declare in the entry signature.
    let param_types: Vec<&'static str> = first
        .inputs
        .iter()
        .map(value_type_str)
        .collect::<Option<Vec<_>>>()?;

    // Solve each column independently.
    let mut col_defs = Vec::with_capacity(k);
    IN_TUPLE.with(|f| f.set(true));
    let solved: Option<Vec<String>> = (0..k)
        .map(|i| solve_column(problem, i))
        .collect();
    IN_TUPLE.with(|f| f.set(false));
    let bodies = solved?;
    for (i, body) in bodies.into_iter().enumerate() {
        col_defs.push(rename_entry_fn(&body, &format!("col{i}")));
    }

    // Assemble the multi-function program (entry FIRST so the verifier calls it).
    let name = {
        let n = problem.function_name();
        if n.is_empty() { "f" } else { n }
    };
    let params: Vec<String> = param_types
        .iter()
        .enumerate()
        .map(|(j, ty)| format!("a{j}: {ty}"))
        .collect();
    let call_args: Vec<String> = (0..param_types.len()).map(|j| format!("a{j}")).collect();
    let call_args = call_args.join(", ");
    let calls: Vec<String> = (0..k).map(|i| format!("col{i}({call_args})")).collect();
    let entry = format!(
        "fn {name}({}) -> [i64] {{\n    return [{}];\n}}\n",
        params.join(", "),
        calls.join(", ")
    );
    let mut code = entry;
    for def in &col_defs {
        code.push('\n');
        code.push_str(def);
    }

    // End-to-end re-verification on the FULL example set. Accept only on a full
    // reproduction — a spurious per-column fit cannot slip through.
    if crate::runtime::code_reproduces_examples(&code, examples) {
        return Some(SolveResult {
            success: true,
            code,
            method: format!("tuple-columns-{k}"),
            error: None,
            metadata: Default::default(),
        });
    }
    None
}

/// Solve output column `i`: return a single-fn Mog body (`fn NAME(params) -> i64
/// { ... }`) whose value reproduces `out[i]` on every example, or `None`.
fn solve_column(problem: &Problem, i: usize) -> Option<String> {
    let col_examples: Vec<Example> = problem
        .examples
        .iter()
        .map(|ex| {
            let Value::Array(v) = &ex.expected else { unreachable!() };
            Example { inputs: ex.inputs.clone(), expected: v[i].clone() }
        })
        .collect();

    // 1. Cheap primitives (constant column, integer-argument passthrough). These
    //    cover columns the op library has no named entry for (e.g. "return the
    //    input unchanged" or "always 0").
    if let Some(code) = primitive_column(&col_examples) {
        return Some(code);
    }

    // 2. FLOAT column: route to the float lanes (affine then poly) — they
    //    self-gate on a `-> f64` signature, carry over-determination guards,
    //    and re-verify the emitted program. Covers parabola-vertex-style pairs.
    if col_examples.iter().any(|e| matches!(e.expected, Value::Float(_))) {
        let sub = Problem {
            name: format!("{}_col{i}", problem.name),
            signature: "fn col() -> f64",
            examples: col_examples,
            ..Problem::default()
        };
        if let Some(res) = super::search_float::search_float_affine(&sub, "col") {
            if res.success {
                return Some(res.code);
            }
        }
        if let Some(res) = super::search_float::search_float_poly(&sub, "col") {
            if res.success {
                return Some(res.code);
            }
        }
        return None;
    }

    // 3. Reuse the verified reference-op library + composition pipeline on the
    //    projected sub-problem. Each already runs a full example verification.
    let sub = Problem {
        name: format!("{}_col{i}", problem.name),
        examples: col_examples,
        ..Problem::default()
    };
    if let Some(res) = crate::op_library::try_library(&sub) {
        return Some(res.code);
    }
    if let Some(res) = crate::op_pipeline::try_pipeline(&sub) {
        return Some(res.code);
    }
    None
}

/// Constant column, or a column equal to one integer input argument.
fn primitive_column(examples: &[Example]) -> Option<String> {
    let first = examples.first()?;
    // Constant: every output identical (and an Int).
    if let Value::Int(c) = &first.expected {
        if examples.iter().all(|e| e.expected == first.expected) {
            return Some(format!("fn col(x: i64) -> i64 {{\n    return {c};\n}}\n"));
        }
    }
    // Integer-argument passthrough: some arg position j is Int in every example
    // and equals the column output everywhere.
    let arity = first.inputs.len();
    for j in 0..arity {
        let matches = examples.iter().all(|e| {
            matches!(&e.inputs.get(j), Some(Value::Int(_))) && e.inputs[j] == e.expected
        });
        if matches {
            // Declare all params (positional call), return the j-th.
            let params: Vec<String> = (0..arity).map(|p| format!("a{p}: i64")).collect();
            return Some(format!(
                "fn col({}) -> i64 {{\n    return a{j};\n}}\n",
                params.join(", ")
            ));
        }
    }
    None
}

/// Rename the FIRST `fn <old>(` in `code` to `fn <new>(`, leaving the body intact.
fn rename_entry_fn(code: &str, new: &str) -> String {
    let Some(old) = code
        .split("fn ")
        .nth(1)
        .and_then(|s| s.split('(').next())
        .map(str::trim)
    else {
        return code.to_string();
    };
    code.replacen(&format!("fn {old}"), &format!("fn {new}"), 1)
}

fn is_scalar(v: &Value) -> bool {
    // Float columns are solved by the float lanes (affine/poly) per column —
    // parabola vertex/focus tasks return fixed pairs of floats.
    matches!(v, Value::Int(_) | Value::Bool(_) | Value::Float(_))
}

/// Mog type string for an input value, or `None` for a shape we can't declare.
fn value_type_str(v: &Value) -> Option<&'static str> {
    match v {
        Value::Int(_) => Some("i64"),
        Value::Bool(_) => Some("bool"),
        Value::Str(_) => Some("string"),
        Value::Array(_) => Some("[i64]"),
        Value::Float(_) => Some("f64"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Value;

    fn prob(name: &str, exs: Vec<(Vec<Value>, Value)>) -> Problem {
        Problem {
            name: name.to_string(),
            examples: exs
                .into_iter()
                .map(|(inputs, expected)| Example { inputs, expected })
                .collect(),
            ..Problem::default()
        }
    }

    fn arr(xs: &[i64]) -> Value {
        Value::Array(xs.iter().map(|&x| Value::Int(x)).collect())
    }

    #[test]
    fn solves_swap_pair_via_argument_passthrough() {
        // (a, b) -> [b, a]: both columns are integer-argument passthroughs, the
        // cheapest column solver. No library op needed.
        let p = prob(
            "swap",
            vec![
                (vec![Value::Int(10), Value::Int(20)], arr(&[20, 10])),
                (vec![Value::Int(15), Value::Int(17)], arr(&[17, 15])),
                (vec![Value::Int(100), Value::Int(200)], arr(&[200, 100])),
            ],
        );
        let r = try_tuple(&p).expect("swap-pair should solve");
        assert!(r.success);
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
        assert!(r.method.starts_with("tuple-columns"));
    }

    #[test]
    fn rejects_variable_length_output() {
        // A list-intersection task only *looks* fixed-K in these examples; the gate
        // must still fire (all K==2 here) but the per-column assembly cannot
        // reproduce a true intersection, so acceptance must fail — never a false solve.
        let p = prob(
            "intersect",
            vec![
                (vec![arr(&[1, 2, 3]), arr(&[2, 3, 9])], arr(&[2, 3])),
                (vec![arr(&[5, 6, 7]), arr(&[6, 7, 1])], arr(&[6, 7])),
                (vec![arr(&[4, 8, 2]), arr(&[8, 2, 0])], arr(&[8, 2])),
            ],
        );
        assert!(try_tuple(&p).is_none(), "must not fabricate an intersection solve");
    }

    #[test]
    fn ignores_scalar_and_plain_array_outputs() {
        // Scalar output: not a tuple task -> instant None (no interference with the
        // scalar tiers).
        let scalar = prob(
            "double",
            vec![
                (vec![Value::Int(1)], Value::Int(2)),
                (vec![Value::Int(2)], Value::Int(4)),
                (vec![Value::Int(3)], Value::Int(6)),
            ],
        );
        assert!(try_tuple(&scalar).is_none());
    }
}
