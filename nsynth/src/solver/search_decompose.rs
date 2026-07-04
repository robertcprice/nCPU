//! STRUCTURAL DECOMPOSITION tier — the first emergent-ops increment.
//!
//! The measured miss tails (MBPP string tail + HumanEval timeouts) are
//! dominated by map/filter/select shapes over lists — especially string-element
//! lists, which the int-array machinery skips entirely. Instead of hand-writing
//! a task-shaped op per miss, this tier synthesizes the STRUCTURE and solves the
//! small hole inside it:
//!
//!   H-MAP     output list, same length as an input list
//!             -> solve ONE element-level function and wrap it in a for-push
//!   H-FILTER  output list is an order-preserving subsequence of an input list
//!             -> synthesize a predicate from the labeled kept/dropped sets
//!   H-SELECT  scalar output that equals one element of an input list
//!             -> synthesize the selecting predicate/extremum
//!
//! Why this is SOUNDER than whole-program search, not looser: decomposition
//! MULTIPLIES EVIDENCE. A 3-example task over 8-element lists yields 24
//! element-level pairs for the sub-solve, and a filter hypothesis yields two
//! labeled sets — far more signal than 3 whole-program checks. On top of that
//! the assembled program is re-verified END-TO-END against the full example set
//! before acceptance (the same contract as `search_tuple`).
//!
//! Why this is EMERGENT, not hand-tuned: the element-level hole is solved by
//! reusing the ENTIRE existing verified op library as a component basis (ops
//! stop being whole answers and become primitives the machine composes), plus a
//! small fixed predicate grammar whose constants are MINED FROM THE TASK'S OWN
//! DATA (never hand-listed). No new task-shaped op is added here.

use crate::benchmark::{Example, Problem, Value};
use crate::solver::SolveResult;

thread_local! {
    static IN_DECOMPOSE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Attempt a structural decomposition solve. Fast `None` when no hypothesis
/// shape matches. Runs its own end-to-end verification before returning.
pub fn try_decompose(problem: &Problem) -> Option<SolveResult> {
    if IN_DECOMPOSE.with(|f| f.get()) {
        return None;
    }
    let examples = &problem.examples;
    let first = examples.first()?;
    // v1 scope: exactly one input, and it is a list. (Scalar pass-through args
    // and multi-list inputs are later increments.)
    if first.inputs.len() != 1 {
        return None;
    }
    let Value::Array(_) = &first.inputs[0] else { return None };

    let name = {
        let n = problem.function_name();
        if n.is_empty() { "f" } else { n }
    };

    IN_DECOMPOSE.with(|f| f.set(true));
    let result = try_map(problem, name)
        .or_else(|| try_filter(problem, name))
        .or_else(|| try_select(problem, name));
    IN_DECOMPOSE.with(|f| f.set(false));

    let (code, method) = result?;
    // End-to-end acceptance on the FULL example set — a plausible per-element
    // fit that does not reproduce the whole task is rejected here.
    if crate::runtime::code_reproduces_examples(&code, examples) {
        return Some(SolveResult {
            success: true,
            code,
            method,
            error: None,
            metadata: Default::default(),
        });
    }
    None
}

// ───────────────────────────── H-MAP ─────────────────────────────

/// Output list with the same length as the input list on EVERY example →
/// flatten to element-level pairs and solve one element function.
fn try_map(problem: &Problem, name: &str) -> Option<(String, String)> {
    let mut elem_examples: Vec<Example> = Vec::new();
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        let Value::Array(output) = &ex.expected else { return None };
        if input.len() != output.len() || input.is_empty() {
            return None;
        }
        for (i, o) in input.iter().zip(output.iter()) {
            elem_examples.push(Example { inputs: vec![i.clone()], expected: o.clone() });
        }
    }
    // The evidence multiplication: elem_examples.len() >> examples.len().
    let body = solve_element_fn(&elem_examples, "elem")?;
    let elem_ty = mog_type(&problem.examples[0].inputs[0], true)?;
    let out_ty = mog_type(&problem.examples[0].expected, false)?;
    let code = format!(
        "fn {name}(xs: {elem_ty}) -> {out_ty} {{\n    out: {out_ty} = [];\n    for x in xs {{\n        out.push(elem(x));\n    }}\n    return out;\n}}\n\n{body}"
    );
    Some((code, "decompose-map".to_string()))
}

// ──────────────────────────── H-FILTER ────────────────────────────

/// Output list is an order-preserving subsequence of the input list on every
/// example → label elements kept/dropped and synthesize a predicate.
fn try_filter(problem: &Problem, name: &str) -> Option<(String, String)> {
    let mut kept: Vec<Value> = Vec::new();
    let mut dropped: Vec<Value> = Vec::new();
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        let Value::Array(output) = &ex.expected else { return None };
        if output.len() >= input.len() {
            return None; // not a strict filter shape (map/identity handled above)
        }
        // Greedy order-preserving subsequence match; ambiguity is fine — any
        // consistent labeling that survives the END-TO-END check is correct.
        let mut oi = 0;
        for v in input {
            if oi < output.len() && v == &output[oi] {
                kept.push(v.clone());
                oi += 1;
            } else {
                dropped.push(v.clone());
            }
        }
        if oi != output.len() {
            return None; // output is not a subsequence — not a filter
        }
    }
    if kept.is_empty() || dropped.is_empty() {
        return None; // degenerate labeling cannot pin a predicate
    }
    let pred_fn = synthesize_predicate(&kept, &dropped)?;
    let elem_ty = mog_type(&problem.examples[0].inputs[0], true)?;
    let code = format!(
        "fn {name}(xs: {elem_ty}) -> {elem_ty} {{\n    out: {elem_ty} = [];\n    for x in xs {{\n        if pred(x) {{\n            out.push(x);\n        }}\n    }}\n    return out;\n}}\n\n{pred_fn}"
    );
    Some((code, "decompose-filter".to_string()))
}

// ──────────────────────────── H-SELECT ────────────────────────────

/// Scalar output that is always an element of the input list → the unique
/// element satisfying a predicate (first match). Extremum selects are already
/// covered by the library; this catches predicate-selects over strings.
fn try_select(problem: &Problem, name: &str) -> Option<(String, String)> {
    let mut kept: Vec<Value> = Vec::new();
    let mut dropped: Vec<Value> = Vec::new();
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        if matches!(ex.expected, Value::Array(_)) {
            return None;
        }
        let hit = input.iter().position(|v| *v == ex.expected)?;
        kept.push(input[hit].clone());
        for (i, v) in input.iter().enumerate() {
            if i != hit {
                dropped.push(v.clone());
            }
        }
    }
    if kept.is_empty() || dropped.is_empty() {
        return None;
    }
    let pred_fn = synthesize_predicate(&kept, &dropped)?;
    let elem_ty = mog_type(&problem.examples[0].inputs[0], true)?;
    let inner = elem_scalar_type(&problem.examples[0].inputs[0])?;
    let code = format!(
        "fn {name}(xs: {elem_ty}) -> {inner} {{\n    for x in xs {{\n        if pred(x) {{\n            return x;\n        }}\n    }}\n    return xs[0];\n}}\n\n{pred_fn}"
    );
    Some((code, "decompose-select".to_string()))
}

// ───────────────────── element-level sub-solve ─────────────────────

/// Solve a single-argument element function that reproduces every element pair.
/// Component basis = the EXISTING verified op library (arity-1, type-matched),
/// tried by behavior — no new hand ops. Falls back to the tiny identity/const
/// primitives the library has no named entry for.
fn solve_element_fn(elem_examples: &[Example], fn_name: &str) -> Option<String> {
    // Identity.
    if elem_examples.iter().all(|e| e.inputs[0] == e.expected) {
        let ty = scalar_ty(&elem_examples[0].inputs[0])?;
        return Some(format!("fn {fn_name}(x: {ty}) -> {ty} {{\n    return x;\n}}\n"));
    }
    // Constant.
    if elem_examples.windows(2).all(|w| w[0].expected == w[1].expected) {
        if let Value::Int(c) = &elem_examples[0].expected {
            let ty = scalar_ty(&elem_examples[0].inputs[0])?;
            return Some(format!("fn {fn_name}(x: {ty}) -> i64 {{\n    return {c};\n}}\n"));
        }
    }
    // The op library as a component basis: any arity-1 op whose behavior
    // reproduces EVERY element pair becomes the element function.
    let sub = Problem {
        name: format!("{fn_name}_sub"),
        examples: elem_examples.to_vec(),
        ..Problem::default()
    };
    if let Some(res) = crate::op_library::try_library(&sub) {
        return Some(rename_entry_fn(&res.code, fn_name));
    }
    None
}

// ─────────────────── predicate synthesis (emergent) ───────────────────

/// A COMPLETE `fn pred(x: T) -> bool` separating `kept` from `dropped`, from a
/// FIXED finite grammar
/// whose constants are mined from the labeled data itself:
///   int elements:    x < c | x > c | x == c | x % 2 == 0|1 | x >= 0 | x < 0
///                    (c mined from the boundary between the two sets)
///   string elements: x.len cmp k (k mined) | all-chars class | contains char c
///                    (c mined from the kept/dropped character sets)
fn synthesize_predicate(kept: &[Value], dropped: &[Value]) -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();
    match kept.first()? {
        Value::Int(_) => {
            let ints = |vs: &[Value]| -> Option<Vec<i64>> {
                vs.iter().map(|v| if let Value::Int(i) = v { Some(*i) } else { None }).collect()
            };
            let k = ints(kept)?;
            let d = ints(dropped)?;
            candidates.push("x % 2 == 0".into());
            candidates.push("x % 2 != 0".into());
            candidates.push("x > 0".into());
            candidates.push("x < 0".into());
            candidates.push("x >= 0".into());
            // Mined thresholds: the boundary values between the two sets.
            for &c in k.iter().chain(d.iter()) {
                candidates.push(format!("x < {c}"));
                candidates.push(format!("x > {c}"));
                candidates.push(format!("x == {c}"));
                candidates.push(format!("x != {c}"));
            }
        }
        Value::Str(_) => {
            let strs = |vs: &[Value]| -> Option<Vec<String>> {
                vs.iter()
                    .map(|v| if let Value::Str(s) = v { Some(s.clone()) } else { None })
                    .collect()
            };
            let k = strs(kept)?;
            let d = strs(dropped)?;
            // Length thresholds mined from the observed lengths.
            for l in k.iter().chain(d.iter()).map(|s| s.len()) {
                candidates.push(format!("x.len >= {l}"));
                candidates.push(format!("x.len > {l}"));
                candidates.push(format!("x.len < {l}"));
                candidates.push(format!("x.len == {l}"));
            }
            // Contains-char: chars that appear in every kept string are the
            // only viable witnesses — mined, not listed.
            if let Some(first) = k.first() {
                for ch in first.chars().filter(|c| c.is_ascii_alphanumeric()) {
                    if k.iter().all(|s| s.contains(ch)) {
                        candidates.push(format!("has_{ch}((x))")); // placeholder, expanded below
                    }
                }
            }
        }
        _ => return None,
    }
    // Evaluate each candidate against BOTH labeled sets; first full separator
    // wins. Evaluation runs through the real interpreter so the accepted
    // condition means exactly what the emitted program will mean.
    let inner = scalar_ty(kept.first()?)?;
    for cond in candidates {
        // contains-char placeholders expand to a scan loop instead of an expr.
        let code = if let Some(ch) = cond.strip_prefix("has_").and_then(|r| r.chars().next()) {
            format!(
                "fn pred(x: {inner}) -> bool {{\n    for ch in x {{\n        if ch == '{ch}' {{\n            return true;\n        }}\n    }}\n    return false;\n}}\n"
            )
        } else {
            format!("fn pred(x: {inner}) -> bool {{\n    return {cond};\n}}\n")
        };
        let ok_kept = kept.iter().all(|v| pred_eval(&code, v) == Some(true));
        let ok_dropped = dropped.iter().all(|v| pred_eval(&code, v) == Some(false));
        if ok_kept && ok_dropped {
            // Contract: return the COMPLETE `fn pred(...) -> bool { ... }`
            // source — exactly what was just evaluated, so the accepted
            // predicate and the emitted predicate cannot diverge.
            return Some(code);
        }
    }
    None
}

/// Run a candidate `pred` fn on one value through the interpreter.
fn pred_eval(code: &str, v: &Value) -> Option<bool> {
    match crate::runtime::execute_function(code, "pred", &[v.clone()], "pred") {
        Ok(crate::runtime::Value::Bool(b)) => Some(b),
        Ok(crate::runtime::Value::Int(i)) => Some(i != 0),
        _ => None,
    }
}

// ───────────────────────────── helpers ─────────────────────────────

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

/// Mog scalar type for an element value.
fn scalar_ty(v: &Value) -> Option<&'static str> {
    match v {
        Value::Int(_) => Some("i64"),
        Value::Str(_) => Some("string"),
        Value::Bool(_) => Some("bool"),
        _ => None,
    }
}

/// Mog type for a LIST value (`[i64]` / `[string]`), or the scalar type of a
/// non-list when `want_list` is false.
fn mog_type(v: &Value, want_list: bool) -> Option<String> {
    match v {
        Value::Array(xs) => {
            let inner = xs.first().and_then(scalar_ty).unwrap_or("i64");
            Some(format!("[{inner}]"))
        }
        _ if !want_list => scalar_ty(v).map(|s| s.to_string()),
        _ => None,
    }
}

/// Scalar type of a list's elements.
fn elem_scalar_type(list: &Value) -> Option<&'static str> {
    match list {
        Value::Array(xs) => xs.first().and_then(scalar_ty),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prob(exs: Vec<(Vec<Value>, Value)>) -> Problem {
        Problem {
            name: "t".to_string(),
            examples: exs
                .into_iter()
                .map(|(inputs, expected)| Example { inputs, expected })
                .collect(),
            ..Problem::default()
        }
    }

    fn sarr(xs: &[&str]) -> Value {
        Value::Array(xs.iter().map(|s| Value::Str(s.to_string())).collect())
    }

    #[test]
    fn filter_strings_by_mined_length_threshold() {
        // Keep words with len >= 4 — the threshold must be MINED from the data,
        // and the predicate must separate both labeled sets exactly.
        let p = prob(vec![
            (vec![sarr(&["hi", "door", "at", "wall"])], sarr(&["door", "wall"])),
            (vec![sarr(&["sun", "moonlight"])], sarr(&["moonlight"])),
            (vec![sarr(&["abcd", "xy", "zzzz"])], sarr(&["abcd", "zzzz"])),
        ]);
        let r = try_decompose(&p).expect("filter shape must solve");
        assert_eq!(r.method, "decompose-filter");
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
    }

    #[test]
    fn map_strings_via_library_component() {
        // Uppercase every word: the element fn comes from the EXISTING op
        // library (toggle_case/keep-style ops) reused as a component — no new
        // hand op. toggle_case on lowercase inputs = uppercase.
        let p = prob(vec![
            (vec![sarr(&["ab", "cd"])], sarr(&["AB", "CD"])),
            (vec![sarr(&["xyz"])], sarr(&["XYZ"])),
            (vec![sarr(&["q", "rs", "t"])], sarr(&["Q", "RS", "T"])),
        ]);
        let r = try_decompose(&p).expect("map shape must solve via a library component");
        assert_eq!(r.method, "decompose-map");
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
    }

    #[test]
    fn select_string_by_predicate() {
        // Return the first word containing 'z' — contains-char witness is mined
        // from the kept set.
        let p = prob(vec![
            (
                vec![sarr(&["ab", "fizz", "cd"])],
                Value::Str("fizz".to_string()),
            ),
            (
                vec![sarr(&["zebra", "cat"])],
                Value::Str("zebra".to_string()),
            ),
            (
                vec![sarr(&["dog", "haze"])],
                Value::Str("haze".to_string()),
            ),
        ]);
        let r = try_decompose(&p).expect("select shape must solve");
        assert_eq!(r.method, "decompose-select");
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
    }

    #[test]
    fn refuses_non_structural_relationship() {
        // Output is unrelated to any structural hypothesis — must refuse, never
        // fabricate.
        let p = prob(vec![
            (vec![sarr(&["ab", "cd"])], sarr(&["qq", "ww", "ee"])),
            (vec![sarr(&["x"])], sarr(&["r", "t"])),
            (vec![sarr(&["m", "n"])], sarr(&["a"])),
        ]);
        assert!(try_decompose(&p).is_none());
    }
}
