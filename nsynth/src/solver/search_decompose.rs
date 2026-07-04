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
    // (list, int-scalar) inputs route to the two-arg schemas: the scalar is a
    // THRESHOLD/OPERAND ("greater than k", "multiply by k"). Same contract:
    // shape hypotheses + data-mined holes + end-to-end verification.
    if first.inputs.len() == 2
        && matches!(first.inputs[0], Value::Array(_))
        && matches!(first.inputs[1], Value::Int(_))
    {
        let name = {
            let n = problem.function_name();
            if n.is_empty() { "f" } else { n }
        };
        IN_DECOMPOSE.with(|f| f.set(true));
        let result = try_map_scalar(problem, name).or_else(|| try_filter_scalar(problem, name));
        IN_DECOMPOSE.with(|f| f.set(false));
        let (code, method) = result?;
        if crate::runtime::code_reproduces_examples(&code, examples) {
            return Some(SolveResult {
                success: true,
                code,
                method,
                error: None,
                metadata: Default::default(),
            });
        }
        return None;
    }
    // Single-input tier: exactly one input, and it is a list.
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
        .or_else(|| try_index_map(problem, name))
        .or_else(|| try_scan(problem, name))
        .or_else(|| try_context_map(problem, name))
        .or_else(|| try_sort_by(problem, name))
        .or_else(|| try_interleave(problem, name))
        .or_else(|| try_median(problem, name))
        .or_else(|| try_filter(problem, name))
        .or_else(|| try_filter_sort(problem, name))
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

// ──────────────────────── H-MAP-SCALAR (2-arg) ────────────────────────

/// (list, k) with same-length list output: out[i] = affine(x, k) — "multiply
/// each by k", "add k to every element". Reuses the exact integer 3-unknown
/// solve with k in the index slot; evidence multiplication applies as usual.
fn try_map_scalar(problem: &Problem, name: &str) -> Option<(String, String)> {
    let mut pairs: Vec<(i64, i64, i64)> = Vec::new(); // (x, k, out)
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        let Value::Int(k) = &ex.inputs[1] else { return None };
        let Value::Array(output) = &ex.expected else { return None };
        if input.len() != output.len() {
            return None;
        }
        for (x, o) in input.iter().zip(output.iter()) {
            let (Value::Int(x), Value::Int(o)) = (x, o) else { return None };
            pairs.push((*x, *k, *o));
        }
    }
    if pairs.len() < 4 {
        return None;
    }
    let fits = |f: &dyn Fn(i64, i64) -> i64| pairs.iter().all(|&(x, k, o)| f(x, k) == o);
    let body: String = if fits(&|x, k| x * k) {
        "x * k".to_string()
    } else {
        let (a, b, c) = solve_affine3(&pairs)?;
        if !fits(&|x, k| a + b * x + c * k) {
            return None;
        }
        let mut terms = Vec::new();
        if b != 0 {
            terms.push(if b == 1 { "x".to_string() } else { format!("{b} * x") });
        }
        if c != 0 {
            terms.push(if c == 1 { "k".to_string() } else { format!("{c} * k") });
        }
        if a != 0 || terms.is_empty() {
            terms.push(a.to_string());
        }
        terms.join(" + ")
    };
    let code = format!(
        "fn {name}(xs: [i64], k: i64) -> [i64] {{\n    out: [i64] = [];\n    for x in xs {{\n        out.push({body});\n    }}\n    return out;\n}}\n"
    );
    Some((code, "decompose-map-scalar".to_string()))
}

// ─────────────────────── H-FILTER-SCALAR (2-arg) ───────────────────────

/// (list, k) with a filtered-subsequence output: the predicate compares each
/// element against the SCALAR ARGUMENT — x > k, x < k, x >= k, x <= k,
/// x == k, x != k, x % k == 0. Labels come per-example (each example carries
/// its own k), so the candidate must separate kept/dropped under EVERY k.
fn try_filter_scalar(problem: &Problem, name: &str) -> Option<(String, String)> {
    // (x, k, kept?) labeled triples from the subsequence walk.
    let mut labeled: Vec<(i64, i64, bool)> = Vec::new();
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        let Value::Int(k) = &ex.inputs[1] else { return None };
        let Value::Array(output) = &ex.expected else { return None };
        if output.len() >= input.len() {
            return None;
        }
        let mut oi = 0;
        for v in input {
            let Value::Int(x) = v else { return None };
            if oi < output.len() && v == &output[oi] {
                labeled.push((*x, *k, true));
                oi += 1;
            } else {
                labeled.push((*x, *k, false));
            }
        }
        if oi != output.len() {
            return None;
        }
    }
    if !labeled.iter().any(|l| l.2) || !labeled.iter().any(|l| !l.2) {
        return None;
    }
    let candidates: [(&str, fn(i64, i64) -> bool); 7] = [
        ("x > k", |x, k| x > k),
        ("x < k", |x, k| x < k),
        ("x >= k", |x, k| x >= k),
        ("x <= k", |x, k| x <= k),
        ("x == k", |x, k| x == k),
        ("x != k", |x, k| x != k),
        ("x % k == 0", |x, k| k != 0 && x % k == 0),
    ];
    for (cond, f) in candidates {
        if labeled.iter().all(|&(x, k, kept)| f(x, k) == kept) {
            let code = format!(
                "fn {name}(xs: [i64], k: i64) -> [i64] {{\n    out: [i64] = [];\n    for x in xs {{\n        if {cond} {{\n            out.push(x);\n        }}\n    }}\n    return out;\n}}\n"
            );
            return Some((code, "decompose-filter-scalar".to_string()));
        }
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
        if input.len() != output.len() {
            return None;
        }
        // Empty lists are vacuous (any program matches []) — skip, don't refuse.
        for (i, o) in input.iter().zip(output.iter()) {
            elem_examples.push(Example { inputs: vec![i.clone()], expected: o.clone() });
        }
    }
    if elem_examples.is_empty() {
        return None; // every example was the empty list — nothing to pin a fn
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

// ─────────────────────────── H-INDEX-MAP ───────────────────────────

/// Same-length list output where the element depends on BOTH value and
/// POSITION: out[i] = f(x, i). The element hole is an exact integer affine
/// c0 + c1·x + c2·i (+ the c1·x·i bilinear special case `derivative`), fitted
/// over the FLATTENED (x, i) -> out pairs — the evidence-multiplication case
/// where a 3-example task hands the fit dozens of points.
fn try_index_map(problem: &Problem, name: &str) -> Option<(String, String)> {
    let mut pairs: Vec<(i64, i64, i64)> = Vec::new(); // (x, i, out)
    let mut skip_first = usize::MAX;
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        let Value::Array(output) = &ex.expected else { return None };
        // Allow out.len == in.len (offset 0) or in.len - 1 (offset 1 — the
        // `derivative` shape drops the constant term).
        let off = input.len().checked_sub(output.len())?;
        if off > 1 || output.is_empty() {
            return None;
        }
        if skip_first == usize::MAX {
            skip_first = off;
        } else if skip_first != off {
            return None;
        }
        for (k, o) in output.iter().enumerate() {
            let idx = k + skip_first;
            let (Value::Int(x), Value::Int(o)) = (&input[idx], o) else { return None };
            pairs.push((*x, idx as i64, *o));
        }
    }
    if pairs.len() < 4 {
        return None; // affine over (x, i) has 3 unknowns — need a spare point
    }
    // Candidate element bodies over (x, i): the bilinear x·i (derivative),
    // then integer affine c0 + c1·x + c2·i solved exactly from 3 points and
    // verified on ALL points.
    let fits = |f: &dyn Fn(i64, i64) -> i64| pairs.iter().all(|&(x, i, o)| f(x, i) == o);
    let body: String = if fits(&|x, i| x * i) {
        "x * i".to_string()
    } else {
        // Solve the 3-unknown integer system from three spanning points.
        let (a, b, c) = solve_affine3(&pairs)?;
        if !fits(&|x, i| a + b * x + c * i) {
            return None;
        }
        let mut terms = Vec::new();
        if b != 0 {
            terms.push(if b == 1 { "x".to_string() } else { format!("{b} * x") });
        }
        if c != 0 {
            terms.push(if c == 1 { "i".to_string() } else { format!("{c} * i") });
        }
        if a != 0 || terms.is_empty() {
            terms.push(a.to_string());
        }
        terms.join(" + ")
    };
    let start = skip_first;
    let code = format!(
        "fn {name}(xs: [i64]) -> [i64] {{\n    out: [i64] = [];\n    i: i64 = {start};\n    while i < xs.len {{\n        x: i64 = xs[i];\n        out.push({body});\n        i = i + 1;\n    }}\n    return out;\n}}\n"
    );
    Some((code, "decompose-index-map".to_string()))
}

/// Exact integer solve of out = a + b·x + c·i from the first three points that
/// span the space (Cramer over i64; None if singular or non-integral).
fn solve_affine3(pairs: &[(i64, i64, i64)]) -> Option<(i64, i64, i64)> {
    for w in 0..pairs.len().saturating_sub(2) {
        let (x1, i1, o1) = pairs[w];
        let (x2, i2, o2) = pairs[w + 1];
        let (x3, i3, o3) = pairs[w + 2];
        let det = (x2 - x1) * (i3 - i1) - (x3 - x1) * (i2 - i1);
        if det == 0 {
            continue;
        }
        let bn = (o2 - o1) * (i3 - i1) - (o3 - o1) * (i2 - i1);
        let cn = (x2 - x1) * (o3 - o1) - (x3 - x1) * (o2 - o1);
        if bn % det != 0 || cn % det != 0 {
            return None;
        }
        let b = bn / det;
        let c = cn / det;
        let a = o1 - b * x1 - c * i1;
        return Some((a, b, c));
    }
    None
}

// ───────────────────────────── H-SCAN ─────────────────────────────

/// Same-length output where out[i] = fold of xs[0..=i] under an associative
/// op — running max/min/sum/product (`rolling_max`). Four candidates checked
/// directly against every prefix; no sub-solve needed.
fn try_scan(problem: &Problem, name: &str) -> Option<(String, String)> {
    let ops: [(&str, fn(i64, i64) -> i64); 4] = [
        ("max", |a, b| a.max(b)),
        ("min", |a, b| a.min(b)),
        ("sum", |a, b| a + b),
        ("prod", |a, b| a.wrapping_mul(b)),
    ];
    'op: for (op_name, f) in ops {
        for ex in &problem.examples {
            let Value::Array(input) = &ex.inputs[0] else { return None };
            let Value::Array(output) = &ex.expected else { return None };
            if input.len() != output.len() {
                return None;
            }
            let mut acc: Option<i64> = None;
            for (x, o) in input.iter().zip(output.iter()) {
                let (Value::Int(x), Value::Int(o)) = (x, o) else { return None };
                let next = match acc {
                    None => *x,
                    Some(a) => f(a, *x),
                };
                if next != *o {
                    continue 'op;
                }
                acc = Some(next);
            }
        }
        let update = match op_name {
            "max" => "if x > acc {\n            acc = x;\n        }",
            "min" => "if x < acc {\n            acc = x;\n        }",
            "sum" => "acc = acc + x;",
            _ => "acc = acc * x;",
        };
        let code = format!(
            "fn {name}(xs: [i64]) -> [i64] {{\n    out: [i64] = [];\n    acc: i64 = 0;\n    first: i64 = 1;\n    for x in xs {{\n        if first == 1 {{\n            acc = x;\n            first = 0;\n        }} else {{\n            {update}\n        }}\n        out.push(acc);\n    }}\n    return out;\n}}\n"
        );
        return Some((code, format!("decompose-scan-{op_name}")));
    }
    None
}

// ─────────────────────────── H-CONTEXT-MAP ───────────────────────────

/// Same-length float/int map where the element transform needs WHOLE-LIST
/// context (min/max/sum/len): generic math idioms tried as fixed templates —
/// rescale-to-unit (x-min)/(max-min), normalize x/sum, x/max, shift x-min.
/// These are universal numeric idioms, not task-shaped ops.
fn try_context_map(problem: &Problem, name: &str) -> Option<(String, String)> {
    let as_f = |v: &Value| -> Option<f64> {
        match v {
            Value::Int(i) => Some(*i as f64),
            Value::Float(b) => Some(f64::from_bits(*b)),
            _ => None,
        }
    };
    // Gather (x, min, max, sum, out) rows.
    let mut rows: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        let Value::Array(output) = &ex.expected else { return None };
        if input.len() != output.len() {
            return None;
        }
        if input.is_empty() {
            continue; // vacuous example
        }
        let xs: Option<Vec<f64>> = input.iter().map(as_f).collect();
        let os: Option<Vec<f64>> = output.iter().map(as_f).collect();
        let (xs, os) = (xs?, os?);
        let (mn, mx, sm) = (
            xs.iter().cloned().fold(f64::INFINITY, f64::min),
            xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            xs.iter().sum::<f64>(),
        );
        for (x, o) in xs.iter().zip(os.iter()) {
            rows.push((*x, mn, mx, sm, *o));
        }
    }
    if rows.len() < 4 {
        return None;
    }
    let eps = 1e-9
        * rows.iter().map(|r| r.4.abs()).fold(1.0f64, f64::max);
    let templates: [(&str, fn(f64, f64, f64, f64) -> f64); 4] = [
        ("rescale", |x, mn, mx, _| (x - mn) / (mx - mn)),
        ("div_sum", |x, _, _, sm| x / sm),
        ("div_max", |x, _, mx, _| x / mx),
        ("sub_min", |x, mn, _, _| x - mn),
    ];
    for (t_name, f) in templates {
        if rows.iter().all(|&(x, mn, mx, sm, o)| {
            let v = f(x, mn, mx, sm);
            v.is_finite() && (v - o).abs() <= eps
        }) {
            let body = match t_name {
                "rescale" => "(x - mn) / (mx - mn)",
                "div_sum" => "x / sm",
                "div_max" => "x / mx",
                _ => "x - mn",
            };
            let code = format!(
                "fn {name}(xs: [i64]) -> [i64] {{\n    mn: f64 = xs[0];\n    mx: f64 = xs[0];\n    sm: f64 = 0.0;\n    for e in xs {{\n        if e < mn {{\n            mn = e;\n        }}\n        if e > mx {{\n            mx = e;\n        }}\n        sm = sm + e;\n    }}\n    out: [i64] = [];\n    for x in xs {{\n        out.push({body});\n    }}\n    return out;\n}}\n"
            );
            return Some((code, format!("decompose-context-{t_name}")));
        }
    }
    None
}

// ──────────────────────────── H-SORT-BY ────────────────────────────

/// Output is a PERMUTATION of the input on every example → try sort keys:
/// value asc/desc, abs asc, and (for strings) length asc/desc. Emitted as a
/// selection sort with the key comparison inlined.
fn try_sort_by(problem: &Problem, name: &str) -> Option<(String, String)> {
    // Permutation check (multiset equality).
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        let Value::Array(output) = &ex.expected else { return None };
        if input.len() != output.len() {
            return None;
        }
        let mut a: Vec<String> = input.iter().map(|v| format!("{v:?}")).collect();
        let mut b: Vec<String> = output.iter().map(|v| format!("{v:?}")).collect();
        a.sort();
        b.sort();
        if a != b {
            return None;
        }
    }
    // Key candidates evaluated in Rust; the winner is emitted as Mog.
    type Key = fn(&Value) -> Option<i64>;
    let val_key: Key = |v| if let Value::Int(i) = v { Some(*i) } else { None };
    let abs_key: Key = |v| if let Value::Int(i) = v { Some(i.abs()) } else { None };
    let len_key: Key = |v| if let Value::Str(s) = v { Some(s.len() as i64) } else { None };
    let candidates: [(&str, Key, bool); 5] = [
        ("val_asc", val_key, false),
        ("val_desc", val_key, true),
        ("abs_asc", abs_key, false),
        ("len_asc", len_key, false),
        ("len_desc", len_key, true),
    ];
    'cand: for (k_name, key, desc) in candidates {
        for ex in &problem.examples {
            let Value::Array(input) = &ex.inputs[0] else { return None };
            let Value::Array(output) = &ex.expected else { return None };
            let mut sorted: Vec<&Value> = input.iter().collect();
            let keyed: Option<Vec<i64>> = sorted.iter().map(|v| key(v)).collect();
            if keyed.is_none() {
                continue 'cand;
            }
            // Stable sort by key preserves original order of equal keys.
            sorted.sort_by_key(|v| {
                let k = key(v).unwrap_or(0);
                if desc { -k } else { k }
            });
            let got: Vec<&Value> = sorted;
            if !got.iter().zip(output.iter()).all(|(g, o)| **g == *o) {
                continue 'cand;
            }
        }
        let elem_ty = elem_scalar_type(&problem.examples[0].inputs[0])?;
        let list_ty = mog_type(&problem.examples[0].inputs[0], true)?;
        let key_expr = match k_name {
            "len_asc" | "len_desc" => "out[j].len",
            _ => "out[j]",
        };
        let key_expr_min = key_expr.replace("[j]", "[m]");
        let cmp = if desc { ">" } else { "<" };
        let code = format!(
            "fn {name}(xs: {list_ty}) -> {list_ty} {{\n    out: {list_ty} = [];\n    for e in xs {{\n        out.push(e);\n    }}\n    i: i64 = 0;\n    while i < out.len {{\n        m: i64 = i;\n        j: i64 = i + 1;\n        while j < out.len {{\n            if {key_expr} {cmp} {key_expr_min} {{\n                m = j;\n            }}\n            j = j + 1;\n        }}\n        t: {elem_ty} = out[i];\n        out[i] = out[m];\n        out[m] = t;\n        i = i + 1;\n    }}\n    return out;\n}}\n"
        );
        return Some((code, format!("decompose-sort-{k_name}")));
    }

    // STRING keys: lexicographic asc/desc and the COMPOUND len-then-alpha
    // (HumanEval 149 class). Mog `<`/`>` are lexicographic on strings by design.
    let str_keys: [(&str, fn(&str, &str) -> std::cmp::Ordering); 3] = [
        ("alpha_asc", |a, b| a.cmp(b)),
        ("alpha_desc", |a, b| b.cmp(a)),
        ("len_alpha_asc", |a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b))),
    ];
    'skey: for (k_name, ord) in str_keys {
        for ex in &problem.examples {
            let Value::Array(input) = &ex.inputs[0] else { return None };
            let Value::Array(output) = &ex.expected else { return None };
            let ins: Option<Vec<&str>> = input
                .iter()
                .map(|v| if let Value::Str(s) = v { Some(s.as_str()) } else { None })
                .collect();
            let Some(mut ins) = ins else { continue 'skey };
            ins.sort_by(|a, b| ord(a, b));
            let outs: Option<Vec<&str>> = output
                .iter()
                .map(|v| if let Value::Str(s) = v { Some(s.as_str()) } else { None })
                .collect();
            let Some(outs) = outs else { continue 'skey };
            if ins != outs {
                continue 'skey;
            }
        }
        let list_ty = mog_type(&problem.examples[0].inputs[0], true)?;
        let swap_cond = match k_name {
            "alpha_asc" => "if out[j] < out[m] {\n                m = j;\n            }".to_string(),
            "alpha_desc" => "if out[j] > out[m] {\n                m = j;\n            }".to_string(),
            _ => "if out[j].len < out[m].len {\n                m = j;\n            } else {\n                if out[j].len == out[m].len {\n                    if out[j] < out[m] {\n                        m = j;\n                    }\n                }\n            }".to_string(),
        };
        let code = format!(
            "fn {name}(xs: {list_ty}) -> {list_ty} {{\n    out: {list_ty} = [];\n    for e in xs {{\n        out.push(e);\n    }}\n    i: i64 = 0;\n    while i < out.len {{\n        m: i64 = i;\n        j: i64 = i + 1;\n        while j < out.len {{\n            {swap_cond}\n            j = j + 1;\n        }}\n        t: string = out[i];\n        out[i] = out[m];\n        out[m] = t;\n        i = i + 1;\n    }}\n    return out;\n}}\n"
        );
        return Some((code, format!("decompose-sort-{k_name}")));
    }
    None
}

// ─────────────────────────── H-INTERLEAVE ───────────────────────────

/// Permutation output built by alternately taking the MIN then MAX of the
/// remaining elements (`strange_sort_list`). Checked directly against every
/// example; emitted as a used-flag selection loop.
fn try_interleave(problem: &Problem, name: &str) -> Option<(String, String)> {
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        let Value::Array(output) = &ex.expected else { return None };
        if input.len() != output.len() {
            return None;
        }
        let mut rest: Vec<i64> = input
            .iter()
            .map(|v| if let Value::Int(i) = v { Some(*i) } else { None })
            .collect::<Option<_>>()?;
        for (k, o) in output.iter().enumerate() {
            let Value::Int(o) = o else { return None };
            let pick = if k % 2 == 0 {
                *rest.iter().min()?
            } else {
                *rest.iter().max()?
            };
            if pick != *o {
                return None;
            }
            let pos = rest.iter().position(|&x| x == pick)?;
            rest.remove(pos);
        }
    }
    let code = format!(
        "fn {name}(xs: [i64]) -> [i64] {{\n    rest: [i64] = [];\n    for e in xs {{\n        rest.push(e);\n    }}\n    out: [i64] = [];\n    take_min: i64 = 1;\n    while rest.len > 0 {{\n        m: i64 = 0;\n        j: i64 = 1;\n        while j < rest.len {{\n            if take_min == 1 {{\n                if rest[j] < rest[m] {{\n                    m = j;\n                }}\n            }} else {{\n                if rest[j] > rest[m] {{\n                    m = j;\n                }}\n            }}\n            j = j + 1;\n        }}\n        out.push(rest[m]);\n        nr: [i64] = [];\n        k: i64 = 0;\n        while k < rest.len {{\n            if k != m {{\n                nr.push(rest[k]);\n            }}\n            k = k + 1;\n        }}\n        rest = nr;\n        take_min = 1 - take_min;\n    }}\n    return out;\n}}\n"
    );
    Some((code, "decompose-interleave-minmax".to_string()))
}

// ───────────────────────────── H-MEDIAN ─────────────────────────────

/// Scalar output equal to the middle of the SORTED input (odd n), or the mean
/// of the two middles (even n — float). A positional idiom, not an element
/// predicate, so H-SELECT cannot express it.
fn try_median(problem: &Problem, name: &str) -> Option<(String, String)> {
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        if input.is_empty() || matches!(ex.expected, Value::Array(_)) {
            return None;
        }
        let mut xs: Vec<i64> = input
            .iter()
            .map(|v| if let Value::Int(i) = v { Some(*i) } else { None })
            .collect::<Option<_>>()?;
        xs.sort_unstable();
        let n = xs.len();
        let want = match &ex.expected {
            Value::Int(o) => *o as f64,
            Value::Float(b) => f64::from_bits(*b),
            _ => return None,
        };
        let got = if n % 2 == 1 {
            xs[n / 2] as f64
        } else {
            (xs[n / 2 - 1] + xs[n / 2]) as f64 / 2.0
        };
        if (got - want).abs() > 1e-9 * want.abs().max(1.0) {
            return None;
        }
    }
    let code = format!(
        "fn {name}(xs: [i64]) -> f64 {{\n    s: [i64] = [];\n    for e in xs {{\n        s.push(e);\n    }}\n    s.sort();\n    n: i64 = s.len;\n    if n % 2 == 1 {{\n        return 1.0 * s[n / 2];\n    }}\n    return (s[n / 2 - 1] + s[n / 2]) / 2.0;\n}}\n"
    );
    Some((code, "decompose-median".to_string()))
}

// ─────────────────────── H-FILTER∘SORT (composed) ───────────────────────

/// First composed schema (depth-2): output is a SORTED, FILTERED subset of the
/// input — `keep the evens, sorted` / sorted_list_sum shapes. The pure filter
/// hypothesis requires an order-preserving subsequence and refuses these; here
/// the labels come from MULTISET difference (order-free), the predicate is
/// synthesized as usual, and the second stage checks the output is the kept
/// elements under one of the sort keys.
fn try_filter_sort(problem: &Problem, name: &str) -> Option<(String, String)> {
    let mut kept: Vec<Value> = Vec::new();
    let mut dropped: Vec<Value> = Vec::new();
    for ex in &problem.examples {
        let Value::Array(input) = &ex.inputs[0] else { return None };
        let Value::Array(output) = &ex.expected else { return None };
        if output.len() > input.len() {
            return None;
        }
        // Multiset difference: every output element must come from the input.
        let mut pool: Vec<&Value> = input.iter().collect();
        for o in output {
            let pos = pool.iter().position(|v| *v == o)?;
            pool.remove(pos);
            kept.push(o.clone());
        }
        for v in pool {
            dropped.push(v.clone());
        }
    }
    if kept.is_empty() || dropped.is_empty() {
        return None; // degenerate — pure sort (H-SORT-BY) or pure identity
    }
    let pred_fn = synthesize_predicate_with_text(&kept, &dropped, problem.description)?;
    // Stage 2: which sort key arranges each example's kept-set into the output?
    let val_key = |v: &Value| if let Value::Int(i) = v { Some(*i) } else { None };
    let len_key = |v: &Value| if let Value::Str(s) = v { Some(s.len() as i64) } else { None };
    let keys: [(&str, &dyn Fn(&Value) -> Option<i64>, bool); 4] = [
        ("val_asc", &val_key, false),
        ("val_desc", &val_key, true),
        ("len_asc", &len_key, false),
        ("len_desc", &len_key, true),
    ];
    // Compound len-then-alpha (HumanEval 149 class): monotone under (len, str).
    let compound_ok = problem.examples.iter().all(|ex| {
        let Value::Array(output) = &ex.expected else { return false };
        let outs: Option<Vec<&str>> = output
            .iter()
            .map(|v| if let Value::Str(s) = v { Some(s.as_str()) } else { None })
            .collect();
        let Some(outs) = outs else { return false };
        outs.windows(2)
            .all(|w| w[0].len() < w[1].len() || (w[0].len() == w[1].len() && w[0] <= w[1]))
    });
    if compound_ok {
        let list_ty = mog_type(&problem.examples[0].inputs[0], true)?;
        let code = format!(
            "fn {name}(xs: {list_ty}) -> {list_ty} {{\n    out: {list_ty} = [];\n    for e in xs {{\n        if pred(e) {{\n            out.push(e);\n        }}\n    }}\n    i: i64 = 0;\n    while i < out.len {{\n        m: i64 = i;\n        j: i64 = i + 1;\n        while j < out.len {{\n            if out[j].len < out[m].len {{\n                m = j;\n            }} else {{\n                if out[j].len == out[m].len {{\n                    if out[j] < out[m] {{\n                        m = j;\n                    }}\n                }}\n            }}\n            j = j + 1;\n        }}\n        t: string = out[i];\n        out[i] = out[m];\n        out[m] = t;\n        i = i + 1;\n    }}\n    return out;\n}}\n\n{pred_fn}"
        );
        return Some((code, "decompose-filter-sort-len_alpha".to_string()));
    }
    'key: for (k_name, key, desc) in keys {
        for ex in &problem.examples {
            let Value::Array(output) = &ex.expected else { return None };
            let mut ks: Vec<i64> = Vec::with_capacity(output.len());
            for o in output {
                ks.push(key(o)?);
            }
            let sorted_ok = if desc {
                ks.windows(2).all(|w| w[0] >= w[1])
            } else {
                ks.windows(2).all(|w| w[0] <= w[1])
            };
            if !sorted_ok {
                continue 'key;
            }
        }
        let elem_ty = elem_scalar_type(&problem.examples[0].inputs[0])?;
        let list_ty = mog_type(&problem.examples[0].inputs[0], true)?;
        let key_expr = match k_name {
            "len_asc" | "len_desc" => "out[j].len",
            _ => "out[j]",
        };
        let key_expr_min = key_expr.replace("[j]", "[m]");
        let cmp = if desc { ">" } else { "<" };
        let code = format!(
            "fn {name}(xs: {list_ty}) -> {list_ty} {{\n    out: {list_ty} = [];\n    for e in xs {{\n        if pred(e) {{\n            out.push(e);\n        }}\n    }}\n    i: i64 = 0;\n    while i < out.len {{\n        m: i64 = i;\n        j: i64 = i + 1;\n        while j < out.len {{\n            if {key_expr} {cmp} {key_expr_min} {{\n                m = j;\n            }}\n            j = j + 1;\n        }}\n        t: {elem_ty} = out[i];\n        out[i] = out[m];\n        out[m] = t;\n        i = i + 1;\n    }}\n    return out;\n}}\n\n{pred_fn}"
        );
        return Some((code, format!("decompose-filter-sort-{k_name}")));
    }
    None
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
    let pred_fn = synthesize_predicate_with_text(&kept, &dropped, problem.description)?;
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
    let pred_fn = synthesize_predicate_with_text(&kept, &dropped, problem.description)?;
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
    synthesize_predicate_with_text(kept, dropped, "")
}

/// Like `synthesize_predicate` but also mines integer constants from the task
/// DESCRIPTION (outside-audit lever #1: the text is part of the spec — "greater
/// than 3" puts 3 in the candidate pool even when no labeled element equals 3).
fn synthesize_predicate_with_text(kept: &[Value], dropped: &[Value], text: &str) -> Option<String> {
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
            // Text-mined constants: integers named in the task description.
            for tok in text.split(|c: char| !c.is_ascii_digit() && c != '-') {
                if let Ok(c) = tok.parse::<i64>() {
                    candidates.push(format!("x < {c}"));
                    candidates.push(format!("x > {c}"));
                    candidates.push(format!("x == {c}"));
                    candidates.push(format!("x % {} == 0", c.max(2)));
                }
            }
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
            candidates.push("x.len % 2 == 0".into());
            candidates.push("x.len % 2 != 0".into());
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
