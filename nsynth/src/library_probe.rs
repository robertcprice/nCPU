//! Behaviour-driven library resolution for repo repair (never-wrong, model-free).
//!
//! The NL front door (`verified_nl_router::answer`) resolves a library op from the
//! PROSE naming it. But a bare "fix the failing test" carries no prose — and the
//! lg-core resolver's `count`→`length` collapse means even descriptive prose can
//! mis-ground. This module sidesteps NL entirely: it takes the failing test's mined
//! I/O examples and asks the VERIFIED op library "which of you reproduces these?".
//!
//! Never-wrong by construction:
//!   1. SHAPE FILTER — only ops whose entry-fn parameter kinds match the repo fn's
//!      declared parameter kinds are even considered (a string op can't reproduce an
//!      array task by coincidence of length-parity — it is excluded before executing).
//!   2. REPRODUCE-ALL — a candidate must reproduce EVERY mined example. A wrong op
//!      fails an example and is dropped.
//!   3. BEHAVIOURAL UNIQUENESS — if two or more ops reproduce all examples, they are
//!      accepted ONLY when they are pairwise INDISTINGUISHABLE on fresh differential
//!      probes (`constraint_oracle::programs_distinguishable`). That means they compute
//!      the SAME function (e.g. the `count_positives` / `count_greater_than_zero`
//!      twins) and returning either is never wrong. If ANY pair DISAGREES on a fresh
//!      input, the examples do not pin a unique behaviour → we DEFER (return None),
//!      never guess.
//!
//! (The literal "exactly one op" rule in the design note is strengthened here to
//! "exactly one BEHAVIOUR": the shipped library contains behaviourally-identical twin
//! ops, so a name-count would defer on tasks the examples pin perfectly. Grouping by
//! fresh-input behaviour is the correct never-wrong formulation and is what lets a bare
//! `count_positives` repair resolve. The caller's real `cargo test` is still the final
//! oracle on top of all of this.)

use crate::agent::repo::{RepairContext, RepairEdit, RepairPatch, RepoTaskSpec};
use crate::benchmark::Example;

/// Canonical type KIND of a declared parameter type, or `None` if unrecognised.
/// Maps both Mog op types (`[i64]`, `i64`, `string`, `bool`) and Rust repo types
/// (`Vec<i64>`, `&[i64]`, `i64`, `usize`, `String`, `&str`, `bool`) onto one of four
/// kinds so the two sides can be compared. Unknown → `None` (fail-safe: an
/// un-mappable type can't be shape-matched, so the op/target is excluded).
pub(crate) fn canonical_kind(ty: &str) -> Option<&'static str> {
    // Strip references / mut / whitespace so `&[i64]`, `& mut [i64]`, ` Vec<i64> ` all normalise.
    let mut t: String = ty.chars().filter(|c| !c.is_whitespace()).collect();
    while let Some(rest) = t.strip_prefix('&') {
        t = rest.to_string();
    }
    if let Some(rest) = t.strip_prefix("mut") {
        // only strip a leading `mut` token when it prefixes a type (e.g. `mut[i64]`)
        if rest.starts_with('[') || rest.starts_with("Vec<") {
            t = rest.to_string();
        }
    }
    // Arrays first (before scalar int check, since `[i64]` contains `i64`).
    if t.starts_with('[') || t.starts_with("Vec<") || t.starts_with("List<") || t.starts_with("Slice<")
    {
        return Some("array");
    }
    match t.as_str() {
        "i64" | "i32" | "i128" | "isize" | "u64" | "u32" | "usize" | "int" | "integer" => {
            Some("int")
        }
        "bool" | "boolean" => Some("bool"),
        "String" | "str" | "string" => Some("string"),
        _ => None,
    }
}

/// The entry (first-declared) function name of a Mog/Rust program, or "" if none.
fn entry_fn_name(src: &str) -> &str {
    src.split("fn ")
        .nth(1)
        .and_then(|s| s.split('(').next())
        .map(str::trim)
        .unwrap_or("")
}

/// Parse the RAW declared type strings of the first `fn NAME(...)`'s parameters in a
/// source string. `fn f(a: i64, b: [i64]) -> i64` → `["i64", "[i64]"]`. Returns an
/// empty vec for a nullary fn and `None` when no parseable header is found.
fn header_param_types(src: &str) -> Option<Vec<String>> {
    let open = src.find('(')?;
    // Match the balanced close of the parameter list.
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    let mut close = None;
    for i in open..bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let inner = src[open + 1..close].trim();
    if inner.is_empty() {
        return Some(vec![]);
    }
    Some(
        split_top_level(inner)
            .into_iter()
            .filter_map(|p| {
                // `name: Type` — take everything after the FIRST colon (types carry no
                // top-level colon in the shapes we support).
                p.split_once(':').map(|(_, ty)| ty.trim().to_string())
            })
            .collect(),
    )
}

/// Split `s` on commas that are NOT nested inside `<>`, `[]`, `()` — so generic and
/// array element types stay intact (`Vec<(i64, i64)>`, `[i64]`).
fn split_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut last = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '[' | '(' => depth += 1,
            '>' | ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[last..i]);
                last = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[last..]);
    parts
}

/// Do this op's entry-fn parameter KINDS equal `target_shape` (already canonicalised)?
fn op_shape_matches(mog: &str, target_shape: &[&'static str]) -> bool {
    let Some(raw) = header_param_types(mog) else {
        return false;
    };
    if raw.len() != target_shape.len() {
        return false;
    }
    raw.iter().zip(target_shape).all(|(ty, want)| canonical_kind(ty) == Some(*want))
}

/// The op's parameter kinds (canonical), or `None` if any is un-mappable.
fn op_kinds(mog: &str) -> Option<Vec<&'static str>> {
    header_param_types(mog)?.iter().map(|t| canonical_kind(t)).collect()
}

/// True when every kind in `shape` is distinct — the only case in which reordering
/// arguments purely BY TYPE is unambiguous (and the exact condition the reshape arg-swap
/// wrapper requires before it will emit a reordered call).
fn all_distinct_kinds(shape: &[&'static str]) -> bool {
    shape.iter().enumerate().all(|(i, k)| !shape[..i].contains(k))
}

/// True when `a` and `b` are the same MULTISET of kinds (a permutation of each other).
fn is_kind_permutation(a: &[&'static str], b: &[&'static str]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let (mut sa, mut sb) = (a.to_vec(), b.to_vec());
    sa.sort_unstable();
    sb.sort_unstable();
    sa == sb
}

/// If `mog`'s parameter kinds are a DISTINCT-kind permutation of `repo_param_types` (the
/// arg-order fallback case), return the mapping from each OP-parameter position to the repo
/// input index feeding it — so callers can present inputs in the op's native order. `None` when
/// the order already matches, types repeat, or shapes differ (i.e. no reordering is needed/safe).
fn op_native_permutation(mog: &str, repo_param_types: &[String]) -> Option<Vec<usize>> {
    let repo: Vec<&'static str> =
        repo_param_types.iter().map(|t| canonical_kind(t)).collect::<Option<_>>()?;
    let op = op_kinds(mog)?;
    if op.len() != repo.len()
        || op == repo
        || !all_distinct_kinds(&repo)
        || !is_kind_permutation(&op, &repo)
    {
        return None;
    }
    Some(op.iter().map(|k| repo.iter().position(|t| t == k).unwrap()).collect())
}

/// BEHAVIOUR-DRIVEN library resolution: return the SINGLE library behaviour that
/// reproduces every example, or `None` when zero reproduce or the examples fail to
/// pin a unique behaviour (see module docs). `target_param_types` are the repo fn's
/// declared parameter types (Rust or Mog spelling); they gate candidates by shape so a
/// coincidental cross-type length match can never be returned.
pub fn library_op_reproducing(
    examples: &[Example],
    target_param_types: &[String],
) -> Option<(String, String)> {
    if examples.is_empty() {
        return None;
    }
    // Canonicalise the target shape; if ANY declared type is un-mappable we cannot
    // shape-filter soundly, so defer rather than guess.
    let target_shape: Option<Vec<&'static str>> =
        target_param_types.iter().map(|t| canonical_kind(t)).collect();
    let target_shape = target_shape?;
    let arity = examples[0].inputs.len();
    if target_shape.len() != arity {
        return None;
    }

    // Every op that (a) matches the target shape and (b) reproduces ALL examples.
    let mut candidates: Vec<(String, String)> = Vec::new();
    for op in crate::op_library::OPS {
        if op.arity != arity {
            continue;
        }
        if !op_shape_matches(op.mog, &target_shape) {
            continue;
        }
        if crate::runtime::code_reproduces_examples(op.mog, examples) {
            candidates.push((op.name.to_string(), op.mog.to_string()));
        }
    }
    // Learned/flywheel ops (bundled distilled + any runtime-grown), gated identically.
    for op in crate::op_library::learned_ops_snapshot() {
        if op.arity != arity {
            continue;
        }
        if !op_shape_matches(&op.mog, &target_shape) {
            continue;
        }
        if crate::runtime::code_reproduces_examples(&op.mog, examples) {
            candidates.push((op.name.clone(), op.mog.clone()));
        }
    }

    // ARG-ORDER FALLBACK: no op matched the target's param order exactly. Try ops whose param
    // KINDS are a permutation of the target's (a scalar-first repo fn `k_largest(k, xs)`
    // resolving to the array-first library op `k_largest(arr, k)`). Each example's inputs are
    // reordered into the op's param order before verifying, and reshape's arg-swap wrapper then
    // reorders the CALL to preserve the repo signature. Gated to ALL-DISTINCT target kinds
    // (unambiguous type reorder) and EXACTLY ONE reproducing op — 0 or ambiguous defers. This
    // only runs when the exact-order pass found nothing, so existing resolutions are unchanged.
    if candidates.is_empty() && all_distinct_kinds(&target_shape) {
        let mut hits: Vec<(String, String)> = Vec::new();
        let mut consider = |name: &str, mog: &str| {
            let Some(op_shape) = op_kinds(mog) else { return };
            if op_shape.len() != arity
                || op_shape.as_slice() == target_shape.as_slice()
                || !is_kind_permutation(&op_shape, &target_shape)
            {
                return;
            }
            // op param position p is fed by the (unique, since distinct) target input whose
            // kind equals op_shape[p].
            let perm: Vec<usize> = op_shape
                .iter()
                .map(|k| target_shape.iter().position(|t| t == k).unwrap())
                .collect();
            let reordered: Vec<Example> = examples
                .iter()
                .map(|e| Example {
                    inputs: perm.iter().map(|&i| e.inputs[i].clone()).collect(),
                    expected: e.expected.clone(),
                })
                .collect();
            if crate::runtime::code_reproduces_examples(mog, &reordered) {
                hits.push((name.to_string(), mog.to_string()));
            }
        };
        for op in crate::op_library::OPS {
            if op.arity == arity {
                consider(op.name, op.mog);
            }
        }
        for op in crate::op_library::learned_ops_snapshot() {
            if op.arity == arity {
                consider(&op.name, &op.mog);
            }
        }
        // Exactly one reproducing op -> confident; 0 or ambiguous -> defer (never guess).
        return (hits.len() == 1).then(|| hits.into_iter().next().unwrap());
    }

    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates.into_iter().next().unwrap());
    }
    // Multiple reproduce: accept ONLY if they are pairwise behaviourally equivalent
    // (no observed difference on fresh differential probes). Any distinguishable pair
    // means the examples under-determine the behaviour → defer.
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let (_, mi) = &candidates[i];
            let (_, mj) = &candidates[j];
            if crate::constraint_oracle::programs_distinguishable(
                mi,
                entry_fn_name(mi),
                mj,
                entry_fn_name(mj),
            ) {
                return None;
            }
        }
    }
    Some(candidates.into_iter().next().unwrap())
}

/// First `fn`/`pub fn` name declared before the test module — the repo function a
/// bare-prompt repair targets when no intent name is available. Mirrors the resolver
/// `synthesis_proposer` uses, kept local so this module owns its logic.
fn first_defined_fn(source: &str) -> Option<String> {
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#[cfg(test)]") {
            break;
        }
        let rest = trimmed.strip_prefix("pub fn ").or_else(|| trimmed.strip_prefix("fn "));
        if let Some(rest) = rest {
            let name: String =
                rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    None
}

/// Repo proposer: solve a bare "fix the failing test" for a function that IS a library
/// op, purely by BEHAVIOUR. Mines the failing asserts across the repair context, requires
/// ≥3 examples, resolves the unique reproducing library behaviour, transpiles + reshapes
/// it onto the repo signature, and strict-verifies before emitting a patch. Returns `None`
/// (a clean decline) whenever anything is ambiguous or unsupported — never a bad patch.
pub fn try_library_behavior_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    description: &str,
) -> Option<RepairPatch> {
    // Resolve the target file. A bare prompt yields no usable intent name, so
    // `pick_target_path` falls through to the writable Rust file, and we take the
    // fn it defines as the repair target.
    let intent = crate::agent::coding_intent::CodingIntent::from_nl_lenient(description).ok();
    let target =
        crate::agent::synthesis_proposer::pick_target_path(task, context, intent.as_ref()).ok()?;
    let old_text = crate::agent::synthesis_proposer::read_relative_file(context, &target).ok()?;
    let repo_fn = first_defined_fn(&old_text)?;

    // Mine the failing test's `assert_eq!` I/O for `repo_fn` across every context file.
    let mut rows: Vec<(Vec<crate::benchmark::Value>, crate::benchmark::Value)> = Vec::new();
    for f in &context.files {
        if let Some(t) = f.text.as_deref() {
            rows.extend(crate::agent::synthesis_proposer::mine_asserts(t, &repo_fn));
        }
    }
    rows.sort();
    rows.dedup();
    // Behaviour resolution needs a strong pin; require at least three examples.
    if rows.len() < 3 {
        return None;
    }
    let exs: Vec<Example> = rows
        .iter()
        .map(|(ins, out)| Example { inputs: ins.clone(), expected: out.clone() })
        .collect();

    // The repo fn's declared parameter types drive the shape filter.
    let target_param_types = header_param_types(&old_text_slice_for_fn(&old_text, &repo_fn)?)?;
    if target_param_types.is_empty() {
        return None;
    }

    let (op_name, mog) = library_op_reproducing(&exs, &target_param_types)?;

    // When the op was resolved via the arg-order fallback, its parameter order is a permutation
    // of the repo's, so strict-verify must see the inputs in the OP's NATIVE order (identity
    // otherwise). reshape's arg-swap wrapper restores the repo call order afterwards.
    let verify_exs: Vec<Example> = match op_native_permutation(&mog, &target_param_types) {
        Some(perm) => exs
            .iter()
            .map(|e| Example {
                inputs: perm.iter().map(|&i| e.inputs[i].clone()).collect(),
                expected: e.expected.clone(),
            })
            .collect(),
        None => exs.clone(),
    };

    // Strict-verify the resolved behaviour against the mined examples (independent of the
    // reproduce-all selection gate), then transpile Mog → Rust and reshape onto the repo
    // signature. Any failure declines gracefully.
    let sig: &'static str = Box::leak(
        crate::linguigenesis_bridge::infer_signature(&repo_fn, &verify_exs).into_boxed_str(),
    );
    let problem = crate::benchmark::Problem {
        name: repo_fn.clone(),
        category: "repo-library-behavior",
        description: "",
        signature: sig,
        examples: verify_exs.clone(),
        ..Default::default()
    };
    crate::runtime::verify_problem_code_strict(&problem, &mog).ok()?;

    let synthesized = crate::agent::synthesis_proposer::rust_code_for_repo_synthesis(&mog);
    // Plain-Rust guard: never emit residual Result/Mog idioms into a repo file.
    if synthesized.contains("ok(")
        || synthesized.contains("err(")
        || synthesized.contains(":=")
        || synthesized.contains("Result<")
    {
        return None;
    }
    let new_text =
        crate::agent::synthesis_proposer::reshape_to_repo_signature(&old_text, &repo_fn, &synthesized)?;
    if new_text == old_text {
        return None;
    }

    Some(
        RepairPatch::new()
            .with_edit(RepairEdit::new(
                target,
                old_text,
                new_text,
                "library-behavior proposer (mined asserts resolve a unique verified library op; no LLM)",
            ))
            .with_metadata("proposer", "nl_library_behavior")
            .with_metadata("synthesis_method", format!("library-behavior:{op_name}")),
    )
}

/// The substring of `old_text` starting at `fn repo_fn(` (or `pub fn repo_fn(`) so
/// `header_param_types` parses THIS fn's params, not the file's first fn.
fn old_text_slice_for_fn<'a>(old_text: &'a str, repo_fn: &str) -> Option<&'a str> {
    let needle = format!("fn {repo_fn}(");
    let pos = old_text.find(&needle)?;
    Some(&old_text[pos..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Value;

    fn iv(xs: &[i64]) -> Value {
        Value::int_array(xs)
    }
    fn ex(inputs: Vec<Value>, expected: Value) -> Example {
        Example { inputs, expected }
    }

    /// Examples that pin the "count elements > 0" behaviour: exactly the twin ops
    /// `count_positives` / `count_greater_than_zero` reproduce them (same function),
    /// so a UNIQUE behaviour is returned even though two op NAMES share it.
    #[test]
    fn unique_reproducer_is_returned() {
        let exs = vec![
            ex(vec![iv(&[5, -2, 3, -4, 5])], Value::Int(3)),
            ex(vec![iv(&[-1, -2, -3])], Value::Int(0)),
            ex(vec![iv(&[1, 2, 3, 4])], Value::Int(4)),
            ex(vec![iv(&[0, -1, 7])], Value::Int(1)),
        ];
        let got = library_op_reproducing(&exs, &["Vec<i64>".to_string()]);
        let (name, mog) = got.expect("a unique positive-count behaviour must resolve");
        assert!(
            crate::runtime::code_reproduces_examples(&mog, &exs),
            "returned op {name} must reproduce every example"
        );
        // The returned behaviour is a positive-count op, never a wrong one.
        assert!(
            name == "count_positives" || name == "count_greater_than_zero",
            "unexpected op resolved: {name}"
        );
    }

    /// Two behaviourally-DISTINCT ops (`array_sum` vs `list_max`) both reproduce
    /// single-element-array examples; they disagree on multi-element inputs, so the
    /// examples under-determine the behaviour → defer (None), never guess.
    #[test]
    fn ambiguous_distinct_behaviours_return_none() {
        // sum([7])==max([7])==7, sum([3])==max([3])==3, sum([9])==max([9])==9.
        let exs = vec![
            ex(vec![iv(&[7])], Value::Int(7)),
            ex(vec![iv(&[3])], Value::Int(3)),
            ex(vec![iv(&[9])], Value::Int(9)),
        ];
        // Sanity: both candidate ops really do reproduce these examples.
        let sum = crate::op_library::OPS.iter().find(|o| o.name == "array_sum").unwrap();
        let max = crate::op_library::OPS.iter().find(|o| o.name == "list_max").unwrap();
        assert!(crate::runtime::code_reproduces_examples(sum.mog, &exs));
        assert!(crate::runtime::code_reproduces_examples(max.mog, &exs));
        assert!(
            library_op_reproducing(&exs, &["Vec<i64>".to_string()]).is_none(),
            "distinct-behaviour reproducers must defer, not guess"
        );
    }

    /// A wrong op (`count_evens`) fails these examples, so the reproduce-all gate
    /// excludes it: only the correct positive-count behaviour can be returned.
    #[test]
    fn wrong_single_op_is_not_returned() {
        let exs = vec![
            ex(vec![iv(&[2, 1, 4, -3, 5])], Value::Int(3)), // positives: 2,1,4,5 = 4? recompute below
            ex(vec![iv(&[-2, -4])], Value::Int(0)),
            ex(vec![iv(&[1, 3, 5, 7])], Value::Int(4)),
            ex(vec![iv(&[0, 2, -1])], Value::Int(1)),
        ];
        // Fix the first example to the true positive count (2,1,4,5 -> 4).
        let exs = vec![
            ex(vec![iv(&[2, 1, 4, -3, 5])], Value::Int(4)),
            ex(vec![iv(&[-2, -4])], Value::Int(0)),
            ex(vec![iv(&[1, 3, 5, 7])], Value::Int(4)),
            ex(vec![iv(&[0, 2, -1])], Value::Int(1)),
        ];
        // count_evens is DISTINCT and does NOT reproduce (e.g. [1,3,5,7] -> 0, not 4).
        let evens = crate::op_library::OPS.iter().find(|o| o.name == "count_evens").unwrap();
        assert!(
            !crate::runtime::code_reproduces_examples(evens.mog, &exs),
            "count_evens must fail these examples"
        );
        let got = library_op_reproducing(&exs, &["Vec<i64>".to_string()]);
        if let Some((name, mog)) = got {
            assert_ne!(name, "count_evens", "a wrong op must never be returned");
            assert!(crate::runtime::code_reproduces_examples(&mog, &exs));
        }
    }

    /// The shape filter excludes type-mismatched ops: array-input examples with a
    /// SCALAR (`i64`) declared target type match no op (all array ops are excluded by
    /// shape), so nothing is returned.
    #[test]
    fn shape_filter_excludes_type_mismatched_ops() {
        let exs = vec![
            ex(vec![iv(&[5, -2, 3, -4, 5])], Value::Int(3)),
            ex(vec![iv(&[-1, -2, -3])], Value::Int(0)),
            ex(vec![iv(&[1, 2, 3, 4])], Value::Int(4)),
        ];
        // With the correct array shape this DOES resolve...
        assert!(library_op_reproducing(&exs, &["Vec<i64>".to_string()]).is_some());
        // ...but a scalar-typed target excludes every array op by shape → None.
        assert!(
            library_op_reproducing(&exs, &["i64".to_string()]).is_none(),
            "scalar target must exclude array ops"
        );
    }

    #[test]
    fn canonical_kind_maps_both_spellings() {
        assert_eq!(canonical_kind("[i64]"), Some("array"));
        assert_eq!(canonical_kind("Vec<i64>"), Some("array"));
        assert_eq!(canonical_kind("&[i64]"), Some("array"));
        assert_eq!(canonical_kind("i64"), Some("int"));
        assert_eq!(canonical_kind("usize"), Some("int"));
        assert_eq!(canonical_kind("string"), Some("string"));
        assert_eq!(canonical_kind("String"), Some("string"));
        assert_eq!(canonical_kind("&str"), Some("string"));
        assert_eq!(canonical_kind("bool"), Some("bool"));
        assert_eq!(canonical_kind("Foo"), None);
    }

    #[test]
    fn header_param_types_parses_signatures() {
        assert_eq!(
            header_param_types("fn f(a: i64, b: [i64]) -> i64 {}"),
            Some(vec!["i64".to_string(), "[i64]".to_string()])
        );
        assert_eq!(
            header_param_types("pub fn count_positives(xs: Vec<i64>) -> i64 { 0 }"),
            Some(vec!["Vec<i64>".to_string()])
        );
        assert_eq!(header_param_types("fn nullary() -> i64 { 0 }"), Some(vec![]));
    }

    /// ARG-ORDER FALLBACK: a SCALAR-FIRST target signature (k: i64, xs: [i64]) -> [i64] resolves
    /// to the ARRAY-FIRST library op `k_largest(arr, k)` — examples given in target order are
    /// reordered to the op's order for verification. Activates the reshape arg-swap wrapper.
    #[test]
    fn arg_order_fallback_resolves_a_reordered_op() {
        let exs = vec![
            ex(vec![Value::Int(2), iv(&[5, 1, 9, 3])], iv(&[9, 5])),
            ex(vec![Value::Int(1), iv(&[7, 2, 5])], iv(&[7])),
            ex(vec![Value::Int(3), iv(&[4, 4, 2, 8, 1])], iv(&[8, 4, 4])),
        ];
        let got = library_op_reproducing(&exs, &["i64".to_string(), "[i64]".to_string()]);
        let (name, _) = got.expect("reordered k_largest must resolve via the multiset fallback");
        assert_eq!(name, "k_largest");
    }

    /// The fallback still DEFERS (never guesses) when no permutation of any op reproduces the
    /// examples — never-wrong holds for the reordered path too.
    #[test]
    fn arg_order_fallback_defers_when_no_op_reproduces() {
        let exs = vec![
            ex(vec![Value::Int(1), iv(&[1, 2, 3])], Value::Int(99)),
            ex(vec![Value::Int(2), iv(&[4, 5])], Value::Int(88)),
            ex(vec![Value::Int(0), iv(&[7])], Value::Int(77)),
        ];
        assert!(library_op_reproducing(&exs, &["i64".to_string(), "[i64]".to_string()]).is_none());
    }

    /// END TO END, ARG-ORDER ACTIVATED: a SCALAR-FIRST repo fn `k_largest(k: i64, xs: Vec<i64>)`
    /// — an argument-order MISMATCH against the array-first library op that could not be repaired
    /// before — is now solved: the behaviour probe resolves the op via the multiset fallback and
    /// reshape's arg-swap wrapper reorders the call to preserve the repo signature.
    #[test]
    fn scalar_first_k_largest_repairs_via_arg_swap() {
        use std::fs;
        let root = std::env::temp_dir().join(format!("nsynth_argswap_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"as\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub fn k_largest(k: i64, xs: Vec<i64>) -> Vec<i64> {\n    Vec::new()\n}\n\n#[cfg(test)]\nmod tests {\n    use super::k_largest;\n    #[test]\n    fn t() {\n        assert_eq!(k_largest(2, vec![5, 1, 9, 3]), vec![9, 5]);\n        assert_eq!(k_largest(1, vec![7, 2, 5]), vec![7]);\n        assert_eq!(k_largest(3, vec![4, 4, 2, 8, 1]), vec![8, 4, 4]);\n    }\n}\n",
        )
        .expect("lib.rs");
        let task = crate::agent::repo::RepoTaskSpec {
            id: "as".into(),
            repo: root.to_string_lossy().to_string(),
            kind: crate::agent::repo::RepoTaskKind::BugFix,
            issue: "the k largest elements".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: crate::agent::repo::HardnessProfile::for_expected_tier(
                crate::agent::repo::HardnessTier::SingleFileBug,
            ),
            signals: Vec::new(),
        };
        let context = crate::agent::repo::RepairContext::build(
            &root,
            &crate::agent::repo::GuardrailPolicy::default(),
        )
        .expect("ctx");
        let patch = try_library_behavior_patch(&task, &context, "the k largest elements")
            .expect("scalar-first k_largest must resolve via the arg-order fallback + swap");
        assert!(
            patch.edits.iter().any(|e| e.path == "src/lib.rs"
                && e.new_text.contains("reordered_k_largest(xs, k)")
                && e.new_text.contains(".sort()")),
            "repo fn must call the reordered impl with swapped args, and the impl must carry the \
             real k-largest logic (sort + take-k)"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// END TO END on a BARE prompt: a repo `count_positives` stub with failing asserts and the
    /// issue EXACTLY "fix the failing test" (NO descriptive prose to name an op) is repaired by
    /// BEHAVIOR — the probe finds the unique library op reproducing the mined asserts. This is the
    /// path that sidesteps the lg-core NL comprehension collapse for name-less repo tasks.
    #[test]
    fn bare_prompt_repairs_count_positives_by_behavior() {
        use std::fs;
        let root = std::env::temp_dir().join(format!("nsynth_libprobe_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"lp\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub fn count_positives(xs: Vec<i64>) -> i64 {\n    0\n}\n\n#[cfg(test)]\nmod tests {\n    use super::count_positives;\n    #[test]\n    fn t() {\n        assert_eq!(count_positives(vec![5, -2, 3, -4, 5]), 3);\n        assert_eq!(count_positives(vec![-1, -2, -3]), 0);\n        assert_eq!(count_positives(vec![1, 2, 3, 4]), 4);\n    }\n}\n",
        )
        .expect("lib.rs");
        let task = crate::agent::repo::RepoTaskSpec {
            id: "lp".into(),
            repo: root.to_string_lossy().to_string(),
            kind: crate::agent::repo::RepoTaskKind::BugFix,
            issue: "fix the failing test".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: crate::agent::repo::HardnessProfile::for_expected_tier(
                crate::agent::repo::HardnessTier::SingleFileBug,
            ),
            signals: Vec::new(),
        };
        let context = crate::agent::repo::RepairContext::build(
            &root,
            &crate::agent::repo::GuardrailPolicy::default(),
        )
        .expect("ctx");
        let patch = try_library_behavior_patch(&task, &context, "fix the failing test")
            .expect("bare-prompt count_positives must resolve by behavior");
        assert!(
            patch.edits.iter().any(|e| e.path == "src/lib.rs" && !e.new_text.contains("    0\n}")),
            "stub body must be replaced by the library op's real count"
        );
        let _ = fs::remove_dir_all(&root);
    }
}
