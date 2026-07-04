//! Typed pipeline search over the verified op library — the composition tier.
//!
//! [`crate::op_library::try_library`] matches ONE known algorithm against the whole
//! example set. But most real tasks are short COMPOSITIONS of known algorithms
//! ("sum of the digits of n!", "is the reversed number prime", "sum of squares of
//! the distinct values") that no single op reproduces and bounded expression search
//! cannot induce. This module searches over CHAINS of verified ops instead:
//!
//! * **Typed**: each stage op carries an input/output type parsed from its own Mog
//!   signature (never hand-annotated), and only signature-compatible ops chain.
//! * **Value-level**: candidate chains are evaluated by propagating the example
//!   VALUES stage by stage; a full program is only emitted for a chain whose final
//!   values match every expected output.
//! * **Observational-equivalence pruned**: a stage whose output value-vector was
//!   already seen (e.g. an identity-like op on these examples) is dropped, so
//!   `reverse_number` on single-digit inputs cannot launder a single-op solve into
//!   a fake "pipeline".
//! * **Depth ≥ 2 only**: single-op solves are `try_library`'s job (and the general
//!   engine's); this tier exists purely for composition, so it never competes for
//!   single-op attribution.
//!
//! Like the library tier, a returned result reproduces EVERY example (whole-program
//! re-check via [`code_reproduces_examples`], not just the value propagation), and a
//! coincidental seed-only match is still caught by the caller's holdout
//! re-verification.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use crate::benchmark::{Problem, Value as BValue};
use crate::op_library::LibOp;
use crate::runtime::{
    benchmark_value_from_runtime, code_reproduces_examples, execute_function, execute_parsed,
    parse_program, Program,
};
use crate::solver::SolveResult;

/// The chain search TERMINATES BY CONSTRUCTION — breadth-first over chains of
/// length ≤ [`MAX_DEPTH`] drawn from a finite, focused stage set, with
/// observational-equivalence dedup (each distinct value-state is expanded once).
/// So there is no attempt cap: exhausting the (small) reachable space is correct,
/// not a runaway. The only unbounded quantity is per-application interpreter TIME
/// (a stage op could be a slow interpreted loop), which the interpreter's own
/// global step budget already bounds; [`SEARCH_BUDGET`] below is a pure safety
/// net over the whole search, not the primary bound.
///
/// Pure safety net. With the curated chain-stage allowlist ([`CHAIN_STAGE_LIBRARY_OPS`])
/// the search is small and O(1) in library size, so a full depth-3 sweep runs in
/// ~0.3s regardless of how many terminal library ops exist. This wall only guards
/// a pathological slow op-application; it is far above the real search cost and
/// does not grow with the library.
const SEARCH_BUDGET: Duration = Duration::from_millis(2000);

/// Magnitude gate for FRONTIER states (checked after the accept test, so a chain
/// whose final values legitimately exceed this can still be returned). Interpreter
/// `while` loops are NOT iteration-capped (only `For*` are), so a stage output
/// like `decimal_to_binary(1234) ≈ 1e10` fed into an O(n) divisor loop runs for
/// MINUTES inside one application — the wall deadline cannot preempt it. Bounding
/// intermediate magnitudes keeps every allowed stage application at ≤~10⁴
/// interpreted steps, so no single application can outrun [`SEARCH_BUDGET`].
const MAX_INT_MAGNITUDE: i64 = 10_000;
const MAX_SEQ_LEN: usize = 512;

/// Library ops (from [`crate::op_library::OPS`]) that are COMPOSITIONAL PRIMITIVES
/// — meaningful as an INTERMEDIATE stage in a chain (`sum_of_digits ∘ factorial`,
/// `is_prime ∘ reverse_number`, `integer_sqrt ∘ …`). Only these join the chain
/// grammar; [`COMPOSE_OPS`] (reshapes/aggregates/2-arg takes) are always in.
///
/// This is an ALLOWLIST, not an exclude-list, and it is the key to SUSTAINED
/// capability: the op library grows every batch, but most new entries are
/// whole-answer TERMINAL solves (figurate numbers, geometry, divisor sums,
/// specific sequences) that a chain would never route THROUGH. Those stay fully
/// available as single-op `try_library` solves — the tasks they target are NOT
/// lost — they simply don't enlarge the chain search, which would otherwise grow
/// unboundedly and push genuine deep chains past the wall. Reach for real tasks
/// is preserved (library solves the terminals; chains use these primitives);
/// only vanishingly-rare terminal-op compositions are forgone. Add a name here
/// only when a real task needs it as an intermediate.
const CHAIN_STAGE_LIBRARY_OPS: &[&str] = &[
    // digit / number transforms
    "sum_of_digits",
    "count_digits",
    "digit_product",
    "largest_digit",
    "reverse_number",
    "last_digit",
    "is_even",
    // predicates useful as filters/intermediates
    "is_prime",
    // bit transforms
    "unset_bits",
    "decimal_to_binary",
    "binary_to_decimal",
    "decimal_to_octal",
    "octal_to_decimal",
    "highest_power_of_2",
    "lowest_set_bit_pos",
    "next_power_of_2",
    // roots / small closed transforms that genuinely compose
    "integer_sqrt",
    "factorial",
    // array reductions/aggregations the base engine misses (compose after reshapes)
    "sum_of_squares",
    "array_range",
    "count_evens",
    "count_odds",
    "count_positives",
    "count_negatives",
];

/// Maximum chain length. Depth 3 already covers reshape→transform→reduce
/// (e.g. digits → unique → sum); deeper chains explode the state space for
/// little measured benefit.
const MAX_DEPTH: usize = 3;

/// The value types stage ops range over. Parsed from Mog signatures; ops using
/// types outside this set simply don't participate in chaining.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ty {
    Int,
    IntArr,
    Str,
}

impl Ty {
    fn from_mog(name: &str) -> Option<Ty> {
        match name {
            "i64" => Some(Ty::Int),
            "[i64]" => Some(Ty::IntArr),
            "string" => Some(Ty::Str),
            _ => None,
        }
    }

    fn mog_name(self) -> &'static str {
        match self {
            Ty::Int => "i64",
            Ty::IntArr => "[i64]",
            Ty::Str => "string",
        }
    }
}

/// One chainable stage with a parsed type signature. `scalar` stages are binary
/// ops `(flow, i64) -> output` that consume the problem's second (scalar) input;
/// a chain may use at most one, and an arity-2 problem's chain must use exactly
/// one (a fit that ignores an argument is overfit laundering, not a solve).
struct StageOp {
    name: &'static str,
    mog: &'static str,
    input: Ty,
    output: Ty,
    scalar: bool,
    /// The op's Mog source parsed ONCE (the search runs each stage over thousands
    /// of intermediate values; re-parsing per call dominated the depth-3 cost).
    prog: Program,
}

/// Compose-only ops: reshape/reorder stages that make chains expressive
/// (array→array transforms, int→array explode). They are deliberately NOT in
/// [`crate::op_library::OPS`] — as single ops they would duplicate what the base
/// engine (`array_transform`, enumerative fold) already solves and steal its
/// method attribution; they exist only as chain stages, and [`try_pipeline`]
/// never returns a depth-1 result.
const COMPOSE_OPS: &[LibOp] = &[
    LibOp { name: "sorted_values", arity: 1, mog:
"fn sorted_values(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    for e in arr {\n        out.push(e);\n    }\n    out.sort();\n    return out;\n}\n" },
    LibOp { name: "reversed_values", arity: 1, mog:
"fn reversed_values(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    i: i64 = arr.len - 1;\n    while i >= 0 {\n        out.push(arr[i]);\n        i = i - 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "unique_values", arity: 1, mog:
"fn unique_values(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    for e in arr {\n        found: i64 = 0;\n        for u in out {\n            if u == e {\n                found = 1;\n            }\n        }\n        if found == 0 {\n            out.push(e);\n        }\n    }\n    return out;\n}\n" },
    LibOp { name: "abs_values", arity: 1, mog:
"fn abs_values(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    for e in arr {\n        x: i64 = e;\n        if x < 0 {\n            x = 0 - x;\n        }\n        out.push(x);\n    }\n    return out;\n}\n" },
    LibOp { name: "digits_of", arity: 1, mog:
"fn digits_of(n: i64) -> [i64] {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    rev: [i64] = [];\n    if x == 0 {\n        rev.push(0);\n    }\n    while x > 0 {\n        rev.push(x % 10);\n        x = x / 10;\n    }\n    out: [i64] = [];\n    i: i64 = rev.len - 1;\n    while i >= 0 {\n        out.push(rev[i]);\n        i = i - 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "max_value", arity: 1, mog:
"fn max_value(arr: [i64]) -> i64 {\n    m: i64 = arr[0];\n    for e in arr {\n        if e > m {\n            m = e;\n        }\n    }\n    return m;\n}\n" },
    LibOp { name: "min_value", arity: 1, mog:
"fn min_value(arr: [i64]) -> i64 {\n    m: i64 = arr[0];\n    for e in arr {\n        if e < m {\n            m = e;\n        }\n    }\n    return m;\n}\n" },
    LibOp { name: "sum_values", arity: 1, mog:
"fn sum_values(arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for e in arr {\n        s = s + e;\n    }\n    return s;\n}\n" },
    LibOp { name: "count_of", arity: 1, mog:
"fn count_of(arr: [i64]) -> i64 {\n    return arr.len;\n}\n" },
    // ── scalar-consuming binary stages (flow, k) — enable the n-largest /
    //    take / index class ("first k of the sorted values" = take_first ∘ sorted).
    LibOp { name: "take_first", arity: 2, mog:
"fn take_first(arr: [i64], k: i64) -> [i64] {\n    out: [i64] = [];\n    i: i64 = 0;\n    while i < k {\n        if i < arr.len {\n            out.push(arr[i]);\n        }\n        i = i + 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "take_last", arity: 2, mog:
"fn take_last(arr: [i64], k: i64) -> [i64] {\n    out: [i64] = [];\n    j: i64 = arr.len - k;\n    if j < 0 {\n        j = 0;\n    }\n    while j < arr.len {\n        out.push(arr[j]);\n        j = j + 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "drop_first", arity: 2, mog:
"fn drop_first(arr: [i64], k: i64) -> [i64] {\n    out: [i64] = [];\n    i: i64 = k;\n    while i < arr.len {\n        out.push(arr[i]);\n        i = i + 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "drop_last", arity: 2, mog:
"fn drop_last(arr: [i64], k: i64) -> [i64] {\n    out: [i64] = [];\n    limit: i64 = arr.len - k;\n    i: i64 = 0;\n    while i < limit {\n        out.push(arr[i]);\n        i = i + 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "element_at", arity: 2, mog:
"fn element_at(arr: [i64], i: i64) -> i64 {\n    return arr[i];\n}\n" },
    LibOp { name: "remove_value", arity: 2, mog:
"fn remove_value(arr: [i64], x: i64) -> [i64] {\n    out: [i64] = [];\n    for e in arr {\n        if e != x {\n            out.push(e);\n        }\n    }\n    return out;\n}\n" },
];

/// Parse `(input types, output type)` from an op's own Mog signature line
/// (`fn name(a: i64) -> [i64] {`). Ops whose signature uses a type outside [`Ty`]
/// return `None` and are skipped — the op set stays self-describing, no
/// hand-maintained annotation table.
fn parse_sig(mog: &str) -> Option<(Vec<Ty>, Ty)> {
    let first = mog.lines().next()?;
    let params = first.split('(').nth(1)?.split(')').next()?;
    let mut ins = Vec::new();
    for p in params.split(',') {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        ins.push(Ty::from_mog(p.split(':').nth(1)?.trim())?);
    }
    let out = first.split("->").nth(1)?.trim().trim_end_matches('{').trim();
    Some((ins, Ty::from_mog(out)?))
}

/// All unary chainable stages: the compose-only reshapers plus the library's
/// arity-1 ops, each with its parsed signature. Compose ops come FIRST because
/// they are uniformly cheap (single pass, no nested trial division), so BFS
/// reaches reshape-based chains before burning budget on expensive number-theory
/// stages like `count_primes_below`.
fn stage_ops() -> Vec<StageOp> {
    COMPOSE_OPS
        .iter()
        .chain(crate::op_library::OPS.iter())
        // COMPOSE_OPS are always chain stages; library OPS join only if they are
        // curated compositional primitives (see the allowlist rationale).
        .filter(|op| {
            COMPOSE_OPS.iter().any(|c| c.name == op.name)
                || CHAIN_STAGE_LIBRARY_OPS.contains(&op.name)
        })
        .filter_map(|op| {
            let (ins, out) = parse_sig(op.mog)?;
            // Every op in OPS/COMPOSE_OPS is valid Mog (guarded by the library's
            // own probe tests), so parse cannot fail here; skip any that somehow
            // don't rather than panic.
            let prog = parse_program(op.mog).ok()?;
            match (op.arity, ins.as_slice()) {
                (1, [input]) => Some(StageOp {
                    name: op.name,
                    mog: op.mog,
                    input: *input,
                    output: out,
                    scalar: false,
                    prog,
                }),
                // Binary stage: second param must be the pass-through scalar.
                (2, [input, Ty::Int]) => Some(StageOp {
                    name: op.name,
                    mog: op.mog,
                    input: *input,
                    output: out,
                    scalar: true,
                    prog,
                }),
                _ => None,
            }
        })
        .collect()
}

/// May this value participate as an INTERMEDIATE chain state? (See
/// [`MAX_INT_MAGNITUDE`] — this is a cost bound, not a semantic one.)
fn small_enough(v: &BValue) -> bool {
    match v {
        BValue::Int(i) => i.abs() <= MAX_INT_MAGNITUDE,
        BValue::Str(s) => s.len() <= MAX_SEQ_LEN,
        BValue::Array(elems) => elems.len() <= MAX_SEQ_LEN && elems.iter().all(small_enough),
        _ => false,
    }
}

/// The [`Ty`] of a wire value, or `None` for values outside the chainable set.
fn ty_of_value(v: &BValue) -> Option<Ty> {
    match v {
        BValue::Int(_) => Some(Ty::Int),
        BValue::Str(_) => Some(Ty::Str),
        BValue::Array(elems) => {
            if elems.iter().all(|e| matches!(e, BValue::Int(_))) {
                Some(Ty::IntArr)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Common [`Ty`] across all values, or `None` if mixed/unsupported.
fn ty_of_values<'a>(mut vals: impl Iterator<Item = &'a BValue>) -> Option<Ty> {
    let first = ty_of_value(vals.next()?)?;
    for v in vals {
        if ty_of_value(v)? != first {
            return None;
        }
    }
    Some(first)
}

/// Run one stage op on every current example value. Any per-value error (type
/// mismatch, fuel exhaustion, overflow) disqualifies the op for this state.
/// Results are memoized per `(op, value)` — different chain prefixes frequently
/// reconverge on the same intermediate values, and re-interpreting an expensive
/// op (trial division, digit loops) for each of them is the dominant search cost.
fn apply_op(
    op_idx: usize,
    op: &StageOp,
    vals: &[BValue],
    scalars: Option<&[BValue]>,
    memo: &mut std::collections::HashMap<(usize, String), Option<BValue>>,
) -> Option<Vec<BValue>> {
    vals.iter()
        .enumerate()
        .map(|(ex, v)| {
            let args: Vec<BValue> = match scalars {
                Some(s) => vec![v.clone(), s[ex].clone()],
                None => vec![v.clone()],
            };
            let key = (op_idx, format!("{args:?}"));
            if let Some(cached) = memo.get(&key) {
                return cached.clone();
            }
            let result = execute_parsed(&op.prog, op.name, &args, op.name)
                .ok()
                .and_then(|out| benchmark_value_from_runtime(&out).ok());
            memo.insert(key, result.clone());
            result
        })
        .collect()
}

/// Deterministic fingerprint of a value-vector for observational-equivalence
/// pruning (Debug form is distinct per variant and content).
fn fingerprint(vals: &[BValue]) -> String {
    format!("{vals:?}")
}

/// Build the full Mog program for a chain (wrapper first — the entry fn is the
/// first `fn` in the file — then each distinct op def once) and re-check it against
/// every example as a whole program. `scalar_arity` = the problem passes a second
/// scalar arg, threaded into the chain's single binary stage.
fn emit_and_check(chain: &[&StageOp], problem: &Problem, scalar_arity: bool) -> Option<SolveResult> {
    let in_ty = chain.first()?.input;
    let out_ty = chain.last()?.output;
    let mut call = "x".to_string();
    for op in chain {
        call = if op.scalar {
            format!("{}({}, k)", op.name, call)
        } else {
            format!("{}({})", op.name, call)
        };
    }
    let params = if scalar_arity {
        format!("x: {}, k: i64", in_ty.mog_name())
    } else {
        format!("x: {}", in_ty.mog_name())
    };
    let mut code = format!(
        "fn pipeline({}) -> {} {{\n    return {};\n}}\n\n",
        params,
        out_ty.mog_name(),
        call
    );
    let mut emitted: Vec<&str> = Vec::new();
    for op in chain {
        if emitted.contains(&op.name) {
            continue;
        }
        emitted.push(op.name);
        code.push_str(op.mog);
        code.push('\n');
    }
    if !code_reproduces_examples(&code, &problem.examples) {
        return None;
    }
    // Anti-laundering: a scalar-arity chain must SEMANTICALLY depend on the
    // scalar, not merely consume it. `sum_values(rotate_left(arr, k))` consumes
    // k but sum is rotation-invariant, so k is vacuous and the "solve" is a
    // coincidental scalar-ignoring fit. Reject unless perturbing k changes the
    // output on at least one example.
    if scalar_arity && !output_depends_on_scalar(&code, problem) {
        return None;
    }
    let names: Vec<&str> = chain.iter().map(|o| o.name).collect();
    Some(SolveResult {
        success: true,
        code,
        method: format!("library-pipeline:{}", names.join("->")),
        error: None,
        metadata: Default::default(),
    })
}

/// True if the emitted 2-arg `pipeline(x, k)` produces a DIFFERENT output for
/// some example when `k` is perturbed — i.e. the scalar genuinely matters. A
/// chain whose output is invariant to `k` (rotation-then-sum, etc.) is a
/// scalar-ignoring overfit and must be rejected.
fn output_depends_on_scalar(code: &str, problem: &Problem) -> bool {
    for ex in &problem.examples {
        let BValue::Int(k) = ex.inputs[1] else { continue };
        // Two perturbations so a single unlucky no-op (e.g. k and k+1 coincide
        // under a modulus) doesn't mask real dependence.
        for delta in [1i64, 2] {
            let mut probe = ex.inputs.clone();
            probe[1] = BValue::Int(k + delta);
            let base = execute_function(code, "pipeline", &ex.inputs, "pipeline");
            let alt = execute_function(code, "pipeline", &probe, "pipeline");
            if let (Ok(b), Ok(a)) = (base, alt) {
                if benchmark_value_from_runtime(&b) != benchmark_value_from_runtime(&a) {
                    return true;
                }
            }
        }
    }
    false
}

/// Search for a chain of 2..=[`MAX_DEPTH`] verified unary ops that reproduces
/// every example. `None` when the problem is not unary, uses types outside [`Ty`],
/// or no chain fits within the budget. Depth-1 fits are deliberately never
/// returned (see module docs).
pub fn try_pipeline(problem: &Problem) -> Option<SolveResult> {
    // Ops kill-switch for clean A/B benchmarking and emergency disable.
    if std::env::var_os("NSYNTH_NO_OP_PIPELINE").is_some() {
        return None;
    }
    if problem.examples.is_empty() {
        return None;
    }
    // OVER-DETERMINATION gate: a multi-op library chain has enormous capacity,
    // so 2 examples are trivially satisfiable by nonsense (observed post-merge:
    // factorial->sum_of_digits->factorial "solving" a 2-example `increment`
    // spec, hijacking the registry-op front door). Require >=3 DISTINCT
    // examples. Zero impact on the MBPP bench (its prep already filters to >=3
    // tests); protects low-example registry ops.
    {
        let mut distinct: Vec<&[BValue]> = Vec::new();
        for e in &problem.examples {
            if !distinct.iter().any(|d| *d == e.inputs.as_slice()) {
                distinct.push(e.inputs.as_slice());
            }
        }
        if distinct.len() < 3 {
            return None;
        }
    }
    // Unary problem: chain over the single input. Binary problem whose SECOND
    // input is a scalar i64: chain over the first input, with exactly one
    // scalar-consuming stage. Anything else is out of scope.
    let arity = problem.examples[0].inputs.len();
    if problem.examples.iter().any(|e| e.inputs.len() != arity) || !(arity == 1 || arity == 2) {
        return None;
    }
    let scalar_arity = arity == 2;
    let scalars: Vec<BValue> = if scalar_arity {
        if problem
            .examples
            .iter()
            .any(|e| !matches!(e.inputs[1], BValue::Int(_)))
        {
            return None;
        }
        problem.examples.iter().map(|e| e.inputs[1].clone()).collect()
    } else {
        Vec::new()
    };
    let in_ty = ty_of_values(problem.examples.iter().map(|e| &e.inputs[0]))?;
    let out_ty = ty_of_values(problem.examples.iter().map(|e| &e.expected))?;
    let ops = stage_ops();
    let deadline = Instant::now() + SEARCH_BUDGET;

    let init: Vec<BValue> = problem.examples.iter().map(|e| e.inputs[0].clone()).collect();
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(format!("false|{}", fingerprint(&init)));

    // state: (current type, current value per example, op path, scalar consumed?)
    let mut frontier: Vec<(Ty, Vec<BValue>, Vec<usize>, bool)> =
        vec![(in_ty, init, Vec::new(), false)];
    let mut memo = std::collections::HashMap::new();

    for depth in 0..MAX_DEPTH {
        // On the final expansion only ops that PRODUCE the goal type can still
        // accept; applying anything else is pure waste.
        let is_last = depth + 1 == MAX_DEPTH;
        let mut next = Vec::new();
        for (ty, vals, path, used_scalar) in &frontier {
            for (i, op) in ops.iter().enumerate() {
                if op.input != *ty {
                    continue;
                }
                // Binary stages: only for scalar-arity problems, at most once.
                if op.scalar && (!scalar_arity || *used_scalar) {
                    continue;
                }
                if is_last && op.output != out_ty {
                    continue;
                }
                // A scalar-arity chain must still be able to consume the scalar.
                if is_last && scalar_arity && !*used_scalar && !op.scalar {
                    continue;
                }
                // Pure safety net: the search terminates on its own; this only
                // guards against a pathological slow op-application run.
                if Instant::now() > deadline {
                    return None;
                }
                let scalar_args = if op.scalar { Some(scalars.as_slice()) } else { None };
                let Some(new_vals) = apply_op(i, op, vals, scalar_args, &mut memo) else {
                    continue;
                };
                let now_used = *used_scalar || op.scalar;
                if !seen.insert(format!("{now_used}|{}", fingerprint(&new_vals))) {
                    continue;
                }
                let mut new_path = path.clone();
                new_path.push(i);
                if op.output == out_ty
                    && new_path.len() >= 2
                    && (!scalar_arity || now_used)
                    && new_vals
                        .iter()
                        .zip(problem.examples.iter())
                        .all(|(v, e)| *v == e.expected)
                {
                    let chain: Vec<&StageOp> = new_path.iter().map(|&j| &ops[j]).collect();
                    if let Some(result) = emit_and_check(&chain, problem, scalar_arity) {
                        return Some(result);
                    }
                }
                if !is_last && new_vals.iter().all(small_enough) {
                    next.push((op.output, new_vals, new_path, now_used));
                }
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }
    // Systematic BFS (depth <= MAX_DEPTH) missed. Try the input-seeded stochastic
    // deep sampler, which can reach DEEPER chains the exhaustive search can't.
    stochastic_deep_search(problem, &ops, in_ty, out_ty, scalar_arity, &scalars, deadline)
}

/// Deterministic xorshift64 RNG (Rust's stdlib has no seedable PRNG, and we need
/// REPRODUCIBILITY — a task must draw the same random sequence every run so the
/// search is deterministic, testable, and flywheel-safe).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

/// Seed the RNG from the problem's examples so the draw sequence is a pure
/// function of the task (deterministic + reproducible).
fn seed_rng(problem: &Problem) -> Rng {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{:?}", problem.examples).hash(&mut h);
    Rng(h.finish() | 1)
}

/// Cap on random draws + chain depth for the stochastic sampler.
const STOCH_MAX_DRAWS: usize = 30_000;
const STOCH_MAX_DEPTH: usize = 6;

#[cfg(test)]
thread_local! {
    /// Per-thread test override for the stochastic stage — avoids the global env
    /// var (which races under parallel tests, flipping other "expect-None" cases).
    static STOCH_TEST_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the stochastic deep search runs. Production: `NSYNTH_STOCHASTIC` env
/// (opt-in for A/B). Tests: a thread-local so each test controls only its own
/// thread, with no cross-test env pollution.
fn stochastic_enabled() -> bool {
    #[cfg(test)]
    {
        if STOCH_TEST_ENABLED.with(|c| c.get()) {
            return true;
        }
    }
    std::env::var_os("NSYNTH_STOCHASTIC").is_some()
}

/// Input-seeded STOCHASTIC deep-chain search ("weighted dice"). The systematic BFS
/// above caps at [`MAX_DEPTH`] (exhaustive → combinatorial), so deep compositions
/// are unreachable. This SAMPLES random type-valid chains up to [`STOCH_MAX_DEPTH`]
/// — random restarts can jump straight to a depth-5/6 program the exhaustive search
/// never reaches in budget ("spontaneous discovery"). Every candidate is still
/// example-matched here and strict-verified in [`emit_and_check`], so recklessness
/// is free — the verifier is the safety net. No model. Deterministic (seeded from
/// the examples). Opt-in via `NSYNTH_STOCHASTIC` for clean A/B.
fn stochastic_deep_search(
    problem: &Problem,
    ops: &[StageOp],
    in_ty: Ty,
    out_ty: Ty,
    scalar_arity: bool,
    scalars: &[BValue],
    deadline: Instant,
) -> Option<SolveResult> {
    if !stochastic_enabled() {
        return None;
    }
    let init: Vec<BValue> = problem.examples.iter().map(|e| e.inputs[0].clone()).collect();
    let mut rng = seed_rng(problem);
    let mut memo = std::collections::HashMap::new();
    for _ in 0..STOCH_MAX_DRAWS {
        if Instant::now() > deadline {
            break;
        }
        let target_depth = 2 + rng.below(STOCH_MAX_DEPTH - 1); // 2..=STOCH_MAX_DEPTH
        let mut vals = init.clone();
        let mut ty = in_ty;
        let mut path: Vec<usize> = Vec::new();
        let mut used_scalar = false;
        let mut alive = true;
        for step in 0..target_depth {
            let is_last = step + 1 == target_depth;
            // Type-valid candidates (the "dice faces"): input type matches, scalar
            // rules honoured, and on the final step only goal-type producers.
            let mut cands: Vec<usize> = Vec::new();
            for (i, op) in ops.iter().enumerate() {
                if op.input != ty {
                    continue;
                }
                if op.scalar && (!scalar_arity || used_scalar) {
                    continue;
                }
                if is_last && op.output != out_ty {
                    continue;
                }
                if is_last && scalar_arity && !used_scalar && !op.scalar {
                    continue;
                }
                cands.push(i);
            }
            if cands.is_empty() {
                alive = false;
                break;
            }
            // "Weighted dice": bias toward ops that MOVE the value type toward the
            // goal (an op whose output type equals the goal, or reshapes toward it),
            // so deep chains still converge instead of wandering. Weight = 3 for a
            // goal-type-producing op, else 1.
            let weights: Vec<u64> =
                cands.iter().map(|&i| if ops[i].output == out_ty { 3 } else { 1 }).collect();
            let total: u64 = weights.iter().sum();
            let mut r = rng.next() % total;
            let mut chosen = cands[0];
            for (k, &i) in cands.iter().enumerate() {
                if r < weights[k] {
                    chosen = i;
                    break;
                }
                r -= weights[k];
            }
            let op = &ops[chosen];
            let scalar_args = if op.scalar { Some(scalars) } else { None };
            let Some(new_vals) = apply_op(chosen, op, &vals, scalar_args, &mut memo) else {
                alive = false;
                break;
            };
            if !is_last && !new_vals.iter().all(small_enough) {
                alive = false;
                break;
            }
            vals = new_vals;
            ty = op.output;
            used_scalar |= op.scalar;
            path.push(chosen);
        }
        if !alive
            || ty != out_ty
            || path.len() < 2
            || (scalar_arity && !used_scalar)
            || !vals.iter().zip(problem.examples.iter()).all(|(v, e)| *v == e.expected)
        {
            continue;
        }
        let chain: Vec<&StageOp> = path.iter().map(|&j| &ops[j]).collect();
        if let Some(mut r) = emit_and_check(&chain, problem, scalar_arity) {
            let names: Vec<&str> = chain.iter().map(|o| o.name).collect();
            r.method = format!("library-pipeline-stochastic:{}", names.join("->"));
            return Some(r);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Example;

    fn problem(examples: Vec<(BValue, BValue)>) -> Problem {
        let mut p = Problem::default();
        p.name = "pipeline_test".to_string();
        p.examples = examples
            .into_iter()
            .map(|(input, expected)| Example { inputs: vec![input], expected })
            .collect();
        p
    }

    fn int_examples(pairs: &[(i64, i64)]) -> Vec<(BValue, BValue)> {
        pairs.iter().map(|&(a, b)| (BValue::Int(a), BValue::Int(b))).collect()
    }

    #[test]
    fn every_compose_op_reproduces_its_probe() {
        let iv = BValue::int_array;
        let cases: &[(&str, BValue, BValue)] = &[
            ("sorted_values", iv(&[3, 1, 2]), iv(&[1, 2, 3])),
            ("reversed_values", iv(&[1, 2, 3]), iv(&[3, 2, 1])),
            ("unique_values", iv(&[2, 1, 2, 3, 1]), iv(&[2, 1, 3])),
            ("abs_values", iv(&[-2, 3, -4]), iv(&[2, 3, 4])),
            ("digits_of", BValue::Int(1203), iv(&[1, 2, 0, 3])),
            ("digits_of", BValue::Int(0), iv(&[0])),
            ("digits_of", BValue::Int(-45), iv(&[4, 5])),
            ("max_value", iv(&[3, 9, 2]), BValue::Int(9)),
            ("min_value", iv(&[3, -9, 2]), BValue::Int(-9)),
            ("sum_values", iv(&[1, 2, 3]), BValue::Int(6)),
            ("count_of", iv(&[7, 7, 7, 7]), BValue::Int(4)),
        ];
        for (name, arg, expect) in cases {
            let op = COMPOSE_OPS.iter().find(|o| o.name == *name).unwrap();
            assert!(
                code_reproduces_examples(
                    op.mog,
                    &[Example { inputs: vec![arg.clone()], expected: expect.clone() }]
                ),
                "compose op {name} failed its probe (expected {expect:?})"
            );
        }
    }

    #[test]
    fn every_binary_compose_op_reproduces_its_probe() {
        let iv = BValue::int_array;
        let cases: &[(&str, Vec<BValue>, BValue)] = &[
            ("take_first", vec![iv(&[5, 6, 7]), BValue::Int(2)], iv(&[5, 6])),
            ("take_first", vec![iv(&[5]), BValue::Int(3)], iv(&[5])),
            ("take_last", vec![iv(&[5, 6, 7]), BValue::Int(2)], iv(&[6, 7])),
            ("take_last", vec![iv(&[5]), BValue::Int(3)], iv(&[5])),
            ("drop_first", vec![iv(&[5, 6, 7]), BValue::Int(1)], iv(&[6, 7])),
            ("drop_last", vec![iv(&[5, 6, 7]), BValue::Int(1)], iv(&[5, 6])),
            ("element_at", vec![iv(&[5, 6, 7]), BValue::Int(1)], BValue::Int(6)),
            ("remove_value", vec![iv(&[1, 2, 1, 3]), BValue::Int(1)], iv(&[2, 3])),
        ];
        for (name, args, expect) in cases {
            let op = COMPOSE_OPS.iter().find(|o| o.name == *name).unwrap();
            assert!(
                code_reproduces_examples(
                    op.mog,
                    &[Example { inputs: args.clone(), expected: expect.clone() }]
                ),
                "binary compose op {name} failed its probe (expected {expect:?})"
            );
        }
    }

    #[test]
    fn stage_ops_parse_their_own_signatures() {
        let ops = stage_ops();
        // Every compose op must be chainable; a sig-parse regression would silently
        // empty the search space.
        for c in COMPOSE_OPS {
            assert!(ops.iter().any(|o| o.name == c.name), "compose op {} not chainable", c.name);
        }
        let digits = ops.iter().find(|o| o.name == "digits_of").unwrap();
        assert_eq!((digits.input, digits.output), (Ty::Int, Ty::IntArr));
        // Allowlisted compositional primitive.
        let prime = ops.iter().find(|o| o.name == "is_prime").unwrap();
        assert_eq!((prime.input, prime.output), (Ty::Int, Ty::Int));
        assert!(!prime.scalar);
        // Binary compose stage: (flow, i64) parses with scalar=true.
        let take = ops.iter().find(|o| o.name == "take_first").unwrap();
        assert!(take.scalar);
        assert_eq!((take.input, take.output), (Ty::IntArr, Ty::IntArr));
        // CURATION: whole-answer TERMINAL library ops are NOT chain stages (they
        // stay single-op library solves — full reach, no search bloat). A few
        // representative terminals must be absent from the chain grammar.
        for terminal in ["nonagonal_number", "volume_cube", "sum_divisors", "lucas_number"] {
            assert!(
                !ops.iter().any(|o| o.name == terminal),
                "terminal op {terminal} must NOT be a chain stage (library-only)"
            );
        }
    }

    #[test]
    fn pipeline_solves_sum_of_digits_of_factorial() {
        // No single op fits: factorial(4)=24 but expected 6; sum_of_digits(4)=4.
        let p = problem(int_examples(&[(1, 1), (2, 2), (3, 6), (4, 6), (5, 3)]));
        assert!(crate::op_library::try_library(&p).is_none(), "single-op library must miss");
        let r = try_pipeline(&p).expect("pipeline should solve sum_of_digits∘factorial");
        assert!(r.method.starts_with("library-pipeline:"), "method was {}", r.method);
        // Un-gameable: the returned program must generalize to an UNSEEN input
        // (8! = 40320 → 4+0+3+2+0 = 9), not merely fit the five seeds.
        assert!(
            code_reproduces_examples(
                &r.code,
                &[Example { inputs: vec![BValue::Int(8)], expected: BValue::Int(9) }]
            ),
            "returned pipeline failed the held-out probe:\n{}",
            r.code
        );
    }

    #[test]
    fn pipeline_solves_primality_of_reversed_number() {
        // is_prime(reverse_number(n)). 16→61 prime distinguishes from is_prime(n).
        let p = problem(int_examples(&[(13, 1), (15, 0), (16, 1), (18, 0), (35, 1), (25, 0)]));
        assert!(crate::op_library::try_library(&p).is_none(), "single-op library must miss");
        let r = try_pipeline(&p).expect("pipeline should solve is_prime∘reverse_number");
        // Held-out: 94→49=7² → 0; 76→67 prime → 1.
        assert!(
            code_reproduces_examples(
                &r.code,
                &[
                    Example { inputs: vec![BValue::Int(94)], expected: BValue::Int(0) },
                    Example { inputs: vec![BValue::Int(76)], expected: BValue::Int(1) },
                ]
            ),
            "returned pipeline failed held-out probes:\n{}",
            r.code
        );
    }

    #[test]
    fn pipeline_solves_sum_of_squares_of_distinct_values() {
        // sum_of_squares(unique_values(arr)): [2,2,3] → 4+9 = 13 (plain
        // sum_of_squares gives 17, so the reshape stage is load-bearing).
        let iv = BValue::int_array;
        let p = problem(vec![
            (iv(&[2, 2, 3]), BValue::Int(13)),
            (iv(&[1, 1, 1]), BValue::Int(1)),
            (iv(&[0, 2, 2]), BValue::Int(4)),
            (iv(&[3, 4]), BValue::Int(25)),
        ]);
        assert!(crate::op_library::try_library(&p).is_none(), "single-op library must miss");
        let r = try_pipeline(&p).expect("pipeline should solve sum_of_squares∘unique_values");
        assert!(
            code_reproduces_examples(
                &r.code,
                &[Example { inputs: vec![iv(&[5, 5, 2])], expected: BValue::Int(29) }]
            ),
            "returned pipeline failed the held-out probe:\n{}",
            r.code
        );
    }

    #[test]
    fn pipeline_solves_depth_three_sum_of_distinct_digits() {
        // sum_values(unique_values(digits_of(n))): 122 → [1,2,2] → [1,2] → 3
        // (sum_of_digits(122)=5 distinguishes the depth-2 alternative).
        let p = problem(int_examples(&[(122, 3), (505, 5), (333, 3), (1234, 10)]));
        assert!(crate::op_library::try_library(&p).is_none(), "single-op library must miss");
        let r = try_pipeline(&p).expect("pipeline should solve depth-3 distinct-digit sum");
        assert!(
            code_reproduces_examples(
                &r.code,
                &[Example { inputs: vec![BValue::Int(7717)], expected: BValue::Int(8) }]
            ),
            "returned pipeline failed the held-out probe:\n{}",
            r.code
        );
    }

    #[test]
    fn pipeline_solves_n_largest_via_scalar_stage() {
        // heap_queue_largest class: n largest, descending =
        // take_first(reversed(sorted(arr)), k) — needs the scalar pass-through.
        let iv = BValue::int_array;
        let p = problem(vec![]);
        let mut p = p;
        p.examples = vec![
            Example {
                inputs: vec![iv(&[25, 35, 22, 85, 14, 65, 75, 25]), BValue::Int(3)],
                expected: iv(&[85, 75, 65]),
            },
            Example { inputs: vec![iv(&[1, 2, 3]), BValue::Int(1)], expected: iv(&[3]) },
            Example { inputs: vec![iv(&[4, 1, 4]), BValue::Int(2)], expected: iv(&[4, 4]) },
        ];
        assert!(crate::op_library::try_library(&p).is_none(), "single-op library must miss");
        let r = try_pipeline(&p).expect("pipeline should solve n-largest");
        assert!(r.method.starts_with("library-pipeline:"), "method was {}", r.method);
        // Held-out generalization probe.
        assert!(
            code_reproduces_examples(
                &r.code,
                &[Example {
                    inputs: vec![iv(&[9, 1, 5, 7]), BValue::Int(2)],
                    expected: iv(&[9, 7]),
                }]
            ),
            "returned pipeline failed the held-out probe:\n{}",
            r.code
        );
    }

    #[test]
    fn pipeline_solves_sum_of_first_k() {
        let iv = BValue::int_array;
        let mut p = problem(vec![]);
        p.examples = vec![
            Example { inputs: vec![iv(&[1, 2, 3, 4]), BValue::Int(2)], expected: BValue::Int(3) },
            Example { inputs: vec![iv(&[5, 5, 5]), BValue::Int(1)], expected: BValue::Int(5) },
            Example { inputs: vec![iv(&[2, 4, 6]), BValue::Int(3)], expected: BValue::Int(12) },
        ];
        assert!(crate::op_library::try_library(&p).is_none(), "single-op library must miss");
        let r = try_pipeline(&p).expect("pipeline should solve sum-of-first-k");
        assert!(
            code_reproduces_examples(
                &r.code,
                &[Example {
                    inputs: vec![iv(&[10, 1, 1]), BValue::Int(2)],
                    expected: BValue::Int(11),
                }]
            ),
            "returned pipeline failed the held-out probe:\n{}",
            r.code
        );
    }

    #[test]
    fn pipeline_rejects_vacuous_scalar_consumption() {
        // sum is rotation-invariant, so sum_values(rotate_left(arr, k)) CONSUMES
        // k structurally but the output never depends on it — a scalar-ignoring
        // overfit dressed up as a scalar chain. The scalar-dependence check must
        // reject it even though a 3-example fit exists.
        let iv = BValue::int_array;
        let mut p = problem(vec![]);
        // Outputs are the full sum (rotation-invariant), with distinctive values
        // and k choices so no genuinely-k-dependent op (take/drop/window/count)
        // coincidentally matches — the ONLY fit is rotate_left∘sum, which the
        // dependence gate must reject as vacuous.
        p.examples = vec![
            Example { inputs: vec![iv(&[10, 20, 30]), BValue::Int(2)], expected: BValue::Int(60) },
            Example { inputs: vec![iv(&[5, 5, 5, 5]), BValue::Int(3)], expected: BValue::Int(20) },
            Example { inputs: vec![iv(&[1, 2, 3, 4]), BValue::Int(1)], expected: BValue::Int(10) },
            Example { inputs: vec![iv(&[7, 8]), BValue::Int(2)], expected: BValue::Int(15) },
        ];
        assert!(
            try_pipeline(&p).is_none(),
            "must reject a chain whose output is invariant to the scalar"
        );
    }

    #[test]
    fn pipeline_rejects_scalar_ignoring_fit() {
        // The outputs are plain sums; the scalar argument is irrelevant. A chain
        // that never consumes the scalar must NOT be returned (ignoring an
        // argument = overfit laundering), and no scalar-consuming chain fits.
        let iv = BValue::int_array;
        let mut p = problem(vec![]);
        p.examples = vec![
            Example { inputs: vec![iv(&[1, 2, 3]), BValue::Int(2)], expected: BValue::Int(6) },
            Example { inputs: vec![iv(&[2, 2]), BValue::Int(3)], expected: BValue::Int(4) },
            Example { inputs: vec![iv(&[1, 1, 1]), BValue::Int(5)], expected: BValue::Int(3) },
        ];
        assert!(try_pipeline(&p).is_none(), "must not solve by ignoring the scalar argument");
    }

    #[test]
    fn stochastic_finds_depth_four_chain_beyond_systematic() {
        // reverse(sort(unique(abs(arr)))): every op is load-bearing, so the
        // ONLY solution is depth-4 — the systematic BFS (MAX_DEPTH=3) cannot
        // reach it. The input-seeded stochastic sampler can.
        let iv = BValue::int_array;
        let f = |a: &[i64]| -> Vec<i64> {
            let mut v: Vec<i64> = a.iter().map(|x| x.abs()).collect();
            v.sort();
            v.dedup();
            v.reverse();
            v
        };
        let inputs: &[&[i64]] = &[&[-3, 2, -3, 1], &[5, -5, 2, 2, 8], &[-1, -4, 4, 1, 0], &[7, 7, -2, 3]];
        let mut p = problem(vec![]);
        p.examples = inputs
            .iter()
            .map(|a| Example { inputs: vec![iv(a)], expected: iv(&f(a)) })
            .collect();

        // With stochastic OFF, the systematic depth-3 search must MISS.
        STOCH_TEST_ENABLED.with(|c| c.set(false));
        assert!(try_pipeline(&p).is_none(), "systematic depth-3 must not reach a depth-4 chain");

        // With stochastic ON (thread-local — no cross-test env pollution).
        STOCH_TEST_ENABLED.with(|c| c.set(true));
        let r = try_pipeline(&p).expect("stochastic sampler should discover the depth-4 chain");
        STOCH_TEST_ENABLED.with(|c| c.set(false));
        assert!(
            r.method.starts_with("library-pipeline-stochastic:"),
            "expected stochastic attribution, got {}",
            r.method
        );
        // Generalises to an unseen input (held-out).
        assert!(
            code_reproduces_examples(
                &r.code,
                &[Example { inputs: vec![iv(&[-9, 4, 4, -9, 6])], expected: iv(&f(&[-9, 4, 4, -9, 6])) }]
            ),
            "stochastically-discovered program must generalise:\n{}",
            r.code
        );
    }

    #[test]
    fn stochastic_is_deterministic() {
        // Same task -> same seed -> same result (reproducible), twice.
        let iv = BValue::int_array;
        let f = |a: &[i64]| -> Vec<i64> {
            let mut v: Vec<i64> = a.iter().map(|x| x.abs()).collect();
            v.sort();
            v.dedup();
            v.reverse();
            v
        };
        let inputs: &[&[i64]] = &[&[-3, 2, -3, 1], &[5, -5, 2, 2, 8], &[-1, -4, 4, 1, 0], &[7, 7, -2, 3]];
        let mut p = problem(vec![]);
        p.examples = inputs
            .iter()
            .map(|a| Example { inputs: vec![iv(a)], expected: iv(&f(a)) })
            .collect();
        STOCH_TEST_ENABLED.with(|c| c.set(true));
        let a = try_pipeline(&p).map(|r| r.code);
        let b = try_pipeline(&p).map(|r| r.code);
        STOCH_TEST_ENABLED.with(|c| c.set(false));
        assert_eq!(a, b, "stochastic search must be deterministic for a fixed task");
    }

    #[test]
    fn pipeline_returns_none_for_unchainable_examples() {
        let p = problem(int_examples(&[(1, 100), (2, -3), (3, 77)]));
        assert!(try_pipeline(&p).is_none());
    }

    #[test]
    fn pipeline_never_returns_a_single_op_fit() {
        // Factorial itself is a SINGLE library op; the pipeline tier must stay out
        // of single-op attribution. Identity-like wrappers (reverse_number on
        // single digits, digits_of→max_value on 1..=5) are killed by the
        // observational-equivalence prune, not by luck.
        let p = problem(int_examples(&[(1, 1), (2, 2), (3, 6), (4, 24), (5, 120)]));
        assert!(
            crate::op_library::try_library(&p).is_some(),
            "precondition: factorial is a single-op library solve"
        );
        assert!(try_pipeline(&p).is_none(), "pipeline must not launder a depth-1 fit");
    }

    #[test]
    fn solve_problem_routes_chain_task_to_pipeline() {
        // End-to-end wiring: the full solver returns the pipeline method for a
        // task neither the single-op library nor bounded search reproduces.
        let p = problem(int_examples(&[(1, 1), (2, 2), (3, 6), (4, 6), (5, 3)]));
        let r = crate::solver::solve_problem(&p);
        assert!(r.success, "solve_problem failed: {:?}", r.error);
        assert!(
            r.method.starts_with("library-pipeline:"),
            "expected pipeline attribution, got {}",
            r.method
        );
    }
}
