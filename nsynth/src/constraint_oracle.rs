//! INDEPENDENT SEMANTIC ORACLE (constraint-oracle).
//!
//! The examples-only synthesis regime has no true oracle: a candidate that fits
//! the seed examples is accepted even when it does not actually implement the
//! requested behavior. That hole WIDENS as specs get more abstract (they are the
//! under-determined, oracle-less ones). This module closes part of it WITHOUT a
//! reference implementation, by turning the request's own RESOLVED OPERATION into
//! a decidable property and checking the candidate honors it on FRESH inputs the
//! example set never saw.
//!
//! The operation name is the semantic contract: an intent that resolved to `max`
//! promises `f(xs) == max(xs)`. If the synthesized code returns the right seed
//! answers but is not actually the maximum, a fresh random input exposes it and
//! the solve is rejected. This is EMERGENT, not a keyword table — the op lemma
//! comes from the same `EntityResolver` the synthesizer already used to choose
//! WHAT to build; this module only maps a small set of unambiguous op contracts
//! to properties, and every check is a decidable predicate over (inputs, output).
//!
//! Deliberately small and high-signal: each contract is a hard invariant an
//! overfit cannot fake across many fresh inputs. Fail-closed — a violation, or a
//! candidate that errors on a valid fresh input, rejects the solve.

use crate::benchmark::Value;
use crate::runtime::{execute_function, Value as RValue};

/// A decidable property the OUTPUT must satisfy given the INPUTS, derived from
/// the request's resolved operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Property {
    /// output == max of the single int-array input
    IsMax,
    /// output == min of the single int-array input
    IsMin,
    /// output >= 0 (absolute value / magnitude / non-negative)
    IsNonNegative,
    /// output array is sorted non-decreasing AND a permutation of the input
    IsSortedAscending,
    /// output == sum of the single int-array input
    IsSum,
    /// output == product of the single int-array input
    IsProduct,
    /// output array is the input reversed (same multiset, reversed order)
    IsReversed,
}

impl Property {
    /// Map a canonical op lemma to its cheap contract. `None` = this op has no
    /// unambiguous decidable output property here (the oracle simply does not
    /// apply — never a rejection on its own).
    pub fn for_op(op_lemma: &str) -> Option<Property> {
        match op_lemma.trim().to_ascii_lowercase().as_str() {
            "max" | "maximum" | "array_max" => Some(Property::IsMax),
            "min" | "minimum" | "array_min" => Some(Property::IsMin),
            "abs" | "absolute" | "absolute_value" | "magnitude" => Some(Property::IsNonNegative),
            "sort" | "sorted" | "sort_ascending" => Some(Property::IsSortedAscending),
            "sum" | "total" | "array_sum" => Some(Property::IsSum),
            "product" | "array_product" => Some(Property::IsProduct),
            "reverse" | "reversed" | "reverse_array" => Some(Property::IsReversed),
            _ => None,
        }
    }

    /// Whether the contract reads a single int-array input (vs a single scalar).
    fn wants_array(&self) -> bool {
        matches!(
            self,
            Property::IsMax
                | Property::IsMin
                | Property::IsSortedAscending
                | Property::IsSum
                | Property::IsProduct
                | Property::IsReversed
        )
    }

    /// Decide the contract on one concrete (inputs, output). `inputs` are the
    /// benchmark-shaped arguments the candidate was called with; `output` is the
    /// runtime value it returned.
    fn holds(&self, inputs: &[Value], output: &RValue) -> Result<(), String> {
        match self {
            Property::IsMax | Property::IsMin => {
                let arr = input_array(inputs).ok_or("expected an int-array input")?;
                let out = rint(output).ok_or("expected an int output")?;
                let want = if *self == Property::IsMax {
                    arr.iter().copied().max()
                } else {
                    arr.iter().copied().min()
                }
                .ok_or("empty array has no extremum")?;
                if out == want {
                    Ok(())
                } else {
                    Err(format!("{self:?}: got {out}, expected {want} for {arr:?}"))
                }
            }
            Property::IsNonNegative => {
                let out = rint(output).ok_or("expected an int output")?;
                if out >= 0 {
                    Ok(())
                } else {
                    Err(format!("IsNonNegative: got negative output {out}"))
                }
            }
            Property::IsSortedAscending => {
                let arr = input_array(inputs).ok_or("expected an int-array input")?;
                let out = rarr(output).ok_or("expected an int-array output")?;
                if out.windows(2).any(|w| w[0] > w[1]) {
                    return Err(format!("IsSortedAscending: output {out:?} not non-decreasing"));
                }
                // Sorting must PRESERVE the multiset — no adds/drops/mutations.
                let mut a = arr.clone();
                let mut b = out.clone();
                a.sort_unstable();
                b.sort_unstable();
                if a != b {
                    return Err(format!(
                        "IsSortedAscending: output {out:?} is not a permutation of input {arr:?}"
                    ));
                }
                Ok(())
            }
            Property::IsSum | Property::IsProduct => {
                let arr = input_array(inputs).ok_or("expected an int-array input")?;
                let out = rint(output).ok_or("expected an int output")?;
                let want: i64 = if *self == Property::IsSum {
                    arr.iter().fold(0i64, |a, &x| a.wrapping_add(x))
                } else {
                    arr.iter().fold(1i64, |a, &x| a.wrapping_mul(x))
                };
                if out == want {
                    Ok(())
                } else {
                    Err(format!("{self:?}: got {out}, expected {want} for {arr:?}"))
                }
            }
            Property::IsReversed => {
                let arr = input_array(inputs).ok_or("expected an int-array input")?;
                let out = rarr(output).ok_or("expected an int-array output")?;
                let want: Vec<i64> = arr.iter().rev().copied().collect();
                if out == want {
                    Ok(())
                } else {
                    Err(format!("IsReversed: output {out:?} != reverse of input {arr:?}"))
                }
            }
        }
    }
}

/// Verify a synthesized candidate against the contract of the operation it claims
/// to implement, on fresh inputs. `Ok(())` means either the op has no contract
/// here (not applicable) or the candidate honored it on every sampled input.
/// `Err` means a decidable violation (or a fresh-input crash) — the caller should
/// fail closed (reject / downgrade the solve).
pub fn check_op_contract(
    code: &str,
    fn_name: &str,
    op_lemma: &str,
    sample_inputs: &[Value],
) -> Result<(), String> {
    let Some(prop) = Property::for_op(op_lemma) else {
        return Ok(());
    };
    // SHAPE GUARD (prevents false rejections): only apply the contract when the
    // task's REAL input shape matches what the contract reads. A problem named
    // "max" that takes two scalars is not array-max — skip rather than reject.
    let shape_ok = sample_inputs.len() == 1
        && if prop.wants_array() {
            matches!(sample_inputs.first(), Some(Value::Array(_)))
        } else {
            matches!(sample_inputs.first(), Some(Value::Int(_)))
        };
    if !shape_ok {
        return Ok(());
    }
    verify_candidate(code, fn_name, &prop, 24, 0x5eed_1234)
}

/// Run `code`'s `fn_name` on `n` fresh inputs and require `prop` to hold on each.
pub fn verify_candidate(
    code: &str,
    fn_name: &str,
    prop: &Property,
    n: usize,
    seed: u64,
) -> Result<(), String> {
    for inputs in fresh_inputs(prop, n, seed) {
        let output = execute_function(code, fn_name, &inputs, "constraint_oracle")
            .map_err(|e| format!("candidate errored on fresh input {inputs:?}: {e}"))?;
        prop.holds(&inputs, &output)?;
    }
    Ok(())
}

/// ORACLE MANUFACTURING. Confirm a bare-NL (no-example) guess reference-free and
/// EMERGENTLY. [`reference_for`] resolves the prompt to the SINGLE library op whose
/// operation-word signature exactly equals the prompt's — the completeness gate that makes
/// "sum of squares" / "second largest" / a two-op compose resolve to nothing — then the
/// candidate is DIFFERENTIALLY executed against that op's verified `.mog` on many fresh
/// inputs. Agreement everywhere means the candidate computes that exact operation, so the
/// guess is confirmed (drop tentative); a wrong candidate (e.g. a mis-built max combinator)
/// disagrees on a fresh input and stays tentative. Works for both a library match (candidate
/// == reference => trivially agrees) and an untrusted synthesis (real differential check).
/// The "spec" is the resolved reference's behavior, not a hand-written property table. Any
/// no-match / shape-mismatch / crash => false (honest: stays tentative, never confident-wrong).
pub fn confirm_from_prompt(code: &str, fn_name: &str, prompt: &str) -> bool {
    let Some(cand_types) = sig_param_types(code) else {
        return false;
    };
    let Some(reference) = crate::verified_nl_router::reference_for(prompt, &cand_types) else {
        return false;
    };
    differential_matches(code, fn_name, reference.mog, 32, 0x00ac_1e50)
}

/// True iff `cand` (its `cand_fn`) computes the SAME function as the reference program
/// `ref_mog` on `n` fresh inputs shaped to the reference's signature. Any disagreement, a
/// crash on either side, or an un-buildable input shape => false (fail-closed).
fn differential_matches(cand: &str, cand_fn: &str, ref_mog: &str, n: usize, seed: u64) -> bool {
    let Some(ref_fn) = crate::site::fn_name_from_mog(ref_mog) else {
        return false;
    };
    let Some(param_types) = sig_param_types(ref_mog) else {
        return false;
    };
    for inputs in fresh_typed_args(&param_types, n, seed) {
        let (Ok(a), Ok(b)) = (
            execute_function(ref_mog, &ref_fn, &inputs, "constraint_oracle"),
            execute_function(cand, cand_fn, &inputs, "constraint_oracle"),
        ) else {
            return false;
        };
        if !outputs_agree(&a, &b) {
            return false;
        }
    }
    true
}

/// True iff `p` (its fn `p_fn`) and `q` (its fn `q_fn`) are DISTINGUISHABLE — they produce
/// a DIFFERENT output for at least one fresh input shaped to `p`'s signature. Reuses the same
/// fresh-input differential as [`confirm_from_prompt`]. Used by the bare-NL composition tier
/// to reject a 2-op chain `b(a(x))` that is INDISTINGUISHABLE from its own inner op `a` (the
/// outer op does nothing observable, so the prompt is really a single-op request). A crash on
/// exactly one side is an observable difference (distinguishable); if `p`'s signature can't be
/// parsed or its inputs can't be sampled, returns false (fail-safe: treat as indistinguishable
/// so the caller refuses the chain rather than shipping an unproven composition).
pub fn programs_distinguishable(p: &str, p_fn: &str, q: &str, q_fn: &str) -> bool {
    let Some(param_types) = sig_param_types(p) else {
        return false;
    };
    let args = fresh_typed_args(&param_types, 32, 0x51a9_bd27_0e13_c4a1);
    if args.is_empty() {
        return false; // unsupported input shape -> cannot sample -> fail-safe
    }
    for inputs in args {
        match (
            execute_function(p, p_fn, &inputs, "constraint_oracle"),
            execute_function(q, q_fn, &inputs, "constraint_oracle"),
        ) {
            (Ok(a), Ok(b)) => {
                if !outputs_agree(&a, &b) {
                    return true; // observed a difference
                }
            }
            // exactly one side errs on a valid input => observably different behaviour;
            // both err => no evidence of a difference (keep scanning / stay closed).
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => return true,
            (Err(_), Err(_)) => {}
        }
    }
    false
}

/// Parse a Mog signature's parameter TYPE strings, e.g. `fn f(a: i64, b: [i64]) -> i64`
/// -> ["i64", "[i64]"]. `None` if the signature can't be parsed. Empty vec = nullary.
fn sig_param_types(mog: &str) -> Option<Vec<String>> {
    let (o, c) = (mog.find('(')?, mog.find(')')?);
    if c < o {
        return None;
    }
    let inner = mog[o + 1..c].trim();
    if inner.is_empty() {
        return Some(vec![]);
    }
    inner
        .split(',')
        .map(|p| p.split(':').nth(1).map(|t| t.trim().to_string()))
        .collect()
}

/// Deterministic fresh argument tuples matching a list of Mog parameter types. Supports the
/// shapes the oracle can build+compare (i64, [i64], string, bool); returns an empty batch if
/// any type is unsupported so confirmation fails closed rather than guessing on a shape it
/// can't sample.
fn fresh_typed_args(param_types: &[String], n: usize, seed: u64) -> Vec<Vec<Value>> {
    let mut state = seed ^ 0x2545_f491_4f6c_dd1d;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut args = Vec::with_capacity(param_types.len());
        for ty in param_types {
            match ty.as_str() {
                // Scalars stay small (|n| <= 20) so exponential-growth ops don't overflow i64
                // on a fresh input and fail closed: 20! < i64::MAX but 21! overflows, and a
                // correct factorial/power op would be wrongly left tentative on a large sample.
                "i64" => args.push(Value::Int((lcg(&mut state) % 41) as i64 - 20)),
                "[i64]" => {
                    let len = 1 + (lcg(&mut state) % 7) as usize; // 1..=7 (non-empty: max/min safe)
                    let arr: Vec<i64> =
                        (0..len).map(|_| (lcg(&mut state) % 41) as i64 - 20).collect();
                    args.push(Value::int_array(&arr));
                }
                "string" => {
                    let len = (lcg(&mut state) % 8) as usize; // 0..=7
                    let s: String = (0..len)
                        .map(|_| (b'a' + (lcg(&mut state) % 26) as u8) as char)
                        .collect();
                    args.push(Value::Str(s));
                }
                "bool" => args.push(Value::Bool(lcg(&mut state) % 2 == 0)),
                _ => return Vec::new(), // unsupported shape -> fail closed
            }
        }
        out.push(args);
    }
    out
}

/// Structural equality of two runtime outputs over the shapes the oracle differential-tests.
fn outputs_agree(a: &RValue, b: &RValue) -> bool {
    match (a, b) {
        (RValue::Int(x), RValue::Int(y)) => x == y,
        (RValue::Bool(x), RValue::Bool(y)) => x == y,
        (RValue::Str(x), RValue::Str(y)) => x == y,
        (RValue::Array(_), RValue::Array(_)) => rarr(a) == rarr(b),
        _ => rt_eq(a, b),
    }
}

/// Deterministic fresh inputs for a property's input shape. Deterministic (LCG,
/// caller-seeded) so verification is reproducible and never flaky.
pub fn fresh_inputs(prop: &Property, n: usize, seed: u64) -> Vec<Vec<Value>> {
    fresh_shaped(prop.wants_array(), n, seed)
}

/// Deterministic fresh single-argument inputs: an int array (`wants_array`) or a
/// scalar int. Shared by the property and metamorphic harnesses.
fn fresh_shaped(wants_array: bool, n: usize, seed: u64) -> Vec<Vec<Value>> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if wants_array {
            let len = 3 + (lcg(&mut state) % 6) as usize; // 3..=8
            let arr: Vec<i64> = (0..len)
                .map(|_| (lcg(&mut state) % 61) as i64 - 30) // -30..=30
                .collect();
            out.push(vec![Value::int_array(&arr)]);
        } else {
            let x = (lcg(&mut state) % 101) as i64 - 50; // -50..=50
            out.push(vec![Value::Int(x)]);
        }
    }
    out
}

/// Deterministic fresh PAIRS of scalar ints (for two-argument relations like
/// commutativity).
fn fresh_scalar_pairs(n: usize, seed: u64) -> Vec<(i64, i64)> {
    let mut state = seed ^ 0x1234_5678_9abc_def0;
    (0..n)
        .map(|_| {
            let a = (lcg(&mut state) % 101) as i64 - 50;
            let b = (lcg(&mut state) % 101) as i64 - 50;
            (a, b)
        })
        .collect()
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 16
}

fn input_array(inputs: &[Value]) -> Option<Vec<i64>> {
    inputs.iter().find_map(|v| match v {
        Value::Array(xs) => xs
            .iter()
            .map(|e| match e {
                Value::Int(i) => Some(*i),
                _ => None,
            })
            .collect(),
        _ => None,
    })
}

fn rint(v: &RValue) -> Option<i64> {
    match v {
        RValue::Int(i) => Some(*i),
        _ => None,
    }
}

fn rarr(v: &RValue) -> Option<Vec<i64>> {
    match v {
        RValue::Array(xs) => xs.iter().map(rint).collect(),
        _ => None,
    }
}

/// Convert a runtime output back into a benchmark input value, so a candidate's
/// OUTPUT can be fed back as its INPUT (idempotence / involution checks). Only
/// the shapes the oracle reasons about (int, int-array).
fn bench_from_runtime(v: &RValue) -> Option<Value> {
    match v {
        RValue::Int(i) => Some(Value::Int(*i)),
        RValue::Array(_) => rarr(v).map(|xs| Value::int_array(&xs)),
        _ => None,
    }
}

/// Structural equality of two runtime outputs over the oracle's shapes.
fn rt_eq(a: &RValue, b: &RValue) -> bool {
    match (a, b) {
        (RValue::Int(x), RValue::Int(y)) => x == y,
        (RValue::Array(_), RValue::Array(_)) => rarr(a) == rarr(b),
        _ => false,
    }
}

// ── METAMORPHIC RELATIONS ────────────────────────────────────────────────────
// A metamorphic relation is an ALGEBRAIC LAW the operation must satisfy — checked
// by running the candidate on RELATED inputs and comparing outputs, with NO
// reference oracle. This is the widest solver-independent, proof-carrying rung:
// it covers any op carrying a law (sort is idempotent + order-invariant, reverse
// is an involution, add/mul are commutative, sum/max are order-invariant), rather
// than a fixed list of named contracts.

/// A law the operation's outputs must obey across related executions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetamorphicRelation {
    /// f(f(x)) == f(x) — sort, dedupe, abs, clamp, normalize.
    Idempotent,
    /// f(f(x)) == x — reverse, negate, transpose (self-inverse).
    Involutive,
    /// f(a, b) == f(b, a) — add, multiply, max, min, gcd (two scalar args).
    Commutative,
    /// f(xs) == f(permute(xs)) — sum, product, max, min, count (order-blind reduce).
    OrderInvariant,
    /// len(f(xs)) == len(xs) — sort, reverse, map (shape-preserving array op).
    LengthPreserving,
}

impl MetamorphicRelation {
    /// The laws an operation is expected to satisfy, derived from its lemma — the
    /// same emergent source (the resolver's chosen op) as the output contracts.
    pub fn for_op(op_lemma: &str) -> Vec<MetamorphicRelation> {
        use MetamorphicRelation::*;
        match op_lemma.trim().to_ascii_lowercase().as_str() {
            "sort" | "sorted" | "sort_ascending" => vec![Idempotent, OrderInvariant, LengthPreserving],
            "reverse" | "reversed" | "reverse_array" => vec![Involutive, LengthPreserving],
            "abs" | "absolute" | "absolute_value" | "magnitude" => vec![Idempotent],
            "negate" | "neg" | "negative" => vec![Involutive],
            "sum" | "total" | "array_sum" | "product" | "array_product" | "max" | "maximum"
            | "min" | "minimum" => vec![OrderInvariant],
            "add" | "plus" | "multiply" | "times" | "gcd" | "lcm" => vec![Commutative],
            _ => vec![],
        }
    }
}

/// Verify a candidate against every metamorphic law its operation should obey, on
/// fresh inputs. Each law is SHAPE-GUARDED — a law that does not fit the task's
/// real signature is skipped, never a false rejection. `Ok(())` = all applicable
/// laws held (or none applied); `Err` = a decidable law violation or a crash.
pub fn check_op_metamorphic(
    code: &str,
    fn_name: &str,
    op_lemma: &str,
    sample_inputs: &[Value],
) -> Result<(), String> {
    for rel in MetamorphicRelation::for_op(op_lemma) {
        verify_relation(code, fn_name, rel, sample_inputs, 24, 0x517e_1a30)?;
    }
    Ok(())
}

fn run(code: &str, fn_name: &str, inputs: &[Value]) -> Result<RValue, String> {
    execute_function(code, fn_name, inputs, "constraint_oracle")
        .map_err(|e| format!("candidate errored on fresh input {inputs:?}: {e}"))
}

fn verify_relation(
    code: &str,
    fn_name: &str,
    rel: MetamorphicRelation,
    sample_inputs: &[Value],
    n: usize,
    seed: u64,
) -> Result<(), String> {
    use MetamorphicRelation::*;
    let one_array =
        sample_inputs.len() == 1 && matches!(sample_inputs.first(), Some(Value::Array(_)));
    let one_scalar =
        sample_inputs.len() == 1 && matches!(sample_inputs.first(), Some(Value::Int(_)));
    let two_scalar = sample_inputs.len() == 2
        && sample_inputs.iter().all(|v| matches!(v, Value::Int(_)));

    match rel {
        Idempotent => {
            // Endomorphism: f's output must be feedable as its input. Applies to
            // array->array (sort) and scalar->scalar (abs); skip otherwise.
            if !one_array && !one_scalar {
                return Ok(());
            }
            for inputs in fresh_shaped(one_array, n, seed) {
                let y = run(code, fn_name, &inputs)?;
                let Some(y_in) = bench_from_runtime(&y) else {
                    return Ok(()); // output shape not feedable — law N/A
                };
                let z = run(code, fn_name, &[y_in])?;
                if !rt_eq(&y, &z) {
                    return Err(format!(
                        "Idempotent: f(f(x)) != f(x) — f({inputs:?})={y:?} but reapplying gives {z:?}"
                    ));
                }
            }
        }
        Involutive => {
            if !one_array && !one_scalar {
                return Ok(());
            }
            for inputs in fresh_shaped(one_array, n, seed) {
                let y = run(code, fn_name, &inputs)?;
                let Some(y_in) = bench_from_runtime(&y) else {
                    return Ok(());
                };
                let z = run(code, fn_name, &[y_in])?;
                // f(f(x)) must equal x (the original input, as a runtime value).
                let x_rt = match &inputs[0] {
                    Value::Int(i) => RValue::Int(*i),
                    Value::Array(_) => {
                        let Some(xs) = input_array(&inputs) else { return Ok(()) };
                        RValue::Array(xs.into_iter().map(RValue::Int).collect())
                    }
                    _ => return Ok(()),
                };
                if !rt_eq(&z, &x_rt) {
                    return Err(format!(
                        "Involutive: f(f(x)) != x — f(f({inputs:?}))={z:?}"
                    ));
                }
            }
        }
        Commutative => {
            if !two_scalar {
                return Ok(());
            }
            for (a, b) in fresh_scalar_pairs(n, seed) {
                let r1 = run(code, fn_name, &[Value::Int(a), Value::Int(b)])?;
                let r2 = run(code, fn_name, &[Value::Int(b), Value::Int(a)])?;
                if !rt_eq(&r1, &r2) {
                    return Err(format!(
                        "Commutative: f({a},{b})={r1:?} != f({b},{a})={r2:?}"
                    ));
                }
            }
        }
        OrderInvariant => {
            if !one_array {
                return Ok(());
            }
            for inputs in fresh_shaped(true, n, seed) {
                let Some(xs) = input_array(&inputs) else { continue };
                let mut rev = xs.clone();
                rev.reverse(); // reversal is a valid permutation
                let r1 = run(code, fn_name, &inputs)?;
                let r2 = run(code, fn_name, &[Value::int_array(&rev)])?;
                if !rt_eq(&r1, &r2) {
                    return Err(format!(
                        "OrderInvariant: f({xs:?})={r1:?} != f(permute)={r2:?}"
                    ));
                }
            }
        }
        LengthPreserving => {
            if !one_array {
                return Ok(());
            }
            for inputs in fresh_shaped(true, n, seed) {
                let Some(xs) = input_array(&inputs) else { continue };
                let y = run(code, fn_name, &inputs)?;
                let Some(ys) = rarr(&y) else {
                    return Ok(()); // not an array output — law N/A
                };
                if ys.len() != xs.len() {
                    return Err(format!(
                        "LengthPreserving: len(f(xs))={} != len(xs)={}",
                        ys.len(),
                        xs.len()
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The FULL semantic gate: an operation's output CONTRACT (decidable output
/// property) AND its metamorphic LAWS, both checked on fresh inputs. This is the
/// single call the trust gate and the flywheel gate use — the widest
/// solver-independent, proof-carrying refutation available for an op.
pub fn check_op_semantics(
    code: &str,
    fn_name: &str,
    op_lemma: &str,
    sample_inputs: &[Value],
) -> Result<(), String> {
    check_op_contract(code, fn_name, op_lemma, sample_inputs)?;
    check_op_metamorphic(code, fn_name, op_lemma, sample_inputs)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_OK: &str = "fn array_max(arr: [i64]) -> i64 {\n    best := arr[0];\n    for item in arr {\n        if item > best {\n            best = item;\n        }\n    }\n    return best;\n}\n";
    // Overfit: returns the FIRST element. Matches any example where arr[0] is the
    // max, but is not actually the maximum.
    const MAX_FAKE: &str = "fn array_max(arr: [i64]) -> i64 {\n    return arr[0];\n}\n";

    const REF_MAX: &str = "fn array_max(arr: [i64]) -> i64 {\n    best := arr[0];\n    for item in arr {\n        if item > best {\n            best = item;\n        }\n    }\n    return best;\n}\n";

    #[test]
    fn differential_confirm_is_vs_reference() {
        // A correct candidate agrees with the reference on every fresh input; a subtle
        // overfit (returns arr[0]) disagrees and stays tentative — no reference oracle needed
        // beyond the resolved op's own verified impl.
        assert!(differential_matches(MAX_OK, "array_max", REF_MAX, 32, 0x00ac_1e50));
        assert!(!differential_matches(MAX_FAKE, "array_max", REF_MAX, 32, 0x00ac_1e50));
    }

    #[test]
    fn programs_distinguishable_detects_observable_difference() {
        // Two identical impls are NOT distinguishable (agree on every fresh input).
        assert!(!programs_distinguishable(MAX_OK, "array_max", REF_MAX, "array_max"));
        // max vs first-element: they disagree on some fresh array -> distinguishable.
        assert!(programs_distinguishable(MAX_OK, "array_max", MAX_FAKE, "array_max"));
    }

    #[test]
    fn sig_param_types_parses_shapes() {
        assert_eq!(sig_param_types(REF_MAX), Some(vec!["[i64]".to_string()]));
        assert_eq!(
            sig_param_types("fn f(a: i64, b: [i64]) -> i64 { return a; }"),
            Some(vec!["i64".to_string(), "[i64]".to_string()])
        );
        assert_eq!(sig_param_types("fn g() -> i64 { return 0; }"), Some(vec![]));
    }

    const ABS_OK: &str = "fn absolute(x: i64) -> i64 {\n    if x < 0 {\n        return 0 - x;\n    }\n    return x;\n}\n";
    // Overfit: identity. Matches non-negative examples, fails on negatives.
    const ABS_FAKE: &str = "fn absolute(x: i64) -> i64 {\n    return x;\n}\n";

    const SORT_OK: &str = "fn sort_ascending(arr: [i64]) -> [i64] {\n    out := arr;\n    n := len(out);\n    for i in 0..n {\n        for j in 0..n {\n            if out[i] < out[j] {\n                tmp := out[i];\n                out[i] = out[j];\n                out[j] = tmp;\n            }\n        }\n    }\n    return out;\n}\n";

    #[test]
    fn op_lemmas_map_to_contracts() {
        assert_eq!(Property::for_op("maximum"), Some(Property::IsMax));
        assert_eq!(Property::for_op("MIN"), Some(Property::IsMin));
        assert_eq!(Property::for_op("magnitude"), Some(Property::IsNonNegative));
        assert_eq!(Property::for_op("sort"), Some(Property::IsSortedAscending));
        assert_eq!(Property::for_op("frobnicate"), None);
    }

    fn arr_input() -> Vec<Value> {
        vec![Value::int_array(&[1, 2, 3])]
    }
    fn scalar_input() -> Vec<Value> {
        vec![Value::Int(-4)]
    }

    #[test]
    fn correct_max_passes_fake_max_is_caught() {
        // The real thing honors the contract on every fresh input.
        assert!(check_op_contract(MAX_OK, "array_max", "max", &arr_input()).is_ok());
        // The overfit that fits arr[0]-is-max examples is exposed on fresh inputs.
        let caught = check_op_contract(MAX_FAKE, "array_max", "max", &arr_input());
        assert!(caught.is_err(), "fake max must be rejected: {caught:?}");
        assert!(caught.unwrap_err().contains("IsMax"));
    }

    #[test]
    fn correct_abs_passes_identity_is_caught() {
        assert!(check_op_contract(ABS_OK, "absolute", "abs", &scalar_input()).is_ok());
        let caught = check_op_contract(ABS_FAKE, "absolute", "abs", &scalar_input());
        assert!(caught.is_err(), "identity-as-abs must be rejected: {caught:?}");
        assert!(caught.unwrap_err().contains("IsNonNegative"));
    }

    #[test]
    fn correct_sort_passes() {
        assert!(
            check_op_contract(SORT_OK, "sort_ascending", "sort", &arr_input()).is_ok(),
            "a real sort honors the ascending+permutation contract"
        );
    }

    const SUM_OK: &str = "fn array_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        total = total + item;\n    }\n    return total;\n}\n";
    // Overfit: returns first element. Matches single-element or crafted examples.
    const SUM_FAKE: &str = "fn array_sum(arr: [i64]) -> i64 {\n    return arr[0];\n}\n";

    #[test]
    fn correct_sum_passes_fake_is_caught() {
        assert!(check_op_contract(SUM_OK, "array_sum", "sum", &arr_input()).is_ok());
        let caught = check_op_contract(SUM_FAKE, "array_sum", "sum", &arr_input());
        assert!(caught.is_err(), "fake sum must be rejected: {caught:?}");
        assert!(caught.unwrap_err().contains("IsSum"));
    }

    #[test]
    fn no_contract_op_is_not_applicable() {
        // An op with no cheap property never rejects on its own.
        assert!(check_op_contract(MAX_OK, "array_max", "some_unknown_op", &arr_input()).is_ok());
    }

    // ── Metamorphic laws ─────────────────────────────────────────────────────
    const NEG_OK: &str = "fn negate(x: i64) -> i64 {\n    return 0 - x;\n}\n";
    // Overfit: |x| passes as an "involution" on non-negative inputs only.
    const NEG_FAKE_ABS: &str = "fn negate(x: i64) -> i64 {\n    if x < 0 {\n        return 0 - x;\n    }\n    return x;\n}\n";
    const ADD_OK: &str = "fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n";
    const SUB_FAKE: &str = "fn add(a: i64, b: i64) -> i64 {\n    return a - b;\n}\n";

    #[test]
    fn metamorphic_laws_map_to_ops() {
        use MetamorphicRelation::*;
        assert_eq!(MetamorphicRelation::for_op("sort"), vec![Idempotent, OrderInvariant, LengthPreserving]);
        assert_eq!(MetamorphicRelation::for_op("reverse"), vec![Involutive, LengthPreserving]);
        assert_eq!(MetamorphicRelation::for_op("add"), vec![Commutative]);
        assert_eq!(MetamorphicRelation::for_op("sum"), vec![OrderInvariant]);
        assert!(MetamorphicRelation::for_op("frobnicate").is_empty());
    }

    #[test]
    fn sort_obeys_idempotence_order_invariance_length() {
        // The real sort honors all three of its laws on fresh inputs.
        assert!(check_op_metamorphic(SORT_OK, "sort_ascending", "sort", &arr_input()).is_ok());
    }

    #[test]
    fn reverse_is_an_involution_but_identity_is_not() {
        const REV_OK: &str = "fn reverse_array(arr: [i64]) -> [i64] {\n    out := arr;\n    n := len(arr);\n    for i in 0..n {\n        out[i] = arr[n - 1 - i];\n    }\n    return out;\n}\n";
        assert!(check_op_metamorphic(REV_OK, "reverse_array", "reverse", &arr_input()).is_ok());
        // Identity-as-reverse: f(f(x))==x holds for identity too, BUT it fails the
        // OUTPUT contract (IsReversed) — caught by check_op_semantics.
        const IDENT: &str = "fn reverse_array(arr: [i64]) -> [i64] {\n    return arr;\n}\n";
        let caught = check_op_semantics(IDENT, "reverse_array", "reverse", &arr_input());
        assert!(caught.is_err(), "identity is not a reverse: {caught:?}");
    }

    #[test]
    fn commutativity_passes_add_catches_subtract() {
        assert!(check_op_metamorphic(ADD_OK, "add", "add", &two_scalar_input()).is_ok());
        let caught = check_op_metamorphic(SUB_FAKE, "add", "add", &two_scalar_input());
        assert!(caught.is_err(), "subtraction is not commutative: {caught:?}");
        assert!(caught.unwrap_err().contains("Commutative"));
    }

    #[test]
    fn involution_catches_a_non_involutive_negate() {
        assert!(check_op_metamorphic(NEG_OK, "negate", "negate", &scalar_input()).is_ok());
        // |x| is not an involution: f(f(-3)) = f(3) = 3 != -3.
        let caught = check_op_metamorphic(NEG_FAKE_ABS, "negate", "negate", &scalar_input());
        assert!(caught.is_err(), "abs is not a negation: {caught:?}");
        assert!(caught.unwrap_err().contains("Involutive"));
    }

    fn two_scalar_input() -> Vec<Value> {
        vec![Value::Int(3), Value::Int(7)]
    }

    #[test]
    fn shape_mismatch_skips_rather_than_false_rejects() {
        // A problem named "max" over TWO SCALARS is not array-max; the oracle
        // must not fire (and must not error trying to feed it an array).
        let two_scalars = vec![Value::Int(3), Value::Int(7)];
        assert!(
            check_op_contract(MAX_FAKE, "array_max", "max", &two_scalars).is_ok(),
            "shape mismatch -> not applicable, never a false rejection"
        );
    }
}
