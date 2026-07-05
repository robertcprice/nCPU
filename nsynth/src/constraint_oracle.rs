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
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// Deterministic fresh inputs for a property's input shape. Deterministic (LCG,
/// caller-seeded) so verification is reproducible and never flaky.
pub fn fresh_inputs(prop: &Property, n: usize, seed: u64) -> Vec<Vec<Value>> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if prop.wants_array() {
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

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_OK: &str = "fn array_max(arr: [i64]) -> i64 {\n    best := arr[0];\n    for item in arr {\n        if item > best {\n            best = item;\n        }\n    }\n    return best;\n}\n";
    // Overfit: returns the FIRST element. Matches any example where arr[0] is the
    // max, but is not actually the maximum.
    const MAX_FAKE: &str = "fn array_max(arr: [i64]) -> i64 {\n    return arr[0];\n}\n";

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
