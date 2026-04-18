//! Run a program against synthetic inputs to extract observed I/O examples.
//!
//! First step toward "learn from a program by observing how it behaves" —
//! the binary-corpus pretraining direction. Today the runtime is the Mog
//! interpreter (see [`crate::runtime`]); a future ARM64 emulator hook can
//! plug in here behind the same trait.
//!
//! Usage pattern:
//!   1. [`InputSampler`] generates synthetic inputs (scalar i64 today).
//!   2. [`trace_function`] executes the program for each input via the
//!      runtime and collects `(inputs, output)` tuples.
//!   3. The resulting [`Vec<(Vec<i64>, i64)>`] is directly compatible with
//!      `MetaRecord.io_examples` and `Problem.examples` (after wrapping in
//!      `Value::Int`).
//!
//! No human curation. No hand-picked input ranges per program. The same
//! sampler runs over every program — the variation comes from the seed.

use crate::benchmark::Value;
use crate::runtime::execute_function;

// ─── Input generation ────────────────────────────────────────────────────────

/// Configuration for synthetic input generation.
///
/// All fields default to small ranges that avoid most overflows in toy
/// programs; tune only the seed for diversity across runs.
#[derive(Clone, Debug)]
pub struct InputSampler {
    /// Inclusive lower bound for each generated i64.
    pub min: i64,
    /// Inclusive upper bound for each generated i64.
    pub max: i64,
    /// LCG seed; advance per call to draw distinct values.
    pub seed: u64,
}

impl Default for InputSampler {
    fn default() -> Self {
        Self {
            min: -10,
            max: 20,
            seed: 0xc0ffee,
        }
    }
}

impl InputSampler {
    /// Draw the next i64 in `[min, max]`. Mutates `seed` so successive calls
    /// produce a deterministic, well-spread sequence.
    pub fn next_i64(&mut self) -> i64 {
        self.seed = self
            .seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let span = (self.max as i128 - self.min as i128 + 1) as u128;
        if span == 0 {
            return self.min;
        }
        let raw = (self.seed >> 33) as u128;
        self.min + (raw % span) as i64
    }

    /// Generate one example input vector of `n_args` scalars.
    pub fn next_inputs(&mut self, n_args: usize) -> Vec<i64> {
        (0..n_args).map(|_| self.next_i64()).collect()
    }
}

// ─── Tracing ─────────────────────────────────────────────────────────────────

/// Execute `code`'s `function_name` against `n_eval` synthetic inputs.
///
/// Returns the observed `(inputs, output)` tuples. Inputs that produce a
/// non-integer return, an overflow, or a runtime error are silently skipped
/// (the function is observation-only — caller decides what to do with a
/// short result list).
pub fn trace_function(
    code: &str,
    function_name: &str,
    n_args: usize,
    n_eval: usize,
    sampler: &mut InputSampler,
) -> Vec<(Vec<i64>, i64)> {
    let mut traced = Vec::with_capacity(n_eval);
    for _ in 0..n_eval {
        let inputs = sampler.next_inputs(n_args);
        let benchmark_inputs: Vec<Value> = inputs.iter().map(|i| Value::Int(*i)).collect();
        match execute_function(code, function_name, &benchmark_inputs, "trace") {
            Ok(crate::runtime::Value::Int(out)) if out.abs() < 1_000_000 => {
                traced.push((inputs, out));
            }
            _ => continue,
        }
    }
    traced
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: trace a trivial Mog program and verify the I/O matches the
    /// hand-derived expected behaviour. Pure observation, no synthesis.
    #[test]
    fn trace_double_function() {
        let code = "fn double(a: i64) -> i64 { return a + a; }\n";
        let mut sampler = InputSampler::default();
        let traces = trace_function(code, "double", 1, 8, &mut sampler);
        assert!(!traces.is_empty(), "expected at least one trace");
        for (inputs, out) in &traces {
            assert_eq!(inputs.len(), 1);
            assert_eq!(
                *out,
                inputs[0] * 2,
                "double({}) should be {}",
                inputs[0],
                inputs[0] * 2
            );
        }
    }

    /// Sampler determinism: same seed produces the same sequence.
    #[test]
    fn sampler_is_deterministic() {
        let mut a = InputSampler {
            min: -5,
            max: 5,
            seed: 42,
        };
        let mut b = a.clone();
        let xs: Vec<i64> = (0..16).map(|_| a.next_i64()).collect();
        let ys: Vec<i64> = (0..16).map(|_| b.next_i64()).collect();
        assert_eq!(xs, ys);
        for v in &xs {
            assert!(*v >= -5 && *v <= 5, "out of range: {v}");
        }
    }
}
