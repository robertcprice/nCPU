#!/usr/bin/env bash
# Offline smoke for modules that do NOT need linguigenesis-core.
# Runs standalone harnesses for schema_miner (+ optionally schema_component).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== schema_miner standalone =="
mkdir -p "$TMP/schema_miner/src"
python3 - "$ROOT" "$TMP" <<'PY'
import sys
from pathlib import Path
root, tmp = Path(sys.argv[1]), Path(sys.argv[2])
src = (root / "src/schema_miner.rs").read_text()
(tmp / "schema_miner/Cargo.toml").write_text("""
[package]
name = "schema_miner_test"
version = "0.1.0"
edition = "2021"
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
""")
lib = (
    "pub mod schema_miner {\n"
    + src.replace("crate::benchmark::", "crate::bench::").replace("crate::runtime::", "crate::rt::")
    + "\n}\n"
    + """
pub mod bench {
    #[derive(Clone, Debug, PartialEq)]
    pub enum Value { Int(i64), Bool(bool), Str(String), Array(Vec<Value>), Other }
    #[derive(Clone, Debug)]
    pub struct Example { pub inputs: Vec<Value>, pub expected: Value }
    #[derive(Clone, Debug, Default)]
    pub struct Problem {
        pub name: String,
        pub examples: Vec<Example>,
        pub holdouts: Vec<Example>,
    }
    impl Problem {
        pub fn function_name(&self) -> &str { if self.name.is_empty() { "f" } else { &self.name } }
    }
}
pub mod rt {
    use super::bench::{Example, Problem};
    pub fn code_reproduces_examples(_code: &str, _examples: &[Example]) -> bool { false }
    pub fn verify_problem_code_strict(_problem: &Problem, _code: &str) -> Result<(), String> {
        Err("offline stub: no Mog oracle".into())
    }
}
"""
)
(tmp / "schema_miner/src/lib.rs").write_text(lib)
PY
( cd "$TMP/schema_miner" && cargo test --lib -q )
echo "schema_miner: OK"

echo "== utbus reduce standalone =="
mkdir -p "$TMP/utbus_reduce/src"
cat > "$TMP/utbus_reduce/Cargo.toml" <<'EOF'
[package]
name = "utbus_reduce_test"
version = "0.1.0"
edition = "2021"
EOF
cat > "$TMP/utbus_reduce/src/lib.rs" <<'EOF'
//! Mirrors nsynth UTBUS Reduce::{Sum,Max,Min,Count} + eval_scalar contract
//! so Phase A expand stays checkable without linguigenesis-core.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reduce { Sum, Max, Min, Count, Product, Xor, BitOr, BitAnd }

impl Reduce {
    fn apply(self, arr: &[i64]) -> i64 {
        match self {
            Reduce::Sum => arr.iter().copied().fold(0i64, i64::saturating_add),
            Reduce::Max => arr.iter().copied().max().unwrap_or(0),
            Reduce::Min => arr.iter().copied().min().unwrap_or(0),
            Reduce::Count => arr.len() as i64,
            Reduce::Product => arr.iter().copied().fold(1i64, i64::saturating_mul),
            Reduce::Xor => arr.iter().copied().fold(0i64, |a, b| a ^ b),
            Reduce::BitOr => arr.iter().copied().fold(0i64, |a, b| a | b),
            Reduce::BitAnd => arr.iter().copied().fold(-1i64, |a, b| a & b),
        }
    }
}

#[derive(Clone, Copy)]
enum Pred { All, Positive, GtK }

fn transform(pred: Pred, input: &[i64], k: Option<i64>) -> Vec<i64> {
    input.iter().copied().filter(|&x| match pred {
        Pred::All => true,
        Pred::Positive => x > 0,
        Pred::GtK => k.map(|k| x > k).unwrap_or(false),
    }).collect()
}

fn eval_scalar(pred: Pred, reduce: Reduce, input: &[i64], k: Option<i64>) -> i64 {
    reduce.apply(&transform(pred, input, k))
}

/// Cheapest-first matching: try all example matches (Sum may agree with Count
/// on examples; holdouts would disambiguate in the real verifier).
fn synthesize(examples: &[(Vec<i64>, i64)]) -> Option<(Pred, Reduce)> {
    let cands = [
        (Pred::All, Reduce::Sum),
        (Pred::All, Reduce::Count),
        (Pred::All, Reduce::Max),
        (Pred::All, Reduce::Min),
        (Pred::All, Reduce::Product),
        (Pred::All, Reduce::Xor),
        (Pred::All, Reduce::BitOr),
        (Pred::All, Reduce::BitAnd),
        (Pred::Positive, Reduce::Count),
        (Pred::Positive, Reduce::Sum),
    ];
    cands.into_iter().find(|&(pred, reduce)| {
        examples.iter().all(|(xs, y)| eval_scalar(pred, reduce, xs, None) == *y)
    })
}

fn synthesize_k(examples: &[(Vec<i64>, i64, i64)]) -> Option<(Pred, Reduce)> {
    let cands = [
        (Pred::GtK, Reduce::Count),
        (Pred::GtK, Reduce::Sum),
        (Pred::All, Reduce::Count),
    ];
    cands.into_iter().find(|&(pred, reduce)| {
        examples
            .iter()
            .all(|(xs, k, y)| eval_scalar(pred, reduce, xs, Some(*k)) == *y)
    })
}

fn dual_range(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let lo = *arr.iter().min()?;
    let hi = *arr.iter().max()?;
    Some(hi - lo)
}

fn dual_second_max(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut first = arr[0];
    let mut second = arr[0];
    for &item in arr {
        if item > first {
            second = first;
            first = item;
        } else if item > second {
            second = item;
        }
    }
    Some(second)
}

fn dual_stock_profit(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut min_price = arr[0];
    let mut best = 0i64;
    for &p in arr {
        if p < min_price { min_price = p; }
        let profit = p - min_price;
        if profit > best { best = profit; }
    }
    Some(best)
}

fn dual_prefix_max_sum(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut running_max = arr[0];
    let mut total = 0i64;
    for &x in arr {
        if x > running_max { running_max = x; }
        total += running_max;
    }
    Some(total)
}

fn dual_max_subarray(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut current = 0i64;
    let mut best = arr[0];
    for &item in arr {
        current = if current > 0 { current + item } else { item };
        if current > best { best = current; }
    }
    Some(best)
}

fn dual_min_subarray(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut current = 0i64;
    let mut best = arr[0];
    for &item in arr {
        current = if current < 0 { current + item } else { item };
        if current < best { best = current; }
    }
    Some(best)
}

fn dual_median(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut s = arr.to_vec();
    s.sort_unstable();
    Some(s[s.len() / 2])
}

fn i64_gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs(); b = b.abs();
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

fn dual_gcd_all(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    Some(arr.iter().copied().fold(arr[0].abs(), i64_gcd))
}

fn dual_lcm_all(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut l = arr[0].abs();
    for &x in &arr[1..] {
        let g = i64_gcd(l, x);
        if g == 0 { return Some(0); }
        l = (l / g).checked_mul(x.abs())?;
    }
    Some(l)
}

fn dual_mean_trunc(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    Some(arr.iter().sum::<i64>() / (arr.len() as i64))
}

fn dual_sum_squares(arr: &[i64]) -> i64 {
    arr.iter().map(|&x| x.saturating_mul(x)).fold(0i64, i64::saturating_add)
}

fn dual_abs_sum(arr: &[i64]) -> i64 {
    arr.iter().map(|&x| x.abs()).fold(0i64, i64::saturating_add)
}

fn dual_max_abs(arr: &[i64]) -> i64 {
    arr.iter().map(|&x| x.abs()).max().unwrap_or(0)
}

fn dual_min_positive(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for &x in arr {
        if x > 0 && (!found || x < best) {
            best = x;
            found = true;
        }
    }
    best
}

fn dual_count_negatives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0).count() as i64
}

fn dual_count_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).count() as i64
}

fn dual_sum_positives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0).copied().sum()
}

fn dual_sum_negatives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0).copied().sum()
}

fn dual_count_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).count() as i64
}

fn dual_any_zero(arr: &[i64]) -> i64 {
    if arr.iter().any(|&x| x == 0) { 1 } else { 0 }
}

fn dual_all_positive(arr: &[i64]) -> i64 {
    if arr.iter().all(|&x| x > 0) { 1 } else { 0 }
}

fn dual_all_negative(arr: &[i64]) -> i64 {
    if arr.iter().all(|&x| x < 0) { 1 } else { 0 }
}

fn dual_count_zeros(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x == 0).count() as i64
}

fn dual_has_duplicate(arr: &[i64]) -> i64 {
    for i in 0..arr.len() {
        if arr[..i].contains(&arr[i]) { return 1; }
    }
    0
}

fn dual_max_negative(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for &x in arr {
        if x < 0 && (!found || x > best) {
            best = x;
            found = true;
        }
    }
    best
}

fn dual_sum_even_values(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).copied().sum()
}

fn dual_sum_odd_values(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).copied().sum()
}

fn dual_all_non_negative(arr: &[i64]) -> i64 {
    if arr.iter().all(|&x| x >= 0) { 1 } else { 0 }
}

fn dual_count_non_zeros(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x != 0).count() as i64
}

fn dual_alternating_sum(arr: &[i64]) -> i64 {
    let mut total = 0i64;
    for (i, &x) in arr.iter().enumerate() {
        if i % 2 == 0 { total += x; } else { total -= x; }
    }
    total
}

fn dual_product_positives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0).copied().fold(1, i64::saturating_mul)
}

fn dual_mean_abs_trunc(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    Some(arr.iter().map(|&x| x.abs()).sum::<i64>() / (arr.len() as i64))
}

fn dual_first_positive(arr: &[i64]) -> i64 {
    for &x in arr { if x > 0 { return x; } }
    0
}

fn dual_last_positive(arr: &[i64]) -> i64 {
    for &x in arr.iter().rev() { if x > 0 { return x; } }
    0
}

fn dual_first_negative(arr: &[i64]) -> i64 {
    for &x in arr { if x < 0 { return x; } }
    0
}

fn dual_last_negative(arr: &[i64]) -> i64 {
    for &x in arr.iter().rev() { if x < 0 { return x; } }
    0
}

fn dual_max_positive(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for &x in arr {
        if x > 0 && (!found || x > best) {
            best = x;
            found = true;
        }
    }
    best
}

fn dual_min_negative(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for &x in arr {
        if x < 0 && (!found || x < best) {
            best = x;
            found = true;
        }
    }
    best
}

fn dual_product_negatives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0).copied().fold(1, i64::saturating_mul)
}

fn dual_sum_cubes(arr: &[i64]) -> i64 {
    arr.iter().map(|&x| x.saturating_mul(x).saturating_mul(x)).fold(0, i64::saturating_add)
}

fn dual_count_gt_mean(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mean = arr.iter().sum::<i64>() / (arr.len() as i64);
    Some(arr.iter().filter(|&&x| x > mean).count() as i64)
}

fn dual_count_lt_mean(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mean = arr.iter().sum::<i64>() / (arr.len() as i64);
    Some(arr.iter().filter(|&&x| x < mean).count() as i64)
}

fn dual_is_palindrome(arr: &[i64]) -> i64 {
    let n = arr.len();
    if (0..n/2).all(|i| arr[i] == arr[n-1-i]) { 1 } else { 0 }
}

fn dual_product_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).copied().fold(1, i64::saturating_mul)
}

fn dual_product_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).copied().fold(1, i64::saturating_mul)
}

fn dual_all_non_positive(arr: &[i64]) -> i64 {
    if arr.iter().all(|&x| x <= 0) { 1 } else { 0 }
}

fn dual_dot_index(arr: &[i64]) -> i64 {
    arr.iter().enumerate().map(|(i,&v)| (i as i64)*v).sum()
}

fn dual_sum_sq_diff_mean(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mean = arr.iter().sum::<i64>() / (arr.len() as i64);
    Some(arr.iter().map(|&x| { let d = x - mean; d*d }).sum())
}

fn dual_any_non_zero(arr: &[i64]) -> i64 {
    if arr.iter().any(|&x| x != 0) { 1 } else { 0 }
}

fn dual_xor_all(arr: &[i64]) -> i64 {
    arr.iter().copied().fold(0, |a, b| a ^ b)
}

fn dual_product_non_zeros(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x != 0).fold(1i64, |a, &b| a.saturating_mul(b))
}

fn dual_or_all(arr: &[i64]) -> i64 {
    arr.iter().copied().fold(0, |a, b| a | b)
}

fn dual_and_all(arr: &[i64]) -> i64 {
    arr.iter().copied().fold(-1i64, |a, b| a & b)
}

fn dual_count_eq_mean(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let sum: i64 = arr.iter().copied().sum();
    let mean = sum / (arr.len() as i64);
    Some(arr.iter().filter(|&&x| x == mean).count() as i64)
}

fn dual_product_abs(arr: &[i64]) -> i64 {
    arr.iter().map(|&x| x.abs()).fold(1i64, i64::saturating_mul)
}

fn dual_count_non_negatives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x >= 0).count() as i64
}

fn dual_count_positives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0).count() as i64
}

fn dual_max_even_value(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for &x in arr {
        if x % 2 == 0 {
            if !found || x > best { best = x; found = true; }
        }
    }
    best
}

fn dual_max_odd_value(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for &x in arr {
        if x % 2 != 0 {
            if !found || x > best { best = x; found = true; }
        }
    }
    best
}

fn dual_min_even_value(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for &x in arr {
        if x % 2 == 0 {
            if !found || x < best { best = x; found = true; }
        }
    }
    best
}

fn dual_min_odd_value(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for &x in arr {
        if x % 2 != 0 {
            if !found || x < best { best = x; found = true; }
        }
    }
    best
}

fn dual_abs_range(arr: &[i64]) -> i64 {
    if arr.is_empty() { return 0; }
    let mut lo = arr[0].abs();
    let mut hi = lo;
    for &x in arr.iter().skip(1) {
        let a = x.abs();
        if a < lo { lo = a; }
        if a > hi { hi = a; }
    }
    hi - lo
}

fn dual_sum_non_negatives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x >= 0).sum()
}

fn dual_sum_non_positives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x <= 0).sum()
}

fn dual_count_non_positives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x <= 0).count() as i64
}

fn dual_product_non_positives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x <= 0).fold(1i64, |a, &b| a.saturating_mul(b))
}

fn dual_sum_abs_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.abs()).sum()
}

fn dual_sum_abs_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.abs()).sum()
}

fn dual_product_abs_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.abs()).fold(1i64, i64::saturating_mul)
}

fn dual_product_abs_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.abs()).fold(1i64, i64::saturating_mul)
}

fn dual_xor_abs_all(arr: &[i64]) -> i64 {
    arr.iter().map(|&x| x.abs()).fold(0i64, |a, b| a ^ b)
}

fn dual_and_abs_all(arr: &[i64]) -> i64 {
    arr.iter().map(|&x| x.abs()).fold(-1i64, |a, b| a & b)
}

fn dual_or_abs_all(arr: &[i64]) -> i64 {
    arr.iter().map(|&x| x.abs()).fold(0i64, |a, b| a | b)
}

fn dual_xor_abs_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.abs()).fold(0i64, |a, b| a ^ b)
}

fn dual_xor_abs_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.abs()).fold(0i64, |a, b| a ^ b)
}

fn dual_and_abs_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.abs()).fold(-1i64, |a, b| a & b)
}

fn dual_and_abs_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.abs()).fold(-1i64, |a, b| a & b)
}

fn dual_or_abs_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.abs()).fold(0i64, |a, b| a | b)
}

fn dual_or_abs_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.abs()).fold(0i64, |a, b| a | b)
}

fn dual_sum_squares_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.saturating_mul(x)).fold(0i64, i64::saturating_add)
}

fn dual_sum_squares_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.saturating_mul(x)).fold(0i64, i64::saturating_add)
}

fn dual_sum_cubes_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.saturating_mul(x).saturating_mul(x)).fold(0i64, i64::saturating_add)
}

fn dual_sum_cubes_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.saturating_mul(x).saturating_mul(x)).fold(0i64, i64::saturating_add)
}

fn dual_product_squares_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.saturating_mul(x)).fold(1i64, i64::saturating_mul)
}

fn dual_product_squares_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.saturating_mul(x)).fold(1i64, i64::saturating_mul)
}

fn dual_max_abs_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.abs()).max().unwrap_or(0)
}

fn dual_max_abs_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.abs()).max().unwrap_or(0)
}

fn dual_min_abs_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.abs()).min().unwrap_or(0)
}

fn dual_min_abs_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.abs()).min().unwrap_or(0)
}

fn dual_count_nonzero_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x != 0 && x % 2 == 0).count() as i64
}

fn dual_count_nonzero_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x != 0 && x % 2 != 0).count() as i64
}

fn dual_sum_nonzero_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x != 0 && x % 2 == 0).copied().fold(0i64, i64::saturating_add)
}

fn dual_sum_nonzero_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x != 0 && x % 2 != 0).copied().fold(0i64, i64::saturating_add)
}

fn dual_product_nonzero_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x != 0 && x % 2 == 0).copied().fold(1i64, i64::saturating_mul)
}

fn dual_product_nonzero_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x != 0 && x % 2 != 0).copied().fold(1i64, i64::saturating_mul)
}

fn dual_mean_abs_evens_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.abs()).collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 }
}

fn dual_mean_abs_odds_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.abs()).collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 }
}

fn dual_gcd_abs_evens(arr: &[i64]) -> i64 {
    let mut g: Option<i64> = None;
    for &x in arr {
        if x % 2 == 0 {
            let a = x.abs();
            g = Some(match g { None => a, Some(prev) => {
                let (mut aa, mut bb) = (prev, a);
                while bb != 0 { let t = bb; bb = aa % bb; aa = t; }
                aa
            }});
        }
    }
    g.unwrap_or(0)
}

fn dual_gcd_abs_odds(arr: &[i64]) -> i64 {
    let mut g: Option<i64> = None;
    for &x in arr {
        if x % 2 != 0 {
            let a = x.abs();
            g = Some(match g { None => a, Some(prev) => {
                let (mut aa, mut bb) = (prev, a);
                while bb != 0 { let t = bb; bb = aa % bb; aa = t; }
                aa
            }});
        }
    }
    g.unwrap_or(0)
}

fn dual_lcm_abs_evens(arr: &[i64]) -> i64 {
    let mut l: Option<i64> = None;
    for &x in arr {
        if x % 2 == 0 {
            let a = x.abs();
            l = Some(match l {
                None => a,
                Some(prev) => {
                    let g = i64_gcd(prev, a);
                    if g == 0 { 0 } else { (prev / g).saturating_mul(a) }
                }
            });
        }
    }
    l.unwrap_or(0)
}

fn dual_lcm_abs_odds(arr: &[i64]) -> i64 {
    let mut l: Option<i64> = None;
    for &x in arr {
        if x % 2 != 0 {
            let a = x.abs();
            l = Some(match l {
                None => a,
                Some(prev) => {
                    let g = i64_gcd(prev, a);
                    if g == 0 { 0 } else { (prev / g).saturating_mul(a) }
                }
            });
        }
    }
    l.unwrap_or(0)
}

fn dual_product_cubes_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| x.saturating_mul(x).saturating_mul(x)).fold(1i64, i64::saturating_mul)
}

fn dual_product_cubes_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| x.saturating_mul(x).saturating_mul(x)).fold(1i64, i64::saturating_mul)
}

fn dual_sum_abs_cubes_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| { let a = x.abs(); a.saturating_mul(a).saturating_mul(a) }).fold(0i64, i64::saturating_add)
}

fn dual_sum_abs_cubes_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| { let a = x.abs(); a.saturating_mul(a).saturating_mul(a) }).fold(0i64, i64::saturating_add)
}

fn dual_product_abs_cubes_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| { let a = x.abs(); a.saturating_mul(a).saturating_mul(a) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_abs_cubes_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| { let a = x.abs(); a.saturating_mul(a).saturating_mul(a) }).fold(1i64, i64::saturating_mul)
}

fn dual_sum_abs_squares_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| { let a = x.abs(); a.saturating_mul(a) }).fold(0i64, i64::saturating_add)
}

fn dual_sum_abs_squares_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| { let a = x.abs(); a.saturating_mul(a) }).fold(0i64, i64::saturating_add)
}

fn dual_product_abs_squares_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0).map(|&x| { let a = x.abs(); a.saturating_mul(a) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_abs_squares_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0).map(|&x| { let a = x.abs(); a.saturating_mul(a) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_abs_squares_evens_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().filter(|&&x| x % 2 == 0).map(|&x| { let a = x.abs(); a.saturating_mul(a) }).collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 }
}

fn dual_mean_abs_squares_odds_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().filter(|&&x| x % 2 != 0).map(|&x| { let a = x.abs(); a.saturating_mul(a) }).collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 }
}

fn dual_count_positive_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0 && x % 2 == 0).count() as i64
}

fn dual_count_positive_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0 && x % 2 != 0).count() as i64
}

fn dual_count_negative_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0 && x % 2 == 0).count() as i64
}

fn dual_count_negative_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0 && x % 2 != 0).count() as i64
}

fn dual_sum_positive_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0 && x % 2 == 0).copied().sum()
}

fn dual_sum_positive_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0 && x % 2 != 0).copied().sum()
}

fn dual_sum_negative_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0 && x % 2 == 0).copied().sum()
}

fn dual_sum_negative_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0 && x % 2 != 0).copied().sum()
}

fn dual_product_positive_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0 && x % 2 == 0).copied().fold(1i64, i64::saturating_mul)
}

fn dual_product_positive_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0 && x % 2 != 0).copied().fold(1i64, i64::saturating_mul)
}

fn dual_product_negative_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0 && x % 2 == 0).copied().fold(1i64, i64::saturating_mul)
}

fn dual_product_negative_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0 && x % 2 != 0).copied().fold(1i64, i64::saturating_mul)
}

fn dual_max_positive_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0 && x % 2 == 0).copied().max().unwrap_or(0)
}

fn dual_max_positive_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0 && x % 2 != 0).copied().max().unwrap_or(0)
}

fn dual_min_positive_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0 && x % 2 == 0).copied().min().unwrap_or(0)
}

fn dual_min_positive_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x > 0 && x % 2 != 0).copied().min().unwrap_or(0)
}

fn dual_max_negative_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0 && x % 2 == 0).copied().max().unwrap_or(0)
}

fn dual_max_negative_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0 && x % 2 != 0).copied().max().unwrap_or(0)
}

fn dual_min_negative_evens(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0 && x % 2 == 0).copied().min().unwrap_or(0)
}

fn dual_min_negative_odds(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x < 0 && x % 2 != 0).copied().min().unwrap_or(0)
}

fn dual_mean_positive_evens_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().filter(|&&x| x > 0 && x % 2 == 0).copied().collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 }
}

fn dual_mean_positive_odds_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().filter(|&&x| x > 0 && x % 2 != 0).copied().collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 }
}

fn dual_mean_negative_evens_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().filter(|&&x| x < 0 && x % 2 == 0).copied().collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 }
}

fn dual_mean_negative_odds_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().filter(|&&x| x < 0 && x % 2 != 0).copied().collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 }
}

fn dual_all_even_positive(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 == 0).all(|&x| x > 0) { 1 } else { 0 }
}

fn dual_all_odd_positive(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 != 0).all(|&x| x > 0) { 1 } else { 0 }
}

fn dual_all_even_negative(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 == 0).all(|&x| x < 0) { 1 } else { 0 }
}

fn dual_all_odd_negative(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 != 0).all(|&x| x < 0) { 1 } else { 0 }
}

fn dual_any_even_positive(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 == 0).any(|&x| x > 0) { 1 } else { 0 }
}

fn dual_any_odd_positive(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 != 0).any(|&x| x > 0) { 1 } else { 0 }
}

fn dual_any_even_negative(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 == 0).any(|&x| x < 0) { 1 } else { 0 }
}

fn dual_any_odd_negative(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 != 0).any(|&x| x < 0) { 1 } else { 0 }
}

fn dual_any_even_non_zero(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 == 0).any(|&x| x != 0) { 1 } else { 0 }
}

fn dual_any_odd_non_zero(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 != 0).any(|&x| x != 0) { 1 } else { 0 }
}

fn dual_all_even_non_zero(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 == 0).all(|&x| x != 0) { 1 } else { 0 }
}

fn dual_all_odd_non_zero(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 != 0).all(|&x| x != 0) { 1 } else { 0 }
}

fn dual_all_even_non_negative(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 == 0).all(|&x| x >= 0) { 1 } else { 0 }
}

fn dual_all_odd_non_negative(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 != 0).all(|&x| x >= 0) { 1 } else { 0 }
}

fn dual_all_even_non_positive(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 == 0).all(|&x| x <= 0) { 1 } else { 0 }
}

fn dual_all_odd_non_positive(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 != 0).all(|&x| x <= 0) { 1 } else { 0 }
}

fn dual_any_even_non_negative(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 == 0).any(|&x| x >= 0) { 1 } else { 0 }
}

fn dual_any_odd_non_negative(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 != 0).any(|&x| x >= 0) { 1 } else { 0 }
}

fn dual_any_even_non_positive(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 == 0).any(|&x| x <= 0) { 1 } else { 0 }
}

fn dual_any_odd_non_positive(arr: &[i64]) -> i64 {
    if arr.iter().filter(|&&x| x % 2 != 0).any(|&x| x <= 0) { 1 } else { 0 }
}

fn dual_max_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).copied().max().unwrap_or(0)
}

fn dual_max_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).copied().max().unwrap_or(0)
}

fn dual_min_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).copied().min().unwrap_or(0)
}

fn dual_min_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).copied().min().unwrap_or(0)
}

fn dual_mean_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_xor_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).fold(0i64, |a, &b| a ^ b)
}

fn dual_xor_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).fold(0i64, |a, &b| a ^ b)
}

fn dual_or_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).fold(0i64, |a, &b| a | b)
}

fn dual_or_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).fold(0i64, |a, &b| a | b)
}

fn dual_and_even_non_zero(arr: &[i64]) -> i64 {
    let mut acc: Option<i64> = None;
    for &x in arr {
        if x % 2 == 0 && x != 0 {
            acc = Some(acc.map_or(x, |a| a & x));
        }
    }
    acc.unwrap_or(-1)
}

fn dual_and_odd_non_zero(arr: &[i64]) -> i64 {
    let mut acc: Option<i64> = None;
    for &x in arr {
        if x % 2 != 0 && x != 0 {
            acc = Some(acc.map_or(x, |a| a & x));
        }
    }
    acc.unwrap_or(-1)
}

fn dual_sum_abs_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| x.abs()).sum()
}

fn dual_sum_abs_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| x.abs()).sum()
}

fn dual_product_abs_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| x.abs()).fold(1i64, i64::saturating_mul)
}

fn dual_product_abs_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| x.abs()).fold(1i64, i64::saturating_mul)
}

fn smoke_gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs(); b = b.abs();
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

fn dual_gcd_abs_even_non_zero(arr: &[i64]) -> i64 {
    let mut g: Option<i64> = None;
    for &x in arr {
        if x % 2 == 0 && x != 0 {
            let a = x.abs();
            g = Some(match g { None => a, Some(g) => smoke_gcd(g, a) });
        }
    }
    g.unwrap_or(0)
}

fn dual_gcd_abs_odd_non_zero(arr: &[i64]) -> i64 {
    let mut g: Option<i64> = None;
    for &x in arr {
        if x % 2 != 0 && x != 0 {
            let a = x.abs();
            g = Some(match g { None => a, Some(g) => smoke_gcd(g, a) });
        }
    }
    g.unwrap_or(0)
}

fn smoke_lcm(a: i64, b: i64) -> i64 {
    let a = a.abs(); let b = b.abs();
    if a == 0 || b == 0 { return 0; }
    (a / smoke_gcd(a, b)) * b
}

fn dual_lcm_abs_even_non_zero(arr: &[i64]) -> i64 {
    let mut l: Option<i64> = None;
    for &x in arr {
        if x % 2 == 0 && x != 0 {
            let a = x.abs();
            l = Some(match l { None => a, Some(l) => smoke_lcm(l, a) });
        }
    }
    l.unwrap_or(1)
}

fn dual_lcm_abs_odd_non_zero(arr: &[i64]) -> i64 {
    let mut l: Option<i64> = None;
    for &x in arr {
        if x % 2 != 0 && x != 0 {
            let a = x.abs();
            l = Some(match l { None => a, Some(l) => smoke_lcm(l, a) });
        }
    }
    l.unwrap_or(1)
}

fn dual_mean_abs_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| x.abs()).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_abs_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| x.abs()).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_max_abs_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| x.abs()).max().unwrap_or(0)
}

fn dual_max_abs_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| x.abs()).max().unwrap_or(0)
}

fn dual_min_abs_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| x.abs()).min().unwrap_or(0)
}

fn dual_min_abs_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| x.abs()).min().unwrap_or(0)
}

fn dual_sum_squares_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| x.saturating_mul(x)).sum()
}

fn dual_sum_squares_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| x.saturating_mul(x)).sum()
}

fn dual_product_squares_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| x.saturating_mul(x)).fold(1i64, i64::saturating_mul)
}

fn dual_product_squares_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| x.saturating_mul(x)).fold(1i64, i64::saturating_mul)
}

fn dual_sum_cubes_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| x.saturating_mul(x).saturating_mul(x)).sum()
}

fn dual_sum_cubes_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| x.saturating_mul(x).saturating_mul(x)).sum()
}

fn dual_product_cubes_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| x.saturating_mul(x).saturating_mul(x)).fold(1i64, i64::saturating_mul)
}

fn dual_product_cubes_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| x.saturating_mul(x).saturating_mul(x)).fold(1i64, i64::saturating_mul)
}

fn dual_sum_fourth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s) }).sum()
}

fn dual_sum_fourth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s) }).sum()
}

fn dual_product_fourth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_fourth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_fourth_powers_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| { let s = x.saturating_mul(x); s.saturating_mul(s) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_fourth_powers_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| { let s = x.saturating_mul(x); s.saturating_mul(s) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_sum_fifth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(x) }).sum()
}

fn dual_sum_fifth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(x) }).sum()
}

fn dual_product_fifth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(x) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_fifth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(x) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_fifth_powers_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(x) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_fifth_powers_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(x) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_sum_sixth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s) }).sum()
}

fn dual_sum_sixth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s) }).sum()
}

fn dual_product_sixth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_sixth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_sixth_powers_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_sixth_powers_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_sum_seventh_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s).saturating_mul(x) }).sum()
}

fn dual_sum_seventh_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s).saturating_mul(x) }).sum()
}

fn dual_product_seventh_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s).saturating_mul(x) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_seventh_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s).saturating_mul(x) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_seventh_powers_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s).saturating_mul(x) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_seventh_powers_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| { let s = x.saturating_mul(x); s.saturating_mul(s).saturating_mul(s).saturating_mul(x) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_sum_eighth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q) }).sum()
}

fn dual_sum_eighth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q) }).sum()
}

fn dual_product_eighth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_eighth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_eighth_powers_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_eighth_powers_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_sum_ninth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(x) }).sum()
}

fn dual_sum_ninth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(x) }).sum()
}

fn dual_product_ninth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(x) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_ninth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(x) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_ninth_powers_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(x) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_ninth_powers_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(x) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_sum_tenth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s) }).sum()
}

fn dual_sum_tenth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s) }).sum()
}

fn dual_product_tenth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_tenth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_tenth_powers_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_tenth_powers_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_sum_eleventh_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(x) }).sum()
}

fn dual_sum_eleventh_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(x) }).sum()
}

fn dual_product_eleventh_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(x) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_eleventh_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(x) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_eleventh_powers_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(x) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_eleventh_powers_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(x) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_sum_twelfth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s) }).sum()
}

fn dual_sum_twelfth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s) }).sum()
}

fn dual_product_twelfth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_twelfth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_twelfth_powers_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_twelfth_powers_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_sum_thirteenth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s).saturating_mul(x) }).sum()
}

fn dual_sum_thirteenth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s).saturating_mul(x) }).sum()
}

fn dual_product_thirteenth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s).saturating_mul(x) }).fold(1i64, i64::saturating_mul)
}

fn dual_product_thirteenth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s).saturating_mul(x) }).fold(1i64, i64::saturating_mul)
}

fn dual_mean_thirteenth_powers_even_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 == 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s).saturating_mul(x) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_mean_thirteenth_powers_odd_non_zero_trunc(arr: &[i64]) -> i64 {
    let xs: Vec<i64> = arr.iter().copied().filter(|&x| x % 2 != 0 && x != 0).map(|x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s).saturating_mul(x) }).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn dual_sum_fourteenth_powers_even_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 == 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s).saturating_mul(s) }).sum()
}

fn dual_sum_fourteenth_powers_odd_non_zero(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x % 2 != 0 && x != 0).map(|&x| { let s = x.saturating_mul(x); let q = s.saturating_mul(s); q.saturating_mul(q).saturating_mul(s).saturating_mul(s).saturating_mul(s) }).sum()
}

fn dual_min_abs(arr: &[i64]) -> i64 {
    arr.iter().map(|&x| x.abs()).min().unwrap_or(0)
}

fn dual_product_non_negatives(arr: &[i64]) -> i64 {
    arr.iter().filter(|&&x| x >= 0).fold(1i64, |a, &b| a.saturating_mul(b))
}

fn dual_len(arr: &[i64]) -> i64 { arr.len() as i64 }
fn dual_is_empty(arr: &[i64]) -> i64 { if arr.is_empty() { 1 } else { 0 } }

fn k_kth_from_end(arr: &[i64], k: i64) -> Option<i64> {
    if k < 1 || k as usize > arr.len() { return None; }
    Some(arr[arr.len() - (k as usize)])
}

fn k_element_at(arr: &[i64], k: i64) -> Option<i64> {
    if k < 0 || k as usize >= arr.len() { return None; }
    Some(arr[k as usize])
}

fn dual_second_min(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut first = arr[0];
    let mut second = arr[0];
    for &item in arr {
        if item < first {
            second = first;
            first = item;
        } else if item < second {
            second = item;
        }
    }
    Some(second)
}

fn pairwise_sum_abs_diff(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    let mut total = 0i64;
    for i in 1..arr.len() {
        let mut d = arr[i] - arr[i - 1];
        if d < 0 { d = -d; }
        total += d;
    }
    total
}

fn index_sum_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i, _)| i % 2 == 0).map(|(_, &v)| v).sum()
}

fn index_product_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i, _)| i % 2 == 0).map(|(_, &v)| v).fold(1, i64::saturating_mul)
}

fn index_count_peaks(arr: &[i64]) -> i64 {
    if arr.len() < 3 { return 0; }
    (1..arr.len() - 1).filter(|&i| arr[i] > arr[i - 1] && arr[i] > arr[i + 1]).count() as i64
}

fn index_count_valleys(arr: &[i64]) -> i64 {
    if arr.len() < 3 { return 0; }
    (1..arr.len() - 1).filter(|&i| arr[i] < arr[i - 1] && arr[i] < arr[i + 1]).count() as i64
}

fn index_count_distinct(arr: &[i64]) -> i64 {
    let mut count = 0i64;
    for i in 0..arr.len() {
        if !arr[..i].contains(&arr[i]) { count += 1; }
    }
    count
}

fn index_argmax(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut best_i = 0usize;
    for i in 1..arr.len() {
        if arr[i] > arr[best_i] { best_i = i; }
    }
    Some(best_i as i64)
}

fn index_argmin(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut best_i = 0usize;
    for i in 1..arr.len() {
        if arr[i] < arr[best_i] { best_i = i; }
    }
    Some(best_i as i64)
}

fn index_mode(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut best_val = arr[0];
    let mut best_count = 1i64;
    for i in 0..arr.len() {
        let count = arr.iter().filter(|&&v| v == arr[i]).count() as i64;
        if count > best_count {
            best_count = count;
            best_val = arr[i];
        }
    }
    Some(best_val)
}

fn k_kth_smallest(arr: &[i64], k: i64) -> Option<i64> {
    if k < 1 || k as usize > arr.len() { return None; }
    let mut s = arr.to_vec();
    s.sort_unstable();
    Some(s[(k as usize) - 1])
}

fn k_first_index_of(arr: &[i64], k: i64) -> i64 {
    arr.iter().position(|&v| v == k).map(|i| i as i64).unwrap_or(-1)
}

fn pairwise_longest_plateau(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut best = 1i64;
    let mut cur = 1i64;
    for i in 1..arr.len() {
        if arr[i] == arr[i - 1] {
            cur += 1;
            if cur > best { best = cur; }
        } else {
            cur = 1;
        }
    }
    Some(best)
}

fn pairwise_max_abs_diff(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    let mut best = 0i64;
    for i in 1..arr.len() {
        let mut d = arr[i] - arr[i - 1];
        if d < 0 { d = -d; }
        if d > best { best = d; }
    }
    best
}

fn pairwise_count_adj_diff(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    (1..arr.len()).filter(|&i| arr[i] != arr[i - 1]).count() as i64
}

fn pairwise_count_increases(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    (1..arr.len()).filter(|&i| arr[i] > arr[i - 1]).count() as i64
}

fn pairwise_count_decreases(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    (1..arr.len()).filter(|&i| arr[i] < arr[i - 1]).count() as i64
}

fn pairwise_strictly_increasing(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 1; }
    if (1..arr.len()).all(|i| arr[i] > arr[i - 1]) { 1 } else { 0 }
}

fn pairwise_strictly_decreasing(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 1; }
    if (1..arr.len()).all(|i| arr[i] < arr[i - 1]) { 1 } else { 0 }
}

fn pairwise_non_increasing(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 1; }
    if (1..arr.len()).all(|i| arr[i] <= arr[i - 1]) { 1 } else { 0 }
}

fn pairwise_longest_inc_run(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut best = 1i64;
    let mut cur = 1i64;
    for i in 1..arr.len() {
        if arr[i] > arr[i - 1] {
            cur += 1;
            if cur > best { best = cur; }
        } else {
            cur = 1;
        }
    }
    Some(best)
}

fn pairwise_longest_dec_run(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut best = 1i64;
    let mut cur = 1i64;
    for i in 1..arr.len() {
        if arr[i] < arr[i - 1] {
            cur += 1;
            if cur > best { best = cur; }
        } else {
            cur = 1;
        }
    }
    Some(best)
}

fn pairwise_count_adj_eq(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    (1..arr.len()).filter(|&i| arr[i] == arr[i - 1]).count() as i64
}

fn pairwise_max_increase(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    let mut best = 0i64;
    for i in 1..arr.len() {
        let rise = arr[i] - arr[i - 1];
        if rise > best { best = rise; }
    }
    best
}

fn pairwise_max_decrease(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    let mut best = 0i64;
    for i in 1..arr.len() {
        let fall = arr[i - 1] - arr[i];
        if fall > best { best = fall; }
    }
    best
}

fn pairwise_longest_nondec_run(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut best = 1i64;
    let mut cur = 1i64;
    for i in 1..arr.len() {
        if arr[i] >= arr[i - 1] {
            cur += 1;
            if cur > best { best = cur; }
        } else {
            cur = 1;
        }
    }
    Some(best)
}

fn pairwise_longest_noninc_run(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut best = 1i64;
    let mut cur = 1i64;
    for i in 1..arr.len() {
        if arr[i] <= arr[i - 1] {
            cur += 1;
            if cur > best { best = cur; }
        } else {
            cur = 1;
        }
    }
    Some(best)
}

fn pairwise_sum_increases(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    let mut total = 0i64;
    for i in 1..arr.len() {
        let rise = arr[i] - arr[i - 1];
        if rise > 0 { total += rise; }
    }
    total
}

fn pairwise_sum_decreases(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    let mut total = 0i64;
    for i in 1..arr.len() {
        let fall = arr[i - 1] - arr[i];
        if fall > 0 { total += fall; }
    }
    total
}

fn pairwise_count_plateaus(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut count = 1i64;
    for i in 1..arr.len() {
        if arr[i] != arr[i - 1] { count += 1; }
    }
    Some(count)
}

fn pairwise_is_zigzag(arr: &[i64]) -> i64 {
    if arr.len() < 3 { return 1; }
    for i in 2..arr.len() {
        let d0 = arr[i - 1] - arr[i - 2];
        let d1 = arr[i] - arr[i - 1];
        if d0 == 0 || d1 == 0 || (d0 > 0) == (d1 > 0) { return 0; }
    }
    1
}

fn pairwise_min_increase(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for i in 1..arr.len() {
        let rise = arr[i] - arr[i - 1];
        if rise > 0 && (!found || rise < best) {
            best = rise;
            found = true;
        }
    }
    best
}

fn pairwise_min_decrease(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for i in 1..arr.len() {
        let fall = arr[i - 1] - arr[i];
        if fall > 0 && (!found || fall < best) {
            best = fall;
            found = true;
        }
    }
    best
}

fn pairwise_mean_abs_diff_trunc(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    let mut total = 0i64;
    for i in 1..arr.len() {
        let mut d = arr[i] - arr[i - 1];
        if d < 0 { d = -d; }
        total += d;
    }
    total / ((arr.len() - 1) as i64)
}

fn pairwise_count_sign_changes(arr: &[i64]) -> i64 {
    let mut count = 0i64;
    let mut prev = 0i64;
    for i in 1..arr.len() {
        let d = arr[i] - arr[i - 1];
        let sign = if d > 0 { 1 } else if d < 0 { -1 } else { 0 };
        if sign != 0 {
            if prev != 0 && sign != prev { count += 1; }
            prev = sign;
        }
    }
    count
}

fn pairwise_sum_sq_diff(arr: &[i64]) -> i64 {
    let mut total = 0i64;
    for i in 1..arr.len() {
        let d = arr[i] - arr[i - 1];
        total += d * d;
    }
    total
}

fn pairwise_mean_sq_diff_trunc(arr: &[i64]) -> i64 {
    if arr.len() < 2 { return 0; }
    pairwise_sum_sq_diff(arr) / ((arr.len() - 1) as i64)
}

fn pairwise_first_increase_idx(arr: &[i64]) -> i64 {
    for i in 1..arr.len() {
        if arr[i] > arr[i - 1] { return i as i64; }
    }
    -1
}

fn pairwise_first_decrease_idx(arr: &[i64]) -> i64 {
    for i in 1..arr.len() {
        if arr[i] < arr[i - 1] { return i as i64; }
    }
    -1
}

fn pairwise_last_increase_idx(arr: &[i64]) -> i64 {
    let mut best = -1i64;
    for i in 1..arr.len() {
        if arr[i] > arr[i - 1] { best = i as i64; }
    }
    best
}

fn pairwise_last_decrease_idx(arr: &[i64]) -> i64 {
    let mut best = -1i64;
    for i in 1..arr.len() {
        if arr[i] < arr[i - 1] { best = i as i64; }
    }
    best
}

fn index_middle(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    Some(arr[arr.len() / 2])
}

fn index_second(arr: &[i64]) -> Option<i64> {
    if arr.len() < 2 { return None; }
    Some(arr[1])
}

fn index_second_last(arr: &[i64]) -> Option<i64> {
    if arr.len() < 2 { return None; }
    Some(arr[arr.len() - 2])
}

fn index_max_even(arr: &[i64]) -> Option<i64> {
    let mut best = None;
    for (i, &v) in arr.iter().enumerate() {
        if i % 2 == 0 {
            best = Some(match best { Some(b) if b >= v => b, _ => v });
        }
    }
    best
}

fn index_min_odd(arr: &[i64]) -> Option<i64> {
    let mut best = None;
    for (i, &v) in arr.iter().enumerate() {
        if i % 2 == 1 {
            best = Some(match best { Some(b) if b <= v => b, _ => v });
        }
    }
    best
}

fn index_argmax_abs(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut best_i = 0usize;
    let mut best_abs = arr[0].abs();
    for i in 1..arr.len() {
        let a = arr[i].abs();
        if a > best_abs { best_abs = a; best_i = i; }
    }
    Some(best_i as i64)
}

fn index_argmin_abs(arr: &[i64]) -> Option<i64> {
    if arr.is_empty() { return None; }
    let mut best_i = 0usize;
    let mut best_abs = arr[0].abs();
    for i in 1..arr.len() {
        let a = arr[i].abs();
        if a < best_abs { best_abs = a; best_i = i; }
    }
    Some(best_i as i64)
}

fn index_sum_abs_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==0).map(|(_,&v)| v.abs()).sum()
}

fn index_sum_abs_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==1).map(|(_,&v)| v.abs()).sum()
}

fn index_count_even_indices(arr: &[i64]) -> i64 {
    ((arr.len() + 1) / 2) as i64
}

fn index_count_odd_indices(arr: &[i64]) -> i64 {
    (arr.len() / 2) as i64
}

fn index_xor_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==0).fold(0i64, |a, (_,&v)| a ^ v)
}

fn index_xor_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==1).fold(0i64, |a, (_,&v)| a ^ v)
}

fn index_or_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==0).fold(0i64, |a, (_,&v)| a | v)
}

fn index_or_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==1).fold(0i64, |a, (_,&v)| a | v)
}

fn index_and_even(arr: &[i64]) -> i64 {
    let mut it = arr.iter().enumerate().filter(|(i,_)| i%2==0);
    match it.next() {
        None => -1,
        Some((_, &v0)) => it.fold(v0, |a, (_,&v)| a & v),
    }
}

fn index_and_odd(arr: &[i64]) -> i64 {
    let mut it = arr.iter().enumerate().filter(|(i,_)| i%2==1);
    match it.next() {
        None => -1,
        Some((_, &v0)) => it.fold(v0, |a, (_,&v)| a & v),
    }
}

fn index_product_abs_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==0).map(|(_,&v)| v.abs()).fold(1i64, i64::saturating_mul)
}

fn index_product_abs_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==1).map(|(_,&v)| v.abs()).fold(1i64, i64::saturating_mul)
}

fn index_sum_squares_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==0).map(|(_,&v)| v*v).sum()
}

fn index_sum_squares_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==1).map(|(_,&v)| v*v).sum()
}

fn index_mean_even_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().enumerate().filter(|(i,_)| i%2==0).map(|(_,&v)| v).collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / (vals.len() as i64) }
}

fn index_mean_odd_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().enumerate().filter(|(i,_)| i%2==1).map(|(_,&v)| v).collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / (vals.len() as i64) }
}

fn index_count_positive_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v > 0).count() as i64
}

fn index_count_positive_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v > 0).count() as i64
}

fn index_count_negative_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v < 0).count() as i64
}

fn index_count_negative_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v < 0).count() as i64
}

fn index_sum_positive_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v > 0).map(|(_,&v)| v).sum()
}

fn index_sum_positive_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v > 0).map(|(_,&v)| v).sum()
}

fn index_sum_negative_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v < 0).map(|(_,&v)| v).sum()
}

fn index_sum_negative_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v < 0).map(|(_,&v)| v).sum()
}

fn index_count_zero_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v == 0).count() as i64
}

fn index_count_zero_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v == 0).count() as i64
}

fn index_max_abs_even(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for (i, &v) in arr.iter().enumerate() {
        if i % 2 == 0 {
            let a = v.abs();
            if !found || a > best { best = a; found = true; }
        }
    }
    best
}

fn index_max_abs_odd(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for (i, &v) in arr.iter().enumerate() {
        if i % 2 == 1 {
            let a = v.abs();
            if !found || a > best { best = a; found = true; }
        }
    }
    best
}

fn index_min_abs_even(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for (i, &v) in arr.iter().enumerate() {
        if i % 2 == 0 {
            let a = v.abs();
            if !found || a < best { best = a; found = true; }
        }
    }
    best
}

fn index_min_abs_odd(arr: &[i64]) -> i64 {
    let mut best = 0i64;
    let mut found = false;
    for (i, &v) in arr.iter().enumerate() {
        if i % 2 == 1 {
            let a = v.abs();
            if !found || a < best { best = a; found = true; }
        }
    }
    best
}

fn index_mean_abs_even_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().enumerate().filter(|(i,_)| i%2==0).map(|(_,&v)| v.abs()).collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 }
}

fn index_mean_abs_odd_trunc(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().enumerate().filter(|(i,_)| i%2==1).map(|(_,&v)| v.abs()).collect();
    if vals.is_empty() { 0 } else { vals.iter().sum::<i64>() / vals.len() as i64 }
}

fn index_count_nonzero_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v != 0).count() as i64
}

fn index_count_nonzero_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v != 0).count() as i64
}

fn index_product_nonzero_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v != 0).map(|(_,&v)| v).fold(1i64, i64::saturating_mul)
}

fn index_product_nonzero_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v != 0).map(|(_,&v)| v).fold(1i64, i64::saturating_mul)
}

fn index_sum_nonzero_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v != 0).map(|(_,&v)| v).sum()
}

fn index_sum_nonzero_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v != 0).map(|(_,&v)| v).sum()
}

fn index_max_nonzero_even(arr: &[i64]) -> i64 {
    let mut best = 0i64; let mut found = false;
    for (i,&v) in arr.iter().enumerate() {
        if i%2==0 && v != 0 {
            if !found || v > best { best = v; found = true; }
        }
    }
    best
}

fn index_max_nonzero_odd(arr: &[i64]) -> i64 {
    let mut best = 0i64; let mut found = false;
    for (i,&v) in arr.iter().enumerate() {
        if i%2==1 && v != 0 {
            if !found || v > best { best = v; found = true; }
        }
    }
    best
}

fn index_min_nonzero_even(arr: &[i64]) -> i64 {
    let mut best = 0i64; let mut found = false;
    for (i,&v) in arr.iter().enumerate() {
        if i%2==0 && v != 0 {
            if !found || v < best { best = v; found = true; }
        }
    }
    best
}

fn index_min_nonzero_odd(arr: &[i64]) -> i64 {
    let mut best = 0i64; let mut found = false;
    for (i,&v) in arr.iter().enumerate() {
        if i%2==1 && v != 0 {
            if !found || v < best { best = v; found = true; }
        }
    }
    best
}

fn index_count_even_value_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v%2==0).count() as i64
}

fn index_count_even_value_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v%2==0).count() as i64
}

fn index_count_odd_value_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v%2!=0).count() as i64
}

fn index_count_odd_value_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v%2!=0).count() as i64
}

fn index_sum_even_value_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v%2==0).map(|(_,&v)| v).sum()
}

fn index_sum_even_value_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v%2==0).map(|(_,&v)| v).sum()
}

fn index_sum_odd_value_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v%2!=0).map(|(_,&v)| v).sum()
}

fn index_sum_odd_value_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v%2!=0).map(|(_,&v)| v).sum()
}

fn index_product_even_value_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v%2==0).map(|(_,&v)| v).fold(1i64, i64::saturating_mul)
}

fn index_product_even_value_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v%2==0).map(|(_,&v)| v).fold(1i64, i64::saturating_mul)
}

fn index_product_odd_value_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v%2!=0).map(|(_,&v)| v).fold(1i64, i64::saturating_mul)
}

fn index_product_odd_value_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v%2!=0).map(|(_,&v)| v).fold(1i64, i64::saturating_mul)
}

fn index_sum_abs_even_value_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v%2==0).map(|(_,&v)| v.abs()).sum()
}

fn index_sum_abs_even_value_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v%2==0).map(|(_,&v)| v.abs()).sum()
}

fn index_sum_abs_odd_value_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==0 && v%2!=0).map(|(_,&v)| v.abs()).sum()
}

fn index_sum_abs_odd_value_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,&v)| i%2==1 && v%2!=0).map(|(_,&v)| v.abs()).sum()
}

fn index_or_abs_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==0).map(|(_,&v)| v.abs()).fold(0i64, |a,b| a|b)
}

fn index_or_abs_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==1).map(|(_,&v)| v.abs()).fold(0i64, |a,b| a|b)
}

fn index_and_abs_even(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().enumerate().filter(|(i,_)| i%2==0).map(|(_,&v)| v.abs()).collect();
    if vals.is_empty() { -1 } else { vals.into_iter().fold(-1i64, |a,b| a&b) }
}

fn index_and_abs_odd(arr: &[i64]) -> i64 {
    let vals: Vec<i64> = arr.iter().enumerate().filter(|(i,_)| i%2==1).map(|(_,&v)| v.abs()).collect();
    if vals.is_empty() { -1 } else { vals.into_iter().fold(-1i64, |a,b| a&b) }
}

fn index_xor_abs_even(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==0).map(|(_,&v)| v.abs()).fold(0i64, |a,b| a^b)
}

fn index_xor_abs_odd(arr: &[i64]) -> i64 {
    arr.iter().enumerate().filter(|(i,_)| i%2==1).map(|(_,&v)| v.abs()).fold(0i64, |a,b| a^b)
}

fn k_count_eq(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v == k).count() as i64
}

fn k_sum_gt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v > k).copied().sum()
}

fn k_count_gt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v > k).count() as i64
}

fn k_sum_lt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v < k).copied().sum()
}

fn k_count_lt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v < k).count() as i64
}

fn k_sum_eq(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v == k).copied().sum()
}

fn k_count_ne(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v != k).count() as i64
}

fn k_max_lt(arr: &[i64], k: i64) -> Option<i64> {
    let mut best = None;
    for &v in arr {
        if v < k {
            best = Some(match best { Some(b) if b >= v => b, _ => v });
        }
    }
    best
}

fn k_min_gt(arr: &[i64], k: i64) -> Option<i64> {
    let mut best = None;
    for &v in arr {
        if v > k {
            best = Some(match best { Some(b) if b <= v => b, _ => v });
        }
    }
    best
}

fn k_sum_ne(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v != k).copied().sum()
}

fn k_max_gt(arr: &[i64], k: i64) -> Option<i64> {
    let mut best = None;
    for &v in arr {
        if v > k {
            best = Some(match best { Some(b) if b >= v => b, _ => v });
        }
    }
    best
}

fn k_min_lt(arr: &[i64], k: i64) -> Option<i64> {
    let mut best = None;
    for &v in arr {
        if v < k {
            best = Some(match best { Some(b) if b <= v => b, _ => v });
        }
    }
    best
}

fn k_count_ge(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v >= k).count() as i64
}

fn k_count_le(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v <= k).count() as i64
}

fn k_sum_ge(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v >= k).copied().sum()
}

fn k_sum_le(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v <= k).copied().sum()
}

fn k_first_ge(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v >= k { return i as i64; }
    }
    -1
}

fn k_first_le(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v <= k { return i as i64; }
    }
    -1
}

fn k_last_ge(arr: &[i64], k: i64) -> i64 {
    for i in (0..arr.len()).rev() {
        if arr[i] >= k { return i as i64; }
    }
    -1
}

fn k_last_le(arr: &[i64], k: i64) -> i64 {
    for i in (0..arr.len()).rev() {
        if arr[i] <= k { return i as i64; }
    }
    -1
}

fn k_sum_abs_gt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v > k).map(|&v| v.abs()).sum()
}

fn k_sum_abs_lt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v < k).map(|&v| v.abs()).sum()
}

fn k_count_abs_gt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() > k).count() as i64
}

fn k_count_abs_lt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() < k).count() as i64
}

fn k_sum_abs_ge(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v >= k).map(|&v| v.abs()).sum()
}

fn k_sum_abs_le(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v <= k).map(|&v| v.abs()).sum()
}

fn k_count_abs_eq(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() == k).count() as i64
}

fn k_count_abs_ne(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() != k).count() as i64
}

fn k_sum_abs_ne(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() != k).map(|&v| v.abs()).sum()
}

fn k_sum_abs_eq(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() == k).map(|&v| v.abs()).sum()
}

fn k_count_abs_ge(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() >= k).count() as i64
}

fn k_count_abs_le(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() <= k).count() as i64
}

fn k_first_abs_ge(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() >= k { return i as i64; }
    }
    -1
}

fn k_last_abs_ge(arr: &[i64], k: i64) -> i64 {
    for i in (0..arr.len()).rev() {
        if arr[i].abs() >= k { return i as i64; }
    }
    -1
}

fn k_first_abs_eq(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() == k { return i as i64; }
    }
    -1
}

fn k_last_abs_eq(arr: &[i64], k: i64) -> i64 {
    for i in (0..arr.len()).rev() {
        if arr[i].abs() == k { return i as i64; }
    }
    -1
}

fn k_first_abs_gt(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() > k { return i as i64; }
    }
    -1
}

fn k_last_abs_gt(arr: &[i64], k: i64) -> i64 {
    for i in (0..arr.len()).rev() {
        if arr[i].abs() > k { return i as i64; }
    }
    -1
}

fn k_first_abs_lt(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() < k { return i as i64; }
    }
    -1
}

fn k_last_abs_lt(arr: &[i64], k: i64) -> i64 {
    for i in (0..arr.len()).rev() {
        if arr[i].abs() < k { return i as i64; }
    }
    -1
}

fn k_max_abs_lt(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        let a = v.abs();
        if a < k {
            if !found || a > best { best = a; found = true; }
        }
    }
    best
}

fn k_min_abs_gt(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        let a = v.abs();
        if a > k {
            if !found || a < best { best = a; found = true; }
        }
    }
    best
}

fn k_max_abs_gt(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        let a = v.abs();
        if a > k {
            if !found || a > best { best = a; found = true; }
        }
    }
    best
}

fn k_min_abs_lt(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        let a = v.abs();
        if a < k {
            if !found || a < best { best = a; found = true; }
        }
    }
    best
}

fn k_first_abs_ne(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() != k { return i as i64; }
    }
    -1
}

fn k_last_abs_ne(arr: &[i64], k: i64) -> i64 {
    for i in (0..arr.len()).rev() {
        if arr[i].abs() != k { return i as i64; }
    }
    -1
}

fn k_sum_where_abs_eq(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() == k).sum()
}

fn k_product_where_abs_eq(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() == k).fold(1i64, |a, &b| a.saturating_mul(b))
}

fn k_max_where_abs_eq(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() == k {
            if !found || v > best { best = v; found = true; }
        }
    }
    best
}

fn k_sum_where_abs_ne(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() != k).copied().fold(0i64, i64::saturating_add)
}

fn k_product_where_abs_ne(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() != k).copied().fold(1i64, i64::saturating_mul)
}

fn k_max_where_abs_ne(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() != k {
            if !found || v > best { best = v; found = true; }
        }
    }
    best
}

fn k_min_where_abs_ne(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() != k {
            if !found || v < best { best = v; found = true; }
        }
    }
    best
}

fn k_sum_where_abs_gt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() > k).copied().fold(0i64, i64::saturating_add)
}

fn k_sum_where_abs_lt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() < k).copied().fold(0i64, i64::saturating_add)
}

fn k_product_where_abs_gt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() > k).copied().fold(1i64, i64::saturating_mul)
}

fn k_product_where_abs_lt(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() < k).copied().fold(1i64, i64::saturating_mul)
}

fn k_max_where_abs_gt(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() > k {
            if !found || v > best { best = v; found = true; }
        }
    }
    best
}

fn k_min_where_abs_gt(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() > k {
            if !found || v < best { best = v; found = true; }
        }
    }
    best
}

fn k_min_where_abs_lt(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() < k {
            if !found || v < best { best = v; found = true; }
        }
    }
    best
}

fn k_max_where_abs_ge(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() >= k {
            if !found || v > best { best = v; found = true; }
        }
    }
    best
}

fn k_min_where_abs_ge(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() >= k {
            if !found || v < best { best = v; found = true; }
        }
    }
    best
}

fn k_sum_where_abs_ge(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&x| x.abs() >= k).copied().sum()
}

fn k_product_where_abs_ge(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() >= k).copied().fold(1i64, i64::saturating_mul)
}

fn k_product_where_abs_le(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() <= k).copied().fold(1i64, i64::saturating_mul)
}

fn k_sum_where_abs_le(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&x| x.abs() <= k).copied().sum()
}

fn k_max_where_abs_le(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() <= k {
            if !found || v > best { best = v; found = true; }
        }
    }
    best
}

fn k_min_where_abs_le(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() <= k {
            if !found || v < best { best = v; found = true; }
        }
    }
    best
}

fn k_count_where_abs_ge(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() >= k).count() as i64
}

fn k_count_where_abs_le(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() <= k).count() as i64
}

fn k_first_where_abs_ge(arr: &[i64], k: i64) -> i64 {
    arr.iter().copied().find(|&v| v.abs() >= k).unwrap_or(0)
}

fn k_last_where_abs_ge(arr: &[i64], k: i64) -> i64 {
    arr.iter().copied().rev().find(|&v| v.abs() >= k).unwrap_or(0)
}

fn k_first_where_abs_le(arr: &[i64], k: i64) -> i64 {
    arr.iter().copied().find(|&v| v.abs() <= k).unwrap_or(0)
}

fn k_last_where_abs_le(arr: &[i64], k: i64) -> i64 {
    arr.iter().copied().rev().find(|&v| v.abs() <= k).unwrap_or(0)
}

fn k_first_where_abs_eq(arr: &[i64], k: i64) -> i64 {
    arr.iter().copied().find(|&v| v.abs() == k).unwrap_or(0)
}

fn k_last_where_abs_eq(arr: &[i64], k: i64) -> i64 {
    arr.iter().copied().rev().find(|&v| v.abs() == k).unwrap_or(0)
}

fn k_first_where_abs_ne(arr: &[i64], k: i64) -> i64 {
    arr.iter().copied().find(|&v| v.abs() != k).unwrap_or(0)
}

fn k_last_where_abs_ne(arr: &[i64], k: i64) -> i64 {
    arr.iter().copied().rev().find(|&v| v.abs() != k).unwrap_or(0)
}

fn k_count_where_abs_ne(arr: &[i64], k: i64) -> i64 {
    arr.iter().filter(|&&v| v.abs() != k).count() as i64
}

fn k_first_index_where_abs_ge(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() >= k { return i as i64; }
    }
    -1
}

fn k_last_index_where_abs_ge(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate().rev() {
        if v.abs() >= k { return i as i64; }
    }
    -1
}

fn k_first_index_where_abs_le(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() <= k { return i as i64; }
    }
    -1
}

fn k_last_index_where_abs_le(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate().rev() {
        if v.abs() <= k { return i as i64; }
    }
    -1
}

fn k_first_index_where_abs_eq(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() == k { return i as i64; }
    }
    -1
}

fn k_last_index_where_abs_eq(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate().rev() {
        if v.abs() == k { return i as i64; }
    }
    -1
}

fn k_first_index_where_abs_ne(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() != k { return i as i64; }
    }
    -1
}

fn k_last_index_where_abs_ne(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate().rev() {
        if v.abs() != k { return i as i64; }
    }
    -1
}

fn k_first_index_where_abs_gt(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() > k { return i as i64; }
    }
    -1
}

fn k_last_index_where_abs_gt(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate().rev() {
        if v.abs() > k { return i as i64; }
    }
    -1
}

fn k_first_index_where_abs_lt(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() < k { return i as i64; }
    }
    -1
}

fn k_last_index_where_abs_lt(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate().rev() {
        if v.abs() < k { return i as i64; }
    }
    -1
}

fn k_count_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v % k == 0).count() as i64
}

fn k_sum_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v % k == 0).copied().sum()
}

fn k_product_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 1; }
    arr.iter().filter(|&&v| v % k == 0).copied().fold(1i64, i64::saturating_mul)
}

fn k_first_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    for &v in arr {
        if v % k == 0 { return v; }
    }
    0
}

fn k_last_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    for &v in arr.iter().rev() {
        if v % k == 0 { return v; }
    }
    0
}

fn k_max_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v % k == 0).copied().max().unwrap_or(0)
}

fn k_min_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v % k == 0).copied().min().unwrap_or(0)
}

fn k_first_index_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return -1; }
    for (i, &v) in arr.iter().enumerate() {
        if v % k == 0 { return i as i64; }
    }
    -1
}

fn k_last_index_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return -1; }
    for (i, &v) in arr.iter().enumerate().rev() {
        if v % k == 0 { return i as i64; }
    }
    -1
}

fn k_abs_sum_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v % k == 0).map(|&v| v.abs()).sum()
}

fn k_abs_product_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 1; }
    arr.iter().filter(|&&v| v % k == 0).map(|&v| v.abs()).fold(1i64, i64::saturating_mul)
}

fn k_max_abs_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v % k == 0).map(|&v| v.abs()).max().unwrap_or(0)
}

fn k_min_abs_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v % k == 0).map(|&v| v.abs()).min().unwrap_or(0)
}

fn k_gcd_abs_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let mut g: Option<i64> = None;
    for &v in arr {
        if v % k == 0 {
            let a = v.abs();
            if a == 0 { continue; }
            g = Some(match g { None => a, Some(g) => smoke_gcd(g, a) });
        }
    }
    g.unwrap_or(0)
}

fn k_lcm_abs_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 1; }
    let mut l: Option<i64> = None;
    for &v in arr {
        if v % k == 0 {
            let a = v.abs();
            if a == 0 { continue; }
            l = Some(match l { None => a, Some(l) => smoke_lcm(l, a) });
        }
    }
    l.unwrap_or(1)
}

fn k_mean_abs_divisible_by_trunc(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let xs: Vec<i64> = arr.iter().copied().filter(|&v| v % k == 0).map(|v| v.abs()).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn k_count_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v != 0 && v % k == 0).count() as i64
}

fn k_sum_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v != 0 && v % k == 0).fold(0i64, |a, &b| a.saturating_add(b))
}

fn k_product_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v != 0 && v % k == 0).fold(1i64, |a, &b| a.saturating_mul(b))
}

fn k_max_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let mut best: Option<i64> = None;
    for &v in arr {
        if v != 0 && v % k == 0 { best = Some(best.map_or(v, |b| b.max(v))); }
    }
    best.unwrap_or(0)
}

fn k_min_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let mut best: Option<i64> = None;
    for &v in arr {
        if v != 0 && v % k == 0 { best = Some(best.map_or(v, |b| b.min(v))); }
    }
    best.unwrap_or(0)
}

fn k_first_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    for &v in arr {
        if v != 0 && v % k == 0 { return v; }
    }
    0
}

fn k_last_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    for &v in arr.iter().rev() {
        if v != 0 && v % k == 0 { return v; }
    }
    0
}

fn k_abs_sum_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v != 0 && v % k == 0).map(|&v| v.abs()).sum()
}

fn k_abs_product_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v != 0 && v % k == 0).map(|&v| v.abs()).fold(1i64, |a, b| a.saturating_mul(b))
}

fn k_mean_non_zero_divisible_by_trunc(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let xs: Vec<i64> = arr.iter().copied().filter(|&v| v != 0 && v % k == 0).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn k_max_abs_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let mut best: Option<i64> = None;
    for &v in arr {
        if v != 0 && v % k == 0 {
            let a = v.abs();
            best = Some(best.map_or(a, |b| b.max(a)));
        }
    }
    best.unwrap_or(0)
}

fn k_min_abs_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let mut best: Option<i64> = None;
    for &v in arr {
        if v != 0 && v % k == 0 {
            let a = v.abs();
            best = Some(best.map_or(a, |b| b.min(a)));
        }
    }
    best.unwrap_or(0)
}

fn k_mean_abs_non_zero_divisible_by_trunc(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let xs: Vec<i64> = arr.iter().copied().filter(|&v| v != 0 && v % k == 0).map(|v| v.abs()).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn k_xor_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v != 0 && v % k == 0).fold(0i64, |a, &b| a ^ b)
}

fn k_or_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v != 0 && v % k == 0).fold(0i64, |a, &b| a | b)
}

fn k_and_non_zero_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let mut found = false;
    let mut acc = -1i64;
    for &v in arr {
        if v != 0 && v % k == 0 {
            if !found { acc = v; found = true; } else { acc &= v; }
        }
    }
    if found { acc } else { -1 }
}

fn k_count_positive_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v > 0 && v % k == 0).count() as i64
}

fn k_count_negative_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v < 0 && v % k == 0).count() as i64
}

fn k_sum_positive_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v > 0 && v % k == 0).fold(0i64, |a, &b| a.saturating_add(b))
}

fn k_sum_negative_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v < 0 && v % k == 0).fold(0i64, |a, &b| a.saturating_add(b))
}

fn k_product_positive_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v > 0 && v % k == 0).fold(1i64, |a, &b| a.saturating_mul(b))
}

fn k_product_negative_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v < 0 && v % k == 0).fold(1i64, |a, &b| a.saturating_mul(b))
}

fn k_max_positive_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v > 0 && v % k == 0).copied().max().unwrap_or(0)
}

fn k_min_positive_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v > 0 && v % k == 0).copied().min().unwrap_or(0)
}

fn k_max_negative_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v < 0 && v % k == 0).copied().max().unwrap_or(0)
}

fn k_min_negative_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    arr.iter().filter(|&&v| v < 0 && v % k == 0).copied().min().unwrap_or(0)
}

fn k_first_positive_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return -1; }
    arr.iter().copied().find(|&v| v > 0 && v % k == 0).unwrap_or(-1)
}

fn k_last_positive_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return -1; }
    arr.iter().copied().rev().find(|&v| v > 0 && v % k == 0).unwrap_or(-1)
}

fn k_first_negative_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return -1; }
    arr.iter().copied().find(|&v| v < 0 && v % k == 0).unwrap_or(-1)
}

fn k_last_negative_divisible_by(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return -1; }
    arr.iter().copied().rev().find(|&v| v < 0 && v % k == 0).unwrap_or(-1)
}

fn k_mean_positive_divisible_by_trunc(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let xs: Vec<i64> = arr.iter().copied().filter(|&v| v > 0 && v % k == 0).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn k_mean_negative_divisible_by_trunc(arr: &[i64], k: i64) -> i64 {
    if k == 0 { return 0; }
    let xs: Vec<i64> = arr.iter().copied().filter(|&v| v < 0 && v % k == 0).collect();
    if xs.is_empty() { 0 } else { xs.iter().sum::<i64>() / xs.len() as i64 }
}

fn k_min_where_abs_eq(arr: &[i64], k: i64) -> i64 {
    let mut best = 0i64; let mut found = false;
    for &v in arr {
        if v.abs() == k {
            if !found || v < best { best = v; found = true; }
        }
    }
    best
}

fn k_first_abs_le(arr: &[i64], k: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() <= k { return i as i64; }
    }
    -1
}

fn k_last_abs_le(arr: &[i64], k: i64) -> i64 {
    for i in (0..arr.len()).rev() {
        if arr[i].abs() <= k { return i as i64; }
    }
    -1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_max() {
        let ex = vec![
            (vec![1, 5, 3], 5),
            (vec![-2, -9, 0], 0),
            (vec![7], 7),
        ];
        let (pred, reduce) = synthesize(&ex).expect("max");
        assert!(matches!(pred, Pred::All));
        assert!(matches!(reduce, Reduce::Max));
    }

    #[test]
    fn count_positives() {
        let ex = vec![
            (vec![-1, 2, -3, 4], 2),
            (vec![-5, -1], 0),
            (vec![1, 2, 3], 3),
        ];
        let (pred, reduce) = synthesize(&ex).expect("count+");
        assert!(matches!(pred, Pred::Positive));
        assert!(matches!(reduce, Reduce::Count));
    }

    #[test]
    fn plain_sum_still_wins() {
        let ex = vec![
            (vec![1, 2, 3], 6),
            (vec![0], 0),
            (vec![-1, 1], 0),
        ];
        let (_, reduce) = synthesize(&ex).expect("sum");
        assert!(matches!(reduce, Reduce::Sum));
    }

    #[test]
    fn array_product() {
        let ex = vec![
            (vec![2, 3, 4], 24),
            (vec![5, 5], 25),
            (vec![1, 2, 3, 4], 24),
        ];
        let (_, reduce) = synthesize(&ex).expect("product");
        assert!(matches!(reduce, Reduce::Product));
    }

    #[test]
    fn array_xor_fold() {
        let ex = vec![
            (vec![1, 2, 3], 0),
            (vec![7, 1], 6),
            (vec![4, 4, 1], 1),
        ];
        let (_, reduce) = synthesize(&ex).expect("xor");
        assert!(matches!(reduce, Reduce::Xor));
    }

    #[test]
    fn array_bitor_and_bitand() {
        // Overlapping bits so OR ≠ SUM (powers-of-two OR equals SUM).
        let or_ex = vec![
            (vec![3, 5], 7),
            (vec![6, 3], 7),
            (vec![12, 10], 14),
        ];
        let (_, r) = synthesize(&or_ex).expect("bitor");
        assert!(matches!(r, Reduce::BitOr));
        // AND must differ from Min and Xor (common collisions).
        let and_ex = vec![
            (vec![9, 12, 10], 8),
            (vec![25, 27, 30], 24),
            (vec![15, 21, 25], 1),
        ];
        let (_, r) = synthesize(&and_ex).expect("bitand");
        assert!(matches!(r, Reduce::BitAnd));
    }

    #[test]
    fn median_not_in_dsl() {
        // Median is outside the Reduce enum (needs sort + index).
        // Cases where median ≠ max/min/sum/xor/product/count.
        let ex = vec![
            (vec![1, 100, 2], 2),
            (vec![10, 1, 5], 5),
            (vec![0, 8, 4], 4),
        ];
        assert!(synthesize(&ex).is_none());
    }

    #[test]
    fn count_greater_than_k() {
        let ex = vec![
            (vec![1, 5, 3, 0], 2, 2),
            (vec![-1, 0, 1], 0, 1),
            (vec![4, 4, 4], 4, 0),
        ];
        let (pred, reduce) = synthesize_k(&ex).expect("gt k");
        assert!(matches!(pred, Pred::GtK));
        assert!(matches!(reduce, Reduce::Count));
    }

    #[test]
    fn dual_range_and_second_max() {
        assert_eq!(dual_range(&[1, 5, 3]), Some(4));
        assert_eq!(dual_second_max(&[3, 1, 4, 1, 5]), Some(4));
        assert_eq!(dual_second_max(&[2, 8, 3]), Some(3));
        assert_eq!(dual_second_max(&[5, 10, 8]), Some(8));
        assert_eq!(dual_second_min(&[3, 1, 4, 1, 5]), Some(1));
        assert_eq!(dual_stock_profit(&[7, 1, 5, 3, 6, 4]), Some(5));
        assert_eq!(dual_stock_profit(&[7, 6, 4, 3, 1]), Some(0));
        assert_eq!(dual_prefix_max_sum(&[1, 3, 2, 5]), Some(12));
        assert_eq!(dual_max_subarray(&[1, -2, 3, 4, -1]), Some(7));
        assert_eq!(dual_min_subarray(&[1, -2, 3, -4]), Some(-4));
        assert_eq!(dual_median(&[1, 100, 2]), Some(2));
        assert_eq!(dual_median(&[10, 1, 5, 0]), Some(5));
        assert_eq!(dual_gcd_all(&[12, 18, 30]), Some(6));
        assert_eq!(dual_lcm_all(&[2, 3, 4]), Some(12));
        assert_eq!(dual_mean_trunc(&[1, 2, 3]), Some(2));
        assert_eq!(dual_mean_trunc(&[10, 20, 30, 40]), Some(25));
        assert_eq!(dual_sum_squares(&[1, 2, 3]), 14);
        assert_eq!(dual_abs_sum(&[-1, 2, -3]), 6);
        assert_eq!(dual_max_abs(&[-1, 2, -5]), 5);
        assert_eq!(dual_min_positive(&[-2, 5, 3, 0]), 3);
        assert_eq!(dual_min_positive(&[-1, 0]), 0);
        assert_eq!(dual_count_negatives(&[-1, 2, -3, 0]), 2);
        assert_eq!(dual_count_evens(&[1, 2, 3, 4, 0]), 3);
        assert_eq!(dual_sum_positives(&[-1, 2, 3, -4]), 5);
        assert_eq!(dual_sum_negatives(&[-1, 2, 3, -4]), -5);
        assert_eq!(dual_count_odds(&[1, 2, 3, 4, 0]), 2);
        assert_eq!(dual_any_zero(&[1, 0, -1]), 1);
        assert_eq!(dual_all_positive(&[1, 2, 3]), 1);
        assert_eq!(dual_all_negative(&[-1, -2]), 1);
        assert_eq!(dual_count_zeros(&[0, 1, 0, 0]), 3);
        assert_eq!(dual_has_duplicate(&[1, 2, 1]), 1);
        assert_eq!(dual_has_duplicate(&[1, 2, 3]), 0);
        assert_eq!(dual_max_negative(&[-5, -1, 3]), -1);
        assert_eq!(dual_sum_even_values(&[1, 2, 3, 4]), 6);
        assert_eq!(dual_sum_odd_values(&[1, 2, 3, 4]), 4);
        assert_eq!(dual_all_non_negative(&[0, 1, 2]), 1);
        assert_eq!(dual_count_non_zeros(&[0, 1, 0, 2]), 2);
        assert_eq!(dual_alternating_sum(&[10, 3, 2, 1]), 8);
        assert_eq!(dual_product_positives(&[-1, 2, 3, 0]), 6);
        assert_eq!(dual_mean_abs_trunc(&[-2, 4, -6]), Some(4));
        assert_eq!(dual_first_positive(&[-2, 0, 5, 3]), 5);
        assert_eq!(dual_last_positive(&[-2, 0, 5, 3]), 3);
        assert_eq!(dual_first_negative(&[2, -4, -1]), -4);
        assert_eq!(dual_last_negative(&[2, -4, -1, 5]), -1);
        assert_eq!(dual_max_positive(&[-2, 5, 3, 0]), 5);
        assert_eq!(dual_min_negative(&[-2, -5, 3]), -5);
        assert_eq!(dual_product_negatives(&[-2, -3, 4]), 6);
        assert_eq!(dual_sum_cubes(&[1, 2, -1]), 8);
        assert_eq!(dual_count_gt_mean(&[1, 2, 3, 10]), Some(1));
        assert_eq!(dual_count_lt_mean(&[1, 2, 3, 10]), Some(3));
        assert_eq!(dual_is_palindrome(&[1, 2, 1]), 1);
        assert_eq!(dual_product_evens(&[1, 2, 3, 4]), 8);
        assert_eq!(dual_product_odds(&[1, 2, 3, 4]), 3);
        assert_eq!(dual_all_non_positive(&[-1, 0, -2]), 1);
        assert_eq!(dual_dot_index(&[10, 20, 30]), 80);
        assert_eq!(dual_sum_sq_diff_mean(&[1, 2, 3]), Some(2));
        assert_eq!(dual_any_non_zero(&[0, 0, 1]), 1);
        assert_eq!(dual_xor_all(&[1, 2, 3]), 0);
        assert_eq!(dual_product_non_zeros(&[0, 2, 3, 0]), 6);
        assert_eq!(dual_or_all(&[1, 2, 4]), 7);
        assert_eq!(dual_and_all(&[7, 3, 1]), 1);
        assert_eq!(dual_count_eq_mean(&[1, 2, 3, 2]), Some(2));
        assert_eq!(dual_product_abs(&[-2, 3, -4]), 24);
        assert_eq!(dual_count_non_negatives(&[-1, 0, 2, -3]), 2);
        assert_eq!(dual_count_positives(&[-1, 0, 2, 3]), 2);
        assert_eq!(dual_max_even_value(&[1, 8, 3, 4]), 8);
        assert_eq!(dual_max_odd_value(&[1, 8, 3, 4]), 3);
        assert_eq!(dual_min_even_value(&[1, 8, 3, 4]), 4);
        assert_eq!(dual_min_odd_value(&[1, 8, 3, 4]), 1);
        assert_eq!(dual_abs_range(&[-5, 2, -1]), 4);
        assert_eq!(dual_product_non_negatives(&[-2, 3, 0, 4]), 0);
        assert_eq!(dual_sum_non_negatives(&[-2, 3, 0, 4]), 7);
        assert_eq!(dual_sum_non_positives(&[-2, 3, 0, 4]), -2);
        assert_eq!(dual_count_non_positives(&[-2, 3, 0, 4]), 2);
        assert_eq!(dual_product_non_positives(&[-2, 3, 0, 4]), 0);
        assert_eq!(dual_sum_abs_evens(&[-4, 3, 2]), 6);
        assert_eq!(dual_sum_abs_odds(&[-4, 3, 2]), 3);
        assert_eq!(dual_product_abs_evens(&[-4, 3, 2]), 8);
        assert_eq!(dual_product_abs_odds(&[-4, 3, 2]), 3);
        assert_eq!(dual_xor_abs_all(&[-3, 5, 1]), 7);
        assert_eq!(dual_and_abs_all(&[-7, 3, 5]), 1);
        assert_eq!(dual_or_abs_all(&[-1, 2, 4]), 7);
        assert_eq!(dual_xor_abs_evens(&[-4, 3, 2]), 6);
        assert_eq!(dual_xor_abs_odds(&[-4, 3, 2]), 3);
        assert_eq!(dual_and_abs_evens(&[-6, 3, 2]), 2);
        assert_eq!(dual_and_abs_odds(&[-7, 3, 2]), 3);
        assert_eq!(dual_or_abs_evens(&[-4, 3, 2]), 6);
        assert_eq!(dual_or_abs_odds(&[-4, 3, 2]), 3);
        assert_eq!(dual_sum_squares_evens(&[-4, 3, 2]), 20);
        assert_eq!(dual_sum_squares_odds(&[-4, 3, 2]), 9);
        assert_eq!(dual_sum_cubes_evens(&[-4, 3, 2]), -56);
        assert_eq!(dual_sum_cubes_odds(&[-4, 3, 2]), 27);
        assert_eq!(dual_product_squares_evens(&[-4, 3, 2]), 64);
        assert_eq!(dual_product_squares_odds(&[-4, 3, 2]), 9);
        assert_eq!(dual_max_abs_evens(&[-4, 3, 2]), 4);
        assert_eq!(dual_max_abs_odds(&[-4, 3, 2]), 3);
        assert_eq!(dual_min_abs_evens(&[-4, 3, 2]), 2);
        assert_eq!(dual_min_abs_odds(&[-4, 3, 2]), 3);
        assert_eq!(dual_count_nonzero_evens(&[-4, 0, 3, 2]), 2);
        assert_eq!(dual_count_nonzero_odds(&[-4, 0, 3, 2]), 1);
        assert_eq!(dual_sum_nonzero_evens(&[-4, 0, 3, 2]), -2);
        assert_eq!(dual_sum_nonzero_odds(&[-4, 0, 3, 2]), 3);
        assert_eq!(dual_product_nonzero_evens(&[-4, 0, 3, 2]), -8);
        assert_eq!(dual_product_nonzero_odds(&[-4, 0, 3, 2]), 3);
        assert_eq!(dual_mean_abs_evens_trunc(&[-4, 3, 2]), 3);
        assert_eq!(dual_mean_abs_odds_trunc(&[-4, 3, 2]), 3);
        assert_eq!(dual_gcd_abs_evens(&[-4, 6, 3, 2]), 2);
        assert_eq!(dual_gcd_abs_odds(&[-4, 6, 3, 9]), 3);
        assert_eq!(dual_lcm_abs_evens(&[-4, 6, 3, 2]), 12);
        assert_eq!(dual_lcm_abs_odds(&[-4, 6, 3, 9]), 9);
        assert_eq!(dual_product_cubes_evens(&[-2, 3, 2]), -64);
        assert_eq!(dual_product_cubes_odds(&[-2, 3, 1]), 27);
        assert_eq!(dual_sum_abs_cubes_evens(&[-2, 3, 2]), 16);
        assert_eq!(dual_sum_abs_cubes_odds(&[-2, 3, 1]), 28);
        assert_eq!(dual_product_abs_cubes_evens(&[-2, 3, 2]), 64);
        assert_eq!(dual_product_abs_cubes_odds(&[-2, 3, 1]), 27);
        assert_eq!(dual_sum_abs_squares_evens(&[-2, 3, 2]), 8);
        assert_eq!(dual_sum_abs_squares_odds(&[-2, 3, 1]), 10);
        assert_eq!(dual_product_abs_squares_evens(&[-2, 3, 2]), 16);
        assert_eq!(dual_product_abs_squares_odds(&[-2, 3, 1]), 9);
        assert_eq!(dual_mean_abs_squares_evens_trunc(&[-2, 3, 2]), 4);
        assert_eq!(dual_mean_abs_squares_odds_trunc(&[-2, 3, 1]), 5);
        assert_eq!(dual_count_positive_evens(&[-2, 3, 2, 4]), 2);
        assert_eq!(dual_count_positive_odds(&[-2, 3, 1]), 2);
        assert_eq!(dual_count_negative_evens(&[-2, 3, -4, 1]), 2);
        assert_eq!(dual_count_negative_odds(&[-2, 3, -1]), 1);
        assert_eq!(dual_sum_positive_evens(&[-2, 3, 2, 4]), 6);
        assert_eq!(dual_sum_positive_odds(&[-2, 3, 1]), 4);
        assert_eq!(dual_sum_negative_evens(&[-2, 3, -4, 1]), -6);
        assert_eq!(dual_sum_negative_odds(&[-3, 2, -1]), -4);
        assert_eq!(dual_product_positive_evens(&[-2, 3, 2, 4]), 8);
        assert_eq!(dual_product_positive_odds(&[-2, 3, 1]), 3);
        assert_eq!(dual_product_negative_evens(&[-2, 3, -4, 1]), 8);
        assert_eq!(dual_product_negative_odds(&[-3, 2, -1]), 3);
        assert_eq!(dual_max_positive_evens(&[-2, 3, 2, 4]), 4);
        assert_eq!(dual_max_positive_odds(&[-2, 3, 1, 5]), 5);
        assert_eq!(dual_min_positive_evens(&[-2, 3, 2, 4]), 2);
        assert_eq!(dual_min_positive_odds(&[-2, 3, 1, 5]), 1);
        assert_eq!(dual_max_negative_evens(&[-4, 3, -2, 1]), -2);
        assert_eq!(dual_max_negative_odds(&[-5, 2, -1, -3]), -1);
        assert_eq!(dual_min_negative_evens(&[-4, 3, -2, 1]), -4);
        assert_eq!(dual_min_negative_odds(&[-5, 2, -1, -3]), -5);
        assert_eq!(dual_mean_positive_evens_trunc(&[-2, 3, 2, 4]), 3);
        assert_eq!(dual_mean_positive_odds_trunc(&[-2, 3, 1, 5]), 3);
        assert_eq!(dual_mean_negative_evens_trunc(&[-4, 3, -2, 1]), -3);
        assert_eq!(dual_mean_negative_odds_trunc(&[-5, 2, -1, -3]), -3);
        assert_eq!(dual_all_even_positive(&[2, 3, 4]), 1);
        assert_eq!(dual_all_odd_positive(&[-2, 3, 5]), 1);
        assert_eq!(dual_all_even_positive(&[-2, 3, 4]), 0);
        assert_eq!(dual_all_even_negative(&[-4, 3, -2]), 1);
        assert_eq!(dual_all_odd_negative(&[-5, 2, -1]), 1);
        assert_eq!(dual_all_even_negative(&[-4, 3, 2]), 0);
        assert_eq!(dual_any_even_positive(&[2, -3, -4]), 1);
        assert_eq!(dual_any_odd_positive(&[-2, 3, -5]), 1);
        assert_eq!(dual_any_even_positive(&[-2, 3, -4]), 0);
        assert_eq!(dual_any_even_negative(&[-2, 3, 4]), 1);
        assert_eq!(dual_any_odd_negative(&[2, -3, 5]), 1);
        assert_eq!(dual_any_even_negative(&[2, 3, 4]), 0);
        assert_eq!(dual_any_even_non_zero(&[0, 3, 2]), 1);
        assert_eq!(dual_any_odd_non_zero(&[0, 2, -1]), 1);
        assert_eq!(dual_any_even_non_zero(&[0, 3, 1]), 0);
        assert_eq!(dual_all_even_non_zero(&[2, 3, 4]), 1);
        assert_eq!(dual_all_odd_non_zero(&[2, 3, 5]), 1);
        assert_eq!(dual_all_even_non_zero(&[0, 3, 2]), 0);
        assert_eq!(dual_all_even_non_negative(&[2, -3, 4]), 1);
        assert_eq!(dual_all_odd_non_negative(&[2, 3, -4]), 1);
        assert_eq!(dual_all_even_non_negative(&[-2, 3, 4]), 0);
        assert_eq!(dual_all_even_non_positive(&[-2, 3, 0]), 1);
        assert_eq!(dual_all_odd_non_positive(&[2, -3, -5]), 1);
        assert_eq!(dual_all_even_non_positive(&[2, -3, 0]), 0);
        assert_eq!(dual_any_even_non_negative(&[-2, 3, 4]), 1);
        assert_eq!(dual_any_odd_non_negative(&[-2, -3, 5]), 1);
        assert_eq!(dual_any_even_non_negative(&[-2, 3, -4]), 0);
        assert_eq!(dual_any_even_non_positive(&[-2, 3, 4]), 1);
        assert_eq!(dual_any_odd_non_positive(&[2, -3, 5]), 1);
        assert_eq!(dual_any_even_non_positive(&[2, 3, 4]), 0);
        assert_eq!(dual_max_even_non_zero(&[0, 2, 3, -4]), 2);
        assert_eq!(dual_max_odd_non_zero(&[0, 2, 3, -5]), 3);
        assert_eq!(dual_max_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_min_even_non_zero(&[0, 2, -4, 3]), -4);
        assert_eq!(dual_min_odd_non_zero(&[0, 5, -3, 2]), -3);
        assert_eq!(dual_min_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_mean_even_non_zero_trunc(&[0, 2, 4, 3]), 3);
        assert_eq!(dual_mean_odd_non_zero_trunc(&[0, 3, 5, 2]), 4);
        assert_eq!(dual_mean_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_xor_even_non_zero(&[0, 2, 6, 3]), 4);
        assert_eq!(dual_xor_odd_non_zero(&[0, 1, 5, 2]), 4);
        assert_eq!(dual_xor_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_or_even_non_zero(&[0, 2, 4, 3]), 6);
        assert_eq!(dual_or_odd_non_zero(&[0, 1, 4, 5]), 5);
        assert_eq!(dual_or_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_and_even_non_zero(&[0, 6, 4, 3]), 4);
        assert_eq!(dual_and_odd_non_zero(&[0, 7, 5, 2]), 5);
        assert_eq!(dual_and_even_non_zero(&[0, 1, 3]), -1);
        assert_eq!(dual_sum_abs_even_non_zero(&[0, -4, 2, 3]), 6);
        assert_eq!(dual_sum_abs_odd_non_zero(&[0, -5, 3, 2]), 8);
        assert_eq!(dual_sum_abs_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_abs_even_non_zero(&[0, -4, 2, 3]), 8);
        assert_eq!(dual_product_abs_odd_non_zero(&[0, -5, 3, 2]), 15);
        assert_eq!(dual_product_abs_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_gcd_abs_even_non_zero(&[0, 12, -18, 5]), 6);
        assert_eq!(dual_gcd_abs_odd_non_zero(&[0, 15, -25, 2]), 5);
        assert_eq!(dual_gcd_abs_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_lcm_abs_even_non_zero(&[0, 4, 6, 3]), 12);
        assert_eq!(dual_lcm_abs_odd_non_zero(&[0, 3, 5, 2]), 15);
        assert_eq!(dual_lcm_abs_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_abs_even_non_zero_trunc(&[0, -4, 2, 3]), 3);
        assert_eq!(dual_mean_abs_odd_non_zero_trunc(&[0, -5, 3, 2]), 4);
        assert_eq!(dual_mean_abs_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_max_abs_even_non_zero(&[0, -8, 2, 3]), 8);
        assert_eq!(dual_max_abs_odd_non_zero(&[0, -7, 5, 2]), 7);
        assert_eq!(dual_max_abs_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_min_abs_even_non_zero(&[0, -8, 2, 3]), 2);
        assert_eq!(dual_min_abs_odd_non_zero(&[0, -7, 5, 2]), 5);
        assert_eq!(dual_min_abs_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_squares_even_non_zero(&[0, -4, 2, 3]), 20);
        assert_eq!(dual_sum_squares_odd_non_zero(&[0, -3, 5, 2]), 34);
        assert_eq!(dual_sum_squares_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_squares_even_non_zero(&[0, -2, 4, 3]), 64);
        assert_eq!(dual_product_squares_odd_non_zero(&[0, -3, 5, 2]), 225);
        assert_eq!(dual_product_squares_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_sum_cubes_even_non_zero(&[0, -2, 4, 3]), 56);
        assert_eq!(dual_sum_cubes_odd_non_zero(&[0, -3, 5, 2]), 98);
        assert_eq!(dual_sum_cubes_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_cubes_even_non_zero(&[0, -2, 2, 3]), -64);
        assert_eq!(dual_product_cubes_odd_non_zero(&[0, -3, 1, 2]), -27);
        assert_eq!(dual_product_cubes_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_sum_fourth_powers_even_non_zero(&[0, -2, 2, 3]), 32);
        assert_eq!(dual_sum_fourth_powers_odd_non_zero(&[0, -3, 1, 2]), 82);
        assert_eq!(dual_sum_fourth_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_fourth_powers_even_non_zero(&[0, -2, 2, 3]), 256);
        assert_eq!(dual_product_fourth_powers_odd_non_zero(&[0, -3, 1, 2]), 81);
        assert_eq!(dual_product_fourth_powers_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_fourth_powers_even_non_zero_trunc(&[0, -2, 2, 3]), 16);
        assert_eq!(dual_mean_fourth_powers_odd_non_zero_trunc(&[0, -3, 1, 2]), 41);
        assert_eq!(dual_mean_fourth_powers_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_fifth_powers_even_non_zero(&[0, -2, 2, 3]), 0);
        assert_eq!(dual_sum_fifth_powers_odd_non_zero(&[0, -3, 1, 2]), -242);
        assert_eq!(dual_sum_fifth_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_fifth_powers_even_non_zero(&[0, -2, 2, 3]), -1024);
        assert_eq!(dual_product_fifth_powers_odd_non_zero(&[0, -3, 1, 2]), -243);
        assert_eq!(dual_product_fifth_powers_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_fifth_powers_even_non_zero_trunc(&[0, -2, 2, 3]), 0);
        assert_eq!(dual_mean_fifth_powers_odd_non_zero_trunc(&[0, -3, 1, 2]), -121);
        assert_eq!(dual_mean_fifth_powers_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_sixth_powers_even_non_zero(&[0, -2, 2, 3]), 128);
        assert_eq!(dual_sum_sixth_powers_odd_non_zero(&[0, -3, 1, 2]), 730);
        assert_eq!(dual_sum_sixth_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_sixth_powers_even_non_zero(&[0, -2, 2, 3]), 4096);
        assert_eq!(dual_product_sixth_powers_odd_non_zero(&[0, -3, 1, 2]), 729);
        assert_eq!(dual_product_sixth_powers_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_sixth_powers_even_non_zero_trunc(&[0, -2, 2, 3]), 64);
        assert_eq!(dual_mean_sixth_powers_odd_non_zero_trunc(&[0, -3, 1, 2]), 365);
        assert_eq!(dual_mean_sixth_powers_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_seventh_powers_even_non_zero(&[0, -2, 2, 3]), 0);
        assert_eq!(dual_sum_seventh_powers_odd_non_zero(&[0, -3, 1, 2]), -2186);
        assert_eq!(dual_sum_seventh_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_seventh_powers_even_non_zero(&[0, -2, 2, 3]), -16384);
        assert_eq!(dual_product_seventh_powers_odd_non_zero(&[0, -3, 1, 2]), -2187);
        assert_eq!(dual_product_seventh_powers_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_seventh_powers_even_non_zero_trunc(&[0, -2, 2, 3]), 0);
        assert_eq!(dual_mean_seventh_powers_odd_non_zero_trunc(&[0, -3, 1, 2]), -1093);
        assert_eq!(dual_mean_seventh_powers_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_eighth_powers_even_non_zero(&[0, -2, 2, 3]), 512);
        assert_eq!(dual_sum_eighth_powers_odd_non_zero(&[0, -3, 1, 2]), 6562);
        assert_eq!(dual_sum_eighth_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_eighth_powers_even_non_zero(&[0, -2, 2, 3]), 65536);
        assert_eq!(dual_product_eighth_powers_odd_non_zero(&[0, -3, 1, 2]), 6561);
        assert_eq!(dual_product_eighth_powers_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_eighth_powers_even_non_zero_trunc(&[0, -2, 2, 3]), 256);
        assert_eq!(dual_mean_eighth_powers_odd_non_zero_trunc(&[0, -3, 1, 2]), 3281);
        assert_eq!(dual_mean_eighth_powers_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_ninth_powers_even_non_zero(&[0, -2, 2, 3]), 0);
        assert_eq!(dual_sum_ninth_powers_odd_non_zero(&[0, -3, 1, 2]), -19682);
        assert_eq!(dual_sum_ninth_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_ninth_powers_even_non_zero(&[0, -2, 2, 3]), -262144);
        assert_eq!(dual_product_ninth_powers_odd_non_zero(&[0, -3, 1, 2]), -19683);
        assert_eq!(dual_product_ninth_powers_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_ninth_powers_even_non_zero_trunc(&[0, -2, 2, 3]), 0);
        assert_eq!(dual_mean_ninth_powers_odd_non_zero_trunc(&[0, -3, 1, 2]), -9841);
        assert_eq!(dual_mean_ninth_powers_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_tenth_powers_even_non_zero(&[0, -2, 2, 3]), 2048);
        assert_eq!(dual_sum_tenth_powers_odd_non_zero(&[0, -3, 1, 2]), 59050);
        assert_eq!(dual_sum_tenth_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_tenth_powers_even_non_zero(&[0, -2, 2, 3]), 1048576);
        assert_eq!(dual_product_tenth_powers_odd_non_zero(&[0, -3, 1, 2]), 59049);
        assert_eq!(dual_product_tenth_powers_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_tenth_powers_even_non_zero_trunc(&[0, -2, 2, 3]), 1024);
        assert_eq!(dual_mean_tenth_powers_odd_non_zero_trunc(&[0, -3, 1, 2]), 29525);
        assert_eq!(dual_mean_tenth_powers_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_eleventh_powers_even_non_zero(&[0, -2, 2, 3]), 0);
        assert_eq!(dual_sum_eleventh_powers_odd_non_zero(&[0, -3, 1, 2]), -177146);
        assert_eq!(dual_sum_eleventh_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_eleventh_powers_even_non_zero(&[0, -2, 2, 3]), -4194304);
        assert_eq!(dual_product_eleventh_powers_odd_non_zero(&[0, -3, 1, 2]), -177147);
        assert_eq!(dual_product_eleventh_powers_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_eleventh_powers_even_non_zero_trunc(&[0, -2, 2, 3]), 0);
        assert_eq!(dual_mean_eleventh_powers_odd_non_zero_trunc(&[0, -3, 1, 2]), -88573);
        assert_eq!(dual_mean_eleventh_powers_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_twelfth_powers_even_non_zero(&[0, -2, 2, 3]), 8192);
        assert_eq!(dual_sum_twelfth_powers_odd_non_zero(&[0, -3, 1, 2]), 531442);
        assert_eq!(dual_sum_twelfth_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_twelfth_powers_even_non_zero(&[0, -2, 2, 3]), 16777216);
        assert_eq!(dual_product_twelfth_powers_odd_non_zero(&[0, -3, 1, 2]), 531441);
        assert_eq!(dual_product_twelfth_powers_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_twelfth_powers_even_non_zero_trunc(&[0, -2, 2, 3]), 4096);
        assert_eq!(dual_mean_twelfth_powers_odd_non_zero_trunc(&[0, -3, 1, 2]), 265721);
        assert_eq!(dual_mean_twelfth_powers_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_thirteenth_powers_even_non_zero(&[0, -2, 2, 3]), 0);
        assert_eq!(dual_sum_thirteenth_powers_odd_non_zero(&[0, -3, 1, 2]), -1594322);
        assert_eq!(dual_sum_thirteenth_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_product_thirteenth_powers_even_non_zero(&[0, -2, 2, 3]), -67108864);
        assert_eq!(dual_product_thirteenth_powers_odd_non_zero(&[0, -3, 1, 2]), -1594323);
        assert_eq!(dual_product_thirteenth_powers_even_non_zero(&[0, 1, 3]), 1);
        assert_eq!(dual_mean_thirteenth_powers_even_non_zero_trunc(&[0, -2, 2, 3]), 0);
        assert_eq!(dual_mean_thirteenth_powers_odd_non_zero_trunc(&[0, -3, 1, 2]), -797161);
        assert_eq!(dual_mean_thirteenth_powers_even_non_zero_trunc(&[0, 1, 3]), 0);
        assert_eq!(dual_sum_fourteenth_powers_even_non_zero(&[0, -2, 2, 3]), 32768);
        assert_eq!(dual_sum_fourteenth_powers_odd_non_zero(&[0, -3, 1, 2]), 4782970);
        assert_eq!(dual_sum_fourteenth_powers_even_non_zero(&[0, 1, 3]), 0);
        assert_eq!(dual_min_abs(&[-3, 9, 2]), 2);
        assert_eq!(dual_len(&[]), 0);
        assert_eq!(dual_is_empty(&[]), 1);
        assert_eq!(k_kth_from_end(&[10, 20, 30, 40], 2), Some(30));
        assert_eq!(k_element_at(&[10, 20, 30], 1), Some(20));
        assert_eq!(index_middle(&[1, 2, 3]), Some(2));
        assert_eq!(index_second(&[10, 20, 30]), Some(20));
        assert_eq!(index_second_last(&[10, 20, 30]), Some(20));
        assert_eq!(index_max_even(&[1, 9, 3, 8, 2]), Some(3));
        assert_eq!(index_min_odd(&[1, 9, 3, 2]), Some(2));
        assert_eq!(index_argmax_abs(&[1, -9, 3]), Some(1));
        assert_eq!(index_argmin_abs(&[5, -1, 3]), Some(1));
        assert_eq!(index_sum_abs_even(&[-1, 2, -3, 4]), 4);
        assert_eq!(index_sum_abs_odd(&[-1, 2, -3, 4]), 6);
        assert_eq!(index_count_even_indices(&[1, 2, 3, 4, 5]), 3);
        assert_eq!(index_count_odd_indices(&[1, 2, 3, 4, 5]), 2);
        assert_eq!(k_count_eq(&[1, 5, 5, 2], 5), 2);
        assert_eq!(k_sum_gt(&[1, 5, 3, 2], 2), 8);
        assert_eq!(k_count_gt(&[1, 5, 3, 2], 2), 2);
        assert_eq!(k_sum_lt(&[1, 5, 3, 2], 3), 3);
        assert_eq!(k_count_lt(&[1, 5, 3, 2], 3), 2);
        assert_eq!(k_sum_eq(&[2, 5, 2, 2], 2), 6);
        assert_eq!(k_count_ne(&[1, 5, 5, 2], 5), 2);
        assert_eq!(k_max_lt(&[1, 5, 3, 2], 4), Some(3));
        assert_eq!(k_min_gt(&[1, 5, 3, 2], 2), Some(3));
        assert_eq!(k_sum_ne(&[1, 5, 5, 2], 5), 3);
        assert_eq!(k_max_gt(&[1, 5, 3, 2], 2), Some(5));
        assert_eq!(k_min_lt(&[1, 5, 3, 2], 4), Some(1));
        assert_eq!(k_count_ge(&[1, 5, 3, 2], 3), 2);
        assert_eq!(k_count_le(&[1, 5, 3, 2], 3), 3);
        assert_eq!(k_sum_ge(&[1, 5, 3, 2], 3), 8);
        assert_eq!(k_sum_le(&[1, 5, 3, 2], 3), 6);
        assert_eq!(k_first_ge(&[1, 5, 3, 2], 3), 1);
        assert_eq!(k_first_le(&[5, 4, 1, 2], 2), 2);
        assert_eq!(k_last_ge(&[1, 5, 3, 2], 3), 2);
        assert_eq!(k_last_le(&[5, 4, 1, 2], 2), 3);
        assert_eq!(k_sum_abs_gt(&[-5, 2, 4], 1), 6);
        assert_eq!(k_sum_abs_lt(&[-5, 2, 4], 3), 7);
        assert_eq!(k_count_abs_gt(&[-5, 2, 4], 2), 2);
        assert_eq!(k_count_abs_lt(&[-5, 2, 4], 3), 1);
        assert_eq!(k_sum_abs_ge(&[-5, 2, 4], 2), 6);
        assert_eq!(k_sum_abs_le(&[-5, 2, 4], 2), 7);
        assert_eq!(k_count_abs_eq(&[-5, 2, 5, 4], 5), 2);
        assert_eq!(k_first_abs_ge(&[1, -5, 2], 4), 1);
        assert_eq!(k_last_abs_ge(&[5, 1, -5, 2], 4), 2);
        assert_eq!(k_first_abs_eq(&[1, -5, 5], 5), 1);
        assert_eq!(k_last_abs_eq(&[5, 1, -5, 2], 5), 2);
        assert_eq!(index_max_abs_even(&[-3, 9, 2, 8]), 3);
        assert_eq!(index_max_abs_odd(&[-3, 9, 2, 8]), 9);
        assert_eq!(index_min_abs_even(&[-3, 9, 2, 8]), 2);
        assert_eq!(index_min_abs_odd(&[-3, 9, 2, 8]), 8);
        assert_eq!(k_count_abs_ge(&[-5, 2, 4], 4), 2);
        assert_eq!(k_count_abs_le(&[-5, 2, 4], 4), 2);
        assert_eq!(index_mean_abs_even_trunc(&[-4, 9, 2, 8]), 3);
        assert_eq!(index_mean_abs_odd_trunc(&[-4, 9, 2, 8]), 8);
        assert_eq!(k_first_abs_le(&[5, 1, -3, 2], 2), 1);
        assert_eq!(k_last_abs_le(&[5, 1, -3, 2], 2), 3);
        assert_eq!(index_count_nonzero_even(&[0, 1, 2, 0]), 1);
        assert_eq!(index_count_nonzero_odd(&[0, 1, 2, 0]), 1);
        assert_eq!(k_sum_abs_eq(&[-5, 2, 5, 4], 5), 10);
        assert_eq!(index_product_nonzero_even(&[0, 9, -3, 8]), -3);
        assert_eq!(index_product_nonzero_odd(&[0, 9, -3, 8]), 72);
        assert_eq!(k_first_abs_gt(&[1, -5, 2], 2), 1);
        assert_eq!(k_last_abs_gt(&[5, 1, -5, 2], 2), 2);
        assert_eq!(index_sum_nonzero_even(&[0, 9, -3, 8]), -3);
        assert_eq!(index_sum_nonzero_odd(&[0, 9, -3, 8]), 17);
        assert_eq!(k_count_abs_ne(&[-5, 2, 5, 4], 5), 2);
        assert_eq!(index_max_nonzero_even(&[0, 9, -3, 8]), -3);
        assert_eq!(index_max_nonzero_odd(&[0, 9, -3, 8]), 9);
        assert_eq!(k_sum_abs_ne(&[-5, 2, 5, 4], 5), 6);
        assert_eq!(index_min_nonzero_even(&[5, 9, -3, 8]), -3);
        assert_eq!(index_min_nonzero_odd(&[5, 9, -3, 8]), 8);
        assert_eq!(k_first_abs_lt(&[5, 1, -3, 2], 2), 1);
        assert_eq!(k_last_abs_lt(&[5, 1, -3, 2], 2), 1);
        assert_eq!(index_count_even_value_even(&[2, 9, 3, 8]), 1);
        assert_eq!(index_count_even_value_odd(&[2, 9, 3, 8]), 1);
        assert_eq!(k_max_abs_lt(&[-5, 2, 4], 4), 2);
        assert_eq!(index_count_odd_value_even(&[2, 9, 3, 8]), 1);
        assert_eq!(index_count_odd_value_odd(&[2, 9, 3, 8]), 1);
        assert_eq!(k_min_abs_gt(&[-5, 2, 4], 2), 4);
        assert_eq!(index_sum_even_value_even(&[2, 9, 4, 8]), 6);
        assert_eq!(index_sum_even_value_odd(&[2, 9, 4, 8]), 8);
        assert_eq!(k_max_abs_gt(&[-5, 2, 4], 2), 5);
        assert_eq!(index_sum_odd_value_even(&[2, 9, 3, 8]), 3);
        assert_eq!(index_sum_odd_value_odd(&[2, 9, 3, 8]), 9);
        assert_eq!(k_min_abs_lt(&[-5, 2, 4], 4), 2);
        assert_eq!(index_product_even_value_even(&[2, 9, 4, 8]), 8);
        assert_eq!(index_product_even_value_odd(&[2, 9, 4, 8]), 8);
        assert_eq!(k_first_abs_ne(&[5, -5, 2], 5), 2);
        assert_eq!(index_product_odd_value_even(&[3, 8, 5, 2]), 15);
        assert_eq!(index_product_odd_value_odd(&[3, 9, 5, 7]), 63);
        assert_eq!(k_last_abs_ne(&[5, 2, -5], 5), 1);
        assert_eq!(k_sum_where_abs_eq(&[5, -5, 2], 5), 0);
        assert_eq!(k_product_where_abs_eq(&[5, -5, 2], 5), -25);
        assert_eq!(index_sum_abs_even_value_even(&[-4, 9, 2, 8]), 6);
        assert_eq!(index_sum_abs_even_value_odd(&[-4, 9, 2, 8]), 8);
        assert_eq!(k_max_where_abs_eq(&[5, -5, 2], 5), 5);
        assert_eq!(k_sum_where_abs_ne(&[5, -5, 2], 5), 2);
        assert_eq!(k_product_where_abs_ne(&[5, -5, 2], 5), 2);
        assert_eq!(k_max_where_abs_ne(&[5, -5, 2], 5), 2);
        assert_eq!(k_min_where_abs_ne(&[5, -5, 2], 5), 2);
        assert_eq!(k_sum_where_abs_gt(&[-5, 2, 4], 2), -1);
        assert_eq!(k_sum_where_abs_lt(&[-5, 2, 4], 4), 2);
        assert_eq!(k_product_where_abs_gt(&[-5, 2, 4], 2), -20);
        assert_eq!(k_product_where_abs_lt(&[-5, 2, 4], 4), 2);
        assert_eq!(k_max_where_abs_gt(&[-5, 2, 4], 2), 4);
        assert_eq!(k_min_where_abs_gt(&[-5, 2, 4], 2), -5);
        assert_eq!(k_min_where_abs_lt(&[-5, 2, 4, 1], 4), 1);
        assert_eq!(k_max_where_abs_ge(&[-5, 2, 4], 4), 4);
        assert_eq!(k_min_where_abs_ge(&[-5, 2, 4], 4), -5);
        assert_eq!(k_sum_where_abs_ge(&[-5, 2, 4], 4), -1);
        assert_eq!(k_product_where_abs_ge(&[-5, 2, 4], 4), -20);
        assert_eq!(k_product_where_abs_le(&[-5, 2, 4], 4), 8);
        assert_eq!(k_sum_where_abs_le(&[-5, 2, 4], 4), 6);
        assert_eq!(k_max_where_abs_le(&[-5, 2, 4], 4), 4);
        assert_eq!(k_min_where_abs_le(&[-5, 2, 4], 4), 2);
        assert_eq!(k_count_where_abs_ge(&[-5, 2, 4], 4), 2);
        assert_eq!(k_count_where_abs_le(&[-5, 2, 4], 4), 2);
        assert_eq!(k_first_where_abs_ge(&[-5, 2, 4], 4), -5);
        assert_eq!(k_last_where_abs_ge(&[-5, 2, 4], 4), 4);
        assert_eq!(k_first_where_abs_le(&[-5, 2, 4], 4), 2);
        assert_eq!(k_last_where_abs_le(&[-5, 2, 4], 4), 4);
        assert_eq!(k_first_where_abs_eq(&[-5, 2, 4], 4), 4);
        assert_eq!(k_last_where_abs_eq(&[-5, 4, -4], 4), -4);
        assert_eq!(k_first_where_abs_ne(&[-4, 2, 4], 4), 2);
        assert_eq!(k_last_where_abs_ne(&[-4, 2, 4], 4), 2);
        assert_eq!(k_count_where_abs_ne(&[-4, 2, 4], 4), 1);
        assert_eq!(k_first_index_where_abs_ge(&[-1, 2, 5], 4), 2);
        assert_eq!(k_last_index_where_abs_ge(&[-1, 5, 2, 6], 4), 3);
        assert_eq!(k_first_index_where_abs_le(&[-5, 2, 4], 4), 1);
        assert_eq!(k_last_index_where_abs_le(&[-5, 2, 4], 4), 2);
        assert_eq!(k_first_index_where_abs_eq(&[-5, 2, 4], 4), 2);
        assert_eq!(k_last_index_where_abs_eq(&[-4, 2, 4], 4), 2);
        assert_eq!(k_first_index_where_abs_ne(&[-4, 2, 4], 4), 1);
        assert_eq!(k_last_index_where_abs_ne(&[-4, 2, 4], 4), 1);
        assert_eq!(k_first_index_where_abs_gt(&[-1, 2, 5], 4), 2);
        assert_eq!(k_last_index_where_abs_gt(&[-5, 2, 6], 4), 2);
        assert_eq!(k_first_index_where_abs_lt(&[-5, 2, 4], 4), 1);
        assert_eq!(k_last_index_where_abs_lt(&[-5, 2, 1], 4), 2);
        assert_eq!(k_count_divisible_by(&[2, 3, 4, 6], 2), 3);
        assert_eq!(k_sum_divisible_by(&[2, 3, 4, 6], 2), 12);
        assert_eq!(k_product_divisible_by(&[2, 3, 4], 2), 8);
        assert_eq!(k_first_divisible_by(&[3, 4, 6], 2), 4);
        assert_eq!(k_last_divisible_by(&[3, 4, 6], 2), 6);
        assert_eq!(k_max_divisible_by(&[3, 8, 4, 6], 2), 8);
        assert_eq!(k_min_divisible_by(&[3, -8, 4, 6], 2), -8);
        assert_eq!(k_first_index_divisible_by(&[3, 4, 6], 2), 1);
        assert_eq!(k_last_index_divisible_by(&[3, 4, 6], 2), 2);
        assert_eq!(k_abs_sum_divisible_by(&[-4, 3, 6], 2), 10);
        assert_eq!(k_abs_product_divisible_by(&[-4, 3, 6], 2), 24);
        assert_eq!(k_max_abs_divisible_by(&[-8, 3, 4], 2), 8);
        assert_eq!(k_min_abs_divisible_by(&[-8, 3, 4], 2), 4);
        assert_eq!(k_gcd_abs_divisible_by(&[12, 18, 5], 2), 6);
        assert_eq!(k_lcm_abs_divisible_by(&[4, 6, 5], 2), 12);
        assert_eq!(k_mean_abs_divisible_by_trunc(&[-4, 2, 6], 2), 4);
        assert_eq!(k_count_non_zero_divisible_by(&[0, 4, 6, 3], 2), 2);
        assert_eq!(k_sum_non_zero_divisible_by(&[0, 4, 6, 3], 2), 10);
        assert_eq!(k_product_non_zero_divisible_by(&[0, 4, 6, 3], 2), 24);
        assert_eq!(k_max_non_zero_divisible_by(&[0, 4, 6, 3], 2), 6);
        assert_eq!(k_min_non_zero_divisible_by(&[-8, 0, 4, 6], 2), -8);
        assert_eq!(k_first_non_zero_divisible_by(&[0, 4, 6, 3], 2), 4);
        assert_eq!(k_last_non_zero_divisible_by(&[0, 4, 6, 3], 2), 6);
        assert_eq!(k_abs_sum_non_zero_divisible_by(&[0, -4, 6, 3], 2), 10);
        assert_eq!(k_abs_product_non_zero_divisible_by(&[0, -4, 6, 3], 2), 24);
        assert_eq!(k_mean_non_zero_divisible_by_trunc(&[0, -4, 6, 3], 2), 1);
        assert_eq!(k_max_abs_non_zero_divisible_by(&[0, -8, 4, 3], 2), 8);
        assert_eq!(k_min_abs_non_zero_divisible_by(&[0, -8, 4, 3], 2), 4);
        assert_eq!(k_mean_abs_non_zero_divisible_by_trunc(&[0, -4, 6, 3], 2), 5);
        assert_eq!(k_xor_non_zero_divisible_by(&[0, 4, 6, 3], 2), 2);
        assert_eq!(k_or_non_zero_divisible_by(&[0, 4, 6, 3], 2), 6);
        assert_eq!(k_and_non_zero_divisible_by(&[0, 4, 6, 3], 2), 4);
        assert_eq!(k_count_positive_divisible_by(&[0, -4, 6, 3], 2), 1);
        assert_eq!(k_count_negative_divisible_by(&[0, -4, 6, 3], 2), 1);
        assert_eq!(k_sum_positive_divisible_by(&[0, -4, 6, 3], 2), 6);
        assert_eq!(k_sum_negative_divisible_by(&[0, -4, 6, 3], 2), -4);
        assert_eq!(k_product_positive_divisible_by(&[0, -4, 6, 8], 2), 48);
        assert_eq!(k_product_negative_divisible_by(&[0, -4, 6, -8], 2), 32);
        assert_eq!(k_max_positive_divisible_by(&[0, -4, 6, 8], 2), 8);
        assert_eq!(k_min_positive_divisible_by(&[0, -4, 6, 8], 2), 6);
        assert_eq!(k_max_negative_divisible_by(&[0, -4, 6, -8], 2), -4);
        assert_eq!(k_min_negative_divisible_by(&[0, -4, 6, -8], 2), -8);
        assert_eq!(k_first_positive_divisible_by(&[0, -4, 6, 8], 2), 6);
        assert_eq!(k_last_positive_divisible_by(&[0, -4, 6, 8], 2), 8);
        assert_eq!(k_first_negative_divisible_by(&[0, -4, 6, -8], 2), -4);
        assert_eq!(k_last_negative_divisible_by(&[0, -4, 6, -8], 2), -8);
        assert_eq!(k_mean_positive_divisible_by_trunc(&[0, -4, 6, 8], 2), 7);
        assert_eq!(k_mean_negative_divisible_by_trunc(&[0, -4, 6, -8], 2), -6);
        assert_eq!(index_sum_abs_odd_value_even(&[-3, 8, 5, 2]), 8);
        assert_eq!(index_sum_abs_odd_value_odd(&[-3, 9, 5, 7]), 16);
        assert_eq!(k_min_where_abs_eq(&[5, -5, 2], 5), -5);
        assert_eq!(index_or_abs_even(&[-1, 2, 4, 8]), 5);
        assert_eq!(index_or_abs_odd(&[-1, 2, 4, 8]), 10);
        assert_eq!(index_and_abs_even(&[-7, 2, 3, 8]), 3);
        assert_eq!(index_and_abs_odd(&[-1, 7, 4, 3]), 3);
        assert_eq!(index_xor_abs_even(&[-1, 2, 4, 8]), 5);
        assert_eq!(index_xor_abs_odd(&[-1, 2, 4, 8]), 10);
        assert_eq!(index_or_even(&[1, 2, 4, 8]), 5);
        assert_eq!(index_or_odd(&[1, 2, 4, 8]), 10);
        assert_eq!(index_and_even(&[7, 2, 3, 8]), 3);
        assert_eq!(index_and_odd(&[1, 7, 4, 3]), 3);
        assert_eq!(index_product_abs_even(&[-2, 9, -3, 8]), 6);
        assert_eq!(index_product_abs_odd(&[-2, 9, -3, 8]), 72);
        assert_eq!(index_sum_squares_even(&[2, 9, 3, 8]), 13);
        assert_eq!(index_sum_squares_odd(&[2, 9, 3, 8]), 145);
        assert_eq!(index_mean_even_trunc(&[2, 9, 4, 8]), 3);
        assert_eq!(index_mean_odd_trunc(&[2, 9, 4, 8]), 8);
        assert_eq!(index_count_positive_even(&[-1, 2, 3, -4]), 1);
        assert_eq!(index_count_positive_odd(&[-1, 2, 3, -4]), 1);
        assert_eq!(index_count_negative_even(&[-1, 2, 3, -4]), 1);
        assert_eq!(index_count_negative_odd(&[-1, 2, 3, -4]), 1);
        assert_eq!(index_sum_positive_even(&[-1, 2, 3, -4]), 3);
        assert_eq!(index_sum_positive_odd(&[-1, 2, 3, -4]), 2);
        assert_eq!(index_sum_negative_even(&[-1, 2, 3, -4]), -1);
        assert_eq!(index_sum_negative_odd(&[-1, 2, 3, -4]), -4);
        assert_eq!(index_count_zero_even(&[0, 1, 2, 0]), 1);
        assert_eq!(index_count_zero_odd(&[0, 1, 2, 0]), 1);
        assert_eq!(index_xor_even(&[1, 2, 4, 8]), 5);
        assert_eq!(index_xor_odd(&[1, 2, 4, 8]), 10);
        assert_eq!(pairwise_sum_abs_diff(&[1, 4, 2]), 5);
        assert_eq!(index_sum_even(&[1, 2, 3, 4]), 4);
        assert_eq!(index_product_even(&[2, 9, 3, 8]), 6);
        assert_eq!(index_count_peaks(&[1, 3, 2, 5, 1]), 2);
        assert_eq!(index_count_valleys(&[3, 1, 4, 0, 2]), 2);
        assert_eq!(index_count_distinct(&[1, 2, 1, 3]), 3);
        assert_eq!(index_argmax(&[1, 5, 3]), Some(1));
        assert_eq!(index_argmin(&[1, 5, 3]), Some(0));
        assert_eq!(index_mode(&[1, 2, 2, 3, 2]), Some(2));
        assert_eq!(index_mode(&[5, 5, 1, 1]), Some(5));
        assert_eq!(k_kth_smallest(&[3, 1, 4, 1, 5], 2), Some(1));
        assert_eq!(k_first_index_of(&[1, 5, 3, 5], 5), 1);
        assert_eq!(k_first_index_of(&[1, 2], 9), -1);
        assert_eq!(pairwise_longest_plateau(&[1, 1, 2, 2, 2, 1]), Some(3));
    }

    #[test]
    fn pairwise_max_abs_and_count_diff() {
        assert_eq!(pairwise_max_abs_diff(&[10, 3, 8]), 7);
        assert_eq!(pairwise_count_adj_diff(&[1, 1, 2, 2, 3]), 2);
        assert_eq!(pairwise_count_increases(&[1, 3, 2, 5]), 2);
        assert_eq!(pairwise_count_decreases(&[5, 3, 4, 1]), 2);
        assert_eq!(pairwise_strictly_increasing(&[1, 2, 4]), 1);
        assert_eq!(pairwise_strictly_increasing(&[1, 2, 2]), 0);
        assert_eq!(pairwise_strictly_decreasing(&[5, 3, 1]), 1);
        assert_eq!(pairwise_non_increasing(&[5, 5, 3]), 1);
        assert_eq!(pairwise_non_increasing(&[5, 6, 3]), 0);
        assert_eq!(pairwise_longest_inc_run(&[1, 2, 0, 3, 4, 5, 1]), Some(4));
        assert_eq!(pairwise_longest_dec_run(&[5, 4, 3, 0, 2, 1]), Some(4));
        assert_eq!(pairwise_count_adj_eq(&[1, 1, 2, 2, 2, 3]), 3);
        assert_eq!(pairwise_max_increase(&[1, 5, 2, 9]), 7);
        assert_eq!(pairwise_max_decrease(&[9, 2, 8, 1]), 7);
        assert_eq!(pairwise_longest_nondec_run(&[1, 2, 2, 0, 3, 4]), Some(3));
        assert_eq!(pairwise_longest_noninc_run(&[5, 4, 4, 1, 9, 8]), Some(4));
        assert_eq!(pairwise_sum_increases(&[1, 5, 2, 9]), 11);
        assert_eq!(pairwise_sum_decreases(&[9, 2, 8, 1]), 14);
        assert_eq!(pairwise_count_plateaus(&[1, 1, 2, 2, 2, 3]), Some(3));
        assert_eq!(pairwise_is_zigzag(&[1, 3, 2, 5, 0]), 1);
        assert_eq!(pairwise_is_zigzag(&[1, 2, 3]), 0);
        assert_eq!(pairwise_min_increase(&[1, 5, 2, 9]), 4);
        assert_eq!(pairwise_min_decrease(&[9, 2, 8, 1]), 7);
        assert_eq!(pairwise_mean_abs_diff_trunc(&[1, 4, 2]), 2);
        assert_eq!(pairwise_count_sign_changes(&[1, 3, 2, 5, 0]), 3);
        assert_eq!(pairwise_sum_sq_diff(&[1, 4, 2]), 13);
        assert_eq!(pairwise_mean_sq_diff_trunc(&[1, 4, 2]), 6);
        assert_eq!(pairwise_first_increase_idx(&[5, 4, 1, 3]), 3);
        assert_eq!(pairwise_first_increase_idx(&[5, 4, 1]), -1);
        assert_eq!(pairwise_first_decrease_idx(&[1, 3, 2, 5]), 2);
        assert_eq!(pairwise_last_increase_idx(&[1, 3, 2, 5]), 3);
        assert_eq!(pairwise_last_decrease_idx(&[1, 3, 2, 5, 0]), 4);
    }
}
EOF
( cd "$TMP/utbus_reduce" && cargo test --lib -q )
echo "utbus_reduce: OK"

echo "== string nonempty_split_count standalone =="
mkdir -p "$TMP/str_count/src"
cat > "$TMP/str_count/Cargo.toml" <<'EOF'
[package]
name = "str_count_test"
version = "0.1.0"
edition = "2021"
EOF
cat > "$TMP/str_count/src/lib.rs" <<'EOF'
fn nonempty_split_count(s: &str, sep: &str) -> i64 {
    s.split(sep).filter(|p| !p.is_empty()).count() as i64
}

fn filter_len_gt2(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() > 2).collect::<Vec<_>>().join(sep)
}

fn dedup_adjacent(s: &str, sep: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for w in s.split(sep) {
        if out.last().copied() != Some(w) { out.push(w); }
    }
    out.join(sep)
}

fn swap_first_last(s: &str, sep: &str) -> String {
    let mut words: Vec<&str> = s.split(sep).collect();
    if words.len() < 2 { return s.to_string(); }
    let last = words.len() - 1;
    words.swap(0, last);
    words.join(sep)
}

fn drop_first(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.is_empty() { return String::new(); }
    words[1..].join(sep)
}

fn first_word(s: &str, sep: &str) -> String {
    s.split(sep).next().unwrap_or("").to_string()
}

fn last_word(s: &str, sep: &str) -> String {
    s.split(sep).last().unwrap_or("").to_string()
}

fn penultimate_word(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() < 2 { String::new() } else { words[words.len()-2].to_string() }
}

fn middle_word(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.is_empty() { String::new() } else { words[words.len()/2].to_string() }
}

fn second_word(s: &str, sep: &str) -> String {
    s.split(sep).nth(1).unwrap_or("").to_string()
}

fn third_word(s: &str, sep: &str) -> String {
    s.split(sep).nth(2).unwrap_or("").to_string()
}

fn duplicate_each(s: &str, sep: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for w in s.split(sep) {
        out.push(w);
        out.push(w);
    }
    out.join(sep)
}

fn filter_len_eq2(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() == 2).collect::<Vec<_>>().join(sep)
}

fn filter_len_gt3(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() > 3).collect::<Vec<_>>().join(sep)
}

fn filter_len_eq3(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() == 3).collect::<Vec<_>>().join(sep)
}

fn filter_len_lt3(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() < 3).collect::<Vec<_>>().join(sep)
}

fn filter_len_eq4(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() == 4).collect::<Vec<_>>().join(sep)
}

fn filter_len_gt4(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() > 4).collect::<Vec<_>>().join(sep)
}

fn filter_len_eq5(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() == 5).collect::<Vec<_>>().join(sep)
}

fn filter_len_gt5(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() > 5).collect::<Vec<_>>().join(sep)
}

fn filter_len_lt5(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() < 5).collect::<Vec<_>>().join(sep)
}

fn filter_len_lt4(s: &str, sep: &str) -> String {
    s.split(sep).filter(|w| w.chars().count() < 4).collect::<Vec<_>>().join(sep)
}

fn dedup_all(s: &str, sep: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for w in s.split(sep) {
        if !out.contains(&w) { out.push(w); }
    }
    out.join(sep)
}

fn sort_by_len(s: &str, sep: &str) -> String {
    let mut owned: Vec<String> = s.split(sep).map(|w| w.to_string()).collect();
    owned.sort_by_key(|w| w.chars().count());
    owned.join(sep)
}

fn sort_by_len_desc(s: &str, sep: &str) -> String {
    let mut owned: Vec<String> = s.split(sep).map(|w| w.to_string()).collect();
    owned.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
    owned.join(sep)
}

fn fourth_word(s: &str, sep: &str) -> String {
    s.split(sep).nth(3).unwrap_or("").to_string()
}

fn fifth_word(s: &str, sep: &str) -> String {
    s.split(sep).nth(4).unwrap_or("").to_string()
}

fn duplicate_first_word(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.is_empty() { String::new() } else {
        let mut out = vec![words[0]];
        out.extend(words.iter().copied());
        out.join(sep)
    }
}

fn duplicate_last_word(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.is_empty() { String::new() } else {
        let mut out = words.clone();
        out.push(words[words.len()-1]);
        out.join(sep)
    }
}

fn swap_second_third(s: &str, sep: &str) -> String {
    let mut words: Vec<&str> = s.split(sep).collect();
    if words.len() >= 3 { words.swap(1, 2); }
    words.join(sep)
}

fn rotate_left_words(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.is_empty() { String::new() } else {
        let mut out = words[1..].to_vec();
        out.push(words[0]);
        out.join(sep)
    }
}

fn rotate_right_words(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.is_empty() { String::new() } else {
        let mut out = vec![words[words.len()-1]];
        out.extend_from_slice(&words[..words.len()-1]);
        out.join(sep)
    }
}

fn take_middle_two(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() < 2 { words.join(sep) } else {
        let start = (words.len() - 2) / 2;
        words[start..start+2].join(sep)
    }
}

fn join_with_hyphen(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("-")
}

fn join_with_underscore(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("_")
}

fn join_with_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("/")
}

fn join_with_comma(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(",")
}

fn join_with_colon(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(":")
}

fn join_with_semicolon(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(";")
}

fn join_with_dot(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(".")
}

fn join_with_plus(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("+")
}

fn join_with_equals(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("=")
}

fn join_with_at(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("@")
}

fn join_with_hash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("#")
}

fn join_with_star(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("*")
}

fn join_with_percent(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("%")
}

fn join_with_ampersand(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("&")
}

fn join_with_caret(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("^")
}

fn join_with_tilde(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("~")
}

fn join_with_tab(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("	")
}

fn join_with_newline(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("
")
}

fn join_with_cr(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("\r")
}

fn join_with_question(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("?")
}

fn join_with_exclamation(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("!")
}

fn join_with_backtick(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("`")
}

fn join_with_dollar(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("$")
}

fn join_with_double_quote(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("\"")
}

fn join_with_space(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(" ")
}

fn join_with_pipe(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("|")
}

fn join_with_brace(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("{")
}

fn join_with_bracket(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("[")
}

fn join_with_paren(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("(")
}

fn join_with_close_brace(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("}")
}

fn join_with_close_bracket(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("]")
}

fn join_with_close_paren(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(")")
}

fn join_with_ellipsis(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("...")
}

fn join_with_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("->")
}

fn join_with_double_colon(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("::")
}

fn join_with_double_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("//")
}

fn join_with_double_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("=>")
}

fn join_with_spaceship(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("<=>")
}

fn join_with_hash_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("#>")
}

fn join_with_colon_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(":->")
}

fn join_with_bang_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("!>")
}

fn join_with_tilde_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("~>")
}

fn join_with_star_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("*>")
}

fn join_with_slash_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("/>")
}

fn join_with_percent_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("%>")
}

fn join_with_amp_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("&>")
}

fn join_with_caret_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("^>")
}

fn join_with_pipe_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("|>")
}

fn join_with_dollar_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("$>")
}

fn join_with_at_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("@>")
}

fn join_with_question_arrow(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("?>")
}

fn join_with_hash_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("#!")
}

fn join_with_colon_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(":=")
}

fn join_with_plus_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("+=")
}

fn join_with_minus_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("-=")
}

fn join_with_star_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("*=")
}

fn join_with_slash_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("/=")
}

fn join_with_percent_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("%=")
}

fn join_with_amp_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("&=")
}

fn join_with_caret_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("^=")
}

fn join_with_pipe_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("|=")
}

fn join_with_tilde_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("~=")
}

fn join_with_dot_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(".=")
}

fn join_with_comma_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(",=")
}

fn join_with_semi_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(";=")
}

fn join_with_colon_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(":/")
}

fn join_with_bang_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("!=")
}

fn join_with_question_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("?=")
}

fn join_with_at_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("@=")
}

fn join_with_hash_eq(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("#=")
}

fn join_with_tilde_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("~/")
}

fn join_with_star_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("*/")
}

fn join_with_percent_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("%/")
}

fn join_with_amp_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("&/")
}

fn join_with_caret_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("^/")
}

fn join_with_pipe_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("|/")
}

fn join_with_dollar_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("$/")
}

fn join_with_bang_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("!/")
}

fn join_with_question_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("?/")
}

fn join_with_at_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("@/")
}

fn join_with_hash_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("#/")
}

fn join_with_dot_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("./")
}

fn join_with_comma_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(",/")
}

fn join_with_semi_slash(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(";/")
}

fn join_with_colon_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join(":!")
}

fn join_with_question_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("?!")
}

fn join_with_bang_question(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("!?")
}

fn join_with_at_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("@!")
}

fn join_with_dollar_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("$!")
}

fn join_with_caret_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("^!")
}

fn join_with_pipe_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("|!")
}

fn join_with_tilde_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("~!")
}

fn join_with_star_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("*!")
}

fn join_with_percent_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("%!")
}

fn join_with_amp_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("&!")
}

fn join_with_eq_bang(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("=!")
}

fn sort_words_desc(s: &str, sep: &str) -> String {
    let mut owned: Vec<String> = s.split(sep).map(|w| w.to_string()).collect();
    owned.sort();
    owned.reverse();
    owned.join(sep)
}

fn take_first_two(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    let n = words.len().min(2);
    words[..n].join(sep)
}

fn take_first_three(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    let n = words.len().min(3);
    words[..n].join(sep)
}

fn take_first_four(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    let n = words.len().min(4);
    words[..n].join(sep)
}

fn drop_first_two(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() <= 2 { String::new() } else { words[2..].join(sep) }
}

fn drop_first_three(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() <= 3 { String::new() } else { words[3..].join(sep) }
}

fn drop_first_four(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() <= 4 { String::new() } else { words[4..].join(sep) }
}

fn take_last_two(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() <= 2 { words.join(sep) } else { words[words.len()-2..].join(sep) }
}

fn take_last_three(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() <= 3 { words.join(sep) } else { words[words.len()-3..].join(sep) }
}

fn take_last_four(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() <= 4 { words.join(sep) } else { words[words.len()-4..].join(sep) }
}

fn drop_last_two(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() <= 2 { String::new() } else { words[..words.len()-2].join(sep) }
}

fn drop_last_three(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() <= 3 { String::new() } else { words[..words.len()-3].join(sep) }
}

fn drop_last_four(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() <= 4 { String::new() } else { words[..words.len()-4].join(sep) }
}

fn digit_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_digit()).count() as i64
}

fn upper_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_uppercase()).count() as i64
}

fn lower_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_lowercase()).count() as i64
}

fn vowel_count(s: &str) -> i64 {
    s.chars().filter(|c| matches!(c.to_ascii_lowercase(), 'a'|'e'|'i'|'o'|'u')).count() as i64
}

fn consonant_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_alphabetic()).filter(|c| !matches!(c.to_ascii_lowercase(), 'a'|'e'|'i'|'o'|'u')).count() as i64
}

fn sum_word_lens(s: &str, sep: &str) -> i64 {
    s.split(sep).filter(|w| !w.is_empty()).map(|w| w.chars().count() as i64).sum()
}

fn space_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == ' ').count() as i64
}

fn alpha_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_alphabetic()).count() as i64
}

fn cap_first(w: &str) -> String {
    let mut c = w.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn cap_last_word(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.is_empty() { return String::new(); }
    let mut out: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    let last = out.len() - 1;
    out[last] = cap_first(&out[last]);
    out.join(sep)
}

fn reverse_first_word(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.is_empty() { return String::new(); }
    let mut out: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    out[0] = out[0].chars().rev().collect();
    out.join(sep)
}

fn cap_first_word(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.is_empty() { return String::new(); }
    let mut out: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    out[0] = cap_first(&out[0]);
    out.join(sep)
}

fn reverse_last_word(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.is_empty() { return String::new(); }
    let mut out: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    let last = out.len() - 1;
    out[last] = out[last].chars().rev().collect();
    out.join(sep)
}

fn concat_words(s: &str, sep: &str) -> String {
    s.split(sep).collect::<Vec<_>>().join("")
}

fn digit_sum(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_digit()).map(|c| (c as u8 - b'0') as i64).sum()
}

fn digit_product(s: &str) -> i64 {
    let digits: Vec<i64> = s.chars().filter(|c| c.is_ascii_digit()).map(|c| (c as u8 - b'0') as i64).collect();
    if digits.is_empty() { 1 } else { digits.iter().product() }
}

fn punctuation_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_punctuation()).count() as i64
}

fn tab_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\t').count() as i64
}

fn newline_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\n').count() as i64
}

fn hex_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_hexdigit()).count() as i64
}

fn whitespace_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_whitespace()).count() as i64
}

fn alnum_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).count() as i64
}

fn non_alnum_count(s: &str) -> i64 {
    s.chars().filter(|c| !c.is_ascii_alphanumeric()).count() as i64
}

fn underscore_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '_').count() as i64
}

fn hyphen_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '-').count() as i64
}

fn slash_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '/').count() as i64
}

fn dot_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '.').count() as i64
}

fn comma_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == ',').count() as i64
}
fn colon_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == ':').count() as i64
}

fn semicolon_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == ';').count() as i64
}

fn question_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '?').count() as i64
}

fn exclamation_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '!').count() as i64
}

fn at_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '@').count() as i64
}

fn hash_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '#').count() as i64
}

fn percent_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '%').count() as i64
}

fn dollar_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '$').count() as i64
}

fn ampersand_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '&').count() as i64
}

fn star_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '*').count() as i64
}

fn plus_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '+').count() as i64
}

fn equals_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '=').count() as i64
}

fn caret_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '^').count() as i64
}

fn tilde_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '~').count() as i64
}

fn pipe_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '|').count() as i64
}

fn brace_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '{').count() as i64
}

fn bracket_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '[').count() as i64
}

fn paren_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '(').count() as i64
}

fn quote_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '"').count() as i64
}

fn backtick_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '`').count() as i64
}

fn apostrophe_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\'').count() as i64
}

fn less_than_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '<').count() as i64
}

fn greater_than_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '>').count() as i64
}

fn backslash_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\\').count() as i64
}

fn cr_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\r').count() as i64
}

fn null_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\0').count() as i64
}

fn ff_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x0c').count() as i64
}

fn vt_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x0b').count() as i64
}

fn bell_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x07').count() as i64
}

fn esc_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x1b').count() as i64
}

fn del_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x7f').count() as i64
}

fn soh_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x01').count() as i64
}

fn stx_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x02').count() as i64
}

fn etx_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x03').count() as i64
}

fn eot_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x04').count() as i64
}

fn enq_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x05').count() as i64
}

fn ack_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x06').count() as i64
}

fn bs_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x08').count() as i64
}

fn so_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x0e').count() as i64
}

fn si_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x0f').count() as i64
}

fn dle_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x10').count() as i64
}

fn dc1_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x11').count() as i64
}

fn dc2_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x12').count() as i64
}

fn dc3_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x13').count() as i64
}

fn dc4_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x14').count() as i64
}

fn nak_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x15').count() as i64
}

fn syn_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x16').count() as i64
}

fn etb_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x17').count() as i64
}

fn can_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x18').count() as i64
}

fn em_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x19').count() as i64
}

fn sub_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x1a').count() as i64
}

fn fs_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x1c').count() as i64
}

fn gs_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x1d').count() as i64
}

fn rs_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x1e').count() as i64
}

fn us_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '\x1f').count() as i64
}

fn bin_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '0' || *c == '1').count() as i64
}

fn oct_count(s: &str) -> i64 {
    s.chars().filter(|c| ('0'..='7').contains(c)).count() as i64
}

fn hex_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_hexdigit()).count() as i64
}

fn control_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_control()).count() as i64
}

fn non_letter_count(s: &str) -> i64 {
    s.chars().filter(|c| !c.is_ascii_alphabetic()).count() as i64
}

fn x_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'x')).count() as i64
}

fn y_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'y')).count() as i64
}

fn z_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'z')).count() as i64
}

fn non_hex_count(s: &str) -> i64 {
    s.chars().filter(|c| !c.is_ascii_hexdigit()).count() as i64
}

fn non_oct_count(s: &str) -> i64 {
    s.chars().filter(|c| !c.is_digit(8)).count() as i64
}

fn non_bin_count(s: &str) -> i64 {
    s.chars().filter(|c| !c.is_digit(2)).count() as i64
}

fn non_space_count(s: &str) -> i64 {
    s.chars().filter(|&c| c != ' ' && c != '\t' && c != '\n' && c != '\r').count() as i64
}

fn non_punct_count(s: &str) -> i64 {
    s.chars().filter(|&c| c != '!' && c != '.' && c != ',' && c != '?' && c != ';' && c != ':').count() as i64
}

fn non_vowel_count(s: &str) -> i64 {
    s.chars().filter(|c| !matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')).count() as i64
}

fn non_consonant_count(s: &str) -> i64 {
    s.chars().filter(|c| {
        let l = c.to_ascii_lowercase();
        !matches!(l, 'b' | 'c' | 'd' | 'f' | 'g' | 'h' | 'j' | 'k' | 'l' | 'm' | 'n' | 'p' | 'q' | 'r' | 's' | 't' | 'v' | 'w' | 'x' | 'y' | 'z')
    }).count() as i64
}

fn non_upper_count(s: &str) -> i64 {
    s.chars().filter(|c| !c.is_ascii_uppercase()).count() as i64
}

fn non_lower_count(s: &str) -> i64 {
    s.chars().filter(|c| !c.is_ascii_lowercase()).count() as i64
}

fn w_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'w')).count() as i64
}

fn v_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'v')).count() as i64
}

fn u_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'u')).count() as i64
}

fn t_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'t')).count() as i64
}

fn s_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'s')).count() as i64
}

fn r_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'r')).count() as i64
}

fn q_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'q')).count() as i64
}

fn p_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'p')).count() as i64
}

fn o_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'o')).count() as i64
}

fn n_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'n')).count() as i64
}

fn m_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'m')).count() as i64
}

fn l_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'l')).count() as i64
}

fn k_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'k')).count() as i64
}

fn j_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'j')).count() as i64
}

fn i_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'i')).count() as i64
}

fn h_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'h')).count() as i64
}

fn g_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'g')).count() as i64
}

fn f_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'f')).count() as i64
}

fn e_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'e')).count() as i64
}

fn d_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'d')).count() as i64
}

fn c_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'c')).count() as i64
}

fn b_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'b')).count() as i64
}

fn a_count(s: &str) -> i64 {
    s.chars().filter(|c| c.eq_ignore_ascii_case(&'a')).count() as i64
}

fn nine_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '9').count() as i64
}

fn eight_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '8').count() as i64
}

fn seven_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '7').count() as i64
}

fn six_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '6').count() as i64
}

fn five_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '5').count() as i64
}

fn four_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '4').count() as i64
}

fn three_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '3').count() as i64
}

fn two_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '2').count() as i64
}

fn one_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '1').count() as i64
}

fn zero_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == '0').count() as i64
}

fn non_digit_count(s: &str) -> i64 {
    s.chars().filter(|c| !c.is_ascii_digit()).count() as i64
}

fn print_count(s: &str) -> i64 {
    s.chars().filter(|c| !c.is_ascii_control()).count() as i64
}

fn non_ascii_count(s: &str) -> i64 {
    s.chars().filter(|c| !c.is_ascii()).count() as i64
}

fn ascii_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii()).count() as i64
}

fn blank_count(s: &str) -> i64 {
    s.chars().filter(|c| *c == ' ' || *c == '\t').count() as i64
}

fn graph_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_graphic()).count() as i64
}

fn punct_count(s: &str) -> i64 {
    s.chars().filter(|c| c.is_ascii_punctuation()).count() as i64
}


fn longest_word_len(s: &str, sep: &str) -> i64 {
    s.split(sep).filter(|w| !w.is_empty()).map(|w| w.chars().count() as i64).max().unwrap_or(0)
}

fn shortest_word_len(s: &str, sep: &str) -> i64 {
    s.split(sep).filter(|w| !w.is_empty()).map(|w| w.chars().count() as i64).min().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn skips_padding() {
        assert_eq!(nonempty_split_count("  two words  ", " "), 2);
        assert_eq!(nonempty_split_count("hello world", " "), 2);
        assert_eq!(nonempty_split_count("a--b--c", "-"), 3);
    }
    #[test]
    fn filter_and_dedup() {
        assert_eq!(filter_len_gt2("a to the moon", " "), "the moon");
        assert_eq!(dedup_adjacent("hi hi there there hi", " "), "hi there hi");
        assert_eq!(swap_first_last("one two three", " "), "three two one");
        assert_eq!(drop_first("keep the rest", " "), "the rest");
        assert_eq!(first_word("alpha beta gamma", " "), "alpha");
        assert_eq!(last_word("alpha beta gamma", " "), "gamma");
        assert_eq!(penultimate_word("alpha beta gamma", " "), "beta");
        assert_eq!(middle_word("alpha beta gamma", " "), "beta");
        assert_eq!(second_word("alpha beta gamma", " "), "beta");
        assert_eq!(third_word("alpha beta gamma delta", " "), "gamma");
        assert_eq!(fourth_word("alpha beta gamma delta", " "), "delta");
        assert_eq!(fifth_word("a b c d e f", " "), "e");
        assert_eq!(duplicate_first_word("a b c", " "), "a a b c");
        assert_eq!(duplicate_last_word("a b c", " "), "a b c c");
        assert_eq!(swap_second_third("a b c d", " "), "a c b d");
        assert_eq!(rotate_left_words("a b c", " "), "b c a");
        assert_eq!(rotate_right_words("a b c", " "), "c a b");
        assert_eq!(take_middle_two("a b c d e", " "), "b c");
        assert_eq!(join_with_hyphen("a b c", " "), "a-b-c");
        assert_eq!(join_with_underscore("a b c", " "), "a_b_c");
        assert_eq!(join_with_slash("a b c", " "), "a/b/c");
        assert_eq!(join_with_comma("a b c", " "), "a,b,c");
        assert_eq!(join_with_colon("a b c", " "), "a:b:c");
        assert_eq!(join_with_semicolon("a b c", " "), "a;b;c");
        assert_eq!(join_with_dot("a b c", " "), "a.b.c");
        assert_eq!(join_with_plus("a b c", " "), "a+b+c");
        assert_eq!(join_with_equals("a b c", " "), "a=b=c");
        assert_eq!(join_with_at("a b c", " "), "a@b@c");
        assert_eq!(join_with_hash("a b c", " "), "a#b#c");
        assert_eq!(join_with_star("a b c", " "), "a*b*c");
        assert_eq!(join_with_percent("a b c", " "), "a%b%c");
        assert_eq!(join_with_ampersand("a b c", " "), "a&b&c");
        assert_eq!(join_with_caret("a b c", " "), "a^b^c");
        assert_eq!(join_with_tilde("a b c", " "), "a~b~c");
        assert_eq!(join_with_tab("a b c", " "), "a	b	c");
        assert_eq!(join_with_newline("a b c", " "), "a
b
c");
                assert_eq!(join_with_cr("a b c", " "), "a\rb\rc");
        assert_eq!(join_with_question("a b c", " "), "a?b?c");
        assert_eq!(join_with_exclamation("a b c", " "), "a!b!c");
        assert_eq!(join_with_backtick("a b c", " "), "a`b`c");
        assert_eq!(join_with_dollar("a b c", " "), "a$b$c");
        assert_eq!(join_with_double_quote("a b c", " "), "a\"b\"c");
        assert_eq!(join_with_space("a-b-c", "-"), "a b c");
        assert_eq!(join_with_pipe("a b c", " "), "a|b|c");
        assert_eq!(join_with_brace("a b c", " "), "a{b{c");
        assert_eq!(join_with_bracket("a b c", " "), "a[b[c");
        assert_eq!(join_with_paren("a b c", " "), "a(b(c");
        assert_eq!(join_with_close_brace("a b c", " "), "a}b}c");
        assert_eq!(join_with_close_bracket("a b c", " "), "a]b]c");
        assert_eq!(join_with_close_paren("a b c", " "), "a)b)c");
        assert_eq!(join_with_ellipsis("a b c", " "), "a...b...c");
        assert_eq!(join_with_arrow("a b c", " "), "a->b->c");
        assert_eq!(join_with_double_colon("a b c", " "), "a::b::c");
        assert_eq!(join_with_double_slash("a b c", " "), "a//b//c");
        assert_eq!(join_with_double_arrow("a b c", " "), "a=>b=>c");
        assert_eq!(join_with_spaceship("a b c", " "), "a<=>b<=>c");
        assert_eq!(join_with_hash_arrow("a b c", " "), "a#>b#>c");
        assert_eq!(join_with_colon_arrow("a b c", " "), "a:->b:->c");
        assert_eq!(join_with_bang_arrow("a b c", " "), "a!>b!>c");
        assert_eq!(join_with_tilde_arrow("a b c", " "), "a~>b~>c");
        assert_eq!(join_with_star_arrow("a b c", " "), "a*>b*>c");
        assert_eq!(join_with_slash_arrow("a b c", " "), "a/>b/>c");
        assert_eq!(join_with_percent_arrow("a b c", " "), "a%>b%>c");
        assert_eq!(join_with_amp_arrow("a b c", " "), "a&>b&>c");
        assert_eq!(join_with_caret_arrow("a b c", " "), "a^>b^>c");
        assert_eq!(join_with_pipe_arrow("a b c", " "), "a|>b|>c");
        assert_eq!(join_with_dollar_arrow("a b c", " "), "a$>b$>c");
        assert_eq!(join_with_at_arrow("a b c", " "), "a@>b@>c");
        assert_eq!(join_with_question_arrow("a b c", " "), "a?>b?>c");
        assert_eq!(join_with_hash_bang("a b c", " "), "a#!b#!c");
        assert_eq!(join_with_colon_eq("a b c", " "), "a:=b:=c");
        assert_eq!(join_with_plus_eq("a b c", " "), "a+=b+=c");
        assert_eq!(join_with_minus_eq("a b c", " "), "a-=b-=c");
        assert_eq!(join_with_star_eq("a b c", " "), "a*=b*=c");
        assert_eq!(join_with_slash_eq("a b c", " "), "a/=b/=c");
        assert_eq!(join_with_percent_eq("a b c", " "), "a%=b%=c");
        assert_eq!(join_with_amp_eq("a b c", " "), "a&=b&=c");
        assert_eq!(join_with_caret_eq("a b c", " "), "a^=b^=c");
        assert_eq!(join_with_pipe_eq("a b c", " "), "a|=b|=c");
        assert_eq!(join_with_tilde_eq("a b c", " "), "a~=b~=c");
        assert_eq!(join_with_dot_eq("a b c", " "), "a.=b.=c");
        assert_eq!(join_with_comma_eq("a b c", " "), "a,=b,=c");
        assert_eq!(join_with_semi_eq("a b c", " "), "a;=b;=c");
        assert_eq!(join_with_colon_slash("a b c", " "), "a:/b:/c");
        assert_eq!(join_with_bang_eq("a b c", " "), "a!=b!=c");
        assert_eq!(join_with_question_eq("a b c", " "), "a?=b?=c");
        assert_eq!(join_with_at_eq("a b c", " "), "a@=b@=c");
        assert_eq!(join_with_hash_eq("a b c", " "), "a#=b#=c");
        assert_eq!(join_with_tilde_slash("a b c", " "), "a~/b~/c");
        assert_eq!(join_with_star_slash("a b c", " "), "a*/b*/c");
        assert_eq!(join_with_percent_slash("a b c", " "), "a%/b%/c");
        assert_eq!(join_with_amp_slash("a b c", " "), "a&/b&/c");
        assert_eq!(join_with_caret_slash("a b c", " "), "a^/b^/c");
        assert_eq!(join_with_pipe_slash("a b c", " "), "a|/b|/c");
        assert_eq!(join_with_dollar_slash("a b c", " "), "a$/b$/c");
        assert_eq!(join_with_bang_slash("a b c", " "), "a!/b!/c");
        assert_eq!(join_with_question_slash("a b c", " "), "a?/b?/c");
        assert_eq!(join_with_at_slash("a b c", " "), "a@/b@/c");
        assert_eq!(join_with_hash_slash("a b c", " "), "a#/b#/c");
        assert_eq!(join_with_dot_slash("a b c", " "), "a./b./c");
        assert_eq!(join_with_comma_slash("a b c", " "), "a,/b,/c");
        assert_eq!(join_with_semi_slash("a b c", " "), "a;/b;/c");
        assert_eq!(join_with_colon_bang("a b c", " "), "a:!b:!c");
        assert_eq!(join_with_question_bang("a b c", " "), "a?!b?!c");
        assert_eq!(join_with_bang_question("a b c", " "), "a!?b!?c");
        assert_eq!(join_with_at_bang("a b c", " "), "a@!b@!c");
        assert_eq!(join_with_dollar_bang("a b c", " "), "a$!b$!c");
        assert_eq!(join_with_caret_bang("a b c", " "), "a^!b^!c");
        assert_eq!(join_with_pipe_bang("a b c", " "), "a|!b|!c");
        assert_eq!(join_with_tilde_bang("a b c", " "), "a~!b~!c");
        assert_eq!(join_with_star_bang("a b c", " "), "a*!b*!c");
        assert_eq!(join_with_percent_bang("a b c", " "), "a%!b%!c");
        assert_eq!(join_with_amp_bang("a b c", " "), "a&!b&!c");
        assert_eq!(join_with_eq_bang("a b c", " "), "a=!b=!c");
        assert_eq!(duplicate_each("a b", " "), "a a b b");
        assert_eq!(filter_len_eq2("to be or not", " "), "to be or");
        assert_eq!(filter_len_gt3("a to the moon", " "), "moon");
        assert_eq!(filter_len_eq3("cat dog hi me", " "), "cat dog");
        assert_eq!(filter_len_lt3("cat dog hi me", " "), "hi me");
        assert_eq!(filter_len_eq4("a to the moon", " "), "moon");
        assert_eq!(filter_len_gt4("a to the moons", " "), "moons");
        assert_eq!(filter_len_lt4("cat dog hi me", " "), "cat dog hi me");
        assert_eq!(filter_len_eq5("a to moons hello", " "), "moons hello");
        assert_eq!(filter_len_gt5("a to planet hello", " "), "planet");
        assert_eq!(filter_len_lt5("a to moons hello", " "), "a to");
        assert_eq!(dedup_all("a b a c b", " "), "a b c");
        assert_eq!(sort_by_len("aaa b cc", " "), "b cc aaa");
        assert_eq!(sort_by_len_desc("aaa b cc", " "), "aaa cc b");
        assert_eq!(sort_words_desc("b aa c", " "), "c b aa");
        assert_eq!(take_first_two("one two three four", " "), "one two");
        assert_eq!(take_first_three("one two three four", " "), "one two three");
        assert_eq!(take_first_four("one two three four five", " "), "one two three four");
        assert_eq!(drop_first_two("one two three four", " "), "three four");
        assert_eq!(drop_first_three("one two three four five", " "), "four five");
        assert_eq!(drop_first_four("one two three four five six", " "), "five six");
        assert_eq!(take_last_two("one two three four", " "), "three four");
        assert_eq!(take_last_three("one two three four", " "), "two three four");
        assert_eq!(take_last_four("one two three four five", " "), "two three four five");
        assert_eq!(drop_last_two("one two three four", " "), "one two");
        assert_eq!(drop_last_three("one two three four five", " "), "one two");
        assert_eq!(drop_last_four("one two three four five six", " "), "one two");
        assert_eq!(digit_count("a1b22"), 3);
        assert_eq!(upper_count("Hi There"), 2);
        assert_eq!(lower_count("Hi There"), 5);
        assert_eq!(vowel_count("Beautiful"), 5);
        assert_eq!(consonant_count("Beautiful"), 4);
        assert_eq!(sum_word_lens("a to moon", " "), 7);
        assert_eq!(space_count("a b c"), 2);
        assert_eq!(alpha_count("Hi 2!"), 2);
        assert_eq!(cap_last_word("hello world", " "), "hello World");
        assert_eq!(reverse_first_word("abc def", " "), "cba def");
        assert_eq!(cap_first_word("hello world", " "), "Hello world");
        assert_eq!(reverse_last_word("abc def", " "), "abc fed");
        assert_eq!(concat_words("a b c", " "), "abc");
        assert_eq!(digit_sum("a12b3"), 6);
        assert_eq!(digit_product("a23b"), 6);
        assert_eq!(punctuation_count("Hi, you!"), 2);
        assert_eq!(tab_count("a\tb\tc"), 2);
        assert_eq!(newline_count("a\nb\nc"), 2);
        assert_eq!(hex_digit_count("xyz0Af!"), 3);
        assert_eq!(whitespace_count("a b\tc\n"), 3);
        assert_eq!(alnum_count("Hi 2!"), 3);
        assert_eq!(non_alnum_count("Hi 2!"), 2);
        assert_eq!(underscore_count("a_b__c"), 3);
        assert_eq!(hyphen_count("a-b--c"), 3);
        assert_eq!(slash_count("a/b//c"), 3);
        assert_eq!(dot_count("a.b..c"), 3);
        assert_eq!(comma_count("a,b,,c"), 3);
        assert_eq!(colon_count("a:b::c"), 3);
        assert_eq!(semicolon_count("a;b;;c"), 3);
        assert_eq!(question_count("a?b??c"), 3);
        assert_eq!(exclamation_count("a!b!!c"), 3);
        assert_eq!(at_count("a@b@@c"), 3);
        assert_eq!(hash_count("a#b##c"), 3);
        assert_eq!(percent_count("a%b%%c"), 3);
        assert_eq!(dollar_count("a$b$$c"), 3);
        assert_eq!(ampersand_count("a&b&&c"), 3);
        assert_eq!(star_count("a*b**c"), 3);
        assert_eq!(plus_count("a+b++c"), 3);
        assert_eq!(equals_count("a=b==c"), 3);
        assert_eq!(caret_count("a^b^^c"), 3);
        assert_eq!(tilde_count("a~b~~c"), 3);
        assert_eq!(pipe_count("a|b||c"), 3);
        assert_eq!(brace_count("a{b{{c"), 3);
        assert_eq!(bracket_count("a[b[[c"), 3);
        assert_eq!(paren_count("a(b((c"), 3);
        assert_eq!(quote_count("a\"b\"\"c"), 3);
        assert_eq!(backtick_count("a`b``c"), 3);
        assert_eq!(apostrophe_count("a'b''c"), 3);
        assert_eq!(less_than_count("a<b<<c"), 3);
        assert_eq!(greater_than_count("a>b>>c"), 3);
        assert_eq!(backslash_count("a\\b\\\\c"), 3);
        assert_eq!(cr_count("a\rb\r\rc"), 3);
        assert_eq!(null_count("a\0b\0\0c"), 3);
        assert_eq!(ff_count("a\x0cb\x0c\x0cc"), 3);
        assert_eq!(vt_count("a\x0bb\x0b\x0bc"), 3);
        assert_eq!(bell_count("a\x07b\x07\x07c"), 3);
        assert_eq!(esc_count("a\x1bb\x1b\x1bc"), 3);
        assert_eq!(del_count("a\x7fb\x7f\x7fc"), 3);
        assert_eq!(soh_count("a\x01b\x01\x01c"), 3);
        assert_eq!(stx_count("a\x02b\x02\x02c"), 3);
        assert_eq!(etx_count("a\x03b\x03\x03c"), 3);
        assert_eq!(eot_count("a\x04b\x04\x04c"), 3);
        assert_eq!(enq_count("a\x05b\x05\x05c"), 3);
        assert_eq!(ack_count("a\x06b\x06\x06c"), 3);
        assert_eq!(bs_count("a\x08b\x08\x08c"), 3);
        assert_eq!(so_count("a\x0eb\x0e\x0ec"), 3);
        assert_eq!(si_count("a\x0fb\x0f\x0fc"), 3);
        assert_eq!(dle_count("a\x10b\x10\x10c"), 3);
        assert_eq!(dc1_count("a\x11b\x11\x11c"), 3);
        assert_eq!(dc2_count("a\x12b\x12\x12c"), 3);
        assert_eq!(dc3_count("a\x13b\x13\x13c"), 3);
        assert_eq!(dc4_count("a\x14b\x14\x14c"), 3);
        assert_eq!(nak_count("a\x15b\x15\x15c"), 3);
        assert_eq!(syn_count("a\x16b\x16\x16c"), 3);
        assert_eq!(etb_count("a\x17b\x17\x17c"), 3);
        assert_eq!(can_count("a\x18b\x18\x18c"), 3);
        assert_eq!(em_count("a\x19b\x19\x19c"), 3);
        assert_eq!(sub_count("a\x1ab\x1a\x1ac"), 3);
        assert_eq!(fs_count("a\x1cb\x1c\x1cc"), 3);
        assert_eq!(gs_count("a\x1db\x1d\x1dc"), 3);
        assert_eq!(rs_count("a\x1eb\x1e\x1ec"), 3);
        assert_eq!(us_count("a\x1fb\x1f\x1fc"), 3);
        assert_eq!(hex_count("deadBEEF!"), 8);
        assert_eq!(oct_count("01234567x"), 8);
        assert_eq!(bin_count("01012x"), 4);
        assert_eq!(punct_count("a,b!c?"), 3);
        assert_eq!(control_count("a\x01b\x7fc"), 2);
        assert_eq!(graph_count("a b!"), 3);
        assert_eq!(blank_count("a b\tc "), 3);
        assert_eq!(ascii_count("ab\u{00e9}c"), 3);
        assert_eq!(non_ascii_count("ab\u{00e9}c"), 1);
        assert_eq!(print_count("a\x01bc"), 3);
        assert_eq!(non_digit_count("a1b2!"), 3);
        assert_eq!(zero_digit_count("a001b0"), 3);
        assert_eq!(one_digit_count("a101b1"), 3);
        assert_eq!(two_digit_count("a202b2"), 3);
        assert_eq!(three_digit_count("a303b3"), 3);
        assert_eq!(four_digit_count("a404b4"), 3);
        assert_eq!(five_digit_count("a505b5"), 3);
        assert_eq!(six_digit_count("a606b6"), 3);
        assert_eq!(seven_digit_count("a707b7"), 3);
        assert_eq!(eight_digit_count("a808b8"), 3);
        assert_eq!(nine_digit_count("a909b9"), 3);
        assert_eq!(a_count("Banana"), 3);
        assert_eq!(b_count("Bubble"), 3);
        assert_eq!(c_count("Circus"), 2);
        assert_eq!(d_count("Daddy"), 3);
        assert_eq!(e_count("Eleven"), 3);
        assert_eq!(f_count("Fluff"), 3);
        assert_eq!(g_count("Gaggle"), 3);
        assert_eq!(h_count("Harsh"), 2);
        assert_eq!(i_count("Initiative"), 4);
        assert_eq!(j_count("Jajaja"), 3);
        assert_eq!(k_count("Kick"), 2);
        assert_eq!(l_count("Lull"), 3);
        assert_eq!(m_count("Mommy"), 3);
        assert_eq!(n_count("Nanny"), 3);
        assert_eq!(o_count("Boobo"), 3);
        assert_eq!(p_count("Pepper"), 3);
        assert_eq!(q_count("QuaQq"), 3);
        assert_eq!(r_count("Error"), 3);
        assert_eq!(s_count("Assess"), 4);
        assert_eq!(t_count("Tattoo"), 3);
        assert_eq!(u_count("Usual"), 2);
        assert_eq!(v_count("Vivid"), 2);
        assert_eq!(w_count("Wow"), 2);
        assert_eq!(x_count("Xerox"), 2);
        assert_eq!(y_count("Yoyo"), 2);
        assert_eq!(z_count("Zestz"), 2);
        assert_eq!(non_hex_count("gg!"), 3);
        assert_eq!(non_oct_count("89a"), 3);
        assert_eq!(non_bin_count("2ab"), 3);
        assert_eq!(non_space_count("a b\tc"), 3);
        assert_eq!(non_punct_count("ab!"), 2);
        assert_eq!(non_vowel_count("aeiouX"), 1);
        assert_eq!(non_consonant_count("bcdA"), 1);
        assert_eq!(non_upper_count("AbC"), 1);
        assert_eq!(non_lower_count("AbC"), 2);
        assert_eq!(non_letter_count("a1B!"), 2);
        assert_eq!(longest_word_len("a to moon", " "), 4);
        assert_eq!(shortest_word_len("a to moon", " "), 1);
    }
}
EOF
( cd "$TMP/str_count" && cargo test --lib -q )
echo "str_count: OK"

if [[ -f "$ROOT/src/schema_component.rs" ]]; then
  echo "== schema_component note =="
  echo "(full schema_component e2e needs mog_synth + linguigenesis; skipped here)"
fi

echo "ALL OFFLINE SMOKES PASSED"
