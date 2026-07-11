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
        assert_eq!(dual_len(&[]), 0);
        assert_eq!(dual_is_empty(&[]), 1);
        assert_eq!(k_kth_from_end(&[10, 20, 30, 40], 2), Some(30));
        assert_eq!(k_element_at(&[10, 20, 30], 1), Some(20));
        assert_eq!(index_middle(&[1, 2, 3]), Some(2));
        assert_eq!(index_second(&[10, 20, 30]), Some(20));
        assert_eq!(index_second_last(&[10, 20, 30]), Some(20));
        assert_eq!(index_max_even(&[1, 9, 3, 8, 2]), Some(3));
        assert_eq!(index_min_odd(&[1, 9, 3, 2]), Some(2));
        assert_eq!(k_count_eq(&[1, 5, 5, 2], 5), 2);
        assert_eq!(k_sum_gt(&[1, 5, 3, 2], 2), 8);
        assert_eq!(k_count_gt(&[1, 5, 3, 2], 2), 2);
        assert_eq!(k_sum_lt(&[1, 5, 3, 2], 3), 3);
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

fn take_first_two(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    let n = words.len().min(2);
    words[..n].join(sep)
}

fn drop_first_two(s: &str, sep: &str) -> String {
    let words: Vec<&str> = s.split(sep).collect();
    if words.len() <= 2 { String::new() } else { words[2..].join(sep) }
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
        assert_eq!(duplicate_each("a b", " "), "a a b b");
        assert_eq!(filter_len_eq2("to be or not", " "), "to be or");
        assert_eq!(filter_len_gt3("a to the moon", " "), "moon");
        assert_eq!(dedup_all("a b a c b", " "), "a b c");
        assert_eq!(sort_by_len("aaa b cc", " "), "b cc aaa");
        assert_eq!(take_first_two("one two three four", " "), "one two");
        assert_eq!(drop_first_two("one two three four", " "), "three four");
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
