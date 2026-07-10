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
    pub struct Problem { pub name: String, pub examples: Vec<Example> }
    impl Problem {
        pub fn function_name(&self) -> &str { if self.name.is_empty() { "f" } else { &self.name } }
    }
}
pub mod rt {
    use super::bench::Example;
    pub fn code_reproduces_examples(_code: &str, _examples: &[Example]) -> bool { false }
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
enum Reduce { Sum, Max, Min, Count }

impl Reduce {
    fn apply(self, arr: &[i64]) -> i64 {
        match self {
            Reduce::Sum => arr.iter().copied().fold(0i64, i64::saturating_add),
            Reduce::Max => arr.iter().copied().max().unwrap_or(0),
            Reduce::Min => arr.iter().copied().min().unwrap_or(0),
            Reduce::Count => arr.len() as i64,
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
    fn product_not_in_dsl() {
        let ex = vec![
            (vec![2, 3, 4], 24),
            (vec![5, 5], 25),
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
}
EOF
( cd "$TMP/utbus_reduce" && cargo test --lib -q )
echo "utbus_reduce: OK"

if [[ -f "$ROOT/src/schema_component.rs" ]]; then
  echo "== schema_component note =="
  echo "(full schema_component e2e needs mog_synth + linguigenesis; skipped here)"
fi

echo "ALL OFFLINE SMOKES PASSED"
