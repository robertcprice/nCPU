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

if [[ -f "$ROOT/src/schema_component.rs" ]]; then
  echo "== schema_component note =="
  echo "(full schema_component e2e needs mog_synth + linguigenesis; skipped here)"
fi

echo "ALL OFFLINE SMOKES PASSED"
