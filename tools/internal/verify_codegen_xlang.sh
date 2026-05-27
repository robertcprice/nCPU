#!/usr/bin/env bash
# Cross-language runtime verifier.
#
# Takes a HumanEval-lite-style spec.json, synthesizes the function in
# Python/Rust/TypeScript via nsynth_codegen, then *runs each version*
# against the test_cases. Reports pass/fail per language so the generated
# code is proven correct in the wild, not just shape-tested.
#
# Required interpreters (graceful skip when missing):
#   - python3 (always required)
#   - rustc + a temp cargo-less single-file setup (rustc direct)
#   - ts-node OR bun OR deno for TypeScript
#
# Usage:
#   tools/verify_codegen_xlang.sh <spec.json>
#   tools/verify_codegen_xlang.sh          # uses the default double spec

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/nsynth"

if ! [[ -x target/release/nsynth_codegen ]]; then
  cargo build --release --bin nsynth_codegen > /dev/null 2>&1
fi

SPEC_FILE="${1:-}"
if [[ -z "$SPEC_FILE" ]]; then
  SPEC_FILE=$(mktemp -t verify_xlang.XXXX.json)
  cat > "$SPEC_FILE" <<'EOF'
{"name":"double","signature":"fn double(a: i64) -> i64",
 "examples":[{"inputs":[0],"expected":0},{"inputs":[3],"expected":6},
             {"inputs":[-2],"expected":-4},{"inputs":[10],"expected":20},
             {"inputs":[5],"expected":10},{"inputs":[-1],"expected":-2}],
 "test_cases":[[4,8],[7,14],[0,0],[-10,-20]]}
EOF
  trap "rm -f $SPEC_FILE" EXIT
fi

NAME=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$SPEC_FILE")
SPEC=$(cat "$SPEC_FILE")

OUT_DIR="$REPO_ROOT/artifacts/codegen_xlang/${NAME}"
mkdir -p "$OUT_DIR"

echo "── cross-language verify: $NAME ──"

# Generate all three languages.
./target/release/nsynth_codegen --lang python     --examples "$SPEC" --out "$OUT_DIR/${NAME}.py" 2>/dev/null
./target/release/nsynth_codegen --lang rust       --examples "$SPEC" --out "$OUT_DIR/${NAME}.rs" 2>/dev/null
./target/release/nsynth_codegen --lang typescript --examples "$SPEC" --out "$OUT_DIR/${NAME}.ts" 2>/dev/null

# ─── Python ────────────────────────────────────────────────────────────────
echo
echo "--- Python ---"
PY_SCRIPT=$(mktemp -t verify_xlang_py.XXXX.py)
trap "rm -f $PY_SCRIPT $SPEC_FILE" EXIT
python3 - "$SPEC_FILE" "$OUT_DIR/${NAME}.py" > "$PY_SCRIPT" <<'PYEOF'
import json, sys, pathlib
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(pathlib.Path(sys.argv[2]).read_text())
print()
print("passed = 0; failed = 0")
for case in spec["test_cases"]:
    args, exp = case[:-1], case[-1]
    print(f"got = {spec['name']}({', '.join(repr(a) for a in args)})")
    print(f"if got == {exp}: passed += 1")
    print(f"else: failed += 1; print(f'  ✗ {args} got {{got}} exp {exp}')")
print(f"print(f'python: {{passed}}/{{passed+failed}} pass')")
PYEOF
python3 "$PY_SCRIPT" || echo "python: exec-error"

# ─── Rust ──────────────────────────────────────────────────────────────────
echo
echo "--- Rust ---"
if command -v rustc >/dev/null 2>&1; then
  RS_FILE=$(mktemp -d -t verify_xlang_rs.XXXX)
  RS_SRC="$RS_FILE/main.rs"
  # Compose a main() that invokes the function on each test case.
  python3 - "$SPEC_FILE" "$OUT_DIR/${NAME}.rs" > "$RS_SRC" <<'RSEOF'
import json, sys, pathlib
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
name = spec["name"]
print(pathlib.Path(sys.argv[2]).read_text())
print()
print("fn main() {")
print("    let mut passed = 0i64;")
print("    let mut failed = 0i64;")
for case in spec["test_cases"]:
    args, exp = case[:-1], case[-1]
    args_str = ", ".join(str(a) + "_i64" for a in args)
    print(f"    let got = {name}({args_str});")
    print(f"    if got == {exp}_i64 {{ passed += 1; }} else {{ failed += 1; eprintln!(\"  ✗ got {{}} exp {exp}\", got); }}")
print(f"    println!(\"rust: {{}}/{{}} pass\", passed, passed + failed);")
print("}")
RSEOF
  if rustc "$RS_SRC" -o "$RS_FILE/bin" 2>/dev/null; then
    "$RS_FILE/bin"
  else
    echo "rust: compile-error"
  fi
  rm -rf "$RS_FILE"
else
  echo "rust: rustc not installed — skipped"
fi

# ─── TypeScript ────────────────────────────────────────────────────────────
echo
echo "--- TypeScript ---"
TS_RUNNER=""
if command -v bun >/dev/null 2>&1; then TS_RUNNER="bun run"
elif command -v deno >/dev/null 2>&1; then TS_RUNNER="deno run --allow-read"
elif command -v ts-node >/dev/null 2>&1; then TS_RUNNER="ts-node --esm"
fi
if [[ -n "$TS_RUNNER" ]]; then
  TS_FILE=$(mktemp -t verify_xlang_ts.XXXX.ts)
  python3 - "$SPEC_FILE" "$OUT_DIR/${NAME}.ts" > "$TS_FILE" <<'TSEOF'
import json, sys, pathlib
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
name = spec["name"]
print(pathlib.Path(sys.argv[2]).read_text())
print()
print("let passed = 0; let failed = 0;")
for case in spec["test_cases"]:
    args, exp = case[:-1], case[-1]
    args_str = ", ".join(str(a) for a in args)
    print(f"let got = {name}({args_str});")
    print(f"if (got === {exp}) passed += 1; else {{ failed += 1; console.log(`  ✗ got ${{got}} exp {exp}`); }}")
print(f"console.log(`typescript: ${{passed}}/${{passed+failed}} pass`);")
TSEOF
  $TS_RUNNER "$TS_FILE" 2>&1 || echo "typescript: exec-error"
  rm -f "$TS_FILE"
else
  echo "typescript: no runner (bun/deno/ts-node) — skipped"
fi

echo
echo "── artifacts in $OUT_DIR ──"
