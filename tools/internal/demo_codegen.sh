#!/usr/bin/env bash
# Practical code-generation demo.
#
# User supplies I/O examples; nsynth synthesizes a Mog program; the
# transpilers emit the same function in Python, Rust, and TypeScript;
# the Python version is then *executed* against the original examples to
# prove end-to-end correctness.
#
# This is the "is this powerful enough to be useful?" question answered
# as a script anyone can run.
#
# Usage:
#   tools/demo_codegen.sh                # runs the default abs_value demo
#   tools/demo_codegen.sh /path/to/spec.json
#
# The spec JSON shape (see bin/nsynth_codegen.rs for the full schema):
#   {"name": "my_fn", "signature": "fn my_fn(x: i64) -> i64",
#    "examples": [{"inputs":[1],"expected":2}, ...]}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/nsynth"

# Ensure the CLI is built.
if ! [[ -x target/release/nsynth_codegen ]]; then
  cargo build --release --bin nsynth_codegen > /dev/null 2>&1
fi

# Default spec: absolute value (common, easily verifiable).
SPEC_FILE="${1:-}"
if [[ -z "$SPEC_FILE" ]]; then
  SPEC_FILE=$(mktemp -t demo_codegen.XXXX.json)
  cat > "$SPEC_FILE" <<'EOF'
{
  "name": "abs_value",
  "signature": "fn abs_value(a: i64) -> i64",
  "examples": [
    {"inputs": [-5],  "expected": 5},
    {"inputs": [0],   "expected": 0},
    {"inputs": [7],   "expected": 7},
    {"inputs": [-12], "expected": 12},
    {"inputs": [-100], "expected": 100},
    {"inputs": [50],  "expected": 50}
  ]
}
EOF
  trap 'rm -f "$SPEC_FILE"' EXIT
fi

SPEC=$(cat "$SPEC_FILE")
NAME=$(echo "$SPEC" | sed -n 's/.*"name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)

OUT_DIR="$REPO_ROOT/artifacts/codegen_demo"
mkdir -p "$OUT_DIR"

echo "── nsynth_codegen demo: $NAME ──"
echo

# Emit all three languages side by side.
for lang in python rust typescript; do
  case "$lang" in
    python) EXT=py ;;
    rust) EXT=rs ;;
    typescript) EXT=ts ;;
  esac
  OUT="$OUT_DIR/${NAME}.${EXT}"
  ./target/release/nsynth_codegen --lang "$lang" --examples "$SPEC" --out "$OUT" 2>/dev/null
  echo "--- $lang ($OUT) ---"
  cat "$OUT"
  echo
done

# Python runtime verification: execute the synthesized function against
# the original examples and confirm every output matches. This is the
# most unambiguous "it works" signal — if the numbers line up, the
# synthesized function is *correct* for the given I/O set.
echo "── Runtime verification (Python) ──"
VERIFY_SCRIPT=$(mktemp -t demo_verify.XXXX.py)
trap "rm -f $VERIFY_SCRIPT" EXIT

# Extract examples as (inputs, expected) tuples the Python verifier can
# loop over. Using Python itself to parse the JSON avoids a jq dep.
python3 - "$SPEC_FILE" "$OUT_DIR/${NAME}.py" > "$VERIFY_SCRIPT" <<'PYEOF'
import json, sys, pathlib
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
code = pathlib.Path(sys.argv[2]).read_text()
print(code)
print()
print("# Auto-generated verification:")
print("checks = [")
for ex in spec["examples"]:
    print(f"    ({tuple(ex['inputs'])}, {ex['expected']}),")
print("]")
print(f"passed = 0")
print(f"failed = 0")
print(f"for inputs, expected in checks:")
print(f"    got = {spec['name']}(*inputs)")
print(f"    if got == expected:")
print(f"        passed += 1")
print(f"    else:")
print(f"        failed += 1")
print(f"        print(f'  ✗ {spec['name']}({{inputs}}) = {{got}}, expected {{expected}}')")
print(f"print(f'{{passed}}/{{passed+failed}} examples match')")
PYEOF

python3 "$VERIFY_SCRIPT"

echo
echo "── Done. Artifacts in $OUT_DIR ──"
