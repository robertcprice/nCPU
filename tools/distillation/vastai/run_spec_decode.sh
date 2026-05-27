#!/usr/bin/env bash
# Turn-key speculative-decoding benchmark on vast.ai.
#
# One command: provision RTX 4090 → sync repo + cache corpus → install
# vLLM → run cache-speculative decoding on the agent loop → pull the
# JSON results → destroy the instance.
#
# Prereqs (set in your shell):
#   export VAST_API_KEY=...              # from vast.ai/console/account
#   export ANTHROPIC_API_KEY=...         # for the cache-compare leg
#   ~/.ssh/id_rsa                  # registered with vast.ai
#
# Usage:
#   tools/vastai/run_spec_decode.sh <target> <corpus_tsv> [<problems_jsonl>]
#
# Example:
#   tools/vastai/run_spec_decode.sh qwen3.5-4b /tmp/retr_v2_corpus.tsv

set -euo pipefail

TARGET="${1:-qwen3.5-4b}"
CORPUS="${2:-}"
PROBLEMS="${3:-tools/benchmarks/humaneval_lite.jsonl}"

if [[ -z "$CORPUS" || ! -f "$CORPUS" ]]; then
  echo "usage: $0 <target> <corpus_tsv> [<problems_jsonl>]" >&2
  echo "  corpus_tsv must be an existing cache file with 6-col rows" >&2
  exit 2
fi

command -v vastai >/dev/null 2>&1 || { echo "vastai CLI required" >&2; exit 2; }
vastai show user --raw >/dev/null 2>&1 || { echo "vastai not authed. run 'vastai set api-key <key>' or export VAST_API_KEY" >&2; exit 2; }
[[ -f ~/.ssh/id_rsa ]] || { echo "~/.ssh/id_rsa missing" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "[spec-decode] launching vast.ai instance for $TARGET..."
"$SCRIPT_DIR/launch.sh" "$TARGET" 2>&1 | tee /tmp/vastai_launch.log
INSTANCE_ID=$(grep -oE 'instance [0-9]+' /tmp/vastai_launch.log | head -1 | awk '{print $2}')

if [[ -z "$INSTANCE_ID" ]]; then
  echo "[spec-decode] failed to provision; check /tmp/vastai_launch.log" >&2
  exit 1
fi
echo "[spec-decode] instance $INSTANCE_ID provisioned"

# Always destroy on exit — even if something below fails. Without this
# a failed mid-script exit leaks a running instance that keeps billing.
cleanup() {
  local ec=$?
  echo "[spec-decode] destroying instance $INSTANCE_ID (exit $ec)..."
  vastai destroy instance "$INSTANCE_ID" 2>&1 || true
}
trap cleanup EXIT

INFO=$(vastai show instance "$INSTANCE_ID" --raw)
HOST=$(echo "$INFO" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["ssh_host"])')
PORT=$(echo "$INFO" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["ssh_port"])')

# launch.sh only verifies the API reports an ssh endpoint, not that
# sshd is actually listening. Poll for real connectivity before rsync.
echo "[spec-decode] waiting for sshd on $HOST:$PORT (up to 12 min)..."
for attempt in $(seq 1 144); do
  if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
         -i ~/.ssh/id_rsa -p "$PORT" "root@$HOST" true 2>/dev/null; then
    echo "[spec-decode] sshd up after ${attempt} tries"
    break
  fi
  if (( attempt == 144 )); then
    echo "[spec-decode] sshd never came up on $HOST:$PORT" >&2
    exit 1
  fi
  sleep 5
done

echo "[spec-decode] packaging minimal tarball (5 files)..."
# Ship only what the benchmark needs. The full repo is 7GB; we only
# need ~20KB of Python and the corpus TSV. This avoids a ~30-min
# rsync over a slow uplink.
TAR=/tmp/spec_decode_payload.tar.gz
( cd "$REPO_ROOT" && tar czf "$TAR" \
    tools/inference/vllm_cache_speculative.py \
    tools/benchmarks/humaneval_lite.jsonl \
    tools/benchmarks/retrieval_prompt.py \
    tools/benchmarks/semantic_cache.py \
    tools/benchmarks/llm_solution_cache.py )
ls -la "$TAR"

echo "[spec-decode] uploading tarball + corpus..."
SSH_OPTS="-P $PORT -i $HOME/.ssh/id_rsa -o StrictHostKeyChecking=no"
scp $SSH_OPTS "$TAR" "root@$HOST:/tmp/spec_decode_payload.tar.gz"
scp $SSH_OPTS "$CORPUS" "root@$HOST:/tmp/cache_corpus.tsv"

echo "[spec-decode] running spec-decode benchmark on remote..."
# The remote runs the stub (corpus sanity) then a real A/B vLLM
# measurement: plain autoregressive vs. cache-hint-in-prompt.
# All output is cat'd back to stdout so the measurement survives
# even if pull_artifacts fails.
ssh -p "$PORT" -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no \
    "root@$HOST" bash <<'REMOTE'
set -ux
mkdir -p /tmp/nsynth/artifacts
cd /tmp/nsynth
tar xzf /tmp/spec_decode_payload.tar.gz

# The pytorch base image's conda env has a corrupted `requests` dist-info
# that kills any pip install. Repair it first with --force-reinstall, then
# install vllm directly into system python (avoids venv overhead).
if ! python3 -c "import vllm" 2>/dev/null; then
  echo "[remote] repairing pip env + installing vllm..."
  pip install -q --force-reinstall --no-deps requests 2>&1 | tail -1
  # Clear the corrupted dist-info if it's still around.
  rm -rf /opt/conda/lib/python3.10/site-packages/requests-2.31.0.dist-info 2>/dev/null || true
  pip install -q vllm 2>&1 | tail -3
fi
export PYBIN=python3
$PYBIN -c "import vllm; print('[remote] vllm', vllm.__version__)"
export NSYNTH_LLM_CACHE_PATH=/tmp/cache_corpus.tsv

echo "=== BEGIN STUB ==="
$PYBIN tools/inference/vllm_cache_speculative.py \
    --stub \
    --problems tools/benchmarks/humaneval_lite.jsonl \
    --corpus $NSYNTH_LLM_CACHE_PATH \
    | tee artifacts/spec_decode_stub.txt
echo "=== END STUB ==="

echo "=== BEGIN SPEC_DECODE_AB ==="
$PYBIN - <<'PYEOF' 2>&1 | tee artifacts/spec_decode_ab.txt
import json, sys, time
sys.path.insert(0, "tools/benchmarks")
import os
os.environ["NSYNTH_LLM_CACHE_PATH"] = "/tmp/cache_corpus.tsv"
from retrieval_prompt import build_retrieval_prefix

from vllm import LLM, SamplingParams
problems = [json.loads(l) for l in
            open("tools/benchmarks/humaneval_lite.jsonl").read().splitlines() if l.strip()]

# Subset for speed: 10 problems is enough to see a timing delta.
problems = problems[:10]
print(f"[ab] {len(problems)} problems")

llm = LLM(model="Qwen/Qwen3-4B-Instruct-2507", dtype="bfloat16",
          max_model_len=2048, gpu_memory_utilization=0.85)
sp = SamplingParams(temperature=0.0, max_tokens=256)

def base_prompt(p):
    ex = "\n".join(f"  {p['name']}({', '.join(repr(x) for x in e['inputs'])}) == {e['expected']}"
                   for e in p["examples"])
    return (f"Write a Python function matching `{p['signature']}`.\n\n"
            f"Examples:\n{ex}\n\nReply with ONLY the function definition.")

# Round 1: plain autoregressive.
plain_prompts = [base_prompt(p) for p in problems]
t0 = time.time()
outs_plain = llm.generate(plain_prompts, sp)
t_plain = time.time() - t0

# Round 2: cache-hint prompt (retrieval prefix).
hinted_prompts = []
for p in problems:
    pref = build_retrieval_prefix(p["examples"], k=1, min_similarity=0.70)
    hinted_prompts.append(pref + base_prompt(p))

t0 = time.time()
outs_hint = llm.generate(hinted_prompts, sp)
t_hint = time.time() - t0

# Count accepted tokens approximation: how many tokens plain vs hint;
# with cache-hint we expect shorter generations (model completes faster
# because the answer is already suggested).
tok_plain = sum(len(o.outputs[0].token_ids) for o in outs_plain)
tok_hint = sum(len(o.outputs[0].token_ids) for o in outs_hint)

print(json.dumps({
    "n_problems": len(problems),
    "plain_wall_s": round(t_plain, 3),
    "hint_wall_s": round(t_hint, 3),
    "plain_tokens": tok_plain,
    "hint_tokens": tok_hint,
    "speedup": round(t_plain / max(t_hint, 1e-9), 3),
    "token_ratio": round(tok_hint / max(tok_plain, 1), 3),
}, indent=2))
PYEOF
echo "=== END SPEC_DECODE_AB ==="
REMOTE

echo "[spec-decode] (tee already captured all output above; skipping pull)"
# The STUB + AB blocks tee to stdout which this script already logged.
# No separate pull needed — simpler, more reliable.

echo "[spec-decode] done."
# trap cleanup() destroys the instance on exit.
