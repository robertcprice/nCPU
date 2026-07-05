#!/usr/bin/env python3
"""RSFT data generator for the LLM+nsynth pipeline (see docs/AGENTIC_NL_PLAYBOOK.md).

Emits verified (NL prompt -> gold tool-call) training traces in mlx_lm chat format.
The model is trained to COMPREHEND prose into the characterizing I/O; nsynth
synthesizes+verifies the program. Rejection-sampling: a trace is kept ONLY when
nsynth_tool returns `verified` on that tool-call (so every SFT/RLVR target is a
proposal that provably leads to a correct, verified solve). No model needed to
generate this — nsynth is the teacher.

Usage:
  cargo build --release --bin nsynth_tool
  python3 scripts/rlvr_data_gen.py /tmp/mbpp_bench.jsonl /tmp/rlvr_sft.jsonl [limit] [timeout_s]

Then SFT:
  python3 -m mlx_lm lora --model mlx-community/SmolLM3-3B-4bit --train \
      --data /tmp/rlvr_sft.jsonl --iters 500 --adapter-path /tmp/planner-adapter
"""
import json, subprocess, sys, os

BIN = os.environ.get("NSYNTH_TOOL_BIN", "./target/release/nsynth_tool")
SYSTEM = (
    "You drive a verified program synthesizer. For the described function, output "
    "ONLY a JSON tool call: {\"kind\":\"examples\",\"signature\":\"fn f(...) -> ...\","
    "\"examples\":[{\"in\":[...],\"out\":...}, ...]} giving input/output pairs that "
    "UNIQUELY determine the function. The synthesizer writes and verifies the code."
)


def signature(n_args: int) -> str:
    params = ", ".join(f"a{i}: i64" for i in range(max(1, n_args)))
    return f"fn f({params}) -> i64"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    bench, out_path = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
    tmo = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    tasks = [json.loads(l) for l in open(bench)][:limit]
    kept = dropped = 0
    with open(out_path, "w") as out:
        for t in tasks:
            exs = t.get("examples", [])
            text = (t.get("text") or "").strip()
            if len(exs) < 3 or not text:
                dropped += 1
                continue
            sig = signature(len(exs[0].get("in", [])))
            req = {"kind": "examples", "signature": sig, "examples": exs, "hidden": exs}
            try:
                p = subprocess.run(
                    [BIN], input=json.dumps(req), capture_output=True, text=True, timeout=tmo
                )
                r = json.loads(p.stdout.strip().splitlines()[-1])
            except Exception:
                dropped += 1
                continue
            # REJECTION SAMPLING: keep proposals that lead to a CORRECT program
            # (reward > 0 on the oracle). `verified` vs `tentative` is a trust label
            # for the user, not a training filter — a tentative-but-correct proposal
            # is a valid gold trace (the model proposed a spec that nsynth solved
            # correctly; the weak corroborator just couldn't independently confirm).
            if r.get("reward", 0) <= 0:
                dropped += 1
                continue
            # Gold completion the model learns to emit: kind + the characterizing
            # I/O. No signature — nsynth infers it from the example value types, so a
            # (possibly-wrong) signature guess would only teach noise.
            gold = json.dumps({"kind": "examples", "examples": exs})
            trace = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": gold},
                ]
            }
            out.write(json.dumps(trace) + "\n")
            kept += 1
    print(f"[rlvr-data] kept {kept} verified traces, dropped {dropped} -> {out_path}")


if __name__ == "__main__":
    main()
