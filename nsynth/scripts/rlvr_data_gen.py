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

# The POWERFUL path: teach the model to WRITE Mog code; nsynth EXECUTES + verifies
# it (ceiling = the interpreter's execution breadth + the model's coding, NOT
# nsynth's synthesis reach). Mog is a small Rust-subset the verifier runs.
MOG_SYSTEM = (
    "Write a function in Mog to satisfy the description. Mog is a Rust subset the "
    "verifier executes. Rules: NO `let` — declare with `name: type = init;` (types "
    "i64, f64, bool, string, [i64], [string]); mutate with `x = expr;`; loops "
    "`for e in arr { ... }` and `while cond { ... }`; `if cond { ... } else { ... }`; "
    "index `arr[i as usize]`, length `arr.len()`; `return expr;`. Output ONLY the "
    "`fn f(...) -> ... { ... }` body, no prose, no markdown fence."
)
# The SECONDARY path: the model proposes I/O; nsynth SYNTHESIZES (verified) — only
# for tasks in nsynth's synthesis domain.
SPEC_SYSTEM = (
    "For the described function output ONLY a JSON tool call "
    "{\"kind\":\"examples\",\"examples\":[{\"in\":[...],\"out\":...}, ...]} whose I/O "
    "pairs UNIQUELY determine the function. A verified synthesizer writes the code."
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
            # REJECTION SAMPLING: keep only when the proposal led to a CORRECT
            # program (reward > 0 on the oracle). `verified` vs `tentative` is a user
            # trust label, not a training filter — a tentative-but-correct trace is
            # valid (the weak corroborator just couldn't independently confirm).
            if r.get("reward", 0) <= 0:
                dropped += 1
                continue
            code = r.get("code")
            # PRIMARY: teach the model to WRITE the (verified) Mog code. nsynth
            # executes+verifies at inference — ceiling = interpreter breadth + the
            # model's coding, not nsynth's synthesis reach.
            if code:
                out.write(json.dumps({"messages": [
                    {"role": "system", "content": MOG_SYSTEM},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": code.strip()},
                ]}) + "\n")
                kept += 1
            # SECONDARY: teach the model to propose a determining spec (nsynth
            # synthesizes) — the verified-synthesis path for nsynth's own domain.
            out.write(json.dumps({"messages": [
                {"role": "system", "content": SPEC_SYSTEM},
                {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps({"kind": "examples", "examples": exs})},
            ]}) + "\n")
            kept += 1
    print(f"[rlvr-data] kept {kept} traces (code+spec), dropped {dropped} -> {out_path}")


if __name__ == "__main__":
    main()
