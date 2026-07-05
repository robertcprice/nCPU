#!/usr/bin/env python3
"""Best-of-N evaluation of the LLM+nsynth pipeline (Phase 1 — inference only, no
training). Drives a local model (mlx_lm.server, OpenAI-compatible) to PROPOSE
tool-calls; nsynth verifies each and scores the first verified against the task's
held-out tests. This measures how far a given model closes the NL->PBE gap (agentic
NL baseline ~16% HumanEval; nsynth PBE ceiling ~38%).

Usage:
  # 1. serve a model (Apple-Silicon MLX):
  python3 -m mlx_lm server --model mlx-community/SmolLM3-3B-4bit --port 8765
  # 2. build the verifier tool:
  cargo build --release --bin nsynth_tool
  # 3. run:
  python3 scripts/bon_eval.py /tmp/he_bench.jsonl [N] [limit] [model] [url]

The model is UNTRUSTED: every proposal is verified by nsynth; a wrong proposal
scores 0. A SOLVED requires nsynth `verified` AND passing the hidden tests.
"""
import json, subprocess, sys, os, urllib.request

BIN = os.environ.get("NSYNTH_TOOL_BIN", "./target/release/nsynth_tool")
# POWERFUL path: the model WRITES Mog code (a Rust subset nsynth executes+verifies).
MOG_SYSTEM = (
    "Write a function in Mog for the description. Mog is a Rust subset the verifier "
    "runs, but NO `as` casts. NO `let` — declare `name: type = init;` (i64,f64,bool,"
    "string,[i64],[string]); mutate `x = expr;`; `for e in arr { }`; `while cond "
    "{ }`; `if cond { } else { }`; index `arr[i]` (i is i64, NO `as usize`); "
    "`arr.len()` returns i64 (NO cast); `arr.push(e)`; `+ - * / % == < > <= >= && "
    "|| !`; `return expr;`. Output ONLY `fn f(...) -> ... { ... }` — no prose, no fence."
)


def chat(url, model, prompt, n, temp):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": MOG_SYSTEM}, {"role": "user", "content": prompt}],
        "n": n, "temperature": temp, "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return [c["message"]["content"] for c in d.get("choices", [])]


def extract_json(s):
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i : j + 1])
    except Exception:
        return None


def verify(proposal, hidden, tmo):
    proposal = dict(proposal)
    proposal["hidden"] = hidden
    try:
        p = subprocess.run([BIN], input=json.dumps(proposal), capture_output=True, text=True, timeout=tmo)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"verdict": "refused", "reward": 0.0}


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "/tmp/he_bench.jsonl"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 154
    model = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("NSYNTH_LOCAL_LLM_MODEL", "local")
    url = sys.argv[5] if len(sys.argv) > 5 else os.environ.get(
        "NSYNTH_LOCAL_LLM_URL", "http://localhost:8765/v1/chat/completions"
    )

    tasks = [json.loads(l) for l in open(bench)][:limit]
    solved = tentative = 0
    for t in tasks:
        text = (t.get("text") or "").strip()
        hidden = t.get("examples", [])
        if not text or len(hidden) < 3:
            continue
        best = {"verdict": "refused", "reward": 0.0}
        try:
            cands = chat(url, model, text, n, 0.7)
        except Exception as e:
            print(f"[bon] model call failed: {e}", file=sys.stderr)
            cands = []
        for c in cands:
            # Strip a markdown fence if present, then wrap the model's Mog code as a
            # VerifyProgram proposal (nsynth executes+verifies it). If the model
            # instead emitted a JSON spec, honour that.
            code = c.strip()
            if code.startswith("```"):
                code = code.strip("`")
                code = code[code.find("fn "):] if "fn " in code else code
            prop = extract_json(c)
            if prop and prop.get("kind") in ("examples", "reference"):
                pass  # spec proposal
            elif "fn " in code:
                # nsynth verifies the model's code against the task's own tests.
                prop = {"kind": "verify", "code": code, "examples": hidden}
            else:
                continue
            r = verify(prop, hidden, 8)
            # first VERIFIED-and-correct wins (reward 1.0); else remember best.
            if r.get("verdict") == "verified" and r.get("reward", 0) >= 1.0:
                best = r
                break
            if r.get("reward", 0) > best.get("reward", 0):
                best = r
        if best.get("verdict") == "verified" and best.get("reward", 0) >= 1.0:
            solved += 1
        elif best.get("reward", 0) > 0:
            tentative += 1
    tot = len(tasks)
    print(f"[bon] best-of-{n} over {tot}: SOLVED={solved} ({100*solved/tot:.1f}%) "
          f"tentative/partial={tentative} | baseline NL~16%, PBE ceiling~38% (HumanEval)")


if __name__ == "__main__":
    main()
