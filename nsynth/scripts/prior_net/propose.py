#!/usr/bin/env python3
"""Prior-net tier-0 proposer bridge (ROADMAP Rung 9, Phase A).

v0 mode (one-shot, default): reads one problem JSON on stdin:
  {"n_scalar": 1, "examples": [{"array": [...], "scalars": [...], "expected": 7}, ...]}
writes one response JSON on stdout and exits.

v1 mode (--serve, persistent): loads the model ONCE, prints a ready line
  {"ready": true}
then loops: one request JSON per stdin line -> one response JSON per stdout
line (line-buffered). EOF on stdin terminates the loop. This removes the
per-problem subprocess + torch-import cost (~1-3 s) that made v0 net-negative.

Response JSON:
  {"proposals": [{"consts": [0,1,-1,c3,c4,c5],
                  "slots": [[op,s1,s2,cmp,gl,gr,el] x6],
                  "body_init": [b0..b3], "ret": r, "confidence": f}, ...],
   "confidence": f,        # argmax-proposal confidence (mean max-softmax)
   "gated": bool}          # true when confidence < --tau (proposals emptied)

Confidence (v1 gate): a per-head statistic aggregated across the 60 heads
after n_scalar masking, for the argmax proposal. The statistic is selected
by --signal (default: the best-AUC signal from
training/prior_net/calibrate.py — see confidence_calibration.json):
  mean_max     mean max-softmax probability
  mean_margin  mean (p_top1 - p_top2)
  mean_logp    mean log max-softmax
  min_max      min max-softmax across heads
Below --tau the server returns no proposals so the Rust caller pays only the
~ms inference cost and falls straight through to its cascade. The prior
proposes only when sure; search disposes.

Proposal 0 is the per-head argmax; the rest are temperature samples
(deterministic via --seed). Invalid pool/lip indices for the problem's
n_scalar are masked out before argmax/sampling, so every proposal is
structurally valid. On ANY per-request error this emits {"proposals": []}
and keeps serving — the Rust caller treats that as a miss (fail-soft).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_runtime(model_path: str):
    """Import torch + model lazily and load the checkpoint once."""
    import torch

    from prior_net_model import (
        CONST_VOCAB,
        HEAD_NAMES,
        HEAD_SIZES,
        N_ARR_BODY,
        N_ARR_SLOTS,
        N_HEADS,
        encode_problem,
        example_mask,
        head_valid_classes,
        load_checkpoint,
    )

    model = load_checkpoint(model_path, "cpu")
    ctx = {
        "torch": torch,
        "model": model,
        "CONST_VOCAB": CONST_VOCAB,
        "HEAD_NAMES": HEAD_NAMES,
        "HEAD_SIZES": HEAD_SIZES,
        "N_ARR_BODY": N_ARR_BODY,
        "N_ARR_SLOTS": N_ARR_SLOTS,
        "N_HEADS": N_HEADS,
        "encode_problem": encode_problem,
        "example_mask": example_mask,
        "head_valid_classes": head_valid_classes,
    }
    return ctx


def _gate_signal(probs: list, n_heads: int, signal: str) -> float:
    """Aggregate per-head argmax statistics into the calibrated gate signal."""
    import math

    tops = []
    for p in probs:
        top2 = p.topk(2)
        tops.append((float(top2.values[0]), float(top2.values[1])))
    if signal == "mean_margin":
        return sum(a - b for a, b in tops) / n_heads
    if signal == "mean_logp":
        return sum(math.log(max(a, 1e-9)) for a, _ in tops) / n_heads
    if signal == "min_max":
        return min(a for a, _ in tops)
    return sum(a for a, _ in tops) / n_heads  # mean_max


def _propose(ctx: dict, req: dict, k: int, temp: float, tau: float,
             signal: str = "mean_max") -> dict:
    """Run one inference and build the response dict (may raise)."""
    torch = ctx["torch"]
    model = ctx["model"]
    n_scalar = int(req["n_scalar"])
    examples = req["examples"]

    x = ctx["encode_problem"](examples, n_scalar).unsqueeze(0)
    m = ctx["example_mask"](len(examples)).unsqueeze(0)
    with torch.no_grad():
        flat = model(x, m)
    logits = [t.squeeze(0) for t in model.split_logits(flat)]

    # Mask classes invalid for this n_scalar (pool / lip heads).
    for h in range(ctx["N_HEADS"]):
        valid = ctx["head_valid_classes"](
            ctx["HEAD_NAMES"][h], ctx["HEAD_SIZES"][h], n_scalar
        )
        if valid < ctx["HEAD_SIZES"][h]:
            logits[h][valid:] = float("-inf")

    probs = [torch.softmax(l, dim=-1) for l in logits]

    def picks_confidence(picks: list[int]) -> float:
        """Mean probability the model assigns to the picked class per head
        (per-proposal diagnostic; the gate uses _gate_signal on the argmax)."""
        return float(
            sum(probs[h][picks[h]].item() for h in range(ctx["N_HEADS"]))
            / ctx["N_HEADS"]
        )

    def decode(picks: list[int], conf: float) -> dict:
        i = 0
        slots = []
        for _ in range(ctx["N_ARR_SLOTS"]):
            slots.append(picks[i : i + 7])
            i += 7
        body_init = picks[i : i + ctx["N_ARR_BODY"]]
        i += ctx["N_ARR_BODY"]
        ret = picks[i]
        i += 1
        consts = [0, 1, -1] + [ctx["CONST_VOCAB"][picks[i + j]] for j in range(3)]
        return {
            "consts": consts,
            "slots": slots,
            "body_init": body_init,
            "ret": ret,
            "confidence": round(conf, 4),
        }

    # Proposal 0: argmax. The calibrated signal over its heads is the gate.
    argmax_picks = [int(p.argmax().item()) for p in probs]
    confidence = _gate_signal(probs, ctx["N_HEADS"], signal)

    if confidence < tau:
        return {"proposals": [], "confidence": round(confidence, 4), "gated": True}

    proposals = [decode(argmax_picks, confidence)]
    seen = {tuple(argmax_picks)}
    attempts = 0
    while len(proposals) < k and attempts < k * 4:
        attempts += 1
        picks = [int(torch.multinomial(p, 1).item()) for p in
                 (torch.softmax(l / temp, dim=-1) for l in logits)]
        key = tuple(picks)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(decode(picks, picks_confidence(picks)))

    return {"proposals": proposals, "confidence": round(confidence, 4), "gated": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0,
                    help="confidence gate: below this, return no proposals")
    ap.add_argument("--signal", default="mean_max",
                    choices=["mean_max", "mean_margin", "mean_logp", "min_max"],
                    help="gate signal (must match the calibration that chose --tau)")
    ap.add_argument("--serve", action="store_true",
                    help="persistent mode: one request JSON per stdin line")
    args = ap.parse_args()

    if not args.serve:
        # v0 one-shot mode (kept for tests / manual probing).
        try:
            ctx = _load_runtime(args.model)
            ctx["torch"].manual_seed(args.seed)
            req = json.loads(sys.stdin.read())
            resp = _propose(ctx, req, args.k, args.temp, args.tau, args.signal)
            print(json.dumps(resp))
        except Exception as e:  # noqa: BLE001 — fail soft by contract
            print(f"[prior_net propose] error: {e}", file=sys.stderr)
            print(json.dumps({"proposals": []}))
        return 0

    # v1 persistent server: load once, then line-protocol until EOF.
    try:
        ctx = _load_runtime(args.model)
        ctx["torch"].manual_seed(args.seed)
    except Exception as e:  # noqa: BLE001
        print(f"[prior_net serve] fatal load error: {e}", file=sys.stderr)
        print(json.dumps({"ready": False, "error": str(e)}), flush=True)
        return 0
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = _propose(ctx, req, args.k, args.temp, args.tau, args.signal)
        except Exception as e:  # noqa: BLE001 — fail soft per request
            print(f"[prior_net serve] request error: {e}", file=sys.stderr)
            resp = {"proposals": [], "confidence": 0.0, "gated": False}
        print(json.dumps(resp), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
