#!/usr/bin/env python3
"""Prior-net tier-0 proposer bridge (ROADMAP Rung 9, Phase A, stage A3).

Reads one problem JSON on stdin:
  {"n_scalar": 1, "examples": [{"array": [...], "scalars": [...], "expected": 7}, ...]}

Writes one response JSON on stdout:
  {"proposals": [{"consts": [0,1,-1,c3,c4,c5],
                  "slots": [[op,s1,s2,cmp,gl,gr,el] x6],
                  "body_init": [b0..b3], "ret": r}, ...]}

Proposal 0 is the per-head argmax; the rest are temperature samples
(deterministic via --seed). Invalid pool/lip indices for the problem's
n_scalar are masked out before argmax/sampling, so every proposal is
structurally valid. On ANY error this prints {"proposals": []} and exits 0 —
the Rust caller treats that as a miss and continues its cascade (fail-soft).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torch

        from prior_net_model import (
            CONST_VOCAB,
            HEAD_NAMES,
            HEAD_SIZES,
            MAX_LIP,
            MAX_POOL,
            N_ARR_BODY,
            N_ARR_SLOTS,
            N_HEADS,
            encode_problem,
            example_mask,
            head_valid_classes,
            load_checkpoint,
        )

        torch.manual_seed(args.seed)
        req = json.loads(sys.stdin.read())
        n_scalar = int(req["n_scalar"])
        examples = req["examples"]

        model = load_checkpoint(args.model, "cpu")
        x = encode_problem(examples, n_scalar).unsqueeze(0)
        m = example_mask(len(examples)).unsqueeze(0)
        with torch.no_grad():
            flat = model(x, m)
        logits = [t.squeeze(0) for t in model.split_logits(flat)]

        # Mask classes invalid for this n_scalar (pool / lip heads).
        for h in range(N_HEADS):
            valid = head_valid_classes(HEAD_NAMES[h], HEAD_SIZES[h], n_scalar)
            if valid < HEAD_SIZES[h]:
                logits[h][valid:] = float("-inf")

        def decode(picks: list[int]) -> dict:
            i = 0
            slots = []
            for _ in range(N_ARR_SLOTS):
                slots.append(picks[i : i + 7])
                i += 7
            body_init = picks[i : i + N_ARR_BODY]
            i += N_ARR_BODY
            ret = picks[i]
            i += 1
            consts = [0, 1, -1] + [CONST_VOCAB[picks[i + j]] for j in range(3)]
            return {"consts": consts, "slots": slots, "body_init": body_init, "ret": ret}

        proposals = []
        seen = set()
        # Proposal 0: argmax.
        argmax_picks = [int(l.argmax().item()) for l in logits]
        proposals.append(decode(argmax_picks))
        seen.add(tuple(argmax_picks))
        # Proposals 1..k-1: temperature samples.
        attempts = 0
        while len(proposals) < args.k and attempts < args.k * 4:
            attempts += 1
            picks = [
                int(torch.multinomial(torch.softmax(l / args.temp, dim=-1), 1).item())
                for l in logits
            ]
            key = tuple(picks)
            if key in seen:
                continue
            seen.add(key)
            proposals.append(decode(picks))

        print(json.dumps({"proposals": proposals}))
        return 0
    except Exception as e:  # noqa: BLE001 — fail soft by contract
        print(f"[prior_net propose] error: {e}", file=sys.stderr)
        print(json.dumps({"proposals": []}))
        return 0


if __name__ == "__main__":
    sys.exit(main())
