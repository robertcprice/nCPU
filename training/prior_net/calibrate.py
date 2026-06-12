#!/usr/bin/env python3
"""Calibrate the prior-net confidence gate (Phase A v1).

Reproduces train.py's exact held-out split (same seed, same randperm), runs
the trained net on the held-out rows, and for each row computes:

  - confidence: mean max-softmax probability across the 60 heads after
    n_scalar masking (identical to the gate signal propose.py --serve emits)
  - exact: whether the per-head argmax reproduces the row's full discrete
    description (the conservative proxy for "proposal verifies verbatim" —
    distinct descriptions can still emit identical Mog code, so true
    verbatim-hit rate is >= this)

Then sweeps thresholds and reports precision (P(exact | conf >= tau)),
fire rate, and recall, and picks the calibrated tau: the smallest threshold
whose held-out precision >= --target-precision. Writes
training/prior_net/confidence_calibration.json.

Usage:
  python3 training/prior_net/calibrate.py \
      --data nsynth/data/prior_net_train.jsonl \
      --model training/prior_net/prior_net_v0.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "nsynth" / "scripts" / "prior_net"))

from prior_net_model import (  # noqa: E402
    HEAD_NAMES,
    HEAD_SIZES,
    N_HEADS,
    encode_problem,
    example_mask,
    head_valid_classes,
    labels_from_desc,
    load_checkpoint,
)

from train import load_dataset  # noqa: E402


@torch.no_grad()
def confidences_and_exact(model, x, m, y, n_scalars, device, batch=1024):
    """Per-row gate confidence + argmax-exact flag, n_scalar-masked."""
    model.eval()
    confs, exacts = [], []
    n = x.size(0)
    # Precompute per-head valid-class counts for each n_scalar value.
    valid = {
        ns: [head_valid_classes(HEAD_NAMES[h], HEAD_SIZES[h], ns) for h in range(N_HEADS)]
        for ns in (0, 1, 2)
    }
    for i in range(0, n, batch):
        xb = x[i : i + batch].to(device)
        mb = m[i : i + batch].to(device)
        yb = y[i : i + batch]
        nsb = n_scalars[i : i + batch]
        flat = model(xb, mb)
        logits = model.split_logits(flat)
        bsz = xb.size(0)
        conf_sum = torch.zeros(bsz)
        row_ok = torch.ones(bsz, dtype=torch.bool)
        for h in range(N_HEADS):
            lh = logits[h].cpu().clone()
            # n_scalar masking, rowwise (matches propose.py).
            for ns in (0, 1, 2):
                v = valid[ns][h]
                if v < HEAD_SIZES[h]:
                    rows = nsb == ns
                    lh[rows, v:] = float("-inf")
            p = torch.softmax(lh, dim=-1)
            pmax, pred = p.max(dim=-1)
            conf_sum += pmax
            row_ok &= pred == yb[:, h]
        confs.extend((conf_sum / N_HEADS).tolist())
        exacts.extend(row_ok.tolist())
    return confs, exacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "nsynth/data/prior_net_train.jsonl"))
    ap.add_argument("--model", default=str(PROJECT_ROOT / "training/prior_net/prior_net_v0.pt"))
    ap.add_argument("--holdout", type=int, default=5000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-precision", type=float, default=0.35)
    # Utility model (measured, Phase A v1): a verbatim hit saves the mean
    # cascade solve time on the 16-problem direct-fallback head-to-head
    # (v0 OFF column, 83.3s/16 = 5.2s); a gate-open miss costs one server
    # round-trip (~12ms median) + K=4 zero-step verifies (~60ms total
    # measured); a warm refine costs ~0.4s (<=120 Adam steps, from v0's
    # per-miss decomposition).
    ap.add_argument("--save-per-hit", type=float, default=5.2)
    ap.add_argument("--cost-zero-miss", type=float, default=0.06)
    ap.add_argument("--cost-warm", type=float, default=0.4)
    ap.add_argument("--out", default=str(Path(__file__).parent / "confidence_calibration.json"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    x, m, y, n_bad = load_dataset(Path(args.data), args.max_rows)
    # n_scalar per row: recover from the one-hot tail of the first example's
    # features is fragile; re-read from JSONL instead (cheap second pass).
    n_scalars = []
    with Path(args.data).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                n_scalars.append(int(json.loads(line)["n_scalar"]))
            except Exception:
                continue
            if args.max_rows and len(n_scalars) >= args.max_rows:
                break
    n_scalars = torch.tensor(n_scalars[: x.size(0)], dtype=torch.long)
    print(f"[calibrate] loaded {x.size(0)} rows ({n_bad} bad) in {time.time()-t0:.1f}s")

    # EXACT same split as train.py: one randperm with the same generator seed.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(x.size(0), generator=g)
    x, m, y, n_scalars = x[perm], m[perm], y[perm], n_scalars[perm]
    holdout = min(args.holdout, x.size(0) // 5)
    xv, mv, yv, nsv = x[-holdout:], m[-holdout:], y[-holdout:], n_scalars[-holdout:]
    print(f"[calibrate] held-out rows: {xv.size(0)}")

    model = load_checkpoint(args.model, device)
    confs, exacts = confidences_and_exact(model, xv, mv, yv, nsv, device)
    confs_t = torch.tensor(confs)
    exacts_t = torch.tensor(exacts)
    base_rate = exacts_t.float().mean().item()
    print(f"[calibrate] base exact rate {base_rate:.4f}; "
          f"conf mean {confs_t.mean():.3f} max {confs_t.max():.3f}")

    sweep = []
    chosen = None
    for tau_i in range(30, 100, 2):
        tau = tau_i / 100.0
        fired = confs_t >= tau
        n_fired = int(fired.sum())
        if n_fired == 0:
            sweep.append({"tau": tau, "fire_rate": 0.0, "precision": None,
                          "recall": 0.0, "n_fired": 0})
            continue
        hits = int((exacts_t & fired).sum())
        precision = hits / n_fired
        recall = hits / max(int(exacts_t.sum()), 1)
        sweep.append({
            "tau": tau,
            "fire_rate": round(n_fired / len(confs), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_fired": n_fired,
            "n_hits": hits,
        })
        if chosen is None and precision >= args.target_precision:
            chosen = tau

    # Confidence distribution split by exact/not (for the report).
    q = lambda t, qs: [round(torch.quantile(t, qq).item(), 4) for qq in qs]
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    dist = {
        "exact_conf_quantiles": q(confs_t[exacts_t], quantiles) if exacts_t.any() else None,
        "miss_conf_quantiles": q(confs_t[~exacts_t], quantiles),
        "quantiles": quantiles,
    }

    report = {
        "model": args.model,
        "data": args.data,
        "holdout_rows": int(xv.size(0)),
        "seed": args.seed,
        "base_exact_rate": round(base_rate, 4),
        "target_precision": args.target_precision,
        "chosen_tau": chosen,
        "sweep": sweep,
        "distribution": dist,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[calibrate] chosen tau={chosen} (target precision {args.target_precision})")
    print(f"[calibrate] wrote {args.out}")
    for row in sweep:
        if row["precision"] is not None and row["fire_rate"] > 0:
            print(f"  tau={row['tau']:.2f} fire={row['fire_rate']:.3f} "
                  f"prec={row['precision']:.3f} recall={row['recall']:.3f}")


if __name__ == "__main__":
    main()
