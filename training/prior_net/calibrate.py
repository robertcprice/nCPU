#!/usr/bin/env python3
"""Calibrate the prior-net confidence gate (Phase A v1).

Reproduces train.py's exact held-out split (same seed, same randperm), runs
the trained net on the held-out rows, and for each row computes candidate
gate signals over the 60 n_scalar-masked heads:

  - mean_max:    mean max-softmax probability      (v0's signal)
  - mean_margin: mean (p_top1 - p_top2)            (margin)
  - mean_logp:   mean log max-softmax              (log geometric mean)
  - min_max:     min max-softmax across heads      (weakest-head)

and `exact`: whether the per-head argmax reproduces the row's full discrete
description (the conservative proxy for "proposal verifies verbatim" —
distinct descriptions can still emit identical or example-satisfying Mog
code, so true verbatim-hit rate is >= this).

Signal selection: rank-AUC of exact-vs-miss for each signal; the sweep and
threshold rules run on the best-AUC signal.

Threshold rules (both reported):
  - utility: tau maximizing expected seconds saved per fallback problem
    under the measured cost model (--save-per-hit / --cost-zero-miss /
    --cost-warm). On a distribution where precision never clears the
    break-even rate this rule yields "never fire" — reported honestly.
  - hit_recall (DEPLOYMENT RULE — chosen_tau): the largest tau that keeps
    >= --hit-recall of held-out exact hits firing (default 0.9, i.e. tau =
    q10 of the exact rows' signal). The holdout's base exact rate (~1-2%,
    random generated programs) badly understates the bench fallback
    population's hit rate (12.5% measured in v0), so utility-on-holdout is
    a lower-bound diagnostic, not the decision signal (it degenerates to an
    extreme-tail tau that fires on ~nothing). The gate's job is to keep
    true hits while cutting the bulk of confident-miss overhead.

Optionally (--bench-requests) runs the model on real bench request rows
(dumped by the ignored Rust test `dump_bench_fallback_requests`) and
reports each problem's gate signal and fire/gated status at the chosen tau
— the deployment-distribution sanity check.

Writes training/prior_net/confidence_calibration.json.

Usage:
  python3 training/prior_net/calibrate.py \
      --data nsynth/data/prior_net_train_300k.jsonl \
      --model training/prior_net/prior_net_v1.pt \
      --bench-requests /tmp/prior_bench_requests.jsonl
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prior_net_model import (  # noqa: E402
    HEAD_NAMES,
    HEAD_SIZES,
    N_HEADS,
    encode_problem,
    example_mask,
    head_valid_classes,
    load_checkpoint,
)

from train import load_dataset  # noqa: E402

SIGNALS = ("mean_max", "mean_margin", "mean_logp", "min_max")


def masked_probs(logits: list[torch.Tensor], n_scalars: torch.Tensor, valid: dict):
    """Yield per-head softmax with n_scalar masking applied rowwise."""
    for h in range(N_HEADS):
        lh = logits[h].cpu().clone()
        for ns in (0, 1, 2):
            v = valid[ns][h]
            if v < HEAD_SIZES[h]:
                rows = n_scalars == ns
                lh[rows, v:] = float("-inf")
        yield h, torch.softmax(lh, dim=-1)


@torch.no_grad()
def signals_and_exact(model, x, m, y, n_scalars, device, batch=1024):
    """Per-row gate signals + argmax-exact flag, n_scalar-masked."""
    model.eval()
    out = {s: [] for s in SIGNALS}
    exacts = []
    n = x.size(0)
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
        max_sum = torch.zeros(bsz)
        margin_sum = torch.zeros(bsz)
        logp_sum = torch.zeros(bsz)
        min_max = torch.ones(bsz)
        row_ok = torch.ones(bsz, dtype=torch.bool)
        for h, p in masked_probs(logits, nsb, valid):
            top2 = p.topk(2, dim=-1).values
            pmax = top2[:, 0]
            max_sum += pmax
            margin_sum += pmax - top2[:, 1]
            logp_sum += torch.log(pmax.clamp_min(1e-9))
            min_max = torch.minimum(min_max, pmax)
            row_ok &= p.argmax(dim=-1) == yb[:, h]
        out["mean_max"].extend((max_sum / N_HEADS).tolist())
        out["mean_margin"].extend((margin_sum / N_HEADS).tolist())
        out["mean_logp"].extend((logp_sum / N_HEADS).tolist())
        out["min_max"].extend(min_max.tolist())
        exacts.extend(row_ok.tolist())
    return {s: torch.tensor(v) for s, v in out.items()}, torch.tensor(exacts)


def rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg)."""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Rank-based (handles ties as 0.5).
    greater = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
    equal = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def sweep_signal(conf: torch.Tensor, exact: torch.Tensor, n_steps: int = 60):
    lo, hi = float(conf.min()), float(conf.max())
    rows = []
    for i in range(n_steps + 1):
        tau = lo + (hi - lo) * i / n_steps
        fired = conf >= tau
        n_fired = int(fired.sum())
        if n_fired == 0:
            rows.append({"tau": round(tau, 4), "fire_rate": 0.0, "precision": None,
                         "recall": 0.0, "n_fired": 0, "n_hits": 0})
            continue
        hits = int((exact & fired).sum())
        rows.append({
            "tau": round(tau, 4),
            "fire_rate": round(n_fired / len(conf), 4),
            "precision": round(hits / n_fired, 4),
            "recall": round(hits / max(int(exact.sum()), 1), 4),
            "n_fired": n_fired,
            "n_hits": hits,
        })
    return rows


@torch.no_grad()
def bench_signals(model, requests: list[dict], device) -> list[dict]:
    """Gate signals for real bench request rows (deployment sanity check)."""
    rows = []
    for req in requests:
        n_scalar = int(req["n_scalar"])
        x = encode_problem(req["examples"], n_scalar).unsqueeze(0).to(device)
        m = example_mask(len(req["examples"])).unsqueeze(0).to(device)
        flat = model(x, m)
        logits = [t.squeeze(0).cpu() for t in model.split_logits(flat)]
        sig = {s: 0.0 for s in SIGNALS}
        sig["min_max"] = 1.0
        for h in range(N_HEADS):
            v = head_valid_classes(HEAD_NAMES[h], HEAD_SIZES[h], n_scalar)
            lh = logits[h].clone()
            if v < HEAD_SIZES[h]:
                lh[v:] = float("-inf")
            p = torch.softmax(lh, dim=-1)
            top2 = p.topk(2).values
            sig["mean_max"] += float(top2[0])
            sig["mean_margin"] += float(top2[0] - top2[1])
            sig["mean_logp"] += float(torch.log(top2[0].clamp_min(1e-9)))
            sig["min_max"] = min(sig["min_max"], float(top2[0]))
        for s in ("mean_max", "mean_margin", "mean_logp"):
            sig[s] /= N_HEADS
        rows.append({"name": req.get("name", "?"),
                     **{s: round(sig[s], 4) for s in SIGNALS}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "nsynth/data/prior_net_train_300k.jsonl"))
    ap.add_argument("--model", default=str(PROJECT_ROOT / "training/prior_net/prior_net_v1.pt"))
    ap.add_argument("--holdout", type=int, default=10000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hit-recall", type=float, default=0.9,
                    help="hit_recall rule: largest tau keeping this fraction of exact hits")
    # Utility model (measured, Phase A v1): a verbatim hit saves the mean
    # cascade solve time on the 16-problem direct-fallback head-to-head
    # (v0 OFF column, 83.3s/16 = 5.2s); a gate-open miss costs one server
    # round-trip (4.3ms median measured) + K=4 zero-step verifies + one
    # warm refine (<=120 Adam steps, argmax only).
    ap.add_argument("--save-per-hit", type=float, default=5.2)
    ap.add_argument("--cost-zero-miss", type=float, default=0.06)
    ap.add_argument("--cost-warm", type=float, default=0.4)
    ap.add_argument("--bench-requests", default=None,
                    help="JSONL of real bench requests (dump_bench_fallback_requests)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "confidence_calibration.json"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    x, m, y, n_bad = load_dataset(Path(args.data), args.max_rows)
    # n_scalar per row, re-read from the JSONL (cheap second pass).
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
    sigs, exacts = signals_and_exact(model, xv, mv, yv, nsv, device)
    base_rate = exacts.float().mean().item()
    print(f"[calibrate] base exact rate {base_rate:.4f}")

    aucs = {s: round(rank_auc(sigs[s], exacts), 4) for s in SIGNALS}
    best_signal = max(aucs, key=lambda s: aucs[s])
    print(f"[calibrate] signal AUCs: {aucs} -> best: {best_signal}")
    conf = sigs[best_signal]

    sweep = sweep_signal(conf, exacts)

    # Rule 1 — utility-max on the holdout (lower bound; see module docstring).
    cost_open_miss = args.cost_zero_miss + args.cost_warm
    best_util, util_tau = 0.0, None
    for row in sweep:
        if not row["n_fired"]:
            continue
        p = row["precision"]
        util = row["fire_rate"] * (p * args.save_per_hit - (1 - p) * cost_open_miss)
        if util > best_util:
            best_util, util_tau = util, row["tau"]

    # Rule 2 — hit-recall: largest tau keeping >= hit-recall of exact hits.
    if exacts.any():
        hit_conf = conf[exacts]
        recall_tau = round(float(torch.quantile(hit_conf, 1.0 - args.hit_recall)), 4)
    else:
        recall_tau = None

    # Deployment rule: hit_recall. The utility rule is reported as a
    # diagnostic only — on the generated-holdout distribution it either
    # finds nothing or picks a degenerate extreme-tail tau (positive utility
    # on a handful of rows, fires on ~nothing real), because the holdout's
    # ~1-2% base exact rate badly understates the bench fallback
    # population's hit rate (12.5% measured in v0).
    chosen_tau = recall_tau if recall_tau is not None else util_tau
    chosen_rule = "hit_recall" if recall_tau is not None else "utility"

    q = lambda t, qs: [round(torch.quantile(t, qq).item(), 4) for qq in qs]
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    dist = {
        "exact_conf_quantiles": q(conf[exacts], quantiles) if exacts.any() else None,
        "miss_conf_quantiles": q(conf[~exacts], quantiles),
        "quantiles": quantiles,
    }

    bench = None
    if args.bench_requests:
        reqs = [json.loads(l) for l in Path(args.bench_requests).read_text().splitlines() if l.strip()]
        bench_rows = bench_signals(model, reqs, device)
        fired = [r["name"] for r in bench_rows if chosen_tau is not None and r[best_signal] >= chosen_tau]
        bench = {"rows": bench_rows, "fired_at_chosen_tau": fired,
                 "fire_rate": round(len(fired) / max(len(bench_rows), 1), 3)}
        print(f"[calibrate] bench fire rate at tau={chosen_tau}: "
              f"{len(fired)}/{len(bench_rows)} -> {fired}")

    report = {
        "model": args.model,
        "data": args.data,
        "holdout_rows": int(xv.size(0)),
        "seed": args.seed,
        "base_exact_rate": round(base_rate, 4),
        "signal_aucs": aucs,
        "signal": best_signal,
        "rules": {
            "utility": {"tau": util_tau, "expected_seconds_saved_per_problem":
                        round(best_util, 4) if util_tau else 0.0,
                        "save_per_hit": args.save_per_hit,
                        "cost_open_miss": cost_open_miss},
            "hit_recall": {"tau": recall_tau, "target_recall": args.hit_recall},
        },
        "chosen_tau": chosen_tau,
        "chosen_rule": chosen_rule,
        "sweep": sweep,
        "distribution": dist,
        "bench_check": bench,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[calibrate] chosen tau={chosen_tau} (rule={chosen_rule}, signal={best_signal})")
    print(f"[calibrate] wrote {args.out}")


if __name__ == "__main__":
    main()
