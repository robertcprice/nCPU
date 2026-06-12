#!/usr/bin/env python3
"""Train Program Prior Net v0 (ROADMAP Rung 9, Phase A, stage A2).

Input: JSONL from `cargo run --release --bin gen_prior_data`
       (default nsynth/data/prior_net_train.jsonl).
Output: training/prior_net/prior_net_v0.pt + eval_report.{json,md}
        (slot accuracies on a held-out split).

Usage:
  python3 training/prior_net/train.py \
      --data nsynth/data/prior_net_train.jsonl \
      --out training/prior_net/prior_net_v0.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "nsynth" / "scripts" / "prior_net"))

from prior_net_model import (  # noqa: E402
    HEAD_NAMES,
    HEAD_SIZES,
    MAX_EXAMPLES,
    N_HEADS,
    PriorNet,
    encode_problem,
    example_mask,
    head_valid_classes,
    labels_from_desc,
    save_checkpoint,
)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dataset(path: Path, max_rows: int | None):
    feats, masks, labels = [], [], []
    n_bad = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                n_scalar = int(row["n_scalar"])
                ex = row["examples"]
                feats.append(encode_problem(ex, n_scalar))
                masks.append(example_mask(len(ex)))
                labels.append(labels_from_desc(row["desc"]))
            except Exception:
                n_bad += 1
                continue
            if max_rows and len(feats) >= max_rows:
                break
    x = torch.stack(feats)
    m = torch.stack(masks)
    y = torch.tensor(labels, dtype=torch.long)
    return x, m, y, n_bad


# Head-name groups for the eval report.
def head_group(name: str) -> str:
    if name.endswith("_op"):
        return "op"
    if name.endswith(("_s1", "_s2")):
        return "src"
    if name.endswith("_cmp"):
        return "cmp"
    if name.endswith(("_gl", "_gr")):
        return "gate"
    if name.endswith("_el"):
        return "else"
    if name.startswith("binit"):
        return "body_init"
    if name == "ret":
        return "ret"
    if name.startswith("const"):
        return "const"
    return "other"


@torch.no_grad()
def evaluate(model, x, m, y, device, batch=1024):
    model.eval()
    head_correct = torch.zeros(N_HEADS)
    all_correct = torch.zeros(0, dtype=torch.bool)
    total_loss = 0.0
    n = x.size(0)
    for i in range(0, n, batch):
        xb = x[i : i + batch].to(device)
        mb = m[i : i + batch].to(device)
        yb = y[i : i + batch].to(device)
        flat = model(xb, mb)
        logits = model.split_logits(flat)
        row_ok = torch.ones(xb.size(0), dtype=torch.bool, device=device)
        for h in range(N_HEADS):
            total_loss += F.cross_entropy(logits[h], yb[:, h], reduction="sum").item()
            pred = logits[h].argmax(dim=-1)
            ok = pred == yb[:, h]
            head_correct[h] += ok.sum().cpu()
            row_ok &= ok
        all_correct = torch.cat([all_correct, row_ok.cpu()])
    head_acc = (head_correct / n).tolist()
    return {
        "loss": total_loss / (n * N_HEADS),
        "head_acc": dict(zip(HEAD_NAMES, head_acc)),
        "full_desc_exact": all_correct.float().mean().item(),
        "n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "nsynth/data/prior_net_train.jsonl"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "training/prior_net/prior_net_v0.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=int, default=5000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    print(f"[prior_net] device={device}")
    if device == "cpu":
        print("[prior_net] WARNING: MPS/CUDA unavailable — CPU fallback (slower)")

    t0 = time.time()
    x, m, y, n_bad = load_dataset(Path(args.data), args.max_rows)
    print(f"[prior_net] loaded {x.size(0)} rows ({n_bad} bad) in {time.time()-t0:.1f}s")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(x.size(0), generator=g)
    x, m, y = x[perm], m[perm], y[perm]
    holdout = min(args.holdout, x.size(0) // 5)
    xt, mt, yt = x[:-holdout], m[:-holdout], y[:-holdout]
    xv, mv, yv = x[-holdout:], m[-holdout:], y[-holdout:]
    print(f"[prior_net] train={xt.size(0)} val={xv.size(0)}")

    model = PriorNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[prior_net] params={n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        ep0 = time.time()
        perm = torch.randperm(xt.size(0), generator=g)
        total = 0.0
        steps = 0
        for i in range(0, xt.size(0), args.batch):
            idx = perm[i : i + args.batch]
            xb = xt[idx].to(device)
            mb = mt[idx].to(device)
            yb = yt[idx].to(device)
            flat = model(xb, mb)
            logits = model.split_logits(flat)
            loss = sum(F.cross_entropy(logits[h], yb[:, h]) for h in range(N_HEADS))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            steps += 1
        sched.step()
        val = evaluate(model, xv, mv, yv, device)
        history.append({"epoch": epoch, "train_loss": total / steps / N_HEADS, "val_loss": val["loss"], "val_exact": val["full_desc_exact"]})
        print(
            f"[prior_net] epoch {epoch:02d} train={total/steps/N_HEADS:.4f} "
            f"val={val['loss']:.4f} exact={val['full_desc_exact']:.3f} "
            f"({time.time()-ep0:.0f}s)"
        )
        if val["loss"] < best_val - 1e-4:
            best_val = val["loss"]
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(model, args.out, extra={"epoch": epoch, "val_loss": val["loss"]})
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[prior_net] early stop at epoch {epoch} (best={best_epoch})")
                break

    # Final eval with the best checkpoint.
    from prior_net_model import load_checkpoint  # noqa: E402

    model = load_checkpoint(args.out, device)
    val = evaluate(model, xv, mv, yv, device)

    groups: dict[str, list[float]] = {}
    for name, acc in val["head_acc"].items():
        groups.setdefault(head_group(name), []).append(acc)
    group_acc = {k: sum(v) / len(v) for k, v in groups.items()}

    report = {
        "model": str(args.out),
        "params": n_params,
        "train_rows": int(xt.size(0)),
        "val_rows": int(xv.size(0)),
        "best_epoch": best_epoch,
        "val_loss": val["loss"],
        "full_desc_exact": val["full_desc_exact"],
        "group_accuracy": group_acc,
        "head_accuracy": val["head_acc"],
        "history": history,
        "wall_seconds": time.time() - t0,
        "device": device,
        "seed": args.seed,
    }
    rep_path = Path(args.out).parent / "eval_report.json"
    rep_path.write_text(json.dumps(report, indent=2))

    md = ["# Prior Net v0 — held-out eval", ""]
    md.append(f"- model: `{args.out}` ({n_params:,} params)")
    md.append(f"- train/val rows: {xt.size(0)} / {xv.size(0)}")
    md.append(f"- best epoch: {best_epoch}, val loss {val['loss']:.4f}")
    md.append(f"- full-description exact match: **{val['full_desc_exact']:.1%}**")
    md.append("")
    md.append("| head group | accuracy |")
    md.append("|---|---|")
    for k in sorted(group_acc):
        md.append(f"| {k} | {group_acc[k]:.1%} |")
    md.append("")
    (Path(args.out).parent / "eval_report.md").write_text("\n".join(md) + "\n")
    print(f"[prior_net] saved {args.out} + eval_report; exact={val['full_desc_exact']:.3f}")
    print(json.dumps({"group_accuracy": group_acc, "full_desc_exact": val["full_desc_exact"]}))


if __name__ == "__main__":
    main()
