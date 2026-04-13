#!/usr/bin/env python3
"""
Meta-learner for program-type classification from I/O examples.

Architecture: TransformerEncoder (d_model=64, n_heads=4, n_layers=3)
  - Each I/O example (inputs..., output) -> linear embed -> d_model
  - TransformerEncoder (permutation-invariant over examples via mean pool)
  - Classification head -> program type logits

Program types (5 classes):
  0: expr           - single arithmetic expression (a+b, a*c, a*a)
  1: two_precomp    - two pre-computations (a*b+c, (a+b)*(a-b))
  2: branch         - conditional expression (max, min, sign, abs)
  3: loop           - iterative computation (sum, factorial, fibonacci, gcd)
  4: chained_branch - two sequential ternary branches (min3, clamp)

Training data: JSONL from generate_training_data.py with fields:
  io_pairs, n_args, method, code

Usage:
  # Generate training data
  python3 scripts/generate_training_data.py --out data/expr_type_train.jsonl --count 600

  # Train
  python3 scripts/train_expr_metalearner.py \\
      --data data/expr_type_train.jsonl \\
      --save models/expr_metalearner.pt \\
      --epochs 100

  # Inference
  python3 scripts/train_expr_metalearner.py \\
      --infer models/expr_metalearner.pt \\
      --io "[[1,2,3],[4,5,9],[0,0,0],[-1,3,2]]"
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Program type definitions ────────────────────────────────────────────────

PROGRAM_TYPES = ["expr", "two_precomp", "branch", "loop", "chained_branch"]
TYPE_TO_IDX = {t: i for i, t in enumerate(PROGRAM_TYPES)}
NUM_TYPES = len(PROGRAM_TYPES)

# Maximum number of function arguments we handle
MAX_ARGS = 4

# Input dimension per I/O example: MAX_ARGS inputs + 1 output + 1 n_args indicator
IO_DIM = MAX_ARGS + 1 + 1  # 6


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(path: str, max_examples: int = 8):
    """
    Load JSONL training data.

    Returns:
        io_list:     list of (max_examples, IO_DIM) float tensors
        label_list:  list of int (program type index)
        n_args_list: list of int
    """
    io_list = []
    label_list = []
    n_args_list = []
    skipped = 0

    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            method = rec.get("method", "")
            if method not in TYPE_TO_IDX:
                skipped += 1
                continue

            io_pairs = rec.get("io_pairs", rec.get("inputs", []))
            n_args = rec.get("n_args", 0)

            if not io_pairs or n_args < 1 or n_args > MAX_ARGS:
                skipped += 1
                continue

            # Encode each I/O pair as a fixed-size vector
            tensors = []
            for pair in io_pairs[:max_examples]:
                if len(pair) != 2:
                    continue
                inputs, output = pair
                if not isinstance(inputs, list):
                    continue
                # Pad inputs to MAX_ARGS
                padded_inputs = list(inputs[:MAX_ARGS])
                while len(padded_inputs) < MAX_ARGS:
                    padded_inputs.append(0)
                # Row: [input0, input1, ..., inputMAX, output, n_args_normalized]
                row = padded_inputs + [output, n_args / MAX_ARGS]
                tensors.append(torch.tensor(row, dtype=torch.float32))

            if len(tensors) < 2:
                skipped += 1
                continue

            # Pad/truncate to max_examples
            while len(tensors) < max_examples:
                tensors.append(tensors[-1].clone())
            tensors = tensors[:max_examples]

            io_list.append(torch.stack(tensors))  # (max_examples, IO_DIM)
            label_list.append(TYPE_TO_IDX[method])
            n_args_list.append(n_args)

    if skipped > 0:
        print(f"  Skipped {skipped} records (unknown method or invalid format)",
              file=sys.stderr, flush=True)

    return io_list, label_list, n_args_list


# ─── Model ────────────────────────────────────────────────────────────────────

class ExprMetaLearner(nn.Module):
    """
    TransformerEncoder meta-learner for program type classification.

    Encodes k I/O examples -> predicts which program type (expr, branch, loop, etc.)
    would best solve the synthesis problem.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Per-example encoder: IO_DIM -> d_model
        self.embed = nn.Sequential(
            nn.Linear(IO_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer encoder (permutation-invariant over examples)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_TYPES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, k, IO_DIM) float tensor of I/O examples

        Returns:
            logits: (batch, NUM_TYPES) classification logits
        """
        # Normalize inputs: log1p(|x|) * sign(x) for large integer stability
        x = torch.sign(x) * torch.log1p(x.abs())

        # Embed each example
        emb = self.embed(x)                         # (b, k, d_model)
        emb = self.transformer(emb)                  # (b, k, d_model)

        # Mean pool over examples (permutation invariant)
        ctx = emb.mean(dim=1)                        # (b, d_model)

        return self.classifier(ctx)                  # (b, NUM_TYPES)

    def predict(self, io_pairs: list, n_args: int) -> tuple[str, dict[str, float]]:
        """
        Predict program type from I/O pairs.

        Args:
            io_pairs: list of [inputs, output] pairs
            n_args: number of function arguments

        Returns:
            (predicted_type, {type: probability})
        """
        tensors = []
        for pair in io_pairs:
            inputs, output = pair
            padded = list(inputs[:MAX_ARGS])
            while len(padded) < MAX_ARGS:
                padded.append(0)
            row = padded + [output, n_args / MAX_ARGS]
            tensors.append(torch.tensor(row, dtype=torch.float32))

        x = torch.stack(tensors).unsqueeze(0)  # (1, k, IO_DIM)

        with torch.no_grad():
            logits = self.forward(x)              # (1, NUM_TYPES)
            probs = F.softmax(logits[0], dim=0)

        pred_idx = logits[0].argmax().item()
        prob_dict = {PROGRAM_TYPES[i]: probs[i].item() for i in range(NUM_TYPES)}

        return PROGRAM_TYPES[pred_idx], prob_dict


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}", flush=True)

    print(f"Loading data from {args.data}...", flush=True)
    io_list, label_list, n_args_list = load_dataset(args.data, max_examples=args.k)
    n = len(io_list)
    print(f"  Loaded {n} records", flush=True)

    if n < 20:
        print("Not enough data. Generate more with generate_training_data.py --count 600",
              file=sys.stderr)
        sys.exit(1)

    # Print class distribution
    counts = {}
    for label in label_list:
        t = PROGRAM_TYPES[label]
        counts[t] = counts.get(t, 0) + 1
    print("  Class distribution:", flush=True)
    for t in PROGRAM_TYPES:
        c = counts.get(t, 0)
        print(f"    {t}: {c} ({100*c/n:.1f}%)", flush=True)

    # Compute class weights for balanced loss
    class_counts = torch.zeros(NUM_TYPES)
    for label in label_list:
        class_counts[label] += 1
    # Inverse frequency weighting (capped)
    class_weights = n / (NUM_TYPES * class_counts.clamp(min=1))
    class_weights = class_weights.clamp(max=5.0)
    class_weights = class_weights.to(device)
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}", flush=True)

    # Split: 90% train, 10% val
    split_idx = int(n * 0.9)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    train_idx = perm[:split_idx]
    val_idx = perm[split_idx:]

    # Pre-stack tensors
    io_tensor = torch.stack(io_list)  # (n, k, IO_DIM)
    label_tensor = torch.tensor(label_list, dtype=torch.long, device=device)

    model = ExprMetaLearner(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)
    print(f"Architecture: d_model={args.d_model}, n_heads={args.n_heads}, "
          f"n_layers={args.n_layers}, dropout={args.dropout}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    batch_size = args.batch_size
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        # ── Training ──
        model.train()
        train_perm = train_idx[torch.randperm(len(train_idx))]
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(train_perm), batch_size):
            idx = train_perm[start:start + batch_size]
            x = io_tensor[idx].to(device)           # (b, k, IO_DIM)
            y = label_tensor[idx]                    # (b,)

            logits = model(x)                        # (b, NUM_TYPES)
            loss = F.cross_entropy(logits, y, weight=class_weights)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += len(idx)

        sched.step()
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        per_class_correct = torch.zeros(NUM_TYPES)
        per_class_total = torch.zeros(NUM_TYPES)

        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                idx = val_idx[start:start + batch_size]
                x = io_tensor[idx].to(device)
                y = label_tensor[idx]

                logits = model(x)
                loss = F.cross_entropy(logits, y)

                val_loss += loss.item() * len(idx)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += len(idx)

                for c in range(NUM_TYPES):
                    mask = y == c
                    per_class_total[c] += mask.sum().item()
                    per_class_correct[c] += ((preds == y) & mask).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        # Save best model
        improved = False
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            improved = True
            torch.save({
                "model_state": model.state_dict(),
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "program_types": PROGRAM_TYPES,
                "val_acc": val_acc,
                "epoch": epoch + 1,
            }, args.save)
        else:
            patience_counter += 1

        # Logging
        if epoch % 5 == 0 or epoch == args.epochs - 1 or improved:
            per_class_str = " | ".join(
                f"{PROGRAM_TYPES[c]}: {100*per_class_correct[c]/max(per_class_total[c],1):.0f}%"
                for c in range(NUM_TYPES)
            )
            marker = " *" if improved else ""
            print(
                f"  epoch {epoch+1:3d}/{args.epochs}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{marker}",
                flush=True,
            )
            if epoch % 20 == 0 or epoch == args.epochs - 1:
                print(f"    per-class: {per_class_str}", flush=True)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)",
                  flush=True)
            break

    print(f"\nTraining complete. Best val_acc={best_val_acc:.3f}. Saved to {args.save}",
          flush=True)

    # Final confusion matrix
    print("\nFinal evaluation on validation set:", flush=True)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for start in range(0, len(val_idx), batch_size):
            idx = val_idx[start:start + batch_size]
            x = io_tensor[idx].to(device)
            y = label_tensor[idx]
            logits = model(x)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # Print confusion matrix
    print("\n  Confusion matrix (rows=true, cols=predicted):", flush=True)
    header = "          " + "".join(f"{t[:8]:>10s}" for t in PROGRAM_TYPES)
    print(header, flush=True)
    for true_c in range(NUM_TYPES):
        row = f"  {PROGRAM_TYPES[true_c][:8]:>8s}"
        for pred_c in range(NUM_TYPES):
            count = sum(1 for p, l in zip(all_preds, all_labels)
                       if l == true_c and p == pred_c)
            row += f"{count:10d}"
        print(row, flush=True)


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> ExprMetaLearner:
    """Load a trained model checkpoint."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ExprMetaLearner(
        d_model=ckpt["d_model"],
        n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"],
        dropout=ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def infer(args):
    """Run single inference from --io JSON."""
    model = load_model(args.infer)

    # Parse io examples: [[i1,i2,...,o], [i1,i2,...,o], ...]
    rows = json.loads(args.io)
    io_pairs = []
    for row in rows:
        if len(row) < 2:
            continue
        inputs = row[:-1]
        output = row[-1]
        io_pairs.append([inputs, output])

    n_args = len(rows[0]) - 1 if rows else 1

    pred_type, probs = model.predict(io_pairs, n_args)

    result = {
        "predicted_type": pred_type,
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "n_args": n_args,
    }
    print(json.dumps(result, indent=2))


def infer_batch(args):
    """Batch inference from JSONL file."""
    model = load_model(args.infer)

    records = []
    with open(args.batch_in) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    out_f = open(args.batch_out, "w") if args.batch_out else sys.stdout

    correct = 0
    total = 0

    for rec in records:
        io_pairs = rec.get("io_pairs", rec.get("inputs", []))
        n_args = rec.get("n_args", 1)

        pred_type, probs = model.predict(io_pairs, n_args)

        output_rec = {
            "name": rec.get("name", ""),
            "predicted_type": pred_type,
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
        }

        # If we have ground truth, compute accuracy
        true_method = rec.get("method", "")
        if true_method:
            output_rec["true_type"] = true_method
            output_rec["correct"] = pred_type == true_method
            if pred_type == true_method:
                correct += 1
            total += 1

        out_f.write(json.dumps(output_rec) + "\n")
        out_f.flush()

    if args.batch_out:
        out_f.close()

    if total > 0:
        print(f"\nBatch accuracy: {correct}/{total} ({100*correct/total:.1f}%)",
              file=sys.stderr, flush=True)
    else:
        print(f"Predicted {len(records)} records.", file=sys.stderr, flush=True)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Train/infer program-type classifier from I/O examples"
    )
    # Data
    p.add_argument("--data", default="data/expr_type_train.jsonl",
                   help="JSONL training data")
    p.add_argument("--save", default="models/expr_metalearner.pt",
                   help="Checkpoint save path")

    # Architecture
    p.add_argument("--d-model", type=int, default=64,
                   help="Transformer hidden dimension")
    p.add_argument("--n-heads", type=int, default=4,
                   help="Number of attention heads")
    p.add_argument("--n-layers", type=int, default=3,
                   help="Number of transformer layers")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate")
    p.add_argument("--k", type=int, default=8,
                   help="Max I/O examples per problem")
    p.add_argument("--patience", type=int, default=30,
                   help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42)

    # Inference
    p.add_argument("--infer", default=None,
                   help="Run inference with this checkpoint")
    p.add_argument("--io", default=None,
                   help="I/O examples JSON for single inference: [[i1,i2,o], ...]")
    p.add_argument("--batch-in", default=None,
                   help="JSONL file for batch inference")
    p.add_argument("--batch-out", default=None,
                   help="Output JSONL path (default: stdout)")

    args = p.parse_args()

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    if args.infer and args.batch_in:
        infer_batch(args)
    elif args.infer:
        if not args.io:
            print("Error: --io required for single inference", file=sys.stderr)
            sys.exit(1)
        infer(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
