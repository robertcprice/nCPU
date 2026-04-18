#!/usr/bin/env python3
"""
Meta-learner for SoftUniversalProgram warm-start.

Architecture: Set Transformer
  • Each I/O example (inputs..., output) → linear embed → d_model
  • Mean-pool over k examples → context vector
  • MLP heads predict each discrete choice:
    - 11 slot descriptions × 7 fields = 77 logit vectors (each size varies by pool/ops)
    - 6 loop_init selections (from lip pool)
    - loop cond (cmp + lhs + rhs)
    - return source
    All capped at max pool size with masking for smaller pools.

Training:
  • Cross-entropy loss on each discrete choice (treated as classification)
  • Load JSONL from --data FILE
  • Save checkpoint to --save MODEL.pt

Usage:
  python3 scripts/train_metalearner.py \\
      --data data/synth_1arg.jsonl \\
      --save models/metalearner_1arg.pt \\
      --n-args 1 --epochs 50

Inference (warm-start):
  python3 scripts/train_metalearner.py \\
      --infer models/metalearner_1arg.pt \\
      --io "[[1],[1],[[2],[8],[[3],[27]"
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

# ─── Architecture constants (must match synthesis.rs) ────────────────────────
N_OPS      = 5   # +,-,*,/,%
N_CMPS     = 6   # <,<=,==,>=,>,!=
N_CONSTS   = 6
N_INIT_SL  = 3
N_LOOP_SL  = 6
N_POST_SL  = 2
N_UNIV_SL  = N_INIT_SL + N_LOOP_SL + N_POST_SL   # 11

def pool_size(n_args):
    return n_args + N_CONSTS + N_UNIV_SL  # args + consts + all slots

def lip_size(n_args):
    return n_args + N_CONSTS + N_INIT_SL  # pre-loop pool

def n_ops_ext():
    return N_OPS + 1   # including identity

# ─── Target logit sizes per description field ─────────────────────────────────
def target_sizes(n_args):
    pool = pool_size(n_args)
    lip  = lip_size(n_args)
    sizes = {}
    # 11 slots × 7 fields
    for s in range(N_UNIV_SL):
        sizes[f"slot_{s}_op"]       = n_ops_ext()
        sizes[f"slot_{s}_s1"]       = pool
        sizes[f"slot_{s}_s2"]       = pool
        sizes[f"slot_{s}_gate_cmp"] = N_CMPS
        sizes[f"slot_{s}_gate_lhs"] = pool
        sizes[f"slot_{s}_gate_rhs"] = pool
        sizes[f"slot_{s}_else_val"] = pool
    # 6 loop_init
    for ls in range(N_LOOP_SL):
        sizes[f"loop_init_{ls}"] = lip
    # loop condition
    sizes["cond_cmp"] = N_CMPS
    sizes["cond_lhs"] = pool
    sizes["cond_rhs"] = pool
    # return
    sizes["ret_src"] = pool
    return sizes


# ─── Data loading ─────────────────────────────────────────────────────────────

def description_to_targets(desc: dict, n_args: int) -> dict[str, int]:
    """Extract integer targets from a description dict."""
    slots = desc["slots"]
    targets = {}
    for i, sd in enumerate(slots):
        targets[f"slot_{i}_op"]       = sd["op"]
        targets[f"slot_{i}_s1"]       = sd["s1"]
        targets[f"slot_{i}_s2"]       = sd["s2"]
        targets[f"slot_{i}_gate_cmp"] = sd["gate_cmp"]
        targets[f"slot_{i}_gate_lhs"] = sd["gate_lhs"]
        targets[f"slot_{i}_gate_rhs"] = sd["gate_rhs"]
        targets[f"slot_{i}_else_val"] = sd["else_val"]
    for ls, src in enumerate(desc["loop_init"]):
        targets[f"loop_init_{ls}"] = src
    targets["cond_cmp"] = desc["cond_cmp"]
    targets["cond_lhs"] = desc["cond_lhs"]
    targets["cond_rhs"] = desc["cond_rhs"]
    targets["ret_src"]  = desc["ret_src"]
    return targets


def load_dataset(path: str, n_args: int, max_examples: int = 8):
    """Load JSONL and return (io_tensors, target_dicts) lists."""
    io_list     = []
    target_list = []
    pool = pool_size(n_args)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["description"]["n_args"] != n_args:
                continue
            ios = rec["io_examples"][:max_examples]
            if len(ios) < 2:
                continue
            # Encode each (inputs, output) as a flat float vector of length n_args+1
            tensors = []
            for (inputs, output) in ios:
                if len(inputs) != n_args:
                    break
                row = torch.tensor(inputs + [output], dtype=torch.float32)
                tensors.append(row)
            else:
                if tensors:
                    # Pad / truncate to max_examples
                    while len(tensors) < max_examples:
                        tensors.append(tensors[-1].clone())
                    io_list.append(torch.stack(tensors[:max_examples]))  # (k, n_args+1)
                    target_list.append(description_to_targets(rec["description"], n_args))
    return io_list, target_list


# ─── Model ────────────────────────────────────────────────────────────────────

class MetaLearner(nn.Module):
    """
    Set Transformer meta-learner.
    Encodes k I/O examples → predicts UniversalProgramDescription.
    """
    def __init__(self, n_args: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_args  = n_args
        self.d_model = d_model
        in_dim = n_args + 1  # inputs + output

        # Per-example encoder
        self.embed = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer encoder (permutation-invariant over examples)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # One classification head per discrete field
        self.sizes = target_sizes(n_args)
        self.heads = nn.ModuleDict({
            k: nn.Linear(d_model, v)
            for k, v in self.sizes.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (batch, k, n_args+1) float tensor
        returns: dict[field_name → (batch, class_size) logits]
        """
        b, k, _ = x.shape
        # Normalize inputs: log1p(|x|) * sign(x) makes large integers manageable
        x = torch.sign(x) * torch.log1p(x.abs())
        # Embed each example
        emb = self.embed(x)                         # (b, k, d)
        emb = self.transformer(emb)                 # (b, k, d)
        ctx = emb.mean(dim=1)                       # (b, d) — mean pool

        return {k: head(ctx) for k, head in self.heads.items()}

    def predict_description(self, io_examples: list[tuple[list[int], int]]) -> dict:
        """Predict description indices from a list of (inputs, output) pairs."""
        tensors = []
        for (inputs, output) in io_examples:
            row = torch.tensor(inputs + [output], dtype=torch.float32)
            tensors.append(row)
        x = torch.stack(tensors).unsqueeze(0)  # (1, k, n_args+1)
        with torch.no_grad():
            logits = self.forward(x)
        slots = []
        for i in range(N_UNIV_SL):
            sd = {
                "op":       logits[f"slot_{i}_op"][0].argmax().item(),
                "s1":       logits[f"slot_{i}_s1"][0].argmax().item(),
                "s2":       logits[f"slot_{i}_s2"][0].argmax().item(),
                "gate_cmp": logits[f"slot_{i}_gate_cmp"][0].argmax().item(),
                "gate_lhs": logits[f"slot_{i}_gate_lhs"][0].argmax().item(),
                "gate_rhs": logits[f"slot_{i}_gate_rhs"][0].argmax().item(),
                "else_val": logits[f"slot_{i}_else_val"][0].argmax().item(),
            }
            slots.append(sd)
        loop_init = [logits[f"loop_init_{ls}"][0].argmax().item() for ls in range(N_LOOP_SL)]
        return {
            "n_args":    self.n_args,
            "slots":     slots,
            "loop_init": loop_init,
            "cond_cmp":  logits["cond_cmp"][0].argmax().item(),
            "cond_lhs":  logits["cond_lhs"][0].argmax().item(),
            "cond_rhs":  logits["cond_rhs"][0].argmax().item(),
            "ret_src":   logits["ret_src"][0].argmax().item(),
            "consts":    [0.0, 1.0, -1.0, 2.0, -2.0, 10.0],  # default
        }


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    n_args = args.n_args
    print(f"Loading data from {args.data} (n_args={n_args})...", flush=True)
    io_list, target_list = load_dataset(args.data, n_args, max_examples=args.k)
    print(f"  Loaded {len(io_list)} records", flush=True)
    if len(io_list) < 10:
        print("Not enough data. Generate more with gen_meta_data --synthetic N")
        sys.exit(1)

    model = MetaLearner(n_args=n_args, d_model=args.d_model, n_heads=args.n_heads).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    sizes = target_sizes(n_args)
    keys  = list(sizes.keys())

    # Pre-stack all I/O tensors and target tensors for fast indexing
    print("  Pre-stacking tensors...", flush=True)
    io_tensor = torch.stack(io_list)  # (n, k, n_args+1) — stays on CPU
    # Per key: (n,) long tensor on device
    target_tensors: dict[str, torch.Tensor] = {
        k: torch.tensor([d[k] for d in target_list], dtype=torch.long, device=device)
        for k in keys
    }
    print("  Done.", flush=True)

    batch_size = args.batch_size
    n = len(io_list)
    best_loss = float("inf")

    # Group keys by output size → batch loss computation (1 CE call per group)
    from collections import defaultdict
    key_groups: dict[int, list[str]] = defaultdict(list)
    for k, v in sizes.items():
        key_groups[v].append(k)
    key_groups = dict(key_groups)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        steps = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            # Build batch — index pre-stacked tensor (fast)
            x = io_tensor[idx].to(device)  # (b, k, n_args+1)
            logits = model(x)

            # Batch CE by output size → fewer autograd nodes → faster backward
            loss = torch.zeros(1, device=device)
            for sz, grp_keys in key_groups.items():
                lg  = torch.stack([logits[k] for k in grp_keys], dim=1)  # (b, g, sz)
                tgt = torch.stack([target_tensors[k][idx] for k in grp_keys], dim=1)  # (b, g)
                loss = loss + F.cross_entropy(lg.reshape(-1, sz), tgt.reshape(-1)) * len(grp_keys)
            loss = loss / len(keys)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            steps += 1

        sched.step()
        avg_loss = total_loss / max(steps, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "n_args": n_args,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
            }, args.save)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            # Accuracy on first 200 examples — use pre-stacked tensors
            model.eval()
            eval_n = min(200, n)
            accs = []
            with torch.no_grad():
                for start in range(0, eval_n, batch_size):
                    end = min(start + batch_size, eval_n)
                    x  = io_tensor[start:end].to(device)
                    lg = model(x)
                    for k in keys:
                        tgt  = target_tensors[k][start:end]  # already on device
                        pred = lg[k].argmax(dim=1)
                        accs.append((pred == tgt).float().mean().item())
            acc = sum(accs) / len(accs) if accs else 0.0
            print(f"  epoch {epoch+1:3d}/{args.epochs}: loss={avg_loss:.4f} acc={acc:.3f} best={best_loss:.4f}", flush=True)

    print(f"Training complete. Best loss={best_loss:.4f}. Saved to {args.save}", flush=True)


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> MetaLearner:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MetaLearner(
        n_args=ckpt["n_args"],
        d_model=ckpt["d_model"],
        n_heads=ckpt["n_heads"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def infer(args):
    model = load_model(args.infer)

    # Parse io examples from --io JSON string: [[input..., output], ...]
    ios = json.loads(args.io)
    io_pairs = [(row[:-1], row[-1]) for row in ios]
    desc = model.predict_description(io_pairs)
    print(json.dumps(desc))


def infer_batch(args):
    """Batch inference: read JSONL of {name, io} records, output JSONL of {name, description}."""
    model = load_model(args.infer)

    input_path = args.batch_in
    output_path = args.batch_out

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    out_f = open(output_path, "w") if output_path else sys.stdout

    for rec in records:
        io_pairs = [(row[:-1], row[-1]) for row in rec["io"]]
        desc = model.predict_description(io_pairs)
        out_f.write(json.dumps({"name": rec["name"], "description": desc}) + "\n")
        out_f.flush()

    if output_path:
        out_f.close()

    print(f"Predicted {len(records)} descriptions.", file=sys.stderr)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Meta-learner for SoftUniversalProgram")
    p.add_argument("--data",       default="data/synth.jsonl",    help="JSONL training data")
    p.add_argument("--save",       default="models/metalearner.pt", help="Checkpoint path")
    p.add_argument("--n-args",     type=int, default=1,            help="Number of function arguments")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--d-model",    type=int, default=128)
    p.add_argument("--n-heads",    type=int, default=4)
    p.add_argument("--k",          type=int, default=8, help="I/O examples per problem")
    p.add_argument("--infer",      default=None,   help="Run inference with this checkpoint")
    p.add_argument("--io",         default=None,   help="I/O examples JSON for single inference")
    p.add_argument("--batch-in",   default=None,   help="JSONL file with {name, io} records for batch inference")
    p.add_argument("--batch-out",  default=None,   help="Output JSONL path (default: stdout)")
    args = p.parse_args()

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    if args.infer and args.batch_in:
        infer_batch(args)
    elif args.infer:
        infer(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
