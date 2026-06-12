"""Program Prior Net v0 — shared model + featurization (Rung 9 Phase A).

Maps I/O examples of a universal-array problem to a distribution over the
discrete program description space used by nsynth's universal-array
synthesizer:

  - 6 slots (1 pre + 4 body + 1 post), each with 7 fields:
      op (6), s1 (pool), s2 (pool), cmp (6), gl (pool), gr (pool), el (pool)
  - 4 body-init pointers (lip = 1 + 6 consts + n_scalar)
  - 1 return pointer (pool)
  - 3 free constant-pool values c3..c5 (anchors [0, 1, -1] are fixed),
    classified over the closed CONST_VOCAB the data generator samples from.

Architecture lineage: the v5 meta-learner (TransformerEncoder over linearly
embedded I/O examples -> program description, 23/24 on the 1-arg scalar
space), scaled to the universal-array space with d_model 256 / 4 layers.

Used by both training/prior_net/train.py and nsynth/scripts/prior_net/propose.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ── program-space geometry (must mirror nsynth/src/synthesis/universal_array.rs) ──
N_OPS = 5          # +, -, *, /, %  (op head has N_OPS+1 classes; last = identity)
N_CMPS = 6
N_CONSTS = 6
N_ARR_PRE = 1
N_ARR_BODY = 4
N_ARR_POST = 1
N_ARR_SLOTS = N_ARR_PRE + N_ARR_BODY + N_ARR_POST
N_ARR_FIXED = 8
MAX_N_SCALAR = 2
MAX_POOL = N_ARR_FIXED + N_CONSTS + MAX_N_SCALAR + N_ARR_SLOTS  # 22
MAX_LIP = 1 + N_CONSTS + MAX_N_SCALAR                            # 9

# Closed vocabulary for the three free constant slots (mirrors
# CONST_CANDIDATES in prior_gen.rs; 2/-2/10 cover the historical defaults).
CONST_VOCAB = [2, -2, 3, -3, 4, -4, 5, -5, 6, 7, 8, 9, 10, 12, 15, 16, 20]
CONST_TO_IDX = {v: i for i, v in enumerate(CONST_VOCAB)}

# ── featurization ──
MAX_ARR_LEN = 12
MAX_EXAMPLES = 6


def _val_feats(v: float) -> list[float]:
    """4 scale-aware features per integer value."""
    av = abs(v)
    return [
        math.tanh(v / 8.0),
        math.tanh(v / 64.0),
        (1.0 if v >= 0 else -1.0) * math.log1p(av) / 14.0,
        float(int(av) % 2),
    ]


# arr (12*4) + arr mask (12) + len (1) + scalars (2*4) + scalar mask (2)
# + expected (4) + n_scalar one-hot (3)
FEAT_DIM = MAX_ARR_LEN * 4 + MAX_ARR_LEN + 1 + MAX_N_SCALAR * 4 + MAX_N_SCALAR + 4 + 3


def example_features(array: list[int], scalars: list[int], expected: int, n_scalar: int) -> list[float]:
    f: list[float] = []
    arr = list(array)[:MAX_ARR_LEN]
    for i in range(MAX_ARR_LEN):
        f.extend(_val_feats(float(arr[i])) if i < len(arr) else [0.0, 0.0, 0.0, 0.0])
    f.extend([1.0 if i < len(arr) else 0.0 for i in range(MAX_ARR_LEN)])
    f.append(len(arr) / MAX_ARR_LEN)
    sc = list(scalars)[:MAX_N_SCALAR]
    for i in range(MAX_N_SCALAR):
        f.extend(_val_feats(float(sc[i])) if i < len(sc) else [0.0, 0.0, 0.0, 0.0])
    f.extend([1.0 if i < len(sc) else 0.0 for i in range(MAX_N_SCALAR)])
    f.extend(_val_feats(float(expected)))
    oh = [0.0, 0.0, 0.0]
    oh[min(n_scalar, 2)] = 1.0
    f.extend(oh)
    assert len(f) == FEAT_DIM
    return f


def encode_problem(examples: list[dict], n_scalar: int) -> torch.Tensor:
    """examples: [{"array": [...], "scalars": [...], "expected": int}] ->
    (MAX_EXAMPLES, FEAT_DIM) tensor, zero-padded; order-invariant enough
    because the transformer has no positional encoding over examples."""
    rows = []
    for ex in examples[:MAX_EXAMPLES]:
        rows.append(example_features(ex["array"], ex.get("scalars", []), ex["expected"], n_scalar))
    while len(rows) < MAX_EXAMPLES:
        rows.append([0.0] * FEAT_DIM)
    return torch.tensor(rows, dtype=torch.float32)


def example_mask(n_examples: int) -> torch.Tensor:
    """True = padding (torch convention for src_key_padding_mask)."""
    m = torch.zeros(MAX_EXAMPLES + 1, dtype=torch.bool)  # +1 for CLS
    m[1 + min(n_examples, MAX_EXAMPLES):] = True
    return m


# ── label/head layout ──
def head_layout() -> list[tuple[str, int]]:
    heads: list[tuple[str, int]] = []
    for s in range(N_ARR_SLOTS):
        heads.append((f"slot{s}_op", N_OPS + 1))
        heads.append((f"slot{s}_s1", MAX_POOL))
        heads.append((f"slot{s}_s2", MAX_POOL))
        heads.append((f"slot{s}_cmp", N_CMPS))
        heads.append((f"slot{s}_gl", MAX_POOL))
        heads.append((f"slot{s}_gr", MAX_POOL))
        heads.append((f"slot{s}_el", MAX_POOL))
    for b in range(N_ARR_BODY):
        heads.append((f"binit{b}", MAX_LIP))
    heads.append(("ret", MAX_POOL))
    for c in range(3):
        heads.append((f"const{c + 3}", len(CONST_VOCAB)))
    return heads


HEADS = head_layout()
HEAD_NAMES = [h[0] for h in HEADS]
HEAD_SIZES = [h[1] for h in HEADS]
N_HEADS = len(HEADS)
TOTAL_LOGITS = sum(HEAD_SIZES)


def labels_from_desc(desc: dict) -> list[int]:
    """Flatten a row's `desc` object into one label per head."""
    labels: list[int] = []
    for s in range(N_ARR_SLOTS):
        op, s1, s2, cmp_, gl, gr, el = desc["slots"][s]
        labels.extend([op, s1, s2, cmp_, gl, gr, el])
    labels.extend(desc["body_init"])
    labels.append(desc["ret"])
    for c in desc["consts"][3:6]:
        labels.append(CONST_TO_IDX.get(int(c), 0))
    return labels


def pool_size(n_scalar: int) -> int:
    return N_ARR_FIXED + N_CONSTS + n_scalar + N_ARR_SLOTS


def lip_size(n_scalar: int) -> int:
    return 1 + N_CONSTS + n_scalar


def head_valid_classes(name: str, size: int, n_scalar: int) -> int:
    """Number of valid classes for a head given the problem's n_scalar."""
    if size == MAX_POOL:
        return pool_size(n_scalar)
    if size == MAX_LIP:
        return lip_size(n_scalar)
    return size


class PriorNet(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Sequential(
            nn.Linear(FEAT_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, TOTAL_LOGITS)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, MAX_EXAMPLES, FEAT_DIM); pad_mask: (B, MAX_EXAMPLES+1) bool.
        Returns flat logits (B, TOTAL_LOGITS)."""
        h = self.embed(x)
        cls = self.cls.expand(h.size(0), 1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.out(h[:, 0])

    def split_logits(self, flat: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(flat, HEAD_SIZES, dim=-1))


def save_checkpoint(model: PriorNet, path: str, extra: dict | None = None) -> None:
    ckpt = {
        "state_dict": model.state_dict(),
        "d_model": model.d_model,
        "feat_dim": FEAT_DIM,
        "head_names": HEAD_NAMES,
        "head_sizes": HEAD_SIZES,
        "const_vocab": CONST_VOCAB,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str = "cpu") -> PriorNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    assert ckpt["feat_dim"] == FEAT_DIM, "feature schema mismatch"
    assert ckpt["head_sizes"] == HEAD_SIZES, "head layout mismatch"
    model = PriorNet(d_model=ckpt["d_model"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
