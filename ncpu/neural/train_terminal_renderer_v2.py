#!/usr/bin/env python3
"""
Training pipeline for the Neural Terminal Renderer V2.

Trains the V2 architecture with:
  - Extended character set (1024 chars: ASCII + Latin-1 + box-drawing + symbols)
  - Full xterm-256 color palette
  - Position-aware glyph generation via sinusoidal positional encoding

Training phases:
  Phase 1: Cell-level -- train glyph generator + palette on individual characters
  Phase 2: Frame-level -- fine-tune full pipeline on complete terminal screens

Ground truth: PIL rendering with system monospace font.

Usage:
    python ncpu/neural/train_terminal_renderer_v2.py
    python ncpu/neural/train_terminal_renderer_v2.py --demo-only
    python ncpu/neural/train_terminal_renderer_v2.py --epochs 3000 --lr 1e-3 --device mps
    python ncpu/neural/train_terminal_renderer_v2.py --efficient-compositor
"""

import sys
import platform
import time
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ncpu.neural.neural_terminal_renderer_v2 import (
    NeuralTerminalRendererV2,
    NeuralColorPaletteV2,
    XTERM_256_PALETTE,
    N_CHARS_V2,
    N_COLORS_V2,
)
from ncpu.neural.neural_terminal_renderer import (
    TerminalState,
    TERM_ROWS, TERM_COLS,
    CELL_H, CELL_W,
    FRAME_H, FRAME_W,
)

__all__ = [
    "ConventionalRendererV2",
    "generate_v2_training_batch",
    "random_terminal_state_v2",
    "train_phase1",
    "train_phase2",
    "demo",
    "main",
    "_build_char_pools",
]

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "display"
MODEL_PATH = MODEL_DIR / "terminal_renderer_v2.pt"


# ═══════════════════════════════════════════════════════════════════════════
# Extended Character Set Definition
# ═══════════════════════════════════════════════════════════════════════════

def _build_extended_char_set() -> list[int]:
    """Build the extended character set for V2 training.

    Returns a list of Unicode code points to train on:
      - ASCII printable (32-126): 95 characters
      - Latin-1 Supplement (160-255): 96 characters (accented letters, symbols)
      - Box-drawing (0x2500-0x257F): 128 characters
      - Block elements (0x2580-0x259F): 32 characters
      - Geometric shapes subset (0x25A0-0x25C0): 33 characters
      - Arrows (0x2190-0x2199): 10 characters
      - Mathematical operators subset (0x2200-0x2230): 49 characters
      - Miscellaneous symbols subset (0x2600-0x2615): 22 characters
    Total: ~465 unique code points, well within the 1024 embedding table.
    """
    chars = []

    # ASCII printable
    chars.extend(range(32, 127))

    # Latin-1 Supplement (accented letters, currency, special symbols)
    chars.extend(range(160, 256))

    # Box-drawing characters
    chars.extend(range(0x2500, 0x2580))

    # Block elements
    chars.extend(range(0x2580, 0x25A0))

    # Geometric shapes (subset)
    chars.extend(range(0x25A0, 0x25C1))

    # Arrows
    chars.extend(range(0x2190, 0x219A))

    # Mathematical operators (subset)
    chars.extend(range(0x2200, 0x2231))

    # Miscellaneous symbols (subset)
    chars.extend(range(0x2600, 0x2616))

    return chars


EXTENDED_CHARS = _build_extended_char_set()

# Mapping: Unicode code point -> V2 embedding index
# Code points < 256 map directly; others get indices starting at 256
_CODEPOINT_TO_INDEX: dict[int, int] = {}
_next_idx = 256  # First 256 indices reserved for code points 0-255
for cp in EXTENDED_CHARS:
    if cp < 256:
        _CODEPOINT_TO_INDEX[cp] = cp  # Direct mapping
    elif cp not in _CODEPOINT_TO_INDEX:
        _CODEPOINT_TO_INDEX[cp] = _next_idx
        _next_idx += 1

# Characters that are trainable (have embedding indices)
TRAINABLE_CHARS = list(_CODEPOINT_TO_INDEX.keys())
TRAINABLE_INDICES = list(_CODEPOINT_TO_INDEX.values())


# ═══════════════════════════════════════════════════════════════════════════
# Font Discovery
# ═══════════════════════════════════════════════════════════════════════════

def _find_monospace_font(prefer_unicode: bool = True):
    """Find a monospace font that fits 8x16 cells and supports extended chars.

    When prefer_unicode is True, fonts with broader Unicode coverage are
    preferred (e.g., DejaVu Sans Mono, Menlo) over fonts with limited
    character sets.
    """
    candidates = []
    if platform.system() == "Darwin":
        candidates = [
            ("/System/Library/Fonts/Menlo.ttc", [13, 14, 12]),
            ("/System/Library/Fonts/Monaco.dfont", [12, 13, 11]),
            ("/Library/Fonts/Courier New.ttf", [13, 14, 12]),
        ]
    elif platform.system() == "Linux":
        candidates = [
            ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", [12, 13, 11]),
            ("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", [13, 14, 12]),
            ("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", [13, 14, 12]),
        ]

    for path, sizes in candidates:
        for size in sizes:
            try:
                font = ImageFont.truetype(path, size)
                bbox = font.getbbox("M")
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w <= CELL_W + 1 and h <= CELL_H + 1:
                    return font
            except (IOError, OSError):
                break

    try:
        return ImageFont.load_default(size=12)
    except TypeError:
        return ImageFont.load_default()


# ═══════════════════════════════════════════════════════════════════════════
# Conventional Renderer V2 (Ground Truth)
# ═══════════════════════════════════════════════════════════════════════════

class ConventionalRendererV2:
    """PIL-based terminal renderer with extended character and color support.

    Generates ground truth for training the V2 neural renderer:
      - Renders all characters in the extended set (ASCII + Latin-1 + Unicode)
      - Uses full xterm-256 color palette
      - Pre-renders glyph alpha masks for all trainable characters
    """

    def __init__(self):
        self.font = _find_monospace_font(prefer_unicode=True)
        self.palette = np.array(XTERM_256_PALETTE, dtype=np.float32) / 255.0

        # Pre-render glyph masks for all embedding indices
        # Index -> alpha mask (cell_h, cell_w)
        self.glyph_table = self._prerender_all_glyphs()

    def _prerender_all_glyphs(self) -> np.ndarray:
        """Pre-render glyph alpha masks for all 1024 embedding indices.

        Returns: (N_CHARS_V2, CELL_H, CELL_W) float32 array
        """
        masks = np.zeros((N_CHARS_V2, CELL_H, CELL_W), dtype=np.float32)

        # Build reverse map: embedding index -> code point
        index_to_cp: dict[int, int] = {}
        for cp, idx in _CODEPOINT_TO_INDEX.items():
            index_to_cp[idx] = cp

        for idx in range(N_CHARS_V2):
            cp = index_to_cp.get(idx, None)
            if cp is None:
                continue  # Unused embedding index -> zero mask

            char = chr(cp)
            img = Image.new("L", (CELL_W, CELL_H), 0)
            draw = ImageDraw.Draw(img)
            try:
                draw.text((0, 0), char, fill=255, font=self.font)
            except Exception:
                pass
            masks[idx] = np.array(img, dtype=np.float32) / 255.0

        return masks

    def render_cell_batch(
        self,
        char_indices: np.ndarray,
        fg_colors: np.ndarray,
        bg_colors: np.ndarray,
    ) -> np.ndarray:
        """Render individual cells for phase 1 training.

        Args:
            char_indices: (N,) int array of embedding indices (not raw code points)
            fg_colors: (N,) int array of xterm-256 color indices
            bg_colors: (N,) int array of xterm-256 color indices

        Returns:
            (N, CELL_H, CELL_W, 3) float32 array in [0, 1]
        """
        # Clamp indices to valid range
        safe_indices = np.clip(char_indices, 0, N_CHARS_V2 - 1)
        safe_fg = np.clip(fg_colors, 0, N_COLORS_V2 - 1)
        safe_bg = np.clip(bg_colors, 0, N_COLORS_V2 - 1)

        alpha = self.glyph_table[safe_indices]      # (N, cell_h, cell_w)
        fg = self.palette[safe_fg]                   # (N, 3)
        bg = self.palette[safe_bg]                   # (N, 3)
        a = alpha[..., None]                         # (N, cell_h, cell_w, 1)
        return a * fg[:, None, None, :] + (1.0 - a) * bg[:, None, None, :]

    def render_cells(
        self,
        char_indices: np.ndarray,
        fg_colors: np.ndarray,
        bg_colors: np.ndarray,
    ) -> np.ndarray:
        """Vectorized full-frame rendering.

        Args:
            char_indices: (R, C) or (B, R, C) int array of embedding indices
            fg_colors: same shape, xterm-256 color indices
            bg_colors: same shape, xterm-256 color indices

        Returns:
            (H, W, 3) or (B, H, W, 3) float32 array in [0, 1]
        """
        batched = char_indices.ndim == 3
        if not batched:
            char_indices = char_indices[None]
            fg_colors = fg_colors[None]
            bg_colors = bg_colors[None]

        B, R, C = char_indices.shape

        safe_indices = np.clip(char_indices, 0, N_CHARS_V2 - 1)
        safe_fg = np.clip(fg_colors, 0, N_COLORS_V2 - 1)
        safe_bg = np.clip(bg_colors, 0, N_COLORS_V2 - 1)

        alpha = self.glyph_table[safe_indices]        # (B, R, C, cell_h, cell_w)
        fg_rgb = self.palette[safe_fg]                 # (B, R, C, 3)
        bg_rgb = self.palette[safe_bg]                 # (B, R, C, 3)

        a = alpha[..., None]                           # (B, R, C, cell_h, cell_w, 1)
        fg = fg_rgb[:, :, :, None, None, :]
        bg = bg_rgb[:, :, :, None, None, :]
        cells = a * fg + (1.0 - a) * bg

        cells = np.transpose(cells, (0, 1, 3, 2, 4, 5))
        frame = cells.reshape(B, R * CELL_H, C * CELL_W, 3)

        return frame if batched else frame[0]


# ═══════════════════════════════════════════════════════════════════════════
# V2 Training Data Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_v2_training_batch(
    batch_size: int = 512,
    rng: np.random.Generator | None = None,
    ascii_weight: float = 0.6,
    extended_weight: float = 0.4,
    char_pool: np.ndarray | None = None,
    n_colors: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a training batch with configurable character and color sets.

    Args:
        batch_size: Number of training samples
        rng: NumPy random generator (default: create new)
        ascii_weight: Fraction of samples using ASCII characters (32-126)
        extended_weight: Fraction using extended characters
        char_pool: If provided, sample characters from this pool of embedding
            indices instead of the default ASCII/extended split. Overrides
            ascii_weight and extended_weight.
        n_colors: Number of colors to sample from (16 for ANSI, 256 for xterm).

    Returns:
        Tuple of (char_indices, fg_colors, bg_colors) each as (N,) int64 arrays.
        char_indices are embedding indices (not raw code points).
    """
    if rng is None:
        rng = np.random.default_rng()

    if char_pool is not None:
        # Sample uniformly from the provided pool
        char_indices = rng.choice(char_pool, size=batch_size).astype(np.int64)
    else:
        n_ascii = int(batch_size * ascii_weight)
        n_extended = batch_size - n_ascii

        # ASCII portion: embedding index = code point for 0-255
        ascii_chars = rng.integers(32, 127, size=n_ascii).astype(np.int64)

        # Extended portion: sample from extended character embedding indices
        extended_indices = np.array(
            [idx for cp, idx in _CODEPOINT_TO_INDEX.items() if cp >= 128],
            dtype=np.int64,
        )
        if len(extended_indices) > 0 and n_extended > 0:
            ext_chars = rng.choice(extended_indices, size=n_extended).astype(np.int64)
        else:
            ext_chars = rng.integers(32, 127, size=n_extended).astype(np.int64)

        char_indices = np.concatenate([ascii_chars, ext_chars])

    # Colors: sample from [0, n_colors)
    if n_colors <= 16:
        fg_colors = rng.integers(0, n_colors, size=batch_size).astype(np.int64)
        bg_colors = rng.integers(0, n_colors, size=batch_size).astype(np.int64)
    else:
        # Bias toward common colors: 50% from ANSI 16, 30% from color cube, 20% from grayscale
        fg_colors = np.empty(batch_size, dtype=np.int64)
        bg_colors = np.empty(batch_size, dtype=np.int64)

        n_ansi = int(batch_size * 0.50)
        n_cube = int(batch_size * 0.30)
        n_gray = batch_size - n_ansi - n_cube

        fg_colors[:n_ansi] = rng.integers(0, 16, size=n_ansi)
        fg_colors[n_ansi:n_ansi + n_cube] = rng.integers(16, 232, size=n_cube)
        fg_colors[n_ansi + n_cube:] = rng.integers(232, 256, size=n_gray)

        bg_colors[:n_ansi] = rng.integers(0, 16, size=n_ansi)
        bg_colors[n_ansi:n_ansi + n_cube] = rng.integers(16, 232, size=n_cube)
        bg_colors[n_ansi + n_cube:] = rng.integers(232, 256, size=n_gray)

    # Shuffle to decorrelate
    shuffle_idx = rng.permutation(batch_size)
    char_indices = char_indices[shuffle_idx]
    fg_colors = fg_colors[shuffle_idx]
    bg_colors = bg_colors[shuffle_idx]

    return char_indices, fg_colors, bg_colors


def random_terminal_state_v2(rng: np.random.Generator | None = None):
    """Generate a random terminal screen using extended characters and 256 colors.

    Produces content similar to V1 but with:
      - Extended characters (box-drawing for TUI elements, Latin-1 for i18n)
      - xterm-256 color indices

    Returns:
        Tuple of (chars, fg, bg) as (TERM_ROWS, TERM_COLS) uint16/uint8 arrays.
        chars are embedding indices (not raw code points).
        fg/bg are xterm-256 color indices.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Use int16 for chars to support indices > 255
    chars = np.full((TERM_ROWS, TERM_COLS), ord(" "), dtype=np.int64)
    fg = np.full((TERM_ROWS, TERM_COLS), 7, dtype=np.int64)
    bg = np.zeros((TERM_ROWS, TERM_COLS), dtype=np.int64)

    # Content types with probabilities
    probs = [0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10]
    types = ["code", "shell", "text", "header", "tui_border", "i18n", "blank"]

    # Box-drawing character indices for TUI borders
    box_h = _CODEPOINT_TO_INDEX.get(0x2500, ord("-"))  # horizontal line
    box_v = _CODEPOINT_TO_INDEX.get(0x2502, ord("|"))  # vertical line
    box_tl = _CODEPOINT_TO_INDEX.get(0x250C, ord("+"))  # top-left corner
    box_tr = _CODEPOINT_TO_INDEX.get(0x2510, ord("+"))  # top-right corner
    box_bl = _CODEPOINT_TO_INDEX.get(0x2514, ord("+"))  # bottom-left corner
    box_br = _CODEPOINT_TO_INDEX.get(0x2518, ord("+"))  # bottom-right corner

    # Extended fg color pool (xterm-256 values commonly seen in terminals)
    code_colors = [2, 3, 6, 7, 11, 14, 34, 70, 106, 142, 178, 214]
    tui_colors = [33, 39, 45, 51, 87, 123, 159, 195]

    for r in range(TERM_ROWS):
        ctype = rng.choice(types, p=probs)

        if ctype == "code":
            indent = rng.integers(0, 16)
            kw = rng.choice(
                ["def ", "if ", "for ", "return ", "class ", "import ", "while ", "with "]
            )
            body = "".join(
                rng.choice(list("abcdefghijklmnopqrstuvwxyz_0123456789, ():.=+-*/"))
                for _ in range(rng.integers(10, 60))
            )
            line = " " * indent + kw + body
            line = line[: TERM_COLS]
            for c, ch in enumerate(line):
                chars[r, c] = ord(ch)
            fg[r, : len(line)] = rng.choice(code_colors, size=len(line))

        elif ctype == "shell":
            prompt = "$ "
            cmd = rng.choice(
                ["ls", "cd", "cat", "python3", "git", "make", "grep", "cargo", "echo"]
            )
            args = "".join(
                rng.choice(list("abcdefghijklmnopqrstuvwxyz_-./ "))
                for _ in range(rng.integers(5, 40))
            )
            line = (prompt + cmd + " " + args)[: TERM_COLS]
            for c, ch in enumerate(line):
                chars[r, c] = ord(ch)
            fg[r, :2] = 10  # green prompt
            fg[r, 2 : len(line)] = 7

        elif ctype == "text":
            pool = list(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:!?-+=()[]{}'\""
            )
            line = "".join(
                rng.choice(pool) for _ in range(rng.integers(20, TERM_COLS))
            )[: TERM_COLS]
            for c, ch in enumerate(line):
                chars[r, c] = ord(ch)
            fg[r, : len(line)] = rng.choice([7, 15, 250, 252], size=len(line))

        elif ctype == "header":
            sep = rng.choice(["=", "-", "#", "*"])
            line = (sep * TERM_COLS)[: TERM_COLS]
            for c, ch in enumerate(line):
                chars[r, c] = ord(ch)
            fg[r] = rng.choice([6, 14, 11, 33, 39])
            if rng.random() < 0.3:
                bg[r] = rng.choice([1, 4, 17, 52, 88])

        elif ctype == "tui_border":
            # Simulate TUI panel borders using box-drawing characters
            is_top = rng.random() < 0.3
            is_bottom = rng.random() < 0.3
            if is_top:
                chars[r, 0] = box_tl
                chars[r, TERM_COLS - 1] = box_tr
                chars[r, 1 : TERM_COLS - 1] = box_h
            elif is_bottom:
                chars[r, 0] = box_bl
                chars[r, TERM_COLS - 1] = box_br
                chars[r, 1 : TERM_COLS - 1] = box_h
            else:
                chars[r, 0] = box_v
                chars[r, TERM_COLS - 1] = box_v
                # Fill interior with spaces or text
                inner_text = "".join(
                    rng.choice(list("abcdefghijklmnopqrstuvwxyz 0123456789"))
                    for _ in range(TERM_COLS - 2)
                )
                for c, ch in enumerate(inner_text):
                    chars[r, c + 1] = ord(ch)
            fg[r] = rng.choice(tui_colors)

        elif ctype == "i18n":
            # Latin-1 accented text (indices map directly for code points < 256)
            latin1_chars = list(range(192, 256))  # accented letters
            line_len = rng.integers(20, TERM_COLS)
            for c in range(line_len):
                if rng.random() < 0.3:
                    # Use accented character
                    cp = rng.choice(latin1_chars)
                    chars[r, c] = cp  # Direct mapping for < 256
                else:
                    chars[r, c] = ord(rng.choice(list("abcdefghijklmnopqrstuvwxyz ")))
            fg[r, :line_len] = rng.choice([7, 15, 188, 224, 230], size=line_len)

        # "blank" type: leave as default spaces

    return chars, fg, bg


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def _build_char_pools() -> dict[str, np.ndarray]:
    """Build character pools for each curriculum stage."""
    # Stage 1: ASCII printable only (95 chars) — same density as V1
    ascii_pool = np.arange(32, 127, dtype=np.int64)

    # Stage 2: ASCII + Latin-1 supplement (191 chars)
    latin1_indices = [
        cp for cp in range(160, 256) if cp in _CODEPOINT_TO_INDEX
    ]
    stage2_pool = np.concatenate([
        ascii_pool,
        np.array(latin1_indices, dtype=np.int64),
    ])

    # Stage 3: Everything (465 chars)
    all_indices = np.array(TRAINABLE_INDICES, dtype=np.int64)

    return {
        "ascii": ascii_pool,
        "latin1": stage2_pool,
        "full": all_indices,
    }


def _run_cell_stage(
    model: NeuralTerminalRendererV2,
    conv: ConventionalRendererV2,
    device: str,
    char_pool: np.ndarray,
    n_colors: int,
    n_steps: int,
    batch_size: int,
    lr: float,
    stage_name: str,
    patience: int = 800,
    rng: np.random.Generator | None = None,
):
    """Run one curriculum stage of cell-level training with early stopping.

    Returns the best loss achieved and the optimizer (for LR continuity).
    """
    n_chars = len(char_pool)
    print(f"\n  ── {stage_name}: {n_chars} chars, {n_colors} colors, "
          f"{n_steps} steps, lr={lr:.1e} ──")

    optimizer = torch.optim.Adam(
        [
            {"params": model.glyphs.parameters(), "lr": lr},
            {"params": model.colors.parameters(), "lr": lr * 0.1},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_steps, eta_min=lr * 0.01
    )
    loss_fn = nn.L1Loss()
    if rng is None:
        rng = np.random.default_rng(42)

    model.train()
    t0 = time.perf_counter()
    best_loss = float("inf")
    steps_without_improvement = 0

    for step in range(n_steps):
        char_indices, fg_colors, bg_colors = generate_v2_training_batch(
            batch_size=batch_size, rng=rng,
            char_pool=char_pool, n_colors=n_colors,
        )

        target = conv.render_cell_batch(char_indices, fg_colors, bg_colors)
        target_t = torch.tensor(target, dtype=torch.float32, device=device)

        ch_t = torch.tensor(char_indices, dtype=torch.long, device=device)
        fg_t = torch.tensor(fg_colors, dtype=torch.long, device=device)
        bg_t = torch.tensor(bg_colors, dtype=torch.long, device=device)

        alpha = model.glyphs(ch_t)
        fg_rgb = model.colors(fg_t)
        bg_rgb = model.colors(bg_t)

        a = alpha.unsqueeze(-1)
        f = fg_rgb[:, None, None, :]
        b = bg_rgb[:, None, None, :]
        pred = a * f + (1.0 - a) * b

        loss = loss_fn(pred, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        if lv < best_loss:
            best_loss = lv
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if step % 500 == 0 or step == n_steps - 1:
            elapsed = time.perf_counter() - t0
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"    step {step:5d}/{n_steps}  loss={lv:.6f}  "
                f"best={best_loss:.6f}  lr={cur_lr:.2e}  ({elapsed:.1f}s)"
            )

        # Early stopping: if no improvement for `patience` steps, move on
        if steps_without_improvement >= patience and step >= 2000:
            elapsed = time.perf_counter() - t0
            print(f"    Early stop at step {step} (no improvement for {patience} steps)")
            print(f"    {stage_name} done: {elapsed:.1f}s, best loss={best_loss:.6f}")
            return best_loss

    elapsed = time.perf_counter() - t0
    print(f"    {stage_name} done: {elapsed:.1f}s, best loss={best_loss:.6f}")
    return best_loss


def train_phase1(
    model: NeuralTerminalRendererV2,
    conv: ConventionalRendererV2,
    device: str,
    n_steps: int = 10000,
    batch_size: int = 512,
    lr: float = 1e-3,
    curriculum: bool = True,
):
    """Phase 1: Cell-level training with progressive curriculum.

    Curriculum stages (when curriculum=True):
      Stage 1: ASCII-95 + 16 ANSI colors — builds sharp glyph foundations
      Stage 2: + Latin-1 (191 chars) + 16 colors — extends to accented chars
      Stage 3: Full 465 chars + 256 colors — complete character/color coverage

    Each stage uses cosine LR annealing and early stopping (patience=800).
    Stage transitions preserve all learned weights — new characters start
    from the embedding's random init while existing glyphs retain their
    trained representations.
    """
    print(f"\n{'=' * 60}")
    if curriculum:
        print(f"Phase 1: Progressive curriculum training")
    else:
        print(f"Phase 1: Cell-level training ({n_steps} steps, batch={batch_size})")
    print(f"  Characters: {len(TRAINABLE_CHARS)} trainable total")
    print(f"  Colors: {N_COLORS_V2} (xterm-256)")
    print(f"{'=' * 60}")

    if not curriculum:
        # Flat training (original behavior)
        _run_cell_stage(
            model, conv, device,
            char_pool=np.array(TRAINABLE_INDICES, dtype=np.int64),
            n_colors=256, n_steps=n_steps, batch_size=batch_size,
            lr=lr, stage_name="Flat (all chars)",
        )
        return

    pools = _build_char_pools()
    rng = np.random.default_rng(42)

    # Allocate steps per stage: 40% / 30% / 30%
    s1_steps = int(n_steps * 0.40)
    s2_steps = int(n_steps * 0.30)
    s3_steps = n_steps - s1_steps - s2_steps

    # Stage 1: ASCII-95 + 16 colors — same char density as V1
    _run_cell_stage(
        model, conv, device,
        char_pool=pools["ascii"], n_colors=16,
        n_steps=s1_steps, batch_size=batch_size, lr=lr,
        stage_name="Stage 1 (ASCII-95, 16 colors)",
        rng=rng,
    )

    # Stage 2: + Latin-1 (191 chars), still 16 colors
    # Lower LR for fine-tuning — don't destroy ASCII glyphs
    _run_cell_stage(
        model, conv, device,
        char_pool=pools["latin1"], n_colors=16,
        n_steps=s2_steps, batch_size=batch_size, lr=lr * 0.5,
        stage_name="Stage 2 (+ Latin-1 = 191 chars, 16 colors)",
        rng=rng,
    )

    # Stage 3: Full 465 chars + 256 colors
    _run_cell_stage(
        model, conv, device,
        char_pool=pools["full"], n_colors=256,
        n_steps=s3_steps, batch_size=batch_size, lr=lr * 0.3,
        stage_name="Stage 3 (full 465 chars, 256 colors)",
        rng=rng,
    )


def train_phase2(
    model: NeuralTerminalRendererV2,
    conv: ConventionalRendererV2,
    device: str,
    n_steps: int = 500,
    batch_size: int = 4,
    lr: float = 3e-4,
):
    """Phase 2: Fine-tune full pipeline on complete terminal frames.

    Uses randomly generated terminal screens with extended characters,
    box-drawing TUI elements, Latin-1 text, and 256-color backgrounds.
    """
    print(f"\n{'=' * 60}")
    print(f"Phase 2: Frame-level fine-tuning ({n_steps} steps, batch={batch_size})")
    print(f"{'=' * 60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_steps, eta_min=lr * 0.01
    )
    loss_fn = nn.L1Loss()
    rng = np.random.default_rng(123)

    model.train()
    t0 = time.perf_counter()
    best_loss = float("inf")

    for step in range(n_steps):
        batch_ch, batch_fg, batch_bg, batch_tgt = [], [], [], []

        for _ in range(batch_size):
            ch, fg_arr, bg_arr = random_terminal_state_v2(rng)
            tgt = conv.render_cells(ch, fg_arr, bg_arr)
            batch_ch.append(ch)
            batch_fg.append(fg_arr)
            batch_bg.append(bg_arr)
            batch_tgt.append(tgt)

        ch_t = torch.tensor(np.array(batch_ch), dtype=torch.long, device=device)
        fg_t = torch.tensor(np.array(batch_fg), dtype=torch.long, device=device)
        bg_t = torch.tensor(np.array(batch_bg), dtype=torch.long, device=device)
        tgt_t = torch.tensor(np.array(batch_tgt), dtype=torch.float32, device=device)

        pred = model(ch_t, fg_t, bg_t)
        loss = loss_fn(pred, tgt_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        best_loss = min(best_loss, lv)
        if step % 50 == 0 or step == n_steps - 1:
            elapsed = time.perf_counter() - t0
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"  step {step:5d}/{n_steps}  loss={lv:.6f}  "
                f"best={best_loss:.6f}  lr={cur_lr:.2e}  ({elapsed:.1f}s)"
            )

    print(f"  Phase 2 done: {time.perf_counter() - t0:.1f}s, final loss={best_loss:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Demo / Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def demo(model: NeuralTerminalRendererV2, conv: ConventionalRendererV2, device: str):
    """Render comparison images and metrics for V2 model."""
    print(f"\n{'=' * 60}")
    print("Demo: Conventional vs Neural V2 rendering")
    print(f"{'=' * 60}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.eval()

    # ── Random screen comparisons ────────────────────────────────────────
    for i, seed in enumerate([999, 42, 7]):
        rng = np.random.default_rng(seed)
        chars, fg_arr, bg_arr = random_terminal_state_v2(rng)

        conv_frame = conv.render_cells(chars, fg_arr, bg_arr)

        with torch.no_grad():
            ch_t = torch.tensor(chars, dtype=torch.long, device=device)
            fg_t = torch.tensor(fg_arr, dtype=torch.long, device=device)
            bg_t = torch.tensor(bg_arr, dtype=torch.long, device=device)
            neural_frame = model(ch_t, fg_t, bg_t).cpu().numpy()

        conv_img = Image.fromarray((conv_frame * 255).astype(np.uint8))
        neural_img = Image.fromarray((neural_frame * 255).astype(np.uint8))

        # Side-by-side comparison
        gap = 20
        comp = Image.new("RGB", (FRAME_W * 2 + gap, FRAME_H + 50), (20, 20, 20))
        comp.paste(conv_img, (0, 50))
        comp.paste(neural_img, (FRAME_W + gap, 50))

        draw = ImageDraw.Draw(comp)
        try:
            label_font = ImageFont.load_default(size=14)
        except TypeError:
            label_font = ImageFont.load_default()
        draw.text(
            (FRAME_W // 2 - 50, 15),
            "Conventional",
            fill=(180, 180, 180),
            font=label_font,
        )
        draw.text(
            (FRAME_W + gap + FRAME_W // 2 - 30, 15),
            "Neural V2",
            fill=(100, 255, 100),
            font=label_font,
        )

        path = MODEL_DIR / f"v2_comparison_{i}.png"
        comp.save(str(path))

        # Metrics
        mse = np.mean((conv_frame - neural_frame) ** 2)
        mae = np.mean(np.abs(conv_frame - neural_frame))
        psnr = -10 * np.log10(mse + 1e-10)
        print(f"  Sample {i}: MSE={mse:.6f}  MAE={mae:.4f}  PSNR={psnr:.1f}dB  -> {path}")

    # ── Shell session with V2 features ───────────────────────────────────
    state = TerminalState()
    shell_text = """\
\x1b[1;32m$ \x1b[0mls -la
total 42
drwxr-xr-x  10 user  staff   320 Apr 12 12:00 \x1b[1;34m.\x1b[0m
drwxr-xr-x   5 user  staff   160 Apr 11 09:15 \x1b[1;34m..\x1b[0m
-rw-r--r--   1 user  staff  4096 Apr 12 11:30 ncpu_paper.md
-rw-r--r--   1 user  staff  2048 Apr 12 10:00 neural_terminal_renderer_v2.py
-rwxr-xr-x   1 user  staff  8192 Apr 12 12:00 \x1b[1;32mtrain_terminal_renderer_v2.py\x1b[0m

\x1b[1;32m$ \x1b[0mpython3 ncpu/neural/train_terminal_renderer_v2.py
Device: mps
Model parameters: 200,547
  glyphs:  199,425  colors:  768  compositor:  354

\x1b[1;36mnCPU Neural Display V2: position-aware glyphs, 256 colors.\x1b[0m
"""
    state.write_str(shell_text)

    with torch.no_grad():
        ch_t, fg_t, bg_t, cur_t = state.to_tensors(device)
        neural_shell = model(ch_t, fg_t, bg_t, cur_t).cpu().numpy()

    shell_img = Image.fromarray((neural_shell * 255).astype(np.uint8))
    shell_path = MODEL_DIR / "v2_neural_shell_demo.png"
    shell_img.save(str(shell_path))
    print(f"  Shell demo: {shell_path}")

    # ── Glyph gallery: extended character set ────────────────────────────
    model.eval()
    with torch.no_grad():
        # Show ASCII printable range
        ascii_codes = torch.arange(32, 127, dtype=torch.long, device=device)
        alpha_ascii = model.glyphs(ascii_codes)  # (95, 16, 8)
        alpha_np = alpha_ascii.cpu().numpy()

    n_display = 95
    cols = 19
    rows_g = (n_display + cols - 1) // cols
    glyph_img = Image.new("L", (cols * (CELL_W + 2), rows_g * (CELL_H + 2)), 0)
    for idx in range(n_display):
        r_g, c_g = divmod(idx, cols)
        glyph = (alpha_np[idx] * 255).astype(np.uint8)
        cell_img = Image.fromarray(glyph, "L")
        glyph_img.paste(cell_img, (c_g * (CELL_W + 2) + 1, r_g * (CELL_H + 2) + 1))

    glyph_path = MODEL_DIR / "v2_neural_font_gallery.png"
    glyph_img.save(str(glyph_path))
    print(f"  Font gallery (ASCII): {glyph_path}")

    # ── Extended glyph gallery: box-drawing characters ───────────────────
    box_range = list(range(0x2500, 0x2580))
    box_indices = [_CODEPOINT_TO_INDEX.get(cp, 63) for cp in box_range]
    with torch.no_grad():
        box_codes = torch.tensor(box_indices, dtype=torch.long, device=device)
        alpha_box = model.glyphs(box_codes)
        alpha_box_np = alpha_box.cpu().numpy()

    n_box = len(box_range)
    cols_box = 16
    rows_box = (n_box + cols_box - 1) // cols_box
    box_img = Image.new("L", (cols_box * (CELL_W + 2), rows_box * (CELL_H + 2)), 0)
    for idx in range(n_box):
        r_b, c_b = divmod(idx, cols_box)
        glyph = (alpha_box_np[idx] * 255).astype(np.uint8)
        cell_img = Image.fromarray(glyph, "L")
        box_img.paste(cell_img, (c_b * (CELL_W + 2) + 1, r_b * (CELL_H + 2) + 1))

    box_path = MODEL_DIR / "v2_box_drawing_gallery.png"
    box_img.save(str(box_path))
    print(f"  Font gallery (box-drawing): {box_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Force unbuffered output so background training shows progress
    import functools
    import builtins
    builtins.print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser(
        description="Train Neural Terminal Renderer V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--epochs", type=int, default=30000, help="Phase 1 total training steps")
    parser.add_argument(
        "--phase2-steps", type=int, default=3000, help="Phase 2 training steps"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Phase 1 learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Phase 1 batch size")
    parser.add_argument(
        "--no-curriculum", action="store_true",
        help="Disable progressive curriculum (train all chars from start)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Custom model save path (default: models/display/terminal_renderer_v2.pt)",
    )
    parser.add_argument("--demo-only", action="store_true", help="Skip training, run demo")
    parser.add_argument(
        "--efficient-compositor",
        action="store_true",
        help="Use depthwise-separable compositor (fewer params)",
    )
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    save_path = Path(args.save_path) if args.save_path else MODEL_PATH
    print(f"Device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = NeuralTerminalRendererV2(
        efficient_compositor=args.efficient_compositor,
    ).to(device)

    # ── Parameter report ─────────────────────────────────────────────────
    total = model.count_params()
    by_component = model.count_params_by_component()
    print(f"Model parameters: {total:,}")
    for name, count in by_component.items():
        print(f"  {name:15s}: {count:>8,}")

    if args.demo_only:
        if save_path.exists():
            model.load_state_dict(
                torch.load(str(save_path), map_location=device, weights_only=True)
            )
            print(f"Loaded: {save_path}")
        else:
            print(f"No model found at {save_path} -- running demo with untrained model")
        conv = ConventionalRendererV2()
        demo(model, conv, device)
        return

    # Build conventional renderer (ground truth)
    print("Building conventional renderer V2 (pre-rendering extended glyphs)...")
    conv = ConventionalRendererV2()

    # Phase 1: cell-level training (with curriculum by default)
    train_phase1(
        model, conv, device,
        n_steps=args.epochs, batch_size=args.batch_size, lr=args.lr,
        curriculum=not args.no_curriculum,
    )

    # Phase 2: frame-level fine-tuning
    train_phase2(model, conv, device, n_steps=args.phase2_steps)

    # Save
    torch.save(model.state_dict(), str(save_path))
    size_kb = save_path.stat().st_size / 1024
    print(f"\nModel saved: {save_path} ({size_kb:.0f} KB)")

    # Demo
    demo(model, conv, device)

    print(f"\n{'=' * 60}")
    print(f"Neural Terminal Renderer V2: {total:,} params, {size_kb:.0f} KB")
    for name, count in by_component.items():
        print(f"  {name:15s}: {count:>8,}")
    print(f"Position-aware glyphs | 1024 chars | 256 colors | neural display")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
