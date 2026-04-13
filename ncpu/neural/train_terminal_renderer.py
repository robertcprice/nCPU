#!/usr/bin/env python3
"""
Training pipeline for the Neural Terminal Renderer.

Phase 1: Cell-level — train glyph generator + palette on individual chars
Phase 2: Frame-level — fine-tune full pipeline on complete terminal screens

Ground truth: PIL rendering with system monospace font.
After training, the neural renderer replaces conventional rendering entirely.

Usage:
    python ncpu/neural/train_terminal_renderer.py
    python ncpu/neural/train_terminal_renderer.py --demo-only
    python ncpu/neural/train_terminal_renderer.py --phase1-steps 3000 --phase2-steps 800
"""

import sys
import platform
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ncpu.neural.neural_terminal_renderer import (
    NeuralTerminalRenderer, TerminalState,
    TERM_ROWS, TERM_COLS, CELL_H, CELL_W,
    FRAME_H, FRAME_W, N_CHARS, N_COLORS, ANSI_PALETTE,
)

MODEL_DIR = Path(__file__).parent.parent.parent / 'models' / 'display'
MODEL_PATH = MODEL_DIR / 'terminal_renderer.pt'


# ═══════════════════════════════════════════════════════════════════════════
# Conventional Renderer (ground truth for training)
# ═══════════════════════════════════════════════════════════════════════════

def _find_monospace_font():
    """Find a monospace font that fits our 8×16 cell size."""
    candidates = []
    if platform.system() == 'Darwin':
        candidates = [
            ('/System/Library/Fonts/Menlo.ttc', [13, 14, 12]),
            ('/System/Library/Fonts/Monaco.dfont', [12, 13, 11]),
            ('/Library/Fonts/Courier New.ttf', [13, 14, 12]),
        ]
    elif platform.system() == 'Linux':
        candidates = [
            ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', [12, 13, 11]),
            ('/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', [13, 14, 12]),
        ]

    for path, sizes in candidates:
        for size in sizes:
            try:
                font = ImageFont.truetype(path, size)
                bbox = font.getbbox('M')
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w <= CELL_W + 1 and h <= CELL_H + 1:
                    return font
            except (IOError, OSError):
                break

    # Fallback to PIL default
    try:
        return ImageFont.load_default(size=12)
    except TypeError:
        return ImageFont.load_default()


class ConventionalRenderer:
    """PIL-based terminal renderer — generates ground truth for training."""

    def __init__(self):
        self.font = _find_monospace_font()
        self.palette = np.array(ANSI_PALETTE, dtype=np.float32) / 255.0
        self.glyph_table = self._prerender_all_glyphs()

    def _prerender_all_glyphs(self):
        """Pre-render all 256 character glyphs as (256, cell_h, cell_w) alpha masks."""
        masks = np.zeros((N_CHARS, CELL_H, CELL_W), dtype=np.float32)
        for ch in range(32, 127):
            img = Image.new('L', (CELL_W, CELL_H), 0)
            draw = ImageDraw.Draw(img)
            try:
                draw.text((0, 0), chr(ch), fill=255, font=self.font)
            except Exception:
                pass
            masks[ch] = np.array(img, dtype=np.float32) / 255.0
        return masks

    def render_cells(self, char_codes, fg_colors, bg_colors):
        """Vectorized full-frame rendering.

        Args: char_codes (R,C) or (B,R,C), fg/bg same shape, dtype uint8
        Returns: frame (H,W,3) or (B,H,W,3) float32 [0,1]
        """
        batched = char_codes.ndim == 3
        if not batched:
            char_codes = char_codes[None]
            fg_colors = fg_colors[None]
            bg_colors = bg_colors[None]

        B, R, C = char_codes.shape

        # Glyph masks: (B, R, C, ch, cw)
        alpha = self.glyph_table[char_codes]

        # Colors: (B, R, C, 3)
        fg_rgb = self.palette[fg_colors]
        bg_rgb = self.palette[bg_colors]

        # Blend: alpha * fg + (1-alpha) * bg → (B, R, C, ch, cw, 3)
        a  = alpha[..., None]                     # (B, R, C, ch, cw, 1)
        fg = fg_rgb[:, :, :, None, None, :]       # (B, R, C, 1,  1,  3)
        bg = bg_rgb[:, :, :, None, None, :]
        cells = a * fg + (1.0 - a) * bg

        # Assemble: (B, R, C, ch, cw, 3) → (B, R*ch, C*cw, 3)
        cells = np.transpose(cells, (0, 1, 3, 2, 4, 5))   # (B, R, ch, C, cw, 3)
        frame = cells.reshape(B, R * CELL_H, C * CELL_W, 3)

        return frame if batched else frame[0]

    def render_cell_batch(self, char_codes, fg_colors, bg_colors):
        """Render individual cells for phase 1.

        Args: char_codes (N,), fg_colors (N,), bg_colors (N,) — int arrays
        Returns: (N, ch, cw, 3) float32
        """
        alpha = self.glyph_table[char_codes]        # (N, ch, cw)
        fg = self.palette[fg_colors]                 # (N, 3)
        bg = self.palette[bg_colors]                 # (N, 3)
        a  = alpha[..., None]                        # (N, ch, cw, 1)
        return a * fg[:, None, None, :] + (1.0 - a) * bg[:, None, None, :]


# ═══════════════════════════════════════════════════════════════════════════
# Data Generation
# ═══════════════════════════════════════════════════════════════════════════

def random_terminal_state(rng=None):
    """Generate a random terminal screen with realistic content."""
    if rng is None:
        rng = np.random.default_rng()

    chars = np.full((TERM_ROWS, TERM_COLS), ord(' '), dtype=np.uint8)
    fg = np.full((TERM_ROWS, TERM_COLS), 7, dtype=np.uint8)
    bg = np.zeros((TERM_ROWS, TERM_COLS), dtype=np.uint8)

    probs = [0.30, 0.25, 0.20, 0.10, 0.15]
    types = ['code', 'shell', 'text', 'header', 'blank']

    for r in range(TERM_ROWS):
        ctype = rng.choice(types, p=probs)

        if ctype == 'code':
            indent = rng.integers(0, 16)
            kw = rng.choice(['def ', 'if ', 'for ', 'return ', 'class ', 'import ', 'while ', 'with '])
            body = ''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz_0123456789, ():.=+-*/'))
                          for _ in range(rng.integers(10, 60)))
            line = ' ' * indent + kw + body
            line = line[:TERM_COLS]
            for c, ch in enumerate(line):
                chars[r, c] = ord(ch)
            fg[r, :len(line)] = rng.choice([2, 3, 6, 7, 11, 14], size=len(line)).astype(np.uint8)

        elif ctype == 'shell':
            prompt = '$ '
            cmd = rng.choice(['ls', 'cd', 'cat', 'python3', 'git', 'make', 'grep', 'cargo', 'echo'])
            args = ''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz_-./ '))
                          for _ in range(rng.integers(5, 40)))
            line = (prompt + cmd + ' ' + args)[:TERM_COLS]
            for c, ch in enumerate(line):
                chars[r, c] = ord(ch)
            fg[r, :2] = 10   # green prompt
            fg[r, 2:len(line)] = 7

        elif ctype == 'text':
            pool = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:!?-+=()[]{}"\'')
            line = ''.join(rng.choice(pool) for _ in range(rng.integers(20, TERM_COLS)))[:TERM_COLS]
            for c, ch in enumerate(line):
                chars[r, c] = ord(ch)
            fg[r, :len(line)] = rng.choice([7, 15], size=len(line)).astype(np.uint8)

        elif ctype == 'header':
            sep = rng.choice(['=', '-', '#', '*'])
            line = (sep * TERM_COLS)[:TERM_COLS]
            for c, ch in enumerate(line):
                chars[r, c] = ord(ch)
            fg[r] = rng.choice([6, 14, 11])
            # Sometimes add colored background
            if rng.random() < 0.3:
                bg[r] = rng.choice([1, 4])

    return chars, fg, bg


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_phase1(model, conv, device, n_steps=2000, batch_size=512, lr=1e-3):
    """Phase 1: Train glyph generator + palette on individual cells."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Cell-level training ({n_steps} steps, batch={batch_size})")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam([
        {'params': model.glyphs.parameters(), 'lr': lr},
        {'params': model.colors.parameters(), 'lr': lr * 0.1},
    ])
    loss_fn = nn.L1Loss()
    rng = np.random.default_rng(42)

    model.train()
    t0 = time.perf_counter()
    best_loss = float('inf')

    for step in range(n_steps):
        # Random (char, fg, bg) triples
        ch = rng.integers(32, 127, size=batch_size).astype(np.int64)
        fg = rng.integers(0, N_COLORS, size=batch_size).astype(np.int64)
        bg = rng.integers(0, N_COLORS, size=batch_size).astype(np.int64)

        # Ground truth
        target = conv.render_cell_batch(ch, fg, bg)
        target_t = torch.tensor(target, dtype=torch.float32, device=device)

        # Neural forward (cell-level, bypass compositor)
        ch_t = torch.tensor(ch, dtype=torch.long, device=device)
        fg_t = torch.tensor(fg, dtype=torch.long, device=device)
        bg_t = torch.tensor(bg, dtype=torch.long, device=device)

        alpha = model.glyphs(ch_t)             # (N, ch, cw)
        fg_rgb = model.colors(fg_t)            # (N, 3)
        bg_rgb = model.colors(bg_t)            # (N, 3)

        a  = alpha.unsqueeze(-1)               # (N, ch, cw, 1)
        f  = fg_rgb[:, None, None, :]          # (N, 1, 1, 3)
        b  = bg_rgb[:, None, None, :]
        pred = a * f + (1.0 - a) * b           # (N, ch, cw, 3)

        loss = loss_fn(pred, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lv = loss.item()
        best_loss = min(best_loss, lv)
        if step % 200 == 0 or step == n_steps - 1:
            elapsed = time.perf_counter() - t0
            print(f"  step {step:5d}/{n_steps}  loss={lv:.6f}  best={best_loss:.6f}  ({elapsed:.1f}s)")

    print(f"  Phase 1 done: {time.perf_counter()-t0:.1f}s, final loss={best_loss:.6f}")


def train_phase2(model, conv, device, n_steps=500, batch_size=4, lr=3e-4):
    """Phase 2: Fine-tune full pipeline on complete terminal frames."""
    print(f"\n{'='*60}")
    print(f"Phase 2: Frame-level fine-tuning ({n_steps} steps, batch={batch_size})")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, eta_min=lr * 0.01)
    loss_fn = nn.L1Loss()
    rng = np.random.default_rng(123)

    model.train()
    t0 = time.perf_counter()
    best_loss = float('inf')

    for step in range(n_steps):
        batch_ch, batch_fg, batch_bg, batch_tgt = [], [], [], []

        for _ in range(batch_size):
            ch, fg, bg = random_terminal_state(rng)
            tgt = conv.render_cells(ch, fg, bg)
            batch_ch.append(ch)
            batch_fg.append(fg)
            batch_bg.append(bg)
            batch_tgt.append(tgt)

        ch_t  = torch.tensor(np.array(batch_ch),  dtype=torch.long,    device=device)
        fg_t  = torch.tensor(np.array(batch_fg),  dtype=torch.long,    device=device)
        bg_t  = torch.tensor(np.array(batch_bg),  dtype=torch.long,    device=device)
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
            print(f"  step {step:5d}/{n_steps}  loss={lv:.6f}  best={best_loss:.6f}  lr={cur_lr:.2e}  ({elapsed:.1f}s)")

    print(f"  Phase 2 done: {time.perf_counter()-t0:.1f}s, final loss={best_loss:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Demo / Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def demo(model, conv, device):
    """Render comparison images: conventional vs neural."""
    print(f"\n{'='*60}")
    print("Demo: Conventional vs Neural rendering")
    print(f"{'='*60}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(999)
    model.eval()

    # ── Render a few sample screens ────────────────────────────────────
    for i, seed in enumerate([999, 42, 7]):
        rng = np.random.default_rng(seed)
        chars, fg_arr, bg_arr = random_terminal_state(rng)

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
        comp = Image.new('RGB', (FRAME_W * 2 + gap, FRAME_H + 50), (20, 20, 20))
        comp.paste(conv_img, (0, 50))
        comp.paste(neural_img, (FRAME_W + gap, 50))

        draw = ImageDraw.Draw(comp)
        try:
            label_font = ImageFont.load_default(size=14)
        except TypeError:
            label_font = ImageFont.load_default()
        draw.text((FRAME_W // 2 - 50, 15), "Conventional", fill=(180, 180, 180), font=label_font)
        draw.text((FRAME_W + gap + FRAME_W // 2 - 30, 15), "Neural", fill=(100, 255, 100), font=label_font)

        path = MODEL_DIR / f'comparison_{i}.png'
        comp.save(str(path))

        # Metrics
        mse = np.mean((conv_frame - neural_frame) ** 2)
        mae = np.mean(np.abs(conv_frame - neural_frame))
        psnr = -10 * np.log10(mse + 1e-10)
        print(f"  Sample {i}: MSE={mse:.6f}  MAE={mae:.4f}  PSNR={psnr:.1f}dB  → {path}")

    # ── Also render a realistic shell session ──────────────────────────
    state = TerminalState()
    shell_text = """\
\x1b[1;32m$ \x1b[0mls -la
total 42
drwxr-xr-x  10 user  staff   320 Apr 11 12:00 \x1b[1;34m.\x1b[0m
drwxr-xr-x   5 user  staff   160 Apr 10 09:15 \x1b[1;34m..\x1b[0m
-rw-r--r--   1 user  staff  4096 Apr 11 11:30 ncpu_paper.md
-rw-r--r--   1 user  staff  2048 Apr 11 10:00 neural_terminal_renderer.py
-rwxr-xr-x   1 user  staff  8192 Apr 11 12:00 \x1b[1;32mtrain_terminal_renderer.py\x1b[0m

\x1b[1;32m$ \x1b[0mpython3 ncpu/neural/train_terminal_renderer.py
Device: mps
Model parameters: 143,539
Building conventional renderer (pre-rendering glyphs)...

============================================================
Phase 1: Cell-level training (2000 steps, batch=512)
============================================================
  step     0/2000  loss=0.156789  best=0.156789  (0.1s)
  \x1b[1;33m[TRAINING COMPLETE]\x1b[0m

\x1b[1;36mnCPU Neural Display: every pixel is neural.\x1b[0m
"""
    state.write_str(shell_text)

    with torch.no_grad():
        ch_t, fg_t, bg_t, cur_t = state.to_tensors(device)
        neural_shell = model(ch_t, fg_t, bg_t, cur_t).cpu().numpy()

    shell_img = Image.fromarray((neural_shell * 255).astype(np.uint8))
    shell_path = MODEL_DIR / 'neural_shell_demo.png'
    shell_img.save(str(shell_path))
    print(f"  Shell demo: {shell_path}")

    # ── Glyph gallery: show all learned character shapes ───────────────
    model.eval()
    with torch.no_grad():
        all_chars = torch.arange(32, 127, dtype=torch.long, device=device)
        alpha = model.glyphs(all_chars)  # (95, 16, 8)
        alpha_np = alpha.cpu().numpy()

    n_chars = 95
    cols = 19
    rows_g = (n_chars + cols - 1) // cols
    glyph_img = Image.new('L', (cols * (CELL_W + 2), rows_g * (CELL_H + 2)), 0)
    for idx in range(n_chars):
        r, c = divmod(idx, cols)
        glyph = (alpha_np[idx] * 255).astype(np.uint8)
        cell_img = Image.fromarray(glyph, 'L')
        glyph_img.paste(cell_img, (c * (CELL_W + 2) + 1, r * (CELL_H + 2) + 1))

    glyph_path = MODEL_DIR / 'neural_font_gallery.png'
    glyph_img.save(str(glyph_path))
    print(f"  Font gallery: {glyph_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Neural Terminal Renderer")
    parser.add_argument('--device', default=None)
    parser.add_argument('--phase1-steps', type=int, default=2000)
    parser.add_argument('--phase2-steps', type=int, default=500)
    parser.add_argument('--demo-only', action='store_true')
    args = parser.parse_args()

    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = NeuralTerminalRenderer().to(device)
    n_params = model.count_params()
    print(f"Model parameters: {n_params:,}")

    if args.demo_only:
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device, weights_only=True))
            print(f"Loaded: {MODEL_PATH}")
        else:
            print(f"No model found at {MODEL_PATH} — running demo with untrained model")
        conv = ConventionalRenderer()
        demo(model, conv, device)
        return

    # Build conventional renderer
    print("Building conventional renderer (pre-rendering glyphs)...")
    conv = ConventionalRenderer()

    # Phase 1: cell-level training
    train_phase1(model, conv, device, n_steps=args.phase1_steps)

    # Phase 2: frame-level fine-tuning
    train_phase2(model, conv, device, n_steps=args.phase2_steps)

    # Save
    torch.save(model.state_dict(), str(MODEL_PATH))
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"\nModel saved: {MODEL_PATH} ({size_kb:.0f} KB)")

    # Demo
    demo(model, conv, device)

    print(f"\n{'='*60}")
    print(f"Neural Terminal Renderer: {n_params:,} params, {size_kb:.0f} KB")
    print(f"Every pixel in the output frame is produced by neural computation.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
