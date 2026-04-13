"""
Neural Terminal Renderer — fully neural text-to-pixel display pipeline.

Replaces conventional font rasterization with a trained neural pipeline:
  char_code → NeuralGlyphGenerator (embedding + MLP) → alpha mask
  color_code → NeuralColorPalette (learned embedding) → RGB
  alpha * fg + (1-a) * bg → cell pixels
  assembled frame → NeuralCompositor (ConvNet) → refined frame

Every pixel in the output frame is produced by neural network forward passes.
No lookup tables, no bitmap fonts, no conventional rasterization.

Components:
  TerminalState        — VT100 state machine tracking char grid from byte stream
  NeuralGlyphGenerator — char embedding → MLP → 8×16 alpha mask (learned font)
  NeuralColorPalette   — ANSI color code → RGB (learned color embedding)
  NeuralCompositor     — ConvNet post-processing (anti-aliasing, refinement)
  NeuralTerminalRenderer — full pipeline orchestrator
  NeuralDisplay        — integration class for wiring into runner.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

# ─── Terminal geometry ──────────────────────────────────────────────────────
TERM_ROWS, TERM_COLS = 24, 80
CELL_H, CELL_W = 16, 8
FRAME_H = TERM_ROWS * CELL_H   # 384
FRAME_W = TERM_COLS * CELL_W    # 640
N_CHARS = 256
N_COLORS = 16

# Standard ANSI terminal palette (sRGB, 8 normal + 8 bright)
ANSI_PALETTE = [
    (0, 0, 0),       (170, 0, 0),     (0, 170, 0),     (170, 85, 0),
    (0, 0, 170),     (170, 0, 170),   (0, 170, 170),   (170, 170, 170),
    (85, 85, 85),    (255, 85, 85),   (85, 255, 85),   (255, 255, 85),
    (85, 85, 255),   (255, 85, 255),  (85, 255, 255),  (255, 255, 255),
]


# ═══════════════════════════════════════════════════════════════════════════
# Terminal State Tracker
# ═══════════════════════════════════════════════════════════════════════════

class TerminalState:
    """Minimal VT100 terminal emulator — tracks character grid from byte stream.

    Handles: printable ASCII, newline, CR, tab, backspace, and basic ANSI
    escape sequences (SGR colors, cursor positioning, erase).
    """

    def __init__(self, rows: int = TERM_ROWS, cols: int = TERM_COLS):
        self.rows = rows
        self.cols = cols
        self.reset()

    # UTF-8 box-drawing → ASCII fallback map
    _UNICODE_FALLBACK = {
        0x2500: ord('-'), 0x2501: ord('='), 0x2502: ord('|'), 0x2503: ord('|'),
        0x250C: ord('+'), 0x250F: ord('+'), 0x2510: ord('+'), 0x2513: ord('+'),
        0x2514: ord('+'), 0x2517: ord('+'), 0x2518: ord('+'), 0x251B: ord('+'),
        0x251C: ord('+'), 0x2523: ord('+'), 0x2524: ord('+'), 0x252B: ord('+'),
        0x252C: ord('+'), 0x2533: ord('+'), 0x2534: ord('+'), 0x253B: ord('+'),
        0x253C: ord('+'), 0x254B: ord('+'),
        0x2550: ord('='), 0x2551: ord('|'),  # ═ ║
        0x2552: ord('+'), 0x2553: ord('+'), 0x2554: ord('+'), 0x2555: ord('+'),
        0x2556: ord('+'), 0x2557: ord('+'), 0x2558: ord('+'), 0x2559: ord('+'),
        0x255A: ord('+'), 0x255B: ord('+'), 0x255C: ord('+'), 0x255D: ord('+'),
        0x255E: ord('+'), 0x255F: ord('+'), 0x2560: ord('+'), 0x2561: ord('+'),
        0x2562: ord('+'), 0x2563: ord('+'), 0x2564: ord('+'), 0x2565: ord('+'),
        0x2566: ord('+'), 0x2567: ord('+'), 0x2568: ord('+'), 0x2569: ord('+'),
        0x256A: ord('+'), 0x256B: ord('+'), 0x256C: ord('+'),
        0x2580: ord('#'), 0x2584: ord('#'), 0x2588: ord('#'),  # block elements
        0x2591: ord('.'), 0x2592: ord(':'), 0x2593: ord('#'),  # shade
        0x25A0: ord('#'), 0x25CF: ord('*'),  # filled shapes
        0x2190: ord('<'), 0x2191: ord('^'), 0x2192: ord('>'), 0x2193: ord('v'),
    }

    def reset(self):
        self.chars = np.full((self.rows, self.cols), ord(' '), dtype=np.uint8)
        self.fg = np.full((self.rows, self.cols), 7, dtype=np.uint8)   # default white
        self.bg = np.zeros((self.rows, self.cols), dtype=np.uint8)     # default black
        self.cr = 0   # cursor row
        self.cc = 0   # cursor col
        self.cur_fg = 7
        self.cur_bg = 0
        self._esc = 0       # 0=normal, 1=got ESC, 2=got CSI '['
        self._buf = []
        self._utf8_buf = []  # accumulator for multi-byte UTF-8
        self._utf8_rem = 0   # remaining bytes in current UTF-8 sequence

    def write(self, data: bytes):
        """Feed a byte stream (from SYS_WRITE) into the terminal."""
        for b in data:
            self._byte(b)

    def write_str(self, text: str):
        """Convenience: feed a Python string (handles UTF-8)."""
        self.write(text.encode('utf-8'))

    # ── Byte processing state machine ──────────────────────────────────

    def _byte(self, b: int):
        # ── UTF-8 multi-byte accumulation ─────────────────────────────
        if self._utf8_rem > 0:
            if (b & 0xC0) == 0x80:              # continuation byte
                self._utf8_buf.append(b)
                self._utf8_rem -= 1
                if self._utf8_rem == 0:
                    self._emit_utf8()
            else:
                # Invalid continuation — discard and reprocess
                self._utf8_buf = []
                self._utf8_rem = 0
                self._byte(b)
            return

        # ── UTF-8 start byte detection ────────────────────────────────
        if self._esc == 0 and (b & 0xE0) == 0xC0:      # 2-byte: 110xxxxx
            self._utf8_buf = [b]
            self._utf8_rem = 1
            return
        elif self._esc == 0 and (b & 0xF0) == 0xE0:    # 3-byte: 1110xxxx
            self._utf8_buf = [b]
            self._utf8_rem = 2
            return
        elif self._esc == 0 and (b & 0xF8) == 0xF0:    # 4-byte: 11110xxx
            self._utf8_buf = [b]
            self._utf8_rem = 3
            return

        # ── Normal single-byte processing ─────────────────────────────
        if self._esc == 0:
            if b == 0x1B:                       # ESC
                self._esc = 1
                self._buf = []
            elif b == 0x0A:                     # LF
                self._newline()
            elif b == 0x0D:                     # CR
                self.cc = 0
            elif b == 0x09:                     # TAB
                self.cc = min(self.cc + (8 - self.cc % 8), self.cols - 1)
            elif b == 0x08:                     # BS
                self.cc = max(0, self.cc - 1)
            elif 0x20 <= b <= 0x7E:             # printable ASCII
                self._put(b)
        elif self._esc == 1:
            self._esc = 2 if b == 0x5B else 0   # '[' → CSI
        elif self._esc == 2:
            if 0x40 <= b <= 0x7E:               # final byte
                self._buf.append(b)
                self._csi()
                self._esc = 0
                self._buf = []
            else:
                self._buf.append(b)             # parameter/intermediate byte

    def _emit_utf8(self):
        """Decode a completed UTF-8 sequence and map to displayable char."""
        try:
            ch = bytes(self._utf8_buf).decode('utf-8')
            cp = ord(ch)
        except (UnicodeDecodeError, ValueError):
            self._utf8_buf = []
            return
        self._utf8_buf = []
        # Map Unicode to ASCII fallback
        mapped = self._UNICODE_FALLBACK.get(cp, ord('?') if cp > 127 else cp)
        if 0x20 <= mapped <= 0x7E:
            self._put(mapped)

    def _put(self, ch: int):
        if self.cc >= self.cols:
            self._newline()
            self.cc = 0
        self.chars[self.cr, self.cc] = ch
        self.fg[self.cr, self.cc] = self.cur_fg
        self.bg[self.cr, self.cc] = self.cur_bg
        self.cc += 1

    def _newline(self):
        self.cr += 1
        self.cc = 0
        if self.cr >= self.rows:
            # Scroll up
            self.chars[:-1] = self.chars[1:]
            self.fg[:-1] = self.fg[1:]
            self.bg[:-1] = self.bg[1:]
            self.chars[-1] = ord(' ')
            self.fg[-1] = 7
            self.bg[-1] = 0
            self.cr = self.rows - 1

    # ── CSI sequence handler ───────────────────────────────────────────

    def _csi(self):
        buf = bytes(self._buf)
        final = chr(buf[-1])
        raw = buf[:-1].decode('ascii', errors='ignore')

        if final == 'm':                        # SGR — colors/attributes
            parts = raw.split(';') if raw else ['0']
            for p in parts:
                p = p.strip()
                code = int(p) if p.isdigit() else 0
                if code == 0:                   # reset
                    self.cur_fg, self.cur_bg = 7, 0
                elif code == 1:                 # bold → bright
                    if self.cur_fg < 8:
                        self.cur_fg += 8
                elif 30 <= code <= 37:          # fg normal
                    self.cur_fg = code - 30
                elif 40 <= code <= 47:          # bg normal
                    self.cur_bg = code - 40
                elif 90 <= code <= 97:          # fg bright
                    self.cur_fg = code - 82
                elif 100 <= code <= 107:        # bg bright
                    self.cur_bg = code - 92

        elif final in ('H', 'f'):               # CUP — cursor position
            parts = raw.split(';')
            r = int(parts[0]) - 1 if parts[0].isdigit() else 0
            c = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else 0
            self.cr = max(0, min(r, self.rows - 1))
            self.cc = max(0, min(c, self.cols - 1))

        elif final == 'J':                      # ED — erase display
            p = int(raw) if raw.isdigit() else 0
            if p == 2:
                self.chars[:] = ord(' ')
                self.fg[:] = 7
                self.bg[:] = 0

        elif final == 'K':                      # EL — erase line
            p = int(raw) if raw.isdigit() else 0
            if p == 0:  # cursor to end
                self.chars[self.cr, self.cc:] = ord(' ')
                self.fg[self.cr, self.cc:] = 7
                self.bg[self.cr, self.cc:] = 0
            elif p == 2:  # entire line
                self.chars[self.cr] = ord(' ')
                self.fg[self.cr] = 7
                self.bg[self.cr] = 0

        elif final == 'A':                      # CUU — cursor up
            n = int(raw) if raw.isdigit() else 1
            self.cr = max(0, self.cr - n)
        elif final == 'B':                      # CUD — cursor down
            n = int(raw) if raw.isdigit() else 1
            self.cr = min(self.rows - 1, self.cr + n)
        elif final == 'C':                      # CUF — cursor forward
            n = int(raw) if raw.isdigit() else 1
            self.cc = min(self.cols - 1, self.cc + n)
        elif final == 'D':                      # CUB — cursor backward
            n = int(raw) if raw.isdigit() else 1
            self.cc = max(0, self.cc - n)

    # ── Tensor export ──────────────────────────────────────────────────

    def to_tensors(self, device='cpu'):
        """Convert current grid state to tensors for the neural renderer."""
        chars = torch.tensor(self.chars.copy(), dtype=torch.long, device=device)
        fg = torch.tensor(self.fg.copy(), dtype=torch.long, device=device)
        bg = torch.tensor(self.bg.copy(), dtype=torch.long, device=device)
        cursor = torch.zeros(self.rows, self.cols, dtype=torch.bool, device=device)
        if 0 <= self.cr < self.rows and 0 <= self.cc < self.cols:
            cursor[self.cr, self.cc] = True
        return chars, fg, bg, cursor


# ═══════════════════════════════════════════════════════════════════════════
# Neural Rendering Models
# ═══════════════════════════════════════════════════════════════════════════

class NeuralGlyphGenerator(nn.Module):
    """Generates character glyph alpha masks via learned embeddings + MLP.

    The character embedding space learns typographic structure: visually
    similar characters (O/0, l/1/I) cluster naturally, and the MLP learns
    to generate their pixel-level shapes.

    Input:  char codes (...,) int tensor
    Output: alpha masks (..., cell_h, cell_w) float tensor in [0, 1]
    """

    def __init__(self, n_chars=N_CHARS, cell_h=CELL_H, cell_w=CELL_W, embed_dim=64):
        super().__init__()
        self.cell_h = cell_h
        self.cell_w = cell_w
        self.embed = nn.Embedding(n_chars, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, cell_h * cell_w),
            nn.Sigmoid(),
        )

    def forward(self, codes):
        shape = codes.shape
        e = self.embed(codes.reshape(-1))
        a = self.net(e)
        return a.view(*shape, self.cell_h, self.cell_w)


class NeuralColorPalette(nn.Module):
    """Learned ANSI color palette — maps 16 terminal color codes to RGB.

    Initialized with standard ANSI colors, but the embeddings are trainable
    so the palette can drift toward whatever produces the best rendering.
    """

    def __init__(self, n_colors=N_COLORS):
        super().__init__()
        self.palette = nn.Embedding(n_colors, 3)
        with torch.no_grad():
            rgb = torch.tensor(ANSI_PALETTE, dtype=torch.float32) / 255.0
            self.palette.weight.copy_(rgb)

    def forward(self, codes):
        return self.palette(codes)


class NeuralCompositor(nn.Module):
    """Learned post-processing ConvNet for inter-cell refinement.

    Takes the cell-assembled frame and applies spatial transformations:
    anti-aliasing between adjacent cells, smoothing, subtle scan effects.
    Initialized near-identity (zero weights) for stable training start.

    Input/output: (B, 3, H, W) float tensors
    """

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 5, padding=2),   # 5×5 for inter-cell effects
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 1),               # 1×1 project back to RGB
        )
        # Initialize near-identity: output starts as input
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Conv2d):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, frame):
        return torch.clamp(frame + self.net(frame), 0.0, 1.0)


class NeuralTerminalRenderer(nn.Module):
    """Complete neural terminal rendering pipeline.

    Every pixel passes through neural computation:
      char → embedding → MLP → alpha mask
      color → embedding → RGB
      alpha * fg + (1-α) * bg → cell
      assembled grid → ConvNet → refined frame

    Input:  char_codes (B, R, C), fg_colors (B, R, C), bg_colors (B, R, C)
    Output: RGB frame (B, H, W, 3) float tensor in [0, 1]

    Also accepts unbatched (R, C) input and returns (H, W, 3).
    """

    def __init__(self, use_compositor=False):
        super().__init__()
        self.glyphs = NeuralGlyphGenerator()
        self.colors = NeuralColorPalette()
        self.compositor = NeuralCompositor()
        self.use_compositor = use_compositor

    def forward(self, char_codes, fg_colors, bg_colors, cursor_mask=None):
        unbatched = char_codes.dim() == 2
        if unbatched:
            char_codes = char_codes.unsqueeze(0)
            fg_colors = fg_colors.unsqueeze(0)
            bg_colors = bg_colors.unsqueeze(0)
            if cursor_mask is not None:
                cursor_mask = cursor_mask.unsqueeze(0)

        B, R, C = char_codes.shape

        # ── Neural glyph generation ────────────────────────────────────
        alpha = self.glyphs(char_codes)           # (B, R, C, ch, cw)

        # ── Neural color lookup ────────────────────────────────────────
        fg_rgb = self.colors(fg_colors)           # (B, R, C, 3)
        bg_rgb = self.colors(bg_colors)           # (B, R, C, 3)

        # ── Alpha blending (all neural values) ─────────────────────────
        a  = alpha.unsqueeze(-1)                  # (B, R, C, ch, cw, 1)
        fg = fg_rgb[:, :, :, None, None, :]       # (B, R, C, 1,  1,  3)
        bg = bg_rgb[:, :, :, None, None, :]
        cells = a * fg + (1.0 - a) * bg           # (B, R, C, ch, cw, 3)

        # ── Cursor: neural inversion at cursor position ────────────────
        if cursor_mask is not None:
            cm = cursor_mask[:, :, :, None, None, None].float()
            cells = cells * (1.0 - cm) + (1.0 - cells) * cm

        # ── Frame assembly ─────────────────────────────────────────────
        # (B, R, C, ch, cw, 3) → (B, R, ch, C, cw, 3) → (B, R*ch, C*cw, 3)
        cells = cells.permute(0, 1, 3, 2, 4, 5)
        frame = cells.contiguous().view(B, R * CELL_H, C * CELL_W, 3)

        # ── Neural compositor (ConvNet refinement) ─────────────────────
        if self.use_compositor:
            frame = frame.permute(0, 3, 1, 2)    # (B, 3, H, W)
            frame = self.compositor(frame)
            frame = frame.permute(0, 2, 3, 1)    # (B, H, W, 3)
        else:
            frame = torch.clamp(frame, 0.0, 1.0)

        if unbatched:
            frame = frame.squeeze(0)
        return frame

    def render_state(self, state: TerminalState, device: str = 'cpu') -> np.ndarray:
        """Convenience: render a TerminalState to numpy RGB (H, W, 3) uint8."""
        self.eval()
        with torch.no_grad():
            chars, fg, bg, cursor = state.to_tensors(device)
            frame = self(chars, fg, bg, cursor)
            return (frame.cpu().numpy() * 255).astype(np.uint8)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# Integration Class
# ═══════════════════════════════════════════════════════════════════════════

class NeuralDisplay:
    """Captures terminal byte stream and renders neurally.

    Wire into runner.py's make_syscall_handler via on_write callback:

        display = NeuralDisplay('models/display/terminal_renderer.pt')

        def on_write(fd, data):
            if fd in (1, 2):
                display.write(data)
            return False  # let default handler also print

        handler = make_syscall_handler(on_write=on_write)
        # ... run program ...
        frame = display.render()   # (384, 640, 3) uint8 numpy
    """

    MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'display' / 'terminal_renderer.pt'

    def __init__(self, model_path=None, device=None):
        if device is None:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = device
        self.terminal = TerminalState()
        self.renderer = NeuralTerminalRenderer().to(device)

        path = Path(model_path) if model_path else self.MODEL_PATH
        if path.exists():
            state = torch.load(str(path), map_location=device, weights_only=True)
            self.renderer.load_state_dict(state)
        self.renderer.eval()

        # Try to load Metal native display (bypass PyTorch entirely)
        self._metal_display = None
        try:
            from ncpu.neural.metal_neural_display import load_metal_neural_display
            self._metal_display = load_metal_neural_display(str(path))
            if self._metal_display is not None:
                self._use_metal = True
            else:
                self._use_metal = False
        except Exception:
            self._use_metal = False

    @property
    def metal_available(self) -> bool:
        """Whether native Metal rendering is active (no PyTorch)."""
        return self._use_metal

    def write(self, data: bytes):
        """Feed raw bytes from SYS_WRITE into the terminal state tracker."""
        self.terminal.write(data)

    @torch.no_grad()
    def render(self) -> np.ndarray:
        """Render current terminal state to RGB frame (H, W, 3) uint8."""
        if self._use_metal:
            return self._metal_display.render(
                self.terminal.chars, self.terminal.fg, self.terminal.bg,
                cursor_row=self.terminal.cr, cursor_col=self.terminal.cc,
            )
        return self.renderer.render_state(self.terminal, self.device)

    def render_rgba(self):
        """Render to RGBA bytes + dimensions (for on_framebuffer callback)."""
        rgb = self.render()
        h, w, _ = rgb.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255
        return rgba.tobytes(), w, h

    def render_text(self, text: str) -> np.ndarray:
        """Convenience: reset, write text, render."""
        self.reset()
        self.terminal.write_str(text)
        return self.render()

    def reset(self):
        self.terminal.reset()
