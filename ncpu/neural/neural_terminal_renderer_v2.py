"""
Neural Terminal Renderer V2 — next-generation text-to-pixel display pipeline.

Improvements over V1 (neural_terminal_renderer.py):
  - Spatial positional encoding: sinusoidal (y, x) injected per-pixel into the
    glyph MLP, enabling position-aware feature learning and sharper edges.
  - Extended character set: 1024 characters (Latin-1, box-drawing, common Unicode
    symbols) with graceful fallback for out-of-range code points.
  - 256-color xterm palette: full xterm-256 support (16 ANSI + 216 color cube +
    24 grayscale), initialized from the standard specification.
  - Optional depthwise-separable compositor for lower parameter count.

Architecture:
  char_code ---> NeuralGlyphGeneratorV2 ---> 8x16 alpha mask (per-pixel positional encoding)
  color_code -> NeuralColorPaletteV2 ----> RGB (256-color xterm)
  alpha * fg + (1-a) * bg ------------> cell pixels
  assembled frame -> NeuralCompositorV2 -> refined frame

Every pixel in the output frame is produced by neural network forward passes.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

# Re-use terminal geometry and state tracker from V1
from ncpu.neural.neural_terminal_renderer import (
    TerminalState,
    TERM_ROWS, TERM_COLS,
    CELL_H, CELL_W,
    FRAME_H, FRAME_W,
)

__all__ = [
    "NeuralGlyphGeneratorV2",
    "NeuralColorPaletteV2",
    "NeuralCompositorV2",
    "NeuralTerminalRendererV2",
    "NeuralDisplayV2",
    "XTERM_256_PALETTE",
    "N_CHARS_V2",
    "N_COLORS_V2",
]

# ─── V2 constants ─────────────────────────────────────────────────────────
N_CHARS_V2 = 1024     # Extended character set (Latin-1, box-drawing, symbols)
N_COLORS_V2 = 256     # Full xterm-256 palette
POS_DIM = 32           # Total positional encoding dimension (8 sin/cos per axis x 2 axes)
EMBED_DIM = 64         # Character embedding dimension
HIDDEN_DIM = 512       # Glyph MLP hidden layer width (512 for sharper glyphs)
N_FREQS = 8            # Sinusoidal frequency count per spatial axis (8 freqs x sin/cos = 16 per axis)


# ═══════════════════════════════════════════════════════════════════════════
# Standard xterm-256 color palette
# ═══════════════════════════════════════════════════════════════════════════

def _build_xterm_256_palette() -> list[tuple[int, int, int]]:
    """Build the standard xterm-256 color palette as (R, G, B) tuples.

    Layout:
      0-7:     Standard ANSI colors
      8-15:    High-intensity (bright) ANSI colors
      16-231:  6x6x6 color cube (R*36 + G*6 + B + 16)
      232-255: Grayscale ramp (24 shades from #080808 to #eeeeee)
    """
    palette = []

    # 0-7: Standard ANSI colors
    ansi_base = [
        (0, 0, 0),       (128, 0, 0),     (0, 128, 0),     (128, 128, 0),
        (0, 0, 128),     (128, 0, 128),   (0, 128, 128),   (192, 192, 192),
    ]
    palette.extend(ansi_base)

    # 8-15: High-intensity (bright) ANSI colors
    ansi_bright = [
        (128, 128, 128), (255, 0, 0),     (0, 255, 0),     (255, 255, 0),
        (0, 0, 255),     (255, 0, 255),   (0, 255, 255),   (255, 255, 255),
    ]
    palette.extend(ansi_bright)

    # 16-231: 6x6x6 color cube
    cube_values = [0, 95, 135, 175, 215, 255]
    for r in range(6):
        for g in range(6):
            for b in range(6):
                palette.append((cube_values[r], cube_values[g], cube_values[b]))

    # 232-255: Grayscale ramp (24 entries)
    for i in range(24):
        v = 8 + i * 10  # 8, 18, 28, ... 238
        palette.append((v, v, v))

    assert len(palette) == 256
    return palette


XTERM_256_PALETTE = _build_xterm_256_palette()


# ═══════════════════════════════════════════════════════════════════════════
# Sinusoidal Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════

def _build_pixel_positions(cell_h: int, cell_w: int, n_freqs: int) -> torch.Tensor:
    """Pre-compute sinusoidal positional encodings for every pixel in a cell.

    For an 8x16 cell, each pixel at (row, col) gets a positional encoding
    vector of length 2 * 2 * n_freqs = 4 * n_freqs. We use n_freqs=4 per
    axis, yielding POS_DIM=16.

    Encoding scheme (transformer-style sinusoidal):
      For each axis (y normalized to [0,1], x normalized to [0,1]):
        sin(2^0 * pi * pos), cos(2^0 * pi * pos),
        sin(2^1 * pi * pos), cos(2^1 * pi * pos),
        ...
        sin(2^(n_freqs-1) * pi * pos), cos(2^(n_freqs-1) * pi * pos)

    Returns: (cell_h * cell_w, 2 * 2 * n_freqs) = (128, 16) tensor
    """
    # Normalized positions in [0, 1]
    y_pos = torch.linspace(0.0, 1.0, cell_h)   # (cell_h,)
    x_pos = torch.linspace(0.0, 1.0, cell_w)   # (cell_w,)

    # Frequency bands: 2^0, 2^1, ..., 2^(n_freqs-1)
    freqs = (2.0 ** torch.arange(n_freqs).float()) * math.pi  # (n_freqs,)

    # Y encoding: (cell_h, 2*n_freqs)
    y_scaled = y_pos.unsqueeze(1) * freqs.unsqueeze(0)  # (cell_h, n_freqs)
    y_enc = torch.cat([torch.sin(y_scaled), torch.cos(y_scaled)], dim=1)  # (cell_h, 2*n_freqs)

    # X encoding: (cell_w, 2*n_freqs)
    x_scaled = x_pos.unsqueeze(1) * freqs.unsqueeze(0)  # (cell_w, n_freqs)
    x_enc = torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=1)  # (cell_w, 2*n_freqs)

    # Outer product: every (row, col) pair gets [y_enc(row) | x_enc(col)]
    # Shape: (cell_h, cell_w, 4*n_freqs)
    y_exp = y_enc.unsqueeze(1).expand(cell_h, cell_w, -1)   # (cell_h, cell_w, 2*n_freqs)
    x_exp = x_enc.unsqueeze(0).expand(cell_h, cell_w, -1)   # (cell_h, cell_w, 2*n_freqs)
    pos_enc = torch.cat([y_exp, x_exp], dim=2)               # (cell_h, cell_w, 4*n_freqs)

    # Flatten spatial dims: (cell_h * cell_w, pos_dim)
    return pos_enc.reshape(cell_h * cell_w, 4 * n_freqs)


# ═══════════════════════════════════════════════════════════════════════════
# V2 Neural Rendering Models
# ═══════════════════════════════════════════════════════════════════════════

class NeuralGlyphGeneratorV2(nn.Module):
    """Position-aware character glyph generator with extended character set.

    Key improvement over V1: instead of an MLP that outputs a flat 128-vector
    which is reshaped into 8x16, each pixel receives its own (y, x) sinusoidal
    positional encoding concatenated with the character embedding. The network
    learns position-dependent features, producing sharper edges and better
    fine-grained structure.

    Architecture:
      Input per pixel: [char_embedding(64) | pos_encoding(32)] = 96-dim
      MLP: Linear(96, 512) + GELU -> Linear(512, 512) + GELU -> Linear(512, 1) + Sigmoid
      All 128 pixels processed in parallel via batched positional encodings.

    Input:  char codes (...,) int tensor, clamped to [0, n_chars)
    Output: alpha masks (..., cell_h, cell_w) float tensor in [0, 1]
    """

    def __init__(
        self,
        n_chars: int = N_CHARS_V2,
        cell_h: int = CELL_H,
        cell_w: int = CELL_W,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_freqs: int = N_FREQS,
    ):
        super().__init__()
        self.n_chars = n_chars
        self.cell_h = cell_h
        self.cell_w = cell_w
        self.n_pixels = cell_h * cell_w  # 128

        # Positional encoding dimension: 4 * n_freqs (sin+cos for both y and x)
        self.pos_dim = 4 * n_freqs
        mlp_input_dim = embed_dim + self.pos_dim  # 64 + 32 = 96

        # Character embedding: 1024 characters for extended Unicode support
        self.embed = nn.Embedding(n_chars, embed_dim)

        # Per-pixel MLP: takes [char_embed | pos_enc] and outputs scalar alpha
        # 512-wide hidden layers with residual skip for sharper glyphs
        self.net = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Pre-computed positional encodings: (n_pixels, pos_dim)
        pos_enc = _build_pixel_positions(cell_h, cell_w, n_freqs)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Generate alpha masks for given character codes.

        Args:
            codes: Integer tensor of shape (...,) with values in [0, n_chars).
                   Out-of-range values are clamped to '?' (code 63).

        Returns:
            Alpha masks of shape (..., cell_h, cell_w) in [0, 1].
        """
        original_shape = codes.shape

        # Clamp out-of-range characters to '?' (ASCII 63)
        safe_codes = codes.clone()
        out_of_range = (safe_codes < 0) | (safe_codes >= self.n_chars)
        safe_codes[out_of_range] = 63  # '?'

        # Flatten batch dimensions
        flat_codes = safe_codes.reshape(-1)  # (N,)
        N = flat_codes.shape[0]
        P = self.n_pixels  # 128

        # Character embeddings: (N, embed_dim) -> expand to (N, P, embed_dim)
        char_emb = self.embed(flat_codes)  # (N, embed_dim)
        char_emb_exp = char_emb.unsqueeze(1).expand(N, P, -1)  # (N, P, embed_dim)

        # Positional encodings: (P, pos_dim) -> expand to (N, P, pos_dim)
        pos_exp = self.pos_enc.unsqueeze(0).expand(N, P, -1)  # (N, P, pos_dim)

        # Concatenate: (N, P, embed_dim + pos_dim) = (N, P, 80)
        mlp_input = torch.cat([char_emb_exp, pos_exp], dim=2)

        # MLP forward: (N, P, 80) -> (N, P, 1) -> (N, P)
        alpha_flat = self.net(mlp_input).squeeze(-1)  # (N, P)

        # Reshape to spatial: (N, cell_h, cell_w) -> (..., cell_h, cell_w)
        alpha = alpha_flat.view(N, self.cell_h, self.cell_w)
        return alpha.view(*original_shape, self.cell_h, self.cell_w)


class NeuralColorPaletteV2(nn.Module):
    """Full xterm-256 color palette via learned embeddings.

    Initialized with standard xterm-256 colors (16 ANSI + 216 color cube +
    24 grayscale) so training starts from a correct baseline. Embeddings
    are trainable, allowing the palette to adapt for optimal rendering.

    Input:  color codes (...,) int tensor in [0, 256)
    Output: RGB values (..., 3) float tensor in [0, 1]
    """

    def __init__(self, n_colors: int = N_COLORS_V2):
        super().__init__()
        self.palette = nn.Embedding(n_colors, 3)

        # Initialize from standard xterm-256 palette
        with torch.no_grad():
            rgb = torch.tensor(XTERM_256_PALETTE[:n_colors], dtype=torch.float32) / 255.0
            if n_colors <= len(XTERM_256_PALETTE):
                self.palette.weight.copy_(rgb)
            else:
                # Pad extra entries with mid-gray
                self.palette.weight[:len(XTERM_256_PALETTE)] = rgb
                self.palette.weight[len(XTERM_256_PALETTE):] = 0.5

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Look up RGB colors for given color codes.

        Args:
            codes: Integer tensor of shape (...,) with values in [0, n_colors).

        Returns:
            RGB values of shape (..., 3) in [0, 1].
        """
        return self.palette(codes)


class NeuralCompositorV2(nn.Module):
    """Learned post-processing ConvNet for inter-cell refinement.

    Two modes:
      - standard (default): Same architecture as V1 for compatibility and
        quality. Conv2d(3, H, 5) + GELU + Conv2d(H, H, 3) + GELU + Conv2d(H, 3, 1).
      - efficient: Depthwise-separable convolutions for lower parameter count.
        Depthwise Conv2d(ch, ch, 5, groups=ch) + Pointwise Conv2d(ch, ch, 1) + GELU
        repeated, then 1x1 projection.

    Initialized near-identity (zero weights) so the output starts as the input,
    allowing stable training from the beginning.

    Input/output: (B, 3, H, W) float tensors
    """

    def __init__(self, hidden: int = 32, efficient: bool = False):
        super().__init__()
        self.efficient = efficient

        if efficient:
            # Depthwise-separable variant
            self.net = nn.Sequential(
                # Layer 1: expand 3 -> hidden via standard conv (can't depthwise-sep from 3)
                nn.Conv2d(3, hidden, 1),
                nn.GELU(),
                # Layer 2: depthwise 5x5 spatial filtering
                nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden),
                # Layer 2: pointwise mixing
                nn.Conv2d(hidden, hidden, 1),
                nn.GELU(),
                # Layer 3: depthwise 3x3 refinement
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                # Layer 3: pointwise mixing
                nn.Conv2d(hidden, hidden, 1),
                nn.GELU(),
                # Output projection
                nn.Conv2d(hidden, 3, 1),
            )
        else:
            # Standard variant (same as V1)
            self.net = nn.Sequential(
                nn.Conv2d(3, hidden, 5, padding=2),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 3, 1),
            )

        # Initialize near-identity: output starts as input (residual = 0)
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Conv2d):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """Apply learned refinement with residual connection.

        Args:
            frame: (B, 3, H, W) float tensor in [0, 1]

        Returns:
            Refined frame (B, 3, H, W) clamped to [0, 1]
        """
        return torch.clamp(frame + self.net(frame), 0.0, 1.0)


class NeuralTerminalRendererV2(nn.Module):
    """Complete V2 neural terminal rendering pipeline.

    Orchestrates the V2 components to render a full terminal frame:
      char -> NeuralGlyphGeneratorV2 (positional encoding) -> alpha mask
      color -> NeuralColorPaletteV2 (256 colors) -> RGB
      alpha * fg + (1-a) * bg -> cell
      assembled grid -> NeuralCompositorV2 -> refined frame

    Interface is identical to V1 for drop-in compatibility:
      forward(char_codes, fg_colors, bg_colors, cursor_mask=None)

    Input:  char_codes (B, R, C), fg_colors (B, R, C), bg_colors (B, R, C)
    Output: RGB frame (B, H, W, 3) float tensor in [0, 1]

    Also accepts unbatched (R, C) input and returns (H, W, 3).
    """

    def __init__(self, use_compositor: bool = False, efficient_compositor: bool = False):
        super().__init__()
        self.glyphs = NeuralGlyphGeneratorV2()
        self.colors = NeuralColorPaletteV2()
        self.compositor = NeuralCompositorV2(efficient=efficient_compositor)
        self.use_compositor = use_compositor

    def forward(
        self,
        char_codes: torch.Tensor,
        fg_colors: torch.Tensor,
        bg_colors: torch.Tensor,
        cursor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Render a complete terminal frame.

        Args:
            char_codes: (B, R, C) or (R, C) character codes
            fg_colors:  (B, R, C) or (R, C) foreground color indices (0-255)
            bg_colors:  (B, R, C) or (R, C) background color indices (0-255)
            cursor_mask: Optional (B, R, C) or (R, C) boolean mask for cursor

        Returns:
            RGB frame (B, H, W, 3) or (H, W, 3) in [0, 1]
        """
        unbatched = char_codes.dim() == 2
        if unbatched:
            char_codes = char_codes.unsqueeze(0)
            fg_colors = fg_colors.unsqueeze(0)
            bg_colors = bg_colors.unsqueeze(0)
            if cursor_mask is not None:
                cursor_mask = cursor_mask.unsqueeze(0)

        B, R, C = char_codes.shape

        # ── Neural glyph generation (position-aware) ─────────────────────
        alpha = self.glyphs(char_codes)               # (B, R, C, cell_h, cell_w)

        # ── Neural color lookup (256 colors) ─────────────────────────────
        fg_rgb = self.colors(fg_colors)               # (B, R, C, 3)
        bg_rgb = self.colors(bg_colors)               # (B, R, C, 3)

        # ── Alpha blending (all neural values) ───────────────────────────
        a  = alpha.unsqueeze(-1)                      # (B, R, C, cell_h, cell_w, 1)
        fg = fg_rgb[:, :, :, None, None, :]           # (B, R, C, 1, 1, 3)
        bg = bg_rgb[:, :, :, None, None, :]
        cells = a * fg + (1.0 - a) * bg               # (B, R, C, cell_h, cell_w, 3)

        # ── Cursor: neural inversion at cursor position ──────────────────
        if cursor_mask is not None:
            cm = cursor_mask[:, :, :, None, None, None].float()
            cells = cells * (1.0 - cm) + (1.0 - cells) * cm

        # ── Frame assembly ───────────────────────────────────────────────
        # (B, R, C, cell_h, cell_w, 3) -> (B, R, cell_h, C, cell_w, 3) -> (B, R*cell_h, C*cell_w, 3)
        cells = cells.permute(0, 1, 3, 2, 4, 5)
        frame = cells.contiguous().view(B, R * CELL_H, C * CELL_W, 3)

        # ── Neural compositor (ConvNet refinement) ───────────────────────
        if self.use_compositor:
            frame = frame.permute(0, 3, 1, 2)         # (B, 3, H, W)
            frame = self.compositor(frame)
            frame = frame.permute(0, 2, 3, 1)         # (B, H, W, 3)
        else:
            frame = torch.clamp(frame, 0.0, 1.0)

        if unbatched:
            frame = frame.squeeze(0)
        return frame

    def render_state(self, state: TerminalState, device: str = "cpu") -> np.ndarray:
        """Render a TerminalState to numpy RGB (H, W, 3) uint8.

        Color codes from the V1 TerminalState (0-15) are passed directly;
        the V2 palette's first 16 entries match standard ANSI colors.
        """
        self.eval()
        with torch.no_grad():
            chars, fg, bg, cursor = state.to_tensors(device)
            frame = self(chars, fg, bg, cursor)
            return (frame.cpu().numpy() * 255).astype(np.uint8)

    def count_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters())

    def count_params_by_component(self) -> dict[str, int]:
        """Parameter count broken down by component."""
        return {
            "glyphs": sum(p.numel() for p in self.glyphs.parameters()),
            "colors": sum(p.numel() for p in self.colors.parameters()),
            "compositor": sum(p.numel() for p in self.compositor.parameters()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Integration Class
# ═══════════════════════════════════════════════════════════════════════════

class NeuralDisplayV2:
    """V2 integration class: captures terminal byte stream and renders neurally.

    Drop-in replacement for NeuralDisplay with extended capabilities:
      - 1024-character support (basic Unicode, box-drawing, Latin-1)
      - 256-color xterm palette
      - Position-aware glyph generation

    Wire into runner.py the same way as V1:

        display = NeuralDisplayV2('models/display/terminal_renderer_v2.pt')

        def on_write(fd, data):
            if fd in (1, 2):
                display.write(data)
            return False

        handler = make_syscall_handler(on_write=on_write)
        frame = display.render()   # (384, 640, 3) uint8 numpy
    """

    MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "display" / "terminal_renderer_v2.pt"

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.terminal = TerminalState()
        self.renderer = NeuralTerminalRendererV2().to(device)

        path = Path(model_path) if model_path else self.MODEL_PATH
        if path.exists():
            state = torch.load(str(path), map_location=device, weights_only=True)
            try:
                self.renderer.load_state_dict(state)
            except RuntimeError:
                # Architecture mismatch (e.g., old narrow checkpoint with new wide model)
                # — try loading with strict=False to get what we can
                self.renderer.load_state_dict(state, strict=False)
        self.renderer.eval()

        # Try to load Metal V2 native display (bypass PyTorch entirely)
        self._metal_display = None
        try:
            from ncpu.neural.metal_neural_display import load_metal_neural_display_v2
            self._metal_display = load_metal_neural_display_v2(str(path))
            if self._metal_display is not None:
                self._use_metal = True
            else:
                self._use_metal = False
        except Exception:
            self._use_metal = False

    @property
    def metal_available(self) -> bool:
        """Whether native Metal V2 rendering is active (no PyTorch)."""
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

    def render_rgba(self) -> tuple[bytes, int, int]:
        """Render to RGBA bytes + dimensions (for on_framebuffer callback)."""
        rgb = self.render()
        h, w, _ = rgb.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255
        return rgba.tobytes(), w, h

    def render_text(self, text: str) -> np.ndarray:
        """Reset terminal, write text, and render."""
        self.reset()
        self.terminal.write_str(text)
        return self.render()

    def reset(self):
        """Reset terminal state."""
        self.terminal.reset()
