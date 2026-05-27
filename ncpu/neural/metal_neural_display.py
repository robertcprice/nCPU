"""Metal Neural Display — runs trained glyph MLP on Metal GPU via native Rust shader.

Exports weights from NeuralTerminalRenderer (.pt checkpoint) and dispatches the
full rendering pipeline (glyph MLP + color palette + alpha blending + optional
compositor ConvNet) entirely on the Metal GPU — no PyTorch inference at runtime.

This is the display analogue of MetalNeuralALU: trained neural network weights
running as native Metal compute shaders.

Supports PyTorch-free inference: weights are cached as a .npy file alongside the
.pt checkpoint. Once cached, the Metal display can render without torch installed.

Uses the generic ``metal_inference.py`` library (WeightCache, MetalKernelLoader)
for weight extraction/caching and Rust .so loading.

Usage:
    from ncpu.neural.metal_neural_display import MetalNeuralDisplay

    display = MetalNeuralDisplay('models/display/terminal_renderer.pt')
    if display.available:
        rgb = display.render(char_codes, fg_codes, bg_codes)  # numpy (384, 640, 3)

    # Batched rendering (amortizes Metal command buffer overhead):
    frames = display.render_batch(
        [chars1, chars2, chars3],
        [fg1, fg2, fg3],
        [bg1, bg2, bg3],
    )  # list of 3 numpy arrays, each (384, 640, 3)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ncpu.neural.metal_inference import WeightCache, MetalKernelLoader

# Terminal geometry (must match Rust constants)
TERM_ROWS, TERM_COLS = 24, 80
FRAME_H, FRAME_W = TERM_ROWS * 16, TERM_COLS * 8  # 384, 640

N_WEIGHT_FLOATS_BASE = 131_760  # embed + FC1-3 + palette
N_WEIGHT_FLOATS_FULL = 143_539  # + compositor ConvNet

# Weight keys in the order the Metal shader expects them in the flat buffer.
_BASE_WEIGHT_KEYS = [
    'glyphs.embed.weight',    # [256, 64]
    'glyphs.net.0.weight',    # [256, 64]   FC1
    'glyphs.net.0.bias',      # [256]
    'glyphs.net.2.weight',    # [256, 256]  FC2
    'glyphs.net.2.bias',      # [256]
    'glyphs.net.4.weight',    # [128, 256]  FC3
    'glyphs.net.4.bias',      # [128]
    'colors.palette.weight',  # [16, 3]
]

_COMPOSITOR_WEIGHT_KEYS = [
    'compositor.net.0.weight',  # [32, 3, 5, 5]  Conv1
    'compositor.net.0.bias',    # [32]
    'compositor.net.2.weight',  # [32, 32, 3, 3] Conv2
    'compositor.net.2.bias',    # [32]
    'compositor.net.4.weight',  # [3, 32, 1, 1]  Conv3
    'compositor.net.4.bias',    # [3]
]

# Shared kernel loader (singleton avoids repeated .so discovery)
_kernel_loader = MetalKernelLoader()


def _load_weights(model_path: str) -> Optional[list[float]]:
    """Load display weights, trying full (with compositor) then base cache.

    Resolution order:
        1. Full .npy cache  (143,539 floats)
        2. Base .npy cache  (131,760 floats) -- backward compatibility
        3. Torch extraction with full keys, falling back to base keys
    """
    # Try full cache first (with compositor)
    full_cache = WeightCache(
        model_path, N_WEIGHT_FLOATS_FULL,
        cache_suffix='.metal_weights_full.npy',
    )
    weights = full_cache.load()
    if weights is not None:
        return weights.tolist()

    # Try base cache (no compositor -- backward compat with older caches)
    base_cache = WeightCache(
        model_path, N_WEIGHT_FLOATS_BASE,
        cache_suffix='.metal_weights.npy',
    )
    weights = base_cache.load()
    if weights is not None:
        return weights.tolist()

    # No cache found. Try torch extraction: full keys first, then base.
    full_keys = _BASE_WEIGHT_KEYS + _COMPOSITOR_WEIGHT_KEYS
    weights = full_cache.extract_from_state_dict(full_keys)
    if weights is not None:
        return weights.tolist()

    weights = base_cache.extract_from_state_dict(_BASE_WEIGHT_KEYS)
    if weights is not None:
        return weights.tolist()

    return None


class MetalNeuralDisplay:
    """Wraps NeuralDisplayKernel (Rust/Metal) with automatic weight loading.

    Supports both base (glyph+palette) and full (+ compositor) weight sets.
    Falls back to None if Metal is unavailable or weights can't be extracted.
    """

    def __init__(self, model_path: str, use_compositor: bool = False):
        """Initialize Metal neural display.

        Args:
            model_path: path to .pt checkpoint
            use_compositor: if True, include compositor ConvNet (6 passes,
                ~16 FPS). Default False for fast 3-pass rendering (~280 FPS).
                Compositor adds <1 pixel difference.
        """
        self._kernel = None
        self._available = False
        self._has_compositor = False

        kernel_cls = _kernel_loader.get_class('NeuralDisplayKernel')
        if kernel_cls is None:
            return

        weights = _load_weights(model_path)
        if weights is None:
            return

        # If compositor not requested, truncate to base weights
        if not use_compositor and len(weights) == N_WEIGHT_FLOATS_FULL:
            weights = weights[:N_WEIGHT_FLOATS_BASE]

        try:
            kernel = kernel_cls()
            kernel.load_weights(weights)
            self._kernel = kernel
            self._available = kernel.is_ready()
            self._has_compositor = kernel.has_compositor()
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available

    @property
    def has_compositor(self) -> bool:
        """Whether compositor ConvNet is running on Metal."""
        return self._has_compositor

    def set_palette(self, colors: list[tuple[int, int, int]]) -> None:
        """Update the 16-color ANSI palette on the GPU for real-time theme switching.

        Writes directly to the palette region of the Metal weight buffer —
        no re-upload of full weights needed. Next render() uses the new colors.

        Args:
            colors: list of 16 (R, G, B) tuples, each 0-255.
        """
        if not self._available:
            raise RuntimeError("Metal neural display not available")
        if len(colors) != 16:
            raise ValueError(f"Need 16 colors, got {len(colors)}")
        flat = []
        for r, g, b in colors:
            flat.extend([r / 255.0, g / 255.0, b / 255.0])
        self._kernel.set_palette(flat)

    def get_palette(self) -> list[tuple[int, int, int]]:
        """Read the current 16-color palette from the GPU buffer.

        Returns:
            list of 16 (R, G, B) tuples, each 0-255.
        """
        if not self._available:
            raise RuntimeError("Metal neural display not available")
        flat = self._kernel.get_palette()
        colors = []
        for i in range(16):
            r = int(flat[i * 3 + 0] * 255 + 0.5)
            g = int(flat[i * 3 + 1] * 255 + 0.5)
            b = int(flat[i * 3 + 2] * 255 + 0.5)
            colors.append((r, g, b))
        return colors

    def render(self, char_codes: np.ndarray, fg_codes: np.ndarray,
               bg_codes: np.ndarray, cursor_row: int = -1,
               cursor_col: int = -1) -> np.ndarray:
        """Render terminal state to RGB frame (384, 640, 3) uint8.

        Args:
            char_codes: (24, 80) uint8 array of character codes
            fg_codes:   (24, 80) uint8 array of foreground color indices
            bg_codes:   (24, 80) uint8 array of background color indices
            cursor_row: cursor row (-1 = no cursor)
            cursor_col: cursor col (-1 = no cursor)

        Returns:
            (384, 640, 3) uint8 numpy array
        """
        if not self._available:
            raise RuntimeError("Metal neural display not available")

        # Flatten to contiguous uint8 byte arrays for the Rust kernel
        chars_flat = np.ascontiguousarray(char_codes.flatten(), dtype=np.uint8)
        fg_flat = np.ascontiguousarray(fg_codes.flatten(), dtype=np.uint8)
        bg_flat = np.ascontiguousarray(bg_codes.flatten(), dtype=np.uint8)

        # Dispatch Metal kernel (pass as bytes for zero-copy)
        rgb_bytes = self._kernel.render(
            bytes(chars_flat), bytes(fg_flat), bytes(bg_flat)
        )

        # Reshape to (H, W, 3)
        frame = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
            FRAME_H, FRAME_W, 3
        ).copy()

        # CPU-side cursor inversion (single cell, trivially fast)
        if 0 <= cursor_row < TERM_ROWS and 0 <= cursor_col < TERM_COLS:
            y0 = cursor_row * 16
            y1 = y0 + 16
            x0 = cursor_col * 8
            x1 = x0 + 8
            frame[y0:y1, x0:x1] = 255 - frame[y0:y1, x0:x1]

        return frame

    def render_batch(
        self,
        char_codes_list: list[np.ndarray],
        fg_codes_list: list[np.ndarray],
        bg_codes_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Render multiple frames in a single Metal command buffer.

        Amortizes GPU command buffer overhead for animation/streaming.

        Args:
            char_codes_list: list of N (24, 80) uint8 arrays
            fg_codes_list:   list of N (24, 80) uint8 arrays
            bg_codes_list:   list of N (24, 80) uint8 arrays

        Returns:
            list of N (384, 640, 3) uint8 numpy arrays
        """
        if not self._available:
            raise RuntimeError("Metal neural display not available")

        n = len(char_codes_list)
        if n != len(fg_codes_list) or n != len(bg_codes_list):
            raise ValueError("All input lists must have the same length")
        if n == 0:
            return []

        max_batch = self._kernel.max_batch_size()

        # Process in chunks of max_batch
        all_frames = []
        for start in range(0, n, max_batch):
            end = min(start + max_batch, n)
            batch_n = end - start

            # Concatenate into flat arrays
            all_chars = np.concatenate([
                np.ascontiguousarray(char_codes_list[i].flatten(), dtype=np.uint8)
                for i in range(start, end)
            ])
            all_fg = np.concatenate([
                np.ascontiguousarray(fg_codes_list[i].flatten(), dtype=np.uint8)
                for i in range(start, end)
            ])
            all_bg = np.concatenate([
                np.ascontiguousarray(bg_codes_list[i].flatten(), dtype=np.uint8)
                for i in range(start, end)
            ])

            # Dispatch batched Metal kernel
            rgb_list = self._kernel.render_batch(
                batch_n, bytes(all_chars), bytes(all_fg), bytes(all_bg)
            )

            for rgb_bytes in rgb_list:
                frame = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
                    FRAME_H, FRAME_W, 3
                ).copy()
                all_frames.append(frame)

        return all_frames


def load_metal_neural_display(model_path: str) -> Optional[MetalNeuralDisplay]:
    """Convenience: load Metal neural display, return None if unavailable."""
    disp = MetalNeuralDisplay(model_path)
    return disp if disp.available else None


# ═══════════════════════════════════════════════════════════════════════════
# V2 Neural Display — position-aware glyph MLP on Metal GPU
# ═══════════════════════════════════════════════════════════════════════════

N_WEIGHT_FLOATS_V2 = 383_233  # pos_enc + embed + FC1-3 + 256-color palette

# Weight keys for V2 in the order the Metal shader expects them in the flat buffer.
# The pos_enc buffer is first because it's a registered buffer in the model.
_V2_WEIGHT_KEYS = [
    'glyphs.pos_enc',            # [128, 32]   = 4096 floats (registered buffer)
    'glyphs.embed.weight',       # [1024, 64]  = 65536 floats
    'glyphs.net.0.weight',       # [512, 96]   = 49152 floats (FC1)
    'glyphs.net.0.bias',         # [512]       = 512 floats
    'glyphs.net.2.weight',       # [512, 512]  = 262144 floats (FC2)
    'glyphs.net.2.bias',         # [512]       = 512 floats
    'glyphs.net.4.weight',       # [1, 512]    = 512 floats (FC3)
    'glyphs.net.4.bias',         # [1]         = 1 float
    'colors.palette.weight',     # [256, 3]    = 768 floats
]


def _load_weights_v2(model_path: str) -> Optional[list[float]]:
    """Load V2 display weights from cache or torch extraction.

    Resolution order:
        1. .npy cache (383,233 floats)
        2. Torch extraction with explicit V2 key ordering
    """
    cache = WeightCache(
        model_path, N_WEIGHT_FLOATS_V2,
        cache_suffix='.metal_weights_v2.npy',
    )

    # Try cache first
    weights = cache.load()
    if weights is not None:
        return weights.tolist()

    # Try torch extraction with explicit keys
    weights = cache.extract_from_state_dict(_V2_WEIGHT_KEYS)
    if weights is not None:
        return weights.tolist()

    return None


class MetalNeuralDisplayV2:
    """Wraps NeuralDisplayKernelV2 (Rust/Metal) with automatic weight loading.

    V2 features over V1:
      - 1024 character embeddings (vs 256) for extended Unicode
      - 256-color xterm palette (vs 16-color ANSI)
      - Per-pixel positional encoding for sharper glyph rendering
      - Two-pass GPU architecture: per-cell partial FC1 + per-pixel completion

    Usage:
        from ncpu.neural.metal_neural_display import MetalNeuralDisplayV2

        display = MetalNeuralDisplayV2('models/display/terminal_renderer_v2.pt')
        if display.available:
            rgb = display.render(char_codes, fg_codes, bg_codes)  # (384, 640, 3) uint8
    """

    def __init__(self, model_path: str):
        """Initialize Metal V2 neural display.

        Args:
            model_path: path to V2 .pt checkpoint
        """
        self._kernel = None
        self._available = False

        kernel_cls = _kernel_loader.get_class('NeuralDisplayKernelV2')
        if kernel_cls is None:
            return

        weights = _load_weights_v2(model_path)
        if weights is None:
            return

        try:
            kernel = kernel_cls()
            kernel.load_weights(weights)
            self._kernel = kernel
            self._available = kernel.is_ready()
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def set_palette(self, colors: list[tuple[int, int, int]]) -> None:
        """Update the 256-color xterm palette on the GPU for real-time theme switching.

        Writes directly to the palette region of the Metal weight buffer.
        Next render() uses the new colors.

        Args:
            colors: list of 256 (R, G, B) tuples, each 0-255.
        """
        if not self._available:
            raise RuntimeError("Metal V2 neural display not available")
        if len(colors) != 256:
            raise ValueError(f"Need 256 colors, got {len(colors)}")
        flat = []
        for r, g, b in colors:
            flat.extend([r / 255.0, g / 255.0, b / 255.0])
        self._kernel.set_palette(flat)

    def get_palette(self) -> list[tuple[int, int, int]]:
        """Read the current 256-color palette from the GPU buffer.

        Returns:
            list of 256 (R, G, B) tuples, each 0-255.
        """
        if not self._available:
            raise RuntimeError("Metal V2 neural display not available")
        flat = self._kernel.get_palette()
        colors = []
        for i in range(256):
            r = int(flat[i * 3 + 0] * 255 + 0.5)
            g = int(flat[i * 3 + 1] * 255 + 0.5)
            b = int(flat[i * 3 + 2] * 255 + 0.5)
            colors.append((r, g, b))
        return colors

    def render(self, char_codes: np.ndarray, fg_codes: np.ndarray,
               bg_codes: np.ndarray, cursor_row: int = -1,
               cursor_col: int = -1) -> np.ndarray:
        """Render terminal state to RGB frame (384, 640, 3) uint8.

        Args:
            char_codes: (24, 80) uint8/uint16 array of character codes (0-1023)
            fg_codes:   (24, 80) uint8 array of foreground color indices (0-255)
            bg_codes:   (24, 80) uint8 array of background color indices (0-255)
            cursor_row: cursor row (-1 = no cursor)
            cursor_col: cursor col (-1 = no cursor)

        Returns:
            (384, 640, 3) uint8 numpy array
        """
        if not self._available:
            raise RuntimeError("Metal V2 neural display not available")

        # Flatten to contiguous uint8 byte arrays for the Rust kernel
        # V2 supports codes up to 1023 but passes as uint8 (Rust expands to uint32)
        chars_flat = np.ascontiguousarray(char_codes.flatten(), dtype=np.uint8)
        fg_flat = np.ascontiguousarray(fg_codes.flatten(), dtype=np.uint8)
        bg_flat = np.ascontiguousarray(bg_codes.flatten(), dtype=np.uint8)

        # Dispatch Metal kernel (pass as bytes for zero-copy)
        rgb_bytes = self._kernel.render(
            bytes(chars_flat), bytes(fg_flat), bytes(bg_flat)
        )

        # Reshape to (H, W, 3)
        frame = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
            FRAME_H, FRAME_W, 3
        ).copy()

        # CPU-side cursor inversion (single cell, trivially fast)
        if 0 <= cursor_row < TERM_ROWS and 0 <= cursor_col < TERM_COLS:
            y0 = cursor_row * 16
            y1 = y0 + 16
            x0 = cursor_col * 8
            x1 = x0 + 8
            frame[y0:y1, x0:x1] = 255 - frame[y0:y1, x0:x1]

        return frame

    def render_batch(
        self,
        char_codes_list: list[np.ndarray],
        fg_codes_list: list[np.ndarray],
        bg_codes_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Render multiple frames in a single Metal command buffer.

        Args:
            char_codes_list: list of N (24, 80) uint8 arrays
            fg_codes_list:   list of N (24, 80) uint8 arrays
            bg_codes_list:   list of N (24, 80) uint8 arrays

        Returns:
            list of N (384, 640, 3) uint8 numpy arrays
        """
        if not self._available:
            raise RuntimeError("Metal V2 neural display not available")

        n = len(char_codes_list)
        if n != len(fg_codes_list) or n != len(bg_codes_list):
            raise ValueError("All input lists must have the same length")
        if n == 0:
            return []

        max_batch = self._kernel.max_batch_size()

        # Process in chunks of max_batch
        all_frames = []
        for start in range(0, n, max_batch):
            end = min(start + max_batch, n)
            batch_n = end - start

            # Concatenate into flat arrays
            all_chars = np.concatenate([
                np.ascontiguousarray(char_codes_list[i].flatten(), dtype=np.uint8)
                for i in range(start, end)
            ])
            all_fg = np.concatenate([
                np.ascontiguousarray(fg_codes_list[i].flatten(), dtype=np.uint8)
                for i in range(start, end)
            ])
            all_bg = np.concatenate([
                np.ascontiguousarray(bg_codes_list[i].flatten(), dtype=np.uint8)
                for i in range(start, end)
            ])

            # Dispatch batched Metal kernel
            rgb_list = self._kernel.render_batch(
                batch_n, bytes(all_chars), bytes(all_fg), bytes(all_bg)
            )

            for rgb_bytes in rgb_list:
                frame = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
                    FRAME_H, FRAME_W, 3
                ).copy()
                all_frames.append(frame)

        return all_frames


def load_metal_neural_display_v2(model_path: str) -> Optional[MetalNeuralDisplayV2]:
    """Convenience: load Metal V2 neural display, return None if unavailable."""
    disp = MetalNeuralDisplayV2(model_path)
    return disp if disp.available else None
