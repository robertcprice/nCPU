#!/usr/bin/env python3
"""nCPU Neural Display example for metal-mlp.

Demonstrates how metal-mlp is used in the nCPU project to deploy a trained
neural terminal renderer on Metal GPU. The renderer converts character codes
and color indices into RGB pixel frames entirely on the GPU, achieving 361
FPS without PyTorch at inference time.

Architecture:
  - Glyph MLP: char_code (uint8) -> 8x16 glyph bitmap (3 FC layers + GELU)
  - Color palette: 16-entry learned RGB lookup
  - Alpha blending: glyph bitmap * fg_color + (1 - glyph) * bg_color
  - Optional compositor ConvNet: 3-layer CNN for sub-pixel refinement

Weight layout (131,760 floats for base, 143,539 with compositor):
  [0       .. 16383 ]  glyphs.embed.weight   [256, 64]
  [16384   .. 32767 ]  glyphs.net.0.weight   [256, 64]   FC1
  [32768   .. 33023 ]  glyphs.net.0.bias     [256]
  [33024   .. 98559 ]  glyphs.net.2.weight   [256, 256]  FC2
  [98560   .. 98815 ]  glyphs.net.2.bias     [256]
  [98816   .. 131583]  glyphs.net.4.weight   [128, 256]  FC3
  [131584  .. 131711]  glyphs.net.4.bias     [128]
  [131712  .. 131759]  colors.palette.weight [16, 3]

This is a real-world example of the metal-mlp pattern in production.
"""

from pathlib import Path

import numpy as np

from metal_mlp import MetalMLPInference, MetalKernelLoader, benchmark_inference

# Paths relative to the nCPU project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"
RUST_METAL_DIR = PROJECT_ROOT / "kernels" / "rust_metal"

# Terminal geometry
TERM_ROWS, TERM_COLS = 24, 80
FRAME_H, FRAME_W = TERM_ROWS * 16, TERM_COLS * 8  # 384x640

# Weight counts
N_WEIGHT_FLOATS_BASE = 131_760  # embed + FC1-3 + palette
N_WEIGHT_FLOATS_FULL = 143_539  # + compositor ConvNet

# Weight keys in the order the Metal shader expects
WEIGHT_KEYS = [
    "glyphs.embed.weight",
    "glyphs.net.0.weight",
    "glyphs.net.0.bias",
    "glyphs.net.2.weight",
    "glyphs.net.2.bias",
    "glyphs.net.4.weight",
    "glyphs.net.4.bias",
    "colors.palette.weight",
]


def main() -> None:
    """Run the neural display example."""
    # Set up the kernel loader pointing at the nCPU Rust build directory
    loader = MetalKernelLoader(
        so_name="ncpu_metal.abi3.so",
        search_paths=[RUST_METAL_DIR],
    )

    # Create the high-level inference object
    display = MetalMLPInference(
        model_path=str(MODEL_PATH),
        kernel_class="NeuralDisplayKernel",
        weight_keys=WEIGHT_KEYS,
        expected_floats=N_WEIGHT_FLOATS_BASE,
        kernel_loader=loader,
        auto_init=True,
    )

    if not display.available:
        print(f"Metal neural display not available: {display.init_error}")
        print("\nThis example requires:")
        print(f"  1. Model checkpoint at: {MODEL_PATH}")
        print(f"  2. Compiled Rust/Metal kernel at: {RUST_METAL_DIR / 'ncpu_metal.abi3.so'}")
        print("  3. Apple Silicon Mac with Metal support")
        return

    print(f"Neural display ready: {display}")
    print(f"  Model: {display.info()['model_path']}")
    print(f"  Cached: {display.info()['is_cached']}")

    # Create a test frame: "Hello, Metal!" centered on the terminal
    chars = np.zeros((TERM_ROWS, TERM_COLS), dtype=np.uint8)
    fg = np.ones((TERM_ROWS, TERM_COLS), dtype=np.uint8) * 7   # white
    bg = np.zeros((TERM_ROWS, TERM_COLS), dtype=np.uint8)       # black

    message = "Hello, Metal!"
    row, col = TERM_ROWS // 2, (TERM_COLS - len(message)) // 2
    for i, ch in enumerate(message):
        chars[row, col + i] = ord(ch)

    # Render via Metal GPU
    kernel = display.kernel
    chars_flat = bytes(np.ascontiguousarray(chars.flatten(), dtype=np.uint8))
    fg_flat = bytes(np.ascontiguousarray(fg.flatten(), dtype=np.uint8))
    bg_flat = bytes(np.ascontiguousarray(bg.flatten(), dtype=np.uint8))

    rgb_bytes = kernel.render(chars_flat, fg_flat, bg_flat)
    frame = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 3)

    print(f"\n  Rendered frame: {frame.shape} {frame.dtype}")
    print(f"  Pixel range: [{frame.min()}, {frame.max()}]")
    print(f"  Non-zero pixels: {np.count_nonzero(frame):,}")

    # Benchmark
    print("\n  Benchmarking (1000 frames)...")
    results = benchmark_inference(
        metal_fn=lambda: kernel.render(chars_flat, fg_flat, bg_flat),
        n_iterations=1000,
        warmup=100,
    )
    print(f"  Metal FPS: {results['metal_fps']:.0f}")
    print(f"  Latency:   {results['metal_ms']:.2f} ms/frame")


if __name__ == "__main__":
    main()
