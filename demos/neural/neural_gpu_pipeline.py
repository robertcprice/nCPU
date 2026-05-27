#!/usr/bin/env python3
"""End-to-end neural computer demo: C → GPU → Neural ALU → Neural Display.

Compiles a C program with aarch64-elf-gcc, executes it on the Metal GPU
kernel, and captures all output through the neural terminal renderer.
Every component in the pipeline is neural:
  - Instruction decode: neural LLM decoder
  - ALU operations: neural Kogge-Stone CLA, truth tables, MUL LUT
  - Memory addressing: neural pointer arithmetic (pointer.pt)
  - Display: neural glyph MLP + color palette + ConvNet compositor

Usage:
    python demos/neural/neural_gpu_pipeline.py
    python demos/neural/neural_gpu_pipeline.py --v2 --device mps
    python demos/neural/neural_gpu_pipeline.py --program ncpu/os/gpu/programs/graphics/mandelbrot.c
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.neural.neural_terminal_renderer import (
    NeuralDisplay,
    FRAME_H,
    FRAME_W,
)
from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
from ncpu.os.gpu.runner import compile_c, load_and_run, make_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


# ──────────────────────────────────────────────────────────────────────────
# Built-in demo programs
# ──────────────────────────────────────────────────────────────────────────

DEMO_HELLO = r'''
#include "arm64_libc.h"

int main(void) {
    printf("╔════════════════════════════════════════╗\n");
    printf("║   nCPU — Fully Neural Computer         ║\n");
    printf("╠════════════════════════════════════════╣\n");
    printf("║                                        ║\n");
    printf("║  Every component is a neural network:  ║\n");
    printf("║    • ALU:     Kogge-Stone CLA (8 pass) ║\n");
    printf("║    • Logic:   Neural truth tables       ║\n");
    printf("║    • Multiply: 256x256 LUT              ║\n");
    printf("║    • Decode:  LLM instruction decoder   ║\n");
    printf("║    • Memory:  Neural pointer arithmetic  ║\n");
    printf("║    • Display: Glyph MLP + Color embed   ║\n");
    printf("║                                        ║\n");
    printf("╚════════════════════════════════════════╝\n");
    printf("\n");

    // Fibonacci sequence — exercises ADD, CMP, branch
    printf("Fibonacci sequence (neural ADD + CMP):\n");
    int a = 0, b = 1;
    for (int i = 0; i < 15; i++) {
        printf("  F(%d) = %d\n", i, a);
        int t = a + b;
        a = b;
        b = t;
    }

    printf("\nAll output rendered by neural networks.\n");
    printf("Zero conventional computation anywhere.\n");
    return 0;
}
'''

DEMO_SIEVE = r'''
#include "arm64_libc.h"

int main(void) {
    printf("Sieve of Eratosthenes (neural ALU)\n");
    printf("==================================\n\n");

    int sieve[100];
    for (int i = 0; i < 100; i++) sieve[i] = 1;
    sieve[0] = sieve[1] = 0;

    for (int i = 2; i < 10; i++) {
        if (sieve[i]) {
            for (int j = i * i; j < 100; j += i) {
                sieve[j] = 0;
            }
        }
    }

    printf("Primes under 100:\n");
    int count = 0;
    for (int i = 2; i < 100; i++) {
        if (sieve[i]) {
            printf("%3d ", i);
            count++;
            if (count % 10 == 0) printf("\n");
        }
    }
    printf("\n\nTotal: %d primes\n", count);
    printf("\nEvery comparison, addition, and multiply\n");
    printf("computed by trained neural networks.\n");
    return 0;
}
'''


def save_png(array: np.ndarray, path: Path) -> None:
    """Save an RGB numpy array as PNG."""
    try:
        from PIL import Image
    except ImportError:
        print(f"  [WARNING] PIL not installed -- cannot save {path}")
        return
    img = Image.fromarray(array)
    img.save(str(path))
    print(f"  Saved: {path}")


def run_demo(
    display,
    program_source: str,
    program_name: str,
    output_path: Path,
    quiet: bool = False,
):
    """Compile, execute on GPU, and render output neurally."""
    import tempfile

    # Write source to temp file
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as f:
        f.write(program_source)
        src_path = f.name

    bin_path = src_path.replace(".c", ".bin")

    print(f"\n  Compiling {program_name}...")
    t0 = time.perf_counter()
    ok = compile_c(src_path, bin_path, quiet=True)
    if not ok:
        print(f"  ERROR: Compilation failed for {program_name}")
        return

    compile_time = time.perf_counter() - t0
    print(f"  Compiled in {compile_time * 1000:.0f} ms")

    # Create syscall handler and pass display to both handler and run()
    fs = GPUFilesystem()
    handler = make_syscall_handler(
        filesystem=fs,
        neural_display=display,
    )

    print(f"  Executing on Metal GPU with neural display...")
    t0 = time.perf_counter()
    result = load_and_run(bin_path, handler=handler, quiet=quiet, neural_display=display)
    exec_time = time.perf_counter() - t0

    cycles = result.get("total_cycles", 0)
    ips = result.get("ips", 0)
    print(f"  Executed: {cycles:,} cycles in {exec_time:.2f}s ({ips:,.0f} IPS)")

    # Render the final frame
    print(f"  Rendering through neural display...")
    t0 = time.perf_counter()
    frame = display.render()
    render_time = time.perf_counter() - t0
    print(f"  Rendered: {frame.shape} in {render_time * 1000:.1f} ms")

    save_png(frame, output_path)

    # Clean up temp files
    Path(src_path).unlink(missing_ok=True)
    Path(bin_path).unlink(missing_ok=True)

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end neural computer: C → GPU → Neural ALU → Neural Display"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device for neural display (default: auto-detect)",
    )
    parser.add_argument(
        "--v2", action="store_true",
        help="Use V2 renderer (512-wide MLP, 1024 chars, 256 colors)",
    )
    parser.add_argument(
        "--program", type=str, default=None,
        help="Path to a custom C source file (default: built-in demos)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for PNG frames",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress GPU execution output",
    )
    args = parser.parse_args()

    # Load neural display
    if args.v2:
        model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer_v2.pt"
        version = "V2 (390K params, 1024 chars, 256 colors)"
    else:
        model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"
        version = "V1 (143K params, 256 chars, 16 colors)"

    print()
    print("=" * 64)
    print("  nCPU End-to-End Neural Computer Pipeline")
    print("  C → GCC → ARM64 → Metal GPU → Neural ALU → Neural Display")
    print("=" * 64)
    print()
    print(f"  Display: {version}")
    print(f"  Model:   {model_path}")

    if args.v2:
        display = NeuralDisplayV2(str(model_path), device=args.device)
    else:
        display = NeuralDisplay(str(model_path), device=args.device)
    print(f"  Device:  {display.device}")
    print()

    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "models" / "display"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.program:
        # Custom program
        src = Path(args.program).read_text()
        name = Path(args.program).stem
        out = output_dir / f"pipeline_{name}.png"
        run_demo(display, src, name, out, quiet=args.quiet)
    else:
        # Built-in demos
        demos = [
            ("hello_neural", DEMO_HELLO, "Hello Neural Computer"),
            ("sieve_neural", DEMO_SIEVE, "Sieve of Eratosthenes"),
        ]
        frames = []
        for slug, source, title in demos:
            out = output_dir / f"pipeline_{slug}.png"
            # Reset display state between demos
            display.terminal.__init__()
            frame = run_demo(display, source, title, out, quiet=args.quiet)
            if frame is not None:
                frames.append((title, frame))

        # Composite
        if len(frames) >= 2:
            from demos.neural.neural_display_demo import compose_grid
            composite = compose_grid([f for _, f in frames], cols=2)
            comp_path = output_dir / "pipeline_composite.png"
            save_png(composite, comp_path)

    print()
    print("  Pipeline: C source → aarch64-elf-gcc → ARM64 binary")
    print("            → Metal GPU kernel → Neural ALU (Kogge-Stone CLA)")
    print("            → Neural display (glyph MLP + color embed)")
    print("  Every computation and every pixel: neural networks.")
    print()


if __name__ == "__main__":
    main()
