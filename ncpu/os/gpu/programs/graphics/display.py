#!/usr/bin/env python3
"""
GPU Framebuffer Display — Renders pixel output from ARM64 programs on Metal GPU.

Uses pygame to create a window that displays framebuffer data sent via
the SYS_FLUSH_FB syscall from programs running on the GPU compute shader.

Usage:
    python demos/graphics/display.py mandelbrot
    python demos/graphics/display.py <c_file>

The C program writes RGBA pixels to a static framebuffer buffer, then calls
sys_flush_fb(width, height, addr) to push the pixels to this display.
"""

import sys
import os
import tempfile
import time
from pathlib import Path

GRAPHICS_DIR = Path(__file__).parent
GPU_OS_DIR = GRAPHICS_DIR.parent.parent
PROJECT_ROOT = GPU_OS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU Framebuffer Display")
    parser.add_argument("program", help="C source file or built-in name (mandelbrot)")
    parser.add_argument("--save", help="Save framebuffer as PNG to this path")
    parser.add_argument("--no-window", action="store_true",
                        help="Don't open a window, just save the image")
    args = parser.parse_args()

    # Resolve program name
    program = args.program
    if program == "mandelbrot":
        c_file = GRAPHICS_DIR / "mandelbrot.c"
    elif os.path.exists(program):
        c_file = Path(program)
    else:
        c_file = GRAPHICS_DIR / program
        if not c_file.exists():
            c_file = GRAPHICS_DIR / (program + ".c")

    if not c_file.exists():
        print(f"Error: source file not found: {c_file}")
        sys.exit(1)

    banner = r"""
 ██████  ██████  ██    ██   ██████  ███████ ██   ██
██       ██   ██ ██    ██  ██       ██       ██ ██
██   ███ ██████  ██    ██  ██   ███ █████     ███
██    ██ ██      ██    ██  ██    ██ ██       ██ ██
 ██████  ██       ██████    ██████  ██      ██   ██

 GPU Framebuffer — ARM64 Pixel Rendering on Metal
 ─────────────────────────────────────────────────
"""
    print(banner)

    # 1. Compile the program
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    print(f"[boot] Compiling {c_file.name}...")
    if not compile_c(str(c_file), bin_path):
        print("[boot] FATAL: Compilation failed")
        sys.exit(1)

    binary = Path(bin_path).read_bytes()
    print(f"[boot] Binary: {len(binary):,} bytes")

    # 2. Load onto GPU
    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    # 3. Set up framebuffer callback
    framebuffer_data = {}

    def on_framebuffer(width, height, data):
        framebuffer_data["width"] = width
        framebuffer_data["height"] = height
        framebuffer_data["data"] = bytes(data)
        print(f"[fb] Received {width}x{height} framebuffer ({len(data):,} bytes)")

    handler = make_syscall_handler(on_framebuffer=on_framebuffer)

    # 4. Run the program
    print(f"[boot] Running on GPU...")
    print("=" * 60)

    start = time.perf_counter()
    results = run(cpu, handler, max_cycles=500_000_000, quiet=True)
    elapsed = time.perf_counter() - start

    print("=" * 60)
    print(f"Cycles: {results['total_cycles']:,}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"IPS: {results['ips']:,.0f}")

    # Clean up binary
    os.unlink(bin_path)

    # 5. Display the framebuffer
    if not framebuffer_data:
        print("\n[fb] No framebuffer data received (program didn't call sys_flush_fb)")
        sys.exit(0)

    width = framebuffer_data["width"]
    height = framebuffer_data["height"]
    data = framebuffer_data["data"]

    # Save as PNG if requested
    if args.save:
        _save_png(width, height, data, args.save)

    if args.no_window:
        print(f"\n[fb] Framebuffer: {width}x{height}")
        if not args.save:
            # Save to default location
            default_path = str(GRAPHICS_DIR / "output.png")
            _save_png(width, height, data, default_path)
        sys.exit(0)

    # Try pygame, fall back to saving PNG
    try:
        _display_pygame(width, height, data)
    except ImportError:
        print("\n[fb] pygame not installed — saving as PNG instead")
        default_path = str(GRAPHICS_DIR / "output.png")
        _save_png(width, height, data, default_path)
    except Exception as e:
        print(f"\n[fb] Display error: {e} — saving as PNG instead")
        default_path = str(GRAPHICS_DIR / "output.png")
        _save_png(width, height, data, default_path)


def _save_png(width, height, data, path):
    """Save framebuffer as PNG using minimal approach."""
    try:
        # Try PIL/Pillow first
        from PIL import Image
        import struct

        img = Image.frombytes("RGBA", (width, height), data)
        img.save(path)
        print(f"[fb] Saved: {path}")
        return
    except ImportError:
        pass

    try:
        # Fall back to pygame
        import pygame
        surf = pygame.Surface((width, height), pygame.SRCALPHA)
        # Unpack RGBA pixels
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 4
                r, g, b, a = data[idx], data[idx+1], data[idx+2], data[idx+3]
                surf.set_at((x, y), (r, g, b, a))
        pygame.image.save(surf, path)
        print(f"[fb] Saved: {path}")
        return
    except ImportError:
        pass

    # Last resort: save raw RGBA
    raw_path = path.replace(".png", ".rgba")
    with open(raw_path, "wb") as f:
        f.write(data)
    print(f"[fb] Saved raw RGBA: {raw_path} ({width}x{height})")


def _display_pygame(width, height, data):
    """Display framebuffer in a pygame window."""
    import pygame

    # Scale up for visibility
    scale = 1
    if width <= 400 and height <= 300:
        scale = 2

    pygame.init()
    screen = pygame.display.set_mode((width * scale, height * scale))
    pygame.display.set_caption(f"GPU Framebuffer — {width}x{height}")

    # Create surface from RGBA data
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    buf = surf.get_buffer()
    buf.write(data)

    if scale > 1:
        surf = pygame.transform.scale(surf, (width * scale, height * scale))

    screen.blit(surf, (0, 0))
    pygame.display.flip()

    print(f"\n[fb] Displaying {width}x{height} (scale {scale}x)")
    print("[fb] Close window or press ESC to exit")

    # Event loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    # Save screenshot
                    path = str(GRAPHICS_DIR / "screenshot.png")
                    pygame.image.save(screen, path)
                    print(f"[fb] Screenshot saved: {path}")

    pygame.quit()


if __name__ == "__main__":
    main()
