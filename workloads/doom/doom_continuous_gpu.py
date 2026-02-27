#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              INTERACTIVE NEURAL DOOM - CONTINUOUS GPU EXECUTION              ║
║                                                                              ║
║  - Uses MetalCPU for continuous GPU execution (no per-cycle sync!)           ║
║  - Real-time framebuffer display                                             ║
║  - Performance measurement                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from kvrm_metal import MetalCPU
import time
import sys

# DOOM display constants
FB_WIDTH = 320
FB_HEIGHT = 200
FB_ADDR = 0x20000
CODE_ADDR = 0x400000

# ASCII grayscale palette (dark to light)
PALETTE = " .:-=+*#%@"

def render_framebuffer(cpu):
    """Render framebuffer to ASCII art."""
    # Read RGBA framebuffer
    fb_data = cpu.read_memory(FB_ADDR, FB_WIDTH * FB_HEIGHT * 4)

    # Convert to ASCII (using red channel)
    print("\n" + "=" * 80)
    for y in range(0, FB_HEIGHT, 4):  # Skip rows for speed
        line = ""
        for x in range(0, FB_WIDTH, 4):  # Skip cols for speed
            offset = (y * FB_WIDTH + x) * 4
            r = fb_data[offset]
            # Map 0-255 to palette
            idx = (r // 32)  # 8 levels
            if idx >= len(PALETTE):
                idx = len(PALETTE) - 1
            line += PALETTE[idx] * 2  # Widen characters
        print(line)
    print("=" * 80)


def run_doom_continuous():
    """Run DOOM with continuous GPU execution for maximum performance."""

    # Create CPU with 8MB memory
    cpu = MetalCPU(memory_size=8*1024*1024)

    # Load DOOM code
    with open("doom_benchmark.elf", "rb") as f:
        f.seek(0x10000)
        code = f.read(0x3000)

    # Load program and set up initial state
    cpu.load_program(bytearray(code), CODE_ADDR)
    cpu.set_pc(CODE_ADDR)
    cpu.set_register(31, 0x700000)  # SP

    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                  NEURAL DOOM - CONTINUOUS GPU MODE                         ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"\n🎮 Running on MetalCPU (GPU: Apple M4 Pro)")
    print(f"📁 Code loaded at: 0x{CODE_ADDR:X}")
    print(f"🖥️  Framebuffer at: 0x{FB_ADDR:X} ({FB_WIDTH}x{FB_HEIGHT})")
    print(f"\n⚡  CONTINUOUS GPU EXECUTION - No per-cycle synchronization!")

    print("\n" + "=" * 80)
    print("DOOM is executing... (continuous GPU mode)")
    print("=" * 80)

    # Run in large batches on GPU (50000 cycles per batch)
    total_cycles = 0
    batch_size = 50000
    target_cycles = 1000000  # 1M cycles

    while total_cycles < target_cycles:
        cycles_this_batch = min(batch_size, target_cycles - total_cycles)

        start = time.time()
        result = cpu.execute(max_cycles=cycles_this_batch)
        elapsed = time.time() - start

        total_cycles += result.cycles
        ips = result.cycles / elapsed if elapsed > 0 else 0

        print(f"\rBatch: {total_cycles}/{target_cycles} cycles | IPS: {ips:8.0f} | PC: 0x{result.final_pc:X}   ",
              end="", flush=True)

        # Check progress
        if result.final_pc >= 0x400134:
            print(f"\n\n🎉 DOOM FRAME COMPLETE!")
            break
        elif result.final_pc >= 0x400120:
            print(f"\n\n✓ Reached z-buffer loop")
            break
        elif result.final_pc >= 0x400100:
            print(f"\n\n✓ Progressing through rendering...")

    print(f"\n\n📊 Final Results:")
    print(f"  Total cycles: {total_cycles}")
    print(f"  Final PC: 0x{result.final_pc:X}")
    print(f"  Instructions executed: {(result.final_pc - CODE_ADDR) // 4}")

    # Render framebuffer
    print("\n🖼️  Rendering framebuffer...")
    render_framebuffer(cpu)

    return cpu


if __name__ == "__main__":
    run_doom_continuous()
