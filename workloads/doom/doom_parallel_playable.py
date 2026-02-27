#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PLAYABLE DOOM ON PARALLELMETALCPU - GPU ACCELERATED               ║
║                                                                              ║
║  Real ARM64 DOOM raycaster running on 128 parallel GPU lanes                 ║
║  611M+ IPS peak performance                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Controls:
    W - Move forward
    S - Move backward
    A - Turn left
    D - Turn right
    Q - Quit
"""

import sys
import importlib.util
import time
import termios
import tty
import select
import os
from pathlib import Path

# Load the shared library
spec = importlib.util.spec_from_file_location(
    'kvrm_metal',
    '/Users/bobbyprice/projects/.venv/lib/python3.13/site-packages/kvrm_metal/kvrm_metal.cpython-313-darwin.so'
)
kvrm_metal = importlib.util.module_from_spec(spec)
sys.modules['kvrm_metal'] = kvrm_metal
spec.loader.exec_module(kvrm_metal)

ParallelMetalCPU = kvrm_metal.ParallelMetalCPU


# DOOM Map (16x16) - 1 = wall, 0 = empty
DOOM_MAP = [
    "################",
    "#..............#",
    "#..####..####..#",
    "#..#........#..#",
    "#..#..####..#..#",
    "#.....#..#.....#",
    "#..#..#..#..#..#",
    "#..#........#..#",
    "#..####..####..#",
    "#..............#",
    "#..####......#.#",
    "#..#..........#.#",
    "#..#..######..#.#",
    "#.....#....#.....#",
    "#..............#",
    "################",
]


class NonBlockingInput:
    """Non-blocking keyboard input handler."""

    def __init__(self):
        self.old_settings = None

    def __enter__(self):
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except:
            pass
        return self

    def __exit__(self, type, value, traceback):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        """Get a key if one is pressed, None otherwise."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None


def setup_map(cpu):
    """Set up the DOOM map in memory."""
    map_data = bytearray()
    for row in DOOM_MAP:
        for cell in row:
            map_data.append(1 if cell == '#' else 0)

    # Write map to 0x30000
    cpu.write_memory(0x30000, list(map_data))


def render_framebuffer(cpu):
    """Render the framebuffer from memory."""
    # Read framebuffer from 0x40000 (80x25 = 2000 bytes)
    fb_data = cpu.read_memory(0x40000, 80 * 25)

    # Clear screen and move cursor to top-left
    print("\033[H\033[J", end="", flush=True)

    # Display framebuffer
    for y in range(25):
        row = "".join(chr(fb_data[y * 80 + x]) for x in range(80))
        print(row)

    # Display stats
    print("\033[30m" + "─" * 80 + "\033[0m")
    print("\033[36mDOOM on ParallelMetalCPU - 128 GPU lanes\033[0m | W/S: Move | A/D: Turn | Q: Quit")


def run_doom_playable(num_lanes=128):
    """Run playable DOOM on ParallelMetalCPU."""

    # Load raycast binary
    script_dir = Path(__file__).parent
    raycast_path = script_dir / "arm64_doom" / "raycast.bin"

    if not raycast_path.exists():
        print(f"ERROR: {raycast_path} not found!")
        print("Please compile the raycaster first:")
        print("  cd arm64_doom")
        print("  aarch64-elf-gcc -O2 -ffreestanding -nostdlib -nostartfiles -mcpu=cortex-a72 -c raycast.c -o raycast.o")
        print("  aarch64-elf-ld -T raycast.ld -nostdlib raycast.o -o raycast.elf")
        print("  aarch64-elf-objcopy -O binary raycast.elf raycast.bin")
        return

    with open(raycast_path, 'rb') as f:
        raycast_binary = f.read()

    print("=" * 80)
    print("  DOOM PARALLEL PLAYABLE - ParallelMetalCPU Edition")
    print("=" * 80)
    print(f"  Binary: {raycast_path}")
    print(f"  Size: {len(raycast_binary)} bytes")
    print(f"  Lanes: {num_lanes}")
    print()

    # Create ParallelMetalCPU with 1MB memory
    memory_size = 1024 * 1024

    print(f"  Creating {num_lanes} parallel ARM64 CPUs with {memory_size/1024}KB memory...")
    cpu = ParallelMetalCPU(num_lanes=num_lanes, memory_size=memory_size)

    # Load raycast binary at 0x10000 (entry point)
    cpu.load_program(list(raycast_binary), 0x10000)
    print(f"  Loaded raycast binary at 0x10000")

    # Set up DOOM map
    setup_map(cpu)
    print(f"  Set up DOOM map (16x16)")

    # Set PC to entry point
    cpu.set_pc_all(0x10000)

    # Clear keyboard input
    cpu.write_memory_u32(0x50000, 0)

    print()
    print("  Starting game...")
    print()

    # Initialize player position (5.5, 5.5 in fixed point)
    # Fixed point 16.16 format: 5.5 = (5 << 16) + (1 << 15) = 327680 + 16384 = 344064
    player_x = (5 << 16) + (1 << 15)  # 5.5 in 16.16 fixed point
    player_y = (5 << 16) + (1 << 15)
    player_angle = 0

    cpu.write_memory_u32(0x20000, player_x & 0xFFFFFFFF)
    cpu.write_memory_u32(0x20004, player_y & 0xFFFFFFFF)
    cpu.write_memory_u32(0x20008, player_angle)

    frame_count = 0
    start_time = time.time()
    last_key_time = time.time()

    with NonBlockingInput() as input_handler:
        while True:
            frame_count += 1

            # Check for keyboard input
            key = input_handler.get_key()
            current_time = time.time()

            # Only process input every 50ms (debounce)
            if key and (current_time - last_key_time > 0.05):
                last_key_time = current_time

                if key == 'q':
                    print("\033[H\033[J", end="", flush=True)
                    print("Quitting...")
                    break

                # Write key to memory (ASCII value)
                cpu.write_memory_u32(0x50000, ord(key))

            # Execute one frame (about 100K instructions per frame)
            result = cpu.execute(100000)

            # Read back player state (in case game modified it)
            # player_x = cpu.read_memory_u32(0x20000)
            # player_y = cpu.read_memory_u32(0x20004)
            # player_angle = cpu.read_memory_u32(0x20008)

            # Render framebuffer every 5 frames (to avoid flicker)
            if frame_count % 5 == 0:
                render_framebuffer(cpu)

                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                ips = result.avg_ips()

                # Display FPS in status line
                print(f"\033[26;1H\033[36mFPS: {fps:.1f} | IPS: {ips/1_000_000:.0f}M | Frame: {frame_count}\033[0m\033[K", end="", flush=True)

    # Final stats
    print()
    print("=" * 80)
    print("  GAME OVER")
    print("=" * 80)
    print(f"  Frames rendered: {frame_count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Average FPS: {fps:.1f}")
    print(f"  Average IPS: {ips:,.0f}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Playable DOOM on ParallelMetalCPU")
    parser.add_argument("--lanes", type=int, default=128, help="Number of parallel lanes")

    args = parser.parse_args()

    try:
        run_doom_playable(num_lanes=args.lanes)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
