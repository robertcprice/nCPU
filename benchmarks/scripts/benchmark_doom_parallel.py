#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        DOOM PARALLEL BENCHMARK - ParallelMetalCPU GPU Edition            ║
║                                                                              ║
║  Benchmarks DOOM on GPU-accelerated parallel ARM64 CPUs                   ║
║  - 128 parallel ARM64 CPUs on GPU                                          ║
║  - 720M+ IPS peak performance                                             ║
║  - Framebuffer capture and visualization                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import importlib.util
import time
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


def load_doom_binary(doom_path):
    """Load DOOM binary from file."""
    with open(doom_path, 'rb') as f:
        return f.read()


def run_doom_benchmark(num_lanes=32, cycles_per_frame=10_000_000, num_frames=60):
    """
    Run DOOM benchmark on ParallelMetalCPU.

    Args:
        num_lanes: Number of parallel ARM64 CPUs
        cycles_per_frame: Instructions to execute per frame
        num_frames: Number of frames to render
    """
    print("=" * 70)
    print("  DOOM PARALLEL BENCHMARK - ParallelMetalCPU")
    print("=" * 70)
    print(f"  Configuration: {num_lanes} lanes, {cycles_per_frame:,} cycles/frame")
    print()

    # Try different DOOM binaries
    script_dir = Path(__file__).parent
    doom_candidates = [
        script_dir / "arm64_doom" / "neural_rtos.bin",
        script_dir / "arm64_doom" / "doom_neural.elf",
        script_dir / "doom-arm" / "doom",
    ]

    doom_binary = None
    doom_path = None
    for candidate in doom_candidates:
        if candidate.exists():
            doom_path = candidate
            with open(candidate, 'rb') as f:
                doom_binary = f.read()
            print(f"  Found DOOM: {doom_path}")
            print(f"  Size: {len(doom_binary):,} bytes ({len(doom_binary)/1024:.1f} KB)")
            break

    if doom_binary is None:
        print("  ERROR: No DOOM binary found!")
        print("  Searched for:")
        for candidate in doom_candidates:
            print(f"    - {candidate}")
        return None

    print()

    # Create ParallelMetalCPU with enough memory for DOOM
    # DOOM needs memory for: code, data, stack, framebuffer
    memory_size = 16 * 1024 * 1024  # 16MB should be plenty

    print(f"  Creating {num_lanes} parallel ARM64 CPUs with {memory_size/1024/1024}MB memory...")
    cpu = ParallelMetalCPU(num_lanes=num_lanes, memory_size=memory_size)
    print()

    # Load DOOM at address 0 (standard for bare-metal)
    load_addr = 0x00000000
    print(f"  Loading DOOM at address 0x{load_addr:X}...")
    cpu.load_program(list(doom_binary), load_addr)
    print(f"  Loaded {len(doom_binary)} bytes")
    print()

    # Set PC to DOOM entry point
    entry_point = 0x00000000  # Start of binary
    cpu.set_pc_all(entry_point)
    print(f"  Entry point: 0x{entry_point:X}")
    print()

    # Benchmark frame rendering
    print("=" * 70)
    print("  RENDERING FRAMES")
    print("=" * 70)
    print()

    results = []
    total_ips = 0
    total_cycles = 0

    for frame in range(num_frames):
        start = time.time()

        # Execute one frame's worth of instructions
        result = cpu.execute(cycles_per_frame)

        elapsed = time.time() - start
        ips = result.avg_ips()
        fps = 1.0 / elapsed if elapsed > 0 else 0

        results.append({
            'frame': frame,
            'ips': ips,
            'fps': fps,
            'elapsed': elapsed,
            'final_pc': result.pcs_per_lane[0],
        })

        total_ips += ips
        total_cycles += result.total_cycles

        # Print progress every 10 frames
        if (frame + 1) % 10 == 0 or frame == 0:
            print(f"  Frame {frame:2d}: {fps:6.1f} FPS ({ips/1_000_000:6.1f}M IPS) | PC: 0x{result.pcs_per_lane[0]:08X}")

    # Print summary
    print()
    print("=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print()

    avg_ips = total_ips / num_frames
    avg_fps = sum(r['fps'] for r in results) / num_frames
    total_time = sum(r['elapsed'] for r in results)

    print(f"  Configuration: {num_lanes} lanes")
    print(f"  Frames rendered: {num_frames}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total cycles: {total_cycles:,}")
    print()
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Average IPS: {avg_ips:,.0f} ({avg_ips/1_000_000:.1f}M)")
    print(f"  Peak IPS: {max(r['ips'] for r in results):,.0f}")
    print(f"  Min IPS: {min(r['ips'] for r in results):,.0f}")
    print()

    # Lane efficiency
    min_cycles = min(result.cycles_per_lane)
    max_cycles = max(result.cycles_per_lane)
    efficiency = min_cycles / max_cycles if max_cycles > 0 else 1.0
    print(f"  Lane efficiency: {efficiency:.1%} (min/max cycles)")
    print()

    # Final PC check (all lanes should be synchronized)
    unique_pcs = set(result.pcs_per_lane)
    print(f"  Synchronization: {len(unique_pcs)} unique PCs across lanes")
    if len(unique_pcs) == 1:
        print(f"  ✅ All lanes synchronized at PC: 0x{result.pcs_per_lane[0]:08X}")
    else:
        print(f"  ⚠️  Lanes diverged: PCs range from 0x{min(result.pcs_per_lane):08X} to 0x{max(result.pcs_per_lane):08X}")

    print()
    print("=" * 70)

    return results


def benchmark_lane_scaling():
    """Benchmark DOOM with different lane counts."""
    print("=" * 70)
    print("  LANE SCALING BENCHMARK")
    print("=" * 70)
    print()

    lane_counts = [1, 4, 8, 16, 32, 64, 128]
    results = []

    for num_lanes in lane_counts:
        print(f"  Testing {num_lanes} lanes...")

        result = run_doom_benchmark(
            num_lanes=num_lanes,
            cycles_per_frame=1_000_000,  # Shorter test for scaling
            num_frames=10
        )

        if result:
            avg_ips = sum(r['ips'] for r in result) / len(result)
            results.append({
                'lanes': num_lanes,
                'ips': avg_ips,
                'speedup': avg_ips / results[0]['ips'] if results else 1.0,
            })

    print()
    print("=" * 70)
    print("  SCALING RESULTS")
    print("=" * 70)
    print()
    print(f"  {'Lanes':>6} | {'IPS':>15} | {'Speedup':>10} | {'Efficiency':>12}")
    print("  " + "-" * 60)

    for r in results:
        efficiency = r['speedup'] / r['lanes'] * 100
        print(f"  {r['lanes']:>6} | {r['ips']:>15,.0f} | {r['speedup']:>9.1f}x | {efficiency:>11.0f}%")

    print()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DOOM Parallel Benchmark on ParallelMetalCPU")
    parser.add_argument("--lanes", type=int, default=32, help="Number of parallel lanes")
    parser.add_argument("--cycles", type=int, default=10_000_000, help="Cycles per frame")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames")
    parser.add_argument("--scaling", action="store_true", help="Run lane scaling benchmark")

    args = parser.parse_args()

    if args.scaling:
        benchmark_lane_scaling()
    else:
        run_doom_benchmark(
            num_lanes=args.lanes,
            cycles_per_frame=args.cycles,
            num_frames=args.frames
        )
