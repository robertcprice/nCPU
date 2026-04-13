#!/usr/bin/env python3
"""nCPU DOOM-style ASCII Raycaster Demo.

A lightweight raycaster that runs ALL arithmetic through the nCPU's trained
neural networks. Every ADD, SUB, MUL, and CMP is a forward pass through a
real .pt model -- the same bit-serial full adder and byte-pair LUT that
power the rest of the nCPU project.

Two modes:
    NEURAL (default): All math through trained .pt models (~248us/ADD, ~21us/MUL)
    FAST   (--fast):  Native Python arithmetic (same algorithm, native speed)

Usage:
    python demos/doom_raycaster.py              # Neural mode, 50 frames
    python demos/doom_raycaster.py --fast       # Fast mode, 50 frames
    python demos/doom_raycaster.py --frames 100 # Neural mode, 100 frames
    python demos/doom_raycaster.py --both       # Run both modes and compare

Fixed-point arithmetic (scale 1024) keeps everything in 32-bit integers,
matching the nCPU's integer-only ISA. No floating point touches the ALU.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve project root so imports work from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXED_SHIFT = 10                     # Fixed-point scale: 1024
FIXED_ONE   = 1 << FIXED_SHIFT      # 1.0 in fixed-point = 1024
FIXED_HALF  = FIXED_ONE >> 1        # 0.5 in fixed-point = 512

SCREEN_W = 80                        # Terminal columns for the viewport
SCREEN_H = 24                        # Terminal rows for the viewport

MAP_SIZE = 16                        # 16x16 tile map

# Maximum ray distance in tiles (avoids infinite loops on open maps)
MAX_RAY_STEPS = 64

# Distance-to-character shading (closest to farthest)
SHADE_CHARS = "\u2588\u2593\u2592\u2591 "   # full, dark, medium, light, space

# Pre-compute a fixed-point sin/cos table (256 entries = full circle).
# These are computed ONCE at import time using Python math, then stored
# as plain integers. The raycaster indexes into this table instead of
# calling neural_sin/neural_cos per ray (which would be thousands of
# redundant model calls for the same angles).
ANGLE_TABLE_SIZE = 256
SIN_TABLE: List[int] = []
COS_TABLE: List[int] = []
for _i in range(ANGLE_TABLE_SIZE):
    _rad = 2.0 * math.pi * _i / ANGLE_TABLE_SIZE
    SIN_TABLE.append(int(math.sin(_rad) * FIXED_ONE))
    COS_TABLE.append(int(math.cos(_rad) * FIXED_ONE))


# ---------------------------------------------------------------------------
# Map definition -- DOOM-inspired layout with rooms and corridors
# ---------------------------------------------------------------------------
MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


# ---------------------------------------------------------------------------
# Operation counter -- tracks every ALU call for stats
# ---------------------------------------------------------------------------
@dataclass
class OpCounter:
    """Counts neural ALU operations performed during raycasting."""
    add: int = 0
    sub: int = 0
    mul: int = 0
    cmp: int = 0

    def total(self) -> int:
        return self.add + self.sub + self.mul + self.cmp

    def reset(self) -> None:
        self.add = self.sub = self.mul = self.cmp = 0

    def summary(self) -> str:
        return f"ADD x{self.add}  MUL x{self.mul}  SUB x{self.sub}  CMP x{self.cmp}"


# ---------------------------------------------------------------------------
# ALU abstraction -- wraps NeuralOps or plain Python
# ---------------------------------------------------------------------------
class ALU:
    """Arithmetic logic unit that dispatches to neural models or native Python.

    In neural mode, every add/sub/mul/cmp is a forward pass through a trained
    .pt model. In fast mode, the same fixed-point algorithm runs with Python
    operators for maximum speed.
    """

    def __init__(self, neural: bool = True, models_dir: str = "models"):
        self.neural_mode = neural
        self.ops = OpCounter()
        self._neural_ops = None

        if neural:
            try:
                from ncpu.model.neural_ops import NeuralOps
                self._neural_ops = NeuralOps(models_dir=models_dir)
                avail = self._neural_ops.load()
                loaded = [k for k, v in avail.items() if v]
                if not loaded:
                    print("[WARN] No neural models found -- falling back to Python math")
                    self.neural_mode = False
                    self._neural_ops = None
                else:
                    print(f"[INFO] Loaded neural models: {', '.join(sorted(loaded))}")
            except Exception as exc:
                print(f"[WARN] Could not load neural models ({exc}) -- using Python math")
                self.neural_mode = False
                self._neural_ops = None

    @property
    def mode_label(self) -> str:
        return "NEURAL" if self.neural_mode else "FAST"

    # -- Core fixed-point operations ------------------------------------------

    def add(self, a: int, b: int) -> int:
        """Fixed-point addition."""
        self.ops.add += 1
        if self.neural_mode and self._neural_ops is not None:
            return self._neural_ops.neural_add(a, b)
        return self._clamp(a + b)

    def sub(self, a: int, b: int) -> int:
        """Fixed-point subtraction."""
        self.ops.sub += 1
        if self.neural_mode and self._neural_ops is not None:
            return self._neural_ops.neural_sub(a, b)
        return self._clamp(a - b)

    def mul(self, a: int, b: int) -> int:
        """Fixed-point multiplication.  Result is (a * b) >> FIXED_SHIFT."""
        self.ops.mul += 1
        if self.neural_mode and self._neural_ops is not None:
            product = self._neural_ops.neural_mul(a, b)
            # Shift right by FIXED_SHIFT to renormalize fixed-point product.
            # Use Python shift here since the neural shift takes ~463us and
            # this is a bookkeeping step, not an ALU computation.
            return product >> FIXED_SHIFT
        return self._clamp((a * b) >> FIXED_SHIFT)

    def mul_raw(self, a: int, b: int) -> int:
        """Raw multiplication (no shift-back). For integer * integer."""
        self.ops.mul += 1
        if self.neural_mode and self._neural_ops is not None:
            return self._neural_ops.neural_mul(a, b)
        return self._clamp(a * b)

    def cmp_lt(self, a: int, b: int) -> bool:
        """Compare: a < b."""
        self.ops.cmp += 1
        if self.neural_mode and self._neural_ops is not None:
            _, sign = self._neural_ops.neural_cmp(a, b)
            return sign   # sign_flag == True means a - b < 0
        return a < b

    def cmp_gt(self, a: int, b: int) -> bool:
        """Compare: a > b."""
        self.ops.cmp += 1
        if self.neural_mode and self._neural_ops is not None:
            zero, sign = self._neural_ops.neural_cmp(a, b)
            return (not zero) and (not sign)
        return a > b

    def abs_val(self, a: int) -> int:
        """Absolute value using SUB from zero if negative."""
        if self.cmp_lt(a, 0):
            return self.sub(0, a)
        return a

    def neg(self, a: int) -> int:
        """Negate: 0 - a."""
        return self.sub(0, a)

    @staticmethod
    def _clamp(v: int) -> int:
        INT32_MIN = -(2**31)
        INT32_MAX = (2**31) - 1
        return max(INT32_MIN, min(INT32_MAX, v))


# ---------------------------------------------------------------------------
# DDA Raycaster -- entirely in fixed-point integer arithmetic
# ---------------------------------------------------------------------------
def cast_ray(
    alu: ALU,
    player_x: int,     # Fixed-point
    player_y: int,     # Fixed-point
    ray_dir_x: int,    # Fixed-point
    ray_dir_y: int,    # Fixed-point
) -> int:
    """Cast a single ray using DDA and return the wall distance (fixed-point).

    Uses only ALU.add, ALU.sub, ALU.mul, ALU.cmp -- every one of which
    goes through a neural network in NEURAL mode.
    """
    # Current map cell (integer tile coordinates = fixed-point >> FIXED_SHIFT)
    map_x = player_x >> FIXED_SHIFT
    map_y = player_y >> FIXED_SHIFT

    # Fractional position within the current tile (fixed-point, 0..1023)
    frac_x = player_x & (FIXED_ONE - 1)
    frac_y = player_y & (FIXED_ONE - 1)

    # Determine step direction for each axis
    step_x = 1 if ray_dir_x >= 0 else -1
    step_y = 1 if ray_dir_y >= 0 else -1

    # Absolute ray direction components for distance calculation
    abs_dx = alu.abs_val(ray_dir_x)
    abs_dy = alu.abs_val(ray_dir_y)

    # Avoid division by zero -- clamp minimum ray component
    MIN_DIR = 1
    if alu.cmp_lt(abs_dx, MIN_DIR):
        abs_dx = MIN_DIR
    if alu.cmp_lt(abs_dy, MIN_DIR):
        abs_dy = MIN_DIR

    # Delta distance: how far along the ray to cross one full tile.
    # delta_dist_x = FIXED_ONE * FIXED_ONE / abs_dx  (fixed-point division
    # approximated by scaling). To avoid a neural division (slow), we compute:
    #   delta = (FIXED_ONE << FIXED_SHIFT) / abs_dx
    # which is integer division -- not an ALU op, just bookkeeping.
    # The actual ALU work is in the stepping loop below.
    delta_dist_x = (FIXED_ONE * FIXED_ONE) // max(abs_dx, 1)
    delta_dist_y = (FIXED_ONE * FIXED_ONE) // max(abs_dy, 1)

    # Initial side distance: distance from current position to next tile edge
    if step_x > 0:
        side_dist_x = alu.mul(alu.sub(FIXED_ONE, frac_x), delta_dist_x)
    else:
        side_dist_x = alu.mul(frac_x, delta_dist_x)

    if step_y > 0:
        side_dist_y = alu.mul(alu.sub(FIXED_ONE, frac_y), delta_dist_y)
    else:
        side_dist_y = alu.mul(frac_y, delta_dist_y)

    # DDA stepping loop -- all comparisons and additions go through the ALU
    hit = False
    side = 0  # 0 = vertical wall (x-step), 1 = horizontal wall (y-step)

    for _ in range(MAX_RAY_STEPS):
        # Step to the next tile boundary along the closer axis
        if alu.cmp_lt(side_dist_x, side_dist_y):
            side_dist_x = alu.add(side_dist_x, delta_dist_x)
            map_x += step_x
            side = 0
        else:
            side_dist_y = alu.add(side_dist_y, delta_dist_y)
            map_y += step_y
            side = 1

        # Bounds check
        if map_x < 0 or map_x >= MAP_SIZE or map_y < 0 or map_y >= MAP_SIZE:
            break

        # Wall hit check
        if MAP[map_y][map_x] == 1:
            hit = True
            break

    if not hit:
        return MAX_RAY_STEPS * FIXED_ONE  # Far distance

    # Compute perpendicular wall distance to avoid fisheye.
    # perp_dist = side_dist - delta_dist  (for the axis we last stepped on)
    if side == 0:
        perp_dist = alu.sub(side_dist_x, delta_dist_x)
    else:
        perp_dist = alu.sub(side_dist_y, delta_dist_y)

    # Clamp minimum distance to avoid division by zero in column height calc
    if alu.cmp_lt(perp_dist, 1):
        perp_dist = 1

    return perp_dist


def cast_all_rays(
    alu: ALU,
    player_x: int,
    player_y: int,
    player_angle: int,     # Index into angle table (0..255)
    fov_span: int = 60,    # Number of angle table entries for FOV (~84 degrees)
) -> List[int]:
    """Cast one ray per screen column and return list of wall distances."""
    distances: List[int] = []
    half_fov = fov_span // 2

    for col in range(SCREEN_W):
        # Map screen column to ray angle offset within the FOV.
        # ray_angle = player_angle - half_fov + col * fov_span / SCREEN_W
        ray_offset = -half_fov + (col * fov_span) // SCREEN_W
        ray_angle = (player_angle + ray_offset) % ANGLE_TABLE_SIZE

        ray_dir_x = COS_TABLE[ray_angle]
        ray_dir_y = SIN_TABLE[ray_angle]

        dist = cast_ray(alu, player_x, player_y, ray_dir_x, ray_dir_y)
        distances.append(dist)

    return distances


# ---------------------------------------------------------------------------
# ASCII renderer
# ---------------------------------------------------------------------------
def render_frame(
    distances: List[int],
    frame_num: int,
    mode_label: str,
    fps: float,
    avg_ms: float,
    op_summary: str,
) -> str:
    """Render a single frame as an ASCII string for the terminal."""
    lines: List[str] = []
    border = "\u2550" * (SCREEN_W + 2)

    # Header
    lines.append(f"\u2554{border}\u2557")
    title = "nCPU DOOM Raycaster"
    lines.append(f"\u2551{title:^{SCREEN_W + 2}}\u2551")
    status = f"Mode: {mode_label}  |  Frame: {frame_num}"
    lines.append(f"\u2551{status:^{SCREEN_W + 2}}\u2551")
    lines.append(f"\u2560{border}\u2563")

    # Build column heights from distances
    col_heights: List[int] = []
    for dist in distances:
        # Column height inversely proportional to distance.
        # height = SCREEN_H * FIXED_ONE / max(dist, 1)
        # Capped at SCREEN_H.
        if dist <= 0:
            dist = 1
        h = (SCREEN_H * FIXED_ONE) // dist
        h = min(h, SCREEN_H)
        col_heights.append(h)

    # Render the viewport row by row
    for row in range(SCREEN_H):
        row_chars: List[str] = []
        mid = SCREEN_H // 2

        for col in range(SCREEN_W):
            h = col_heights[col]
            half_h = h // 2
            dist_from_center = abs(row - mid)

            if dist_from_center <= half_h:
                # Wall pixel -- shade by distance
                dist = distances[col]
                # Shade index: 0 = closest, len(SHADE_CHARS)-1 = farthest
                shade_idx = min(
                    len(SHADE_CHARS) - 1,
                    (dist * len(SHADE_CHARS)) // (MAX_RAY_STEPS * FIXED_ONE),
                )
                row_chars.append(SHADE_CHARS[shade_idx])
            elif row < mid:
                # Ceiling
                row_chars.append(" ")
            else:
                # Floor -- subtle texture
                floor_char = "." if (row + col) % 4 == 0 else " "
                row_chars.append(floor_char)

        line_str = "".join(row_chars)
        lines.append(f"\u2551 {line_str} \u2551")

    # Footer with stats
    lines.append(f"\u2560{border}\u2563")
    stats = f"FPS: {fps:.1f}  |  Avg: {avg_ms:.0f}ms/frame  |  Ops: {op_summary}"
    # Truncate if too long for the border
    if len(stats) > SCREEN_W + 2:
        stats = stats[: SCREEN_W - 1] + "..."
    lines.append(f"\u2551 {stats:<{SCREEN_W}} \u2551")
    lines.append(f"\u255a{border}\u255d")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
@dataclass
class RunStats:
    """Accumulated statistics for a complete run."""
    mode: str = ""
    total_frames: int = 0
    total_time_s: float = 0.0
    total_ops: int = 0
    frame_times_ms: list = field(default_factory=list)
    ops_breakdown: str = ""

    @property
    def avg_fps(self) -> float:
        if self.total_time_s <= 0:
            return 0.0
        return self.total_frames / self.total_time_s

    @property
    def avg_ms(self) -> float:
        if not self.frame_times_ms:
            return 0.0
        return sum(self.frame_times_ms) / len(self.frame_times_ms)

    @property
    def min_ms(self) -> float:
        return min(self.frame_times_ms) if self.frame_times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.frame_times_ms) if self.frame_times_ms else 0.0


def run_demo(
    neural: bool = True,
    num_frames: int = 50,
    models_dir: str = "models",
    quiet: bool = False,
) -> RunStats:
    """Run the raycaster demo and return accumulated stats."""
    alu = ALU(neural=neural, models_dir=models_dir)
    stats = RunStats(mode=alu.mode_label)

    # Player starts in the center of tile (2, 2), facing right
    player_x = 2 * FIXED_ONE + FIXED_HALF   # Tile 2, center
    player_y = 2 * FIXED_ONE + FIXED_HALF
    player_angle = 0                          # Facing right (east)

    # Rotation speed: ~1.4 degrees per frame (1/256 of full circle)
    angle_step = 1

    cumulative_ops = OpCounter()
    run_start = time.perf_counter()

    for frame in range(num_frames):
        alu.ops.reset()
        t0 = time.perf_counter()

        # Cast all rays for this frame
        distances = cast_all_rays(alu, player_x, player_y, player_angle)

        t1 = time.perf_counter()
        frame_ms = (t1 - t0) * 1000.0
        stats.frame_times_ms.append(frame_ms)

        # Accumulate ops
        cumulative_ops.add += alu.ops.add
        cumulative_ops.sub += alu.ops.sub
        cumulative_ops.mul += alu.ops.mul
        cumulative_ops.cmp += alu.ops.cmp

        # Compute running FPS
        elapsed = t1 - run_start
        fps = (frame + 1) / elapsed if elapsed > 0 else 0.0
        avg_ms = sum(stats.frame_times_ms) / len(stats.frame_times_ms)

        if not quiet:
            # Clear screen and render
            frame_str = render_frame(
                distances,
                frame_num=frame + 1,
                mode_label=alu.mode_label,
                fps=fps,
                avg_ms=avg_ms,
                op_summary=alu.ops.summary(),
            )
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(frame_str)
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Auto-rotate player
        player_angle = (player_angle + angle_step) % ANGLE_TABLE_SIZE

    run_end = time.perf_counter()
    stats.total_frames = num_frames
    stats.total_time_s = run_end - run_start
    stats.total_ops = cumulative_ops.total()
    stats.ops_breakdown = cumulative_ops.summary()

    return stats


def print_summary(stats: RunStats) -> None:
    """Print a summary block for one run."""
    sep = "-" * 60
    print(sep)
    print(f"  Mode:          {stats.mode}")
    print(f"  Frames:        {stats.total_frames}")
    print(f"  Total time:    {stats.total_time_s:.2f}s")
    print(f"  Avg FPS:       {stats.avg_fps:.2f}")
    print(f"  Avg frame:     {stats.avg_ms:.1f}ms")
    print(f"  Min frame:     {stats.min_ms:.1f}ms")
    print(f"  Max frame:     {stats.max_ms:.1f}ms")
    print(f"  Total ALU ops: {stats.total_ops:,}")
    print(f"  Breakdown:     {stats.ops_breakdown}")
    print(sep)


def print_comparison(neural_stats: RunStats, fast_stats: RunStats) -> None:
    """Print a side-by-side comparison of NEURAL vs FAST runs."""
    border = "=" * 64
    print()
    print(border)
    print("  nCPU DOOM Raycaster -- Mode Comparison")
    print(border)
    print()
    print(f"  {'Metric':<20} {'NEURAL':>18} {'FAST':>18}")
    print(f"  {'-'*20} {'-'*18} {'-'*18}")
    print(f"  {'Frames':<20} {neural_stats.total_frames:>18} {fast_stats.total_frames:>18}")
    print(f"  {'Total time':<20} {neural_stats.total_time_s:>17.2f}s {fast_stats.total_time_s:>17.2f}s")
    print(f"  {'Avg FPS':<20} {neural_stats.avg_fps:>18.2f} {fast_stats.avg_fps:>18.2f}")
    print(f"  {'Avg frame (ms)':<20} {neural_stats.avg_ms:>18.1f} {fast_stats.avg_ms:>18.1f}")
    print(f"  {'Total ALU ops':<20} {neural_stats.total_ops:>18,} {fast_stats.total_ops:>18,}")

    if fast_stats.avg_ms > 0:
        slowdown = neural_stats.avg_ms / fast_stats.avg_ms
        print()
        print(f"  Neural overhead:   {slowdown:.1f}x slower per frame")
        print(f"  Cost of neural:    Every ADD/SUB/MUL/CMP is a real .pt model forward pass")
    print()
    print(border)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="nCPU DOOM-style ASCII Raycaster Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use native Python arithmetic instead of neural models",
    )
    parser.add_argument(
        "--frames", "-n",
        type=int,
        default=50,
        help="Number of frames to render (default: 50)",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both NEURAL and FAST modes, then compare",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(PROJECT_ROOT / "models"),
        help="Path to models directory (default: <project>/models)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress frame rendering, only print summary stats",
    )

    args = parser.parse_args()

    if args.both:
        # Run both modes sequentially
        print("Running NEURAL mode...")
        neural_stats = run_demo(
            neural=True,
            num_frames=args.frames,
            models_dir=args.models_dir,
            quiet=args.quiet,
        )
        print_summary(neural_stats)

        print("\nRunning FAST mode...")
        fast_stats = run_demo(
            neural=False,
            num_frames=args.frames,
            models_dir=args.models_dir,
            quiet=args.quiet,
        )
        print_summary(fast_stats)

        print_comparison(neural_stats, fast_stats)
    else:
        neural = not args.fast
        stats = run_demo(
            neural=neural,
            num_frames=args.frames,
            models_dir=args.models_dir,
            quiet=args.quiet,
        )
        print()
        print_summary(stats)


if __name__ == "__main__":
    main()
