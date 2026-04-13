#!/usr/bin/env python3
"""Neural Game of Life -- Conway's Game of Life rendered through neural networks.

Runs Conway's Game of Life on the 80x24 terminal grid, rendering each
generation through the nCPU neural terminal renderer. Cells are drawn
using block characters with ANSI colors; the neural glyph MLP and color
embeddings produce every pixel.

Seed patterns:
  random      -- random initial state (default)
  glider      -- single glider traveling southeast
  r-pentomino -- R-pentomino (long-lived methuselah)
  acorn       -- Acorn pattern (5206 generations to stabilize)
  gosper      -- Gosper glider gun (produces gliders)
  pulsar      -- Period-3 oscillator

Usage:
    python demos/neural/neural_game_of_life.py
    python demos/neural/neural_game_of_life.py --seed glider --generations 200
    python demos/neural/neural_game_of_life.py --seed r-pentomino --fps 15
    python demos/neural/neural_game_of_life.py --output /tmp/life.gif
    python demos/neural/neural_game_of_life.py --live --scale 2
"""

from __future__ import annotations

import argparse
import random
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
    TERM_ROWS,
    TERM_COLS,
)


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

ESC = "\033["

def sgr(code: int) -> str:
    return f"{ESC}{code}m"

def fg(c: int) -> str:
    return sgr(c)

def bg(c: int) -> str:
    return sgr(c)

def bold() -> str:
    return sgr(1)

def reset() -> str:
    return sgr(0)

def cursor_pos(row: int, col: int) -> str:
    return f"{ESC}{row};{col}H"

def clear() -> str:
    return f"{ESC}2J{ESC}1;1H"


# ---------------------------------------------------------------------------
# Game of Life grid
# ---------------------------------------------------------------------------

# Reserve row 0 for header and row 23 for status
GRID_ROWS = TERM_ROWS - 2  # 22 playable rows
GRID_COLS = TERM_COLS       # 80 columns


def make_grid(rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    """Create an empty grid."""
    return np.zeros((rows, cols), dtype=np.int8)


def place_pattern(grid: np.ndarray, pattern: list[tuple[int, int]],
                  offset_r: int = 0, offset_c: int = 0) -> None:
    """Place a pattern on the grid at the given offset."""
    rows, cols = grid.shape
    for dr, dc in pattern:
        r = (offset_r + dr) % rows
        c = (offset_c + dc) % cols
        grid[r, c] = 1


def seed_grid(name: str, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    """Create a grid with the named seed pattern."""
    grid = make_grid(rows, cols)
    mid_r, mid_c = rows // 2, cols // 2

    if name == "random":
        for r in range(rows):
            for c in range(cols):
                grid[r, c] = 1 if random.random() < 0.3 else 0

    elif name == "glider":
        # Classic glider
        pattern = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        place_pattern(grid, pattern, mid_r - 5, mid_c - 5)

    elif name == "r-pentomino":
        # R-pentomino: long-lived methuselah
        pattern = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]
        place_pattern(grid, pattern, mid_r, mid_c)

    elif name == "acorn":
        # Acorn: stabilizes after 5206 generations
        pattern = [(0, 1), (1, 3), (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)]
        place_pattern(grid, pattern, mid_r, mid_c - 3)

    elif name == "gosper":
        # Gosper glider gun
        pattern = [
            (0, 24),
            (1, 22), (1, 24),
            (2, 12), (2, 13), (2, 20), (2, 21), (2, 34), (2, 35),
            (3, 11), (3, 15), (3, 20), (3, 21), (3, 34), (3, 35),
            (4, 0), (4, 1), (4, 10), (4, 16), (4, 20), (4, 21),
            (5, 0), (5, 1), (5, 10), (5, 14), (5, 16), (5, 17), (5, 22), (5, 24),
            (6, 10), (6, 16), (6, 24),
            (7, 11), (7, 15),
            (8, 12), (8, 13),
        ]
        place_pattern(grid, pattern, 2, 2)

    elif name == "pulsar":
        # Period-3 oscillator
        quarter = [
            (1, 2), (1, 3), (1, 4),
            (2, 1), (3, 1), (4, 1),
            (2, 6), (3, 6), (4, 6),
            (6, 2), (6, 3), (6, 4),
        ]
        # Full pulsar from quarter pattern reflected
        pattern = []
        for dr, dc in quarter:
            pattern.append((dr, dc))
            pattern.append((dr, -dc))
            pattern.append((-dr, dc))
            pattern.append((-dr, -dc))
        # Deduplicate
        pattern = list(set(pattern))
        place_pattern(grid, pattern, mid_r, mid_c)

    else:
        raise ValueError(f"Unknown seed pattern: {name}")

    return grid


def step_grid(grid: np.ndarray) -> np.ndarray:
    """Advance the Game of Life by one generation using toroidal boundary."""
    rows, cols = grid.shape
    # Count neighbors using rolled sums (toroidal wrap)
    neighbors = np.zeros_like(grid, dtype=np.int8)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbors += np.roll(np.roll(grid, dr, axis=0), dc, axis=1)

    # Conway's rules
    new_grid = np.zeros_like(grid)
    # Birth: dead cell with exactly 3 neighbors
    new_grid[(grid == 0) & (neighbors == 3)] = 1
    # Survival: live cell with 2 or 3 neighbors
    new_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
    return new_grid


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# Color cycle for generation-based coloring
LIFE_COLORS = [92, 93, 91, 96, 95, 94]  # green, yellow, red, cyan, magenta, blue


def render_generation(
    display: NeuralDisplay,
    grid: np.ndarray,
    generation: int,
    population: int,
    seed_name: str,
) -> np.ndarray:
    """Render a single generation to a neural display frame.

    Uses cursor positioning to write directly into the terminal state
    for maximum efficiency (no full-screen rewrite each frame).
    """
    display.reset()
    rows, cols = grid.shape

    # Header
    color = LIFE_COLORS[generation % len(LIFE_COLORS)]
    header = (
        f"{cursor_pos(1, 1)}"
        f"{fg(color)}{bold()}Game of Life{reset()}"
        f"  {fg(97)}Gen:{reset()} {fg(93)}{generation:<5}{reset()}"
        f"  {fg(97)}Pop:{reset()} {fg(92)}{population:<5}{reset()}"
        f"  {fg(97)}Seed:{reset()} {fg(36)}{seed_name}{reset()}"
    )
    display.terminal.write_str(header)

    # Grid body (rows 1..22 in 0-indexed, or terminal rows 2..23)
    for r in range(rows):
        display.terminal.write_str(cursor_pos(r + 2, 1))
        line_parts = []
        for c in range(cols):
            if grid[r, c]:
                # Use different colors based on neighbor count for visual interest
                neighbors = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = (r + dr) % rows
                        nc = (c + dc) % cols
                        neighbors += grid[nr, nc]
                if neighbors <= 2:
                    line_parts.append(f"{fg(92)}#{reset()}")  # green - sparse
                elif neighbors == 3:
                    line_parts.append(f"{fg(93)}#{reset()}")  # yellow - birth zone
                else:
                    line_parts.append(f"{fg(91)}#{reset()}")  # red - crowded
            else:
                line_parts.append(" ")
        display.terminal.write_str("".join(line_parts))

    # Status bar
    display.terminal.write_str(
        f"{cursor_pos(TERM_ROWS, 1)}"
        f"{fg(90)}Neural Display | 143K params | "
        f"Every pixel from neural networks{reset()}"
    )

    return display.render()


def render_generation_fast(
    display: NeuralDisplay,
    grid: np.ndarray,
    generation: int,
    population: int,
    seed_name: str,
) -> np.ndarray:
    """Fast rendering path: directly manipulate terminal state arrays.

    Instead of writing ANSI escape sequences character by character, this
    writes directly into the TerminalState char/fg/bg arrays. Much faster
    for large grids.
    """
    ts = display.terminal
    ts.reset()
    rows, cols = grid.shape

    # Header (row 0)
    header = f"Game of Life  Gen: {generation:<5}  Pop: {population:<5}  Seed: {seed_name}"
    for i, ch in enumerate(header[:TERM_COLS]):
        ts.chars[0, i] = ord(ch)
        ts.fg[0, i] = 97 if i < 12 else (93 if 18 <= i < 24 else (92 if 30 <= i < 36 else 36))

    # Grid body (terminal rows 1..22)
    for r in range(rows):
        tr = r + 1  # terminal row
        for c in range(cols):
            if grid[r, c]:
                ts.chars[tr, c] = ord('#')
                # Color based on neighbor density
                neighbors = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = (r + dr) % rows
                        nc = (c + dc) % cols
                        neighbors += grid[nr, nc]
                if neighbors <= 2:
                    ts.fg[tr, c] = 2   # green
                elif neighbors == 3:
                    ts.fg[tr, c] = 3   # yellow
                else:
                    ts.fg[tr, c] = 1   # red
            # else: already space with default colors from reset()

    # Status bar (row 23)
    status = "Neural Display | 143K params | Every pixel from neural networks"
    for i, ch in enumerate(status[:TERM_COLS]):
        ts.chars[TERM_ROWS - 1, i] = ord(ch)
        ts.fg[TERM_ROWS - 1, i] = 8  # bright black (gray)

    return display.render()


# ---------------------------------------------------------------------------
# GIF saving
# ---------------------------------------------------------------------------

def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Save frames as animated GIF."""
    try:
        from PIL import Image
    except ImportError:
        print(f"  [WARNING] PIL not installed -- cannot save GIF to {path}")
        print("  Install with: pip install Pillow")
        return

    images = [Image.fromarray(f) for f in frames]
    duration = max(1, int(1000 / fps))
    images[0].save(
        str(path),
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    size_kb = path.stat().st_size / 1024
    print(f"  Saved: {path} ({len(frames)} frames, {size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Live display
# ---------------------------------------------------------------------------

def run_live(
    display: NeuralDisplay,
    seed_name: str,
    generations: int,
    scale: int,
    target_fps: int,
) -> None:
    """Run Game of Life in a live pygame window."""
    try:
        import pygame
    except ImportError:
        print()
        print("  [ERROR] pygame is required for --live mode.")
        print("  Install with: pip install pygame")
        print()
        sys.exit(1)

    pygame.init()
    win_w = FRAME_W * scale
    win_h = FRAME_H * scale
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("nCPU Neural Game of Life")
    clock = pygame.time.Clock()

    grid = seed_grid(seed_name)
    gen = 0
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Reset with new random seed
                    grid = seed_grid(seed_name)
                    gen = 0

        if not paused:
            population = int(grid.sum())
            frame = render_generation_fast(display, grid, gen, population, seed_name)

            surface = pygame.surfarray.make_surface(
                np.transpose(frame, (1, 0, 2))
            )
            if scale != 1:
                surface = pygame.transform.scale(surface, (win_w, win_h))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            grid = step_grid(grid)
            gen += 1

            if generations > 0 and gen >= generations:
                running = False

        pygame.display.set_caption(
            f"nCPU Neural Game of Life -- Gen {gen} | {'PAUSED' if paused else 'Running'}"
        )
        clock.tick(target_fps)

    pygame.quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neural Game of Life -- Conway's Life rendered through neural networks"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "models" / "display" / "neural_game_of_life.gif"),
        help="Output GIF path",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: cpu, mps, cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--generations", type=int, default=100,
        help="Number of generations to simulate (default: 100)",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second (default: 10)",
    )
    parser.add_argument(
        "--seed", type=str, default="random",
        choices=["random", "glider", "r-pentomino", "acorn", "gosper", "pulsar"],
        help="Initial pattern (default: random)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run in a live pygame window instead of saving GIF",
    )
    parser.add_argument(
        "--scale", type=int, default=2,
        help="Window scale for --live mode (default: 2)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=None,
        help="Random number seed for reproducible initial states",
    )
    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"

    print()
    print("=" * 60)
    print("  nCPU Neural Game of Life")
    print("  Conway's Life rendered through neural networks")
    print("=" * 60)
    print()
    print(f"  Model:       {model_path}")

    display = NeuralDisplay(str(model_path), device=args.device)
    print(f"  Device:      {display.device}")
    print(f"  Metal:       {display.metal_available}")
    print(f"  Seed:        {args.seed}")
    print(f"  Generations: {args.generations}")
    print(f"  Grid:        {GRID_COLS}x{GRID_ROWS}")
    print()

    if args.live:
        print(f"  Mode: Live display (scale={args.scale}x, fps={args.fps})")
        print("  SPACE=pause, R=reset, ESC/Q=quit")
        print()
        run_live(display, args.seed, args.generations, args.scale, args.fps)
    else:
        print(f"  Mode: GIF capture (fps={args.fps})")
        print()

        grid = seed_grid(args.seed)
        initial_pop = int(grid.sum())
        print(f"  Initial population: {initial_pop}")
        print("  Simulating...")

        frames = []
        t0 = time.perf_counter()

        for gen in range(args.generations):
            population = int(grid.sum())
            frame = render_generation_fast(display, grid, gen, population, args.seed)
            frames.append(frame)

            if (gen + 1) % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    Gen {gen + 1:>4}/{args.generations}: pop={population}, "
                      f"{elapsed:.1f}s ({(gen + 1) / elapsed:.1f} gen/s)")

            grid = step_grid(grid)

            # Stop if population dies
            if population == 0 and gen > 0:
                print(f"  Population extinct at generation {gen}.")
                break

        dt = time.perf_counter() - t0
        print(f"  Captured {len(frames)} frames in {dt:.1f}s "
              f"({len(frames) / dt:.1f} frames/s)")
        print()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, output_path, args.fps)

    print()
    print("  Done.")
    print()


if __name__ == "__main__":
    main()
