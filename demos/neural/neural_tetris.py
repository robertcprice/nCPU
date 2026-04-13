#!/usr/bin/env python3
"""Neural Tetris — classic Tetris rendered entirely through neural networks.

Every pixel on screen is produced by the nCPU neural terminal renderer: character
embeddings generate glyph alpha masks via MLPs, a learned color palette provides
RGB values, and alpha blending composites the final 640x384 frame. The game uses
ANSI colors and block characters on a 24x80 virtual terminal, which the neural
pipeline renders to a pixel display.

Features:
  - Standard Tetris on a 10x20 board with 7 piece types (I, O, T, S, Z, J, L)
  - ANSI color-coded pieces: I=cyan, O=yellow, T=magenta, S=green, Z=red, J=blue, L=orange
  - Next piece preview, score, level, lines cleared
  - Keyboard: arrows=move/rotate, space=hard drop, down=soft drop, P=pause
  - Auto-demo mode (--auto): greedy AI plays automatically
  - GIF recording (--output, --record): save gameplay as animated GIF

Usage:
    python demos/neural/neural_tetris.py
    python demos/neural/neural_tetris.py --auto
    python demos/neural/neural_tetris.py --auto --output tetris.gif --record 300
    python demos/neural/neural_tetris.py --fps 15 --scale 2
    python demos/neural/neural_tetris.py --device mps --v2
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import pygame
except ImportError:
    print("ERROR: pygame is required for neural Tetris.")
    print("Install with: pip install pygame")
    sys.exit(1)

from ncpu.neural.neural_terminal_renderer import (
    NeuralDisplay, FRAME_H, FRAME_W, TERM_ROWS, TERM_COLS,
)


# ---------------------------------------------------------------------------
# Tetris constants
# ---------------------------------------------------------------------------

BOARD_W = 10
BOARD_H = 20
BOARD_X = 2    # Column offset for the board on the 80-col terminal
BOARD_Y = 2    # Row offset for the board on the 24-row terminal

# Piece shapes as list of (row, col) offsets from anchor. Each piece has 4 rotations.
PIECES = {
    'I': [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
    'O': [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    'T': [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (1, -1)],
    ],
    'S': [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    'Z': [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    'J': [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, -1)],
    ],
    'L': [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}

# ANSI SGR color codes for each piece type (foreground)
PIECE_COLORS = {
    'I': 36,   # Cyan
    'O': 33,   # Yellow
    'T': 35,   # Magenta
    'S': 32,   # Green
    'Z': 31,   # Red
    'J': 34,   # Blue
    'L': 91,   # Bright red (closest to orange in 16-color ANSI)
}

PIECE_NAMES = list(PIECES.keys())

# Scoring (NES-style)
LINE_SCORES = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class TetrisGame:
    """Core Tetris game logic, decoupled from rendering."""

    def __init__(self):
        self.board: list[list[Optional[str]]] = [
            [None] * BOARD_W for _ in range(BOARD_H)
        ]
        self.score = 0
        self.level = 1
        self.lines = 0
        self.game_over = False
        self.paused = False

        self._bag: list[str] = []
        self.current_piece = self._next_piece()
        self.current_rot = 0
        self.current_row = 0
        self.current_col = BOARD_W // 2 - 1
        self.next_piece = self._next_piece()

    def _next_piece(self) -> str:
        """7-bag randomizer: shuffle all 7 pieces, deal one at a time."""
        if not self._bag:
            self._bag = list(PIECE_NAMES)
            random.shuffle(self._bag)
        return self._bag.pop()

    def _cells(self, piece: str, rot: int, row: int, col: int) -> list[tuple[int, int]]:
        """Get absolute (row, col) coordinates for a piece placement."""
        return [(row + dr, col + dc) for dr, dc in PIECES[piece][rot % 4]]

    def _valid(self, piece: str, rot: int, row: int, col: int) -> bool:
        """Check if a piece placement is valid (in bounds and no collision)."""
        for r, c in self._cells(piece, rot, row, col):
            if r < 0 or r >= BOARD_H or c < 0 or c >= BOARD_W:
                return False
            if self.board[r][c] is not None:
                return False
        return True

    def _lock(self) -> None:
        """Lock current piece into the board and spawn next."""
        cells = self._cells(
            self.current_piece, self.current_rot,
            self.current_row, self.current_col,
        )
        for r, c in cells:
            if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                self.board[r][c] = self.current_piece

        # Clear complete lines
        cleared = 0
        new_board = []
        for row in self.board:
            if all(cell is not None for cell in row):
                cleared += 1
            else:
                new_board.append(row)
        while len(new_board) < BOARD_H:
            new_board.insert(0, [None] * BOARD_W)
        self.board = new_board

        self.lines += cleared
        self.score += LINE_SCORES.get(cleared, 0) * self.level
        self.level = 1 + self.lines // 10

        # Spawn next piece
        self.current_piece = self.next_piece
        self.next_piece = self._next_piece()
        self.current_rot = 0
        self.current_row = 0
        self.current_col = BOARD_W // 2 - 1

        if not self._valid(self.current_piece, self.current_rot,
                           self.current_row, self.current_col):
            self.game_over = True

    def move_left(self) -> bool:
        if self._valid(self.current_piece, self.current_rot,
                       self.current_row, self.current_col - 1):
            self.current_col -= 1
            return True
        return False

    def move_right(self) -> bool:
        if self._valid(self.current_piece, self.current_rot,
                       self.current_row, self.current_col + 1):
            self.current_col += 1
            return True
        return False

    def rotate(self) -> bool:
        new_rot = (self.current_rot + 1) % 4
        if self._valid(self.current_piece, new_rot,
                       self.current_row, self.current_col):
            self.current_rot = new_rot
            return True
        # Wall kick: try shifting left/right
        for dx in [-1, 1, -2, 2]:
            if self._valid(self.current_piece, new_rot,
                           self.current_row, self.current_col + dx):
                self.current_rot = new_rot
                self.current_col += dx
                return True
        return False

    def soft_drop(self) -> bool:
        if self._valid(self.current_piece, self.current_rot,
                       self.current_row + 1, self.current_col):
            self.current_row += 1
            return True
        return False

    def hard_drop(self) -> int:
        """Drop piece to bottom, return number of rows dropped."""
        rows = 0
        while self._valid(self.current_piece, self.current_rot,
                          self.current_row + 1, self.current_col):
            self.current_row += 1
            rows += 1
        self._lock()
        return rows

    def tick(self) -> None:
        """Gravity tick: move piece down one row, lock if can't."""
        if self.game_over or self.paused:
            return
        if not self.soft_drop():
            self._lock()

    def get_drop_row(self) -> int:
        """Get the row where the piece would land (for ghost piece)."""
        r = self.current_row
        while self._valid(self.current_piece, self.current_rot,
                          r + 1, self.current_col):
            r += 1
        return r


# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------

def render_game(game: TetrisGame, terminal) -> None:
    """Render the Tetris game state into a TerminalState via ANSI sequences."""
    # Clear screen
    terminal.write_str("\x1b[2J\x1b[H")

    # Board border and title
    terminal.write_str(f"\x1b[1;{BOARD_X}H\x1b[1;37m NEURAL TETRIS\x1b[0m")

    # Draw the board
    for row in range(BOARD_H):
        tr = BOARD_Y + row  # terminal row (1-indexed for ANSI)
        tc = BOARD_X        # terminal col

        terminal.write_str(f"\x1b[{tr};{tc}H\x1b[90m|\x1b[0m")

        for col in range(BOARD_W):
            cell = game.board[row][col]
            if cell is not None:
                color = PIECE_COLORS[cell]
                terminal.write_str(f"\x1b[{color}m##\x1b[0m")
            else:
                terminal.write_str("  ")

        terminal.write_str(f"\x1b[90m|\x1b[0m")

    # Bottom border
    terminal.write_str(f"\x1b[{BOARD_Y + BOARD_H};{BOARD_X}H\x1b[90m+{'--' * BOARD_W}+\x1b[0m")

    # Draw current piece
    cells = game._cells(
        game.current_piece, game.current_rot,
        game.current_row, game.current_col,
    )
    color = PIECE_COLORS[game.current_piece]
    for r, c in cells:
        if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
            tr = BOARD_Y + r
            tc = BOARD_X + 1 + c * 2
            terminal.write_str(f"\x1b[{tr};{tc}H\x1b[{color}m##\x1b[0m")

    # Ghost piece (faint)
    ghost_row = game.get_drop_row()
    if ghost_row != game.current_row:
        ghost_cells = game._cells(
            game.current_piece, game.current_rot,
            ghost_row, game.current_col,
        )
        for r, c in ghost_cells:
            if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                # Only draw ghost if cell not already occupied
                if game.board[r][c] is None:
                    tr = BOARD_Y + r
                    tc = BOARD_X + 1 + c * 2
                    terminal.write_str(f"\x1b[{tr};{tc}H\x1b[90m..\x1b[0m")

    # Side panel — next piece, score, etc.
    panel_x = BOARD_X + 2 + BOARD_W * 2 + 3

    terminal.write_str(f"\x1b[{BOARD_Y};{panel_x}H\x1b[1;37mNEXT:\x1b[0m")
    next_color = PIECE_COLORS[game.next_piece]
    next_cells = PIECES[game.next_piece][0]
    for dr, dc in next_cells:
        tr = BOARD_Y + 1 + dr
        tc = panel_x + dc * 2
        terminal.write_str(f"\x1b[{tr};{tc}H\x1b[{next_color}m##\x1b[0m")

    info_y = BOARD_Y + 6
    terminal.write_str(f"\x1b[{info_y};{panel_x}H\x1b[1;37mSCORE\x1b[0m")
    terminal.write_str(f"\x1b[{info_y + 1};{panel_x}H\x1b[97m{game.score:>8}\x1b[0m")

    terminal.write_str(f"\x1b[{info_y + 3};{panel_x}H\x1b[1;37mLEVEL\x1b[0m")
    terminal.write_str(f"\x1b[{info_y + 4};{panel_x}H\x1b[97m{game.level:>8}\x1b[0m")

    terminal.write_str(f"\x1b[{info_y + 6};{panel_x}H\x1b[1;37mLINES\x1b[0m")
    terminal.write_str(f"\x1b[{info_y + 7};{panel_x}H\x1b[97m{game.lines:>8}\x1b[0m")

    # Controls reminder
    ctrl_y = BOARD_Y + BOARD_H + 1
    if ctrl_y < TERM_ROWS:
        terminal.write_str(
            f"\x1b[{ctrl_y};{BOARD_X}H"
            "\x1b[90mArrows:move Up:rot Space:drop P:pause\x1b[0m"
        )

    if game.game_over:
        # Center "GAME OVER" on the board
        gy = BOARD_Y + BOARD_H // 2
        gx = BOARD_X + BOARD_W - 4
        terminal.write_str(f"\x1b[{gy};{gx}H\x1b[1;91m GAME OVER \x1b[0m")

    if game.paused and not game.game_over:
        py = BOARD_Y + BOARD_H // 2
        px = BOARD_X + BOARD_W - 3
        terminal.write_str(f"\x1b[{py};{px}H\x1b[1;93m PAUSED \x1b[0m")


# ---------------------------------------------------------------------------
# Greedy AI for auto-demo mode
# ---------------------------------------------------------------------------

def ai_decide(game: TetrisGame) -> tuple[int, int]:
    """Simple greedy AI: find placement that minimizes total height + holes.

    Returns (target_col, target_rot).
    """
    best_score = float('inf')
    best_col = game.current_col
    best_rot = 0

    for rot in range(4):
        for col in range(-2, BOARD_W + 2):
            if not game._valid(game.current_piece, rot, 0, col):
                continue

            # Simulate drop
            r = 0
            while game._valid(game.current_piece, rot, r + 1, col):
                r += 1

            cells = game._cells(game.current_piece, rot, r, col)

            # Check all cells are in bounds
            valid = all(0 <= cr < BOARD_H and 0 <= cc < BOARD_W for cr, cc in cells)
            if not valid:
                continue

            # Simulate board after placement
            test_board = [row[:] for row in game.board]
            for cr, cc in cells:
                test_board[cr][cc] = game.current_piece

            # Score: lower is better
            # Count complete lines (good — subtract)
            complete = sum(
                1 for row in test_board if all(c is not None for c in row)
            )

            # Aggregate height
            heights = []
            for c in range(BOARD_W):
                h = 0
                for r2 in range(BOARD_H):
                    if test_board[r2][c] is not None:
                        h = BOARD_H - r2
                        break
                heights.append(h)
            total_height = sum(heights)

            # Count holes (empty cells below a filled cell)
            holes = 0
            for c in range(BOARD_W):
                found_block = False
                for r2 in range(BOARD_H):
                    if test_board[r2][c] is not None:
                        found_block = True
                    elif found_block:
                        holes += 1

            # Bumpiness (height differences between adjacent columns)
            bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(BOARD_W - 1))

            score = total_height * 1.0 + holes * 4.0 + bumpiness * 1.0 - complete * 10.0

            if score < best_score:
                best_score = score
                best_col = col
                best_rot = rot

    return best_col, best_rot


# ---------------------------------------------------------------------------
# GIF recording
# ---------------------------------------------------------------------------

def save_gif(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    """Save a list of RGB frames as an animated GIF."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow is required for GIF export.")
        print("Install with: pip install Pillow")
        return

    if not frames:
        print("No frames to save.")
        return

    images = [Image.fromarray(f) for f in frames]
    duration_ms = max(1, 1000 // fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Saved {len(frames)} frames to {output_path} ({fps} FPS, "
          f"{Path(output_path).stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Display loading
# ---------------------------------------------------------------------------

def _load_display(args: argparse.Namespace) -> NeuralDisplay:
    """Load V1 or V2 neural display."""
    if args.v2:
        from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
        path = PROJECT_ROOT / "models" / "display" / "terminal_renderer_v2.pt"
        return NeuralDisplayV2(str(path), device=args.device)
    path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"
    return NeuralDisplay(str(path), device=args.device)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Neural Tetris — rendered by neural networks")
    parser.add_argument("--scale", type=int, default=2, help="Window scale (default: 2)")
    parser.add_argument("--device", type=str, default=None, help="Render device")
    parser.add_argument("--v2", action="store_true", help="Use V2 renderer")
    parser.add_argument("--fps", type=int, default=10, help="Game ticks per second (default: 10)")
    parser.add_argument("--auto", action="store_true", help="AI-controlled auto-demo mode")
    parser.add_argument("--output", type=str, default=None, help="Path to save GIF of gameplay")
    parser.add_argument("--record", type=int, default=0, help="Record N frames then save GIF")
    args = parser.parse_args()

    scale = max(1, args.scale)
    win_w = FRAME_W * scale
    win_h = FRAME_H * scale

    print(f"Loading neural renderer ({'V2' if args.v2 else 'V1'})...")
    display = _load_display(args)
    backend = "Metal GPU" if getattr(display, 'metal_available', False) else "PyTorch"
    mode_str = "AI AUTO" if args.auto else "PLAYER"
    print(f"Backend: {backend} | Mode: {mode_str} | Game FPS: {args.fps}")

    pygame.init()
    pygame.display.set_caption("nCPU Neural Tetris")
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    game = TetrisGame()
    random.seed(int(time.time()))

    # Timing for gravity
    tick_interval = 1.0 / max(1, args.fps)
    last_tick = time.perf_counter()

    # AI state
    ai_target_col = game.current_col
    ai_target_rot = 0
    ai_move_delay = 0.05  # seconds between AI moves
    last_ai_move = time.perf_counter()
    ai_decided = False

    # Recording
    recording = args.record > 0
    recorded_frames: list[np.ndarray] = []
    max_record = args.record if args.record > 0 else float('inf')

    frame_times: deque[float] = deque(maxlen=30)
    running = True

    while running:
        t0 = time.perf_counter()

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or (event.key == pygame.K_c and event.mod & pygame.KMOD_CTRL):
                    running = False
                    break
                if event.key == pygame.K_p:
                    game.paused = not game.paused
                if not game.game_over and not game.paused and not args.auto:
                    if event.key == pygame.K_LEFT:
                        game.move_left()
                    elif event.key == pygame.K_RIGHT:
                        game.move_right()
                    elif event.key == pygame.K_UP:
                        game.rotate()
                    elif event.key == pygame.K_DOWN:
                        game.soft_drop()
                    elif event.key == pygame.K_SPACE:
                        game.hard_drop()
                # Allow restart on game over
                if game.game_over and event.key == pygame.K_r:
                    game = TetrisGame()
                    ai_decided = False

        if not running:
            break

        now = time.perf_counter()

        # --- AI moves ---
        if args.auto and not game.game_over and not game.paused:
            if not ai_decided:
                ai_target_col, ai_target_rot = ai_decide(game)
                ai_decided = True

            if now - last_ai_move >= ai_move_delay:
                last_ai_move = now
                # Rotate toward target
                if game.current_rot != ai_target_rot % 4:
                    game.rotate()
                # Move toward target column
                elif game.current_col < ai_target_col:
                    game.move_right()
                elif game.current_col > ai_target_col:
                    game.move_left()
                else:
                    # In position — hard drop
                    game.hard_drop()
                    ai_decided = False

        # --- Gravity tick ---
        if now - last_tick >= tick_interval:
            last_tick = now
            if not game.game_over and not game.paused:
                game.tick()
                # Recalculate AI decision after tick (piece may have changed)
                if args.auto:
                    ai_decided = False

            # Speed up with level
            tick_interval = max(0.05, 1.0 / (args.fps + game.level * 0.5))

        # --- Render to terminal state ---
        render_game(game, display.terminal)

        # --- Neural render ---
        frame = display.render()

        # --- Recording ---
        if recording and len(recorded_frames) < max_record:
            recorded_frames.append(frame.copy())
            if len(recorded_frames) >= max_record:
                # Auto-save and stop recording
                if args.output:
                    save_gif(recorded_frames, args.output, args.fps)
                recording = False

        # --- Display ---
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        if scale != 1:
            surf = pygame.transform.scale(surf, (win_w, win_h))
        screen.blit(surf, (0, 0))

        # FPS overlay
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        avg = sum(frame_times) / len(frame_times) if frame_times else 1.0
        render_fps = 1.0 / avg if avg > 0 else 0.0
        font = pygame.font.SysFont("monospace", max(12, 10 * scale))
        fps_text = font.render(f"{render_fps:.0f} FPS", True, (0, 255, 0), (0, 0, 0))
        screen.blit(fps_text, (win_w - fps_text.get_width() - 4, 4))

        pygame.display.flip()
        clock.tick(60)  # Render at up to 60 FPS, game ticks are independent

    # Save GIF if we have recorded frames and haven't already saved
    if recorded_frames and args.output:
        save_gif(recorded_frames, args.output, args.fps)

    pygame.quit()
    print(f"Game over! Score: {game.score} | Level: {game.level} | Lines: {game.lines}")


if __name__ == "__main__":
    main()
