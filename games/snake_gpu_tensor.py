#!/usr/bin/env python3
"""Tensor-first GPU Snake demo.

This variant keeps game state in torch tensors on GPU/MPS when available and
steps the environment with tensor operations. It is intended as a lightweight,
reproducible GPU workload for runtime validation.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch


ACTIONS = torch.tensor(
    [
        [0, -1],  # up
        [1, 0],   # right
        [0, 1],   # down
        [-1, 0],  # left
    ],
    dtype=torch.int64,
)


@dataclass
class SnakeStats:
    steps: int
    score: int
    alive: bool
    device: str


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TensorSnakeGPU:
    """Snake environment backed by tensor state."""

    def __init__(self, width: int = 40, height: int = 20, max_length: int = 256, device: Optional[torch.device] = None):
        if width < 8 or height < 8:
            raise ValueError("width and height must both be >= 8")

        self.width = width
        self.height = height
        self.max_length = max_length
        self.device = device or pick_device()

        self.actions = ACTIONS.to(self.device)

        # Tensor state
        self.snake = torch.zeros((max_length, 2), dtype=torch.int64, device=self.device)
        self.length = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.direction = torch.zeros(2, dtype=torch.int64, device=self.device)
        self.food = torch.zeros(2, dtype=torch.int64, device=self.device)
        self.score = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.steps = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.alive = torch.tensor(True, dtype=torch.bool, device=self.device)

        self.reset()

    def reset(self) -> None:
        cx = self.width // 2
        cy = self.height // 2

        self.snake.zero_()
        self.snake[0] = torch.tensor([cx, cy], dtype=torch.int64, device=self.device)
        self.snake[1] = torch.tensor([cx - 1, cy], dtype=torch.int64, device=self.device)
        self.snake[2] = torch.tensor([cx - 2, cy], dtype=torch.int64, device=self.device)

        self.length.fill_(3)
        self.direction[:] = torch.tensor([1, 0], dtype=torch.int64, device=self.device)
        self.score.zero_()
        self.steps.zero_()
        self.alive.fill_(True)
        self.food[:] = self._sample_food()

    def _sample_food(self) -> torch.Tensor:
        x = torch.randint(1, self.width - 1, (1,), dtype=torch.int64, device=self.device)
        y = torch.randint(1, self.height - 1, (1,), dtype=torch.int64, device=self.device)
        return torch.cat([x, y])

    def _occupied(self) -> torch.Tensor:
        length = int(self.length.item())
        return self.snake[:length]

    def step(self, action_idx: Optional[int] = None) -> bool:
        """Advance one tick. Returns whether snake is alive after this tick."""
        if not bool(self.alive.item()):
            return False

        if action_idx is None:
            action = torch.randint(0, 4, (1,), dtype=torch.int64, device=self.device).squeeze(0)
        else:
            action = torch.tensor(action_idx % 4, dtype=torch.int64, device=self.device)

        proposed_dir = self.actions[action]
        reverse = (proposed_dir + self.direction == 0).all()
        self.direction = torch.where(reverse, self.direction, proposed_dir)

        new_head = self.snake[0] + self.direction

        # Collision checks
        wall_hit = (
            (new_head[0] <= 0)
            | (new_head[0] >= self.width - 1)
            | (new_head[1] <= 0)
            | (new_head[1] >= self.height - 1)
        )

        body = self._occupied()
        self_hit = (body == new_head).all(dim=1).any()
        dead = wall_hit | self_hit

        # Move body
        length = int(self.length.item())
        shifted = torch.roll(self.snake, shifts=1, dims=0)
        shifted[0] = new_head
        self.snake[:length] = shifted[:length]

        ate_food = (new_head == self.food).all()
        can_grow = self.length < self.max_length
        grew = ate_food & (~dead) & can_grow

        if bool(grew.item()) and length < self.max_length:
            self.snake[length] = self.snake[length - 1]

        self.length = torch.where(grew, self.length + 1, self.length)
        self.score = torch.where(ate_food & (~dead), self.score + 10, self.score)

        replace_food = ate_food & (~dead)
        if bool(replace_food.item()):
            candidate = self._sample_food()
            on_snake = (self._occupied() == candidate).all(dim=1).any()
            self.food = torch.where(on_snake, self.food, candidate)

        self.alive = self.alive & (~dead)
        self.steps += 1
        return bool(self.alive.item())

    def stats(self) -> SnakeStats:
        return SnakeStats(
            steps=int(self.steps.item()),
            score=int(self.score.item()),
            alive=bool(self.alive.item()),
            device=str(self.device),
        )

    def render_ascii(self) -> str:
        """Render a text frame (CPU conversion only for display)."""
        board = torch.full((self.height, self.width), ord(" "), dtype=torch.uint8, device=self.device)

        # Borders
        board[0, :] = ord("#")
        board[-1, :] = ord("#")
        board[:, 0] = ord("#")
        board[:, -1] = ord("#")

        length = int(self.length.item())
        body = self.snake[:length]

        board[self.food[1], self.food[0]] = ord("*")
        if length > 0:
            board[body[:, 1], body[:, 0]] = ord("o")
            head = body[0]
            board[head[1], head[0]] = ord("@")

        cpu_board = board.cpu().numpy()
        return "\n".join("".join(chr(c) for c in row) for row in cpu_board)


def run_demo(steps: int, fps: float, render_every: int) -> SnakeStats:
    game = TensorSnakeGPU()
    delay = 1.0 / max(fps, 1.0)

    for step_idx in range(steps):
        alive = game.step()
        if render_every > 0 and step_idx % render_every == 0:
            print("\x1b[2J\x1b[H", end="")
            print(game.render_ascii())
            s = game.stats()
            print(f"device={s.device} steps={s.steps} score={s.score} alive={s.alive}")

        if not alive:
            break

        if render_every > 0:
            time.sleep(delay)

    return game.stats()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tensor-first GPU snake demo")
    parser.add_argument("--steps", type=int, default=500, help="Max steps to run")
    parser.add_argument("--fps", type=float, default=12.0, help="Render FPS")
    parser.add_argument("--render-every", type=int, default=1, help="Render every N steps; 0 = no render")
    args = parser.parse_args()

    stats = run_demo(args.steps, args.fps, args.render_every)
    print(f"final: device={stats.device} steps={stats.steps} score={stats.score} alive={stats.alive}")


if __name__ == "__main__":
    main()
