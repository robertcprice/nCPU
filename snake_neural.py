#!/usr/bin/env python3
"""Backward-compatible wrapper for GPU snake demo.

Use `games/snake_gpu_tensor.py` for the canonical source.
"""

from games.snake_gpu_tensor import *  # noqa: F401,F403

if __name__ == "__main__":
    from games.snake_gpu_tensor import main
    main()
