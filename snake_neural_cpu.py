#!/usr/bin/env python3
"""Backward-compatible wrapper for full-model snake demo.

Use `games/snake_full_model.py` for the canonical source.
"""

from games.snake_full_model import *  # noqa: F401,F403

if __name__ == "__main__":
    from games.snake_full_model import main
    main()
