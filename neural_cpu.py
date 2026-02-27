#!/usr/bin/env python3
"""Backward-compatible wrapper for full-model runtime.

Use `runtimes/full_model/neural_cpu_full.py` for the canonical source.
"""

from runtimes.full_model.neural_cpu_full import *  # noqa: F401,F403

if __name__ == "__main__":
    try:
        from runtimes.full_model.neural_cpu_full import main
        main()
    except ImportError as exc:
        try:
            from runtimes.full_model.neural_cpu_full import benchmark
            benchmark()
        except ImportError:
            raise SystemExit(str(exc))
