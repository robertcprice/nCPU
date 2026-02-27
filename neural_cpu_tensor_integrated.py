#!/usr/bin/env python3
"""Backward-compatible wrapper for tensor-native kernel runtime.

Use `runtimes/tensor_optimized/tensor_native_kernel.py` for the canonical source.
"""

from runtimes.tensor_optimized.tensor_native_kernel import *  # noqa: F401,F403

if __name__ == "__main__":
    try:
        from runtimes.tensor_optimized.tensor_native_kernel import main
        main()
    except ImportError as exc:
        try:
            from runtimes.tensor_optimized.tensor_native_kernel import benchmark
            benchmark()
        except ImportError:
            raise SystemExit(str(exc))
