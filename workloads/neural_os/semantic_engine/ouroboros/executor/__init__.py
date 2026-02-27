"""
OUROBOROS Executor
===================
Run both tracks in parallel and compare results.
"""

from .parallel_runner import ParallelRunner, RunConfig, RunResult

__all__ = [
    "ParallelRunner",
    "RunConfig",
    "RunResult",
]
