"""Benchmark: NPCoT soft-forward vs library fast-path (NV3).

Measures the time-per-call ratio between:

* **Soft forward**: the M2 differentiable array-thought head running a
  temperature-softmaxed reduction with autograd disabled (inference mode).
* **Library hit**: the M3 fast path where every sample is served by a
  cached `DiscreteArrayProgram` executed as pure tensor ops.

Expected result: library hits should be 5-50x faster than the soft forward on
moderate batch sizes, because they skip the 5 softmaxes, the 4-way soft
reduction, and the trace encoder / state patch projections that the head
runs even at inference.

Usage::

    python3 -m benchmarks.benchmark_npcot_library
    python3 -m benchmarks.benchmark_npcot_library --device mps --batch 64 --iters 200
    python3 -m benchmarks.benchmark_npcot_library --json out.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ncpu.self_optimizing.array_executable_thought_head import (
    ArrayExecutableThoughtHead,
    ArrayExecutableThoughtHeadConfig,
    build_array_thought_smoke_batch,
    run_array_thought_smoke_train,
)
from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
)


@dataclass
class BenchmarkResult:
    device: str
    batch_size: int
    iters: int
    soft_forward_seconds: float
    library_hit_seconds: float
    speedup: float
    library_hit_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def _bench_soft(
    head: ArrayExecutableThoughtHead,
    hidden: torch.Tensor,
    arrays: torch.Tensor,
    lengths: torch.Tensor,
    iters: int,
) -> float:
    head.eval()
    with torch.no_grad():
        for _ in range(3):
            head(hidden, arrays, lengths=lengths, temperature=0.1)
        if hidden.is_mps:
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            _ = head(hidden, arrays, lengths=lengths, temperature=0.1)
        if hidden.is_mps:
            torch.mps.synchronize()
        return time.perf_counter() - start


def _bench_library(
    head: ArrayExecutableThoughtHead,
    hidden: torch.Tensor,
    arrays: torch.Tensor,
    lengths: torch.Tensor,
    library: ArrayProgramLibrary,
    iters: int,
) -> tuple[float, float]:
    head.eval()
    with torch.no_grad():
        # Warm up and measure hit rate.
        result = head.consult_library(
            hidden,
            arrays,
            library,
            lengths=lengths,
            temperature=0.1,
            auto_cache=False,
        )
        hit_rate = sum(result.library_hits) / max(len(result.library_hits), 1)
        for _ in range(3):
            head.consult_library(
                hidden,
                arrays,
                library,
                lengths=lengths,
                temperature=0.1,
                auto_cache=False,
            )
        if hidden.is_mps:
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            _ = head.consult_library(
                hidden,
                arrays,
                library,
                lengths=lengths,
                temperature=0.1,
                auto_cache=False,
            )
        if hidden.is_mps:
            torch.mps.synchronize()
        return time.perf_counter() - start, hit_rate


def run_benchmark(
    *,
    device: str = "cpu",
    batch_size: int = 18,
    iters: int = 200,
    train_steps: int = 400,
) -> BenchmarkResult:
    torch.manual_seed(0)
    config = ArrayExecutableThoughtHeadConfig(
        hidden_dim=8, array_max_len=6
    )
    head = ArrayExecutableThoughtHead(config).to(device)
    hidden, arrays, lengths, targets, _ = build_array_thought_smoke_batch(
        hidden_dim=8,
        array_max_len=6,
        samples_per_op=max(1, batch_size // 3),
        seed=0,
        device=device,
    )

    # Train to convergence so the library can capture real skills.
    run_array_thought_smoke_train(
        head,
        hidden_state=hidden,
        array_inputs=arrays,
        lengths=lengths,
        targets=targets,
        steps=train_steps,
        learning_rate=5e-2,
    )

    # Build library from converged programs.
    library = ArrayProgramLibrary(
        ArrayProgramLibraryConfig(similarity_threshold=0.85, max_entries=64)
    )
    head.consult_library(
        hidden,
        arrays,
        library,
        lengths=lengths,
        temperature=0.1,
        auto_cache=True,
        convergence_gap_threshold=1.0,
    )

    soft_seconds = _bench_soft(head, hidden, arrays, lengths, iters)
    library_seconds, hit_rate = _bench_library(
        head, hidden, arrays, lengths, library, iters
    )

    speedup = soft_seconds / max(library_seconds, 1e-9)
    return BenchmarkResult(
        device=device,
        batch_size=int(hidden.shape[0]),
        iters=iters,
        soft_forward_seconds=soft_seconds,
        library_hit_seconds=library_seconds,
        speedup=speedup,
        library_hit_rate=hit_rate,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--batch", type=int, default=18)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--json", type=Path, help="Write results JSON here")
    args = parser.parse_args(argv)

    if args.device == "mps" and not (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ):
        print("MPS not available on this system; falling back to cpu")
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available on this system; falling back to cpu")
        args.device = "cpu"

    result = run_benchmark(
        device=args.device,
        batch_size=args.batch,
        iters=args.iters,
        train_steps=args.train_steps,
    )

    print(f"NPCoT library benchmark  ({result.device}, batch={result.batch_size}, iters={result.iters})")
    print(f"  soft forward total:  {result.soft_forward_seconds:.3f} s  "
          f"(per-call {result.soft_forward_seconds / result.iters * 1000:.3f} ms)")
    print(f"  library hit total:   {result.library_hit_seconds:.3f} s  "
          f"(per-call {result.library_hit_seconds / result.iters * 1000:.3f} ms)")
    print(f"  speedup:             {result.speedup:.2f}x")
    print(f"  library hit rate:    {100 * result.library_hit_rate:.1f}%")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
