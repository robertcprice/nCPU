"""Smoke tests for organized runtime variants and GPU snake workload."""

import importlib
import sys
from pathlib import Path

import pytest

# Ensure imports resolve when pytest rootdir is a parent workspace folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_full_model_runtime_imports() -> None:
    module = importlib.import_module("runtimes.full_model.neural_cpu_full")
    assert hasattr(module, "NeuralCPU")


def test_tensor_runtime_imports() -> None:
    module = importlib.import_module("runtimes.tensor_optimized.tensor_native_cpu")
    assert hasattr(module, "TensorNativeCPU")


def test_gpu_snake_runs_short_episode() -> None:
    torch = pytest.importorskip("torch")

    from games.snake_gpu_tensor import TensorSnakeGPU

    game = TensorSnakeGPU(width=20, height=12, max_length=64)

    # Keep stepping until death or limit, validating tensor-backed counters.
    for _ in range(30):
        if not game.step():
            break

    stats = game.stats()
    assert stats.steps >= 1
    assert stats.score >= 0
    assert stats.device in {"cpu", "mps", "cuda"}
    assert torch.is_tensor(game.snake)
