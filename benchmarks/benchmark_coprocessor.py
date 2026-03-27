#!/usr/bin/env python3
"""Benchmark: nCPU coprocessor vs vanilla transformer on arithmetic.

Compares:
  1. Vanilla model (no coprocessor)
  2. Model with coprocessor injected at last layer
  3. Model with coprocessor injected at multiple layers

Metrics:
  - Arithmetic accuracy (ADD, SUB, MUL, AND, OR, XOR)
  - Inference latency per token
  - Router gate activation rate
  - Parameter overhead

Usage:
    # Quick benchmark with tiny model (no GPU needed):
    python benchmarks/benchmark_coprocessor.py --synthetic

    # Full benchmark with Qwen:
    python benchmarks/benchmark_coprocessor.py \
        --model Qwen/Qwen2.5-0.5B \
        --num-samples 500 \
        --output benchmark_results/coprocessor_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncpu.coprocessor.config import NCPUCoprocessorConfig
from ncpu.coprocessor.inject import (
    inject_ncpu_coprocessor,
    collect_aux_losses,
    freeze_backbone,
    get_coprocessor_params,
)
from ncpu.coprocessor.coprocessor_layer import NCPUCoprocessorMLP
from ncpu.coprocessor.data import ArithmeticDataset

logger = logging.getLogger(__name__)


def benchmark_synthetic(
    num_samples: int = 500,
    max_value: int = 99,
    layers_configs: Optional[list] = None,
) -> dict:
    """Benchmark with tiny synthetic model (no HF download needed).

    Tests the coprocessor injection overhead and gate behavior.
    """
    from ncpu.coprocessor.train import TinyTransformer

    hidden_dim = 128
    vocab_size = 1000
    results = {}

    if layers_configs is None:
        layers_configs = [
            ("vanilla", []),
            ("last_layer", [-1]),
            ("two_layers", [-1, -2]),
            ("all_layers", [0, 1, 2, 3]),
        ]

    for name, layer_indices in layers_configs:
        model = TinyTransformer(vocab_size=vocab_size, hidden_dim=hidden_dim, n_layers=4)
        total_params = sum(p.numel() for p in model.parameters())

        copro_params = 0
        if layer_indices:
            config = NCPUCoprocessorConfig(layer_indices=layer_indices)
            injected = inject_ncpu_coprocessor(model, config)
            copro_params = sum(
                p.numel()
                for m in model.modules()
                if isinstance(m, NCPUCoprocessorMLP)
                for p in m.parameters()
            )

        model.eval()
        batch_size = 16
        seq_len = 16

        # Warmup
        for _ in range(5):
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            with torch.no_grad():
                model(input_ids=x)

        # Benchmark forward pass latency
        latencies = []
        gate_values = []

        for _ in range(num_samples // batch_size):
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            start = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=x)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

            # Collect gate stats
            for m in model.modules():
                if isinstance(m, NCPUCoprocessorMLP) and m._aux_loss is not None:
                    gate_values.append(m.router.gate_proj.bias.sigmoid().item())

        mean_latency = sum(latencies) / len(latencies) if latencies else 0
        p50_latency = sorted(latencies)[len(latencies)//2] if latencies else 0
        p99_latency = sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0
        mean_gate = sum(gate_values) / len(gate_values) if gate_values else 0

        results[name] = {
            "total_params": total_params + copro_params,
            "copro_params": copro_params,
            "overhead_pct": 100 * copro_params / total_params if total_params > 0 else 0,
            "mean_latency_ms": round(mean_latency, 3),
            "p50_latency_ms": round(p50_latency, 3),
            "p99_latency_ms": round(p99_latency, 3),
            "mean_gate": round(mean_gate, 4),
            "layers_injected": len(layer_indices),
        }

    return results


def benchmark_real_model(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    num_samples: int = 200,
    max_value: int = 99,
) -> dict:
    """Benchmark with real transformer model.

    Compares vanilla vs coprocessor on actual arithmetic generation.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ncpu.coprocessor.train import evaluate_arithmetic_accuracy

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    # Load model
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # --- Vanilla baseline ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} params")

    logger.info("Evaluating vanilla model...")
    start = time.time()
    vanilla_eval = evaluate_arithmetic_accuracy(
        model, tokenizer, num_samples=num_samples, max_value=max_value, device=device
    )
    vanilla_time = time.time() - start

    results["vanilla"] = {
        **vanilla_eval,
        "total_params": total_params,
        "copro_params": 0,
        "overhead_pct": 0,
        "eval_time_s": round(vanilla_time, 1),
    }
    logger.info(f"Vanilla accuracy: {vanilla_eval['overall_accuracy']:.1%}")

    # --- Single layer coprocessor ---
    config_single = NCPUCoprocessorConfig(layer_indices=[-1])
    injected_single = inject_ncpu_coprocessor(model, config_single)
    copro_params_single = sum(p.numel() for p in get_coprocessor_params(model))

    logger.info("Evaluating single-layer coprocessor (untrained)...")
    start = time.time()
    single_eval = evaluate_arithmetic_accuracy(
        model, tokenizer, num_samples=num_samples, max_value=max_value, device=device
    )
    single_time = time.time() - start

    results["single_layer_untrained"] = {
        **single_eval,
        "total_params": total_params + copro_params_single,
        "copro_params": copro_params_single,
        "overhead_pct": round(100 * copro_params_single / total_params, 4),
        "eval_time_s": round(single_time, 1),
    }
    logger.info(f"Single-layer accuracy: {single_eval['overall_accuracy']:.1%}")

    # --- Multi-layer coprocessor ---
    # Reload model fresh
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to(device)

    n_layers = len(model.model.layers) if hasattr(model, "model") else 4
    multi_indices = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    config_multi = NCPUCoprocessorConfig(layer_indices=multi_indices)
    injected_multi = inject_ncpu_coprocessor(model, config_multi)
    copro_params_multi = sum(p.numel() for p in get_coprocessor_params(model))

    logger.info(f"Evaluating multi-layer coprocessor at layers {multi_indices}...")
    start = time.time()
    multi_eval = evaluate_arithmetic_accuracy(
        model, tokenizer, num_samples=num_samples, max_value=max_value, device=device
    )
    multi_time = time.time() - start

    results["multi_layer_untrained"] = {
        **multi_eval,
        "total_params": total_params + copro_params_multi,
        "copro_params": copro_params_multi,
        "overhead_pct": round(100 * copro_params_multi / total_params, 4),
        "eval_time_s": round(multi_time, 1),
        "layers": multi_indices,
    }
    logger.info(f"Multi-layer accuracy: {multi_eval['overall_accuracy']:.1%}")

    return results


def print_results(results: dict, title: str = "Benchmark Results") -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

    for name, metrics in results.items():
        print(f"\n--- {name} ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark nCPU coprocessor")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-value", type=int, default=99)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use tiny synthetic model (no HF download)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.synthetic:
        results = benchmark_synthetic(num_samples=args.num_samples)
        print_results(results, "Synthetic Benchmark (Tiny Model)")
    else:
        results = benchmark_real_model(
            model_name=args.model,
            num_samples=args.num_samples,
            max_value=args.max_value,
        )
        print_results(results, f"Coprocessor Benchmark ({args.model})")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
