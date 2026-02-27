#!/usr/bin/env python3
"""
LLM Benchmark for OUROBOROS
============================
Tests different Ollama models for speed and quality.
"""

import subprocess
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class BenchmarkResult:
    model: str
    prompt_type: str
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    response_quality: str  # "good", "partial", "failed"
    response_preview: str


class LLMBenchmark:
    """Benchmark Ollama models for OUROBOROS use case."""

    # Test prompts representative of OUROBOROS usage
    TEST_PROMPTS = {
        "code_generation": """You are an AI agent solving a coding problem.

PROBLEM: Write a Python function called 'find_peaks' that finds all local peaks in a list of integers.

Your response format:
REASONING: [your thought process]
SOLUTION: [your code]

Be concise.""",

        "analysis": """You are the Meta-Narrator observing an agent swarm.

SWARM STATE:
- 3 agents active
- Average fitness: 0.45
- Best agent: agent_2 at 0.72
- Alert: Fitness plateau detected

Provide brief strategic guidance (2-3 sentences) to help agents improve.""",

        "evaluation": """Evaluate this code solution for correctness:

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```

Rate: GOOD, PARTIAL, or BAD. Explain in one sentence.""",
    }

    def __init__(self, models: List[str] = None):
        self.models = models or ["qwen3:8b", "deepseek-r1:8b", "llama3.1:8b"]
        self.results: List[BenchmarkResult] = []

    def check_available_models(self) -> List[str]:
        """Check which models are available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            available = []
            for line in result.stdout.split("\n")[1:]:
                if line.strip():
                    model_name = line.split()[0]
                    available.append(model_name)
            return available
        except Exception as e:
            print(f"Error checking models: {e}")
            return []

    def run_single_benchmark(
        self,
        model: str,
        prompt_type: str,
        prompt: str,
        timeout: int = 120
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        print(f"  Testing {model} on {prompt_type}...", end=" ", flush=True)

        start_time = time.time()
        try:
            result = subprocess.run(
                ["ollama", "run", model, "--nowordwrap", prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            latency = (time.time() - start_time) * 1000

            response = result.stdout.strip()
            tokens = len(response.split())
            tps = tokens / (latency / 1000) if latency > 0 else 0

            # Assess quality
            quality = "failed"
            if result.returncode == 0 and response:
                if prompt_type == "code_generation":
                    if "def " in response or "SOLUTION" in response:
                        quality = "good"
                    elif len(response) > 50:
                        quality = "partial"
                elif prompt_type == "analysis":
                    if len(response) > 20:
                        quality = "good"
                elif prompt_type == "evaluation":
                    if any(x in response.upper() for x in ["GOOD", "BAD", "PARTIAL"]):
                        quality = "good"
                    elif len(response) > 10:
                        quality = "partial"

            print(f"{latency:.0f}ms, {tps:.1f} tok/s, {quality}")

            return BenchmarkResult(
                model=model,
                prompt_type=prompt_type,
                latency_ms=latency,
                tokens_generated=tokens,
                tokens_per_second=tps,
                response_quality=quality,
                response_preview=response[:200]
            )

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT ({timeout}s)")
            return BenchmarkResult(
                model=model,
                prompt_type=prompt_type,
                latency_ms=timeout * 1000,
                tokens_generated=0,
                tokens_per_second=0,
                response_quality="failed",
                response_preview="[TIMEOUT]"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            return BenchmarkResult(
                model=model,
                prompt_type=prompt_type,
                latency_ms=0,
                tokens_generated=0,
                tokens_per_second=0,
                response_quality="failed",
                response_preview=f"[ERROR: {e}]"
            )

    def run_all_benchmarks(self) -> Dict:
        """Run all benchmarks."""
        print("=" * 70)
        print("       OUROBOROS LLM BENCHMARK")
        print("=" * 70)

        # Check available models
        available = self.check_available_models()
        print(f"\nAvailable models: {available}")

        models_to_test = [m for m in self.models if any(m in a for a in available)]
        print(f"Testing models: {models_to_test}")

        if not models_to_test:
            print("No matching models found!")
            return {}

        print("\n" + "-" * 70)

        # Run benchmarks
        for model in models_to_test:
            print(f"\n[{model}]")
            for prompt_type, prompt in self.TEST_PROMPTS.items():
                result = self.run_single_benchmark(model, prompt_type, prompt)
                self.results.append(result)

        # Generate summary
        return self.generate_summary()

    def generate_summary(self) -> Dict:
        """Generate benchmark summary."""
        print("\n" + "=" * 70)
        print("       BENCHMARK RESULTS")
        print("=" * 70)

        # Group by model
        by_model = {}
        for r in self.results:
            if r.model not in by_model:
                by_model[r.model] = []
            by_model[r.model].append(r)

        summary = {}
        for model, results in by_model.items():
            avg_latency = sum(r.latency_ms for r in results) / len(results)
            avg_tps = sum(r.tokens_per_second for r in results) / len(results)
            good_count = sum(1 for r in results if r.response_quality == "good")
            total = len(results)

            summary[model] = {
                "avg_latency_ms": avg_latency,
                "avg_tokens_per_second": avg_tps,
                "quality_score": good_count / total,
                "tests_passed": good_count,
                "tests_total": total,
            }

            print(f"\n{model}:")
            print(f"  Avg latency: {avg_latency:.0f}ms")
            print(f"  Avg speed: {avg_tps:.1f} tokens/sec")
            print(f"  Quality: {good_count}/{total} tests passed")

        # Recommendation
        print("\n" + "-" * 70)
        print("RECOMMENDATION:")

        if summary:
            # Score: lower latency + higher quality = better
            scored = []
            for model, stats in summary.items():
                # Normalize: latency penalty, quality bonus
                score = stats["quality_score"] * 100 - stats["avg_latency_ms"] / 1000
                scored.append((model, score, stats))

            scored.sort(key=lambda x: x[1], reverse=True)
            best_model, best_score, best_stats = scored[0]

            print(f"  Best model: {best_model}")
            print(f"  Avg latency: {best_stats['avg_latency_ms']:.0f}ms")
            print(f"  Quality: {best_stats['quality_score']*100:.0f}%")

            summary["recommendation"] = best_model

        print("=" * 70)
        return summary


def main():
    benchmark = LLMBenchmark()
    results = benchmark.run_all_benchmarks()

    # Save results
    output_path = "/tmp/ouroboros_llm_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
