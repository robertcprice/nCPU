"""
Expanded Benchmark Suite for SOME

More comprehensive benchmarks testing various scenarios.
"""

import time
import json
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run"""
    name: str
    description: str
    num_runs: int
    max_retries: int
    timeout_seconds: int


@dataclass
class BenchmarkMetrics:
    """Metrics from a benchmark run"""
    name: str
    timestamp: str
    duration_seconds: float
    success_rate: float
    avg_attempts: float
    total_cost: Optional[float] = None
    errors: list = field(default_factory=list)


class BenchmarkSuite:
    """Comprehensive benchmark suite for SOME"""

    def __init__(self):
        self.results: list = []

    def run_all(self, provider_fn: Optional[Callable] = None) -> dict:
        """Run all benchmarks"""
        benchmarks = [
            self.benchmark_algorithm_problems,
            self.benchmark_error_recovery,
            self.benchmark_code_complexity,
            self.benchmark_token_efficiency,
        ]

        all_results = []

        for bench_fn in benchmarks:
            print(f"\n{'='*60}")
            print(f"Running: {bench_fn.__name__}")
            print(f"{'='*60}")

            result = bench_fn(provider_fn)
            all_results.append(result)

        return all_results

    def benchmark_algorithm_problems(
        self,
        provider_fn: Optional[Callable] = None,
    ) -> BenchmarkMetrics:
        """Benchmark on classic algorithm problems"""
        from ncpu.self_optimizing.llm_benchmark import LLMBenchmark
        from ncpu.self_optimizing.code_verifier import (
            CodeVerifier, FIBONACCI_TESTS, FACTORIAL_TESTS,
            PALINDROME_TESTS, BINARY_SEARCH_TESTS
        )

        prompts = [
            # Fibonacci
            "Write a Python function called 'fib' that returns the nth Fibonacci number recursively.",
            # Factorial
            "Write a Python function called 'factorial' that returns n factorial.",
            # Palindrome
            "Write a Python function called 'is_palindrome' that returns True if string is palindrome.",
            # Binary search
            "Write a Python function called 'binary_search' that finds index of target in sorted array.",
            # Reverse list (not in test cases but common)
            "Write a Python function called 'reverse_list' that reverses a list in place.",
            # Max subarray
            "Write a Python function called 'max_subarray' that finds the contiguous subarray with largest sum.",
            # Merge sort
            "Write a Python function called 'merge_sort' that sorts a list using merge sort.",
            # Quick sort
            "Write a Python function called 'quick_sort' that sorts a list using quick sort.",
            # Two sum
            "Write a Python function called 'two_sum' that finds indices of two numbers that add to target.",
            # Valid parentheses
            "Write a Python function called 'valid_parentheses' that checks if parentheses are balanced.",
        ]

        verifier = CodeVerifier()
        benchmark = LLMBenchmark(
            llm_provider=provider_fn,
            verify_fn=lambda code: (verifier.verify(code).success, None)
        )

        start = time.time()
        result = benchmark.run_comparison(prompts, max_retries=3)
        duration = time.time() - start

        return BenchmarkMetrics(
            name="algorithm_problems",
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            success_rate=result['some'].success_rate,
            avg_attempts=result['some'].avg_attempts,
        )

    def benchmark_error_recovery(
        self,
        provider_fn: Optional[Callable] = None,
    ) -> BenchmarkMetrics:
        """Benchmark error recovery capability"""
        from ncpu.self_optimizing.llm_benchmark import LLMBenchmark

        # Test prompts that commonly cause issues
        prompts = [
            "Write a Python function that divides two numbers.",
            "Write a Python function that accesses list index.",
            "Write a Python function that calls a non-existent function.",
            "Write a Python function with syntax error.",
            "Write a Python function with undefined variable.",
        ]

        # Mock LLM that sometimes generates bad code
        def mock_llm(prompt: str) -> str:
            import random
            if "divide" in prompt.lower():
                return random.choice([
                    "def div(a, b): return a / b",  # OK but might fail on 0
                    "def div(a, b): return a / (b - b)",  # Always fails
                ])
            elif "index" in prompt.lower():
                return random.choice([
                    "def get_item(lst, i): return lst[i]",
                    "def get_item(lst, i): return lst[i + 1]",  # Off by one
                ])
            elif "syntax" in prompt.lower():
                return random.choice([
                    "def test(): return 42",
                    "def test() return 42",  # Syntax error
                ])
            else:
                return "def test(): return 42"

        benchmark = LLMBenchmark(
            llm_provider=provider_fn or mock_llm,
        )

        start = time.time()
        result = benchmark.run_comparison(prompts, max_retries=3)
        duration = time.time() - start

        return BenchmarkMetrics(
            name="error_recovery",
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            success_rate=result['some'].success_rate,
            avg_attempts=result['some'].avg_attempts,
        )

    def benchmark_code_complexity(
        self,
        provider_fn: Optional[Callable] = None,
    ) -> BenchmarkMetrics:
        """Benchmark with varying code complexity"""
        from ncpu.self_optimizing.llm_benchmark import LLMBenchmark

        prompts = [
            # Simple (1-5 lines)
            "Return the sum of two numbers.",

            # Medium (5-20 lines)
            "Implement a class for a bank account with deposit, withdraw, and balance methods.",

            # Complex (20+ lines)
            "Implement a binary search tree with insert, delete, search, and traversal methods. Include handling for duplicates.",

            # Very complex
            "Implement a full decorator system that supports timing, caching, and retry logic with configurable parameters.",
        ]

        benchmark = LLMBenchmark(llm_provider=provider_fn)

        start = time.time()
        result = benchmark.run_comparison(prompts, max_retries=2)
        duration = time.time() - start

        return BenchmarkMetrics(
            name="code_complexity",
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            success_rate=result['some'].success_rate,
            avg_attempts=result['some'].avg_attempts,
        )

    def benchmark_token_efficiency(
        self,
        provider_fn: Optional[Callable] = None,
    ) -> BenchmarkMetrics:
        """Benchmark token efficiency with SOME"""
        from ncpu.self_optimizing.llm_benchmark import LLMBenchmark

        prompts = [
            f"Write function {i}: return the square of a number."
            for i in range(20)
        ]

        benchmark = LLMBenchmark(llm_provider=provider_fn)

        start = time.time()
        result = benchmark.run_comparison(prompts, max_retries=3)
        duration = time.time() - start

        # Estimate token usage
        # Standard: 1 prompt per task
        # SOME: 1 prompt + retries
        estimated_tokens = len(prompts) * 100  # Rough estimate
        estimated_with_retry = estimated_tokens * result['some'].avg_attempts

        return BenchmarkMetrics(
            name="token_efficiency",
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            success_rate=result['some'].success_rate,
            avg_attempts=result['some'].avg_attempts,
            total_cost=estimated_with_retry * 0.00001,  # Rough cost estimate
        )


def run_expanded_benchmark():
    """Run expanded benchmark suite"""
    print("=" * 60)
    print("EXPANDED SOME BENCHMARK SUITE")
    print("=" * 60)

    suite = BenchmarkSuite()
    results = suite.run_all()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for result in results:
        print(f"\n{result.name}:")
        print(f"  Success rate: {result.success_rate:.1%}")
        print(f"  Avg attempts: {result.avg_attempts:.1f}")
        print(f"  Duration: {result.duration_seconds:.1f}s")

    return results


if __name__ == "__main__":
    run_expanded_benchmark()
