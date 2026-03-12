#!/usr/bin/env python3
"""
SOME LLM Benchmark Runner

Run this script to benchmark LLM code generation with and without SOME.

Usage:
    # With OpenAI
    export OPENAI_API_KEY="your-key"
    python run_benchmark.py --provider openai --model gpt-4

    # With Anthropic
    export ANTHROPIC_API_KEY="your-key"
    python run_benchmark.py --provider anthropic --model claude-3-opus-20240229

    # With local model
    python run_benchmark.py --provider local --model llama-2-7b --base-url http://localhost:8080/v1
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncpu.self_optimizing.llm_benchmark import BenchmarkTask, LLMBenchmark
from ncpu.self_optimizing.llm_provider import LLMProviderFactory
from ncpu.self_optimizing.code_verifier import (
    CodeVerifier,
    FIBONACCI_TESTS,
    FACTORIAL_TESTS,
    PALINDROME_TESTS,
    REVERSE_LIST_TESTS,
    BINARY_SEARCH_TESTS,
    CODE_PROMPTS,
)


TASK_TEST_CASES = {
    "fibonacci": FIBONACCI_TESTS,
    "factorial": FACTORIAL_TESTS,
    "palindrome": PALINDROME_TESTS,
    "reverse_list": REVERSE_LIST_TESTS,
    "binary_search": BINARY_SEARCH_TESTS,
}


def main():
    parser = argparse.ArgumentParser(description="SOME LLM Benchmark")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "anthropic", "local", "glm", "minimax"],
                        help="LLM provider")
    parser.add_argument("--model", default="gpt-4",
                        help="Model name")
    parser.add_argument("--api-key",
                        help="API key (or set via env var)")
    parser.add_argument("--base-url",
                        help="Base URL for local provider")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retry attempts")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of test cases")
    parser.add_argument("--output",
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key
    if args.provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not set")
            print("Run: export OPENAI_API_KEY='your-key'")
            sys.exit(1)
    elif args.provider == "anthropic":
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set")
            print("Run: export ANTHROPIC_API_KEY='your-key'")
            sys.exit(1)

    print("=" * 60)
    print("SOME LLM BENCHMARK")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Max retries: {args.max_retries}")
    print()

    # Create LLM provider
    try:
        llm_provider = LLMProviderFactory.create_provider(
            provider=args.provider,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
        )
        print(f"✓ Connected to {args.provider}")
    except Exception as e:
        print(f"Error connecting to {args.provider}: {e}")
        sys.exit(1)

    # Create verifier
    verifier = CodeVerifier()

    # Define benchmark tasks with actual verification hooks
    tasks = []
    for name, prompt in list(CODE_PROMPTS.items())[:args.num_samples]:
        test_cases = TASK_TEST_CASES.get(name)
        tasks.append(
            BenchmarkTask(
                name=name,
                prompt=prompt,
                verify_fn=lambda code, cases=test_cases, local_verifier=verifier: local_verifier.verify(
                    code,
                    cases,
                ),
            )
        )

    # Create benchmark
    benchmark = LLMBenchmark(
        llm_provider=llm_provider,
    )

    # Run benchmark
    print(f"\nRunning benchmark with {len(tasks)} prompts...")
    print("-" * 60)

    start_time = time.time()
    results = benchmark.run_comparison(tasks, max_retries=args.max_retries)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Standard success: {results['standard'].success_rate:.1%}")
    print(f"SOME success:      {results['some'].success_rate:.1%}")
    print(f"Improvement:       {(results['some'].success_rate - results['standard'].success_rate):.1%}")
    print()

    # Save results
    if args.output:
        import json
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "provider": args.provider,
            "model": args.model,
            "elapsed_seconds": elapsed,
            "standard": {
                "success_rate": results['standard'].success_rate,
                "num_samples": results['standard'].num_samples,
                "errors": results['standard'].errors,
                "avg_execution_time": results['standard'].avg_execution_time,
            },
            "some": {
                "success_rate": results['some'].success_rate,
                "num_samples": results['some'].num_samples,
                "avg_attempts": results['some'].avg_attempts,
                "avg_execution_time": results['some'].avg_execution_time,
                "errors": results['some'].errors,
            },
            "improvement": results['improvement'],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
