#!/usr/bin/env python3
"""Quick benchmark for specific models."""

import subprocess
import time
import json

MODELS = ["devstral", "mistral-small", "llama3.1:8b"]

PROMPT = """You are an AI agent solving a coding problem.

PROBLEM: Write a Python function called 'is_prime' that returns True if n is prime.

Your response format:
SOLUTION: [your code]

Be concise. Output only the function."""

def benchmark_model(model: str) -> dict:
    print(f"\n[{model}]")

    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, "--nowordwrap", PROMPT],
            capture_output=True,
            text=True,
            timeout=90
        )
        latency = time.time() - start
        response = result.stdout.strip()
        tokens = len(response.split())

        # Check quality
        quality = "good" if "def " in response and "return" in response else "partial"

        print(f"  Latency: {latency:.1f}s")
        print(f"  Tokens: {tokens}")
        print(f"  Quality: {quality}")
        print(f"  Preview: {response[:100]}...")

        return {
            "model": model,
            "latency_s": latency,
            "tokens": tokens,
            "quality": quality,
        }
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (90s)")
        return {"model": model, "latency_s": 90, "tokens": 0, "quality": "timeout"}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"model": model, "latency_s": 0, "tokens": 0, "quality": "error"}

def main():
    print("=" * 60)
    print("       QUICK LLM BENCHMARK")
    print("=" * 60)

    results = []
    for model in MODELS:
        results.append(benchmark_model(model))

    print("\n" + "=" * 60)
    print("       RESULTS")
    print("=" * 60)

    # Sort by latency (lower is better), excluding failures
    valid = [r for r in results if r["quality"] not in ["timeout", "error"]]
    valid.sort(key=lambda x: x["latency_s"])

    for r in valid:
        print(f"\n{r['model']}:")
        print(f"  Latency: {r['latency_s']:.1f}s")
        print(f"  Quality: {r['quality']}")

    if valid:
        best = valid[0]
        print(f"\n{'=' * 60}")
        print(f"BEST MODEL: {best['model']} ({best['latency_s']:.1f}s)")
        print(f"{'=' * 60}")

        # Save result
        with open("/tmp/best_model.txt", "w") as f:
            f.write(best["model"])

        return best["model"]

    return "llama3.1:8b"  # fallback

if __name__ == "__main__":
    best = main()
    print(f"\nBest model saved to /tmp/best_model.txt: {best}")
