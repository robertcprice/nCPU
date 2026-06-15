#!/usr/bin/env python3
"""
Stage 4 End-to-End Synthesis Verification

Verifies time-parameterized benchmarks (fibonacci, factorial, triangular, linear, poly)
by extracting problem definitions, calling mog_synth, verifying output, and reporting
solve times and methods used.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class SynthesisResult:
    problem_name: str
    variant: int
    success: bool
    solve_time: float
    method: str
    code: Optional[str]
    error: Optional[str]

def get_benchmark_list() -> list[dict]:
    """Extract benchmark list from Rust benchmark module."""
    # Run the binary to extract benchmark metadata
    result = subprocess.run(
        ["./target/release/mog_synth", "--list-benchmarks"],
        capture_output=True,
        text=True,
        cwd="/Users/bobbyprice/projects/nCPU/nsynth"
    )

    if result.returncode != 0:
        # Fallback: manually define the representative benchmarks
        return [
            {"name": "fibonacci", "variant": 0},
            {"name": "factorial", "variant": 0},
            {"name": "triangular_number", "variant": 0},
            {"name": "linear_series", "variant": 0},
            {"name": "polynomial_eval", "variant": 0},
        ]

    try:
        return json.loads(result.stdout)
    except:
        return [
            {"name": "fibonacci", "variant": 0},
            {"name": "factorial", "variant": 0},
            {"name": "triangular_number", "variant": 0},
            {"name": "linear_series", "variant": 0},
            {"name": "polynomial_eval", "variant": 0},
        ]

def run_synthesis(problem_name: str, variant: int) -> SynthesisResult:
    """Run mog_synth on a single problem and return results."""
    problem_spec = {
        "name": problem_name,
        "variant": variant,
    }

    start = time.time()
    try:
        result = subprocess.run(
            ["./target/release/mog_synth",
             "--problem-name", problem_name,
             "--problem-variant", str(variant),
             "--json"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/Users/bobbyprice/projects/nCPU/nsynth"
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            return SynthesisResult(
                problem_name=problem_name,
                variant=variant,
                success=False,
                solve_time=elapsed,
                method="error",
                code=None,
                error=result.stderr or result.stdout
            )

        try:
            output = json.loads(result.stdout)
            return SynthesisResult(
                problem_name=problem_name,
                variant=variant,
                success=output.get("success", False),
                solve_time=output.get("solve_time", elapsed),
                method=output.get("method", "unknown"),
                code=output.get("code"),
                error=output.get("error")
            )
        except json.JSONDecodeError:
            return SynthesisResult(
                problem_name=problem_name,
                variant=variant,
                success=False,
                solve_time=elapsed,
                method="parse_error",
                code=None,
                error=f"Failed to parse JSON output: {result.stdout}"
            )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return SynthesisResult(
            problem_name=problem_name,
            variant=variant,
            success=False,
            solve_time=elapsed,
            method="timeout",
            code=None,
            error="Synthesis timed out after 300 seconds"
        )
    except Exception as e:
        elapsed = time.time() - start
        return SynthesisResult(
            problem_name=problem_name,
            variant=variant,
            success=False,
            solve_time=elapsed,
            method="exception",
            code=None,
            error=str(e)
        )

def verify_compiled_code(code: str, problem_name: str) -> bool:
    """Try to compile and run the synthesized code."""
    if not code:
        return False

    # Create a temporary test file
    test_file = Path(f"/tmp/test_{problem_name}.mog")
    try:
        test_file.write_text(code)

        # Try to run it with mog_synth interpreter
        result = subprocess.run(
            ["./target/release/mog_synth", "--run-file", str(test_file)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="/Users/bobbyprice/projects/nCPU/nsynth"
        )

        return result.returncode == 0
    except Exception as e:
        print(f"  Verification failed: {e}", file=sys.stderr)
        return False
    finally:
        try:
            test_file.unlink()
        except:
            pass

def main():
    print("=" * 70)
    print("Stage 4 Time-Parameterized Synthesis Verification")
    print("=" * 70)

    # Representative benchmarks to verify
    benchmarks = [
        ("fibonacci", 0),
        ("factorial", 0),
        ("triangular_number", 0),
        ("linear_series", 0),
        ("polynomial_eval", 0),
    ]

    results: list[SynthesisResult] = []
    method_counts: dict[str, int] = {}

    for problem_name, variant in benchmarks:
        print(f"\nSynthesizing {problem_name} (variant {variant})...")
        result = run_synthesis(problem_name, variant)
        results.append(result)

        # Update method distribution
        method = result.method if result.success else "failed"
        method_counts[method] = method_counts.get(method, 0) + 1

        # Print result
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"  {status} | Time: {result.solve_time:.2f}s | Method: {result.method}")

        if result.error:
            print(f"  Error: {result.error[:100]}")

        if result.code:
            print(f"  Code length: {len(result.code)} bytes")
            # Verify compilation
            if verify_compiled_code(result.code, problem_name):
                print(f"  ✓ Code compiles successfully")
            else:
                print(f"  ✗ Code compilation/execution failed")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.success)
    total = len(results)
    mean_time = sum(r.solve_time for r in results) / total if total > 0 else 0

    print(f"\nPass Rate: {passed}/{total} ({100*passed//total}%)")
    print(f"Mean Solve Time: {mean_time:.2f}s")
    print(f"\nMethod Distribution:")
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method:20s}: {count:3d} problems")

    print(f"\nDetailed Results:")
    for result in results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.problem_name:20s} | {result.solve_time:7.2f}s | {result.method:20s}")

    # Failures
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  - {f.problem_name}: {f.error[:80] if f.error else 'unknown error'}")

    # JSON output for automation
    output = {
        "pass_count": passed,
        "total": total,
        "mean_solve_time": mean_time,
        "method_distribution": method_counts,
        "results": [asdict(r) for r in results],
        "any_failures": len(failures) > 0,
    }

    output_file = Path("/Users/bobbyprice/projects/nCPU/nsynth/verify_stage4_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
