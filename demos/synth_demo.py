#!/usr/bin/env python3
"""
synth_demo.py — Interactive program synthesis demo.

Three modes:
  1. Natural language: "write a function that computes factorial"
  2. I/O examples: "f(3)=6, f(5)=120, f(0)=1"
  3. Binary reverse-engineering: point at a compiled binary

Usage:
  python3 demos/synth_demo.py                  # interactive REPL
  python3 demos/synth_demo.py --binary /path   # reverse-engineer a binary
  python3 demos/synth_demo.py --batch file.txt # batch mode
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

MOG_SYNTH = str(ROOT / "mog_synth" / "target" / "release" / "mog_synth")


def synth_from_examples(fn_name: str, signature: str, examples: list[dict], timeout: int = 60) -> dict:
    """Call mog_synth --problem-json."""
    problem = {"name": fn_name, "signature": signature, "examples": examples, "holdouts": []}
    try:
        r = subprocess.run(
            [MOG_SYNTH, "--problem-json", "-"],
            input=json.dumps(problem), capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0:
            return json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "method": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e), "method": "error"}
    return {"success": False, "error": "synthesis failed", "method": "none"}


def parse_io_examples(text: str) -> tuple[str, str, list[dict]] | None:
    """Parse I/O examples from natural text.

    Formats:
      f(3) = 6, f(5) = 120
      f(1,2) = 3; f(3,4) = 7
      Input: 3 -> Output: 6 | Input: 5 -> Output: 120
    """
    # Try: f(args) = result pattern
    pattern = r'(\w+)\(([^)]+)\)\s*=\s*(-?\d+)'
    matches = re.findall(pattern, text)
    if matches:
        fn_name = matches[0][0]
        examples = []
        for _, args_str, result in matches:
            args = [int(x.strip()) for x in args_str.split(",")]
            examples.append({"inputs": args, "expected": int(result)})
        n_args = len(examples[0]["inputs"])
        arg_names = ["a", "b", "c", "d", "e", "f"][:n_args]
        sig = f"fn {fn_name}({', '.join(f'{n}: i64' for n in arg_names)}) -> i64"
        return fn_name, sig, examples

    # Try: input -> output pattern
    pattern2 = r'(\d+(?:\s*,\s*\d+)*)\s*(?:->|→|=>)\s*(-?\d+)'
    matches2 = re.findall(pattern2, text)
    if matches2:
        examples = []
        for args_str, result in matches2:
            args = [int(x.strip()) for x in args_str.split(",")]
            examples.append({"inputs": args, "expected": int(result)})
        n_args = len(examples[0]["inputs"])
        arg_names = ["a", "b", "c", "d", "e", "f"][:n_args]
        sig = f"fn f({', '.join(f'{n}: i64' for n in arg_names)}) -> i64"
        return "f", sig, examples

    return None


def generate_examples_from_description(description: str) -> tuple[str, str, list[dict]] | None:
    """Generate I/O examples from a natural language description.

    Uses pattern matching for common function descriptions.
    """
    desc = description.lower().strip()

    patterns = {
        r"factorial|n!": ("factorial", "fn factorial(n: i64) -> i64",
            [{"inputs": [0], "expected": 1}, {"inputs": [1], "expected": 1},
             {"inputs": [3], "expected": 6}, {"inputs": [5], "expected": 120},
             {"inputs": [7], "expected": 5040}, {"inputs": [10], "expected": 3628800}]),

        r"fibonacci|fib": ("fibonacci", "fn fibonacci(n: i64) -> i64",
            [{"inputs": [0], "expected": 0}, {"inputs": [1], "expected": 1},
             {"inputs": [2], "expected": 1}, {"inputs": [5], "expected": 5},
             {"inputs": [10], "expected": 55}, {"inputs": [15], "expected": 610}]),

        r"gcd|greatest common": ("gcd", "fn gcd(a: i64, b: i64) -> i64",
            [{"inputs": [12, 8], "expected": 4}, {"inputs": [7, 3], "expected": 1},
             {"inputs": [100, 75], "expected": 25}, {"inputs": [48, 18], "expected": 6},
             {"inputs": [1, 1], "expected": 1}, {"inputs": [0, 5], "expected": 5}]),

        r"prime|is.?prime|primality": ("is_prime", "fn is_prime(n: i64) -> i64",
            [{"inputs": [1], "expected": 0}, {"inputs": [2], "expected": 1},
             {"inputs": [3], "expected": 1}, {"inputs": [4], "expected": 0},
             {"inputs": [17], "expected": 1}, {"inputs": [25], "expected": 0},
             {"inputs": [97], "expected": 1}]),

        r"collatz|3n\+1|hailstone": ("collatz_steps", "fn collatz_steps(n: i64) -> i64",
            [{"inputs": [1], "expected": 0}, {"inputs": [2], "expected": 1},
             {"inputs": [3], "expected": 7}, {"inputs": [6], "expected": 8},
             {"inputs": [27], "expected": 111}]),

        r"abs.*diff|absolute.*diff|diff.*abs": ("abs_diff", "fn abs_diff(a: i64, b: i64) -> i64",
            [{"inputs": [5, 3], "expected": 2}, {"inputs": [3, 5], "expected": 2},
             {"inputs": [0, 0], "expected": 0}, {"inputs": [-2, 3], "expected": 5}]),

        r"square|x\^2|x\*x": ("square", "fn square(n: i64) -> i64",
            [{"inputs": [0], "expected": 0}, {"inputs": [3], "expected": 9},
             {"inputs": [-4], "expected": 16}, {"inputs": [10], "expected": 100}]),

        r"cube|x\^3": ("cube", "fn cube(n: i64) -> i64",
            [{"inputs": [0], "expected": 0}, {"inputs": [2], "expected": 8},
             {"inputs": [3], "expected": 27}, {"inputs": [-2], "expected": -8}]),

        r"double|2\*?x|twice": ("double", "fn double(n: i64) -> i64",
            [{"inputs": [0], "expected": 0}, {"inputs": [5], "expected": 10},
             {"inputs": [-3], "expected": -6}, {"inputs": [100], "expected": 200}]),

        r"triple|3\*?x|thrice": ("triple", "fn triple(n: i64) -> i64",
            [{"inputs": [0], "expected": 0}, {"inputs": [5], "expected": 15},
             {"inputs": [-3], "expected": -9}, {"inputs": [100], "expected": 300}]),

        r"max.*two|max\(a.*b\)|maximum": ("max2", "fn max2(a: i64, b: i64) -> i64",
            [{"inputs": [3, 5], "expected": 5}, {"inputs": [5, 3], "expected": 5},
             {"inputs": [0, 0], "expected": 0}, {"inputs": [-1, -3], "expected": -1}]),

        r"min.*two|min\(a.*b\)|minimum": ("min2", "fn min2(a: i64, b: i64) -> i64",
            [{"inputs": [3, 5], "expected": 3}, {"inputs": [5, 3], "expected": 3},
             {"inputs": [0, 0], "expected": 0}, {"inputs": [-1, -3], "expected": -3}]),

        r"sum.*to.*n|1\+2\+.*\+n|triangular": ("sum_to_n", "fn sum_to_n(n: i64) -> i64",
            [{"inputs": [0], "expected": 0}, {"inputs": [1], "expected": 1},
             {"inputs": [5], "expected": 15}, {"inputs": [10], "expected": 55},
             {"inputs": [100], "expected": 5050}]),

        r"popcount|count.*bits|hamming.*weight": ("popcount", "fn popcount(n: i64) -> i64",
            [{"inputs": [0], "expected": 0}, {"inputs": [1], "expected": 1},
             {"inputs": [7], "expected": 3}, {"inputs": [255], "expected": 8},
             {"inputs": [1023], "expected": 10}]),

        r"power|exponent|a\^b|a\*\*b": ("power", "fn power(a: i64, b: i64) -> i64",
            [{"inputs": [2, 0], "expected": 1}, {"inputs": [2, 10], "expected": 1024},
             {"inputs": [3, 3], "expected": 27}, {"inputs": [5, 2], "expected": 25}]),

        r"digit.*sum|sum.*digits": ("digit_sum", "fn digit_sum(n: i64) -> i64",
            [{"inputs": [0], "expected": 0}, {"inputs": [123], "expected": 6},
             {"inputs": [999], "expected": 27}, {"inputs": [1000], "expected": 1}]),

        r"reverse.*digits|digit.*reverse": ("reverse_digits", "fn reverse_digits(n: i64) -> i64",
            [{"inputs": [123], "expected": 321}, {"inputs": [100], "expected": 1},
             {"inputs": [9], "expected": 9}, {"inputs": [1234], "expected": 4321}]),

        r"clamp|bound|limit.*range": ("clamp", "fn clamp(v: i64, lo: i64, hi: i64) -> i64",
            [{"inputs": [5, 0, 10], "expected": 5}, {"inputs": [-5, 0, 10], "expected": 0},
             {"inputs": [15, 0, 10], "expected": 10}, {"inputs": [3, 3, 3], "expected": 3}]),

        r"manhattan|city.*block.*dist": ("manhattan", "fn manhattan(x1: i64, y1: i64, x2: i64, y2: i64) -> i64",
            [{"inputs": [0,0,3,4], "expected": 7}, {"inputs": [1,1,1,1], "expected": 0},
             {"inputs": [5,0,0,5], "expected": 10}]),

        r"add|sum|plus|a\+b": ("add", "fn add(a: i64, b: i64) -> i64",
            [{"inputs": [1, 2], "expected": 3}, {"inputs": [0, 0], "expected": 0},
             {"inputs": [-5, 5], "expected": 0}, {"inputs": [100, 200], "expected": 300}]),

        r"multiply|product|a\*b|times": ("multiply", "fn multiply(a: i64, b: i64) -> i64",
            [{"inputs": [3, 4], "expected": 12}, {"inputs": [0, 5], "expected": 0},
             {"inputs": [-2, 3], "expected": -6}, {"inputs": [7, 7], "expected": 49}]),

        r"sign|signum": ("sign", "fn sign(n: i64) -> i64",
            [{"inputs": [5], "expected": 1}, {"inputs": [-3], "expected": -1},
             {"inputs": [0], "expected": 0}, {"inputs": [100], "expected": 1}]),

        r"even|is.?even|parity": ("is_even", "fn is_even(n: i64) -> i64",
            [{"inputs": [0], "expected": 1}, {"inputs": [1], "expected": 0},
             {"inputs": [2], "expected": 1}, {"inputs": [7], "expected": 0},
             {"inputs": [-4], "expected": 1}]),
    }

    for pattern, (fn_name, sig, examples) in patterns.items():
        if re.search(pattern, desc):
            return fn_name, sig, examples

    return None


def reverse_engineer_binary(binary_path: str, fn_name: str, arg_types: list[str]) -> dict | None:
    """Extract I/O from a binary and synthesize."""
    from egdc.binary_io_extract import IOExtractor
    ext = IOExtractor.from_binary(binary_path, arg_types, fn_name=fn_name)
    problem = ext.extract(n_examples=12, n_holdouts=4, seed=42)
    examples = [{"inputs": ex.inputs, "expected": ex.expected} for ex in problem.examples]
    return synth_from_examples(fn_name, problem.signature, examples)


def interactive_repl():
    """Interactive REPL for program synthesis."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           NEURAL PROGRAM SYNTHESIS ENGINE                   ║")
    print("║                                                             ║")
    print("║  Describe a function in natural language, provide I/O       ║")
    print("║  examples, or point at a binary to reverse-engineer.        ║")
    print("║                                                             ║")
    print("║  Examples:                                                  ║")
    print("║    > write a factorial function                             ║")
    print("║    > f(3)=6, f(5)=120, f(0)=1                             ║")
    print("║    > reverse /tmp/my_binary my_func int int                ║")
    print("║    > help                                                   ║")
    print("║    > quit                                                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            user_input = input("synth> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if user_input.lower() == "help":
            print("""
  Commands:
    <description>          Natural language: "write a fibonacci function"
    f(1,2)=3, f(3,4)=7    I/O examples with function calls
    1,2 -> 3 | 3,4 -> 7   I/O examples with arrow notation
    reverse <binary> <fn> <arg_types...>   Reverse-engineer a binary
    help                   Show this help
    quit                   Exit
            """)
            continue

        # Try: reverse-engineer binary
        if user_input.lower().startswith("reverse "):
            parts = user_input.split()
            if len(parts) < 4:
                print("  Usage: reverse <binary_path> <function_name> <arg_type1> [arg_type2] ...")
                continue
            binary_path = parts[1]
            fn_name = parts[2]
            arg_types = parts[3:]
            print(f"  Reverse-engineering {fn_name} from {binary_path}...")
            t0 = time.time()
            result = reverse_engineer_binary(binary_path, fn_name, arg_types)
            elapsed = time.time() - t0
            if result and result.get("success"):
                print(f"  SOLVED in {elapsed:.1f}s via {result['method']}:\n")
                print(result["code"])
            else:
                print(f"  Failed ({elapsed:.1f}s): {result.get('error', 'unknown')}")
            continue

        # Try: parse I/O examples
        parsed = parse_io_examples(user_input)
        if parsed:
            fn_name, sig, examples = parsed
            print(f"  Synthesizing {fn_name} from {len(examples)} I/O examples...")
            t0 = time.time()
            result = synth_from_examples(fn_name, sig, examples)
            elapsed = time.time() - t0
            if result.get("success"):
                print(f"  SOLVED in {elapsed:.1f}s via {result['method']}:\n")
                print(result["code"])
            else:
                print(f"  Failed ({elapsed:.1f}s): {result.get('error', 'unknown')}")
            continue

        # Try: natural language description
        generated = generate_examples_from_description(user_input)
        if generated:
            fn_name, sig, examples = generated
            print(f"  Recognized: {fn_name}")
            print(f"  Generating from {len(examples)} I/O examples...")
            t0 = time.time()
            result = synth_from_examples(fn_name, sig, examples)
            elapsed = time.time() - t0
            if result.get("success"):
                print(f"  SOLVED in {elapsed:.1f}s via {result['method']}:\n")
                print(result["code"])
            else:
                print(f"  Failed ({elapsed:.1f}s): {result.get('error', 'unknown')}")
            continue

        # Nothing matched
        print("  I don't understand that. Try:")
        print('    "write a factorial function"')
        print('    "f(3)=6, f(5)=120, f(0)=1"')
        print('    "reverse /tmp/binary func_name int int"')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive program synthesis demo")
    parser.add_argument("--binary", help="Reverse-engineer a binary")
    parser.add_argument("--fn-name", default="program", help="Function name for binary mode")
    parser.add_argument("--arg-types", nargs="+", default=["int"], help="Arg types for binary")
    parser.add_argument("--batch", help="Batch mode: file with one query per line")
    args = parser.parse_args()

    if args.binary:
        result = reverse_engineer_binary(args.binary, args.fn_name, args.arg_types)
        if result and result.get("success"):
            print(result["code"])
        else:
            print(f"Failed: {result}")
    elif args.batch:
        with open(args.batch) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    print(f"\n> {line}")
                    generated = generate_examples_from_description(line)
                    if generated:
                        fn_name, sig, examples = generated
                        result = synth_from_examples(fn_name, sig, examples)
                        if result.get("success"):
                            print(result["code"])
                        else:
                            print(f"  Failed: {result.get('error')}")
    else:
        interactive_repl()
