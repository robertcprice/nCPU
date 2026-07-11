#!/usr/bin/env python3
"""Held-out generalization bench for Python repo repair (VibeThinker + a real test oracle).

The Python analog of repair_bench.rs: plant a realistic bug in a Python module, run the never-wrong
model repair against a SHOWN test, then re-check with HELD-OUT asserts the model never saw. Passing
the shown test is not correctness; held-out-green means the fix GENERALIZES.

Serve VibeThinker, then:
    NSYNTH_LOCAL_LLM_URL=http://127.0.0.1:8080/v1/chat/completions \
    NSYNTH_LOCAL_LLM_MODEL=mlx-community/VibeThinker-3B-4bit python3 py_repair_bench.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import py_repair  # noqa: E402

# (id, class, buggy sol.py, shown test body, held-out test body). Tests import `sol` and assert;
# they exit non-zero on failure (no pytest needed).
TASKS = [
    (
        "factorial_basecase",
        "recursion base case returns 0 instead of 1",
        "def factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)\n",
        "from sol import factorial\nassert factorial(0) == 1\nassert factorial(3) == 6\n",
        "from sol import factorial\nassert factorial(1) == 1\nassert factorial(5) == 120\nassert factorial(6) == 720\n",
    ),
    (
        "binary_search_boundary",
        "off-by-one: `lo < hi` misses the single-element window",
        "def bsearch(xs, t):\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == t:\n            return mid\n        if xs[mid] < t:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
        "from sol import bsearch\nassert bsearch([1, 3, 5, 7], 5) == 2\nassert bsearch([1, 3, 5, 7], 1) == 0\n",
        "from sol import bsearch\nassert bsearch([2, 4, 6], 6) == 2\nassert bsearch([9], 9) == 0\nassert bsearch([1, 2, 3], 4) == -1\n",
    ),
    (
        "kadane_reset",
        "Kadane missing the max(x, cur+x) reset",
        "def max_subarray(xs):\n    best = cur = xs[0]\n    for x in xs[1:]:\n        cur = cur + x\n        best = max(best, cur)\n    return best\n",
        "from sol import max_subarray\nassert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\nassert max_subarray([1, 2, 3]) == 6\n",
        "from sol import max_subarray\nassert max_subarray([-1, -2, -3]) == -1\nassert max_subarray([5, -9, 6]) == 6\nassert max_subarray([4]) == 4\n",
    ),
    (
        "title_case_rewrite",
        "returns upper() where per-word title case is meant (a real rewrite)",
        "def title_case(s):\n    return s.upper()\n",
        "from sol import title_case\nassert title_case('hello world') == 'Hello World'\nassert title_case('a b') == 'A B'\n",
        "from sol import title_case\nassert title_case('the QUICK') == 'The Quick'\nassert title_case('x') == 'X'\nassert title_case('one two three') == 'One Two Three'\n",
    ),
]


def run(repo, script):
    p = subprocess.run(
        [sys.executable, script], cwd=repo, capture_output=True, text=True, timeout=60
    )
    return p.returncode == 0


def main():
    model = os.environ.get("NSYNTH_LOCAL_LLM_URL", "")
    print(f"Python repair bench (model lane: {model or 'OFF — will fail, harness is model-only'})\n")
    print(f"{'TASK':<26}{'shown':>8}{'heldout':>9}   class")
    shown_ok = generalized = total = 0
    base = tempfile.mkdtemp(prefix="py_repair_bench_")
    for tid, cls, buggy, shown, held in TASKS:
        total += 1
        repo = os.path.join(base, tid)
        os.makedirs(repo)
        open(os.path.join(repo, "sol.py"), "w").write(buggy)
        open(os.path.join(repo, "test_shown.py"), "w").write(shown)
        # The model sees only sol.py + the shown-test failure. Never the held-out asserts.
        success, _iters, _note = py_repair.repair(
            repo, "sol.py", f"{sys.executable} test_shown.py", iters=3
        )
        shown_green = success and run(repo, "test_shown.py")
        held_green = False
        if shown_green:
            open(os.path.join(repo, "test_held.py"), "w").write(held)
            held_green = run(repo, "test_held.py")
        shown_ok += shown_green
        generalized += held_green
        flag = "  <- OVERFIT" if shown_green and not held_green else ("" if shown_green else "  <- unfixed")
        print(f"{tid:<26}{('green' if shown_green else 'red'):>8}{('GREEN' if held_green else 'red'):>9}   {cls}{flag}")
    shutil.rmtree(base, ignore_errors=True)
    print(
        f"\nPYTHON REPAIR (VibeThinker, never-wrong): {generalized}/{total} GENERALIZE (held-out) | "
        f"{shown_ok}/{total} passed the shown test. Every shipped fix was pytest/oracle-verified; "
        f"unverified proposals were reverted."
    )


if __name__ == "__main__":
    main()
