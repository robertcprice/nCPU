#!/usr/bin/env python3
"""SWE-bench-lite-style sweep: inject a realistic bug into a REAL repo's source, then repair it with
py_repair (auto-localize + VibeThinker + the repo's OWN tests as the oracle). Reports pass@1 over
several bugs across different modules. Every fix is test-verified; the file is reverted after each
task so runs are independent.

Setup (once):
    git clone --depth 1 https://github.com/keon/algorithms.git /tmp/algos
Run:
    NSYNTH_LOCAL_LLM_URL=http://127.0.0.1:8080/v1/chat/completions \
    NSYNTH_LOCAL_LLM_MODEL=mlx-community/VibeThinker-3B-4bit \
    python3 py_swebench_sweep.py /tmp/algos
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import py_repair  # noqa: E402

# (id, relpath, test_command, find, replace) — `find`->`replace` injects a realistic single-edit bug.
TASKS = [
    (
        "binary_search:boundary",
        "algorithms/searching/binary_search.py",
        "python3 -m pytest tests/test_searching.py -q",
        "while low <= high:",
        "while low < high:",
    ),
    (
        "bubble_sort:wrong_compare",
        "algorithms/sorting/bubble_sort.py",
        "python3 -m pytest tests/test_sorting.py -q",
        "if array[i - 1] > array[i]:",
        "if array[i - 1] < array[i]:",
    ),
    (
        "selection_sort:wrong_compare",
        "algorithms/sorting/selection_sort.py",
        "python3 -m pytest tests/test_sorting.py -q",
        "if array[j] < array[minimum]:",
        "if array[j] > array[minimum]:",
    ),
    (
        "insertion_sort:wrong_compare",
        "algorithms/sorting/insertion_sort.py",
        "python3 -m pytest tests/test_sorting.py -q",
        "while pos > 0 and array[pos - 1] > cursor:",
        "while pos > 0 and array[pos - 1] < cursor:",
    ),
]


def run(repo, cmd):
    ok, _ = py_repair.run_tests(repo, cmd)
    return ok


def main(repo):
    model = os.environ.get("NSYNTH_LOCAL_LLM_URL", "")
    print(f"SWE-bench-lite sweep on {repo}\nmodel: {model or 'OFF (harness is model-only)'}\n")
    print(f"{'TASK':<30}{'baseline':>10}{'repaired':>10}   note")
    fixed = considered = 0
    for tid, rel, cmd, find, repl in TASKS:
        path = os.path.join(repo, rel)
        if not os.path.exists(path):
            print(f"{tid:<30}{'':>10}{'':>10}   MISSING {rel}")
            continue
        orig = open(path).read()
        if find not in orig:
            print(f"{tid:<30}{'':>10}{'':>10}   inject-miss (source changed)")
            continue
        open(path, "w").write(orig.replace(find, repl, 1))
        try:
            baseline_fail = not run(repo, cmd)  # the injected bug must be caught by the repo's tests
            if not baseline_fail:
                print(f"{tid:<30}{'PASS?!':>10}{'':>10}   bug not caught by tests — skip")
                continue
            considered += 1
            success, iters, note = py_repair.repair(repo, "auto", cmd, iters=3)
            repaired = success and run(repo, cmd)
            fixed += repaired
            print(
                f"{tid:<30}{'fail':>10}{('GREEN' if repaired else 'red'):>10}   "
                f"{note} (iters={iters})"
            )
        finally:
            open(path, "w").write(orig)  # revert to pristine for the next task
    print(
        f"\nSWE-BENCH-LITE (real repo, auto-localize, never-wrong): {fixed}/{considered} REPAIRED "
        f"pass@1 (repo's own tests). Unverified proposals were reverted."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
