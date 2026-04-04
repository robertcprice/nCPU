"""Test that memory actually persists across separate Python process invocations."""

import subprocess
import sys
import tempfile
from pathlib import Path


def test_memory_persists_across_processes():
    tmp = Path(tempfile.mkdtemp())
    python = sys.executable

    # Process 1: synthesize, save memory
    p1_code = f'''
import sys; sys.path.insert(0, ".")
from egdc.mog_adaptive_router import AdaptiveMogRouter
from egdc.mog_benchmark import get_benchmark

router = AdaptiveMogRouter(memory_root="{tmp}")
problems = get_benchmark(seed=42, variants_per_factory=1)[:3]
summary = router.evaluate(problems, use_real_compiler=True)
print(f"P1: solved {{summary['num_solved']}}/{{summary['num_problems']}}")
print(f"P1: stored {{router.memory.total_successes()}} successes")
'''
    r1 = subprocess.run([python, "-c", p1_code], capture_output=True, text=True, timeout=60,
                         cwd="/Users/bobbyprice/projects/nCPU")
    assert r1.returncode == 0, r1.stderr
    assert "P1: solved 3/3" in r1.stdout
    assert "P1: stored 3 successes" in r1.stdout

    # Process 2: load memory, verify it's there, use it for retrieval
    p2_code = f'''
import sys; sys.path.insert(0, ".")
from egdc.mog_pathways import PathwayMemory

mem = PathwayMemory(root="{tmp}")
print(f"P2: loaded {{mem.total_successes()}} successes")
print(f"P2: families {{mem.successes_by_family()}}")

hits = mem.retrieve_similar("Return the sum of two integers", "fn add(a: i64, b: i64) -> i64", top_k=1)
if hits:
    print(f"P2: best match: {{hits[0]['problem_name']}} (sim={{hits[0]['similarity']:.2f}})")
else:
    print("P2: no matches")
'''
    r2 = subprocess.run([python, "-c", p2_code], capture_output=True, text=True, timeout=30,
                         cwd="/Users/bobbyprice/projects/nCPU")
    assert r2.returncode == 0, r2.stderr
    assert "P2: loaded 3 successes" in r2.stdout
    assert "P2: best match:" in r2.stdout


def test_repl_demo_runs_and_produces_output():
    """The REPL demo should run, synthesize programs, and compile them."""
    python = sys.executable
    r = subprocess.run(
        [python, "-m", "egdc.mog_repl", "--demo"],
        capture_output=True, text=True, timeout=120,
        cwd="/Users/bobbyprice/projects/nCPU",
    )
    assert r.returncode == 0, r.stderr
    assert "max2(100, 1) = 100" in r.stdout
    assert "lcm(12, 18) = 36" in r.stdout
    assert "sum_to_n(100) = 5050" in r.stdout
    assert "return x + x" in r.stdout or "return x * 2" in r.stdout  # double refined
