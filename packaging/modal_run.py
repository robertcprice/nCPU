"""Modal-based runner for NPCoT benchmarks (DEPLOY-2 — Modal).

Modal (modal.com) gives you GPU compute by the second without having to
manage SSH keys or worry about idle instances — containers spin up on
demand, run, and shut down. Cost is usually higher per-GPU-hour than
vast.ai but much lower operational friction.

Usage:

    # Install modal
    pip install modal
    modal setup

    # Run
    modal run packaging/modal_run.py::run_tests
    modal run packaging/modal_run.py::run_bench3
    modal run packaging/modal_run.py::run_humaneval --model Qwen/Qwen3.5-1.5B --library /path/to/lib.json

This file is intentionally minimal — it imports modal lazily so the file
is importable on hosts without modal installed, and it delegates all
NPCoT logic to the existing runners.
"""

from __future__ import annotations

import os
from pathlib import Path


def _build_image():
    """Build a modal image with everything NPCoT needs."""
    import modal

    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.5.1",
            "transformers>=4.45",
            "datasets",
            "pytest",
            "hypothesis",
            "pytest-asyncio",
        )
        .run_commands(
            "pip install --upgrade pip",
        )
        .add_local_python_source("ncpu", "benchmarks", "tests")
    )


def _build_app():
    import modal

    app = modal.App("npcot-bench")
    image = _build_image()

    @app.function(image=image, gpu="A10G", timeout=3600)
    def _run_tests_remote():
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/self_optimizing/", "-q"],
            capture_output=True, text=True,
        )
        return {"returncode": result.returncode, "output": result.stdout[-4000:]}

    @app.function(image=image, gpu="A10G", timeout=1800)
    def _run_bench3_remote():
        import json, subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.benchmark_npcot_coding_bench",
             "--n-problems", "200", "--json", "/tmp/bench3.json"],
            capture_output=True, text=True,
        )
        with open("/tmp/bench3.json") as f:
            report = json.load(f)
        return {"stdout": result.stdout, "report": report}

    @app.function(image=image, gpu="A10G", timeout=7200)
    def _run_humaneval_remote(model: str, library_bytes: bytes | None):
        import json, subprocess, sys
        args = [sys.executable, "-m", "ncpu.self_optimizing.humaneval_runner",
                "--model", model, "--out", "/tmp/he.json"]
        if library_bytes is None:
            args.append("--no-library")
        else:
            with open("/tmp/library.json", "wb") as f:
                f.write(library_bytes)
            args.extend(["--library", "/tmp/library.json"])
        result = subprocess.run(args, capture_output=True, text=True)
        try:
            with open("/tmp/he.json") as f:
                report = json.load(f)
        except FileNotFoundError:
            report = None
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-1500:],
            "report": report,
        }

    return app, _run_tests_remote, _run_bench3_remote, _run_humaneval_remote


def run_tests():
    """modal run packaging/modal_run.py::run_tests"""
    app, _run_tests_remote, _, _ = _build_app()
    with app.run():
        result = _run_tests_remote.remote()
    print(result["output"])
    return result["returncode"]


def run_bench3():
    """modal run packaging/modal_run.py::run_bench3"""
    app, _, _run_bench3_remote, _ = _build_app()
    with app.run():
        result = _run_bench3_remote.remote()
    print(result["stdout"])
    results = result["report"]["results"]
    print(f"\nNPCoT: {results['npcot_library']['pass_at_1']:.1%}")
    print(f"Baseline: {results['llm_baseline_emulated']['pass_at_1']:.1%}")
    print(f"Delta: {result['report']['pass_at_1_delta']:+.1%}")
    return 0


def run_humaneval(model: str = "Qwen/Qwen3.5-1.5B", library: str | None = None):
    """modal run packaging/modal_run.py::run_humaneval --model X --library Y"""
    app, _, _, _run_humaneval_remote = _build_app()
    lib_bytes = None
    if library is not None:
        lib_bytes = Path(library).read_bytes()
    with app.run():
        result = _run_humaneval_remote.remote(model, lib_bytes)
    print(result["stdout"])
    if result["report"] is not None:
        r = result["report"]["results"]
        print(f"\npass@1 = {r['pass_at_1']:.2%} ({r['pass_count']}/{r['total_problems']})")
    return result["returncode"]
