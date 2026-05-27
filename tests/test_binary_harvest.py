"""Regression tests for the coreutils binary-harvest pipeline.

Pin the mechanism so future edits to the probes or reference impls
can't silently ship a wrong-implementation that contaminates the
distillation dataset. We test each registered tool end-to-end:
  1. Generate probes for a seeded RNG.
  2. Run the real binary (if present) on each probe.
  3. Exec the generated reference implementation on the same stdin.
  4. Assert the two produce byte-identical stdout.

These tests require the underlying binaries to exist. If a binary
is missing on the test machine, that tool's test xfails with a
clear message rather than poisoning the suite."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS_HARVEST = Path(__file__).resolve().parent.parent / "tools" / "binary_harvest"
sys.path.insert(0, str(TOOLS_HARVEST))

from harvest import TOOLS, reference_python  # noqa: E402


def _binary_available(spec: dict) -> bool:
    return Path(spec["bin"]).exists()


@pytest.mark.parametrize("tool_name", sorted(TOOLS.keys()))
def test_reference_impl_matches_binary(tool_name):
    spec = TOOLS[tool_name]
    if not _binary_available(spec):
        pytest.xfail(f"binary {spec['bin']} not present on this machine")
    rng = random.Random(tool_name)  # deterministic per tool
    probes = spec["probe"](rng)
    assert probes, f"tool {tool_name} generated no probes"
    # Subsample to avoid slow tests on tools with many probes.
    if len(probes) > 5:
        probes = probes[:5]

    mismatches = []
    for args, stdin in probes:
        r = subprocess.run(
            [spec["bin"], *args], input=stdin,
            capture_output=True, text=True, timeout=5,
        )
        # grep returns rc=1 on no-match; accept it.
        if tool_name == "grep":
            if r.returncode not in (0, 1):
                continue
        elif r.returncode != 0:
            continue
        expected = r.stdout
        code = reference_python(tool_name, args, stdin, expected)
        ns = {"__builtins__": __builtins__}
        exec(code, ns)
        got = ns["solve"](stdin)
        if got != expected:
            mismatches.append({
                "args": args, "expected": expected[:80], "got": got[:80],
            })
    assert not mismatches, (
        f"{tool_name} reference impl diverges on {len(mismatches)} probes: "
        f"{mismatches[:2]}"
    )
