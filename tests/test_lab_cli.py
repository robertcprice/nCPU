import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_lab(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "ncpu.lab", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_lab_help_exits_zero():
    result = run_lab("--help")
    assert result.returncode == 0
    assert "interactive discovery" in result.stdout.lower() or "flagship launcher" in result.stdout.lower()


def test_lab_demos_lists_flagship_entries():
    result = run_lab("demos")
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "flagship interactive" in out
    assert "interactive discovery repl" in out
    assert "neural text machine" in out
    assert "busybox" in out


def test_lab_demos_verbose_shows_platform_and_weight():
    result = run_lab("demos", "--verbose")
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "best platform" in out
    assert "weight" in out


def test_lab_doctor_exits_zero():
    result = run_lab("doctor")
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "environment doctor" in out
    assert "recommended first run" in out
    assert "demo,dev" in out


def test_lab_systems_help_exits_zero():
    result = run_lab("systems", "--help")
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "busybox" in out
    assert "alpine" in out


def test_lab_coprocessor_help_exits_zero():
    result = run_lab("coprocessor", "--help")
    assert result.returncode == 0
    assert "coprocessor" in result.stdout.lower()


def test_lab_info_shows_demo_summary():
    result = run_lab("info", "discover")
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "interactive discovery repl" in out
    assert "best platform" in out
    assert "highlights" in out
    assert "next suggested step" in out


def test_lab_path_shows_recommended_flow():
    result = run_lab("path")
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "recommended path" in out
    assert "discover" in out
    assert "text --interactive" in out
    assert "walkthrough" in out


def test_lab_walkthrough_shows_guided_steps():
    result = run_lab("walkthrough")
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "step 1" in out
    assert "preset add" in out
    assert "cipher hello khoor" in out
