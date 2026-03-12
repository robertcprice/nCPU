"""Pure-Python tests for backend availability probes."""

import subprocess

from kernels.mlx import availability


def _clear_probe_caches():
    availability.probe_python_module.cache_clear()
    availability.mlx_probe.cache_clear()
    availability.rust_metal_probe.cache_clear()


def test_probe_python_module_missing_module():
    _clear_probe_caches()
    ok, error = availability.probe_python_module("definitely_missing_ncpu_module")
    assert ok is False
    assert "not found" in error.lower()


def test_probe_python_module_stdlib_module():
    _clear_probe_caches()
    ok, error = availability.probe_python_module("json")
    assert ok is True
    assert error is None


def test_rust_probe_reports_runtime_backend_failure(monkeypatch):
    _clear_probe_caches()

    monkeypatch.setattr(availability.importlib.util, "find_spec", lambda name: object())

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="RuntimeError: No Metal device found",
        )

    monkeypatch.setattr(availability.subprocess, "run", fake_run)

    ok, error = availability.rust_metal_probe()
    assert ok is False
    assert "No Metal device found" in error


def test_has_gpu_backend_false_when_forced_off(monkeypatch):
    _clear_probe_caches()
    monkeypatch.setenv("NCPU_FORCE_MLX_USABLE", "0")
    monkeypatch.setenv("NCPU_FORCE_RUST_USABLE", "0")

    assert availability.has_gpu_backend() is False
