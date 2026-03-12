"""Safe backend availability probes for MLX and Rust Metal.

These probes run in a subprocess because some backends can abort the Python
interpreter instead of raising a normal ImportError.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from functools import lru_cache


_DEFAULT_TIMEOUT = float(os.environ.get("NCPU_BACKEND_PROBE_TIMEOUT", "5.0"))


def _forced_result(name: str):
    raw = os.environ.get(f"NCPU_FORCE_{name.upper()}_USABLE")
    if raw is None:
        return None
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _run_probe(snippet: str, timeout: float) -> tuple[bool, str | None]:
    cmd = [
        sys.executable,
        "-c",
        snippet,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "Timed out running backend probe"

    if result.returncode == 0:
        return True, None

    message = (result.stderr or result.stdout or "").strip()
    if not message:
        message = f"Import exited with status {result.returncode}"
    return False, message


@lru_cache(maxsize=16)
def probe_python_module(module_name: str, timeout: float = _DEFAULT_TIMEOUT) -> tuple[bool, str | None]:
    """Probe whether a Python module can be imported without crashing."""
    if importlib.util.find_spec(module_name) is None:
        return False, f"Module {module_name!r} not found"

    return _run_probe(
        "import importlib; "
        f"importlib.import_module({module_name!r})",
        timeout,
    )


@lru_cache(maxsize=1)
def mlx_probe() -> tuple[bool, str | None]:
    forced = _forced_result("mlx")
    if forced is not None:
        return forced, None if forced else "Forced unavailable by env"
    return probe_python_module("mlx.core")


@lru_cache(maxsize=1)
def rust_metal_probe() -> tuple[bool, str | None]:
    forced = _forced_result("rust")
    if forced is not None:
        return forced, None if forced else "Forced unavailable by env"
    if importlib.util.find_spec("ncpu_metal") is None:
        return False, "Module 'ncpu_metal' not found"
    return _run_probe(
        (
            "import ncpu_metal; "
            "ncpu_metal.FullARM64CPU(memory_size=65536, cycles_per_batch=1)"
        ),
        _DEFAULT_TIMEOUT,
    )


def is_mlx_usable() -> bool:
    return mlx_probe()[0]


def is_rust_metal_usable() -> bool:
    return rust_metal_probe()[0]


def has_gpu_backend() -> bool:
    return is_rust_metal_usable() or is_mlx_usable()


__all__ = [
    "probe_python_module",
    "mlx_probe",
    "rust_metal_probe",
    "is_mlx_usable",
    "is_rust_metal_usable",
    "has_gpu_backend",
]
