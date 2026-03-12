"""GPU UNIX OS package with lazy exports to avoid backend side effects."""

from __future__ import annotations

import importlib


_EXPORTS = {
    "compile_c": ("ncpu.os.gpu.runner", "compile_c"),
    "compile_c_from_string": ("ncpu.os.gpu.runner", "compile_c_from_string"),
    "run": ("ncpu.os.gpu.runner", "run"),
    "make_syscall_handler": ("ncpu.os.gpu.runner", "make_syscall_handler"),
    "ProcessManager": ("ncpu.os.gpu.runner", "ProcessManager"),
    "run_multiprocess": ("ncpu.os.gpu.runner", "run_multiprocess"),
    "GPUFilesystem": ("ncpu.os.gpu.filesystem", "GPUFilesystem"),
    "PipeBuffer": ("ncpu.os.gpu.filesystem", "PipeBuffer"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
