#!/usr/bin/env python3
"""Backends for the neural-vs-Meta comparison demo."""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from demos.neural.neural_terminal_pty import ShellProcess
from ncpu.neural.neural_terminal_renderer import TERM_COLS, TERM_ROWS
from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2


logging.getLogger("ncpu.neural.metal_inference").setLevel(logging.ERROR)


def _load_display(device: str | None) -> NeuralDisplayV2:
    model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer_v2.pt"
    return NeuralDisplayV2(str(model_path), device=device)


class PtyPaneRuntime:
    """Host PTY shell with neural pixels."""

    runtime_name = "pty"
    runtime_label = "Host PTY shell"
    compute_path = "host-pty-shell"
    runtime_owned_by_ncpu = False

    def __init__(self, *, shell: str, device: str | None, script_mode: bool):
        self.display = _load_display(device)
        self.shell = shell
        self.script_mode = script_mode
        self.proc = ShellProcess(
            shell,
            TERM_ROWS,
            TERM_COLS,
            login=not script_mode,
            extra_env={"PS1": "$ "} if script_mode else None,
        )

    def start(self) -> None:
        self.proc.start()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        return True

    def poll(self) -> list[bytes]:
        chunks: list[bytes] = []
        while True:
            data = self.proc.read()
            if not data:
                break
            self.display.write(data)
            chunks.append(data)
        return chunks

    def render(self):
        return self.display.render()

    def send(self, data: bytes) -> None:
        self.proc.write(data)

    @property
    def alive(self) -> bool:
        return self.proc.alive

    def terminate(self) -> None:
        self.proc.terminate()

    def metadata(self) -> dict[str, Any]:
        return {
            "left_runtime": self.runtime_name,
            "left_runtime_label": self.runtime_label,
            "left_runtime_owned_by_ncpu": self.runtime_owned_by_ncpu,
            "left_runtime_compute_path": self.compute_path,
            "left_runtime_shell": self.shell,
            "left_runtime_interactive": True,
        }


class NeuralOSPaneRuntime:
    """Headless nCPU GPU shell that feeds a neural display."""

    runtime_name = "neural-os"
    runtime_label = "nCPU GPU shell"
    compute_path = "ncpu-gpu-arm64-shell"
    runtime_owned_by_ncpu = True

    def __init__(self, *, device: str | None):
        self.display = _load_display(device)
        self.shell = "/bin/sh (ARM64 C shell on nCPU GPU OS)"
        self._display_lock = threading.Lock()
        self._input_queue: queue.Queue[bytes] = queue.Queue()
        self._output_queue: queue.Queue[bytes] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._alive = False
        self._ready = threading.Event()
        self._boot_error: str | None = None
        self._shell_binary: bytes | None = None
        self.total_cycles = 0
        self.elapsed_s = 0.0

    def start(self) -> None:
        try:
            self._shell_binary = self._compile_shell_binary()
        except Exception as exc:  # noqa: BLE001
            self._boot_error = f"{type(exc).__name__}: {exc}"
            self._emit_output(
                (
                    "\r\n\x1b[1;31m[nCPU runtime boot failed]\x1b[0m\r\n"
                    f"{self._boot_error}\r\n"
                ).encode("utf-8", errors="replace")
            )
            self._ready.set()
            self._alive = False
            self._running = False
            return
        self._running = True
        self._alive = True
        self._thread = threading.Thread(target=self._os_thread, daemon=True)
        self._thread.start()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout)

    def _emit_output(self, data: bytes) -> None:
        if not data:
            return
        payload = bytes(data)
        with self._display_lock:
            self.display.write(payload)
        self._output_queue.put(payload)

    def poll(self) -> list[bytes]:
        chunks: list[bytes] = []
        while True:
            try:
                chunks.append(self._output_queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    def render(self):
        with self._display_lock:
            return self.display.render()

    def send(self, data: bytes) -> None:
        if self._running:
            self._input_queue.put(bytes(data))

    @property
    def alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._running = False
        self._input_queue.put(b"")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def metadata(self) -> dict[str, Any]:
        data = {
            "left_runtime": self.runtime_name,
            "left_runtime_label": self.runtime_label,
            "left_runtime_owned_by_ncpu": self.runtime_owned_by_ncpu,
            "left_runtime_compute_path": self.compute_path,
            "left_runtime_shell": self.shell,
            "left_runtime_interactive": True,
            "left_runtime_total_cycles": int(self.total_cycles),
            "left_runtime_elapsed_s": float(self.elapsed_s),
        }
        if self._boot_error:
            data["left_runtime_boot_error"] = self._boot_error
        return data

    def _compile_shell_binary(self) -> bytes:
        from ncpu.os.gpu.runner import compile_c

        c_file = PROJECT_ROOT / "ncpu" / "os" / "gpu" / "src" / "arm64_unix_shell.c"
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            bin_path = f.name
        try:
            if not compile_c(str(c_file), bin_path, quiet=True):
                raise RuntimeError("shell compilation failed")
            return Path(bin_path).read_bytes()
        finally:
            if os.path.exists(bin_path):
                os.unlink(bin_path)

    def _bootstrap_filesystem(self):
        from ncpu.os.gpu.filesystem import GPUFilesystem

        fs = GPUFilesystem()
        for path in ["/etc", "/home", "/home/user", "/var/log", "/usr/lib", "/usr/include", "/tmp", "/bin"]:
            fs.mkdir(path)

        fs.write_file(
            "/etc/motd",
            "Welcome to nCPU Neural Computer\n"
            "Left pane runtime: ARM64 shell on the nCPU GPU OS\n"
            "Visible pixels: NeuralDisplayV2\n"
            "Type 'help' for commands.\n",
        )
        fs.write_file(
            "/etc/os-release",
            "NAME=\"nCPU GPU Shell\"\n"
            "VERSION=\"meta-compare\"\n"
            "ARCH=\"ARM64 Metal\"\n",
        )
        fs.write_file("/etc/hostname", "ncpu\n")
        fs.write_file(
            "/home/user/hello.c",
            '#include "arm64_libc.h"\n'
            "\n"
            "int main(void) {\n"
            '    printf("Hello from the nCPU GPU shell.\\n");\n'
            '    printf("This binary was compiled and executed inside our stack.\\n");\n'
            "    return 0;\n"
            "}\n",
        )
        fs.write_file(
            "/home/user/README.txt",
            "nCPU Meta comparison runtime\n"
            "- ARM64 shell running on our GPU OS\n"
            "- Neural screen output\n",
        )
        fs.chdir("/home/user")
        return fs

    def _os_thread(self) -> None:
        try:
            from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2, StopReasonV2
            from ncpu.os.gpu.runner import make_syscall_handler

            shell_binary = self._shell_binary
            if not shell_binary:
                raise RuntimeError("shell binary missing")
            fs = self._bootstrap_filesystem()

            cpu = MLXKernelCPUv2()
            cpu.load_program(shell_binary, address=0x10000)
            cpu.set_pc(0x10000)

            def on_read(fd: int, max_len: int):
                if fd != 0:
                    return None
                while self._running:
                    try:
                        return self._input_queue.get(timeout=0.1)[:max_len]
                    except queue.Empty:
                        continue
                return b""

            def on_exec(bin_path_str: str) -> bool:
                resolved = fs.resolve_path(bin_path_str)
                binary_data = fs.read_file(resolved)
                if binary_data:
                    cpu.load_program(binary_data, address=0x10000)
                    cpu.set_pc(0x10000)
                    return True
                return False

            def on_write(fd: int, data: bytes) -> bool:
                if fd in (1, 2):
                    self._emit_output(bytes(data))
                    return True
                return False

            handler = make_syscall_handler(
                filesystem=fs,
                on_read=on_read,
                on_write=on_write,
                on_exec=on_exec,
                neural_display=None,
            )

            if cpu.memory_size > cpu.SVC_BUF_BASE + 0x10000:
                cpu.init_svc_buffer()

            self._ready.set()
            start_time = time.perf_counter()
            total_cycles = 0

            while self._running and total_cycles < 500_000_000:
                result = cpu.execute(max_cycles=100_000)
                total_cycles += int(result.cycles)
                self.total_cycles = total_cycles
                self.elapsed_s = time.perf_counter() - start_time

                if cpu.memory_size > cpu.SVC_BUF_BASE:
                    for fd, data in cpu.drain_svc_buffer():
                        if fd in (1, 2):
                            self._emit_output(bytes(data))

                if result.stop_reason == StopReasonV2.HALT:
                    break
                if result.stop_reason == StopReasonV2.SYSCALL:
                    ret = handler(cpu)
                    if ret is False:
                        break
                    if ret == "exec":
                        continue
                    cpu.set_pc(cpu.pc + 4)
                    continue
                if result.stop_reason == StopReasonV2.MAX_CYCLES:
                    continue

            self.total_cycles = total_cycles
            self.elapsed_s = time.perf_counter() - start_time
        except Exception as exc:  # noqa: BLE001
            self._boot_error = f"{type(exc).__name__}: {exc}"
            self._emit_output(
                (
                    "\r\n\x1b[1;31m[nCPU runtime boot failed]\x1b[0m\r\n"
                    f"{self._boot_error}\r\n"
                ).encode("utf-8", errors="replace")
            )
        finally:
            self._alive = False
            self._running = False
            self._ready.set()


def build_left_runtime(
    runtime_name: str,
    *,
    shell: str,
    device: str | None,
    script_mode: bool,
):
    if runtime_name == "neural-os":
        return NeuralOSPaneRuntime(device=device)
    if runtime_name == "pty":
        return PtyPaneRuntime(shell=shell, device=device, script_mode=script_mode)
    raise ValueError(f"Unknown left runtime: {runtime_name}")
