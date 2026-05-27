import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmarks.export_gpu_only_matrix as exporter


def _benchmark_record(
    workload: str,
    *,
    backend: str,
    avg_ips: float,
    backend_ok: bool | None = None,
) -> dict:
    return {
        "workload": workload,
        "avg_ips": avg_ips,
        "peak_ips": avg_ips,
        "min_ips": avg_ips,
        "backend": backend,
        "backend_ok": backend_ok,
        "backend_requirement": "rust" if backend_ok is not None else None,
        "hotloop_segments": 2 if "adjacent" in workload or "bytecopy" in workload else 1,
        "hotloop_pre_sync_bytes": 0,
        "hotloop_post_sync_bytes": 0,
        "hotloop_reused_state_segments": 0,
        "hotloop_detector_attempts": 1,
        "hotloop_policy_rejections": 0,
        "result_ok": True,
        "result_check": "OK",
        "executed_count": 100,
        "expected_executed_count": 100,
        "insts_ok": True,
        "best_rust_backend": None,
        "best_rust_avg_ips": None,
        "best_rust_speedup_vs_neural": None,
        "hotloop_samples": [],
        "hotloop_trace": [],
    }


def test_run_matrix_rust_only_skips_torch_baseline(monkeypatch, tmp_path):
    calls = []

    def fake_run(workload, overrides, *, compare_rust, timeout_seconds, require_backend_prefix=None):
        calls.append(
            {
                "workload": workload,
                "overrides": dict(overrides),
                "compare_rust": compare_rust,
                "timeout_seconds": timeout_seconds,
                "require_backend_prefix": require_backend_prefix,
            }
        )
        return {
            "status": "completed",
            "record": _benchmark_record(
                workload,
                backend="rust-hotloop",
                avg_ips=1234.0,
                backend_ok=True,
            ),
            "error_message": None,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(exporter, "_run_benchmark_subprocess", fake_run)

    result = exporter.run_matrix(
        tmp_path,
        workloads=["counted"],
        primary_backend="rust",
        require_backend_prefix="rust",
        include_torch_baseline=False,
        compare_rust=False,
    )

    row = result["records"][0]
    assert row["status"] == "completed"
    assert row["primary_completed"] is True
    assert row["torch_baseline_status"] == "skipped"
    assert row["backend"] == "rust-hotloop"
    payload = json.loads((tmp_path / "gpu_only_matrix.json").read_text())
    assert payload["benchmark_env"]["NCPU_GPU_ONLY_HOTLOOP_BACKEND"] == "rust"
    assert payload["summary"]["strict_rust_only"] is True
    assert payload["summary"]["passing_rows"] == 1
    assert calls == [
        {
            "workload": "counted",
            "overrides": {"NCPU_GPU_ONLY_HOTLOOP_BACKEND": "rust"},
            "compare_rust": False,
            "timeout_seconds": exporter.DEFAULT_TIMEOUT_SECONDS,
            "require_backend_prefix": "rust",
        }
    ]


def test_run_matrix_resume_retries_only_missing_torch_baseline(monkeypatch, tmp_path):
    first_calls = []

    def first_fake_run(workload, overrides, *, compare_rust, timeout_seconds, require_backend_prefix=None):
        phase = overrides["NCPU_GPU_ONLY_HOTLOOP_BACKEND"]
        first_calls.append(phase)
        if phase == "0":
            return {
                "status": "timeout",
                "record": None,
                "error_message": "timed out after 5.0s",
                "timeout_seconds": timeout_seconds,
            }
        return {
            "status": "completed",
            "record": _benchmark_record(
                workload,
                backend="rust-hotloop",
                avg_ips=1000.0,
                backend_ok=True,
            ),
            "error_message": None,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(exporter, "_run_benchmark_subprocess", first_fake_run)

    first_result = exporter.run_matrix(
        tmp_path,
        workloads=["counted"],
        timeout_seconds=10.0,
        torch_timeout_seconds=5.0,
        primary_backend="rust",
        require_backend_prefix="rust",
        include_torch_baseline=True,
        compare_rust=False,
    )

    first_row = first_result["records"][0]
    assert first_calls == ["rust", "0"]
    assert first_row["primary_completed"] is True
    assert first_row["status"] == "baseline-timeout"
    assert first_row["torch_baseline_status"] == "timeout"
    assert first_row["backend"] == "rust-hotloop"

    second_calls = []

    def second_fake_run(workload, overrides, *, compare_rust, timeout_seconds, require_backend_prefix=None):
        phase = overrides["NCPU_GPU_ONLY_HOTLOOP_BACKEND"]
        second_calls.append(phase)
        assert phase == "0"
        return {
            "status": "completed",
            "record": _benchmark_record(
                workload,
                backend="torch-gpu-only",
                avg_ips=100.0,
            ),
            "error_message": None,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(exporter, "_run_benchmark_subprocess", second_fake_run)

    resumed = exporter.run_matrix(
        tmp_path,
        workloads=["counted"],
        resume=True,
        timeout_seconds=10.0,
        torch_timeout_seconds=5.0,
        primary_backend="rust",
        require_backend_prefix="rust",
        include_torch_baseline=True,
        compare_rust=False,
    )

    resumed_row = resumed["records"][0]
    assert second_calls == ["0"]
    assert resumed_row["status"] == "completed"
    assert resumed_row["primary_completed"] is True
    assert resumed_row["torch_baseline_status"] == "completed"
    assert resumed_row["torch_baseline_backend"] == "torch-gpu-only"
    assert resumed_row["hotloop_speedup_vs_torch"] == pytest.approx(10.0, rel=1e-6)

    payload = json.loads((tmp_path / "gpu_only_matrix.json").read_text())
    assert payload["results"][0]["status"] == "completed"


def test_run_benchmark_subprocess_retries_transient_signal_exit(monkeypatch):
    attempts = []

    def fake_run(cmd, **kwargs):
        attempts.append(list(cmd))
        json_output_path = Path(cmd[cmd.index("--json-output") + 1])
        if len(attempts) < 3:
            return SimpleNamespace(returncode=-10)
        json_output_path.write_text(
            json.dumps(
                _benchmark_record(
                    "counted",
                    backend="rust-hotloop",
                    avg_ips=1234.0,
                    backend_ok=True,
                )
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    monkeypatch.setattr(exporter.time, "sleep", lambda _seconds: None)

    outcome = exporter._run_benchmark_subprocess(
        "counted",
        {"NCPU_GPU_ONLY_HOTLOOP_BACKEND": "rust"},
        compare_rust=False,
        timeout_seconds=10.0,
        require_backend_prefix="rust",
        retry_attempts=4,
    )

    assert len(attempts) == 3
    assert outcome["status"] == "completed"
    assert outcome["record"]["backend"] == "rust-hotloop"
    assert outcome["record"]["backend_ok"] is True


def test_run_matrix_inprocess_skips_subprocess(monkeypatch, tmp_path):
    subprocess_calls = []
    inprocess_calls = []

    def fake_subprocess_run(workload, overrides, **kwargs):
        subprocess_calls.append(workload)
        raise AssertionError("subprocess runner should not be called in in-process mode")

    def fake_inprocess_run(
        workload,
        overrides,
        *,
        compare_rust,
        require_backend_prefix=None,
        timeout_seconds=None,
        cpu=None,
    ):
        inprocess_calls.append(
            {
                "workload": workload,
                "overrides": dict(overrides),
                "compare_rust": compare_rust,
                "require_backend_prefix": require_backend_prefix,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "status": "completed",
            "record": _benchmark_record(
                workload,
                backend="rust-hotloop",
                avg_ips=4321.0,
                backend_ok=True,
            ),
            "error_message": None,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(exporter, "_run_benchmark_subprocess", fake_subprocess_run)
    monkeypatch.setattr(exporter, "_run_benchmark_inprocess", fake_inprocess_run)

    result = exporter.run_matrix(
        tmp_path,
        workloads=["counted"],
        primary_backend="rust",
        require_backend_prefix="rust",
        include_torch_baseline=False,
        compare_rust=False,
        inprocess=True,
    )

    assert subprocess_calls == []
    assert inprocess_calls == [
        {
            "workload": "counted",
            "overrides": {"NCPU_GPU_ONLY_HOTLOOP_BACKEND": "rust"},
            "compare_rust": False,
            "require_backend_prefix": "rust",
            "timeout_seconds": exporter.DEFAULT_TIMEOUT_SECONDS,
        }
    ]
    row = result["records"][0]
    assert row["status"] == "completed"
    assert row["backend"] == "rust-hotloop"
    payload = json.loads((tmp_path / "gpu_only_matrix.json").read_text())
    assert payload["exporter_config"]["inprocess"] is True
    assert payload["summary"]["strict_rust_only"] is True


def test_run_matrix_inprocess_routes_torch_baseline_through_inprocess(monkeypatch, tmp_path):
    subprocess_calls = []
    inprocess_calls = []

    def fake_subprocess_run(workload, overrides, **kwargs):
        subprocess_calls.append(dict(overrides))
        raise AssertionError("subprocess runner should not be used in in-process mode")

    def fake_inprocess_run(
        workload,
        overrides,
        *,
        compare_rust,
        require_backend_prefix=None,
        timeout_seconds=None,
        cpu=None,
    ):
        overrides = dict(overrides)
        inprocess_calls.append(overrides)
        if overrides.get("NCPU_GPU_ONLY_HOTLOOP_BACKEND") == "0":
            record = _benchmark_record(workload, backend="torch-gpu-only", avg_ips=100.0)
        else:
            record = _benchmark_record(
                workload,
                backend="rust-hotloop",
                avg_ips=1000.0,
                backend_ok=True,
            )
        return {
            "status": "completed",
            "record": record,
            "error_message": None,
            "timeout_seconds": None,
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(exporter, "_run_benchmark_subprocess", fake_subprocess_run)
    monkeypatch.setattr(exporter, "_run_benchmark_inprocess", fake_inprocess_run)

    result = exporter.run_matrix(
        tmp_path,
        workloads=["counted"],
        primary_backend="auto",
        require_backend_prefix=None,
        include_torch_baseline=True,
        compare_rust=False,
        inprocess=True,
    )

    assert subprocess_calls == []
    phases = [overrides["NCPU_GPU_ONLY_HOTLOOP_BACKEND"] for overrides in inprocess_calls]
    assert phases == ["auto", "0"], phases
    row = result["records"][0]
    assert row["torch_baseline_status"] == "completed"
    assert row["torch_baseline_backend"] == "torch-gpu-only"
    assert row["hotloop_speedup_vs_torch"] == pytest.approx(10.0, rel=1e-6)


def test_run_benchmark_inprocess_captures_stdout_and_restores_env(monkeypatch):
    captured_env: dict[str, str | None] = {}

    def fake_benchmark(workload, *, compare_rust, require_backend_prefix, cpu=None):
        captured_env["during"] = os.environ.get("NCPU_GPU_ONLY_HOTLOOP_BACKEND")
        print(f"noisy benchmark output for {workload}")
        return _benchmark_record(workload, backend="rust-hotloop", avg_ips=42.0, backend_ok=True)

    fake_module = type(sys)("benchmarks.benchmark_gpu_only")
    fake_module.benchmark = fake_benchmark
    monkeypatch.setitem(sys.modules, "benchmarks.benchmark_gpu_only", fake_module)

    monkeypatch.delenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", raising=False)
    outcome = exporter._run_benchmark_inprocess(
        "counted",
        {"NCPU_GPU_ONLY_HOTLOOP_BACKEND": "rust"},
        compare_rust=False,
        require_backend_prefix="rust",
    )

    assert outcome["status"] == "completed"
    assert outcome["record"]["backend"] == "rust-hotloop"
    assert captured_env["during"] == "rust"
    assert "NCPU_GPU_ONLY_HOTLOOP_BACKEND" not in os.environ


def test_run_matrix_reuse_cpu_builds_one_cpu_and_threads_it(monkeypatch, tmp_path):
    build_calls = []
    received_cpus = []

    sentinel_cpu = object()

    def fake_build():
        build_calls.append(True)
        return sentinel_cpu

    def fake_inprocess_run(
        workload,
        overrides,
        *,
        compare_rust,
        require_backend_prefix=None,
        timeout_seconds=None,
        cpu=None,
    ):
        received_cpus.append(cpu)
        return {
            "status": "completed",
            "record": _benchmark_record(
                workload,
                backend="rust-hotloop",
                avg_ips=100.0,
                backend_ok=True,
            ),
            "error_message": None,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(exporter, "_build_reusable_cpu", fake_build)
    monkeypatch.setattr(exporter, "_run_benchmark_inprocess", fake_inprocess_run)

    result = exporter.run_matrix(
        tmp_path,
        workloads=["counted", "bytecopy"],
        primary_backend="rust",
        require_backend_prefix="rust",
        include_torch_baseline=False,
        compare_rust=False,
        inprocess=True,
        reuse_cpu=True,
    )

    assert build_calls == [True], "CPU must be constructed exactly once"
    assert received_cpus == [sentinel_cpu, sentinel_cpu]
    payload = json.loads((tmp_path / "gpu_only_matrix.json").read_text())
    assert payload["exporter_config"]["reuse_cpu"] is True


def test_run_matrix_reuse_cpu_ignored_without_inprocess(monkeypatch, tmp_path):
    build_calls = []

    def fake_build():
        build_calls.append(True)
        raise AssertionError("CPU must not be built when inprocess=False")

    def fake_subprocess_run(workload, overrides, **kwargs):
        return {
            "status": "completed",
            "record": _benchmark_record(
                workload,
                backend="rust-hotloop",
                avg_ips=100.0,
                backend_ok=True,
            ),
            "error_message": None,
            "timeout_seconds": kwargs.get("timeout_seconds"),
        }

    monkeypatch.setattr(exporter, "_build_reusable_cpu", fake_build)
    monkeypatch.setattr(exporter, "_run_benchmark_subprocess", fake_subprocess_run)

    result = exporter.run_matrix(
        tmp_path,
        workloads=["counted"],
        primary_backend="rust",
        require_backend_prefix="rust",
        include_torch_baseline=False,
        compare_rust=False,
        inprocess=False,
        reuse_cpu=True,
    )

    payload = json.loads((tmp_path / "gpu_only_matrix.json").read_text())
    assert payload["exporter_config"]["reuse_cpu"] is False
    assert build_calls == []


def test_run_benchmark_inprocess_times_out_pathological_workload(monkeypatch):
    """A workload that exceeds its budget must surface as status='timeout', not hang."""

    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM not available on this platform")

    def fake_benchmark(workload, *, compare_rust, require_backend_prefix, cpu=None):
        # Simulate an infinite loop that the SIGALRM handler must interrupt.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            time.sleep(0.01)
        return _benchmark_record(workload, backend="rust-hotloop", avg_ips=1.0, backend_ok=True)

    fake_module = type(sys)("benchmarks.benchmark_gpu_only")
    fake_module.benchmark = fake_benchmark
    monkeypatch.setitem(sys.modules, "benchmarks.benchmark_gpu_only", fake_module)

    outcome = exporter._run_benchmark_inprocess(
        "counted",
        {"NCPU_GPU_ONLY_HOTLOOP_BACKEND": "rust"},
        compare_rust=False,
        require_backend_prefix="rust",
        timeout_seconds=0.5,
    )

    assert outcome["status"] == "timeout"
    assert outcome["record"] is None
    assert "deadline" in outcome["error_message"]
    # Deadline should fire roughly at the timeout, not at the 5s simulated runtime
    assert outcome["elapsed_seconds"] < 3.0
    # SIGALRM handler must have been restored after the run
    assert signal.getsignal(signal.SIGALRM) in (signal.SIG_DFL, signal.SIG_IGN) or callable(
        signal.getsignal(signal.SIGALRM)
    )


def test_run_benchmark_inprocess_zero_timeout_is_no_op(monkeypatch):
    """timeout_seconds=0 or None must NOT arm the deadline timer."""

    def fake_benchmark(workload, *, compare_rust, require_backend_prefix, cpu=None):
        return _benchmark_record(workload, backend="rust-hotloop", avg_ips=1.0, backend_ok=True)

    fake_module = type(sys)("benchmarks.benchmark_gpu_only")
    fake_module.benchmark = fake_benchmark
    monkeypatch.setitem(sys.modules, "benchmarks.benchmark_gpu_only", fake_module)

    for timeout in (None, 0, 0.0, -5.0):
        outcome = exporter._run_benchmark_inprocess(
            "counted",
            {},
            compare_rust=False,
            timeout_seconds=timeout,
        )
        assert outcome["status"] == "completed", f"timeout={timeout!r} should be no-op"


def test_run_benchmark_inprocess_returns_error_on_exception(monkeypatch):
    def fake_benchmark(workload, *, compare_rust, require_backend_prefix, cpu=None):
        raise RuntimeError("simulated ALU failure")

    fake_module = type(sys)("benchmarks.benchmark_gpu_only")
    fake_module.benchmark = fake_benchmark
    monkeypatch.setitem(sys.modules, "benchmarks.benchmark_gpu_only", fake_module)

    outcome = exporter._run_benchmark_inprocess(
        "counted",
        {},
        compare_rust=False,
    )

    assert outcome["status"] == "error"
    assert outcome["record"] is None
    assert "simulated ALU failure" in outcome["error_message"]


def test_run_benchmark_subprocess_retries_transient_timeout(monkeypatch):
    attempts = []

    def fake_run(cmd, **kwargs):
        attempts.append(list(cmd))
        json_output_path = Path(cmd[cmd.index("--json-output") + 1])
        if len(attempts) < 3:
            raise exporter.subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])
        json_output_path.write_text(
            json.dumps(
                _benchmark_record(
                    "counted",
                    backend="rust-hotloop",
                    avg_ips=1234.0,
                    backend_ok=True,
                )
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    monkeypatch.setattr(exporter.time, "sleep", lambda _seconds: None)

    outcome = exporter._run_benchmark_subprocess(
        "counted",
        {"NCPU_GPU_ONLY_HOTLOOP_BACKEND": "rust"},
        compare_rust=False,
        timeout_seconds=10.0,
        require_backend_prefix="rust",
        retry_attempts=4,
    )

    assert len(attempts) == 3
    assert outcome["status"] == "completed"
    assert outcome["record"]["backend"] == "rust-hotloop"
