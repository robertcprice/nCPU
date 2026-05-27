#!/usr/bin/env python3
"""Export a publication-facing GPU-only benchmark matrix.

The exporter runs each workload in an isolated subprocess so a single slow or
pathological benchmark does not block the entire matrix. Results are written
incrementally after each workload, which makes ``--resume`` reliable for long
publication sweeps.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_gpu_only import BENCHMARK_WORKLOADS
from ncpu.utils.gpu_only_matrix import summarize_gpu_only_matrix

BENCHMARK_ENV_DEFAULTS = {
    "NCPU_GPU_ONLY_HOTLOOP_BACKEND": "auto",
    "NCPU_GPU_ONLY_AUTO_ALLOW_CPU": "1",
    "NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS": "1",
}

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_TORCH_TIMEOUT_SECONDS = 45.0
DEFAULT_RETRY_ATTEMPTS = 8

CSV_FIELDS = [
    "workload",
    "status",
    "primary_completed",
    "avg_ips",
    "peak_ips",
    "min_ips",
    "backend",
    "backend_ok",
    "backend_requirement",
    "hotloop_segments",
    "hotloop_pre_sync_bytes",
    "hotloop_post_sync_bytes",
    "hotloop_reused_state_segments",
    "hotloop_detector_attempts",
    "hotloop_policy_rejections",
    "result_ok",
    "result_check",
    "executed_count",
    "expected_executed_count",
    "insts_ok",
    "best_rust_backend",
    "best_rust_avg_ips",
    "best_rust_speedup_vs_neural",
    "torch_baseline_status",
    "torch_baseline_backend",
    "torch_baseline_avg_ips",
    "torch_baseline_peak_ips",
    "torch_baseline_min_ips",
    "torch_baseline_result_ok",
    "torch_baseline_insts_ok",
    "hotloop_speedup_vs_torch",
    "error_message",
    "timeout_seconds",
]


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _benchmark_command(
    workload: str,
    *,
    compare_rust: bool,
    require_backend_prefix: str | None,
    json_output_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmarks" / "benchmark_gpu_only.py"),
        "--workload",
        workload,
        "--json-output",
        str(json_output_path),
    ]
    cmd.append("--compare-rust" if compare_rust else "--no-compare-rust")
    if require_backend_prefix:
        cmd.extend(["--require-backend-prefix", require_backend_prefix])
    return cmd


def _extract_json_record(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("benchmark subprocess produced no JSON output")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("benchmark subprocess did not emit a JSON object")


def _run_benchmark_subprocess(
    workload: str,
    overrides: dict[str, str],
    *,
    compare_rust: bool,
    timeout_seconds: float,
    require_backend_prefix: str | None = None,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
) -> dict[str, Any]:
    env = os.environ.copy()
    for key, value in BENCHMARK_ENV_DEFAULTS.items():
        env[key] = value
    for key, value in overrides.items():
        env[key] = value
    attempts = max(int(retry_attempts), 1)
    last_error_message = None
    last_status = "error"
    for attempt in range(1, attempts + 1):
        with tempfile.TemporaryDirectory(prefix="gpu-only-matrix-") as tmpdir:
            json_output_path = Path(tmpdir) / f"{workload}.json"
            stdout_path = Path(tmpdir) / f"{workload}.stdout.log"
            cmd = _benchmark_command(
                workload,
                compare_rust=compare_rust,
                require_backend_prefix=require_backend_prefix,
                json_output_path=json_output_path,
            )
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout_handle:
                    proc = subprocess.run(
                        cmd,
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=stdout_handle,
                        check=False,
                        timeout=max(float(timeout_seconds), 0.1),
                    )
            except subprocess.TimeoutExpired:
                last_status = "timeout"
                last_error_message = f"timed out after {float(timeout_seconds):.1f}s"
                if attempt >= attempts:
                    return {
                        "status": "timeout",
                        "record": None,
                        "error_message": last_error_message,
                        "timeout_seconds": float(timeout_seconds),
                    }
                time.sleep(0.25 * attempt)
                continue

            parsed_record = None
            if json_output_path.exists():
                try:
                    parsed_record = json.loads(json_output_path.read_text())
                except Exception:
                    parsed_record = None
            if parsed_record is None:
                try:
                    parsed_record = _extract_json_record(stdout_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    if proc.returncode == 0:
                        return {
                            "status": "error",
                            "record": None,
                            "error_message": str(exc),
                            "timeout_seconds": float(timeout_seconds),
                        }

            if parsed_record is not None:
                if proc.returncode != 0:
                    parsed_record.setdefault("benchmark_cli_returncode", int(proc.returncode))
                return {
                    "status": "completed",
                    "record": parsed_record,
                    "error_message": None,
                    "timeout_seconds": float(timeout_seconds),
                }

            stdout = stdout_path.read_text(encoding="utf-8").strip() if stdout_path.exists() else ""
            last_error_message = stdout or f"benchmark exited with code {proc.returncode}"
            last_status = "error"
            if proc.returncode >= 0 or attempt >= attempts:
                return {
                    "status": "error",
                    "record": None,
                    "error_message": last_error_message,
                    "timeout_seconds": float(timeout_seconds),
                }
        time.sleep(0.25 * attempt)

    return {
        "status": last_status,
        "record": None,
        "error_message": last_error_message or "benchmark subprocess failed",
        "timeout_seconds": float(timeout_seconds),
    }


class _InprocessTimeoutError(Exception):
    """Raised inside the in-process runner when a workload exceeds its budget."""


def _timeout_supported() -> bool:
    """True if SIGALRM-based timeouts can be armed in the current interpreter."""
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )


@contextmanager
def _inprocess_timeout(timeout_seconds: float | None) -> Iterator[None]:
    """Arm a SIGALRM-based deadline for the duration of the block.

    When the interpreter cannot install a signal handler (non-main thread,
    Windows, etc.) or the timeout is non-positive, the context is a no-op. The
    inner workload is expected to propagate ``_InprocessTimeoutError`` so the
    caller can convert it to a ``status="timeout"`` record.
    """
    if timeout_seconds is None or timeout_seconds <= 0 or not _timeout_supported():
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise _InprocessTimeoutError(
            f"workload exceeded {float(timeout_seconds):.1f}s deadline"
        )

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_benchmark_inprocess(
    workload: str,
    overrides: dict[str, str],
    *,
    compare_rust: bool,
    require_backend_prefix: str | None = None,
    timeout_seconds: float | None = None,
    cpu: Any = None,
) -> dict[str, Any]:
    """Run a single workload in this process, bypassing subprocess/retry overhead.

    Env overrides are applied for the duration of the call and restored on exit.
    Stdout from the benchmark is captured so matrix output stays clean. When
    ``timeout_seconds`` is set and SIGALRM is available on the current thread,
    the workload is aborted with ``status="timeout"`` after the deadline.
    When ``cpu`` is provided, it is reused instead of constructing a fresh
    NeuralCPU so model-loading cost amortizes across workloads.
    """
    from benchmarks.benchmark_gpu_only import benchmark

    saved_env: dict[str, str | None] = {}
    for key, value in overrides.items():
        saved_env[key] = os.environ.get(key)
        os.environ[key] = value
    start = time.perf_counter()
    try:
        with redirect_stdout(io.StringIO()), _inprocess_timeout(timeout_seconds):
            record = benchmark(
                workload,
                compare_rust=compare_rust,
                require_backend_prefix=require_backend_prefix,
                cpu=cpu,
            )
    except _InprocessTimeoutError as exc:
        return {
            "status": "timeout",
            "record": None,
            "error_message": str(exc),
            "timeout_seconds": float(timeout_seconds) if timeout_seconds else None,
            "elapsed_seconds": time.perf_counter() - start,
        }
    except Exception as exc:  # noqa: BLE001 — surface any benchmark failure as an error record
        return {
            "status": "error",
            "record": None,
            "error_message": f"{type(exc).__name__}: {exc}",
            "timeout_seconds": None,
            "elapsed_seconds": time.perf_counter() - start,
        }
    finally:
        for key, original in saved_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    return {
        "status": "completed",
        "record": record,
        "error_message": None,
        "timeout_seconds": float(timeout_seconds) if timeout_seconds else None,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _init_record(workload: str, *, include_torch_baseline: bool) -> dict[str, Any]:
    return {
        "workload": workload,
        "status": "pending",
        "primary_completed": False,
        "torch_baseline_status": "pending" if include_torch_baseline else "skipped",
        "torch_baseline_backend": None,
        "torch_baseline_avg_ips": None,
        "torch_baseline_peak_ips": None,
        "torch_baseline_min_ips": None,
        "torch_baseline_result_ok": None,
        "torch_baseline_insts_ok": None,
        "hotloop_speedup_vs_torch": None,
        "error_message": None,
        "timeout_seconds": None,
        "backend_ok": None,
        "backend_requirement": None,
    }


def _apply_primary_result(record: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    status = outcome["status"]
    record["timeout_seconds"] = outcome.get("timeout_seconds")
    if status != "completed":
        record["status"] = f"primary-{status}"
        record["error_message"] = outcome.get("error_message")
        record["primary_completed"] = False
        record["result_ok"] = False
        record["insts_ok"] = False
        return record

    payload = dict(outcome["record"])
    record.update(payload)
    record["primary_completed"] = True
    if payload.get("backend_ok") is False:
        record["status"] = "backend-mismatch"
        record["error_message"] = (
            f"required backend prefix {payload.get('backend_requirement')!r}, "
            f"observed {payload.get('backend')!r}"
        )
    else:
        record["status"] = "primary-completed"
        record["error_message"] = None
    return record


def _apply_torch_baseline_result(record: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    status = outcome["status"]
    if status != "completed":
        record["torch_baseline_status"] = status
        record["status"] = f"baseline-{status}"
        record["error_message"] = outcome.get("error_message")
        return record

    baseline = dict(outcome["record"])
    torch_avg_ips = float(baseline.get("avg_ips") or 0.0)
    hotloop_avg_ips = float(record.get("avg_ips") or 0.0)
    record["torch_baseline_status"] = "completed"
    record["torch_baseline_backend"] = baseline.get("backend")
    record["torch_baseline_avg_ips"] = torch_avg_ips
    record["torch_baseline_peak_ips"] = float(baseline.get("peak_ips") or 0.0)
    record["torch_baseline_min_ips"] = float(baseline.get("min_ips") or 0.0)
    record["torch_baseline_result_ok"] = bool(baseline.get("result_ok", False))
    record["torch_baseline_insts_ok"] = bool(baseline.get("insts_ok", False))
    record["hotloop_speedup_vs_torch"] = (
        hotloop_avg_ips / torch_avg_ips if hotloop_avg_ips > 0.0 and torch_avg_ips > 0.0 else None
    )
    if record["status"] not in {"backend-mismatch"}:
        record["status"] = "completed"
        record["error_message"] = None
    return record


def _is_resume_complete(record: dict[str, Any], *, include_torch_baseline: bool) -> bool:
    if not isinstance(record, dict) or not record.get("primary_completed"):
        return False
    if record.get("status") == "backend-mismatch":
        return True
    if not include_torch_baseline:
        return True
    return record.get("torch_baseline_status") in {"completed", "skipped"}


def _fmt_int(value: Any) -> str:
    if value in (None, ""):
        return "---"
    return f"{int(round(float(value))):,}"


def _fmt_speedup(value: Any) -> str:
    if value in (None, ""):
        return "---"
    return f"{float(value):.2f}x"


def _write_matrix_artifacts(output_dir: Path, payload: dict, records: list[dict]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gpu_only_matrix.json"
    csv_path = output_dir / "gpu_only_matrix.csv"
    md_path = output_dir / "gpu_only_matrix.md"

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    lines = [
        "# GPU-Only Benchmark Matrix",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| Workload | Status | Avg IPS | Torch IPS | Speedup vs Torch | Backend | Hotloops | Result | Insts |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in records:
        lines.append(
            "| {workload} | {status} | {avg_ips} | {torch_avg_ips} | {speedup_vs_torch} | {backend} | "
            "{hotloop_segments} | {result_status} | {inst_status} |".format(
                workload=row["workload"],
                status=row.get("status", "---"),
                avg_ips=_fmt_int(row.get("avg_ips")),
                torch_avg_ips=_fmt_int(row.get("torch_baseline_avg_ips")),
                speedup_vs_torch=_fmt_speedup(row.get("hotloop_speedup_vs_torch")),
                backend=row.get("backend") or "---",
                hotloop_segments=row.get("hotloop_segments") if row.get("hotloop_segments") is not None else "---",
                result_status="OK" if row.get("result_ok") else "BAD",
                inst_status="OK" if row.get("insts_ok") else "BAD",
            )
        )
        error_message = row.get("error_message")
        if error_message:
            lines.append(f"  - Note: {error_message}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}


def _build_reusable_cpu() -> Any:
    """Construct a single NeuralCPU to share across in-process workloads."""
    from ncpu.neural.cpu import NeuralCPU

    return NeuralCPU(fast_mode=False)


def run_matrix(
    output_dir: Path,
    workloads: list[str] | None = None,
    *,
    resume: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    torch_timeout_seconds: float = DEFAULT_TORCH_TIMEOUT_SECONDS,
    primary_backend: str = "auto",
    require_backend_prefix: str | None = None,
    include_torch_baseline: bool = True,
    compare_rust: bool = True,
    inprocess: bool = False,
    reuse_cpu: bool = False,
) -> dict[str, Any]:
    selected_workloads = list(workloads or BENCHMARK_WORKLOADS)
    json_path = output_dir / "gpu_only_matrix.json"
    existing_by_workload: dict[str, dict[str, Any]] = {}
    if resume and json_path.exists():
        try:
            existing_payload = json.loads(json_path.read_text())
            for row in existing_payload.get("results", []):
                workload = row.get("workload")
                if isinstance(workload, str):
                    existing_by_workload[workload] = dict(row)
        except Exception:
            existing_by_workload = {}

    effective_benchmark_env = {
        "NCPU_GPU_ONLY_HOTLOOP_BACKEND": primary_backend,
        "NCPU_GPU_ONLY_AUTO_ALLOW_CPU": os.environ.get(
            "NCPU_GPU_ONLY_AUTO_ALLOW_CPU",
            BENCHMARK_ENV_DEFAULTS["NCPU_GPU_ONLY_AUTO_ALLOW_CPU"],
        ),
        "NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS": os.environ.get(
            "NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS",
            BENCHMARK_ENV_DEFAULTS["NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS"],
        ),
    }

    reuse_cpu_effective = bool(reuse_cpu) and bool(inprocess)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "benchmark_env": effective_benchmark_env,
        "workloads": selected_workloads,
        "exporter_config": {
            "timeout_seconds": float(timeout_seconds),
            "torch_timeout_seconds": float(torch_timeout_seconds),
            "primary_backend": primary_backend,
            "require_backend_prefix": require_backend_prefix,
            "include_torch_baseline": bool(include_torch_baseline),
            "compare_rust": bool(compare_rust),
            "inprocess": bool(inprocess),
            "reuse_cpu": reuse_cpu_effective,
        },
        "summary": summarize_gpu_only_matrix(None),
        "results": [],
    }

    reusable_cpu: Any = None
    if reuse_cpu_effective:
        reusable_cpu = _build_reusable_cpu()

    records: list[dict[str, Any]] = []
    for workload in selected_workloads:
        existing = dict(existing_by_workload.get(workload, {}))
        if resume and _is_resume_complete(existing, include_torch_baseline=include_torch_baseline):
            record = existing
        else:
            record = existing or _init_record(workload, include_torch_baseline=include_torch_baseline)
            primary_overrides = {"NCPU_GPU_ONLY_HOTLOOP_BACKEND": primary_backend}
            if not record.get("primary_completed"):
                if inprocess:
                    outcome = _run_benchmark_inprocess(
                        workload,
                        primary_overrides,
                        compare_rust=compare_rust,
                        require_backend_prefix=require_backend_prefix,
                        timeout_seconds=timeout_seconds,
                        cpu=reusable_cpu,
                    )
                else:
                    outcome = _run_benchmark_subprocess(
                        workload,
                        primary_overrides,
                        compare_rust=compare_rust,
                        timeout_seconds=timeout_seconds,
                        require_backend_prefix=require_backend_prefix,
                    )
                record = _apply_primary_result(record, outcome)

            if record.get("primary_completed") and include_torch_baseline:
                if record.get("torch_baseline_status") != "completed":
                    if inprocess:
                        baseline = _run_benchmark_inprocess(
                            workload,
                            {"NCPU_GPU_ONLY_HOTLOOP_BACKEND": "0"},
                            compare_rust=False,
                            require_backend_prefix=None,
                            timeout_seconds=torch_timeout_seconds,
                            cpu=reusable_cpu,
                        )
                    else:
                        baseline = _run_benchmark_subprocess(
                            workload,
                            {"NCPU_GPU_ONLY_HOTLOOP_BACKEND": "0"},
                            compare_rust=False,
                            timeout_seconds=torch_timeout_seconds,
                            require_backend_prefix=None,
                        )
                    record = _apply_torch_baseline_result(record, baseline)
            elif record.get("primary_completed") and not include_torch_baseline and record.get("status") != "backend-mismatch":
                record["torch_baseline_status"] = "skipped"
                record["status"] = "completed"
                record["error_message"] = None

        records.append(record)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload["results"] = records
        payload["summary"] = summarize_gpu_only_matrix(payload)
        _write_matrix_artifacts(output_dir, payload, records)

    payload["summary"] = summarize_gpu_only_matrix(payload)
    paths = _write_matrix_artifacts(output_dir, payload, records)
    return {
        "json": paths["json"],
        "csv": paths["csv"],
        "md": paths["md"],
        "records": records,
        "payload": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the GPU-only benchmark matrix")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "results",
        help="Directory for gpu_only_matrix.{json,csv,md} (default: benchmarks/results)",
    )
    parser.add_argument(
        "--workload",
        dest="workloads",
        action="append",
        choices=BENCHMARK_WORKLOADS,
        help="Restrict export to one or more specific workloads",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing gpu_only_matrix.json in the output directory",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout for each primary workload run (default: {DEFAULT_TIMEOUT_SECONDS:.0f}s)",
    )
    parser.add_argument(
        "--torch-timeout-seconds",
        type=float,
        default=DEFAULT_TORCH_TIMEOUT_SECONDS,
        help=f"Timeout for each torch-baseline workload run (default: {DEFAULT_TORCH_TIMEOUT_SECONDS:.0f}s)",
    )
    parser.add_argument(
        "--primary-backend",
        choices=("auto", "rust", "0"),
        default="auto",
        help="Primary NCPU_GPU_ONLY_HOTLOOP_BACKEND value (default: auto)",
    )
    parser.add_argument(
        "--require-backend-prefix",
        default=None,
        help="Require the primary run to report a backend with this prefix",
    )
    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="Require Rust/Metal execution, skip torch baselines, and disable extra comparison sweeps",
    )
    parser.add_argument(
        "--inprocess",
        dest="inprocess",
        action="store_true",
        default=None,
        help="Run primary workloads in-process (default when --rust-only is set)",
    )
    parser.add_argument(
        "--no-inprocess",
        dest="inprocess",
        action="store_false",
        help="Force primary workloads to run as isolated subprocesses",
    )
    parser.add_argument(
        "--reuse-cpu",
        dest="reuse_cpu",
        action="store_true",
        default=False,
        help=(
            "Share a single NeuralCPU across in-process workloads to amortize "
            "model-loading cost (opt-in; may leak per-workload state)"
        ),
    )
    parser.add_argument(
        "--include-torch-baseline",
        dest="include_torch_baseline",
        action="store_true",
        default=True,
        help="Include torch-only baseline rows for speedup reporting",
    )
    parser.add_argument(
        "--no-torch-baseline",
        dest="include_torch_baseline",
        action="store_false",
        help="Skip torch-only baseline runs",
    )
    parser.add_argument(
        "--compare-rust",
        dest="compare_rust",
        action="store_true",
        default=True,
        help="Include the benchmark script's extra Rust/Metal comparison sweep",
    )
    parser.add_argument(
        "--no-compare-rust",
        dest="compare_rust",
        action="store_false",
        help="Skip the benchmark script's extra Rust/Metal comparison sweep",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    primary_backend = args.primary_backend
    require_backend_prefix = args.require_backend_prefix
    include_torch_baseline = args.include_torch_baseline
    compare_rust = args.compare_rust
    if args.rust_only:
        primary_backend = "rust"
        require_backend_prefix = "rust"
        include_torch_baseline = False
        compare_rust = False

    if args.inprocess is None:
        inprocess = bool(args.rust_only)
    else:
        inprocess = bool(args.inprocess)

    result = run_matrix(
        output_dir,
        workloads=args.workloads,
        resume=args.resume,
        timeout_seconds=args.timeout_seconds,
        torch_timeout_seconds=args.torch_timeout_seconds,
        primary_backend=primary_backend,
        require_backend_prefix=require_backend_prefix,
        include_torch_baseline=include_torch_baseline,
        compare_rust=compare_rust,
        inprocess=inprocess,
        reuse_cpu=args.reuse_cpu,
    )
    if args.rust_only:
        summary = result["payload"]["summary"]
        if not summary.get("strict_rust_only", False):
            print("[gpu-only-matrix] Strict Rust/Metal validation failed", file=sys.stderr)
            for issue in summary.get("contract_issues", []):
                print(f"  - {issue}", file=sys.stderr)
            return 1
    print(f"Wrote {result['json']}")
    print(f"Wrote {result['csv']}")
    print(f"Wrote {result['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
