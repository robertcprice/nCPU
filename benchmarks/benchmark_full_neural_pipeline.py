#!/usr/bin/env python3
"""Benchmark the strict bottom-up full-neural pipeline.

Runs a small workload suite through ``demos/neural/full_neural_demo.py`` and
collects machine-readable summaries for paper-facing benchmarking. Each
workload emits:

- a rendered terminal PNG
- a per-run JSON summary from the full-neural demo
- a stdout/stderr log

The benchmark then aggregates those per-run artifacts into:

- ``full_neural_pipeline.json``: aggregate metrics + per-workload records
- ``full_neural_pipeline.md``: compact markdown table for review and paper prep
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from demos.neural.full_neural_demo import available_workloads, resolve_workload
from ncpu.utils.provenance import collect_provenance


BENCHMARK_NAME = "full_neural_pipeline"
DEMO_SCRIPT = PROJECT_ROOT / "demos" / "neural" / "full_neural_demo.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "local" / BENCHMARK_NAME


def _fmt_int(value: Any) -> str:
    if value in (None, ""):
        return "---"
    return f"{int(round(float(value))):,}"


def _fmt_float(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return "---"
    return f"{float(value):.{digits}f}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_workload(
    workload: str,
    *,
    output_dir: Path,
    device: str | None = None,
    max_instructions: int | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    spec = resolve_workload(workload)
    run_dir = output_dir / workload
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    frame_path = run_dir / "frame.png"
    log_path = run_dir / "run.log"

    argv = [
        python_executable or sys.executable,
        str(DEMO_SCRIPT),
        "--workload",
        workload,
        "--output",
        str(frame_path),
        "--summary-json",
        str(summary_path),
    ]
    if device:
        argv.extend(["--device", device])
    if max_instructions is not None:
        argv.extend(["--max-instructions", str(max_instructions)])

    proc = subprocess.run(
        argv,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    log_chunks = [proc.stdout.rstrip()]
    if proc.stderr:
        log_chunks.append("[stderr]")
        log_chunks.append(proc.stderr.rstrip())
    log_path.write_text("\n".join(chunk for chunk in log_chunks if chunk) + "\n", encoding="utf-8")

    record: dict[str, Any] = {
        "workload": workload,
        "workload_title": spec.title,
        "loop_counts": list(spec.loop_counts),
        "expected_total_iterations": spec.expected_total,
        "status": "failed",
        "returncode": proc.returncode,
        "command": argv,
        "frame_path": str(frame_path),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
    }

    if proc.returncode != 0:
        record["error_message"] = f"demo exited with return code {proc.returncode}"
        return record

    if not summary_path.exists():
        record["error_message"] = "demo completed without writing summary JSON"
        return record

    try:
        summary = _load_json(summary_path)
    except Exception as exc:
        record["error_message"] = f"failed to parse summary JSON: {exc}"
        return record

    record.update(
        {
            "status": "completed" if summary.get("counter_verified") else "verification_failed",
            "device": summary.get("device"),
            "display_params": summary.get("display_params"),
            "executed_instructions": summary.get("executed_instructions"),
            "execution_time_s": summary.get("execution_time_s"),
            "render_time_ms": summary.get("render_time_ms"),
            "throughput_ips": summary.get("throughput_ips"),
            "counter_expected": summary.get("counter_expected"),
            "counter_x10": summary.get("counter_x10"),
            "counter_x12": summary.get("counter_x12"),
            "counter_verified": summary.get("counter_verified"),
            "demo_output_path": summary.get("output_path"),
            "workload_note": summary.get("workload_note"),
        }
    )
    return record


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in records if row.get("status") == "completed"]
    throughputs = [float(row["throughput_ips"]) for row in completed if row.get("throughput_ips") is not None]
    render_times = [float(row["render_time_ms"]) for row in completed if row.get("render_time_ms") is not None]
    execution_times = [float(row["execution_time_s"]) for row in completed if row.get("execution_time_s") is not None]

    return {
        "requested": len(records),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "all_counters_verified": bool(records) and len(completed) == len(records),
        "avg_throughput_ips": statistics.mean(throughputs) if throughputs else None,
        "median_throughput_ips": statistics.median(throughputs) if throughputs else None,
        "max_throughput_ips": max(throughputs) if throughputs else None,
        "min_throughput_ips": min(throughputs) if throughputs else None,
        "avg_render_time_ms": statistics.mean(render_times) if render_times else None,
        "avg_execution_time_s": statistics.mean(execution_times) if execution_times else None,
        "total_executed_instructions": sum(
            int(row.get("executed_instructions", 0) or 0) for row in completed
        ),
    }


def build_payload(records: list[dict[str, Any]], *, requested_workloads: list[str]) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_workloads": list(requested_workloads),
        "provenance": collect_provenance(
            PROJECT_ROOT,
            argv=[sys.argv[0], *sys.argv[1:]],
            extra={"benchmark": BENCHMARK_NAME, "workloads": list(requested_workloads)},
        ),
        "summary": _build_summary(records),
        "results": records,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Full Neural Pipeline Benchmark",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Requested workloads: {payload['summary']['requested']}",
        f"- Completed workloads: {payload['summary']['completed']}",
        f"- Failed workloads: {payload['summary']['failed']}",
        f"- All counters verified: `{payload['summary']['all_counters_verified']}`",
        f"- Avg throughput: {_fmt_int(payload['summary']['avg_throughput_ips'])} IPS",
        f"- Median throughput: {_fmt_int(payload['summary']['median_throughput_ips'])} IPS",
        f"- Avg render time: {_fmt_float(payload['summary']['avg_render_time_ms'], 1)} ms",
        "",
        "## Workloads",
        "",
        "| Workload | Loops | Expected | Instr | IPS | Exec s | Render ms | Counter | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["results"]:
        loops = "+".join(str(count) for count in row.get("loop_counts", [])) or "---"
        counter = "OK" if row.get("counter_verified") else "BAD"
        lines.append(
            "| {workload} | {loops} | {expected} | {inst} | {ips} | {exec_s} | {render_ms} | {counter} | {status} |".format(
                workload=row["workload"],
                loops=loops,
                expected=_fmt_int(row.get("expected_total_iterations")),
                inst=_fmt_int(row.get("executed_instructions")),
                ips=_fmt_int(row.get("throughput_ips")),
                exec_s=_fmt_float(row.get("execution_time_s"), 3),
                render_ms=_fmt_float(row.get("render_time_ms"), 1),
                counter=counter,
                status=row.get("status", "---"),
            )
        )
        if row.get("error_message"):
            lines.append(f"  - Note: {row['error_message']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_artifacts(output_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{BENCHMARK_NAME}.json"
    md_path = output_dir / f"{BENCHMARK_NAME}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"json": json_path, "md": md_path}


def run_suite(
    *,
    workloads: list[str],
    output_dir: Path,
    device: str | None = None,
    max_instructions: int | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for workload in workloads:
        print(f"[full-neural] running workload: {workload}")
        record = run_workload(
            workload,
            output_dir=output_dir,
            device=device,
            max_instructions=max_instructions,
        )
        records.append(record)
        status = record.get("status", "unknown")
        ips = record.get("throughput_ips")
        if ips is not None and status == "completed":
            print(f"[full-neural] {workload}: {_fmt_int(ips)} IPS")
        else:
            print(f"[full-neural] {workload}: {status}")
        if fail_fast and status != "completed":
            break

    payload = build_payload(records, requested_workloads=workloads)
    paths = write_artifacts(output_dir, payload)
    print(f"[full-neural] wrote {paths['json']}")
    print(f"[full-neural] wrote {paths['md']}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the strict bottom-up full-neural pipeline")
    parser.add_argument(
        "--workloads",
        nargs="*",
        default=available_workloads(),
        help="Named workloads to run (default: full suite)",
    )
    parser.add_argument("--device", help="Forwarded device override for the full-neural demo")
    parser.add_argument("--max-instructions", type=int, default=None, help="Forwarded instruction budget")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for aggregate artifacts and per-workload outputs",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed workload")
    parser.add_argument("--list-workloads", action="store_true", help="Print available workloads and exit")
    args = parser.parse_args()

    if args.list_workloads:
        print("Available workloads:")
        for name in available_workloads():
            spec = resolve_workload(name)
            print(f"- {name}: {spec.title} | loops={','.join(str(count) for count in spec.loop_counts)}")
        return 0

    invalid = [name for name in args.workloads if name not in available_workloads()]
    if invalid:
        parser.error(f"Unknown workloads: {', '.join(invalid)}")

    payload = run_suite(
        workloads=list(args.workloads),
        output_dir=args.output_dir.resolve(),
        device=args.device,
        max_instructions=args.max_instructions,
        fail_fast=args.fail_fast,
    )
    return 0 if payload["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
