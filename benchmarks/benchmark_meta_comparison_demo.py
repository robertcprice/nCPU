#!/usr/bin/env python3
"""Generate a reproducible artifact bundle for the neural-vs-Meta comparison demo.

Runs ``demos/neural/meta_comparison_demo.py`` in scripted mode so the left pane
executes a fixed shell transcript while the visible content area remains the
neural-rendered comparison window. The benchmark captures:

- per-step PNGs
- a final composite PNG
- a machine-readable summary from the demo
- a decoded shell log
- stdout/stderr from the run

It then writes aggregate artifacts:

- ``meta_comparison_demo.json``
- ``meta_comparison_demo.md``
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.utils.provenance import collect_provenance


BENCHMARK_NAME = "meta_comparison_demo"
DEMO_SCRIPT = PROJECT_ROOT / "demos" / "neural" / "meta_comparison_demo.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "local" / BENCHMARK_NAME
DEFAULT_SHELL = "/bin/sh"
DEFAULT_LEFT_RUNTIME = "pty"
DEFAULT_COMMANDS_BY_RUNTIME = {
    "pty": [
        "pwd",
        "echo \"2+2=$((2+2))\"",
        "python3 --version",
        "printf '%s\\n' beta alpha | sort",
    ],
    "neural-os": [
        "pwd",
        "echo 2+2=4",
        "ls /home/user",
        "cat /home/user/README.txt",
    ],
}


def default_commands_for_runtime(left_runtime: str) -> list[str]:
    try:
        return list(DEFAULT_COMMANDS_BY_RUNTIME[left_runtime])
    except KeyError as exc:
        known = ", ".join(sorted(DEFAULT_COMMANDS_BY_RUNTIME))
        raise ValueError(f"Unknown left runtime '{left_runtime}'. Known runtimes: {known}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "---"


def run_demo(
    *,
    output_dir: Path,
    shell: str = DEFAULT_SHELL,
    left_runtime: str = DEFAULT_LEFT_RUNTIME,
    commands: list[str] | None = None,
    device: str | None = None,
    python_executable: str | None = None,
    boot_delay_ms: int = 1200,
    step_delay_ms: int = 1000,
    final_hold_ms: int = 800,
) -> dict[str, Any]:
    commands = list(commands or default_commands_for_runtime(left_runtime))
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_dir = output_dir / "captures"
    summary_path = output_dir / "summary.json"
    final_frame_path = output_dir / "final.png"
    shell_log_path = output_dir / "shell.log"
    run_log_path = output_dir / "run.log"

    argv = [
        python_executable or sys.executable,
        str(DEMO_SCRIPT),
        "--left-runtime",
        left_runtime,
        "--shell",
        shell,
        "--capture-dir",
        str(capture_dir),
        "--summary-json",
        str(summary_path),
        "--shell-log",
        str(shell_log_path),
        "--output",
        str(final_frame_path),
        "--boot-delay-ms",
        str(boot_delay_ms),
        "--step-delay-ms",
        str(step_delay_ms),
        "--final-hold-ms",
        str(final_hold_ms),
    ]
    if device:
        argv.extend(["--device", device])
    for command in commands:
        argv.extend(["--command", command])

    env = dict(os.environ)
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("SDL_AUDIODRIVER", "dummy")

    proc = subprocess.run(
        argv,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    log_chunks = [proc.stdout.rstrip()]
    if proc.stderr:
        log_chunks.append("[stderr]")
        log_chunks.append(proc.stderr.rstrip())
    run_log_path.write_text("\n".join(chunk for chunk in log_chunks if chunk) + "\n", encoding="utf-8")

    record: dict[str, Any] = {
        "status": "failed",
        "returncode": proc.returncode,
        "command": argv,
        "left_runtime": left_runtime,
        "shell": shell,
        "script_commands": commands,
        "capture_dir": str(capture_dir),
        "summary_path": str(summary_path),
        "final_frame_path": str(final_frame_path),
        "shell_log_path": str(shell_log_path),
        "run_log_path": str(run_log_path),
    }

    if proc.returncode != 0:
        record["error_message"] = f"demo exited with return code {proc.returncode}"
        return record
    if not summary_path.exists():
        record["error_message"] = "demo completed without writing summary JSON"
        return record

    summary = _load_json(summary_path)
    record.update(
        {
            "status": "completed",
            "mode": summary.get("mode"),
            "device": summary.get("device"),
            "backend": summary.get("backend"),
            "metal_active": summary.get("metal_active"),
            "renderer_params": summary.get("renderer_params"),
            "frames_rendered": summary.get("frames_rendered"),
            "elapsed_s": summary.get("elapsed_s"),
            "interactive_left_pane": summary.get("interactive_left_pane"),
            "visible_content_neural_only": summary.get("visible_content_neural_only"),
            "reference_right_pane_not_meta_output": summary.get("reference_right_pane_not_meta_output"),
            "left_pane_computation_owned_by_ncpu": summary.get("left_pane_computation_owned_by_ncpu"),
            "left_runtime_label": summary.get("left_runtime_label"),
            "left_runtime_compute_path": summary.get("left_runtime_compute_path"),
            "avg_first_output_latency_ms": summary.get("avg_first_output_latency_ms"),
            "avg_capture_latency_ms": summary.get("avg_capture_latency_ms"),
            "captures": summary.get("captures", []),
            "commands_run": summary.get("commands", []),
            "output_path": summary.get("output_path"),
        }
    )
    return record


def build_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": collect_provenance(
            PROJECT_ROOT,
            argv=[sys.argv[0], *sys.argv[1:]],
            extra={
                "benchmark": BENCHMARK_NAME,
                "left_runtime": record.get("left_runtime"),
                "commands": record.get("script_commands", []),
            },
        ),
        "result": record,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    result = payload["result"]
    commands = result.get("commands_run", [])
    captures = result.get("captures", [])

    lines = [
        "# Meta Comparison Demo Benchmark",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Left runtime: `{result.get('left_runtime')}`",
        f"- Left runtime label: `{result.get('left_runtime_label')}`",
        f"- Shell: `{result.get('shell')}`",
        f"- Device: `{result.get('device')}`",
        f"- Backend: `{result.get('backend')}`",
        f"- Mode: `{result.get('mode')}`",
        f"- Frames rendered: `{result.get('frames_rendered')}`",
        f"- Elapsed seconds: `{result.get('elapsed_s')}`",
        f"- Left pane interactive: `{_fmt_bool(result.get('interactive_left_pane'))}`",
        f"- Left pane computation owned by nCPU: `{_fmt_bool(result.get('left_pane_computation_owned_by_ncpu'))}`",
        f"- Visible content neural-only: `{_fmt_bool(result.get('visible_content_neural_only'))}`",
        f"- Right pane is reference, not Meta output: `{_fmt_bool(result.get('reference_right_pane_not_meta_output'))}`",
        f"- Avg first-output latency ms: `{result.get('avg_first_output_latency_ms')}`",
        f"- Avg capture latency ms: `{result.get('avg_capture_latency_ms')}`",
        f"- Captures written: `{len(captures)}`",
        "",
        "## Commands",
        "",
        "| # | Command | First Output ms | Capture ms | Capture |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in commands:
        lines.append(
            "| {index} | `{command}` | {first_output} | {capture_latency} | `{capture}` |".format(
                index=row.get("index", "---"),
                command=str(row.get("command", "")).replace("`", "\\`"),
                first_output=(
                    f"{float(row['first_output_latency_ms']):.1f}"
                    if row.get("first_output_latency_ms") is not None
                    else "---"
                ),
                capture_latency=(
                    f"{float(row['capture_latency_ms']):.1f}"
                    if row.get("capture_latency_ms") is not None
                    else "---"
                ),
                capture=row.get("capture_path", "---"),
            )
        )
    if result.get("error_message"):
        lines.extend(["", f"Note: {result['error_message']}"])
    lines.append("")
    return "\n".join(lines) + "\n"


def write_artifacts(output_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{BENCHMARK_NAME}.json"
    md_path = output_dir / f"{BENCHMARK_NAME}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"json": json_path, "md": md_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the neural-vs-Meta comparison demo")
    parser.add_argument("--shell", default=DEFAULT_SHELL, help="Shell to use in the interactive pane")
    parser.add_argument(
        "--left-runtime",
        choices=tuple(sorted(DEFAULT_COMMANDS_BY_RUNTIME)),
        default=DEFAULT_LEFT_RUNTIME,
        help="Left pane runtime to benchmark",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Scripted command to run; may be repeated (default: built-in transcript)",
    )
    parser.add_argument("--device", help="Forwarded device override for the comparison demo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for aggregate artifacts and the captured demo bundle",
    )
    parser.add_argument("--boot-delay-ms", type=int, default=None, help="Delay before first scripted capture/send")
    parser.add_argument("--step-delay-ms", type=int, default=None, help="Delay between scripted send and capture")
    parser.add_argument("--final-hold-ms", type=int, default=None, help="Delay after the final scripted capture")
    args = parser.parse_args()

    default_boot_delay_ms = 1800 if args.left_runtime == "neural-os" else 1200
    default_step_delay_ms = 1800 if args.left_runtime == "neural-os" else 1000
    default_final_hold_ms = 1200 if args.left_runtime == "neural-os" else 800
    record = run_demo(
        output_dir=args.output_dir.resolve(),
        shell=args.shell,
        left_runtime=args.left_runtime,
        commands=list(args.command) or default_commands_for_runtime(args.left_runtime),
        device=args.device,
        boot_delay_ms=args.boot_delay_ms if args.boot_delay_ms is not None else default_boot_delay_ms,
        step_delay_ms=args.step_delay_ms if args.step_delay_ms is not None else default_step_delay_ms,
        final_hold_ms=args.final_hold_ms if args.final_hold_ms is not None else default_final_hold_ms,
    )
    payload = build_payload(record)
    paths = write_artifacts(args.output_dir.resolve(), payload)
    print(f"[meta-compare] wrote {paths['json']}")
    print(f"[meta-compare] wrote {paths['md']}")
    return 0 if record.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
