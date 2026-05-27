#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUST_DIR = ROOT / "kernels" / "rust_metal"
RUNNER_RS = RUST_DIR / "bin" / "ncpu_run.rs"
README = ROOT / "README.md"
RUST_PLAN = ROOT / "docs" / "plans" / "2026-03-26-rust-gpu-next-iterations.md"


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def count_warnings(stderr_lines: list[str]) -> int:
    return sum(1 for line in stderr_lines if line.startswith("warning:"))


def feature_score() -> tuple[float, dict[str, bool]]:
    text = RUNNER_RS.read_text() if RUNNER_RS.exists() else ""
    readme = README.read_text() if README.exists() else ""
    plan = RUST_PLAN.read_text() if RUST_PLAN.exists() else ""

    checks = {
        "inspect_flag": "--inspect" in text,
        "json_report_flag": "--json-report" in text,
        "benchmark_flag": "--benchmark" in text,
        "repeat_flag": "--repeat" in text,
        "aggregate_reporting": "mean_" in text or "aggregate" in text.lower(),
        "rust_docs": "ncpu_run" in readme and "inspect --json-report" in readme,
        "rust_plan": "benchmark" in plan.lower() and "aggregate" in plan.lower(),
    }
    score = sum(1.0 for ok in checks.values() if ok)
    return score, checks


def main() -> int:
    proc = run(["cargo", "check", "--bin", "ncpu_run"], RUST_DIR)
    ok = proc.returncode == 0

    stderr_lines = [line for line in proc.stderr.strip().splitlines() if line.strip()]
    stdout_lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    warnings = count_warnings(stderr_lines)
    feat_score, checks = feature_score()

    # Score with headroom so Hermes Lab can keep iterating:
    # - 10 points for passing cargo check
    # - +1 per Rust-side launcher/reporting milestone present
    # - small warning penalty
    value = (10.0 if ok else 0.0) + feat_score - min(warnings * 0.01, 5.0)

    payload = {
        "metric": "rust_ncpu_run_iteration_score",
        "value": round(value, 4),
        "ok": ok,
        "command": "cargo check --bin ncpu_run",
        "workdir": str(RUST_DIR),
        "returncode": proc.returncode,
        "warning_count": warnings,
        "feature_checks": checks,
        "stdout_tail": stdout_lines[-10:],
        "stderr_tail": stderr_lines[-20:],
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
