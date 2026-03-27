#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUST_DIR = ROOT / "kernels" / "rust_metal"


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main() -> int:
    proc = run(["cargo", "check", "--bin", "ncpu_run"], RUST_DIR)
    ok = proc.returncode == 0

    stderr_lines = [line for line in proc.stderr.strip().splitlines() if line.strip()]
    stdout_lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]

    payload = {
        "metric": "rust_ncpu_run_check",
        "value": 1.0 if ok else 0.0,
        "ok": ok,
        "command": "cargo check --bin ncpu_run",
        "workdir": str(RUST_DIR),
        "returncode": proc.returncode,
        "stdout_tail": stdout_lines[-10:],
        "stderr_tail": stderr_lines[-20:],
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
