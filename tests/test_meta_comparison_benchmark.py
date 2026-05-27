from __future__ import annotations

import json
from pathlib import Path

from benchmarks import benchmark_meta_comparison_demo as bench


def test_default_meta_compare_commands_are_present():
    assert bench.default_commands_for_runtime("pty") == [
        "pwd",
        "echo \"2+2=$((2+2))\"",
        "python3 --version",
        "printf '%s\\n' beta alpha | sort",
    ]
    assert bench.default_commands_for_runtime("neural-os") == [
        "pwd",
        "echo 2+2=4",
        "ls /home/user",
        "cat /home/user/README.txt",
    ]


def test_build_payload_and_write_artifacts(tmp_path: Path):
    record = {
        "status": "completed",
        "left_runtime": "neural-os",
        "left_runtime_label": "nCPU GPU shell",
        "shell": "/bin/sh",
        "device": "cpu",
        "backend": "PyTorch",
        "mode": "scripted",
        "frames_rendered": 123,
        "elapsed_s": 4.25,
        "interactive_left_pane": True,
        "left_pane_computation_owned_by_ncpu": True,
        "visible_content_neural_only": True,
        "reference_right_pane_not_meta_output": True,
        "avg_first_output_latency_ms": 88.0,
        "avg_capture_latency_ms": 125.0,
        "captures": [
            {"label": "boot", "path": "/tmp/boot.png"},
            {"label": "01_pwd", "path": "/tmp/01_pwd.png"},
        ],
        "commands_run": [
            {"index": 1, "command": "pwd", "first_output_latency_ms": 50.0, "capture_latency_ms": 75.0, "capture_path": "/tmp/01_pwd.png"},
            {"index": 2, "command": "cc hello.c", "first_output_latency_ms": 126.0, "capture_latency_ms": 175.0, "capture_path": "/tmp/02_cc-hello-c.png"},
        ],
        "script_commands": ["pwd", "cc hello.c"],
    }

    payload = bench.build_payload(record)
    paths = bench.write_artifacts(tmp_path, payload)

    data = json.loads(paths["json"].read_text())
    markdown = paths["md"].read_text()

    assert data["benchmark"] == "meta_comparison_demo"
    assert data["result"]["status"] == "completed"
    assert data["result"]["visible_content_neural_only"] is True
    assert data["result"]["left_runtime"] == "neural-os"
    assert "## Commands" in markdown
    assert "| 1 | `pwd` | 50.0 | 75.0 | `/tmp/01_pwd.png` |" in markdown
    assert "Visible content neural-only: `yes`" in markdown
    assert "Left pane computation owned by nCPU: `yes`" in markdown
