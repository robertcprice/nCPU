import json
from pathlib import Path
import subprocess
import sys

from ncpu.utils.paper_tables import write_paper_tables


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _make_artifact_dir(root: Path) -> None:
    provenance = {
        "platform": {
            "system": "Darwin",
            "machine": "arm64",
            "release": "test",
            "platform": "Darwin-arm64",
        },
        "mog": {"commit": "abc123"},
    }

    _write_json(
        root / "benchmarks/baseline_comparison_results.json",
        {
            "metadata": {"trials": 3, "workload_commands": 5},
            "provenance": provenance,
            "conventional": {"ips_gpu_only": 400000, "ips_gpu_only_ci95": 10000, "neural_inferences": 0},
            "neural_enhanced": {
                "ips_gpu_only": 300000,
                "ips_gpu_only_ci95": 8000,
                "neural_inferences": 239,
                "models_active": 9,
            },
            "overhead_pct": 25.0,
            "output_comparison": {"match_percentage": 100.0, "identical": True},
        },
    )
    _write_json(
        root / "benchmarks/ablation_results.json",
        {
            "metadata": {"trials": 3, "configurations": ["baseline (0 models)", "all 9 models"]},
            "provenance": provenance,
            "results": [
                {
                    "name": "baseline (0 models)",
                    "n_models": 0,
                    "ips_gpu_only": 400000,
                    "ips_gpu_only_ci95": 10000,
                    "neural_inferences": 0,
                    "gpu_time_s": 0.100,
                },
                {
                    "name": "all 9 models",
                    "n_models": 9,
                    "ips_gpu_only": 300000,
                    "ips_gpu_only_ci95": 8000,
                    "neural_inferences": 239,
                    "gpu_time_s": 0.150,
                },
            ],
        },
    )
    _write_json(
        root / "benchmarks/real_workload_results.json",
        {
            "metadata": {"trials": 3},
            "provenance": provenance,
            "aggregate_overhead": {
                "mean": 20.0,
                "ci95_lo": 18.0,
                "ci95_hi": 22.0,
            },
            "shell_workload": {
                "conventional": {
                    "ips_gpu_only": {"mean": 410000},
                    "neural_inferences": {"mean": 0},
                },
                "neural": {
                    "ips_gpu_only": {"mean": 305000},
                    "neural_inferences": {"mean": 33},
                },
            },
            "compiler_workloads": {
                "fibonacci": {
                    "conventional": {
                        "ips": {"mean": 800000},
                        "compile_time_s": {"mean": 0.120},
                        "exec_time_s": {"mean": 0.010},
                        "compiled_ok": True,
                        "exec_ok": True,
                    },
                    "neural": {
                        "ips": {"mean": 620000, "ci95_lo": 600000, "ci95_hi": 640000},
                        "compile_time_s": {"mean": 0.160},
                        "exec_time_s": {"mean": 0.011},
                        "neural_inferences": {"mean": 15},
                        "compiled_ok": True,
                        "exec_ok": True,
                    },
                }
            },
        },
    )
    _write_json(
        root / "benchmarks/gpu_only_matrix.json",
        {
            "generated_at": "2026-04-16T00:00:00+00:00",
            "benchmark_env": {
                "NCPU_GPU_ONLY_HOTLOOP_BACKEND": "rust",
                "NCPU_GPU_ONLY_AUTO_ALLOW_CPU": "1",
                "NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS": "1",
            },
            "exporter_config": {
                "primary_backend": "rust",
                "require_backend_prefix": "rust",
                "include_torch_baseline": False,
                "compare_rust": False,
            },
            "results": [
                {
                    "workload": "adjacent-bytecopy",
                    "avg_ips": 898334.4,
                    "status": "completed",
                    "primary_completed": True,
                    "backend": "rust-hotloop",
                    "backend_ok": True,
                    "hotloop_segments": 2,
                    "result_ok": True,
                    "insts_ok": True,
                    "torch_baseline_status": "skipped",
                },
                {
                    "workload": "adjacent-bytecopy-bge-exit",
                    "avg_ips": 994394.1,
                    "status": "completed",
                    "primary_completed": True,
                    "backend": "rust-hotloop",
                    "backend_ok": True,
                    "hotloop_segments": 2,
                    "result_ok": True,
                    "insts_ok": True,
                    "torch_baseline_status": "skipped",
                },
            ],
        },
    )


def test_write_paper_tables_creates_metrics_and_markdown(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    _make_artifact_dir(artifact_dir)

    paths = write_paper_tables(artifact_dir)

    metrics = json.loads(paths["metrics"].read_text())
    markdown = paths["markdown"].read_text()

    assert metrics["baseline_comparison"]["neural_overhead_pct"] == 25.0
    assert metrics["real_workload"]["aggregate_overhead"]["mean"] == 20.0
    assert metrics["gpu_only_matrix"]["rows"][0]["workload"] == "adjacent-bytecopy"
    assert metrics["gpu_only_matrix"]["strict_rust_only"] is True
    assert "## Baseline Comparison" in markdown
    assert "## GPU-Only Hotloop Matrix" in markdown
    assert "| Neural-enhanced | 300,000 | 300,000 +/- 8,000 | 239 | 100.0% | 25.0% |" in markdown
    assert "| adjacent-bytecopy | 898,334 | rust-hotloop | 2 | OK | OK |" in markdown
    assert "Strict Rust/Metal mode: `True`." in markdown
    assert "Aggregate compute-workload overhead: 20.0%" in markdown


def test_extract_paper_tables_cli_writes_outputs(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    output_dir = tmp_path / "tables"
    _make_artifact_dir(artifact_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/release/extract_paper_tables.py"),
            "--artifact-dir",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "paper_metrics.json").is_file()
    assert (output_dir / "paper_tables.md").is_file()
