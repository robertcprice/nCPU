import json
from pathlib import Path
import subprocess
import sys

from ncpu.utils.artifact_compare import compare_artifact_dirs


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _make_artifact_dir(
    root: Path,
    *,
    platform_system: str = "Darwin",
    platform_machine: str = "arm64",
    mog_commit: str = "abc123",
    baseline_conv_ips: float = 1000.0,
    baseline_neural_ips: float = 800.0,
    baseline_overhead: float = 20.0,
    output_match_percentage: float = 100.0,
    output_identical: bool = True,
    ablation_all_models_ips: float = 720.0,
    real_workload_overhead: float = 24.0,
    shell_neural_ips: float = 760.0,
    workload_neural_ips: float = 700.0,
    workload_neural_exec_ok: bool = True,
    gpu_only_counted_ips: float = 900.0,
    gpu_only_adjacent_ips: float = 850.0,
    gpu_only_adjacent_segments: int = 2,
    gpu_only_backend: str = "rust-hotloop",
    gpu_only_backend_ok: bool = True,
    gpu_only_primary_backend: str = "rust",
    gpu_only_require_backend_prefix: str | None = "rust",
    gpu_only_include_torch_baseline: bool = False,
) -> None:
    provenance = {
        "platform": {
            "system": platform_system,
            "machine": platform_machine,
            "release": "test-release",
            "platform": f"{platform_system}-{platform_machine}",
        },
        "mog": {"commit": mog_commit},
    }

    _write_json(
        root / "artifact_manifest.json",
        {
            "label": "test-artifacts",
            "output_dir": str(root),
            "provenance": provenance,
            "files": [],
        },
    )
    _write_json(
        root / "benchmarks/baseline_comparison_results.json",
        {
            "metadata": {"trials": 1, "workload_commands": 1},
            "provenance": provenance,
            "conventional": {"ips_gpu_only": baseline_conv_ips},
            "neural_enhanced": {"ips_gpu_only": baseline_neural_ips},
            "overhead_pct": baseline_overhead,
            "output_comparison": {
                "conv_lines": 10,
                "neural_lines": 10,
                "matching_lines": 10,
                "match_percentage": output_match_percentage,
                "identical": output_identical,
            },
        },
    )
    _write_json(
        root / "benchmarks/ablation_results.json",
        {
            "metadata": {
                "trials": 1,
                "workload_commands": 1,
                "configurations": ["baseline (0 models)", "all 9 models"],
            },
            "provenance": provenance,
            "results": [
                {"name": "baseline (0 models)", "ips_gpu_only": baseline_conv_ips},
                {"name": "all 9 models", "ips_gpu_only": ablation_all_models_ips},
            ],
        },
    )
    _write_json(
        root / "benchmarks/real_workload_results.json",
        {
            "metadata": {"trials": 1},
            "provenance": provenance,
            "aggregate_overhead": {
                "mean": real_workload_overhead,
                "std": 0,
                "stderr": 0,
                "ci95_lo": real_workload_overhead,
                "ci95_hi": real_workload_overhead,
                "n": 1,
            },
            "shell_workload": {
                "conventional": {
                    "ips_gpu_only": {"mean": baseline_conv_ips},
                },
                "neural": {
                    "ips_gpu_only": {"mean": shell_neural_ips},
                },
            },
            "compiler_workloads": {
                "fibonacci": {
                    "conventional": {
                        "ips": {"mean": baseline_conv_ips},
                        "compiled_ok": True,
                        "exec_ok": True,
                    },
                    "neural": {
                        "ips": {"mean": workload_neural_ips},
                        "compiled_ok": True,
                        "exec_ok": workload_neural_exec_ok,
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
                "NCPU_GPU_ONLY_HOTLOOP_BACKEND": gpu_only_primary_backend,
                "NCPU_GPU_ONLY_AUTO_ALLOW_CPU": "1",
                "NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS": "1",
            },
            "exporter_config": {
                "primary_backend": gpu_only_primary_backend,
                "require_backend_prefix": gpu_only_require_backend_prefix,
                "include_torch_baseline": gpu_only_include_torch_baseline,
                "compare_rust": False,
            },
            "results": [
                {
                    "workload": "counted",
                    "status": "completed",
                    "primary_completed": True,
                    "avg_ips": gpu_only_counted_ips,
                    "backend": gpu_only_backend,
                    "backend_ok": gpu_only_backend_ok,
                    "hotloop_segments": 1,
                    "result_ok": True,
                    "insts_ok": True,
                    "torch_baseline_status": "skipped" if not gpu_only_include_torch_baseline else "completed",
                },
                {
                    "workload": "adjacent-counted",
                    "status": "completed",
                    "primary_completed": True,
                    "avg_ips": gpu_only_adjacent_ips,
                    "backend": gpu_only_backend,
                    "backend_ok": gpu_only_backend_ok,
                    "hotloop_segments": gpu_only_adjacent_segments,
                    "result_ok": True,
                    "insts_ok": True,
                    "torch_baseline_status": "skipped" if not gpu_only_include_torch_baseline else "completed",
                },
            ],
        },
    )


def test_compare_artifact_dirs_passes_for_identical_runs(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _make_artifact_dir(baseline_dir)
    _make_artifact_dir(candidate_dir)

    report = compare_artifact_dirs(baseline_dir, candidate_dir)

    assert report["summary"]["failed"] == 0
    assert report["summary"]["overall_status"] == "pass"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["gpu_only_matrix.candidate.strict_rust_only"]["status"] == "pass"


def test_compare_artifact_dirs_flags_overhead_regression(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _make_artifact_dir(baseline_dir, baseline_overhead=20.0)
    _make_artifact_dir(candidate_dir, baseline_overhead=31.0)

    report = compare_artifact_dirs(baseline_dir, candidate_dir)
    failed_checks = {check["name"] for check in report["checks"] if check["status"] == "fail"}

    assert "baseline.overhead_pct" in failed_checks


def test_compare_artifact_dirs_flags_platform_mismatch_by_default(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _make_artifact_dir(baseline_dir, platform_system="Darwin", platform_machine="arm64")
    _make_artifact_dir(candidate_dir, platform_system="Linux", platform_machine="x86_64")

    report = compare_artifact_dirs(baseline_dir, candidate_dir)
    checks = {check["name"]: check for check in report["checks"]}

    assert checks["environment.platform"]["status"] == "fail"
    assert checks["baseline.neural_ips_gpu_only"]["status"] == "skip"
    assert checks["gpu_only_matrix.counted.avg_ips"]["status"] == "skip"


def test_compare_artifact_dirs_flags_gpu_only_matrix_regression(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _make_artifact_dir(baseline_dir, gpu_only_counted_ips=950.0, gpu_only_adjacent_ips=900.0)
    _make_artifact_dir(candidate_dir, gpu_only_counted_ips=780.0, gpu_only_adjacent_ips=700.0, gpu_only_adjacent_segments=1)

    report = compare_artifact_dirs(baseline_dir, candidate_dir)
    checks = {check["name"]: check for check in report["checks"]}

    assert checks["gpu_only_matrix.counted.avg_ips"]["status"] == "fail"
    assert checks["gpu_only_matrix.adjacent-counted.hotloop_segments"]["status"] == "warn"


def test_compare_artifact_dirs_flags_candidate_non_rust_gpu_only_contract(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _make_artifact_dir(baseline_dir)
    _make_artifact_dir(
        candidate_dir,
        gpu_only_backend="torch-gpu-only",
        gpu_only_backend_ok=False,
        gpu_only_primary_backend="auto",
        gpu_only_require_backend_prefix=None,
        gpu_only_include_torch_baseline=True,
    )

    report = compare_artifact_dirs(baseline_dir, candidate_dir)
    checks = {check["name"]: check for check in report["checks"]}

    assert checks["gpu_only_matrix.candidate.strict_rust_only"]["status"] == "fail"


def test_compare_artifacts_cli_writes_reports(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    _make_artifact_dir(baseline_dir)
    _make_artifact_dir(candidate_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/release/compare_artifacts.py"),
            "--baseline-dir",
            str(baseline_dir),
            "--candidate-dir",
            str(candidate_dir),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json_output.is_file()
    assert markdown_output.is_file()
    assert "Artifact Comparison Report" in markdown_output.read_text()
