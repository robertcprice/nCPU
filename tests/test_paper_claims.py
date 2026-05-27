import json
from pathlib import Path
import subprocess
import sys

from ncpu.utils.paper_claims import render_updated_paper, verify_paper_claims, write_updated_paper


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _make_artifact_dir(root: Path) -> None:
    _write_json(
        root / "benchmarks/baseline_comparison_results.json",
        {
            "metadata": {"trials": 3, "workload_commands": 36},
            "conventional": {
                "total_cycles": 64355,
                "gpu_time_s": 0.125,
                "ips_gpu_only": 516703,
                "neural_inferences": 0,
                "peak_rss_kb": 245137408,
            },
            "neural_enhanced": {
                "total_cycles": 64355,
                "gpu_time_s": 0.329,
                "ips_gpu_only": 195484,
                "neural_inferences": 239,
                "models_active": 9,
                "peak_rss_kb": 368197632,
            },
            "overhead_pct": 62.2,
            "output_comparison": {
                "conv_lines": 114,
                "neural_lines": 114,
                "match_percentage": 100.0,
                "identical": True,
            },
        },
    )
    _write_json(
        root / "benchmarks/ablation_results.json",
        {
            "metadata": {"trials": 3},
            "results": [
                {"name": "baseline (0 models)", "n_models": 0, "ips_gpu_only": 642503, "neural_inferences": 0},
                {"name": "+display (1 model)", "n_models": 1, "ips_gpu_only": 655875, "neural_inferences": 1},
                {"name": "+display +cache (3 models)", "n_models": 3, "ips_gpu_only": 625698, "neural_inferences": 8},
                {"name": "+5 models (core)", "n_models": 6, "ips_gpu_only": 397299, "neural_inferences": 30},
                {"name": "all 9 models", "n_models": 9, "ips_gpu_only": 324259, "neural_inferences": 239},
            ],
        },
    )
    _write_json(
        root / "benchmarks/real_workload_results.json",
        {
            "aggregate_overhead": {"mean": 24.1},
            "compiler_workloads": {
                "fibonacci": {
                    "conventional": {"compiled_ok": True, "exec_ok": True},
                    "neural": {"compiled_ok": True, "exec_ok": True},
                },
                "sieve": {
                    "conventional": {"compiled_ok": True, "exec_ok": True},
                    "neural": {"compiled_ok": True, "exec_ok": True},
                },
                "sort": {
                    "conventional": {"compiled_ok": True, "exec_ok": True},
                    "neural": {"compiled_ok": True, "exec_ok": True},
                },
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
                    "workload": "counted",
                    "avg_ips": 950000,
                    "status": "completed",
                    "primary_completed": True,
                    "backend": "rust-hotloop",
                    "backend_ok": True,
                    "hotloop_segments": 1,
                    "result_ok": True,
                    "insts_ok": True,
                    "torch_baseline_status": "skipped",
                },
                {
                    "workload": "adjacent-bytecopy",
                    "avg_ips": 898334,
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


def _paper_fixture() -> str:
    return """### 21.9 Ablation Study

old ablation body

### 21.10 Baseline Comparison

old baseline body

### 21.11 Conclusions and Future Work

old summary paragraph

**Novel contributions.**

1. Example

**Limitations.**

- The full neural OS adds old overhead.

**Future directions.**

1. Example

## 22. Conclusion

Some intro.

**Neural OS models enhance without altering.** old conclusion sentence.

The system comprises 27+ trained models.
"""


def test_render_updated_paper_rewrites_targeted_sections(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    _make_artifact_dir(artifact_dir)

    updated = render_updated_paper(_paper_fixture(), artifact_dir)

    assert "36 shell commands including in-shell C compilation and execution" in updated
    assert "| Baseline (0 models) | 0 | 643K | 0 | 0.0% |" in updated
    assert "| GPU-Only IPS | 517K | 195K | -62.2% |" in updated
    assert "byte-identical output (114 lines, 100.0% match)" in updated
    assert "the full 9-model stack adds 49.5% overhead relative to the ablation baseline" in updated
    assert "compute-heavy real-workload benchmark reports 24.1% mean overhead" in updated
    assert "The strict Rust/Metal GPU-only hotloop matrix reports 2/2 workloads passing both result and instruction checks" in updated


def test_update_paper_claims_cli_writes_output(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    paper_path = tmp_path / "paper.md"
    output_path = tmp_path / "paper.updated.md"
    _make_artifact_dir(artifact_dir)
    paper_path.write_text(_paper_fixture())

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/release/update_paper_claims.py"),
            "--artifact-dir",
            str(artifact_dir),
            "--paper-path",
            str(paper_path),
            "--output-path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert output_path.is_file()
    assert "### 21.10 Baseline Comparison" in output_path.read_text()


def test_write_updated_paper_defaults_to_preview_file(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    paper_path = tmp_path / "paper.md"
    _make_artifact_dir(artifact_dir)
    paper_path.write_text(_paper_fixture())

    target = write_updated_paper(artifact_dir, paper_path)

    assert target == tmp_path / "paper.updated.md"
    assert target.is_file()


def test_verify_paper_claims_detects_drift_and_writes_preview(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    paper_path = tmp_path / "paper.md"
    preview_path = tmp_path / "expected.md"
    _make_artifact_dir(artifact_dir)
    paper_path.write_text(_paper_fixture())

    matches, preview = verify_paper_claims(artifact_dir, paper_path, output_path=preview_path)

    assert not matches
    assert preview == preview_path
    assert preview_path.is_file()


def test_verify_paper_claims_passes_when_paper_is_current(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    paper_path = tmp_path / "paper.md"
    _make_artifact_dir(artifact_dir)
    paper_path.write_text(render_updated_paper(_paper_fixture(), artifact_dir))

    matches, preview = verify_paper_claims(artifact_dir, paper_path)

    assert matches
    assert preview is None
