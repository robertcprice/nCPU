from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_ARTIFACTS = PROJECT_ROOT / "scripts" / "release" / "paper_artifacts.sh"
BUILD_SUBMISSION_BUNDLE = PROJECT_ROOT / "scripts" / "release" / "build_submission_bundle.sh"
PAPER = PROJECT_ROOT / "paper" / "ncpu_paper.md"


def test_paper_artifacts_includes_meta_comparison_benchmark() -> None:
    text = PAPER_ARTIFACTS.read_text(encoding="utf-8")

    assert "benchmarks/benchmark_meta_comparison_demo.py" in text
    assert '--output-dir "$output_dir/benchmarks/meta_comparison_demo"' in text
    assert "scripts/release/export_meta_comparison_figure.py" in text
    assert '--output-dir "$repo_root/paper/generated/meta_comparison_demo_latest"' in text
    assert "--no-inprocess" in text
    assert "--reuse-cpu" not in text


def test_submission_bundle_recursively_copies_benchmark_trees() -> None:
    text = BUILD_SUBMISSION_BUNDLE.read_text(encoding="utf-8")

    assert 'find "$source_dir" -type f | sort' in text
    assert 'rel_path="${source#"${source_dir}/"}"' in text
    assert 'copied_files+=("${target_subdir}/${rel_path}")' in text
    assert 'copy_tree_to_exact_path "${repo_root}/paper/generated/meta_comparison_demo_latest" "paper/generated/meta_comparison_demo_latest"' in text


def test_paper_references_stable_meta_comparison_figure() -> None:
    text = PAPER.read_text(encoding="utf-8")

    assert "paper/generated/meta_comparison_demo_latest/final.png" in text
    assert "Figure 19.1. Scripted neural-vs-Meta comparison artifact." in text
