"""Comparison helpers for publication artifact directories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .gpu_only_matrix import summarize_gpu_only_matrix


REQUIRED_ARTIFACT_FILES = {
    "manifest": "artifact_manifest.json",
    "baseline": "benchmarks/baseline_comparison_results.json",
    "ablation": "benchmarks/ablation_results.json",
    "real_workload": "benchmarks/real_workload_results.json",
    "gpu_only_matrix": "benchmarks/gpu_only_matrix.json",
}


@dataclass(frozen=True)
class ComparisonThresholds:
    max_baseline_overhead_increase_points: float = 5.0
    max_baseline_ips_regression_pct: float = 10.0
    max_ablation_ips_regression_pct: float = 10.0
    max_real_workload_ips_regression_pct: float = 10.0
    max_real_workload_overhead_increase_points: float = 10.0
    max_gpu_only_matrix_ips_regression_pct: float = 10.0


@dataclass
class ArtifactRun:
    root: Path
    manifest: dict[str, Any] | None
    baseline: dict[str, Any] | None
    ablation: dict[str, Any] | None
    real_workload: dict[str, Any] | None
    gpu_only_matrix: dict[str, Any] | None
    errors: list[dict[str, str]]

    @classmethod
    def load(cls, root: Path) -> "ArtifactRun":
        root = root.resolve()
        loaded: dict[str, dict[str, Any] | None] = {}
        errors: list[dict[str, str]] = []

        for key, rel_path in REQUIRED_ARTIFACT_FILES.items():
            path = root / rel_path
            if not path.is_file():
                loaded[key] = None
                continue
            try:
                loaded[key] = json.loads(path.read_text())
            except Exception as exc:  # pragma: no cover - exercised by CLI on bad inputs
                loaded[key] = None
                errors.append({"path": str(path), "error": str(exc)})

        return cls(root=root, errors=errors, **loaded)

    @property
    def provenance(self) -> dict[str, Any]:
        for payload in (self.manifest, self.baseline, self.ablation, self.real_workload, self.gpu_only_matrix):
            if isinstance(payload, dict) and isinstance(payload.get("provenance"), dict):
                return payload["provenance"]
        return {}


def _nested_get(payload: dict[str, Any] | None, *keys: str) -> Any:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _make_check(name: str, status: str, summary: str, **details: Any) -> dict[str, Any]:
    check = {"name": name, "status": status, "summary": summary}
    for key, value in details.items():
        if value is not None:
            check[key] = value
    return check


def _compare_higher_is_better(
    *,
    name: str,
    label: str,
    baseline_value: Any,
    candidate_value: Any,
    max_regression_pct: float,
) -> dict[str, Any]:
    if baseline_value is None or candidate_value is None:
        return _make_check(
            name,
            "fail",
            f"{label} metric missing from one or both artifact sets",
            baseline=baseline_value,
            candidate=candidate_value,
        )

    if baseline_value <= 0:
        return _make_check(
            name,
            "warn",
            f"{label} baseline is non-positive; skipping regression threshold",
            baseline=baseline_value,
            candidate=candidate_value,
        )

    regression_pct = ((baseline_value - candidate_value) / baseline_value) * 100.0
    status = "fail" if regression_pct > max_regression_pct else "pass"
    if regression_pct > 0:
        summary = f"{label} regressed by {regression_pct:.2f}%"
    else:
        summary = f"{label} improved by {abs(regression_pct):.2f}%"

    return _make_check(
        name,
        status,
        summary,
        baseline=baseline_value,
        candidate=candidate_value,
        delta=candidate_value - baseline_value,
        regression_pct=regression_pct,
        threshold_pct=max_regression_pct,
    )


def _compare_lower_is_better(
    *,
    name: str,
    label: str,
    baseline_value: Any,
    candidate_value: Any,
    max_increase_points: float,
) -> dict[str, Any]:
    if baseline_value is None or candidate_value is None:
        return _make_check(
            name,
            "fail",
            f"{label} metric missing from one or both artifact sets",
            baseline=baseline_value,
            candidate=candidate_value,
        )

    increase = candidate_value - baseline_value
    status = "fail" if increase > max_increase_points else "pass"
    if increase > 0:
        summary = f"{label} increased by {increase:.2f} points"
    else:
        summary = f"{label} improved by {abs(increase):.2f} points"

    return _make_check(
        name,
        status,
        summary,
        baseline=baseline_value,
        candidate=candidate_value,
        delta=increase,
        threshold_points=max_increase_points,
    )


def _skip_check(name: str, summary: str) -> dict[str, Any]:
    return _make_check(name, "skip", summary)


def _platform_descriptor(provenance: dict[str, Any]) -> dict[str, Any]:
    platform_payload = provenance.get("platform", {})
    return {
        "system": platform_payload.get("system"),
        "machine": platform_payload.get("machine"),
        "release": platform_payload.get("release"),
        "platform": platform_payload.get("platform"),
    }


def _summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1

    overall = "fail" if counts["fail"] else "warn" if counts["warn"] else "pass"
    return {
        "overall_status": overall,
        "passed": counts["pass"],
        "failed": counts["fail"],
        "warnings": counts["warn"],
        "skipped": counts["skip"],
        "total_checks": len(checks),
    }


def compare_artifact_dirs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    thresholds: ComparisonThresholds | None = None,
    allow_platform_mismatch: bool = False,
) -> dict[str, Any]:
    thresholds = thresholds or ComparisonThresholds()
    baseline = ArtifactRun.load(Path(baseline_dir))
    candidate = ArtifactRun.load(Path(candidate_dir))
    checks: list[dict[str, Any]] = []

    for run_name, run in (("baseline", baseline), ("candidate", candidate)):
        for error in run.errors:
            checks.append(
                _make_check(
                    f"artifacts.{run_name}.load_error",
                    "fail",
                    f"{run_name} artifact failed to load",
                    path=error["path"],
                    error=error["error"],
                )
            )

    for key, rel_path in REQUIRED_ARTIFACT_FILES.items():
        base_present = getattr(baseline, key) is not None
        cand_present = getattr(candidate, key) is not None
        if base_present and cand_present:
            checks.append(
                _make_check(
                    f"artifacts.{rel_path}",
                    "pass",
                    f"Both artifact sets contain {rel_path}",
                )
            )
        elif base_present and not cand_present:
            checks.append(
                _make_check(
                    f"artifacts.{rel_path}",
                    "fail",
                    f"Candidate artifact set is missing {rel_path}",
                )
            )
        elif not base_present and cand_present:
            checks.append(
                _make_check(
                    f"artifacts.{rel_path}",
                    "warn",
                    f"Baseline artifact set is missing {rel_path}",
                )
            )
        else:
            checks.append(
                _make_check(
                    f"artifacts.{rel_path}",
                    "warn",
                    f"Neither artifact set contains {rel_path}",
                )
            )

    baseline_platform = _platform_descriptor(baseline.provenance)
    candidate_platform = _platform_descriptor(candidate.provenance)
    same_platform = (
        baseline_platform.get("system"),
        baseline_platform.get("machine"),
    ) == (
        candidate_platform.get("system"),
        candidate_platform.get("machine"),
    )
    performance_comparable = same_platform or allow_platform_mismatch
    platform_status = "pass" if same_platform else "warn" if allow_platform_mismatch else "fail"
    platform_summary = (
        "Baseline and candidate artifacts were captured on the same platform"
        if same_platform
        else "Platform mismatch; use --allow-platform-mismatch only for non-gating review"
    )
    checks.append(
        _make_check(
            "environment.platform",
            platform_status,
            platform_summary,
            baseline=baseline_platform,
            candidate=candidate_platform,
        )
    )

    baseline_mog = _nested_get(baseline.provenance, "mog", "commit")
    candidate_mog = _nested_get(candidate.provenance, "mog", "commit")
    if baseline_mog and candidate_mog:
        mog_status = "pass" if baseline_mog == candidate_mog else "warn"
        mog_summary = (
            "Mog toolchain commits match"
            if baseline_mog == candidate_mog
            else "Mog toolchain commit changed between artifact runs"
        )
        checks.append(
            _make_check(
                "environment.mog_commit",
                mog_status,
                mog_summary,
                baseline=baseline_mog,
                candidate=candidate_mog,
            )
        )

    if baseline.baseline and candidate.baseline:
        base_output = _nested_get(baseline.baseline, "output_comparison") or {}
        cand_output = _nested_get(candidate.baseline, "output_comparison") or {}
        base_match = base_output.get("match_percentage")
        cand_match = cand_output.get("match_percentage")
        base_identical = bool(base_output.get("identical"))
        cand_identical = bool(cand_output.get("identical"))

        if cand_match is None:
            checks.append(
                _make_check(
                    "baseline.output_correctness",
                    "fail",
                    "Candidate baseline comparison is missing output correctness data",
                )
            )
        else:
            degraded = (base_match is not None and cand_match < base_match) or (base_identical and not cand_identical)
            status = "fail" if degraded else "pass"
            summary = (
                f"Output correctness dropped from {base_match:.1f}% to {cand_match:.1f}%"
                if degraded and base_match is not None
                else f"Output correctness preserved at {cand_match:.1f}%"
            )
            checks.append(
                _make_check(
                    "baseline.output_correctness",
                    status,
                    summary,
                    baseline=base_output,
                    candidate=cand_output,
                )
            )

        if performance_comparable:
            checks.append(
                _compare_higher_is_better(
                    name="baseline.conventional_ips_gpu_only",
                    label="Baseline conventional GPU-only IPS",
                    baseline_value=_nested_get(baseline.baseline, "conventional", "ips_gpu_only"),
                    candidate_value=_nested_get(candidate.baseline, "conventional", "ips_gpu_only"),
                    max_regression_pct=thresholds.max_baseline_ips_regression_pct,
                )
            )
            checks.append(
                _compare_higher_is_better(
                    name="baseline.neural_ips_gpu_only",
                    label="Baseline neural GPU-only IPS",
                    baseline_value=_nested_get(baseline.baseline, "neural_enhanced", "ips_gpu_only"),
                    candidate_value=_nested_get(candidate.baseline, "neural_enhanced", "ips_gpu_only"),
                    max_regression_pct=thresholds.max_baseline_ips_regression_pct,
                )
            )
            checks.append(
                _compare_lower_is_better(
                    name="baseline.overhead_pct",
                    label="Baseline neural overhead",
                    baseline_value=_nested_get(baseline.baseline, "overhead_pct"),
                    candidate_value=_nested_get(candidate.baseline, "overhead_pct"),
                    max_increase_points=thresholds.max_baseline_overhead_increase_points,
                )
            )
        else:
            checks.append(_skip_check("baseline.conventional_ips_gpu_only", "Skipped because platforms differ"))
            checks.append(_skip_check("baseline.neural_ips_gpu_only", "Skipped because platforms differ"))
            checks.append(_skip_check("baseline.overhead_pct", "Skipped because platforms differ"))

    if baseline.ablation and candidate.ablation:
        base_results = {
            entry["name"]: entry
            for entry in baseline.ablation.get("results", [])
            if isinstance(entry, dict) and entry.get("name")
        }
        cand_results = {
            entry["name"]: entry
            for entry in candidate.ablation.get("results", [])
            if isinstance(entry, dict) and entry.get("name")
        }
        missing_configs = sorted(set(base_results) - set(cand_results))
        extra_configs = sorted(set(cand_results) - set(base_results))

        config_status = "pass"
        if missing_configs:
            config_status = "fail"
        elif extra_configs:
            config_status = "warn"

        summary_parts = []
        if missing_configs:
            summary_parts.append(f"missing configs: {', '.join(missing_configs)}")
        if extra_configs:
            summary_parts.append(f"extra configs: {', '.join(extra_configs)}")
        if not summary_parts:
            summary_parts.append("Ablation configuration set matches")
        checks.append(
            _make_check(
                "ablation.configurations",
                config_status,
                "; ".join(summary_parts),
            )
        )

        if performance_comparable:
            for config_name in sorted(set(base_results) & set(cand_results)):
                checks.append(
                    _compare_higher_is_better(
                        name=f"ablation.{config_name}.ips_gpu_only",
                        label=f"Ablation {config_name} GPU-only IPS",
                        baseline_value=_nested_get(base_results[config_name], "ips_gpu_only"),
                        candidate_value=_nested_get(cand_results[config_name], "ips_gpu_only"),
                        max_regression_pct=thresholds.max_ablation_ips_regression_pct,
                    )
                )
        else:
            checks.append(_skip_check("ablation.performance", "Skipped because platforms differ"))

    if baseline.real_workload and candidate.real_workload:
        if performance_comparable:
            checks.append(
                _compare_lower_is_better(
                    name="real_workload.aggregate_overhead",
                    label="Real-workload aggregate overhead",
                    baseline_value=_nested_get(baseline.real_workload, "aggregate_overhead", "mean"),
                    candidate_value=_nested_get(candidate.real_workload, "aggregate_overhead", "mean"),
                    max_increase_points=thresholds.max_real_workload_overhead_increase_points,
                )
            )
            checks.append(
                _compare_higher_is_better(
                    name="real_workload.shell.neural_ips_gpu_only",
                    label="Shell workload neural GPU-only IPS",
                    baseline_value=_nested_get(baseline.real_workload, "shell_workload", "neural", "ips_gpu_only", "mean"),
                    candidate_value=_nested_get(candidate.real_workload, "shell_workload", "neural", "ips_gpu_only", "mean"),
                    max_regression_pct=thresholds.max_real_workload_ips_regression_pct,
                )
            )
        else:
            checks.append(_skip_check("real_workload.aggregate_overhead", "Skipped because platforms differ"))
            checks.append(_skip_check("real_workload.shell.neural_ips_gpu_only", "Skipped because platforms differ"))

        base_programs = baseline.real_workload.get("compiler_workloads", {})
        cand_programs = candidate.real_workload.get("compiler_workloads", {})
        missing_programs = sorted(set(base_programs) - set(cand_programs))
        extra_programs = sorted(set(cand_programs) - set(base_programs))

        program_status = "pass"
        if missing_programs:
            program_status = "fail"
        elif extra_programs:
            program_status = "warn"

        summary_parts = []
        if missing_programs:
            summary_parts.append(f"missing programs: {', '.join(missing_programs)}")
        if extra_programs:
            summary_parts.append(f"extra programs: {', '.join(extra_programs)}")
        if not summary_parts:
            summary_parts.append("Real-workload program set matches")
        checks.append(
            _make_check(
                "real_workload.programs",
                program_status,
                "; ".join(summary_parts),
            )
        )

        for program_name in sorted(set(base_programs) & set(cand_programs)):
            base_program = base_programs.get(program_name, {})
            cand_program = cand_programs.get(program_name, {})

            correctness_ok = True
            details: dict[str, Any] = {"baseline": {}, "candidate": {}}
            for config_name in ("conventional", "neural"):
                base_cfg = base_program.get(config_name, {})
                cand_cfg = cand_program.get(config_name, {})
                details["baseline"][config_name] = {
                    "compiled_ok": base_cfg.get("compiled_ok"),
                    "exec_ok": base_cfg.get("exec_ok"),
                }
                details["candidate"][config_name] = {
                    "compiled_ok": cand_cfg.get("compiled_ok"),
                    "exec_ok": cand_cfg.get("exec_ok"),
                }
                if base_cfg.get("compiled_ok") and not cand_cfg.get("compiled_ok"):
                    correctness_ok = False
                if base_cfg.get("exec_ok") and not cand_cfg.get("exec_ok"):
                    correctness_ok = False

            checks.append(
                _make_check(
                    f"real_workload.{program_name}.correctness",
                    "pass" if correctness_ok else "fail",
                    "Compile/exec health preserved" if correctness_ok else "Compile/exec health regressed",
                    **details,
                )
            )

            if performance_comparable:
                checks.append(
                    _compare_higher_is_better(
                        name=f"real_workload.{program_name}.neural_ips",
                        label=f"Real-workload {program_name} neural IPS",
                        baseline_value=_nested_get(base_program, "neural", "ips", "mean"),
                        candidate_value=_nested_get(cand_program, "neural", "ips", "mean"),
                        max_regression_pct=thresholds.max_real_workload_ips_regression_pct,
                    )
                )

    if baseline.gpu_only_matrix and candidate.gpu_only_matrix:
        base_matrix_summary = summarize_gpu_only_matrix(baseline.gpu_only_matrix)
        cand_matrix_summary = summarize_gpu_only_matrix(candidate.gpu_only_matrix)
        base_issue_text = "; ".join(base_matrix_summary["contract_issues"]) or "strict Rust/Metal mode was not requested"
        cand_issue_text = "; ".join(cand_matrix_summary["contract_issues"]) or "strict Rust/Metal mode was not requested"
        checks.append(
            _make_check(
                "gpu_only_matrix.baseline.strict_rust_only",
                "pass" if base_matrix_summary["strict_rust_only"] else "warn",
                "Baseline GPU-only matrix satisfies the strict Rust/Metal contract"
                if base_matrix_summary["strict_rust_only"]
                else f"Baseline GPU-only matrix is not strict Rust/Metal: {base_issue_text}",
            )
        )
        checks.append(
            _make_check(
                "gpu_only_matrix.candidate.strict_rust_only",
                "pass" if cand_matrix_summary["strict_rust_only"] else "fail",
                "Candidate GPU-only matrix satisfies the strict Rust/Metal contract"
                if cand_matrix_summary["strict_rust_only"]
                else f"Candidate GPU-only matrix is not strict Rust/Metal: {cand_issue_text}",
            )
        )
        base_rows = {
            entry["workload"]: entry
            for entry in baseline.gpu_only_matrix.get("results", [])
            if isinstance(entry, dict) and entry.get("workload")
        }
        cand_rows = {
            entry["workload"]: entry
            for entry in candidate.gpu_only_matrix.get("results", [])
            if isinstance(entry, dict) and entry.get("workload")
        }
        missing_workloads = sorted(set(base_rows) - set(cand_rows))
        extra_workloads = sorted(set(cand_rows) - set(base_rows))

        workload_status = "pass"
        if missing_workloads:
            workload_status = "fail"
        elif extra_workloads:
            workload_status = "warn"

        summary_parts = []
        if missing_workloads:
            summary_parts.append(f"missing workloads: {', '.join(missing_workloads)}")
        if extra_workloads:
            summary_parts.append(f"extra workloads: {', '.join(extra_workloads)}")
        if not summary_parts:
            summary_parts.append("GPU-only matrix workload set matches")
        checks.append(
            _make_check(
                "gpu_only_matrix.workloads",
                workload_status,
                "; ".join(summary_parts),
            )
        )

        for workload in sorted(set(base_rows) & set(cand_rows)):
            base_row = base_rows[workload]
            cand_row = cand_rows[workload]
            correctness_ok = True
            if bool(base_row.get("result_ok")) and not bool(cand_row.get("result_ok")):
                correctness_ok = False
            if bool(base_row.get("insts_ok")) and not bool(cand_row.get("insts_ok")):
                correctness_ok = False
            if bool(base_row.get("backend_ok")) and not bool(cand_row.get("backend_ok")):
                correctness_ok = False
            checks.append(
                _make_check(
                    f"gpu_only_matrix.{workload}.correctness",
                    "pass" if correctness_ok else "fail",
                    "Result/instruction checks preserved" if correctness_ok else "Result/instruction checks regressed",
                    baseline={
                        "result_ok": base_row.get("result_ok"),
                        "insts_ok": base_row.get("insts_ok"),
                        "backend_ok": base_row.get("backend_ok"),
                        "backend": base_row.get("backend"),
                        "hotloop_segments": base_row.get("hotloop_segments"),
                    },
                    candidate={
                        "result_ok": cand_row.get("result_ok"),
                        "insts_ok": cand_row.get("insts_ok"),
                        "backend_ok": cand_row.get("backend_ok"),
                        "backend": cand_row.get("backend"),
                        "hotloop_segments": cand_row.get("hotloop_segments"),
                    },
                )
            )
            if performance_comparable:
                checks.append(
                    _compare_higher_is_better(
                        name=f"gpu_only_matrix.{workload}.avg_ips",
                        label=f"GPU-only matrix {workload} average IPS",
                        baseline_value=base_row.get("avg_ips"),
                        candidate_value=cand_row.get("avg_ips"),
                        max_regression_pct=thresholds.max_gpu_only_matrix_ips_regression_pct,
                    )
                )
            else:
                checks.append(
                    _skip_check(
                        f"gpu_only_matrix.{workload}.avg_ips",
                        "Skipped because platforms differ",
                    )
                )

            base_backend = base_row.get("backend")
            cand_backend = cand_row.get("backend")
            if base_backend is not None and cand_backend is not None:
                checks.append(
                    _make_check(
                        f"gpu_only_matrix.{workload}.backend",
                        "pass" if base_backend == cand_backend else "warn",
                        "GPU-only matrix backend preserved"
                        if base_backend == cand_backend
                        else f"GPU-only matrix backend changed from {base_backend} to {cand_backend}",
                        baseline=base_backend,
                        candidate=cand_backend,
                    )
                )

            base_segments = base_row.get("hotloop_segments")
            cand_segments = cand_row.get("hotloop_segments")
            if base_segments is not None and cand_segments is not None:
                segment_status = "pass"
                if int(cand_segments) < int(base_segments):
                    segment_status = "warn"
                checks.append(
                    _make_check(
                        f"gpu_only_matrix.{workload}.hotloop_segments",
                        segment_status,
                        "GPU-only matrix hotloop segment count preserved"
                        if int(cand_segments) == int(base_segments)
                        else f"GPU-only matrix hotloop segments changed from {base_segments} to {cand_segments}",
                        baseline=base_segments,
                        candidate=cand_segments,
                    )
                )

    report = {
        "baseline_dir": str(baseline.root),
        "candidate_dir": str(candidate.root),
        "allow_platform_mismatch": allow_platform_mismatch,
        "thresholds": asdict(thresholds),
        "comparability": {
            "performance_comparable": performance_comparable,
            "baseline_platform": baseline_platform,
            "candidate_platform": candidate_platform,
            "baseline_mog_commit": baseline_mog,
            "candidate_mog_commit": candidate_mog,
        },
        "checks": checks,
    }
    report["summary"] = _summarize(checks)
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Artifact Comparison Report",
        "",
        f"- Baseline: `{report['baseline_dir']}`",
        f"- Candidate: `{report['candidate_dir']}`",
        f"- Overall status: **{summary['overall_status'].upper()}**",
        f"- Checks: {summary['total_checks']} total, {summary['passed']} pass, {summary['failed']} fail, {summary['warnings']} warn, {summary['skipped']} skip",
        f"- Performance comparable: `{report['comparability']['performance_comparable']}`",
        "",
        "| Check | Status | Summary |",
        "| --- | --- | --- |",
    ]

    for check in report["checks"]:
        lines.append(f"| `{check['name']}` | `{check['status']}` | {check['summary']} |")

    lines.append("")
    return "\n".join(lines)
