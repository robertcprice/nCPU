"""Generate paper-ready benchmark tables from artifact directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gpu_only_matrix import summarize_gpu_only_matrix


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fmt_int(value: Any) -> str:
    if value is None:
        return "---"
    return f"{int(round(float(value))):,}"


def _fmt_float(value: Any, digits: int = 1) -> str:
    if value is None:
        return "---"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "---"
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.{digits}f}%"


def _fmt_ci(mean: Any, ci95: Any, digits: int = 0) -> str:
    if mean is None:
        return "---"
    mean_fmt = f"{float(mean):,.{digits}f}" if digits > 0 else f"{int(round(float(mean))):,}"
    if ci95 in (None, 0, 0.0):
        return mean_fmt
    ci_fmt = f"{float(ci95):,.{digits}f}" if digits > 0 else f"{int(round(float(ci95))):,}"
    return f"{mean_fmt} +/- {ci_fmt}"


def _fmt_interval(lo: Any, hi: Any, digits: int = 1, suffix: str = "") -> str:
    if lo is None or hi is None:
        return "---"
    return f"[{float(lo):.{digits}f}{suffix}, {float(hi):.{digits}f}{suffix}]"


def _calc_overhead(conv_ips: Any, neural_ips: Any) -> float | None:
    if conv_ips in (None, 0, 0.0) or neural_ips is None:
        return None
    return ((float(conv_ips) - float(neural_ips)) / float(conv_ips)) * 100.0


def _section(title: str, lines: list[str]) -> str:
    body = "\n".join(lines).rstrip()
    return f"## {title}\n\n{body}\n"


def extract_paper_tables(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    benchmark_dir = artifact_dir / "benchmarks"
    baseline_data = _read_json(benchmark_dir / "baseline_comparison_results.json")
    ablation_data = _read_json(benchmark_dir / "ablation_results.json")
    real_data = _read_json(benchmark_dir / "real_workload_results.json")
    gpu_only_path = benchmark_dir / "gpu_only_matrix.json"
    gpu_only_data = _read_json(gpu_only_path) if gpu_only_path.is_file() else None

    baseline_conv = baseline_data.get("conventional", {})
    baseline_neural = baseline_data.get("neural_enhanced", {})
    output_comparison = baseline_data.get("output_comparison", {})

    ablation_rows = []
    ablation_results = ablation_data.get("results", [])
    baseline_ablation_ips = next(
        (row.get("ips_gpu_only") for row in ablation_results if row.get("n_models") == 0),
        None,
    )
    for row in ablation_results:
        ablation_rows.append(
            {
                "name": row.get("name"),
                "n_models": row.get("n_models"),
                "ips_gpu_only": row.get("ips_gpu_only"),
                "ips_gpu_only_ci95": row.get("ips_gpu_only_ci95"),
                "neural_inferences": row.get("neural_inferences"),
                "gpu_time_s": row.get("gpu_time_s"),
                "overhead_pct": _calc_overhead(baseline_ablation_ips, row.get("ips_gpu_only"))
                if baseline_ablation_ips is not None
                else None,
            }
        )

    compiler_programs = []
    for program_name, program_data in sorted(real_data.get("compiler_workloads", {}).items()):
        conv = program_data.get("conventional", {})
        neural = program_data.get("neural", {})
        conv_ips = conv.get("ips", {}).get("mean")
        neural_ips = neural.get("ips", {}).get("mean")
        compiler_programs.append(
            {
                "program": program_name,
                "conventional_ips": conv_ips,
                "neural_ips": neural_ips,
                "neural_ci95": {
                    "lo": neural.get("ips", {}).get("ci95_lo"),
                    "hi": neural.get("ips", {}).get("ci95_hi"),
                },
                "compile_time_s": {
                    "conventional": conv.get("compile_time_s", {}).get("mean"),
                    "neural": neural.get("compile_time_s", {}).get("mean"),
                },
                "exec_time_s": {
                    "conventional": conv.get("exec_time_s", {}).get("mean"),
                    "neural": neural.get("exec_time_s", {}).get("mean"),
                },
                "neural_inferences": neural.get("neural_inferences", {}).get("mean"),
                "overhead_pct": _calc_overhead(conv_ips, neural_ips),
                "compiled_ok": {
                    "conventional": conv.get("compiled_ok"),
                    "neural": neural.get("compiled_ok"),
                },
                "exec_ok": {
                    "conventional": conv.get("exec_ok"),
                    "neural": neural.get("exec_ok"),
                },
            }
        )

    shell_conv = real_data.get("shell_workload", {}).get("conventional", {})
    shell_neural = real_data.get("shell_workload", {}).get("neural", {})
    shell_overhead = _calc_overhead(
        shell_conv.get("ips_gpu_only", {}).get("mean"),
        shell_neural.get("ips_gpu_only", {}).get("mean"),
    )
    gpu_only_rows = []
    gpu_only_summary = summarize_gpu_only_matrix(gpu_only_data)
    if gpu_only_data is not None:
        for row in gpu_only_data.get("results", []):
            gpu_only_rows.append(
                {
                    "workload": row.get("workload"),
                    "avg_ips": row.get("avg_ips"),
                    "backend": row.get("backend"),
                    "hotloop_segments": row.get("hotloop_segments"),
                    "result_ok": row.get("result_ok"),
                    "insts_ok": row.get("insts_ok"),
                }
            )

    metrics = {
        "baseline_comparison": {
            "trials": baseline_data.get("metadata", {}).get("trials"),
            "conventional_ips_gpu_only": baseline_conv.get("ips_gpu_only"),
            "conventional_ips_gpu_only_ci95": baseline_conv.get("ips_gpu_only_ci95"),
            "neural_ips_gpu_only": baseline_neural.get("ips_gpu_only"),
            "neural_ips_gpu_only_ci95": baseline_neural.get("ips_gpu_only_ci95"),
            "neural_overhead_pct": baseline_data.get("overhead_pct"),
            "output_match_percentage": output_comparison.get("match_percentage"),
            "outputs_identical": output_comparison.get("identical"),
            "neural_inferences": baseline_neural.get("neural_inferences"),
            "models_active": baseline_neural.get("models_active"),
        },
        "ablation": {
            "trials": ablation_data.get("metadata", {}).get("trials"),
            "rows": ablation_rows,
        },
        "real_workload": {
            "trials": real_data.get("metadata", {}).get("trials"),
            "aggregate_overhead": real_data.get("aggregate_overhead"),
            "shell_workload": {
                "conventional_ips_gpu_only": shell_conv.get("ips_gpu_only", {}).get("mean"),
                "neural_ips_gpu_only": shell_neural.get("ips_gpu_only", {}).get("mean"),
                "neural_overhead_pct": shell_overhead,
                "neural_inferences": shell_neural.get("neural_inferences", {}).get("mean"),
            },
            "compiler_programs": compiler_programs,
        },
    }
    if gpu_only_data is not None:
        metrics["gpu_only_matrix"] = {
            "generated_at": gpu_only_data.get("generated_at"),
            "primary_backend": gpu_only_summary.get("primary_backend"),
            "require_backend_prefix": gpu_only_summary.get("require_backend_prefix"),
            "strict_rust_only": gpu_only_summary.get("strict_rust_only"),
            "passing_rows": gpu_only_summary.get("passing_rows"),
            "multi_segment_rows": gpu_only_summary.get("multi_segment_rows"),
            "max_hotloop_segments": gpu_only_summary.get("max_hotloop_segments"),
            "rows": gpu_only_rows,
        }

    baseline_lines = [
        "| Configuration | GPU-only IPS | 95% CI | Neural Inferences | Output Match | Overhead |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Conventional | {_fmt_int(baseline_conv.get('ips_gpu_only'))} | "
            f"{_fmt_ci(baseline_conv.get('ips_gpu_only'), baseline_conv.get('ips_gpu_only_ci95'))} | 0 | "
            f"100.0% | 0.0% |"
        ),
        (
            f"| Neural-enhanced | {_fmt_int(baseline_neural.get('ips_gpu_only'))} | "
            f"{_fmt_ci(baseline_neural.get('ips_gpu_only'), baseline_neural.get('ips_gpu_only_ci95'))} | "
            f"{_fmt_int(baseline_neural.get('neural_inferences'))} | "
            f"{_fmt_pct(output_comparison.get('match_percentage'))} | "
            f"{_fmt_pct(baseline_data.get('overhead_pct'))} |"
        ),
        "",
        (
            f"Identical output: `{bool(output_comparison.get('identical'))}`. "
            f"Models active: {_fmt_int(baseline_neural.get('models_active'))}."
        ),
    ]

    ablation_lines = [
        "| Configuration | Models | GPU-only IPS | 95% CI | Neural Inferences | Overhead | GPU Time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ablation_rows:
        ablation_lines.append(
            f"| {row['name']} | {_fmt_int(row['n_models'])} | {_fmt_int(row['ips_gpu_only'])} | "
            f"{_fmt_ci(row['ips_gpu_only'], row['ips_gpu_only_ci95'])} | {_fmt_int(row['neural_inferences'])} | "
            f"{_fmt_pct(row['overhead_pct'])} | {_fmt_float(row['gpu_time_s'], 3)} |"
        )

    real_lines = [
        "| Program | Conventional IPS | Neural IPS | Neural 95% CI | Neural Inferences | Overhead | Compile/Exec Health |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in compiler_programs:
        health = (
            f"conv(c={row['compiled_ok']['conventional']}, e={row['exec_ok']['conventional']}), "
            f"neural(c={row['compiled_ok']['neural']}, e={row['exec_ok']['neural']})"
        )
        real_lines.append(
            f"| {row['program']} | {_fmt_int(row['conventional_ips'])} | {_fmt_int(row['neural_ips'])} | "
            f"{_fmt_interval(row['neural_ci95']['lo'], row['neural_ci95']['hi'], 0)} | "
            f"{_fmt_int(row['neural_inferences'])} | {_fmt_pct(row['overhead_pct'])} | {health} |"
        )

    real_lines.extend(
        [
            "",
            "### Shell Workload",
            "",
            "| Configuration | GPU-only IPS | Neural Inferences | Overhead |",
            "| --- | ---: | ---: | ---: |",
            f"| Conventional | {_fmt_int(shell_conv.get('ips_gpu_only', {}).get('mean'))} | 0 | 0.0% |",
            f"| Neural | {_fmt_int(shell_neural.get('ips_gpu_only', {}).get('mean'))} | "
            f"{_fmt_int(shell_neural.get('neural_inferences', {}).get('mean'))} | {_fmt_pct(shell_overhead)} |",
            "",
            (
                "Aggregate compute-workload overhead: "
                f"{_fmt_pct(real_data.get('aggregate_overhead', {}).get('mean'))} "
                f"(95% CI {_fmt_interval(real_data.get('aggregate_overhead', {}).get('ci95_lo'), real_data.get('aggregate_overhead', {}).get('ci95_hi'), 1, '%')})."
            ),
        ]
    )

    gpu_only_lines = []
    if gpu_only_rows:
        gpu_only_lines.extend(
            [
                "| Workload | Avg IPS | Backend | Hotloops | Result | Insts |",
                "| --- | ---: | --- | ---: | --- | --- |",
            ]
        )
        for row in gpu_only_rows:
            gpu_only_lines.append(
                f"| {row['workload']} | {_fmt_int(row['avg_ips'])} | {row['backend']} | "
                f"{_fmt_int(row['hotloop_segments'])} | "
                f"{'OK' if row['result_ok'] else 'BAD'} | "
                f"{'OK' if row['insts_ok'] else 'BAD'} |"
            )
        gpu_only_lines.extend(
            [
                "",
                (
                    "Strict Rust/Metal mode: "
                    f"`{bool(gpu_only_summary.get('strict_rust_only'))}`. "
                    f"Passing rows: {_fmt_int(gpu_only_summary.get('passing_rows'))}/"
                    f"{_fmt_int(gpu_only_summary.get('row_count'))}. "
                    f"Multi-segment rows: {_fmt_int(gpu_only_summary.get('multi_segment_rows'))}. "
                    f"Max chained hotloops: {_fmt_int(gpu_only_summary.get('max_hotloop_segments'))}."
                ),
            ]
        )

    claim_lines = [
        (
            "Baseline shell workload: "
            f"{_fmt_int(baseline_conv.get('ips_gpu_only'))} GPU-only IPS conventional vs "
            f"{_fmt_int(baseline_neural.get('ips_gpu_only'))} neural, "
            f"{_fmt_pct(baseline_data.get('overhead_pct'))} overhead, "
            f"{_fmt_pct(output_comparison.get('match_percentage'))} output match."
        ),
        (
            "Full ablation stack: "
            f"{_fmt_int(ablation_rows[-1]['n_models'])} models, "
            f"{_fmt_int(ablation_rows[-1]['ips_gpu_only'])} GPU-only IPS, "
            f"{_fmt_pct(ablation_rows[-1]['overhead_pct'])} overhead relative to the ablation baseline."
        ) if ablation_rows else "Full ablation stack: unavailable.",
        (
            "Real compute workloads: aggregate neural overhead "
            f"{_fmt_pct(real_data.get('aggregate_overhead', {}).get('mean'))} "
            f"with 95% CI {_fmt_interval(real_data.get('aggregate_overhead', {}).get('ci95_lo'), real_data.get('aggregate_overhead', {}).get('ci95_hi'), 1, '%')}."
        ),
    ]

    markdown = "\n".join(
        [
            "# Paper Tables",
            "",
            _section("Baseline Comparison", baseline_lines).rstrip(),
            "",
            _section("Ablation Study", ablation_lines).rstrip(),
            "",
            _section("Real Workloads", real_lines).rstrip(),
            "",
            _section("GPU-Only Hotloop Matrix", gpu_only_lines).rstrip() if gpu_only_lines else "",
            "" if gpu_only_lines else "",
            _section("Benchmark Claims", claim_lines).rstrip(),
            "",
        ]
    )

    return {"metrics": metrics, "markdown": markdown}


def write_paper_tables(artifact_dir: Path, output_dir: Path | None = None) -> dict[str, Path]:
    artifact_dir = artifact_dir.resolve()
    output_dir = (artifact_dir / "paper_tables" if output_dir is None else output_dir.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = extract_paper_tables(artifact_dir)
    metrics_path = output_dir / "paper_metrics.json"
    tables_path = output_dir / "paper_tables.md"

    metrics_path.write_text(json.dumps(generated["metrics"], indent=2) + "\n")
    tables_path.write_text(generated["markdown"])

    return {
        "metrics": metrics_path,
        "markdown": tables_path,
    }
