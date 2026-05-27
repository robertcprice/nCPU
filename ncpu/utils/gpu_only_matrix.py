"""Helpers for exported GPU-only hotloop matrix artifacts."""

from __future__ import annotations

from typing import Any


def _rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def summarize_gpu_only_matrix(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return a compact summary plus strict Rust/Metal contract status."""
    config = payload.get("exporter_config", {}) if isinstance(payload, dict) else {}
    benchmark_env = payload.get("benchmark_env", {}) if isinstance(payload, dict) else {}
    rows = _rows(payload)

    row_count = len(rows)
    completed_count = sum(1 for row in rows if row.get("status") == "completed")
    passing_count = sum(
        1 for row in rows if bool(row.get("result_ok")) and bool(row.get("insts_ok"))
    )
    rust_backend_count = sum(
        1
        for row in rows
        if isinstance(row.get("backend"), str) and row["backend"].startswith("rust")
    )
    backend_ok_count = sum(1 for row in rows if row.get("backend_ok") is True)
    multi_segment_count = sum(1 for row in rows if int(row.get("hotloop_segments") or 0) > 1)
    max_hotloop_segments = max((int(row.get("hotloop_segments") or 0) for row in rows), default=0)
    torch_skipped_count = sum(1 for row in rows if row.get("torch_baseline_status") == "skipped")

    primary_backend = None
    if isinstance(config, dict):
        primary_backend = config.get("primary_backend")
    if primary_backend is None and isinstance(benchmark_env, dict):
        primary_backend = benchmark_env.get("NCPU_GPU_ONLY_HOTLOOP_BACKEND")

    require_backend_prefix = config.get("require_backend_prefix") if isinstance(config, dict) else None
    include_torch_baseline = (
        config.get("include_torch_baseline") if isinstance(config, dict) else None
    )

    strict_requested = primary_backend == "rust" or require_backend_prefix == "rust"
    contract_issues: list[str] = []
    if strict_requested:
        if primary_backend != "rust":
            contract_issues.append(
                f"primary backend is {primary_backend!r}, expected 'rust'"
            )
        if require_backend_prefix != "rust":
            contract_issues.append(
                f"backend prefix requirement is {require_backend_prefix!r}, expected 'rust'"
            )
        if include_torch_baseline is not False:
            contract_issues.append("torch baseline is enabled")
        if row_count == 0:
            contract_issues.append("matrix contains no workloads")
        if completed_count != row_count:
            contract_issues.append(
                f"{completed_count}/{row_count} workloads completed successfully"
            )
        if passing_count != row_count:
            contract_issues.append(
                f"{passing_count}/{row_count} workloads passed result/instruction checks"
            )
        if rust_backend_count != row_count:
            contract_issues.append(
                f"{rust_backend_count}/{row_count} workloads reported a rust backend"
            )
        if backend_ok_count != row_count:
            contract_issues.append(
                f"{backend_ok_count}/{row_count} workloads satisfied the backend prefix check"
            )
        if include_torch_baseline is False and torch_skipped_count != row_count:
            contract_issues.append(
                f"{torch_skipped_count}/{row_count} workloads skipped the torch baseline"
            )

    return {
        "row_count": row_count,
        "completed_rows": completed_count,
        "passing_rows": passing_count,
        "rust_backend_rows": rust_backend_count,
        "backend_ok_rows": backend_ok_count,
        "multi_segment_rows": multi_segment_count,
        "max_hotloop_segments": max_hotloop_segments,
        "torch_baseline_skipped_rows": torch_skipped_count,
        "primary_backend": primary_backend,
        "require_backend_prefix": require_backend_prefix,
        "include_torch_baseline": include_torch_baseline,
        "strict_requested": strict_requested,
        "strict_rust_only": strict_requested and not contract_issues,
        "contract_issues": contract_issues,
    }
