"""Compliance report generator (NV5).

Combines:

* `ArrayProgramLibrary.audit_report()` (skill inventory + hit counts),
* `program_verifier.verify_library(...)` (per-skill static safety analysis),
* Session-level diff (added/removed/changed/hits_since_snapshot), and
* A single risk-aggregate summary for the whole library,

into one artifact suitable for regulated-workflow review. The output is a
JSON-serializable dict; `render_markdown` produces a human-readable form.

The envelope is deliberately narrow: everything the report describes is a
property of the *library* and the *discrete programs* it holds, not of
the underlying LLM's weights. That distinction is exactly the one auditors
care about — the library is what actually runs on every fast-path hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary
from ncpu.self_optimizing.program_verifier import (
    RISK_HIGH,
    RISK_SAFE,
    RISK_WARN,
    VerifierConfig,
    verify_library,
)


@dataclass
class ComplianceReportConfig:
    """How to compile a compliance report."""

    verifier: VerifierConfig = None
    library_name: Optional[str] = None
    report_version: str = "1.0"


def build_compliance_report(
    library: ArrayProgramLibrary,
    *,
    config: Optional[ComplianceReportConfig] = None,
    diff: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a machine-readable compliance report for the whole library."""
    cfg = config or ComplianceReportConfig()
    verifier_config = cfg.verifier or VerifierConfig()

    audit = library.audit_report()
    verification = verify_library(
        library.entries, config=verifier_config
    )

    risk_counts = {
        RISK_SAFE: 0,
        RISK_WARN: 0,
        RISK_HIGH: 0,
    }
    safe_entries = 0
    for report in verification:
        risk_counts[report["worst_risk"]] = (
            risk_counts.get(report["worst_risk"], 0) + 1
        )
        if report["overall_safe"]:
            safe_entries += 1

    aggregate_risk = RISK_SAFE
    if risk_counts.get(RISK_HIGH, 0) > 0:
        aggregate_risk = RISK_HIGH
    elif risk_counts.get(RISK_WARN, 0) > 0:
        aggregate_risk = RISK_WARN

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_version": cfg.report_version,
        "library_name": cfg.library_name,
        "library_summary": audit["summary"],
        "per_skill_verification": verification,
        "aggregate": {
            "entry_count": audit["summary"]["entry_count"],
            "safe_entries": safe_entries,
            "unsafe_entries": audit["summary"]["entry_count"] - safe_entries,
            "risk_counts": risk_counts,
            "aggregate_risk": aggregate_risk,
        },
        "session_diff": diff,
        "verifier_assumptions": {
            "input_bound": verifier_config.input_bound.to_dict(),
            "max_length": int(verifier_config.max_length),
            "overflow_threshold": float(
                verifier_config.overflow_threshold
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compliance report as human-readable markdown."""
    lines: list[str] = []
    name = report.get("library_name") or "(unnamed library)"
    lines.append(f"# Compliance Report: {name}")
    lines.append("")
    lines.append(f"- **Report version**: {report['report_version']}")
    lines.append(f"- **Generated at**: {report['generated_at']}")
    lines.append("")
    lines.append("## Aggregate")
    agg = report["aggregate"]
    lines.append(f"- **Entries**: {agg['entry_count']}")
    lines.append(f"- **Safe entries**: {agg['safe_entries']}")
    lines.append(f"- **Unsafe entries**: {agg['unsafe_entries']}")
    lines.append(f"- **Aggregate risk**: `{agg['aggregate_risk']}`")
    risks = agg["risk_counts"]
    lines.append(
        f"- **Risk breakdown**: safe={risks.get(RISK_SAFE, 0)}, "
        f"warn={risks.get(RISK_WARN, 0)}, high={risks.get(RISK_HIGH, 0)}"
    )

    lines.append("")
    lines.append("## Verifier Assumptions")
    asm = report["verifier_assumptions"]
    lines.append(
        f"- Input bound: [{asm['input_bound']['lower']:.3g}, "
        f"{asm['input_bound']['upper']:.3g}]"
    )
    lines.append(f"- Max array length: {asm['max_length']}")
    lines.append(f"- Overflow threshold: {asm['overflow_threshold']:.3g}")

    if report.get("session_diff") is not None:
        lines.append("")
        lines.append("## Session Diff")
        diff = report["session_diff"]
        lines.append(f"- Added: {len(diff.get('added', []))}")
        lines.append(f"- Removed: {len(diff.get('removed', []))}")
        lines.append(f"- Changed: {len(diff.get('changed', []))}")
        lines.append(f"- Unchanged: {diff.get('unchanged', 0)}")
        lines.append(
            f"- Hits since snapshot: {diff.get('hits_since_snapshot', 0)}"
        )

    lines.append("")
    lines.append("## Per-Skill Analysis")
    if not report["per_skill_verification"]:
        lines.append("_(library is empty)_")
        return "\n".join(lines)

    for index, entry in enumerate(report["per_skill_verification"], start=1):
        task = entry.get("task_name") or "(unnamed)"
        lines.append(f"### Skill {index}: `{task}`")
        lines.append("")
        lines.append(
            f"- Worst risk: `{entry['worst_risk']}`, overall_safe: "
            f"`{entry['overall_safe']}`"
        )
        lines.append(f"- Hit count: {entry.get('hit_count', 0)}")
        if entry.get("output_bound") is not None:
            bound = entry["output_bound"]
            lines.append(
                f"- Output bound: [{bound['lower']:.3g}, "
                f"{bound['upper']:.3g}]"
            )
        lines.append("")
        lines.append("**Program**:")
        lines.append("")
        lines.append("```rust")
        lines.append(entry["program"]["program_text"])
        lines.append("```")
        lines.append("")
        lines.append("**Claims**:")
        lines.append("")
        for claim in entry["claims"]:
            verdict = "OK" if claim["verdict"] else "FAIL"
            lines.append(
                f"- `{claim['name']}` [{claim['risk_level']}] {verdict}: "
                f"{claim['message']}"
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "ComplianceReportConfig",
    "build_compliance_report",
    "render_markdown",
]
