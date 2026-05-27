"""Compliance report tests (NV5)."""

from __future__ import annotations

import json
import unittest

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.compliance_report import (
    ComplianceReportConfig,
    build_compliance_report,
    render_markdown,
)
from ncpu.self_optimizing.program_verifier import (
    RISK_SAFE,
    RISK_WARN,
    RangeBound,
    VerifierConfig,
)


def _populate_mixed_library() -> ArrayProgramLibrary:
    lib = ArrayProgramLibrary(
        ArrayProgramLibraryConfig(similarity_threshold=0.85)
    )
    lib.record(
        torch.tensor([1.0, 0.0, 0.0]),
        DiscreteArrayProgram(0, 0, 0, 0, 0.0),
        task_name="sum",
    )
    lib.record(
        torch.tensor([0.0, 1.0, 0.0]),
        DiscreteArrayProgram(1, 0, 1, 0, 0.0),
        task_name="naive_product",
    )
    return lib


class TestBuildComplianceReport(unittest.TestCase):
    def test_report_has_expected_top_level_keys(self):
        report = build_compliance_report(_populate_mixed_library())
        for key in (
            "generated_at",
            "report_version",
            "library_summary",
            "per_skill_verification",
            "aggregate",
            "verifier_assumptions",
        ):
            self.assertIn(key, report)

    def test_aggregate_classifies_mixed_library_as_warn(self):
        # naive_product triggers product_stability warning → aggregate=warn
        report = build_compliance_report(
            _populate_mixed_library(),
            config=ComplianceReportConfig(
                verifier=VerifierConfig(
                    input_bound=RangeBound(-3.0, 3.0),
                    max_length=8,
                )
            ),
        )
        self.assertEqual(report["aggregate"]["aggregate_risk"], RISK_WARN)
        self.assertEqual(report["aggregate"]["entry_count"], 2)
        self.assertEqual(report["aggregate"]["safe_entries"], 1)
        self.assertEqual(report["aggregate"]["unsafe_entries"], 1)

    def test_safe_library_has_safe_aggregate(self):
        lib = ArrayProgramLibrary()
        lib.record(
            torch.tensor([1.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),  # sum
            task_name="sum",
        )
        report = build_compliance_report(lib)
        self.assertEqual(report["aggregate"]["aggregate_risk"], RISK_SAFE)
        self.assertEqual(report["aggregate"]["safe_entries"], 1)

    def test_empty_library_produces_valid_report(self):
        report = build_compliance_report(ArrayProgramLibrary())
        self.assertEqual(report["aggregate"]["entry_count"], 0)
        self.assertEqual(report["per_skill_verification"], [])
        self.assertEqual(report["aggregate"]["aggregate_risk"], RISK_SAFE)

    def test_report_accepts_session_diff(self):
        diff = {
            "added": [{"task_name": "new"}],
            "removed": [],
            "changed": [],
            "unchanged": 1,
            "hits_since_snapshot": 3,
        }
        report = build_compliance_report(
            _populate_mixed_library(),
            diff=diff,
        )
        self.assertEqual(report["session_diff"], diff)

    def test_report_is_json_serializable(self):
        report = build_compliance_report(_populate_mixed_library())
        dumped = json.dumps(report)
        parsed = json.loads(dumped)
        self.assertEqual(parsed["aggregate"]["entry_count"], 2)

    def test_library_name_passed_through(self):
        report = build_compliance_report(
            _populate_mixed_library(),
            config=ComplianceReportConfig(library_name="sales_agent_v1"),
        )
        self.assertEqual(report["library_name"], "sales_agent_v1")


class TestRenderMarkdown(unittest.TestCase):
    def test_markdown_contains_sections(self):
        report = build_compliance_report(
            _populate_mixed_library(),
            config=ComplianceReportConfig(library_name="demo"),
        )
        md = render_markdown(report)
        self.assertIn("# Compliance Report", md)
        self.assertIn("## Aggregate", md)
        self.assertIn("## Verifier Assumptions", md)
        self.assertIn("## Per-Skill Analysis", md)
        self.assertIn("### Skill 1", md)

    def test_markdown_renders_empty_library(self):
        report = build_compliance_report(ArrayProgramLibrary())
        md = render_markdown(report)
        self.assertIn("(library is empty)", md)

    def test_markdown_includes_session_diff(self):
        diff = {
            "added": [{"task_name": "x"}],
            "removed": [],
            "changed": [],
            "unchanged": 0,
            "hits_since_snapshot": 5,
        }
        report = build_compliance_report(
            _populate_mixed_library(),
            diff=diff,
        )
        md = render_markdown(report)
        self.assertIn("## Session Diff", md)
        self.assertIn("Added: 1", md)
        self.assertIn("Hits since snapshot: 5", md)


if __name__ == "__main__":
    unittest.main()
