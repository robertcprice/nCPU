"""Tests for executable-thought proof orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ncpu.self_optimizing.run_executable_thought_proof import run_executable_thought_proof


class TestExecutableThoughtProof(unittest.TestCase):
    def test_run_executable_thought_proof_writes_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "proof"
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            bundle_path.write_text(
                json.dumps(
                    {
                        "name": "demo",
                        "base_model": "stub-base-model",
                        "response": {"provider": "hf_fast_weights", "model": "response-adapter", "temperature": 0.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            benchmark_report = {
                "provider": "hf_fast_weights",
                "model": "response-adapter",
                "baseline": {"summary": {"success_rate": 0.5}},
                "some": {"summary": {"success_rate": 0.75}},
                "delta": {"overall_success_delta": 0.25},
            }
            executable_bundle = {
                "train_path": str(Path(tmpdir) / "executable_thought_train.jsonl"),
                "val_path": str(Path(tmpdir) / "executable_thought_val.jsonl"),
                "train_examples": 6,
                "val_examples": 2,
            }
            Path(executable_bundle["train_path"]).write_text(
                '{"prompt_text":"x","register_inputs":[0.0],"target_vector":[0.0]}\n',
                encoding="utf-8",
            )
            Path(executable_bundle["val_path"]).write_text(
                '{"prompt_text":"x","register_inputs":[0.0],"target_vector":[0.0]}\n',
                encoding="utf-8",
            )
            training_metrics = {"trained": True, "train_examples": 6, "val_examples": 2}
            evaluation_report = {"splits": {"val": {"program_summary": {"unique_programs": 2}}}}

            with (
                patch(
                    "ncpu.self_optimizing.run_executable_thought_proof.run_model_benchmark",
                    return_value=benchmark_report,
                ),
                patch(
                    "ncpu.self_optimizing.run_executable_thought_proof.build_executable_thought_training_bundle",
                    return_value=executable_bundle,
                ),
                patch(
                    "ncpu.self_optimizing.run_executable_thought_proof.train_executable_thought_head",
                    return_value=training_metrics,
                ),
                patch(
                    "ncpu.self_optimizing.run_executable_thought_proof.evaluate_executable_thought_head",
                    return_value=evaluation_report,
                ),
            ):
                result = run_executable_thought_proof(
                    controller_bundle_path=bundle_path,
                    output_dir=output_dir,
                    repeats=1,
                )

            self.assertEqual(result.provider, "hf_fast_weights")
            self.assertEqual(result.model, "response-adapter")
            self.assertEqual(result.benchmark_summary["delta"]["overall_success_delta"], 0.25)
            self.assertEqual(result.executable_thought_eval["splits"]["val"]["program_summary"]["unique_programs"], 2)
            self.assertTrue((output_dir / "executable_thought_proof.json").exists())
            self.assertTrue((output_dir / "benchmark_report.json").exists())


if __name__ == "__main__":
    unittest.main()
