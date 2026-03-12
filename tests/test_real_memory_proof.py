"""Tests for end-to-end real memory proof orchestration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ncpu.self_optimizing.run_real_memory_proof import run_real_memory_proof


class TestRealMemoryProof(unittest.TestCase):
    def test_run_real_memory_proof_writes_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "proof"
            benchmark_report = {
                "baseline": {"summary": {"success_rate": 0.5}},
                "some": {"summary": {"success_rate": 0.75}},
                "delta": {"overall_success_delta": 0.25},
            }
            memory_bundle = {
                "train_path": str(Path(tmpdir) / "latent_memory_train.jsonl"),
                "val_path": str(Path(tmpdir) / "latent_memory_val.jsonl"),
                "train_examples": 10,
                "val_examples": 4,
            }
            Path(memory_bundle["train_path"]).write_text('{"feature_vector":[0.0],"target_vector":[0.0]}\n', encoding="utf-8")
            Path(memory_bundle["val_path"]).write_text('{"feature_vector":[0.0],"target_vector":[0.0]}\n', encoding="utf-8")
            training_metrics = {"trained": True, "train_examples": 10, "val_examples": 4}
            evaluation_report = {"splits": {"val": {"improvement": {"relative_mse_improvement": 0.5}}}}

            with patch("ncpu.self_optimizing.run_real_memory_proof.run_model_benchmark", return_value=benchmark_report), \
                 patch("ncpu.self_optimizing.run_real_memory_proof.build_latent_memory_training_bundle", return_value=memory_bundle), \
                 patch("ncpu.self_optimizing.run_real_memory_proof.train_latent_memory_head", return_value=training_metrics), \
                 patch("ncpu.self_optimizing.run_real_memory_proof.evaluate_latent_memory_head", return_value=evaluation_report):
                result = run_real_memory_proof(
                    provider_name="local",
                    model="qwen3.5:4b",
                    output_dir=output_dir,
                    repeats=1,
                )

            self.assertEqual(result.provider, "local")
            self.assertEqual(result.model, "qwen3.5:4b")
            self.assertEqual(result.benchmark_summary["delta"]["overall_success_delta"], 0.25)
            self.assertEqual(result.memory_eval["splits"]["val"]["improvement"]["relative_mse_improvement"], 0.5)
            self.assertTrue((output_dir / "real_memory_proof.json").exists())
            self.assertTrue((output_dir / "benchmark_report.json").exists())


if __name__ == "__main__":
    unittest.main()
