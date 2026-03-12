"""Tests for controller training bundle preparation."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ncpu.self_optimizing import (
    BufferedInternalController,
    ControllerBundle,
    InternalControllerConfig,
    InternalDeliberationTask,
    TrajectoryLogger,
    build_action_policy_examples,
    build_controller_training_bundle,
    load_controller_bundle,
    load_weight_cpu_blueprint,
    load_trajectory,
)


FIB_TESTS = [
    {"input": {"n": 0}, "expected": 0},
    {"input": {"n": 1}, "expected": 1},
    {"input": {"n": 6}, "expected": 8},
]


class TestControllerTraining(unittest.TestCase):
    """Validate action-policy extraction and bundle preparation."""

    def _write_successful_trajectory(self, path: Path, *, task_name: str, function_name: str) -> None:
        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return f"Use recursion for {function_name}."
            if "repairing a hidden candidate" in prompt.lower():
                return (
                    f"def {function_name}(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    f"    return {function_name}(n - 1) + {function_name}(n - 2)\n"
                )
            return f"def {function_name}(n):\n    return 0\n"

        controller = BufferedInternalController(
            llm_provider=fake_provider,
            config=InternalControllerConfig(max_generation_attempts=3),
            trajectory_logger=TrajectoryLogger(str(path)),
        )
        task = InternalDeliberationTask(
            name=task_name,
            prompt=f"Write {function_name}(n) in Python.",
            test_cases=FIB_TESTS,
        )
        workspace = controller.run_task(task)
        self.assertEqual(workspace.status, "committed")
        self.assertTrue(workspace.committed_verified)

    def test_build_action_policy_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            self._write_successful_trajectory(trajectory_path, task_name="fibonacci", function_name="fib")

            trajectory = load_trajectory(trajectory_path)
            examples = build_action_policy_examples(trajectory)

            self.assertEqual([example.target_action for example in examples], ["think", "write", "patch", "commit"])
            self.assertIn("Choose the single best next action", examples[0].prompt)
            self.assertEqual(examples[-1].messages[-1]["content"], "commit")

    def test_build_controller_training_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "trajectories"
            root.mkdir()
            self._write_successful_trajectory(root / "fib.jsonl", task_name="fibonacci", function_name="fib")
            self._write_successful_trajectory(root / "trib.jsonl", task_name="tribonacci", function_name="trib")

            bundle = build_controller_training_bundle(root, Path(tmpdir) / "bundle", val_ratio=0.5)

            self.assertTrue(Path(bundle.response_train_path).exists())
            self.assertTrue(Path(bundle.response_val_path).exists())
            self.assertTrue(Path(bundle.action_train_path or "").exists())
            self.assertTrue(Path(bundle.action_val_path or "").exists())
            self.assertTrue(Path(bundle.latent_action_train_path or "").exists())
            self.assertTrue(Path(bundle.latent_action_val_path or "").exists())
            self.assertTrue(Path(bundle.latent_descriptor_train_path or "").exists())
            self.assertTrue(Path(bundle.latent_descriptor_val_path or "").exists())
            self.assertTrue(Path(bundle.latent_memory_train_path or "").exists())
            self.assertTrue(Path(bundle.latent_memory_val_path or "").exists())
            self.assertTrue(Path(bundle.latent_halt_train_path or "").exists())
            self.assertTrue(Path(bundle.latent_halt_val_path or "").exists())
            self.assertTrue(Path(bundle.state_patch_train_path or "").exists())
            self.assertTrue(Path(bundle.state_patch_val_path or "").exists())
            manifest = json.loads(Path(bundle.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["response_sft"]["train_source_files"], 1)
            self.assertEqual(manifest["response_sft"]["val_source_files"], 1)
            self.assertEqual(manifest["action_policy"]["action_labels"], ["think", "write", "patch", "commit", "fail"])
            self.assertIn("latent_descriptor_head", manifest)
            self.assertIn("latent_memory_head", manifest)
            self.assertEqual(manifest["latent_halt_policy"]["action_labels"], ["continue", "commit", "fail"])
            self.assertIn("state_patch_head", manifest)

    def test_prepare_only_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "trajectories"
            root.mkdir()
            self._write_successful_trajectory(root / "fib.jsonl", task_name="fibonacci", function_name="fib")
            self._write_successful_trajectory(root / "trib.jsonl", task_name="tribonacci", function_name="trib")

            output_dir = Path(tmpdir) / "training_out"
            script_path = Path(__file__).resolve().parents[1] / "ncpu" / "self_optimizing" / "train_internal_controller.py"
            result = subprocess.run(
                [
                    "python3",
                    str(script_path),
                    "--trajectory-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--prepare-only",
                    "--val-ratio",
                    "0.5",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("Prepared controller training bundle", result.stdout)
            manifest_path = output_dir / "prepared_datasets" / "training_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["action_policy"]["train_source_files"], 1)
            self.assertIn("latent_action_policy", manifest)
            self.assertGreaterEqual(manifest["latent_action_policy"]["train_examples"], 1)
            self.assertIn("latent_halt_policy", manifest)
            self.assertGreaterEqual(manifest["latent_halt_policy"]["train_examples"], 1)
            bundle_path = output_dir / "controller_bundle.template.json"
            self.assertTrue(bundle_path.exists())
            bundle = load_controller_bundle(bundle_path)
            self.assertIsInstance(bundle, ControllerBundle)
            self.assertEqual(bundle.response.provider, "hf_local")
            self.assertIn("adapters/response", bundle.response.model)
            self.assertTrue(bundle.metadata["prepared_only"])
            self.assertEqual(bundle.latent_action_head_path, str(output_dir / "latent_action_head.pt"))
            self.assertEqual(bundle.latent_action_head_config["hidden_dim"], 64)
            self.assertEqual(bundle.latent_memory_head_path, str(output_dir / "latent_memory_head.pt"))
            self.assertEqual(bundle.latent_memory_head_config["hidden_dim"], 64)
            self.assertEqual(bundle.latent_halt_head_path, str(output_dir / "latent_halt_head.pt"))
            self.assertEqual(bundle.latent_halt_head_config["hidden_dim"], 48)
            fast_weight_bundle_path = output_dir / "controller_bundle.fast_weights.template.json"
            self.assertTrue(fast_weight_bundle_path.exists())
            fast_weight_bundle = load_controller_bundle(fast_weight_bundle_path)
            self.assertEqual(fast_weight_bundle.response.provider, "hf_fast_weights")
            self.assertEqual(fast_weight_bundle.response.provider_kwargs["fast_weights_rank"], 8)
            self.assertTrue(fast_weight_bundle.response.provider_kwargs["fast_weights_use_ncpu_adaptation"])
            self.assertEqual(fast_weight_bundle.response.provider_kwargs["decode_backend"], "segmented_kv")
            self.assertEqual(fast_weight_bundle.latent_action_head_path, str(output_dir / "latent_action_head.pt"))
            self.assertEqual(fast_weight_bundle.latent_memory_head_path, str(output_dir / "latent_memory_head.pt"))
            self.assertEqual(fast_weight_bundle.latent_halt_head_path, str(output_dir / "latent_halt_head.pt"))
            self.assertEqual(
                fast_weight_bundle.response.provider_kwargs["latent_descriptor_head_path"],
                str(output_dir / "latent_descriptor_head.pt"),
            )
            self.assertEqual(
                fast_weight_bundle.response.provider_kwargs["latent_descriptor_head_config"]["hidden_dim"],
                64,
            )
            self.assertEqual(
                fast_weight_bundle.response.provider_kwargs["state_patch_head_path"],
                str(output_dir / "state_patch_head.pt"),
            )
            self.assertEqual(
                fast_weight_bundle.response.provider_kwargs["state_patch_head_config"]["hidden_dim"],
                64,
            )
            self.assertFalse(fast_weight_bundle.controller_config["fast_weight_updates_on_plan"])
            self.assertTrue(fast_weight_bundle.controller_config["descriptor_updates_on_plan"])
            self.assertEqual(
                fast_weight_bundle.response.provider_kwargs["fast_weights_target_modules"],
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            blueprint_path = output_dir / "weight_cpu_blueprint.template.json"
            self.assertTrue(blueprint_path.exists())
            blueprint = load_weight_cpu_blueprint(blueprint_path)
            self.assertEqual(blueprint.controller_bundle_path, str(bundle_path))
            self.assertEqual(blueprint.training_stages[0].name, "stage1_hidden_controller_distillation")


if __name__ == "__main__":
    unittest.main()
