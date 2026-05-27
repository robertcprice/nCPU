"""Tests for trajectory-backed executable-thought supervision."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from ncpu.self_optimizing import (
    ExecutableThoughtHeadConfig,
    build_executable_thought_training_examples,
    load_executable_thought_head,
    load_trajectory,
    train_executable_thought_head,
    write_executable_thought_dataset,
)


def _write_executable_thought_trajectory(path: Path) -> None:
    events = [
        {
            "event": "workspace_initialized",
            "task_name": "fibonacci",
            "category": "coding",
            "status": "running",
            "max_generation_attempts": 3,
            "latent_state": {
                "confidence": 0.15,
                "verification_passes": 0,
                "verification_failures": 0,
                "descriptor_updates_used": 0,
                "fast_weight_updates_used": 0,
                "recent_actions": [],
                "failure_patterns": [],
                "verified_constraints": ["output_format=python"],
                "memory_vector": [0.0] * 16,
            },
        },
        {
            "event": "workspace_step",
            "task_name": "fibonacci",
            "status": "running",
            "step_index": 1,
            "action": "write",
            "success": True,
            "error": None,
            "metadata": {},
            "latent_state": {
                "confidence": 0.2,
                "verification_passes": 0,
                "verification_failures": 0,
                "descriptor_updates_used": 0,
                "fast_weight_updates_used": 0,
                "recent_actions": ["write"],
                "failure_patterns": [],
                "verified_constraints": ["output_format=python"],
                "memory_vector": [0.05] * 16,
            },
            "prompt": "Write fib(n).",
            "response_text": "def fib(n):\n    return 0\n",
        },
        {
            "event": "workspace_step",
            "task_name": "fibonacci",
            "status": "running",
            "step_index": 2,
            "action": "verify",
            "success": False,
            "error": "expected 8, got 0",
            "metadata": {"test_index": 2},
            "latent_state": {
                "confidence": 0.05,
                "verification_passes": 0,
                "verification_failures": 1,
                "descriptor_updates_used": 0,
                "fast_weight_updates_used": 0,
                "recent_actions": ["write", "verify"],
                "failure_patterns": ["expected 8, got 0"],
                "verified_constraints": ["output_format=python"],
                "memory_vector": [0.1, -0.05, 0.04, 0.0] + [0.0] * 12,
            },
            "prompt": "Verify fib(n).",
            "response_text": "expected 8, got 0",
        },
        {
            "event": "workspace_step",
            "task_name": "fibonacci",
            "status": "running",
            "step_index": 3,
            "action": "descriptor_update",
            "success": True,
            "error": None,
            "metadata": {
                "kind": "verify_failure_descriptor",
                "adaptation_descriptor": {
                    "signal_projection": [0.6, -0.25, 0.15, 0.05],
                    "update_kind": "verify_failure_descriptor",
                    "source": "latent_state+hf_hidden_state+executable_thought_head",
                },
            },
            "latent_state": {
                "confidence": 0.12,
                "verification_passes": 0,
                "verification_failures": 1,
                "descriptor_updates_used": 1,
                "fast_weight_updates_used": 0,
                "recent_actions": ["write", "verify", "descriptor_update"],
                "failure_patterns": ["expected 8, got 0"],
                "verified_constraints": [
                    "output_format=python",
                    "descriptor_update=verify_failure_descriptor",
                ],
                "memory_vector": [0.1, -0.05, 0.04, 0.0] + [0.0] * 12,
            },
            "prompt": "Apply descriptor update.",
            "response_text": "applied",
        },
        {
            "event": "workspace_committed",
            "task_name": "fibonacci",
            "status": "committed",
            "committed_verified": True,
            "generation_attempts": 1,
            "last_error": None,
            "latent_state": {},
            "committed_output": "def fib(n):\n    return n\n",
        },
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


class _TinyBatch(dict):
    def to(self, device: str) -> "_TinyBatch":
        for key, value in list(self.items()):
            self[key] = value.to(device)
        return self


class _TinyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False) -> _TinyBatch:
        del return_tensors, add_special_tokens
        token_ids = [((ord(ch) - 31) % 48) + 1 for ch in text][:64] or [1]
        return _TinyBatch({"input_ids": torch.tensor([token_ids], dtype=torch.long)})


class _TinyHiddenModel(nn.Module):
    def __init__(self, hidden_dim: int = 6):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_dim})()
        self.embed = nn.Embedding(64, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        **_kwargs,
    ):
        del attention_mask, use_cache
        hidden0 = self.embed(input_ids)
        hidden1 = torch.tanh(self.proj(hidden0))
        payload = type("Outputs", (), {})()
        payload.hidden_states = (hidden0, hidden1) if output_hidden_states else None
        payload.last_hidden_state = hidden1
        return payload


class TestExecutableThoughtTraining(unittest.TestCase):
    def test_build_executable_thought_training_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            _write_executable_thought_trajectory(trajectory_path)
            trajectory = load_trajectory(trajectory_path)

            examples = build_executable_thought_training_examples(
                trajectory,
                num_registers=8,
                output_dim=4,
            )

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.update_kind, "verify_failure_descriptor")
        self.assertEqual(example.register_inputs[:3], [8.0, 0.0, 0.0])
        self.assertEqual(example.target_vector, [0.6, -0.25, 0.15, 0.05])
        self.assertIn("expected 8, got 0", example.prompt_text)
        self.assertIn("def fib(n):", example.prompt_text)
        self.assertEqual(example.metadata["error_text"], "expected 8, got 0")

    def test_train_executable_thought_head_from_prompt_hidden_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            _write_executable_thought_trajectory(trajectory_path)
            trajectory = load_trajectory(trajectory_path)
            examples = build_executable_thought_training_examples(
                trajectory,
                num_registers=8,
                output_dim=4,
            )

            train_path = Path(tmpdir) / "train.jsonl"
            val_path = Path(tmpdir) / "val.jsonl"
            write_executable_thought_dataset(examples, train_path)
            write_executable_thought_dataset(examples, val_path)

            checkpoint_path = Path(tmpdir) / "executable_thought_head.pt"
            metrics = train_executable_thought_head(
                output_path=checkpoint_path,
                config=ExecutableThoughtHeadConfig(
                    hidden_dim=6,
                    compiler_d_model=16,
                    compiler_max_program_len=4,
                    num_registers=8,
                    execution_max_steps=4,
                    output_register=2,
                    trace_projection_dim=8,
                    trace_hidden_dim=16,
                    state_patch_dim=4,
                    allowed_opcodes=("NOP", "MOV_IMM", "MOV_REG", "ADD", "SUB", "MUL", "HALT"),
                ),
                steps=6,
                batch_size=1,
                learning_rate=1e-2,
                device="cpu",
                train_path=train_path,
                val_path=val_path,
                model=_TinyHiddenModel(hidden_dim=6),
                tokenizer=_TinyTokenizer(),
                max_prompt_tokens=256,
            )
            loaded = load_executable_thought_head(path=checkpoint_path, device="cpu")
            checkpoint_exists = checkpoint_path.exists()

        self.assertTrue(checkpoint_exists)
        self.assertTrue(metrics["trained"])
        self.assertEqual(metrics["objective"], "patch_signal_supervision")
        self.assertEqual(metrics["train_examples"], 1)
        self.assertEqual(metrics["val_examples"], 1)
        self.assertEqual(metrics["config"]["hidden_dim"], 6)
        self.assertEqual(loaded.config.state_patch_dim, 4)


if __name__ == "__main__":
    unittest.main()
