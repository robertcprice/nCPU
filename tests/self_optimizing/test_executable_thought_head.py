"""Tests for the executable-thought M1 module."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import torch

from ncpu.self_optimizing.executable_thought_head import (
    ExecutableThoughtHead,
    ExecutableThoughtHeadConfig,
    load_executable_thought_head,
    run_executable_thought_smoke_train,
    train_executable_thought_head,
)


class TestExecutableThoughtHead(unittest.TestCase):
    def _make_head(self) -> ExecutableThoughtHead:
        config = ExecutableThoughtHeadConfig(
            hidden_dim=4,
            compiler_d_model=16,
            compiler_max_program_len=4,
            num_registers=4,
            execution_max_steps=4,
            output_register=2,
            trace_projection_dim=8,
            trace_hidden_dim=16,
            state_patch_dim=8,
            allowed_opcodes=("NOP", "ADD", "SUB", "MUL", "HALT"),
        )
        return ExecutableThoughtHead(config)

    def test_forward_returns_execution_artifacts(self):
        torch.manual_seed(0)
        head = self._make_head()
        hidden = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        register_inputs = torch.tensor(
            [
                [2.0, 3.0, 0.0, 0.0],
                [7.0, 2.0, 0.0, 0.0],
            ]
        )

        result = head(hidden, register_inputs, temperature=1.0)

        self.assertEqual(tuple(result.compiler_context.shape), (2, 16))
        self.assertEqual(tuple(result.predicted_output.shape), (2,))
        self.assertEqual(tuple(result.next_hidden_state.shape), (2, 4))
        self.assertEqual(tuple(result.trace_projection.shape), (2, 8))
        self.assertEqual(tuple(result.patch_signal.shape), (2, 8))
        self.assertEqual(tuple(result.execution_registers.shape), (2, 4))
        self.assertEqual(tuple(result.execution_flags.shape), (2, 4))
        self.assertEqual(len(result.program_texts), 2)
        self.assertEqual(len(result.mog_previews), 2)
        self.assertTrue(all("fn thought" in preview for preview in result.mog_previews))

    def test_backward_reaches_hidden_state_and_projector(self):
        torch.manual_seed(1)
        head = self._make_head()
        hidden = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        register_inputs = torch.tensor([3.0, 5.0, 0.0, 0.0])

        result = head(hidden, register_inputs, temperature=1.0)
        loss = (result.predicted_output - torch.tensor(8.0)) ** 2
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertGreater(float(hidden.grad.abs().sum().item()), 0.0)
        first_linear = head.context_projector[0]
        self.assertIsNotNone(first_linear.weight.grad)
        self.assertGreater(float(first_linear.weight.grad.abs().sum().item()), 0.0)

    def test_smoke_training_reduces_loss(self):
        torch.manual_seed(2)
        head = self._make_head()
        hidden = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # add
                [1.0, 0.0, 0.0, 0.0],  # add
                [0.0, 1.0, 0.0, 0.0],  # sub
                [0.0, 1.0, 0.0, 0.0],  # sub
                [0.0, 0.0, 1.0, 0.0],  # mul
                [0.0, 0.0, 1.0, 0.0],  # mul
            ]
        )
        register_inputs = torch.tensor(
            [
                [2.0, 3.0, 0.0, 0.0],
                [4.0, 1.0, 0.0, 0.0],
                [7.0, 2.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0],
                [2.0, 3.0, 0.0, 0.0],
                [4.0, 2.0, 0.0, 0.0],
            ]
        )
        targets = torch.tensor([5.0, 5.0, 5.0, 2.0, 6.0, 8.0])

        metrics = run_executable_thought_smoke_train(
            head,
            hidden_state=hidden,
            register_inputs=register_inputs,
            targets=targets,
            steps=60,
            learning_rate=5e-2,
            start_temperature=1.25,
            end_temperature=0.35,
        )

        self.assertLess(metrics.final_loss, metrics.initial_loss)
        self.assertLess(metrics.final_loss, metrics.initial_loss * 0.6)
        self.assertLess(metrics.final_mae, 2.0)
        self.assertEqual(len(metrics.final_program_texts), 6)

    def test_train_checkpoint_can_infer_hidden_dim_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = f"{tmpdir}/executable_thought_head.pt"
            config = ExecutableThoughtHeadConfig(
                hidden_dim=0,
                compiler_d_model=16,
                compiler_max_program_len=4,
                num_registers=4,
                execution_max_steps=4,
                output_register=2,
                trace_projection_dim=8,
                trace_hidden_dim=16,
                state_patch_dim=8,
                allowed_opcodes=("NOP", "ADD", "SUB", "MUL", "HALT"),
            )
            with mock.patch("transformers.AutoConfig.from_pretrained", return_value=mock.Mock(hidden_size=6)):
                metrics = train_executable_thought_head(
                    output_path=checkpoint_path,
                    config=config,
                    model_name_or_path="stub-model",
                    steps=30,
                    learning_rate=5e-2,
                    samples_per_op=4,
                    seed=7,
                    device="cpu",
                )

            self.assertTrue(metrics["trained"])
            self.assertEqual(metrics["config"]["hidden_dim"], 6)
            self.assertLess(metrics["final_loss"], metrics["initial_loss"])

            loaded = load_executable_thought_head(path=checkpoint_path, device="cpu")
            self.assertEqual(loaded.config.hidden_dim, 6)
            hidden = torch.zeros(6)
            hidden[0] = 1.0
            register_inputs = torch.tensor([2.0, 3.0, 0.0, 0.0])
            result = loaded(hidden, register_inputs)
            self.assertEqual(tuple(result.next_hidden_state.shape), (6,))
            self.assertEqual(len(result.program_texts), 1)


if __name__ == "__main__":
    unittest.main()
