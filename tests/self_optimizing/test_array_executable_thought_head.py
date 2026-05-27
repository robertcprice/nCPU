"""Tests for the array-valued executable thought head (NPCoT milestone M2)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.array_executable_thought_head import (
    ArrayExecutableThoughtHead,
    ArrayExecutableThoughtHeadConfig,
    build_array_thought_smoke_batch,
    load_array_thought_head,
    run_array_thought_smoke_train,
    train_array_thought_head,
)


class TestArrayExecutableThoughtHead(unittest.TestCase):
    def _make_head(self, hidden_dim: int = 8, array_max_len: int = 6) -> ArrayExecutableThoughtHead:
        config = ArrayExecutableThoughtHeadConfig(
            hidden_dim=hidden_dim,
            array_max_len=array_max_len,
            trace_projection_dim=8,
            trace_hidden_dim=16,
            state_patch_dim=8,
        )
        return ArrayExecutableThoughtHead(config)

    def test_forward_shapes_and_program_count(self):
        torch.manual_seed(0)
        head = self._make_head()
        hidden = torch.randn(3, 8)
        arrays = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                [-1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, -3.0, 2.0, 1.0, 0.0, 0.0],
            ]
        )
        lengths = torch.tensor([3.0, 2.0, 4.0])
        result = head(hidden, arrays, lengths=lengths, temperature=1.0)

        self.assertEqual(result.predicted_output.shape, (3,))
        self.assertEqual(result.next_hidden_state.shape, (3, 8))
        self.assertEqual(result.trace_projection.shape, (3, head.config.trace_projection_dim))
        self.assertEqual(result.patch_signal.shape, (3, head.config.state_patch_dim))
        self.assertEqual(len(result.program_texts), 3)
        for text in result.program_texts:
            self.assertIn("fn array_thought", text)
            self.assertIn("for i in 0..arr.len()", text)

    def test_forward_rank1_hidden_state(self):
        torch.manual_seed(0)
        head = self._make_head()
        hidden = torch.randn(8)
        arrays = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])
        lengths = torch.tensor([3.0])
        result = head(hidden, arrays, lengths=lengths)
        self.assertEqual(result.predicted_output.ndim, 0)
        self.assertEqual(result.next_hidden_state.shape, (8,))

    def test_gradient_flows_to_param_projector(self):
        torch.manual_seed(0)
        head = self._make_head()
        hidden = torch.randn(2, 8, requires_grad=True)
        arrays = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                [-1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        lengths = torch.tensor([3.0, 2.0])
        result = head(hidden, arrays, lengths=lengths, temperature=1.0)
        loss = result.predicted_output.pow(2).sum()
        loss.backward()
        self.assertIsNotNone(head.param_projector.weight.grad)
        self.assertGreater(head.param_projector.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(hidden.grad)

    def test_default_length_uses_full_array(self):
        torch.manual_seed(0)
        head = self._make_head()
        hidden = torch.randn(1, 8)
        arrays = torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
        with_len = head(hidden, arrays, lengths=torch.tensor([6.0]))
        without_len = head(hidden, arrays)
        # Both should be equivalent when length == array_max_len.
        self.assertAlmostEqual(
            float(with_len.predicted_output.item()),
            float(without_len.predicted_output.item()),
            places=5,
        )

    def test_smoke_training_reduces_loss(self):
        torch.manual_seed(0)
        head = self._make_head()
        hidden, arrays, lengths, targets, _ = build_array_thought_smoke_batch(
            hidden_dim=8,
            array_max_len=6,
            samples_per_op=6,
            seed=0,
        )
        metrics = run_array_thought_smoke_train(
            head,
            hidden_state=hidden,
            array_inputs=arrays,
            lengths=lengths,
            targets=targets,
            steps=150,
            learning_rate=5e-2,
        )
        # Smoke run must drop loss substantially and reach a reasonable MAE.
        self.assertLess(metrics.final_loss, metrics.initial_loss * 0.25)
        self.assertLess(metrics.final_mae, 1.5)
        self.assertEqual(len(metrics.final_program_texts), 18)

    def test_max_program_discretizes_correctly(self):
        # Longer run: with enough steps the "max" hidden prototype should
        # converge to a discrete program containing "max(acc, arr[i])".
        torch.manual_seed(0)
        head = self._make_head()
        hidden, arrays, lengths, targets, labels = build_array_thought_smoke_batch(
            hidden_dim=8,
            array_max_len=6,
            samples_per_op=6,
            seed=0,
        )
        run_array_thought_smoke_train(
            head,
            hidden_state=hidden,
            array_inputs=arrays,
            lengths=lengths,
            targets=targets,
            steps=300,
            learning_rate=5e-2,
        )
        # Decode at very low temperature for crisp argmax.
        with torch.no_grad():
            result = head(hidden, arrays, lengths=lengths, temperature=0.1)

        # Collect discrete program texts for the "max" samples only.
        max_texts = [
            text for text, label in zip(result.program_texts, labels) if label == "max"
        ]
        self.assertTrue(any("max(acc, arr[i])" in text for text in max_texts))

    def test_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "array_thought.pt"
            config = ArrayExecutableThoughtHeadConfig(hidden_dim=8, array_max_len=6)
            summary = train_array_thought_head(
                output_path=checkpoint,
                config=config,
                samples_per_op=4,
                steps=60,
                learning_rate=5e-2,
                seed=0,
            )
            self.assertTrue(checkpoint.exists())
            self.assertTrue(summary["trained"])
            head = load_array_thought_head(path=checkpoint, device="cpu")
            self.assertIsInstance(head, ArrayExecutableThoughtHead)
            self.assertEqual(head.config.hidden_dim, 8)
            self.assertEqual(head.config.array_max_len, 6)

            # Loaded head produces deterministic outputs given fixed inputs.
            hidden = torch.randn(2, 8, generator=torch.Generator().manual_seed(1))
            arrays = torch.tensor(
                [
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.0, -2.0, 0.0, 0.0],
                ]
            )
            lengths = torch.tensor([3.0, 4.0])
            with torch.no_grad():
                output_a = head(hidden, arrays, lengths=lengths, temperature=0.3).predicted_output
                output_b = head(hidden, arrays, lengths=lengths, temperature=0.3).predicted_output
            self.assertTrue(torch.allclose(output_a, output_b))


class TestSmokeBatchGeneration(unittest.TestCase):
    def test_batch_shapes_match_operations(self):
        hidden, arrays, lengths, targets, labels = build_array_thought_smoke_batch(
            hidden_dim=4,
            array_max_len=5,
            samples_per_op=3,
            seed=0,
        )
        self.assertEqual(hidden.shape, (9, 4))
        self.assertEqual(arrays.shape, (9, 5))
        self.assertEqual(lengths.shape, (9,))
        self.assertEqual(targets.shape, (9,))
        self.assertEqual(labels, ["sum"] * 3 + ["max"] * 3 + ["count_positive"] * 3)
        # Lengths must be within [1, array_max_len]
        self.assertTrue(bool((lengths >= 1.0).all()))
        self.assertTrue(bool((lengths <= 5.0).all()))

    def test_targets_match_ground_truth(self):
        _, arrays, lengths, targets, labels = build_array_thought_smoke_batch(
            hidden_dim=4,
            array_max_len=5,
            samples_per_op=2,
            seed=0,
        )
        for idx, label in enumerate(labels):
            length = int(lengths[idx].item())
            row = arrays[idx, :length]
            if label == "sum":
                self.assertAlmostEqual(float(targets[idx].item()), float(row.sum().item()), places=5)
            elif label == "max":
                self.assertAlmostEqual(float(targets[idx].item()), float(row.max().item()), places=5)
            else:
                expected = float((row > 0).to(torch.float32).sum().item())
                self.assertAlmostEqual(float(targets[idx].item()), expected, places=5)


if __name__ == "__main__":
    unittest.main()
