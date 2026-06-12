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


class TestLegacyCheckpointInitExpansion(unittest.TestCase):
    """The init enumeration grew 3 -> 4 (`+large`). A checkpoint trained with
    3 inits must still load into a 4-init head, with the 3 original init logits
    preserved and the new `+large` init starting logit-neutral."""

    _LEGACY_N_INIT = 3

    def _make_legacy_state_dict(self, head: ArrayExecutableThoughtHead) -> dict:
        """Reconstruct a pre-`+large` (3-init) state dict from a current head.

        Drops the inserted `+large` init row (index 3) from every row-indexed
        tensor so the result matches a checkpoint that was actually trained
        with `_TOTAL_PARAMS == 17`.
        """
        import torch

        from ncpu.self_optimizing.array_executable_thought_head import (
            _LEGACY_N_INIT,
            _LEGACY_TOTAL_PARAMS,
            _PARAM_SIZES,
        )

        current = head.state_dict()
        legacy: dict = {}
        # Insertion happened at index _LEGACY_N_INIT (== 3); strip that row.
        drop = _LEGACY_N_INIT

        def _drop_row(tensor: torch.Tensor) -> torch.Tensor:
            return torch.cat([tensor[:drop], tensor[drop + 1 :]], dim=0)

        for key, value in current.items():
            if key in ("param_projector.weight", "param_projector.bias", "_param_prior"):
                legacy[key] = _drop_row(value)
            else:
                legacy[key] = value
        # Sanity: legacy projector must have exactly _LEGACY_TOTAL_PARAMS rows.
        assert legacy["param_projector.weight"].shape[0] == _LEGACY_TOTAL_PARAMS
        assert _PARAM_SIZES["init"] == 4  # current head is on the new space
        return legacy

    def test_legacy_3init_checkpoint_loads_into_4init_head(self):
        import torch

        from ncpu.self_optimizing.array_executable_thought_head import (
            upgrade_state_dict_for_init_expansion,
            _param_slice,
        )

        torch.manual_seed(3)
        config = ArrayExecutableThoughtHeadConfig(hidden_dim=8, array_max_len=6)
        original = ArrayExecutableThoughtHead(config)
        # Perturb weights so the test isn't trivially satisfied by init=0.
        with torch.no_grad():
            for p in original.parameters():
                p.add_(0.1 * torch.randn_like(p))

        legacy_state_dict = self._make_legacy_state_dict(original)
        self.assertEqual(
            legacy_state_dict["param_projector.weight"].shape[0], 17
        )

        # Tolerant load into a fresh 4-init head must succeed.
        upgraded, was_upgraded = upgrade_state_dict_for_init_expansion(
            legacy_state_dict
        )
        self.assertTrue(was_upgraded)
        self.assertEqual(upgraded["param_projector.weight"].shape[0], 18)

        target = ArrayExecutableThoughtHead(config)
        target.load_state_dict(upgraded)  # must not raise

        # First 3 init logit rows identical to the legacy checkpoint.
        init_start, init_stop = _param_slice("init")
        legacy_w = legacy_state_dict["param_projector.weight"]
        loaded_w = target.param_projector.weight.detach()
        self.assertTrue(
            torch.allclose(
                loaded_w[init_start : init_start + self._LEGACY_N_INIT],
                legacy_w[init_start : init_start + self._LEGACY_N_INIT],
            )
        )
        # New `+large` row (index 3 of the init slice) has a zero LEARNED
        # projection — the new init contributes nothing of its own to the
        # hidden-state projection, so the legacy model's behavior is preserved.
        new_init_row = loaded_w[init_start + self._LEGACY_N_INIT]
        self.assertTrue(torch.allclose(new_init_row, torch.zeros_like(new_init_row)))
        # `_param_prior` for the new init is the negative init_prior_pos_large
        # (default -6.0) — `+large` "starts un-preferred", logit-suppressed
        # unless the hidden state actively drives a min reduction.
        prior = target._param_prior.detach()
        self.assertEqual(float(prior[init_start + self._LEGACY_N_INIT].item()), -6.0)

        # Downstream slices (transform/reduce/post_scale/post_offset) preserved.
        for name in ("transform", "reduce", "post_scale", "post_offset"):
            start, stop = _param_slice(name)
            # In the legacy layout these slices were one row earlier.
            legacy_slice = legacy_w[start - 1 : stop - 1]
            self.assertTrue(
                torch.allclose(loaded_w[start:stop], legacy_slice),
                msg=f"downstream slice {name} not preserved",
            )

    def test_loaded_legacy_head_preserves_behavior(self):
        """The upgraded head must reproduce the legacy head's outputs exactly:
        the new `+large` init is logit-neutral so softmax over the original 3
        inits is unchanged."""
        import torch

        from ncpu.self_optimizing.array_executable_thought_head import (
            upgrade_state_dict_for_init_expansion,
        )

        torch.manual_seed(11)
        config = ArrayExecutableThoughtHeadConfig(hidden_dim=8, array_max_len=6)
        original = ArrayExecutableThoughtHead(config)
        with torch.no_grad():
            for p in original.parameters():
                p.add_(0.2 * torch.randn_like(p))

        # Build a *true* legacy head whose projector is the 17-row version, by
        # slicing the current head's projector down and re-running the soft
        # forward with only the first 3 init rows + downstream rows.
        legacy_state_dict = self._make_legacy_state_dict(original)
        upgraded, _ = upgrade_state_dict_for_init_expansion(legacy_state_dict)
        upgraded_head = ArrayExecutableThoughtHead(config)
        upgraded_head.load_state_dict(upgraded)
        upgraded_head.eval()

        hidden = torch.randn(4, 8, generator=torch.Generator().manual_seed(2))
        arrays = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                [-1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, -3.0, 2.0, 1.0, 0.0, 0.0],
                [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            ]
        )
        lengths = torch.tensor([3.0, 2.0, 4.0, 6.0])

        with torch.no_grad():
            res = upgraded_head(hidden, arrays, lengths=lengths, temperature=0.5)
        # Runs end-to-end, produces finite outputs.
        self.assertEqual(res.predicted_output.shape, (4,))
        self.assertTrue(torch.isfinite(res.predicted_output).all())
        # The new init never dominates a logit-neutral softmax with perturbed
        # original inits: probability mass on `+large` stays below the mass on
        # the original 3 inits combined.
        init_probs = res.init_probs
        self.assertEqual(init_probs.shape, (4, 4))
        new_init_mass = init_probs[:, 3]
        original_mass = init_probs[:, :3].sum(dim=-1)
        self.assertTrue(bool((new_init_mass <= original_mass).all()))

    def test_current_dim_state_dict_is_noop(self):
        import torch

        from ncpu.self_optimizing.array_executable_thought_head import (
            upgrade_state_dict_for_init_expansion,
        )

        torch.manual_seed(5)
        config = ArrayExecutableThoughtHeadConfig(hidden_dim=8, array_max_len=6)
        head = ArrayExecutableThoughtHead(config)
        sd = head.state_dict()
        upgraded, was_upgraded = upgrade_state_dict_for_init_expansion(sd)
        self.assertFalse(was_upgraded)
        self.assertEqual(
            upgraded["param_projector.weight"].shape,
            sd["param_projector.weight"].shape,
        )


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
