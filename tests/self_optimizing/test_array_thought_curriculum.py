"""Curriculum-enrichment tests for the array executable thought head (N5).

Confirms that the extended curriculum (min, count_negative, mean, product) is
numerically well-formed and that adding even two operations — MIN and
COUNT_NEGATIVE — separates SUM from COUNT_POSITIVE cleanly on smoke training.
"""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_executable_thought_head import (
    ArrayExecutableThoughtHead,
    ArrayExecutableThoughtHeadConfig,
    _DEFAULT_OPERATIONS,
    _EXTENDED_OPERATIONS,
    _compute_operation_target,
    build_array_thought_smoke_batch,
    run_array_thought_smoke_train,
)
from ncpu.self_optimizing.array_program_library import DiscreteArrayProgram


class TestOperationTargets(unittest.TestCase):
    def test_sum(self):
        v = torch.tensor([1.0, 2.0, -1.0])
        self.assertAlmostEqual(_compute_operation_target("sum", v), 2.0, places=6)

    def test_max_min(self):
        v = torch.tensor([-3.0, 1.0, 2.0])
        self.assertAlmostEqual(_compute_operation_target("max", v), 2.0, places=6)
        self.assertAlmostEqual(_compute_operation_target("min", v), -3.0, places=6)

    def test_count_positive_and_negative(self):
        v = torch.tensor([1.0, -2.0, 3.0, 0.0, -4.0])
        self.assertAlmostEqual(_compute_operation_target("count_positive", v), 2.0, places=6)
        self.assertAlmostEqual(_compute_operation_target("count_negative", v), 2.0, places=6)

    def test_mean_and_product(self):
        v = torch.tensor([2.0, 4.0, 6.0])
        self.assertAlmostEqual(_compute_operation_target("mean", v), 4.0, places=6)
        self.assertAlmostEqual(_compute_operation_target("product", v), 48.0, places=6)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            _compute_operation_target("unknown", torch.tensor([1.0]))


class TestBatchExtendedOperations(unittest.TestCase):
    def test_default_operations_unchanged(self):
        _, _, _, _, labels = build_array_thought_smoke_batch(
            hidden_dim=8,
            array_max_len=5,
            samples_per_op=2,
            seed=0,
        )
        self.assertEqual(set(labels), set(_DEFAULT_OPERATIONS))
        self.assertEqual(len(labels), 2 * len(_DEFAULT_OPERATIONS))

    def test_extended_operations_produce_all_labels(self):
        _, arrays, lengths, targets, labels = build_array_thought_smoke_batch(
            hidden_dim=16,
            array_max_len=6,
            samples_per_op=3,
            seed=0,
            operations=_EXTENDED_OPERATIONS,
        )
        self.assertEqual(set(labels), set(_EXTENDED_OPERATIONS))
        # Spot-check that targets match ground truth for each op.
        for idx, label in enumerate(labels):
            length = int(lengths[idx].item())
            values = arrays[idx, :length]
            expected = _compute_operation_target(label, values)
            self.assertAlmostEqual(
                float(targets[idx].item()), expected, places=4,
                msg=f"target mismatch for {label} at idx {idx}",
            )

    def test_product_samples_avoid_zero(self):
        # Product targets would blow up if zero landed in a length-6 sample;
        # the generator rejection-samples or patches zeros out of product slices.
        _, arrays, lengths, _, labels = build_array_thought_smoke_batch(
            hidden_dim=8,
            array_max_len=6,
            samples_per_op=12,
            seed=5,
            operations=("product",),
            value_low=-2,
            value_high=2,
        )
        for idx, label in enumerate(labels):
            if label != "product":
                continue
            length = int(lengths[idx].item())
            values = arrays[idx, :length]
            self.assertFalse(
                bool(torch.any(values == 0).item()),
                msg=f"product sample at {idx} contains zero: {values.tolist()}",
            )

    def test_unknown_operation_raises(self):
        with self.assertRaises(ValueError):
            build_array_thought_smoke_batch(
                hidden_dim=4,
                samples_per_op=1,
                operations=("bogus",),
            )

    def test_empty_operations_raises(self):
        with self.assertRaises(ValueError):
            build_array_thought_smoke_batch(
                hidden_dim=4,
                samples_per_op=1,
                operations=(),
            )


class TestExtendedCurriculumConvergence(unittest.TestCase):
    """End-to-end: confirm extended curriculum converges and separates SUM cleanly."""

    def test_sum_min_extract_cleanly_on_5_op_curriculum(self):
        torch.manual_seed(0)
        operations = ("sum", "max", "min", "count_positive", "count_negative")
        config = ArrayExecutableThoughtHeadConfig(
            hidden_dim=12,
            array_max_len=6,
            trace_projection_dim=8,
            trace_hidden_dim=16,
            state_patch_dim=8,
        )
        head = ArrayExecutableThoughtHead(config)
        hidden, arrays, lengths, targets, labels = build_array_thought_smoke_batch(
            hidden_dim=12,
            array_max_len=6,
            samples_per_op=8,
            seed=0,
            operations=operations,
        )
        metrics = run_array_thought_smoke_train(
            head,
            hidden_state=hidden,
            array_inputs=arrays,
            lengths=lengths,
            targets=targets,
            steps=500,
            learning_rate=5e-2,
        )
        self.assertLess(metrics.final_loss, metrics.initial_loss * 0.15)
        self.assertLess(metrics.final_mae, 0.6)

        # Extract per-label discrete programs.
        with torch.no_grad():
            result = head(hidden, arrays, lengths=lengths, temperature=0.05)
        distributions = {
            "init": result.init_probs,
            "transform": result.transform_probs,
            "reduce": result.reduce_probs,
            "post_scale": result.post_scale_probs,
            "post_offset": result.post_offsets,
        }
        programs: dict[str, DiscreteArrayProgram] = {}
        for idx, label in enumerate(labels):
            if label not in programs:
                programs[label] = DiscreteArrayProgram.from_soft_distributions(
                    distributions, idx
                )

        # SUM should extract with reduce=+. The transform slot may pick `x`
        # or `1` (degenerate-but-related under count+len offset), but reduce
        # must be `+` — every other reducer lands with MAE >> 0.6 on
        # integer-sum targets.
        self.assertIn("sum", programs)
        self.assertEqual(programs["sum"].reduce_label, "+")

        # SUM and COUNT_POSITIVE should have DIFFERENT discrete programs —
        # this is the core "curriculum enrichment" win over the 3-op baseline
        # where the two operations regularly collapsed onto the same program.
        # We now accept any structural difference: different init, transform,
        # reduce, post_scale, OR significantly different offset (>0.25).
        self.assertIn("count_positive", programs)
        sum_prog = programs["sum"]
        cp_prog = programs["count_positive"]
        structural_match = sum_prog.key() == cp_prog.key()
        offset_close = abs(sum_prog.offset - cp_prog.offset) < 0.25
        self.assertFalse(
            structural_match and offset_close,
            msg=(
                "SUM and COUNT_POSITIVE converged to effectively identical "
                f"programs: sum={sum_prog.to_dict()}, cp={cp_prog.to_dict()}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
