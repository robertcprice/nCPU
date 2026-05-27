"""Tests for the array-program library (NPCoT milestone M3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.array_executable_thought_head import (
    ArrayExecutableThoughtHead,
    ArrayExecutableThoughtHeadConfig,
    build_array_thought_smoke_batch,
    run_array_thought_smoke_train,
)
from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    ArrayThoughtLibraryResult,
    DiscreteArrayProgram,
    LibraryEntry,
)


class TestDiscreteArrayProgram(unittest.TestCase):
    def test_execute_sum_matches_ground_truth(self):
        program = DiscreteArrayProgram(
            init_idx=0,
            transform_idx=0,
            reduce_idx=0,
            post_scale_idx=0,
            offset=0.0,
        )
        arrays = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0],
                [4.0, 5.0, 0.0, 0.0, 0.0],
            ]
        )
        lengths = torch.tensor([3.0, 2.0])
        result = program.execute(arrays, lengths)
        self.assertTrue(torch.allclose(result, torch.tensor([6.0, 9.0])))

    def test_execute_max_ignores_padding(self):
        program = DiscreteArrayProgram(
            init_idx=2,
            transform_idx=0,
            reduce_idx=2,
            post_scale_idx=0,
            offset=0.0,
        )
        arrays = torch.tensor(
            [
                [-1.0, -3.0, -2.0, 99.0, 99.0],
                [5.0, 0.0, -4.0, 99.0, 99.0],
            ]
        )
        lengths = torch.tensor([3.0, 3.0])
        result = program.execute(arrays, lengths)
        self.assertTrue(torch.allclose(result, torch.tensor([-1.0, 5.0])))

    def test_execute_count_positive(self):
        program = DiscreteArrayProgram(
            init_idx=0,
            transform_idx=4,
            reduce_idx=0,
            post_scale_idx=0,
            offset=0.0,
        )
        arrays = torch.tensor(
            [
                [1.0, -2.0, 3.0, 0.0, 0.0],
                [-1.0, -2.0, -3.0, 0.0, 0.0],
            ]
        )
        lengths = torch.tensor([3.0, 3.0])
        result = program.execute(arrays, lengths)
        self.assertTrue(torch.allclose(result, torch.tensor([2.0, 0.0])))

    def test_execute_mean(self):
        program = DiscreteArrayProgram(
            init_idx=0,
            transform_idx=0,
            reduce_idx=0,
            post_scale_idx=1,
            offset=0.0,
        )
        arrays = torch.tensor(
            [
                [2.0, 4.0, 6.0, 0.0, 0.0],
            ]
        )
        lengths = torch.tensor([3.0])
        result = program.execute(arrays, lengths)
        self.assertTrue(torch.allclose(result, torch.tensor([4.0])))

    def test_execute_rejects_wrong_shape(self):
        program = DiscreteArrayProgram(
            init_idx=0,
            transform_idx=0,
            reduce_idx=0,
            post_scale_idx=0,
            offset=0.0,
        )
        with self.assertRaises(ValueError):
            program.execute(torch.zeros(3), torch.tensor([3.0]))
        with self.assertRaises(ValueError):
            program.execute(torch.zeros(2, 3), torch.tensor([3.0]))

    def test_render_contains_canonical_shape(self):
        program = DiscreteArrayProgram(
            init_idx=0,
            transform_idx=0,
            reduce_idx=0,
            post_scale_idx=0,
            offset=0.0,
        )
        text = program.render()
        self.assertIn("fn array_thought", text)
        self.assertIn("acc += arr[i]", text)
        self.assertIn("return acc", text)

    def test_to_from_dict_roundtrip(self):
        program = DiscreteArrayProgram(
            init_idx=1,
            transform_idx=2,
            reduce_idx=3,
            post_scale_idx=1,
            offset=-0.5,
        )
        recovered = DiscreteArrayProgram.from_dict(program.to_dict())
        self.assertEqual(program.key(), recovered.key())
        self.assertAlmostEqual(program.offset, recovered.offset)

    def test_from_soft_distributions_matches_argmax(self):
        distributions = {
            "init": torch.tensor([[0.1, 0.7, 0.2]]),
            "transform": torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.6]]),
            "reduce": torch.tensor([[0.8, 0.1, 0.05, 0.05]]),
            "post_scale": torch.tensor([[0.3, 0.7]]),
            "post_offset": torch.tensor([0.25]),
        }
        program = DiscreteArrayProgram.from_soft_distributions(distributions, 0)
        self.assertEqual(program.init_idx, 1)
        self.assertEqual(program.transform_idx, 4)
        self.assertEqual(program.reduce_idx, 0)
        self.assertEqual(program.post_scale_idx, 1)
        self.assertAlmostEqual(program.offset, 0.25, places=6)


class TestArrayProgramLibrary(unittest.TestCase):
    def _sum_program(self) -> DiscreteArrayProgram:
        return DiscreteArrayProgram(0, 0, 0, 0, 0.0)

    def _max_program(self) -> DiscreteArrayProgram:
        return DiscreteArrayProgram(2, 0, 2, 0, 0.0)

    def test_lookup_returns_none_on_empty_library(self):
        library = ArrayProgramLibrary()
        self.assertIsNone(library.lookup(torch.tensor([1.0, 0.0])))

    def test_record_then_lookup_hits_on_aligned_hidden(self):
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.9)
        )
        hidden = torch.tensor([1.0, 0.0, 0.0])
        entry = library.record(hidden, self._sum_program(), task_name="sum")
        self.assertEqual(len(library), 1)
        self.assertEqual(entry.task_name, "sum")

        similar = torch.tensor([0.95, 0.1, 0.05])
        hit = library.lookup(similar)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.program.key(), self._sum_program().key())
        self.assertEqual(hit.hit_count, 1)

    def test_lookup_miss_below_threshold(self):
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.98)
        )
        library.record(torch.tensor([1.0, 0.0, 0.0]), self._sum_program())
        distant = torch.tensor([0.3, 0.9, 0.3])
        self.assertIsNone(library.lookup(distant))

    def test_record_overwrites_near_duplicate_signature(self):
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.9)
        )
        library.record(torch.tensor([1.0, 0.0]), self._sum_program())
        library.record(torch.tensor([0.99, 0.05]), self._max_program())
        self.assertEqual(len(library), 1)
        entry = library.lookup(torch.tensor([1.0, 0.0]))
        self.assertIsNotNone(entry)
        self.assertEqual(entry.program.key(), self._max_program().key())

    def test_record_rejects_zero_hidden_state(self):
        library = ArrayProgramLibrary()
        with self.assertRaises(ValueError):
            library.record(torch.zeros(4), self._sum_program())

    def test_capacity_eviction_prefers_hit_entries(self):
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.95, max_entries=2)
        )
        library.record(torch.tensor([1.0, 0.0, 0.0]), self._sum_program(), task_name="a")
        hit_entry = library.record(
            torch.tensor([0.0, 1.0, 0.0]), self._max_program(), task_name="b"
        )
        # Warm up entry B so it survives eviction.
        for _ in range(3):
            library.lookup(torch.tensor([0.0, 1.0, 0.0]))
        library.record(
            torch.tensor([0.0, 0.0, 1.0]),
            self._sum_program(),
            task_name="c",
        )
        self.assertLessEqual(len(library), 2)
        task_names = {entry.task_name for entry in library.entries}
        self.assertIn("b", task_names)
        self.assertEqual(hit_entry.hit_count, 3)

    def test_json_roundtrip_preserves_entries(self):
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.91, max_entries=64)
        )
        library.record(torch.tensor([1.0, 2.0, 3.0]), self._sum_program(), task_name="sum")
        library.record(torch.tensor([0.0, 0.0, 5.0]), self._max_program(), task_name="max")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            payload = json.loads(path.read_text())
            self.assertEqual(len(payload["entries"]), 2)
            restored = ArrayProgramLibrary.load(path)

        self.assertEqual(len(restored), 2)
        self.assertAlmostEqual(
            restored.config.similarity_threshold,
            library.config.similarity_threshold,
        )
        self.assertEqual(
            {entry.task_name for entry in restored.entries},
            {"sum", "max"},
        )


class TestHeadConsultLibrary(unittest.TestCase):
    def _make_head(self) -> ArrayExecutableThoughtHead:
        config = ArrayExecutableThoughtHeadConfig(
            hidden_dim=8,
            array_max_len=6,
            trace_projection_dim=8,
            trace_hidden_dim=16,
            state_patch_dim=8,
        )
        return ArrayExecutableThoughtHead(config)

    def _train(self, head: ArrayExecutableThoughtHead):
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
        return hidden, arrays, lengths, targets, labels

    def test_consult_library_cold_runs_soft_forward(self):
        torch.manual_seed(0)
        head = self._make_head()
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(
                similarity_threshold=0.9, max_entries=32
            )
        )
        hidden = torch.randn(2, 8)
        arrays = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                [-1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        lengths = torch.tensor([3.0, 2.0])
        result = head.consult_library(
            hidden,
            arrays,
            library,
            lengths=lengths,
            temperature=1.0,
            auto_cache=False,
        )
        self.assertIsInstance(result, ArrayThoughtLibraryResult)
        self.assertEqual(result.library_hits, [False, False])
        self.assertEqual(len(result.programs), 2)
        self.assertEqual(result.predicted_output.shape, (2,))

    def test_converged_program_is_cached_and_hits_on_second_visit(self):
        torch.manual_seed(0)
        head = self._make_head()
        hidden, arrays, lengths, _, labels = self._train(head)

        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(
                similarity_threshold=0.85, max_entries=32
            )
        )

        sum_mask = [index for index, label in enumerate(labels) if label == "sum"]
        self.assertTrue(sum_mask)
        sum_hidden = hidden[sum_mask]
        sum_arrays = arrays[sum_mask]
        sum_lengths = lengths[sum_mask]

        first = head.consult_library(
            sum_hidden,
            sum_arrays,
            library,
            lengths=sum_lengths,
            temperature=0.1,
            auto_cache=True,
            convergence_gap_threshold=0.5,
            task_name="sum",
        )
        self.assertTrue(any(first.newly_cached))
        self.assertGreaterEqual(len(library), 1)

        # Second visit on the same hidden state should hit the library for at
        # least one sample (the most-converged one).
        second = head.consult_library(
            sum_hidden,
            sum_arrays,
            library,
            lengths=sum_lengths,
            temperature=0.1,
            auto_cache=False,
        )
        self.assertTrue(any(second.library_hits))

    def test_all_hits_short_circuits_without_soft_forward(self):
        head = self._make_head()
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.9)
        )
        hidden = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        arrays = torch.tensor([[2.0, 3.0, 5.0, 0.0, 0.0, 0.0]])
        lengths = torch.tensor([3.0])
        cached = DiscreteArrayProgram(0, 0, 0, 0, 0.0)
        library.record(hidden[0], cached, task_name="sum")

        result = head.consult_library(
            hidden,
            arrays,
            library,
            lengths=lengths,
            temperature=0.1,
            auto_cache=False,
        )
        self.assertEqual(result.library_hits, [True])
        self.assertEqual(result.newly_cached, [False])
        self.assertTrue(
            torch.allclose(
                result.predicted_output,
                torch.tensor([10.0]),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.equal(result.next_hidden_state, hidden),
        )

    def test_non_converged_sample_is_not_cached(self):
        # Extremely small convergence_gap_threshold → nothing should ever
        # qualify unless the head has been trained to absolute convergence.
        # A freshly-initialized head is guaranteed to fail this bar.
        torch.manual_seed(0)
        head = self._make_head()
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85, max_entries=16)
        )
        hidden = torch.randn(3, 8)
        arrays = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 2.0, 0.0, 0.0],
            ]
        )
        lengths = torch.tensor([3.0, 3.0, 4.0])
        result = head.consult_library(
            hidden,
            arrays,
            library,
            lengths=lengths,
            temperature=1.0,
            auto_cache=True,
            convergence_gap_threshold=1e-8,
        )
        self.assertEqual(result.newly_cached, [False, False, False])
        self.assertEqual(len(library), 0)
        # Every convergence gap should be strictly positive for a fresh head.
        self.assertTrue(all(gap > 0.0 for gap in result.convergence_gaps))


if __name__ == "__main__":
    unittest.main()
