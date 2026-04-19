"""Continual library growth tests."""

from __future__ import annotations

import unittest

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
)
from ncpu.self_optimizing.continual_library import (
    ContinualGrowthReport,
    attach_verifier_hook,
    record_successful_generation,
)


class TestContinualGrowth(unittest.TestCase):
    def _trained_head(self) -> ArrayExecutableThoughtHead:
        torch.manual_seed(0)
        head = ArrayExecutableThoughtHead(
            ArrayExecutableThoughtHeadConfig(
                hidden_dim=8,
                array_max_len=6,
                trace_projection_dim=8,
                trace_hidden_dim=16,
                state_patch_dim=8,
            )
        )
        hidden, arrays, lengths, targets, _ = build_array_thought_smoke_batch(
            hidden_dim=8, array_max_len=6, samples_per_op=6, seed=0,
        )
        run_array_thought_smoke_train(
            head, hidden_state=hidden, array_inputs=arrays,
            lengths=lengths, targets=targets, steps=300, learning_rate=5e-2,
        )
        return head

    def test_records_matching_generation(self):
        head = self._trained_head()
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85)
        )
        # Use a sum-prototype hidden state (first slot) and sum-like input.
        hidden = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        array = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        length = torch.tensor([3.0])
        report = record_successful_generation(
            library, head,
            hidden_state=hidden, array_inputs=array, lengths=length,
            ground_truth_scalar=6.0,
            task_name="sum",
            convergence_gap_threshold=2.0,
        )
        self.assertIsInstance(report, ContinualGrowthReport)
        self.assertTrue(report.grew or report.after_entries > 0)
        self.assertGreaterEqual(len(library), 1)

    def test_rejects_nonmatching_generation(self):
        head = self._trained_head()
        library = ArrayProgramLibrary()
        hidden = torch.randn(8)
        array = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        length = torch.tensor([3.0])
        # Ground truth deliberately far from any sensible output.
        report = record_successful_generation(
            library, head,
            hidden_state=hidden, array_inputs=array, lengths=length,
            ground_truth_scalar=1e9,  # impossible target
            task_name="bogus",
            convergence_gap_threshold=0.1,
        )
        self.assertFalse(report.grew)
        self.assertEqual(len(library), 0)
        self.assertIn("differs", report.reason)


class TestVerifierHook(unittest.TestCase):
    def test_hook_grows_on_pass(self):
        head = self._trained_head()
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85)
        )
        hook = attach_verifier_hook(
            library,
            verifier_fn=lambda artifact: (True, 6.0),
        )
        hidden = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        array = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        length = torch.tensor([3.0])
        report = hook(head, hidden, array, length, "some_artifact", task_name="t")
        self.assertIsInstance(report, ContinualGrowthReport)

    def test_hook_skips_on_fail(self):
        library = ArrayProgramLibrary()
        hook = attach_verifier_hook(
            library,
            verifier_fn=lambda artifact: (False, None),
        )
        dummy_head = None  # never called if verifier rejects
        report = hook(dummy_head, None, None, None, "artifact")
        self.assertFalse(report.grew)
        self.assertEqual(report.reason, "verifier rejected generation")

    def _trained_head(self) -> ArrayExecutableThoughtHead:
        torch.manual_seed(0)
        head = ArrayExecutableThoughtHead(
            ArrayExecutableThoughtHeadConfig(
                hidden_dim=8,
                array_max_len=6,
                trace_projection_dim=8,
                trace_hidden_dim=16,
                state_patch_dim=8,
            )
        )
        hidden, arrays, lengths, targets, _ = build_array_thought_smoke_batch(
            hidden_dim=8, array_max_len=6, samples_per_op=6, seed=0,
        )
        run_array_thought_smoke_train(
            head, hidden_state=hidden, array_inputs=arrays,
            lengths=lengths, targets=targets, steps=300, learning_rate=5e-2,
        )
        return head


if __name__ == "__main__":
    unittest.main()
