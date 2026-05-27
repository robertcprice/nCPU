"""Distillation utility tests (NV3-next)."""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.library_distillation import (
    distill_library,
    fit_projection,
)


def _seed_teacher() -> ArrayProgramLibrary:
    lib = ArrayProgramLibrary(
        ArrayProgramLibraryConfig(similarity_threshold=0.85)
    )
    lib.record(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        DiscreteArrayProgram(0, 0, 0, 0, 0.0),
        task_name="sum",
    )
    lib.record(
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        DiscreteArrayProgram(2, 0, 2, 0, 0.0),
        task_name="max",
    )
    lib.record(
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        DiscreteArrayProgram(0, 4, 0, 0, 0.0),
        task_name="count",
    )
    return lib


class TestFitProjection(unittest.TestCase):
    def test_identity_mapping_trivial(self):
        torch.manual_seed(0)
        teacher = torch.randn(20, 4)
        student = teacher.clone()
        proj, residual = fit_projection(teacher, student)
        self.assertEqual(proj.shape, (4, 4))
        self.assertLess(residual, 1e-2)

    def test_dim_change_mapping(self):
        torch.manual_seed(0)
        teacher = torch.randn(40, 6)
        # Student = known linear map of teacher + small noise.
        W_true = torch.randn(3, 6)
        student = teacher @ W_true.t() + 0.01 * torch.randn(40, 3)
        proj, residual = fit_projection(teacher, student)
        self.assertEqual(proj.shape, (3, 6))
        # Predicted projection should be close to W_true
        self.assertLess((proj - W_true).norm().item(), 2.0)
        # Residual should be small (noise only).
        self.assertLess(residual, 5.0)

    def test_invalid_shapes_raise(self):
        with self.assertRaises(ValueError):
            fit_projection(torch.zeros(3), torch.zeros(3, 4))
        with self.assertRaises(ValueError):
            fit_projection(torch.zeros(3, 4), torch.zeros(5, 3))
        with self.assertRaises(ValueError):
            fit_projection(torch.zeros(0, 4), torch.zeros(0, 3))


class TestDistillLibrary(unittest.TestCase):
    def test_identity_distillation_preserves_library(self):
        torch.manual_seed(0)
        teacher = _seed_teacher()
        teacher_samples = torch.randn(30, 4)
        student_samples = teacher_samples.clone()
        student_lib, report = distill_library(
            teacher,
            teacher_hidden_samples=teacher_samples,
            student_hidden_samples=student_samples,
        )
        self.assertEqual(len(student_lib), len(teacher))
        self.assertEqual(report.teacher_dim, 4)
        self.assertEqual(report.student_dim, 4)
        # Programs are preserved unchanged.
        for t_entry, s_entry in zip(teacher.entries, student_lib.entries):
            self.assertEqual(t_entry.program.key(), s_entry.program.key())

    def test_distillation_to_smaller_dim(self):
        torch.manual_seed(0)
        teacher = _seed_teacher()
        teacher_samples = torch.randn(40, 4)
        # Student is a linear projection of teacher into 3-D
        W = torch.randn(3, 4)
        student_samples = teacher_samples @ W.t()
        student_lib, report = distill_library(
            teacher,
            teacher_hidden_samples=teacher_samples,
            student_hidden_samples=student_samples,
        )
        self.assertEqual(report.student_dim, 3)
        # All 3 entries transferred.
        self.assertEqual(len(student_lib), 3)
        # Each student entry has a 3-D signature.
        for entry in student_lib.entries:
            self.assertEqual(len(entry.signature), 3)

    def test_task_names_get_suffix(self):
        torch.manual_seed(0)
        teacher = _seed_teacher()
        t_samples = torch.randn(20, 4)
        s_samples = torch.randn(20, 4)
        student_lib, _ = distill_library(
            teacher,
            teacher_hidden_samples=t_samples,
            student_hidden_samples=s_samples,
            task_name_suffix="_student",
        )
        tasks = {e.task_name for e in student_lib.entries}
        self.assertEqual(tasks, {"sum_student", "max_student", "count_student"})

    def test_report_fields(self):
        torch.manual_seed(0)
        teacher = _seed_teacher()
        t_samples = torch.randn(25, 4)
        s_samples = torch.randn(25, 4)
        _, report = distill_library(
            teacher,
            teacher_hidden_samples=t_samples,
            student_hidden_samples=s_samples,
        )
        self.assertEqual(report.num_samples, 25)
        self.assertEqual(report.teacher_dim, 4)
        self.assertEqual(report.student_dim, 4)
        self.assertEqual(report.transferred_entries, 3)
        self.assertGreater(report.projection_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
