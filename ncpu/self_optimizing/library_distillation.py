"""Teacher→student library distillation (NV3-next).

`ArrayProgramLibrary.transfer_library` (in `array_program_library.py`)
already reprojects signatures through a caller-supplied projection matrix.
This module adds the fitted-projection path: given a sample of hidden
states from a student model, find the best linear map from teacher's
hidden space to student's hidden space, then apply `transfer_library` with
that map.

The projection is fit via closed-form least-squares on the paired hidden
samples. Callers supply:

* `teacher_library` — the source `ArrayProgramLibrary`.
* `teacher_hidden_samples` — rank-2 `[N, T]` tensor of teacher hiddens.
* `student_hidden_samples` — rank-2 `[N, S]` tensor of student hiddens
  (aligned: row i of both tensors corresponds to the same input token).

The output is a new `ArrayProgramLibrary` whose signatures live in the
student's hidden space. Programs themselves carry over unchanged.

Typical usage::

    from ncpu.self_optimizing.library_distillation import distill_library
    student_library = distill_library(
        teacher_library=teacher_library,
        teacher_hidden_samples=teacher_hidden_samples,
        student_hidden_samples=student_hidden_samples,
    )
    student_library.save("./student.json")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    transfer_library,
)


@dataclass
class DistillationReport:
    """Diagnostic summary of a distillation run."""

    teacher_dim: int
    student_dim: int
    num_samples: int
    projection_residual: float
    projection_norm: float
    transferred_entries: int


def fit_projection(
    teacher_samples: torch.Tensor,
    student_samples: torch.Tensor,
    *,
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, float]:
    """Fit a least-squares linear map from teacher hidden → student hidden.

    Solves `min ||W @ teacher^T - student^T||_F^2 + ridge * ||W||_F^2`
    where W has shape `(student_dim, teacher_dim)`.

    Returns `(W, residual)`.
    """
    if teacher_samples.ndim != 2 or student_samples.ndim != 2:
        raise ValueError("samples must be rank-2 [N, D]")
    if teacher_samples.shape[0] != student_samples.shape[0]:
        raise ValueError(
            "teacher and student must have the same number of samples"
        )
    if teacher_samples.shape[0] == 0:
        raise ValueError("need at least one sample to fit a projection")

    teacher = teacher_samples.detach().to(torch.float32).cpu()
    student = student_samples.detach().to(torch.float32).cpu()

    # Solve for W: W @ T^T ≈ S^T  →  W = S^T @ T @ (T^T @ T + λI)^-1
    t_dim = teacher.shape[1]
    gram = teacher.t() @ teacher + ridge * torch.eye(t_dim, dtype=torch.float32)
    pinv = torch.linalg.inv(gram)
    projection = student.t() @ teacher @ pinv  # (S, T)
    predicted = projection @ teacher.t()  # (S, N)
    residual = float((predicted - student.t()).norm().item())
    return projection, residual


def distill_library(
    teacher_library: ArrayProgramLibrary,
    *,
    teacher_hidden_samples: torch.Tensor,
    student_hidden_samples: torch.Tensor,
    target_config: Optional[ArrayProgramLibraryConfig] = None,
    ridge: float = 1e-4,
    task_name_suffix: str = "_distilled",
) -> tuple[ArrayProgramLibrary, DistillationReport]:
    """Distill a teacher's skill library into the student's hidden space."""
    projection, residual = fit_projection(
        teacher_hidden_samples,
        student_hidden_samples,
        ridge=ridge,
    )
    transferred = transfer_library(
        teacher_library,
        projection=projection,
        target_config=target_config,
        task_name_suffix=task_name_suffix,
    )
    report = DistillationReport(
        teacher_dim=int(teacher_hidden_samples.shape[1]),
        student_dim=int(student_hidden_samples.shape[1]),
        num_samples=int(teacher_hidden_samples.shape[0]),
        projection_residual=float(residual),
        projection_norm=float(projection.norm().item()),
        transferred_entries=len(transferred),
    )
    return transferred, report


__all__ = [
    "DistillationReport",
    "distill_library",
    "fit_projection",
]
