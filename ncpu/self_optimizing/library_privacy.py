"""Differential privacy for library signatures (NV5b).

Before publishing a library, a curator may want to blur the signature
vectors so that reverse-engineering the training data from the library is
harder. This module adds calibrated Gaussian noise to every signature
such that the resulting library satisfies an approximate `(ε, δ)`
differential-privacy guarantee under the Gaussian mechanism.

The guarantee, precisely:

* We treat each signature vector as the L2-sensitive output of a
  (signature-producing) query with L2 sensitivity 1 — signatures are
  unit-norm by construction, so swapping one entry for another changes
  the `entries[i].signature` vector by at most L2 norm 2. To be
  conservative we use sensitivity 2.
* We inject zero-mean Gaussian noise with standard deviation
  `sigma = sqrt(2 * ln(1.25 / delta)) * sensitivity / epsilon` to each
  coordinate (Dwork & Roth 2014, Theorem 3.22).
* Signatures are re-normalized after perturbation so they remain
  unit-norm (lookups still work).

This is a post-hoc sanitizer — it does NOT replace training with DP-SGD
or anything like that. It's the simpler guarantee: the PUBLISHED library
preserves (ε, δ)-DP over the choice of which signatures to include.
Programs themselves are categorical and left unperturbed — they are not
considered sensitive in this model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    LibraryEntry,
)


@dataclass
class DPCertificate:
    """Provable privacy budget for a perturbed library."""

    epsilon: float
    delta: float
    sigma: float
    sensitivity: float
    entries_perturbed: int
    signature_dim: int


def dp_perturb_library(
    library: ArrayProgramLibrary,
    *,
    epsilon: float,
    delta: float,
    sensitivity: float = 2.0,
    seed: Optional[int] = None,
    target_config: Optional[ArrayProgramLibraryConfig] = None,
) -> tuple[ArrayProgramLibrary, DPCertificate]:
    """Return a perturbed copy of `library` satisfying (ε, δ)-DP on signatures.

    Programs themselves are preserved unchanged — the library's *what* (the
    skills) stays identical; only the *where* (the hidden-state signatures
    that trigger them) is blurred.

    Raises:
        ValueError: if `epsilon <= 0` or `delta <= 0 or delta >= 1`.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    sigma = math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity / epsilon

    rng = random.Random(seed)
    perturbed_config = target_config or ArrayProgramLibraryConfig(
        similarity_threshold=library.config.similarity_threshold,
        max_entries=library.config.max_entries,
        normalize_epsilon=library.config.normalize_epsilon,
    )
    perturbed = ArrayProgramLibrary(perturbed_config)

    signature_dim = 0
    for entry in library.entries:
        if not entry.signature:
            continue
        signature_dim = len(entry.signature)
        noisy = [
            float(v) + rng.gauss(0.0, sigma) for v in entry.signature
        ]
        norm = math.sqrt(sum(x * x for x in noisy))
        if norm < library.config.normalize_epsilon:
            # Degenerate: skip rather than record a zero-norm signature.
            continue
        unit = [x / norm for x in noisy]
        perturbed._entries.append(
            LibraryEntry(
                signature=unit,
                program=entry.program,
                hit_count=0,
                task_name=entry.task_name,
                cached_at_step=entry.cached_at_step,
                convergence_gap=entry.convergence_gap,
            )
        )

    certificate = DPCertificate(
        epsilon=float(epsilon),
        delta=float(delta),
        sigma=float(sigma),
        sensitivity=float(sensitivity),
        entries_perturbed=len(perturbed._entries),
        signature_dim=int(signature_dim),
    )
    return perturbed, certificate


__all__ = ["DPCertificate", "dp_perturb_library"]
