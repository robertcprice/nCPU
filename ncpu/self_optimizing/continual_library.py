"""Continual library growth — grow the library from verified generations.

The idea: every problem the model solves successfully at inference time
adds a new (hidden-state signature → discovered program) pair to the
library. The library monotonically grows across sessions, so the model
gets better at problem shapes it has seen before without retraining.

Two hooks:

* `record_successful_generation(library, head, hidden_states, array_inputs,
  lengths, ground_truth)` — after an LLM generation is verified, call this
  with the final-layer hidden state and the scalar the array-thought head
  should produce for this problem. The head crystallizes a program that
  matches, and caches it into the library keyed by the hidden signature.

* `attach_verifier_hook(library, verifier_fn)` — wire a generation
  pipeline to verify + record automatically.

This is the simplest possible "never fails harder than baseline" safety
rail when combined with confidence_gate=True: on any problem where the
library doesn't yet have a matching skill, the wrapper outputs zero
(baseline behavior). On any problem the model has successfully solved
before, the library fires with a cached discrete program that was
crystallized from the specific hidden-state pattern of that problem
class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from ncpu.self_optimizing.array_executable_thought_head import (
    ArrayExecutableThoughtHead,
)
from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    DiscreteArrayProgram,
)


@dataclass
class ContinualGrowthReport:
    """What happened when we tried to grow the library."""

    grew: bool
    before_entries: int
    after_entries: int
    new_entry_task: Optional[str]
    reason: str


def record_successful_generation(
    library: ArrayProgramLibrary,
    head: ArrayExecutableThoughtHead,
    *,
    hidden_state: torch.Tensor,
    array_inputs: torch.Tensor,
    lengths: torch.Tensor,
    ground_truth_scalar: float,
    task_name: Optional[str] = None,
    convergence_gap_threshold: float = 0.5,
) -> ContinualGrowthReport:
    """Grow the library with a new entry crystallized from a verified solve.

    Steps:
    1. Run the array-thought head at low temperature → get the discrete
       program it would extract from `hidden_state`.
    2. Check whether that discrete program, executed on `array_inputs`,
       produces `ground_truth_scalar` within tolerance. If so, it's a
       genuinely useful skill — add it to the library.
    3. If the program already exists (same key + near-identical signature),
       just refresh the hit count (it validates an existing skill).
    """
    before = len(library)
    with torch.no_grad():
        # Ensure rank-2 hidden state.
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.unsqueeze(0)
        if array_inputs.ndim == 1:
            array_inputs = array_inputs.unsqueeze(0)
        if lengths.ndim == 0:
            lengths = lengths.unsqueeze(0)

        result = head(
            hidden_state,
            array_inputs,
            lengths=lengths,
            temperature=0.05,
        )
        # Extract the argmax discrete program.
        init_probs = result.init_probs
        if init_probs.ndim == 1:
            init_probs = init_probs.unsqueeze(0)
            tr = result.transform_probs.unsqueeze(0)
            rd = result.reduce_probs.unsqueeze(0)
            ps = result.post_scale_probs.unsqueeze(0)
            po = result.post_offsets.unsqueeze(0)
        else:
            tr = result.transform_probs
            rd = result.reduce_probs
            ps = result.post_scale_probs
            po = result.post_offsets
        distributions = {
            "init": init_probs,
            "transform": tr,
            "reduce": rd,
            "post_scale": ps,
            "post_offset": po,
        }
        program = DiscreteArrayProgram.from_soft_distributions(distributions, 0)

        # Verify the discrete program matches the ground-truth scalar.
        predicted = program.execute(array_inputs, lengths).item()
        error = abs(predicted - ground_truth_scalar)
        if error > convergence_gap_threshold:
            return ContinualGrowthReport(
                grew=False,
                before_entries=before,
                after_entries=before,
                new_entry_task=task_name,
                reason=(
                    f"discrete program output {predicted:.4f} differs from "
                    f"ground truth {ground_truth_scalar:.4f} by {error:.4f} "
                    f"> threshold {convergence_gap_threshold}"
                ),
            )

        entry = library.record(
            hidden_state.squeeze(0),
            program,
            task_name=task_name,
            convergence_gap=error,
        )

    after = len(library)
    return ContinualGrowthReport(
        grew=after > before,
        before_entries=before,
        after_entries=after,
        new_entry_task=entry.task_name,
        reason="added new entry" if after > before else "refreshed existing entry",
    )


def attach_verifier_hook(
    library: ArrayProgramLibrary,
    *,
    verifier_fn: Callable[[Any], tuple[bool, Optional[float]]],
) -> Callable:
    """Wire a continual-growth hook to a user-supplied verifier.

    `verifier_fn(generation_artifact) -> (passed, ground_truth_scalar)`:
    returns whether the generation passed verification, and if so, the
    scalar the library entry should produce. For coding tasks this might
    be the number of test cases passed, or a hash of the passing solution.

    Returns a callable `(head, hidden_state, array_inputs, lengths,
    generation_artifact, task_name) -> ContinualGrowthReport`. Call it
    after every generation to optionally grow the library.
    """
    def hook(
        head,
        hidden_state,
        array_inputs,
        lengths,
        generation_artifact,
        task_name=None,
    ):
        passed, ground_truth = verifier_fn(generation_artifact)
        if not passed or ground_truth is None:
            return ContinualGrowthReport(
                grew=False,
                before_entries=len(library),
                after_entries=len(library),
                new_entry_task=task_name,
                reason="verifier rejected generation",
            )
        return record_successful_generation(
            library,
            head,
            hidden_state=hidden_state,
            array_inputs=array_inputs,
            lengths=lengths,
            ground_truth_scalar=ground_truth,
            task_name=task_name,
        )
    return hook


__all__ = [
    "ContinualGrowthReport",
    "record_successful_generation",
    "attach_verifier_hook",
]
