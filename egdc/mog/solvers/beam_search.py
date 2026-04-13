"""Top-k beam discretization for soft programs.

Replaces greedy argmax with a beam search that explores the top-k most likely
discrete choices at each decision point, scoring each candidate against the
Mog interpreter. This closes the soft-to-hard gap that causes the discretization
problem: when two choices have similar logits, argmax picks the wrong one.

Strategy:
1. Extract top-k indices for each parameter at each slot
2. Build a beam of k partial programs, expanding one slot at a time
3. Score each complete candidate via the Mog interpreter
4. Return the best-scoring program
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from egdc.mog.solvers.program_search import (
    SoftMogProgram, SoftBranchingProgram,
    _eval_code_on_examples, _eval_discrete,
    _discrete_refinement,
    STMT_TYPES, OPS, CMP_OPS,
)


@dataclass
class BeamResult:
    code: str
    loss: float
    beam_position: int  # which beam candidate found this


class BeamDiscretizer:
    """Beam-search discretization for SoftMogProgram."""

    def __init__(self, beam_width: int = 8, top_k_per_slot: int = 3):
        self.beam_width = beam_width
        self.top_k_per_slot = top_k_per_slot

    def discretize(
        self,
        prog: SoftMogProgram,
        arg_names: Sequence[str],
        examples: Sequence[tuple[tuple[float, ...], float]],
        function_name: str = "program",
    ) -> list[BeamResult]:
        """Discretize a soft program using beam search.

        Returns up to beam_width candidates, sorted by loss (best first).
        """
        # Start with the argmax baseline
        baseline_code, baseline_loss = _eval_discrete(prog, arg_names, examples)
        baseline_code = baseline_code.replace("fn program(", f"fn {function_name}(")
        results = [BeamResult(baseline_code, baseline_loss, 0)]

        if baseline_loss < 1e-6:
            return results

        # Top-k sampling: for key parameters, get the top choices
        state = prog.state_dict()

        # Extract top-k for each slot
        top_k_data = self._extract_top_k(state, prog.num_slots, prog.num_sources)

        # Generate candidates by flipping top-k choices on key parameters
        candidates = self._generate_candidates(
            state, prog, arg_names, examples, function_name, top_k_data
        )
        results.extend(candidates)

        # Sort by loss, deduplicate by code
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x.loss):
            if r.code not in seen:
                seen.add(r.code)
                unique.append(r)

        return unique[:self.beam_width]

    def _extract_top_k(
        self, state: dict, num_slots: int, num_sources: int
    ) -> dict[str, list[tuple[list[int], list[float]]]]:
        """Extract top-k indices and probabilities for each parameter group."""
        top_k = {}
        k = min(self.top_k_per_slot, num_sources)

        for param_name in ["stmt_logits", "src1_logits", "src2_logits", "op_logits", "return_src_logits"]:
            if param_name not in state:
                continue
            tensor = state[param_name]
            slots_data = []
            for slot in range(tensor.shape[0]):
                probs = F.softmax(tensor[slot], dim=0)
                actual_k = min(k, probs.shape[0])
                top_vals, top_idx = torch.topk(probs, actual_k)
                slots_data.append((top_idx.tolist(), top_vals.tolist()))
            top_k[param_name] = slots_data
        return top_k

    def _generate_candidates(
        self,
        state: dict,
        prog: SoftMogProgram,
        arg_names: Sequence[str],
        examples: Sequence[tuple[tuple[float, ...], float]],
        function_name: str,
        top_k_data: dict,
    ) -> list[BeamResult]:
        """Generate candidate programs by trying top-k alternatives."""
        results = []

        # Strategy 1: Try top-k for each parameter at each slot
        for param_name, slots_data in top_k_data.items():
            if param_name not in state:
                continue
            tensor = state[param_name]
            for slot in range(min(len(slots_data), tensor.shape[0])):
                current = int(torch.argmax(tensor[slot]).item())
                top_indices, _ = slots_data[slot]

                for alt in top_indices:
                    if alt == current:
                        continue
                    saved = tensor[slot].clone()
                    tensor[slot] = torch.full_like(tensor[slot], -10.0)
                    tensor[slot][alt] = 10.0
                    prog.load_state_dict(state)

                    code, loss = _eval_discrete(prog, arg_names, examples)
                    code = code.replace("fn program(", f"fn {function_name}(")
                    results.append(BeamResult(code, loss, len(results) + 1))

                    tensor[slot] = saved
                    if loss < 1e-6:
                        prog.load_state_dict(state)
                        return results

        prog.load_state_dict(state)

        # Strategy 2: Pair perturbation on correlated parameters
        if results and min(r.loss for r in results) < 1e-6:
            return results

        pairs = [
            ("src1_logits", "op_logits"),
            ("src1_logits", "src2_logits"),
            ("src2_logits", "op_logits"),
        ]
        for p1, p2 in pairs:
            if p1 not in top_k_data or p2 not in top_k_data:
                continue
            if p1 not in state or p2 not in state:
                continue
            t1, t2 = state[p1], state[p2]
            for slot in range(min(len(top_k_data[p1]), t1.shape[0])):
                cur1 = int(torch.argmax(t1[slot]).item())
                cur2 = int(torch.argmax(t2[slot]).item())
                indices1 = top_k_data[p1][slot][0]
                indices2 = top_k_data[p2][slot][0]

                for a1 in indices1[:2]:
                    if a1 == cur1:
                        continue
                    for a2 in indices2[:2]:
                        if a2 == cur2:
                            continue
                        s1 = t1[slot].clone()
                        s2 = t2[slot].clone()
                        t1[slot] = torch.full_like(t1[slot], -10.0)
                        t1[slot][a1] = 10.0
                        t2[slot] = torch.full_like(t2[slot], -10.0)
                        t2[slot][a2] = 10.0
                        prog.load_state_dict(state)

                        code, loss = _eval_discrete(prog, arg_names, examples)
                        code = code.replace("fn program(", f"fn {function_name}(")
                        results.append(BeamResult(code, loss, len(results) + 1))

                        t1[slot] = s1
                        t2[slot] = s2
                        if loss < 1e-6:
                            prog.load_state_dict(state)
                            return results

        prog.load_state_dict(state)
        return results


def beam_discretize(
    prog: SoftMogProgram,
    arg_names: Sequence[str],
    examples: Sequence[tuple[tuple[float, ...], float]],
    function_name: str = "program",
    beam_width: int = 8,
) -> tuple[str, float]:
    """Convenience function: beam discretize and return best result.

    Falls back to the existing _discrete_refinement if beam doesn't find
    a perfect solution.
    """
    discretizer = BeamDiscretizer(beam_width=beam_width)
    results = discretizer.discretize(prog, arg_names, examples, function_name)

    if results and results[0].loss < 1e-6:
        return results[0].code, results[0].loss

    # Fall back to the more expensive _discrete_refinement for harder cases
    best = results[0] if results else None
    if best is None:
        baseline_code, baseline_loss = _eval_discrete(prog, arg_names, examples)
        baseline_code = baseline_code.replace("fn program(", f"fn {function_name}(")
        best = BeamResult(baseline_code, baseline_loss, 0)

    code, loss = _discrete_refinement(
        prog, arg_names, examples, best.code, best.loss, function_name
    )
    return code, loss
