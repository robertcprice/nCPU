"""Differentiable Mog compiler.

Compiles I/O examples into verified Mog source code using gradient descent
through a differentiable program representation. Adds grammar-aware constraints
and beam discretization to close the soft-to-hard gap.

Architecture:
    I/O Examples
      -> Structure detection (which program shape to try)
      -> Constrained soft training (gradient descent with grammar penalties)
      -> Beam discretization (top-k search over discrete choices)
      -> Concrete verification (parse + execute via Mog interpreter)

This is the "soft AST" layer that the differentiable Mog executor needs:
instead of just optimizing execution loss, the compiler injects knowledge
about what valid Mog programs look like, producing compilable output.

Reuses:
- SoftMogProgram from mog_program_search.py (the differentiable IR)
- grammar_penalty from mog_grammar.py (compilation-aware loss)
- beam_discretize from mog_beam_search.py (tighter discretization)
- StructureSelector from mog_meta_selector.py (structure detection)
- _mine_constants from mog_two_phase.py (data-aware initialization)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from egdc.mog.solvers.grammar import grammar_penalty, validate_discrete
from egdc.mog.solvers.beam_search import beam_discretize
from egdc.mog.solvers.program_search import (
    SoftMogProgram, SoftBranchingProgram,
    _eval_code_on_examples, _discrete_refinement,
    _branching_refinement, _two_branch_refinement,
    _loop_accum_refinement, _gcd_loop_refinement,
    STMT_TYPES, OPS, CMP_OPS,
)
from egdc.mog.routing.meta_selector import StructureSelector
from egdc.mog.solvers.two_phase import _mine_constants


@dataclass
class DiffCompilerResult:
    success: bool
    code: str
    soft_loss: float
    discrete_loss: float
    structure: str
    verified: bool
    steps: int
    metadata: dict[str, Any] | None = None


class MogDiffCompiler:
    """Differentiable compiler: I/O examples -> verified Mog source code.

    The compiler tries multiple program structures in order:
    1. The structure predicted by StructureSelector
    2. Fallback structures if the prediction fails
    For each structure, it runs constrained gradient descent with grammar
    penalties, then beam-search discretizes the result.
    """

    def __init__(
        self,
        max_steps: int = 500,
        lr: float = 0.05,
        gram_weight: float = 0.1,
        ent_weight: float = 0.01,
        beam_width: int = 8,
    ):
        self.max_steps = max_steps
        self.lr = lr
        self.gram_weight = gram_weight
        self.ent_weight = ent_weight
        self.beam_width = beam_width
        self.selector = StructureSelector()

    def compile(
        self,
        arg_names: Sequence[str],
        examples: Sequence[tuple[tuple[float, ...], float]],
        function_name: str = "program",
        num_slots: int = 8,
        num_restarts: int = 3,
    ) -> DiffCompilerResult:
        """Compile I/O examples into Mog source code.

        Tries fast combinatorial searches first, then falls back to
        grammar-constrained gradient descent for harder programs.
        """
        if not examples:
            return DiffCompilerResult(False, "", float("inf"), float("inf"), "none", False, 0)

        # Phase 1: Fast combinatorial searches (instant to ~2s)
        for search_fn, method_name in self._fast_searches(arg_names, examples, function_name):
            code, loss = search_fn()
            if loss < 1e-6:
                interp_loss = _eval_code_on_examples(code, arg_names, examples)
                if interp_loss < 1e-6:
                    return DiffCompilerResult(
                        success=True, code=code, soft_loss=0.0,
                        discrete_loss=0.0, structure=method_name,
                        verified=True, steps=0,
                    )

        # Phase 2: Grammar-constrained gradient descent (for harder programs)
        predicted = self.selector.predict_structure(arg_names, examples)
        structures = [predicted]
        for fb in ["arithmetic", "branch", "loop", "multi_branch", "general"]:
            if fb not in structures:
                structures.append(fb)

        best_result = None
        for structure in structures:
            for restart in range(num_restarts):
                result = self._try_structure(
                    structure, arg_names, examples, function_name,
                    num_slots, seed=restart * 42,
                )
                if result.success:
                    return result
                if best_result is None or result.discrete_loss < best_result.discrete_loss:
                    best_result = result

        return best_result or DiffCompilerResult(
            False, "", float("inf"), float("inf"), "failed", False, 0,
        )

    def _fast_searches(self, arg_names, examples, fn_name):
        """Yield (search_fn, name) pairs for fast combinatorial searches."""
        # Arithmetic: instant
        yield (lambda: self._fast_arithmetic(arg_names, examples, fn_name), "arithmetic")

        # GCD loop: instant
        yield (lambda: _gcd_loop_refinement(list(arg_names), examples, fn_name), "gcd_loop")

        # Single-branch differentiable: ~2s
        yield (lambda: _branching_refinement(
            SoftBranchingProgram(num_args=len(arg_names)),
            list(arg_names), examples, fn_name,
        ), "single_branch")

        # Loop accumulator: instant
        yield (lambda: _loop_accum_refinement(list(arg_names), examples, fn_name), "loop_accum")

        # Two-branch: ~60s for 1-arg
        if len(arg_names) <= 1:
            yield (lambda: _two_branch_refinement(list(arg_names), examples, fn_name), "two_branch")

    def _try_structure(
        self,
        structure: str,
        arg_names: Sequence[str],
        examples: Sequence[tuple[tuple[float, ...], float]],
        function_name: str,
        num_slots: int,
        seed: int,
    ) -> DiffCompilerResult:
        """Try to compile with a specific program structure."""

        if structure == "branch":
            return self._try_branching(arg_names, examples, function_name, seed)
        if structure == "multi_branch":
            return self._try_multi_branch(arg_names, examples, function_name, seed)
        if structure == "loop":
            return self._try_loop(arg_names, examples, function_name, seed)

        # Default: general SoftMogProgram with grammar constraints
        return self._try_general(arg_names, examples, function_name, num_slots, seed)

    def _fast_arithmetic(self, arg_names, examples, fn_name):
        """Fast arithmetic search: return src1 OP src2."""
        CONSTS = [0, 1, -1, 2, 100]
        from egdc.mog.solvers.program_search import _py_eval_expr
        names = list(arg_names) + [str(c) for c in CONSTS]
        params = ", ".join(f"{a}: i64" for a in arg_names)
        best_loss, best_code = float("inf"), ""
        for s1 in names:
            for s2 in names:
                for op in ["+", "-", "*", "/", "%"]:
                    loss = 0.0
                    for args, target in examples:
                        env = {n: float(v) for n, v in zip(arg_names, args)}
                        try:
                            v1 = _py_eval_expr(s1, env)
                            v2 = _py_eval_expr(s2, env)
                            if op == "+": pred = v1 + v2
                            elif op == "-": pred = v1 - v2
                            elif op == "*": pred = v1 * v2
                            elif op == "/": pred = v1 / v2 if v2 != 0 else 9999
                            elif op == "%": pred = v1 % v2 if v2 != 0 else 9999
                            else: pred = 0
                            loss += (pred - target) ** 2
                        except Exception:
                            loss += 1e8
                    loss /= max(len(examples), 1)
                    if loss < best_loss:
                        best_loss = loss
                        best_code = f"fn {fn_name}({params}) -> i64 {{\n    return {s1} {op} {s2};\n}}\n"
                        if loss < 1e-6:
                            return best_code, best_loss
        return best_code, best_loss

    def _try_general(
        self,
        arg_names: Sequence[str],
        examples: Sequence[tuple[tuple[float, ...], float]],
        function_name: str,
        num_slots: int,
        seed: int,
    ) -> DiffCompilerResult:
        """Train a SoftMogProgram with grammar-constrained gradient descent."""

        torch.manual_seed(seed)
        num_args = len(arg_names)
        prog = SoftMogProgram(num_args=num_args, num_slots=num_slots)

        # Initialize constants from examples
        consts = _mine_constants(list(arg_names), examples)
        if consts:
            with torch.no_grad():
                for s in range(min(num_slots, len(consts))):
                    prog.const_values[s] = float(consts[s % len(consts)])

        optimizer = torch.optim.Adam(prog.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_steps, eta_min=self.lr * 0.1
        )

        best_soft_loss = float("inf")
        best_code = ""
        best_discrete_loss = float("inf")

        for step in range(self.max_steps):
            # Temperature annealing: 2.0 -> 0.1
            progress = step / max(self.max_steps - 1, 1)
            temperature = 2.0 * (1.0 - progress) + 0.1 * progress

            total_loss = torch.tensor(0.0)
            for args, target in examples:
                x = torch.tensor(args, dtype=torch.float32)
                y = torch.tensor(float(target), dtype=torch.float32)
                pred = prog(x, temperature=temperature)
                total_loss = total_loss + (pred - y) ** 2

            exec_loss = total_loss / len(examples)

            # Grammar constraint loss
            gram_loss = grammar_penalty(
                prog.stmt_logits, prog.op_logits, prog.src2_logits,
                num_slots, prog.num_sources,
            )

            # Entropy regularization
            entropy = torch.tensor(0.0)
            for s in range(num_slots):
                p = F.softmax(prog.stmt_logits[s], dim=0)
                entropy = entropy - (p * torch.log(p + 1e-8)).sum()

            loss = exec_loss + self.gram_weight * gram_loss + self.ent_weight * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prog.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            soft_loss_val = float(exec_loss.item())
            if soft_loss_val < best_soft_loss:
                best_soft_loss = soft_loss_val

                # Periodic discretization check (every 50 steps or at end)
                if step % 50 == 49 or step == self.max_steps - 1:
                    code, d_loss = beam_discretize(
                        prog, arg_names, examples, function_name,
                        beam_width=self.beam_width,
                    )
                    if d_loss < best_discrete_loss:
                        best_code = code
                        best_discrete_loss = d_loss
                    if d_loss < 1e-6:
                        break

        # Final beam discretization
        if best_discrete_loss > 1e-6:
            code, d_loss = beam_discretize(
                prog, arg_names, examples, function_name,
                beam_width=self.beam_width,
            )
            if d_loss < best_discrete_loss:
                best_code = code
                best_discrete_loss = d_loss

        # Verify via interpreter
        verified = False
        if best_discrete_loss < 1e-6:
            interp_loss = _eval_code_on_examples(best_code, arg_names, examples)
            verified = interp_loss < 1e-6

        return DiffCompilerResult(
            success=verified,
            code=best_code,
            soft_loss=best_soft_loss,
            discrete_loss=best_discrete_loss,
            structure="general",
            verified=verified,
            steps=self.max_steps,
            metadata={"seed": seed, "temperature_final": temperature},
        )

    def _try_branching(
        self,
        arg_names: Sequence[str],
        examples: Sequence[tuple[tuple[float, ...], float]],
        function_name: str,
        seed: int,
    ) -> DiffCompilerResult:
        """Try a branching program (if-then-else) via SoftBranchingProgram."""
        torch.manual_seed(seed)
        num_args = len(arg_names)
        prog = SoftBranchingProgram(num_args=num_args)

        optimizer = torch.optim.Adam(prog.parameters(), lr=self.lr)

        best_soft_loss = float("inf")
        best_code = ""
        best_discrete_loss = float("inf")

        for step in range(min(self.max_steps, 300)):
            progress = step / 299
            temperature = 2.0 * (1.0 - progress) + 0.1 * progress

            total_loss = torch.tensor(0.0)
            for args, target in examples:
                x = torch.tensor(args, dtype=torch.float32)
                y = torch.tensor(float(target), dtype=torch.float32)
                pred = prog(x, temperature=temperature)
                total_loss = total_loss + (pred - y) ** 2

            loss = total_loss / len(examples)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prog.parameters(), max_norm=5.0)
            optimizer.step()

            soft_loss_val = float(loss.item())
            if soft_loss_val < best_soft_loss:
                best_soft_loss = soft_loss_val

            if step % 50 == 49 or step == 299:
                code = prog.discretize(list(arg_names))
                code = code.replace("fn program(", f"fn {function_name}(")
                d_loss = _eval_code_on_examples(code, arg_names, examples)
                if d_loss < best_discrete_loss:
                    best_code = code
                    best_discrete_loss = d_loss
                if d_loss < 1e-6:
                    break

        # Discrete refinement as fallback
        if best_discrete_loss > 1e-6:
            from egdc.mog.solvers.program_search import _discrete_refinement, _eval_discrete
            base_code, base_loss = _eval_discrete(prog, arg_names, examples)
            base_code = base_code.replace("fn program(", f"fn {function_name}(")
            refined_code, refined_loss = _discrete_refinement(
                prog, arg_names, examples, base_code, base_loss, function_name
            )
            if refined_loss < best_discrete_loss:
                best_code = refined_code
                best_discrete_loss = refined_loss

        verified = False
        if best_discrete_loss < 1e-6:
            interp_loss = _eval_code_on_examples(best_code, arg_names, examples)
            verified = interp_loss < 1e-6

        return DiffCompilerResult(
            success=verified,
            code=best_code,
            soft_loss=best_soft_loss,
            discrete_loss=best_discrete_loss,
            structure="branch",
            verified=verified,
            steps=300,
        )

    def _try_multi_branch(
        self,
        arg_names: Sequence[str],
        examples: Sequence[tuple[tuple[float, ...], float]],
        function_name: str,
        seed: int,
    ) -> DiffCompilerResult:
        """Try multi-branch programs via combinatorial search (fast)."""
        from egdc.mog.solvers.program_search import _two_branch_refinement

        code, loss = _two_branch_refinement(list(arg_names), examples, function_name)
        verified = False
        if loss < 1e-6:
            interp_loss = _eval_code_on_examples(code, arg_names, examples)
            verified = interp_loss < 1e-6

        return DiffCompilerResult(
            success=verified,
            code=code,
            soft_loss=loss,
            discrete_loss=loss,
            structure="multi_branch",
            verified=verified,
            steps=0,
        )

    def _try_loop(
        self,
        arg_names: Sequence[str],
        examples: Sequence[tuple[tuple[float, ...], float]],
        function_name: str,
        seed: int,
    ) -> DiffCompilerResult:
        """Try loop accumulator programs via combinatorial search (fast)."""
        from egdc.mog.solvers.program_search import _loop_accum_refinement

        code, loss = _loop_accum_refinement(list(arg_names), examples, function_name)
        verified = False
        if loss < 1e-6:
            interp_loss = _eval_code_on_examples(code, arg_names, examples)
            verified = interp_loss < 1e-6

        return DiffCompilerResult(
            success=verified,
            code=code,
            soft_loss=loss,
            discrete_loss=loss,
            structure="loop",
            verified=verified,
            steps=0,
        )
