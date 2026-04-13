"""
unified_synth.py — Unified program synthesis engine.

Three-layer approach:
  1. Differentiable synthesis (Rust mog_synth) — gradient descent discovers programs
  2. LLM synthesis (Claude) — handles arbitrary complexity
  3. Binary I/O extraction — generates training data from any existing code

The engine tries differentiable first (fastest, no API cost), falls back to LLM,
and can accept arbitrary functions/binaries as input via I/O extraction.

Usage:
    from egdc.unified_synth import UnifiedSynthesizer

    synth = UnifiedSynthesizer()

    # From I/O examples
    result = synth.synthesize("add_two", examples=[(3, 5), (0, 2), (-1, 1)])

    # From an existing function (extracts I/O automatically)
    result = synth.from_function(my_func, n_examples=20)

    # From a binary
    result = synth.from_binary("./my_program", arg_types=["int", "int"])
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class SynthResult:
    solved: bool
    code: str | None
    language: str
    method: str  # "differentiable", "register_machine", "llm", "llm_mog"
    time_s: float = 0.0
    attempts: int = 0
    error: str | None = None


class UnifiedSynthesizer:
    """Three-layer program synthesis: differentiable → LLM → verified output."""

    def __init__(
        self,
        mog_synth_binary: str | None = None,
        llm_model: str = "claude-sonnet-4-20250514",
        enable_differentiable: bool = True,
        enable_llm: bool = True,
    ):
        # Find mog_synth binary
        if mog_synth_binary:
            self.mog_binary = mog_synth_binary
        else:
            # Try common locations
            candidates = [
                Path(__file__).parent.parent / "mog_synth" / "target" / "release" / "mog_synth",
                Path(__file__).parent.parent / "mog_synth" / "target" / "debug" / "mog_synth",
            ]
            self.mog_binary = None
            for c in candidates:
                if c.exists():
                    self.mog_binary = str(c)
                    break

        self.llm_model = llm_model
        self.enable_differentiable = enable_differentiable
        self.enable_llm = enable_llm

        # Load meta-learner for program type prediction
        self.meta_learner = None
        try:
            model_path = Path(__file__).parent.parent / "mog_synth" / "models" / "expr_type_classifier.pt"
            if model_path.exists():
                sys.path.insert(0, str(Path(__file__).parent.parent / "mog_synth" / "scripts"))
                from train_expr_metalearner import load_model
                self.meta_learner = load_model(str(model_path))
        except Exception:
            pass  # Meta-learner is optional

    # ── Main entry point ──────────────────────────────────────────────────

    def synthesize(
        self,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        description: str = "",
        signature: str | None = None,
        holdouts: list[tuple[Any, Any]] | None = None,
    ) -> SynthResult:
        """Synthesize a function from I/O examples.

        Tries differentiable synthesis first (fast, free), then LLM (powerful, costs API).

        Args:
            fn_name: function name
            examples: list of (input, expected_output) pairs
            description: plain-English description
            signature: Mog-style type signature (e.g., "fn foo(a: i64, b: i64) -> i64")
            holdouts: extra test cases for validation (not shown to LLM)
        """
        results = []

        # Meta-learner prediction: predict which program type to try first
        predicted_type = None
        if self.meta_learner:
            try:
                n_args = len(examples[0][0]) if isinstance(examples[0][0], (list, tuple)) else 1
                io_pairs = []
                for args, exp in examples[:8]:
                    if isinstance(args, (list, tuple)):
                        io_pairs.append([list(args), exp])
                    else:
                        io_pairs.append([[args], exp])
                predicted_type, probs = self.meta_learner.predict(io_pairs, n_args)
            except Exception:
                pass

        # Layer 1: Differentiable synthesis via mog_synth
        if self.enable_differentiable and self.mog_binary:
            t0 = time.time()
            result = self._try_differentiable(fn_name, examples, signature, holdouts)
            result.time_s = time.time() - t0
            if result.solved:
                return result
            results.append(result)

        # Layer 2: LLM synthesis (Python)
        if self.enable_llm:
            t0 = time.time()
            result = self._try_llm_python(fn_name, examples, description, signature, holdouts)
            result.time_s = time.time() - t0
            if result.solved:
                return result
            results.append(result)

        # Layer 3: LLM synthesis (Mog — for integration with nCPU)
        if self.enable_llm:
            t0 = time.time()
            result = self._try_llm_mog(fn_name, examples, description, signature, holdouts)
            result.time_s = time.time() - t0
            if result.solved:
                return result
            results.append(result)

        # Return best failing result
        if results:
            return results[-1]
        return SynthResult(solved=False, code=None, language="none", method="none",
                           error="All synthesis methods disabled")

    # ── From function/binary ──────────────────────────────────────────────

    def from_function(
        self,
        fn: Callable,
        arg_types: list[str] | None = None,
        n_examples: int = 12,
        n_holdouts: int = 4,
        description: str = "",
    ) -> SynthResult:
        """Extract I/O from a Python function and synthesize."""
        from egdc.binary_io_extract import IOExtractor
        ext = IOExtractor.from_function(fn, arg_types=arg_types, description=description)
        problem = ext.extract(n_examples=n_examples, n_holdouts=n_holdouts)

        examples = [(ex.inputs if len(ex.inputs) > 1 else ex.inputs[0], ex.expected)
                     for ex in problem.examples]
        holdouts = [(ex.inputs if len(ex.inputs) > 1 else ex.inputs[0], ex.expected)
                     for ex in problem.holdouts]

        return self.synthesize(
            fn_name=problem.name,
            examples=examples,
            description=problem.description,
            signature=problem.signature,
            holdouts=holdouts,
        )

    def from_binary(
        self,
        binary_path: str,
        arg_types: list[str],
        fn_name: str = "program",
        n_examples: int = 12,
        n_holdouts: int = 4,
        description: str = "",
    ) -> SynthResult:
        """Extract I/O from a compiled binary and synthesize."""
        from egdc.binary_io_extract import IOExtractor
        ext = IOExtractor.from_binary(binary_path, arg_types, fn_name=fn_name,
                                       description=description)
        problem = ext.extract(n_examples=n_examples, n_holdouts=n_holdouts)

        examples = [(ex.inputs if len(ex.inputs) > 1 else ex.inputs[0], ex.expected)
                     for ex in problem.examples]
        holdouts = [(ex.inputs if len(ex.inputs) > 1 else ex.inputs[0], ex.expected)
                     for ex in problem.holdouts]

        return self.synthesize(
            fn_name=problem.name,
            examples=examples,
            description=problem.description,
            signature=problem.signature,
            holdouts=holdouts,
        )

    # ── Layer 1: Differentiable ───────────────────────────────────────────

    def _try_differentiable(
        self,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        signature: str | None,
        holdouts: list[tuple[Any, Any]] | None,
    ) -> SynthResult:
        """Call mog_synth binary for gradient-based synthesis."""
        # Build problem JSON
        problem_json = self._build_problem_json(fn_name, examples, signature, holdouts)

        try:
            result = subprocess.run(
                [self.mog_binary, "--problem-json", "-"],
                input=problem_json,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                output = json.loads(result.stdout)
                if output.get("success"):
                    return SynthResult(
                        solved=True,
                        code=output["code"],
                        language="mog",
                        method=output.get("method", "differentiable"),
                    )
            return SynthResult(
                solved=False, code=None, language="mog", method="differentiable",
                error=result.stderr[:500] if result.stderr else "synthesis failed",
            )
        except FileNotFoundError:
            return SynthResult(
                solved=False, code=None, language="mog", method="differentiable",
                error=f"mog_synth binary not found: {self.mog_binary}",
            )
        except subprocess.TimeoutExpired:
            return SynthResult(
                solved=False, code=None, language="mog", method="differentiable",
                error="differentiable synthesis timed out (120s)",
            )
        except Exception as e:
            return SynthResult(
                solved=False, code=None, language="mog", method="differentiable",
                error=str(e),
            )

    # ── Layer 2: LLM Python ───────────────────────────────────────────────

    def _try_llm_python(
        self,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        description: str,
        signature: str | None,
        holdouts: list[tuple[Any, Any]] | None,
    ) -> SynthResult:
        """Use Claude to generate Python code, verify against examples + holdouts."""
        try:
            from egdc.llm_synth import LLMSynthesizer
        except ImportError:
            return SynthResult(solved=False, code=None, language="python", method="llm",
                               error="llm_synth not available")

        synth = LLMSynthesizer(model=self.llm_model)
        result = synth.synthesize(
            fn_name=fn_name,
            examples=examples,
            description=description,
            language="python",
            signature=signature,
            max_retries=3,
        )

        if result.solved and holdouts:
            # Extra validation against holdouts
            ok, failing = synth._verify_python(result.code, fn_name, holdouts)
            if not ok:
                return SynthResult(
                    solved=False, code=result.code, language="python", method="llm",
                    attempts=result.attempts,
                    error=f"Passed examples but failed holdouts: {failing[:3]}",
                )

        return SynthResult(
            solved=result.solved,
            code=result.code,
            language="python",
            method="llm",
            attempts=result.attempts,
            error=result.error,
        )

    # ── Layer 3: LLM Mog ─────────────────────────────────────────────────

    def _try_llm_mog(
        self,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        description: str,
        signature: str | None,
        holdouts: list[tuple[Any, Any]] | None,
    ) -> SynthResult:
        """Use Claude to generate Mog code for nCPU integration."""
        try:
            from egdc.llm_synth import LLMSynthesizer
        except ImportError:
            return SynthResult(solved=False, code=None, language="mog", method="llm_mog",
                               error="llm_synth not available")

        synth = LLMSynthesizer(model=self.llm_model)
        result = synth.synthesize_mog(
            fn_name=fn_name,
            examples=examples,
            description=description,
            signature=signature,
            max_retries=3,
        )

        return SynthResult(
            solved=result.solved,
            code=result.code,
            language="mog",
            method="llm_mog",
            attempts=result.attempts,
            error=result.error,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_problem_json(
        fn_name: str,
        examples: list[tuple[Any, Any]],
        signature: str | None,
        holdouts: list[tuple[Any, Any]] | None,
    ) -> str:
        """Build JSON compatible with mog_synth --problem-json."""
        def serialize_inputs(args: Any) -> list:
            if isinstance(args, (list, tuple)):
                out = []
                for a in args:
                    if isinstance(a, list):
                        out.append(a)
                    else:
                        out.append(int(a))
                return out
            return [int(args)]

        ex_list = [{"inputs": serialize_inputs(a), "expected": int(e)} for a, e in examples]
        ho_list = [{"inputs": serialize_inputs(a), "expected": int(e)} for a, e in (holdouts or [])]

        if not signature:
            # Infer from first example
            first_args = examples[0][0]
            if isinstance(first_args, (list, tuple)):
                params = []
                names = ["a", "b", "c", "d"]
                for i, a in enumerate(first_args):
                    n = names[i] if i < len(names) else f"x{i}"
                    if isinstance(a, list):
                        params.append(f"arr: [i64]")
                    else:
                        params.append(f"{n}: i64")
                signature = f"fn {fn_name}({', '.join(params)}) -> i64"
            else:
                signature = f"fn {fn_name}(a: i64) -> i64"

        return json.dumps({
            "name": fn_name,
            "signature": signature,
            "examples": ex_list,
            "holdouts": ho_list,
        })


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Demo: synthesize a function from I/O examples or a Python function."""
    import argparse
    parser = argparse.ArgumentParser(description="Unified program synthesis engine")
    sub = parser.add_subparsers(dest="command")

    # From examples
    ex_parser = sub.add_parser("examples", help="Synthesize from I/O examples")
    ex_parser.add_argument("fn_name")
    ex_parser.add_argument("--examples", required=True, help="JSON array of [input, output] pairs")
    ex_parser.add_argument("--description", default="")
    ex_parser.add_argument("--no-differentiable", action="store_true")
    ex_parser.add_argument("--no-llm", action="store_true")

    # From function
    fn_parser = sub.add_parser("function", help="Synthesize from a Python function")
    fn_parser.add_argument("source", help="file:function_name")
    fn_parser.add_argument("--arg-types", nargs="+", default=["int"])
    fn_parser.add_argument("--n-examples", type=int, default=12)

    # From binary
    bin_parser = sub.add_parser("binary", help="Synthesize from a compiled binary")
    bin_parser.add_argument("binary_path")
    bin_parser.add_argument("--arg-types", nargs="+", default=["int"])
    bin_parser.add_argument("--fn-name", default="program")
    bin_parser.add_argument("--n-examples", type=int, default=12)

    args = parser.parse_args()

    synth = UnifiedSynthesizer()

    if args.command == "examples":
        examples = json.loads(args.examples)
        result = synth.synthesize(
            fn_name=args.fn_name,
            examples=[(e[0], e[1]) for e in examples],
            description=args.description,
        )
    elif args.command == "function":
        file_path, func_name = args.source.rsplit(":", 1)
        from egdc.binary_io_extract import IOExtractor
        ext = IOExtractor.from_module(file_path, func_name, args.arg_types)
        problem = ext.extract(n_examples=args.n_examples)
        examples = [(ex.inputs if len(ex.inputs) > 1 else ex.inputs[0], ex.expected)
                     for ex in problem.examples]
        result = synth.synthesize(fn_name=func_name, examples=examples,
                                  signature=problem.signature)
    elif args.command == "binary":
        result = synth.from_binary(
            args.binary_path, args.arg_types, fn_name=args.fn_name,
            n_examples=args.n_examples,
        )
    else:
        parser.print_help()
        return

    print(f"\n{'='*60}")
    print(f"Result: {'SOLVED' if result.solved else 'FAILED'}")
    print(f"Method: {result.method}")
    print(f"Language: {result.language}")
    print(f"Time: {result.time_s:.2f}s")
    if result.code:
        print(f"\nCode:\n{result.code}")
    if result.error:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    main()
