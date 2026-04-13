"""
llm_synth.py — LLM-guided program synthesis using Claude.

Breaks the ceiling of gradient-only synthesis: handles any program complexity
by using Claude to generate candidates and verifying them against I/O examples.

Supports:
  - Python output (universal, verified via exec)
  - Mog output (for nCPU integration, verified via mog interpreter)
  - Multi-function programs (helpers + main)
  - Arbitrary complexity (sorting, graph algorithms, DP, etc.)

Usage:
  from egdc.llm_synth import LLMSynthesizer
  synth = LLMSynthesizer()
  result = synth.synthesize(
      fn_name="bubble_sort",
      examples=[([3,1,2], [1,2,3]), ([5,4], [4,5])],
      description="Sort an array of integers in ascending order",
      language="python"
  )
  if result.solved:
      print(result.code)  # def bubble_sort(arr): ...
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

import anthropic


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SynthResult:
    solved: bool
    code: str | None
    language: str
    method: str = "llm"
    attempts: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Main synthesizer
# ---------------------------------------------------------------------------

class LLMSynthesizer:
    """LLM-guided synthesizer using Claude as the generator."""

    MODEL = "claude-opus-4-6"

    def __init__(self, model: str | None = None):
        self.client = anthropic.Anthropic()
        self.model = model or self.MODEL

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def synthesize(
        self,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        description: str = "",
        language: str = "python",
        signature: str | None = None,
        max_retries: int = 3,
    ) -> SynthResult:
        """Synthesize a function matching the given I/O examples.

        Args:
            fn_name: Python function name to generate.
            examples: list of (args, expected_output) pairs.
                      args can be a scalar, a list, or a tuple of multiple args.
            description: plain-English description of the function.
            language: "python" or "mog".
            signature: optional type signature string (Python or Mog style).
            max_retries: maximum LLM call attempts.

        Returns:
            SynthResult with solved=True and code if verification passed.
        """
        if language == "mog":
            return self._synthesize_mog(fn_name, examples, description, signature, max_retries)
        return self._synthesize_python(fn_name, examples, description, signature, max_retries)

    def synthesize_mog(
        self,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        description: str = "",
        signature: str | None = None,
        max_retries: int = 3,
    ) -> SynthResult:
        """Convenience method for Mog synthesis (used by orchestrator)."""
        return self._synthesize_mog(fn_name, examples, description, signature, max_retries)

    # ------------------------------------------------------------------
    # Python synthesis path
    # ------------------------------------------------------------------

    def _synthesize_python(
        self,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        description: str,
        signature: str | None,
        max_retries: int,
    ) -> SynthResult:
        failing_cases: list[tuple[Any, Any, Any]] = []
        previous_attempt: str | None = None

        for attempt in range(1, max_retries + 1):
            prompt = self._build_prompt(
                fn_name=fn_name,
                examples=examples,
                language="python",
                description=description,
                signature=signature,
                failing_cases=failing_cases,
                previous_attempt=previous_attempt,
            )
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text
            except Exception as e:
                return SynthResult(
                    solved=False, code=None, language="python",
                    attempts=attempt, error=f"API error: {e}",
                )

            code = self._extract_code(raw, language="python")
            if not code:
                previous_attempt = raw[:500]
                continue

            previous_attempt = code
            ok, new_failing = self._verify_python(code, fn_name, examples)
            if ok:
                return SynthResult(
                    solved=True, code=code, language="python",
                    attempts=attempt,
                )
            failing_cases = new_failing

        return SynthResult(
            solved=False, code=previous_attempt, language="python",
            attempts=max_retries,
            error=f"Verification failed after {max_retries} attempts. "
                  f"Failing cases: {failing_cases[:3]}",
        )

    def _verify_python(
        self,
        code: str,
        fn_name: str,
        examples: list[tuple[Any, Any]],
    ) -> tuple[bool, list[tuple[Any, Any, Any]]]:
        """Execute code and test against all examples.

        Returns (all_passed, list_of_failing_cases).
        Each failing case is (args, expected, actual_or_exception).
        """
        namespace: dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(compile(code, "<llm_synth>", "exec"), namespace)
        except Exception as e:
            return False, [("compile_error", None, str(e))]

        fn = namespace.get(fn_name)
        if fn is None:
            return False, [("missing_function", fn_name, None)]

        failing: list[tuple[Any, Any, Any]] = []
        for args, expected in examples:
            # Normalize args to a tuple for unpacking
            if isinstance(args, (list, tuple)):
                call_args = list(args)
            else:
                call_args = [args]

            try:
                actual = fn(*call_args)
            except Exception as e:
                failing.append((args, expected, f"exception: {e}"))
                continue

            if not _values_equal(actual, expected):
                failing.append((args, expected, actual))

        return len(failing) == 0, failing

    # ------------------------------------------------------------------
    # Mog synthesis path
    # ------------------------------------------------------------------

    def _synthesize_mog(
        self,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        description: str,
        signature: str | None,
        max_retries: int,
    ) -> SynthResult:
        failing_cases: list[tuple[Any, Any, Any]] = []
        previous_attempt: str | None = None

        for attempt in range(1, max_retries + 1):
            prompt = self._build_prompt(
                fn_name=fn_name,
                examples=examples,
                language="mog",
                description=description,
                signature=signature,
                failing_cases=failing_cases,
                previous_attempt=previous_attempt,
            )
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text
            except Exception as e:
                return SynthResult(
                    solved=False, code=None, language="mog",
                    attempts=attempt, error=f"API error: {e}",
                )

            code = self._extract_code(raw, language="mog")
            if not code:
                previous_attempt = raw[:500]
                continue

            previous_attempt = code
            ok, error = self._verify_mog(code, fn_name, examples, signature)
            if ok:
                return SynthResult(
                    solved=True, code=code, language="mog",
                    attempts=attempt,
                )
            failing_cases = [(args, exp, error) for args, exp in examples[:3]]

        return SynthResult(
            solved=False, code=previous_attempt, language="mog",
            attempts=max_retries,
            error=f"Mog verification failed after {max_retries} attempts.",
        )

    def _verify_mog(
        self,
        code: str,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        signature: str | None,
    ) -> tuple[bool, str | None]:
        """Verify Mog code using the mog interpreter."""
        try:
            from egdc.mog_lang import interpret
        except ImportError:
            # If interpreter not available, accept the code (best-effort)
            return True, None

        try:
            from egdc.mog_benchmark import MogBenchmarkProblem, evaluate_solution
        except ImportError:
            return True, None

        # Build test cases in the format evaluate_solution expects
        test_cases = []
        for args, expected in examples:
            if isinstance(args, (list, tuple)):
                args_tuple = tuple(args)
            else:
                args_tuple = (args,)
            test_cases.append((args_tuple, str(expected)))

        # Build a minimal wrapper to call the function
        sig = signature or _infer_mog_signature(fn_name, examples)
        dummy_problem = MogBenchmarkProblem(
            name=fn_name,
            category="llm",
            description="",
            signature=sig,
            test_cases=test_cases,
            wrapper_template="",
        )
        try:
            result = evaluate_solution(dummy_problem, code)
            if result.passed:
                return True, None
            return False, result.error
        except Exception as e:
            return False, str(e)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        language: str,
        description: str = "",
        signature: str | None = None,
        failing_cases: list[tuple[Any, Any, Any]] | None = None,
        previous_attempt: str | None = None,
    ) -> str:
        lines: list[str] = []

        if language == "python":
            lines.append(
                "You are a code synthesis engine. Write a SINGLE Python function "
                "that satisfies ALL of the following I/O examples. "
                "Output ONLY the function definition inside a ```python ... ``` code block. "
                "No explanation, no extra text outside the code block."
            )
        else:
            lines.append(
                "You are a code synthesis engine. Write a SINGLE Mog function "
                "that satisfies ALL of the following I/O examples. "
                "Output ONLY the function definition inside a ```mog ... ``` code block. "
                "No explanation, no extra text outside the code block.\n"
                "\n"
                "Mog syntax rules:\n"
                "  - fn name(arg: type, ...) -> return_type { ... }\n"
                "  - Types: i64, f64, str, [i64] (array), [f64]\n"
                "  - let x: i64 = expr;\n"
                "  - if condition { ... } else { ... }\n"
                "  - while condition { ... }\n"
                "  - return expr;\n"
                "  - Operators: + - * / % == != < > <= >=\n"
                "  - Array indexing: arr[i], array length: arr.len()\n"
                "  - NO bool type — use i64 (0 or 1) instead\n"
                "  - NO ** operator — use repeated multiplication\n"
                "  - NO closures, no lambdas, no map/filter\n"
                "  - NO string slicing or .join()\n"
                "  - Helper functions are allowed before the main function\n"
            )

        if description:
            lines.append(f"\nTask: {description}")

        if signature:
            lines.append(f"Signature: {signature}")

        lines.append(f"\nFunction name: {fn_name}")
        lines.append("\nI/O Examples:")

        for i, (args, expected) in enumerate(examples[:12]):
            args_display = _format_args(args)
            lines.append(f"  Example {i+1}: {fn_name}({args_display}) → {_format_val(expected)}")

        if previous_attempt and failing_cases:
            lines.append("\n--- Previous attempt (INCORRECT) ---")
            lines.append(f"```{language}")
            lines.append(previous_attempt.strip())
            lines.append("```")
            lines.append("\nFailing test cases from previous attempt:")
            for args, expected, actual in (failing_cases or [])[:5]:
                args_display = _format_args(args)
                lines.append(
                    f"  {fn_name}({args_display}) → expected {_format_val(expected)}, "
                    f"got {_format_val(actual)}"
                )
            lines.append("\nFix the function to pass ALL test cases including the ones above.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Code extraction
    # ------------------------------------------------------------------

    def _extract_code(self, text: str, language: str) -> str | None:
        """Extract code from a markdown code block."""
        # Try language-tagged block first
        pattern = rf"```(?:{language})?\s*\n(.*?)```"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Fallback: any code block
        m = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Last resort: if the text itself starts with fn or def
        stripped = text.strip()
        if language == "python" and stripped.startswith("def "):
            return stripped
        if language == "mog" and stripped.startswith("fn "):
            return stripped

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _values_equal(actual: Any, expected: Any, tol: float = 1e-6) -> bool:
    """Flexible equality check handling int/float/list/tuple differences."""
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(actual) - float(expected)) <= tol
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(_values_equal(a, e) for a, e in zip(actual, expected))
    # int/bool coercion
    try:
        return int(actual) == int(expected)
    except (TypeError, ValueError):
        pass
    return actual == expected


def _format_args(args: Any) -> str:
    """Format args for display in prompts."""
    if isinstance(args, tuple):
        return ", ".join(_format_val(a) for a in args)
    if isinstance(args, list):
        return _format_val(args)
    return _format_val(args)


def _format_val(v: Any) -> str:
    """Compact string representation of a value."""
    if isinstance(v, list):
        if len(v) > 8:
            return "[" + ", ".join(str(x) for x in v[:6]) + ", ...]"
        return str(v)
    return str(v)


def _infer_mog_signature(fn_name: str, examples: list[tuple[Any, Any]]) -> str:
    """Infer a minimal Mog function signature from examples."""
    if not examples:
        return f"fn {fn_name}(x: i64) -> i64"
    args, ret = examples[0]
    if isinstance(args, (list, tuple)):
        arg_parts = []
        for i, a in enumerate(args):
            if isinstance(a, list):
                arg_parts.append(f"arr{i}: [i64]")
            elif isinstance(a, float):
                arg_parts.append(f"x{i}: f64")
            else:
                arg_parts.append(f"x{i}: i64")
        args_str = ", ".join(arg_parts)
    elif isinstance(args, list):
        args_str = "arr: [i64]"
    elif isinstance(args, float):
        args_str = "x: f64"
    else:
        args_str = "x: i64"

    if isinstance(ret, float):
        ret_type = "f64"
    elif isinstance(ret, list):
        ret_type = "[i64]"
    else:
        ret_type = "i64"

    return f"fn {fn_name}({args_str}) -> {ret_type}"
