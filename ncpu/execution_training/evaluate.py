"""Evaluation harness for execution-grounded code model training.

Measures multiple dimensions:
1. Arithmetic accuracy: does the model's code produce correct numeric outputs?
2. Execution-grounded accuracy: does the differentiable engine reach the right state?
3. Parse success rate: what fraction of generated code can be parsed to nCPU ISA?
4. Per-category breakdown: arithmetic vs variable-tracking vs loop problems

Usage:
    evaluator = ExecutionEvaluator(engine=engine)
    result = evaluator.evaluate(
        model, tokenizer, test_samples,
        temperature=0.1, max_new_tokens=128
    )
    print(result.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from ncpu.differentiable.execution import DifferentiableEngine

from .code_parser import CodeToISAParser, ParseError
from .execution_loss import ExecutionLoss
from .data import ExecutionTrainingSample

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results."""

    # Overall
    total_samples: int = 0
    total_correct: int = 0  # Code produces correct output
    total_parseable: int = 0  # Code parses to nCPU ISA
    total_executable: int = 0  # Parsed program executes without error
    total_exec_correct: int = 0  # Differentiable execution matches expected

    # Per category
    category_results: dict[str, dict] = field(default_factory=dict)

    # Per difficulty
    difficulty_results: dict[str, dict] = field(default_factory=dict)

    # Losses
    mean_execution_loss: float = 0.0
    mean_output_loss: float = 0.0

    # Timing
    eval_time_seconds: float = 0.0

    # Individual results for debugging
    per_sample: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.total_correct / max(self.total_samples, 1)

    @property
    def parse_rate(self) -> float:
        return self.total_parseable / max(self.total_samples, 1)

    @property
    def exec_accuracy(self) -> float:
        return self.total_exec_correct / max(self.total_executable, 1)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "EXECUTION-GROUNDED EVALUATION RESULTS",
            "=" * 60,
            f"Total samples:        {self.total_samples}",
            f"Code correct:         {self.total_correct}/{self.total_samples} "
            f"({self.accuracy:.1%})",
            f"Parse success:        {self.total_parseable}/{self.total_samples} "
            f"({self.parse_rate:.1%})",
            f"Executable:           {self.total_executable}/{self.total_parseable}",
            f"Exec correct:         {self.total_exec_correct}/{self.total_executable} "
            f"({self.exec_accuracy:.1%})",
            f"Mean execution loss:  {self.mean_execution_loss:.6f}",
            f"Mean output loss:     {self.mean_output_loss:.6f}",
            f"Eval time:            {self.eval_time_seconds:.1f}s",
            "",
        ]

        if self.category_results:
            lines.append("Per-category breakdown:")
            for cat, res in sorted(self.category_results.items()):
                n = res.get("total", 0)
                correct = res.get("correct", 0)
                parsed = res.get("parseable", 0)
                exec_correct = res.get("exec_correct", 0)
                lines.append(
                    f"  {cat:20s}: "
                    f"correct={correct}/{n} ({correct / max(n, 1):.1%}), "
                    f"parsed={parsed}/{n} ({parsed / max(n, 1):.1%}), "
                    f"exec_correct={exec_correct}"
                )

        if self.difficulty_results:
            lines.append("\nPer-difficulty breakdown:")
            for diff, res in sorted(self.difficulty_results.items()):
                n = res.get("total", 0)
                correct = res.get("correct", 0)
                lines.append(
                    f"  {diff:10s}: correct={correct}/{n} ({correct / max(n, 1):.1%})"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


class ExecutionEvaluator:
    """Evaluates code model quality using differentiable execution.

    Generates code from the model, parses it, executes on the differentiable
    engine, and compares against expected outputs.
    """

    def __init__(
        self,
        engine: Optional[DifferentiableEngine] = None,
        execution_loss: Optional[ExecutionLoss] = None,
        parser: Optional[CodeToISAParser] = None,
        correctness_tolerance: float = 0.5,
        device: str = "cpu",
    ):
        self.device = device
        self.engine = engine or DifferentiableEngine(device=device)
        self.execution_loss = execution_loss or ExecutionLoss(
            engine=self.engine,
            correctness_tolerance=correctness_tolerance,
            device=device,
        )
        self.parser = parser or CodeToISAParser()
        self.tolerance = correctness_tolerance

    @torch.no_grad()
    def evaluate(
        self,
        model,
        tokenizer,
        test_samples: list[ExecutionTrainingSample],
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        batch_size: int = 1,
    ) -> EvaluationResult:
        """Run full evaluation.

        Args:
            model: HuggingFace language model
            tokenizer: Corresponding tokenizer
            test_samples: List of ExecutionTrainingSample
            max_new_tokens: Max tokens to generate
            temperature: Generation temperature (low = more deterministic)
            batch_size: Generation batch size

        Returns:
            EvaluationResult with detailed metrics
        """
        start_time = time.time()
        result = EvaluationResult(total_samples=len(test_samples))
        total_exec_loss = 0.0
        total_output_loss = 0.0
        n_exec = 0

        for sample in test_samples:
            sample_result = self._evaluate_single(
                model, tokenizer, sample, max_new_tokens, temperature
            )
            result.per_sample.append(sample_result)

            # Aggregate
            if sample_result["code_correct"]:
                result.total_correct += 1
            if sample_result["parseable"]:
                result.total_parseable += 1
            if sample_result["executable"]:
                result.total_executable += 1
            if sample_result["exec_correct"]:
                result.total_exec_correct += 1
            if sample_result.get("exec_loss") is not None:
                total_exec_loss += sample_result["exec_loss"]
                total_output_loss += sample_result.get("output_loss", 0.0)
                n_exec += 1

            # Per category
            cat = sample.category
            if cat not in result.category_results:
                result.category_results[cat] = {
                    "total": 0, "correct": 0, "parseable": 0, "exec_correct": 0
                }
            result.category_results[cat]["total"] += 1
            if sample_result["code_correct"]:
                result.category_results[cat]["correct"] += 1
            if sample_result["parseable"]:
                result.category_results[cat]["parseable"] += 1
            if sample_result["exec_correct"]:
                result.category_results[cat]["exec_correct"] += 1

            # Per difficulty
            diff = sample.difficulty
            if diff not in result.difficulty_results:
                result.difficulty_results[diff] = {"total": 0, "correct": 0}
            result.difficulty_results[diff]["total"] += 1
            if sample_result["code_correct"]:
                result.difficulty_results[diff]["correct"] += 1

        result.mean_execution_loss = total_exec_loss / max(n_exec, 1)
        result.mean_output_loss = total_output_loss / max(n_exec, 1)
        result.eval_time_seconds = time.time() - start_time

        return result

    def evaluate_reference_only(
        self,
        test_samples: list[ExecutionTrainingSample],
    ) -> EvaluationResult:
        """Evaluate only the reference code (no model generation).

        Useful for validating the data pipeline: parse reference code,
        execute it, check that expected outputs match.
        """
        result = EvaluationResult(total_samples=len(test_samples))
        total_exec_loss = 0.0
        total_output_loss = 0.0
        n_exec = 0

        for sample in test_samples:
            sr = self._evaluate_code(
                sample.reference_code, sample, is_reference=True
            )
            result.per_sample.append(sr)

            if sr["code_correct"]:
                result.total_correct += 1
            if sr["parseable"]:
                result.total_parseable += 1
            if sr["executable"]:
                result.total_executable += 1
            if sr["exec_correct"]:
                result.total_exec_correct += 1
            if sr.get("exec_loss") is not None:
                total_exec_loss += sr["exec_loss"]
                total_output_loss += sr.get("output_loss", 0.0)
                n_exec += 1

            cat = sample.category
            if cat not in result.category_results:
                result.category_results[cat] = {
                    "total": 0, "correct": 0, "parseable": 0, "exec_correct": 0
                }
            result.category_results[cat]["total"] += 1
            if sr["code_correct"]:
                result.category_results[cat]["correct"] += 1
            if sr["parseable"]:
                result.category_results[cat]["parseable"] += 1
            if sr["exec_correct"]:
                result.category_results[cat]["exec_correct"] += 1

        result.mean_execution_loss = total_exec_loss / max(n_exec, 1)
        result.mean_output_loss = total_output_loss / max(n_exec, 1)
        return result

    def _evaluate_single(
        self,
        model,
        tokenizer,
        sample: ExecutionTrainingSample,
        max_new_tokens: int,
        temperature: float,
    ) -> dict:
        """Evaluate a single sample: generate code, parse, execute."""
        # Generate code from model
        prompt = sample.prompt + "\n\n```python\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                }
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature

                output_ids = model.generate(**inputs, **gen_kwargs)

            # Decode generated text
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Extract code (up to ```)
            if "```" in generated:
                generated = generated[:generated.index("```")]
            generated = generated.strip()

        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            return {
                "code_correct": False,
                "parseable": False,
                "executable": False,
                "exec_correct": False,
                "generated_code": "",
                "error": str(e),
            }

        return self._evaluate_code(generated, sample)

    def _evaluate_code(
        self,
        code: str,
        sample: ExecutionTrainingSample,
        is_reference: bool = False,
    ) -> dict:
        """Evaluate a code string against sample's test cases."""
        result = {
            "code_correct": False,
            "parseable": False,
            "executable": False,
            "exec_correct": False,
            "generated_code": code,
            "category": sample.category,
            "difficulty": sample.difficulty,
        }

        if not code.strip():
            result["error"] = "Empty code"
            return result

        # ── 1. Direct Python execution check ──
        try:
            code_correct = self._check_python_execution(code, sample)
            result["code_correct"] = code_correct
        except Exception as e:
            result["python_error"] = str(e)

        # ── 2. Parse to nCPU ISA ──
        try:
            if sample.is_function:
                parse_result = self.parser.parse_function(code)
            else:
                parse_result = self.parser.parse_block(
                    code,
                    arg_names=sample.arg_names if sample.arg_names else None,
                    output_var=sample.output_var,
                )
            result["parseable"] = True
            result["parse_warnings"] = parse_result.warnings
            result["asm"] = parse_result.to_asm()
        except (ParseError, Exception) as e:
            result["parse_error"] = str(e)
            return result

        # ── 3. Execute on differentiable engine ──
        try:
            tc = sample.test_cases[0]
            inputs = {}
            for var_name, val in tc.get("inputs", {}).items():
                reg = parse_result.variable_map.get(var_name)
                if reg is not None:
                    inputs[reg] = float(val)

            expected = {}
            if "expected" in tc:
                for var_name, val in tc["expected"].items():
                    reg = parse_result.variable_map.get(var_name)
                    if reg is not None:
                        expected[reg] = float(val)
            elif "output" in tc:
                expected[parse_result.output_register] = float(tc["output"])

            if not expected:
                result["error"] = "No expected values mapped to registers"
                return result

            fixed_prog = parse_result.to_fixed_program()
            loss_result = self.execution_loss.compute_fixed(
                fixed_prog, inputs=inputs, expected=expected
            )

            result["executable"] = True
            result["exec_loss"] = loss_result.total_loss.item()
            result["output_loss"] = loss_result.output_loss.item()
            result["exec_correct"] = loss_result.accuracy >= 1.0
            result["per_register_loss"] = loss_result.per_register_loss

            if loss_result.execution_result:
                result["final_registers"] = (
                    loss_result.execution_result.registers.detach().tolist()
                )
                result["steps_executed"] = loss_result.execution_result.steps_executed
                result["halted"] = loss_result.execution_result.halted

        except Exception as e:
            result["exec_error"] = str(e)

        return result

    def _check_python_execution(
        self, code: str, sample: ExecutionTrainingSample
    ) -> bool:
        """Execute the code in Python and check against expected output."""
        tc = sample.test_cases[0]
        inputs = tc.get("inputs", {})
        expected = tc.get("expected", {})
        if not expected and "output" in tc:
            expected = {sample.output_var: tc["output"]}

        # Build execution environment
        env = dict(inputs)
        try:
            exec(code, {"__builtins__": {}}, env)
        except Exception:
            return False

        # Check expected values
        for var, val in expected.items():
            if var not in env:
                return False
            if abs(env[var] - val) > self.tolerance:
                return False

        return True
