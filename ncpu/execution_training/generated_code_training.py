"""Mode 3: Train on model-GENERATED code.

The model generates code, we parse it to nCPU ISA, execute differentiably,
and backpropagate the execution error. This trains the model on its own output.

Challenge: text generation (model.generate()) is non-differentiable — you can't
backprop through argmax token sampling. We use a two-phase approach:

  Phase 1 (non-differentiable): model.generate() produces code text
  Phase 2 (differentiable): parse code → SoftProgram → execute → loss → backprop

The execution loss updates the coprocessor weights to be better at arithmetic,
which improves the model's code output indirectly. Optionally, we also compute
a REINFORCE gradient estimate on the generation logits using the execution loss
as a reward signal.

Usage:
    trainer = GeneratedCodeTrainer(
        model=model, tokenizer=tokenizer, engine=engine,
        exec_loss_fn=exec_loss_fn, config=config,
    )

    # Single step: generate, parse, execute, compute loss
    result = trainer.generate_and_evaluate(prompt, test_cases, arg_names)

    # REINFORCE gradient (optional)
    reinforce = trainer.reinforce_loss(prompt, test_cases, arg_names, baseline=0.5)

    # Full training step (for integration with train.py)
    step_result = trainer.training_step(sample, optimizer)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.differentiable.execution import DifferentiableEngine, SoftProgram

from .code_parser import CodeToISAParser, ParseError, ParseResult
from .execution_loss import ExecutionLoss, ExecutionLossResult, ExecutionLossWithParsing
from .data import ExecutionTrainingSample

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of generating and evaluating code."""

    generated_code: str  # The code the model generated
    parse_success: bool  # Whether the code parsed to ISA
    execution_loss: Optional[ExecutionLossResult] = None  # Differentiable execution loss
    generation_log_probs: Optional[torch.Tensor] = None  # Log-probs of generated tokens
    generation_token_ids: Optional[torch.Tensor] = None  # Generated token ids
    parse_result: Optional[ParseResult] = None  # Parsed ISA program
    reward: float = 0.0  # Scalar reward (for REINFORCE)
    error: Optional[str] = None  # Error message if something failed


@dataclass
class GeneratedTrainingStepResult:
    """Result of a single Mode 3 training step."""

    exec_loss: torch.Tensor  # Differentiable execution loss (through SoftProgram)
    reinforce_loss: Optional[torch.Tensor] = None  # REINFORCE policy gradient loss
    lm_loss: Optional[torch.Tensor] = None  # Standard LM next-token loss on reference
    total_loss: torch.Tensor = None  # Combined loss for backprop
    generated_code: str = ""
    parse_success: bool = False
    reward: float = 0.0
    n_generated_tokens: int = 0

    def __post_init__(self):
        if self.total_loss is None:
            self.total_loss = self.exec_loss


class GeneratedCodeTrainer(nn.Module):
    """Mode 3 trainer: generate code, parse, execute, backprop.

    Two-phase approach:
      Phase 1: model.generate() produces code tokens (non-differentiable)
      Phase 2: parse to ISA → SoftProgram → differentiable execution → loss

    The execution loss provides dense gradients to the coprocessor weights.
    Optional REINFORCE loss provides a gradient signal to the generation policy.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        exec_loss_fn: Optional[ExecutionLossWithParsing] = None,
        engine: Optional[DifferentiableEngine] = None,
        parser: Optional[CodeToISAParser] = None,
        device: str = "cpu",
        # Generation config
        max_gen_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        # Loss config
        exec_loss_weight: float = 1.0,
        reinforce_weight: float = 0.01,
        lm_loss_weight: float = 1.0,
        reinforce_baseline: float = 0.0,
        entropy_bonus: float = 0.01,
        # Execution config
        exec_temperature: float = 1.0,
        use_soft_programs: bool = True,
        fallback_loss: float = 10.0,
        max_exec_steps: int = 64,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Execution pipeline
        self.engine = engine or DifferentiableEngine(device=device)
        self.parser = parser or CodeToISAParser()
        self.exec_loss_fn = exec_loss_fn or ExecutionLossWithParsing(
            execution_loss=ExecutionLoss(
                engine=self.engine,
                max_exec_steps=max_exec_steps,
                device=device,
            ),
            use_soft_programs=use_soft_programs,
            temperature=exec_temperature,
            fallback_loss=fallback_loss,
            device=device,
        )

        # Generation config
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # Loss config
        self.exec_loss_weight = exec_loss_weight
        self.reinforce_weight = reinforce_weight
        self.lm_loss_weight = lm_loss_weight
        self.reinforce_baseline = reinforce_baseline
        self.entropy_bonus = entropy_bonus

        # Execution config
        self.exec_temperature = exec_temperature
        self.use_soft_programs = use_soft_programs
        self.fallback_loss_value = fallback_loss

        # Running statistics for adaptive baseline
        self._reward_ema = 0.0
        self._reward_count = 0

    def generate_code(
        self,
        prompt: str,
        return_log_probs: bool = False,
    ) -> tuple[str, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Generate code from the model given a prompt.

        Args:
            prompt: Text prompt describing the coding task
            return_log_probs: Whether to compute log-probs for REINFORCE

        Returns:
            (generated_code, log_probs_tensor_or_None, token_ids_or_None)
        """
        # Encode prompt
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        # Generate with the model
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_gen_tokens,
                temperature=max(self.temperature, 1e-3),
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        generated_ids = outputs.sequences[0, prompt_len:]  # Strip prompt tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract code block if present
        code = self._extract_code(generated_text)

        log_probs = None
        if return_log_probs and len(generated_ids) > 0:
            # Re-run forward pass to get log-probs (needed for REINFORCE)
            log_probs = self._compute_generation_log_probs(
                input_ids, generated_ids, prompt_len
            )

        return code, log_probs, generated_ids

    def _extract_code(self, text: str) -> str:
        """Extract Python code from generated text.

        Handles code blocks (```python ... ```) or raw code.
        """
        # Try to extract from markdown code block
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].strip()

        # Otherwise treat the whole thing as code
        return text.strip()

    def _compute_generation_log_probs(
        self,
        prompt_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute log-probabilities of generated tokens for REINFORCE.

        This requires a forward pass with teacher forcing on the generated
        sequence to get the logits at each position.
        """
        # Concatenate prompt + generated
        full_ids = torch.cat([
            prompt_ids[0],
            generated_ids,
        ]).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids=full_ids)

        # Get logits for the generated positions
        # logits[i] predicts token at position i+1
        gen_logits = outputs.logits[0, prompt_len - 1:-1]  # Shape: [gen_len, vocab]

        # Compute log-probs of the actually generated tokens
        log_probs = F.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)

        return token_log_probs  # Shape: [gen_len]

    def generate_and_evaluate(
        self,
        prompt: str,
        test_cases: list[dict],
        arg_names: Optional[list[str]] = None,
        output_var: Optional[str] = None,
        is_function: bool = False,
        return_log_probs: bool = False,
    ) -> GenerationResult:
        """Generate code from prompt, parse it, execute it, compute loss.

        This is the core Mode 3 operation:
        1. Model generates code from prompt
        2. Parse generated code to nCPU ISA
        3. Execute differentiably on test cases
        4. Return execution loss (differentiable through SoftProgram)

        Args:
            prompt: Task description
            test_cases: [{\"inputs\": {...}, \"expected\": {...}}, ...]
            arg_names: Input variable names
            output_var: Output variable name
            is_function: Whether generated code should be a function
            return_log_probs: Whether to compute REINFORCE log-probs

        Returns:
            GenerationResult with differentiable execution loss
        """
        # Phase 1: Generate code (non-differentiable)
        try:
            code, log_probs, token_ids = self.generate_code(
                prompt, return_log_probs=return_log_probs
            )
        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            return GenerationResult(
                generated_code="",
                parse_success=False,
                error=f"Generation failed: {e}",
            )

        # Phase 2: Parse and execute (differentiable)
        return self._evaluate_code(
            code=code,
            test_cases=test_cases,
            arg_names=arg_names,
            output_var=output_var,
            is_function=is_function,
            log_probs=log_probs,
            token_ids=token_ids,
        )

    def evaluate_code(
        self,
        code: str,
        test_cases: list[dict],
        arg_names: Optional[list[str]] = None,
        output_var: Optional[str] = None,
        is_function: bool = False,
    ) -> GenerationResult:
        """Evaluate pre-generated code (no model generation step).

        Useful for testing or when code comes from an external source.
        """
        return self._evaluate_code(
            code=code,
            test_cases=test_cases,
            arg_names=arg_names,
            output_var=output_var,
            is_function=is_function,
        )

    def _evaluate_code(
        self,
        code: str,
        test_cases: list[dict],
        arg_names: Optional[list[str]] = None,
        output_var: Optional[str] = None,
        is_function: bool = False,
        log_probs: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
    ) -> GenerationResult:
        """Internal: parse code to ISA, execute differentiably, compute loss."""
        if not code.strip():
            return GenerationResult(
                generated_code=code,
                parse_success=False,
                error="Empty code",
            )

        # Parse to ISA
        try:
            exec_result = self.exec_loss_fn(
                code=code,
                test_cases=test_cases,
                arg_names=arg_names,
                output_var=output_var,
                is_function=is_function,
            )
            parse_success = True
        except Exception as e:
            logger.debug(f"Parse/execute failed for generated code: {e}")
            return GenerationResult(
                generated_code=code,
                parse_success=False,
                error=str(e),
            )

        # Compute reward for REINFORCE (lower loss = higher reward)
        loss_val = exec_result.total_loss.item()
        reward = self._loss_to_reward(loss_val)

        # Parse separately for the result
        parse_result = None
        try:
            if is_function:
                parse_result = self.parser.parse_function(code)
            else:
                parse_result = self.parser.parse_block(
                    code, arg_names=arg_names, output_var=output_var
                )
        except ParseError:
            pass

        return GenerationResult(
            generated_code=code,
            parse_success=parse_success,
            execution_loss=exec_result,
            generation_log_probs=log_probs,
            generation_token_ids=token_ids,
            parse_result=parse_result,
            reward=reward,
        )

    def reinforce_loss(
        self,
        generation_result: GenerationResult,
        baseline: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """Compute REINFORCE policy gradient loss from a generation result.

        REINFORCE gradient: ∇θ J ≈ (R - b) * ∇θ log π(a|s)

        Where:
          R = reward (from execution quality)
          b = baseline (to reduce variance)
          π(a|s) = probability of generating the code tokens
          θ = model parameters

        Args:
            generation_result: Output of generate_and_evaluate()
            baseline: Reward baseline (None = use running EMA)

        Returns:
            REINFORCE loss tensor (differentiable through generation logits)
            or None if log-probs aren't available
        """
        log_probs = generation_result.generation_log_probs
        if log_probs is None or len(log_probs) == 0:
            return None

        reward = generation_result.reward

        # Adaptive baseline: exponential moving average of rewards
        if baseline is None:
            baseline = self.reinforce_baseline
            if self._reward_count > 0:
                baseline = self._reward_ema

        # Update running baseline
        self._reward_count += 1
        alpha = min(0.1, 1.0 / self._reward_count)
        self._reward_ema = (1 - alpha) * self._reward_ema + alpha * reward

        # REINFORCE loss: -advantage * sum(log_probs)
        advantage = reward - baseline
        policy_loss = -advantage * log_probs.sum()

        # Entropy bonus (encourage exploration)
        if self.entropy_bonus > 0 and log_probs.requires_grad:
            # Approximate entropy from log-probs
            entropy = -(log_probs * log_probs.exp()).sum()
            policy_loss = policy_loss - self.entropy_bonus * entropy

        return policy_loss

    def training_step(
        self,
        sample: ExecutionTrainingSample,
        use_reinforce: bool = False,
        use_lm_loss: bool = True,
    ) -> GeneratedTrainingStepResult:
        """Full Mode 3 training step.

        1. Generate code from sample prompt
        2. Parse generated code → ISA → SoftProgram
        3. Execute differentiably, compute execution loss
        4. Optionally: REINFORCE gradient on generation logits
        5. Optionally: standard LM loss on reference code
        6. Return combined loss for backprop

        Args:
            sample: Training sample with prompt + test cases
            use_reinforce: Whether to add REINFORCE gradient
            use_lm_loss: Whether to add LM loss on reference code

        Returns:
            GeneratedTrainingStepResult with combined loss
        """
        device = self.device

        # Phase 1: Generate code and evaluate
        gen_result = self.generate_and_evaluate(
            prompt=sample.prompt,
            test_cases=sample.test_cases,
            arg_names=sample.arg_names if sample.arg_names else None,
            output_var=sample.output_var,
            is_function=sample.is_function,
            return_log_probs=use_reinforce,
        )

        # Phase 2: Compute execution loss
        if gen_result.execution_loss is not None:
            exec_loss = gen_result.execution_loss.total_loss
        else:
            # Fallback: try executing reference code instead
            try:
                ref_result = self.exec_loss_fn(
                    code=sample.reference_code,
                    test_cases=sample.test_cases,
                    arg_names=sample.arg_names if sample.arg_names else None,
                    output_var=sample.output_var,
                    is_function=sample.is_function,
                )
                exec_loss = ref_result.total_loss
            except Exception:
                exec_loss = torch.tensor(
                    self.fallback_loss_value, device=device, requires_grad=False
                )

        # Phase 3: Optional REINFORCE loss
        reinforce_loss = None
        if use_reinforce and gen_result.generation_log_probs is not None:
            reinforce_loss = self.reinforce_loss(gen_result)

        # Phase 4: Optional LM loss on reference code
        lm_loss = None
        if use_lm_loss and hasattr(self.model, "forward"):
            lm_loss = self._compute_lm_loss(sample)

        # Combine losses
        total_loss = self.exec_loss_weight * exec_loss
        if reinforce_loss is not None and reinforce_loss.requires_grad:
            total_loss = total_loss + self.reinforce_weight * reinforce_loss
        if lm_loss is not None and lm_loss.requires_grad:
            total_loss = total_loss + self.lm_loss_weight * lm_loss

        n_tokens = 0
        if gen_result.generation_token_ids is not None:
            n_tokens = len(gen_result.generation_token_ids)

        return GeneratedTrainingStepResult(
            exec_loss=exec_loss,
            reinforce_loss=reinforce_loss,
            lm_loss=lm_loss,
            total_loss=total_loss,
            generated_code=gen_result.generated_code,
            parse_success=gen_result.parse_success,
            reward=gen_result.reward,
            n_generated_tokens=n_tokens,
        )

    def _compute_lm_loss(self, sample: ExecutionTrainingSample) -> Optional[torch.Tensor]:
        """Compute standard LM next-token loss on the reference code.

        This provides a secondary training signal: the model should also
        learn to *generate* good code, not just have a good coprocessor.
        """
        try:
            text = f"{sample.prompt}\n\n```python\n{sample.reference_code}\n```"
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False,
            )
            input_ids = encoding["input_ids"].to(self.device)
            labels = input_ids.clone()

            outputs = self.model(input_ids=input_ids, labels=labels)
            return outputs.loss
        except Exception as e:
            logger.debug(f"LM loss computation failed: {e}")
            return None

    def _loss_to_reward(self, loss_val: float) -> float:
        """Convert execution loss to a reward signal for REINFORCE.

        Maps loss ∈ [0, ∞) to reward ∈ (-1, 1):
          - loss=0 → reward≈1 (perfect execution)
          - loss=fallback → reward≈0 (baseline/failed parse)
          - loss→∞ → reward→-1

        Uses: reward = exp(-loss/scale) * 2 - 1
        This gives reward=1 at loss=0 and decays toward -1.
        """
        # Clamp for numerical stability
        loss_val = max(0.0, min(loss_val, 1e6))
        scale = max(self.fallback_loss_value, 1.0)
        # Exponential decay: 1 at loss=0, approaches 0 as loss→∞
        decay = math.exp(-loss_val / scale)
        # Map [0,1] → [-1, 1]
        reward = 2.0 * decay - 1.0
        return reward

    def update_baseline(self, reward: float):
        """Manually update the REINFORCE baseline."""
        self._reward_count += 1
        alpha = min(0.1, 1.0 / self._reward_count)
        self._reward_ema = (1 - alpha) * self._reward_ema + alpha * reward


def create_generated_trainer(
    model: nn.Module,
    tokenizer,
    config,
    device: str = "cpu",
) -> GeneratedCodeTrainer:
    """Factory function to create a GeneratedCodeTrainer from a training config.

    Integrates with train.py's ExecutionTrainingConfig for mode='generated'.
    """
    engine = DifferentiableEngine(device=device)
    exec_loss_fn = ExecutionLossWithParsing(
        execution_loss=ExecutionLoss(
            engine=engine,
            output_weight=1.0,
            trace_weight=getattr(config, "trace_loss_weight", 0.1),
            structure_weight=0.01,
            correctness_tolerance=getattr(config, "correctness_tolerance", 0.5),
            max_exec_steps=getattr(config, "max_exec_steps", 64),
            device=device,
        ),
        use_soft_programs=getattr(config, "use_soft_programs", True),
        temperature=getattr(config, "exec_temperature", 1.0),
        device=device,
    )

    return GeneratedCodeTrainer(
        model=model,
        tokenizer=tokenizer,
        exec_loss_fn=exec_loss_fn,
        engine=engine,
        device=device,
        max_gen_tokens=getattr(config, "max_length", 128),
        exec_loss_weight=getattr(config, "exec_loss_weight", 1.0),
        lm_loss_weight=getattr(config, "lm_loss_weight", 1.0),
        exec_temperature=getattr(config, "exec_temperature", 1.0),
        use_soft_programs=getattr(config, "use_soft_programs", True),
        max_exec_steps=getattr(config, "max_exec_steps", 64),
    )
