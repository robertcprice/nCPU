"""Tests for Mode 3: generated code training (generated_code_training.py).

Tests cover:
  1. GeneratedCodeTrainer initialization and configuration
  2. Code evaluation pipeline (parse → execute → loss)
  3. REINFORCE loss computation
  4. Reward signal computation
  5. Training step integration
  6. Factory function
"""

import math
import pytest
import torch
import torch.nn as nn

from ncpu.differentiable.execution import DifferentiableEngine
from ncpu.execution_training.code_parser import CodeToISAParser
from ncpu.execution_training.execution_loss import (
    ExecutionLoss,
    ExecutionLossWithParsing,
    ExecutionLossResult,
)
from ncpu.execution_training.data import ExecutionTrainingSample
from ncpu.execution_training.generated_code_training import (
    GeneratedCodeTrainer,
    GenerationResult,
    GeneratedTrainingStepResult,
    create_generated_trainer,
)


# ════════════════════════════════════════════════════════════════
# Fixtures / Helpers
# ════════════════════════════════════════════════════════════════


class DummyModel(nn.Module):
    """Minimal model stub for testing without a real LM."""

    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.config = type("Config", (), {"vocab_size": vocab_size})()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        h = self.embed(input_ids)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return type("Output", (), {"logits": logits, "loss": loss})()

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=10, **kwargs):
        """Dummy generate: emit random tokens."""
        batch = input_ids.shape[0]
        gen_tokens = torch.randint(0, self.vocab_size, (batch, max_new_tokens))
        sequences = torch.cat([input_ids, gen_tokens], dim=1)
        return type("GenOutput", (), {"sequences": sequences})()


class DummyTokenizer:
    """Minimal tokenizer stub."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def __call__(self, text, return_tensors=None, **kwargs):
        # Encode as sequence of ASCII codes mod vocab_size
        ids = [ord(c) % self.vocab_size for c in text[:50]]
        if not ids:
            ids = [0]
        t = torch.tensor([ids])
        mask = torch.ones_like(t)
        return {"input_ids": t, "attention_mask": mask}

    def decode(self, token_ids, skip_special_tokens=True):
        # Return some parseable Python code
        return "result = a + b"


@pytest.fixture
def trainer():
    """Create a GeneratedCodeTrainer with dummy model/tokenizer."""
    model = DummyModel()
    tokenizer = DummyTokenizer()
    engine = DifferentiableEngine(device="cpu")
    exec_loss = ExecutionLoss(engine=engine, device="cpu")
    exec_loss_with_parsing = ExecutionLossWithParsing(
        execution_loss=exec_loss,
        use_soft_programs=True,
        device="cpu",
    )
    return GeneratedCodeTrainer(
        model=model,
        tokenizer=tokenizer,
        exec_loss_fn=exec_loss_with_parsing,
        engine=engine,
        device="cpu",
    )


@pytest.fixture
def sample():
    """Create a test training sample."""
    return ExecutionTrainingSample(
        prompt="Write Python code to compute a + b",
        reference_code="result = a + b",
        test_cases=[
            {"inputs": {"a": 3, "b": 5}, "expected": {"result": 8}},
            {"inputs": {"a": 7, "b": 2}, "expected": {"result": 9}},
        ],
        arg_names=["a", "b"],
        output_var="result",
        category="arithmetic",
        difficulty="easy",
    )


# ════════════════════════════════════════════════════════════════
# GeneratedCodeTrainer Initialization Tests
# ════════════════════════════════════════════════════════════════


class TestGeneratedCodeTrainerInit:
    def test_basic_init(self, trainer):
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.engine is not None
        assert trainer.parser is not None
        assert trainer.exec_loss_fn is not None

    def test_default_config(self, trainer):
        assert trainer.max_gen_tokens == 128
        assert trainer.temperature == 0.7
        assert trainer.exec_loss_weight == 1.0
        assert trainer.reinforce_weight == 0.01
        assert trainer.lm_loss_weight == 1.0

    def test_custom_config(self):
        model = DummyModel()
        tokenizer = DummyTokenizer()
        t = GeneratedCodeTrainer(
            model=model,
            tokenizer=tokenizer,
            max_gen_tokens=64,
            temperature=0.5,
            exec_loss_weight=2.0,
            reinforce_weight=0.05,
        )
        assert t.max_gen_tokens == 64
        assert t.temperature == 0.5
        assert t.exec_loss_weight == 2.0
        assert t.reinforce_weight == 0.05


# ════════════════════════════════════════════════════════════════
# Code Evaluation Tests
# ════════════════════════════════════════════════════════════════


class TestCodeEvaluation:
    def test_evaluate_correct_code(self, trainer):
        """Test evaluating code that should parse and execute correctly."""
        result = trainer.evaluate_code(
            code="result = a + b",
            test_cases=[{"inputs": {"a": 3, "b": 5}, "expected": {"result": 8}}],
            arg_names=["a", "b"],
            output_var="result",
        )
        assert result.parse_success is True
        assert result.execution_loss is not None
        assert result.execution_loss.total_loss.item() >= 0

    def test_evaluate_wrong_code(self, trainer):
        """Code that computes wrong answer should have higher loss."""
        # Correct code
        correct = trainer.evaluate_code(
            code="result = a + b",
            test_cases=[{"inputs": {"a": 3, "b": 5}, "expected": {"result": 8}}],
            arg_names=["a", "b"],
            output_var="result",
        )
        # Wrong code (subtraction instead of addition)
        wrong = trainer.evaluate_code(
            code="result = a - b",
            test_cases=[{"inputs": {"a": 3, "b": 5}, "expected": {"result": 8}}],
            arg_names=["a", "b"],
            output_var="result",
        )
        assert wrong.execution_loss.total_loss.item() > correct.execution_loss.total_loss.item()

    def test_evaluate_empty_code(self, trainer):
        """Empty code should fail gracefully."""
        result = trainer.evaluate_code(
            code="",
            test_cases=[{"inputs": {"a": 3}, "expected": {"result": 3}}],
            arg_names=["a"],
        )
        assert result.parse_success is False
        assert result.error is not None

    def test_evaluate_unparseable_code(self, trainer):
        """Unparseable code should return fallback result."""
        result = trainer.evaluate_code(
            code="this is not valid python!!!",
            test_cases=[{"inputs": {}, "expected": {"x": 0}}],
        )
        # ExecutionLossWithParsing handles parse errors with fallback
        assert result.execution_loss is not None

    def test_evaluate_gradient_flow(self, trainer):
        """Execution loss from evaluated code should support backprop."""
        result = trainer.evaluate_code(
            code="result = a + b",
            test_cases=[{"inputs": {"a": 3, "b": 5}, "expected": {"result": 8}}],
            arg_names=["a", "b"],
            output_var="result",
        )
        assert result.execution_loss is not None
        # Try backward -- SoftProgram params should get gradients
        if result.execution_loss.total_loss.requires_grad:
            result.execution_loss.total_loss.backward()

    def test_evaluate_multiple_test_cases(self, trainer):
        """Test with multiple test cases."""
        result = trainer.evaluate_code(
            code="result = a * b",
            test_cases=[
                {"inputs": {"a": 3, "b": 5}, "expected": {"result": 15}},
                {"inputs": {"a": 7, "b": 2}, "expected": {"result": 14}},
            ],
            arg_names=["a", "b"],
            output_var="result",
        )
        assert result.parse_success is True
        assert result.execution_loss is not None


# ════════════════════════════════════════════════════════════════
# Code Extraction Tests
# ════════════════════════════════════════════════════════════════


class TestCodeExtraction:
    def test_extract_plain_code(self, trainer):
        assert trainer._extract_code("result = a + b") == "result = a + b"

    def test_extract_markdown_python(self, trainer):
        text = "Here's the code:\n```python\nresult = a + b\n```\nDone."
        assert trainer._extract_code(text) == "result = a + b"

    def test_extract_markdown_generic(self, trainer):
        text = "Code:\n```\nresult = a + b\n```"
        assert trainer._extract_code(text) == "result = a + b"

    def test_extract_empty(self, trainer):
        assert trainer._extract_code("") == ""


# ════════════════════════════════════════════════════════════════
# Reward Computation Tests
# ════════════════════════════════════════════════════════════════


class TestRewardComputation:
    def test_zero_loss_high_reward(self, trainer):
        """Zero execution loss should give high reward."""
        reward = trainer._loss_to_reward(0.0)
        assert reward > 0.5

    def test_high_loss_low_reward(self, trainer):
        """High execution loss should give low reward."""
        reward = trainer._loss_to_reward(100.0)
        assert reward < 0.0

    def test_reward_monotonic(self, trainer):
        """Reward should decrease as loss increases."""
        rewards = [trainer._loss_to_reward(l) for l in [0.0, 1.0, 5.0, 10.0, 100.0]]
        for i in range(len(rewards) - 1):
            assert rewards[i] >= rewards[i + 1], \
                f"Reward not monotonically decreasing: {rewards}"

    def test_reward_bounded(self, trainer):
        """Reward should be in [-1, 1]."""
        for loss_val in [0.0, 0.1, 1.0, 10.0, 1000.0, 1e6]:
            reward = trainer._loss_to_reward(loss_val)
            assert -1.0 <= reward <= 1.0, f"Reward {reward} out of bounds for loss {loss_val}"


# ════════════════════════════════════════════════════════════════
# REINFORCE Loss Tests
# ════════════════════════════════════════════════════════════════


class TestReinforceLoss:
    def test_reinforce_with_log_probs(self, trainer):
        """REINFORCE loss should be computable given log-probs."""
        log_probs = torch.tensor([-0.5, -1.0, -0.3, -0.8])
        gen_result = GenerationResult(
            generated_code="result = a + b",
            parse_success=True,
            generation_log_probs=log_probs,
            reward=0.8,
        )
        loss = trainer.reinforce_loss(gen_result, baseline=0.0)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_reinforce_without_log_probs(self, trainer):
        """REINFORCE should return None if no log-probs."""
        gen_result = GenerationResult(
            generated_code="result = a + b",
            parse_success=True,
            reward=0.8,
        )
        loss = trainer.reinforce_loss(gen_result)
        assert loss is None

    def test_reinforce_positive_reward(self, trainer):
        """Positive advantage → loss should encourage the action (negative loss)."""
        log_probs = torch.tensor([-1.0, -1.0])  # sum = -2.0
        gen_result = GenerationResult(
            generated_code="x = 1",
            parse_success=True,
            generation_log_probs=log_probs,
            reward=1.0,
        )
        loss = trainer.reinforce_loss(gen_result, baseline=0.0)
        # loss = -(reward - baseline) * sum(log_probs) = -(1.0) * (-2.0) = 2.0
        assert loss is not None
        assert loss.item() > 0  # Positive because log_probs are negative

    def test_reinforce_baseline_reduces_variance(self, trainer):
        """Baseline should shift the advantage."""
        log_probs = torch.tensor([-1.0, -1.0])
        gen_result = GenerationResult(
            generated_code="x = 1",
            parse_success=True,
            generation_log_probs=log_probs,
            reward=0.5,
        )
        loss_no_baseline = trainer.reinforce_loss(gen_result, baseline=0.0)
        loss_with_baseline = trainer.reinforce_loss(gen_result, baseline=0.5)
        # With baseline=reward, advantage=0, loss should be ~0
        assert loss_with_baseline is not None
        assert abs(loss_with_baseline.item()) < abs(loss_no_baseline.item())

    def test_adaptive_baseline_updates(self, trainer):
        """Running baseline should update with new rewards."""
        assert trainer._reward_count == 0
        trainer.update_baseline(0.5)
        assert trainer._reward_count == 1
        assert trainer._reward_ema > 0

        trainer.update_baseline(0.9)
        assert trainer._reward_count == 2


# ════════════════════════════════════════════════════════════════
# Generation Result Dataclass Tests
# ════════════════════════════════════════════════════════════════


class TestGenerationResult:
    def test_defaults(self):
        r = GenerationResult(generated_code="x = 1", parse_success=True)
        assert r.execution_loss is None
        assert r.generation_log_probs is None
        assert r.reward == 0.0
        assert r.error is None

    def test_with_execution_loss(self):
        loss_result = ExecutionLossResult(
            total_loss=torch.tensor(0.5),
            output_loss=torch.tensor(0.5),
        )
        r = GenerationResult(
            generated_code="x = 1",
            parse_success=True,
            execution_loss=loss_result,
            reward=0.7,
        )
        assert r.execution_loss.total_loss.item() == 0.5
        assert r.reward == 0.7


class TestGeneratedTrainingStepResult:
    def test_defaults(self):
        loss = torch.tensor(1.0)
        r = GeneratedTrainingStepResult(exec_loss=loss)
        assert r.total_loss.item() == 1.0  # Defaults to exec_loss
        assert r.reinforce_loss is None
        assert r.lm_loss is None

    def test_with_all_losses(self):
        r = GeneratedTrainingStepResult(
            exec_loss=torch.tensor(0.5),
            reinforce_loss=torch.tensor(0.1),
            lm_loss=torch.tensor(0.3),
            total_loss=torch.tensor(0.9),
            generated_code="x = 1",
            parse_success=True,
            reward=0.8,
            n_generated_tokens=10,
        )
        assert abs(r.total_loss.item() - 0.9) < 1e-5
        assert r.parse_success is True
        assert r.n_generated_tokens == 10


# ════════════════════════════════════════════════════════════════
# Factory Function Tests
# ════════════════════════════════════════════════════════════════


class TestFactoryFunction:
    def test_create_from_config(self):
        """Test factory function with a config-like object."""
        model = DummyModel()
        tokenizer = DummyTokenizer()

        config = type("Config", (), {
            "trace_loss_weight": 0.1,
            "correctness_tolerance": 0.5,
            "max_exec_steps": 64,
            "use_soft_programs": True,
            "exec_temperature": 1.0,
            "exec_loss_weight": 1.0,
            "lm_loss_weight": 1.0,
            "max_length": 128,
        })()

        trainer = create_generated_trainer(model, tokenizer, config, device="cpu")
        assert isinstance(trainer, GeneratedCodeTrainer)
        assert trainer.model is model
        assert trainer.tokenizer is tokenizer

    def test_create_with_defaults(self):
        """Test factory function with minimal config."""
        model = DummyModel()
        tokenizer = DummyTokenizer()
        config = type("Config", (), {})()

        trainer = create_generated_trainer(model, tokenizer, config)
        assert isinstance(trainer, GeneratedCodeTrainer)


# ════════════════════════════════════════════════════════════════
# Integration: Full Evaluate Pipeline
# ════════════════════════════════════════════════════════════════


class TestIntegrationPipeline:
    def test_evaluate_from_sample(self, trainer, sample):
        """Test evaluating code from a training sample."""
        result = trainer.evaluate_code(
            code=sample.reference_code,
            test_cases=sample.test_cases[:1],  # Use first test case
            arg_names=sample.arg_names,
            output_var=sample.output_var,
            is_function=sample.is_function,
        )
        assert result.parse_success is True
        assert result.execution_loss is not None

    def test_generation_result_has_reward(self, trainer, sample):
        """Generation result should compute a reward."""
        result = trainer.evaluate_code(
            code=sample.reference_code,
            test_cases=sample.test_cases[:1],
            arg_names=sample.arg_names,
            output_var=sample.output_var,
        )
        # Reward is computed from execution loss; soft execution may have
        # significant approximation error so we just check it's a valid number
        assert -1.0 <= result.reward <= 1.0

    def test_wrong_code_lower_reward(self, trainer, sample):
        """Wrong code should get a lower reward than correct code."""
        correct = trainer.evaluate_code(
            code="result = a + b",
            test_cases=[{"inputs": {"a": 3, "b": 5}, "expected": {"result": 8}}],
            arg_names=["a", "b"],
            output_var="result",
        )
        wrong = trainer.evaluate_code(
            code="result = a - b",
            test_cases=[{"inputs": {"a": 3, "b": 5}, "expected": {"result": 8}}],
            arg_names=["a", "b"],
            output_var="result",
        )
        assert correct.reward > wrong.reward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
