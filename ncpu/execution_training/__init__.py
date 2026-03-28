"""Differentiable execution as a training signal for code generation models.

This package bridges nCPU's differentiable CPU with language model training.
Instead of sparse pass/fail rewards, execution error gradients flow directly
back into the model's weights — providing dense, per-operation training signal.

Three training modes:
  Mode 1: Coprocessor + Execution Loss (simplest, most practical)
  Mode 2: Differentiable Compilation (end-to-end, most ambitious)
  Mode 3: Program Optimization Feedback (inference-time constant tuning)

Quick start:
    from ncpu.execution_training import (
        CodeToISAParser,
        ExecutionLoss,
        ExecutionTrainer,
        ExecutionEvaluator,
        ExecutionTrainingDataset,
    )
"""

from .code_parser import CodeToISAParser, ParseResult, VariableMap
from .execution_loss import ExecutionLoss, ExecutionLossResult
from .data import (
    ExecutionTrainingDataset,
    ExecutionTrainingSample,
    ArithmeticFunctionGenerator,
    VariableTrackingGenerator,
    LoopProblemGenerator,
)
from .evaluate import ExecutionEvaluator, EvaluationResult
from .train import train_execution_grounded, ExecutionTrainingConfig

__all__ = [
    "CodeToISAParser",
    "ParseResult",
    "VariableMap",
    "ExecutionLoss",
    "ExecutionLossResult",
    "ExecutionTrainingDataset",
    "ExecutionTrainingSample",
    "ArithmeticFunctionGenerator",
    "VariableTrackingGenerator",
    "LoopProblemGenerator",
    "ExecutionEvaluator",
    "EvaluationResult",
    "train_execution_grounded",
    "ExecutionTrainingConfig",
]
