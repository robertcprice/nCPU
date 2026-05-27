"""Differentiable program execution, optimization, synthesis, and compilation.

This package provides fully differentiable CPU execution that supports
gradient-based program optimization, program synthesis, and differentiable
compilation --- the central capabilities that a differentiable computer enables.

Main entry points:

- DifferentiableEngine: Execute programs with full gradient flow
- ProgramOptimizer: Optimize program parameters via backpropagation
- ProgramSynthesizer: Discover programs from input-output specifications
- NeuralISADiscovery: Learn optimal instruction sets via gradient descent
- NeuralFloatALU: Differentiable floating-point arithmetic
- SelfModifyingProgram / SelfModifyingEngine: Programs that rewrite their
  own instruction memory during execution, with full gradient flow
- DifferentiableCompiler: Neural compiler with gradient flow through compilation
- DifferentiableCompilationPipeline: End-to-end source -> compile -> execute
- SynthesisOSPipeline: End-to-end synthesis + differentiable OS execution
- GradientGuidedSynthesis: Use OS gradients to steer program synthesis search
"""

from .execution import (
    DifferentiableEngine,
    FixedProgram,
    SoftProgram,
    Instruction,
    ExecutionResult,
    OPCODES,
)
from .program_optimizer import ProgramOptimizer, OptimizationResult
from .program_synthesis import (
    ProgramSynthesizer,
    SynthesisResult,
    SynthesisSpec,
    make_addition_spec,
    make_multiply_spec,
    make_polynomial_spec,
    make_max_spec,
)
from .isa_discovery import NeuralISADiscovery, ISAConfig, ISADiscoveryResult
from .float_alu import NeuralFloatALU, FloatPrecision
from .self_modifying import (
    SelfModifyingProgram,
    SelfModifyingEngine,
    SelfModifyingResult,
)
from .diff_compiler import (
    AutoregressiveDecoder,
    DifferentiableCompiler,
    DifferentiableCompilationPipeline,
    CompilationResult,
    SimpleTokenizer,
)
from .synthesis_integration import (
    DifferentiableProgramExecutor,
    SynthesisOSPipeline,
    GradientGuidedSynthesis,
    ExecutorResult,
    PipelineResult,
    GradientGuidance,
)

__all__ = [
    "DifferentiableEngine",
    "FixedProgram",
    "SoftProgram",
    "Instruction",
    "ExecutionResult",
    "OPCODES",
    "ProgramOptimizer",
    "OptimizationResult",
    "ProgramSynthesizer",
    "SynthesisResult",
    "SynthesisSpec",
    "make_addition_spec",
    "make_multiply_spec",
    "make_polynomial_spec",
    "make_max_spec",
    "NeuralISADiscovery",
    "ISAConfig",
    "ISADiscoveryResult",
    "NeuralFloatALU",
    "FloatPrecision",
    "SelfModifyingProgram",
    "SelfModifyingEngine",
    "SelfModifyingResult",
    "AutoregressiveDecoder",
    "DifferentiableCompiler",
    "DifferentiableCompilationPipeline",
    "CompilationResult",
    "SimpleTokenizer",
    "DifferentiableProgramExecutor",
    "SynthesisOSPipeline",
    "GradientGuidedSynthesis",
    "ExecutorResult",
    "PipelineResult",
    "GradientGuidance",
]
