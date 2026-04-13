"""EGDC - Execution-Guided Diffusion for Code.

Combines masked discrete diffusion with nCPU's differentiable execution
engine to generate provably-correct code.
"""

from egdc.core.tokenizer import NCPUTokenizer
from egdc.core.data_generator import NCPUDataGenerator
from egdc.core.dataset import NCPUDataset
from egdc.core.model import MaskedDiffusionTransformer, ModelConfig
from egdc.core.noise_schedule import cosine_masking_rate, get_mask, apply_mask
from egdc.core.sampler import generate

# Mog language support
from egdc.mog.tokenizer import MogCodeTokenizer
from egdc.mog.dataset import MogDataset, MogProgramGenerator
from egdc.mog.eval import evaluate_mog_program, evaluate_batch
from egdc.mog.model import MogMaskedDiffusion, MogDiffusionConfig
from egdc.mog.execute import compile_mog, execute_mog, check_mog_output, evaluate_mog_programs
from egdc.mog.benchmark import get_benchmark, evaluate_solution, evaluate_solutions_batch
from egdc.mog.solvers.differentiable import DifferentiableMogExecutor
from egdc.mog.training.execution_guidance import MogExecutionGuidedScorer, execution_guided_generate_mog
from egdc.mog.training.grpo import MogRewardModel, MogRewardConfig
from egdc.mog.training.completion import mask_function_bodies, build_completion_tokens, complete_mog_from_initial
from egdc.mog.training.completion_dataset import MogBenchmarkCompletionDataset
from egdc.mog.training.supervised_dataset import MogBenchmarkSupervisedDataset
from egdc.mog.solvers.direct_synth import synthesize_expression_program, DirectSynthResult
from egdc.mog.routing.direct_router import solve_problem_direct, evaluate_direct_solver
from egdc.mog.routing.pathways import PathwayMemory
from egdc.mog.routing.adaptive_router import AdaptiveMogRouter
from egdc.mog.routing.orchestrator import MogOrchestrator

__all__ = [
    "NCPUTokenizer",
    "NCPUDataGenerator", 
    "NCPUDataset",
    "MaskedDiffusionTransformer",
    "ModelConfig",
    "cosine_masking_rate",
    "get_mask",
    "apply_mask",
    "generate",
    # Mog
    "MogCodeTokenizer",
    "MogDataset",
    "MogProgramGenerator",
    "evaluate_mog_program",
    "evaluate_batch",
    "MogMaskedDiffusion",
    "MogDiffusionConfig",
    "compile_mog",
    "execute_mog",
    "check_mog_output",
    "evaluate_mog_programs",
    "get_benchmark",
    "evaluate_solution",
    "evaluate_solutions_batch",
    "DifferentiableMogExecutor",
    "MogExecutionGuidedScorer",
    "execution_guided_generate_mog",
    "MogRewardModel",
    "MogRewardConfig",
    "mask_function_bodies",
    "build_completion_tokens",
    "complete_mog_from_initial",
    "MogBenchmarkCompletionDataset",
    "MogBenchmarkSupervisedDataset",
    "synthesize_expression_program",
    "DirectSynthResult",
    "solve_problem_direct",
    "evaluate_direct_solver",
    "PathwayMemory",
    "AdaptiveMogRouter",
    "MogOrchestrator",
]
