"""EGDC - Execution-Guided Diffusion for Code.

Combines masked discrete diffusion with nCPU's differentiable execution
engine to generate provably-correct code.
"""

from egdc.tokenizer import NCPUTokenizer
from egdc.data_generator import NCPUDataGenerator
from egdc.dataset import NCPUDataset
from egdc.model import MaskedDiffusionTransformer, ModelConfig
from egdc.noise_schedule import cosine_masking_rate, get_mask, apply_mask
from egdc.sampler import generate

# Mog language support
from egdc.mog_tokenizer import MogCodeTokenizer
from egdc.mog_dataset import MogDataset, MogProgramGenerator
from egdc.mog_eval import evaluate_mog_program, evaluate_batch
from egdc.mog_model import MogMaskedDiffusion, MogDiffusionConfig
from egdc.mog_execute import compile_mog, execute_mog, check_mog_output, evaluate_mog_programs
from egdc.mog_benchmark import get_benchmark, evaluate_solution, evaluate_solutions_batch
from egdc.mog_differentiable import DifferentiableMogExecutor
from egdc.mog_execution_guidance import MogExecutionGuidedScorer, execution_guided_generate_mog
from egdc.mog_grpo import MogRewardModel, MogRewardConfig
from egdc.mog_completion import mask_function_bodies, build_completion_tokens, complete_mog_from_initial
from egdc.mog_completion_dataset import MogBenchmarkCompletionDataset
from egdc.mog_supervised_dataset import MogBenchmarkSupervisedDataset
from egdc.mog_direct_synth import synthesize_expression_program, DirectSynthResult
from egdc.mog_direct_router import solve_problem_direct, evaluate_direct_solver
from egdc.mog_pathways import PathwayMemory
from egdc.mog_adaptive_router import AdaptiveMogRouter

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
]
