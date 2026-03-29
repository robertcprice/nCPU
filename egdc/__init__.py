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
]
