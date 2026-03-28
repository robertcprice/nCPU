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
]
