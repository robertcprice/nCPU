"""KVRM-CPU: Model-Native CPU Emulator with Semantic Instruction Decode.

This package implements a CPU emulator where instruction decoding is handled by
a semantic micro-LLM that emits verified registry keys. This proves the KVRM
(Key-Value Response Mapping) paradigm at the lowest level of computing.

Core Thesis:
    Traditional CPU: fetch -> decode -> execute (hardcoded silicon)
    KVRM-CPU: fetch -> decode_llm -> key -> verified_execute (semantic understanding)

Architecture:
    MEMORY -> FETCH -> DECODE_LLM -> KEY -> REGISTRY -> EXECUTE -> STATE
               |          |           |        |           |
           [PC-based] [MicroLLM]   [JSON]  [Verified]  [Immutable]
                                  {"op":"OP_ADD",...}  Primitives  Audit Trail

Modules:
    state: CPUState dataclass for immutable state management
    registry: Verified CPU primitives (OP_ADD, OP_MOV, etc.)
    decode_llm: Semantic instruction decoder (mock + real modes)
    cpu: Main KVRMCPU orchestrator
"""

__version__ = "0.1.0"
__author__ = "KVRM Project"

from .state import CPUState
from .registry import CPURegistry
from .decode_llm import DecodeLLM
from .cpu import KVRMCPU

__all__ = ["CPUState", "CPURegistry", "DecodeLLM", "KVRMCPU"]
