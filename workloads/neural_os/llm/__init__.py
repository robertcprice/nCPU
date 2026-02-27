"""
LLM MODULE - Intelligent Mutation with Ollama

This module provides LLM-powered code mutation capabilities:
- OllamaMutator: Async client for Ollama API
- ModelSelector: Complexity-based model selection
- FallbackMutator: AST-based mutations when LLM unavailable

The LLM is used to suggest intelligent code transformations,
but all suggestions are validated by the Judge before acceptance.
"""

from .ollama_mutator import OllamaMutator, OllamaConfig, MutationRequest, MutationResponse
from .model_selector import ModelSelector, ComplexityLevel
from .fallback_mutator import FallbackMutator

__all__ = [
    'OllamaMutator', 'OllamaConfig', 'MutationRequest', 'MutationResponse',
    'ModelSelector', 'ComplexityLevel',
    'FallbackMutator',
]
