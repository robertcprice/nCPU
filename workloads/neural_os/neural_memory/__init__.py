"""
KVRM Neural Memory System
=========================

Learned memory management that adapts to workload patterns.

Components:
- NeuralCache: Learned cache replacement policy
- NeuralPrefetcher: Predicts future memory accesses
- NeuralAllocator: Predictive memory allocation
"""

from .neural_cache import NeuralCache
from .neural_prefetcher import NeuralPrefetcher

__all__ = ['NeuralCache', 'NeuralPrefetcher']
