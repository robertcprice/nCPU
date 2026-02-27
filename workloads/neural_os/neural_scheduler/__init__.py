"""
KVRM Neural Scheduler
=====================

Learned process scheduling using reinforcement learning.

Components:
- NeuralScheduler: RL-based process selection
- WorkloadPredictor: Predicts future resource needs
- PriorityLearner: Learns implicit priorities from behavior
"""

from .neural_scheduler import NeuralScheduler

__all__ = ['NeuralScheduler']
