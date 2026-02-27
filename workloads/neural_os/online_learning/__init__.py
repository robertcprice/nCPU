"""
KVRM Online Learning Framework
==============================

Provides safe, continuous learning capabilities for all neural components.

Components:
- SafeOnlineLearner: Universal wrapper for safe weight updates
- CheckpointManager: State management and rollback
- ValidationFramework: Continuous accuracy monitoring
"""

from .safe_learner import SafeOnlineLearner
from .checkpoint_manager import CheckpointManager
from .validation_framework import ValidationFramework

__all__ = ['SafeOnlineLearner', 'CheckpointManager', 'ValidationFramework']
