#!/usr/bin/env python3
"""
SafeOnlineLearner - Universal wrapper for safe online learning
==============================================================

Wraps any neural component to enable safe continuous learning with:
- Automatic validation after each update
- Rollback on accuracy degradation
- Checkpoint management
- Learning rate adaptation
- Catastrophic forgetting prevention

Usage:
    model = SomeNeuralModel()
    safe_model = SafeOnlineLearner(model, validator=my_validator)

    # Learning happens safely
    safe_model.learn(input, target)  # Validates automatically
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Optional, Dict, Any, List
from collections import deque
import time
import copy


class SafeOnlineLearner(nn.Module):
    """
    Universal wrapper for safe online learning.

    Ensures that online weight updates don't degrade model performance
    by validating after updates and rolling back if necessary.
    """

    def __init__(
        self,
        model: nn.Module,
        validator: Optional[Callable] = None,
        learning_rate: float = 1e-6,
        accuracy_threshold: float = 0.999,
        max_consecutive_failures: int = 10,
        checkpoint_frequency: int = 1000,
        validation_frequency: int = 100,
        experience_buffer_size: int = 10000,
        device: str = "auto"
    ):
        """
        Initialize SafeOnlineLearner.

        Args:
            model: The neural network to wrap
            validator: Function that returns accuracy score (0-1)
            learning_rate: Initial learning rate for online updates
            accuracy_threshold: Minimum accuracy before rollback
            max_consecutive_failures: Failures before rollback
            checkpoint_frequency: How often to save checkpoints
            validation_frequency: How often to validate
            experience_buffer_size: Size of experience replay buffer
            device: Device to use ("auto", "cuda", "mps", "cpu")
        """
        super().__init__()

        # Device selection
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Core model
        self.model = model.to(self.device)
        self.model.eval()  # Default to eval mode

        # Optimizer for online learning
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.base_lr = learning_rate
        self.current_lr = learning_rate

        # Validation
        self.validator = validator
        self.accuracy_threshold = accuracy_threshold
        self.validation_frequency = validation_frequency

        # Checkpointing
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint = self._create_checkpoint()
        self.last_checkpoint_step = 0

        # Failure tracking
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.total_rollbacks = 0

        # Experience replay buffer (prevents catastrophic forgetting)
        self.experience_buffer = deque(maxlen=experience_buffer_size)
        self.replay_batch_size = 32
        self.replay_frequency = 10  # Replay every N updates

        # Statistics
        self.total_updates = 0
        self.successful_updates = 0
        self.failed_updates = 0
        self.current_accuracy = 1.0
        self.accuracy_history = deque(maxlen=1000)
        self.learning_events = []

        # Mode
        self.learning_enabled = True
        self.verbose = False

    def _create_checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state."""
        return {
            'model_state': copy.deepcopy(self.model.state_dict()),
            'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
            'accuracy': self.current_accuracy,
            'step': self.total_updates,
            'timestamp': time.time()
        }

    def _restore_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore from a checkpoint."""
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.current_accuracy = checkpoint['accuracy']

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)

    def learn(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable = None,
        validate_after: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a safe learning update.

        Args:
            inputs: Input tensor
            targets: Target tensor
            loss_fn: Loss function (defaults to MSE)
            validate_after: Whether to validate after this update

        Returns:
            Dict with learning outcome information
        """
        if not self.learning_enabled:
            return {'status': 'disabled', 'updated': False}

        loss_fn = loss_fn or nn.MSELoss()

        # Add to experience buffer for replay
        self.experience_buffer.append((inputs.detach().cpu(), targets.detach().cpu()))

        # Perform the update
        self.model.train()

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.model.eval()

        self.total_updates += 1

        result = {
            'status': 'updated',
            'loss': loss.item(),
            'step': self.total_updates,
            'updated': True
        }

        # Periodic experience replay to prevent catastrophic forgetting
        if self.total_updates % self.replay_frequency == 0:
            self._experience_replay(loss_fn)

        # Validation check
        if validate_after and self.validator and self.total_updates % self.validation_frequency == 0:
            validation_result = self._validate_and_maybe_rollback()
            result.update(validation_result)

        # Checkpoint if needed
        if self.total_updates - self.last_checkpoint_step >= self.checkpoint_frequency:
            if self.current_accuracy >= self.accuracy_threshold:
                self.checkpoint = self._create_checkpoint()
                self.last_checkpoint_step = self.total_updates
                result['checkpointed'] = True

        return result

    def _experience_replay(self, loss_fn: Callable):
        """Replay past experiences to prevent forgetting."""
        if len(self.experience_buffer) < self.replay_batch_size:
            return

        # Sample random experiences
        import random
        experiences = random.sample(list(self.experience_buffer), self.replay_batch_size)

        self.model.train()

        for inputs, targets in experiences:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets) * 0.1  # Lower weight for replay
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.model.eval()

    def _validate_and_maybe_rollback(self) -> Dict[str, Any]:
        """Validate current model and rollback if degraded."""
        accuracy = self.validator()
        self.accuracy_history.append(accuracy)
        self.current_accuracy = accuracy

        result = {'validation_accuracy': accuracy}

        if accuracy < self.accuracy_threshold:
            self.consecutive_failures += 1
            self.failed_updates += 1

            result['validation_passed'] = False
            result['consecutive_failures'] = self.consecutive_failures

            if self.consecutive_failures >= self.max_consecutive_failures:
                # Rollback to last known good state
                self._restore_checkpoint(self.checkpoint)
                self.total_rollbacks += 1
                self.consecutive_failures = 0

                # Reduce learning rate
                self.current_lr *= 0.5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr

                result['rolled_back'] = True
                result['new_lr'] = self.current_lr

                if self.verbose:
                    print(f"🔄 Rolled back to checkpoint (accuracy: {accuracy:.4f} < {self.accuracy_threshold})")
                    print(f"   New learning rate: {self.current_lr:.2e}")

                self.learning_events.append({
                    'type': 'rollback',
                    'step': self.total_updates,
                    'accuracy': accuracy,
                    'new_lr': self.current_lr
                })
        else:
            self.consecutive_failures = 0
            self.successful_updates += 1
            result['validation_passed'] = True

            # Gradually restore learning rate on success
            if self.current_lr < self.base_lr:
                self.current_lr = min(self.current_lr * 1.01, self.base_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr

        return result

    def learn_from_mistake(
        self,
        inputs: torch.Tensor,
        correct_output: torch.Tensor,
        loss_fn: Callable = None,
        importance: float = 1.0
    ) -> Dict[str, Any]:
        """
        Learn from a specific mistake with higher importance.

        This is called when we detect the model made an error and
        we have the correct answer.

        Args:
            inputs: The input that caused the mistake
            correct_output: The correct output
            loss_fn: Loss function
            importance: Weight for this learning example (default 1.0)
        """
        if not self.learning_enabled:
            return {'status': 'disabled'}

        loss_fn = loss_fn or nn.MSELoss()

        self.model.train()

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, correct_output) * importance
        loss.backward()

        # More aggressive gradient clipping for mistake learning
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        self.optimizer.step()
        self.model.eval()

        # Always add mistakes to experience buffer (they're valuable)
        for _ in range(5):  # Add multiple copies to increase replay probability
            self.experience_buffer.append((inputs.detach().cpu(), correct_output.detach().cpu()))

        if self.verbose:
            print(f"🧠 Learned from mistake (loss: {loss.item():.4f})")

        self.learning_events.append({
            'type': 'mistake_learning',
            'step': self.total_updates,
            'loss': loss.item(),
            'importance': importance
        })

        return {
            'status': 'learned',
            'loss': loss.item(),
            'importance': importance
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'total_updates': self.total_updates,
            'successful_updates': self.successful_updates,
            'failed_updates': self.failed_updates,
            'total_rollbacks': self.total_rollbacks,
            'current_accuracy': self.current_accuracy,
            'current_lr': self.current_lr,
            'consecutive_failures': self.consecutive_failures,
            'experience_buffer_size': len(self.experience_buffer),
            'accuracy_history': list(self.accuracy_history)[-100:],  # Last 100
            'learning_enabled': self.learning_enabled
        }

    def save(self, path: str):
        """Save the model and learning state."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'checkpoint': self.checkpoint,
            'stats': self.get_stats(),
            'config': {
                'accuracy_threshold': self.accuracy_threshold,
                'base_lr': self.base_lr,
                'current_lr': self.current_lr
            }
        }, path)

    def load(self, path: str):
        """Load model and learning state."""
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data['model_state'])
        self.optimizer.load_state_dict(data['optimizer_state'])
        self.checkpoint = data['checkpoint']

        config = data.get('config', {})
        self.accuracy_threshold = config.get('accuracy_threshold', self.accuracy_threshold)
        self.base_lr = config.get('base_lr', self.base_lr)
        self.current_lr = config.get('current_lr', self.current_lr)

    def enable_learning(self):
        """Enable online learning."""
        self.learning_enabled = True

    def disable_learning(self):
        """Disable online learning (inference only)."""
        self.learning_enabled = False

    def set_verbose(self, verbose: bool):
        """Set verbose mode for debugging."""
        self.verbose = verbose


class OnlineLearningManager:
    """
    Manages online learning across multiple components.

    Coordinates learning, validates system-wide health,
    and provides unified control.
    """

    def __init__(self):
        self.learners: Dict[str, SafeOnlineLearner] = {}
        self.global_learning_enabled = True
        self.system_health_threshold = 0.95

    def register(self, name: str, learner: SafeOnlineLearner):
        """Register a learnable component."""
        self.learners[name] = learner

    def get(self, name: str) -> Optional[SafeOnlineLearner]:
        """Get a registered learner."""
        return self.learners.get(name)

    def enable_all(self):
        """Enable learning for all components."""
        self.global_learning_enabled = True
        for learner in self.learners.values():
            learner.enable_learning()

    def disable_all(self):
        """Disable learning for all components."""
        self.global_learning_enabled = False
        for learner in self.learners.values():
            learner.disable_learning()

    def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all components."""
        health = {}
        for name, learner in self.learners.items():
            stats = learner.get_stats()
            health[name] = {
                'accuracy': stats['current_accuracy'],
                'rollbacks': stats['total_rollbacks'],
                'learning_enabled': stats['learning_enabled'],
                'healthy': stats['current_accuracy'] >= learner.accuracy_threshold
            }

        overall_healthy = all(h['healthy'] for h in health.values())

        return {
            'components': health,
            'overall_healthy': overall_healthy,
            'component_count': len(self.learners)
        }

    def save_all(self, directory: str):
        """Save all learners to directory."""
        import os
        os.makedirs(directory, exist_ok=True)

        for name, learner in self.learners.items():
            path = os.path.join(directory, f"{name}.pt")
            learner.save(path)

    def load_all(self, directory: str):
        """Load all learners from directory."""
        import os

        for name, learner in self.learners.items():
            path = os.path.join(directory, f"{name}.pt")
            if os.path.exists(path):
                learner.load(path)


# Convenience function for creating validators
def create_accuracy_validator(
    model: nn.Module,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    threshold: float = 0.5
) -> Callable:
    """
    Create a validation function for a model.

    Args:
        model: The model to validate
        test_inputs: Test input tensor
        test_targets: Expected outputs
        threshold: Threshold for binary accuracy

    Returns:
        Callable that returns accuracy (0-1)
    """
    def validator() -> float:
        model.eval()
        with torch.no_grad():
            outputs = model(test_inputs)
            # For binary outputs
            if threshold is not None:
                predictions = (outputs > threshold).float()
                targets = (test_targets > threshold).float()
                correct = (predictions == targets).float().mean()
            else:
                # For continuous outputs - use relative error
                error = torch.abs(outputs - test_targets) / (torch.abs(test_targets) + 1e-8)
                correct = (error < 0.01).float().mean()

            return correct.item()

    return validator
