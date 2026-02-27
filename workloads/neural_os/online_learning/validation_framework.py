#!/usr/bin/env python3
"""
ValidationFramework - Continuous accuracy monitoring for online learning
=========================================================================

Provides comprehensive validation with:
- Multiple validation strategies
- Statistical significance testing
- Trend detection
- Alert generation
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, Any, List, Optional, Tuple
from collections import deque
import time
import math


class ValidationFramework:
    """
    Comprehensive validation framework for online learning.

    Monitors model accuracy continuously and provides alerts
    when accuracy degrades or trends downward.
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.999,
        window_size: int = 100,
        trend_sensitivity: float = 0.01,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize ValidationFramework.

        Args:
            accuracy_threshold: Minimum acceptable accuracy
            window_size: Window size for trend detection
            trend_sensitivity: Sensitivity for trend detection
            alert_callback: Function to call when alert triggered
        """
        self.accuracy_threshold = accuracy_threshold
        self.window_size = window_size
        self.trend_sensitivity = trend_sensitivity
        self.alert_callback = alert_callback

        # History tracking
        self.accuracy_history = deque(maxlen=window_size * 2)
        self.validation_times = deque(maxlen=window_size)
        self.alerts = []

        # Statistics
        self.total_validations = 0
        self.passed_validations = 0
        self.failed_validations = 0

    def validate(
        self,
        model: nn.Module,
        test_inputs: torch.Tensor,
        test_targets: torch.Tensor,
        loss_fn: Callable = None,
        accuracy_fn: Callable = None
    ) -> Dict[str, Any]:
        """
        Perform validation on a model.

        Args:
            model: Model to validate
            test_inputs: Test inputs
            test_targets: Expected outputs
            loss_fn: Loss function (optional)
            accuracy_fn: Custom accuracy function (optional)

        Returns:
            Validation results dict
        """
        start_time = time.time()
        model.eval()

        with torch.no_grad():
            outputs = model(test_inputs)

            # Compute loss
            if loss_fn:
                loss = loss_fn(outputs, test_targets).item()
            else:
                loss = nn.MSELoss()(outputs, test_targets).item()

            # Compute accuracy
            if accuracy_fn:
                accuracy = accuracy_fn(outputs, test_targets)
            else:
                accuracy = self._default_accuracy(outputs, test_targets)

        validation_time = time.time() - start_time
        self.validation_times.append(validation_time)

        # Record history
        self.accuracy_history.append(accuracy)
        self.total_validations += 1

        # Check threshold
        passed = accuracy >= self.accuracy_threshold
        if passed:
            self.passed_validations += 1
        else:
            self.failed_validations += 1

        # Compute trend
        trend = self._compute_trend()

        # Generate result
        result = {
            'accuracy': accuracy,
            'loss': loss,
            'passed': passed,
            'threshold': self.accuracy_threshold,
            'trend': trend,
            'validation_time': validation_time,
            'total_validations': self.total_validations
        }

        # Check for alerts
        alert = self._check_alerts(accuracy, trend)
        if alert:
            result['alert'] = alert
            self.alerts.append(alert)
            if self.alert_callback:
                self.alert_callback(alert)

        return result

    def _default_accuracy(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Default accuracy computation (bit-wise for binary outputs)."""
        predictions = (outputs > 0.5).float()
        expected = (targets > 0.5).float()
        correct = (predictions == expected).float().mean()
        return correct.item()

    def _compute_trend(self) -> Dict[str, Any]:
        """Compute accuracy trend over time."""
        if len(self.accuracy_history) < self.window_size:
            return {'direction': 'unknown', 'slope': 0.0, 'confidence': 0.0}

        # Get recent window
        recent = list(self.accuracy_history)[-self.window_size:]

        # Compute linear regression
        n = len(recent)
        x = list(range(n))
        y = recent

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Compute R-squared for confidence
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
        if ss_tot == 0:
            r_squared = 0
        else:
            y_pred = [y_mean + slope * (x[i] - x_mean) for i in range(n)]
            ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
            r_squared = 1 - (ss_res / ss_tot)

        # Determine direction
        if abs(slope) < self.trend_sensitivity:
            direction = 'stable'
        elif slope > 0:
            direction = 'improving'
        else:
            direction = 'degrading'

        return {
            'direction': direction,
            'slope': slope,
            'confidence': abs(r_squared),
            'current_mean': y_mean,
            'window_size': n
        }

    def _check_alerts(
        self,
        accuracy: float,
        trend: Dict
    ) -> Optional[Dict[str, Any]]:
        """Check if any alert conditions are met."""
        alerts = []

        # Below threshold
        if accuracy < self.accuracy_threshold:
            alerts.append({
                'type': 'threshold_violation',
                'severity': 'high',
                'message': f'Accuracy {accuracy:.4f} below threshold {self.accuracy_threshold}'
            })

        # Degrading trend with high confidence
        if trend['direction'] == 'degrading' and trend['confidence'] > 0.7:
            alerts.append({
                'type': 'degrading_trend',
                'severity': 'medium',
                'message': f'Accuracy trending down (slope: {trend["slope"]:.6f})'
            })

        # Sudden drop
        if len(self.accuracy_history) >= 2:
            prev = self.accuracy_history[-2]
            if accuracy < prev - 0.05:  # 5% drop
                alerts.append({
                    'type': 'sudden_drop',
                    'severity': 'high',
                    'message': f'Sudden accuracy drop: {prev:.4f} → {accuracy:.4f}'
                })

        if alerts:
            return {
                'timestamp': time.time(),
                'accuracy': accuracy,
                'alerts': alerts
            }

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validations': self.total_validations,
            'passed': self.passed_validations,
            'failed': self.failed_validations,
            'pass_rate': self.passed_validations / max(1, self.total_validations),
            'current_accuracy': self.accuracy_history[-1] if self.accuracy_history else None,
            'mean_accuracy': sum(self.accuracy_history) / max(1, len(self.accuracy_history)),
            'min_accuracy': min(self.accuracy_history) if self.accuracy_history else None,
            'max_accuracy': max(self.accuracy_history) if self.accuracy_history else None,
            'alert_count': len(self.alerts),
            'avg_validation_time': sum(self.validation_times) / max(1, len(self.validation_times))
        }

    def reset(self):
        """Reset validation statistics."""
        self.accuracy_history.clear()
        self.validation_times.clear()
        self.alerts.clear()
        self.total_validations = 0
        self.passed_validations = 0
        self.failed_validations = 0


class CompositeValidator:
    """
    Validates multiple aspects of a model.

    Combines multiple validation functions into a single validator.
    """

    def __init__(self):
        self.validators: Dict[str, Callable] = {}
        self.weights: Dict[str, float] = {}

    def add_validator(
        self,
        name: str,
        validator: Callable,
        weight: float = 1.0
    ):
        """Add a validation function."""
        self.validators[name] = validator
        self.weights[name] = weight

    def validate(self) -> Dict[str, Any]:
        """Run all validators and return combined result."""
        results = {}
        weighted_sum = 0
        total_weight = 0

        for name, validator in self.validators.items():
            try:
                score = validator()
                results[name] = {
                    'score': score,
                    'weight': self.weights[name],
                    'passed': score >= 0.99  # Default threshold
                }
                weighted_sum += score * self.weights[name]
                total_weight += self.weights[name]
            except Exception as e:
                results[name] = {
                    'score': 0,
                    'error': str(e),
                    'passed': False
                }

        overall = weighted_sum / max(total_weight, 1)

        return {
            'overall_score': overall,
            'overall_passed': all(r.get('passed', False) for r in results.values()),
            'component_results': results
        }
