"""
Gradient-Guided Search for Autoresearch

Uses gradient signals to guide experiment search instead of random exploration.
Learns from past experiments to find promising modification directions.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
import numpy as np
import time
from collections import defaultdict


class SearchDirection(Enum):
    """Directions to search"""
    INCREASE = "increase"
    DECREASE = "decrease"
    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"


@dataclass
class SearchPattern:
    """A learned search pattern"""
    pattern_id: str
    modification_type: str
    direction: SearchDirection
    magnitude: float
    success_rate: float = 0.0
    count: int = 0


@dataclass
class SearchSuggestion:
    """A suggested modification"""
    modification_type: str
    target: str  # e.g., "n_layer", "lr", "n_head"
    direction: SearchDirection
    magnitude: float
    reason: str
    expected_improvement: float


class GradientGuidedSearch:
    """
    Uses gradient signals to guide experiment search.

    Instead of random modifications, uses learned patterns
    from previous experiments to find promising directions.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.2,
    ):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate

        # Learned patterns
        self.patterns: dict[str, SearchPattern] = {}

        # Experiment history
        self.history: list[dict] = []

        # Best results tracking
        self.best_val_bpb: Optional[float] = None
        self.best_modification: Optional[dict] = None

        # Statistics
        self.total_experiments = 0
        self.successful_experiments = 0

    def get_suggestion(self) -> SearchSuggestion:
        """
        Get the next modification suggestion based on learned patterns.

        Combines:
        - Exploitation: Use successful patterns
        - Exploration: Try new directions
        """
        # Decide: exploit or explore
        if np.random.random() < self.exploration_rate:
            return self._explore()
        else:
            return self._exploit()

    def learn_from_result(
        self,
        modification: dict,
        val_bpb: float,
        status: str,
    ):
        """
        Learn from an experiment result.

        Updates patterns based on success/failure.
        """
        self.total_experiments += 1

        if status == "keep" and val_bpb < 3.0:  # Sanity check
            self.successful_experiments += 1
            is_success = True
        else:
            is_success = False

        # Update best
        if self.best_val_bpb is None or val_bpb < self.best_val_bpb:
            self.best_val_bpb = val_bpb
            self.best_modification = modification

        # Extract pattern from modification
        pattern_id = self._extract_pattern_id(modification)

        if pattern_id not in self.patterns:
            # Map ImprovementDirection values to SearchDirection
            direction_map = {
                "positive": "increase",
                "negative": "decrease",
                "neutral": "modify",
                "refinement": "modify",
            }
            direction_str = modification.get("direction", "modify").lower()
            search_direction = direction_map.get(direction_str, direction_str)
            self.patterns[pattern_id] = SearchPattern(
                pattern_id=pattern_id,
                modification_type=modification.get("type", "unknown"),
                direction=SearchDirection(search_direction),
                magnitude=modification.get("magnitude", 1.0),
            )

        # Update pattern statistics
        pattern = self.patterns[pattern_id]
        pattern.count += 1

        # Update success rate (running average)
        if is_success:
            pattern.success_rate = (
                pattern.success_rate * (pattern.count - 1) + 1.0
            ) / pattern.count
        else:
            pattern.success_rate = (
                pattern.success_rate * (pattern.count - 1) + 0.0
            ) / pattern.count

        # Store in history
        self.history.append({
            "modification": modification,
            "val_bpb": val_bpb,
            "status": status,
            "is_success": is_success,
            "pattern_id": pattern_id,
        })

    def get_gradient_signal(self) -> np.ndarray:
        """
        Get aggregated gradient signal for model improvement.

        Returns a vector representing learned search patterns.
        """
        if not self.patterns:
            return np.zeros(10, dtype=np.float32)

        # Build gradient from patterns
        gradient = []

        # Success rate features
        rates = [p.success_rate for p in self.patterns.values()]
        gradient.append(np.mean(rates) if rates else 0.0)
        gradient.append(np.max(rates) if rates else 0.0)

        # Count features
        counts = [p.count for p in self.patterns.values()]
        gradient.append(np.sum(counts))
        gradient.append(np.max(counts))

        # Recent success (last 5 experiments)
        recent = self.history[-5:] if self.history else []
        recent_success = sum(1 for h in recent if h.get("is_success")) / max(len(recent), 1)
        gradient.append(recent_success)

        # Direction features
        increases = sum(1 for p in self.patterns.values() if p.direction == SearchDirection.INCREASE)
        decreases = sum(1 for p in self.patterns.values() if p.direction == SearchDirection.DECREASE)
        gradient.append(increases / max(len(self.patterns), 1))
        gradient.append(decreases / max(len(self.patterns), 1))

        # Best improvement
        if self.best_val_bpb:
            gradient.append(2.5 - self.best_val_bpb)  # Normalized
        else:
            gradient.append(0.0)

        # Exploration vs exploitation
        success_rate = self.successful_experiments / max(self.total_experiments, 1)
        gradient.append(success_rate)

        # Pad to fixed size
        while len(gradient) < 10:
            gradient.append(0.0)

        return np.array(gradient[:10], dtype=np.float32)

    def _exploit(self) -> SearchSuggestion:
        """Exploit known successful patterns"""
        # Find patterns with high success rate
        successful_patterns = [
            p for p in self.patterns.values()
            if p.success_rate >= 0.5
        ]

        if successful_patterns:
            # Use the best one
            best = max(successful_patterns, key=lambda p: p.success_rate)

            return SearchSuggestion(
                modification_type=best.modification_type,
                target=self._infer_target(best.modification_type),
                direction=best.direction,
                magnitude=best.magnitude * (1 + self.learning_rate),
                reason=f"Exploiting pattern {best.pattern_id} with {best.success_rate:.1%} success rate",
                expected_improvement=best.success_rate * 0.1,
            )

        # Fall back to default
        return self._default_suggestion()

    def _explore(self) -> SearchSuggestion:
        """Explore new directions"""
        # Try different modification types
        modification_types = [
            ("layer", "n_layer", SearchDirection.INCREASE),
            ("head", "n_head", SearchDirection.INCREASE),
            ("embedding", "n_embd", SearchDirection.INCREASE),
            ("learning_rate", "lr", SearchDirection.MODIFY),
            ("checkpoint", "use_checkpointing", SearchDirection.ADD),
            ("window", "window_pattern", SearchDirection.MODIFY),
        ]

        # Pick random
        mod_type, target, direction = modification_types[
            int(np.random.random() * len(modification_types))
        ]

        return SearchSuggestion(
            modification_type=mod_type,
            target=target,
            direction=direction,
            magnitude=np.random.uniform(0.5, 2.0),
            reason=f"Exploring new direction: {mod_type}",
            expected_improvement=0.0,  # Unknown
        )

    def _default_suggestion(self) -> SearchSuggestion:
        """Default suggestion when no patterns learned"""
        suggestions = [
            ("layer", "n_layer", SearchDirection.INCREASE, 1),
            ("head", "n_head", SearchDirection.INCREASE, 2),
            ("embedding", "n_embd", SearchDirection.INCREASE, 128),
        ]

        mod_type, target, direction, magnitude = suggestions[
            self.total_experiments % len(suggestions)
        ]

        return SearchSuggestion(
            modification_type=mod_type,
            target=target,
            direction=direction,
            magnitude=magnitude,
            reason="Default suggestion based on common improvements",
            expected_improvement=0.05,
        )

    def _extract_pattern_id(self, modification: dict) -> str:
        """Extract pattern ID from modification"""
        mod_type = modification.get("type", "unknown")
        direction = modification.get("direction", "modify")
        return f"{mod_type}_{direction}"

    def _infer_target(self, modification_type: str) -> str:
        """Infer target parameter from modification type"""
        mapping = {
            "layer": "n_layer",
            "head": "n_head",
            "embedding": "n_embd",
            "learning_rate": "lr",
            "checkpoint": "use_checkpointing",
            "window": "window_pattern",
        }
        return mapping.get(modification_type, "unknown")

    def get_statistics(self) -> dict:
        """Get search statistics"""
        return {
            "total_experiments": self.total_experiments,
            "successful_experiments": self.successful_experiments,
            "success_rate": self.successful_experiments / max(self.total_experiments, 1),
            "patterns_learned": len(self.patterns),
            "best_val_bpb": self.best_val_bpb,
            "patterns": [
                {
                    "id": p.pattern_id,
                    "success_rate": p.success_rate,
                    "count": p.count,
                }
                for p in self.patterns.values()
            ]
        }


class MultiObjectiveSearch(GradientGuidedSearch):
    """
    Gradient-guided search with multiple objectives.

    Optimizes for:
    - val_bpb (main metric)
    - Training speed
    - Memory efficiency
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Weights for objectives
        self.weights = {
            "val_bpb": 1.0,
            "speed": 0.3,
            "memory": 0.2,
        }

    def learn_from_result(
        self,
        modification: dict,
        val_bpb: float,
        training_seconds: float,
        peak_vram_mb: float,
        status: str,
    ):
        """Learn from multi-objective results"""
        # Create composite score
        # Lower val_bpb is better, lower time is better, lower memory is better

        # Normalize
        score = (
            self.weights["val_bpb"] * val_bpb +
            self.weights["speed"] * (training_seconds / 300) +
            self.weights["memory"] * (peak_vram_mb / 8000)
        )

        # Determine success
        is_success = status == "keep" and val_bpb < 3.0

        # Store extra metrics
        result = modification.copy()
        result["val_bpb"] = val_bpb
        result["training_seconds"] = training_seconds
        result["peak_vram_mb"] = peak_vram_mb
        result["composite_score"] = score

        super().learn_from_result(result, val_bpb, status)


def demo():
    """Demo of gradient-guided search"""
    print("=== Gradient-Guided Search Demo ===\n")

    # Create search
    search = GradientGuidedSearch()

    # Simulate learning
    print("Simulating experiments...")

    experiments = [
        {"type": "layer", "direction": "increase", "magnitude": 1},
        {"type": "head", "direction": "increase", "magnitude": 2},
        {"type": "layer", "direction": "increase", "magnitude": 1},  # Success!
        {"type": "embedding", "direction": "increase", "magnitude": 128},
        {"type": "embedding", "direction": "increase", "magnitude": 128},  # Success!
    ]

    val_bpbs = [2.1, 2.0, 1.95, 2.05, 1.98]

    for i, (mod, vbp) in enumerate(zip(experiments, val_bpbs)):
        status = "keep" if vbp < 2.05 else "discard"
        search.learn_from_result(mod, vbp, status)
        print(f"  Experiment {i+1}: val_bpb={vbp:.3f}, status={status}")

    # Get suggestions
    print("\nGetting suggestions...")

    for _ in range(3):
        suggestion = search.get_suggestion()
        print(f"  {suggestion.modification_type}: {suggestion.target}")
        print(f"    Reason: {suggestion.reason}")

    # Gradient signal
    gradient = search.get_gradient_signal()
    print(f"\nGradient signal: {gradient}")

    # Statistics
    stats = search.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total experiments: {stats['total_experiments']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Best val_bpb: {stats['best_val_bpb']}")
    print(f"  Patterns learned: {stats['patterns_learned']}")


if __name__ == "__main__":
    demo()
