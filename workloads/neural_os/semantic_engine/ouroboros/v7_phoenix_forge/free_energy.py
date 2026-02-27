"""
Free Energy Minimization (V7 Core)
====================================
Active Inference engine: F = Surprise + Complexity

Agents minimize free energy by:
1. LEARNING: Update beliefs to reduce surprise
2. ACTION: Modify code to match expectations

This is how biological brains work - and how V7 agents create.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime

from .world_model import HierarchicalWorldModel, SurpriseResult


@dataclass
class FreeEnergyResult:
    """Result of free energy computation."""
    total_F: float
    surprise: float
    complexity: float
    accuracy: float  # How well code achieves goal
    details: Dict[str, Any]


@dataclass
class ActionCandidate:
    """A candidate action to reduce free energy."""
    action_type: str  # "learn" or "act"
    description: str
    expected_F_reduction: float
    code_change: Optional[str] = None
    belief_change: Optional[Dict[str, Any]] = None


class FreeEnergyMinimizer:
    """
    Free Energy Minimization engine.

    F = Surprise + Complexity - Accuracy

    Agents reduce F by:
    - Reducing surprise (making code more predictable)
    - Reducing complexity (simpler solutions)
    - Increasing accuracy (better goal achievement)
    """

    # Weights for free energy components
    SURPRISE_WEIGHT = 0.4
    COMPLEXITY_WEIGHT = 0.3
    ACCURACY_WEIGHT = 0.3

    def __init__(self, world_model: HierarchicalWorldModel):
        self.world_model = world_model
        self._history: List[FreeEnergyResult] = []

    def compute_free_energy(
        self,
        code: str,
        goal: str,
        test_fn: Optional[Callable[[str], float]] = None
    ) -> FreeEnergyResult:
        """
        Compute free energy for a piece of code.

        F = w1 * Surprise + w2 * Complexity - w3 * Accuracy
        """
        # 1. Compute surprise using world model
        surprise_result = self.world_model.compute_surprise(code)
        surprise = surprise_result.total

        # 2. Compute complexity
        complexity = self._compute_complexity(code)

        # 3. Compute accuracy (goal achievement)
        if test_fn:
            accuracy = test_fn(code)
        else:
            accuracy = self._estimate_accuracy(code, goal)

        # 4. Compute free energy
        total_F = (
            self.SURPRISE_WEIGHT * surprise +
            self.COMPLEXITY_WEIGHT * complexity -
            self.ACCURACY_WEIGHT * accuracy
        )

        result = FreeEnergyResult(
            total_F=total_F,
            surprise=surprise,
            complexity=complexity,
            accuracy=accuracy,
            details={
                "surprise_by_level": surprise_result.by_level,
                "algorithm_patterns": surprise_result.details.get("algorithm_patterns", []),
            }
        )

        self._history.append(result)

        return result

    def _compute_complexity(self, code: str) -> float:
        """
        Compute code complexity.

        Based on:
        - Lines of code
        - Nesting depth
        - Number of branches
        - Cyclomatic complexity approximation
        """
        lines = [l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
        loc = len(lines)

        # Nesting depth
        max_depth = 0
        current_depth = 0
        for char in code:
            if char in '({[':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ')}]':
                current_depth = max(0, current_depth - 1)

        # Branch count
        branches = code.count('if ') + code.count('elif ') + code.count('else:')
        branches += code.count('for ') + code.count('while ')
        branches += code.count('try:') + code.count('except')

        # Normalize to 0-1
        complexity = (
            min(loc / 50, 1.0) * 0.3 +        # LOC contribution
            min(max_depth / 5, 1.0) * 0.3 +   # Nesting contribution
            min(branches / 10, 1.0) * 0.4     # Branch contribution
        )

        return complexity

    def _estimate_accuracy(self, code: str, goal: str) -> float:
        """Estimate how well code achieves goal (without running tests)."""
        accuracy = 0.5  # Base

        # Check for return statement
        if 'return' in code:
            accuracy += 0.1

        # Check for error handling
        if 'try' in code or 'if ' in code:
            accuracy += 0.1

        # Check if goal keywords appear in code
        goal_words = set(goal.lower().split())
        code_words = set(code.lower().split())
        overlap = len(goal_words & code_words)
        accuracy += min(overlap / max(len(goal_words), 1), 0.2)

        return min(1.0, accuracy)

    def propose_actions(
        self,
        code: str,
        goal: str,
        current_F: FreeEnergyResult
    ) -> List[ActionCandidate]:
        """
        Propose actions to reduce free energy.

        Two strategies:
        1. LEARN: Update world model beliefs
        2. ACT: Modify the code
        """
        actions = []

        # Strategy 1: Learning actions (reduce surprise)
        if current_F.surprise > 0.3:
            actions.append(ActionCandidate(
                action_type="learn",
                description="Update world model to better predict this code pattern",
                expected_F_reduction=current_F.surprise * 0.1,
                belief_change={"update_priors": True},
            ))

        # Strategy 2: Simplification (reduce complexity)
        if current_F.complexity > 0.5:
            actions.append(ActionCandidate(
                action_type="act",
                description="Simplify code structure",
                expected_F_reduction=current_F.complexity * 0.2,
                code_change="simplify",
            ))

        # Strategy 3: Goal alignment (increase accuracy)
        if current_F.accuracy < 0.7:
            actions.append(ActionCandidate(
                action_type="act",
                description="Improve goal alignment",
                expected_F_reduction=(1 - current_F.accuracy) * 0.3,
                code_change="improve_accuracy",
            ))

        # Strategy 4: Pattern application
        if not current_F.details.get("algorithm_patterns"):
            actions.append(ActionCandidate(
                action_type="act",
                description="Apply a known algorithm pattern",
                expected_F_reduction=0.15,
                code_change="apply_pattern",
            ))

        # Sort by expected reduction
        actions.sort(key=lambda a: a.expected_F_reduction, reverse=True)

        return actions

    def minimize_step(
        self,
        code: str,
        goal: str,
        test_fn: Optional[Callable[[str], float]] = None
    ) -> Tuple[str, FreeEnergyResult]:
        """
        Take one step to minimize free energy.

        Returns: (new_code, new_F)
        """
        # Compute current F
        current_F = self.compute_free_energy(code, goal, test_fn)

        # Get action proposals
        actions = self.propose_actions(code, goal, current_F)

        if not actions:
            return code, current_F

        # Try the best action
        best_action = actions[0]

        if best_action.action_type == "learn":
            # Update world model
            self.world_model.update_beliefs(code, {"goal": goal})
            # Recompute F with updated beliefs
            new_F = self.compute_free_energy(code, goal, test_fn)
            return code, new_F

        else:  # action_type == "act"
            # Apply code transformation
            new_code = self._apply_transformation(code, best_action.code_change, goal)
            new_F = self.compute_free_energy(new_code, goal, test_fn)

            # Only accept if F decreased
            if new_F.total_F < current_F.total_F:
                return new_code, new_F
            else:
                return code, current_F

    def _apply_transformation(self, code: str, transform_type: str, goal: str) -> str:
        """Apply a code transformation."""
        if transform_type == "simplify":
            # Remove unnecessary complexity
            lines = code.split('\n')
            # Remove empty lines and redundant passes
            lines = [l for l in lines if l.strip() and l.strip() != 'pass']
            return '\n'.join(lines)

        elif transform_type == "improve_accuracy":
            # Add return statement if missing
            if 'return' not in code:
                lines = code.split('\n')
                # Find last line of function body
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() and not lines[i].strip().startswith('#'):
                        indent = len(lines[i]) - len(lines[i].lstrip())
                        lines.insert(i + 1, ' ' * indent + 'return result')
                        break
                return '\n'.join(lines)
            return code

        elif transform_type == "apply_pattern":
            # This would be more sophisticated in production
            return code

        return code

    def get_optimization_history(self) -> List[Dict[str, float]]:
        """Get history of F values."""
        return [
            {
                "total_F": r.total_F,
                "surprise": r.surprise,
                "complexity": r.complexity,
                "accuracy": r.accuracy,
            }
            for r in self._history
        ]
