"""
Hierarchical World Model (V7)
==============================
Multi-level expectations about what good code looks like.

Used for Active Inference - agents minimize surprise relative
to their world model.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import ast
import re


@dataclass
class Prediction:
    """A prediction at a specific level."""
    level: str
    expected: Any
    confidence: float


@dataclass
class SurpriseResult:
    """Result of surprise computation."""
    total: float
    by_level: Dict[str, float]
    details: Dict[str, Any]


class TokenLanguageModel:
    """Syntax-level expectations."""

    def __init__(self):
        # Expected token frequencies
        self.token_probs: Dict[str, float] = {
            "def": 0.05,
            "return": 0.04,
            "if": 0.06,
            "for": 0.05,
            "while": 0.02,
            "class": 0.02,
            "import": 0.03,
            "from": 0.02,
            "=": 0.08,
            "(": 0.10,
            ")": 0.10,
            ":": 0.06,
        }

    def predict(self, code: str) -> Dict[str, float]:
        """Predict token distribution."""
        return self.token_probs.copy()

    def compute_surprise(self, code: str) -> float:
        """Compute surprise at token level."""
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
        if not tokens:
            return 0.0

        # Count actual tokens
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        total = len(tokens)
        surprise = 0.0

        for token, count in counts.items():
            actual_freq = count / total
            expected_freq = self.token_probs.get(token, 0.01)

            # Surprise from deviation
            surprise += abs(actual_freq - expected_freq)

        return min(1.0, surprise)


class ASTStructureModel:
    """Structure-level expectations."""

    def __init__(self):
        # Expected AST patterns
        self.expected_ratios = {
            "functions_to_classes": 3.0,
            "loops_to_conditionals": 0.5,
            "returns_to_functions": 1.0,
            "max_nesting": 4,
        }

    def predict(self, code: str) -> Dict[str, Any]:
        """Predict AST structure."""
        return self.expected_ratios.copy()

    def compute_surprise(self, code: str) -> float:
        """Compute surprise at structure level."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.5

        # Count structures
        functions = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
        classes = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))
        loops = sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.For, ast.While)))
        conditionals = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.If))
        returns = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.Return))

        surprise = 0.0

        # Check ratios
        if classes > 0:
            actual_ratio = functions / classes
            expected = self.expected_ratios["functions_to_classes"]
            surprise += abs(actual_ratio - expected) / expected * 0.2

        if conditionals > 0:
            actual_ratio = loops / conditionals
            expected = self.expected_ratios["loops_to_conditionals"]
            surprise += abs(actual_ratio - expected) / expected * 0.2

        if functions > 0:
            actual_ratio = returns / functions
            expected = self.expected_ratios["returns_to_functions"]
            surprise += abs(actual_ratio - expected) / expected * 0.2

        return min(1.0, surprise)


class SemanticTraceModel:
    """Behavior-level expectations."""

    def __init__(self):
        # Expected behavior patterns
        self.expected_patterns = {
            "has_input_validation": True,
            "has_error_handling": True,
            "has_return_statement": True,
            "pure_function": True,
        }

    def predict(self, code: str) -> Dict[str, bool]:
        """Predict semantic properties."""
        return self.expected_patterns.copy()

    def compute_surprise(self, code: str) -> float:
        """Compute surprise at semantic level."""
        surprise = 0.0

        # Check for expected patterns
        if "if " not in code and "assert" not in code:
            surprise += 0.2  # No input validation

        if "try" not in code and "except" not in code:
            surprise += 0.1  # No error handling

        if "return" not in code:
            surprise += 0.3  # No return statement

        # Check for side effects (impure)
        if "global " in code or "print(" in code:
            surprise += 0.2  # Side effects

        return min(1.0, surprise)


class AlgorithmPatternModel:
    """Algorithm-level expectations."""

    def __init__(self):
        # Known algorithm patterns
        self.patterns = {
            "divide_conquer": r"def\s+\w+\(.*\).*\1\s*\(",
            "dynamic_programming": r"memo|cache|dp\[",
            "greedy": r"max\(|min\(|sorted\(",
            "recursion": r"def\s+(\w+).*\1\s*\(",
            "iteration": r"for\s+|while\s+",
        }

    def detect_patterns(self, code: str) -> List[str]:
        """Detect algorithm patterns in code."""
        detected = []
        for name, pattern in self.patterns.items():
            if re.search(pattern, code, re.DOTALL):
                detected.append(name)
        return detected

    def compute_surprise(self, code: str) -> float:
        """Compute surprise at algorithm level."""
        patterns = self.detect_patterns(code)

        # More patterns = less surprise (more structured)
        if len(patterns) >= 2:
            return 0.1
        elif len(patterns) == 1:
            return 0.3
        else:
            return 0.5  # No clear pattern


class HierarchicalWorldModel:
    """
    Multi-level world model for Active Inference.

    Combines expectations at different abstraction levels:
    - Token (syntax)
    - AST (structure)
    - Semantic (behavior)
    - Algorithm (patterns)
    """

    def __init__(self):
        self.token_prior = TokenLanguageModel()
        self.ast_prior = ASTStructureModel()
        self.semantic_prior = SemanticTraceModel()
        self.algorithm_prior = AlgorithmPatternModel()

        # Weights for each level
        self.level_weights = {
            "token": 0.1,
            "ast": 0.2,
            "semantic": 0.3,
            "algorithm": 0.4,
        }

        # Learning history
        self._observations: List[Dict[str, Any]] = []

    def predict(self, code: str) -> Dict[str, Prediction]:
        """Generate predictions at all levels."""
        return {
            "token": Prediction(
                level="token",
                expected=self.token_prior.predict(code),
                confidence=0.7,
            ),
            "ast": Prediction(
                level="ast",
                expected=self.ast_prior.predict(code),
                confidence=0.8,
            ),
            "semantic": Prediction(
                level="semantic",
                expected=self.semantic_prior.predict(code),
                confidence=0.6,
            ),
            "algorithm": Prediction(
                level="algorithm",
                expected=self.algorithm_prior.detect_patterns(code),
                confidence=0.5,
            ),
        }

    def compute_surprise(self, code: str, observations: Optional[Dict[str, Any]] = None) -> SurpriseResult:
        """
        Compute total surprise across all levels.

        Lower surprise = code matches expectations better.
        """
        by_level = {
            "token": self.token_prior.compute_surprise(code),
            "ast": self.ast_prior.compute_surprise(code),
            "semantic": self.semantic_prior.compute_surprise(code),
            "algorithm": self.algorithm_prior.compute_surprise(code),
        }

        # Weighted total
        total = sum(
            by_level[level] * self.level_weights[level]
            for level in by_level
        )

        return SurpriseResult(
            total=total,
            by_level=by_level,
            details={
                "algorithm_patterns": self.algorithm_prior.detect_patterns(code),
            }
        )

    def update_beliefs(self, code: str, observations: Dict[str, Any]) -> None:
        """
        Update world model based on observations.

        This is the LEARNING step in Active Inference.
        """
        self._observations.append({
            "code": code[:500],
            "observations": observations,
            "timestamp": datetime.now().isoformat(),
        })

        # Update token priors based on observed code
        tokens = re.findall(r'\b\w+\b', code)
        for token in tokens:
            current = self.token_prior.token_probs.get(token, 0.01)
            # Slowly adjust toward observed frequency
            self.token_prior.token_probs[token] = current * 0.95 + 0.05 * (1.0 / max(len(tokens), 1))

        # Keep observations bounded
        self._observations = self._observations[-100:]

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about model learning."""
        return {
            "observations_count": len(self._observations),
            "token_vocab_size": len(self.token_prior.token_probs),
            "level_weights": self.level_weights,
        }
