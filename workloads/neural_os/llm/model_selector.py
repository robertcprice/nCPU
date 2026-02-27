#!/usr/bin/env python3
"""
MODEL SELECTOR - Complexity-Based Model Selection

Selects the appropriate LLM model based on code complexity:
- Level 1-3: Simple model (fast)
- Level 4-5: Balanced model (quality + speed)
- Level 6-7: Advanced model (highest quality)

Complexity is determined by:
- Lines of code
- Cyclomatic complexity
- Nesting depth
- Number of functions/classes
- Import count
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import IntEnum
import ast
import logging

logger = logging.getLogger(__name__)


class ComplexityLevel(IntEnum):
    """Code complexity levels."""
    TRIVIAL = 1      # Simple functions (sum, count)
    SIMPLE = 2       # Basic looping (sorting, filtering)
    MODERATE = 3     # Recursive functions
    INTERMEDIATE = 4  # Multi-function modules
    COMPLEX = 5      # Class-based code
    ADVANCED = 6     # Algorithmic (graphs, DP)
    SELF = 7         # SPNC's own code (highest complexity)


@dataclass
class ComplexityMetrics:
    """Detailed complexity metrics for code."""
    level: ComplexityLevel
    lines: int
    functions: int
    classes: int
    max_nesting: int
    cyclomatic: int
    imports: int
    has_recursion: bool
    has_generators: bool
    has_async: bool
    raw_score: float


class ModelSelector:
    """
    Selects the appropriate model based on code complexity.

    Usage:
        selector = ModelSelector()
        model = selector.select_model(source_code)
        metrics = selector.analyze_complexity(source_code)
    """

    # Model mapping by complexity level
    DEFAULT_MODELS = {
        ComplexityLevel.TRIVIAL: "codellama:7b",
        ComplexityLevel.SIMPLE: "codellama:7b",
        ComplexityLevel.MODERATE: "codellama:7b",
        ComplexityLevel.INTERMEDIATE: "deepseek-coder:6.7b",
        ComplexityLevel.COMPLEX: "deepseek-coder:6.7b",
        ComplexityLevel.ADVANCED: "mistral:7b",
        ComplexityLevel.SELF: "mistral:7b",
    }

    def __init__(self, model_mapping: Optional[Dict[ComplexityLevel, str]] = None):
        self.model_mapping = model_mapping or self.DEFAULT_MODELS.copy()

    def analyze_complexity(self, source_code: str) -> ComplexityMetrics:
        """
        Analyze the complexity of source code.

        Returns detailed metrics including the complexity level.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # If can't parse, assume moderate complexity
            return ComplexityMetrics(
                level=ComplexityLevel.MODERATE,
                lines=len(source_code.split('\n')),
                functions=0,
                classes=0,
                max_nesting=0,
                cyclomatic=1,
                imports=0,
                has_recursion=False,
                has_generators=False,
                has_async=False,
                raw_score=3.0,
            )

        lines = len(source_code.split('\n'))
        functions = 0
        classes = 0
        max_nesting = 0
        cyclomatic = 1  # Base complexity
        imports = 0
        has_recursion = False
        has_generators = False
        has_async = False

        # Collect function names for recursion detection
        function_names = set()

        def visit_node(node: ast.AST, depth: int = 0) -> None:
            nonlocal functions, classes, max_nesting, cyclomatic, imports
            nonlocal has_recursion, has_generators, has_async

            max_nesting = max(max_nesting, depth)

            if isinstance(node, ast.FunctionDef):
                functions += 1
                function_names.add(node.name)

            elif isinstance(node, ast.AsyncFunctionDef):
                functions += 1
                has_async = True
                function_names.add(node.name)

            elif isinstance(node, ast.ClassDef):
                classes += 1

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1

            elif isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                cyclomatic += 1

            elif isinstance(node, ast.BoolOp):
                cyclomatic += len(node.values) - 1

            elif isinstance(node, (ast.Yield, ast.YieldFrom)):
                has_generators = True

            elif isinstance(node, ast.Call):
                # Check for recursion
                if isinstance(node.func, ast.Name) and node.func.id in function_names:
                    has_recursion = True

            # Recurse into children
            for child in ast.iter_child_nodes(node):
                new_depth = depth + 1 if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)) else depth
                visit_node(child, new_depth)

        visit_node(tree)

        # Calculate raw score
        raw_score = (
            (lines / 50) +  # ~50 lines per complexity point
            (functions * 0.5) +
            (classes * 1.0) +
            (max_nesting * 0.3) +
            (cyclomatic * 0.2) +
            (imports * 0.1) +
            (2.0 if has_recursion else 0) +
            (1.0 if has_generators else 0) +
            (1.5 if has_async else 0)
        )

        # Map to complexity level
        if raw_score < 1.5:
            level = ComplexityLevel.TRIVIAL
        elif raw_score < 3.0:
            level = ComplexityLevel.SIMPLE
        elif raw_score < 5.0:
            level = ComplexityLevel.MODERATE
        elif raw_score < 8.0:
            level = ComplexityLevel.INTERMEDIATE
        elif raw_score < 12.0:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.ADVANCED

        return ComplexityMetrics(
            level=level,
            lines=lines,
            functions=functions,
            classes=classes,
            max_nesting=max_nesting,
            cyclomatic=cyclomatic,
            imports=imports,
            has_recursion=has_recursion,
            has_generators=has_generators,
            has_async=has_async,
            raw_score=raw_score,
        )

    def select_model(self, source_code: str) -> str:
        """
        Select the appropriate model for the given code.

        Returns the model name to use with Ollama.
        """
        metrics = self.analyze_complexity(source_code)
        return self.model_mapping.get(metrics.level, self.DEFAULT_MODELS[ComplexityLevel.MODERATE])

    def get_model_for_level(self, level: ComplexityLevel) -> str:
        """Get model for a specific complexity level."""
        return self.model_mapping.get(level, self.DEFAULT_MODELS[level])

    def set_model_for_level(self, level: ComplexityLevel, model: str) -> None:
        """Set model for a specific complexity level."""
        self.model_mapping[level] = model

    def is_self_modification(self, source_code: str, self_patterns: Optional[List[str]] = None) -> bool:
        """
        Check if code appears to be part of the SPNC system itself.

        This triggers the highest complexity level (SELF) for extra caution.
        """
        patterns = self_patterns or [
            'Governor',
            'Arena',
            'Judge',
            'KillSwitch',
            'Population',
            'CodeSurgeon',
            'OllamaMutator',
            'semantic_engine',
            'constitution',
            'evolution',
        ]

        for pattern in patterns:
            if pattern in source_code:
                return True

        return False

    def select_model_with_self_check(
        self,
        source_code: str,
        self_patterns: Optional[List[str]] = None,
    ) -> str:
        """
        Select model with additional check for self-modification.

        If code appears to be part of SPNC, uses the SELF level model.
        """
        if self.is_self_modification(source_code, self_patterns):
            return self.model_mapping.get(ComplexityLevel.SELF, self.DEFAULT_MODELS[ComplexityLevel.SELF])

        return self.select_model(source_code)


def analyze_code_complexity(source_code: str) -> Dict[str, Any]:
    """
    Convenience function to analyze code complexity.

    Returns a dictionary with all metrics.
    """
    selector = ModelSelector()
    metrics = selector.analyze_complexity(source_code)

    return {
        'level': metrics.level.name,
        'level_value': int(metrics.level),
        'lines': metrics.lines,
        'functions': metrics.functions,
        'classes': metrics.classes,
        'max_nesting': metrics.max_nesting,
        'cyclomatic_complexity': metrics.cyclomatic,
        'imports': metrics.imports,
        'has_recursion': metrics.has_recursion,
        'has_generators': metrics.has_generators,
        'has_async': metrics.has_async,
        'raw_score': metrics.raw_score,
        'recommended_model': selector.select_model(source_code),
    }
