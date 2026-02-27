"""
Curiosity Engine (V6)
======================
Surprise-weighted mutation selection for guided exploration.

Key Concept: NOT random mutation - target HIGH SURPRISE regions
where information gain is greatest.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import random
import ast
import re


@dataclass
class CodeRegion:
    """A region of code to potentially mutate."""
    start_line: int
    end_line: int
    region_type: str  # function, loop, condition, expression
    content: str
    surprise_score: float = 0.0
    information_gain: float = 0.0


@dataclass
class MutationResult:
    """Result of a mutation attempt."""
    original_code: str
    mutated_code: str
    region: CodeRegion
    mutation_type: str
    surprise_before: float
    surprise_after: float
    success: bool
    fitness_delta: float = 0.0


class WorldModel:
    """
    Agent's internal model of what "good code" looks like.

    Used to compute surprise - how unexpected is this code?
    """

    def __init__(self):
        # Pattern frequencies from past experience
        self.pattern_counts: Dict[str, int] = {}
        self.total_patterns = 0

        # Expected distributions
        self.expected_node_types: Dict[str, float] = {
            "FunctionDef": 0.1,
            "If": 0.15,
            "For": 0.1,
            "While": 0.05,
            "Return": 0.1,
            "Assign": 0.2,
            "Call": 0.15,
            "BinOp": 0.1,
            "Compare": 0.05,
        }

    def update(self, code: str) -> None:
        """Update world model with observed code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                node_type = type(node).__name__
                self.pattern_counts[node_type] = self.pattern_counts.get(node_type, 0) + 1
                self.total_patterns += 1
        except SyntaxError:
            pass

    def compute_surprise(self, code: str) -> float:
        """
        Compute how surprising this code is relative to world model.

        Higher surprise = code deviates more from expectations.
        """
        if not code:
            return 0.0

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.5  # Neutral if can't parse

        # Count node types in this code
        node_counts: Dict[str, int] = {}
        total = 0
        for node in ast.walk(tree):
            node_type = type(node).__name__
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
            total += 1

        if total == 0:
            return 0.0

        # Compute KL-divergence-like surprise
        surprise = 0.0
        for node_type, count in node_counts.items():
            observed_freq = count / total
            expected_freq = self.expected_node_types.get(node_type, 0.01)

            # Surprise from deviation
            if observed_freq > 0:
                surprise += observed_freq * abs(observed_freq - expected_freq)

        # Check for rare patterns (more surprising)
        for node_type, count in node_counts.items():
            if node_type not in self.expected_node_types:
                surprise += 0.1 * count  # Bonus surprise for unexpected nodes

        return min(1.0, surprise)


class CuriosityEngine:
    """
    Curiosity-driven mutation selection.

    Instead of random mutations, target regions where:
    1. Surprise is high (unexpected patterns)
    2. Information gain is high (learning opportunity)
    3. Past mutations were successful
    """

    def __init__(self):
        self.world_model = WorldModel()
        self.mutation_history: List[MutationResult] = []
        self.region_success_rates: Dict[str, float] = {}

    def identify_regions(self, code: str) -> List[CodeRegion]:
        """Identify mutable regions in code."""
        regions = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return regions

        lines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                regions.append(CodeRegion(
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    region_type="function",
                    content=ast.unparse(node) if hasattr(ast, 'unparse') else "",
                ))
            elif isinstance(node, (ast.For, ast.While)):
                regions.append(CodeRegion(
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    region_type="loop",
                    content=ast.unparse(node) if hasattr(ast, 'unparse') else "",
                ))
            elif isinstance(node, ast.If):
                regions.append(CodeRegion(
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    region_type="condition",
                    content=ast.unparse(node) if hasattr(ast, 'unparse') else "",
                ))

        return regions

    def score_regions(self, code: str, regions: List[CodeRegion]) -> List[CodeRegion]:
        """Score regions by surprise and information gain."""
        for region in regions:
            # Compute surprise for this region
            region.surprise_score = self.world_model.compute_surprise(region.content)

            # Information gain based on past success
            region.information_gain = self.region_success_rates.get(
                region.region_type, 0.5
            )

        return regions

    def select_mutation_region(
        self,
        code: str,
        energy_budget: float
    ) -> Optional[CodeRegion]:
        """
        Select a region to mutate based on curiosity.

        Higher surprise + higher information gain = higher priority.
        """
        regions = self.identify_regions(code)
        if not regions:
            return None

        regions = self.score_regions(code, regions)

        # Combine surprise and information gain
        for region in regions:
            # Curiosity = surprise * (1 + information_gain)
            region.curiosity = region.surprise_score * (1 + region.information_gain)

        # Sort by curiosity (highest first)
        regions.sort(key=lambda r: getattr(r, 'curiosity', 0), reverse=True)

        # Filter by energy budget (larger regions cost more)
        affordable = [
            r for r in regions
            if (r.end_line - r.start_line + 1) * 0.5 <= energy_budget
        ]

        if not affordable:
            return regions[0] if regions else None

        # Probabilistic selection weighted by curiosity
        total_curiosity = sum(getattr(r, 'curiosity', 0.1) for r in affordable)
        if total_curiosity == 0:
            return random.choice(affordable)

        weights = [getattr(r, 'curiosity', 0.1) / total_curiosity for r in affordable]
        selected = random.choices(affordable, weights=weights, k=1)[0]

        return selected

    def calculate_surprise(
        self,
        original: str,
        mutated: str,
        outcome_fitness: float
    ) -> float:
        """
        Calculate surprise from a mutation.

        Surprise = how unexpected was the outcome?
        """
        original_surprise = self.world_model.compute_surprise(original)
        mutated_surprise = self.world_model.compute_surprise(mutated)

        # Surprise is the change in unexpectedness
        structural_surprise = abs(mutated_surprise - original_surprise)

        # Plus surprise from fitness change
        # (assuming fitness changes are somewhat unexpected)
        fitness_surprise = abs(outcome_fitness)

        return (structural_surprise + fitness_surprise) / 2

    def record_mutation(self, result: MutationResult) -> None:
        """Record a mutation result for learning."""
        self.mutation_history.append(result)

        # Update world model
        self.world_model.update(result.mutated_code)

        # Update success rates
        region_type = result.region.region_type
        current = self.region_success_rates.get(region_type, 0.5)

        if result.success:
            self.region_success_rates[region_type] = current * 0.9 + 0.1
        else:
            self.region_success_rates[region_type] = current * 0.9

    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get curiosity exploration statistics."""
        return {
            "mutations_recorded": len(self.mutation_history),
            "region_success_rates": self.region_success_rates,
            "patterns_learned": self.world_model.total_patterns,
            "unique_patterns": len(self.world_model.pattern_counts),
        }


class MutationGenerator:
    """Generate mutations for selected regions."""

    MUTATION_TYPES = [
        "swap_operator",
        "change_constant",
        "add_condition",
        "remove_statement",
        "duplicate_statement",
        "swap_statements",
        "add_loop",
        "change_variable",
    ]

    def __init__(self, brain=None):
        self.brain = brain  # Optional LLM brain for smart mutations

    def generate_mutation(
        self,
        code: str,
        region: CodeRegion,
        mutation_type: Optional[str] = None
    ) -> str:
        """Generate a mutation for the given region."""
        if mutation_type is None:
            mutation_type = random.choice(self.MUTATION_TYPES)

        if self.brain:
            return self._llm_mutation(code, region, mutation_type)
        else:
            return self._heuristic_mutation(code, region, mutation_type)

    def _llm_mutation(
        self,
        code: str,
        region: CodeRegion,
        mutation_type: str
    ) -> str:
        """Use LLM to generate smart mutation."""
        prompt = f"""Mutate this code region using {mutation_type}:

Code:
```
{code}
```

Region to mutate (lines {region.start_line}-{region.end_line}):
```
{region.content}
```

Output ONLY the complete modified code."""

        return self.brain.reason(prompt, max_tokens=500)

    def _heuristic_mutation(
        self,
        code: str,
        region: CodeRegion,
        mutation_type: str
    ) -> str:
        """Use heuristics to generate mutation."""
        lines = code.split('\n')

        if mutation_type == "swap_operator":
            # Swap arithmetic operators
            operators = [('+', '-'), ('*', '/'), ('<', '>'), ('==', '!=')]
            for old, new in operators:
                if old in code:
                    return code.replace(old, new, 1)

        elif mutation_type == "change_constant":
            # Find and modify numeric constants
            match = re.search(r'\b(\d+)\b', code)
            if match:
                old_num = match.group(1)
                new_num = str(int(old_num) + random.choice([-1, 1, 2, -2]))
                return code.replace(old_num, new_num, 1)

        elif mutation_type == "duplicate_statement":
            # Duplicate a random statement
            if region.start_line <= len(lines):
                line = lines[region.start_line - 1]
                lines.insert(region.start_line, line)
                return '\n'.join(lines)

        elif mutation_type == "swap_statements":
            # Swap two adjacent statements
            start = region.start_line - 1
            end = min(region.end_line, len(lines)) - 1
            if end > start:
                lines[start], lines[start + 1] = lines[start + 1], lines[start]
                return '\n'.join(lines)

        # Default: return original if no mutation applied
        return code
