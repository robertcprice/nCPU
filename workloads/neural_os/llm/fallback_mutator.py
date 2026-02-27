#!/usr/bin/env python3
"""
FALLBACK MUTATOR - AST-Based Mutations When LLM Unavailable

Provides reliable mutations using pure AST transformations.
No external dependencies (LLM, network, etc.).

Used when:
- Ollama is not running
- LLM request times out
- LLM returns invalid code repeatedly

Mutation strategies:
- Constant tweaking
- Operator swapping
- Loop transformations
- Statement reordering
- Expression simplification
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum, auto
import ast
import random
import logging

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Available fallback mutation strategies."""
    TWEAK_CONSTANTS = auto()
    SWAP_OPERATORS = auto()
    REORDER_STATEMENTS = auto()
    SIMPLIFY_EXPRESSIONS = auto()
    LOOP_TRANSFORM = auto()
    ADD_CACHE = auto()
    INLINE_SIMPLE = auto()


@dataclass
class FallbackResult:
    """Result of a fallback mutation."""
    success: bool
    strategy: FallbackStrategy
    original_code: str
    mutated_code: str
    description: str = ""
    error: Optional[str] = None


class FallbackMutator:
    """
    AST-based fallback mutator.

    Provides reliable mutations when LLM is not available.
    All mutations are syntactically valid (AST → code → parse check).
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self._strategies = [
            (FallbackStrategy.TWEAK_CONSTANTS, self._tweak_constants),
            (FallbackStrategy.SWAP_OPERATORS, self._swap_operators),
            (FallbackStrategy.REORDER_STATEMENTS, self._reorder_statements),
            (FallbackStrategy.SIMPLIFY_EXPRESSIONS, self._simplify_expressions),
            (FallbackStrategy.LOOP_TRANSFORM, self._loop_transform),
            (FallbackStrategy.ADD_CACHE, self._add_cache),
        ]

        self._mutation_count = 0
        self._success_count = 0

    def mutate(
        self,
        source_code: str,
        strategy: Optional[FallbackStrategy] = None,
    ) -> FallbackResult:
        """
        Apply a mutation to the source code.

        If strategy is None, tries random strategies until one succeeds.
        """
        self._mutation_count += 1

        if strategy:
            # Apply specific strategy
            for strat, func in self._strategies:
                if strat == strategy:
                    return func(source_code)

            return FallbackResult(
                success=False,
                strategy=strategy,
                original_code=source_code,
                mutated_code=source_code,
                error=f"Unknown strategy: {strategy}",
            )

        # Try random strategies
        strategies = list(self._strategies)
        random.shuffle(strategies)

        for strat, func in strategies:
            result = func(source_code)
            if result.success and result.mutated_code != source_code:
                self._success_count += 1
                return result

        return FallbackResult(
            success=False,
            strategy=FallbackStrategy.TWEAK_CONSTANTS,
            original_code=source_code,
            mutated_code=source_code,
            error="All strategies failed",
        )

    def multi_mutate(self, source_code: str, count: int = 3) -> FallbackResult:
        """Apply multiple mutations in sequence."""
        current_code = source_code
        descriptions = []

        for _ in range(count):
            result = self.mutate(current_code)
            if result.success:
                current_code = result.mutated_code
                descriptions.append(result.description)
            else:
                break

        return FallbackResult(
            success=current_code != source_code,
            strategy=FallbackStrategy.TWEAK_CONSTANTS,
            original_code=source_code,
            mutated_code=current_code,
            description="; ".join(descriptions) if descriptions else "No changes",
        )

    def _tweak_constants(self, source_code: str) -> FallbackResult:
        """Modify numeric constants slightly."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.TWEAK_CONSTANTS,
                original_code=source_code,
                mutated_code=source_code,
                error=str(e),
            )

        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value != 0 and node.value != 1:  # Skip common constants
                    constants.append(node)

        if not constants:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.TWEAK_CONSTANTS,
                original_code=source_code,
                mutated_code=source_code,
                error="No modifiable constants",
            )

        target = random.choice(constants)
        old_value = target.value

        # Small tweaks
        tweaks = [
            lambda x: x + 1,
            lambda x: x - 1,
            lambda x: int(x * 1.1) if isinstance(x, int) else x * 1.1,
            lambda x: int(x * 0.9) if isinstance(x, int) else x * 0.9,
        ]

        target.value = random.choice(tweaks)(old_value)

        try:
            new_code = ast.unparse(tree)
            ast.parse(new_code)  # Validate

            return FallbackResult(
                success=True,
                strategy=FallbackStrategy.TWEAK_CONSTANTS,
                original_code=source_code,
                mutated_code=new_code,
                description=f"Tweaked constant {old_value} → {target.value}",
            )
        except Exception as e:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.TWEAK_CONSTANTS,
                original_code=source_code,
                mutated_code=source_code,
                error=str(e),
            )

    def _swap_operators(self, source_code: str) -> FallbackResult:
        """Swap compatible operators."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.SWAP_OPERATORS,
                original_code=source_code,
                mutated_code=source_code,
                error=str(e),
            )

        # Find binary operations
        binops = []
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                binops.append(node)

        if not binops:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.SWAP_OPERATORS,
                original_code=source_code,
                mutated_code=source_code,
                error="No binary operations",
            )

        # Safe swaps
        swaps = {
            ast.Add: [ast.Sub],
            ast.Sub: [ast.Add],
            ast.Mult: [ast.FloorDiv, ast.Pow],
            ast.FloorDiv: [ast.Mult],
        }

        target = random.choice(binops)
        old_op = type(target.op).__name__

        if type(target.op) in swaps:
            new_op_type = random.choice(swaps[type(target.op)])
            target.op = new_op_type()
            new_op = type(target.op).__name__

            try:
                new_code = ast.unparse(tree)
                ast.parse(new_code)

                return FallbackResult(
                    success=True,
                    strategy=FallbackStrategy.SWAP_OPERATORS,
                    original_code=source_code,
                    mutated_code=new_code,
                    description=f"Swapped operator {old_op} → {new_op}",
                )
            except Exception as e:
                return FallbackResult(
                    success=False,
                    strategy=FallbackStrategy.SWAP_OPERATORS,
                    original_code=source_code,
                    mutated_code=source_code,
                    error=str(e),
                )

        return FallbackResult(
            success=False,
            strategy=FallbackStrategy.SWAP_OPERATORS,
            original_code=source_code,
            mutated_code=source_code,
            error="No swappable operators",
        )

    def _reorder_statements(self, source_code: str) -> FallbackResult:
        """Reorder independent statements."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.REORDER_STATEMENTS,
                original_code=source_code,
                mutated_code=source_code,
                error=str(e),
            )

        # Find function bodies
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        if not functions:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.REORDER_STATEMENTS,
                original_code=source_code,
                mutated_code=source_code,
                error="No functions to reorder",
            )

        func = random.choice(functions)

        if len(func.body) < 2:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.REORDER_STATEMENTS,
                original_code=source_code,
                mutated_code=source_code,
                error="Function too short",
            )

        # Find swappable adjacent statements
        swappable = []
        for i in range(len(func.body) - 1):
            # Simple heuristic: assignments that don't depend on each other
            if isinstance(func.body[i], ast.Assign) and isinstance(func.body[i + 1], ast.Assign):
                swappable.append(i)

        if not swappable:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.REORDER_STATEMENTS,
                original_code=source_code,
                mutated_code=source_code,
                error="No swappable statements",
            )

        idx = random.choice(swappable)
        func.body[idx], func.body[idx + 1] = func.body[idx + 1], func.body[idx]

        try:
            new_code = ast.unparse(tree)
            ast.parse(new_code)

            return FallbackResult(
                success=True,
                strategy=FallbackStrategy.REORDER_STATEMENTS,
                original_code=source_code,
                mutated_code=new_code,
                description=f"Swapped statements {idx} ↔ {idx + 1}",
            )
        except Exception as e:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.REORDER_STATEMENTS,
                original_code=source_code,
                mutated_code=source_code,
                error=str(e),
            )

    def _simplify_expressions(self, source_code: str) -> FallbackResult:
        """Simplify redundant expressions."""
        # Simple pattern-based simplifications
        patterns = [
            ('x + 0', 'x'),
            ('x * 1', 'x'),
            ('x - 0', 'x'),
            ('x // 1', 'x'),
            ('not not ', ''),
            ('True and ', ''),
            (' and True', ''),
            ('False or ', ''),
            (' or False', ''),
        ]

        new_code = source_code
        applied = []

        for old, new in patterns:
            if old in new_code:
                new_code = new_code.replace(old, new, 1)
                applied.append(f"{old} → {new}")

        if new_code != source_code:
            try:
                ast.parse(new_code)
                return FallbackResult(
                    success=True,
                    strategy=FallbackStrategy.SIMPLIFY_EXPRESSIONS,
                    original_code=source_code,
                    mutated_code=new_code,
                    description=f"Simplified: {', '.join(applied)}",
                )
            except SyntaxError:
                pass

        return FallbackResult(
            success=False,
            strategy=FallbackStrategy.SIMPLIFY_EXPRESSIONS,
            original_code=source_code,
            mutated_code=source_code,
            error="No simplifiable patterns",
        )

    def _loop_transform(self, source_code: str) -> FallbackResult:
        """Transform loop structures."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.LOOP_TRANSFORM,
                original_code=source_code,
                mutated_code=source_code,
                error=str(e),
            )

        # Find for loops with range
        for_loops = []
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        for_loops.append(node)

        if not for_loops:
            return FallbackResult(
                success=False,
                strategy=FallbackStrategy.LOOP_TRANSFORM,
                original_code=source_code,
                mutated_code=source_code,
                error="No transformable loops",
            )

        # Modify range bounds slightly
        target = random.choice(for_loops)

        if target.iter.args:
            # Find a constant argument to tweak
            for arg in target.iter.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    old_value = arg.value
                    arg.value = arg.value + random.choice([-1, 1])

                    try:
                        new_code = ast.unparse(tree)
                        ast.parse(new_code)

                        return FallbackResult(
                            success=True,
                            strategy=FallbackStrategy.LOOP_TRANSFORM,
                            original_code=source_code,
                            mutated_code=new_code,
                            description=f"Modified loop bound {old_value} → {arg.value}",
                        )
                    except Exception:
                        arg.value = old_value

        return FallbackResult(
            success=False,
            strategy=FallbackStrategy.LOOP_TRANSFORM,
            original_code=source_code,
            mutated_code=source_code,
            error="Could not transform loop",
        )

    def _add_cache(self, source_code: str) -> FallbackResult:
        """Add simple caching to functions (memoization hint)."""
        # Look for recursive function patterns
        if 'def ' in source_code and ('fib' in source_code.lower() or 'factorial' in source_code.lower()):
            # Add lru_cache decorator if not present
            if 'lru_cache' not in source_code:
                new_code = "from functools import lru_cache\n\n@lru_cache(maxsize=128)\n" + source_code

                try:
                    ast.parse(new_code)
                    return FallbackResult(
                        success=True,
                        strategy=FallbackStrategy.ADD_CACHE,
                        original_code=source_code,
                        mutated_code=new_code,
                        description="Added lru_cache decorator",
                    )
                except SyntaxError:
                    pass

        return FallbackResult(
            success=False,
            strategy=FallbackStrategy.ADD_CACHE,
            original_code=source_code,
            mutated_code=source_code,
            error="No cacheable patterns",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get mutation statistics."""
        return {
            'total_mutations': self._mutation_count,
            'successful_mutations': self._success_count,
            'success_rate': self._success_count / self._mutation_count if self._mutation_count > 0 else 0,
        }
