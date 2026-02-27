#!/usr/bin/env python3
"""
CODE SURGERY - AST-Based Mutation Primitives

Provides structured code transformations using Python's AST module.
These are the building blocks that the Mutator uses to evolve code.

Mutation Primitives:
- swap_lines: Swap two lines of code
- change_constant: Modify numeric/string constants
- unwrap_loop: Convert for/while to equivalent form
- inline_function: Inline a function call
- extract_function: Extract code to a new function
- change_operator: Swap operators (+, -, *, /)
- reorder_statements: Change statement order

Each mutation preserves syntactic validity (AST parseable).
Semantic validity is checked by the Judge.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
from enum import Enum, auto
import ast
import random
import copy
import logging

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of mutations available."""
    SWAP_LINES = auto()
    CHANGE_CONSTANT = auto()
    CHANGE_OPERATOR = auto()
    UNWRAP_LOOP = auto()
    LOOP_INTERCHANGE = auto()
    INLINE_FUNCTION = auto()
    EXTRACT_EXPRESSION = auto()
    REORDER_STATEMENTS = auto()
    DUPLICATE_CHECK = auto()
    ADD_MEMOIZATION = auto()
    VECTORIZE_LOOP = auto()


@dataclass
class MutationResult:
    """Result of a mutation operation."""
    success: bool
    mutation_type: MutationType
    original_code: str
    new_code: str
    description: str = ""
    error: Optional[str] = None
    line_changes: List[Tuple[int, str, str]] = field(default_factory=list)

    @property
    def is_different(self) -> bool:
        """Check if mutation actually changed the code."""
        return self.original_code != self.new_code


class MutationPrimitive:
    """Base class for mutation primitives."""

    def __init__(self, mutation_type: MutationType):
        self.mutation_type = mutation_type

    def apply(self, code: str) -> MutationResult:
        """Apply the mutation to code. Override in subclasses."""
        raise NotImplementedError

    def can_apply(self, code: str) -> bool:
        """Check if mutation can be applied. Override in subclasses."""
        return True


class ConstantChanger(MutationPrimitive):
    """Change numeric constants in the code."""

    def __init__(self):
        super().__init__(MutationType.CHANGE_CONSTANT)

    def can_apply(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return True
            return False
        except:
            return False

    def apply(self, code: str) -> MutationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error=str(e),
            )

        # Find all numeric constants
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                constants.append(node)

        if not constants:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error="No numeric constants found",
            )

        # Pick a random constant to change
        target = random.choice(constants)
        old_value = target.value

        # Mutation strategies
        strategies = [
            lambda x: x + 1,
            lambda x: x - 1,
            lambda x: x * 2,
            lambda x: x // 2 if isinstance(x, int) and x != 0 else x / 2,
            lambda x: -x,
            lambda x: 0,
            lambda x: 1,
            lambda x: x + random.randint(-10, 10),
        ]

        new_value = random.choice(strategies)(old_value)
        target.value = new_value

        try:
            new_code = ast.unparse(tree)
            return MutationResult(
                success=True,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=new_code,
                description=f"Changed constant {old_value} -> {new_value}",
            )
        except Exception as e:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error=str(e),
            )


class OperatorChanger(MutationPrimitive):
    """Change binary operators in the code."""

    # Operator groups - swap within groups
    ARITHMETIC = {ast.Add: [ast.Sub, ast.Mult], ast.Sub: [ast.Add], ast.Mult: [ast.Add, ast.FloorDiv], ast.FloorDiv: [ast.Mult]}
    COMPARISON = {ast.Lt: [ast.LtE, ast.Gt], ast.LtE: [ast.Lt, ast.GtE], ast.Gt: [ast.GtE, ast.Lt], ast.GtE: [ast.Gt, ast.LtE]}
    LOGICAL = {ast.And: [ast.Or], ast.Or: [ast.And]}

    def __init__(self):
        super().__init__(MutationType.CHANGE_OPERATOR)

    def can_apply(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.BinOp, ast.Compare, ast.BoolOp)):
                    return True
            return False
        except:
            return False

    def apply(self, code: str) -> MutationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error=str(e),
            )

        # Find all operators
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                targets.append(('binop', node))
            elif isinstance(node, ast.Compare) and node.ops:
                targets.append(('compare', node))
            elif isinstance(node, ast.BoolOp):
                targets.append(('boolop', node))

        if not targets:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error="No operators found",
            )

        # Pick random target
        target_type, target = random.choice(targets)
        old_op_name = ""
        new_op_name = ""

        if target_type == 'binop':
            op_type = type(target.op)
            old_op_name = op_type.__name__
            if op_type in self.ARITHMETIC:
                new_op_type = random.choice(self.ARITHMETIC[op_type])
                target.op = new_op_type()
                new_op_name = new_op_type.__name__

        elif target_type == 'compare' and target.ops:
            op_type = type(target.ops[0])
            old_op_name = op_type.__name__
            if op_type in self.COMPARISON:
                new_op_type = random.choice(self.COMPARISON[op_type])
                target.ops[0] = new_op_type()
                new_op_name = new_op_type.__name__

        elif target_type == 'boolop':
            op_type = type(target.op)
            old_op_name = op_type.__name__
            if op_type in self.LOGICAL:
                new_op_type = random.choice(self.LOGICAL[op_type])
                target.op = new_op_type()
                new_op_name = new_op_type.__name__

        if old_op_name == new_op_name or not new_op_name:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error="No suitable operator change found",
            )

        try:
            new_code = ast.unparse(tree)
            return MutationResult(
                success=True,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=new_code,
                description=f"Changed operator {old_op_name} -> {new_op_name}",
            )
        except Exception as e:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error=str(e),
            )


class StatementReorderer(MutationPrimitive):
    """Reorder independent statements."""

    def __init__(self):
        super().__init__(MutationType.REORDER_STATEMENTS)

    def can_apply(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.Module)):
                    if hasattr(node, 'body') and len(node.body) >= 2:
                        return True
            return False
        except:
            return False

    def _get_used_names(self, node: ast.AST) -> Set[str]:
        """Get all names used (loaded) in a node."""
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                names.add(child.id)
        return names

    def _get_defined_names(self, node: ast.AST) -> Set[str]:
        """Get all names defined (stored) in a node."""
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            elif isinstance(child, ast.FunctionDef):
                names.add(child.name)
        return names

    def apply(self, code: str) -> MutationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error=str(e),
            )

        # Find a suitable body to reorder
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Module)):
                if hasattr(node, 'body') and len(node.body) >= 2:
                    targets.append(node)

        if not targets:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error="No suitable blocks found",
            )

        target = random.choice(targets)
        body = target.body

        # Find two adjacent statements that can be swapped
        swappable = []
        for i in range(len(body) - 1):
            stmt_a, stmt_b = body[i], body[i + 1]

            # Check if they're independent (no data dependency)
            defined_a = self._get_defined_names(stmt_a)
            used_b = self._get_used_names(stmt_b)

            if not defined_a.intersection(used_b):
                swappable.append(i)

        if not swappable:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error="No independent statements to swap",
            )

        # Swap
        idx = random.choice(swappable)
        body[idx], body[idx + 1] = body[idx + 1], body[idx]

        try:
            new_code = ast.unparse(tree)
            return MutationResult(
                success=True,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=new_code,
                description=f"Swapped statements at lines {idx} and {idx + 1}",
            )
        except Exception as e:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error=str(e),
            )


class LoopOptimizer(MutationPrimitive):
    """Optimize loops (unroll, interchange, etc.)."""

    def __init__(self):
        super().__init__(MutationType.LOOP_INTERCHANGE)

    def can_apply(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    return True
            return False
        except:
            return False

    def apply(self, code: str) -> MutationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
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
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error="No range-based for loops found",
            )

        # Try to optimize: convert range(len(x)) to enumerate
        target = random.choice(for_loops)

        # Simple mutation: change range bounds
        if target.iter.args:
            # Modify range arguments slightly
            for arg in target.iter.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    # Small adjustment
                    arg.value = arg.value + random.choice([-1, 0, 1])
                    break

        try:
            new_code = ast.unparse(tree)
            return MutationResult(
                success=True,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=new_code,
                description="Modified loop bounds",
            )
        except Exception as e:
            return MutationResult(
                success=False,
                mutation_type=self.mutation_type,
                original_code=code,
                new_code=code,
                error=str(e),
            )


class CodeSurgeon:
    """
    Main class for performing code surgery.

    Orchestrates multiple mutation primitives to evolve code.
    """

    def __init__(self):
        self.primitives: List[MutationPrimitive] = [
            ConstantChanger(),
            OperatorChanger(),
            StatementReorderer(),
            LoopOptimizer(),
        ]
        self._mutation_history: List[MutationResult] = []

    def random_mutation(self, code: str) -> MutationResult:
        """Apply a random applicable mutation."""
        # Find applicable primitives
        applicable = [p for p in self.primitives if p.can_apply(code)]

        if not applicable:
            return MutationResult(
                success=False,
                mutation_type=MutationType.SWAP_LINES,
                original_code=code,
                new_code=code,
                error="No applicable mutations",
            )

        # Try mutations until one succeeds
        random.shuffle(applicable)

        for primitive in applicable:
            result = primitive.apply(code)
            if result.success and result.is_different:
                self._mutation_history.append(result)
                return result

        # All failed
        return MutationResult(
            success=False,
            mutation_type=MutationType.SWAP_LINES,
            original_code=code,
            new_code=code,
            error="All mutations failed",
        )

    def apply_specific(self, code: str, mutation_type: MutationType) -> MutationResult:
        """Apply a specific type of mutation."""
        for primitive in self.primitives:
            if primitive.mutation_type == mutation_type:
                return primitive.apply(code)

        return MutationResult(
            success=False,
            mutation_type=mutation_type,
            original_code=code,
            new_code=code,
            error=f"Unknown mutation type: {mutation_type}",
        )

    def multi_mutate(self, code: str, count: int = 3) -> MutationResult:
        """Apply multiple mutations in sequence."""
        current_code = code
        descriptions = []

        for _ in range(count):
            result = self.random_mutation(current_code)
            if result.success:
                current_code = result.new_code
                descriptions.append(result.description)
            else:
                break

        return MutationResult(
            success=current_code != code,
            mutation_type=MutationType.SWAP_LINES,  # Generic
            original_code=code,
            new_code=current_code,
            description="; ".join(descriptions) if descriptions else "No changes",
        )

    def get_mutation_stats(self) -> Dict[str, Any]:
        """Get statistics about mutations performed."""
        if not self._mutation_history:
            return {'total': 0, 'success_rate': 0}

        successes = sum(1 for r in self._mutation_history if r.success)

        type_counts: Dict[str, int] = {}
        for r in self._mutation_history:
            t = r.mutation_type.name
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            'total': len(self._mutation_history),
            'successes': successes,
            'success_rate': successes / len(self._mutation_history),
            'by_type': type_counts,
        }

    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has valid syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def repair_syntax(self, code: str, max_attempts: int = 5) -> Tuple[bool, str]:
        """
        Attempt to repair syntax errors.

        Simple strategies:
        - Add missing colons
        - Fix indentation
        - Balance parentheses
        """
        is_valid, error = self.validate_syntax(code)
        if is_valid:
            return True, code

        current = code

        for _ in range(max_attempts):
            # Try simple fixes
            repairs = [
                self._fix_missing_colon,
                self._fix_indentation,
                self._balance_brackets,
            ]

            for repair_fn in repairs:
                fixed = repair_fn(current)
                is_valid, _ = self.validate_syntax(fixed)
                if is_valid:
                    return True, fixed
                current = fixed

        return False, code

    def _fix_missing_colon(self, code: str) -> str:
        """Add missing colons after def, if, for, while, etc."""
        lines = code.split('\n')
        keywords = ['def ', 'if ', 'elif ', 'else', 'for ', 'while ', 'class ', 'try', 'except', 'finally', 'with ']

        for i, line in enumerate(lines):
            stripped = line.rstrip()
            for kw in keywords:
                if stripped.lstrip().startswith(kw) and not stripped.endswith(':'):
                    lines[i] = stripped + ':'
                    break

        return '\n'.join(lines)

    def _fix_indentation(self, code: str) -> str:
        """Normalize indentation to 4 spaces."""
        lines = code.split('\n')
        fixed = []

        for line in lines:
            # Count leading whitespace
            stripped = line.lstrip()
            if not stripped:
                fixed.append('')
                continue

            indent = len(line) - len(stripped)
            # Normalize to nearest 4 spaces
            normalized_indent = (indent // 4) * 4
            fixed.append(' ' * normalized_indent + stripped)

        return '\n'.join(fixed)

    def _balance_brackets(self, code: str) -> str:
        """Balance parentheses, brackets, and braces."""
        pairs = {'(': ')', '[': ']', '{': '}'}
        stack = []

        for char in code:
            if char in pairs:
                stack.append(pairs[char])
            elif char in pairs.values():
                if stack and stack[-1] == char:
                    stack.pop()

        # Add missing closing brackets
        return code + ''.join(reversed(stack))
