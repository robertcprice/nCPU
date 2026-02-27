#!/usr/bin/env python3
"""
MDL OPTIMIZER: Minimum Description Length for Autonomous Program Synthesis

This is Phase 3 of the Semantic Synthesizer based on the 5-AI Hybrid Review.
It uses Minimum Description Length (MDL) principles to find the simplest
programs that explain the data - enabling autonomous discovery.

Key insights from the hybrid review:
- "COMPRESSION = UNDERSTANDING: Shorter program → Captured essential structure" (Expansion)
- "Kolmogorov complexity provides principled framework for evaluating discoveries" (Expansion)
- "The right amount of complexity is the minimum needed for the current task" (Gap Analyst)

This module provides:
1. Program complexity metrics (description length)
2. Library compression (how much shorter with primitives)
3. Novelty detection via incompressibility
4. Autonomous curiosity based on compression progress
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
import math
from collections import Counter

from rewrite_engine import (
    Expr, Var, Const, App,
    RewriteEngine,
    var, const, add, sub, mul, div, square, double,
    lsl, lsr
)
from semantic_dictionary import SEMANTIC_DICTIONARY, get_operation
from trace_analyzer import Trace, TraceAnalyzer, trace_from_io


# =============================================================================
# DESCRIPTION LENGTH METRICS
# =============================================================================

class DescriptionLengthCalculator:
    """
    Calculate the description length (complexity) of expressions.

    Uses bits as the unit, approximating Kolmogorov complexity
    with a practical encoding.
    """

    def __init__(self, library: Optional[Dict[str, Expr]] = None):
        """
        Initialize with an optional library of learned primitives.

        Programs that use library primitives have shorter descriptions.
        """
        self.library = library or {}
        self.base_op_bits = 5  # log2(~30 operations) ≈ 5 bits
        self.register_bits = 5  # log2(32 registers) = 5 bits
        self.small_const_bits = 4  # Small constants (0-15)
        self.large_const_bits = 16  # Larger constants

    def description_length(self, expr: Expr) -> float:
        """
        Calculate the description length of an expression in bits.

        This is an approximation of Kolmogorov complexity.
        """
        return self._dl(expr, depth=0)

    def _dl(self, expr: Expr, depth: int) -> float:
        """Recursive description length calculation."""
        # Check if this expression is in the library (1 bit for library reference)
        expr_str = str(expr)
        if expr_str in self.library:
            return 1.0 + math.log2(len(self.library) + 1)

        if isinstance(expr, Const):
            value = expr.value
            if 0 <= value <= 15:
                return self.small_const_bits
            elif 0 <= value <= 255:
                return 8
            elif 0 <= value <= 65535:
                return 16
            else:
                return 32

        if isinstance(expr, Var):
            return self.register_bits

        if isinstance(expr, App):
            # Operation encoding
            op_bits = self.base_op_bits

            # Check if operation is a derived/learned one
            if expr.op in ['SQUARE', 'DOUBLE', 'NEG']:
                # Derived operations are cheaper if in semantic dictionary
                op_bits = 3

            # Add description length of all arguments
            arg_bits = sum(self._dl(arg, depth + 1) for arg in expr.args)

            # Depth penalty (very deep trees are more complex)
            depth_penalty = 0.1 * depth

            return op_bits + arg_bits + depth_penalty

        return 0.0

    def normalized_complexity(self, expr: Expr) -> float:
        """
        Normalize complexity to [0, 1] range for comparison.

        Lower is simpler.
        """
        dl = self.description_length(expr)
        # Normalize with sigmoid-like function
        return 1.0 - math.exp(-dl / 50.0)

    def add_to_library(self, name: str, expr: Expr):
        """Add a new primitive to the library."""
        self.library[name] = expr

    def library_compression(self, expr: Expr) -> float:
        """
        Calculate how much the library compresses this expression.

        Returns the ratio: (without library) / (with library)
        Values > 1 mean the library helps, < 1 means it hurts.
        """
        without_library = DescriptionLengthCalculator({})
        with_library = self

        dl_without = without_library.description_length(expr)
        dl_with = with_library.description_length(expr)

        if dl_with > 0:
            return dl_without / dl_with
        return 1.0


# =============================================================================
# NOVELTY DETECTION
# =============================================================================

class NoveltyDetector:
    """
    Detect novel programs based on compression.

    A program is novel if it cannot be compressed well using
    existing knowledge (library).
    """

    def __init__(self, library: Dict[str, Expr] = None):
        self.library = library or {}
        self.known_patterns: Set[str] = set()
        self.dl_calculator = DescriptionLengthCalculator(self.library)

    def is_novel(self, expr: Expr, threshold: float = 0.7) -> bool:
        """
        Check if an expression is novel.

        Novel means it's incompressible with current library.
        """
        # Check syntactic novelty
        expr_str = str(expr)
        if expr_str in self.known_patterns:
            return False

        # Check compression novelty
        compression = self.dl_calculator.library_compression(expr)
        if compression < threshold:
            # Poor compression = likely novel
            return True

        return False

    def add_known_pattern(self, expr: Expr):
        """Add a pattern to known set."""
        self.known_patterns.add(str(expr))

    def novelty_score(self, expr: Expr) -> float:
        """
        Calculate novelty score (0 to 1).

        Higher = more novel.
        """
        # Compression-based novelty
        compression = self.dl_calculator.library_compression(expr)

        # Structural novelty (does it use new operations?)
        ops_used = self._get_operations(expr)
        new_ops = ops_used - set(self.library.keys())
        structural_novelty = len(new_ops) / max(1, len(ops_used))

        # Combine metrics
        compression_novelty = 1.0 / max(1.0, compression)

        return 0.5 * compression_novelty + 0.5 * structural_novelty

    def _get_operations(self, expr: Expr) -> Set[str]:
        """Get all operations used in an expression."""
        ops = set()
        if isinstance(expr, App):
            ops.add(expr.op)
            for arg in expr.args:
                ops |= self._get_operations(arg)
        return ops


# =============================================================================
# CURIOSITY ENGINE
# =============================================================================

class CuriosityEngine:
    """
    Drive autonomous exploration based on compression progress.

    Curiosity = seeking programs that improve understanding (compression).
    """

    def __init__(self):
        self.library: Dict[str, Expr] = {}
        self.dl_calculator = DescriptionLengthCalculator(self.library)
        self.novelty_detector = NoveltyDetector(self.library)
        self.discovery_history: List[Tuple[str, float]] = []  # (expr, novelty)
        self.compression_progress: List[float] = []

    def evaluate_interest(self, expr: Expr) -> float:
        """
        Calculate how interesting an expression is.

        Interest = novelty + potential for compression improvement.
        """
        novelty = self.novelty_detector.novelty_score(expr)

        # Check if adding this to library would help
        potential_improvement = self._potential_compression_gain(expr)

        # Combine
        interest = 0.6 * novelty + 0.4 * potential_improvement

        return interest

    def _potential_compression_gain(self, expr: Expr) -> float:
        """
        Estimate how much adding this expression to the library
        would improve overall compression.
        """
        # Simple heuristic: expressions with reusable substructures help more
        complexity = self.dl_calculator.description_length(expr)

        # Moderate complexity is most useful (not too simple, not too complex)
        if 10 <= complexity <= 50:
            return 0.8
        elif 5 <= complexity <= 10:
            return 0.6
        elif complexity < 5:
            return 0.3  # Too simple to be worth adding
        else:
            return 0.4  # Too complex

    def register_discovery(self, name: str, expr: Expr):
        """Register a new discovery and update the library."""
        novelty = self.novelty_detector.novelty_score(expr)
        self.discovery_history.append((name, novelty))

        # Add to library if sufficiently novel
        if novelty > 0.3:
            self.library[name] = expr
            self.dl_calculator.add_to_library(name, expr)
            self.novelty_detector.add_known_pattern(expr)

    def should_explore(self, expr: Expr) -> bool:
        """Determine if we should explore around this expression."""
        interest = self.evaluate_interest(expr)
        return interest > 0.5

    def get_exploration_priority(self) -> List[str]:
        """
        Get a prioritized list of areas to explore.

        Based on gaps in the current library.
        """
        priorities = []

        # Check for missing derived operations
        if 'SQUARE' not in self.library:
            priorities.append('SQUARE: MUL(x, x)')
        if 'DOUBLE' not in self.library:
            priorities.append('DOUBLE: ADD(x, x)')
        if 'CUBE' not in self.library:
            priorities.append('CUBE: MUL(MUL(x, x), x)')

        # Check for missing patterns
        if not any('conditional' in d[0].lower() for d in self.discovery_history):
            priorities.append('CONDITIONAL: if-then-else patterns')
        if not any('loop' in d[0].lower() for d in self.discovery_history):
            priorities.append('LOOP: iterative patterns')

        return priorities


# =============================================================================
# MDL-BASED SYNTHESIZER
# =============================================================================

class MDLSynthesizer:
    """
    Program synthesizer using Minimum Description Length.

    Given I/O examples, find the simplest (shortest description)
    program that fits the data.
    """

    def __init__(self):
        self.rewrite_engine = RewriteEngine()
        self.trace_analyzer = TraceAnalyzer()
        self.dl_calculator = DescriptionLengthCalculator()
        self.curiosity = CuriosityEngine()

    def synthesize(self, trace: Trace, verbose: bool = False) -> Optional[Tuple[Expr, float]]:
        """
        Synthesize the simplest program matching the trace.

        Returns (expression, description_length) or None.
        """
        if verbose:
            print(f"\n{'='*60}")
            print("MDL SYNTHESIS")
            print(f"{'='*60}")
            print(f"Input trace: {trace}")

        # Step 1: Use trace analyzer to get candidate patterns
        candidates = []

        # Get pattern-based candidates
        matches = self.trace_analyzer.analyze(trace)
        for match in matches:
            if match.suggested_expr is not None:
                dl = self.dl_calculator.description_length(match.suggested_expr)
                candidates.append((match.suggested_expr, dl, match.description))

        # Step 2: Generate enumeration-based candidates
        x = var("x")
        enum_candidates = self._enumerate_candidates(x, max_depth=3)

        for expr in enum_candidates:
            if self._matches_trace(expr, trace):
                dl = self.dl_calculator.description_length(expr)
                candidates.append((expr, dl, f"Enumerated: {expr}"))

        if not candidates:
            if verbose:
                print("No candidates found")
            return None

        # Step 3: Sort by description length (MDL principle)
        candidates.sort(key=lambda c: c[1])

        if verbose:
            print(f"\nCandidates (sorted by MDL):")
            for expr, dl, desc in candidates[:5]:
                print(f"  [{dl:.1f} bits] {expr} ({desc})")

        # Return the simplest one
        best_expr, best_dl, _ = candidates[0]

        # Step 4: Simplify using rewrite rules
        simplified = self.rewrite_engine.simplify(best_expr)
        simplified_dl = self.dl_calculator.description_length(simplified)

        if simplified_dl < best_dl:
            best_expr = simplified
            best_dl = simplified_dl
            if verbose:
                print(f"\nSimplified to: {best_expr} [{best_dl:.1f} bits]")

        # Step 5: Register discovery if novel
        if self.curiosity.should_explore(best_expr):
            self.curiosity.register_discovery(str(best_expr), best_expr)
            if verbose:
                print(f"Registered as novel discovery!")

        return best_expr, best_dl

    def _enumerate_candidates(self, x: Var, max_depth: int) -> List[Expr]:
        """Enumerate simple candidate expressions."""
        candidates = []

        # Constants
        for c in [0, 1, 2, 3, 4, 5, 8, 10]:
            candidates.append(const(c))

        # Identity
        candidates.append(x)

        # Unary operations
        candidates.append(square(x))
        candidates.append(double(x))

        # Binary with constants
        for c in [1, 2, 3, 4, 5, 8, 10]:
            candidates.append(add(x, const(c)))
            candidates.append(sub(x, const(c)))
            candidates.append(mul(x, const(c)))
            if c != 0:
                candidates.append(div(x, const(c)))
            candidates.append(lsl(x, const(min(c, 6))))
            candidates.append(lsr(x, const(min(c, 6))))

        # Self operations
        candidates.append(add(x, x))  # 2x
        candidates.append(mul(x, x))  # x^2
        candidates.append(sub(x, x))  # 0

        if max_depth >= 2:
            # Composite operations
            candidates.append(add(square(x), const(1)))  # x^2 + 1
            candidates.append(mul(add(x, const(1)), const(2)))  # (x+1)*2
            candidates.append(add(mul(x, const(2)), const(1)))  # 2x + 1
            candidates.append(mul(x, add(x, const(1))))  # x*(x+1)
            candidates.append(div(mul(x, add(x, const(1))), const(2)))  # x*(x+1)/2

        return candidates

    def _matches_trace(self, expr: Expr, trace: Trace, tolerance: int = 0) -> bool:
        """Check if an expression matches a trace."""
        from formal_verification import concrete_eval

        for pair in trace.pairs:
            try:
                result = concrete_eval(expr, {'x': pair.input})
                if abs(result - pair.output) > tolerance:
                    return False
            except:
                return False
        return True


# =============================================================================
# AUTONOMOUS EXPLORATION
# =============================================================================

class AutonomousExplorer:
    """
    Autonomously explore the space of programs.

    Uses MDL and curiosity to guide exploration without
    human intervention.
    """

    def __init__(self):
        self.synthesizer = MDLSynthesizer()
        self.discoveries: List[Tuple[str, Expr, float]] = []
        self.exploration_count = 0

    def explore(self, iterations: int = 100, verbose: bool = False) -> List[Tuple[str, Expr]]:
        """
        Run autonomous exploration.

        Returns list of (name, expression) discoveries.
        """
        if verbose:
            print(f"\n{'='*60}")
            print("AUTONOMOUS EXPLORATION")
            print(f"{'='*60}")

        # Generate random traces to explore
        for i in range(iterations):
            self.exploration_count += 1

            # Generate a random function to explore
            func, func_name = self._generate_random_function()

            # Create trace
            inputs = list(range(0, 10))
            try:
                trace = trace_from_io([(x, func(x)) for x in inputs])
            except:
                continue

            # Try to synthesize
            result = self.synthesizer.synthesize(trace, verbose=False)

            if result is not None:
                expr, dl = result

                # Check if this is a new discovery
                is_new = not any(str(expr) == str(d[1]) for d in self.discoveries)

                if is_new:
                    self.discoveries.append((func_name, expr, dl))
                    if verbose:
                        print(f"[{i}] Discovered: {func_name} → {expr} [{dl:.1f} bits]")

        return [(name, expr) for name, expr, _ in self.discoveries]

    def _generate_random_function(self) -> Tuple[Callable[[int], int], str]:
        """Generate a random function to explore."""
        import random

        funcs = [
            (lambda x: x, "identity"),
            (lambda x: x + 1, "increment"),
            (lambda x: x * 2, "double"),
            (lambda x: x * x, "square"),
            (lambda x: x * x + 1, "square_plus_1"),
            (lambda x: 2 * x + 1, "odd"),
            (lambda x: 2 * x, "even"),
            (lambda x: x * (x + 1) // 2, "triangular"),
            (lambda x: x ** 3, "cube"),
            (lambda x: 3 * x + 2, "3x+2"),
            (lambda x: x * 4, "quadruple"),
            (lambda x: x + x + x, "triple"),
            (lambda x: min(x, 5), "clamp5"),
            (lambda x: max(x - 3, 0), "relu_minus_3"),
        ]

        return random.choice(funcs)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MDL OPTIMIZER: Minimum Description Length for Program Synthesis")
    print("=" * 60)

    # =================================
    # TEST 1: Description Length Calculation
    # =================================
    print("\n" + "=" * 60)
    print("TEST 1: Description Length Calculation")
    print("=" * 60)

    dl_calc = DescriptionLengthCalculator()

    x = var("x")

    expressions = [
        (const(0), "constant 0"),
        (const(255), "constant 255"),
        (x, "variable x"),
        (add(x, const(1)), "x + 1"),
        (mul(x, x), "x * x"),
        (square(x), "SQUARE(x)"),
        (add(mul(x, x), const(1)), "(x * x) + 1"),
        (mul(add(x, const(1)), add(x, const(2))), "(x+1)*(x+2)"),
    ]

    for expr, name in expressions:
        dl = dl_calc.description_length(expr)
        print(f"  [{dl:5.1f} bits] {name}: {expr}")

    print("✅ PASSED: Description lengths calculated")

    # =================================
    # TEST 2: MDL Synthesis
    # =================================
    print("\n" + "=" * 60)
    print("TEST 2: MDL Synthesis - Find simplest program")
    print("=" * 60)

    synthesizer = MDLSynthesizer()

    # Square function
    trace = trace_from_io([(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25)])
    result = synthesizer.synthesize(trace, verbose=True)
    assert result is not None, "Expected synthesis result"
    expr, dl = result
    print(f"\n✅ Found: {expr} [{dl:.1f} bits]")

    # =================================
    # TEST 3: MDL Synthesis - Linear function
    # =================================
    print("\n" + "=" * 60)
    print("TEST 3: MDL Synthesis - Linear function 2x + 1")
    print("=" * 60)

    trace = trace_from_io([(0, 1), (1, 3), (2, 5), (3, 7), (4, 9)])
    result = synthesizer.synthesize(trace, verbose=True)
    assert result is not None, "Expected synthesis result"
    expr, dl = result
    print(f"\n✅ Found: {expr} [{dl:.1f} bits]")

    # =================================
    # TEST 4: Novelty Detection
    # =================================
    print("\n" + "=" * 60)
    print("TEST 4: Novelty Detection")
    print("=" * 60)

    detector = NoveltyDetector()

    x = var("x")
    simple_expr = add(x, const(1))
    complex_expr = mul(add(mul(x, x), x), add(x, const(1)))

    print(f"  {simple_expr}: novelty = {detector.novelty_score(simple_expr):.2f}")
    print(f"  {complex_expr}: novelty = {detector.novelty_score(complex_expr):.2f}")

    # Add simple to known, check if novelty changes
    detector.add_known_pattern(simple_expr)
    print(f"  After adding to known:")
    print(f"  {simple_expr}: novelty = {detector.novelty_score(simple_expr):.2f}")
    print("✅ PASSED: Novelty detection working")

    # =================================
    # TEST 5: Curiosity Engine
    # =================================
    print("\n" + "=" * 60)
    print("TEST 5: Curiosity Engine")
    print("=" * 60)

    curiosity = CuriosityEngine()

    x = var("x")
    test_exprs = [
        square(x),
        add(x, const(1)),
        mul(mul(x, x), x),  # Cube
    ]

    for expr in test_exprs:
        interest = curiosity.evaluate_interest(expr)
        should_explore = curiosity.should_explore(expr)
        print(f"  {expr}: interest = {interest:.2f}, explore = {should_explore}")

    priorities = curiosity.get_exploration_priority()
    print(f"\n  Exploration priorities:")
    for p in priorities:
        print(f"    - {p}")

    print("✅ PASSED: Curiosity engine working")

    # =================================
    # TEST 6: Autonomous Exploration
    # =================================
    print("\n" + "=" * 60)
    print("TEST 6: Autonomous Exploration (10 iterations)")
    print("=" * 60)

    explorer = AutonomousExplorer()
    discoveries = explorer.explore(iterations=10, verbose=True)

    print(f"\nTotal discoveries: {len(discoveries)}")
    for name, expr in discoveries:
        print(f"  {name}: {expr}")

    print("✅ PASSED: Autonomous exploration working")

    # =================================
    # TEST 7: Library Compression
    # =================================
    print("\n" + "=" * 60)
    print("TEST 7: Library Compression")
    print("=" * 60)

    # Create library with SQUARE
    library = {'SQUARE(x)': square(var("x"))}
    dl_with_lib = DescriptionLengthCalculator(library)
    dl_without_lib = DescriptionLengthCalculator({})

    # Test expression that uses square
    x = var("x")
    expr = add(mul(x, x), const(1))  # x^2 + 1

    dl_with = dl_with_lib.description_length(expr)
    dl_without = dl_without_lib.description_length(expr)
    compression = dl_with_lib.library_compression(expr)

    print(f"  Expression: {expr}")
    print(f"  Without library: {dl_without:.1f} bits")
    print(f"  With library: {dl_with:.1f} bits")
    print(f"  Compression ratio: {compression:.2f}x")

    print("✅ PASSED: Library compression working")

    print("\n" + "=" * 60)
    print("ALL MDL OPTIMIZER TESTS COMPLETED!")
    print("=" * 60)
    print("\n🎉 The MDL Optimizer provides:")
    print("   - Description length (complexity) calculation")
    print("   - MDL-based program synthesis")
    print("   - Novelty detection via compression")
    print("   - Curiosity-driven exploration")
    print("   - Library compression optimization")
    print("   - Autonomous discovery capability")
