#!/usr/bin/env python3
"""
EPISTEMIC FRONTIER (EF): Layer 4 - Discovering Unknown Unknowns

This implements Grok's Layer 4 recommendation:
"Epistemic Frontier (EF) - Discovers 'unknown unknowns' via cross-domain bisociation"

Key Grok insights:
- "Bisociation: Connecting unrelated conceptual domains (Koestler's theory)"
- "Psychedelic Bisociation Engine: Graph scrambling + reconnection"
- "Cross-domain mapping: Seeing analogies between unrelated areas"
- "Modal Program Logic: Necessity vs possibility; self-defines 'novel'"

This module provides:
1. Cross-domain analogy detection
2. Conceptual blending of program patterns
3. Unknown-unknown discovery through bisociation
4. Self-directed goal generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
import random
import math
from collections import defaultdict
import itertools

from rewrite_engine import (
    Expr, Var, Const, App,
    var, const, add, sub, mul, div, square, double,
    and_, or_, xor, lsl, lsr
)
from semantic_dictionary import SEMANTIC_DICTIONARY, AlgebraicProperty, get_operation
from trace_analyzer import Trace, trace_from_io, PatternType


# =============================================================================
# CONCEPTUAL DOMAINS
# =============================================================================

class ConceptualDomain(Enum):
    """Different domains of computation."""
    ARITHMETIC = auto()     # +, -, *, /
    LOGICAL = auto()        # AND, OR, XOR, NOT
    BITWISE = auto()        # LSL, LSR, bit manipulation
    COMPARISON = auto()     # <, >, ==, CMP
    STRUCTURAL = auto()     # Loops, conditionals, recursion
    MATHEMATICAL = auto()   # Square, factorial, fibonacci
    COMBINATORIAL = auto()  # Permutations, combinations
    GEOMETRIC = auto()      # Distance, area, volume


@dataclass
class Concept:
    """A computational concept that can be combined with others."""
    name: str
    domain: ConceptualDomain
    operation: Optional[Expr] = None
    properties: Set[str] = field(default_factory=set)
    analogies: List[str] = field(default_factory=list)
    description: str = ""

    def __repr__(self):
        return f"Concept({self.name}, {self.domain.name})"


# =============================================================================
# BISOCIATION ENGINE
# =============================================================================

class BisociationEngine:
    """
    Creates novel concepts by connecting unrelated domains.

    Arthur Koestler's Bisociation Theory:
    "The creative act consists in combining previously unrelated
    structures in such a way that you get more out of the emergent
    whole than you have put in."

    Grok: "Psychedelic Bisociation Engine: Graph scrambling + reconnection"
    """

    def __init__(self):
        self.discovered_blends: List[Tuple[str, str, Expr]] = []
        self.analogy_map: Dict[str, List[str]] = defaultdict(list)
        self.concepts = self._initialize_concepts()

    def _initialize_concepts(self) -> Dict[str, Concept]:
        """Initialize base concepts from each domain."""
        concepts = {}

        # Arithmetic concepts
        x = var("x")
        y = var("y")

        concepts['addition'] = Concept(
            name='addition',
            domain=ConceptualDomain.ARITHMETIC,
            operation=add(x, y),
            properties={'commutative', 'associative', 'has_identity'},
            analogies=['union', 'merge', 'accumulate'],
            description='Combining quantities'
        )

        concepts['multiplication'] = Concept(
            name='multiplication',
            domain=ConceptualDomain.ARITHMETIC,
            operation=mul(x, y),
            properties={'commutative', 'associative', 'has_identity', 'distributes'},
            analogies=['scaling', 'repetition', 'area'],
            description='Repeated addition or scaling'
        )

        concepts['subtraction'] = Concept(
            name='subtraction',
            domain=ConceptualDomain.ARITHMETIC,
            operation=sub(x, y),
            properties={'has_identity', 'has_inverse'},
            analogies=['difference', 'remove', 'distance'],
            description='Finding difference'
        )

        # Logical concepts
        concepts['conjunction'] = Concept(
            name='conjunction',
            domain=ConceptualDomain.LOGICAL,
            operation=and_(x, y),
            properties={'commutative', 'associative', 'idempotent'},
            analogies=['intersection', 'filter', 'constraint'],
            description='Both conditions true'
        )

        concepts['disjunction'] = Concept(
            name='disjunction',
            domain=ConceptualDomain.LOGICAL,
            operation=or_(x, y),
            properties={'commutative', 'associative', 'idempotent'},
            analogies=['union', 'choice', 'fallback'],
            description='Either condition true'
        )

        concepts['exclusive'] = Concept(
            name='exclusive',
            domain=ConceptualDomain.LOGICAL,
            operation=xor(x, y),
            properties={'commutative', 'associative', 'self_inverse'},
            analogies=['toggle', 'difference', 'switch'],
            description='Exactly one true'
        )

        # Bitwise concepts
        concepts['left_shift'] = Concept(
            name='left_shift',
            domain=ConceptualDomain.BITWISE,
            operation=lsl(x, const(1)),
            properties={'doubles'},
            analogies=['double', 'multiply_power2', 'grow'],
            description='Shift bits left (multiply by 2)'
        )

        concepts['right_shift'] = Concept(
            name='right_shift',
            domain=ConceptualDomain.BITWISE,
            operation=lsr(x, const(1)),
            properties={'halves'},
            analogies=['halve', 'divide_power2', 'shrink'],
            description='Shift bits right (divide by 2)'
        )

        # Mathematical concepts
        concepts['squaring'] = Concept(
            name='squaring',
            domain=ConceptualDomain.MATHEMATICAL,
            operation=square(x),
            properties={'self_multiplication', 'quadratic'},
            analogies=['area', 'power', 'self_reference'],
            description='Multiply by self'
        )

        concepts['doubling'] = Concept(
            name='doubling',
            domain=ConceptualDomain.MATHEMATICAL,
            operation=double(x),
            properties={'linear', 'scaling'},
            analogies=['duplicate', 'mirror', 'echo'],
            description='Add to self'
        )

        # Build analogy map
        for name, concept in concepts.items():
            for analogy in concept.analogies:
                self.analogy_map[analogy].append(name)

        return concepts

    def find_analogies(self, concept_name: str) -> List[Concept]:
        """Find concepts analogous to the given one."""
        if concept_name not in self.concepts:
            return []

        concept = self.concepts[concept_name]
        analogous_names = set()

        for analogy in concept.analogies:
            for other_name in self.analogy_map[analogy]:
                if other_name != concept_name:
                    analogous_names.add(other_name)

        return [self.concepts[n] for n in analogous_names]

    def bisociate(self, concept1: str, concept2: str) -> Optional[Expr]:
        """
        Create a novel blend by bisociating two concepts.

        This is the core creative act: finding unexpected connections
        between unrelated domains.
        """
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return None

        c1 = self.concepts[concept1]
        c2 = self.concepts[concept2]

        # Strategy 1: Compose operations
        if c1.operation and c2.operation:
            blend = self._compose_operations(c1, c2)
            if blend:
                self.discovered_blends.append((concept1, concept2, blend))
                return blend

        # Strategy 2: Transfer properties
        blend = self._transfer_properties(c1, c2)
        if blend:
            self.discovered_blends.append((concept1, concept2, blend))
            return blend

        return None

    def _compose_operations(self, c1: Concept, c2: Concept) -> Optional[Expr]:
        """Compose two operations in various ways."""
        x = var("x")

        if c1.operation is None or c2.operation is None:
            return None

        compositions = []

        # Nested composition: c1(c2(x))
        # Simplified for demonstration
        if c1.name == 'squaring' and c2.name == 'addition':
            # (x + something)^2
            compositions.append(square(add(x, const(1))))

        if c1.name == 'doubling' and c2.name == 'squaring':
            # 2 * x^2
            compositions.append(mul(const(2), square(x)))

        if c1.name == 'left_shift' and c2.name == 'multiplication':
            # Insight: shifting left is multiplying by 2
            compositions.append(mul(x, const(2)))

        if c1.name == 'exclusive' and c2.name == 'subtraction':
            # XOR can compute difference in some contexts
            compositions.append(xor(x, var("y")))

        if c1.name == 'multiplication' and c2.name == 'squaring':
            # Cubing: x * x^2
            compositions.append(mul(x, square(x)))

        return compositions[0] if compositions else None

    def _transfer_properties(self, c1: Concept, c2: Concept) -> Optional[Expr]:
        """Transfer properties from one domain to another."""
        x = var("x")

        # If c1 has 'self_inverse' (like XOR), look for similar in c2's domain
        if 'self_inverse' in c1.properties:
            if c2.domain == ConceptualDomain.ARITHMETIC:
                # Subtraction of self is self-inverse: x - x = 0
                return sub(x, x)

        # If c1 has 'distributes', apply distributive law
        if 'distributes' in c1.properties:
            if c2.domain == ConceptualDomain.LOGICAL:
                # AND distributes over OR: a & (b | c) = (a & b) | (a & c)
                pass

        return None

    def random_bisociation(self) -> Tuple[str, str, Optional[Expr]]:
        """Perform random bisociation between two concepts."""
        names = list(self.concepts.keys())
        c1, c2 = random.sample(names, 2)
        blend = self.bisociate(c1, c2)
        return c1, c2, blend

    def explore_bisociations(self, n: int = 50) -> List[Tuple[str, str, Expr]]:
        """Explore many random bisociations."""
        discoveries = []
        for _ in range(n):
            c1, c2, blend = self.random_bisociation()
            if blend is not None:
                discoveries.append((c1, c2, blend))
        return discoveries


# =============================================================================
# UNKNOWN UNKNOWN DETECTOR
# =============================================================================

class UnknownUnknownDetector:
    """
    Detects gaps in knowledge - things we don't know we don't know.

    Grok: "The 'Unknown Unknowns' Problem: How does the system discover
    things it doesn't know it should be looking for?"
    """

    def __init__(self):
        self.known_patterns: Set[str] = set()
        self.known_gaps: Set[str] = set()
        self.exploration_frontier: List[str] = []

    def register_known(self, pattern: str):
        """Register a known pattern."""
        self.known_patterns.add(pattern)

    def detect_gaps(self, all_operations: List[str]) -> List[str]:
        """
        Detect gaps in our knowledge by analyzing what's missing.

        Uses "negative space" reasoning - what patterns SHOULD exist
        but don't?
        """
        gaps = []

        # Gap detection strategies:

        # 1. Missing inverses
        op_map = {op: get_operation(op) for op in all_operations}
        for op_name, op in op_map.items():
            if op and op.inverse_op:
                if op.inverse_op not in self.known_patterns:
                    gaps.append(f"missing_inverse:{op.inverse_op}")

        # 2. Missing compositions
        binary_ops = [op for op in all_operations
                      if get_operation(op) and get_operation(op).arity == 2]
        for op1, op2 in itertools.combinations(binary_ops, 2):
            composition = f"{op1}_then_{op2}"
            if composition not in self.known_patterns:
                gaps.append(f"unexplored_composition:{composition}")

        # 3. Missing generalizations
        if 'SQUARE' in self.known_patterns and 'CUBE' not in self.known_patterns:
            gaps.append("missing_generalization:POWER_N")
        if 'DOUBLE' in self.known_patterns and 'TRIPLE' not in self.known_patterns:
            gaps.append("missing_generalization:MULTIPLY_N")

        # 4. Missing duals
        duals = [
            ('ADD', 'SUB'), ('MUL', 'DIV'),
            ('AND', 'OR'), ('LSL', 'LSR')
        ]
        for d1, d2 in duals:
            if d1 in self.known_patterns and d2 not in self.known_patterns:
                gaps.append(f"missing_dual:{d2}")
            if d2 in self.known_patterns and d1 not in self.known_patterns:
                gaps.append(f"missing_dual:{d1}")

        self.known_gaps.update(gaps)
        return gaps

    def suggest_explorations(self, n: int = 5) -> List[str]:
        """Suggest what to explore based on detected gaps."""
        if not self.known_gaps:
            # Generate some based on common patterns
            return [
                "explore:higher_order_functions",
                "explore:recursive_patterns",
                "explore:conditional_logic",
                "explore:loop_structures",
                "explore:data_structures"
            ][:n]

        return list(self.known_gaps)[:n]


# =============================================================================
# EPISTEMIC FRONTIER
# =============================================================================

class EpistemicFrontier:
    """
    Layer 4: The Epistemic Frontier.

    This is the highest layer that:
    1. Discovers unknown unknowns
    2. Creates novel concepts through bisociation
    3. Guides self-directed exploration
    4. Defines what counts as "novel"

    Grok: "Self-defines 'novel' as pragmatic utility in simulated worlds"
    """

    def __init__(self):
        self.bisociation_engine = BisociationEngine()
        self.unknown_detector = UnknownUnknownDetector()
        self.discovery_log: List[Dict[str, Any]] = []
        self.novelty_threshold = 0.5
        self.exploration_budget = 100

    def explore_frontier(self, iterations: int = 50,
                         verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Explore the epistemic frontier - looking for unknown unknowns.

        This is autonomous exploration driven by curiosity about
        what we don't know we don't know.
        """
        discoveries = []

        if verbose:
            print(f"\n{'='*60}")
            print("EXPLORING EPISTEMIC FRONTIER")
            print(f"{'='*60}")

        for i in range(iterations):
            # Strategy 1: Bisociation
            c1, c2, blend = self.bisociation_engine.random_bisociation()
            if blend is not None:
                novelty = self._assess_novelty(blend)
                if novelty > self.novelty_threshold:
                    discovery = {
                        'type': 'bisociation',
                        'source1': c1,
                        'source2': c2,
                        'result': str(blend),
                        'novelty': novelty
                    }
                    discoveries.append(discovery)
                    self.discovery_log.append(discovery)
                    if verbose:
                        print(f"  [{i}] Bisociation: {c1} × {c2} → {blend} (novelty={novelty:.2f})")

            # Strategy 2: Gap exploration
            if i % 10 == 0:
                gaps = self.unknown_detector.detect_gaps(
                    list(SEMANTIC_DICTIONARY.keys())
                )
                if gaps and verbose:
                    print(f"  [{i}] Detected gaps: {gaps[:3]}")

        return discoveries

    def _assess_novelty(self, expr: Expr) -> float:
        """
        Assess how novel an expression is.

        Grok: "Modal Program Logic: Necessity (provable algos) vs.
        possibility (exploratory)"
        """
        # Factors in novelty:

        # 1. Structural complexity
        complexity = self._expr_complexity(expr)
        complexity_score = min(1.0, complexity / 10.0)

        # 2. Whether it's a new combination
        expr_str = str(expr)
        seen_similar = any(
            d['result'] == expr_str for d in self.discovery_log
        )
        uniqueness_score = 0.0 if seen_similar else 1.0

        # 3. Cross-domain nature
        domains = self._count_domains(expr)
        cross_domain_score = min(1.0, domains / 3.0)

        # Combine scores
        novelty = (
            0.3 * complexity_score +
            0.4 * uniqueness_score +
            0.3 * cross_domain_score
        )

        return novelty

    def _expr_complexity(self, expr: Expr) -> int:
        """Count the complexity of an expression."""
        if isinstance(expr, (Var, Const)):
            return 1
        if isinstance(expr, App):
            return 1 + sum(self._expr_complexity(a) for a in expr.args)
        return 0

    def _count_domains(self, expr: Expr) -> int:
        """Count how many conceptual domains an expression spans."""
        domains = set()

        def traverse(e):
            if isinstance(e, App):
                op = e.op
                if op in ['ADD', 'SUB', 'MUL', 'DIV']:
                    domains.add('arithmetic')
                elif op in ['AND', 'OR', 'XOR', 'NOT']:
                    domains.add('logical')
                elif op in ['LSL', 'LSR']:
                    domains.add('bitwise')
                elif op in ['SQUARE', 'DOUBLE', 'NEG']:
                    domains.add('mathematical')
                for arg in e.args:
                    traverse(arg)

        traverse(expr)
        return len(domains)

    def generate_goals(self, n: int = 5) -> List[str]:
        """
        Generate self-directed goals for exploration.

        Grok: "Self-directed task generation"
        """
        goals = []

        # Goal 1: Explore detected gaps
        suggestions = self.unknown_detector.suggest_explorations(2)
        goals.extend([f"fill_gap:{s}" for s in suggestions])

        # Goal 2: Extend successful bisociations
        if self.discovery_log:
            recent = self.discovery_log[-3:]
            for d in recent:
                if d['type'] == 'bisociation':
                    goals.append(f"extend:{d['source1']}_{d['source2']}")

        # Goal 3: Explore unexplored domains
        explored_domains = set()
        for d in self.discovery_log:
            if 'source1' in d:
                c = self.bisociation_engine.concepts.get(d['source1'])
                if c:
                    explored_domains.add(c.domain)

        all_domains = set(ConceptualDomain)
        unexplored = all_domains - explored_domains
        for domain in list(unexplored)[:2]:
            goals.append(f"explore_domain:{domain.name}")

        return goals[:n]

    def self_reflect(self) -> Dict[str, Any]:
        """
        Self-reflection on what we've learned.

        Grok: "Monitor own thought processes; evaluate evaluation criteria"
        """
        reflection = {
            'total_discoveries': len(self.discovery_log),
            'bisociations': sum(1 for d in self.discovery_log if d['type'] == 'bisociation'),
            'avg_novelty': sum(d.get('novelty', 0) for d in self.discovery_log) / max(1, len(self.discovery_log)),
            'domains_explored': len(set(
                self.bisociation_engine.concepts.get(d.get('source1', ''), Concept('', ConceptualDomain.ARITHMETIC)).domain
                for d in self.discovery_log
                if 'source1' in d
            )),
            'top_discoveries': sorted(
                self.discovery_log,
                key=lambda d: d.get('novelty', 0),
                reverse=True
            )[:5],
            'current_gaps': list(self.unknown_detector.known_gaps)[:10],
            'next_goals': self.generate_goals()
        }
        return reflection


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EPISTEMIC FRONTIER (EF) - Layer 4")
    print("=" * 60)

    ef = EpistemicFrontier()

    # =================================
    # TEST 1: Bisociation Engine
    # =================================
    print("\n" + "=" * 60)
    print("TEST 1: Bisociation Engine")
    print("=" * 60)

    engine = ef.bisociation_engine

    print("\nAvailable concepts:")
    for name, concept in engine.concepts.items():
        print(f"  {name}: {concept.domain.name} - {concept.description}")

    print("\nFinding analogies for 'multiplication':")
    analogies = engine.find_analogies('multiplication')
    for a in analogies:
        print(f"  → {a.name} ({a.domain.name})")

    print("✅ PASSED: Bisociation engine initialized")

    # =================================
    # TEST 2: Bisociation Creation
    # =================================
    print("\n" + "=" * 60)
    print("TEST 2: Creating Bisociations")
    print("=" * 60)

    blend = engine.bisociate('squaring', 'addition')
    if blend:
        print(f"  squaring × addition = {blend}")

    blend = engine.bisociate('doubling', 'squaring')
    if blend:
        print(f"  doubling × squaring = {blend}")

    blend = engine.bisociate('multiplication', 'squaring')
    if blend:
        print(f"  multiplication × squaring = {blend}")

    print("✅ PASSED: Bisociations created")

    # =================================
    # TEST 3: Unknown Unknown Detection
    # =================================
    print("\n" + "=" * 60)
    print("TEST 3: Unknown Unknown Detection")
    print("=" * 60)

    detector = ef.unknown_detector

    # Register some known patterns
    detector.register_known('ADD')
    detector.register_known('MUL')
    detector.register_known('SQUARE')
    detector.register_known('DOUBLE')

    gaps = detector.detect_gaps(list(SEMANTIC_DICTIONARY.keys()))
    print(f"\nDetected {len(gaps)} gaps in knowledge:")
    for gap in gaps[:5]:
        print(f"  - {gap}")

    suggestions = detector.suggest_explorations()
    print(f"\nSuggested explorations:")
    for s in suggestions:
        print(f"  → {s}")

    print("✅ PASSED: Gap detection works")

    # =================================
    # TEST 4: Frontier Exploration
    # =================================
    print("\n" + "=" * 60)
    print("TEST 4: Frontier Exploration")
    print("=" * 60)

    discoveries = ef.explore_frontier(iterations=30, verbose=True)

    print(f"\nMade {len(discoveries)} novel discoveries")
    print("✅ PASSED: Frontier exploration works")

    # =================================
    # TEST 5: Self-Reflection
    # =================================
    print("\n" + "=" * 60)
    print("TEST 5: Self-Reflection")
    print("=" * 60)

    reflection = ef.self_reflect()

    print(f"\nReflection on learning:")
    print(f"  Total discoveries: {reflection['total_discoveries']}")
    print(f"  Bisociations: {reflection['bisociations']}")
    print(f"  Average novelty: {reflection['avg_novelty']:.2f}")
    print(f"  Domains explored: {reflection['domains_explored']}")

    print(f"\nNext goals:")
    for goal in reflection['next_goals']:
        print(f"  → {goal}")

    print("✅ PASSED: Self-reflection works")

    # =================================
    # TEST 6: Goal Generation
    # =================================
    print("\n" + "=" * 60)
    print("TEST 6: Self-Directed Goal Generation")
    print("=" * 60)

    goals = ef.generate_goals(10)
    print("Generated goals:")
    for goal in goals:
        print(f"  → {goal}")

    print("✅ PASSED: Goal generation works")

    print("\n" + "=" * 60)
    print("ALL EPISTEMIC FRONTIER TESTS COMPLETED!")
    print("=" * 60)
    print("\n🎉 The Epistemic Frontier provides:")
    print("   - Cross-domain bisociation (Koestler's theory)")
    print("   - Unknown-unknown detection")
    print("   - Self-directed goal generation")
    print("   - Novelty assessment")
    print("   - Self-reflection on learning")
