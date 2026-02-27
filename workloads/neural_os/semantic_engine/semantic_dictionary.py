#!/usr/bin/env python3
"""
SEMANTIC DICTIONARY: Operations as Algebraic Objects

This is Phase 1 of the Semantic Synthesizer based on the 5-AI Hybrid Review.
Each KVRM operation is defined not just as executable code, but as a mathematical
object with properties that enable automatic discovery.

Key insight from Claude/Gemini: "The system fails at x*x because it sees x and *
as tokens, not mathematical entities. Operations must know their algebraic properties."
"""

from dataclasses import dataclass, field
from typing import List, Set, Callable, Optional, Dict, Any, Tuple
from enum import Enum, auto
from abc import ABC, abstractmethod


class AlgebraicProperty(Enum):
    """Mathematical properties that operations can have."""
    COMMUTATIVE = auto()      # a ⊕ b = b ⊕ a
    ASSOCIATIVE = auto()      # (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
    IDEMPOTENT = auto()       # a ⊕ a = a
    IDENTITY_EXISTS = auto()  # ∃e: a ⊕ e = a
    INVERSE_EXISTS = auto()   # ∃a⁻¹: a ⊕ a⁻¹ = e
    DISTRIBUTIVE_OVER = auto()  # a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
    ABSORBING = auto()        # a ⊕ 0 = 0 (for multiplication)
    SELF_INVERSE = auto()     # a ⊕ a = e (like XOR)
    NILPOTENT = auto()        # a^n = 0 for some n
    MONOTONIC = auto()        # a ≤ b → f(a) ≤ f(b)


@dataclass
class SemanticOperation:
    """
    An operation defined as a mathematical object with full semantic metadata.

    This is the core innovation: operations KNOW what they mean mathematically,
    enabling automatic discovery of patterns like x*x → square(x).
    """
    name: str
    symbol: str
    arity: int  # Number of arguments (1=unary, 2=binary)

    # Algebraic properties
    properties: Set[AlgebraicProperty] = field(default_factory=set)

    # Identity element (if exists)
    identity: Optional[int] = None

    # Absorbing element (if exists) - e.g., 0 for multiplication
    absorbing: Optional[int] = None

    # Inverse operation (if exists)
    inverse_op: Optional[str] = None

    # Operations this distributes over
    distributes_over: Set[str] = field(default_factory=set)

    # Special case patterns: (condition, result_op)
    # e.g., for MUL: ("a == b", "SQUARE") means MUL(a,a) → SQUARE(a)
    special_cases: List[Tuple[str, str]] = field(default_factory=list)

    # Simplification rules: (pattern, simplified)
    # e.g., ("a + 0", "a") means ADD(a, 0) → a
    simplifications: List[Tuple[str, str]] = field(default_factory=list)

    # The actual computation function
    compute: Optional[Callable] = None

    # Type signature for formal verification
    type_signature: str = "int -> int -> int"

    def has_property(self, prop: AlgebraicProperty) -> bool:
        return prop in self.properties

    def is_commutative(self) -> bool:
        return AlgebraicProperty.COMMUTATIVE in self.properties

    def is_associative(self) -> bool:
        return AlgebraicProperty.ASSOCIATIVE in self.properties


# =============================================================================
# SEMANTIC DICTIONARY: All KVRM operations with full algebraic metadata
# =============================================================================

SEMANTIC_DICTIONARY: Dict[str, SemanticOperation] = {}


def register_operation(op: SemanticOperation) -> SemanticOperation:
    """Register an operation in the semantic dictionary."""
    SEMANTIC_DICTIONARY[op.name] = op
    return op


# -----------------------------------------------------------------------------
# ARITHMETIC OPERATIONS
# -----------------------------------------------------------------------------

ADD = register_operation(SemanticOperation(
    name="ADD",
    symbol="+",
    arity=2,
    properties={
        AlgebraicProperty.COMMUTATIVE,
        AlgebraicProperty.ASSOCIATIVE,
        AlgebraicProperty.IDENTITY_EXISTS,
        AlgebraicProperty.INVERSE_EXISTS,
    },
    identity=0,
    inverse_op="SUB",
    distributes_over=set(),  # Nothing distributes ADD over anything
    special_cases=[
        ("a == 0", "IDENTITY"),      # a + 0 = a
        ("b == 0", "IDENTITY"),      # 0 + b = b
        ("a == b", "DOUBLE"),        # a + a = 2*a (DOUBLE)
    ],
    simplifications=[
        ("a + 0", "a"),
        ("0 + a", "a"),
        ("a + (-a)", "0"),
        ("a + a", "MUL(a, 2)"),      # Key insight: a+a = 2a
    ],
    compute=lambda a, b: (a + b) & ((1 << 64) - 1),
    type_signature="int -> int -> int"
))


SUB = register_operation(SemanticOperation(
    name="SUB",
    symbol="-",
    arity=2,
    properties={
        AlgebraicProperty.IDENTITY_EXISTS,  # a - 0 = a
    },
    identity=0,  # Right identity only
    inverse_op="ADD",
    special_cases=[
        ("b == 0", "IDENTITY"),      # a - 0 = a
        ("a == b", "ZERO"),          # a - a = 0
    ],
    simplifications=[
        ("a - 0", "a"),
        ("a - a", "0"),
        ("0 - a", "NEG(a)"),
    ],
    compute=lambda a, b: (a - b) & ((1 << 64) - 1),
    type_signature="int -> int -> int"
))


MUL = register_operation(SemanticOperation(
    name="MUL",
    symbol="*",
    arity=2,
    properties={
        AlgebraicProperty.COMMUTATIVE,
        AlgebraicProperty.ASSOCIATIVE,
        AlgebraicProperty.IDENTITY_EXISTS,
        AlgebraicProperty.ABSORBING,
    },
    identity=1,
    absorbing=0,
    inverse_op="DIV",
    distributes_over={"ADD", "SUB"},  # a*(b+c) = a*b + a*c
    special_cases=[
        ("a == 1", "IDENTITY_B"),    # 1 * b = b
        ("b == 1", "IDENTITY_A"),    # a * 1 = a
        ("a == 0", "ZERO"),          # 0 * b = 0
        ("b == 0", "ZERO"),          # a * 0 = 0
        ("a == b", "SQUARE"),        # a * a = a² ← KEY DISCOVERY!
        ("b == 2", "DOUBLE"),        # a * 2 = a + a
        ("is_power_of_2(b)", "LSL"), # a * 2^n = a << n
    ],
    simplifications=[
        ("a * 0", "0"),
        ("0 * a", "0"),
        ("a * 1", "a"),
        ("1 * a", "a"),
        ("a * a", "SQUARE(a)"),      # THE KEY INSIGHT
        ("a * 2", "ADD(a, a)"),
        ("a * 2", "LSL(a, 1)"),
    ],
    compute=lambda a, b: (a * b) & ((1 << 64) - 1),
    type_signature="int -> int -> int"
))


DIV = register_operation(SemanticOperation(
    name="DIV",
    symbol="/",
    arity=2,
    properties={
        AlgebraicProperty.IDENTITY_EXISTS,  # a / 1 = a
    },
    identity=1,  # Right identity only
    inverse_op="MUL",
    special_cases=[
        ("b == 1", "IDENTITY"),      # a / 1 = a
        ("a == b", "ONE"),           # a / a = 1
        ("a == 0", "ZERO"),          # 0 / b = 0
        ("is_power_of_2(b)", "LSR"), # a / 2^n = a >> n
    ],
    simplifications=[
        ("a / 1", "a"),
        ("a / a", "1"),
        ("0 / a", "0"),
    ],
    compute=lambda a, b: a // b if b != 0 else 0,
    type_signature="int -> int -> int"
))


# -----------------------------------------------------------------------------
# DERIVED OPERATIONS (discovered through algebraic rules)
# -----------------------------------------------------------------------------

SQUARE = register_operation(SemanticOperation(
    name="SQUARE",
    symbol="²",
    arity=1,
    properties={
        AlgebraicProperty.MONOTONIC,  # For non-negative integers
    },
    special_cases=[
        ("a == 0", "ZERO"),
        ("a == 1", "ONE"),
    ],
    simplifications=[
        ("SQUARE(0)", "0"),
        ("SQUARE(1)", "1"),
    ],
    compute=lambda a: (a * a) & ((1 << 64) - 1),
    type_signature="int -> int"
))


DOUBLE = register_operation(SemanticOperation(
    name="DOUBLE",
    symbol="2×",
    arity=1,
    properties={
        AlgebraicProperty.MONOTONIC,
    },
    special_cases=[
        ("a == 0", "ZERO"),
    ],
    simplifications=[
        ("DOUBLE(0)", "0"),
    ],
    compute=lambda a: (a * 2) & ((1 << 64) - 1),
    type_signature="int -> int"
))


NEG = register_operation(SemanticOperation(
    name="NEG",
    symbol="-",
    arity=1,
    properties={
        AlgebraicProperty.SELF_INVERSE,  # NEG(NEG(a)) = a
    },
    special_cases=[
        ("a == 0", "ZERO"),
    ],
    simplifications=[
        ("NEG(NEG(a))", "a"),
        ("NEG(0)", "0"),
    ],
    compute=lambda a: (-a) & ((1 << 64) - 1),
    type_signature="int -> int"
))


# -----------------------------------------------------------------------------
# LOGICAL OPERATIONS
# -----------------------------------------------------------------------------

AND = register_operation(SemanticOperation(
    name="AND",
    symbol="&",
    arity=2,
    properties={
        AlgebraicProperty.COMMUTATIVE,
        AlgebraicProperty.ASSOCIATIVE,
        AlgebraicProperty.IDEMPOTENT,      # a & a = a
        AlgebraicProperty.IDENTITY_EXISTS,
        AlgebraicProperty.ABSORBING,
    },
    identity=(1 << 64) - 1,  # All 1s
    absorbing=0,
    distributes_over={"OR"},  # a & (b | c) = (a & b) | (a & c)
    special_cases=[
        ("a == b", "IDENTITY"),      # a & a = a
        ("b == 0", "ZERO"),          # a & 0 = 0
        ("b == ALL_ONES", "IDENTITY_A"),  # a & ~0 = a
    ],
    simplifications=[
        ("a & a", "a"),
        ("a & 0", "0"),
        ("a & ~0", "a"),
    ],
    compute=lambda a, b: a & b,
    type_signature="int -> int -> int"
))


OR = register_operation(SemanticOperation(
    name="OR",
    symbol="|",
    arity=2,
    properties={
        AlgebraicProperty.COMMUTATIVE,
        AlgebraicProperty.ASSOCIATIVE,
        AlgebraicProperty.IDEMPOTENT,      # a | a = a
        AlgebraicProperty.IDENTITY_EXISTS,
        AlgebraicProperty.ABSORBING,
    },
    identity=0,
    absorbing=(1 << 64) - 1,  # All 1s
    distributes_over={"AND"},  # a | (b & c) = (a | b) & (a | c)
    special_cases=[
        ("a == b", "IDENTITY"),      # a | a = a
        ("b == 0", "IDENTITY_A"),    # a | 0 = a
        ("b == ALL_ONES", "ALL_ONES"),  # a | ~0 = ~0
    ],
    simplifications=[
        ("a | a", "a"),
        ("a | 0", "a"),
    ],
    compute=lambda a, b: a | b,
    type_signature="int -> int -> int"
))


XOR = register_operation(SemanticOperation(
    name="XOR",
    symbol="^",
    arity=2,
    properties={
        AlgebraicProperty.COMMUTATIVE,
        AlgebraicProperty.ASSOCIATIVE,
        AlgebraicProperty.IDENTITY_EXISTS,
        AlgebraicProperty.SELF_INVERSE,    # a ^ a = 0
    },
    identity=0,
    special_cases=[
        ("a == b", "ZERO"),          # a ^ a = 0
        ("b == 0", "IDENTITY_A"),    # a ^ 0 = a
    ],
    simplifications=[
        ("a ^ a", "0"),
        ("a ^ 0", "a"),
        ("a ^ ~0", "NOT(a)"),
    ],
    compute=lambda a, b: a ^ b,
    type_signature="int -> int -> int"
))


NOT = register_operation(SemanticOperation(
    name="NOT",
    symbol="~",
    arity=1,
    properties={
        AlgebraicProperty.SELF_INVERSE,    # NOT(NOT(a)) = a
    },
    special_cases=[],
    simplifications=[
        ("NOT(NOT(a))", "a"),
    ],
    compute=lambda a: ~a & ((1 << 64) - 1),
    type_signature="int -> int"
))


# -----------------------------------------------------------------------------
# SHIFT OPERATIONS
# -----------------------------------------------------------------------------

LSL = register_operation(SemanticOperation(
    name="LSL",
    symbol="<<",
    arity=2,
    properties={},
    special_cases=[
        ("b == 0", "IDENTITY"),      # a << 0 = a
        ("a == 0", "ZERO"),          # 0 << b = 0
        ("b == 1", "DOUBLE"),        # a << 1 = 2*a
    ],
    simplifications=[
        ("a << 0", "a"),
        ("0 << b", "0"),
        ("a << 1", "DOUBLE(a)"),
        ("a << 1", "MUL(a, 2)"),
    ],
    compute=lambda a, b: (a << (b & 63)) & ((1 << 64) - 1),
    type_signature="int -> int -> int"
))


LSR = register_operation(SemanticOperation(
    name="LSR",
    symbol=">>",
    arity=2,
    properties={},
    special_cases=[
        ("b == 0", "IDENTITY"),      # a >> 0 = a
        ("a == 0", "ZERO"),          # 0 >> b = 0
        ("b == 1", "HALF"),          # a >> 1 = a/2
    ],
    simplifications=[
        ("a >> 0", "a"),
        ("0 >> b", "0"),
        ("a >> 1", "DIV(a, 2)"),
    ],
    compute=lambda a, b: a >> (b & 63),
    type_signature="int -> int -> int"
))


# -----------------------------------------------------------------------------
# COMPARISON OPERATIONS
# -----------------------------------------------------------------------------

CMP = register_operation(SemanticOperation(
    name="CMP",
    symbol="<=>",
    arity=2,
    properties={},
    special_cases=[
        ("a == b", "EQUAL"),         # CMP(a, a) → EQUAL
    ],
    simplifications=[],
    compute=lambda a, b: (1 if a > b else (-1 if a < b else 0)),
    type_signature="int -> int -> Ordering"
))


# -----------------------------------------------------------------------------
# CONSTANT GENERATORS
# -----------------------------------------------------------------------------

ZERO = register_operation(SemanticOperation(
    name="ZERO",
    symbol="0",
    arity=0,
    properties={},
    special_cases=[],
    simplifications=[],
    compute=lambda: 0,
    type_signature="() -> int"
))


ONE = register_operation(SemanticOperation(
    name="ONE",
    symbol="1",
    arity=0,
    properties={},
    special_cases=[],
    simplifications=[],
    compute=lambda: 1,
    type_signature="() -> int"
))


# =============================================================================
# SEMANTIC QUERIES
# =============================================================================

def get_operation(name: str) -> Optional[SemanticOperation]:
    """Get an operation by name."""
    return SEMANTIC_DICTIONARY.get(name)


def find_operations_with_property(prop: AlgebraicProperty) -> List[SemanticOperation]:
    """Find all operations with a given algebraic property."""
    return [op for op in SEMANTIC_DICTIONARY.values() if prop in op.properties]


def find_commutative_operations() -> List[SemanticOperation]:
    """Find all commutative operations."""
    return find_operations_with_property(AlgebraicProperty.COMMUTATIVE)


def find_operations_distributing_over(op_name: str) -> List[SemanticOperation]:
    """Find all operations that distribute over the given operation."""
    return [op for op in SEMANTIC_DICTIONARY.values() if op_name in op.distributes_over]


def check_special_case(op_name: str, a: Any, b: Any = None) -> Optional[str]:
    """
    Check if an operation application matches a special case.

    This is how we discover things like MUL(x, x) → SQUARE(x).
    """
    op = SEMANTIC_DICTIONARY.get(op_name)
    if not op:
        return None

    for condition, result in op.special_cases:
        # Simple condition evaluation
        if condition == "a == b" and a == b:
            return result
        elif condition == "a == 0" and a == 0:
            return result
        elif condition == "b == 0" and b == 0:
            return result
        elif condition == "a == 1" and a == 1:
            return result
        elif condition == "b == 1" and b == 1:
            return result
        elif condition == "b == 2" and b == 2:
            return result
        elif condition.startswith("is_power_of_2") and b is not None:
            if b > 0 and (b & (b - 1)) == 0:
                return result

    return None


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC DICTIONARY: Operations as Algebraic Objects")
    print("=" * 60)

    print(f"\nRegistered {len(SEMANTIC_DICTIONARY)} operations:")
    for name, op in SEMANTIC_DICTIONARY.items():
        props = [p.name for p in op.properties]
        print(f"  {name} ({op.symbol}): {', '.join(props) if props else 'no properties'}")

    print("\n" + "=" * 60)
    print("COMMUTATIVE OPERATIONS:")
    for op in find_commutative_operations():
        print(f"  {op.name}: {op.symbol}")

    print("\n" + "=" * 60)
    print("OPERATIONS DISTRIBUTING OVER ADD:")
    for op in find_operations_distributing_over("ADD"):
        print(f"  {op.name}: {op.symbol}")

    print("\n" + "=" * 60)
    print("SPECIAL CASE DETECTION:")

    # The key test: Does MUL(x, x) get recognized as SQUARE?
    result = check_special_case("MUL", "x", "x")
    print(f"  MUL(x, x) → {result}")  # Should print SQUARE

    result = check_special_case("ADD", "x", "x")
    print(f"  ADD(x, x) → {result}")  # Should print DOUBLE

    result = check_special_case("SUB", "x", "x")
    print(f"  SUB(x, x) → {result}")  # Should print ZERO

    result = check_special_case("XOR", "x", "x")
    print(f"  XOR(x, x) → {result}")  # Should print ZERO

    result = check_special_case("MUL", 5, 0)
    print(f"  MUL(5, 0) → {result}")  # Should print ZERO

    result = check_special_case("MUL", 7, 8)  # 8 is power of 2
    print(f"  MUL(7, 8) → {result}")  # Should print LSL

    print("\n✅ Semantic Dictionary ready for Rewrite Engine!")
