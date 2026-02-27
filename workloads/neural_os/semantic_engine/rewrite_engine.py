#!/usr/bin/env python3
"""
ALGEBRAIC REWRITE ENGINE: Automatic Program Simplification & Discovery

This is the core innovation from Claude's analysis:
"Instead of random mutation, use algebraic rewrite rules. If the system knows
the laws of algebra, it doesn't need to 'search' for x*x; the simplification
rule a * a -> square(a) makes the discovery automatic."

The engine:
1. Takes a program expression tree
2. Applies algebraic rewrite rules exhaustively
3. Discovers patterns like MUL(x,x) → SQUARE(x) automatically
4. Produces the simplest equivalent program
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
import copy

from semantic_dictionary import (
    SEMANTIC_DICTIONARY, SemanticOperation, AlgebraicProperty,
    check_special_case, get_operation
)


# =============================================================================
# EXPRESSION TREE REPRESENTATION
# =============================================================================

@dataclass
class Expr:
    """Base class for expression tree nodes."""
    pass


@dataclass
class Var(Expr):
    """A variable (e.g., x, y)."""
    name: str

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(("Var", self.name))


@dataclass
class Const(Expr):
    """A constant value."""
    value: int

    def __repr__(self):
        return str(self.value)

    def __eq__(self, other):
        return isinstance(other, Const) and self.value == other.value

    def __hash__(self):
        return hash(("Const", self.value))


@dataclass
class App(Expr):
    """Function application (operation applied to arguments)."""
    op: str  # Operation name (e.g., "MUL", "ADD")
    args: List[Expr]

    def __repr__(self):
        if len(self.args) == 1:
            return f"{self.op}({self.args[0]})"
        elif len(self.args) == 2:
            op_info = get_operation(self.op)
            if op_info and op_info.symbol in "+-*/<>&|^":
                return f"({self.args[0]} {op_info.symbol} {self.args[1]})"
            return f"{self.op}({self.args[0]}, {self.args[1]})"
        return f"{self.op}({', '.join(str(a) for a in self.args)})"

    def __eq__(self, other):
        return isinstance(other, App) and self.op == other.op and self.args == other.args

    def __hash__(self):
        return hash(("App", self.op, tuple(self.args)))


# =============================================================================
# REWRITE RULES
# =============================================================================

@dataclass
class RewriteRule:
    """
    A rewrite rule that transforms one expression pattern into another.

    Pattern matching uses:
    - Var("_a"), Var("_b") as pattern variables that match any expression
    - Const values match specific constants
    - App matches operation applications
    """
    name: str
    pattern: Expr       # What to match
    replacement: Expr   # What to replace with
    condition: Optional[str] = None  # Optional condition (e.g., "a == b")

    def __repr__(self):
        return f"{self.name}: {self.pattern} → {self.replacement}"


def match_pattern(pattern: Expr, expr: Expr, bindings: Dict[str, Expr] = None) -> Optional[Dict[str, Expr]]:
    """
    Match an expression against a pattern, returning variable bindings if successful.

    Pattern variables start with "_" (e.g., Var("_a")).
    """
    if bindings is None:
        bindings = {}

    # Pattern variable - matches anything
    if isinstance(pattern, Var) and pattern.name.startswith("_"):
        var_name = pattern.name[1:]  # Remove "_" prefix
        if var_name in bindings:
            # Must match existing binding
            return bindings if bindings[var_name] == expr else None
        else:
            # New binding
            new_bindings = bindings.copy()
            new_bindings[var_name] = expr
            return new_bindings

    # Literal variable - must match exactly
    if isinstance(pattern, Var) and isinstance(expr, Var):
        return bindings if pattern.name == expr.name else None

    # Constant - must match exactly
    if isinstance(pattern, Const) and isinstance(expr, Const):
        return bindings if pattern.value == expr.value else None

    # Application - must match op and all args
    if isinstance(pattern, App) and isinstance(expr, App):
        if pattern.op != expr.op:
            return None
        if len(pattern.args) != len(expr.args):
            return None

        current_bindings = bindings
        for p_arg, e_arg in zip(pattern.args, expr.args):
            result = match_pattern(p_arg, e_arg, current_bindings)
            if result is None:
                return None
            current_bindings = result

        return current_bindings

    return None


def apply_bindings(expr: Expr, bindings: Dict[str, Expr]) -> Expr:
    """Replace pattern variables with their bound values."""
    if isinstance(expr, Var):
        if expr.name.startswith("_"):
            var_name = expr.name[1:]
            if var_name in bindings:
                return copy.deepcopy(bindings[var_name])
        return expr

    if isinstance(expr, Const):
        return expr

    if isinstance(expr, App):
        return App(expr.op, [apply_bindings(arg, bindings) for arg in expr.args])

    return expr


def check_condition(condition: str, bindings: Dict[str, Expr]) -> bool:
    """Check if a rewrite condition is satisfied."""
    if condition is None:
        return True

    if condition == "a == b":
        return bindings.get("a") == bindings.get("b")

    if condition == "a == 0":
        a = bindings.get("a")
        return isinstance(a, Const) and a.value == 0

    if condition == "b == 0":
        b = bindings.get("b")
        return isinstance(b, Const) and b.value == 0

    if condition == "b == 1":
        b = bindings.get("b")
        return isinstance(b, Const) and b.value == 1

    if condition == "b == 2":
        b = bindings.get("b")
        return isinstance(b, Const) and b.value == 2

    if condition == "is_power_of_2(b)":
        b = bindings.get("b")
        if isinstance(b, Const) and b.value > 0:
            return (b.value & (b.value - 1)) == 0
        return False

    return True


# =============================================================================
# BUILT-IN REWRITE RULES (from Semantic Dictionary)
# =============================================================================

def build_rewrite_rules() -> List[RewriteRule]:
    """
    Build rewrite rules from the semantic dictionary.

    This is where the magic happens: algebraic properties become
    automatic program transformations.
    """
    rules = []

    # Pattern variables
    _a = Var("_a")
    _b = Var("_b")

    # =================================
    # IDENTITY RULES
    # =================================

    # ADD identity: a + 0 = a
    rules.append(RewriteRule(
        "add_identity_right",
        App("ADD", [_a, Const(0)]),
        _a
    ))
    rules.append(RewriteRule(
        "add_identity_left",
        App("ADD", [Const(0), _a]),
        _a
    ))

    # MUL identity: a * 1 = a
    rules.append(RewriteRule(
        "mul_identity_right",
        App("MUL", [_a, Const(1)]),
        _a
    ))
    rules.append(RewriteRule(
        "mul_identity_left",
        App("MUL", [Const(1), _a]),
        _a
    ))

    # =================================
    # ABSORBING ELEMENT RULES
    # =================================

    # MUL absorbing: a * 0 = 0
    rules.append(RewriteRule(
        "mul_zero_right",
        App("MUL", [_a, Const(0)]),
        Const(0)
    ))
    rules.append(RewriteRule(
        "mul_zero_left",
        App("MUL", [Const(0), _a]),
        Const(0)
    ))

    # AND absorbing: a & 0 = 0
    rules.append(RewriteRule(
        "and_zero_right",
        App("AND", [_a, Const(0)]),
        Const(0)
    ))
    rules.append(RewriteRule(
        "and_zero_left",
        App("AND", [Const(0), _a]),
        Const(0)
    ))

    # =================================
    # SELF-INVERSE RULES
    # =================================

    # XOR self-inverse: a ^ a = 0
    rules.append(RewriteRule(
        "xor_self",
        App("XOR", [_a, _a]),
        Const(0)
        # No condition needed - pattern matching handles _a == _a
    ))

    # SUB self: a - a = 0
    rules.append(RewriteRule(
        "sub_self",
        App("SUB", [_a, _a]),
        Const(0)
    ))

    # =================================
    # IDEMPOTENT RULES
    # =================================

    # AND idempotent: a & a = a
    rules.append(RewriteRule(
        "and_idempotent",
        App("AND", [_a, _a]),
        _a
    ))

    # OR idempotent: a | a = a
    rules.append(RewriteRule(
        "or_idempotent",
        App("OR", [_a, _a]),
        _a
    ))

    # =================================
    # DOUBLE NEGATION
    # =================================

    # NOT(NOT(a)) = a
    rules.append(RewriteRule(
        "double_negation",
        App("NOT", [App("NOT", [_a])]),
        _a
    ))

    # NEG(NEG(a)) = a
    rules.append(RewriteRule(
        "double_neg",
        App("NEG", [App("NEG", [_a])]),
        _a
    ))

    # =================================
    # ★★★ THE KEY DISCOVERY RULES ★★★
    # =================================

    # MUL(a, a) → SQUARE(a)  ← THIS IS HOW WE DISCOVER x*x!
    rules.append(RewriteRule(
        "mul_self_to_square",
        App("MUL", [_a, _a]),
        App("SQUARE", [_a])
    ))

    # ADD(a, a) → DOUBLE(a)
    rules.append(RewriteRule(
        "add_self_to_double",
        App("ADD", [_a, _a]),
        App("DOUBLE", [_a])
    ))

    # MUL(a, 2) → DOUBLE(a)
    rules.append(RewriteRule(
        "mul_2_to_double",
        App("MUL", [_a, Const(2)]),
        App("DOUBLE", [_a])
    ))

    # LSL(a, 1) → DOUBLE(a)
    rules.append(RewriteRule(
        "lsl_1_to_double",
        App("LSL", [_a, Const(1)]),
        App("DOUBLE", [_a])
    ))

    # SUB(0, a) → NEG(a)
    rules.append(RewriteRule(
        "sub_from_zero",
        App("SUB", [Const(0), _a]),
        App("NEG", [_a])
    ))

    # =================================
    # SHIFT-MULTIPLY EQUIVALENCE
    # =================================

    # MUL(a, 2^n) → LSL(a, n) for small powers of 2
    for n in range(1, 7):
        rules.append(RewriteRule(
            f"mul_pow2_{n}",
            App("MUL", [_a, Const(1 << n)]),
            App("LSL", [_a, Const(n)])
        ))

    # DIV(a, 2^n) → LSR(a, n) for small powers of 2
    for n in range(1, 7):
        rules.append(RewriteRule(
            f"div_pow2_{n}",
            App("DIV", [_a, Const(1 << n)]),
            App("LSR", [_a, Const(n)])
        ))

    return rules


# =============================================================================
# REWRITE ENGINE
# =============================================================================

class RewriteEngine:
    """
    The algebraic rewrite engine that simplifies expressions.

    This is the core of the Semantic Synthesizer: instead of searching
    for programs, we derive them through algebraic simplification.
    """

    def __init__(self):
        self.rules = build_rewrite_rules()
        self.trace: List[Tuple[str, Expr, Expr]] = []  # (rule_name, before, after)

    def add_rule(self, rule: RewriteRule):
        """Add a custom rewrite rule."""
        self.rules.append(rule)

    def apply_rule(self, rule: RewriteRule, expr: Expr) -> Optional[Expr]:
        """Try to apply a rule to an expression (at the root)."""
        bindings = match_pattern(rule.pattern, expr)
        if bindings is not None:
            if check_condition(rule.condition, bindings):
                return apply_bindings(rule.replacement, bindings)
        return None

    def rewrite_once(self, expr: Expr) -> Tuple[Expr, bool, Optional[str]]:
        """
        Try to apply one rewrite rule to the expression.
        Returns (new_expr, changed, rule_name).
        """
        # Try to rewrite at root
        for rule in self.rules:
            result = self.apply_rule(rule, expr)
            if result is not None:
                return result, True, rule.name

        # Try to rewrite in subexpressions
        if isinstance(expr, App):
            new_args = []
            changed = False
            rule_used = None
            for arg in expr.args:
                new_arg, arg_changed, rule = self.rewrite_once(arg)
                new_args.append(new_arg)
                if arg_changed:
                    changed = True
                    rule_used = rule
            if changed:
                return App(expr.op, new_args), True, rule_used

        return expr, False, None

    def simplify(self, expr: Expr, max_iterations: int = 100, verbose: bool = False) -> Expr:
        """
        Simplify an expression by exhaustively applying rewrite rules.

        This is where MUL(x, x) becomes SQUARE(x) automatically!
        """
        self.trace = []
        current = expr

        for i in range(max_iterations):
            new_expr, changed, rule_name = self.rewrite_once(current)

            if changed:
                if verbose:
                    print(f"  [{rule_name}] {current} → {new_expr}")
                self.trace.append((rule_name, current, new_expr))
                current = new_expr
            else:
                break

        return current

    def discover_pattern(self, expr: Expr, verbose: bool = True) -> Tuple[Expr, List[str]]:
        """
        Discover the simplest form of an expression and return the rules used.

        This is the "discovery" mechanism - given a complex expression,
        it automatically finds simpler equivalent forms.
        """
        if verbose:
            print(f"\nDiscovering pattern for: {expr}")
            print("-" * 40)

        simplified = self.simplify(expr, verbose=verbose)
        rules_used = [r[0] for r in self.trace]

        if verbose:
            print(f"\nResult: {simplified}")
            print(f"Rules used: {rules_used}")

        return simplified, rules_used


# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================

def var(name: str) -> Var:
    return Var(name)

def const(value: int) -> Const:
    return Const(value)

def add(a: Expr, b: Expr) -> App:
    return App("ADD", [a, b])

def sub(a: Expr, b: Expr) -> App:
    return App("SUB", [a, b])

def mul(a: Expr, b: Expr) -> App:
    return App("MUL", [a, b])

def div(a: Expr, b: Expr) -> App:
    return App("DIV", [a, b])

def square(a: Expr) -> App:
    return App("SQUARE", [a])

def double(a: Expr) -> App:
    return App("DOUBLE", [a])

def neg(a: Expr) -> App:
    return App("NEG", [a])

def and_(a: Expr, b: Expr) -> App:
    return App("AND", [a, b])

def or_(a: Expr, b: Expr) -> App:
    return App("OR", [a, b])

def xor(a: Expr, b: Expr) -> App:
    return App("XOR", [a, b])

def not_(a: Expr) -> App:
    return App("NOT", [a])

def lsl(a: Expr, b: Expr) -> App:
    return App("LSL", [a, b])

def lsr(a: Expr, b: Expr) -> App:
    return App("LSR", [a, b])


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ALGEBRAIC REWRITE ENGINE")
    print("=" * 60)

    engine = RewriteEngine()

    print(f"\nLoaded {len(engine.rules)} rewrite rules")

    # =================================
    # TEST 1: The key discovery - MUL(x, x) → SQUARE(x)
    # =================================
    print("\n" + "=" * 60)
    print("TEST 1: MUL(x, x) → SQUARE(x)")
    print("=" * 60)

    x = var("x")
    expr = mul(x, x)  # x * x
    result, rules = engine.discover_pattern(expr)
    assert str(result) == "SQUARE(x)", f"Expected SQUARE(x), got {result}"
    print("✅ PASSED: x*x automatically becomes SQUARE(x)!")

    # =================================
    # TEST 2: ADD(x, x) → DOUBLE(x)
    # =================================
    print("\n" + "=" * 60)
    print("TEST 2: ADD(x, x) → DOUBLE(x)")
    print("=" * 60)

    expr = add(x, x)  # x + x
    result, rules = engine.discover_pattern(expr)
    assert str(result) == "DOUBLE(x)", f"Expected DOUBLE(x), got {result}"
    print("✅ PASSED: x+x automatically becomes DOUBLE(x)!")

    # =================================
    # TEST 3: Identity elimination
    # =================================
    print("\n" + "=" * 60)
    print("TEST 3: ADD(x, 0) → x")
    print("=" * 60)

    expr = add(x, const(0))  # x + 0
    result, rules = engine.discover_pattern(expr)
    assert str(result) == "x", f"Expected x, got {result}"
    print("✅ PASSED: x+0 automatically simplifies to x!")

    # =================================
    # TEST 4: Zero absorption
    # =================================
    print("\n" + "=" * 60)
    print("TEST 4: MUL(x, 0) → 0")
    print("=" * 60)

    expr = mul(x, const(0))  # x * 0
    result, rules = engine.discover_pattern(expr)
    assert str(result) == "0", f"Expected 0, got {result}"
    print("✅ PASSED: x*0 automatically becomes 0!")

    # =================================
    # TEST 5: Shift-multiply equivalence
    # =================================
    print("\n" + "=" * 60)
    print("TEST 5: MUL(x, 8) → LSL(x, 3)")
    print("=" * 60)

    expr = mul(x, const(8))  # x * 8
    result, rules = engine.discover_pattern(expr)
    assert "LSL" in str(result), f"Expected LSL, got {result}"
    print("✅ PASSED: x*8 automatically becomes x<<3!")

    # =================================
    # TEST 6: Nested simplification
    # =================================
    print("\n" + "=" * 60)
    print("TEST 6: ADD(MUL(x, x), 0) → SQUARE(x)")
    print("=" * 60)

    expr = add(mul(x, x), const(0))  # (x * x) + 0
    result, rules = engine.discover_pattern(expr)
    assert str(result) == "SQUARE(x)", f"Expected SQUARE(x), got {result}"
    print("✅ PASSED: Nested expressions simplify correctly!")

    # =================================
    # TEST 7: Self-cancellation
    # =================================
    print("\n" + "=" * 60)
    print("TEST 7: SUB(x, x) → 0")
    print("=" * 60)

    expr = sub(x, x)  # x - x
    result, rules = engine.discover_pattern(expr)
    assert str(result) == "0", f"Expected 0, got {result}"
    print("✅ PASSED: x-x automatically becomes 0!")

    # =================================
    # TEST 8: XOR self-cancellation
    # =================================
    print("\n" + "=" * 60)
    print("TEST 8: XOR(x, x) → 0")
    print("=" * 60)

    expr = xor(x, x)  # x ^ x
    result, rules = engine.discover_pattern(expr)
    assert str(result) == "0", f"Expected 0, got {result}"
    print("✅ PASSED: x^x automatically becomes 0!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\n🎉 The Semantic Rewrite Engine can automatically discover:")
    print("   - SQUARE from MUL(x, x)")
    print("   - DOUBLE from ADD(x, x) or MUL(x, 2)")
    print("   - Shift optimizations from power-of-2 multiplications")
    print("   - Identity and absorption simplifications")
    print("   - Self-cancellation patterns")
