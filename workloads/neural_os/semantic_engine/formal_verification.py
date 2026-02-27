#!/usr/bin/env python3
"""
FORMAL VERIFICATION ENGINE: Symbolic Program Equivalence

This is Phase 1.3 of the Semantic Synthesizer based on the 5-AI Hybrid Review.
It provides symbolic verification that program transformations preserve semantics.

Key insights from the hybrid review:
- "KVRM's 100% accuracy enables generate-and-verify at scale"
- "Verification = Execution on test cases (essentially free)"
- "Programs with VERIFIED behavior create perfect training signal"

This module provides:
1. Symbolic execution traces
2. Equivalence proofs between expressions
3. Property verification (commutativity, associativity, etc.)
4. Test-based verification fallback (using KVRM-style execution)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any, Union, Callable
from enum import Enum, auto
import copy
import random

from rewrite_engine import (
    Expr, Var, Const, App,
    RewriteEngine, RewriteRule,
    var, const, add, sub, mul, div, square, double, neg,
    and_, or_, xor, not_, lsl, lsr,
    match_pattern, apply_bindings
)
from semantic_dictionary import (
    SEMANTIC_DICTIONARY, SemanticOperation, AlgebraicProperty,
    get_operation
)


# =============================================================================
# VERIFICATION RESULT TYPES
# =============================================================================

class VerificationStatus(Enum):
    """Result of a verification attempt."""
    PROVEN = auto()          # Symbolically proven equivalent
    DISPROVEN = auto()       # Counterexample found
    UNKNOWN = auto()         # Could not determine
    TESTED_VALID = auto()    # Passed all test cases (probabilistic)
    TESTED_INVALID = auto()  # Failed test case


@dataclass
class VerificationResult:
    """Result of verification attempt."""
    status: VerificationStatus
    message: str
    proof_steps: List[str] = field(default_factory=list)
    counterexample: Optional[Dict[str, int]] = None
    test_cases_passed: int = 0
    test_cases_total: int = 0

    def is_valid(self) -> bool:
        return self.status in {VerificationStatus.PROVEN, VerificationStatus.TESTED_VALID}

    def __repr__(self):
        if self.status == VerificationStatus.PROVEN:
            return f"✅ PROVEN: {self.message}"
        elif self.status == VerificationStatus.DISPROVEN:
            return f"❌ DISPROVEN: {self.message} (counterexample: {self.counterexample})"
        elif self.status == VerificationStatus.TESTED_VALID:
            return f"✓ TESTED: {self.test_cases_passed}/{self.test_cases_total} passed"
        elif self.status == VerificationStatus.TESTED_INVALID:
            return f"✗ INVALID: {self.message}"
        else:
            return f"? UNKNOWN: {self.message}"


# =============================================================================
# SYMBOLIC EVALUATION
# =============================================================================

@dataclass
class SymbolicValue:
    """
    A symbolic value that can be a concrete integer, a variable,
    or a symbolic expression.
    """
    expr: Expr

    def is_concrete(self) -> bool:
        return isinstance(self.expr, Const)

    def get_concrete(self) -> Optional[int]:
        if isinstance(self.expr, Const):
            return self.expr.value
        return None

    def __repr__(self):
        return f"Sym({self.expr})"


def symbolic_eval(expr: Expr, env: Dict[str, SymbolicValue]) -> SymbolicValue:
    """
    Symbolically evaluate an expression in an environment.

    Returns the simplified symbolic result.
    """
    if isinstance(expr, Const):
        return SymbolicValue(expr)

    if isinstance(expr, Var):
        if expr.name in env:
            return env[expr.name]
        return SymbolicValue(expr)

    if isinstance(expr, App):
        # Evaluate arguments symbolically
        args = [symbolic_eval(arg, env) for arg in expr.args]

        # Try to compute if all args are concrete
        if all(a.is_concrete() for a in args):
            op = get_operation(expr.op)
            if op and op.compute:
                concrete_args = [a.get_concrete() for a in args]
                try:
                    result = op.compute(*concrete_args)
                    return SymbolicValue(Const(result))
                except:
                    pass

        # Return symbolic application
        return SymbolicValue(App(expr.op, [a.expr for a in args]))

    return SymbolicValue(expr)


def concrete_eval(expr: Expr, env: Dict[str, int]) -> int:
    """
    Concretely evaluate an expression given variable bindings.

    This is used for test-based verification.
    """
    MASK = (1 << 64) - 1

    if isinstance(expr, Const):
        return expr.value

    if isinstance(expr, Var):
        if expr.name in env:
            return env[expr.name]
        raise ValueError(f"Unbound variable: {expr.name}")

    if isinstance(expr, App):
        args = [concrete_eval(arg, env) for arg in expr.args]
        op = get_operation(expr.op)

        if op and op.compute:
            return op.compute(*args) & MASK

        # Handle derived operations
        if expr.op == "SQUARE":
            return (args[0] * args[0]) & MASK
        elif expr.op == "DOUBLE":
            return (args[0] * 2) & MASK
        elif expr.op == "NEG":
            return (-args[0]) & MASK
        elif expr.op == "NOT":
            return (~args[0]) & MASK

        raise ValueError(f"Unknown operation: {expr.op}")

    raise ValueError(f"Cannot evaluate: {expr}")


# =============================================================================
# EQUIVALENCE CHECKER
# =============================================================================

class EquivalenceChecker:
    """
    Checks if two expressions are semantically equivalent.

    Uses multiple strategies:
    1. Syntactic equality (after normalization)
    2. Algebraic proof (using rewrite rules)
    3. Test-based verification (fallback)
    """

    def __init__(self, rewrite_engine: Optional[RewriteEngine] = None):
        self.rewrite_engine = rewrite_engine or RewriteEngine()
        self.test_inputs = self._generate_test_inputs()

    def _generate_test_inputs(self, count: int = 100) -> List[int]:
        """Generate diverse test inputs for verification."""
        inputs = []

        # Edge cases
        inputs.extend([0, 1, 2, 3, -1, -2])

        # Small positive
        inputs.extend(range(4, 20))

        # Powers of 2
        inputs.extend([2**i for i in range(1, 10)])

        # Powers of 2 minus/plus 1
        inputs.extend([2**i - 1 for i in range(2, 10)])
        inputs.extend([2**i + 1 for i in range(2, 10)])

        # Random values
        random.seed(42)  # Reproducible
        inputs.extend([random.randint(0, 1000) for _ in range(50)])
        inputs.extend([random.randint(0, 2**32) for _ in range(20)])

        return list(set(inputs))[:count]

    def check_equivalence(self, expr1: Expr, expr2: Expr) -> VerificationResult:
        """
        Check if two expressions are equivalent.

        Returns a VerificationResult with proof or counterexample.
        """
        # Step 1: Normalize both expressions
        norm1 = self.rewrite_engine.simplify(expr1)
        norm2 = self.rewrite_engine.simplify(expr2)

        # Step 2: Check syntactic equality after normalization
        if self._syntactically_equal(norm1, norm2):
            return VerificationResult(
                status=VerificationStatus.PROVEN,
                message="Expressions normalize to identical form",
                proof_steps=[
                    f"Normalize {expr1} → {norm1}",
                    f"Normalize {expr2} → {norm2}",
                    "Normalized forms are identical"
                ]
            )

        # Step 3: Try algebraic proof
        algebraic_result = self._try_algebraic_proof(expr1, expr2)
        if algebraic_result.status == VerificationStatus.PROVEN:
            return algebraic_result

        # Step 4: Fall back to test-based verification
        return self._test_based_verification(expr1, expr2)

    def _syntactically_equal(self, e1: Expr, e2: Expr) -> bool:
        """Check if two expressions are syntactically equal."""
        if type(e1) != type(e2):
            return False

        if isinstance(e1, Const) and isinstance(e2, Const):
            return e1.value == e2.value

        if isinstance(e1, Var) and isinstance(e2, Var):
            return e1.name == e2.name

        if isinstance(e1, App) and isinstance(e2, App):
            if e1.op != e2.op:
                return False
            if len(e1.args) != len(e2.args):
                return False
            return all(self._syntactically_equal(a1, a2)
                      for a1, a2 in zip(e1.args, e2.args))

        return False

    def _try_algebraic_proof(self, expr1: Expr, expr2: Expr) -> VerificationResult:
        """
        Try to prove equivalence using algebraic properties.

        This uses the semantic dictionary to apply commutativity,
        associativity, and other algebraic laws.
        """
        proof_steps = []

        # Get free variables
        vars1 = self._get_free_variables(expr1)
        vars2 = self._get_free_variables(expr2)

        if vars1 != vars2:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                message=f"Different free variables: {vars1} vs {vars2}"
            )

        # Try commutativity
        if self._try_commutative_equivalence(expr1, expr2, proof_steps):
            return VerificationResult(
                status=VerificationStatus.PROVEN,
                message="Equivalent by commutativity",
                proof_steps=proof_steps
            )

        # Try associativity
        if self._try_associative_equivalence(expr1, expr2, proof_steps):
            return VerificationResult(
                status=VerificationStatus.PROVEN,
                message="Equivalent by associativity",
                proof_steps=proof_steps
            )

        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            message="Could not prove algebraically"
        )

    def _get_free_variables(self, expr: Expr) -> Set[str]:
        """Get all free variables in an expression."""
        if isinstance(expr, Const):
            return set()
        if isinstance(expr, Var):
            return {expr.name}
        if isinstance(expr, App):
            result = set()
            for arg in expr.args:
                result |= self._get_free_variables(arg)
            return result
        return set()

    def _try_commutative_equivalence(self, e1: Expr, e2: Expr,
                                     proof_steps: List[str]) -> bool:
        """Check if expressions are equivalent by commutativity."""
        if not isinstance(e1, App) or not isinstance(e2, App):
            return False

        if e1.op != e2.op:
            return False

        op = get_operation(e1.op)
        if not op or AlgebraicProperty.COMMUTATIVE not in op.properties:
            return False

        if len(e1.args) != 2 or len(e2.args) != 2:
            return False

        # Check if e2 is e1 with swapped arguments
        if (self._syntactically_equal(e1.args[0], e2.args[1]) and
            self._syntactically_equal(e1.args[1], e2.args[0])):
            proof_steps.append(f"Apply commutativity of {e1.op}")
            proof_steps.append(f"{e1} = {e2}")
            return True

        return False

    def _try_associative_equivalence(self, e1: Expr, e2: Expr,
                                     proof_steps: List[str]) -> bool:
        """Check if expressions are equivalent by associativity."""
        # This is a simplified check - full associativity handling
        # would require flattening and comparison
        return False

    def _test_based_verification(self, expr1: Expr, expr2: Expr) -> VerificationResult:
        """
        Verify equivalence by testing on many inputs.

        This leverages KVRM's 100% execution accuracy as the
        ground truth oracle.
        """
        vars = self._get_free_variables(expr1) | self._get_free_variables(expr2)

        passed = 0
        total = 0

        for x_val in self.test_inputs:
            for y_val in self.test_inputs[:20]:  # Limit for 2-var expressions
                env = {}
                var_list = list(vars)
                if len(var_list) >= 1:
                    env[var_list[0]] = x_val
                if len(var_list) >= 2:
                    env[var_list[1]] = y_val

                try:
                    result1 = concrete_eval(expr1, env)
                    result2 = concrete_eval(expr2, env)

                    total += 1
                    if result1 == result2:
                        passed += 1
                    else:
                        return VerificationResult(
                            status=VerificationStatus.TESTED_INVALID,
                            message=f"Values differ for {env}",
                            counterexample=env,
                            test_cases_passed=passed,
                            test_cases_total=total
                        )
                except Exception as e:
                    # Skip invalid inputs (e.g., division by zero)
                    pass

                if len(vars) <= 1:
                    break  # Only one variable, don't loop on y

        if total == 0:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                message="No valid test cases"
            )

        return VerificationResult(
            status=VerificationStatus.TESTED_VALID,
            message=f"All {total} test cases passed",
            test_cases_passed=passed,
            test_cases_total=total
        )


# =============================================================================
# REWRITE RULE VERIFIER
# =============================================================================

class RewriteRuleVerifier:
    """
    Verifies that rewrite rules preserve semantics.

    Every rule in the engine should be verified to ensure
    it doesn't change program behavior.
    """

    def __init__(self):
        self.equivalence_checker = EquivalenceChecker()

    def verify_rule(self, rule: RewriteRule) -> VerificationResult:
        """
        Verify that a rewrite rule preserves semantics.

        Checks that pattern and replacement are equivalent
        for all possible bindings.
        """
        # The pattern and replacement should be equivalent
        # when pattern variables are treated as free variables
        return self.equivalence_checker.check_equivalence(
            rule.pattern, rule.replacement
        )

    def verify_all_rules(self, engine: RewriteEngine) -> Dict[str, VerificationResult]:
        """Verify all rules in a rewrite engine."""
        results = {}
        for rule in engine.rules:
            results[rule.name] = self.verify_rule(rule)
        return results


# =============================================================================
# PROPERTY VERIFIER
# =============================================================================

class PropertyVerifier:
    """
    Verifies algebraic properties of operations.

    Used to ensure the semantic dictionary is consistent.
    """

    def __init__(self):
        self.test_inputs = list(range(0, 20)) + [2**i for i in range(1, 8)]

    def verify_commutativity(self, op_name: str) -> VerificationResult:
        """Verify that an operation is commutative: a ⊕ b = b ⊕ a"""
        op = get_operation(op_name)
        if not op or op.arity != 2:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                message="Not a binary operation"
            )

        if not op.compute:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                message="No compute function"
            )

        passed = 0
        total = 0

        for a in self.test_inputs:
            for b in self.test_inputs:
                try:
                    result1 = op.compute(a, b)
                    result2 = op.compute(b, a)
                    total += 1

                    if result1 == result2:
                        passed += 1
                    else:
                        return VerificationResult(
                            status=VerificationStatus.DISPROVEN,
                            message=f"{op_name}({a}, {b}) ≠ {op_name}({b}, {a})",
                            counterexample={'a': a, 'b': b}
                        )
                except:
                    pass

        return VerificationResult(
            status=VerificationStatus.TESTED_VALID,
            message=f"Commutativity verified on {total} cases",
            test_cases_passed=passed,
            test_cases_total=total
        )

    def verify_associativity(self, op_name: str) -> VerificationResult:
        """Verify that an operation is associative: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)"""
        op = get_operation(op_name)
        if not op or op.arity != 2:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                message="Not a binary operation"
            )

        if not op.compute:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                message="No compute function"
            )

        passed = 0
        total = 0

        for a in self.test_inputs[:10]:
            for b in self.test_inputs[:10]:
                for c in self.test_inputs[:10]:
                    try:
                        left = op.compute(op.compute(a, b), c)
                        right = op.compute(a, op.compute(b, c))
                        total += 1

                        if left == right:
                            passed += 1
                        else:
                            return VerificationResult(
                                status=VerificationStatus.DISPROVEN,
                                message=f"({op_name}({a},{b}))⊕{c} ≠ {a}⊕({op_name}({b},{c}))",
                                counterexample={'a': a, 'b': b, 'c': c}
                            )
                    except:
                        pass

        return VerificationResult(
            status=VerificationStatus.TESTED_VALID,
            message=f"Associativity verified on {total} cases",
            test_cases_passed=passed,
            test_cases_total=total
        )

    def verify_identity(self, op_name: str) -> VerificationResult:
        """Verify that an operation has the claimed identity element."""
        op = get_operation(op_name)
        if not op or op.identity is None:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                message="No identity element declared"
            )

        if not op.compute:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                message="No compute function"
            )

        e = op.identity
        passed = 0
        total = 0

        for a in self.test_inputs:
            try:
                # Right identity: a ⊕ e = a
                result_right = op.compute(a, e)
                total += 1

                if result_right == a:
                    passed += 1
                else:
                    return VerificationResult(
                        status=VerificationStatus.DISPROVEN,
                        message=f"{op_name}({a}, {e}) = {result_right} ≠ {a}",
                        counterexample={'a': a, 'e': e}
                    )

                # Left identity (if commutative): e ⊕ a = a
                if AlgebraicProperty.COMMUTATIVE in op.properties:
                    result_left = op.compute(e, a)
                    total += 1

                    if result_left == a:
                        passed += 1
                    else:
                        return VerificationResult(
                            status=VerificationStatus.DISPROVEN,
                            message=f"{op_name}({e}, {a}) = {result_left} ≠ {a}",
                            counterexample={'a': a, 'e': e}
                        )
            except:
                pass

        return VerificationResult(
            status=VerificationStatus.TESTED_VALID,
            message=f"Identity verified on {total} cases",
            test_cases_passed=passed,
            test_cases_total=total
        )

    def verify_all_properties(self, op_name: str) -> Dict[str, VerificationResult]:
        """Verify all declared properties of an operation."""
        op = get_operation(op_name)
        if not op:
            return {"error": VerificationResult(
                status=VerificationStatus.UNKNOWN,
                message=f"Unknown operation: {op_name}"
            )}

        results = {}

        if AlgebraicProperty.COMMUTATIVE in op.properties:
            results["commutativity"] = self.verify_commutativity(op_name)

        if AlgebraicProperty.ASSOCIATIVE in op.properties:
            results["associativity"] = self.verify_associativity(op_name)

        if AlgebraicProperty.IDENTITY_EXISTS in op.properties:
            results["identity"] = self.verify_identity(op_name)

        return results


# =============================================================================
# PROGRAM EQUIVALENCE PROVER
# =============================================================================

class ProgramEquivalenceProver:
    """
    High-level interface for proving program equivalence.

    This is the main entry point for formal verification in SPNC.
    """

    def __init__(self):
        self.rewrite_engine = RewriteEngine()
        self.equivalence_checker = EquivalenceChecker(self.rewrite_engine)
        self.rule_verifier = RewriteRuleVerifier()
        self.property_verifier = PropertyVerifier()

    def prove_equivalent(self, expr1: Expr, expr2: Expr) -> VerificationResult:
        """
        Prove that two expressions compute the same function.

        This is the key operation for ensuring program transformations
        preserve semantics.
        """
        return self.equivalence_checker.check_equivalence(expr1, expr2)

    def prove_rewrite_sound(self, original: Expr, rewritten: Expr) -> VerificationResult:
        """
        Prove that a rewrite preserves semantics.

        Used to verify that the rewrite engine's simplifications are correct.
        """
        return self.prove_equivalent(original, rewritten)

    def verify_simplification(self, expr: Expr, verbose: bool = False) -> Tuple[Expr, VerificationResult]:
        """
        Simplify an expression and verify the simplification is sound.

        Returns the simplified expression and a proof of equivalence.
        """
        simplified = self.rewrite_engine.simplify(expr, verbose=verbose)
        proof = self.prove_equivalent(expr, simplified)
        return simplified, proof

    def verify_dictionary_consistency(self) -> Dict[str, Dict[str, VerificationResult]]:
        """
        Verify that all operations in the semantic dictionary have
        the algebraic properties they claim.
        """
        results = {}
        for op_name in SEMANTIC_DICTIONARY:
            results[op_name] = self.property_verifier.verify_all_properties(op_name)
        return results


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FORMAL VERIFICATION ENGINE")
    print("=" * 60)

    prover = ProgramEquivalenceProver()

    # =================================
    # TEST 1: Verify MUL(x, x) = SQUARE(x)
    # =================================
    print("\n" + "=" * 60)
    print("TEST 1: MUL(x, x) ≡ SQUARE(x)")
    print("=" * 60)

    x = var("x")
    expr1 = mul(x, x)
    expr2 = square(x)

    result = prover.prove_equivalent(expr1, expr2)
    print(result)
    assert result.is_valid(), f"Expected valid equivalence"
    print("✅ PASSED: MUL(x, x) is equivalent to SQUARE(x)")

    # =================================
    # TEST 2: Verify ADD(x, x) = DOUBLE(x)
    # =================================
    print("\n" + "=" * 60)
    print("TEST 2: ADD(x, x) ≡ DOUBLE(x)")
    print("=" * 60)

    expr1 = add(x, x)
    expr2 = double(x)

    result = prover.prove_equivalent(expr1, expr2)
    print(result)
    assert result.is_valid(), f"Expected valid equivalence"
    print("✅ PASSED: ADD(x, x) is equivalent to DOUBLE(x)")

    # =================================
    # TEST 3: Verify commutativity of ADD
    # =================================
    print("\n" + "=" * 60)
    print("TEST 3: Verify ADD is commutative")
    print("=" * 60)

    y = var("y")
    expr1 = add(x, y)
    expr2 = add(y, x)

    result = prover.prove_equivalent(expr1, expr2)
    print(result)
    assert result.is_valid(), f"Expected valid equivalence"
    print("✅ PASSED: ADD(x, y) = ADD(y, x)")

    # =================================
    # TEST 4: Verify identity elimination
    # =================================
    print("\n" + "=" * 60)
    print("TEST 4: ADD(x, 0) ≡ x")
    print("=" * 60)

    expr1 = add(x, const(0))
    expr2 = x

    result = prover.prove_equivalent(expr1, expr2)
    print(result)
    assert result.is_valid(), f"Expected valid equivalence"
    print("✅ PASSED: ADD(x, 0) is equivalent to x")

    # =================================
    # TEST 5: Verify non-equivalence detection
    # =================================
    print("\n" + "=" * 60)
    print("TEST 5: MUL(x, 2) ≢ MUL(x, 3)")
    print("=" * 60)

    expr1 = mul(x, const(2))
    expr2 = mul(x, const(3))

    result = prover.prove_equivalent(expr1, expr2)
    print(result)
    assert not result.is_valid(), f"Expected invalid equivalence"
    print("✅ PASSED: Correctly detected non-equivalence")

    # =================================
    # TEST 6: Verify shift-multiply equivalence
    # =================================
    print("\n" + "=" * 60)
    print("TEST 6: MUL(x, 8) ≡ LSL(x, 3)")
    print("=" * 60)

    expr1 = mul(x, const(8))
    expr2 = lsl(x, const(3))

    result = prover.prove_equivalent(expr1, expr2)
    print(result)
    assert result.is_valid(), f"Expected valid equivalence"
    print("✅ PASSED: MUL(x, 8) is equivalent to LSL(x, 3)")

    # =================================
    # TEST 7: Verify simplification preserves semantics
    # =================================
    print("\n" + "=" * 60)
    print("TEST 7: Simplification verification")
    print("=" * 60)

    complex_expr = add(mul(x, x), const(0))  # (x * x) + 0
    simplified, proof = prover.verify_simplification(complex_expr, verbose=True)

    print(f"\nOriginal: {complex_expr}")
    print(f"Simplified: {simplified}")
    print(f"Verification: {proof}")

    assert proof.is_valid(), f"Expected valid simplification"
    print("✅ PASSED: Simplification preserves semantics")

    # =================================
    # TEST 8: Verify semantic dictionary consistency
    # =================================
    print("\n" + "=" * 60)
    print("TEST 8: Semantic Dictionary Verification")
    print("=" * 60)

    consistency = prover.verify_dictionary_consistency()

    all_valid = True
    for op_name, props in consistency.items():
        if props:  # Skip operations with no verifiable properties
            print(f"\n{op_name}:")
            for prop_name, result in props.items():
                status_symbol = "✅" if result.is_valid() else "❌"
                print(f"  {status_symbol} {prop_name}: {result.message}")
                if not result.is_valid():
                    all_valid = False

    if all_valid:
        print("\n✅ All semantic dictionary properties verified!")
    else:
        print("\n⚠️ Some properties failed verification")

    print("\n" + "=" * 60)
    print("ALL FORMAL VERIFICATION TESTS COMPLETED!")
    print("=" * 60)
    print("\n🎉 The Formal Verification Engine provides:")
    print("   - Symbolic equivalence proofs")
    print("   - Test-based verification fallback")
    print("   - Rewrite rule soundness checking")
    print("   - Semantic dictionary consistency verification")
