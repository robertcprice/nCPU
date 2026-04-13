"""Test fixes for the three identified gaps:
1. abs_diff via two-phase branch training
2. Learnable constant discovery
3. Structure meta-selection
"""

from egdc.mog.solvers.gradient_solver import gradient_solve
from egdc.mog.lang import interpret


def _verify(code, fn_name, cases):
    for args, expected in cases:
        arg_str = ", ".join(str(int(a)) for a in args)
        test = code + f"\nfn main() -> i64 {{ println_i64({fn_name}({arg_str})); return 0; }}"
        r = interpret(test)
        if not r.success or r.output.strip() != str(int(expected)):
            return False
    return True


def test_gradient_discovers_abs_diff():
    """abs_diff requires different ops in each branch arm."""
    from egdc.mog.solvers.two_phase import two_phase_branch_solve

    examples = [
        ((2., 3.), 1.), ((10., -4.), 14.), ((7., 7.), 0.),
        ((-3., -2.), 1.), ((1., 5.), 4.), ((3., 10.), 7.),
    ]
    result = two_phase_branch_solve(["a", "b"], examples, "abs_diff", seed=42)
    print(f"abs_diff: loss={result.loss:.4f}")
    print(result.code)
    assert result.success
    assert result.loss < 1e-6
    assert _verify(result.code, "abs_diff", [((8., 3.), 5.), ((3., 8.), 5.)])


def test_gradient_discovers_positive_or_zero():
    """if x > 0 return x else return 0 — two-phase finds this."""
    from egdc.mog.solvers.two_phase import two_phase_branch_solve

    examples = [
        ((-5.,), 0.), ((0.,), 0.), ((3.,), 3.), ((10.,), 10.), ((-1.,), 0.),
    ]
    result = two_phase_branch_solve(["x"], examples, "pos_or_zero", seed=42)
    print(f"pos_or_zero: loss={result.loss:.4f}")
    print(result.code)
    assert result.success


def test_gradient_discovers_sign_recursive():
    """sign(x): if x>0 return 1; if x<0 return -1; return 0. Recursive branch."""
    from egdc.mog.solvers.two_phase import two_phase_branch_solve

    examples = [
        ((-5.,), -1.), ((0.,), 0.), ((7.,), 1.), ((3.,), 1.), ((-1.,), -1.),
    ]
    result = two_phase_branch_solve(["x"], examples, "sign", seed=42, num_restarts=8)
    print(f"sign: loss={result.loss:.4f}")
    print(result.code)
    assert result.success


def test_gradient_discovers_clamp():
    """clamp(x,0,100): constants mined from examples should include 100."""
    from egdc.mog.solvers.two_phase import two_phase_branch_solve

    examples = [
        ((-5.,), 0.), ((-10.,), 0.), ((-1.,), 0.), ((0.,), 0.),
        ((50.,), 50.), ((100.,), 100.), ((120.,), 100.), ((200.,), 100.),
        ((1.,), 1.), ((99.,), 99.),
    ]
    result = two_phase_branch_solve(["x"], examples, "clamp_0_100", seed=42, num_restarts=15)
    print(f"clamp: loss={result.loss:.4f}")
    print(result.code)
    assert result.success


def test_gradient_discovers_safe_div():
    """safe_div: if b==0 return -1; else return a/b. Constants mined include 0, -1."""
    from egdc.mog.solvers.two_phase import two_phase_branch_solve

    examples = [
        ((10., 2.), 5.), ((7., 0.), -1.), ((9., 3.), 3.),
        ((5., 0.), -1.), ((20., 5.), 4.), ((0., 0.), -1.),
    ]
    result = two_phase_branch_solve(["a", "b"], examples, "safe_div", seed=42, num_restarts=10)
    print(f"safe_div: loss={result.loss:.4f}")
    print(result.code)
    assert result.success


def test_gradient_discovers_is_even():
    """is_even requires modulo operation discovery."""
    from egdc.mog.solvers.two_phase import two_phase_branch_solve

    examples = [
        ((0.,), 1.), ((1.,), 0.), ((2.,), 1.), ((3.,), 0.),
        ((8.,), 1.), ((11.,), 0.), ((100.,), 1.),
    ]
    result = two_phase_branch_solve(["x"], examples, "is_even", seed=42)
    print(f"is_even: loss={result.loss:.4f}")
    print(result.code)
    assert result.success


def test_gradient_discovers_max_or_zero():
    """if a > b return a else return 0 — two-phase finds different arm types."""
    from egdc.mog.solvers.two_phase import two_phase_branch_solve

    examples = [
        ((5., 3.), 5.), ((2., 7.), 0.), ((10., 10.), 0.), ((8., 1.), 8.),
    ]
    result = two_phase_branch_solve(["a", "b"], examples, "max_or_zero", seed=42)
    print(f"max_or_zero: loss={result.loss:.4f}")
    print(result.code)
    assert result.success


def test_structure_selector_picks_right_type():
    """Meta-selector should predict loop for sum_to_n, branch for max2."""
    from egdc.mog.routing.meta_selector import StructureSelector

    selector = StructureSelector()

    # sum_to_n examples — should suggest loop
    loop_examples = [((0.,), 0.), ((1.,), 1.), ((5.,), 15.), ((10.,), 55.)]
    pred = selector.predict_structure(["n"], loop_examples)
    print(f"sum_to_n structure: {pred}")
    assert pred in ("loop", "multi_branch")

    # max2 examples — should suggest branch
    branch_examples = [((2., 3.), 3.), ((10., -4.), 10.), ((7., 7.), 7.)]
    pred = selector.predict_structure(["a", "b"], branch_examples)
    print(f"max2 structure: {pred}")
    assert pred in ("branch", "multi_branch")

    # add examples — should suggest arithmetic
    arith_examples = [((1., 2.), 3.), ((5., -3.), 2.), ((0., 0.), 0.)]
    pred = selector.predict_structure(["a", "b"], arith_examples)
    print(f"add structure: {pred}")
    assert pred == "arithmetic"
