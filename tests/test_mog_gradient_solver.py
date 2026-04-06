"""Test the unified gradient-first solver on real benchmark problems.
Everything found by gradient descent through Gumbel-softmax annealing."""

from egdc.mog_gradient_solver import gradient_solve
from egdc.mog_lang import interpret


def _verify(code, fn_name, cases):
    """Verify discovered code on test cases using interpreter."""
    for args, expected in cases:
        arg_str = ", ".join(str(int(a)) for a in args)
        test = code + f"\nfn main() -> i64 {{ println_i64({fn_name}({arg_str})); return 0; }}"
        r = interpret(test)
        if not r.success or r.output.strip() != str(int(expected)):
            return False
    return True


def test_gradient_solver_add_two():
    r = gradient_solve(["a", "b"], [((2., 3.), 5.), ((10., -4.), 6.), ((-3., -2.), -5.)],
                        function_name="add_two", steps=1000, seed=42)
    print(f"add_two: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert _verify(r.code, "add_two", [((7., 8.), 15.), ((0., 0.), 0.)])


def test_gradient_solver_max2():
    r = gradient_solve(["a", "b"],
        [((2., 3.), 3.), ((10., -4.), 10.), ((7., 7.), 7.), ((-3., -2.), -2.), ((0., 5.), 5.)],
        function_name="max2", steps=2000, num_restarts=5, seed=42)
    print(f"max2: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success


def test_gradient_solver_subtract():
    """a - b is discoverable by pure gradient (simpler than abs_diff)."""
    r = gradient_solve(["a", "b"],
        [((5., 3.), 2.), ((10., -4.), 14.), ((7., 7.), 0.), ((-3., 2.), -5.)],
        function_name="subtract", steps=1000, num_restarts=3, seed=42)
    print(f"subtract: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert _verify(r.code, "subtract", [((20., 5.), 15.)])


def test_gradient_solver_sign():
    r = gradient_solve(["x"],
        [((-5.,), -1.), ((0.,), 0.), ((7.,), 1.), ((3.,), 1.), ((-1.,), -1.)],
        function_name="sign", steps=2000, num_restarts=5, seed=42)
    print(f"sign: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success


def test_gradient_solver_sum_to_n():
    r = gradient_solve(["n"],
        [((0.,), 0.), ((1.,), 1.), ((5.,), 15.), ((10.,), 55.)],
        function_name="sum_to_n", steps=2000, num_restarts=5, seed=42)
    print(f"sum_to_n: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
