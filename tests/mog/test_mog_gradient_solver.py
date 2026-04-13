"""Test the unified gradient-first solver on real benchmark problems.
Everything found by gradient descent through Gumbel-softmax annealing."""

from egdc.mog.solvers.gradient_solver import gradient_solve
from egdc.mog.lang import interpret


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
    assert _verify(r.code, "sign", [((-8.,), -1.), ((0.,), 0.), ((15.,), 1.)])


def test_gradient_solver_clamp():
    r = gradient_solve(["x"],
        [((-5.,), 0.), ((0.,), 0.), ((37.,), 37.), ((140.,), 100.)],
        function_name="clamp_0_100", steps=1200, num_restarts=3, seed=42)
    print(f"clamp: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert _verify(r.code, "clamp_0_100", [((-1.,), 0.), ((42.,), 42.), ((101.,), 100.)])


def test_gradient_solver_safe_div():
    r = gradient_solve(["a", "b"],
        [((10., 2.), 5.), ((7., 0.), -1.), ((9., 3.), 3.), ((5., 0.), -1.), ((20., 5.), 4.)],
        function_name="safe_div_or_neg1", steps=1200, num_restarts=3, seed=42)
    print(f"safe_div: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert _verify(r.code, "safe_div_or_neg1", [((9., 0.), -1.), ((21., 7.), 3.)])


def test_gradient_solver_is_even():
    r = gradient_solve(["x"],
        [((0.,), 1.), ((1.,), 0.), ((2.,), 1.), ((3.,), 0.), ((8.,), 1.), ((11.,), 0.)],
        function_name="is_even", steps=1200, num_restarts=3, seed=42)
    print(f"is_even: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert _verify(r.code, "is_even", [((-6.,), 1.), ((20.,), 1.), ((105.,), 0.)])


def test_gradient_solver_sum_to_n():
    r = gradient_solve(["n"],
        [((0.,), 0.), ((1.,), 1.), ((5.,), 15.), ((10.,), 55.)],
        function_name="sum_to_n", steps=2000, num_restarts=5, seed=42)
    print(f"sum_to_n: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert _verify(r.code, "sum_to_n", [((7.,), 28.), ((-3.,), 0.)])


def test_gradient_solver_factorial():
    r = gradient_solve(["n"],
        [((0.,), 1.), ((1.,), 1.), ((4.,), 24.), ((5.,), 120.)],
        function_name="factorial", steps=2000, num_restarts=5, seed=42)
    print(f"factorial: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert _verify(r.code, "factorial", [((3.,), 6.), ((6.,), 720.)])


def test_gradient_solver_digit_sum():
    r = gradient_solve(["n"],
        [((0.,), 0.), ((123.,), 6.), ((999.,), 27.), ((1002.,), 3.)],
        function_name="digit_sum", steps=1200, num_restarts=3, seed=42)
    print(f"digit_sum: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert _verify(r.code, "digit_sum", [((405.,), 9.), ((7001.,), 8.)])


def test_gradient_solver_reverse_digits():
    r = gradient_solve(["n"],
        [((0.,), 0.), ((120.,), 21.), ((907.,), 709.), ((4005.,), 5004.)],
        function_name="reverse_digits", steps=1500, num_restarts=3, seed=42)
    print(f"reverse_digits: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert r.structure == "digit_loop"
    assert _verify(r.code, "reverse_digits", [((81.,), 18.), ((12030.,), 3021.)])


def test_gradient_solver_digit_count():
    r = gradient_solve(["n"],
        [((0.,), 1.), ((7.,), 1.), ((120.,), 3.), ((4005.,), 4.)],
        function_name="digit_count", steps=1200, num_restarts=3, seed=42)
    print(f"digit_count: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert r.structure == "digit_loop"
    assert _verify(r.code, "digit_count", [((81.,), 2.), ((12030.,), 5.)])


def test_gradient_solver_count_even_digits():
    r = gradient_solve(["n"],
        [((0.,), 1.), ((7.,), 0.), ((120.,), 2.), ((4005.,), 3.)],
        function_name="count_even_digits", steps=1500, num_restarts=3, seed=42)
    print(f"count_even_digits: loss={r.loss:.4f} structure={r.structure}")
    print(r.code)
    assert r.success
    assert r.structure == "digit_loop"
    assert _verify(r.code, "count_even_digits", [((81.,), 1.), ((12030.,), 3.), ((24680.,), 5.)])


def test_bridge_kadane_max_consecutive_sum():
    """End-to-end test: gradient bridge synthesizes Kadane's algorithm via interactive two-register path."""
    import json
    import subprocess

    payload = {
        "mode": "interactive",
        "interactive_traces": [
            {"input_stream": [1, -2, 3], "expected_output": [1, 1, 3]},
            {"input_stream": [3, -1, 2], "expected_output": [3, 3, 4]},
            {"input_stream": [-1, -2, -3], "expected_output": [-1, -1, -1]},
            {"input_stream": [2, 3, -1, 4], "expected_output": [2, 5, 5, 8]},
        ],
    }
    result = subprocess.run(
        ["python3", "egdc/mog/solvers/gradient_bridge.py"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"bridge failed: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["success"], f"synthesis failed: {data.get('error')}"
    assert data["loss"] == 0.0, f"non-zero loss: {data['loss']}"
    assert "kadane" in data["structure"], f"unexpected structure: {data['structure']}"
    # Confirm reg_b is emitted (global max tracking)
    assert "reg_b" in data["code"], f"reg_b not found in generated code"
