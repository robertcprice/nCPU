from egdc.mog_execute import execute_mog
from egdc.mog_lang import interpret
from egdc.mog_dataset import MogProgramGenerator
from egdc.mog_benchmark import get_benchmark, evaluate_solution, evaluate_solution_with_compiler
from egdc.mog_differentiable import DifferentiableMogExecutor
from egdc.mog_grpo import MogRewardModel


def test_mog_execute_smoke():
    code = 'fn main() -> int { println_i64(42); return 0; }'
    result = execute_mog(code)
    assert result.success
    assert result.stdout.strip() == '42'


def test_mog_interpreter_matches_expected_output():
    code = '''
fn factorial(n: i64) -> i64 {
    if (n <= 1) { return 1; }
    return n * factorial(n - 1);
}
fn main() -> int {
    println_i64(factorial(6));
    return 0;
}
'''
    result = interpret(code)
    assert result.success
    assert result.output.strip() == '720'


def test_safe_generator_produces_executable_programs():
    gen = MogProgramGenerator(seed=123)
    for _ in range(5):
        code, spec, expected = gen.generate_one()
        interp = interpret(code)
        assert interp.success, interp.error
        assert interp.output.strip() == expected.strip()
        comp = execute_mog(code)
        assert comp.success, (comp.compile_stderr or comp.stderr or comp.error)
        assert comp.stdout.strip() == expected.strip()


def test_benchmark_reference_solutions_pass_interpreter_and_compiler():
    problems = get_benchmark(seed=42, variants_per_factory=1)[:8]
    for p in problems:
        interp = evaluate_solution(p, p.reference_solution or '')
        assert interp.passed, (p.name, interp.error, interp.actual_output, interp.expected_output)
        comp = evaluate_solution_with_compiler(p, p.reference_solution or '')
        assert comp.passed, (p.name, comp.error, comp.actual_output, comp.expected_output)


def test_differentiable_executor_and_reward_smoke():
    problem = get_benchmark(seed=42, variants_per_factory=1)[0]
    code = problem.reference_solution or ''
    fn_name = problem.signature.split('fn ', 1)[1].split('(', 1)[0]
    args = list(problem.test_cases[0][0])

    ex = DifferentiableMogExecutor()
    result = ex.evaluate_function(code, fn_name, args)
    assert result.success
    assert result.return_value is not None

    reward = MogRewardModel().compute_reward(code, problem)
    assert reward.interpreter_pass == 1.0
    assert reward.compiler_pass == 1.0
    assert reward.reward > 1.0
