"""
Shared Judge (Verification)
============================
Fair evaluation of solutions from both V6 and V7 tracks.

Same judge evaluates both tracks to ensure fair comparison.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
import traceback
import sys
import io
import resource
import signal
from contextlib import contextmanager


@dataclass
class TestCase:
    """A single test case for verification."""
    name: str
    input_data: Any
    expected_output: Any
    timeout_seconds: float = 5.0
    description: str = ""


@dataclass
class VerificationResult:
    """Result of code verification."""
    code_hash: str
    passed: bool
    score: float  # 0.0 to 1.0
    tests_passed: int
    tests_total: int
    test_results: List[Dict[str, Any]]
    execution_time_ms: float
    memory_used_mb: float
    errors: List[str]
    timestamp: datetime


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds: float):
    """Context manager for timeout."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds}s")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextmanager
def memory_limit(mb: int):
    """Context manager for memory limit."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (mb * 1024 * 1024, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


@contextmanager
def capture_output():
    """Capture stdout and stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class SharedJudge:
    """
    Fair judge for evaluating code from both tracks.

    Properties:
    - Deterministic evaluation
    - Same test cases for both tracks
    - Resource-limited execution
    - Detailed result reporting
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        memory_limit_mb: int = 256
    ):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self._evaluations = 0

    def verify(
        self,
        code: str,
        test_cases: List[TestCase],
        function_name: str = "solution"
    ) -> VerificationResult:
        """
        Verify code against test cases.

        Returns detailed verification result.
        """
        import hashlib
        import time

        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        start_time = time.time()

        test_results = []
        errors = []
        tests_passed = 0

        # Try to compile the code first
        try:
            compiled = compile(code, "<agent_code>", "exec")
        except SyntaxError as e:
            return VerificationResult(
                code_hash=code_hash,
                passed=False,
                score=0.0,
                tests_passed=0,
                tests_total=len(test_cases),
                test_results=[],
                execution_time_ms=0,
                memory_used_mb=0,
                errors=[f"Syntax error: {e}"],
                timestamp=datetime.now(),
            )

        # Create execution namespace
        namespace = {}

        # Execute the code to define functions
        try:
            exec(compiled, namespace)
        except Exception as e:
            return VerificationResult(
                code_hash=code_hash,
                passed=False,
                score=0.0,
                tests_passed=0,
                tests_total=len(test_cases),
                test_results=[],
                execution_time_ms=0,
                memory_used_mb=0,
                errors=[f"Execution error: {e}"],
                timestamp=datetime.now(),
            )

        # Check if function exists
        if function_name not in namespace:
            return VerificationResult(
                code_hash=code_hash,
                passed=False,
                score=0.0,
                tests_passed=0,
                tests_total=len(test_cases),
                test_results=[],
                execution_time_ms=0,
                memory_used_mb=0,
                errors=[f"Function '{function_name}' not found"],
                timestamp=datetime.now(),
            )

        func = namespace[function_name]

        # Run each test case
        for test in test_cases:
            result = self._run_test(func, test)
            test_results.append(result)

            if result["passed"]:
                tests_passed += 1
            if result.get("error"):
                errors.append(f"{test.name}: {result['error']}")

        # Calculate score
        score = tests_passed / max(len(test_cases), 1)

        execution_time = (time.time() - start_time) * 1000

        self._evaluations += 1

        return VerificationResult(
            code_hash=code_hash,
            passed=tests_passed == len(test_cases),
            score=score,
            tests_passed=tests_passed,
            tests_total=len(test_cases),
            test_results=test_results,
            execution_time_ms=execution_time,
            memory_used_mb=0,  # TODO: actual memory tracking
            errors=errors,
            timestamp=datetime.now(),
        )

    def _run_test(
        self,
        func: Callable,
        test: TestCase
    ) -> Dict[str, Any]:
        """Run a single test case."""
        import time

        result = {
            "name": test.name,
            "passed": False,
            "actual_output": None,
            "expected_output": test.expected_output,
            "execution_time_ms": 0,
            "error": None,
        }

        start_time = time.time()

        try:
            with time_limit(test.timeout_seconds):
                with capture_output() as (stdout, stderr):
                    # Handle different input types
                    if isinstance(test.input_data, tuple):
                        actual = func(*test.input_data)
                    elif isinstance(test.input_data, dict):
                        actual = func(**test.input_data)
                    else:
                        actual = func(test.input_data)

                    result["actual_output"] = actual
                    result["stdout"] = stdout.getvalue()
                    result["stderr"] = stderr.getvalue()

                    # Check result
                    if actual == test.expected_output:
                        result["passed"] = True
                    elif self._fuzzy_match(actual, test.expected_output):
                        result["passed"] = True
                        result["note"] = "fuzzy match"

        except TimeoutError as e:
            result["error"] = str(e)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            result["traceback"] = traceback.format_exc()

        result["execution_time_ms"] = (time.time() - start_time) * 1000

        return result

    def _fuzzy_match(self, actual: Any, expected: Any) -> bool:
        """Fuzzy matching for floating point and similar cases."""
        # Float comparison with tolerance
        if isinstance(actual, float) and isinstance(expected, float):
            return abs(actual - expected) < 1e-6

        # List comparison with fuzzy float matching
        if isinstance(actual, list) and isinstance(expected, list):
            if len(actual) != len(expected):
                return False
            return all(
                self._fuzzy_match(a, e)
                for a, e in zip(actual, expected)
            )

        # String comparison ignoring whitespace
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.strip() == expected.strip()

        return False

    def create_test_suite(
        self,
        problem_type: str,
        difficulty: str = "medium"
    ) -> List[TestCase]:
        """Create a test suite for a problem type."""
        # Example test suites for common problem types

        if problem_type == "sorting":
            return [
                TestCase("empty", [], []),
                TestCase("single", [1], [1]),
                TestCase("sorted", [1, 2, 3], [1, 2, 3]),
                TestCase("reverse", [3, 2, 1], [1, 2, 3]),
                TestCase("random", [3, 1, 4, 1, 5], [1, 1, 3, 4, 5]),
                TestCase("duplicates", [2, 2, 2], [2, 2, 2]),
                TestCase("negative", [-1, -3, -2], [-3, -2, -1]),
            ]

        elif problem_type == "fibonacci":
            return [
                TestCase("fib_0", 0, 0),
                TestCase("fib_1", 1, 1),
                TestCase("fib_2", 2, 1),
                TestCase("fib_5", 5, 5),
                TestCase("fib_10", 10, 55),
            ]

        elif problem_type == "factorial":
            return [
                TestCase("fact_0", 0, 1),
                TestCase("fact_1", 1, 1),
                TestCase("fact_5", 5, 120),
                TestCase("fact_10", 10, 3628800),
            ]

        elif problem_type == "prime":
            return [
                TestCase("prime_2", 2, True),
                TestCase("prime_3", 3, True),
                TestCase("prime_4", 4, False),
                TestCase("prime_17", 17, True),
                TestCase("prime_100", 100, False),
            ]

        else:
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get judge statistics."""
        return {
            "evaluations": self._evaluations,
            "timeout_seconds": self.timeout_seconds,
            "memory_limit_mb": self.memory_limit_mb,
        }


# Singleton judge
_JUDGE: Optional[SharedJudge] = None


def get_judge() -> SharedJudge:
    """Get the global shared judge."""
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = SharedJudge()
    return _JUDGE
