"""Top-level Mog interpreter interface."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .lexer import lex, LexError
from .parser import parse, ParseError
from .evaluator import Evaluator, MogError


@dataclass
class InterpreterResult:
    output: str = ""
    return_value: Optional[int] = None
    error: Optional[str] = None
    success: bool = False


def interpret(code: str, input_data: list[str] | None = None) -> InterpreterResult:
    """Interpret a Mog program and return the result.

    Args:
        code: Mog source code string.
        input_data: Optional list of input values (consumed by read_i64 / read_string).

    Returns:
        InterpreterResult with captured output, return value, and any error.
    """
    try:
        tokens = lex(code)
        program = parse(tokens)
        evaluator = Evaluator()
        if input_data:
            evaluator.input_queue = list(input_data)
        ret = evaluator.run(program)
        return InterpreterResult(
            output="\n".join(evaluator.output),
            return_value=ret if isinstance(ret, int) else 0,
            success=True,
        )
    except (LexError, ParseError, MogError) as e:
        return InterpreterResult(error=str(e), success=False)
    except RecursionError:
        return InterpreterResult(error="maximum recursion depth exceeded", success=False)
    except Exception as e:
        return InterpreterResult(error=f"internal error: {e}", success=False)


if __name__ == "__main__":
    import sys

    # Run tests
    tests = [
        (
            "hello",
            'fn main() -> int { println_i64(42); return 0; }',
            "42",
        ),
        (
            "factorial",
            """
            fn factorial(n: i64) -> i64 {
                if (n <= 1) { return 1; }
                return n * factorial(n - 1);
            }
            fn main() -> int { println_i64(factorial(10)); return 0; }
            """,
            "3628800",
        ),
        (
            "fibonacci",
            """
            fn fibonacci(n: i64) -> i64 {
                if (n <= 0) { return 0; }
                if (n == 1) { return 1; }
                a: i64 = 0;
                b: i64 = 1;
                i: i64 = 2;
                while (i <= n) {
                    tmp := a + b;
                    a = b;
                    b = tmp;
                    i = i + 1;
                }
                return b;
            }
            fn main() -> int { println_i64(fibonacci(20)); return 0; }
            """,
            "6765",
        ),
        (
            "struct",
            """
            struct Point { x: f64, y: f64, }
            fn main() -> int {
                p := Point { x: 3.0, y: 4.0 };
                d := (p.x * p.x) + (p.y * p.y);
                print_f64(d);
                return 0;
            }
            """,
            "25.0000000",
        ),
        (
            "match + result",
            """
            fn safe_div(a: i64, b: i64) -> Result<i64> {
                if (b == 0) { return err("division by zero"); }
                return ok(a / b);
            }
            fn main() -> int {
                r := safe_div(10, 2);
                v: i64 = match r {
                    ok(x) => x,
                    err(e) => -1,
                };
                println_i64(v);
                r2 := safe_div(10, 0);
                v2: i64 = match r2 {
                    ok(x) => x,
                    err(e) => -1,
                };
                println_i64(v2);
                return 0;
            }
            """,
            "5\n-1",
        ),
        (
            "for..to loop",
            """
            fn main() -> int {
                acc: i64 = 0;
                for i := 1 to 11 {
                    acc = acc + i;
                }
                println_i64(acc);
                return 0;
            }
            """,
            "55",
        ),
        (
            "for..in range",
            """
            fn main() -> int {
                acc: i64 = 0;
                for i in 0..10 {
                    acc = acc + i;
                }
                println_i64(acc);
                return 0;
            }
            """,
            "45",
        ),
        (
            "arrays",
            """
            fn main() -> int {
                nums := [1, 2, 3, 4, 5];
                println_i64(nums.len);
                nums.push(6);
                println_i64(nums.len);
                total: i64 = 0;
                for item in nums {
                    total = total + item;
                }
                println_i64(total);
                return 0;
            }
            """,
            "5\n6\n21",
        ),
        (
            "closures + map",
            """
            fn main() -> int {
                nums := [1, 2, 3];
                doubled := nums.map(fn(x: i64) -> i64 { x * 2 });
                for item in doubled {
                    println_i64(item);
                }
                return 0;
            }
            """,
            "2\n4\n6",
        ),
        (
            "optional",
            """
            fn find_positive(n: i64) -> ?i64 {
                if (n > 0) { return some(n); }
                return none;
            }
            fn main() -> int {
                r := find_positive(42);
                v: i64 = match r {
                    some(x) => x,
                    none => -1,
                };
                println_i64(v);
                r2 := find_positive(-5);
                v2: i64 = match r2 {
                    some(x) => x,
                    none => -1,
                };
                println_i64(v2);
                return 0;
            }
            """,
            "42\n-1",
        ),
    ]

    passed = 0
    failed = 0
    for name, code, expected in tests:
        result = interpret(code)
        actual = result.output.strip()
        expected = expected.strip()
        if result.success and actual == expected:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            if result.error:
                print(f"        error: {result.error}")
            else:
                print(f"        expected: {expected!r}")
                print(f"        got:      {actual!r}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(1 if failed else 0)
