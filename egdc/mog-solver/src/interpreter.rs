/*! Top-level interpreter: lex → parse → evaluate. */

use crate::evaluator::Evaluator;
use crate::lexer::Lexer;
use crate::parser;

/// Result of interpreting a Mog program.
pub struct InterpretResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub return_value: i64,
}

/// Interpret a Mog source string, returns captured stdout.
pub fn interpret(source: &str) -> InterpretResult {
    interpret_with_input(source, Vec::new())
}

/// Interpret a Mog source string with input queue.
pub fn interpret_with_input(source: &str, input: Vec<String>) -> InterpretResult {
    // Lex
    let tokens = Lexer::new(source).tokenize();

    // Parse
    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            return InterpretResult {
                success: false,
                output: String::new(),
                error: Some(format!("{}", e)),
                return_value: 0,
            };
        }
    };

    // Evaluate
    let mut ev = if input.is_empty() {
        Evaluator::new()
    } else {
        Evaluator::with_input(input)
    };

    match ev.run(&program) {
        Ok(val) => {
            let return_value = val.to_i64();
            InterpretResult {
                success: true,
                output: ev.output.join("\n"),
                error: None,
                return_value,
            }
        }
        Err(e) => InterpretResult {
            success: false,
            output: ev.output.join("\n"),
            error: Some(format!("{}", e)),
            return_value: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_two() {
        let code = r#"
fn add_two(a: i64, b: i64) -> i64 {
    return a + b;
}

fn main() -> i64 {
    println_i64(add_two(3, 4));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "7");
    }

    #[test]
    fn test_factorial() {
        let code = r#"
fn factorial(n: i64) -> i64 {
    if n <= 1 { return 1; }
    return n * factorial(n - 1);
}

fn main() -> i64 {
    println_i64(factorial(5));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "120");
    }

    #[test]
    fn test_loop_sum() {
        let code = r#"
fn sum_to_n(n: i64) -> i64 {
    total: i64 = 0;
    i: i64 = 1;
    while i <= n {
        total = total + i;
        i = i + 1;
    }
    return total;
}

fn main() -> i64 {
    println_i64(sum_to_n(10));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "55");
    }

    #[test]
    fn test_array_sum() {
        let code = r#"
fn array_sum(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        total = total + item;
    }
    return total;
}

fn main() -> i64 {
    println_i64(array_sum([1, 2, 3, 4, 5]));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "15");
    }

    #[test]
    fn test_string_methods() {
        let code = r#"
fn main() -> i64 {
    s := "  hello world  ";
    t := s.trim();
    println(t.len);
    println(t.contains("world"));
    println(t.starts_with("hello"));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        let lines: Vec<&str> = result.output.trim().lines().collect();
        assert_eq!(lines[0], "11");
        assert_eq!(lines[1], "true");
        assert_eq!(lines[2], "true");
    }

    #[test]
    fn test_struct() {
        let code = r#"
struct Point {
    x: i64,
    y: i64,
}

fn point_sum(p: Point) -> i64 {
    return p.x + p.y;
}

fn main() -> i64 {
    println_i64(point_sum(Point { x: 3, y: 7 }));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "10");
    }

    #[test]
    fn test_match_result() {
        let code = r#"
fn safe_div(a: i64, b: i64) -> i64 {
    if b == 0 { return -1; }
    return a / b;
}

fn main() -> i64 {
    println_i64(safe_div(10, 3));
    println_i64(safe_div(10, 0));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        let lines: Vec<&str> = result.output.trim().lines().collect();
        assert_eq!(lines[0], "3");
        assert_eq!(lines[1], "-1");
    }

    #[test]
    fn test_for_range() {
        let code = r#"
fn main() -> i64 {
    total: i64 = 0;
    for i := 1 to 6 {
        total = total + i;
    }
    println_i64(total);
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "15");
    }

    #[test]
    fn test_fib_iterative() {
        let code = r#"
fn fib_iter(n: i64) -> i64 {
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    a: i64 = 0;
    b: i64 = 1;
    i: i64 = 2;
    while i <= n {
        tmp: i64 = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    return b;
}

fn main() -> i64 {
    println_i64(fib_iter(10));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "55");
    }

    #[test]
    fn test_gcd() {
        let code = r#"
fn gcd(a: i64, b: i64) -> i64 {
    x: i64 = a;
    y: i64 = b;
    while y != 0 {
        tmp := y;
        y = x % y;
        x = tmp;
    }
    return x;
}

fn main() -> i64 {
    println_i64(gcd(12, 8));
    println_i64(gcd(35, 14));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        let lines: Vec<&str> = result.output.trim().lines().collect();
        assert_eq!(lines[0], "4");
        assert_eq!(lines[1], "7");
    }

    #[test]
    fn test_is_prime() {
        let code = r#"
fn is_prime(n: i64) -> i64 {
    if n < 2 { return 0; }
    if n == 2 { return 1; }
    if n % 2 == 0 { return 0; }
    i: i64 = 3;
    while i * i <= n {
        if n % i == 0 { return 0; }
        i = i + 2;
    }
    return 1;
}

fn main() -> i64 {
    println_i64(is_prime(7));
    println_i64(is_prime(10));
    println_i64(is_prime(13));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        let lines: Vec<&str> = result.output.trim().lines().collect();
        assert_eq!(lines[0], "1");
        assert_eq!(lines[1], "0");
        assert_eq!(lines[2], "1");
    }

    #[test]
    fn test_vowel_count() {
        let code = r#"
fn vowel_count(s: string) -> i64 {
    chars := s.split("");
    total: i64 = 0;
    for ch in chars {
        if ch == "a" { total = total + 1; }
        if ch == "e" { total = total + 1; }
        if ch == "i" { total = total + 1; }
        if ch == "o" { total = total + 1; }
        if ch == "u" { total = total + 1; }
    }
    return total;
}

fn main() -> i64 {
    println_i64(vowel_count("banana"));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "3");
    }

    #[test]
    fn test_count_positive() {
        let code = r#"
fn count_positive(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        if item > 0 {
            total = total + 1;
        }
    }
    return total;
}

fn main() -> i64 {
    println_i64(count_positive([1, -2, 3, -4, 5]));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "3");
    }

    #[test]
    fn test_if_else_chain() {
        let code = r#"
fn sign(x: i64) -> i64 {
    if x < 0 { return -1; }
    if x > 0 { return 1; }
    return 0;
}

fn main() -> i64 {
    println_i64(sign(-5));
    println_i64(sign(0));
    println_i64(sign(7));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        let lines: Vec<&str> = result.output.trim().lines().collect();
        assert_eq!(lines[0], "-1");
        assert_eq!(lines[1], "0");
        assert_eq!(lines[2], "1");
    }

    #[test]
    fn test_collatz() {
        let code = r#"
fn collatz_steps(n: i64) -> i64 {
    x: i64 = n;
    steps: i64 = 0;
    while x > 1 {
        if x % 2 == 0 {
            x = x / 2;
        } else {
            x = 3 * x + 1;
        }
        steps = steps + 1;
    }
    return steps;
}

fn main() -> i64 {
    println_i64(collatz_steps(6));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "8");
    }

    #[test]
    fn test_euler_totient() {
        let code = r#"
fn euler_totient(n: i64) -> i64 {
    result: i64 = n;
    p: i64 = 2;
    temp: i64 = n;
    while p * p <= temp {
        if temp % p == 0 {
            while temp % p == 0 {
                temp = temp / p;
            }
            result = result - result / p;
        }
        p = p + 1;
    }
    if temp > 1 {
        result = result - result / temp;
    }
    return result;
}

fn main() -> i64 {
    println_i64(euler_totient(12));
    return 0;
}
"#;
        let result = interpret(code);
        assert!(result.success, "error: {:?}", result.error);
        assert_eq!(result.output.trim(), "4");
    }
}
