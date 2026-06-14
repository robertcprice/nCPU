#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum Value {
    Int(i64),
    Str(String),
    Array(Vec<i64>),
    Pair(i64, i64),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{v}"),
            Value::Str(s) => write!(f, "{s}"),
            Value::Array(a) => write!(
                f,
                "[{}]",
                a.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Value::Pair(a, b) => write!(f, "({a}, {b})"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Example {
    pub inputs: Vec<Value>,
    pub expected: Value,
}

impl Example {
    /// The expected output as an i64. Numeric solvers operate only on integer
    /// problems, so this returns the int payload (0 for non-int outputs, which
    /// those solvers never see).
    pub fn expected_int(&self) -> i64 {
        match &self.expected {
            Value::Int(i) => *i,
            Value::Pair(a, _) => *a,
            _ => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    pub name: String,
    pub category: &'static str,
    pub description: &'static str,
    pub signature: &'static str,
    pub examples: Vec<Example>,
    pub holdouts: Vec<Example>,
    pub reference_code: &'static str,
}

impl Problem {
    pub fn function_name(&self) -> &str {
        self.signature
            .split_once("fn ")
            .and_then(|(_, rest)| rest.split_once('('))
            .map(|(name, _)| name.trim())
            .unwrap_or("")
    }

    /// True when the function returns a string (used to pick the right print
    /// builtin and expected-output rendering).
    fn returns_string(&self) -> bool {
        self.signature.replace(' ', "").contains("->string")
    }

    pub fn expected_stdout(&self) -> String {
        self.examples
            .iter()
            .map(|example| render_expected(&example.expected))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn build_wrapper(&self) -> Result<String, String> {
        let fn_name = self.function_name();
        // String-returning functions print with the generic `println` (raw
        // string); integer functions print with `println_i64`.
        let print = if self.returns_string() {
            "println"
        } else {
            "println_i64"
        };
        let mut lines = vec!["fn main() -> i64 {".to_string()];
        for example in &self.examples {
            let args = example
                .inputs
                .iter()
                .map(|value| render_value(self, value))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            lines.push(format!("    {print}({fn_name}({args}));"));
        }
        lines.push("    return 0;".to_string());
        lines.push("}".to_string());
        Ok(lines.join("\n"))
    }

    pub fn wrap_program(&self, generated_code: &str) -> Result<String, String> {
        Ok(format!(
            "{}\n\n{}\n",
            generated_code.trim_end(),
            self.build_wrapper()?
        ))
    }
}

fn example(inputs: Vec<Value>, expected: i64) -> Example {
    Example {
        inputs,
        expected: Value::Int(expected),
    }
}

/// An example with a string expected output (for string-returning problems).
#[allow(dead_code)]
fn example_str(inputs: Vec<Value>, expected: &str) -> Example {
    Example {
        inputs,
        expected: Value::Str(expected.to_string()),
    }
}

/// Render an expected output to match what the wrapper's print builtin emits.
fn render_expected(value: &Value) -> String {
    match value {
        Value::Int(v) => v.to_string(),
        Value::Str(s) => s.clone(),
        Value::Array(a) => format!(
            "[{}]",
            a.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Pair(a, b) => format!("({a}, {b})"),
    }
}

fn int(v: i64) -> Value {
    Value::Int(v)
}

fn string(v: &str) -> Value {
    Value::Str(v.to_string())
}

fn array(v: &[i64]) -> Value {
    Value::Array(v.to_vec())
}

fn pair(a: i64, b: i64) -> Value {
    Value::Pair(a, b)
}

fn render_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn render_value(problem: &Problem, value: &Value) -> Result<String, String> {
    match value {
        Value::Int(v) => Ok(v.to_string()),
        Value::Str(v) => Ok(render_string(v)),
        Value::Array(values) => Ok(format!(
            "[{}]",
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
        Value::Pair(a, b) => {
            if problem.signature.contains("Point") {
                Ok(format!("Point {{ x: {a}, y: {b} }}"))
            } else if problem.signature.contains("Rectangle") {
                Ok(format!("Rectangle {{ width: {a}, height: {b} }}"))
            } else {
                Err(format!(
                    "cannot render pair literal for {} with signature {}",
                    problem.name, problem.signature
                ))
            }
        }
    }
}

fn problem(
    name: &str,
    variant: usize,
    category: &'static str,
    description: &'static str,
    signature: &'static str,
    examples: Vec<Example>,
    holdouts: Vec<Example>,
    reference_code: &'static str,
) -> Problem {
    Problem {
        name: format!("{name}_v{variant}"),
        category,
        description,
        signature,
        examples,
        holdouts,
        reference_code,
    }
}

pub type Factory = fn(usize) -> Problem;

pub fn get_benchmark(variants_per_factory: usize) -> Vec<Problem> {
    let mut problems = Vec::new();
    for variant in 0..variants_per_factory {
        for factory in FACTORIES {
            problems.push(factory(variant));
        }
    }
    problems
}

pub fn factory_count() -> usize {
    FACTORIES.len()
}

pub fn generated_holdouts(problem: &Problem) -> Vec<Example> {
    problem.holdouts.clone()
}

fn make_add_two(variant: usize) -> Problem {
    problem(
        "add_two",
        variant,
        "arithmetic",
        "Return the sum of two i64 integers.",
        "fn add_two(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(2), int(3)], 5),
            example(vec![int(10), int(-4)], 6),
            example(vec![int(7), int(8)], 15),
            example(vec![int(-3), int(-2)], -5),
        ],
        vec![
            example(vec![int(100), int(-37)], 63),
            example(vec![int(-12), int(-8)], -20),
        ],
        "fn add_two(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n",
    )
}

fn make_abs_diff(variant: usize) -> Problem {
    problem(
        "abs_diff",
        variant,
        "arithmetic",
        "Return the absolute difference between two integers.",
        "fn abs_diff(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(3), int(7)], 4),
            example(vec![int(7), int(3)], 4),
            example(vec![int(0), int(5)], 5),
            example(vec![int(-3), int(2)], 5),
        ],
        vec![
            example(vec![int(-10), int(7)], 17),
            example(vec![int(9), int(-4)], 13),
        ],
        "fn abs_diff(a: i64, b: i64) -> i64 {\n    if a > b {\n        return a - b;\n    } else {\n        return b - a;\n    }\n}\n",
    )
}

fn make_max2(variant: usize) -> Problem {
    problem(
        "max2",
        variant,
        "control_flow",
        "Return the larger of two integers.",
        "fn max2(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(2), int(3)], 3),
            example(vec![int(10), int(-4)], 10),
            example(vec![int(7), int(7)], 7),
            example(vec![int(-3), int(-2)], -2),
        ],
        vec![
            example(vec![int(-3), int(9)], 9),
            example(vec![int(12), int(12)], 12),
        ],
        "fn max2(a: i64, b: i64) -> i64 {\n    if a > b {\n        return a;\n    } else {\n        return b;\n    }\n}\n",
    )
}

fn make_clamp(variant: usize) -> Problem {
    problem(
        "clamp_0_100",
        variant,
        "control_flow",
        "Clamp x into the closed range [0, 100].",
        "fn clamp_0_100(x: i64) -> i64",
        vec![
            example(vec![int(-5)], 0),
            example(vec![int(0)], 0),
            example(vec![int(37)], 37),
            example(vec![int(140)], 100),
        ],
        vec![
            example(vec![int(-1)], 0),
            example(vec![int(101)], 100),
            example(vec![int(42)], 42),
        ],
        "fn clamp_0_100(x: i64) -> i64 {\n    if x < 0 {\n        return 0;\n    }\n    if x > 100 {\n        return 100;\n    }\n    return x;\n}\n",
    )
}

fn make_sign(variant: usize) -> Problem {
    problem(
        "sign",
        variant,
        "control_flow",
        "Return -1 for negative, 0 for zero, and 1 for positive.",
        "fn sign(x: i64) -> i64",
        vec![
            example(vec![int(-5)], -1),
            example(vec![int(0)], 0),
            example(vec![int(7)], 1),
            example(vec![int(3)], 1),
        ],
        vec![
            example(vec![int(-8)], -1),
            example(vec![int(0)], 0),
            example(vec![int(15)], 1),
        ],
        "fn sign(x: i64) -> i64 {\n    if x < 0 {\n        return -1;\n    }\n    if x > 0 {\n        return 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_sum_to_n(variant: usize) -> Problem {
    problem(
        "sum_to_n",
        variant,
        "arithmetic",
        "Return 1 + 2 + ... + n. For n <= 0 return 0.",
        "fn sum_to_n(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 1),
            example(vec![int(5)], 15),
            example(vec![int(10)], 55),
        ],
        vec![example(vec![int(7)], 28), example(vec![int(-3)], 0)],
        "fn sum_to_n(n: i64) -> i64 {\n    if n <= 0 {\n        return 0;\n    }\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        total = total + i;\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_gcd(variant: usize) -> Problem {
    problem(
        "gcd",
        variant,
        "arithmetic",
        "Return the greatest common divisor of two positive integers.",
        "fn gcd(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(12), int(18)], 6),
            example(vec![int(21), int(14)], 7),
            example(vec![int(9), int(28)], 1),
            example(vec![int(48), int(18)], 6),
        ],
        vec![
            example(vec![int(270), int(192)], 6),
            example(vec![int(17), int(13)], 1),
        ],
        "fn gcd(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }\n    return x;\n}\n",
    )
}

fn make_lcm(variant: usize) -> Problem {
    problem(
        "lcm",
        variant,
        "arithmetic",
        "Return the least common multiple of two positive integers.",
        "fn lcm(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(3), int(4)], 12),
            example(vec![int(6), int(8)], 24),
            example(vec![int(5), int(10)], 10),
            example(vec![int(7), int(9)], 63),
        ],
        vec![
            example(vec![int(8), int(12)], 24),
            example(vec![int(9), int(6)], 18),
        ],
        "fn gcd_inner(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }\n    return x;\n}\n\nfn lcm(a: i64, b: i64) -> i64 {\n    return (a * b) / gcd_inner(a, b);\n}\n",
    )
}

fn make_array_sum(variant: usize) -> Problem {
    problem(
        "array_sum",
        variant,
        "arrays",
        "Return the sum of all elements in an array of i64 values.",
        "fn array_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 6),
            example(vec![array(&[5])], 5),
            example(vec![array(&[4, 4])], 8),
            example(vec![array(&[2, 7, 1, 0])], 10),
        ],
        vec![
            example(vec![array(&[10, -5, 2])], 7),
            example(vec![array(&[1, 2, 3, 4])], 10),
        ],
        "fn array_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        total = total + item;\n    }\n    return total;\n}\n",
    )
}

fn make_array_max(variant: usize) -> Problem {
    problem(
        "array_max",
        variant,
        "arrays",
        "Return the largest element in a non-empty array.",
        "fn array_max(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 3),
            example(vec![array(&[5])], 5),
            example(vec![array(&[4, 9])], 9),
            example(vec![array(&[2, 7, 1, 0])], 7),
        ],
        vec![
            example(vec![array(&[-3, -9, -1])], -1),
            example(vec![array(&[10, 2, 10])], 10),
        ],
        "fn array_max(arr: [i64]) -> i64 {\n    best := arr[0];\n    for item in arr {\n        if item > best {\n            best = item;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_count_occurrences(variant: usize) -> Problem {
    problem(
        "count_occurrences",
        variant,
        "arrays",
        "Count how many times target appears in arr.",
        "fn count_occurrences(arr: [i64], target: i64) -> i64",
        vec![
            example(vec![array(&[1, 2, 1]), int(1)], 2),
            example(vec![array(&[5]), int(5)], 1),
            example(vec![array(&[4, 9]), int(1)], 0),
            example(vec![array(&[2, 7, 2, 0]), int(2)], 2),
        ],
        vec![
            example(vec![array(&[4, 1, 4, 4]), int(4)], 3),
            example(vec![array(&[2, 3]), int(5)], 0),
        ],
        "fn count_occurrences(arr: [i64], target: i64) -> i64 {\n    count: i64 = 0;\n    for item in arr {\n        if item == target {\n            count = count + 1;\n        }\n    }\n    return count;\n}\n",
    )
}

fn make_trimmed_len(variant: usize) -> Problem {
    problem(
        "trimmed_len",
        variant,
        "strings",
        "Trim leading and trailing spaces and return the remaining length.",
        "fn trimmed_len(s: string) -> i64",
        vec![
            example(vec![string(" mog ")], 3),
            example(vec![string("  diffusion")], 9),
            example(vec![string("compiler  ")], 8),
            example(vec![string("  hello world  ")], 11),
        ],
        vec![
            example(vec![string("   hi there   ")], 8),
            example(vec![string("      ")], 0),
        ],
        "fn trimmed_len(s: string) -> i64 {\n    t := s.trim();\n    return t.len;\n}\n",
    )
}

fn make_vowel_count(variant: usize) -> Problem {
    problem(
        "vowel_count",
        variant,
        "strings",
        "Count vowels (a, e, i, o, u) in a lowercase ASCII string.",
        "fn vowel_count(s: string) -> i64",
        vec![
            example(vec![string("mog")], 1),
            example(vec![string("aeiou")], 5),
            example(vec![string("banana")], 3),
            example(vec![string("rhythm")], 0),
        ],
        vec![
            example(vec![string("queue")], 4),
            example(vec![string("sky")], 0),
        ],
        "fn vowel_count(s: string) -> i64 {\n    chars := s.split(\"\");\n    total: i64 = 0;\n    for ch in chars {\n        if ch == \"a\" { total = total + 1; }\n        if ch == \"e\" { total = total + 1; }\n        if ch == \"i\" { total = total + 1; }\n        if ch == \"o\" { total = total + 1; }\n        if ch == \"u\" { total = total + 1; }\n    }\n    return total;\n}\n",
    )
}

fn make_contains_cat(variant: usize) -> Problem {
    problem(
        "contains_cat",
        variant,
        "strings",
        "Return 1 if the string contains the substring 'cat', else 0.",
        "fn contains_cat(s: string) -> i64",
        vec![
            example(vec![string("cat")], 1),
            example(vec![string("scatter")], 1),
            example(vec![string("dog")], 0),
            example(vec![string("hello")], 0),
        ],
        vec![
            example(vec![string("bobcat")], 1),
            example(vec![string("atlas")], 0),
        ],
        "fn contains_cat(s: string) -> i64 {\n    if s.contains(\"cat\") {\n        return 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_point_sum(variant: usize) -> Problem {
    problem(
        "point_sum",
        variant,
        "structs",
        "Define struct Point { x: i64, y: i64 } and return x + y.",
        "fn point_sum(p: Point) -> i64",
        vec![
            example(vec![pair(3, 4)], 7),
            example(vec![pair(-1, 2)], 1),
            example(vec![pair(0, 0)], 0),
            example(vec![pair(9, 8)], 17),
        ],
        vec![
            example(vec![pair(12, -5)], 7),
            example(vec![pair(-3, -4)], -7),
        ],
        "struct Point {\n    x: i64,\n    y: i64,\n}\n\nfn point_sum(p: Point) -> i64 {\n    return p.x + p.y;\n}\n",
    )
}

fn make_safe_div_or_neg1(variant: usize) -> Problem {
    problem(
        "safe_div_or_neg1",
        variant,
        "result_optional",
        "Divide a by b. If b is zero, return -1.",
        "fn safe_div_or_neg1(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(10), int(2)], 5),
            example(vec![int(7), int(0)], -1),
            example(vec![int(9), int(3)], 3),
            example(vec![int(5), int(0)], -1),
        ],
        vec![
            example(vec![int(9), int(0)], -1),
            example(vec![int(21), int(7)], 3),
        ],
        "fn helper_div(a: i64, b: i64) -> Result<i64> {\n    if b == 0 {\n        return err(\"division by zero\");\n    }\n    return ok(a / b);\n}\n\nfn safe_div_or_neg1(a: i64, b: i64) -> i64 {\n    r := helper_div(a, b);\n    out: i64 = match r {\n        ok(v) => v,\n        err(e) => -1,\n    };\n    return out;\n}\n",
    )
}

fn make_positive_or_default(variant: usize) -> Problem {
    problem(
        "positive_or_default",
        variant,
        "result_optional",
        "Return x if x is positive, otherwise 0.",
        "fn positive_or_default(x: i64) -> i64",
        vec![
            example(vec![int(10)], 10),
            example(vec![int(0)], 0),
            example(vec![int(-5)], 0),
            example(vec![int(3)], 3),
        ],
        vec![example(vec![int(-4)], 0), example(vec![int(19)], 19)],
        "fn maybe_positive(x: i64) -> ?i64 {\n    if x > 0 {\n        return some(x);\n    }\n    return none;\n}\n\nfn positive_or_default(x: i64) -> i64 {\n    r := maybe_positive(x);\n    out: i64 = match r {\n        some(v) => v,\n        none => 0,\n    };\n    return out;\n}\n",
    )
}

fn make_factorial(variant: usize) -> Problem {
    problem(
        "factorial",
        variant,
        "recursion",
        "Return n! recursively.",
        "fn factorial(n: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(1)], 1),
            example(vec![int(5)], 120),
            example(vec![int(7)], 5040),
        ],
        vec![
            example(vec![int(3)], 6),
            example(vec![int(6)], 720),
        ],
        "fn factorial(n: i64) -> i64 {\n    if n <= 1 {\n        return 1;\n    }\n    return n * factorial(n - 1);\n}\n",
    )
}

fn make_fibonacci(variant: usize) -> Problem {
    problem(
        "fibonacci",
        variant,
        "recursion",
        "Return the nth Fibonacci number recursively or iteratively.",
        "fn fibonacci(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 1),
            example(vec![int(7)], 13),
            example(vec![int(10)], 55),
        ],
        vec![
            example(vec![int(5)], 5),
            example(vec![int(8)], 21),
        ],
        "fn fibonacci(n: i64) -> i64 {\n    if n <= 0 { return 0; }\n    if n == 1 { return 1; }\n    return fibonacci(n - 1) + fibonacci(n - 2);\n}\n",
    )
}

fn make_closure_map_sum(variant: usize) -> Problem {
    problem(
        "closure_map_sum",
        variant,
        "higher_order",
        "Double every array element with .map() and return the sum of the doubled values.",
        "fn closure_map_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 12),
            example(vec![array(&[2, 2, 2])], 12),
            example(vec![array(&[5, 1, 4])], 20),
            example(vec![array(&[3, 5, 1])], 18),
        ],
        vec![
            example(vec![array(&[4, 6])], 20),
            example(vec![array(&[1, 1, 1, 1])], 8),
        ],
        "fn closure_map_sum(arr: [i64]) -> i64 {\n    doubled := arr.map(fn(x: i64) -> i64 { x * 2 });\n    total: i64 = 0;\n    for item in doubled {\n        total = total + item;\n    }\n    return total;\n}\n",
    )
}

fn make_count_positive(variant: usize) -> Problem {
    problem(
        "count_positive",
        variant,
        "arrays",
        "Count how many elements in the array are greater than zero.",
        "fn count_positive(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 3),
            example(vec![array(&[-5])], 0),
            example(vec![array(&[4, -9])], 1),
            example(vec![array(&[2, 7, -1, 0])], 2),
        ],
        vec![
            example(vec![array(&[0, 5, 0])], 1),
            example(vec![array(&[-1, -2, 3, 4])], 2),
        ],
        "fn count_positive(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        if item > 0 {\n            total = total + 1;\n        }\n    }\n    return total;\n}\n",
    )
}

fn make_is_even(variant: usize) -> Problem {
    problem(
        "is_even",
        variant,
        "control_flow",
        "Return 1 if x is even, otherwise 0.",
        "fn is_even(x: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(1)], 0),
            example(vec![int(8)], 1),
            example(vec![int(11)], 0),
        ],
        vec![
            example(vec![int(4)], 1),
            example(vec![int(7)], 0),
        ],
        "fn is_even(x: i64) -> i64 {\n    if (x % 2) == 0 {\n        return 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_digit_sum(variant: usize) -> Problem {
    problem(
        "digit_sum",
        variant,
        "arithmetic",
        "Return the sum of the decimal digits of n.",
        "fn digit_sum(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(7)], 7),
            example(vec![int(123)], 6),
            example(vec![int(9081)], 18),
        ],
        vec![
            example(vec![int(456)], 15),
            example(vec![int(100)], 1),
        ],
        "fn digit_sum(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    total: i64 = 0;\n    while x > 0 {\n        total = total + (x % 10);\n        x = x / 10;\n    }\n    return total;\n}\n",
    )
}

fn make_reverse_digits(variant: usize) -> Problem {
    problem(
        "reverse_digits",
        variant,
        "loops",
        "Reverse the decimal digits of n.",
        "fn reverse_digits(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(120)], 21),
            example(vec![int(907)], 709),
            example(vec![int(4005)], 5004),
        ],
        vec![
            example(vec![int(123)], 321),
            example(vec![int(100)], 1),
        ],
        "fn reverse_digits(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    acc: i64 = 0;\n    while x > 0 {\n        acc = (acc * 10) + (x % 10);\n        x = x / 10;\n    }\n    return acc;\n}\n",
    )
}

fn make_digit_count(variant: usize) -> Problem {
    problem(
        "digit_count",
        variant,
        "loops",
        "Count how many decimal digits n contains.",
        "fn digit_count(n: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(7)], 1),
            example(vec![int(120)], 3),
            example(vec![int(4005)], 4),
        ],
        vec![
            example(vec![int(99)], 2),
            example(vec![int(1000)], 4),
        ],
        "fn digit_count(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    if x == 0 {\n        return 1;\n    }\n    acc: i64 = 0;\n    while x > 0 {\n        acc = acc + 1;\n        x = x / 10;\n    }\n    return acc;\n}\n",
    )
}

fn make_count_even_digits(variant: usize) -> Problem {
    problem(
        "count_even_digits",
        variant,
        "loops",
        "Count how many decimal digits of n are even.",
        "fn count_even_digits(n: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(7)], 0),
            example(vec![int(120)], 2),
            example(vec![int(4005)], 3),
        ],
        vec![
            example(vec![int(246)], 3),
            example(vec![int(135)], 0),
        ],
        "fn count_even_digits(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    if x == 0 {\n        return 1;\n    }\n    acc: i64 = 0;\n    while x > 0 {\n        if ((x % 10) % 2) == 0 {\n            acc = acc + 1;\n        }\n        x = x / 10;\n    }\n    return acc;\n}\n",
    )
}

fn make_starts_with_m(variant: usize) -> Problem {
    problem(
        "starts_with_m",
        variant,
        "strings",
        "Return 1 if s starts with the lowercase letter m, else 0.",
        "fn starts_with_m(s: string) -> i64",
        vec![
            example(vec![string("mog")], 1),
            example(vec![string("metal")], 1),
            example(vec![string("apple")], 0),
            example(vec![string("map")], 1),
        ],
        vec![
            example(vec![string("moon")], 1),
            example(vec![string("sun")], 0),
        ],
        "fn starts_with_m(s: string) -> i64 {\n    if s.starts_with(\"m\") {\n        return 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_rectangle_area(variant: usize) -> Problem {
    problem(
        "rectangle_area",
        variant,
        "structs",
        "Define struct Rectangle { width: i64, height: i64 } and return its area.",
        "fn rectangle_area(r: Rectangle) -> i64",
        vec![
            example(vec![pair(3, 4)], 12),
            example(vec![pair(1, 2)], 2),
            example(vec![pair(5, 5)], 25),
            example(vec![pair(9, 8)], 72),
        ],
        vec![
            example(vec![pair(7, 3)], 21),
            example(vec![pair(6, 6)], 36),
        ],
        "struct Rectangle {\n    width: i64,\n    height: i64,\n}\n\nfn rectangle_area(r: Rectangle) -> i64 {\n    return r.width * r.height;\n}\n",
    )
}

fn make_power(variant: usize) -> Problem {
    problem(
        "power",
        variant,
        "arithmetic",
        "Compute base raised to the power exp (non-negative).",
        "fn power(base: i64, exp: i64) -> i64",
        vec![
            example(vec![int(2), int(0)], 1),
            example(vec![int(2), int(3)], 8),
            example(vec![int(5), int(2)], 25),
            example(vec![int(3), int(4)], 81),
        ],
        vec![
            example(vec![int(4), int(3)], 64),
            example(vec![int(10), int(2)], 100),
        ],
        "fn power(base: i64, exp: i64) -> i64 {\n    if exp == 0 { return 1; }\n    result: i64 = 1;\n    i: i64 = 0;\n    while i < exp {\n        result = result * base;\n        i = i + 1;\n    }\n    return result;\n}\n",
    )
}

fn make_polynomial(variant: usize) -> Problem {
    problem(
        "polynomial",
        variant,
        "arithmetic",
        "Evaluate the polynomial 2*x*x + 3*x + 1.",
        "fn polynomial(x: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(1)], 6),
            example(vec![int(2)], 15),
            example(vec![int(5)], 66),
        ],
        vec![example(vec![int(3)], 28), example(vec![int(-1)], 0)],
        "fn polynomial(x: i64) -> i64 {\n    return 2 * x * x + 3 * x + 1;\n}\n",
    )
}

fn make_collatz_steps(variant: usize) -> Problem {
    problem(
        "collatz_steps",
        variant,
        "loops",
        "Count how many steps it takes for the Collatz sequence starting at n to reach 1.",
        "fn collatz_steps(n: i64) -> i64",
        vec![
            example(vec![int(1)], 0),
            example(vec![int(2)], 1),
            example(vec![int(3)], 7),
            example(vec![int(10)], 6),
        ],
        vec![
            example(vec![int(6)], 8),
            example(vec![int(7)], 16),
        ],
        "fn collatz_steps(n: i64) -> i64 {\n    x: i64 = n;\n    steps: i64 = 0;\n    while x > 1 {\n        if x % 2 == 0 {\n            x = x / 2;\n        } else {\n            x = 3 * x + 1;\n        }\n        steps = steps + 1;\n    }\n    return steps;\n}\n",
    )
}

fn make_min3(variant: usize) -> Problem {
    problem(
        "min3",
        variant,
        "control_flow",
        "Return the minimum of three integers.",
        "fn min3(a: i64, b: i64, c: i64) -> i64",
        vec![
            example(vec![int(2), int(3), int(4)], 2),
            example(vec![int(10), int(-4), int(8)], -4),
            example(vec![int(7), int(7), int(7)], 7),
            example(vec![int(-3), int(-2), int(-9)], -9),
        ],
        vec![
            example(vec![int(5), int(1), int(3)], 1),
            example(vec![int(0), int(-1), int(2)], -1),
        ],
        "fn min3(a: i64, b: i64, c: i64) -> i64 {\n    m: i64 = a;\n    if b < m { m = b; }\n    if c < m { m = c; }\n    return m;\n}\n",
    )
}

fn make_reverse_sum(variant: usize) -> Problem {
    problem(
        "reverse_sum",
        variant,
        "arrays",
        "Sum all elements of an array.",
        "fn reverse_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 6),
            example(vec![array(&[5, 1])], 6),
            example(vec![array(&[4, 4, 4])], 12),
            example(vec![array(&[2, 7, 1, 0])], 10),
        ],
        vec![
            example(vec![array(&[10, -5, 2])], 7),
            example(vec![array(&[1, 2, 3, 4])], 10),
        ],
        "fn reverse_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        total = total + item;\n    }\n    return total;\n}\n",
    )
}

fn make_array_max_elem(variant: usize) -> Problem {
    problem(
        "array_max_elem",
        variant,
        "arrays",
        "Find the maximum element in an array.",
        "fn array_max_elem(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 3),
            example(vec![array(&[5, 1])], 5),
            example(vec![array(&[4, 9])], 9),
            example(vec![array(&[2, 7, 1, 0])], 7),
        ],
        vec![
            example(vec![array(&[-3, -9, -1])], -1),
            example(vec![array(&[10, 2, 10])], 10),
        ],
        "fn array_max_elem(arr: [i64]) -> i64 {\n    best := arr[0];\n    for item in arr {\n        if item > best {\n            best = item;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_is_prime(variant: usize) -> Problem {
    problem(
        "is_prime",
        variant,
        "loops",
        "Return 1 if the number is prime, 0 otherwise.",
        "fn is_prime(n: i64) -> i64",
        vec![
            example(vec![int(2)], 1),
            example(vec![int(4)], 0),
            example(vec![int(11)], 1),
            example(vec![int(15)], 0),
        ],
        vec![
            example(vec![int(7)], 1),
            example(vec![int(9)], 0),
        ],
        "fn is_prime(n: i64) -> i64 {\n    if n < 2 { return 0; }\n    if n == 2 { return 1; }\n    if n % 2 == 0 { return 0; }\n    i: i64 = 3;\n    while i * i <= n {\n        if n % i == 0 { return 0; }\n        i = i + 2;\n    }\n    return 1;\n}\n",
    )
}

fn make_nth_triangle(variant: usize) -> Problem {
    problem(
        "nth_triangle",
        variant,
        "loops",
        "Return the nth triangular number: 1+2+...+n.",
        "fn nth_triangle(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 1),
            example(vec![int(5)], 15),
            example(vec![int(10)], 55),
        ],
        vec![
            example(vec![int(7)], 28),
            example(vec![int(4)], 10),
        ],
        "fn nth_triangle(n: i64) -> i64 {\n    if n <= 0 {\n        return 0;\n    }\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        total = total + i;\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_fib_iter(variant: usize) -> Problem {
    problem(
        "fib_iter",
        variant,
        "loops",
        "Return the nth Fibonacci number using iterative multi-variable update.",
        "fn fib_iter(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 1),
            example(vec![int(7)], 13),
            example(vec![int(10)], 55),
        ],
        vec![
            example(vec![int(5)], 5),
            example(vec![int(8)], 21),
        ],
        "fn fib_iter(n: i64) -> i64 {\n    if n == 0 { return 0; }\n    if n == 1 { return 1; }\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= n {\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }\n    return b;\n}\n",
    )
}

fn make_palindrome_check(variant: usize) -> Problem {
    problem(
        "palindrome_check",
        variant,
        "strings",
        "Return 1 if the string is a palindrome, 0 otherwise.",
        "fn palindrome_check(s: string) -> i64",
        vec![
            example(vec![string("racecar")], 1),
            example(vec![string("hello")], 0),
            example(vec![string("aba")], 1),
            example(vec![string("ab")], 0),
            example(vec![string("a")], 1),
            example(vec![string("")], 1),
        ],
        vec![
            example(vec![string("level")], 1),
            example(vec![string("world")], 0),
        ],
        "fn palindrome_check(s: string) -> i64 {\n    chars := s.split(\"\");\n    left: i64 = 0;\n    right: i64 = s.len - 1;\n    while left < right {\n        if chars[left] != chars[right] { return 0; }\n        left = left + 1;\n        right = right - 1;\n    }\n    return 1;\n}\n",
    )
}

fn make_count_words(variant: usize) -> Problem {
    problem(
        "count_words",
        variant,
        "strings",
        "Count the number of space-separated words in a string.",
        "fn count_words(s: string) -> i64",
        vec![
            example(vec![string("hello world")], 2),
            example(vec![string("one")], 1),
            example(vec![string("a b c d")], 4),
            example(vec![string("  two words  ")], 2),
            example(vec![string("")], 0),
        ],
        vec![
            example(vec![string("foo bar baz")], 3),
            example(vec![string("  spaced  ")], 1),
        ],
        "fn count_words(s: string) -> i64 {\n    t := s.trim();\n    if t.len == 0 { return 0; }\n    parts := t.split(\" \");\n    count: i64 = 0;\n    for p in parts {\n        if p.len > 0 {\n            count = count + 1;\n        }\n    }\n    return count;\n}\n",
    )
}

fn make_euler_totient(variant: usize) -> Problem {
    problem(
        "euler_totient",
        variant,
        "algorithms",
        "Compute Euler's totient function phi(n).",
        "fn euler_totient(n: i64) -> i64",
        vec![
            example(vec![int(1)], 1),
            example(vec![int(2)], 1),
            example(vec![int(9)], 6),
            example(vec![int(12)], 4),
        ],
        vec![
            example(vec![int(5)], 4),
            example(vec![int(10)], 4),
        ],
        "fn euler_totient(n: i64) -> i64 {\n    result: i64 = n;\n    p: i64 = 2;\n    temp: i64 = n;\n    while p * p <= temp {\n        if temp % p == 0 {\n            while temp % p == 0 {\n                temp = temp / p;\n            }\n            result = result - result / p;\n        }\n        p = p + 1;\n    }\n    if temp > 1 {\n        result = result - result / temp;\n    }\n    return result;\n}\n",
    )
}

fn make_sum_squares(variant: usize) -> Problem {
    problem(
        "sum_squares",
        variant,
        "loops",
        "Compute the sum of squares from 1 to n.",
        "fn sum_squares(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 1),
            example(vec![int(3)], 14),
            example(vec![int(5)], 55),
        ],
        vec![
            example(vec![int(4)], 30),
            example(vec![int(2)], 5),
        ],
        "fn sum_squares(n: i64) -> i64 {\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        total = total + i * i;\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_product_1_to_n(variant: usize) -> Problem {
    problem(
        "product_1_to_n",
        variant,
        "loops",
        "Compute the product of all integers from 1 to n.",
        "fn product_1_to_n(n: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(1)], 1),
            example(vec![int(4)], 24),
            example(vec![int(6)], 720),
        ],
        vec![
            example(vec![int(3)], 6),
            example(vec![int(5)], 120),
        ],
        "fn product_1_to_n(n: i64) -> i64 {\n    if n == 0 { return 1; }\n    total: i64 = 1;\n    i: i64 = 1;\n    while i <= n {\n        total = total * i;\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_count_divisors(variant: usize) -> Problem {
    problem(
        "count_divisors",
        variant,
        "loops",
        "Count how many positive divisors n has.",
        "fn count_divisors(n: i64) -> i64",
        vec![
            example(vec![int(1)], 1),
            example(vec![int(2)], 2),
            example(vec![int(6)], 4),
            example(vec![int(12)], 6),
        ],
        vec![
            example(vec![int(4)], 3),
            example(vec![int(9)], 3),
        ],
        "fn count_divisors(n: i64) -> i64 {\n    count: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        if n % i == 0 {\n            count = count + 1;\n        }\n        i = i + 1;\n    }\n    return count;\n}\n",
    )
}

fn make_triangular_check(variant: usize) -> Problem {
    problem(
        "triangular_check",
        variant,
        "algorithms",
        "Return 1 if n is a triangular number, 0 otherwise.",
        "fn triangular_check(n: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(1)], 1),
            example(vec![int(3)], 1),
            example(vec![int(4)], 0),
        ],
        vec![
            example(vec![int(6)], 1),
            example(vec![int(5)], 0),
        ],
        "fn triangular_check(n: i64) -> i64 {\n    k: i64 = 0;\n    while k * (k + 1) / 2 <= n {\n        if k * (k + 1) / 2 == n { return 1; }\n        k = k + 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_max_pair_diff(variant: usize) -> Problem {
    problem(
        "max_pair_diff",
        variant,
        "arrays",
        "Find the maximum absolute difference between consecutive elements.",
        "fn max_pair_diff(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 5, 3])], 4),
            example(vec![array(&[10, 3, 9, 2])], 7),
            example(vec![array(&[7, 7, 7])], 0),
            example(vec![array(&[2, 11, 4, 20])], 16),
        ],
        vec![
            example(vec![array(&[1, 10])], 9),
            example(vec![array(&[5, 5, 5, 5])], 0),
        ],
        "fn max_pair_diff(arr: [i64]) -> i64 {\n    best: i64 = 0;\n    i: i64 = 1;\n    while i < arr.len {\n        diff: i64 = arr[i] - arr[i - 1];\n        if diff < 0 { diff = 0 - diff; }\n        if diff > best { best = diff; }\n        i = i + 1;\n    }\n    return best;\n}\n",
    )
}

fn make_sum_negatives(variant: usize) -> Problem {
    problem(
        "sum_negatives",
        variant,
        "arrays",
        "Sum all negative numbers in the array.",
        "fn sum_negatives(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[-1, 2, -3])], -4),
            example(vec![array(&[5, 1, 4])], 0),
            example(vec![array(&[-4, -9])], -13),
            example(vec![array(&[2, -7, 1, 0])], -7),
        ],
        vec![
            example(vec![array(&[-2, -2, 2])], -4),
            example(vec![array(&[1, 2, 3])], 0),
        ],
        "fn sum_negatives(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        if item < 0 {\n            total = total + item;\n        }\n    }\n    return total;\n}\n",
    )
}

fn make_gcd_extended(variant: usize) -> Problem {
    problem(
        "gcd_extended",
        variant,
        "algorithms",
        "Compute the GCD of two non-negative integers using Euclidean algorithm with variable swap.",
        "fn gcd_extended(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(12), int(8)], 4),
            example(vec![int(35), int(14)], 7),
            example(vec![int(7), int(13)], 1),
            example(vec![int(100), int(75)], 25),
            example(vec![int(0), int(5)], 5),
            example(vec![int(6), int(0)], 6),
        ],
        vec![
            example(vec![int(18), int(12)], 6),
            example(vec![int(17), int(5)], 1),
        ],
        "fn gcd_extended(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }\n    return x;\n}\n",
    )
}

fn make_harmonic_sum(variant: usize) -> Problem {
    problem(
        "harmonic_sum",
        variant,
        "loops",
        "Compute integer harmonic sum: sum of 1000/i for i from 1 to n.",
        "fn harmonic_sum(n: i64) -> i64",
        vec![
            example(vec![int(1)], 1000),
            example(vec![int(2)], 1500),
            example(vec![int(5)], 2283),
            example(vec![int(10)], 2927),
        ],
        vec![
            example(vec![int(3)], 1833),
            example(vec![int(4)], 2083),
        ],
        "fn harmonic_sum(n: i64) -> i64 {\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        total = total + 1000 / i;\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_second_max(variant: usize) -> Problem {
    problem(
        "second_max",
        variant,
        "arrays",
        "Return the second largest element in the array.",
        "fn second_max(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[3, 1, 4, 1, 5])], 4),
            example(vec![array(&[2, 8, 3])], 3),
            example(vec![array(&[7, 7, 2, 9])], 7),
            example(vec![array(&[1, 3])], 1),
        ],
        vec![
            example(vec![array(&[5, 10, 8])], 8),
            example(vec![array(&[4, 4, 4])], 4),
        ],
        "fn second_max(arr: [i64]) -> i64 {\n    first: i64 = arr[0];\n    second: i64 = arr[0];\n    for item in arr {\n        if item > first {\n            second = first;\n            first = item;\n        } else {\n            if item > second {\n                second = item;\n            }\n        }\n    }\n    return second;\n}\n",
    )
}

fn make_array_range(variant: usize) -> Problem {
    problem(
        "array_range",
        variant,
        "arrays",
        "Return max - min of the array.",
        "fn array_range(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[3, 1, 5])], 4),
            example(vec![array(&[7, 2, 9, 4])], 7),
            example(vec![array(&[5, 5])], 0),
            example(vec![array(&[1, 8, 3, 2])], 7),
        ],
        vec![
            example(vec![array(&[0, 10, 5])], 10),
            example(vec![array(&[-3, 3])], 6),
        ],
        "fn array_range(arr: [i64]) -> i64 {\n    lo: i64 = arr[0];\n    hi: i64 = arr[0];\n    for item in arr {\n        if item < lo {\n            lo = item;\n        }\n        if item > hi {\n            hi = item;\n        }\n    }\n    return hi - lo;\n}\n",
    )
}

fn make_sum_of_divisors(variant: usize) -> Problem {
    problem(
        "sum_of_divisors",
        variant,
        "loops",
        "Sum all positive divisors of n including 1 and n.",
        "fn sum_of_divisors(n: i64) -> i64",
        vec![
            example(vec![int(6)], 12),
            example(vec![int(12)], 28),
            example(vec![int(7)], 8),
            example(vec![int(1)], 1),
        ],
        vec![
            example(vec![int(4)], 7),
            example(vec![int(9)], 13),
        ],
        "fn sum_of_divisors(n: i64) -> i64 {\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        if n % i == 0 {\n            total = total + i;\n        }\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_sum_odd_digits(variant: usize) -> Problem {
    problem(
        "sum_odd_digits",
        variant,
        "loops",
        "Sum the digits of n that are odd (1,3,5,7,9).",
        "fn sum_odd_digits(n: i64) -> i64",
        vec![
            example(vec![int(135)], 9),
            example(vec![int(248)], 0),
            example(vec![int(19)], 10),
            example(vec![int(0)], 0),
        ],
        vec![
            example(vec![int(357)], 15),
            example(vec![int(246)], 0),
        ],
        "fn sum_odd_digits(n: i64) -> i64 {\n    x: i64 = n;\n    acc: i64 = 0;\n    while x > 0 {\n        d: i64 = x % 10;\n        if (d % 2) == 1 {\n            acc = acc + d;\n        }\n        x = x / 10;\n    }\n    return acc;\n}\n",
    )
}

fn make_count_zeros(variant: usize) -> Problem {
    problem(
        "count_zeros",
        variant,
        "arrays",
        "Count the number of zeros in the array.",
        "fn count_zeros(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 0, 2, 0])], 2),
            example(vec![array(&[0])], 1),
            example(vec![array(&[3, 5, 1])], 0),
            example(vec![array(&[0, 0, 0, 1])], 3),
        ],
        vec![
            example(vec![array(&[0, 1, 0])], 2),
            example(vec![array(&[1, 2, 3])], 0),
        ],
        "fn count_zeros(arr: [i64]) -> i64 {\n    count: i64 = 0;\n    for item in arr {\n        if item == 0 {\n            count = count + 1;\n        }\n    }\n    return count;\n}\n",
    )
}

fn make_max_consecutive_sum(variant: usize) -> Problem {
    problem(
        "max_consecutive_sum",
        variant,
        "arrays",
        "Return the maximum sum of any non-empty contiguous subarray (Kadane's algorithm).",
        "fn max_consecutive_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, -2, 3])], 3),
            example(vec![array(&[3, -1, 2])], 4),
            example(vec![array(&[-1, -2, -3])], -1),
            example(vec![array(&[2, 3, -1, 4])], 8),
        ],
        vec![
            example(vec![array(&[1, 2, 3])], 6),
            example(vec![array(&[-5, 4, -1, 2])], 5),
        ],
        "fn max_consecutive_sum(arr: [i64]) -> i64 {\n    current: i64 = 0;\n    best: i64 = arr[0];\n    for item in arr {\n        if current > 0 {\n            current = current + item;\n        } else {\n            current = item;\n        }\n        if current > best {\n            best = current;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_min_consecutive_sum(variant: usize) -> Problem {
    problem(
        "min_consecutive_sum",
        variant,
        "arrays",
        "Return the minimum sum of any non-empty contiguous subarray (anti-Kadane's algorithm).",
        "fn min_consecutive_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, -2, 3])], -2),
            example(vec![array(&[3, -1, -2, 5])], -3),
            example(vec![array(&[1, 2, 3, 4])], 1),
            example(vec![array(&[-2, -3, 1, -4])], -8),
        ],
        vec![
            example(vec![array(&[-1, -2, -3])], -6),
            example(vec![array(&[5, -1, 3])], -1),
        ],
        "fn min_consecutive_sum(arr: [i64]) -> i64 {\n    current: i64 = 0;\n    best: i64 = arr[0];\n    for item in arr {\n        if current < 0 {\n            current = current + item;\n        } else {\n            current = item;\n        }\n        if current < best {\n            best = current;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_interactive_sum(variant: usize) -> Problem {
    problem(
        "interactive_sum",
        variant,
        "arrays",
        "Return the sum of all integers in an array.",
        "fn interactive_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 6),
            example(vec![array(&[5, 1])], 6),
            example(vec![array(&[4, 4, 4])], 12),
            example(vec![array(&[2, 7, 1, 0])], 10),
        ],
        vec![
            example(vec![array(&[10, -5, 2])], 7),
            example(vec![array(&[1, 2, 3, 4])], 10),
        ],
        "fn interactive_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        total = total + item;\n    }\n    return total;\n}\n",
    )
}

pub const FACTORIES: &[Factory] = &[
    make_add_two,
    make_abs_diff,
    make_max2,
    make_clamp,
    make_sign,
    make_sum_to_n,
    make_gcd,
    make_lcm,
    make_array_sum,
    make_array_max,
    make_count_occurrences,
    make_trimmed_len,
    make_vowel_count,
    make_contains_cat,
    make_point_sum,
    make_safe_div_or_neg1,
    make_positive_or_default,
    make_factorial,
    make_fibonacci,
    make_closure_map_sum,
    make_count_positive,
    make_is_even,
    make_digit_sum,
    make_reverse_digits,
    make_digit_count,
    make_count_even_digits,
    make_starts_with_m,
    make_rectangle_area,
    make_power,
    make_polynomial,
    make_collatz_steps,
    make_min3,
    make_reverse_sum,
    make_array_max_elem,
    make_is_prime,
    make_nth_triangle,
    make_fib_iter,
    make_palindrome_check,
    make_count_words,
    make_euler_totient,
    make_sum_squares,
    make_product_1_to_n,
    make_count_divisors,
    make_triangular_check,
    make_max_pair_diff,
    make_sum_negatives,
    make_gcd_extended,
    make_harmonic_sum,
    make_interactive_sum,
    make_second_max,
    make_array_range,
    make_sum_of_divisors,
    make_sum_odd_digits,
    make_count_zeros,
    make_max_consecutive_sum,
    make_min_consecutive_sum,
    make_kth_smallest,
    make_max_stock_profit,
    make_is_sorted,
    make_longest_increasing_run,
    make_digital_root,
    make_two_sum_exists,
    make_count_distinct,
    make_binary_search,
    make_longest_plateau,
    make_prefix_max_sum,
    make_cube,
    make_square_plus_n,
    make_bilinear3,
    make_scaled_sum,
    make_product_offset,
    make_arr_sum_squares,
    make_min_element,
    make_sum_absolute,
    make_count_evens,
    make_sum_positives,
    make_sum_at_even_indices,
    make_kth_from_end,
    make_max_abs,
    make_digit_product,
    make_max_digit,
    make_min_positive,
    make_lucas_number,
    make_celsius_to_fahrenheit,
    make_is_perfect_square,
    make_next_power_of_2,
    make_count_peaks,
    make_alternating_sum,
    make_count_greater_than,
    make_dot_product,
    make_leading_digit,
    make_popcount,
    make_prefix_sum_k,
    make_is_palindrome_arr,
    make_sum_odd_indexed,
    // Tier-2: game-adjacent problems (April 2026)
    make_score_tracker,
    make_vending_change,
    make_combat_resolve,
    make_traffic_light_phase,
    make_run_length_decode_sum,
    make_count_adjacent_diff,
    make_priority_pop,
    make_turn_order_rotate,
    make_grid_bounds_check,
    make_simulate_gravity,
];

fn make_kth_smallest(variant: usize) -> Problem {
    problem(
        "kth_smallest",
        variant,
        "sorting",
        "Return the kth smallest element of the array (1-indexed).",
        "fn kth_smallest(arr: [i64], k: i64) -> i64",
        vec![
            example(vec![array(&[3, 1, 4, 1, 5]), int(2)], 1),
            example(vec![array(&[7, 2, 9, 4]), int(3)], 7),
            example(vec![array(&[5]), int(1)], 5),
            example(vec![array(&[1, 2, 3, 4, 5]), int(4)], 4),
        ],
        vec![
            example(vec![array(&[10, 3, 7, 1]), int(2)], 3),
            example(vec![array(&[-2, 0, 4]), int(1)], -2),
        ],
        "fn kth_smallest(arr: [i64], k: i64) -> i64 {\n    arr.sort();\n    return arr[k - 1];\n}\n",
    )
}

fn make_max_stock_profit(variant: usize) -> Problem {
    problem(
        "max_stock_profit",
        variant,
        "arrays",
        "Given daily prices, return the maximum profit from buying once then selling later. If no profit possible, return 0.",
        "fn max_stock_profit(prices: [i64]) -> i64",
        vec![
            example(vec![array(&[7, 1, 5, 3, 6, 4])], 5),
            example(vec![array(&[7, 6, 4, 3, 1])], 0),
            example(vec![array(&[1, 2])], 1),
            example(vec![array(&[2, 4, 1, 7])], 6),
        ],
        vec![
            example(vec![array(&[3, 3, 3])], 0),
            example(vec![array(&[1, 5, 2, 8])], 7),
        ],
        "fn max_stock_profit(prices: [i64]) -> i64 {\n    min_price: i64 = prices[0];\n    best: i64 = 0;\n    for p in prices {\n        if p < min_price { min_price = p; }\n        profit: i64 = p - min_price;\n        if profit > best { best = profit; }\n    }\n    return best;\n}\n",
    )
}

fn make_is_sorted(variant: usize) -> Problem {
    problem(
        "is_sorted",
        variant,
        "arrays",
        "Return 1 if the array is sorted in non-decreasing order, 0 otherwise.",
        "fn is_sorted(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 1),
            example(vec![array(&[3, 2, 1])], 0),
            example(vec![array(&[1, 1, 2])], 1),
            example(vec![array(&[5])], 1),
        ],
        vec![
            example(vec![array(&[1, 3, 2])], 0),
            example(vec![array(&[-2, 0, 4])], 1),
        ],
        "fn is_sorted(arr: [i64]) -> i64 {\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] < arr[i - 1] { return 0; }\n        i = i + 1;\n    }\n    return 1;\n}\n",
    )
}

fn make_longest_increasing_run(variant: usize) -> Problem {
    problem(
        "longest_increasing_run",
        variant,
        "arrays",
        "Return the length of the longest strictly increasing consecutive run.",
        "fn longest_increasing_run(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 3, 2, 4, 5])], 3),
            example(vec![array(&[5, 4, 3, 2, 1])], 1),
            example(vec![array(&[1, 2, 3, 4])], 4),
            example(vec![array(&[1, 2, 1, 2, 3])], 3),
        ],
        vec![
            example(vec![array(&[3, 1, 2])], 2),
            example(vec![array(&[1, 1, 1])], 1),
        ],
        "fn longest_increasing_run(arr: [i64]) -> i64 {\n    best: i64 = 1;\n    cur: i64 = 1;\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] > arr[i - 1] {\n            cur = cur + 1;\n            if cur > best { best = cur; }\n        } else {\n            cur = 1;\n        }\n        i = i + 1;\n    }\n    return best;\n}\n",
    )
}

fn make_digital_root(variant: usize) -> Problem {
    problem(
        "digital_root",
        variant,
        "loops",
        "Repeatedly sum the digits of n until a single digit remains.",
        "fn digital_root(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(9)], 9),
            example(vec![int(493)], 7),
            example(vec![int(942)], 6),
        ],
        vec![
            example(vec![int(18)], 9),
            example(vec![int(11)], 2),
        ],
        "fn digital_root(n: i64) -> i64 {\n    x: i64 = n;\n    while x >= 10 {\n        s: i64 = 0;\n        while x > 0 {\n            s = s + x % 10;\n            x = x / 10;\n        }\n        x = s;\n    }\n    return x;\n}\n",
    )
}

fn make_two_sum_exists(variant: usize) -> Problem {
    problem(
        "two_sum_exists",
        variant,
        "arrays",
        "Return 1 if any two distinct elements sum to target, 0 otherwise.",
        "fn two_sum_exists(arr: [i64], target: i64) -> i64",
        vec![
            example(vec![array(&[2, 7, 11, 15]), int(9)], 1),
            example(vec![array(&[3, 5, 8]), int(4)], 0),
            example(vec![array(&[1, 2, 3]), int(5)], 1),
            example(vec![array(&[1, 2, 3]), int(7)], 0),
        ],
        vec![
            example(vec![array(&[4, 4]), int(8)], 1),
            example(vec![array(&[1, 5, 3]), int(6)], 1),
        ],
        "fn two_sum_exists(arr: [i64], target: i64) -> i64 {\n    i: i64 = 0;\n    while i < arr.len {\n        j: i64 = i + 1;\n        while j < arr.len {\n            if arr[i] + arr[j] == target { return 1; }\n            j = j + 1;\n        }\n        i = i + 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_count_distinct(variant: usize) -> Problem {
    problem(
        "count_distinct",
        variant,
        "sorting",
        "Count the number of distinct values in an array.",
        "fn count_distinct(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 2, 3, 3, 3])], 3),
            example(vec![array(&[5, 5, 5])], 1),
            example(vec![array(&[1, 2, 3, 4])], 4),
            example(vec![array(&[7])], 1),
        ],
        vec![
            example(vec![array(&[1, 1, 2, 3])], 3),
            example(vec![array(&[0, 0, 0, 1, 1])], 2),
        ],
        "fn count_distinct(arr: [i64]) -> i64 {\n    arr.sort();\n    count: i64 = 1;\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] != arr[i - 1] {\n            count = count + 1;\n        }\n        i = i + 1;\n    }\n    return count;\n}\n",
    )
}

fn make_binary_search(variant: usize) -> Problem {
    problem(
        "binary_search",
        variant,
        "sorting",
        "Return the index of target in a sorted array, or -1 if not found.",
        "fn binary_search(arr: [i64], target: i64) -> i64",
        vec![
            example(vec![array(&[1, 3, 5, 7, 9]), int(5)], 2),
            example(vec![array(&[1, 3, 5, 7, 9]), int(6)], -1),
            example(vec![array(&[2, 4, 6, 8]), int(2)], 0),
            example(vec![array(&[2, 4, 6, 8]), int(8)], 3),
        ],
        vec![
            example(vec![array(&[10, 20, 30]), int(20)], 1),
            example(vec![array(&[1, 2, 3, 4, 5]), int(0)], -1),
        ],
        "fn binary_search(arr: [i64], target: i64) -> i64 {\n    lo: i64 = 0;\n    hi: i64 = arr.len - 1;\n    while lo <= hi {\n        mid: i64 = (lo + hi) / 2;\n        if arr[mid] == target { return mid; }\n        if arr[mid] < target { lo = mid + 1; }\n        if arr[mid] > target { hi = mid - 1; }\n    }\n    return -1;\n}\n",
    )
}

fn make_longest_plateau(variant: usize) -> Problem {
    problem(
        "longest_plateau",
        variant,
        "arrays",
        "Return the length of the longest run of equal consecutive elements.",
        "fn longest_plateau(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 1, 2, 2, 2, 1])], 3),
            example(vec![array(&[5, 5, 5, 5])], 4),
            example(vec![array(&[1, 2, 3])], 1),
            example(vec![array(&[3, 3, 1, 1, 1, 2])], 3),
        ],
        vec![
            example(vec![array(&[7, 7, 3, 3])], 2),
            example(vec![array(&[1])], 1),
        ],
        "fn longest_plateau(arr: [i64]) -> i64 {\n    best: i64 = 1;\n    cur: i64 = 1;\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] == arr[i - 1] {\n            cur = cur + 1;\n            if cur > best { best = cur; }\n        } else {\n            cur = 1;\n        }\n        i = i + 1;\n    }\n    return best;\n}\n",
    )
}

fn make_prefix_max_sum(variant: usize) -> Problem {
    problem(
        "prefix_max_sum",
        variant,
        "arrays",
        "Sum the running maximum: for each position i, add the max of arr[0..=i].",
        "fn prefix_max_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 3, 2, 5])], 12),
            example(vec![array(&[5, 4, 3])], 15),
            example(vec![array(&[1, 1, 1])], 3),
            example(vec![array(&[2, 5, 3, 8])], 20),
        ],
        vec![
            example(vec![array(&[3, 1, 4, 2])], 14),
            example(vec![array(&[7])], 7),
        ],
        "fn prefix_max_sum(arr: [i64]) -> i64 {\n    running_max: i64 = arr[0];\n    total: i64 = 0;\n    for x in arr {\n        if x > running_max { running_max = x; }\n        total = total + running_max;\n    }\n    return total;\n}\n",
    )
}

fn make_cube(variant: usize) -> Problem {
    problem(
        "cube",
        variant,
        "arithmetic",
        "Return x cubed (x * x * x).",
        "fn cube(x: i64) -> i64",
        vec![
            example(vec![int(2)], 8),
            example(vec![int(3)], 27),
            example(vec![int(4)], 64),
            example(vec![int(1)], 1),
        ],
        vec![example(vec![int(0)], 0), example(vec![int(-2)], -8)],
        "fn cube(x: i64) -> i64 {\n    return x * x * x;\n}\n",
    )
}

fn make_square_plus_n(variant: usize) -> Problem {
    problem(
        "square_plus_n",
        variant,
        "arithmetic",
        "Return n*n + n (equivalently n*(n+1)).",
        "fn square_plus_n(n: i64) -> i64",
        vec![
            example(vec![int(1)], 2),
            example(vec![int(2)], 6),
            example(vec![int(3)], 12),
            example(vec![int(4)], 20),
        ],
        vec![example(vec![int(0)], 0), example(vec![int(5)], 30)],
        "fn square_plus_n(n: i64) -> i64 {\n    return (n * n) + n;\n}\n",
    )
}

fn make_bilinear3(variant: usize) -> Problem {
    problem(
        "bilinear3",
        variant,
        "arithmetic",
        "Return a*b + c.",
        "fn bilinear3(a: i64, b: i64, c: i64) -> i64",
        vec![
            example(vec![int(2), int(3), int(1)], 7),
            example(vec![int(3), int(2), int(5)], 11),
            example(vec![int(4), int(1), int(0)], 4),
            example(vec![int(1), int(4), int(2)], 6),
        ],
        vec![
            example(vec![int(5), int(2), int(3)], 13),
            example(vec![int(2), int(5), int(1)], 11),
        ],
        "fn bilinear3(a: i64, b: i64, c: i64) -> i64 {\n    return a * b + c;\n}\n",
    )
}

fn make_scaled_sum(variant: usize) -> Problem {
    problem(
        "scaled_sum",
        variant,
        "arithmetic",
        "Return 2*a + b.",
        "fn scaled_sum(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(3), int(1)], 7),
            example(vec![int(2), int(4)], 8),
            example(vec![int(5), int(0)], 10),
            example(vec![int(1), int(3)], 5),
        ],
        vec![
            example(vec![int(4), int(3)], 11),
            example(vec![int(0), int(6)], 6),
        ],
        "fn scaled_sum(a: i64, b: i64) -> i64 {\n    return 2 * a + b;\n}\n",
    )
}

fn make_product_offset(variant: usize) -> Problem {
    problem(
        "product_offset",
        variant,
        "arithmetic",
        "Return a*b - a (equivalently a*(b-1)).",
        "fn product_offset(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(3), int(4)], 9),
            example(vec![int(2), int(5)], 8),
            example(vec![int(5), int(2)], 5),
            example(vec![int(4), int(3)], 8),
        ],
        vec![
            example(vec![int(1), int(7)], 6),
            example(vec![int(6), int(0)], -6),
        ],
        "fn product_offset(a: i64, b: i64) -> i64 {\n    return a * b - a;\n}\n",
    )
}

fn make_arr_sum_squares(variant: usize) -> Problem {
    problem(
        "arr_sum_squares",
        variant,
        "arrays",
        "Return the sum of squares of all elements.",
        "fn arr_sum_squares(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 14),
            example(vec![array(&[4])], 16),
            example(vec![array(&[0, 5])], 25),
            example(vec![array(&[-3, 4])], 25),
        ],
        vec![
            example(vec![array(&[2, 2, 2])], 12),
            example(vec![array(&[1, 1, 1, 1])], 4),
        ],
        "fn arr_sum_squares(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for x in arr {\n        acc = acc + x * x;\n    }\n    return acc;\n}\n",
    )
}

fn make_min_element(variant: usize) -> Problem {
    problem(
        "min_element",
        variant,
        "arrays",
        "Return the minimum element of the array.",
        "fn min_element(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[3, 1, 4, 2])], 1),
            example(vec![array(&[5, 9, 2, 6])], 2),
            example(vec![array(&[7])], 7),
            example(vec![array(&[-3, 0, 2])], -3),
        ],
        vec![
            example(vec![array(&[4, 4, 4])], 4),
            example(vec![array(&[10, -5, 3])], -5),
        ],
        "fn min_element(arr: [i64]) -> i64 {\n    best: i64 = arr[0];\n    for x in arr {\n        if x < best {\n            best = x;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_sum_absolute(variant: usize) -> Problem {
    problem(
        "sum_absolute",
        variant,
        "arrays",
        "Return the sum of absolute values of all elements.",
        "fn sum_absolute(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, -2, 3])], 6),
            example(vec![array(&[-4, -5])], 9),
            example(vec![array(&[0, 3, -3])], 6),
            example(vec![array(&[2, 2, 2])], 6),
        ],
        vec![
            example(vec![array(&[-1, 1, -1, 1])], 4),
            example(vec![array(&[10, -3])], 13),
        ],
        "fn sum_absolute(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for x in arr {\n        if x < 0 {\n            acc = acc + (0 - x);\n        } else {\n            acc = acc + x;\n        }\n    }\n    return acc;\n}\n",
    )
}

fn make_count_evens(variant: usize) -> Problem {
    problem(
        "count_evens",
        variant,
        "arrays",
        "Return the count of even elements in the array.",
        "fn count_evens(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4])], 2),
            example(vec![array(&[1, 3, 5])], 0),
            example(vec![array(&[2, 4, 6])], 3),
            example(vec![array(&[0, 1, 2, 3, 4])], 3),
        ],
        vec![
            example(vec![array(&[2, 2, 1, 1])], 2),
            example(vec![array(&[7, 8, 9])], 1),
        ],
        "fn count_evens(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for x in arr {\n        if (x % 2) == 0 {\n            acc = acc + 1;\n        }\n    }\n    return acc;\n}\n",
    )
}

fn make_sum_positives(variant: usize) -> Problem {
    problem(
        "sum_positives",
        variant,
        "arrays",
        "Return the sum of all positive elements (excluding zero).",
        "fn sum_positives(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, -2, 3, -4])], 4),
            example(vec![array(&[-1, -2, -3])], 0),
            example(vec![array(&[5, 0, 3])], 8),
            example(vec![array(&[2, 4, 6])], 12),
        ],
        vec![
            example(vec![array(&[0, 0, 1])], 1),
            example(vec![array(&[-5, 10, -3, 2])], 12),
        ],
        "fn sum_positives(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for x in arr {\n        if x > 0 {\n            acc = acc + x;\n        }\n    }\n    return acc;\n}\n",
    )
}

fn make_sum_at_even_indices(variant: usize) -> Problem {
    problem(
        "sum_at_even_indices",
        variant,
        "arrays",
        "Return the sum of elements at even indices (0, 2, 4, ...).",
        "fn sum_at_even_indices(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4, 5])], 9),
            example(vec![array(&[10, 20, 30])], 40),
            example(vec![array(&[5, 5])], 5),
            example(vec![array(&[1, 2, 3, 4])], 4),
        ],
        vec![
            example(vec![array(&[2, 1, 4, 3])], 6),
            example(vec![array(&[7, 0, 3, 0, 1])], 11),
        ],
        "fn sum_at_even_indices(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        acc = acc + arr[i];\n        i = i + 2;\n    }\n    return acc;\n}\n",
    )
}

fn make_kth_from_end(variant: usize) -> Problem {
    problem(
        "kth_from_end",
        variant,
        "arrays",
        "Return the kth element from the end (k=1 is the last).",
        "fn kth_from_end(arr: [i64], k: i64) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4, 5]), int(1)], 5),
            example(vec![array(&[1, 2, 3, 4, 5]), int(3)], 3),
            example(vec![array(&[10, 20]), int(2)], 10),
            example(vec![array(&[7, 8, 9]), int(2)], 8),
        ],
        vec![
            example(vec![array(&[1, 2, 3]), int(1)], 3),
            example(vec![array(&[5, 10, 15, 20]), int(3)], 10),
        ],
        "fn kth_from_end(arr: [i64], k: i64) -> i64 {\n    return arr[arr.len - k];\n}\n",
    )
}

fn make_max_abs(variant: usize) -> Problem {
    problem(
        "max_abs",
        variant,
        "arrays",
        "Return the largest absolute value in the array.",
        "fn max_abs(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, -5, 3])], 5),
            example(vec![array(&[-2, -7, 4])], 7),
            example(vec![array(&[3, 3, 3])], 3),
            example(vec![array(&[0, -1])], 1),
        ],
        vec![
            example(vec![array(&[-10, 5, -3])], 10),
            example(vec![array(&[4, -4])], 4),
        ],
        "fn max_abs(arr: [i64]) -> i64 {\n    best: i64 = 0;\n    for x in arr {\n        v: i64 = x;\n        if v < 0 {\n            v = 0 - v;\n        }\n        if v > best {\n            best = v;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_alternating_sum(variant: usize) -> Problem {
    problem(
        "alternating_sum",
        variant,
        "array",
        "Return the alternating sum: arr[0] - arr[1] + arr[2] - arr[3] + ...",
        "fn alternating_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4])], -2),
            example(vec![array(&[5])], 5),
            example(vec![array(&[3, 1])], 2),
            example(vec![array(&[2, 4, 6, 8])], -4),
        ],
        vec![
            example(vec![array(&[1, 1, 1, 1])], 0),
            example(vec![array(&[10, 3, 2, 1])], 8),
        ],
        "fn alternating_sum(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    i: i64 = 0;\n    sign: i64 = 1;\n    while i < arr.len {\n        acc = acc + sign * arr[i];\n        sign = 0 - sign;\n        i = i + 1;\n    }\n    return acc;\n}\n",
    )
}

fn make_count_greater_than(variant: usize) -> Problem {
    problem(
        "count_greater_than",
        variant,
        "array",
        "Return the count of elements greater than k.",
        "fn count_greater_than(arr: [i64], k: i64) -> i64",
        vec![
            example(vec![array(&[1, 5, 3, 7, 2]), int(4)], 2),
            example(vec![array(&[1, 2, 3]), int(0)], 3),
            example(vec![array(&[4, 4, 4]), int(4)], 0),
            example(vec![array(&[10, 20, 30]), int(15)], 2),
        ],
        vec![
            example(vec![array(&[1, 2, 3, 4, 5]), int(3)], 2),
            example(vec![array(&[0, -1, 1]), int(0)], 1),
        ],
        "fn count_greater_than(arr: [i64], k: i64) -> i64 {\n    acc: i64 = 0;\n    for item in arr {\n        if item > k {\n            acc = acc + 1;\n        }\n    }\n    return acc;\n}\n",
    )
}

fn make_dot_product(variant: usize) -> Problem {
    problem(
        "dot_product",
        variant,
        "array",
        "Return the dot product: sum of a[i] * b[i] for all i.",
        "fn dot_product(a: [i64], b: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3]), array(&[4, 5, 6])], 32),
            example(vec![array(&[1, 0]), array(&[0, 1])], 0),
            example(vec![array(&[2, 3]), array(&[3, 2])], 12),
            example(vec![array(&[1, 1, 1, 1]), array(&[1, 2, 3, 4])], 10),
        ],
        vec![
            example(vec![array(&[5, 0, 1]), array(&[1, 1, 1])], 6),
            example(vec![array(&[2, 2]), array(&[3, 3])], 12),
        ],
        "fn dot_product(a: [i64], b: [i64]) -> i64 {\n    acc: i64 = 0;\n    i: i64 = 0;\n    while i < a.len {\n        acc = acc + a[i] * b[i];\n        i = i + 1;\n    }\n    return acc;\n}\n",
    )
}

fn make_leading_digit(variant: usize) -> Problem {
    problem(
        "leading_digit",
        variant,
        "integer",
        "Return the leading (most significant) digit of a positive integer.",
        "fn leading_digit(n: i64) -> i64",
        vec![
            example(vec![int(1234)], 1),
            example(vec![int(5)], 5),
            example(vec![int(100)], 1),
            example(vec![int(73)], 7),
        ],
        vec![
            example(vec![int(999)], 9),
            example(vec![int(42)], 4),
        ],
        "fn leading_digit(n: i64) -> i64 {\n    x: i64 = n;\n    while x >= 10 {\n        x = x / 10;\n    }\n    return x;\n}\n",
    )
}

fn make_popcount(variant: usize) -> Problem {
    problem(
        "popcount",
        variant,
        "integer",
        "Return the number of 1-bits in the binary representation of n (n >= 0).",
        "fn popcount(n: i64) -> i64",
        vec![
            example(vec![int(5)], 2),   // 101
            example(vec![int(7)], 3),   // 111
            example(vec![int(8)], 1),   // 1000
            example(vec![int(15)], 4),  // 1111
        ],
        vec![
            example(vec![int(0)], 0),
            example(vec![int(255)], 8),
        ],
        "fn popcount(n: i64) -> i64 {\n    x: i64 = n;\n    acc: i64 = 0;\n    while x > 0 {\n        acc = acc + x % 2;\n        x = x / 2;\n    }\n    return acc;\n}\n",
    )
}

fn make_prefix_sum_k(variant: usize) -> Problem {
    problem(
        "prefix_sum_k",
        variant,
        "array",
        "Return the sum of the first k elements.",
        "fn prefix_sum_k(arr: [i64], k: i64) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4, 5]), int(3)], 6),
            example(vec![array(&[10, 20, 30]), int(2)], 30),
            example(vec![array(&[5]), int(1)], 5),
            example(vec![array(&[1, 2, 3, 4, 5]), int(5)], 15),
        ],
        vec![
            example(vec![array(&[3, 1, 4, 1, 5]), int(2)], 4),
            example(vec![array(&[7, 7, 7]), int(3)], 21),
        ],
        "fn prefix_sum_k(arr: [i64], k: i64) -> i64 {\n    acc: i64 = 0;\n    i: i64 = 0;\n    while i < k {\n        acc = acc + arr[i];\n        i = i + 1;\n    }\n    return acc;\n}\n",
    )
}

fn make_is_palindrome_arr(variant: usize) -> Problem {
    problem(
        "is_palindrome_arr",
        variant,
        "array",
        "Return 1 if the array reads the same forwards and backwards, 0 otherwise.",
        "fn is_palindrome_arr(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 1])], 1),
            example(vec![array(&[1, 2, 3])], 0),
            example(vec![array(&[3, 1, 3])], 1),
            example(vec![array(&[1])], 1),
        ],
        vec![
            example(vec![array(&[1, 2, 2, 1])], 1),
            example(vec![array(&[1, 2, 3, 4])], 0),
        ],
        "fn is_palindrome_arr(arr: [i64]) -> i64 {\n    i: i64 = 0;\n    j: i64 = arr.len - 1;\n    while i < j {\n        if arr[i] != arr[j] {\n            return 0;\n        }\n        i = i + 1;\n        j = j - 1;\n    }\n    return 1;\n}\n",
    )
}

fn make_sum_odd_indexed(variant: usize) -> Problem {
    problem(
        "sum_odd_indexed",
        variant,
        "array",
        "Return the sum of elements at odd indices (1, 3, 5, ...).",
        "fn sum_odd_indexed(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4])], 6),   // 2+4
            example(vec![array(&[5, 3])], 3),
            example(vec![array(&[1, 10, 1, 10, 1])], 20),
            example(vec![array(&[0, 7, 0, 8])], 15),
        ],
        vec![
            example(vec![array(&[1, 1, 1, 1, 1, 1])], 3),
            example(vec![array(&[0, 5, 0, 3])], 8),
        ],
        "fn sum_odd_indexed(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    i: i64 = 1;\n    while i < arr.len {\n        acc = acc + arr[i];\n        i = i + 2;\n    }\n    return acc;\n}\n",
    )
}

fn make_digit_product(variant: usize) -> Problem {
    problem(
        "digit_product",
        variant,
        "integer",
        "Return the product of the digits of a positive integer.",
        "fn digit_product(n: i64) -> i64",
        vec![
            example(vec![int(123)], 6),
            example(vec![int(24)], 8),
            example(vec![int(100)], 0),
            example(vec![int(9)], 9),
        ],
        vec![
            example(vec![int(11)], 1),
            example(vec![int(55)], 25),
        ],
        "fn digit_product(n: i64) -> i64 {\n    x: i64 = n;\n    acc: i64 = 1;\n    while x > 0 {\n        acc = acc * (x % 10);\n        x = x / 10;\n    }\n    return acc;\n}\n",
    )
}

fn make_max_digit(variant: usize) -> Problem {
    problem(
        "max_digit",
        variant,
        "integer",
        "Return the maximum digit of a non-negative integer.",
        "fn max_digit(n: i64) -> i64",
        vec![
            example(vec![int(1234)], 4),
            example(vec![int(9000)], 9),
            example(vec![int(123)], 3),
            example(vec![int(55)], 5),
        ],
        vec![
            example(vec![int(999)], 9),
            example(vec![int(21)], 2),
        ],
        "fn max_digit(n: i64) -> i64 {\n    x: i64 = n;\n    best: i64 = 0;\n    while x > 0 {\n        d: i64 = x % 10;\n        if d > best {\n            best = d;\n        }\n        x = x / 10;\n    }\n    return best;\n}\n",
    )
}

fn make_min_positive(variant: usize) -> Problem {
    problem(
        "min_positive",
        variant,
        "array",
        "Return the minimum positive element, or 0 if no positive elements exist.",
        "fn min_positive(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 5, 3])], 1),
            example(vec![array(&[2, 8, 6])], 2),
            example(vec![array(&[-1, -2, -3])], 0),
            example(vec![array(&[10, 1, 5])], 1),
        ],
        vec![
            example(vec![array(&[3, 7, 2, 4])], 2),
            example(vec![array(&[0, 0, 0])], 0),
        ],
        "fn min_positive(arr: [i64]) -> i64 {\n    best: i64 = 0;\n    found: i64 = 0;\n    for x in arr {\n        if x > 0 {\n            if found == 0 {\n                best = x;\n                found = 1;\n            } else {\n                if x < best {\n                    best = x;\n                }\n            }\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_lucas_number(variant: usize) -> Problem {
    problem(
        "lucas_number",
        variant,
        "math",
        "Return the nth Lucas number: L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2).",
        "fn lucas_number(n: i64) -> i64",
        vec![
            example(vec![int(0)], 2),
            example(vec![int(1)], 1),
            example(vec![int(4)], 7),
            example(vec![int(6)], 18),
        ],
        vec![
            example(vec![int(3)], 4),
            example(vec![int(7)], 29),
        ],
        "fn lucas_number(n: i64) -> i64 {\n    if n == 0 {\n        return 2;\n    }\n    if n == 1 {\n        return 1;\n    }\n    a: i64 = 2;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= n {\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }\n    return b;\n}\n",
    )
}

fn make_celsius_to_fahrenheit(variant: usize) -> Problem {
    problem(
        "celsius_to_fahrenheit",
        variant,
        "math",
        "Convert Celsius to Fahrenheit using integer arithmetic: c * 9 / 5 + 32.",
        "fn celsius_to_fahrenheit(c: i64) -> i64",
        vec![
            example(vec![int(0)], 32),
            example(vec![int(100)], 212),
            example(vec![int(20)], 68),
            example(vec![int(37)], 98),
        ],
        vec![example(vec![int(-40)], -40), example(vec![int(25)], 77)],
        "fn celsius_to_fahrenheit(c: i64) -> i64 {\n    return c * 9 / 5 + 32;\n}\n",
    )
}

fn make_is_perfect_square(variant: usize) -> Problem {
    problem(
        "is_perfect_square",
        variant,
        "math",
        "Return 1 if n is a perfect square, 0 otherwise (n >= 0).",
        "fn is_perfect_square(n: i64) -> i64",
        vec![
            example(vec![int(4)], 1),
            example(vec![int(9)], 1),
            example(vec![int(10)], 0),
            example(vec![int(1)], 1),
        ],
        vec![
            example(vec![int(25)], 1),
            example(vec![int(26)], 0),
        ],
        "fn is_perfect_square(n: i64) -> i64 {\n    i: i64 = 0;\n    while i * i <= n {\n        if i * i == n {\n            return 1;\n        }\n        i = i + 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_next_power_of_2(variant: usize) -> Problem {
    problem(
        "next_power_of_2",
        variant,
        "math",
        "Return the smallest power of 2 that is >= n.",
        "fn next_power_of_2(n: i64) -> i64",
        vec![
            example(vec![int(1)], 1),
            example(vec![int(3)], 4),
            example(vec![int(5)], 8),
            example(vec![int(8)], 8),
        ],
        vec![
            example(vec![int(9)], 16),
            example(vec![int(16)], 16),
        ],
        "fn next_power_of_2(n: i64) -> i64 {\n    p: i64 = 1;\n    while p < n {\n        p = p * 2;\n    }\n    return p;\n}\n",
    )
}

fn make_count_peaks(variant: usize) -> Problem {
    problem(
        "count_peaks",
        variant,
        "array",
        "Count elements strictly greater than both their neighbors.",
        "fn count_peaks(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 3, 2])], 1),
            example(vec![array(&[1, 2, 3])], 0),
            example(vec![array(&[1, 3, 2, 4, 1])], 2),
            example(vec![array(&[5, 1, 5, 1, 5])], 1),
        ],
        vec![
            example(vec![array(&[3, 1, 4, 1, 5, 9, 2])], 2),
            example(vec![array(&[1, 2, 3, 2, 1])], 1),
        ],
        "fn count_peaks(arr: [i64]) -> i64 {\n    count: i64 = 0;\n    i: i64 = 1;\n    while i < arr.len - 1 {\n        if arr[i] > arr[i - 1] {\n            if arr[i] > arr[i + 1] {\n                count = count + 1;\n            }\n        }\n        i = i + 1;\n    }\n    return count;\n}\n",
    )
}

// ---------------------------------------------------------------------------
// Tier-2: game-adjacent problems (April 2026)
//
// These push past the single-clean-formula shapes of Tier-1 and require
// multi-branch dispatch, multi-arg coupling, or compositional array work.
// They are starting points toward game logic and app-style control flow.
// ---------------------------------------------------------------------------

fn make_score_tracker(variant: usize) -> Problem {
    problem(
        "score_tracker",
        variant,
        "game",
        "Update score: event 0 adds 1, event 1 adds 5, event 2 resets to 0, else unchanged.",
        "fn score_tracker(score: i64, event: i64) -> i64",
        vec![
            example(vec![int(10), int(0)], 11),
            example(vec![int(10), int(1)], 15),
            example(vec![int(10), int(2)], 0),
            example(vec![int(10), int(7)], 10),
            example(vec![int(0), int(1)], 5),
        ],
        vec![
            example(vec![int(42), int(0)], 43),
            example(vec![int(99), int(2)], 0),
        ],
        "fn score_tracker(score: i64, event: i64) -> i64 {\n    if event == 0 { return score + 1; }\n    if event == 1 { return score + 5; }\n    if event == 2 { return 0; }\n    return score;\n}\n",
    )
}

fn make_vending_change(variant: usize) -> Problem {
    problem(
        "vending_change",
        variant,
        "game",
        "Return coins_in - price if coins_in >= price, else -1.",
        "fn vending_change(coins_in: i64, price: i64) -> i64",
        vec![
            example(vec![int(100), int(75)], 25),
            example(vec![int(50), int(50)], 0),
            example(vec![int(30), int(75)], -1),
            example(vec![int(200), int(125)], 75),
            example(vec![int(10), int(20)], -1),
        ],
        vec![
            example(vec![int(80), int(80)], 0),
            example(vec![int(0), int(5)], -1),
        ],
        "fn vending_change(coins_in: i64, price: i64) -> i64 {\n    if coins_in >= price { return coins_in - price; }\n    return -1;\n}\n",
    )
}

fn make_combat_resolve(variant: usize) -> Problem {
    problem(
        "combat_resolve",
        variant,
        "game",
        "Damage dealt is max(attack - defense, 0).",
        "fn combat_resolve(attack: i64, defense: i64) -> i64",
        vec![
            example(vec![int(15), int(10)], 5),
            example(vec![int(5), int(10)], 0),
            example(vec![int(20), int(20)], 0),
            example(vec![int(100), int(1)], 99),
            example(vec![int(0), int(50)], 0),
        ],
        vec![
            example(vec![int(7), int(3)], 4),
            example(vec![int(50), int(75)], 0),
        ],
        "fn combat_resolve(attack: i64, defense: i64) -> i64 {\n    d: i64 = attack - defense;\n    if d < 0 { return 0; }\n    return d;\n}\n",
    )
}

fn make_traffic_light_phase(variant: usize) -> Problem {
    problem(
        "traffic_light_phase",
        variant,
        "game",
        "7-step cycle: steps 0-2 → 0 (red), 3-5 → 1 (green), 6 → 2 (yellow).",
        "fn traffic_light_phase(step: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 0),
            example(vec![int(2)], 0),
            example(vec![int(3)], 1),
            example(vec![int(5)], 1),
            example(vec![int(6)], 2),
            example(vec![int(7)], 0),
            example(vec![int(13)], 2),
        ],
        vec![
            example(vec![int(14)], 0),
            example(vec![int(20)], 2),
        ],
        "fn traffic_light_phase(step: i64) -> i64 {\n    m: i64 = step % 7;\n    if m < 3 { return 0; }\n    if m < 6 { return 1; }\n    return 2;\n}\n",
    )
}

fn make_run_length_decode_sum(variant: usize) -> Problem {
    problem(
        "run_length_decode_sum",
        variant,
        "array",
        "Interpret array as (count, value) pairs; return sum of count*value products.",
        "fn run_length_decode_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[3, 5, 2, 10])], 35),
            example(vec![array(&[1, 7])], 7),
            example(vec![array(&[4, 2, 3, 1])], 11),
            example(vec![array(&[2, 3, 2, 4, 2, 5])], 24),
        ],
        vec![
            example(vec![array(&[5, 1, 5, 1])], 10),
            example(vec![array(&[1, 100])], 100),
        ],
        "fn run_length_decode_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        total = total + arr[i] * arr[i + 1];\n        i = i + 2;\n    }\n    return total;\n}\n",
    )
}

fn make_count_adjacent_diff(variant: usize) -> Problem {
    problem(
        "count_adjacent_diff",
        variant,
        "array",
        "Count positions i > 0 where arr[i] != arr[i-1].",
        "fn count_adjacent_diff(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 1, 2, 2, 3])], 2),
            example(vec![array(&[5, 5, 5, 5])], 0),
            example(vec![array(&[1, 2, 1, 2, 1])], 4),
            example(vec![array(&[7])], 0),
        ],
        vec![
            example(vec![array(&[0, 0, 1, 0, 0])], 2),
            example(vec![array(&[1, 2, 3, 4])], 3),
        ],
        "fn count_adjacent_diff(arr: [i64]) -> i64 {\n    count: i64 = 0;\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] != arr[i - 1] { count = count + 1; }\n        i = i + 1;\n    }\n    return count;\n}\n",
    )
}

fn make_priority_pop(variant: usize) -> Problem {
    problem(
        "priority_pop",
        variant,
        "array",
        "Return the maximum element (simulated pop from a heap).",
        "fn priority_pop(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[3, 1, 4, 1, 5, 9, 2, 6])], 9),
            example(vec![array(&[7])], 7),
            example(vec![array(&[1, 2, 3, 4])], 4),
            example(vec![array(&[5, 5, 5, 5])], 5),
        ],
        vec![
            example(vec![array(&[10, 20, 15])], 20),
            example(vec![array(&[-5, -2, -10])], -2),
        ],
        "fn priority_pop(arr: [i64]) -> i64 {\n    best: i64 = arr[0];\n    for x in arr { if x > best { best = x; } }\n    return best;\n}\n",
    )
}

fn make_turn_order_rotate(variant: usize) -> Problem {
    problem(
        "turn_order_rotate",
        variant,
        "game",
        "Advance current player index in a ring of num_players.",
        "fn turn_order_rotate(current: i64, num_players: i64) -> i64",
        vec![
            example(vec![int(0), int(4)], 1),
            example(vec![int(3), int(4)], 0),
            example(vec![int(1), int(3)], 2),
            example(vec![int(2), int(3)], 0),
            example(vec![int(0), int(2)], 1),
        ],
        vec![
            example(vec![int(4), int(5)], 0),
            example(vec![int(0), int(1)], 0),
        ],
        "fn turn_order_rotate(current: i64, num_players: i64) -> i64 {\n    return (current + 1) % num_players;\n}\n",
    )
}

fn make_grid_bounds_check(variant: usize) -> Problem {
    problem(
        "grid_bounds_check",
        variant,
        "game",
        "Return 1 if (x,y) is inside an w×h grid (0 ≤ x < w, 0 ≤ y < h), else 0.",
        "fn grid_bounds_check(x: i64, y: i64, w: i64, h: i64) -> i64",
        vec![
            example(vec![int(0), int(0), int(5), int(5)], 1),
            example(vec![int(4), int(4), int(5), int(5)], 1),
            example(vec![int(5), int(0), int(5), int(5)], 0),
            example(vec![int(-1), int(0), int(5), int(5)], 0),
            example(vec![int(0), int(-1), int(5), int(5)], 0),
            example(vec![int(2), int(3), int(10), int(10)], 1),
        ],
        vec![
            example(vec![int(10), int(10), int(10), int(10)], 0),
            example(vec![int(3), int(3), int(5), int(3)], 0),
        ],
        "fn grid_bounds_check(x: i64, y: i64, w: i64, h: i64) -> i64 {\n    if x < 0 { return 0; }\n    if y < 0 { return 0; }\n    if x >= w { return 0; }\n    if y >= h { return 0; }\n    return 1;\n}\n",
    )
}

fn make_simulate_gravity(variant: usize) -> Problem {
    problem(
        "simulate_gravity",
        variant,
        "game",
        "Integrate velocity: v + g*t, then clamp to terminal velocity 100.",
        "fn simulate_gravity(v: i64, g: i64, t: i64) -> i64",
        vec![
            example(vec![int(0), int(10), int(3)], 30),
            example(vec![int(5), int(2), int(4)], 13),
            example(vec![int(50), int(20), int(5)], 100),
            example(vec![int(-10), int(5), int(2)], 0),
            example(vec![int(0), int(0), int(10)], 0),
        ],
        vec![
            example(vec![int(20), int(10), int(10)], 100),
            example(vec![int(3), int(1), int(7)], 10),
        ],
        "fn simulate_gravity(v: i64, g: i64, t: i64) -> i64 {\n    r: i64 = v + g * t;\n    if r > 100 { return 100; }\n    if r < 0 { return 0; }\n    return r;\n}\n",
    )
}
