use super::*;

#[test]
fn native_scalar_teacher_distillation_solves_add_two() {
    let problem = Problem {
        name: "add_two_teacher_v0".to_string(),
        category: "test",
        description: "Return the sum of a and b.",
        signature: "fn add_two_teacher(a: i64, b: i64) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Int(1), Value::Int(2)],
                expected: Value::Int(3),
            },
            Example {
                inputs: vec![Value::Int(-5), Value::Int(8)],
                expected: Value::Int(3),
            },
            Example {
                inputs: vec![Value::Int(0), Value::Int(0)],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Int(10), Value::Int(-4)],
                expected: Value::Int(6),
            },
        ],
        holdouts: vec![
            Example {
                inputs: vec![Value::Int(7), Value::Int(9)],
                expected: Value::Int(16),
            },
            Example {
                inputs: vec![Value::Int(-3), Value::Int(-4)],
                expected: Value::Int(-7),
            },
        ],
        reference_code: "",
    };
    let teacher_code = "fn add_two_teacher(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n";
    let result = crate::synthesis::synthesize_scalar_from_teacher(&problem, teacher_code)
        .expect("teacher-guided scalar synthesis should produce a native program");
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "synth_gradient");
}

#[test]
fn native_scalar_teacher_distillation_solves_digit_sum_loop() {
    let problem = Problem {
        name: "digit_sum_teacher_v0".to_string(),
        category: "test",
        description: "Return the sum of the digits of n.",
        signature: "fn digit_sum_teacher(n: i64) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Int(0)],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Int(405)],
                expected: Value::Int(9),
            },
            Example {
                inputs: vec![Value::Int(7001)],
                expected: Value::Int(8),
            },
            Example {
                inputs: vec![Value::Int(999)],
                expected: Value::Int(27),
            },
        ],
        holdouts: vec![
            Example {
                inputs: vec![Value::Int(12345)],
                expected: Value::Int(15),
            },
            Example {
                inputs: vec![Value::Int(90)],
                expected: Value::Int(9),
            },
        ],
        reference_code: "",
    };
    let teacher_code = "fn digit_sum_teacher(n: i64) -> i64 {\n    x: i64 = n;\n    acc: i64 = 0;\n    while x > 0 {\n        acc = acc + x % 10;\n        x = x / 10;\n    }\n    return acc;\n}\n";
    let result = crate::synthesis::synthesize_scalar_from_teacher(&problem, teacher_code)
        .expect("loop teacher should distill into native digit-loop family");
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "synth_gradient");
    assert!(result.code.contains("x % 10"), "{}", result.code);
}

#[test]
fn native_scalar_teacher_distillation_solves_is_prime_loop() {
    let problem = Problem {
        name: "is_prime_teacher_v0".to_string(),
        category: "test",
        description: "Return 1 if n is prime, else 0.",
        signature: "fn is_prime_teacher(n: i64) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Int(1)],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Int(2)],
                expected: Value::Int(1),
            },
            Example {
                inputs: vec![Value::Int(4)],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Int(17)],
                expected: Value::Int(1),
            },
        ],
        holdouts: vec![
            Example {
                inputs: vec![Value::Int(21)],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Int(29)],
                expected: Value::Int(1),
            },
        ],
        reference_code: "",
    };
    let teacher_code = "fn is_prime_teacher(n: i64) -> i64 {\n    count: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        if n % i == 0 { count = count + 1; }\n        i = i + 1;\n    }\n    if count == 2 { return 1; }\n    return 0;\n}\n";
    let result = crate::synthesis::synthesize_scalar_from_teacher(&problem, teacher_code)
        .expect("loop teacher should distill into native prime family");
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "synth_gradient");
    assert!(result.code.contains("acc == 2"), "{}", result.code);
}

#[test]
fn native_scalar_teacher_distillation_solves_gcd_loop() {
    let problem = Problem {
        name: "gcd_teacher_v0".to_string(),
        category: "test",
        description: "Return the greatest common divisor of a and b.",
        signature: "fn gcd_teacher(a: i64, b: i64) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Int(18), Value::Int(24)],
                expected: Value::Int(6),
            },
            Example {
                inputs: vec![Value::Int(7), Value::Int(5)],
                expected: Value::Int(1),
            },
            Example {
                inputs: vec![Value::Int(42), Value::Int(56)],
                expected: Value::Int(14),
            },
            Example {
                inputs: vec![Value::Int(81), Value::Int(27)],
                expected: Value::Int(27),
            },
        ],
        holdouts: vec![
            Example {
                inputs: vec![Value::Int(270), Value::Int(192)],
                expected: Value::Int(6),
            },
            Example {
                inputs: vec![Value::Int(54), Value::Int(24)],
                expected: Value::Int(6),
            },
        ],
        reference_code: "",
    };
    let teacher_code = "fn gcd_teacher(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }\n    return x;\n}\n";
    let result = crate::synthesis::synthesize_scalar_from_teacher(&problem, teacher_code)
        .expect("loop teacher should distill into native predicate-loop family");
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "synth_gradient");
    assert!(result.code.contains("x0 % x1"), "{}", result.code);
}

#[test]
fn native_scalar_reference_distillation_solves_is_prime_benchmark() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "is_prime_v0")
        .expect("is_prime_v0 not found");
    let result = crate::synthesis::synthesize_scalar_from_teacher(&problem, problem.reference_code)
        .expect("benchmark reference loop should distill into native scalar family");
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "synth_gradient");
    assert!(result.code.contains("acc == 2"), "{}", result.code);
}

#[test]
fn native_scalar_reference_distillation_loop_family_benchmarks() {
    let problems = get_benchmark(1);
    let targets = [
        "digit_sum_v0",
        "digit_product_v0",
        "reverse_digits_v0",
        "digit_count_v0",
        "count_even_digits_v0",
        "sum_odd_digits_v0",
        "popcount_v0",
        "max_digit_v0",
        "power_v0",
        "harmonic_sum_v0",
        "count_divisors_v0",
        "sum_of_divisors_v0",
        "is_perfect_square_v0",
        "is_prime_v0",
        "gcd_v0",
        "leading_digit_v0",
        "next_power_of_2_v0",
        "triangular_check_v0",
        "collatz_steps_v0",
    ];

    let mut failed = Vec::new();
    for name in targets {
        println!("teacher distill check: {name}");
        let problem = problems
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} not found"));
        let result =
            crate::synthesis::synthesize_scalar_from_teacher(problem, problem.reference_code);
        match result {
            Some(result) if result.success && result.method == "synth_gradient" => {
                println!("teacher distill ok: {name}");
            }
            Some(result) => failed.push(format!("{name}: {}", result.method)),
            None => failed.push(format!("{name}: no native teacher result")),
        }
    }

    assert!(
        failed.is_empty(),
        "native scalar reference distillation failed for: {}",
        failed.join(", ")
    );
}
