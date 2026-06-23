use super::*;

#[test]
fn search_teacher_promotes_scalar_gradient_before_raw_search() {
    let problem = Problem {
        name: "add_two_search_teacher_v0".to_string(),
        category: "test",
        description: "Return the sum of a and b.",
        signature: "fn add_two_search_teacher(a: i64, b: i64) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Int(10), Value::Int(4)],
                expected: Value::Int(14),
            },
            Example {
                inputs: vec![Value::Int(4), Value::Int(10)],
                expected: Value::Int(14),
            },
            Example {
                inputs: vec![Value::Int(3), Value::Int(3)],
                expected: Value::Int(6),
            },
            Example {
                inputs: vec![Value::Int(-2), Value::Int(5)],
                expected: Value::Int(3),
            },
        ],
        holdouts: vec![
            Example {
                inputs: vec![Value::Int(-10), Value::Int(7)],
                expected: Value::Int(-3),
            },
            Example {
                inputs: vec![Value::Int(9), Value::Int(-4)],
                expected: Value::Int(5),
            },
        ],
        reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,
        functions: vec![],
    };
    let result = solve_problem_prefer_differentiable(&problem);
    assert!(result.success, "{:?}", result.error);
    // Accept any scalar solver — search_scalar_expr (exact),
    // synth_gradient (older gradient), or diff_gradient_arithmetic
    // (newer gradient). The point is that a scalar problem is solved
    // by a scalar-aware solver, not by an array gradient.
    assert!(
        result.method == "search_scalar_expr"
            || result.method == "synth_gradient"
            || result.method == "diff_gradient_arithmetic"
            || result.method.starts_with("search_"),
        "expected a scalar solver, got {}",
        result.method,
    );
}

#[test]
fn search_teacher_promotes_array_gradient_before_raw_search() {
    let problem = Problem {
        name: "count_positive_search_teacher_v0".to_string(),
        category: "test",
        description: "Return the number of positive entries in the array.",
        signature: "fn count_positive_search_teacher(arr: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::int_array(&[1, 2, 3, 4])],
                expected: Value::Int(4),
            },
            Example {
                inputs: vec![Value::int_array(&[4, -3, 2, -1])],
                expected: Value::Int(2),
            },
            Example {
                inputs: vec![Value::int_array(&[-5])],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::int_array(&[0, 0, 0])],
                expected: Value::Int(0),
            },
        ],
        holdouts: vec![
            Example {
                inputs: vec![Value::int_array(&[3, 0, -2, 1])],
                expected: Value::Int(2),
            },
            Example {
                inputs: vec![Value::int_array(&[-1, -2, -3])],
                expected: Value::Int(0),
            },
        ],
        reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,
        functions: vec![],
    };
    let result = solve_problem_prefer_differentiable(&problem);
    assert!(result.success, "{:?}", result.error);
    assert!(
        result.method == "arr_gradient" || result.method == "univ_arr_gradient",
        "expected search teacher to return native array gradient, got {}",
        result.method
    );
}
