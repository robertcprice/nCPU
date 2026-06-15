use super::*;

#[test]
fn array_teacher_distillation_solves_count_positive() {
    let problem = Problem {
        name: "count_positive_teacher_v0".to_string(),
        category: "test",
        description: "Return the number of positive entries in the array.",
        signature: "fn count_positive_teacher(arr: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Array(vec![1, 2, 3, 4])],
                expected: Value::Int(4),
            },
            Example {
                inputs: vec![Value::Array(vec![4, -3, 2, -1])],
                expected: Value::Int(2),
            },
            Example {
                inputs: vec![Value::Array(vec![-5])],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Array(vec![0, 0, 0])],
                expected: Value::Int(0),
            },
        ],
        holdouts: vec![
            Example {
                inputs: vec![Value::Array(vec![3, 0, -2, 1])],
                expected: Value::Int(2),
            },
            Example {
                inputs: vec![Value::Array(vec![-1, -2, -3])],
                expected: Value::Int(0),
            },
        ],
        reference_code: "",

    synthetic_args: Vec::new(),

    synthetic_values: Vec::new(),

    recursive_allowed: false,

    tree_input: false,

    explicit_stack: false,

    };
    let teacher_code = "fn count_positive_teacher(arr: [i64]) -> i64 {\n    count: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        if arr[i] > 0 { count = count + 1; }\n        i = i + 1;\n    }\n    return count;\n}\n";
    let result = crate::synthesis::synthesize_array_from_teacher(&problem, teacher_code)
        .expect("teacher-guided array synthesis should produce a native program");
    assert!(result.success, "{:?}", result.error);
    assert!(
        result.method == "arr_gradient" || result.method == "univ_arr_gradient",
        "expected teacher distillation to land on native array gradient, got {}",
        result.method
    );
}
