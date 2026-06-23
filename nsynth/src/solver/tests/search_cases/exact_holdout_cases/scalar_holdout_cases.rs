use super::*;

#[test]
fn search_abs_diff_generalizes_beyond_examples() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|problem| problem.name == "abs_diff_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success);

    let program = format!(
        "{}\nfn main() -> i64 {{\n    println_i64(abs_diff(-10, 7));\n    println_i64(abs_diff(9, -4));\n    return 0;\n}}\n",
        result.code.trim_end()
    );
    let run = crate::runtime::execute_program(&program).unwrap();
    assert_eq!(run.output, "17\n13");
}

#[test]
fn search_only_generalizes_on_holdout_cases() {
    assert_search_generalizes(
        "add_two_v0",
        vec![
            (vec![Value::Int(100), Value::Int(-37)], 63),
            (vec![Value::Int(-12), Value::Int(-8)], -20),
        ],
    );
    assert_search_generalizes(
        "max2_v0",
        vec![
            (vec![Value::Int(-3), Value::Int(9)], 9),
            (vec![Value::Int(12), Value::Int(12)], 12),
        ],
    );
    assert_search_generalizes(
        "clamp_0_100_v0",
        vec![
            (vec![Value::Int(-5)], 0),
            (vec![Value::Int(101)], 100),
            (vec![Value::Int(42)], 42),
        ],
    );
    assert_search_generalizes(
        "sign_v0",
        vec![
            (vec![Value::Int(-8)], -1),
            (vec![Value::Int(0)], 0),
            (vec![Value::Int(15)], 1),
        ],
    );
    assert_search_generalizes(
        "safe_div_or_neg1_v0",
        vec![
            (vec![Value::Int(9), Value::Int(0)], -1),
            (vec![Value::Int(21), Value::Int(7)], 3),
        ],
    );
    assert_search_generalizes(
        "positive_or_default_v0",
        vec![(vec![Value::Int(-4)], 0), (vec![Value::Int(19)], 19)],
    );
    assert_search_generalizes(
        "is_even_v0",
        vec![(vec![Value::Int(-6)], 1), (vec![Value::Int(105)], 0)],
    );
    assert_search_generalizes(
        "array_sum_v0",
        vec![
            (vec![Value::int_array(&[10, -5, 2])], 7),
            (vec![Value::int_array(&[1, 2, 3, 4])], 10),
        ],
    );
    assert_search_generalizes(
        "count_positive_v0",
        vec![
            (vec![Value::int_array(&[0, 1, -1, 3])], 2),
            (vec![Value::int_array(&[-5, -2, 0])], 0),
        ],
    );
    assert_search_generalizes(
        "count_occurrences_v0",
        vec![
            (vec![Value::int_array(&[4, 1, 4, 4]), Value::Int(4)], 3),
            (vec![Value::int_array(&[2, 3]), Value::Int(5)], 0),
        ],
    );
    assert_search_generalizes(
        "gcd_extended_v0",
        vec![
            (vec![Value::Int(270), Value::Int(192)], 6),
            (vec![Value::Int(17), Value::Int(13)], 1),
        ],
    );
    assert_search_generalizes(
        "point_sum_v0",
        vec![
            (vec![Value::Pair(5, -7)], -2),
            (vec![Value::Pair(8, 9)], 17),
        ],
    );
    assert_search_generalizes(
        "rectangle_area_v0",
        vec![
            (vec![Value::Pair(9, 11)], 99),
            (vec![Value::Pair(3, 7)], 21),
        ],
    );
}
