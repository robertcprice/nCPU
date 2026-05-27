use super::*;

#[test]
fn search_only_generalizes_on_array_holdout_cases() {
    assert_search_generalizes(
        "array_max_v0",
        vec![
            (vec![Value::Array(vec![-3, -9, -1])], -1),
            (vec![Value::Array(vec![10, 2, 10])], 10),
        ],
    );
    assert_search_generalizes(
        "closure_map_sum_v0",
        vec![
            (vec![Value::Array(vec![0, -1, 4])], 6),
            (vec![Value::Array(vec![5])], 10),
        ],
    );
    assert_search_generalizes(
        "reverse_sum_v0",
        vec![
            (vec![Value::Array(vec![9, -2, 4])], 11),
            (vec![Value::Array(vec![0, 0, 1])], 1),
        ],
    );
    assert_search_generalizes(
        "array_max_elem_v0",
        vec![
            (vec![Value::Array(vec![-1, -5, -3])], -1),
            (vec![Value::Array(vec![10, 2, 10])], 10),
        ],
    );
    assert_search_generalizes(
        "max_pair_diff_v0",
        vec![
            (vec![Value::Array(vec![1, 10, 3, 20])], 17),
            (vec![Value::Array(vec![5, 5, 5])], 0),
        ],
    );
    assert_search_generalizes(
        "sum_negatives_v0",
        vec![
            (vec![Value::Array(vec![-5, 2, -1, 0])], -6),
            (vec![Value::Array(vec![1, 2, 3])], 0),
        ],
    );
    assert_search_generalizes(
        "interactive_sum_v0",
        vec![
            (vec![Value::Array(vec![10, -5, 3])], 8),
            (vec![Value::Array(vec![7])], 7),
        ],
    );
}

#[test]
fn search_only_generalizes_on_aliased_struct_holdouts() {
    let point_problem = aliased_problem(
        "point_sum",
        "mystery_point_holdout_v0",
        "fn mystery_point_holdout(p: Point) -> i64",
        "struct_search",
        "Return the sum of the point coordinates.",
    );
    assert_search_generalizes_problem(
        point_problem,
        vec![
            (vec![Value::Pair(12, -5)], 7),
            (vec![Value::Pair(-3, -4)], -7),
        ],
    );

    let rectangle_problem = aliased_problem(
        "rectangle_area",
        "mystery_rect_holdout_v0",
        "fn mystery_rect_holdout(r: Rectangle) -> i64",
        "struct_search",
        "Return the rectangle area.",
    );
    assert_search_generalizes_problem(
        rectangle_problem,
        vec![
            (vec![Value::Pair(6, 7)], 42),
            (vec![Value::Pair(11, 3)], 33),
        ],
    );
}
