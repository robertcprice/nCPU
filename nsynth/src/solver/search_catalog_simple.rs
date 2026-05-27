use super::search_catalog_codegen::*;
use super::search_catalog_runtime::*;
use super::search_codegen::verified_result;
use super::*;

pub(super) fn search_arr_sum_squares(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| arr.iter().map(|x| x * x).sum()) {
        return None;
    }
    verified_result(
        problem,
        code_arr_sum_squares(fn_name),
        "search_arr_sum_squares",
    )
}

pub(super) fn search_min_element(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| arr.iter().copied().min().unwrap_or(0)) {
        return None;
    }
    verified_result(problem, code_min_element(fn_name), "search_min_element")
}

pub(super) fn search_sum_absolute(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| arr.iter().map(|x| x.abs()).sum()) {
        return None;
    }
    verified_result(problem, code_sum_absolute(fn_name), "search_sum_absolute")
}

pub(super) fn search_count_evens(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        arr.iter().filter(|value| *value % 2 == 0).count() as i64
    }) {
        return None;
    }
    verified_result(problem, code_count_evens(fn_name), "search_count_evens")
}

pub(super) fn search_sum_positives(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| arr.iter().filter(|x| **x > 0).sum()) {
        return None;
    }
    verified_result(problem, code_sum_positives(fn_name), "search_sum_positives")
}

pub(super) fn search_sum_at_even_indices(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| arr.iter().step_by(2).sum()) {
        return None;
    }
    verified_result(
        problem,
        code_sum_at_even_indices(fn_name),
        "search_sum_at_even_indices",
    )
}

pub(super) fn search_kth_from_end(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64, ParamType::I64] {
        return None;
    }
    if !validate_array_and_int(problem, kth_from_end_rust) {
        return None;
    }
    verified_result(problem, code_kth_from_end(fn_name), "search_kth_from_end")
}

pub(super) fn search_max_abs(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        arr.iter().map(|x| x.abs()).max().unwrap_or(0)
    }) {
        return None;
    }
    verified_result(problem, code_max_abs(fn_name), "search_max_abs")
}

pub(super) fn search_lucas_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    let max_input = problem
        .examples
        .iter()
        .filter_map(|example| int_value(&example.inputs[0]))
        .map(|value| value.abs())
        .max()
        .unwrap_or(0);
    if max_input > 92 {
        return None;
    }
    if !validate_unary_int(problem, |n| {
        if n == 0 {
            return 2;
        }
        if n == 1 {
            return 1;
        }
        let mut a = 2i64;
        let mut b = 1i64;
        for _ in 2..=n {
            let tmp = a + b;
            a = b;
            b = tmp;
        }
        b
    }) {
        return None;
    }
    verified_result(problem, code_lucas_number(fn_name), "search_lucas_loop")
}

pub(super) fn search_celsius_to_fahrenheit(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |c| c * 9 / 5 + 32) {
        return None;
    }
    verified_result(
        problem,
        code_celsius_to_fahrenheit(fn_name),
        "search_celsius_to_fahrenheit",
    )
}

pub(super) fn search_is_perfect_square(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |n| {
        let root = (n as f64).sqrt() as i64;
        if root * root == n {
            1
        } else {
            0
        }
    }) {
        return None;
    }
    verified_result(
        problem,
        code_is_perfect_square(fn_name),
        "search_is_perfect_square",
    )
}

pub(super) fn search_next_power_of_2(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |n| {
        let mut power = 1i64;
        while power < n {
            power *= 2;
        }
        power
    }) {
        return None;
    }
    verified_result(
        problem,
        code_next_power_of_2(fn_name),
        "search_next_power_of_2",
    )
}

pub(super) fn search_min_positive(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        arr.iter().filter(|&&x| x > 0).copied().min().unwrap_or(0)
    }) {
        return None;
    }
    verified_result(problem, code_min_positive(fn_name), "search_min_positive")
}

pub(super) fn search_count_peaks(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        let mut count = 0i64;
        for index in 1..arr.len().saturating_sub(1) {
            if arr[index] > arr[index - 1] && arr[index] > arr[index + 1] {
                count += 1;
            }
        }
        count
    }) {
        return None;
    }
    verified_result(problem, code_count_peaks(fn_name), "search_count_peaks")
}

pub(super) fn search_alternating_sum(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        arr.iter()
            .enumerate()
            .map(|(index, &x)| if index % 2 == 0 { x } else { -x })
            .sum()
    }) {
        return None;
    }
    verified_result(
        problem,
        code_alternating_sum(fn_name),
        "search_alternating_sum",
    )
}

pub(super) fn search_dot_product(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64, ParamType::ArrayI64] {
        return None;
    }
    if !validate_two_arrays(problem, |a, b| {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }) {
        return None;
    }
    verified_result(problem, code_dot_product(fn_name), "search_dot_product")
}

pub(super) fn search_leading_digit(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |mut n| {
        while n >= 10 {
            n /= 10;
        }
        n
    }) {
        return None;
    }
    verified_result(problem, code_leading_digit(fn_name), "search_leading_digit")
}

pub(super) fn search_popcount(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |n| n.count_ones() as i64) {
        return None;
    }
    verified_result(problem, code_popcount(fn_name), "search_popcount")
}

pub(super) fn search_is_palindrome_arr(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        if arr.iter().zip(arr.iter().rev()).all(|(a, b)| a == b) {
            1
        } else {
            0
        }
    }) {
        return None;
    }
    verified_result(
        problem,
        code_is_palindrome_arr(fn_name),
        "search_is_palindrome_arr",
    )
}

pub(super) fn search_sum_odd_indexed(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        arr.iter()
            .enumerate()
            .filter(|(index, _)| index % 2 == 1)
            .map(|(_, &x)| x)
            .sum()
    }) {
        return None;
    }
    verified_result(
        problem,
        code_sum_odd_indexed(fn_name),
        "search_sum_odd_indexed",
    )
}
