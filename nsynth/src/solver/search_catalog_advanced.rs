use super::search_catalog_runtime::*;
use super::search_codegen::verified_result;
use super::*;

fn code_count_zeros(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    count: i64 = 0;
    for item in arr {
        if item == 0 {
            count = count + 1;
        }
    }
    return count;
}
"#,
        fn_name,
    )
}

fn code_max_consecutive_sum(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    current: i64 = 0;
    best: i64 = arr[0];
    for item in arr {
        if current > 0 {
            current = current + item;
        } else {
            current = item;
        }
        if current > best {
            best = current;
        }
    }
    return best;
}
"#,
        fn_name,
    )
}

fn code_min_consecutive_sum(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    current: i64 = 0;
    best: i64 = arr[0];
    for item in arr {
        if current < 0 {
            current = current + item;
        } else {
            current = item;
        }
        if current < best {
            best = current;
        }
    }
    return best;
}
"#,
        fn_name,
    )
}

pub(super) fn search_min_consecutive_sum(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, min_consecutive_sum) {
        return None;
    }
    verified_result(
        problem,
        code_min_consecutive_sum(fn_name),
        "search_min_consecutive_sum",
    )
}

fn code_kth_smallest(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64], k: i64) -> i64 {
    arr.sort();
    return arr[k - 1];
}
"#,
        fn_name,
    )
}

pub(super) fn search_kth_smallest(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64, ParamType::I64] {
        return None;
    }
    if !validate_array_and_int(problem, kth_smallest_rust) {
        return None;
    }
    verified_result(problem, code_kth_smallest(fn_name), "search_kth_smallest")
}

fn code_max_stock_profit(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(prices: [i64]) -> i64 {
    min_price: i64 = prices[0];
    best: i64 = 0;
    for p in prices {
        if p < min_price { min_price = p; }
        profit: i64 = p - min_price;
        if profit > best { best = profit; }
    }
    return best;
}
"#,
        fn_name,
    )
}

pub(super) fn search_max_stock_profit(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, max_stock_profit_rust) {
        return None;
    }
    verified_result(
        problem,
        code_max_stock_profit(fn_name),
        "search_max_stock_profit",
    )
}

fn code_is_sorted(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    i: i64 = 1;
    while i < arr.len {
        if arr[i] < arr[i - 1] { return 0; }
        i = i + 1;
    }
    return 1;
}
"#,
        fn_name,
    )
}

pub(super) fn search_is_sorted(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, is_sorted_rust) {
        return None;
    }
    verified_result(problem, code_is_sorted(fn_name), "search_is_sorted")
}

fn code_longest_increasing_run(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    best: i64 = 1;
    cur: i64 = 1;
    i: i64 = 1;
    while i < arr.len {
        if arr[i] > arr[i - 1] {
            cur = cur + 1;
            if cur > best { best = cur; }
        } else {
            cur = 1;
        }
        i = i + 1;
    }
    return best;
}
"#,
        fn_name,
    )
}

pub(super) fn search_longest_increasing_run(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, longest_increasing_run_rust) {
        return None;
    }
    verified_result(
        problem,
        code_longest_increasing_run(fn_name),
        "search_longest_increasing_run",
    )
}

fn code_digital_root(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    x: i64 = n;
    while x >= 10 {
        s: i64 = 0;
        while x > 0 {
            s = s + x % 10;
            x = x / 10;
        }
        x = s;
    }
    return x;
}
"#,
        fn_name,
    )
}

pub(super) fn search_digital_root(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, digital_root_rust) {
        return None;
    }
    verified_result(problem, code_digital_root(fn_name), "search_digital_root")
}

fn code_two_sum_exists(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64], target: i64) -> i64 {
    i: i64 = 0;
    while i < arr.len {
        j: i64 = i + 1;
        while j < arr.len {
            if arr[i] + arr[j] == target { return 1; }
            j = j + 1;
        }
        i = i + 1;
    }
    return 0;
}
"#,
        fn_name,
    )
}

pub(super) fn search_two_sum_exists(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64, ParamType::I64] {
        return None;
    }
    if !validate_array_and_int(problem, two_sum_exists_rust) {
        return None;
    }
    verified_result(
        problem,
        code_two_sum_exists(fn_name),
        "search_two_sum_exists",
    )
}

fn code_count_distinct(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    arr.sort();
    count: i64 = 1;
    i: i64 = 1;
    while i < arr.len {
        if arr[i] != arr[i - 1] {
            count = count + 1;
        }
        i = i + 1;
    }
    return count;
}
"#,
        fn_name,
    )
}

pub(super) fn search_count_distinct(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, count_distinct_rust) {
        return None;
    }
    verified_result(
        problem,
        code_count_distinct(fn_name),
        "search_count_distinct",
    )
}

fn code_binary_search(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64], target: i64) -> i64 {
    lo: i64 = 0;
    hi: i64 = arr.len - 1;
    while lo <= hi {
        mid: i64 = (lo + hi) / 2;
        if arr[mid] == target { return mid; }
        if arr[mid] < target { lo = mid + 1; }
        if arr[mid] > target { hi = mid - 1; }
    }
    return -1;
}
"#,
        fn_name,
    )
}

pub(super) fn search_binary_search(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64, ParamType::I64] {
        return None;
    }
    if !validate_array_and_int(problem, binary_search_rust) {
        return None;
    }
    verified_result(problem, code_binary_search(fn_name), "search_binary_search")
}

fn code_longest_plateau(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    best: i64 = 1;
    cur: i64 = 1;
    i: i64 = 1;
    while i < arr.len {
        if arr[i] == arr[i - 1] {
            cur = cur + 1;
            if cur > best { best = cur; }
        } else {
            cur = 1;
        }
        i = i + 1;
    }
    return best;
}
"#,
        fn_name,
    )
}

pub(super) fn search_longest_plateau(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, longest_plateau_rust) {
        return None;
    }
    verified_result(
        problem,
        code_longest_plateau(fn_name),
        "search_longest_plateau",
    )
}

fn code_prefix_max_sum(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    running_max: i64 = arr[0];
    total: i64 = 0;
    for x in arr {
        if x > running_max { running_max = x; }
        total = total + running_max;
    }
    return total;
}
"#,
        fn_name,
    )
}

pub(super) fn search_prefix_max_sum(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, prefix_max_sum_rust) {
        return None;
    }
    verified_result(
        problem,
        code_prefix_max_sum(fn_name),
        "search_prefix_max_sum",
    )
}

pub(super) fn search_count_zeros(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        arr.iter().filter(|value| **value == 0).count() as i64
    }) {
        return None;
    }
    verified_result(problem, code_count_zeros(fn_name), "search_count_zeros")
}

pub(super) fn search_max_consecutive_sum(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, max_consecutive_sum) {
        return None;
    }
    verified_result(
        problem,
        code_max_consecutive_sum(fn_name),
        "search_max_consecutive_sum",
    )
}
