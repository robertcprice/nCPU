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

fn min_consecutive_sum(arr: &[i64]) -> i64 {
    if arr.is_empty() {
        return 0;
    }
    let mut current = 0i64;
    let mut best = arr[0];
    for &item in arr {
        current = if current < 0 { current + item } else { item };
        best = best.min(current);
    }
    best
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

fn kth_smallest_rust(arr: &[i64], k: i64) -> i64 {
    if k < 1 || k as usize > arr.len() {
        return i64::MIN;
    }
    let mut values = arr.to_vec();
    values.sort();
    values[(k - 1) as usize]
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

fn max_stock_profit_rust(prices: &[i64]) -> i64 {
    let mut min_price = prices[0];
    let mut best = 0i64;
    for &price in prices {
        if price < min_price {
            min_price = price;
        }
        let profit = price - min_price;
        if profit > best {
            best = profit;
        }
    }
    best
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

fn is_sorted_rust(arr: &[i64]) -> i64 {
    if arr.windows(2).all(|window| window[0] <= window[1]) {
        1
    } else {
        0
    }
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

fn longest_increasing_run_rust(arr: &[i64]) -> i64 {
    let mut best = 1i64;
    let mut current = 1i64;
    for index in 1..arr.len() {
        if arr[index] > arr[index - 1] {
            current += 1;
            if current > best {
                best = current;
            }
        } else {
            current = 1;
        }
    }
    best
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

fn digital_root_rust(mut n: i64) -> i64 {
    while n >= 10 {
        let mut sum = 0i64;
        while n > 0 {
            sum += n % 10;
            n /= 10;
        }
        n = sum;
    }
    n
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

fn two_sum_exists_rust(arr: &[i64], target: i64) -> i64 {
    for i in 0..arr.len() {
        for j in (i + 1)..arr.len() {
            if arr[i] + arr[j] == target {
                return 1;
            }
        }
    }
    0
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

fn count_distinct_rust(arr: &[i64]) -> i64 {
    let mut values = arr.to_vec();
    values.sort();
    values.dedup();
    values.len() as i64
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

fn binary_search_rust(arr: &[i64], target: i64) -> i64 {
    let mut lo = 0i64;
    let mut hi = arr.len() as i64 - 1;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        if arr[mid as usize] == target {
            return mid;
        }
        if arr[mid as usize] < target {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    -1
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

fn longest_plateau_rust(arr: &[i64]) -> i64 {
    let mut best = 1i64;
    let mut current = 1i64;
    for index in 1..arr.len() {
        if arr[index] == arr[index - 1] {
            current += 1;
            if current > best {
                best = current;
            }
        } else {
            current = 1;
        }
    }
    best
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

fn prefix_max_sum_rust(arr: &[i64]) -> i64 {
    let mut running_max = arr[0];
    let mut total = 0i64;
    for &value in arr {
        if value > running_max {
            running_max = value;
        }
        total += running_max;
    }
    total
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

fn code_arr_sum_squares(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    acc: i64 = 0;
    for x in arr {
        acc = acc + x * x;
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_min_element(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    best: i64 = arr[0];
    for x in arr {
        if x < best {
            best = x;
        }
    }
    return best;
}
"#,
        fn_name,
    )
}

fn code_sum_absolute(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    acc: i64 = 0;
    for x in arr {
        if x < 0 {
            acc = acc + (0 - x);
        } else {
            acc = acc + x;
        }
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_count_evens(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    acc: i64 = 0;
    for x in arr {
        if (x % 2) == 0 {
            acc = acc + 1;
        }
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_sum_positives(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    acc: i64 = 0;
    for x in arr {
        if x > 0 {
            acc = acc + x;
        }
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_sum_at_even_indices(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    acc: i64 = 0;
    i: i64 = 0;
    while i < arr.len {
        acc = acc + arr[i];
        i = i + 2;
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_kth_from_end(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64], k: i64) -> i64 {
    return arr[arr.len - k];
}
"#,
        fn_name,
    )
}

fn code_max_abs(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    best: i64 = 0;
    for x in arr {
        v: i64 = x;
        if v < 0 {
            v = 0 - v;
        }
        if v > best {
            best = v;
        }
    }
    return best;
}
"#,
        fn_name,
    )
}

fn code_alternating_sum(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    acc: i64 = 0;
    i: i64 = 0;
    sign: i64 = 1;
    while i < arr.len {
        acc = acc + sign * arr[i];
        sign = 0 - sign;
        i = i + 1;
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_dot_product(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(a: [i64], b: [i64]) -> i64 {
    acc: i64 = 0;
    i: i64 = 0;
    while i < a.len {
        acc = acc + a[i] * b[i];
        i = i + 1;
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_leading_digit(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    x: i64 = n;
    while x >= 10 {
        x = x / 10;
    }
    return x;
}
"#,
        fn_name,
    )
}

fn code_popcount(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    x: i64 = n;
    acc: i64 = 0;
    while x > 0 {
        acc = acc + x % 2;
        x = x / 2;
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_is_palindrome_arr(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    i: i64 = 0;
    j: i64 = arr.len - 1;
    while i < j {
        if arr[i] != arr[j] {
            return 0;
        }
        i = i + 1;
        j = j - 1;
    }
    return 1;
}
"#,
        fn_name,
    )
}

fn code_sum_odd_indexed(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    acc: i64 = 0;
    i: i64 = 1;
    while i < arr.len {
        acc = acc + arr[i];
        i = i + 2;
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_min_positive(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    best: i64 = 0;
    found: i64 = 0;
    for x in arr {
        if x > 0 {
            if found == 0 {
                best = x;
                found = 1;
            } else {
                if x < best {
                    best = x;
                }
            }
        }
    }
    return best;
}
"#,
        fn_name,
    )
}

fn code_lucas_number(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    if n == 0 {
        return 2;
    }
    if n == 1 {
        return 1;
    }
    a: i64 = 2;
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
"#,
        fn_name,
    )
}

fn code_celsius_to_fahrenheit(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(c: i64) -> i64 {
    return c * 9 / 5 + 32;
}
"#,
        fn_name,
    )
}

fn code_is_perfect_square(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    i: i64 = 0;
    while i * i <= n {
        if i * i == n {
            return 1;
        }
        i = i + 1;
    }
    return 0;
}
"#,
        fn_name,
    )
}

fn code_next_power_of_2(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    p: i64 = 1;
    while p < n {
        p = p * 2;
    }
    return p;
}
"#,
        fn_name,
    )
}

fn code_count_peaks(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    count: i64 = 0;
    i: i64 = 1;
    while i < arr.len - 1 {
        if arr[i] > arr[i - 1] {
            if arr[i] > arr[i + 1] {
                count = count + 1;
            }
        }
        i = i + 1;
    }
    return count;
}
"#,
        fn_name,
    )
}

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

fn kth_from_end_rust(arr: &[i64], k: i64) -> i64 {
    if k < 1 || k as usize > arr.len() {
        return i64::MIN;
    }
    arr[arr.len() - k as usize]
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

fn max_consecutive_sum(arr: &[i64]) -> i64 {
    if arr.is_empty() {
        return 0;
    }
    let mut current = 0i64;
    let mut best = arr[0];
    for &item in arr {
        current = if current > 0 { current + item } else { item };
        best = best.max(current);
    }
    best
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
