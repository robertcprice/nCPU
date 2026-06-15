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

fn code_strictly_increasing(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    i: i64 = 1;
    while i < arr.len {
        if arr[i] <= arr[i - 1] { return 0; }
        i = i + 1;
    }
    return 1;
}
"#,
        fn_name,
    )
}

pub(super) fn search_strictly_increasing(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, strictly_increasing_rust) {
        return None;
    }
    verified_result(
        problem,
        code_strictly_increasing(fn_name),
        "search_strictly_increasing",
    )
}

fn code_has_strictly_increasing_run(fn_name: &str, length: i64) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    run: i64 = 1;
    i: i64 = 1;
    while i < arr.len {
        if arr[i] > arr[i - 1] {
            run = run + 1;
            if run >= __L__ { return 1; }
        } else {
            run = 1;
        }
        i = i + 1;
    }
    return 0;
}
"#,
        fn_name,
    )
    .replace("__L__", &length.to_string())
}

pub(super) fn search_has_strictly_increasing_run(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    const CANDIDATE_LENGTHS: &[i64] = &[2, 3, 4, 5];

    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !problem
        .examples
        .iter()
        .all(|e| e.expected_int() == 0 || e.expected_int() == 1)
    {
        return None;
    }

    let _arrays = unary_array_examples(problem)?;
    for &length in CANDIDATE_LENGTHS {
        let candidate = |arr: &[i64]| has_strictly_increasing_run_rust(arr, length);
        if validate_unary_array(problem, candidate) {
            return verified_result(
                problem,
                code_has_strictly_increasing_run(fn_name, length),
                "search_has_strictly_increasing_run",
            );
        }
    }
    None
}

fn code_first_index_of(fn_name: &str, target: i64) -> String {
    let target_str = target.to_string();
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    i: i64 = 0;
    while i < arr.len {
        if arr[i] == __T__ { return i; }
        i = i + 1;
    }
    return 0 - 1;
}
"#,
        fn_name,
    )
    .replace("__T__", &target_str)
}

pub(super) fn search_first_index_of(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    const CANDIDATE_TARGETS: &[i64] = &[
        0, 1, -1, 2, 3, 5, 7, 10, -2, 100, 42, 13, 17, -5,
    ];

    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }

    let arrays = unary_array_examples(problem)?;
    for &target in CANDIDATE_TARGETS {
        let candidate = |arr: &[i64]| first_index_of_rust(arr, target);
        if validate_unary_array(problem, candidate) {
            let _ = arrays;
            return verified_result(
                problem,
                code_first_index_of(fn_name, target),
                "search_first_index_of",
            );
        }
    }
    None
}

fn code_last_index_of(fn_name: &str, target: i64) -> String {
    let target_str = target.to_string();
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    i: i64 = arr.len - 1;
    while i >= 0 {
        if arr[i] == __T__ { return i; }
        i = i - 1;
    }
    return 0 - 1;
}
"#,
        fn_name,
    )
    .replace("__T__", &target_str)
}

pub(super) fn search_last_index_of(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    const CANDIDATE_TARGETS: &[i64] = &[
        0, 1, -1, 2, 3, 5, 7, 10, -2, 100, 42, 13, 17, -5,
    ];

    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }

    let _arrays = unary_array_examples(problem)?;
    for &target in CANDIDATE_TARGETS {
        let candidate = |arr: &[i64]| last_index_of_rust(arr, target);
        if validate_unary_array(problem, candidate) {
            return verified_result(
                problem,
                code_last_index_of(fn_name, target),
                "search_last_index_of",
            );
        }
    }
    None
}

// Note: `search_count_distinct` already exists in this file (line ~501).
// It uses sort + adjacent-unique counting, which is the canonical
// implementation. We don't add a table-lookup variant with a different
// name — same problem shape, same solution, just two codegen bodies.

fn code_is_anagram(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(a: [i64], b: [i64]) -> i64 {
    if a.len != b.len { return 0; }
    sa: [i64] = a;
    sb: [i64] = b;
    sa.sort();
    sb.sort();
    i: i64 = 0;
    while i < a.len {
        if sa[i] != sb[i] { return 0; }
        i = i + 1;
    }
    return 1;
}
"#,
        fn_name,
    )
}

pub(super) fn search_is_anagram(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64, ParamType::ArrayI64] {
        return None;
    }
    if !validate_two_arrays(problem, is_anagram_rust) {
        return None;
    }
    verified_result(problem, code_is_anagram(fn_name), "search_is_anagram")
}

fn code_longest_run(fn_name: &str, target: i64) -> String {
    let target_str = target.to_string();
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    best: i64 = 0;
    cur: i64 = 0;
    for v in arr {
        if v == __T__ {
            cur = cur + 1;
            if cur > best { best = cur; }
        } else {
            cur = 0;
        }
    }
    return best;
}
"#,
        fn_name,
    )
    .replace("__T__", &target_str)
}

pub(super) fn search_longest_run(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    const CANDIDATE_TARGETS: &[i64] = &[
        0, 1, -1, 2, 3, 5, 7, 10, -2, 100, 42, 13, 17, -5,
    ];

    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    let _arrays = unary_array_examples(problem)?;
    for &target in CANDIDATE_TARGETS {
        let candidate = |arr: &[i64]| longest_run_rust(arr, target);
        if validate_unary_array(problem, candidate) {
            return verified_result(
                problem,
                code_longest_run(fn_name, target),
                "search_longest_run",
            );
        }
    }
    None
}

fn code_intersects(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(a: [i64], b: [i64]) -> i64 {
    for x in a {
        for y in b {
            if x == y { return 1; }
        }
    }
    return 0;
}
"#,
        fn_name,
    )
}

pub(super) fn search_intersects(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64, ParamType::ArrayI64] {
        return None;
    }
    if !validate_two_arrays(problem, intersects_rust) {
        return None;
    }
    verified_result(problem, code_intersects(fn_name), "search_intersects")
}

// Note: `search_kth_smallest` (the arr + k binary version) already
// exists in this file at line ~91. We don't add a second copy with a
// fixed-k signature; the binary version is more general and covers the
// same use cases via Mog's overloaded signatures.

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
    if arr.len == 0 { return 0; }
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
