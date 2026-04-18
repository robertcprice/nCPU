use crate::benchmark::{Problem, Value};
use crate::differentiable::{
    solve_problem_differentiable_only as solve_problem_differentiable_bridge,
    DifferentiableMetadata,
};
use crate::method_router;
use crate::runtime::verify_problem_code_strict;

mod benchmarking;
mod helpers;
mod legacy_fallback;
mod pipeline;
mod post_enumerative;
mod routing;
mod scalar_search;
mod search;
mod search_catalog;
mod signature;

use self::helpers::{
    family_name, int_value, templ, validate_array_and_int, validate_binary_int,
    validate_quaternary_int, validate_ternary_int, validate_two_arrays, validate_unary_array,
    validate_unary_int, validate_unary_pair, validate_unary_str,
};
use self::signature::{
    parse_param_types, scalar_param_names, scalar_params_decl, unary_pair_examples,
    unary_string_examples, ParamType,
};

#[cfg(test)]
use self::benchmarking::find_python_warmstart_model;
#[cfg(test)]
use self::post_enumerative::search_result_preempts_native_gradient;
use self::post_enumerative::{
    solve_problem_after_enumeration, solve_problem_from_preemptive_search_teacher,
};
#[cfg(test)]
use self::routing::planned_post_enumerative_routes;
use self::routing::{
    normalized_router_stats, post_enumerative_context, should_bypass_solved_cache,
    should_try_enumerative, ROUTE_ENUMERATIVE,
};
#[cfg(test)]
use self::routing::{ROUTE_ARRAY_GRADIENT, ROUTE_SCALAR_GRADIENT, ROUTE_SEARCH_TEACHER};
use self::search::solve_by_search;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolveResult {
    pub success: bool,
    pub code: String,
    pub method: String,
    pub error: Option<String>,
    pub metadata: DifferentiableMetadata,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkSummary {
    pub total: usize,
    pub solved: usize,
    pub failures: Vec<String>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PostEnumerativeStage {
    ScalarGradientOnly,
    ArrayGradient,
    ExprOnly,
    SearchTeacher,
    ExprTemplates,
    ScalarTemplates,
    RegisterMachine,
    BridgeGradient,
    ReferenceDistillation,
    NativeScalarTeacherDistillation,
    ArrayTeacherDistillation,
    TemplateReference,
    Search,
}

#[cfg(test)]
fn post_enumerative_stage_order(problem: &Problem) -> Vec<PostEnumerativeStage> {
    let fn_name = problem.function_name();
    let search_teacher_preempts = solve_by_search(problem, fn_name)
        .map(|result| search_result_preempts_native_gradient(&result))
        .unwrap_or(false);
    let n_args = problem
        .examples
        .first()
        .map(|e| e.inputs.len())
        .unwrap_or(0);
    let is_external = problem.category == "external";
    let has_array_input = problem
        .examples
        .first()
        .map(|ex| ex.inputs.iter().any(|v| matches!(v, Value::Array(_))))
        .unwrap_or(false);
    let scalar_only_inputs = problem
        .examples
        .iter()
        .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_))));
    let mut stages = Vec::new();
    if search_teacher_preempts {
        stages.push(PostEnumerativeStage::SearchTeacher);
    }
    if scalar_only_inputs && (!is_external || n_args <= 3) {
        stages.push(PostEnumerativeStage::ScalarGradientOnly);
    } else if !has_array_input {
        // scalar gradient stage is skipped for non-scalar, non-array problems
    }
    stages.push(PostEnumerativeStage::ArrayGradient);
    stages.push(PostEnumerativeStage::ExprOnly);
    if !search_teacher_preempts {
        stages.push(PostEnumerativeStage::SearchTeacher);
    }
    stages.push(PostEnumerativeStage::RegisterMachine);
    if scalar_only_inputs && (!is_external || n_args <= 3) {
        stages.push(PostEnumerativeStage::BridgeGradient);
    }
    if scalar_only_inputs && (!is_external || n_args <= 3) && !problem.reference_code.is_empty() {
        stages.push(PostEnumerativeStage::ReferenceDistillation);
    }
    if scalar_only_inputs && !problem.reference_code.is_empty() {
        stages.push(PostEnumerativeStage::NativeScalarTeacherDistillation);
    }
    if has_array_input && !problem.reference_code.is_empty() {
        stages.push(PostEnumerativeStage::ArrayTeacherDistillation);
    }
    stages.push(PostEnumerativeStage::ExprTemplates);
    if scalar_only_inputs && (!is_external || n_args <= 3) {
        stages.push(PostEnumerativeStage::ScalarTemplates);
    }
    stages.push(PostEnumerativeStage::TemplateReference);
    if !(is_external && n_args > 3) {
        stages.push(PostEnumerativeStage::Search);
    }
    stages
}

fn code_power_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(a: i64, b: i64) -> i64 {{\n    acc: i64 = 1;\n    i: i64 = 0;\n    while i < b {{\n        acc = acc * a;\n        i = i + 1;\n    }}\n    return acc;\n}}\n"
    )
}

fn code_digit_sum_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{\n        x = 0 - x;\n    }}\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = acc + (x % 10);\n        x = x / 10;\n    }}\n    return acc;\n}}\n"
    )
}

fn code_reverse_digits_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{\n        x = 0 - x;\n    }}\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = (acc * 10) + (x % 10);\n        x = x / 10;\n    }}\n    return acc;\n}}\n"
    )
}

fn code_digit_count_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{\n        x = 0 - x;\n    }}\n    if x == 0 {{\n        return 1;\n    }}\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = acc + 1;\n        x = x / 10;\n    }}\n    return acc;\n}}\n"
    )
}

fn code_count_even_digits_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{\n        x = 0 - x;\n    }}\n    if x == 0 {{\n        return 1;\n    }}\n    acc: i64 = 0;\n    while x > 0 {{\n        if ((x % 10) % 2) == 0 {{\n            acc = acc + 1;\n        }}\n        x = x / 10;\n    }}\n    return acc;\n}}\n"
    )
}

fn code_fib_iter_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    if n == 0 {{ return 0; }}\n    if n == 1 {{ return 1; }}\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= n {{\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }}\n    return b;\n}}\n"
    )
}

fn code_quadratic_search(fn_name: &str, a: i64, b: i64, c: i64) -> String {
    format!("fn {fn_name}(x: i64) -> i64 {{\n    return ({a} * x * x) + ({b} * x) + {c};\n}}\n")
}

fn code_contains_literal_search(fn_name: &str, literal: &str) -> String {
    let literal = literal.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "fn {fn_name}(s: string) -> i64 {{\n    if s.contains(\"{literal}\") {{\n        return 1;\n    }}\n    return 0;\n}}\n"
    )
}

fn code_starts_with_literal_search(fn_name: &str, literal: &str) -> String {
    let literal = literal.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "fn {fn_name}(s: string) -> i64 {{\n    if s.starts_with(\"{literal}\") {{\n        return 1;\n    }}\n    return 0;\n}}\n"
    )
}

fn verified_result(problem: &Problem, code: String, method: &str) -> Option<SolveResult> {
    verify_problem_code_strict(problem, &code).ok()?;
    Some(SolveResult {
        success: true,
        code,
        method: method.to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    })
}

fn code_abs_diff(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(a: i64, b: i64) -> i64 {
    if a > b {
        return a - b;
    } else {
        return b - a;
    }
}
"#,
        fn_name,
    )
}

fn code_max2(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(a: i64, b: i64) -> i64 {
    if a > b {
        return a;
    } else {
        return b;
    }
}
"#,
        fn_name,
    )
}

fn code_clamp(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(x: i64) -> i64 {
    if x < 0 {
        return 0;
    }
    if x > 100 {
        return 100;
    }
    return x;
}
"#,
        fn_name,
    )
}

fn code_sign(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(x: i64) -> i64 {
    if x < 0 {
        return -1;
    }
    if x > 0 {
        return 1;
    }
    return 0;
}
"#,
        fn_name,
    )
}

fn code_combat_resolve(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(attack: i64, defense: i64) -> i64 {
    damage: i64 = attack - defense;
    if damage < 0 {
        return 0;
    }
    return damage;
}
"#,
        fn_name,
    )
}

fn code_score_tracker(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(score: i64, event: i64) -> i64 {
    if event == 0 {
        return score + 1;
    }
    if event == 1 {
        return score + 5;
    }
    if event == 2 {
        return 0;
    }
    return score;
}
"#,
        fn_name,
    )
}

fn code_vending_change(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(coins_in: i64, price: i64) -> i64 {
    if coins_in >= price {
        return coins_in - price;
    }
    return -1;
}
"#,
        fn_name,
    )
}

fn code_turn_order_rotate(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(current: i64, num_players: i64) -> i64 {
    return (current + 1) % num_players;
}
"#,
        fn_name,
    )
}

fn code_grid_bounds_check(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(x: i64, y: i64, w: i64, h: i64) -> i64 {
    if x < 0 {
        return 0;
    }
    if y < 0 {
        return 0;
    }
    if x >= w {
        return 0;
    }
    if y >= h {
        return 0;
    }
    return 1;
}
"#,
        fn_name,
    )
}

fn code_simulate_gravity(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(v: i64, g: i64, t: i64) -> i64 {
    r: i64 = v + g * t;
    if r > 100 {
        return 100;
    }
    if r < 0 {
        return 0;
    }
    return r;
}
"#,
        fn_name,
    )
}

fn code_gcd(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(a: i64, b: i64) -> i64 {
    x: i64 = a;
    y: i64 = b;
    while y != 0 {
        tmp := y;
        y = x % y;
        x = tmp;
    }
    return x;
}
"#,
        fn_name,
    )
}

fn code_lcm(fn_name: &str) -> String {
    templ(
        r#"fn gcd_inner(a: i64, b: i64) -> i64 {
    x: i64 = a;
    y: i64 = b;
    while y != 0 {
        tmp := y;
        y = x % y;
        x = tmp;
    }
    return x;
}

fn __FN__(a: i64, b: i64) -> i64 {
    return (a * b) / gcd_inner(a, b);
}
"#,
        fn_name,
    )
}

fn code_array_sum(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        total = total + item;
    }
    return total;
}
"#,
        fn_name,
    )
}

fn code_array_max(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    best := arr[0];
    for item in arr {
        if item > best {
            best = item;
        }
    }
    return best;
}
"#,
        fn_name,
    )
}

fn code_count_occurrences(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64], target: i64) -> i64 {
    count: i64 = 0;
    for item in arr {
        if item == target {
            count = count + 1;
        }
    }
    return count;
}
"#,
        fn_name,
    )
}

fn code_trimmed_len(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(s: string) -> i64 {
    t := s.trim();
    return t.len;
}
"#,
        fn_name,
    )
}

fn code_vowel_count(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(s: string) -> i64 {
    chars := s.split("");
    total: i64 = 0;
    for ch in chars {
        if ch == "a" { total = total + 1; }
        if ch == "e" { total = total + 1; }
        if ch == "i" { total = total + 1; }
        if ch == "o" { total = total + 1; }
        if ch == "u" { total = total + 1; }
    }
    return total;
}
"#,
        fn_name,
    )
}

fn code_point_sum(fn_name: &str) -> String {
    templ(
        r#"struct Point {
    x: i64,
    y: i64,
}

fn __FN__(p: Point) -> i64 {
    return p.x + p.y;
}
"#,
        fn_name,
    )
}

fn code_safe_div_or_neg1(fn_name: &str) -> String {
    templ(
        r#"fn helper_div(a: i64, b: i64) -> Result<i64> {
    if b == 0 {
        return err("division by zero");
    }
    return ok(a / b);
}

fn __FN__(a: i64, b: i64) -> i64 {
    r := helper_div(a, b);
    out: i64 = match r {
        ok(v) => v,
        err(e) => -1,
    };
    return out;
}
"#,
        fn_name,
    )
}

fn code_positive_or_default(fn_name: &str) -> String {
    templ(
        r#"fn maybe_positive(x: i64) -> ?i64 {
    if x > 0 {
        return some(x);
    }
    return none;
}

fn __FN__(x: i64) -> i64 {
    r := maybe_positive(x);
    out: i64 = match r {
        some(v) => v,
        none => 0,
    };
    return out;
}
"#,
        fn_name,
    )
}

fn code_closure_map_sum(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    doubled := arr.map(fn(x: i64) -> i64 { x * 2 });
    total: i64 = 0;
    for item in doubled {
        total = total + item;
    }
    return total;
}
"#,
        fn_name,
    )
}

fn code_count_positive(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        if item > 0 {
            total = total + 1;
        }
    }
    return total;
}
"#,
        fn_name,
    )
}

fn code_is_even(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(x: i64) -> i64 {
    if (x % 2) == 0 {
        return 1;
    }
    return 0;
}
"#,
        fn_name,
    )
}

fn code_rectangle_area(fn_name: &str) -> String {
    templ(
        r#"struct Rectangle {
    width: i64,
    height: i64,
}

fn __FN__(r: Rectangle) -> i64 {
    return r.width * r.height;
}
"#,
        fn_name,
    )
}

fn code_collatz_steps(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    x: i64 = n;
    steps: i64 = 0;
    while x > 1 {
        if x % 2 == 0 {
            x = x / 2;
        } else {
            x = 3 * x + 1;
        }
        steps = steps + 1;
    }
    return steps;
}
"#,
        fn_name,
    )
}

fn code_min3(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(a: i64, b: i64, c: i64) -> i64 {
    m: i64 = a;
    if b < m { m = b; }
    if c < m { m = c; }
    return m;
}
"#,
        fn_name,
    )
}

fn code_is_prime(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    if n < 2 { return 0; }
    if n == 2 { return 1; }
    if n % 2 == 0 { return 0; }
    i: i64 = 3;
    while i * i <= n {
        if n % i == 0 { return 0; }
        i = i + 2;
    }
    return 1;
}
"#,
        fn_name,
    )
}

fn code_palindrome_check(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(s: string) -> i64 {
    chars := s.split("");
    left: i64 = 0;
    right: i64 = s.len - 1;
    while left < right {
        if chars[left] != chars[right] { return 0; }
        left = left + 1;
        right = right - 1;
    }
    return 1;
}
"#,
        fn_name,
    )
}

fn code_count_words(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(s: string) -> i64 {
    t := s.trim();
    if t.len == 0 { return 0; }
    parts := t.split(" ");
    count: i64 = 0;
    for p in parts {
        if p.len > 0 {
            count = count + 1;
        }
    }
    return count;
}
"#,
        fn_name,
    )
}

fn code_euler_totient(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    result: i64 = n;
    p: i64 = 2;
    temp: i64 = n;
    while p * p <= temp {
        if temp % p == 0 {
            while temp % p == 0 {
                temp = temp / p;
            }
            result = result - result / p;
        }
        p = p + 1;
    }
    if temp > 1 {
        result = result - result / temp;
    }
    return result;
}
"#,
        fn_name,
    )
}

fn code_count_divisors(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    count: i64 = 0;
    i: i64 = 1;
    while i <= n {
        if n % i == 0 {
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

fn code_triangular_check(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    k: i64 = 0;
    while k * (k + 1) / 2 <= n {
        if k * (k + 1) / 2 == n { return 1; }
        k = k + 1;
    }
    return 0;
}
"#,
        fn_name,
    )
}

fn code_max_pair_diff(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    best: i64 = 0;
    i: i64 = 1;
    while i < arr.len {
        diff: i64 = arr[i] - arr[i - 1];
        if diff < 0 { diff = 0 - diff; }
        if diff > best { best = diff; }
        i = i + 1;
    }
    return best;
}
"#,
        fn_name,
    )
}

fn code_sum_negatives(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        if item < 0 {
            total = total + item;
        }
    }
    return total;
}
"#,
        fn_name,
    )
}

fn code_harmonic_sum(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    total: i64 = 0;
    i: i64 = 1;
    while i <= n {
        total = total + 1000 / i;
        i = i + 1;
    }
    return total;
}
"#,
        fn_name,
    )
}

fn code_second_max(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    first: i64 = arr[0];
    second: i64 = arr[0];
    for item in arr {
        if item > first {
            second = first;
            first = item;
        } else {
            if item > second {
                second = item;
            }
        }
    }
    return second;
}
"#,
        fn_name,
    )
}

fn code_array_range(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    lo: i64 = arr[0];
    hi: i64 = arr[0];
    for item in arr {
        if item < lo {
            lo = item;
        }
        if item > hi {
            hi = item;
        }
    }
    return hi - lo;
}
"#,
        fn_name,
    )
}

fn code_run_length_decode_sum(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    total: i64 = 0;
    i: i64 = 0;
    while i < arr.len {
        total = total + arr[i] * arr[i + 1];
        i = i + 2;
    }
    return total;
}
"#,
        fn_name,
    )
}

fn code_count_adjacent_diff(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    count: i64 = 0;
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

fn code_sum_of_divisors(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    total: i64 = 0;
    i: i64 = 1;
    while i <= n {
        if n % i == 0 {
            total = total + i;
        }
        i = i + 1;
    }
    return total;
}
"#,
        fn_name,
    )
}

fn code_sum_odd_digits(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    x: i64 = n;
    acc: i64 = 0;
    while x > 0 {
        d: i64 = x % 10;
        if (d % 2) == 1 {
            acc = acc + d;
        }
        x = x / 10;
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_count_greater_than(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64], k: i64) -> i64 {
    acc: i64 = 0;
    for item in arr {
        if item > k {
            acc = acc + 1;
        }
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_prefix_sum_k(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64], k: i64) -> i64 {
    acc: i64 = 0;
    i: i64 = 0;
    while i < k {
        acc = acc + arr[i];
        i = i + 1;
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_digit_product(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    x: i64 = n;
    acc: i64 = 1;
    while x > 0 {
        acc = acc * (x % 10);
        x = x / 10;
    }
    return acc;
}
"#,
        fn_name,
    )
}

fn code_max_digit(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    x: i64 = n;
    best: i64 = 0;
    while x > 0 {
        d: i64 = x % 10;
        if d > best {
            best = d;
        }
        x = x / 10;
    }
    return best;
}
"#,
        fn_name,
    )
}

pub fn solve_problem_with_legacy_fallback(problem: &Problem) -> SolveResult {
    let result = solve_problem_search_only(problem);
    if result.success {
        return result;
    }
    legacy_fallback::solve(problem)
}

pub fn solve_problem_legacy_only(problem: &Problem) -> SolveResult {
    legacy_fallback::solve(problem)
}

pub fn solve_problem_differentiable_only(problem: &Problem) -> SolveResult {
    let result = solve_problem_differentiable_bridge(problem);
    SolveResult {
        success: result.success,
        code: result.code,
        method: result.method,
        error: result.error,
        metadata: result.metadata,
    }
}

pub fn solve_problem_prefer_differentiable(problem: &Problem) -> SolveResult {
    post_enumerative::solve_problem_prefer_differentiable(problem)
}

pub fn solve_problem(problem: &Problem) -> SolveResult {
    pipeline::solve_problem(problem)
}

pub fn solve_problem_search_only(problem: &Problem) -> SolveResult {
    search::solve_problem_search_only(problem)
}

pub fn solve_benchmark_with_legacy_fallback(problems: &[Problem]) -> BenchmarkSummary {
    benchmarking::run_benchmark(problems, solve_problem_with_legacy_fallback)
}

pub fn solve_benchmark_legacy_only(problems: &[Problem]) -> BenchmarkSummary {
    benchmarking::run_benchmark(problems, solve_problem_legacy_only)
}

pub fn solve_benchmark_differentiable_only(problems: &[Problem]) -> BenchmarkSummary {
    benchmarking::run_benchmark(problems, solve_problem_differentiable_only)
}

pub fn solve_benchmark_prefer_differentiable(problems: &[Problem]) -> BenchmarkSummary {
    benchmarking::solve_benchmark_prefer_differentiable(problems)
}

pub fn solve_benchmark(problems: &[Problem]) -> BenchmarkSummary {
    solve_benchmark_prefer_differentiable(problems)
}

pub fn solve_benchmark_search_only(problems: &[Problem]) -> BenchmarkSummary {
    benchmarking::run_benchmark(problems, solve_problem_search_only)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{factory_count, generated_holdouts, get_benchmark, Example, Value};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_MODEL_ROOT_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn aliased_problem(
        source_prefix: &str,
        name: &str,
        signature: &'static str,
        category: &'static str,
        description: &'static str,
    ) -> Problem {
        let source = get_benchmark(1)
            .into_iter()
            .find(|p| p.name.starts_with(source_prefix))
            .unwrap();
        Problem {
            name: name.to_string(),
            category,
            description,
            signature,
            examples: source.examples,
            holdouts: vec![],
            reference_code: "",
        }
    }

    fn assert_search_generalizes_problem(problem: Problem, holdouts: Vec<(Vec<Value>, i64)>) {
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "search failed for {}", problem.name);

        for (inputs, expected) in holdouts {
            let actual = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &inputs,
                &problem,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "execution failed for {} on {:?}: {err}",
                    problem.name, inputs
                )
            });
            match actual {
                crate::runtime::Value::Int(value) => {
                    assert_eq!(
                        value, expected,
                        "wrong result for {} on {:?}",
                        problem.name, inputs
                    );
                }
                other => panic!("expected int result for {}, got {:?}", problem.name, other),
            }
        }
    }

    fn assert_search_generalizes(problem_name: &str, holdouts: Vec<(Vec<Value>, i64)>) {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == problem_name)
            .unwrap();
        assert_search_generalizes_problem(problem, holdouts);
    }

    fn temp_model_root() -> std::path::PathBuf {
        let nonce = TEMP_MODEL_ROOT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "mog-warmstart-models-{}-{}",
            std::process::id(),
            nonce
        ));
        fs::create_dir_all(root.join("models")).unwrap();
        root
    }

    fn with_scratch_method_router<R>(f: impl FnOnce() -> R) -> R {
        crate::method_router::with_test_lock(|| {
            crate::solved_cache::with_test_lock(|| {
                let scratch = std::env::temp_dir().join(format!(
                    "nsynth_router_solver_test_{}_{:?}.json",
                    std::process::id(),
                    std::thread::current().id(),
                ));
                std::env::set_var("NSYNTH_METHOD_ROUTER_PATH", &scratch);
                crate::method_router::reset_for_tests();
                crate::solved_cache::reset_for_tests();
                let _ = fs::remove_file(&scratch);
                let result = f();
                std::env::remove_var("NSYNTH_METHOD_ROUTER_PATH");
                crate::method_router::reset_for_tests();
                crate::solved_cache::reset_for_tests();
                let _ = fs::remove_file(&scratch);
                result
            })
        })
    }

    fn with_scratch_router_and_cache<R>(f: impl FnOnce() -> R) -> R {
        crate::method_router::with_test_lock(|| {
            crate::solved_cache::with_test_lock(|| {
                let router = std::env::temp_dir().join(format!(
                    "nsynth_router_solver_test_{}_{:?}.json",
                    std::process::id(),
                    std::thread::current().id(),
                ));
                let cache = std::env::temp_dir().join(format!(
                    "nsynth_cache_solver_test_{}_{:?}.json",
                    std::process::id(),
                    std::thread::current().id(),
                ));
                std::env::set_var("NSYNTH_METHOD_ROUTER_PATH", &router);
                std::env::set_var("NSYNTH_CACHE_PATH", &cache);
                crate::method_router::reset_for_tests();
                crate::solved_cache::reset_for_tests();
                let _ = fs::remove_file(&router);
                let _ = fs::remove_file(&cache);
                let result = f();
                std::env::remove_var("NSYNTH_METHOD_ROUTER_PATH");
                std::env::remove_var("NSYNTH_CACHE_PATH");
                crate::method_router::reset_for_tests();
                crate::solved_cache::reset_for_tests();
                let _ = fs::remove_file(&router);
                let _ = fs::remove_file(&cache);
                result
            })
        })
    }

    fn with_scratch_search_family_router<R>(f: impl FnOnce() -> R) -> R {
        crate::search_family_router::with_test_lock(|| {
            let router = std::env::temp_dir().join(format!(
                "nsynth_search_family_solver_test_{}_{:?}.json",
                std::process::id(),
                std::thread::current().id(),
            ));
            std::env::set_var("NSYNTH_SEARCH_FAMILY_ROUTER_PATH", &router);
            crate::search_family_router::reset_for_tests();
            let _ = fs::remove_file(&router);
            let result = f();
            std::env::remove_var("NSYNTH_SEARCH_FAMILY_ROUTER_PATH");
            crate::search_family_router::reset_for_tests();
            let _ = fs::remove_file(&router);
            result
        })
    }

    #[test]
    fn benchmark_factory_count_matches_generated_benchmark() {
        assert_eq!(factory_count(), get_benchmark(1).len());
    }

    #[test]
    fn benchmark_generated_holdouts_cover_full_benchmark() {
        for problem in get_benchmark(1) {
            assert!(
                !generated_holdouts(&problem).is_empty(),
                "missing generated holdouts for {}",
                problem.name
            );
        }
    }

    #[test]
    fn python_warmstart_prefers_latest_available_model() {
        let root = temp_model_root();
        fs::write(root.join("models/metalearner_1arg_v3.pt"), b"v3").unwrap();
        fs::write(root.join("models/metalearner_1arg_v5.pt"), b"v5").unwrap();

        let selected = find_python_warmstart_model(&root).unwrap();
        assert_eq!(selected, root.join("models/metalearner_1arg_v5.pt"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn python_warmstart_falls_back_when_latest_model_is_missing() {
        let root = temp_model_root();
        fs::write(root.join("models/metalearner_1arg_v3.pt"), b"v3").unwrap();

        let selected = find_python_warmstart_model(&root).unwrap();
        assert_eq!(selected, root.join("models/metalearner_1arg_v3.pt"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn solves_count_positive() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name.starts_with("count_positive"))
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert!(result.code.contains("for item in arr"));
    }

    #[test]
    fn differentiable_only_solves_add_two() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "add_two_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(result.method.starts_with("diff_gradient_"));
        assert!(result.code.contains("return a + b;"));
    }

    #[test]
    fn prefer_differentiable_keeps_gradient_for_supported_family() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "add_two_v0")
            .unwrap();
        let result = solve_problem_prefer_differentiable(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(
            result.method == "diff_gradient_arithmetic"
                || result.method == "search_scalar_expr"
                || result.method == "synth_gradient",
            "expected differentiable solve when the bridge is available, otherwise native/search fallback; got {}",
            result.method
        );
    }

    #[test]
    fn prefer_differentiable_skips_probe_for_positive_or_default() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "positive_or_default_v0")
            .unwrap();
        let result = solve_problem_prefer_differentiable(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(
            result.method.starts_with("search_")
                || result.method.starts_with("diff_gradient_")
                || result.method == "synth_gradient",
            "expected search/native differentiable fallback, got {}",
            result.method
        );
    }

    #[test]
    fn prefer_differentiable_skips_probe_for_is_prime() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "is_prime_v0")
            .unwrap();
        let result = solve_problem_prefer_differentiable(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(
            result.method == "search_is_prime_loop"
                || result.method.starts_with("diff_gradient_")
                || result.method == "synth_gradient",
            "expected search/native differentiable fallback, got {}",
            result.method
        );
    }

    #[test]
    fn differentiable_only_solves_abs_diff() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "abs_diff_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_branch");
        assert!(result.code.contains("return a - b;"));
        assert!(result.code.contains("return b - a;"));
    }

    #[test]
    fn differentiable_only_rejects_array_problem() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "array_sum_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(!result.success);
        assert_eq!(result.method, "diff_gradient_unsupported");
    }

    #[test]
    fn differentiable_only_solves_sign() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "sign_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_soft_multi_branch");

        for (input, expected) in [(-8, -1), (0, 0), (15, 1)] {
            let exec = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &[Value::Int(input)],
                &problem,
            )
            .unwrap();
            match exec {
                crate::runtime::Value::Int(value) => assert_eq!(value, expected),
                other => panic!("expected int result, got {:?}", other),
            }
        }
    }

    #[test]
    fn differentiable_only_solves_clamp() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "clamp_0_100_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_soft_multi_branch");

        for (input, expected) in [(-1, 0), (42, 42), (101, 100)] {
            let exec = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &[Value::Int(input)],
                &problem,
            )
            .unwrap();
            match exec {
                crate::runtime::Value::Int(value) => assert_eq!(value, expected),
                other => panic!("expected int result, got {:?}", other),
            }
        }
    }

    #[test]
    fn differentiable_only_solves_safe_div() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "safe_div_or_neg1_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_branch");

        for ((a, b), expected) in [((9, 0), -1), ((21, 7), 3)] {
            let exec = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &[Value::Int(a), Value::Int(b)],
                &problem,
            )
            .unwrap();
            match exec {
                crate::runtime::Value::Int(value) => assert_eq!(value, expected),
                other => panic!("expected int result, got {:?}", other),
            }
        }
    }

    #[test]
    fn differentiable_only_solves_is_even() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "is_even_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_soft_multi_branch");

        for (input, expected) in [(-6, 1), (20, 1), (105, 0)] {
            let exec = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &[Value::Int(input)],
                &problem,
            )
            .unwrap();
            match exec {
                crate::runtime::Value::Int(value) => assert_eq!(value, expected),
                other => panic!("expected int result, got {:?}", other),
            }
        }
    }

    #[test]
    fn differentiable_only_solves_sum_to_n() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "sum_to_n_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_loop");

        for (input, expected) in [(7, 28), (-3, 0)] {
            let exec = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &[Value::Int(input)],
                &problem,
            )
            .unwrap();
            match exec {
                crate::runtime::Value::Int(value) => assert_eq!(value, expected),
                other => panic!("expected int result, got {:?}", other),
            }
        }
    }

    #[test]
    fn differentiable_only_solves_factorial() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "factorial_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_loop");

        for (input, expected) in [(3, 6), (8, 40320)] {
            let exec = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &[Value::Int(input)],
                &problem,
            )
            .unwrap();
            match exec {
                crate::runtime::Value::Int(value) => assert_eq!(value, expected),
                other => panic!("expected int result, got {:?}", other),
            }
        }
    }

    #[test]
    fn differentiable_only_solves_digit_sum() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "digit_sum_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_digit_loop");

        for (input, expected) in [(405, 9), (7001, 8)] {
            let exec = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &[Value::Int(input)],
                &problem,
            )
            .unwrap();
            match exec {
                crate::runtime::Value::Int(value) => assert_eq!(value, expected),
                other => panic!("expected int result, got {:?}", other),
            }
        }
    }

    #[test]
    fn differentiable_only_solves_reverse_digits() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "reverse_digits_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_digit_loop");

        for (input, expected) in [(81, 18), (12030, 3021)] {
            let exec = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &[Value::Int(input)],
                &problem,
            )
            .unwrap();
            match exec {
                crate::runtime::Value::Int(value) => assert_eq!(value, expected),
                other => panic!("expected int result, got {:?}", other),
            }
        }
    }

    #[test]
    fn differentiable_only_solves_digit_count() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "digit_count_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_digit_loop");

        for (input, expected) in [(81, 2), (12030, 5)] {
            let exec = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &[Value::Int(input)],
                &problem,
            )
            .unwrap();
            match exec {
                crate::runtime::Value::Int(value) => assert_eq!(value, expected),
                other => panic!("expected int result, got {:?}", other),
            }
        }
    }

    #[test]
    fn differentiable_only_solves_count_even_digits() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "count_even_digits_v0")
            .unwrap();
        let result = solve_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "diff_gradient_digit_loop");

        for (input, expected) in [(81, 1), (12030, 3), (24680, 5)] {
            let exec = crate::runtime::execute_function_for_problem(
                &result.code,
                problem.function_name(),
                &[Value::Int(input)],
                &problem,
            )
            .unwrap();
            match exec {
                crate::runtime::Value::Int(value) => assert_eq!(value, expected),
                other => panic!("expected int result, got {:?}", other),
            }
        }
    }

    #[test]
    fn default_solver_prefers_differentiable_when_supported() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "add_two_v0")
            .unwrap();
        let result = solve_problem(&problem);
        assert!(result.success, "{:?}", result.error);
        // Exact scalar expression search now preempts the gradient stack for
        // simple closed-form arithmetic, but older routes remain acceptable if
        // search heuristics ever change.
        assert!(
            result.method == "search_scalar_expr"
                || result.method == "search_abs_diff_formula"
                || result.method.starts_with("diff_gradient_")
                || result.method == "synth_gradient"
                || result.method == "enumerative"
                || result.method == "template"
                || result.method == "template_reference",
            "unexpected method: {}",
            result.method
        );
    }

    #[test]
    fn default_solver_prefers_discovery_for_count_positive() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "count_positive_v0")
            .unwrap();
        let result = solve_problem(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(
            result.method == "enumerative-array"
                || result.method == "arr_gradient"
                || result.method == "univ_arr_gradient",
            "unexpected method: {}",
            result.method
        );
    }

    #[test]
    fn default_solver_uses_gradient_for_dot_product() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "dot_product_v0")
            .unwrap();
        let result = solve_problem(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "arr_gradient");
        assert!(result.code.contains("a[i] * b[i]"), "{}", result.code);
    }

    #[test]
    fn default_solver_uses_structured_gradient_for_hard_array_families() {
        let problems = get_benchmark(1);
        let targets = [
            ("kth_smallest_v0", "arr_gradient_kth_smallest"),
            ("two_sum_exists_v0", "arr_gradient_two_sum_exists"),
            ("count_distinct_v0", "arr_gradient_count_distinct"),
            ("binary_search_v0", "arr_gradient_binary_search"),
        ];
        for (name, expected_method) in targets {
            let problem = problems
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("{name} not found"));
            let result = solve_problem(problem);
            assert!(result.success, "{name}: {:?}", result.error);
            assert_eq!(result.method, expected_method, "{name}: {}", result.code);
        }
    }

    #[test]
    fn default_solver_short_circuits_lcm_to_search_teacher() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "lcm_v0")
            .unwrap();
        let result = solve_problem(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_lcm_formula");
        assert!(result.code.contains("gcd_inner"), "{}", result.code);
    }

    #[test]
    fn default_solver_short_circuits_euler_totient_to_search_teacher() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "euler_totient_v0")
            .unwrap();
        let result = solve_problem(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_euler_totient_loop");
        assert!(
            result.code.contains("result = result - result / p;"),
            "{}",
            result.code
        );
    }

    #[test]
    fn preemptive_search_teacher_solves_slow_exact_search_cases() {
        let problems = get_benchmark(1);
        let targets = [
            ("abs_diff_v0", "search_abs_diff_formula"),
            ("max2_v0", "search_max2_formula"),
            ("safe_div_or_neg1_v0", "search_safe_div_or_neg1_branch"),
            (
                "positive_or_default_v0",
                "search_positive_or_default_branch",
            ),
            ("clamp_0_100_v0", "search_clamp_formula"),
            ("sign_v0", "search_sign_branch"),
            ("is_even_v0", "search_is_even_formula"),
            ("gcd_v0", "search_gcd_loop"),
            ("next_power_of_2_v0", "search_next_power_of_2"),
            ("triangular_check_v0", "search_triangular_check_loop"),
            ("collatz_steps_v0", "search_collatz_loop"),
            ("lucas_number_v0", "search_lucas_loop"),
            ("celsius_to_fahrenheit_v0", "search_celsius_to_fahrenheit"),
            ("is_perfect_square_v0", "search_is_perfect_square"),
            ("leading_digit_v0", "search_leading_digit"),
            ("popcount_v0", "search_popcount"),
            ("polynomial_v0", "search_polynomial_quadratic"),
            ("scaled_sum_v0", "search_scalar_expr"),
            ("product_offset_v0", "search_scalar_expr"),
            ("bilinear3_v0", "search_scalar_expr"),
            ("min3_v0", "search_min3_branch"),
            ("digit_sum_v0", "search_digit_sum_loop"),
            ("reverse_digits_v0", "search_reverse_digits_loop"),
            ("digit_count_v0", "search_digit_count_loop"),
            ("digital_root_v0", "search_digital_root"),
            ("fibonacci_v0", "search_fib_iter_loop"),
            ("count_divisors_v0", "search_count_divisors_loop"),
            ("sum_of_divisors_v0", "search_sum_of_divisors_loop"),
            ("sum_odd_digits_v0", "search_sum_odd_digits_loop"),
            ("max_digit_v0", "search_max_digit_loop"),
            ("product_1_to_n_v0", "search_unary_range_loop"),
            ("is_prime_v0", "search_is_prime_loop"),
            ("harmonic_sum_v0", "search_harmonic_sum_loop"),
            ("run_length_decode_sum_v0", "search_run_length_decode_sum"),
            ("count_adjacent_diff_v0", "search_count_adjacent_diff"),
            ("prefix_sum_k_v0", "search_prefix_sum_k"),
            ("second_max_v0", "search_second_max"),
            ("array_range_v0", "search_array_range"),
            ("arr_sum_squares_v0", "search_arr_sum_squares"),
            ("min_element_v0", "search_min_element"),
            ("sum_absolute_v0", "search_sum_absolute"),
            ("count_evens_v0", "search_count_evens"),
            ("sum_positives_v0", "search_sum_positives"),
            ("max_consecutive_sum_v0", "search_max_consecutive_sum"),
            ("min_consecutive_sum_v0", "search_min_consecutive_sum"),
            ("max_stock_profit_v0", "search_max_stock_profit"),
            ("is_sorted_v0", "search_is_sorted"),
            ("longest_increasing_run_v0", "search_longest_increasing_run"),
            ("longest_plateau_v0", "search_longest_plateau"),
            ("prefix_max_sum_v0", "search_prefix_max_sum"),
            ("sum_at_even_indices_v0", "search_sum_at_even_indices"),
            ("kth_from_end_v0", "search_kth_from_end"),
            ("max_abs_v0", "search_max_abs"),
            ("min_positive_v0", "search_min_positive"),
            ("count_peaks_v0", "search_count_peaks"),
            ("alternating_sum_v0", "search_alternating_sum"),
            ("is_palindrome_arr_v0", "search_is_palindrome_arr"),
            ("sum_odd_indexed_v0", "search_sum_odd_indexed"),
            ("max_pair_diff_v0", "search_max_pair_diff"),
            ("combat_resolve_v0", "search_combat_resolve_branch"),
            ("score_tracker_v0", "search_score_tracker_branch"),
            ("vending_change_v0", "search_vending_change_branch"),
            ("turn_order_rotate_v0", "search_turn_order_rotate"),
            ("grid_bounds_check_v0", "search_grid_bounds_check_branch"),
            ("simulate_gravity_v0", "search_simulate_gravity_clamp"),
        ];

        for (name, expected_method) in targets {
            let problem = problems
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("{name} not found"));
            let result = solve_problem_from_preemptive_search_teacher(problem)
                .unwrap_or_else(|| panic!("{name}: missing preemptive search route"));
            assert!(result.success, "{name}: {:?}", result.error);
            assert_eq!(result.method, expected_method, "{name}: {}", result.code);
        }
    }

    #[test]
    fn search_family_router_reorders_exact_search_candidates() {
        with_scratch_search_family_router(|| {
            let problem = get_benchmark(1)
                .into_iter()
                .find(|p| p.name == "max2_v0")
                .unwrap();

            let baseline = search::ranked_search_candidate_keys(&problem);
            assert!(
                baseline
                    .iter()
                    .position(|key| *key == "search_max2_formula")
                    .unwrap()
                    < baseline
                        .iter()
                        .position(|key| *key == "search_single_branch")
                        .unwrap()
            );

            crate::search_family_router::record_attempt(
                &problem,
                &["search_single_branch"],
                Some("search_single_branch"),
            );
            crate::search_family_router::record_attempt(
                &problem,
                &["search_single_branch"],
                Some("search_single_branch"),
            );

            let reranked = search::ranked_search_candidate_keys(&problem);
            assert_eq!(reranked.first().copied(), Some("search_single_branch"));
            assert!(
                reranked
                    .iter()
                    .position(|key| *key == "search_single_branch")
                    .unwrap()
                    < reranked
                        .iter()
                        .position(|key| *key == "search_max2_formula")
                        .unwrap()
            );

            let result = solve_problem_search_only(&problem);
            assert!(result.success, "{:?}", result.error);
            assert_eq!(result.method, "search_single_branch");
        });
    }

    #[test]
    fn search_solves_exact_game_logic_families() {
        let problems = get_benchmark(1);
        let targets = [
            ("combat_resolve_v0", "search_combat_resolve_branch"),
            ("score_tracker_v0", "search_score_tracker_branch"),
            ("vending_change_v0", "search_vending_change_branch"),
            ("turn_order_rotate_v0", "search_turn_order_rotate"),
            ("grid_bounds_check_v0", "search_grid_bounds_check_branch"),
            ("simulate_gravity_v0", "search_simulate_gravity_clamp"),
        ];

        for (name, expected_method) in targets {
            let problem = problems
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("{name} not found"));
            let result = solve_problem_search_only(problem);
            assert!(result.success, "{name}: {:?}", result.error);
            assert_eq!(result.method, expected_method, "{name}: {}", result.code);
        }
    }

    #[test]
    fn search_solves_exact_array_loop_families() {
        let problems = get_benchmark(1);
        let targets = [
            ("run_length_decode_sum_v0", "search_run_length_decode_sum"),
            ("count_adjacent_diff_v0", "search_count_adjacent_diff"),
        ];

        for (name, expected_method) in targets {
            let problem = problems
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("{name} not found"));
            let result = solve_problem_search_only(problem);
            assert!(result.success, "{name}: {:?}", result.error);
            assert_eq!(result.method, expected_method, "{name}: {}", result.code);
        }
    }

    #[test]
    fn solves_gcd_extended() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name.starts_with("gcd_extended"))
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert!(result.code.contains("while y != 0"));
    }

    #[test]
    fn search_solves_aliased_array_sum_without_family_name() {
        let problem = aliased_problem(
            "array_sum",
            "mystery_reduce_v0",
            "fn mystery_reduce(xs: [i64]) -> i64",
            "array_search",
            "Return the total of all elements in xs.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_array_sum");
        assert!(result.code.contains("for item in arr"));
        assert!(result.code.contains("fn mystery_reduce"));
    }

    #[test]
    fn search_solves_aliased_lcm_without_family_name() {
        let problem = aliased_problem(
            "lcm",
            "mystery_lcm_v0",
            "fn mystery_lcm(a: i64, b: i64) -> i64",
            "scalar_search",
            "Return the least common multiple of a and b.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_lcm_formula");
        assert!(result.code.contains("gcd_inner"));
        assert!(result.code.contains("fn mystery_lcm"));
    }

    #[test]
    fn search_solves_aliased_add_two_without_family_name() {
        let problem = aliased_problem(
            "add_two",
            "mystery_plus_v0",
            "fn mystery_plus(left: i64, right: i64) -> i64",
            "scalar_search",
            "Return the sum of the two inputs.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_scalar_expr");
        assert!(result.code.contains('+'));
        assert!(result.code.contains("fn mystery_plus"));
    }

    #[test]
    fn search_solves_aliased_abs_diff_without_family_name() {
        let problem = aliased_problem(
            "abs_diff",
            "mystery_gap_v0",
            "fn mystery_gap(left: i64, right: i64) -> i64",
            "scalar_search",
            "Return the absolute difference between the two inputs.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_abs_diff_formula");
        assert!(result.code.contains("fn mystery_gap"));
    }

    #[test]
    fn search_solves_aliased_polynomial_without_family_name() {
        let problem = aliased_problem(
            "polynomial",
            "mystery_quadratic_v0",
            "fn mystery_quadratic(x: i64) -> i64",
            "scalar_search",
            "Evaluate a small quadratic polynomial of x.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_polynomial_quadratic");
        assert!(result.code.contains("x * x"));
        assert!(result.code.contains("fn mystery_quadratic"));
    }

    #[test]
    fn search_solves_aliased_sum_to_n_without_family_name() {
        let problem = aliased_problem(
            "sum_to_n",
            "mystery_series_v0",
            "fn mystery_series(value: i64) -> i64",
            "scalar_search",
            "Return the total from 1 through value.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_unary_range_loop");
        assert!(result.code.contains("while i <= n"));
        assert!(result.code.contains("acc = acc + i;"));
        assert!(result.code.contains("fn mystery_series"));
    }

    #[test]
    fn search_solves_aliased_sum_squares_without_family_name() {
        let problem = aliased_problem(
            "sum_squares",
            "mystery_square_series_v0",
            "fn mystery_square_series(value: i64) -> i64",
            "scalar_search",
            "Return the sum of the squares from 1 through value.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_unary_range_loop");
        assert!(result.code.contains("acc = acc + (i * i);"));
        assert!(result.code.contains("fn mystery_square_series"));
    }

    #[test]
    fn search_solves_aliased_product_without_family_name() {
        let problem = aliased_problem(
            "product_1_to_n",
            "mystery_product_v0",
            "fn mystery_product(value: i64) -> i64",
            "scalar_search",
            "Return the product of all integers from 1 through value.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_unary_range_loop");
        assert!(result.code.contains("acc = acc * i;"));
        assert!(result.code.contains("fn mystery_product"));
    }

    #[test]
    fn search_solves_aliased_min3_without_family_name() {
        let problem = aliased_problem(
            "min3",
            "mystery_min3_v0",
            "fn mystery_min3(a: i64, b: i64, c: i64) -> i64",
            "scalar_search",
            "Return the minimum of three integers.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_min3_branch");
        assert!(result.code.contains("if b < m"));
        assert!(result.code.contains("fn mystery_min3"));
    }

    #[test]
    fn search_solves_aliased_count_positive_without_family_name() {
        let problem = aliased_problem(
            "count_positive",
            "mystery_positive_counter_v0",
            "fn mystery_positive_counter(xs: [i64]) -> i64",
            "array_search",
            "Count how many entries in xs are strictly above zero.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_array_count_positive");
        assert!(result.code.contains("if item > 0"));
        assert!(result.code.contains("fn mystery_positive_counter"));
    }

    #[test]
    fn search_solves_aliased_count_occurrences_without_family_name() {
        let problem = aliased_problem(
            "count_occurrences",
            "mystery_matches_v0",
            "fn mystery_matches(xs: [i64], needle: i64) -> i64",
            "array_search",
            "Count how many entries in xs equal needle.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_array_count_occurrences");
        assert!(result.code.contains("if item == target"));
        assert!(result.code.contains("fn mystery_matches"));
    }

    #[test]
    fn search_solves_aliased_closure_map_sum_without_family_name() {
        let problem = aliased_problem(
            "closure_map_sum",
            "mystery_map_sum_v0",
            "fn mystery_map_sum(arr: [i64]) -> i64",
            "array_search",
            "Double each array element and return the sum of the doubled values.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_closure_map_sum");
        assert!(result.code.contains("arr.map"));
        assert!(result.code.contains("fn mystery_map_sum"));
    }

    #[test]
    fn search_solves_aliased_safe_div_without_family_name() {
        let mut problem = aliased_problem(
            "safe_div_or_neg1",
            "mystery_safe_div_v0",
            "fn mystery_safe_div(a: i64, b: i64) -> i64",
            "scalar_search",
            "Return a divided by b, or -1 when b is zero.",
        );
        problem.examples.push(Example {
            inputs: vec![Value::Int(20), Value::Int(4)],
            expected: 5,
        });
        problem.examples.push(Example {
            inputs: vec![Value::Int(8), Value::Int(2)],
            expected: 4,
        });
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_safe_div_or_neg1_branch");
        assert!(result.code.contains("helper_div"));
        assert!(result.code.contains("=> -1"));
        assert!(result.code.contains(" / "));
        assert!(result.code.contains("fn mystery_safe_div"));
    }

    #[test]
    fn search_solves_aliased_trimmed_len_without_family_name() {
        let problem = aliased_problem(
            "trimmed_len",
            "mystery_trim_v0",
            "fn mystery_trim(s: string) -> i64",
            "string_search",
            "Trim spaces from s and return the resulting length.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_trimmed_len");
        assert!(result.code.contains("s.trim()"));
        assert!(result.code.contains("fn mystery_trim"));
    }

    #[test]
    fn search_solves_aliased_contains_literal_without_family_name() {
        let problem = aliased_problem(
            "contains_cat",
            "mystery_contains_v0",
            "fn mystery_contains(s: string) -> i64",
            "string_search",
            "Return 1 when s contains a learned literal substring.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_contains_literal");
        assert!(result.code.contains(".contains(\"cat\")"));
        assert!(result.code.contains("fn mystery_contains"));
    }

    #[test]
    fn search_solves_aliased_starts_with_literal_without_family_name() {
        let problem = aliased_problem(
            "starts_with_m",
            "mystery_prefix_v0",
            "fn mystery_prefix(s: string) -> i64",
            "string_search",
            "Return 1 when s starts with a learned prefix.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_starts_with_literal");
        assert!(result.code.contains(".starts_with(\"m\")"));
        assert!(result.code.contains("fn mystery_prefix"));
    }

    #[test]
    fn search_solves_aliased_vowel_count_without_family_name() {
        let problem = aliased_problem(
            "vowel_count",
            "mystery_vowels_v0",
            "fn mystery_vowels(s: string) -> i64",
            "string_search",
            "Count vowels in s.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_vowel_count");
        assert!(result.code.contains("if ch == \"a\""));
        assert!(result.code.contains("fn mystery_vowels"));
    }

    #[test]
    fn search_solves_aliased_count_words_without_family_name() {
        let problem = aliased_problem(
            "count_words",
            "mystery_words_v0",
            "fn mystery_words(s: string) -> i64",
            "string_search",
            "Count the number of words in s.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_count_words");
        assert!(result.code.contains("split(\" \")"));
        assert!(result.code.contains("fn mystery_words"));
    }

    #[test]
    fn search_solves_aliased_palindrome_without_family_name() {
        let problem = aliased_problem(
            "palindrome_check",
            "mystery_palindrome_v0",
            "fn mystery_palindrome(s: string) -> i64",
            "string_search",
            "Return 1 when s is a palindrome.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_palindrome");
        assert!(result.code.contains("left < right"));
        assert!(result.code.contains("fn mystery_palindrome"));
    }

    #[test]
    fn search_solves_aliased_power_without_family_name() {
        let problem = aliased_problem(
            "power",
            "mystery_power_v0",
            "fn mystery_power(base: i64, exp: i64) -> i64",
            "scalar_search",
            "Raise base to the non-negative exponent exp.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_power_loop");
        assert!(result.code.contains("while i < b"));
        assert!(result.code.contains("acc = acc * a;"));
        assert!(result.code.contains("fn mystery_power"));
    }

    #[test]
    fn search_solves_aliased_collatz_without_family_name() {
        let problem = aliased_problem(
            "collatz_steps",
            "mystery_collatz_v0",
            "fn mystery_collatz(value: i64) -> i64",
            "scalar_search",
            "Return how many Collatz steps are needed for value to reach one.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_collatz_loop");
        assert!(result.code.contains("while x > 1"));
        assert!(result.code.contains("x = 3 * x + 1;"));
        assert!(result.code.contains("fn mystery_collatz"));
    }

    #[test]
    fn search_solves_aliased_is_prime_without_family_name() {
        let problem = aliased_problem(
            "is_prime",
            "mystery_prime_v0",
            "fn mystery_prime(value: i64) -> i64",
            "scalar_search",
            "Return 1 when value is prime and 0 otherwise.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_is_prime_loop");
        assert!(result.code.contains("while i * i <= n"));
        assert!(result.code.contains("return 1;"));
        assert!(result.code.contains("fn mystery_prime"));
    }

    #[test]
    fn search_solves_aliased_digit_sum_without_family_name() {
        let problem = aliased_problem(
            "digit_sum",
            "mystery_digits_v0",
            "fn mystery_digits(value: i64) -> i64",
            "scalar_search",
            "Return the sum of the base-10 digits of value.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_digit_sum_loop");
        assert!(result.code.contains("x % 10"));
        assert!(result.code.contains("x = x / 10;"));
        assert!(result.code.contains("fn mystery_digits"));
    }

    #[test]
    fn search_solves_aliased_reverse_digits_without_family_name() {
        let problem = aliased_problem(
            "reverse_digits",
            "mystery_reverse_digits_v0",
            "fn mystery_reverse_digits(value: i64) -> i64",
            "scalar_search",
            "Reverse the base-10 digits of value.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_reverse_digits_loop");
        assert!(result.code.contains("acc = (acc * 10) + (x % 10);"));
        assert!(result.code.contains("fn mystery_reverse_digits"));
    }

    #[test]
    fn search_solves_aliased_digit_count_without_family_name() {
        let problem = aliased_problem(
            "digit_count",
            "mystery_digit_count_v0",
            "fn mystery_digit_count(value: i64) -> i64",
            "scalar_search",
            "Count how many base-10 digits value contains.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_digit_count_loop");
        assert!(result.code.contains("acc = acc + 1;"));
        assert!(result.code.contains("fn mystery_digit_count"));
    }

    #[test]
    fn search_solves_aliased_count_even_digits_without_family_name() {
        let problem = aliased_problem(
            "count_even_digits",
            "mystery_count_even_digits_v0",
            "fn mystery_count_even_digits(value: i64) -> i64",
            "scalar_search",
            "Count how many base-10 digits of value are even.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_count_even_digits_loop");
        assert!(result.code.contains("((x % 10) % 2) == 0"));
        assert!(result.code.contains("fn mystery_count_even_digits"));
    }

    #[test]
    fn search_solves_aliased_gcd_without_family_name() {
        let problem = aliased_problem(
            "gcd_extended",
            "mystery_euclid_v0",
            "fn mystery_euclid(a: i64, b: i64) -> i64",
            "scalar_search",
            "Return the Euclidean greatest common divisor of a and b.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_gcd_loop");
        assert!(result.code.contains("while y != 0"));
        assert!(result.code.contains("fn mystery_euclid"));
    }

    #[test]
    fn search_solves_aliased_point_sum_without_family_name() {
        let problem = aliased_problem(
            "point_sum",
            "mystery_point_v0",
            "fn mystery_point(p: Point) -> i64",
            "struct_search",
            "Return the sum of the point coordinates.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_struct_pair");
        assert!(result.code.contains("struct Point"));
        assert!(result.code.contains("return p.x + p.y;"));
    }

    #[test]
    fn search_solves_aliased_rectangle_area_without_family_name() {
        let problem = aliased_problem(
            "rectangle_area",
            "mystery_rect_v0",
            "fn mystery_rect(r: Rectangle) -> i64",
            "struct_search",
            "Return the rectangle area.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_struct_pair");
        assert!(result.code.contains("struct Rectangle"));
        assert!(result.code.contains("return r.width * r.height;"));
    }

    #[test]
    fn search_solves_aliased_count_divisors_without_family_name() {
        let problem = aliased_problem(
            "count_divisors",
            "mystery_divisors_v0",
            "fn mystery_divisors(value: i64) -> i64",
            "scalar_search",
            "Count the number of positive divisors of value.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_count_divisors_loop");
        assert!(result.code.contains("while i <= n"));
        assert!(result.code.contains("if n % i == 0"));
        assert!(result.code.contains("fn mystery_divisors"));
    }

    #[test]
    fn search_solves_aliased_fib_iter_without_family_name() {
        let problem = aliased_problem(
            "fib_iter",
            "mystery_fib_v0",
            "fn mystery_fib(value: i64) -> i64",
            "scalar_search",
            "Return the iterative Fibonacci number for value.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_fib_iter_loop");
        assert!(result.code.contains("tmp: i64 = a + b;"));
        assert!(result.code.contains("while i <= n"));
        assert!(result.code.contains("fn mystery_fib"));
    }

    #[test]
    fn search_solves_aliased_max_pair_diff_without_family_name() {
        let problem = aliased_problem(
            "max_pair_diff",
            "mystery_pair_diff_v0",
            "fn mystery_pair_diff(arr: [i64]) -> i64",
            "array_search",
            "Return the maximum absolute gap between consecutive elements.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_max_pair_diff");
        assert!(result.code.contains("arr[i] - arr[i - 1]"));
        assert!(result.code.contains("fn mystery_pair_diff"));
    }

    #[test]
    fn search_solves_aliased_harmonic_sum_without_family_name() {
        let problem = aliased_problem(
            "harmonic_sum",
            "mystery_harmonic_v0",
            "fn mystery_harmonic(value: i64) -> i64",
            "scalar_search",
            "Return the scaled harmonic sum 1000/1 + ... + 1000/value.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_harmonic_sum_loop");
        assert!(result.code.contains("total = total + 1000 / i;"));
        assert!(result.code.contains("fn mystery_harmonic"));
    }

    #[test]
    fn search_solves_aliased_triangular_check_without_family_name() {
        let problem = aliased_problem(
            "triangular_check",
            "mystery_triangular_v0",
            "fn mystery_triangular(value: i64) -> i64",
            "scalar_search",
            "Return 1 when value is triangular and 0 otherwise.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_triangular_check_loop");
        assert!(result.code.contains("k * (k + 1) / 2"));
        assert!(result.code.contains("fn mystery_triangular"));
    }

    #[test]
    fn search_solves_aliased_euler_totient_without_family_name() {
        let problem = aliased_problem(
            "euler_totient",
            "mystery_totient_v0",
            "fn mystery_totient(value: i64) -> i64",
            "scalar_search",
            "Compute Euler's totient function of value.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_euler_totient_loop");
        assert!(result.code.contains("while p * p <= temp"));
        assert!(result.code.contains("result = result - result / p;"));
        assert!(result.code.contains("fn mystery_totient"));
    }

    #[test]
    fn search_solves_aliased_clamp_without_family_name() {
        let problem = aliased_problem(
            "clamp_0_100",
            "mystery_clamp_v0",
            "fn mystery_clamp(value: i64) -> i64",
            "scalar_search",
            "Clamp value into the inclusive range from 0 to 100.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_clamp_formula");
        assert!(result.code.matches("if ").count() >= 2);
        assert!(result.code.contains("return 100;"));
        assert!(result.code.contains("fn mystery_clamp"));
    }

    #[test]
    fn search_solves_aliased_sign_without_family_name() {
        let problem = aliased_problem(
            "sign",
            "mystery_sign_v0",
            "fn mystery_sign(value: i64) -> i64",
            "scalar_search",
            "Return -1 for negative values, 0 for zero, and 1 for positive values.",
        );
        let result = solve_problem_search_only(&problem);
        assert!(result.success);
        assert_eq!(result.method, "search_sign_branch");
        assert!(result.code.matches("if ").count() >= 2);
        assert!(result.code.contains("return -1;"));
        assert!(result.code.contains("return 1;"));
        assert!(result.code.contains("fn mystery_sign"));
    }

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
                (vec![Value::Array(vec![10, -5, 2])], 7),
                (vec![Value::Array(vec![1, 2, 3, 4])], 10),
            ],
        );
        assert_search_generalizes(
            "count_positive_v0",
            vec![
                (vec![Value::Array(vec![0, 1, -1, 3])], 2),
                (vec![Value::Array(vec![-5, -2, 0])], 0),
            ],
        );
        assert_search_generalizes(
            "count_occurrences_v0",
            vec![
                (vec![Value::Array(vec![4, 1, 4, 4]), Value::Int(4)], 3),
                (vec![Value::Array(vec![2, 3]), Value::Int(5)], 0),
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

    #[test]
    fn search_only_generalizes_on_string_holdout_cases() {
        assert_search_generalizes(
            "trimmed_len_v0",
            vec![
                (vec![Value::Str("   hi there   ".to_string())], 8),
                (vec![Value::Str("      ".to_string())], 0),
            ],
        );
        assert_search_generalizes(
            "vowel_count_v0",
            vec![
                (vec![Value::Str("queue".to_string())], 4),
                (vec![Value::Str("sky".to_string())], 0),
            ],
        );
        assert_search_generalizes(
            "contains_cat_v0",
            vec![
                (vec![Value::Str("bobcat".to_string())], 1),
                (vec![Value::Str("atlas".to_string())], 0),
            ],
        );
        assert_search_generalizes(
            "starts_with_m_v0",
            vec![
                (vec![Value::Str("m".to_string())], 1),
                (vec![Value::Str("Map".to_string())], 0),
                (vec![Value::Str("moss".to_string())], 1),
            ],
        );
        assert_search_generalizes(
            "palindrome_check_v0",
            vec![
                (vec![Value::Str("abba".to_string())], 1),
                (vec![Value::Str("abca".to_string())], 0),
            ],
        );
        assert_search_generalizes(
            "count_words_v0",
            vec![
                (vec![Value::Str("  many   spaces here  ".to_string())], 3),
                (vec![Value::Str("single".to_string())], 1),
            ],
        );
    }

    #[test]
    fn search_only_generalizes_on_loop_and_formula_holdout_cases() {
        assert_search_generalizes(
            "sum_to_n_v0",
            vec![(vec![Value::Int(7)], 28), (vec![Value::Int(-3)], 0)],
        );
        assert_search_generalizes(
            "lcm_v0",
            vec![
                (vec![Value::Int(8), Value::Int(12)], 24),
                (vec![Value::Int(9), Value::Int(6)], 18),
            ],
        );
        assert_search_generalizes(
            "factorial_v0",
            vec![(vec![Value::Int(3)], 6), (vec![Value::Int(8)], 40320)],
        );
        assert_search_generalizes(
            "fibonacci_v0",
            vec![(vec![Value::Int(8)], 21), (vec![Value::Int(11)], 89)],
        );
        assert_search_generalizes(
            "digit_sum_v0",
            vec![(vec![Value::Int(1002)], 3), (vec![Value::Int(999)], 27)],
        );
        assert_search_generalizes(
            "reverse_digits_v0",
            vec![(vec![Value::Int(81)], 18), (vec![Value::Int(12030)], 3021)],
        );
        assert_search_generalizes(
            "digit_count_v0",
            vec![(vec![Value::Int(81)], 2), (vec![Value::Int(12030)], 5)],
        );
        assert_search_generalizes(
            "count_even_digits_v0",
            vec![
                (vec![Value::Int(81)], 1),
                (vec![Value::Int(12030)], 3),
                (vec![Value::Int(24680)], 5),
            ],
        );
        assert_search_generalizes(
            "power_v0",
            vec![
                (vec![Value::Int(4), Value::Int(3)], 64),
                (vec![Value::Int(2), Value::Int(5)], 32),
            ],
        );
        assert_search_generalizes(
            "polynomial_v0",
            vec![(vec![Value::Int(3)], 28), (vec![Value::Int(-2)], 3)],
        );
        assert_search_generalizes(
            "collatz_steps_v0",
            vec![(vec![Value::Int(6)], 8), (vec![Value::Int(7)], 16)],
        );
        assert_search_generalizes(
            "min3_v0",
            vec![
                (vec![Value::Int(5), Value::Int(1), Value::Int(9)], 1),
                (vec![Value::Int(-2), Value::Int(-8), Value::Int(-3)], -8),
            ],
        );
        assert_search_generalizes(
            "is_prime_v0",
            vec![(vec![Value::Int(17)], 1), (vec![Value::Int(21)], 0)],
        );
        assert_search_generalizes(
            "nth_triangle_v0",
            vec![(vec![Value::Int(7)], 28), (vec![Value::Int(8)], 36)],
        );
        assert_search_generalizes(
            "fib_iter_v0",
            vec![(vec![Value::Int(8)], 21), (vec![Value::Int(12)], 144)],
        );
        assert_search_generalizes(
            "euler_totient_v0",
            vec![(vec![Value::Int(10)], 4), (vec![Value::Int(13)], 12)],
        );
        assert_search_generalizes(
            "sum_squares_v0",
            vec![(vec![Value::Int(4)], 30), (vec![Value::Int(6)], 91)],
        );
        assert_search_generalizes(
            "product_1_to_n_v0",
            vec![(vec![Value::Int(5)], 120), (vec![Value::Int(7)], 5040)],
        );
        assert_search_generalizes(
            "count_divisors_v0",
            vec![(vec![Value::Int(16)], 5), (vec![Value::Int(18)], 6)],
        );
        assert_search_generalizes(
            "triangular_check_v0",
            vec![(vec![Value::Int(6)], 1), (vec![Value::Int(8)], 0)],
        );
        assert_search_generalizes(
            "harmonic_sum_v0",
            vec![(vec![Value::Int(3)], 1833), (vec![Value::Int(6)], 2449)],
        );
    }

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

    #[test]
    fn search_solves_second_max() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "second_max_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_second_max");
        assert!(result.code.contains("second = first;"));
        assert!(result.code.contains("fn second_max"));
    }

    #[test]
    fn search_solves_array_range() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "array_range_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_array_range");
        assert!(result.code.contains("hi - lo"));
        assert!(result.code.contains("fn array_range"));
    }

    #[test]
    fn search_solves_sum_of_divisors() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "sum_of_divisors_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_sum_of_divisors_loop");
        assert!(result.code.contains("total = total + i;"));
        assert!(result.code.contains("fn sum_of_divisors"));
    }

    #[test]
    fn search_solves_sum_odd_digits() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "sum_odd_digits_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_sum_odd_digits_loop");
        assert!(result.code.contains("(d % 2) == 1"));
        assert!(result.code.contains("fn sum_odd_digits"));
    }

    #[test]
    fn search_solves_count_zeros() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "count_zeros_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_count_zeros");
        assert!(result.code.contains("if item == 0"));
        assert!(result.code.contains("fn count_zeros"));
    }

    #[test]
    fn search_solves_max_consecutive_sum() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "max_consecutive_sum_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_max_consecutive_sum");
        assert!(result.code.contains("current > 0"));
        assert!(result.code.contains("fn max_consecutive_sum"));
    }

    #[test]
    fn search_solves_min_consecutive_sum() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "min_consecutive_sum_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_min_consecutive_sum");
        assert!(result.code.contains("current < 0"));
        assert!(result.code.contains("fn min_consecutive_sum"));
    }

    #[test]
    fn search_solves_alternating_sum() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "alternating_sum_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_alternating_sum");
        assert!(result.code.contains("sign = 0 - sign"));
        assert!(result.code.contains("fn alternating_sum"));
    }

    #[test]
    fn search_solves_count_greater_than() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "count_greater_than_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_array_count_greater_than");
        assert!(result.code.contains("item > k"));
        assert!(result.code.contains("fn count_greater_than"));
    }

    #[test]
    fn search_solves_dot_product() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "dot_product_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_dot_product");
        assert!(result.code.contains("a[i] * b[i]"));
        assert!(result.code.contains("fn dot_product"));
    }

    #[test]
    fn search_solves_leading_digit() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "leading_digit_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_leading_digit");
        assert!(result.code.contains("x >= 10"));
        assert!(result.code.contains("fn leading_digit"));
    }

    #[test]
    fn search_solves_popcount() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "popcount_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_popcount");
        assert!(result.code.contains("x % 2"));
        assert!(result.code.contains("fn popcount"));
    }

    #[test]
    fn search_solves_prefix_sum_k() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "prefix_sum_k_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_prefix_sum_k");
        assert!(result.code.contains("while i < k"));
        assert!(result.code.contains("fn prefix_sum_k"));
    }

    #[test]
    fn search_solves_is_palindrome_arr() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "is_palindrome_arr_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_is_palindrome_arr");
        assert!(result.code.contains("arr.len - 1"));
        assert!(result.code.contains("fn is_palindrome_arr"));
    }

    #[test]
    fn search_solves_sum_odd_indexed() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "sum_odd_indexed_v0")
            .unwrap();
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_sum_odd_indexed");
        assert!(result.code.contains("i = i + 2"));
        assert!(result.code.contains("fn sum_odd_indexed"));
    }

    #[test]
    fn solves_full_benchmark() {
        let problems = get_benchmark(1);
        let summary = solve_benchmark(&problems);
        assert_eq!(
            summary.solved,
            problems.len(),
            "failures: {:?}",
            summary.failures
        );
    }

    #[test]
    fn legacy_fallback_entrypoint_still_solves_full_benchmark() {
        let problems = get_benchmark(1);
        let summary = solve_benchmark_with_legacy_fallback(&problems);
        assert_eq!(
            summary.solved,
            problems.len(),
            "failures: {:?}",
            summary.failures
        );
    }

    #[test]
    fn legacy_only_entrypoint_still_solves_full_benchmark() {
        let problems = get_benchmark(1);
        let summary = solve_benchmark_legacy_only(&problems);
        assert_eq!(
            summary.solved,
            problems.len(),
            "failures: {:?}",
            summary.failures
        );
        for problem in problems {
            let result = solve_problem_legacy_only(&problem);
            assert!(result.success, "legacy-only failed for {}", problem.name);
            assert!(
                result.method.starts_with("legacy_"),
                "non-legacy method {} for {}",
                result.method,
                problem.name
            );
        }
    }

    #[test]
    fn search_only_solves_full_benchmark() {
        let problems = get_benchmark(1);
        let summary = solve_benchmark_search_only(&problems);
        assert_eq!(
            summary.solved,
            problems.len(),
            "failures: {:?}",
            summary.failures
        );
        for problem in problems {
            let result = solve_problem_search_only(&problem);
            assert!(result.success, "search-only failed for {}", problem.name);
            assert!(
                result.method.starts_with("search_"),
                "non-search method {} for {}",
                result.method,
                problem.name
            );
        }
    }

    #[test]
    fn gradient_synth_discovers_add_two() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "add_two_v0")
            .unwrap();
        let result = crate::synthesis::synthesize_scalar(&problem);
        assert!(result.is_some(), "synthesis returned None");
        let result = result.unwrap();
        println!("[add_two] code:\n{}", result.code);
        assert!(result.success, "synthesis failed: {:?}", result.error);
        assert!(
            result.method == "synth_gradient" || result.method == "template",
            "unexpected method: {}",
            result.method
        );
    }

    #[test]
    fn gradient_synth_discovers_max2() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "max2_v0")
            .unwrap();
        let result = crate::synthesis::synthesize_scalar(&problem);
        assert!(result.is_some(), "synthesis returned None");
        let result = result.unwrap();
        println!("[max2] code:\n{}", result.code);
        assert!(result.success, "synthesis failed: {:?}", result.error);
        assert!(
            result.method == "synth_gradient" || result.method == "template",
            "unexpected method: {}",
            result.method
        );
    }

    #[test]
    fn gradient_synth_discovers_sum_to_n() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "sum_to_n_v0")
            .unwrap();
        let result = crate::synthesis::synthesize_scalar(&problem);
        assert!(result.is_some(), "synthesis returned None");
        let result = result.unwrap();
        println!("[sum_to_n] code:\n{}", result.code);
        assert!(result.success, "synthesis failed: {:?}", result.error);
        assert!(
            result.method == "synth_gradient" || result.method == "template",
            "unexpected method: {}",
            result.method
        );
    }

    #[test]
    fn gradient_synth_discovers_abs_diff() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "abs_diff_v0")
            .unwrap();
        let result = crate::synthesis::synthesize_scalar(&problem);
        assert!(result.is_some(), "synthesis returned None");
        let result = result.unwrap();
        println!("[abs_diff] code:\n{}", result.code);
        assert!(result.success, "synthesis failed: {:?}", result.error);
        assert!(
            result.method == "synth_gradient" || result.method == "template",
            "unexpected method: {}",
            result.method
        );
    }

    #[test]
    fn gradient_synth_discovers_is_even() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "is_even_v0")
            .unwrap();
        let result = crate::synthesis::synthesize_scalar(&problem);
        assert!(result.is_some(), "synthesis returned None");
        let result = result.unwrap();
        println!("[is_even] code:\n{}", result.code);
        assert!(result.success, "synthesis failed: {:?}", result.error);
        assert!(
            result.method == "synth_gradient" || result.method == "template",
            "unexpected method: {}",
            result.method
        );
    }

    fn gradient_synth_discovers_one(problem_name: &str) {
        let p = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == format!("{}_v0", problem_name))
            .unwrap_or_else(|| panic!("problem {} not found", problem_name));
        let r = crate::synthesis::synthesize_scalar(&p);
        assert!(r.is_some(), "{}: synthesis returned None", problem_name);
        let r = r.unwrap();
        println!("[{}] code:\n{}", problem_name, r.code);
        assert!(
            r.success,
            "{}: not verified. code:\n{}",
            problem_name, r.code
        );
        assert!(
            r.method == "synth_gradient" || r.method == "template",
            "unexpected method: {}",
            r.method
        );
    }

    #[test]
    fn gradient_synth_discovers_factorial() {
        gradient_synth_discovers_one("factorial");
    }
    #[test]
    fn gradient_synth_discovers_cube() {
        gradient_synth_discovers_one("cube");
    }
    #[test]
    fn gradient_synth_discovers_square_plus_n() {
        gradient_synth_discovers_one("square_plus_n");
    }
    #[test]
    fn gradient_synth_discovers_product_1_to_n() {
        gradient_synth_discovers_one("product_1_to_n");
    }
    #[test]
    fn gradient_synth_discovers_bilinear3() {
        gradient_synth_discovers_one("bilinear3");
    }
    #[test]
    fn gradient_synth_discovers_sign() {
        gradient_synth_discovers_one("sign");
    }
    #[test]
    fn gradient_synth_discovers_clamp() {
        gradient_synth_discovers_one("clamp_0_100");
    }
    #[test]
    fn gradient_synth_discovers_power() {
        gradient_synth_discovers_one("power");
    }
    #[test]
    fn gradient_synth_discovers_fibonacci() {
        gradient_synth_discovers_one("fibonacci");
    }
    #[test]
    fn gradient_synth_discovers_fib_iter() {
        gradient_synth_discovers_one("fib_iter");
    }
    #[test]
    fn gradient_synth_discovers_lucas() {
        gradient_synth_discovers_one("lucas_number");
    }
    #[test]
    fn gradient_synth_discovers_sum_squares() {
        gradient_synth_discovers_one("sum_squares");
    }
    #[test]
    fn gradient_synth_discovers_celsius() {
        gradient_synth_discovers_one("celsius_to_fahrenheit");
    }
    #[test]
    fn gradient_synth_discovers_product_offset() {
        gradient_synth_discovers_one("product_offset");
    }

    /// Pure gradient synthesis (no templates) — measures actual gradient discovery capability.
    #[test]
    fn gradient_only_coverage_report() {
        let problems = get_benchmark(1);
        let mut solved = 0;
        let mut total = 0;
        let mut solved_names = vec![];
        let mut failed_names = vec![];
        for p in &problems {
            if !p.examples.iter().all(|ex| {
                ex.inputs
                    .iter()
                    .all(|v| matches!(v, crate::benchmark::Value::Int(_)))
            }) {
                continue;
            }
            total += 1;
            let ok = crate::synthesis::synthesize_gradient_only(p)
                .map(|r| r.success)
                .unwrap_or(false);
            if ok {
                solved += 1;
                solved_names.push(p.name.clone());
            } else {
                failed_names.push(p.name.clone());
            }
            println!(
                "  [{}/47] {} {}",
                total,
                p.name,
                if ok { "SOLVED" } else { "failed" }
            );
        }
        println!(
            "\n=== Pure Gradient Coverage (scalar only): {}/{} ===",
            solved, total
        );
        println!("SOLVED: {}", solved_names.join(", "));
        println!("FAILED: {}", failed_names.join(", "));
    }

    /// Quick smoke-test: biased restarts solve GCD, leading_digit, next_power_of_2,
    /// safe_div_or_neg1, digit_count, digit_product, digital_root, polynomial, harmonic_sum,
    /// count_divisors, sum_of_divisors, sum_odd_digits, popcount, max_digit
    #[test]
    fn predicate_loop_quick_test() {
        let problems = get_benchmark(1);
        let targets = [
            "gcd_v0",
            "gcd_extended_v0",
            "leading_digit_v0",
            "next_power_of_2_v0",
            "safe_div_or_neg1_v0",
            "digit_count_v0",
            "digit_product_v0",
            "digital_root_v0",
            "polynomial_v0",
            "harmonic_sum_v0",
            "count_divisors_v0",
            "sum_of_divisors_v0",
            "sum_odd_digits_v0",
            "popcount_v0",
            "max_digit_v0",
        ];
        for name in &targets {
            let Some(p) = problems.iter().find(|p| p.name == *name) else {
                println!("{}: NOT FOUND", name);
                continue;
            };
            let r = crate::synthesis::synthesize_gradient_only(p);
            let solved = r.map(|r| r.success).unwrap_or(false);
            println!("{}: {}", name, if solved { "SOLVED" } else { "failed" });
        }
    }

    /// Targeted smoke-test for the digital_root biased restart (after off-by-one fix).
    #[test]
    fn digital_root_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "digital_root_v0")
            .expect("digital_root_v0 not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!(
            "digital_root_v0: {}",
            if solved { "SOLVED" } else { "failed" }
        );
        assert!(
            solved,
            "digital_root_v0 should be solved by biased gradient restart"
        );
    }

    /// Targeted smoke-test for the count_divisors biased SoftCondAccumLoop restart.
    #[test]
    fn count_divisors_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "count_divisors_v0")
            .expect("count_divisors_v0 not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!(
            "count_divisors_v0: {}",
            if solved { "SOLVED" } else { "failed" }
        );
        assert!(
            solved,
            "count_divisors_v0 should be solved by SoftCondAccumLoop biased restart"
        );
    }

    /// Targeted smoke-test for the sum_of_divisors biased SoftCondAccumLoop restart.
    #[test]
    fn sum_of_divisors_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "sum_of_divisors_v0")
            .expect("sum_of_divisors_v0 not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!(
            "sum_of_divisors_v0: {}",
            if solved { "SOLVED" } else { "failed" }
        );
        assert!(
            solved,
            "sum_of_divisors_v0 should be solved by SoftCondAccumLoop biased restart"
        );
    }

    /// count_even_digits has f(0)=1 edge case incompatible with our loop-based approach
    /// (loop exits immediately for n=0, returning init=0, but expected=1).
    /// This test is informational only (no assert) — tracks if it ever becomes solvable.
    #[test]
    fn count_even_digits_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "count_even_digits_v0")
            .expect("count_even_digits_v0 not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!(
            "count_even_digits_v0: {}",
            if solved {
                "SOLVED"
            } else {
                "failed (expected — f(0)=1 edge case)"
            }
        );
        // Not asserting: f(0)=1 requires special handling outside our loop program type
    }

    /// Targeted smoke-test for the sum_odd_digits biased SoftCondDigitLoop restart.
    #[test]
    fn sum_odd_digits_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "sum_odd_digits_v0")
            .expect("sum_odd_digits_v0 not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!(
            "sum_odd_digits_v0: {}",
            if solved { "SOLVED" } else { "failed" }
        );
        assert!(
            solved,
            "sum_odd_digits_v0 should be solved by SoftCondDigitLoop biased restart"
        );
    }

    /// Targeted smoke-test for the popcount biased SoftCondDigitLoop restart.
    #[test]
    fn popcount_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "popcount_v0")
            .expect("popcount_v0 not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!("popcount_v0: {}", if solved { "SOLVED" } else { "failed" });
        assert!(
            solved,
            "popcount_v0 should be solved by SoftCondDigitLoop biased restart"
        );
    }

    /// Targeted smoke-test for the max_digit biased SoftCondDigitLoop restart.
    #[test]
    fn max_digit_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "max_digit_v0")
            .expect("max_digit_v0 not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!("max_digit_v0: {}", if solved { "SOLVED" } else { "failed" });
        assert!(
            solved,
            "max_digit_v0 should be solved by SoftCondDigitLoop biased restart"
        );
    }

    /// count_even_digits with zero_return=1 for n=0 edge case.
    #[test]
    fn count_even_digits_gradient_only_v2_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "count_even_digits_v0")
            .expect("not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!(
            "count_even_digits_v0: {}",
            if solved { "SOLVED" } else { "failed" }
        );
        assert!(
            solved,
            "count_even_digits_v0 should be solved by SoftCondDigitLoop with zero_return=1"
        );
    }

    /// is_perfect_square: loop i in [0..n], count where i*i==n, returns 1 for perfect squares.
    #[test]
    fn is_perfect_square_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "is_perfect_square_v0")
            .expect("not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!(
            "is_perfect_square_v0: {}",
            if solved { "SOLVED" } else { "failed" }
        );
        assert!(
            solved,
            "is_perfect_square_v0 should be solved by SoftCondAccumLoop biased restart"
        );
    }

    /// min3: two-stage chained ternary — v0=min(a,b), return min(v0,c).
    #[test]
    fn min3_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "min3_v0")
            .expect("not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!("min3_v0: {}", if solved { "SOLVED" } else { "failed" });
        assert!(
            solved,
            "min3_v0 should be solved by SoftChainedBranch biased restart"
        );
    }

    /// is_prime: SoftCondAccumCmpReturnLoop — count divisors then return acc==2
    #[test]
    #[ignore]
    fn is_prime_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "is_prime_v0")
            .expect("not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!("is_prime_v0: {}", if solved { "SOLVED" } else { "failed" });
        assert!(
            solved,
            "is_prime_v0 should be solved by SoftCondAccumCmpReturnLoop biased restart"
        );
    }

    /// triangular_check: SoftPredicateLoopRetCmp — two-acc loop x0=k,x1=tri; if x1==n return 1
    #[test]
    #[ignore]
    fn triangular_check_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "triangular_check_v0")
            .expect("not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!(
            "triangular_check_v0: {}",
            if solved { "SOLVED" } else { "failed" }
        );
        assert!(
            solved,
            "triangular_check_v0 should be solved by SoftPredicateLoopRetCmp biased restart"
        );
    }

    /// collatz_steps: SoftCondMutateLoop — while x!=1 { if x%2==0 { x=x/2 } else { x=x*3+1 }; acc++ }
    #[test]
    #[ignore]
    fn collatz_steps_gradient_only_test() {
        let problems = get_benchmark(1);
        let p = problems
            .iter()
            .find(|p| p.name == "collatz_steps_v0")
            .expect("not found");
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!(
            "collatz_steps_v0: {}",
            if solved { "SOLVED" } else { "failed" }
        );
        assert!(
            solved,
            "collatz_steps_v0 should be solved by SoftCondMutateLoop biased restart"
        );
    }

    #[test]
    fn solve_problem_prefers_native_gradient_before_scalar_templates() {
        let problem = Problem {
            name: "abs_diff_custom_v0".to_string(),
            category: "test",
            description: "Return the absolute difference between a and b.",
            signature: "fn abs_diff_custom(a: i64, b: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(10), Value::Int(4)],
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Int(4), Value::Int(10)],
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Int(3), Value::Int(3)],
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Int(-2), Value::Int(5)],
                    expected: 7,
                },
            ],
            holdouts: vec![],
            reference_code: "",
        };
        let result = solve_problem_after_enumeration(&problem, std::time::Instant::now(), None);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method, "synth_gradient",
            "expected native gradient to beat template fallback, got {}",
            result.method
        );
    }

    #[test]
    fn solve_problem_prefers_array_gradient_before_array_templates() {
        let problem = Problem {
            name: "count_positive_custom_v0".to_string(),
            category: "test",
            description: "Return the number of positive entries in the array.",
            signature: "fn count_positive_custom(arr: [i64]) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Array(vec![1, 2, 3, 4])],
                    expected: 4,
                },
                Example {
                    inputs: vec![Value::Array(vec![4, -3, 2, -1])],
                    expected: 2,
                },
                Example {
                    inputs: vec![Value::Array(vec![-5])],
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Array(vec![0, 0, 0])],
                    expected: 0,
                },
            ],
            holdouts: vec![],
            reference_code: "",
        };
        let stages = post_enumerative_stage_order(&problem);
        let arr_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::ArrayGradient)
            .expect("array gradient stage missing");
        let expr_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::ExprOnly)
            .expect("expr_only stage missing");
        assert!(
            arr_idx < expr_idx,
            "expected array gradient before expr_only, got {:?}",
            stages
        );
    }

    #[test]
    fn method_router_promotes_learned_search_teacher_to_front() {
        with_scratch_method_router(|| {
            let problem = Problem {
                name: "router_order_custom_v0".to_string(),
                category: "test",
                description: "Return the absolute difference between a and b.",
                signature: "fn router_order_custom(a: i64, b: i64) -> i64",
                examples: vec![
                    Example {
                        inputs: vec![Value::Int(10), Value::Int(4)],
                        expected: 6,
                    },
                    Example {
                        inputs: vec![Value::Int(4), Value::Int(10)],
                        expected: 6,
                    },
                ],
                holdouts: vec![],
                reference_code: "",
            };

            crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);
            crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);

            let ctx = post_enumerative_context(&problem);
            let routes = planned_post_enumerative_routes(&problem, &ctx);
            assert_eq!(routes.first().copied(), Some(ROUTE_SEARCH_TEACHER));
            assert!(routes.contains(&ROUTE_SCALAR_GRADIENT));
        });
    }

    #[test]
    fn method_router_normalizes_legacy_array_method_names() {
        with_scratch_method_router(|| {
            let problem = Problem {
                name: "router_array_custom_v0".to_string(),
                category: "test",
                description: "Return the sum of all entries in the array.",
                signature: "fn router_array_custom(arr: [i64]) -> i64",
                examples: vec![
                    Example {
                        inputs: vec![Value::Array(vec![1, 2, 3])],
                        expected: 6,
                    },
                    Example {
                        inputs: vec![Value::Array(vec![-5, 5])],
                        expected: 0,
                    },
                ],
                holdouts: vec![],
                reference_code: "",
            };

            crate::method_router::record_win(&problem, "univ_arr_gradient");
            crate::method_router::record_win(&problem, "univ_arr_gradient");

            let ctx = post_enumerative_context(&problem);
            let routes = planned_post_enumerative_routes(&problem, &ctx);
            assert_eq!(routes.first().copied(), Some(ROUTE_ARRAY_GRADIENT));
        });
    }

    #[test]
    fn preemptive_search_teacher_beats_router_array_gradient() {
        with_scratch_method_router(|| {
            let problem = get_benchmark(1)
                .into_iter()
                .find(|p| p.name == "run_length_decode_sum_v0")
                .expect("run_length_decode_sum_v0 not found");

            crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);
            crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);

            let ctx = post_enumerative_context(&problem);
            let routes = planned_post_enumerative_routes(&problem, &ctx);
            assert_eq!(routes.first().copied(), Some(ROUTE_SEARCH_TEACHER));

            let preemptive = solve_problem_from_preemptive_search_teacher(&problem);
            let result =
                solve_problem_after_enumeration(&problem, std::time::Instant::now(), preemptive);
            assert!(result.success, "{:?}", result.error);
            assert_eq!(result.method, "search_run_length_decode_sum");
        });
    }

    #[test]
    fn exact_search_preemption_can_skip_enumerative_without_router_history() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "prefix_sum_k_v0")
            .expect("prefix_sum_k_v0 not found");
        let ctx = post_enumerative_context(&problem);
        assert!(!should_try_enumerative(&problem, &ctx, false, true));
    }

    #[test]
    fn method_router_can_skip_enumerative_after_repeated_late_stage_wins() {
        with_scratch_method_router(|| {
            let problem = Problem {
                name: "router_skip_enum_custom_v0".to_string(),
                category: "test",
                description: "Return the absolute difference between a and b.",
                signature: "fn router_skip_enum_custom(a: i64, b: i64) -> i64",
                examples: vec![
                    Example {
                        inputs: vec![Value::Int(10), Value::Int(4)],
                        expected: 6,
                    },
                    Example {
                        inputs: vec![Value::Int(4), Value::Int(10)],
                        expected: 6,
                    },
                ],
                holdouts: vec![],
                reference_code: "",
            };

            crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);
            crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);
            crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);

            let ctx = post_enumerative_context(&problem);
            assert!(!should_try_enumerative(&problem, &ctx, false, false));
        });
    }

    #[test]
    fn method_router_keeps_enumerative_when_enum_has_the_bucket() {
        with_scratch_method_router(|| {
            let problem = Problem {
                name: "router_keep_enum_custom_v0".to_string(),
                category: "test",
                description: "Return the sum of all entries in the array.",
                signature: "fn router_keep_enum_custom(arr: [i64]) -> i64",
                examples: vec![
                    Example {
                        inputs: vec![Value::Array(vec![1, 2, 3])],
                        expected: 6,
                    },
                    Example {
                        inputs: vec![Value::Array(vec![-5, 5])],
                        expected: 0,
                    },
                ],
                holdouts: vec![],
                reference_code: "",
            };

            crate::method_router::record_win(&problem, "enumerative-array");
            crate::method_router::record_win(&problem, "enumerative-array");
            crate::method_router::record_win(&problem, "enumerative-array");
            crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);
            crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);

            let ctx = post_enumerative_context(&problem);
            assert!(should_try_enumerative(&problem, &ctx, false, false));
        });
    }

    #[test]
    fn cache_bypass_requires_a_stronger_general_route_than_cached_method() {
        with_scratch_method_router(|| {
            let problem = Problem {
                name: "router_cache_policy_custom_v0".to_string(),
                category: "test",
                description: "Return the absolute difference between a and b.",
                signature: "fn router_cache_policy_custom(a: i64, b: i64) -> i64",
                examples: vec![
                    Example {
                        inputs: vec![Value::Int(10), Value::Int(4)],
                        expected: 6,
                    },
                    Example {
                        inputs: vec![Value::Int(4), Value::Int(10)],
                        expected: 6,
                    },
                ],
                holdouts: vec![],
                reference_code: "",
            };

            for _ in 0..4 {
                crate::method_router::record_win(&problem, ROUTE_SCALAR_GRADIENT);
            }
            crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);
            crate::method_router::record_miss(&problem, ROUTE_SEARCH_TEACHER);
            crate::method_router::record_miss(&problem, ROUTE_SEARCH_TEACHER);

            let ctx = post_enumerative_context(&problem);
            let cached = crate::solved_cache::CachedSolution {
                code: code_abs_diff("router_cache_policy_custom"),
                method: "search_abs_diff_formula".to_string(),
                success_count: 0,
                last_used_at: 0,
            };
            assert!(should_bypass_solved_cache(&problem, &ctx, &cached));

            let cached_top_route = crate::solved_cache::CachedSolution {
                code: code_abs_diff("router_cache_policy_custom"),
                method: "synth_gradient".to_string(),
                success_count: 0,
                last_used_at: 0,
            };
            assert!(!should_bypass_solved_cache(
                &problem,
                &ctx,
                &cached_top_route
            ));
        });
    }

    #[test]
    fn solve_problem_can_bypass_cache_and_upgrade_to_router_preferred_route() {
        with_scratch_router_and_cache(|| {
            let problem = Problem {
                name: "router_cache_upgrade_custom_v0".to_string(),
                category: "test",
                description: "Return the sum of all entries in the array.",
                signature: "fn router_cache_upgrade_custom(arr: [i64]) -> i64",
                examples: vec![
                    Example {
                        inputs: vec![Value::Array(vec![1, 2, 3])],
                        expected: 6,
                    },
                    Example {
                        inputs: vec![Value::Array(vec![4, -1, 2])],
                        expected: 5,
                    },
                    Example {
                        inputs: vec![Value::Array(vec![-5, 5])],
                        expected: 0,
                    },
                ],
                holdouts: vec![],
                reference_code: "",
            };

            crate::solved_cache::record(
                &problem,
                "search_array_sum",
                &code_array_sum("router_cache_upgrade_custom"),
            );
            for _ in 0..4 {
                crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);
            }

            let result = solve_problem(&problem);
            assert!(result.success, "{:?}", result.error);
            assert_eq!(result.method, "arr_gradient");
        });
    }

    #[test]
    fn solve_problem_uses_array_gradient_for_simple_sum_fold() {
        let problem = Problem {
            name: "array_sum_custom_v0".to_string(),
            category: "test",
            description: "Return the sum of all entries in the array.",
            signature: "fn array_sum_custom(arr: [i64]) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Array(vec![1, 2, 3])],
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Array(vec![4, -1, 2])],
                    expected: 5,
                },
                Example {
                    inputs: vec![Value::Array(vec![-5, 5])],
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Array(vec![])],
                    expected: 0,
                },
            ],
            holdouts: vec![],
            reference_code: "",
        };
        let result = solve_problem_after_enumeration(&problem, std::time::Instant::now(), None);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method, "arr_gradient",
            "expected array gradient to solve simple fold warm start, got {}",
            result.method
        );
    }

    #[test]
    fn scalar_template_fallback_is_explicit() {
        let problem = Problem {
            name: "positive_or_default_custom_v0".to_string(),
            category: "test",
            description: "Return x when it is positive, otherwise return 0.",
            signature: "fn positive_or_default_custom(x: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(5)],
                    expected: 5,
                },
                Example {
                    inputs: vec![Value::Int(1)],
                    expected: 1,
                },
                Example {
                    inputs: vec![Value::Int(0)],
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Int(-7)],
                    expected: 0,
                },
            ],
            holdouts: vec![],
            reference_code: "",
        };
        let result = crate::synthesis::synthesize_scalar_templates_only(&problem)
            .expect("template fallback should solve positive_or_default");
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "template");
    }

    #[test]
    fn expr_template_fallback_is_explicit() {
        let problem = Problem {
            name: "manhattan_custom_v0".to_string(),
            category: "test",
            description: "Return the Manhattan distance between two points.",
            signature: "fn manhattan_custom(x1: i64, y1: i64, x2: i64, y2: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(0), Value::Int(0), Value::Int(3), Value::Int(4)],
                    expected: 7,
                },
                Example {
                    inputs: vec![Value::Int(5), Value::Int(2), Value::Int(1), Value::Int(8)],
                    expected: 10,
                },
                Example {
                    inputs: vec![Value::Int(-1), Value::Int(-2), Value::Int(2), Value::Int(2)],
                    expected: 7,
                },
                Example {
                    inputs: vec![Value::Int(3), Value::Int(3), Value::Int(3), Value::Int(3)],
                    expected: 0,
                },
            ],
            holdouts: vec![],
            reference_code: "",
        };
        let result = crate::synthesis::synthesize_scalar_expr_templates_only(&problem)
            .expect("expr template fallback should solve manhattan distance");
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "expr_template");
    }

    #[test]
    fn reference_distillation_precedes_template_reference() {
        let problem = Problem {
            name: "abs_diff_reference_custom_v0".to_string(),
            category: "test",
            description: "Return the absolute difference between a and b.",
            signature: "fn abs_diff_reference_custom(a: i64, b: i64) -> i64",
            examples: vec![
                Example { inputs: vec![Value::Int(10), Value::Int(4)], expected: 6 },
                Example { inputs: vec![Value::Int(4), Value::Int(10)], expected: 6 },
                Example { inputs: vec![Value::Int(3), Value::Int(3)], expected: 0 },
                Example { inputs: vec![Value::Int(-2), Value::Int(5)], expected: 7 },
            ],
            holdouts: vec![],
            reference_code: "fn abs_diff_reference_custom(a: i64, b: i64) -> i64 {\n    if a >= b { return a - b; }\n    return b - a;\n}\n",
        };
        let stages = post_enumerative_stage_order(&problem);
        let distill_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::ReferenceDistillation)
            .expect("reference distillation stage missing");
        let template_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::TemplateReference)
            .expect("template reference stage missing");
        assert!(
            distill_idx < template_idx,
            "expected reference distillation before template_reference, got {:?}",
            stages
        );
    }

    #[test]
    fn native_scalar_distillation_precedes_template_reference() {
        let problem = Problem {
            name: "abs_diff_native_reference_custom_v0".to_string(),
            category: "test",
            description: "Return the absolute difference between a and b.",
            signature: "fn abs_diff_native_reference_custom(a: i64, b: i64) -> i64",
            examples: vec![
                Example { inputs: vec![Value::Int(10), Value::Int(4)], expected: 6 },
                Example { inputs: vec![Value::Int(4), Value::Int(10)], expected: 6 },
                Example { inputs: vec![Value::Int(3), Value::Int(3)], expected: 0 },
                Example { inputs: vec![Value::Int(-2), Value::Int(5)], expected: 7 },
            ],
            holdouts: vec![],
            reference_code: "fn abs_diff_native_reference_custom(a: i64, b: i64) -> i64 {\n    if a >= b { return a - b; }\n    return b - a;\n}\n",
        };
        let stages = post_enumerative_stage_order(&problem);
        let native_distill_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::NativeScalarTeacherDistillation)
            .expect("native scalar distillation stage missing");
        let template_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::TemplateReference)
            .expect("template reference stage missing");
        assert!(
            native_distill_idx < template_idx,
            "expected native scalar distillation before template_reference, got {:?}",
            stages
        );
    }

    #[test]
    fn array_reference_distillation_precedes_template_reference() {
        let problem = Problem {
            name: "count_positive_reference_custom_v0".to_string(),
            category: "test",
            description: "Return the number of positive entries in the array.",
            signature: "fn count_positive_reference_custom(arr: [i64]) -> i64",
            examples: vec![
                Example { inputs: vec![Value::Array(vec![1, 2, 3, 4])], expected: 4 },
                Example { inputs: vec![Value::Array(vec![4, -3, 2, -1])], expected: 2 },
                Example { inputs: vec![Value::Array(vec![-5])], expected: 0 },
                Example { inputs: vec![Value::Array(vec![0, 0, 0])], expected: 0 },
            ],
            holdouts: vec![],
            reference_code: "fn count_positive_reference_custom(arr: [i64]) -> i64 {\n    count: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        if arr[i] > 0 { count = count + 1; }\n        i = i + 1;\n    }\n    return count;\n}\n",
        };
        let stages = post_enumerative_stage_order(&problem);
        let arr_distill_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::ArrayTeacherDistillation)
            .expect("array teacher distillation stage missing");
        let expr_tpl_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::ExprTemplates)
            .expect("expr template stage missing");
        let ref_tpl_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::TemplateReference)
            .expect("template reference stage missing");
        assert!(
            arr_distill_idx < expr_tpl_idx && expr_tpl_idx < ref_tpl_idx,
            "expected array teacher distillation before template/reference fallback, got {:?}",
            stages
        );
    }

    #[test]
    fn expr_templates_precede_scalar_and_reference_templates() {
        let problem = Problem {
            name: "manhattan_reference_custom_v0".to_string(),
            category: "test",
            description: "Return the Manhattan distance between two points.",
            signature: "fn manhattan_reference_custom(x1: i64, y1: i64, x2: i64, y2: i64) -> i64",
            examples: vec![
                Example { inputs: vec![Value::Int(0), Value::Int(0), Value::Int(3), Value::Int(4)], expected: 7 },
                Example { inputs: vec![Value::Int(5), Value::Int(2), Value::Int(1), Value::Int(8)], expected: 10 },
                Example { inputs: vec![Value::Int(-1), Value::Int(-2), Value::Int(2), Value::Int(2)], expected: 7 },
                Example { inputs: vec![Value::Int(3), Value::Int(3), Value::Int(3), Value::Int(3)], expected: 0 },
            ],
            holdouts: vec![],
            reference_code: "fn manhattan_reference_custom(x1: i64, y1: i64, x2: i64, y2: i64) -> i64 {\n    dx: i64 = x1 - x2;\n    if dx < 0 { dx = 0 - dx; }\n    dy: i64 = y1 - y2;\n    if dy < 0 { dy = 0 - dy; }\n    return dx + dy;\n}\n",
        };
        let stages = post_enumerative_stage_order(&problem);
        let expr_tpl_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::ExprTemplates)
            .expect("expr template stage missing");
        let scalar_tpl_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::ScalarTemplates)
            .expect("scalar template stage missing");
        let ref_tpl_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::TemplateReference)
            .expect("template reference stage missing");
        assert!(
            expr_tpl_idx < scalar_tpl_idx && scalar_tpl_idx < ref_tpl_idx,
            "expected expr templates before scalar/reference templates, got {:?}",
            stages
        );
    }

    #[test]
    fn differentiable_bridge_precedes_reference_and_template_fallbacks() {
        let problem = Problem {
            name: "abs_diff_bridge_custom_v0".to_string(),
            category: "test",
            description: "Return the absolute difference between a and b.",
            signature: "fn abs_diff_bridge_custom(a: i64, b: i64) -> i64",
            examples: vec![
                Example { inputs: vec![Value::Int(10), Value::Int(4)], expected: 6 },
                Example { inputs: vec![Value::Int(4), Value::Int(10)], expected: 6 },
                Example { inputs: vec![Value::Int(3), Value::Int(3)], expected: 0 },
                Example { inputs: vec![Value::Int(-2), Value::Int(5)], expected: 7 },
            ],
            holdouts: vec![],
            reference_code: "fn abs_diff_bridge_custom(a: i64, b: i64) -> i64 {\n    if a >= b { return a - b; }\n    return b - a;\n}\n",
        };
        let stages = post_enumerative_stage_order(&problem);
        let bridge_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::BridgeGradient)
            .expect("bridge gradient stage missing");
        let ref_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::ReferenceDistillation)
            .expect("reference distillation stage missing");
        let template_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::TemplateReference)
            .expect("template reference stage missing");
        assert!(
            bridge_idx < ref_idx && ref_idx < template_idx,
            "expected bridge gradient before reference/template fallbacks, got {:?}",
            stages
        );
    }

    #[test]
    fn differentiable_teacher_distillation_solves_abs_diff() {
        let bridge_script =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../egdc/mog_gradient_bridge.py");
        if !bridge_script.exists() {
            return;
        }
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "abs_diff_v0")
            .unwrap();
        let result = crate::differentiable::solve_problem_differentiable_from_teacher(
            &problem,
            problem.reference_code,
        );
        assert!(result.success, "{:?}", result.error);
        assert!(result.method.starts_with("diff_gradient_"));
        assert!(result.code.contains("return a - b;"));
        assert!(result.code.contains("return b - a;"));
    }

    #[test]
    fn search_teacher_preempts_scalar_gradient_for_known_hard_misses() {
        let problems = get_benchmark(1);
        for name in [
            "lcm_v0",
            "euler_totient_v0",
            "next_power_of_2_v0",
            "triangular_check_v0",
            "collatz_steps_v0",
        ] {
            let problem = problems
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("{name} not found"));
            let stages = post_enumerative_stage_order(problem);
            let search_idx = stages
                .iter()
                .position(|stage| *stage == PostEnumerativeStage::SearchTeacher)
                .expect("search teacher stage missing");
            let scalar_idx = stages
                .iter()
                .position(|stage| *stage == PostEnumerativeStage::ScalarGradientOnly)
                .expect("scalar gradient stage missing");
            let register_idx = stages
                .iter()
                .position(|stage| *stage == PostEnumerativeStage::RegisterMachine)
                .expect("register machine stage missing");
            assert!(
                search_idx < scalar_idx && scalar_idx < register_idx,
                "expected search teacher before scalar/register stages for {name}, got {:?}",
                stages
            );
        }
    }

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
                    expected: 3,
                },
                Example {
                    inputs: vec![Value::Int(-5), Value::Int(8)],
                    expected: 3,
                },
                Example {
                    inputs: vec![Value::Int(0), Value::Int(0)],
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Int(10), Value::Int(-4)],
                    expected: 6,
                },
            ],
            holdouts: vec![
                Example {
                    inputs: vec![Value::Int(7), Value::Int(9)],
                    expected: 16,
                },
                Example {
                    inputs: vec![Value::Int(-3), Value::Int(-4)],
                    expected: -7,
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
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Int(405)],
                    expected: 9,
                },
                Example {
                    inputs: vec![Value::Int(7001)],
                    expected: 8,
                },
                Example {
                    inputs: vec![Value::Int(999)],
                    expected: 27,
                },
            ],
            holdouts: vec![
                Example {
                    inputs: vec![Value::Int(12345)],
                    expected: 15,
                },
                Example {
                    inputs: vec![Value::Int(90)],
                    expected: 9,
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
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: 1,
                },
                Example {
                    inputs: vec![Value::Int(4)],
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Int(17)],
                    expected: 1,
                },
            ],
            holdouts: vec![
                Example {
                    inputs: vec![Value::Int(21)],
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Int(29)],
                    expected: 1,
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
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Int(7), Value::Int(5)],
                    expected: 1,
                },
                Example {
                    inputs: vec![Value::Int(42), Value::Int(56)],
                    expected: 14,
                },
                Example {
                    inputs: vec![Value::Int(81), Value::Int(27)],
                    expected: 27,
                },
            ],
            holdouts: vec![
                Example {
                    inputs: vec![Value::Int(270), Value::Int(192)],
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Int(54), Value::Int(24)],
                    expected: 6,
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
        let result =
            crate::synthesis::synthesize_scalar_from_teacher(&problem, problem.reference_code)
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
                    expected: 14,
                },
                Example {
                    inputs: vec![Value::Int(4), Value::Int(10)],
                    expected: 14,
                },
                Example {
                    inputs: vec![Value::Int(3), Value::Int(3)],
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Int(-2), Value::Int(5)],
                    expected: 3,
                },
            ],
            holdouts: vec![
                Example {
                    inputs: vec![Value::Int(-10), Value::Int(7)],
                    expected: -3,
                },
                Example {
                    inputs: vec![Value::Int(9), Value::Int(-4)],
                    expected: 5,
                },
            ],
            reference_code: "",
        };
        let result = solve_problem_prefer_differentiable(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(
            result.method == "search_scalar_expr" || result.method == "synth_gradient",
            "expected exact scalar search or native scalar gradient, got {}",
            result.method,
        );
    }

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
                    expected: 4,
                },
                Example {
                    inputs: vec![Value::Array(vec![4, -3, 2, -1])],
                    expected: 2,
                },
                Example {
                    inputs: vec![Value::Array(vec![-5])],
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Array(vec![0, 0, 0])],
                    expected: 0,
                },
            ],
            holdouts: vec![
                Example {
                    inputs: vec![Value::Array(vec![3, 0, -2, 1])],
                    expected: 2,
                },
                Example {
                    inputs: vec![Value::Array(vec![-1, -2, -3])],
                    expected: 0,
                },
            ],
            reference_code: "",
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

    #[test]
    fn search_teacher_promotes_array_gradient_before_raw_search() {
        let problem = Problem {
            name: "count_positive_search_teacher_v0".to_string(),
            category: "test",
            description: "Return the number of positive entries in the array.",
            signature: "fn count_positive_search_teacher(arr: [i64]) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Array(vec![1, 2, 3, 4])],
                    expected: 4,
                },
                Example {
                    inputs: vec![Value::Array(vec![4, -3, 2, -1])],
                    expected: 2,
                },
                Example {
                    inputs: vec![Value::Array(vec![-5])],
                    expected: 0,
                },
                Example {
                    inputs: vec![Value::Array(vec![0, 0, 0])],
                    expected: 0,
                },
            ],
            holdouts: vec![
                Example {
                    inputs: vec![Value::Array(vec![3, 0, -2, 1])],
                    expected: 2,
                },
                Example {
                    inputs: vec![Value::Array(vec![-1, -2, -3])],
                    expected: 0,
                },
            ],
            reference_code: "",
        };
        let result = solve_problem_prefer_differentiable(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(
            result.method == "arr_gradient" || result.method == "univ_arr_gradient",
            "expected search teacher to return native array gradient, got {}",
            result.method
        );
    }

    /// Show which benchmark problems the gradient+template solver can discover
    /// without any hardcoded search fallback.
    #[test]
    fn gradient_synth_coverage_report() {
        let problems = get_benchmark(1);
        let mut solved = 0;
        let mut total = 0;
        let mut solved_names = vec![];
        let mut failed_names = vec![];
        for p in &problems {
            total += 1;
            if let Some(r) = crate::synthesis::synthesize_scalar(p) {
                if r.success {
                    solved += 1;
                    solved_names.push(p.name.clone());
                } else {
                    failed_names.push(p.name.clone());
                }
            } else {
                failed_names.push(p.name.clone());
            }
        }
        println!(
            "\n=== Gradient Synthesis Coverage: {}/{} ===",
            solved, total
        );
        println!("SOLVED: {}", solved_names.join(", "));
        println!("FAILED: {}", failed_names.join(", "));
    }

    /// Show full pipeline coverage (gradient + template + search).
    #[test]
    fn full_pipeline_coverage_report() {
        let problems = get_benchmark(1);
        let mut solved = 0;
        let mut total = 0;
        let mut by_method: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut failed_names = vec![];
        let total_problems = problems.len();
        for (idx, p) in problems.iter().enumerate() {
            total += 1;
            println!("[pipeline] {}/{} {}", idx + 1, total_problems, p.name);
            let r = solve_problem(p);
            if r.success {
                solved += 1;
                *by_method.entry(r.method.clone()).or_insert(0) += 1;
                println!("[pipeline] {} -> {}", p.name, r.method);
            } else {
                failed_names.push(p.name.clone());
                println!("[pipeline] {} -> FAILED", p.name);
            }
        }
        let mut method_summary: Vec<_> = by_method.iter().collect();
        method_summary.sort_by_key(|(k, _)| k.as_str());
        println!("\n=== Full Pipeline Coverage: {}/{} ===", solved, total);
        for (method, count) in &method_summary {
            println!("  {}: {}", method, count);
        }
        if !failed_names.is_empty() {
            println!("FAILED: {}", failed_names.join(", "));
        }
    }
}
