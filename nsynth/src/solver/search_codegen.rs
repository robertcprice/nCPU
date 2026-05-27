use crate::runtime::verify_problem_code_strict;

use super::*;

pub(super) fn code_power_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(a: i64, b: i64) -> i64 {{\n    acc: i64 = 1;\n    i: i64 = 0;\n    while i < b {{\n        acc = acc * a;\n        i = i + 1;\n    }}\n    return acc;\n}}\n"
    )
}

pub(super) fn code_digit_sum_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{\n        x = 0 - x;\n    }}\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = acc + (x % 10);\n        x = x / 10;\n    }}\n    return acc;\n}}\n"
    )
}

pub(super) fn code_reverse_digits_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{\n        x = 0 - x;\n    }}\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = (acc * 10) + (x % 10);\n        x = x / 10;\n    }}\n    return acc;\n}}\n"
    )
}

pub(super) fn code_digit_count_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{\n        x = 0 - x;\n    }}\n    if x == 0 {{\n        return 1;\n    }}\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = acc + 1;\n        x = x / 10;\n    }}\n    return acc;\n}}\n"
    )
}

pub(super) fn code_count_even_digits_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{\n        x = 0 - x;\n    }}\n    if x == 0 {{\n        return 1;\n    }}\n    acc: i64 = 0;\n    while x > 0 {{\n        if ((x % 10) % 2) == 0 {{\n            acc = acc + 1;\n        }}\n        x = x / 10;\n    }}\n    return acc;\n}}\n"
    )
}

pub(super) fn code_fib_iter_loop_search(fn_name: &str) -> String {
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    if n == 0 {{ return 0; }}\n    if n == 1 {{ return 1; }}\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= n {{\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }}\n    return b;\n}}\n"
    )
}

pub(super) fn code_quadratic_search(fn_name: &str, a: i64, b: i64, c: i64) -> String {
    format!("fn {fn_name}(x: i64) -> i64 {{\n    return ({a} * x * x) + ({b} * x) + {c};\n}}\n")
}

pub(super) fn code_contains_literal_search(fn_name: &str, literal: &str) -> String {
    let literal = literal.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "fn {fn_name}(s: string) -> i64 {{\n    if s.contains(\"{literal}\") {{\n        return 1;\n    }}\n    return 0;\n}}\n"
    )
}

pub(super) fn code_starts_with_literal_search(fn_name: &str, literal: &str) -> String {
    let literal = literal.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "fn {fn_name}(s: string) -> i64 {{\n    if s.starts_with(\"{literal}\") {{\n        return 1;\n    }}\n    return 0;\n}}\n"
    )
}

pub(super) fn verified_result(
    problem: &Problem,
    code: String,
    method: &str,
) -> Option<SolveResult> {
    verify_problem_code_strict(problem, &code).ok()?;
    Some(SolveResult {
        success: true,
        code,
        method: method.to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    })
}

pub(super) fn code_abs_diff(fn_name: &str) -> String {
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

pub(super) fn code_max2(fn_name: &str) -> String {
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

pub(super) fn code_clamp(fn_name: &str) -> String {
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

pub(super) fn code_sign(fn_name: &str) -> String {
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

pub(super) fn code_combat_resolve(fn_name: &str) -> String {
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

pub(super) fn code_score_tracker(fn_name: &str) -> String {
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

pub(super) fn code_vending_change(fn_name: &str) -> String {
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

pub(super) fn code_turn_order_rotate(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(current: i64, num_players: i64) -> i64 {
    return (current + 1) % num_players;
}
"#,
        fn_name,
    )
}

pub(super) fn code_grid_bounds_check(fn_name: &str) -> String {
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

pub(super) fn code_simulate_gravity(fn_name: &str) -> String {
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

pub(super) fn code_gcd(fn_name: &str) -> String {
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

pub(super) fn code_lcm(fn_name: &str) -> String {
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

pub(super) fn code_array_sum(fn_name: &str) -> String {
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

pub(super) fn code_array_max(fn_name: &str) -> String {
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

pub(super) fn code_count_occurrences(fn_name: &str) -> String {
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

pub(super) fn code_trimmed_len(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(s: string) -> i64 {
    t := s.trim();
    return t.len;
}
"#,
        fn_name,
    )
}

pub(super) fn code_vowel_count(fn_name: &str) -> String {
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

pub(super) fn code_point_sum(fn_name: &str) -> String {
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

pub(super) fn code_safe_div_or_neg1(fn_name: &str) -> String {
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

pub(super) fn code_positive_or_default(fn_name: &str) -> String {
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

pub(super) fn code_closure_map_sum(fn_name: &str) -> String {
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

pub(super) fn code_count_positive(fn_name: &str) -> String {
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

pub(super) fn code_is_even(fn_name: &str) -> String {
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

pub(super) fn code_rectangle_area(fn_name: &str) -> String {
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

pub(super) fn code_collatz_steps(fn_name: &str) -> String {
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

pub(super) fn code_min3(fn_name: &str) -> String {
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

pub(super) fn code_is_prime(fn_name: &str) -> String {
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

pub(super) fn code_palindrome_check(fn_name: &str) -> String {
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

pub(super) fn code_count_words(fn_name: &str) -> String {
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

pub(super) fn code_euler_totient(fn_name: &str) -> String {
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

pub(super) fn code_count_divisors(fn_name: &str) -> String {
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

pub(super) fn code_triangular_check(fn_name: &str) -> String {
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

pub(super) fn code_max_pair_diff(fn_name: &str) -> String {
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

pub(super) fn code_sum_negatives(fn_name: &str) -> String {
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

pub(super) fn code_harmonic_sum(fn_name: &str) -> String {
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

pub(super) fn code_second_max(fn_name: &str) -> String {
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

pub(super) fn code_array_range(fn_name: &str) -> String {
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

pub(super) fn code_run_length_decode_sum(fn_name: &str) -> String {
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

pub(super) fn code_count_adjacent_diff(fn_name: &str) -> String {
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

pub(super) fn code_sum_of_divisors(fn_name: &str) -> String {
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

pub(super) fn code_sum_odd_digits(fn_name: &str) -> String {
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

pub(super) fn code_count_greater_than(fn_name: &str) -> String {
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

pub(super) fn code_prefix_sum_k(fn_name: &str) -> String {
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

pub(super) fn code_digit_product(fn_name: &str) -> String {
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

pub(super) fn code_max_digit(fn_name: &str) -> String {
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
