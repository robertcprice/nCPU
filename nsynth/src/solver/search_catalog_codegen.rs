use super::*;

pub(super) fn code_arr_sum_squares(fn_name: &str) -> String {
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

pub(super) fn code_min_element(fn_name: &str) -> String {
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

pub(super) fn code_sum_absolute(fn_name: &str) -> String {
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

pub(super) fn code_count_evens(fn_name: &str) -> String {
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

pub(super) fn code_sum_positives(fn_name: &str) -> String {
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

pub(super) fn code_sum_at_even_indices(fn_name: &str) -> String {
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

pub(super) fn code_kth_from_end(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64], k: i64) -> i64 {
    return arr[arr.len - k];
}
"#,
        fn_name,
    )
}

pub(super) fn code_max_abs(fn_name: &str) -> String {
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

pub(super) fn code_alternating_sum(fn_name: &str) -> String {
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

pub(super) fn code_dot_product(fn_name: &str) -> String {
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

pub(super) fn code_leading_digit(fn_name: &str) -> String {
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

pub(super) fn code_popcount(fn_name: &str) -> String {
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

pub(super) fn code_is_palindrome_arr(fn_name: &str) -> String {
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

pub(super) fn code_sum_odd_indexed(fn_name: &str) -> String {
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

pub(super) fn code_min_positive(fn_name: &str) -> String {
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

pub(super) fn code_lucas_number(fn_name: &str) -> String {
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

pub(super) fn code_celsius_to_fahrenheit(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(c: i64) -> i64 {
    return c * 9 / 5 + 32;
}
"#,
        fn_name,
    )
}

pub(super) fn code_is_perfect_square(fn_name: &str) -> String {
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

pub(super) fn code_next_power_of_2(fn_name: &str) -> String {
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

pub(super) fn code_count_peaks(fn_name: &str) -> String {
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
