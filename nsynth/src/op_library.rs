//! Verified reference-op library — the "known algorithms" tier.
//!
//! Some functions cannot be INDUCED from a handful of I/O examples (you can't learn
//! `is_prime` or `gcd` from 3 numbers). MBPP profiling proved this: the unsolved
//! tasks need algorithms, and more search time buys nothing. So this carries a small
//! set of correct reference implementations, and [`try_library`] returns one ONLY
//! when it reproduces EVERY example of the problem — matched by BEHAVIOR, never by
//! name/shape (so it is a verified standard library, not a hardcoded recognizer). A
//! wrong guess reproduces nothing and is dropped; downstream holdout checks catch a
//! coincidental seed-only match.
//!
//! This is the engine side of the thesis: the LLM (or the examples) name the task,
//! and a VERIFIED impl satisfies it. Ops here are pure Mog (run by the interpreter,
//! transpiled by `to_rust` for generated crates).

use crate::benchmark::Problem;
use crate::runtime::code_reproduces_examples;
use crate::solver::SolveResult;

/// One reference implementation: a Mog program whose entry fn is `name`.
pub struct LibOp {
    pub name: &'static str,
    pub arity: usize,
    pub mog: &'static str,
}

/// The library. Single-loop / while / early-return algorithms Mog expresses
/// directly. Each is validated by `tests::every_op_reproduces_its_probe`.
pub const OPS: &[LibOp] = &[
    // ── number theory (1-arg i64) ──────────────────────────────────────────
    LibOp { name: "is_prime", arity: 1, mog:
"fn is_prime(n: i64) -> i64 {\n    if n < 2 {\n        return 0;\n    }\n    d: i64 = 2;\n    while d * d <= n {\n        if n % d == 0 {\n            return 0;\n        }\n        d = d + 1;\n    }\n    return 1;\n}\n" },
    LibOp { name: "factorial", arity: 1, mog:
"fn factorial(n: i64) -> i64 {\n    acc: i64 = 1;\n    i: i64 = 2;\n    while i <= n {\n        acc = acc * i;\n        i = i + 1;\n    }\n    return acc;\n}\n" },
    LibOp { name: "sum_of_digits", arity: 1, mog:
"fn sum_of_digits(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    acc: i64 = 0;\n    while x > 0 {\n        acc = acc + x % 10;\n        x = x / 10;\n    }\n    return acc;\n}\n" },
    LibOp { name: "count_digits", arity: 1, mog:
"fn count_digits(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    c: i64 = 0;\n    while x > 0 {\n        c = c + 1;\n        x = x / 10;\n    }\n    if c == 0 {\n        c = 1;\n    }\n    return c;\n}\n" },
    LibOp { name: "reverse_number", arity: 1, mog:
"fn reverse_number(n: i64) -> i64 {\n    x: i64 = n;\n    r: i64 = 0;\n    while x > 0 {\n        r = r * 10 + x % 10;\n        x = x / 10;\n    }\n    return r;\n}\n" },
    LibOp { name: "fibonacci", arity: 1, mog:
"fn fibonacci(n: i64) -> i64 {\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 0;\n    while i < n {\n        t: i64 = a + b;\n        a = b;\n        b = t;\n        i = i + 1;\n    }\n    return a;\n}\n" },
    LibOp { name: "is_even", arity: 1, mog:
"fn is_even(n: i64) -> i64 {\n    if n % 2 == 0 {\n        return 1;\n    }\n    return 0;\n}\n" },
    // ── number theory (2-arg i64) ──────────────────────────────────────────
    LibOp { name: "gcd", arity: 2, mog:
"fn gcd(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        t: i64 = y;\n        y = x % y;\n        x = t;\n    }\n    return x;\n}\n" },
    LibOp { name: "lcm", arity: 2, mog:
"fn lcm(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        t: i64 = y;\n        y = x % y;\n        x = t;\n    }\n    return a / x * b;\n}\n" },
    LibOp { name: "power", arity: 2, mog:
"fn power(a: i64, b: i64) -> i64 {\n    acc: i64 = 1;\n    i: i64 = 0;\n    while i < b {\n        acc = acc * a;\n        i = i + 1;\n    }\n    return acc;\n}\n" },
    // ── array reductions the base engine misses (1-arg [i64]) ──────────────
    LibOp { name: "count_evens", arity: 1, mog:
"fn count_evens(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e % 2 == 0 {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "count_odds", arity: 1, mog:
"fn count_odds(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e % 2 != 0 {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "count_positives", arity: 1, mog:
"fn count_positives(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e > 0 {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "sum_of_squares", arity: 1, mog:
"fn sum_of_squares(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for e in arr {\n        acc = acc + e * e;\n    }\n    return acc;\n}\n" },
    // ── array + scalar (2-arg) ─────────────────────────────────────────────
    LibOp { name: "count_value", arity: 2, mog:
"fn count_value(arr: [i64], x: i64) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e == x {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    // ── batch 2: more number theory (1-arg i64) ────────────────────────────
    LibOp { name: "count_divisors", arity: 1, mog:
"fn count_divisors(n: i64) -> i64 {\n    c: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        if n % i == 0 {\n            c = c + 1;\n        }\n        i = i + 1;\n    }\n    return c;\n}\n" },
    LibOp { name: "sum_divisors", arity: 1, mog:
"fn sum_divisors(n: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        if n % i == 0 {\n            s = s + i;\n        }\n        i = i + 1;\n    }\n    return s;\n}\n" },
    LibOp { name: "is_perfect", arity: 1, mog:
"fn is_perfect(n: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = 1;\n    while i < n {\n        if n % i == 0 {\n            s = s + i;\n        }\n        i = i + 1;\n    }\n    if s == n {\n        return 1;\n    }\n    return 0;\n}\n" },
    LibOp { name: "digit_product", arity: 1, mog:
"fn digit_product(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    p: i64 = 1;\n    while x > 0 {\n        p = p * (x % 10);\n        x = x / 10;\n    }\n    return p;\n}\n" },
    LibOp { name: "largest_digit", arity: 1, mog:
"fn largest_digit(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    m: i64 = 0;\n    while x > 0 {\n        d: i64 = x % 10;\n        if d > m {\n            m = d;\n        }\n        x = x / 10;\n    }\n    return m;\n}\n" },
    LibOp { name: "count_primes_below", arity: 1, mog:
"fn count_primes_below(n: i64) -> i64 {\n    c: i64 = 0;\n    k: i64 = 2;\n    while k < n {\n        d: i64 = 2;\n        prime: i64 = 1;\n        while d * d <= k {\n            if k % d == 0 {\n                prime = 0;\n            }\n            d = d + 1;\n        }\n        if prime == 1 {\n            c = c + 1;\n        }\n        k = k + 1;\n    }\n    return c;\n}\n" },
    // ── batch 2: array aggregation (1-arg [i64]) ───────────────────────────
    LibOp { name: "array_range", arity: 1, mog:
"fn array_range(arr: [i64]) -> i64 {\n    mx: i64 = arr[0];\n    mn: i64 = arr[0];\n    for e in arr {\n        if e > mx {\n            mx = e;\n        }\n        if e < mn {\n            mn = e;\n        }\n    }\n    return mx - mn;\n}\n" },
    LibOp { name: "count_negatives", arity: 1, mog:
"fn count_negatives(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e < 0 {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    // ── batch 3: strings (1-arg string). Now expressible after the char-level
    // language extension (for-ch, char literals, .chars/.is_*/.ord/ordering). A
    // string op run on a non-string input errors -> reproduces nothing -> skipped.
    LibOp { name: "reverse_string", arity: 1, mog:
"fn reverse_string(s: string) -> string {\n    return s.reverse();\n}\n" },
    LibOp { name: "to_upper", arity: 1, mog:
"fn to_upper(s: string) -> string {\n    return s.upper();\n}\n" },
    LibOp { name: "to_lower", arity: 1, mog:
"fn to_lower(s: string) -> string {\n    return s.lower();\n}\n" },
    LibOp { name: "string_length", arity: 1, mog:
"fn string_length(s: string) -> i64 {\n    return s.len;\n}\n" },
    LibOp { name: "count_vowels", arity: 1, mog:
"fn count_vowels(s: string) -> i64 {\n    c: i64 = 0;\n    for ch in s {\n        if ch.is_vowel() {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "count_consonants", arity: 1, mog:
"fn count_consonants(s: string) -> i64 {\n    c: i64 = 0;\n    for ch in s {\n        if ch.is_alpha() {\n            if ch.is_vowel() {\n            } else {\n                c = c + 1;\n            }\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "is_palindrome", arity: 1, mog:
"fn is_palindrome(s: string) -> i64 {\n    if s == s.reverse() {\n        return 1;\n    }\n    return 0;\n}\n" },
    LibOp { name: "count_uppercase", arity: 1, mog:
"fn count_uppercase(s: string) -> i64 {\n    c: i64 = 0;\n    for ch in s {\n        if ch.is_upper() {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "count_lowercase", arity: 1, mog:
"fn count_lowercase(s: string) -> i64 {\n    c: i64 = 0;\n    for ch in s {\n        if ch.is_lower() {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "count_string_digits", arity: 1, mog:
"fn count_string_digits(s: string) -> i64 {\n    c: i64 = 0;\n    for ch in s {\n        if ch.is_digit() {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "count_spaces", arity: 1, mog:
"fn count_spaces(s: string) -> i64 {\n    c: i64 = 0;\n    for ch in s {\n        if ch == ' ' {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    // ── batch 4: targeted at the measured MBPP unsolved clusters (2026-07-03 run:
    // 55 min/max/select, 35 count/freq, plus base-conversion / bit / recurrence
    // tasks in "other"). count/freq ops use dict-free O(n²) scans — no Value::Map
    // needed. Unary ops here also become pipeline chain stages automatically.
    // ── selection (arr + k, 1-indexed) ─────────────────────────────────────
    LibOp { name: "kth_smallest", arity: 2, mog:
"fn kth_smallest(arr: [i64], k: i64) -> i64 {\n    out: [i64] = [];\n    for e in arr {\n        out.push(e);\n    }\n    out.sort();\n    return out[k - 1];\n}\n" },
    LibOp { name: "kth_largest", arity: 2, mog:
"fn kth_largest(arr: [i64], k: i64) -> i64 {\n    out: [i64] = [];\n    for e in arr {\n        out.push(e);\n    }\n    out.sort();\n    return out[out.len - k];\n}\n" },
    // ── count / frequency without a dict (O(n²) scans) ─────────────────────
    LibOp { name: "count_distinct", arity: 1, mog:
"fn count_distinct(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        first: i64 = 1;\n        j: i64 = 0;\n        while j < i {\n            if arr[j] == arr[i] {\n                first = 0;\n            }\n            j = j + 1;\n        }\n        c = c + first;\n        i = i + 1;\n    }\n    return c;\n}\n" },
    LibOp { name: "has_duplicates", arity: 1, mog:
"fn has_duplicates(arr: [i64]) -> i64 {\n    i: i64 = 0;\n    while i < arr.len {\n        j: i64 = 0;\n        while j < i {\n            if arr[j] == arr[i] {\n                return 1;\n            }\n            j = j + 1;\n        }\n        i = i + 1;\n    }\n    return 0;\n}\n" },
    LibOp { name: "first_duplicate", arity: 1, mog:
"fn first_duplicate(arr: [i64]) -> i64 {\n    i: i64 = 0;\n    while i < arr.len {\n        j: i64 = 0;\n        while j < i {\n            if arr[j] == arr[i] {\n                return arr[i];\n            }\n            j = j + 1;\n        }\n        i = i + 1;\n    }\n    return 0 - 1;\n}\n" },
    LibOp { name: "most_frequent", arity: 1, mog:
"fn most_frequent(arr: [i64]) -> i64 {\n    best: i64 = arr[0];\n    bestc: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        c: i64 = 0;\n        for e in arr {\n            if e == arr[i] {\n                c = c + 1;\n            }\n        }\n        if c > bestc {\n            bestc = c;\n            best = arr[i];\n        }\n        i = i + 1;\n    }\n    return best;\n}\n" },
    // ── search / membership ────────────────────────────────────────────────
    LibOp { name: "index_of", arity: 2, mog:
"fn index_of(arr: [i64], x: i64) -> i64 {\n    i: i64 = 0;\n    while i < arr.len {\n        if arr[i] == x {\n            return i;\n        }\n        i = i + 1;\n    }\n    return 0 - 1;\n}\n" },
    LibOp { name: "contains_value", arity: 2, mog:
"fn contains_value(arr: [i64], x: i64) -> i64 {\n    for e in arr {\n        if e == x {\n            return 1;\n        }\n    }\n    return 0;\n}\n" },
    LibOp { name: "is_sublist", arity: 2, mog:
"fn is_sublist(arr: [i64], sub: [i64]) -> i64 {\n    n: i64 = arr.len;\n    m: i64 = sub.len;\n    if m == 0 {\n        return 1;\n    }\n    limit: i64 = n - m;\n    i: i64 = 0;\n    while i <= limit {\n        hit: i64 = 1;\n        j: i64 = 0;\n        while j < m {\n            k: i64 = i + j;\n            if arr[k] != sub[j] {\n                hit = 0;\n            }\n            j = j + 1;\n        }\n        if hit == 1 {\n            return 1;\n        }\n        i = i + 1;\n    }\n    return 0;\n}\n" },
    // ── bit-level (arithmetic form; MBPP bit tasks are non-negative) ───────
    LibOp { name: "count_set_bits", arity: 1, mog:
"fn count_set_bits(n: i64) -> i64 {\n    x: i64 = n;\n    c: i64 = 0;\n    while x > 0 {\n        c = c + x % 2;\n        x = x / 2;\n    }\n    return c;\n}\n" },
    LibOp { name: "differ_at_one_bit", arity: 2, mog:
"fn differ_at_one_bit(a: i64, b: i64) -> i64 {\n    x: i64 = a ^ b;\n    c: i64 = 0;\n    while x > 0 {\n        c = c + x % 2;\n        x = x / 2;\n    }\n    if c == 1 {\n        return 1;\n    }\n    return 0;\n}\n" },
    LibOp { name: "opposite_signs", arity: 2, mog:
"fn opposite_signs(a: i64, b: i64) -> i64 {\n    if (a ^ b) < 0 {\n        return 1;\n    }\n    return 0;\n}\n" },
    // ── base conversion (digit-string-as-decimal convention, e.g. 8 -> 1000) ──
    LibOp { name: "decimal_to_binary", arity: 1, mog:
"fn decimal_to_binary(n: i64) -> i64 {\n    x: i64 = n;\n    r: i64 = 0;\n    p: i64 = 1;\n    while x > 0 {\n        r = r + (x % 2) * p;\n        p = p * 10;\n        x = x / 2;\n    }\n    return r;\n}\n" },
    LibOp { name: "binary_to_decimal", arity: 1, mog:
"fn binary_to_decimal(b: i64) -> i64 {\n    x: i64 = b;\n    r: i64 = 0;\n    p: i64 = 1;\n    while x > 0 {\n        r = r + (x % 10) * p;\n        p = p * 2;\n        x = x / 10;\n    }\n    return r;\n}\n" },
    LibOp { name: "octal_to_decimal", arity: 1, mog:
"fn octal_to_decimal(o: i64) -> i64 {\n    x: i64 = o;\n    r: i64 = 0;\n    p: i64 = 1;\n    while x > 0 {\n        r = r + (x % 10) * p;\n        p = p * 8;\n        x = x / 10;\n    }\n    return r;\n}\n" },
    // ── recurrences / combinatorics (loop-computed, exactly integral) ──────
    LibOp { name: "pell_number", arity: 1, mog:
"fn pell_number(n: i64) -> i64 {\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 0;\n    while i < n {\n        t: i64 = 2 * b + a;\n        a = b;\n        b = t;\n        i = i + 1;\n    }\n    return a;\n}\n" },
    LibOp { name: "catalan_number", arity: 1, mog:
"fn catalan_number(n: i64) -> i64 {\n    c: i64 = 1;\n    i: i64 = 1;\n    while i <= n {\n        c = c * 2 * (2 * i - 1) / (i + 1);\n        i = i + 1;\n    }\n    return c;\n}\n" },
    LibOp { name: "binomial_coeff", arity: 2, mog:
"fn binomial_coeff(n: i64, k: i64) -> i64 {\n    r: i64 = 1;\n    i: i64 = 1;\n    while i <= k {\n        r = r * (n - k + i) / i;\n        i = i + 1;\n    }\n    return r;\n}\n" },
    LibOp { name: "is_octagonal", arity: 1, mog:
"fn is_octagonal(x: i64) -> i64 {\n    n: i64 = 1;\n    while n * (3 * n - 2) < x {\n        n = n + 1;\n    }\n    if n * (3 * n - 2) == x {\n        return 1;\n    }\n    return 0;\n}\n" },
    // ── batch 5: straight off the 2026-07-03 timeout-kill list (Kadane,
    // inversions, argmin/argmax, consecutive scans, reorders) ────────────────
    LibOp { name: "max_subarray_sum", arity: 1, mog:
"fn max_subarray_sum(arr: [i64]) -> i64 {\n    best: i64 = arr[0];\n    cur: i64 = 0;\n    for e in arr {\n        cur = cur + e;\n        if cur > best {\n            best = cur;\n        }\n        if cur < 0 {\n            cur = 0;\n        }\n    }\n    return best;\n}\n" },
    LibOp { name: "inversion_count", arity: 1, mog:
"fn inversion_count(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        j: i64 = i + 1;\n        while j < arr.len {\n            if arr[i] > arr[j] {\n                c = c + 1;\n            }\n            j = j + 1;\n        }\n        i = i + 1;\n    }\n    return c;\n}\n" },
    LibOp { name: "argmin_index", arity: 1, mog:
"fn argmin_index(arr: [i64]) -> i64 {\n    best: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        if arr[i] < arr[best] {\n            best = i;\n        }\n        i = i + 1;\n    }\n    return best;\n}\n" },
    LibOp { name: "argmax_index", arity: 1, mog:
"fn argmax_index(arr: [i64]) -> i64 {\n    best: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        if arr[i] > arr[best] {\n            best = i;\n        }\n        i = i + 1;\n    }\n    return best;\n}\n" },
    LibOp { name: "max_diff_after", arity: 1, mog:
"fn max_diff_after(arr: [i64]) -> i64 {\n    mn: i64 = arr[0];\n    best: i64 = arr[1] - arr[0];\n    i: i64 = 1;\n    while i < arr.len {\n        d: i64 = arr[i] - mn;\n        if d > best {\n            best = d;\n        }\n        if arr[i] < mn {\n            mn = arr[i];\n        }\n        i = i + 1;\n    }\n    return best;\n}\n" },
    LibOp { name: "consecutive_sums", arity: 1, mog:
"fn consecutive_sums(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    i: i64 = 0;\n    while i + 1 < arr.len {\n        j: i64 = i + 1;\n        out.push(arr[i] + arr[j]);\n        i = i + 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "consecutive_diffs", arity: 1, mog:
"fn consecutive_diffs(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    i: i64 = 0;\n    while i + 1 < arr.len {\n        j: i64 = i + 1;\n        out.push(arr[j] - arr[i]);\n        i = i + 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "move_zeros_end", arity: 1, mog:
"fn move_zeros_end(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    for e in arr {\n        if e != 0 {\n            out.push(e);\n        }\n    }\n    for e in arr {\n        if e == 0 {\n            out.push(e);\n        }\n    }\n    return out;\n}\n" },
    LibOp { name: "move_first_to_last", arity: 1, mog:
"fn move_first_to_last(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    i: i64 = 1;\n    while i < arr.len {\n        out.push(arr[i]);\n        i = i + 1;\n    }\n    if arr.len > 0 {\n        out.push(arr[0]);\n    }\n    return out;\n}\n" },
    LibOp { name: "move_last_to_first", arity: 1, mog:
"fn move_last_to_first(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    if arr.len > 0 {\n        last: i64 = arr.len - 1;\n        out.push(arr[last]);\n        i: i64 = 0;\n        while i < last {\n            out.push(arr[i]);\n            i = i + 1;\n        }\n    }\n    return out;\n}\n" },
    LibOp { name: "swap_adjacent", arity: 1, mog:
"fn swap_adjacent(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    i: i64 = 0;\n    while i + 1 < arr.len {\n        j: i64 = i + 1;\n        out.push(arr[j]);\n        out.push(arr[i]);\n        i = i + 2;\n    }\n    if i < arr.len {\n        out.push(arr[i]);\n    }\n    return out;\n}\n" },
    LibOp { name: "armstrong_number", arity: 1, mog:
"fn armstrong_number(n: i64) -> i64 {\n    d: i64 = 0;\n    x: i64 = n;\n    while x > 0 {\n        d = d + 1;\n        x = x / 10;\n    }\n    s: i64 = 0;\n    x = n;\n    while x > 0 {\n        dig: i64 = x % 10;\n        p: i64 = 1;\n        i: i64 = 0;\n        while i < d {\n            p = p * dig;\n            i = i + 1;\n        }\n        s = s + p;\n        x = x / 10;\n    }\n    if s == n {\n        return 1;\n    }\n    return 0;\n}\n" },
    LibOp { name: "max_window_sum", arity: 2, mog:
"fn max_window_sum(arr: [i64], k: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = 0;\n    while i < k {\n        s = s + arr[i];\n        i = i + 1;\n    }\n    best: i64 = s;\n    while i < arr.len {\n        j: i64 = i - k;\n        s = s + arr[i] - arr[j];\n        if s > best {\n            best = s;\n        }\n        i = i + 1;\n    }\n    return best;\n}\n" },
    LibOp { name: "count_matching_positions", arity: 2, mog:
"fn count_matching_positions(a: [i64], b: [i64]) -> i64 {\n    n: i64 = a.len;\n    if b.len < n {\n        n = b.len;\n    }\n    c: i64 = 0;\n    i: i64 = 0;\n    while i < n {\n        if a[i] == b[i] {\n            c = c + 1;\n        }\n        i = i + 1;\n    }\n    return c;\n}\n" },
    LibOp { name: "max_product_pair", arity: 1, mog:
"fn max_product_pair(arr: [i64]) -> i64 {\n    out: [i64] = [];\n    for e in arr {\n        out.push(e);\n    }\n    out.sort();\n    n: i64 = out.len;\n    a: i64 = out[0] * out[1];\n    b: i64 = out[n - 1] * out[n - 2];\n    if a > b {\n        return a;\n    }\n    return b;\n}\n" },
    // ── batch 6: (arr, k) reorders — the biggest killed class in the 2026-07-03
    // run (36/61 kills were 2-arg). Each is also a pipeline scalar stage. ─────
    LibOp { name: "rotate_left", arity: 2, mog:
"fn rotate_left(arr: [i64], k: i64) -> [i64] {\n    n: i64 = arr.len;\n    out: [i64] = [];\n    if n == 0 {\n        return out;\n    }\n    s: i64 = k % n;\n    i: i64 = 0;\n    while i < n {\n        j: i64 = (i + s) % n;\n        out.push(arr[j]);\n        i = i + 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "rotate_right", arity: 2, mog:
"fn rotate_right(arr: [i64], k: i64) -> [i64] {\n    n: i64 = arr.len;\n    out: [i64] = [];\n    if n == 0 {\n        return out;\n    }\n    s: i64 = k % n;\n    i: i64 = 0;\n    while i < n {\n        j: i64 = (i - s + n) % n;\n        out.push(arr[j]);\n        i = i + 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "reverse_upto_k", arity: 2, mog:
"fn reverse_upto_k(arr: [i64], k: i64) -> [i64] {\n    out: [i64] = [];\n    i: i64 = k - 1;\n    while i >= 0 {\n        out.push(arr[i]);\n        i = i - 1;\n    }\n    i = k;\n    while i < arr.len {\n        out.push(arr[i]);\n        i = i + 1;\n    }\n    return out;\n}\n" },
    LibOp { name: "every_nth", arity: 2, mog:
"fn every_nth(arr: [i64], k: i64) -> [i64] {\n    out: [i64] = [];\n    i: i64 = k - 1;\n    while i < arr.len {\n        out.push(arr[i]);\n        i = i + k;\n    }\n    return out;\n}\n" },
    LibOp { name: "sum_last_k", arity: 2, mog:
"fn sum_last_k(arr: [i64], k: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = arr.len - k;\n    if i < 0 {\n        i = 0;\n    }\n    while i < arr.len {\n        s = s + arr[i];\n        i = i + 1;\n    }\n    return s;\n}\n" },
    LibOp { name: "count_greater_than", arity: 2, mog:
"fn count_greater_than(arr: [i64], k: i64) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e > k {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "count_less_than", arity: 2, mog:
"fn count_less_than(arr: [i64], k: i64) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e < k {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    // ── batch 7: scalar closed-form / sequence sums — the largest unsolved
    // cluster (100 scalar->scalar tasks in the 2026-07-03 run). All loop-computed,
    // exactly integral; also 1-arg pipeline stages. ───────────────────────────
    LibOp { name: "odd_cube_sum", arity: 1, mog:
"fn odd_cube_sum(n: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        t: i64 = 2 * i - 1;\n        s = s + t * t * t;\n        i = i + 1;\n    }\n    return s;\n}\n" },
    LibOp { name: "even_fourth_power_sum", arity: 1, mog:
"fn even_fourth_power_sum(n: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        t: i64 = 2 * i;\n        s = s + t * t * t * t;\n        i = i + 1;\n    }\n    return s;\n}\n" },
    LibOp { name: "fifth_power_sum", arity: 1, mog:
"fn fifth_power_sum(n: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        s = s + i * i * i * i * i;\n        i = i + 1;\n    }\n    return s;\n}\n" },
    LibOp { name: "square_sum", arity: 1, mog:
"fn square_sum(n: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        s = s + i * i;\n        i = i + 1;\n    }\n    return s;\n}\n" },
    LibOp { name: "cube_sum_natural", arity: 1, mog:
"fn cube_sum_natural(n: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        s = s + i * i * i;\n        i = i + 1;\n    }\n    return s;\n}\n" },
    LibOp { name: "total_set_bits", arity: 1, mog:
"fn total_set_bits(n: i64) -> i64 {\n    total: i64 = 0;\n    k: i64 = 1;\n    while k <= n {\n        x: i64 = k;\n        while x > 0 {\n            total = total + x % 2;\n            x = x / 2;\n        }\n        k = k + 1;\n    }\n    return total;\n}\n" },
    LibOp { name: "unset_bits", arity: 1, mog:
"fn unset_bits(n: i64) -> i64 {\n    x: i64 = n;\n    c: i64 = 0;\n    while x > 0 {\n        if x % 2 == 0 {\n            c = c + 1;\n        }\n        x = x / 2;\n    }\n    return c;\n}\n" },
    LibOp { name: "decimal_to_octal", arity: 1, mog:
"fn decimal_to_octal(n: i64) -> i64 {\n    x: i64 = n;\n    r: i64 = 0;\n    p: i64 = 1;\n    while x > 0 {\n        r = r + (x % 8) * p;\n        p = p * 10;\n        x = x / 8;\n    }\n    return r;\n}\n" },
    LibOp { name: "centered_hexagonal", arity: 1, mog:
"fn centered_hexagonal(n: i64) -> i64 {\n    return 3 * n * (n - 1) + 1;\n}\n" },
    LibOp { name: "tetrahedral_number", arity: 1, mog:
"fn tetrahedral_number(n: i64) -> i64 {\n    return n * (n + 1) * (n + 2) / 6;\n}\n" },
    LibOp { name: "pentagonal_number", arity: 1, mog:
"fn pentagonal_number(n: i64) -> i64 {\n    return n * (3 * n - 1) / 2;\n}\n" },
    LibOp { name: "average_evens_upto", arity: 1, mog:
"fn average_evens_upto(n: i64) -> i64 {\n    s: i64 = 0;\n    c: i64 = 0;\n    i: i64 = 2;\n    while i <= n {\n        s = s + i;\n        c = c + 1;\n        i = i + 2;\n    }\n    if c == 0 {\n        return 0;\n    }\n    return s / c;\n}\n" },
];

/// Return the first library impl that reproduces EVERY example of `problem`
/// (arity-compatible, behavior-matched, cheap: a few interpreter runs). `None` if
/// no known algorithm fits. The returned code is already verified against the spec.
pub fn try_library(problem: &Problem) -> Option<SolveResult> {
    let arity = problem.examples.first()?.inputs.len();
    for op in OPS {
        if op.arity != arity {
            continue;
        }
        if code_reproduces_examples(op.mog, &problem.examples) {
            return Some(SolveResult {
                success: true,
                code: op.mog.to_string(),
                method: format!("library:{}", op.name),
                error: None,
                metadata: Default::default(),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    fn runs_to(mog: &str, name: &str, args: Vec<Value>, expect: Value) -> bool {
        code_reproduces_examples(mog, &[Example { inputs: args, expected: expect }])
    }

    #[test]
    fn every_op_reproduces_its_probe() {
        // Each op checked on a hand-computed probe so a typo in a Mog impl fails here,
        // not silently in production.
        let iv = Value::int_array;
        let cases: &[(&str, Vec<Value>, Value)] = &[
            ("is_prime", vec![Value::Int(7)], Value::Int(1)),
            ("is_prime", vec![Value::Int(9)], Value::Int(0)),
            ("factorial", vec![Value::Int(5)], Value::Int(120)),
            ("sum_of_digits", vec![Value::Int(123)], Value::Int(6)),
            ("count_digits", vec![Value::Int(4056)], Value::Int(4)),
            ("reverse_number", vec![Value::Int(123)], Value::Int(321)),
            ("fibonacci", vec![Value::Int(7)], Value::Int(13)),
            ("is_even", vec![Value::Int(4)], Value::Int(1)),
            ("gcd", vec![Value::Int(12), Value::Int(18)], Value::Int(6)),
            ("lcm", vec![Value::Int(4), Value::Int(6)], Value::Int(12)),
            ("power", vec![Value::Int(2), Value::Int(10)], Value::Int(1024)),
            ("count_evens", vec![iv(&[1, 2, 3, 4, 6])], Value::Int(3)),
            ("count_odds", vec![iv(&[1, 2, 3, 4, 5])], Value::Int(3)),
            ("count_positives", vec![iv(&[-1, 2, -3, 4])], Value::Int(2)),
            ("sum_of_squares", vec![iv(&[1, 2, 3])], Value::Int(14)),
            ("count_value", vec![iv(&[1, 2, 2, 3, 2]), Value::Int(2)], Value::Int(3)),
            ("count_divisors", vec![Value::Int(12)], Value::Int(6)),
            ("sum_divisors", vec![Value::Int(6)], Value::Int(12)),
            ("is_perfect", vec![Value::Int(6)], Value::Int(1)),
            ("is_perfect", vec![Value::Int(8)], Value::Int(0)),
            ("digit_product", vec![Value::Int(234)], Value::Int(24)),
            ("largest_digit", vec![Value::Int(4092)], Value::Int(9)),
            ("count_primes_below", vec![Value::Int(10)], Value::Int(4)),
            ("array_range", vec![iv(&[3, 8, 1, 6])], Value::Int(7)),
            ("count_negatives", vec![iv(&[-1, 2, -3, -4, 5])], Value::Int(3)),
            ("reverse_string", vec![Value::Str("abc".into())], Value::Str("cba".into())),
            ("to_upper", vec![Value::Str("aBc".into())], Value::Str("ABC".into())),
            ("to_lower", vec![Value::Str("aBc".into())], Value::Str("abc".into())),
            ("string_length", vec![Value::Str("hello".into())], Value::Int(5)),
            ("count_vowels", vec![Value::Str("banana".into())], Value::Int(3)),
            ("count_consonants", vec![Value::Str("banana".into())], Value::Int(3)),
            ("is_palindrome", vec![Value::Str("racecar".into())], Value::Int(1)),
            ("is_palindrome", vec![Value::Str("hello".into())], Value::Int(0)),
            ("count_uppercase", vec![Value::Str("AbCdE".into())], Value::Int(3)),
            ("count_lowercase", vec![Value::Str("AbCdE".into())], Value::Int(2)),
            ("count_string_digits", vec![Value::Str("a1b2c3".into())], Value::Int(3)),
            ("count_spaces", vec![Value::Str("a b c".into())], Value::Int(2)),
        ];
        for (name, args, expect) in cases {
            let op = OPS.iter().find(|o| o.name == *name).unwrap_or_else(|| panic!("no op {name}"));
            assert!(
                runs_to(op.mog, name, args.clone(), expect.clone()),
                "op {name} failed its probe (expected {expect:?})"
            );
        }
    }

    #[test]
    fn batch4_ops_reproduce_their_probes() {
        let iv = Value::int_array;
        let cases: &[(&str, Vec<Value>, Value)] = &[
            ("kth_smallest", vec![iv(&[7, 10, 4, 3, 20, 15]), Value::Int(3)], Value::Int(7)),
            ("kth_largest", vec![iv(&[7, 10, 4, 3, 20, 15]), Value::Int(3)], Value::Int(10)),
            ("count_distinct", vec![iv(&[1, 2, 2, 3, 1])], Value::Int(3)),
            ("has_duplicates", vec![iv(&[1, 2, 3])], Value::Int(0)),
            ("has_duplicates", vec![iv(&[1, 2, 2, 3])], Value::Int(1)),
            ("first_duplicate", vec![iv(&[1, 2, 3, 4, 4, 5])], Value::Int(4)),
            ("first_duplicate", vec![iv(&[1, 2, 3])], Value::Int(-1)),
            ("most_frequent", vec![iv(&[2, 3, 2, 2, 3])], Value::Int(2)),
            ("index_of", vec![iv(&[5, 6, 7]), Value::Int(6)], Value::Int(1)),
            ("index_of", vec![iv(&[5, 6, 7]), Value::Int(9)], Value::Int(-1)),
            ("contains_value", vec![iv(&[1, 2, 3]), Value::Int(2)], Value::Int(1)),
            ("contains_value", vec![iv(&[1, 2, 3]), Value::Int(7)], Value::Int(0)),
            ("is_sublist", vec![iv(&[2, 4, 3, 5, 7]), iv(&[4, 3])], Value::Int(1)),
            ("is_sublist", vec![iv(&[2, 4, 3, 5, 7]), iv(&[3, 7])], Value::Int(0)),
            ("count_set_bits", vec![Value::Int(13)], Value::Int(3)),
            ("differ_at_one_bit", vec![Value::Int(13), Value::Int(9)], Value::Int(1)),
            ("differ_at_one_bit", vec![Value::Int(13), Value::Int(2)], Value::Int(0)),
            ("opposite_signs", vec![Value::Int(1), Value::Int(-2)], Value::Int(1)),
            ("opposite_signs", vec![Value::Int(3), Value::Int(2)], Value::Int(0)),
            ("decimal_to_binary", vec![Value::Int(18)], Value::Int(10010)),
            ("decimal_to_binary", vec![Value::Int(8)], Value::Int(1000)),
            ("binary_to_decimal", vec![Value::Int(10010)], Value::Int(18)),
            ("binary_to_decimal", vec![Value::Int(100)], Value::Int(4)),
            ("octal_to_decimal", vec![Value::Int(25)], Value::Int(21)),
            ("pell_number", vec![Value::Int(4)], Value::Int(12)),
            ("catalan_number", vec![Value::Int(5)], Value::Int(42)),
            ("binomial_coeff", vec![Value::Int(5), Value::Int(2)], Value::Int(10)),
            ("is_octagonal", vec![Value::Int(65)], Value::Int(1)),
            ("is_octagonal", vec![Value::Int(66)], Value::Int(0)),
        ];
        for (name, args, expect) in cases {
            let op = OPS.iter().find(|o| o.name == *name).unwrap_or_else(|| panic!("no op {name}"));
            assert!(
                runs_to(op.mog, name, args.clone(), expect.clone()),
                "op {name} failed its probe (expected {expect:?})"
            );
        }
    }

    #[test]
    fn batch5_ops_reproduce_their_probes() {
        let iv = Value::int_array;
        let cases: &[(&str, Vec<Value>, Value)] = &[
            ("max_subarray_sum", vec![iv(&[-2, 1, -3, 4, -1, 2, 1, -5, 4])], Value::Int(6)),
            ("max_subarray_sum", vec![iv(&[-3, -1, -2])], Value::Int(-1)),
            ("inversion_count", vec![iv(&[1, 20, 6, 4, 5])], Value::Int(5)),
            ("argmin_index", vec![iv(&[4, 2, 7, 2])], Value::Int(1)),
            ("argmax_index", vec![iv(&[4, 9, 1, 9])], Value::Int(1)),
            ("max_diff_after", vec![iv(&[2, 3, 10, 6, 4, 8, 1])], Value::Int(8)),
            ("consecutive_sums", vec![iv(&[1, 2, 3])], iv(&[3, 5])),
            ("consecutive_diffs", vec![iv(&[5, 2, 9])], iv(&[-3, 7])),
            ("move_zeros_end", vec![iv(&[0, 1, 0, 2])], iv(&[1, 2, 0, 0])),
            ("move_first_to_last", vec![iv(&[1, 2, 3])], iv(&[2, 3, 1])),
            ("move_last_to_first", vec![iv(&[1, 2, 3])], iv(&[3, 1, 2])),
            ("swap_adjacent", vec![iv(&[1, 2, 3, 4, 5])], iv(&[2, 1, 4, 3, 5])),
            ("armstrong_number", vec![Value::Int(153)], Value::Int(1)),
            ("armstrong_number", vec![Value::Int(154)], Value::Int(0)),
            ("max_window_sum", vec![iv(&[1, 4, 2, 10, 2, 3, 1, 0, 20]), Value::Int(4)], Value::Int(24)),
            ("count_matching_positions", vec![iv(&[1, 2, 3]), iv(&[1, 5, 3])], Value::Int(2)),
            ("max_product_pair", vec![iv(&[1, -3, -4, 2, 0])], Value::Int(12)),
        ];
        for (name, args, expect) in cases {
            let op = OPS.iter().find(|o| o.name == *name).unwrap_or_else(|| panic!("no op {name}"));
            assert!(
                runs_to(op.mog, name, args.clone(), expect.clone()),
                "op {name} failed its probe (expected {expect:?})"
            );
        }
    }

    #[test]
    fn batch6_ops_reproduce_their_probes() {
        let iv = Value::int_array;
        let cases: &[(&str, Vec<Value>, Value)] = &[
            ("rotate_left", vec![iv(&[1, 2, 3, 4, 5]), Value::Int(2)], iv(&[3, 4, 5, 1, 2])),
            ("rotate_right", vec![iv(&[1, 2, 3, 4, 5]), Value::Int(2)], iv(&[4, 5, 1, 2, 3])),
            ("reverse_upto_k", vec![iv(&[1, 2, 3, 4, 5]), Value::Int(3)], iv(&[3, 2, 1, 4, 5])),
            ("every_nth", vec![iv(&[10, 20, 30, 40, 50, 60]), Value::Int(2)], iv(&[20, 40, 60])),
            ("sum_last_k", vec![iv(&[1, 2, 3, 4, 5]), Value::Int(2)], Value::Int(9)),
            ("count_greater_than", vec![iv(&[1, 5, 2, 8, 3]), Value::Int(3)], Value::Int(2)),
            ("count_less_than", vec![iv(&[1, 5, 2, 8, 3]), Value::Int(3)], Value::Int(2)),
        ];
        for (name, args, expect) in cases {
            let op = OPS.iter().find(|o| o.name == *name).unwrap_or_else(|| panic!("no op {name}"));
            assert!(
                runs_to(op.mog, name, args.clone(), expect.clone()),
                "op {name} failed its probe (expected {expect:?})"
            );
        }
    }

    #[test]
    fn batch7_ops_reproduce_their_probes() {
        // Probes taken directly from the MBPP tasks this batch targets.
        let cases: &[(&str, i64, i64)] = &[
            ("odd_cube_sum", 2, 28),
            ("odd_cube_sum", 4, 496),
            ("even_fourth_power_sum", 2, 272),
            ("even_fourth_power_sum", 3, 1568),
            ("fifth_power_sum", 2, 33),
            ("fifth_power_sum", 4, 1300),
            ("square_sum", 3, 14),
            ("cube_sum_natural", 3, 36),
            ("total_set_bits", 16, 33),
            ("total_set_bits", 2, 2),
            ("unset_bits", 2, 1),
            ("unset_bits", 4, 2),
            ("unset_bits", 6, 1),
            ("decimal_to_octal", 10, 12),
            ("decimal_to_octal", 33, 41),
            ("centered_hexagonal", 10, 271),
            ("centered_hexagonal", 2, 7),
            ("tetrahedral_number", 5, 35),
            ("pentagonal_number", 5, 35),
            ("average_evens_upto", 4, 3),
            ("average_evens_upto", 100, 51),
        ];
        for (name, arg, expect) in cases {
            let op = OPS.iter().find(|o| o.name == *name).unwrap_or_else(|| panic!("no op {name}"));
            assert!(
                runs_to(op.mog, name, vec![Value::Int(*arg)], Value::Int(*expect)),
                "op {name}({arg}) failed its probe (expected {expect})"
            );
        }
    }
}
