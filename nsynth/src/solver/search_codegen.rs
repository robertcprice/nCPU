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

pub(super) fn code_suffix_class_search(fn_name: &str, suffixes: &[String]) -> String {
    let mut body = String::new();
    for suffix in suffixes {
        let suffix = suffix.replace('\\', "\\\\").replace('"', "\\\"");
        body.push_str(&format!(
            "    if s.ends_with(\"{suffix}\") {{\n        return 1;\n    }}\n"
        ));
    }
    format!("fn {fn_name}(s: string) -> i64 {{\n{body}    return 0;\n}}\n")
}

pub(super) fn code_array_member_class_search(fn_name: &str, consts: &[i64]) -> String {
    let mut checks = String::new();
    for c in consts {
        checks.push_str(&format!(
            "        if x == {c} {{\n            return 1;\n        }}\n"
        ));
    }
    format!(
        "fn {fn_name}(arr: [i64]) -> i64 {{\n    for x in arr {{\n{checks}    }}\n    return 0;\n}}\n"
    )
}

pub(super) fn code_array_conjunction_search(
    fn_name: &str,
    required: &[i64],
    forbidden: &[i64],
) -> String {
    // One flag per required/forbidden token, set during a single pass.
    let mut decls = String::new();
    let mut sets = String::new();
    for (i, t) in required.iter().enumerate() {
        decls.push_str(&format!("    r{i}: i64 = 0;\n"));
        sets.push_str(&format!(
            "        if x == {t} {{\n            r{i} = 1;\n        }}\n"
        ));
    }
    for (i, t) in forbidden.iter().enumerate() {
        decls.push_str(&format!("    f{i}: i64 = 0;\n"));
        sets.push_str(&format!(
            "        if x == {t} {{\n            f{i} = 1;\n        }}\n"
        ));
    }
    // Guard: all required flags 1, all forbidden flags 0. Mog has no `&&`, so the
    // conjunction is a nest of single-condition `if`s; the innermost returns 1.
    let mut conds: Vec<String> = Vec::new();
    for i in 0..required.len() {
        conds.push(format!("r{i} == 1"));
    }
    for i in 0..forbidden.len() {
        conds.push(format!("f{i} == 0"));
    }
    let mut guard = String::from("return 1;");
    for cond in conds.iter().rev() {
        guard = format!("if {cond} {{ {guard} }}");
    }
    format!(
        "fn {fn_name}(arr: [i64]) -> i64 {{\n{decls}    for x in arr {{\n{sets}    }}\n    {guard}\n    return 0;\n}}\n"
    )
}

pub(super) fn code_array_dnf_search(fn_name: &str, rules: &[Vec<(i64, bool)>]) -> String {
    // One flag per distinct token referenced, set in a single pass; then one
    // nested-if block per conjunctive rule (OR across rules).
    let mut toks: Vec<i64> = rules.iter().flatten().map(|&(t, _)| t).collect();
    toks.sort_unstable();
    toks.dedup();
    let idx = |t: i64| toks.iter().position(|&x| x == t).unwrap();

    let mut decls = String::new();
    let mut sets = String::new();
    for (i, t) in toks.iter().enumerate() {
        decls.push_str(&format!("    t{i}: i64 = 0;\n"));
        sets.push_str(&format!(
            "        if x == {t} {{\n            t{i} = 1;\n        }}\n"
        ));
    }

    let mut blocks = String::new();
    for rule in rules {
        let mut guard = String::from("return 1;");
        for &(tok, want) in rule.iter().rev() {
            let val = if want { 1 } else { 0 };
            guard = format!("if t{} == {val} {{ {guard} }}", idx(tok));
        }
        blocks.push_str(&format!("    {guard}\n"));
    }
    format!(
        "fn {fn_name}(arr: [i64]) -> i64 {{\n{decls}    for x in arr {{\n{sets}    }}\n{blocks}    return 0;\n}}\n"
    )
}

pub(super) fn code_array_sequence_search(fn_name: &str, rules: &[(i64, i64)]) -> String {
    let mut decls = String::new();
    let mut loop_body = String::new();
    for (i, &(a, b)) in rules.iter().enumerate() {
        decls.push_str(&format!("    seen_{i}: i64 = 0;\n"));
        loop_body.push_str(&format!(
            "        if x == {b} {{\n            if seen_{i} == 1 {{\n                return 1;\n            }}\n        }}\n"
        ));
        loop_body.push_str(&format!(
            "        if x == {a} {{\n            seen_{i} = 1;\n        }}\n"
        ));
    }
    format!(
        "fn {fn_name}(arr: [i64]) -> i64 {{\n{decls}    for x in arr {{\n{loop_body}    }}\n    return 0;\n}}\n"
    )
}

pub(super) fn code_array_feature_dnf_search(
    fn_name: &str,
    features: &[super::search_families::ArrayFeature],
    rules: &[Vec<usize>],
) -> String {
    let mut decls = String::new();
    let mut init = String::new();
    let mut loop_body = String::new();
    let mut post_loop = String::new();
    let mut needs_prev_value = false;

    for (idx, feature) in features.iter().enumerate() {
        decls.push_str(&format!("    f{idx}: i64 = 0;\n"));
        match feature {
            super::search_families::ArrayFeature::Contains(tok) => {
                loop_body.push_str(&format!(
                    "        if x == {tok} {{\n            f{idx} = 1;\n        }}\n"
                ));
            }
            super::search_families::ArrayFeature::Adjacent(a, b) => {
                if !needs_prev_value {
                    init.push_str("    prev_value: i64 = 0;\n    has_prev_value: i64 = 0;\n");
                    needs_prev_value = true;
                }
                loop_body.push_str(&format!(
                    "        if has_prev_value == 1 && prev_value == {a} && x == {b} {{\n            f{idx} = 1;\n        }}\n"
                ));
            }
            super::search_families::ArrayFeature::Sequence(a, b) => {
                init.push_str(&format!("    seen_f{idx}_a: i64 = 0;\n"));
                loop_body.push_str(&format!(
                    "        if seen_f{idx}_a == 1 && x == {b} {{\n            f{idx} = 1;\n        }}\n"
                ));
                loop_body.push_str(&format!(
                    "        if x == {a} {{\n            seen_f{idx}_a = 1;\n        }}\n"
                ));
            }
            super::search_families::ArrayFeature::CountAtLeast(tok, threshold) => {
                init.push_str(&format!("    count_f{idx}: i64 = 0;\n"));
                loop_body.push_str(&format!(
                    "        if x == {tok} {{\n            count_f{idx} = count_f{idx} + 1;\n        }}\n"
                ));
                post_loop.push_str(&format!(
                    "    if count_f{idx} >= {threshold} {{\n        f{idx} = 1;\n    }}\n"
                ));
            }
            super::search_families::ArrayFeature::CountExactly(tok, threshold) => {
                init.push_str(&format!("    count_f{idx}: i64 = 0;\n"));
                loop_body.push_str(&format!(
                    "        if x == {tok} {{\n            count_f{idx} = count_f{idx} + 1;\n        }}\n"
                ));
                post_loop.push_str(&format!(
                    "    if count_f{idx} == {threshold} {{\n        f{idx} = 1;\n    }}\n"
                ));
            }
            super::search_families::ArrayFeature::RunAtLeast(tok, length) => {
                init.push_str(&format!(
                    "    run_value_f{idx}: i64 = 0;\n    run_len_f{idx}: i64 = 0;\n"
                ));
                loop_body.push_str(&format!(
                    "        if x == {tok} {{\n            if run_value_f{idx} == {tok} {{\n                run_len_f{idx} = run_len_f{idx} + 1;\n            }} else {{\n                run_value_f{idx} = {tok};\n                run_len_f{idx} = 1;\n            }}\n            if run_len_f{idx} >= {length} {{\n                f{idx} = 1;\n            }}\n        }} else {{\n            run_len_f{idx} = 0;\n        }}\n"
                ));
            }
            super::search_families::ArrayFeature::AnyGreater(threshold) => {
                loop_body.push_str(&format!(
                    "        if x > {threshold} {{\n            f{idx} = 1;\n        }}\n"
                ));
            }
            super::search_families::ArrayFeature::AnyLess(threshold) => {
                loop_body.push_str(&format!(
                    "        if x < {threshold} {{\n            f{idx} = 1;\n        }}\n"
                ));
            }
            super::search_families::ArrayFeature::AllGreater(threshold) => {
                init.push_str(&format!(
                    "    bad_f{idx}: i64 = 0;\n    count_f{idx}: i64 = 0;\n"
                ));
                loop_body.push_str(&format!(
                    "        count_f{idx} = count_f{idx} + 1;\n        if x <= {threshold} {{\n            bad_f{idx} = 1;\n        }}\n"
                ));
                post_loop.push_str(&format!(
                    "    if bad_f{idx} == 0 && count_f{idx} > 0 {{\n        f{idx} = 1;\n    }}\n"
                ));
            }
            super::search_families::ArrayFeature::AllLess(threshold) => {
                init.push_str(&format!(
                    "    bad_f{idx}: i64 = 0;\n    count_f{idx}: i64 = 0;\n"
                ));
                loop_body.push_str(&format!(
                    "        count_f{idx} = count_f{idx} + 1;\n        if x >= {threshold} {{\n            bad_f{idx} = 1;\n        }}\n"
                ));
                post_loop.push_str(&format!(
                    "    if bad_f{idx} == 0 && count_f{idx} > 0 {{\n        f{idx} = 1;\n    }}\n"
                ));
            }
        }
    }
    if needs_prev_value {
        loop_body.push_str("        prev_value = x;\n        has_prev_value = 1;\n");
    }

    let mut guards = String::new();
    for rule in rules {
        let mut guard = String::from("return 1;");
        for &feature_idx in rule.iter().rev() {
            guard = format!("if f{feature_idx} == 1 {{ {guard} }}");
        }
        guards.push_str(&format!("    {guard}\n"));
    }

    format!(
        "fn {fn_name}(arr: [i64]) -> i64 {{\n{decls}{init}    for x in arr {{\n{loop_body}    }}\n{post_loop}{guards}    return 0;\n}}\n"
    )
}

pub(super) fn code_string_subsequence_class_search(
    fn_name: &str,
    subsequences: &[Vec<String>],
) -> String {
    let mut flags = String::new();
    let mut checks = String::new();

    for (idx, subseq) in subsequences.iter().enumerate() {
        flags.push_str(&format!(
            "    done_{idx}: i64 = 0;\n    cursor_{idx}: i64 = 0;\n"
        ));
        let mut body = String::new();
        for (pos, token) in subseq.iter().enumerate() {
            let escaped = escape_string(token);
            body.push_str(&format!(
                "        found_{idx}_{pos}: i64 = 0;\n        search_{idx}_{pos}: i64 = cursor_{idx};\n        while search_{idx}_{pos} < s.len {{\n            if s[search_{idx}_{pos}] == \"{escaped}\" {{\n                found_{idx}_{pos} = 1;\n                cursor_{idx} = search_{idx}_{pos} + 1;\n                break;\n            }}\n            search_{idx}_{pos} = search_{idx}_{pos} + 1;\n        }}\n        if found_{idx}_{pos} == 1 {{\n            done_{idx} = done_{idx} + 1;\n        }}\n"
            ));
        }
        checks.push_str(&body);
        checks.push_str(&format!(
            "    if done_{idx} == {} {{\n        return 1;\n    }}\n",
            subseq.len()
        ));
    }

    format!("fn {fn_name}(s: string) -> i64 {{\n{flags}{checks}    return 0;\n}}\n")
}

fn escape_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
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

/// 2-arg stateful reducer: `f(state, arr) = state op g(arr)`. The
/// `op` is one of `+ - * min max`; `g` is one of `sum / max / min /
/// count_positive / count_zero / count_negative / all_zero`. This
/// is the per-tick shape the rest of the nsynth search catalogue
/// doesn't cover — see `docs/stateful_synthesis_status.md` Stage 1.
pub(super) fn code_stateful_reducer(
    fn_name: &str,
    state_arg: &str,
    arr_arg: &str,
    op_token: &str,
    reducer_kind: &str,
) -> String {
    let reduction = match reducer_kind {
        "sum" => format!(
            "    s: i64 = 0;\n    for v in {arr_arg} {{\n        s = s + v;\n    }}\n    r := s;\n",
            arr_arg = arr_arg
        ),
        "max" => format!(
            "    r: i64 = {arr_arg}[0];\n    for v in {arr_arg} {{\n        if v > r {{ r = v; }}\n    }}\n",
            arr_arg = arr_arg
        ),
        "min" => format!(
            "    r: i64 = {arr_arg}[0];\n    for v in {arr_arg} {{\n        if v < r {{ r = v; }}\n    }}\n",
            arr_arg = arr_arg
        ),
        "count_positive" => format!(
            "    s: i64 = 0;\n    for v in {arr_arg} {{\n        if v > 0 {{ s = s + 1; }}\n    }}\n    r := s;\n",
            arr_arg = arr_arg
        ),
        "count_zero" => format!(
            "    s: i64 = 0;\n    for v in {arr_arg} {{\n        if v == 0 {{ s = s + 1; }}\n    }}\n    r := s;\n",
            arr_arg = arr_arg
        ),
        "count_negative" => format!(
            "    s: i64 = 0;\n    for v in {arr_arg} {{\n        if v < 0 {{ s = s + 1; }}\n    }}\n    r := s;\n",
            arr_arg = arr_arg
        ),
        "all_zero" => {
            // Reducer that returns 1 if every element is zero, else 0.
            "    r: i64 = 1;\n    for v in arr {\n        if v != 0 { r = 0; }\n    }\n".to_string()
        }
        other => format!("    r: i64 = 0; // unknown reducer kind: {}\n", other),
    };
    // Combine: result = (state) op (reduction result).
    let combine = match op_token {
        "+" => format!("    return {state} + r;\n", state = state_arg),
        "-" => format!("    return {state} - r;\n", state = state_arg),
        "*" => format!("    return {state} * r;\n", state = state_arg),
        "min" => {
            if state_arg == arr_arg {
                "    if r > state { r = state; }\n    return r;\n".to_string()
            } else {
                format!("    if r > {state} {{ r = {state}; }}\n    return r;\n", state = state_arg)
            }
        }
        "max" => {
            if state_arg == arr_arg {
                "    if r < state { r = state; }\n    return r;\n".to_string()
            } else {
                format!("    if r < {state} {{ r = {state}; }}\n    return r;\n", state = state_arg)
            }
        }
        other => format!("    return r; // unknown op: {}\n", other),
    };
    format!(
        "fn {fn_name}({state_arg}: i64, {arr_arg}: [i64]) -> i64 {{\n{reduction}{combine}}}\n",
        fn_name = fn_name,
        state_arg = state_arg,
        arr_arg = arr_arg
    )
}

/// Emit Mog for `state = state OP1 r(a) OP2 r(b)` with two named
/// reducers (sum/max/min/count_*). Pattern is `state +/- ra +/- rb`
/// chained left-to-right.
pub(super) fn code_stateful_reducer_dual(
    fn_name: &str,
    state_arg: &str,
    arr_a: &str,
    arr_b: &str,
    op1: &str,
    reducer_a: &str,
    op2: &str,
    reducer_b: &str,
) -> String {
    let ra = reducer_body(arr_a, reducer_a, "ra");
    let rb = reducer_body(arr_b, reducer_b, "rb");
    let op1_tok = match op1 {
        "+" => "+",
        "-" => "-",
        _ => "+",
    };
    let op2_tok = match op2 {
        "+" => "+",
        "-" => "-",
        _ => "+",
    };
    format!(
        "fn {fn_name}({state_arg}: i64, {arr_a}: [i64], {arr_b}: [i64]) -> i64 {{\n{ra}{rb}    return {state_arg} {op1} ra {op2} rb;\n}}\n",
        fn_name = fn_name,
        state_arg = state_arg,
        arr_a = arr_a,
        arr_b = arr_b,
        ra = ra,
        rb = rb,
        op1 = op1_tok,
        op2 = op2_tok,
    )
}

/// Emit Mog for the event-modulated 3-arg reducer
/// `(state, event, arr) -> state`.
///
/// Combine kinds (as encoded in the search teacher):
///   * `add_arr`   — `state OP r`
///   * `mul_event` — `state OP event * r`
///   * `add_event` — `state OP r OP event`
///
/// Gate kinds (optional; empty = no gate):
///   * `event_gt_0` — `if event > 0 then combined else state`
///   * `event_eq_0` — `if event == 0 then state else combined`
pub(super) fn code_stateful_reducer_event(
    fn_name: &str,
    state_arg: &str,
    event_arg: &str,
    arr_arg: &str,
    combine: &str,
    op_token: &str,
    reducer_kind: &str,
    gate_kind: &str,
) -> String {
    let r_body = reducer_body(arr_arg, reducer_kind, "r");
    // Build the combined expression. The grammar we emit is one of:
    //   state + r
    //   state - r
    //   state + event * r
    //   state - event * r
    //   state + r + event
    //   state - r - event
    let combined_expr: String = match (combine, op_token) {
        ("add_arr", "+") => format!("{state} + r", state = state_arg),
        ("add_arr", "-") => format!("{state} - r", state = state_arg),
        ("mul_event", "+") => {
            format!("{state} + {event} * r", state = state_arg, event = event_arg)
        }
        ("mul_event", "-") => {
            format!("{state} - {event} * r", state = state_arg, event = event_arg)
        }
        ("add_event", "+") => {
            format!("{state} + r + {event}", state = state_arg, event = event_arg)
        }
        ("add_event", "-") => {
            format!("{state} - r - {event}", state = state_arg, event = event_arg)
        }
        _ => format!("{state} + r", state = state_arg),
    };
    // Apply the gate, if any. The gate branches on the event scalar.
    let return_body = match gate_kind {
        "" => format!("    return {expr};\n", expr = combined_expr),
        "event_gt_0" => format!(
            "    if {event} > 0 {{\n        return {expr};\n    }}\n    return {state};\n",
            event = event_arg,
            expr = combined_expr,
            state = state_arg,
        ),
        "event_eq_0" => format!(
            "    if {event} == 0 {{\n        return {state};\n    }}\n    return {expr};\n",
            event = event_arg,
            state = state_arg,
            expr = combined_expr,
        ),
        "event_le_0" => format!(
            "    if {event} <= 0 {{\n        return {expr};\n    }}\n    return {state};\n",
            event = event_arg,
            expr = combined_expr,
            state = state_arg,
        ),
        "event_lt_0" => format!(
            "    if {event} < 0 {{\n        return {expr};\n    }}\n    return {state};\n",
            event = event_arg,
            expr = combined_expr,
            state = state_arg,
        ),
        _ => format!("    return {expr};\n", expr = combined_expr),
    };
    format!(
        "fn {fn_name}({state_arg}: i64, {event_arg}: i64, {arr_arg}: [i64]) -> i64 {{\n{r_body}{return_body}}}\n",
        fn_name = fn_name,
        state_arg = state_arg,
        event_arg = event_arg,
        arr_arg = arr_arg,
        r_body = r_body,
        return_body = return_body,
    )
}

/// Emit Mog for the composite two-reducer event-modulated form
/// `f(state, event, arr) = state OP_OUTER (r_a(arr) OP_INNER r_b(arr))`.
/// The event scalar is accepted but ignored by design — the array
/// signal dominates.
pub(super) fn code_stateful_reducer_event_composite(
    fn_name: &str,
    state_arg: &str,
    event_arg: &str,
    arr_arg: &str,
    reducer_a: &str,
    reducer_b: &str,
    op_inner: &str,
    op_outer: &str,
) -> String {
    let ra = reducer_body(arr_arg, reducer_a, "ra");
    let rb = reducer_body(arr_arg, reducer_b, "rb");
    let op_inner_tok = match op_inner {
        "+" => "+",
        "-" => "-",
        _ => "+",
    };
    let op_outer_tok = match op_outer {
        "+" => "+",
        "-" => "-",
        _ => "+",
    };
    format!(
        "fn {fn_name}({state_arg}: i64, {event_arg}: i64, {arr_arg}: [i64]) -> i64 {{\n{ra}{rb}    return {state_arg} {op_outer} ra {op_inner_tok} rb;\n}}\n",
        fn_name = fn_name,
        state_arg = state_arg,
        event_arg = event_arg,
        arr_arg = arr_arg,
        ra = ra,
        rb = rb,
        op_outer = op_outer_tok,
        op_inner_tok = op_inner_tok,
    )
}

/// Emit Mog for `if pred(arr) then state = new_value else state`.
///
/// Predicates are encoded as a single pass over `arr` checking the
/// relevant property; new_value is one of the supported constant or
/// state-derived expressions.
pub(super) fn code_stateful_replace(
    fn_name: &str,
    pred: &str,
    new_value: &str,
) -> String {
    // Each predicate maps to a guard like `p = 0/1`. The condition
    // selects when the *new value* should be returned — i.e. when
    // the predicate holds. For "any_pos", p=1 means "found a
    // positive", so the condition is p == 1.
    let (guard_init, guard_cond) = match pred {
        "any_pos" => ("    p: i64 = 0;\n", "p == 1"),
        "any_neg" => ("    p: i64 = 0;\n", "p == 1"),
        "all_pos" => ("    p: i64 = 1;\n", "p == 1"),
        "all_neg" => ("    p: i64 = 1;\n", "p == 1"),
        "max_gt_zero" => ("    p: i64 = 0;\n", "p == 1"),
        "min_lt_zero" => ("    p: i64 = 0;\n", "p == 1"),
        "any_eq_zero" => ("    p: i64 = 0;\n", "p == 1"),
        "any_eq_neg1" => ("    p: i64 = 0;\n", "p == 1"),
        "any_eq_pos1" => ("    p: i64 = 0;\n", "p == 1"),
        _ => ("    p: i64 = 0;\n", "0 == 1"),
    };
    let guard_body = match pred {
        "any_pos" => "    for v in arr { if v > 0 { p = 1; } }\n",
        "any_neg" => "    for v in arr { if v < 0 { p = 1; } }\n",
        "all_pos" => {
            "    for v in arr { if v <= 0 { p = 0; } }\n"
        }
        "all_neg" => {
            "    for v in arr { if v >= 0 { p = 0; } }\n"
        }
        "max_gt_zero" => {
            "    p: i64 = 0;\n    for v in arr { if v > p { p = v; } }\n    if p > 0 { p = 1; } else { p = 0; }\n"
        }
        "min_lt_zero" => {
            "    p: i64 = 0;\n    for v in arr { if v < p { p = v; } }\n    if p < 0 { p = 1; } else { p = 0; }\n"
        }
        "any_eq_zero" => "    for v in arr { if v == 0 { p = 1; } }\n",
        "any_eq_neg1" => "    for v in arr { if v == -1 { p = 1; } }\n",
        "any_eq_pos1" => "    for v in arr { if v == 1 { p = 1; } }\n",
        _ => "",
    };
    let new_expr = match new_value {
        "max" => "    r: i64 = arr[0];\n    for v in arr { if v > r { r = v; } }\n",
        "min" => "    r: i64 = arr[0];\n    for v in arr { if v < r { r = v; } }\n",
        "first" => "    r: i64 = arr[0];\n",
        "last" => {
            "    r: i64 = 0;\n    for v in arr { r = v; }\n"
        }
        "zero" => "    r: i64 = 0;\n",
        "one" => "    r: i64 = 1;\n",
        "neg_one" => "    r: i64 = -1;\n",
        "state_plus_one" => "    r: i64 = state + 1;\n",
        "state_minus_one" => "    r: i64 = state - 1;\n",
        "neg_state" => "    r: i64 = -state;\n",
        _ => "    r: i64 = 0;\n",
    };
    // Note: guard_init may be unused for max_gt_zero/min_lt_zero
    // because their guard_body is self-contained. We always emit it
    // for symmetry; redundant init is harmless.
    let _ = guard_init;
    format!(
        "fn {fn_name}(state: i64, arr: [i64]) -> i64 {{\n{new_expr}{guard_init}{guard_body}    if {cond} {{\n        return r;\n    }}\n    return state;\n}}\n",
        fn_name = fn_name,
        new_expr = new_expr,
        guard_init = guard_init,
        guard_body = guard_body,
        cond = guard_cond,
    )
}

/// Helper: emit the body of a single reducer into a result variable.
fn reducer_body(arr: &str, reducer: &str, result_var: &str) -> String {
    match reducer {
        "sum" => format!(
            "    {result_var}: i64 = 0;\n    for v in {arr} {{\n        {result_var} = {result_var} + v;\n    }}\n",
            result_var = result_var, arr = arr
        ),
        "max" => format!(
            "    {result_var}: i64 = {arr}[0];\n    for v in {arr} {{\n        if v > {result_var} {{ {result_var} = v; }}\n    }}\n",
            result_var = result_var, arr = arr
        ),
        "min" => format!(
            "    {result_var}: i64 = {arr}[0];\n    for v in {arr} {{\n        if v < {result_var} {{ {result_var} = v; }}\n    }}\n",
            result_var = result_var, arr = arr
        ),
        "count_positive" => format!(
            "    {result_var}: i64 = 0;\n    for v in {arr} {{\n        if v > 0 {{ {result_var} = {result_var} + 1; }}\n    }}\n",
            result_var = result_var, arr = arr
        ),
        "count_zero" => format!(
            "    {result_var}: i64 = 0;\n    for v in {arr} {{\n        if v == 0 {{ {result_var} = {result_var} + 1; }}\n    }}\n",
            result_var = result_var, arr = arr
        ),
        "count_negative" => format!(
            "    {result_var}: i64 = 0;\n    for v in {arr} {{\n        if v < 0 {{ {result_var} = {result_var} + 1; }}\n    }}\n",
            result_var = result_var, arr = arr
        ),
        other => format!(
            "    {result_var}: i64 = 0; // unknown reducer: {}\n",
            other
        ),
    }
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

/// **Stage 2: Tensor Broadcast Teacher Code Generator**
///
/// Generates code that replicates a scalar input across a fixed-size array.
/// Template: broadcast a scalar by reading it and returning it as array elements.
/// This is a placeholder; real tensor operations will be defined in tensor_codegen.rs.
pub(super) fn code_broadcast_pattern(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(scalar: i64) -> [i64] {
    result: [i64] = [scalar, scalar, scalar, scalar];
    return result;
}
"#,
        fn_name,
    )
}

/// **Stage 2: Tensor Dot Product Teacher Code Generator**
///
/// Generates code for element-wise multiplication and summation.
/// Computes: result = sum(a[i] * b[i] for all i).
pub(super) fn code_dot_product_search(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(a: [i64], b: [i64]) -> i64 {
    acc: i64 = 0;
    i: i64 = 0;
    while i < a.len() {
        acc = acc + (a[i] * b[i]);
        i = i + 1;
    }
    return acc;
}
"#,
        fn_name,
    )
}

/// **Stage 2: Tensor Matrix Multiplication Teacher Code Generator**
///
/// Generates code for N×M @ M×K -> N×K matrix multiplication.
/// Assumes flattened row-major layout: a is N*M elements, b is M*K, result is N*K.
/// Computes: result[i][j] = sum(a[i][k] * b[k][j] for k in 0..M).
pub(super) fn code_matmul_template(fn_name: &str, n: usize, m: usize, k: usize) -> String {
    format!(
        r#"fn {fn_name}(a: [i64], b: [i64]) -> [i64] {{
    result: [i64] = [];
    i: i64 = 0;
    while i < {n} {{
        j: i64 = 0;
        while j < {k} {{
            acc: i64 = 0;
            l: i64 = 0;
            while l < {m} {{
                acc = acc + (a[(i * {m}) + l] * b[(l * {k}) + j]);
                l = l + 1;
            }}
            result.push(acc);
            j = j + 1;
        }}
        i = i + 1;
    }}
    return result;
}}
"#,
        fn_name = fn_name,
        n = n,
        k = k,
        m = m
    )
}

/// Stage 3: struct-of-state with independent field reductions.
/// Each field evolves as: field_new = field_old OP reducer(arr)
/// where each field may use a different reducer (sum, max, min, count_*).
pub(super) fn code_struct_field_reduction(
    fn_name: &str,
    state_arg: &str,
    arr_arg: &str,
    fields: &[(&str, &str, &str)],
) -> String {
    let mut field_updates = String::new();
    for (field_name, op, reducer) in fields {
        let reduce_fn = match *reducer {
            "sum" => "arr.iter().sum::<i64>()",
            "max" => "arr.iter().copied().max().unwrap_or(0)",
            "min" => "arr.iter().copied().min().unwrap_or(0)",
            "count_positive" => "arr.iter().filter(|&&x| x > 0).count() as i64",
            "count_negative" => "arr.iter().filter(|&&x| x < 0).count() as i64",
            "count_zero" => "arr.iter().filter(|&&x| x == 0).count() as i64",
            _ => "0",
        };
        let update = format!(
            "    s.{field_name} = s.{field_name} {op} ({reduce_fn});\n"
        );
        field_updates.push_str(&update);
    }

    format!(
        "fn {fn_name}({state_arg}: State, {arr_arg}: [i64]) -> State {{\n\
        mut s: State = {state_arg};\n\
        {field_updates}\
        return s;\n\
        }}\n"
    )
}

/// Stage 3: struct-of-state with coupled field dependencies.
/// Two fields evolve with mutual dependence: both read from each other
/// and from reductions. Pattern: f1_new = f1_old OP1 r1(arr), f2_new = f2_old OP2 r2(arr)
/// where cross-field coupling can appear (e.g., f1 and f2 both gated by the same condition).
pub(super) fn code_struct_coupled_fields(
    fn_name: &str,
    state_arg: &str,
    arr_arg: &str,
    field1: &str,
    op1: &str,
    reducer1: &str,
    field2: &str,
    op2: &str,
    reducer2: &str,
) -> String {
    let reduce1_fn = match reducer1 {
        "sum" => "arr.iter().sum::<i64>()",
        "max" => "arr.iter().copied().max().unwrap_or(0)",
        "min" => "arr.iter().copied().min().unwrap_or(0)",
        "count_positive" => "arr.iter().filter(|&&x| x > 0).count() as i64",
        "count_negative" => "arr.iter().filter(|&&x| x < 0).count() as i64",
        "count_zero" => "arr.iter().filter(|&&x| x == 0).count() as i64",
        _ => "0",
    };

    let reduce2_fn = match reducer2 {
        "sum" => "arr.iter().sum::<i64>()",
        "max" => "arr.iter().copied().max().unwrap_or(0)",
        "min" => "arr.iter().copied().min().unwrap_or(0)",
        "count_positive" => "arr.iter().filter(|&&x| x > 0).count() as i64",
        "count_negative" => "arr.iter().filter(|&&x| x < 0).count() as i64",
        "count_zero" => "arr.iter().filter(|&&x| x == 0).count() as i64",
        _ => "0",
    };

    let apply_op = |op: &str, lhs: &str, rhs: &str| -> String {
        match op {
            "+" => format!("{lhs} + {rhs}"),
            "-" => format!("{lhs} - {rhs}"),
            "*" => format!("{lhs} * {rhs}"),
            "min" => format!("{lhs}.min({rhs})"),
            "max" => format!("{lhs}.max({rhs})"),
            _ => lhs.to_string(),
        }
    };

    let update1 = apply_op(op1, &format!("s.{field1}"), reduce1_fn);
    let update2 = apply_op(op2, &format!("s.{field2}"), reduce2_fn);

    format!(
        "fn {fn_name}({state_arg}: State, {arr_arg}: [i64]) -> State {{\n\
        mut s: State = {state_arg};\n\
        s.{field1} = {update1};\n\
        s.{field2} = {update2};\n\
        return s;\n\
        }}\n"
    )
}

/// Stage 3: struct-of-state with conditional field logic.
/// field_new = if cond(arr) then field_true(field_old, arr) else field_false(field_old, arr)
/// Captures gated updates and selective state mutations based on array properties.
pub(super) fn code_struct_conditional_fields(
    fn_name: &str,
    state_arg: &str,
    arr_arg: &str,
    field_name: &str,
    condition: &str,
    update_true: &str,
    update_false: &str,
) -> String {
    let cond_code = match condition {
        "any_positive" => format!("{arr_arg}.iter().any(|&x| x > 0)"),
        "all_positive" => format!("!{arr_arg}.is_empty() && {arr_arg}.iter().all(|&x| x > 0)"),
        "any_negative" => format!("{arr_arg}.iter().any(|&x| x < 0)"),
        "any_zero" => format!("{arr_arg}.iter().any(|&x| x == 0)"),
        "is_empty" => format!("{arr_arg}.is_empty()"),
        "sum_positive" => format!("{arr_arg}.iter().sum::<i64>() > 0"),
        _ => "true".to_string(),
    };

    format!(
        "fn {fn_name}({state_arg}: State, {arr_arg}: [i64]) -> State {{\n\
        mut s: State = {state_arg};\n\
        if {cond_code} {{\n\
            s.{field_name} = {update_true};\n\
        }} else {{\n\
            s.{field_name} = {update_false};\n\
        }}\n\
        return s;\n\
        }}\n"
    )
}

/// Emit Mog for temporal stateful reducer: state combined with reduction of array,
/// parameterized by time variable t.
            "    r: i64 = {arr_arg}[0];\n    for v in {arr_arg} {{\n        if v > r {{ r = v; }}\n    }}\n",
            arr_arg = arr_arg
        ),
        "min" => format!(
            "    r: i64 = {arr_arg}[0];\n    for v in {arr_arg} {{\n        if v < r {{ r = v; }}\n    }}\n",
            arr_arg = arr_arg
        ),
        "count_positive" => format!(
            "    s: i64 = 0;\n    for v in {arr_arg} {{\n        if v > 0 {{ s = s + 1; }}\n    }}\n    r := s;\n",
            arr_arg = arr_arg
        ),
        _ => format!("    r: i64 = 0; // reducer: {}\n", reducer_kind),
    };

    // Time adjustment
    let time_adjust = match time_kind {
        "exponential_decay" => match time_op {
            "multiply" => format!("    r = r * {time_arg} / ({time_arg} + 1);\n", time_arg = time_arg),
            "divide" => format!("    r = r / ({time_arg} + 1);\n", time_arg = time_arg),
            _ => format!("    r = r * {time_arg};\n", time_arg = time_arg),
        },
        "polynomial" => match time_op {
            "power2" => format!("    r = r * {time_arg} * {time_arg};\n", time_arg = time_arg),
            "power3" => format!("    r = r * {time_arg} * {time_arg} * {time_arg};\n", time_arg = time_arg),
            _ => format!("    r = r * {time_arg};\n", time_arg = time_arg),
        },
        _ => format!("    // time_kind: {}\n", time_kind),
    };

    // Combine with state
    let combine = match op_state {
        "+" => format!("    return {state} + r;\n", state = state_arg),
        "-" => format!("    return {state} - r;\n", state = state_arg),
        "*" => format!("    return {state} * r;\n", state = state_arg),
        _ => format!("    return r;\n"),
    };

    format!(
        "fn {fn_name}({state_arg}: i64, {time_arg}: i64, {arr_arg}: [i64]) -> i64 {{\n{reduction}{time_adjust}{combine}}}\n",
        fn_name = fn_name,
        state_arg = state_arg,
        time_arg = time_arg,
        arr_arg = arr_arg,
    )
}

/// Emit Mog for temporal stateful update without reducer: pure state + time transformation.
pub(super) fn code_stateful_reducer_temporal_no_reducer(
    fn_name: &str,
    state_arg: &str,
    time_arg: &str,
    op_state: &str,
    time_kind: &str,
) -> String {
    // Time transformation
    let time_expr = match time_kind {
        "exponential_decay" => format!("{time_arg} / ({time_arg} + 1)", time_arg = time_arg),
        "polynomial_linear" => format!("{time_arg}", time_arg = time_arg),
        "polynomial_square" => format!("{time_arg} * {time_arg}", time_arg = time_arg),
        "polynomial_cube" => format!("{time_arg} * {time_arg} * {time_arg}", time_arg = time_arg),
        _ => format!("{time_arg}", time_arg = time_arg),
    };

    // Combine state with time
    let result = match op_state {
        "+" => format!("    return {state} + {time_expr};\n", state = state_arg, time_expr = time_expr),
        "-" => format!("    return {state} - {time_expr};\n", state = state_arg, time_expr = time_expr),
        "*" => format!("    return {state} * {time_expr};\n", state = state_arg, time_expr = time_expr),
        "divide" => format!("    return {state} / ({time_expr} + 1);\n", state = state_arg, time_expr = time_expr),
        _ => format!("    return {state};\n", state = state_arg),
    };

    format!(
        "fn {fn_name}({state_arg}: i64, {time_arg}: i64) -> i64 {{\n{result}}}\n",
        fn_name = fn_name,
        state_arg = state_arg,
        time_arg = time_arg,
    )
}

