use super::*;

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
