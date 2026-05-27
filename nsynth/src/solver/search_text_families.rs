use super::search_codegen::*;
use super::search_runtime::count_words;
use super::*;

pub(super) fn search_trimmed_len(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    if unary_string_examples(problem).is_none() {
        return None;
    }
    if !validate_unary_str(problem, |s| s.trim().chars().count() as i64) {
        return None;
    }
    verified_result(problem, code_trimmed_len(fn_name), "search_trimmed_len")
}

pub(super) fn search_contains_literal(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let strings = unary_string_examples(problem)?;
    let mut candidates = Vec::new();
    for (example, value) in problem.examples.iter().zip(strings.iter()) {
        if example.expected != 1 {
            continue;
        }
        let chars = value.chars().collect::<Vec<_>>();
        for start in 0..chars.len() {
            for end in (start + 1)..=chars.len().min(start + 4) {
                candidates.push(chars[start..end].iter().collect::<String>());
            }
        }
    }
    candidates.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    candidates.dedup();

    for candidate in candidates {
        let matches = problem
            .examples
            .iter()
            .zip(strings.iter())
            .all(|(example, value)| {
                (if value.contains(&candidate) { 1 } else { 0 }) == example.expected
            });
        if !matches {
            continue;
        }
        let code = code_contains_literal_search(fn_name, &candidate);
        if let Some(result) = verified_result(problem, code, "search_contains_literal") {
            return Some(result);
        }
    }
    None
}

pub(super) fn search_starts_with_literal(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let strings = unary_string_examples(problem)?;
    let mut candidates = Vec::new();
    for (example, value) in problem.examples.iter().zip(strings.iter()) {
        if example.expected != 1 {
            continue;
        }
        let chars = value.chars().collect::<Vec<_>>();
        for end in 1..=chars.len().min(4) {
            candidates.push(chars[..end].iter().collect::<String>());
        }
    }
    candidates.sort();
    candidates.dedup();

    for candidate in candidates {
        let matches = problem
            .examples
            .iter()
            .zip(strings.iter())
            .all(|(example, value)| {
                (if value.starts_with(&candidate) { 1 } else { 0 }) == example.expected
            });
        if !matches {
            continue;
        }
        let code = code_starts_with_literal_search(fn_name, &candidate);
        if let Some(result) = verified_result(problem, code, "search_starts_with_literal") {
            return Some(result);
        }
    }
    None
}

pub(super) fn search_vowel_count(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    if unary_string_examples(problem).is_none() {
        return None;
    }
    if !validate_unary_str(problem, |s| {
        s.chars()
            .filter(|c| matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u'))
            .count() as i64
    }) {
        return None;
    }
    verified_result(problem, code_vowel_count(fn_name), "search_vowel_count")
}

pub(super) fn search_count_words(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    if unary_string_examples(problem).is_none() {
        return None;
    }
    if !validate_unary_str(problem, count_words) {
        return None;
    }
    verified_result(problem, code_count_words(fn_name), "search_count_words")
}

pub(super) fn search_palindrome(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    if unary_string_examples(problem).is_none() {
        return None;
    }
    if !validate_unary_str(problem, |s| {
        let chars: Vec<char> = s.chars().collect();
        if chars.iter().eq(chars.iter().rev()) {
            1
        } else {
            0
        }
    }) {
        return None;
    }
    verified_result(problem, code_palindrome_check(fn_name), "search_palindrome")
}
