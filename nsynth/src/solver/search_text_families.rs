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

/// General disjunctive-suffix classifier learner.
///
/// Hypothesis class: `s.ends_with(suf_1) || ... || s.ends_with(suf_k) -> 1 else 0`.
///
/// Unlike the single-literal teachers (`search_starts_with_literal`,
/// `search_contains_literal`), this discovers a *learned disjunction* of suffixes
/// from labeled `(string -> 0/1)` examples. It is the first teacher whose hypothesis
/// class can express a real morphological rule — e.g. English `-es` pluralization
/// after a sibilant (s / x / z / ch / sh).
///
/// Method: greedy clean-suffix set cover. A candidate suffix is *admissible* only if
/// it never fires on a negative example (no false positives). We greedily union
/// admissible suffixes — always taking the one covering the most still-uncovered
/// positives — until every positive is covered. If the positives cannot be covered by
/// admissible suffixes alone, the rule is not a suffix class and we return None.
/// The resulting program is correct on the training set by construction; `verified_result`
/// re-checks it and the solver re-checks held-out examples.
pub(super) fn search_suffix_class(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    const MAX_SUFFIX_LEN: usize = 3;
    const MAX_DISJUNCTS: usize = 8;

    let strings = unary_string_examples(problem)?;

    // Labels must be binary 0/1.
    if !problem
        .examples
        .iter()
        .all(|e| e.expected == 0 || e.expected == 1)
    {
        return None;
    }

    let positives: Vec<&String> = problem
        .examples
        .iter()
        .zip(strings.iter())
        .filter(|(e, _)| e.expected == 1)
        .map(|(_, s)| s)
        .collect();
    let negatives: Vec<&String> = problem
        .examples
        .iter()
        .zip(strings.iter())
        .filter(|(e, _)| e.expected == 0)
        .map(|(_, s)| s)
        .collect();
    // Need both classes present to learn a discriminative rule.
    if positives.is_empty() || negatives.is_empty() {
        return None;
    }

    // Candidate suffixes (len 1..=MAX_SUFFIX_LEN) mined from the positive examples.
    let mut candidates: Vec<String> = Vec::new();
    for p in &positives {
        let chars: Vec<char> = p.chars().collect();
        for len in 1..=MAX_SUFFIX_LEN.min(chars.len()) {
            let suffix: String = chars[chars.len() - len..].iter().collect();
            candidates.push(suffix);
        }
    }
    // Deterministic order: shorter (more general) first, then lexicographic.
    candidates.sort_by(|a, b| {
        a.chars()
            .count()
            .cmp(&b.chars().count())
            .then_with(|| a.cmp(b))
    });
    candidates.dedup();

    // Admissible = never fires on a negative (no false positives).
    let admissible: Vec<String> = candidates
        .into_iter()
        .filter(|suf| !negatives.iter().any(|n| n.ends_with(suf.as_str())))
        .collect();
    if admissible.is_empty() {
        return None;
    }

    // Greedy set cover over the positives using admissible suffixes.
    let mut uncovered: Vec<bool> = vec![true; positives.len()];
    let mut chosen: Vec<String> = Vec::new();
    while uncovered.iter().any(|&u| u) && chosen.len() < MAX_DISJUNCTS {
        let mut best: Option<(usize, &String)> = None;
        for suf in &admissible {
            if chosen.iter().any(|c| c == suf) {
                continue;
            }
            let cover = positives
                .iter()
                .enumerate()
                .filter(|(i, p)| uncovered[*i] && p.ends_with(suf.as_str()))
                .count();
            if cover == 0 {
                continue;
            }
            match best {
                Some((best_cover, _)) if cover <= best_cover => {}
                _ => best = Some((cover, suf)),
            }
        }
        // No admissible suffix covers any remaining positive -> not a clean suffix class.
        let (_, suf) = best?;
        for (i, p) in positives.iter().enumerate() {
            if p.ends_with(suf.as_str()) {
                uncovered[i] = false;
            }
        }
        chosen.push(suf.clone());
    }
    if uncovered.iter().any(|&u| u) {
        return None;
    }

    chosen.sort();
    let code = code_suffix_class_search(fn_name, &chosen);
    verified_result(problem, code, "search_suffix_class")
}
