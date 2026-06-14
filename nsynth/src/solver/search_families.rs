use super::search_codegen::*;
use super::search_runtime::*;
use super::*;

/// Minimum examples for the general membership/DNF array classifiers to fire.
/// Every structural benchmark problem has <= 10 examples; requiring more keeps
/// these general (and easily-overfit) teachers from shadowing the exact solvers,
/// while the curriculum language-classification tasks (30+ examples) are well
/// above it.
const MIN_CLASSIFIER_EXAMPLES: usize = 12;

pub(super) fn search_struct_pair_patterns(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    let ParamType::Other(type_name) = param_types.first()?.clone() else {
        return None;
    };
    let _pairs = unary_pair_examples(problem)?;

    if type_name == "Point" && validate_unary_pair(problem, |x, y| x + y) {
        return verified_result(problem, code_point_sum(fn_name), "search_struct_pair");
    }
    if type_name == "Rectangle" && validate_unary_pair(problem, |w, h| w * h) {
        return verified_result(problem, code_rectangle_area(fn_name), "search_struct_pair");
    }
    None
}

pub(super) fn search_closure_map_sum(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| arr.iter().map(|x| x * 2).sum()) {
        return None;
    }
    verified_result(
        problem,
        code_closure_map_sum(fn_name),
        "search_closure_map_sum",
    )
}

pub(super) fn search_max_pair_diff(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        arr.windows(2)
            .map(|w| (w[0] - w[1]).abs())
            .max()
            .unwrap_or(0)
    }) {
        return None;
    }
    verified_result(problem, code_max_pair_diff(fn_name), "search_max_pair_diff")
}

pub(super) fn search_array_item_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);

    match param_types.as_slice() {
        [ParamType::ArrayI64] => {
            if validate_unary_array(problem, |arr| arr.iter().sum()) {
                return verified_result(problem, code_array_sum(fn_name), "search_array_sum");
            }
            if validate_unary_array(problem, |arr| *arr.iter().max().unwrap_or(&0)) {
                return verified_result(problem, code_array_max(fn_name), "search_array_max");
            }
            if validate_unary_array(problem, |arr| arr.iter().filter(|x| **x > 0).count() as i64) {
                return verified_result(
                    problem,
                    code_count_positive(fn_name),
                    "search_array_count_positive",
                );
            }
            if validate_unary_array(problem, |arr| arr.iter().filter(|x| **x < 0).sum()) {
                return verified_result(
                    problem,
                    code_sum_negatives(fn_name),
                    "search_array_sum_negatives",
                );
            }
        }
        [ParamType::ArrayI64, ParamType::I64] => {
            if validate_array_and_int(problem, |arr, target| {
                arr.iter().filter(|x| **x == target).count() as i64
            }) {
                return verified_result(
                    problem,
                    code_count_occurrences(fn_name),
                    "search_array_count_occurrences",
                );
            }
            if validate_array_and_int(problem, |arr, k| {
                arr.iter().filter(|&&x| x > k).count() as i64
            }) {
                return verified_result(
                    problem,
                    code_count_greater_than(fn_name),
                    "search_array_count_greater_than",
                );
            }
            if validate_array_and_int(problem, |arr, k| arr.iter().take(k as usize).sum()) {
                return verified_result(problem, code_prefix_sum_k(fn_name), "search_prefix_sum_k");
            }
        }
        _ => {}
    }

    None
}

pub(super) fn search_run_length_decode_sum(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        let mut total = 0i64;
        let mut i = 0usize;
        while i + 1 < arr.len() {
            total += arr[i] * arr[i + 1];
            i += 2;
        }
        total
    }) {
        return None;
    }
    verified_result(
        problem,
        code_run_length_decode_sum(fn_name),
        "search_run_length_decode_sum",
    )
}

pub(super) fn search_count_adjacent_diff(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        let mut count = 0i64;
        for i in 1..arr.len() {
            if arr[i] != arr[i - 1] {
                count += 1;
            }
        }
        count
    }) {
        return None;
    }
    verified_result(
        problem,
        code_count_adjacent_diff(fn_name),
        "search_count_adjacent_diff",
    )
}

pub(super) fn search_second_max(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, second_max) {
        return None;
    }
    verified_result(problem, code_second_max(fn_name), "search_second_max")
}

pub(super) fn search_array_range(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, array_range) {
        return None;
    }
    verified_result(problem, code_array_range(fn_name), "search_array_range")
}

/// General "array contains any of a learned constant set" classifier.
///
/// The array analog of `search_suffix_class`: hypothesis class is
/// `(arr contains c_1 || ... || arr contains c_k) -> 1 else 0`. Mines candidate
/// constants from positive arrays, keeps only *admissible* ones (a constant that
/// never appears in a negative array — no false positives), then greedy
/// set-covers the positives.
///
/// This is what lets nsynth classify morpheme-tokenized sentences: feed the
/// token-id array of a sentence, and the rule "grammatical iff the array carries
/// a valid inflection-suffix token id" is exactly a member-class disjunction.
pub(super) fn search_array_member_class(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    const MAX_DISJUNCTS: usize = 8;

    let arrays = unary_array_examples(problem)?;
    // A membership/DNF rule from few examples is untrustworthy and would
    // shadow exact structural solvers; require a substantial example set.
    if problem.examples.len() < MIN_CLASSIFIER_EXAMPLES {
        return None;
    }
    if !problem
        .examples
        .iter()
        .all(|e| e.expected_int() == 0 || e.expected_int() == 1)
    {
        return None;
    }

    let positives: Vec<&Vec<i64>> = problem
        .examples
        .iter()
        .zip(arrays.iter())
        .filter(|(e, _)| e.expected_int() == 1)
        .map(|(_, a)| a)
        .collect();
    let negatives: Vec<&Vec<i64>> = problem
        .examples
        .iter()
        .zip(arrays.iter())
        .filter(|(e, _)| e.expected_int() == 0)
        .map(|(_, a)| a)
        .collect();
    if positives.is_empty() || negatives.is_empty() {
        return None;
    }

    // Candidate constants = values that occur in some positive array.
    let mut candidates: Vec<i64> = positives.iter().flat_map(|a| a.iter().copied()).collect();
    candidates.sort_unstable();
    candidates.dedup();

    // Admissible = never occurs in a negative array (no false positive).
    let admissible: Vec<i64> = candidates
        .into_iter()
        .filter(|c| !negatives.iter().any(|n| n.contains(c)))
        .collect();
    if admissible.is_empty() {
        return None;
    }

    // Greedy set cover over positives by membership.
    let mut uncovered: Vec<bool> = vec![true; positives.len()];
    let mut chosen: Vec<i64> = Vec::new();
    while uncovered.iter().any(|&u| u) && chosen.len() < MAX_DISJUNCTS {
        let mut best: Option<(usize, i64)> = None;
        for &c in &admissible {
            if chosen.contains(&c) {
                continue;
            }
            let cover = positives
                .iter()
                .enumerate()
                .filter(|(i, p)| uncovered[*i] && p.contains(&c))
                .count();
            if cover == 0 {
                continue;
            }
            match best {
                Some((best_cover, _)) if cover <= best_cover => {}
                _ => best = Some((cover, c)),
            }
        }
        let (_, c) = best?;
        for (i, p) in positives.iter().enumerate() {
            if p.contains(&c) {
                uncovered[i] = false;
            }
        }
        chosen.push(c);
    }
    if uncovered.iter().any(|&u| u) {
        return None;
    }

    chosen.sort_unstable();
    let code = code_array_member_class_search(fn_name, &chosen);
    verified_result(problem, code, "search_array_member_class")
}

/// Conjunctive membership classifier: label 1 iff the array contains ALL of a
/// learned *required* set of tokens AND NONE of a learned *forbidden* set.
///
/// This is strictly more expressive than `search_array_member_class` (an OR over
/// memberships) — it can express conjunctions like "grammatical gerund iff the
/// token stream carries both the auxiliary `is` AND the `<+ing>` suffix", which a
/// pure OR cannot. (Errors that hinge on a wrong *stem*, e.g. "copyed", are not a
/// function of the token set at all and remain out of reach — that needs lexical
/// or positional features.)
pub(super) fn search_array_conjunction(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    use std::collections::HashSet;

    let arrays = unary_array_examples(problem)?;
    // A membership/DNF rule from few examples is untrustworthy and would
    // shadow exact structural solvers; require a substantial example set.
    if problem.examples.len() < MIN_CLASSIFIER_EXAMPLES {
        return None;
    }
    if !problem
        .examples
        .iter()
        .all(|e| e.expected_int() == 0 || e.expected_int() == 1)
    {
        return None;
    }

    let sets: Vec<HashSet<i64>> = arrays.iter().map(|a| a.iter().copied().collect()).collect();
    let pos: Vec<&HashSet<i64>> = sets
        .iter()
        .zip(problem.examples.iter())
        .filter(|(_, e)| e.expected_int() == 1)
        .map(|(s, _)| s)
        .collect();
    let neg: Vec<&HashSet<i64>> = sets
        .iter()
        .zip(problem.examples.iter())
        .filter(|(_, e)| e.expected_int() == 0)
        .map(|(s, _)| s)
        .collect();
    if pos.is_empty() || neg.is_empty() {
        return None;
    }

    // required = tokens present in EVERY positive.
    let mut required: Vec<i64> = pos[0]
        .iter()
        .copied()
        .filter(|t| pos.iter().all(|p| p.contains(t)))
        .collect();
    // forbidden = tokens present in some negative but in NO positive.
    let pos_union: HashSet<i64> = pos.iter().flat_map(|p| p.iter().copied()).collect();
    let mut forbidden: Vec<i64> = neg
        .iter()
        .flat_map(|n| n.iter().copied())
        .filter(|t| !pos_union.contains(t))
        .collect::<HashSet<i64>>()
        .into_iter()
        .collect();

    let predicate = |set: &HashSet<i64>, req: &[i64], forb: &[i64]| -> bool {
        req.iter().all(|t| set.contains(t)) && !forb.iter().any(|t| set.contains(t))
    };
    let separates = |req: &[i64], forb: &[i64]| -> bool {
        sets.iter().zip(problem.examples.iter()).all(|(s, e)| {
            (predicate(s, req, forb) as i64) == e.expected_int()
        })
    };

    if !separates(&required, &forbidden) {
        return None;
    }

    // Minimize: drop any required/forbidden token whose removal still separates.
    required.sort_unstable();
    forbidden.sort_unstable();
    let mut i = required.len();
    while i > 0 {
        i -= 1;
        let t = required.remove(i);
        if !separates(&required, &forbidden) {
            required.insert(i, t);
        }
    }
    let mut j = forbidden.len();
    while j > 0 {
        j -= 1;
        let t = forbidden.remove(j);
        if !separates(&required, &forbidden) {
            forbidden.insert(j, t);
        }
    }
    // A bare "always 1" (no required, no forbidden) is not a meaningful rule.
    if required.is_empty() && forbidden.is_empty() {
        return None;
    }

    let code = code_array_conjunction_search(fn_name, &required, &forbidden);
    verified_result(problem, code, "search_array_conjunction")
}

/// DNF membership classifier (separate-and-conquer rule learner).
///
/// Learns label = 1 iff the array matches ANY of a small set of conjunctive
/// rules, each an AND of membership literals ("contains T" / "does not contain
/// T"). This is the keystone array classifier: it subsumes both
/// `search_array_member_class` (a disjunction of single positive literals) and
/// `search_array_conjunction` (a single conjunction), and can express genuine
/// DNF — e.g. logical-argument validity = "(modus ponens pattern) OR (modus
/// tollens pattern)", which neither simpler teacher can.
///
/// Each round greedily grows one pure conjunction (covering only positives) by
/// adding the literal that best removes still-matched negatives while retaining
/// positives, then removes the positives it covers and repeats until all are
/// covered. Standard separate-and-conquer (RIPPER-style) induction.
pub(super) fn search_array_dnf(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    use std::collections::HashSet;

    let arrays = unary_array_examples(problem)?;
    // A membership/DNF rule from few examples is untrustworthy and would
    // shadow exact structural solvers; require a substantial example set.
    if problem.examples.len() < MIN_CLASSIFIER_EXAMPLES {
        return None;
    }
    if !problem
        .examples
        .iter()
        .all(|e| e.expected_int() == 0 || e.expected_int() == 1)
    {
        return None;
    }
    let sets: Vec<HashSet<i64>> = arrays.iter().map(|a| a.iter().copied().collect()).collect();
    let labels: Vec<i64> = problem.examples.iter().map(|e| e.expected_int()).collect();
    let pos_idx: Vec<usize> = (0..sets.len()).filter(|&i| labels[i] == 1).collect();
    let neg_idx: Vec<usize> = (0..sets.len()).filter(|&i| labels[i] == 0).collect();
    if pos_idx.is_empty() || neg_idx.is_empty() {
        return None;
    }

    // Candidate tokens: any token appearing in a positive (literals are built over these).
    let mut tokens: Vec<i64> = pos_idx
        .iter()
        .flat_map(|&i| sets[i].iter().copied())
        .collect::<HashSet<i64>>()
        .into_iter()
        .collect();
    tokens.sort_unstable();

    let lit_matches = |set: &HashSet<i64>, (tok, want): (i64, bool)| set.contains(&tok) == want;
    let rule_matches =
        |set: &HashSet<i64>, rule: &[(i64, bool)]| rule.iter().all(|&l| lit_matches(set, l));

    let mut rules: Vec<Vec<(i64, bool)>> = Vec::new();
    let mut covered = vec![false; sets.len()];

    const MAX_RULES: usize = 8;
    const MAX_LITS: usize = 6;
    while pos_idx.iter().any(|&i| !covered[i]) && rules.len() < MAX_RULES {
        // Grow one pure conjunction over the still-uncovered positives.
        let mut rule: Vec<(i64, bool)> = Vec::new();
        loop {
            let neg_hit: Vec<usize> = neg_idx
                .iter()
                .copied()
                .filter(|&i| rule_matches(&sets[i], &rule))
                .collect();
            if neg_hit.is_empty() {
                break; // pure
            }
            if rule.len() >= MAX_LITS {
                return None;
            }
            // Pick the literal maximizing kept-uncovered-positives, then fewest
            // negatives remaining, that makes strict progress on negatives.
            let mut best: Option<(usize, usize, (i64, bool))> = None;
            for &tok in &tokens {
                for want in [true, false] {
                    let lit = (tok, want);
                    if rule.contains(&lit) {
                        continue;
                    }
                    let pos_keep = pos_idx
                        .iter()
                        .filter(|&&i| !covered[i] && rule_matches(&sets[i], &rule) && lit_matches(&sets[i], lit))
                        .count();
                    if pos_keep == 0 {
                        continue;
                    }
                    let neg_keep = neg_hit.iter().filter(|&&i| lit_matches(&sets[i], lit)).count();
                    if neg_keep >= neg_hit.len() {
                        continue; // no progress on negatives
                    }
                    let score = (pos_keep, usize::MAX - neg_keep);
                    if best.map(|(bp, bn, _)| score > (bp, bn)).unwrap_or(true) {
                        best = Some((score.0, score.1, lit));
                    }
                }
            }
            let (_, _, lit) = best?; // cannot purify -> not DNF-separable here
            rule.push(lit);
        }
        // Which uncovered positives does this pure rule cover?
        let newly: Vec<usize> = pos_idx
            .iter()
            .copied()
            .filter(|&i| !covered[i] && rule_matches(&sets[i], &rule))
            .collect();
        if newly.is_empty() {
            return None;
        }
        for i in newly {
            covered[i] = true;
        }
        // Minimal-ish: drop literals whose removal keeps the rule pure.
        let mut k = rule.len();
        while k > 0 {
            k -= 1;
            let removed = rule.remove(k);
            let still_pure = !neg_idx.iter().any(|&i| rule_matches(&sets[i], &rule));
            let still_covers = pos_idx
                .iter()
                .any(|&i| labels[i] == 1 && rule_matches(&sets[i], &rule));
            if !(still_pure && still_covers) {
                rule.insert(k, removed);
            }
        }
        rule.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        if !rules.contains(&rule) {
            rules.push(rule);
        }
    }

    if pos_idx.iter().any(|&i| !covered[i]) {
        return None;
    }
    // A single 1-literal positive rule is just member_class; let that teacher own it.
    if rules.len() == 1 && rules[0].len() <= 1 {
        return None;
    }

    let code = code_array_dnf_search(fn_name, &rules);
    verified_result(problem, code, "search_array_dnf")
}
