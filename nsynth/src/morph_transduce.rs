//! Generative morphology: string -> string transduction synthesis.
//!
//! This is the additive, low-blast-radius path for *producing* inflected forms
//! (cat -> cats, box -> boxes) rather than merely classifying them. It does not
//! touch `Example` / the i64 verify path: string examples live in their own
//! `StrExample`, and verification runs the emitted Mog program and compares the
//! returned `Value::Str` directly.
//!
//! Scope: suffix-conditioned APPEND rules — `output = input + append(input)`,
//! where the appended string depends only on the input's trailing characters.
//! This is exactly the productive English plural for regular + sibilant nouns
//! (the curriculum's `pluralize` for the non-stem-changing cases): default `+s`,
//! sibilant `+es`. Stem-changing rules (city -> cities, knife -> knives) are not
//! pure appends and are reported unsupported — they need substring ops, a
//! natural next extension.

use crate::runtime::execute_str_function;

#[derive(Clone, Debug)]
pub struct StrExample {
    pub input: String,
    pub expected: String,
}

#[derive(Clone, Debug)]
pub struct MorphResult {
    pub success: bool,
    pub code: String,
    pub method: String,
    pub error: Option<String>,
}

fn fail(error: &str) -> MorphResult {
    MorphResult {
        success: false,
        code: String::new(),
        method: "morph_transduce_unsupported".to_string(),
        error: Some(error.to_string()),
    }
}

/// A transduction action: drop the last `strip` chars of the input, then append
/// `append`. `strip == 0` is a pure append (cat -> cats); `strip > 0` is a
/// stem change (city -> cit + "ies", knife -> kni + "ves").
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Action {
    strip: usize,
    append: String,
}

/// Derive the action turning `input` into `output`: drop the chars after their
/// longest common prefix, append the remainder of the output.
fn action_for(input: &str, output: &str) -> Action {
    let common = input
        .chars()
        .zip(output.chars())
        .take_while(|(a, b)| a == b)
        .count();
    let strip = input.chars().count() - common;
    let append: String = output.chars().skip(common).collect();
    Action { strip, append }
}

/// Learn a suffix-conditioned append transduction from (input -> output) pairs.
pub fn solve_morph_transduction(
    fn_name: &str,
    train: &[StrExample],
    holdouts: &[StrExample],
) -> MorphResult {
    const MAX_SUFFIX_LEN: usize = 3;
    if train.is_empty() {
        return fail("no examples");
    }

    // 1. Each example becomes a (input, action) pair, where the action is a
    //    drop-last-k-then-append (pure append when k == 0, stem change when k > 0).
    let pairs: Vec<(String, Action)> = train
        .iter()
        .map(|ex| (ex.input.clone(), action_for(&ex.input, &ex.expected)))
        .collect();

    // 2. Learn an ordered ends_with decision list over the actions: specific
    //    (longer) suffix rules first, then a default action. This lets a general
    //    short rule coexist with longer exception rules above it — e.g.
    //    "ay/ey/oy -> +s" placed above "y -> drop1 + ies" generalizes the
    //    consonant+y rule to unseen words (lady -> ladies) while keeping vowel+y
    //    regular (play -> plays).
    //
    //    Which action is the default is decided by generalization, not by sample
    //    frequency (a sibilant-heavy sample has more `+es` than `+s`): try each
    //    distinct action AS the default and keep the first whose program verifies
    //    on train + holdouts. Candidates ordered by frequency desc as a heuristic.
    let mut counts: std::collections::HashMap<&Action, usize> = std::collections::HashMap::new();
    for (_, act) in &pairs {
        *counts.entry(act).or_insert(0) += 1;
    }
    let mut default_candidates: Vec<Action> = counts.keys().map(|a| (*a).clone()).collect();
    default_candidates.sort_by(|a, b| {
        counts[b]
            .cmp(&counts[a])
            .then_with(|| a.strip.cmp(&b.strip))
            .then_with(|| a.append.cmp(&b.append))
    });

    let mut last_err = "not suffix-separable (over-determined / needs char classes)".to_string();
    for default in &default_candidates {
        let Some(rules) = learn_suffix_decision_list(&pairs, default, MAX_SUFFIX_LEN) else {
            continue;
        };
        // A stem-change rule whose conditioning suffix is shorter than its strip
        // would slice off chars the suffix never matched — reject as unsafe.
        if rules.iter().any(|(suf, act)| act.strip > suf.chars().count()) {
            last_err = "stem-change action needs a suffix >= strip length".to_string();
            continue;
        }
        let code = emit_transduce(fn_name, &rules, default);
        match verify_transduction(&code, fn_name, train, holdouts) {
            Ok(()) => {
                return MorphResult {
                    success: true,
                    code,
                    method: "morph_transduce_suffix_append".to_string(),
                    error: None,
                }
            }
            Err(e) => last_err = e,
        }
    }
    fail(&last_err)
}

/// Greedy ends_with decision-list induction over an arbitrary label type.
///
/// Repeatedly peels off the highest-coverage *pure* suffix rule (all remaining
/// examples ending in that suffix share a label); ties break toward the shorter
/// (more general) suffix. Rules are then emitted specific-first (longest suffix
/// first) so exceptions override generals, and redundant rules are pruned.
/// Returns the ordered rules (default supplied by the caller), or None if the
/// non-default examples can't be peeled by pure suffix rules.
pub(crate) fn learn_suffix_decision_list<L: Clone + Eq + std::hash::Hash>(
    pairs: &[(String, L)],
    default: &L,
    max_suffix_len: usize,
) -> Option<Vec<(String, L)>> {
    use std::collections::HashSet;
    if pairs.is_empty() {
        return None;
    }

    let mut remaining: Vec<usize> = (0..pairs.len()).collect();
    let mut rules: Vec<(String, L)> = Vec::new();

    // Peel pure suffix rules until everything left is the default label. Picking
    // a default-labeled exception rule (e.g. "ay" -> +s) is allowed: it removes
    // examples that would otherwise block a more general rule (e.g. "y" -> ies)
    // from becoming pure.
    while remaining.iter().any(|&i| pairs[i].1 != *default) {
        let mut cands: HashSet<String> = HashSet::new();
        for &i in &remaining {
            let chars: Vec<char> = pairs[i].0.chars().collect();
            for len in 1..=max_suffix_len.min(chars.len()) {
                cands.insert(chars[chars.len() - len..].iter().collect());
            }
        }
        // Best pure rule: max coverage, then shorter suffix, then lexical.
        let mut best: Option<(usize, usize, String, L)> = None;
        for suf in &cands {
            let grp: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&i| pairs[i].0.ends_with(suf.as_str()))
                .collect();
            let g_labels: HashSet<&L> = grp.iter().map(|&i| &pairs[i].1).collect();
            if g_labels.len() != 1 {
                continue;
            }
            let label = pairs[grp[0]].1.clone();
            let cover = grp.len();
            let len = suf.chars().count();
            let better = match &best {
                None => true,
                Some((bc, bl, bs, _)) => {
                    cover > *bc
                        || (cover == *bc && len < *bl)
                        || (cover == *bc && len == *bl && suf < bs)
                }
            };
            if better {
                best = Some((cover, len, suf.clone(), label));
            }
        }
        let (_, _, suf, label) = best?; // not suffix-separable
        remaining.retain(|&i| !pairs[i].0.ends_with(suf.as_str()));
        rules.push((suf, label));
    }

    // Emit specific-first: longest suffix first, then lexical.
    rules.sort_by(|a, b| {
        b.0.chars()
            .count()
            .cmp(&a.0.chars().count())
            .then_with(|| a.0.cmp(&b.0))
    });
    prune_suffix_rules(&mut rules, default, pairs);
    Some(rules)
}

/// Classify with the ordered ends_with list: first matching rule wins, else default.
fn classify_suffix<'a, L>(rules: &'a [(String, L)], default: &'a L, input: &str) -> &'a L {
    for (suf, label) in rules {
        if input.ends_with(suf.as_str()) {
            return label;
        }
    }
    default
}

/// Drop rules whose removal leaves the training set still perfectly classified
/// (lowest-priority / shortest rules removed first).
fn prune_suffix_rules<L: Clone + Eq>(
    rules: &mut Vec<(String, L)>,
    default: &L,
    pairs: &[(String, L)],
) {
    let mut i = rules.len();
    while i > 0 {
        i -= 1;
        let removed = rules.remove(i);
        let ok = pairs
            .iter()
            .all(|(inp, lbl)| classify_suffix(rules, default, inp) == lbl);
        if !ok {
            rules.insert(i, removed);
        }
    }
}

fn verify_transduction(
    code: &str,
    fn_name: &str,
    train: &[StrExample],
    holdouts: &[StrExample],
) -> Result<(), String> {
    for ex in train.iter().chain(holdouts.iter()) {
        match execute_str_function(code, fn_name, &ex.input) {
            Ok(out) if out == ex.expected => {}
            Ok(out) => {
                return Err(format!(
                    "verify failed: {} -> {:?}, expected {:?}",
                    ex.input, out, ex.expected
                ))
            }
            Err(e) => return Err(format!("execution failed on {}: {e}", ex.input)),
        }
    }
    Ok(())
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Emit the expression that applies an action to `s`: a pure append is
/// `s + "X"`; a stem change drops the last `strip` chars first via slice:
/// `s.slice(0, s.len - strip) + "X"`.
fn apply_expr(action: &Action) -> String {
    if action.strip == 0 {
        format!("s + \"{}\"", esc(&action.append))
    } else {
        format!(
            "s.slice(0, s.len - {}) + \"{}\"",
            action.strip,
            esc(&action.append)
        )
    }
}

fn emit_transduce(fn_name: &str, rules: &[(String, Action)], default_action: &Action) -> String {
    let mut body = String::new();
    for (suf, action) in rules {
        body.push_str(&format!(
            "    if s.ends_with(\"{}\") {{\n        return {};\n    }}\n",
            esc(suf),
            apply_expr(action)
        ));
    }
    body.push_str(&format!("    return {};\n", apply_expr(default_action)));
    format!("fn {fn_name}(s: string) -> string {{\n{body}}}\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(input: &str, expected: &str) -> StrExample {
        StrExample {
            input: input.to_string(),
            expected: expected.to_string(),
        }
    }

    #[test]
    fn learns_regular_and_sibilant_plural_append() {
        let train = vec![
            ex("cat", "cats"),
            ex("dog", "dogs"),
            ex("book", "books"),
            ex("car", "cars"),
            ex("bus", "buses"),
            ex("box", "boxes"),
            ex("watch", "watches"),
            ex("dish", "dishes"),
            ex("buzz", "buzzes"),
        ];
        let holdouts = vec![
            ex("tree", "trees"),
            ex("fox", "foxes"),
            ex("brush", "brushes"),
            ex("glass", "glasses"),
        ];
        let result = solve_morph_transduction("pluralize", &train, &holdouts);
        assert!(result.success, "failed: {:?}", result.error);
        assert_eq!(result.method, "morph_transduce_suffix_append");
        // sanity: the emitted program actually produces the inflected forms
        assert_eq!(
            execute_str_function(&result.code, "pluralize", "fox").unwrap(),
            "foxes"
        );
        assert_eq!(
            execute_str_function(&result.code, "pluralize", "tree").unwrap(),
            "trees"
        );
    }

    #[test]
    fn learns_stem_changing_y_to_ies() {
        // city -> cit + "ies" (strip 1), with vowel+y staying regular (+s).
        let train = vec![
            ex("city", "cities"),
            ex("baby", "babies"),
            ex("story", "stories"),
            ex("party", "parties"),
            ex("army", "armies"),
            ex("puppy", "puppies"),
            ex("cat", "cats"),
            ex("dog", "dogs"),
            ex("play", "plays"), // vowel+y -> regular +s, NOT +ies
            ex("day", "days"),
            ex("key", "keys"),
            ex("boy", "boys"), // -oy exception, so the holdout toy generalizes
        ];
        let holdouts = vec![
            ex("lady", "ladies"),
            ex("berry", "berries"),
            ex("tree", "trees"),
            ex("toy", "toys"), // vowel+y holdout
        ];
        let result = solve_morph_transduction("pluralize", &train, &holdouts);
        assert!(result.success, "failed: {:?}", result.error);
        assert_eq!(
            execute_str_function(&result.code, "pluralize", "lady").unwrap(),
            "ladies"
        );
        assert_eq!(
            execute_str_function(&result.code, "pluralize", "toy").unwrap(),
            "toys"
        );
    }

    #[test]
    fn reports_unseparable_unsupported() {
        // Both inputs share their last 3 chars ("xyz") but need different
        // outputs; the distinguishing character is beyond MAX_SUFFIX_LEN, so the
        // rule is not suffix-separable.
        let train = vec![ex("axyz", "axyzs"), ex("bxyz", "bxyze")];
        let result = solve_morph_transduction("xduce", &train, &[]);
        assert!(!result.success);
        assert_eq!(result.method, "morph_transduce_unsupported");
    }
}
