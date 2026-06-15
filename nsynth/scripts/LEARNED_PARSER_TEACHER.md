# Learned-parser teacher: `search_string_equality_map`

Durable copy of the Rust synthesis-teacher that lets nSynth **learn a verified
lexicon** (string → integer label) as an executable Mog program — the core of the
learned comprehension parser. Kept here because a concurrent editing session on
this repo reverted the uncommitted Rust edits; re-apply the three hunks below to
restore the capability in source. (The compiled binary already contains it.)

This teacher is what makes the comprehension pipeline (scripts/comprehend.py +
the `noun_animacy` / `roles_rule` / `agreement_rule` / `ends_s` bridge tasks)
fully synthesized: animacy and noun-membership are arbitrary lexical facts with
no orthographic signal, so they must be *stored*, and nSynth recovers the closed
lexicon directly from I/O instead of relying on a hand-written Python lookup.

## Hunk 1 — `src/solver/search_codegen.rs`
Insert after `code_suffix_class_search` (before `code_array_member_class_search`):

```rust
/// A learned lexical lookup: maps specific input strings to integer labels via
/// string-equality, falling back to `default` for anything unlisted. This is how
/// nSynth recovers an *arbitrary lexicon* (e.g. animacy: "teacher" -> 1) from
/// I/O examples — facts with no orthographic rule must be stored, not derived,
/// so the synthesized program IS the lexicon, verified against every example.
pub(super) fn code_string_equality_map(
    fn_name: &str,
    default: i64,
    branches: &[(String, i64)],
) -> String {
    let mut body = String::new();
    for (literal, value) in branches {
        let literal = literal.replace('\\', "\\\\").replace('"', "\\\"");
        body.push_str(&format!(
            "    if s == \"{literal}\" {{\n        return {value};\n    }}\n"
        ));
    }
    format!("fn {fn_name}(s: string) -> i64 {{\n{body}    return {default};\n}}\n")
}
```

## Hunk 2 — `src/solver/search_text_families.rs`
Change the imports line to bring in `HashMap`:

```rust
use std::collections::{HashMap, HashSet};
```

Add the teacher (anywhere in the file, e.g. after `search_suffix_class`):

```rust
/// Learned lexical lookup: synthesize `fn f(s: string) -> i64` as a verified
/// string-equality table — the parser's *lexicon*. Arbitrary facts (animacy,
/// irregular class membership) carry no orthographic signal and cannot be
/// derived by a rule, so they must be stored. The teacher recovers the table
/// from I/O, compresses it around the majority label, and verifies it end to end.
pub(super) fn search_string_equality_map(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    let strings = unary_string_examples(problem)?;
    if problem.examples.len() < 3 {
        return None;
    }

    // Build a consistent string -> label map. Conflicting labels for the same
    // surface form mean the answer is not a function of the word alone — refuse.
    let mut map: HashMap<String, i64> = HashMap::new();
    for (example, word) in problem.examples.iter().zip(strings.iter()) {
        let label = example.expected_int();
        match map.get(word) {
            Some(prev) if *prev != label => return None,
            _ => {
                map.insert(word.clone(), label);
            }
        }
    }

    // A single-label map is a constant function — leave that to other teachers.
    let distinct_labels: HashSet<i64> = map.values().copied().collect();
    if distinct_labels.len() < 2 {
        return None;
    }

    // Last-resort guarantee, independent of portfolio/router ordering: if a
    // general orthographic-rule teacher already explains these examples, defer to
    // it — a rule (suffix / prefix / substring) generalizes to unseen words; a
    // lookup table does not. This teacher only claims a problem when the
    // string->label map really is an arbitrary lexicon no rule can capture.
    for rule_teacher in [
        search_suffix_class as fn(&Problem, &str) -> Option<SolveResult>,
        search_contains_literal,
        search_starts_with_literal,
    ] {
        if rule_teacher(problem, fn_name).is_some() {
            return None;
        }
    }

    // Default = the most frequent label across examples, so the common class
    // needs no branches and the table stays as small as the data allows.
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for example in &problem.examples {
        *counts.entry(example.expected_int()).or_insert(0) += 1;
    }
    let default = counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(label, _)| *label)?;

    // Emit a branch only for words whose label differs from the default.
    let mut branches: Vec<(String, i64)> =
        map.into_iter().filter(|(_, label)| *label != default).collect();
    branches.sort();

    let code = code_string_equality_map(fn_name, default, &branches);
    verified_result(problem, code, "search_string_equality_map")
}
```

## Hunk 3 — `src/solver/search.rs`
Register it in `SEARCH_CANDIDATES`, AFTER `search_string_subsequence_class`
(so every orthographic-rule string teacher runs first):

```rust
    SearchCandidate {
        key: "search_string_equality_map",
        func: search_string_equality_map,
    },
```

## Unit test — `src/solver/tests.rs`
Uses the existing `str_class_problem` helper. Data must contain suffix collisions
(animate and inanimate words sharing endings) so the spurious suffix rule fails
and the true lexicon lookup wins:

```rust
#[test]
fn search_string_equality_map_learns_animacy_lexicon() {
    let problem = str_class_problem(
        "is_animate",
        "fn is_animate(s: string) -> i64",
        &[
            ("teacher", 1), ("doctor", 1), ("actor", 1), ("singer", 1),
            ("painter", 1), ("baker", 1), ("dog", 1), ("cat", 1),
            ("weather", 0), ("tractor", 0), ("finger", 0), ("printer", 0),
            ("marker", 0), ("fog", 0), ("mat", 0), ("report", 0),
            ("song", 0), ("door", 0),
        ],
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_string_equality_map");
    assert!(result.code.contains("if s == \"teacher\""));
}
```
