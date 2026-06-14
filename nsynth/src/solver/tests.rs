use super::search_codegen::*;
use super::*;
use crate::benchmark::{factory_count, generated_holdouts, get_benchmark, Example, Value};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_MODEL_ROOT_COUNTER: AtomicU64 = AtomicU64::new(0);

fn aliased_problem(
    source_prefix: &str,
    name: &str,
    signature: &'static str,
    category: &'static str,
    description: &'static str,
) -> Problem {
    let source = get_benchmark(1)
        .into_iter()
        .find(|p| p.name.starts_with(source_prefix))
        .unwrap();
    Problem {
        name: name.to_string(),
        category,
        description,
        signature,
        examples: source.examples,
        holdouts: vec![],
        reference_code: "",
    }
}

fn assert_search_generalizes_problem(problem: Problem, holdouts: Vec<(Vec<Value>, i64)>) {
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "search failed for {}", problem.name);

    for (inputs, expected) in holdouts {
        let actual = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &inputs,
            &problem,
        )
        .unwrap_or_else(|err| {
            panic!(
                "execution failed for {} on {:?}: {err}",
                problem.name, inputs
            )
        });
        match actual {
            crate::runtime::Value::Int(value) => {
                assert_eq!(
                    value, expected,
                    "wrong result for {} on {:?}",
                    problem.name, inputs
                );
            }
            other => panic!("expected int result for {}, got {:?}", problem.name, other),
        }
    }
}

fn assert_search_generalizes(problem_name: &str, holdouts: Vec<(Vec<Value>, i64)>) {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|problem| problem.name == problem_name)
        .unwrap();
    assert_search_generalizes_problem(problem, holdouts);
}

fn temp_model_root() -> std::path::PathBuf {
    let nonce = TEMP_MODEL_ROOT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "mog-warmstart-models-{}-{}",
        std::process::id(),
        nonce
    ));
    fs::create_dir_all(root.join("models")).unwrap();
    root
}

fn is_differentiable_only_capability_gap(error: &str) -> bool {
    error.contains("python bridge script not found")
        || error.contains("meta-learner checkpoint only covers")
        || error.contains("No 1-field correction found (fast mode)")
        || error.contains("unexpected token in expression: If")
}

fn solve_problem_differentiable_only_or_skip(problem: &Problem) -> Option<SolveResult> {
    let result = solve_problem_differentiable_only(problem);
    if result.success {
        return Some(result);
    }
    if let Some(error) = result.error.as_deref() {
        if is_differentiable_only_capability_gap(error) {
            eprintln!(
                "[skip] differentiable-only unavailable for {}: {error}",
                problem.name
            );
            return None;
        }
    }
    panic!(
        "differentiable-only solve failed for {}: {:?}",
        problem.name, result.error
    );
}

fn with_scratch_method_router<R>(f: impl FnOnce() -> R) -> R {
    crate::method_router::with_test_lock(|| {
        crate::solved_cache::with_test_lock(|| {
            let scratch = std::env::temp_dir().join(format!(
                "nsynth_router_solver_test_{}_{:?}.json",
                std::process::id(),
                std::thread::current().id(),
            ));
            std::env::set_var("NSYNTH_METHOD_ROUTER_PATH", &scratch);
            crate::method_router::reset_for_tests();
            crate::solved_cache::reset_for_tests();
            let _ = fs::remove_file(&scratch);
            let result = f();
            std::env::remove_var("NSYNTH_METHOD_ROUTER_PATH");
            crate::method_router::reset_for_tests();
            crate::solved_cache::reset_for_tests();
            let _ = fs::remove_file(&scratch);
            result
        })
    })
}

fn with_scratch_router_and_cache<R>(f: impl FnOnce() -> R) -> R {
    crate::method_router::with_test_lock(|| {
        crate::solved_cache::with_test_lock(|| {
            let router = std::env::temp_dir().join(format!(
                "nsynth_router_solver_test_{}_{:?}.json",
                std::process::id(),
                std::thread::current().id(),
            ));
            let cache = std::env::temp_dir().join(format!(
                "nsynth_cache_solver_test_{}_{:?}.json",
                std::process::id(),
                std::thread::current().id(),
            ));
            std::env::set_var("NSYNTH_METHOD_ROUTER_PATH", &router);
            std::env::set_var("NSYNTH_CACHE_PATH", &cache);
            crate::method_router::reset_for_tests();
            crate::solved_cache::reset_for_tests();
            let _ = fs::remove_file(&router);
            let _ = fs::remove_file(&cache);
            let result = f();
            std::env::remove_var("NSYNTH_METHOD_ROUTER_PATH");
            std::env::remove_var("NSYNTH_CACHE_PATH");
            crate::method_router::reset_for_tests();
            crate::solved_cache::reset_for_tests();
            let _ = fs::remove_file(&router);
            let _ = fs::remove_file(&cache);
            result
        })
    })
}

fn with_scratch_search_family_router<R>(f: impl FnOnce() -> R) -> R {
    let router = std::env::temp_dir().join(format!(
        "nsynth_search_family_solver_test_{}_{:?}.json",
        std::process::id(),
        std::thread::current().id(),
    ));
    crate::search_family_router::with_test_router_path(Some(router.clone()), || {
        let _ = fs::remove_file(&router);
        let result = f();
        let _ = fs::remove_file(&router);
        result
    })
}

/// Build a unary-string classification Problem from (word, label) pairs.
fn str_class_problem(
    name: &'static str,
    signature: &'static str,
    rows: &[(&str, i64)],
) -> Problem {
    Problem {
        name: name.to_string(),
        category: "morphology",
        description: "",
        signature,
        examples: rows
            .iter()
            .map(|(w, label)| Example {
                inputs: vec![Value::Str((*w).to_string())],
                expected: *label,
            })
            .collect(),
        holdouts: vec![],
        reference_code: "",
    }
}

/// LinguaGenesis x nsynth: the synthesizer must rediscover the regular English
/// `-es` pluralization rule (sibilant rule: -es after s/sh/ch/x/z) as a verified
/// Mog program, from labeled nouns alone. The non-sibilant `-h` words (month,
/// path, cough) are the hard negatives that force the precise `sh`/`ch` suffixes
/// rather than the over-general `ends_with("h")`.
#[test]
fn search_suffix_class_learns_sibilant_plural_rule() {
    let problem = str_class_problem(
        "takes_es_plural",
        "fn takes_es_plural(s: string) -> i64",
        &[
            // sibilant positives
            ("bus", 1), ("glass", 1), ("kiss", 1), ("class", 1),
            ("dish", 1), ("brush", 1), ("wish", 1), ("crash", 1),
            ("bench", 1), ("watch", 1), ("branch", 1), ("church", 1),
            ("box", 1), ("fox", 1), ("tax", 1), ("index", 1),
            ("buzz", 1), ("quiz", 1), ("fizz", 1),
            // hard negatives: non-sibilant -h, take +s
            ("month", 0), ("path", 0), ("bath", 0), ("cloth", 0),
            ("truth", 0), ("depth", 0), ("length", 0), ("cough", 0),
            ("laugh", 0), ("graph", 0),
            // easy negatives
            ("cat", 0), ("dog", 0), ("book", 0), ("tree", 0),
            ("car", 0), ("pen", 0), ("hand", 0), ("map", 0),
            ("table", 0), ("star", 0), ("road", 0), ("cup", 0),
        ],
    );

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "search failed to learn sibilant rule");
    assert_eq!(
        result.method, "search_suffix_class",
        "expected the suffix-class teacher to fire, got {}",
        result.method
    );

    // Generalize to held-out nouns the miner never saw. The -h negatives prove the
    // rule learned `ch`/`sh`, not the over-general bare `h`.
    assert_search_generalizes_problem(
        problem,
        vec![
            (vec![Value::Str("lunch".into())], 1),
            (vec![Value::Str("flash".into())], 1),
            (vec![Value::Str("prefix".into())], 1),
            (vec![Value::Str("waltz".into())], 1),
            (vec![Value::Str("kiss".into())], 1),
            (vec![Value::Str("mouth".into())], 0), // -th hard negative
            (vec![Value::Str("myth".into())], 0),  // -th hard negative
            (vec![Value::Str("breath".into())], 0), // -th hard negative
            (vec![Value::Str("door".into())], 0),
            (vec![Value::Str("chair".into())], 0),
        ],
    );
}

/// Build a unary-array classification Problem from (array, label) pairs.
fn arr_class_problem(
    name: &'static str,
    signature: &'static str,
    rows: &[(&[i64], i64)],
) -> Problem {
    Problem {
        name: name.to_string(),
        category: "morphology",
        description: "",
        signature,
        examples: rows
            .iter()
            .map(|(arr, label)| Example {
                inputs: vec![Value::Array(arr.to_vec())],
                expected: *label,
            })
            .collect(),
        holdouts: vec![],
        reference_code: "",
    }
}

/// Array analog of the suffix-class test, and the core of the morpheme-tokenized
/// sentence path: a sentence is encoded to a token-id array, and grammaticality
/// is "the array carries a valid inflection-suffix token id". Here token id 104
/// stands in for `<+es>` and 101 for `<+ies>` (the valid 3sg markers); 106
/// (`<+s>`, emitted by tokenizer over-split on bare stems) and other ids are the
/// poison the learner must NOT key on.
#[test]
fn search_array_member_class_learns_inflection_membership() {
    let problem = arr_class_problem(
        "sentence_es_3sg",
        "fn sentence_es_3sg(arr: [i64]) -> i64",
        &[
            // positives: carry a valid -es-family inflection token (104 or 101)
            (&[111, 4, 284, 104], 1),
            (&[111, 4, 554, 104], 1),
            (&[111, 4, 657, 101], 1),
            (&[111, 4, 812, 104], 1),
            (&[111, 4, 876, 101], 1),
            (&[111, 4, 596, 104], 1),
            // negatives: bare stem (no suffix), or over-split 106 (<+s>), or wrong 107 (<+d>)
            (&[111, 4, 284], 0),
            (&[111, 4, 554, 106], 0), // over-split bare stem -> <+s>
            (&[111, 4, 657], 0),
            (&[111, 4, 812, 107], 0), // wrong suffix <+d>
            (&[111, 4, 876], 0),
            (&[111, 4, 659, 106], 0),
        ],
    );

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "search failed to learn inflection membership");
    assert_eq!(
        result.method, "search_array_member_class",
        "expected the array member-class teacher, got {}",
        result.method
    );

    // Generalize to held-out token arrays the miner never saw. The 106/107
    // negatives prove it keyed on the valid markers (104/101), not bare-stem noise.
    assert_search_generalizes_problem(
        problem,
        vec![
            (vec![Value::Array(vec![111, 4, 999, 104])], 1),
            (vec![Value::Array(vec![111, 4, 888, 101])], 1),
            (vec![Value::Array(vec![111, 4, 999])], 0),
            (vec![Value::Array(vec![111, 4, 888, 106])], 0), // over-split <+s>
            (vec![Value::Array(vec![111, 4, 777, 107])], 0), // wrong <+d>
        ],
    );
}

/// Conjunctive membership: grammatical gerund iff the token array carries BOTH
/// the auxiliary `is` (109) AND the `<+ing>` suffix (103). Neither token alone
/// separates (each appears in some negative), so the OR-only member-class teacher
/// cannot — only the conjunction does. Mirrors bridge task `sentence_gerund`.
#[test]
fn search_array_conjunction_learns_auxiliary_agreement() {
    let problem = arr_class_problem(
        "gerund_ok",
        "fn gerund_ok(arr: [i64]) -> i64",
        &[
            // positives: contain BOTH is(109) and <+ing>(103)
            (&[111, 4, 109, 659, 103], 1),
            (&[111, 4, 109, 272, 103], 1),
            (&[111, 4, 109, 812, 103], 1),
            (&[111, 4, 109, 540, 103], 1),
            (&[111, 4, 109, 661, 103], 1),
            // negatives missing the auxiliary (have 103, no 109)
            (&[111, 4, 659, 103], 0),
            (&[111, 4, 812, 103], 0),
            (&[111, 4, 540, 103], 0),
            // negatives missing the -ing (have 109, no 103)
            (&[111, 4, 109, 659], 0),
            (&[111, 4, 109, 272], 0),
            (&[111, 4, 109, 540], 0),
            (&[111, 4, 109, 661], 0),
        ],
    );

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "conjunction not learned");
    assert_eq!(
        result.method, "search_array_conjunction",
        "expected the conjunction teacher, got {}",
        result.method
    );

    assert_search_generalizes_problem(
        problem,
        vec![
            (vec![Value::Array(vec![111, 4, 109, 999, 103])], 1), // is + ing
            (vec![Value::Array(vec![111, 4, 999, 103])], 0),      // missing is
            (vec![Value::Array(vec![111, 4, 109, 999])], 0),      // missing ing
        ],
    );
}

/// DNF membership: logical-argument validity. Feature tokens — assertA=1,
/// assertB=2, assertNeg=3, concludeA=4, concludeB=5, concludeNeg=6. Valid =
/// modus ponens {assert A, conclude B} OR modus tollens {assert ~B, conclude ~A};
/// invalid = affirming-consequent / denying-antecedent. This is a true DNF that
/// neither member-class (OR of singletons) nor the single-conjunction teacher can
/// express — mirrors bridge task `formal_logic`.
#[test]
fn search_array_dnf_learns_inference_validity() {
    let mut rows: Vec<(Vec<i64>, i64)> = Vec::new();
    for k in 0..6 {
        let filler = 90 + k; // distinct per row, ignored by the rule
        rows.push((vec![filler, 1, 5], 1)); // modus ponens — valid
        rows.push((vec![filler, 2, 3, 4, 6], 1)); // modus tollens — valid
        rows.push((vec![filler, 2, 4], 0)); // affirming consequent — invalid
        rows.push((vec![filler, 1, 3, 5, 6], 0)); // denying antecedent — invalid
    }
    let problem = Problem {
        name: "valid_argument".to_string(),
        category: "reasoning",
        description: "",
        signature: "fn valid_argument(arr: [i64]) -> i64",
        examples: rows
            .iter()
            .take(16)
            .map(|(a, l)| Example {
                inputs: vec![Value::Array(a.clone())],
                expected: *l,
            })
            .collect(),
        holdouts: vec![],
        reference_code: "",
    };

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "DNF validity rule not learned");
    assert_eq!(
        result.method, "search_array_dnf",
        "expected the DNF teacher, got {}",
        result.method
    );
    // Generalize to the held-out forms (different filler tokens).
    assert_search_generalizes_problem(
        problem,
        vec![
            (vec![Value::Array(vec![77, 1, 5])], 1),       // MP
            (vec![Value::Array(vec![77, 2, 3, 4, 6])], 1), // MT
            (vec![Value::Array(vec![77, 2, 4])], 0),       // AC
            (vec![Value::Array(vec![77, 1, 3, 5, 6])], 0), // DA
        ],
    );
}

#[path = "tests/benchmark_diff_cases.rs"]
mod benchmark_diff_cases;
#[path = "tests/search_cases.rs"]
mod exact_cases;
#[path = "tests/gradient_cases.rs"]
mod gradient_cases;
#[path = "tests/routing_cases.rs"]
mod routing_cases;
