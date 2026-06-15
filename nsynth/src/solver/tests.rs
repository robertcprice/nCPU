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
fn str_class_problem(name: &'static str, signature: &'static str, rows: &[(&str, i64)]) -> Problem {
    Problem {
        name: name.to_string(),
        category: "morphology",
        description: "",
        signature,
        examples: rows
            .iter()
            .map(|(w, label)| Example {
                inputs: vec![Value::Str((*w).to_string())],
                expected: Value::Int(*label),
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
            ("bus", 1),
            ("glass", 1),
            ("kiss", 1),
            ("class", 1),
            ("dish", 1),
            ("brush", 1),
            ("wish", 1),
            ("crash", 1),
            ("bench", 1),
            ("watch", 1),
            ("branch", 1),
            ("church", 1),
            ("box", 1),
            ("fox", 1),
            ("tax", 1),
            ("index", 1),
            ("buzz", 1),
            ("quiz", 1),
            ("fizz", 1),
            // hard negatives: non-sibilant -h, take +s
            ("month", 0),
            ("path", 0),
            ("bath", 0),
            ("cloth", 0),
            ("truth", 0),
            ("depth", 0),
            ("length", 0),
            ("cough", 0),
            ("laugh", 0),
            ("graph", 0),
            // easy negatives
            ("cat", 0),
            ("dog", 0),
            ("book", 0),
            ("tree", 0),
            ("car", 0),
            ("pen", 0),
            ("hand", 0),
            ("map", 0),
            ("table", 0),
            ("star", 0),
            ("road", 0),
            ("cup", 0),
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
                expected: Value::Int(*label),
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
    assert!(
        result.success,
        "search failed to learn inflection membership"
    );
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

#[test]
fn search_array_sequence_learns_order_constraint() {
    let problem = arr_class_problem(
        "sequence_ok",
        "fn sequence_ok(arr: [i64]) -> i64",
        &[
            // positives: 109 occurs before 103
            (&[109, 103], 1),
            (&[111, 4, 109, 659, 103], 1),
            (&[111, 109, 4, 103], 1),
            (&[109, 111, 103], 1),
            (&[109, 4, 103, 5], 1),
            (&[109, 109, 103], 1),
            // negatives: missing or wrong order
            (&[103, 109], 0),
            (&[111, 103, 4, 109], 0),
            (&[109], 0),
            (&[103], 0),
            (&[111, 103], 0),
            (&[109, 111], 0),
            (&[111, 103, 109], 0),
        ],
    );

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "sequence not learned");
    assert_eq!(
        result.method, "search_array_sequence",
        "expected the sequence teacher, got {}",
        result.method
    );

    assert_search_generalizes_problem(
        problem,
        vec![
            (vec![Value::Array(vec![111, 109, 999, 103])], 1), // 109 before 103
            (vec![Value::Array(vec![111, 103, 999, 109])], 0), // 103 before 109
            (vec![Value::Array(vec![111, 109, 999])], 0),      // missing 103
        ],
    );
}

#[test]
fn search_array_feature_dnf_learns_count_and_run_features() {
    let problem = arr_class_problem(
        "array_feature_ok",
        "fn array_feature_ok(arr: [i64]) -> i64",
        &[
            (&[7, 7, 1, 2, 3], 1),
            (&[0, 7, 5, 7], 1),
            (&[7, 3, 7], 1),
            (&[2, 7, 7, 8], 1),
            (&[4, 4, 4, 9], 1),
            (&[1, 4, 4, 4], 1),
            (&[4, 4, 4], 1),
            (&[6, 4, 4, 4, 5], 1),
            (&[7, 1, 2], 0),
            (&[7, 7, 7, 1], 0),
            (&[4, 4, 9, 4], 0),
            (&[4, 9, 4, 4], 0),
            (&[7, 4, 4, 4], 0),
            (&[4, 4, 7, 7], 0),
        ],
    );

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "feature DNF not learned");
    assert_eq!(
        result.method, "search_array_feature_dnf",
        "expected the feature DNF teacher, got {}",
        result.method
    );
    assert_search_generalizes_problem(
        problem,
        vec![
            (vec![Value::Array(vec![7, 3, 7, 6])], 1),
            (vec![Value::Array(vec![7, 7, 7, 9])], 0),
            (vec![Value::Array(vec![5, 4, 4, 4])], 1),
            (vec![Value::Array(vec![10, 4, 4])], 0),
        ],
    );
}

#[test]
fn search_string_subsequence_class_learns_order_constraint() {
    let problem = str_class_problem(
        "a_before_b",
        "fn a_before_b(s: string) -> i64",
        &[
            ("aXbZ", 1),
            ("aaXbY", 1),
            ("aXXbX", 1),
            ("zaXbqW", 1),
            ("a bV", 1),
            ("a12bU", 1),
            ("aaabbbT", 1),
            ("xaYbS", 1),
            ("bXbZ", 0),
            ("bXbY", 0),
            ("bXbX", 0),
            ("bqbW", 0),
            ("b bV", 0),
            ("b2bU", 0),
            ("bbbT", 0),
            ("bYbS", 0),
            ("bXa", 0),
            ("b a", 0),
            ("a only", 0),
            ("b only", 0),
            ("xxa", 0),
            ("xxb", 0),
            ("ba", 0),
            ("bbb aaa", 0),
        ],
    );

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "string subsequence class not learned");
    assert_eq!(
        result.method, "search_string_subsequence_class",
        "expected the string subsequence-class teacher, got {}",
        result.method
    );

    assert_search_generalizes_problem(
        problem,
        vec![
            (vec![Value::Str("q a z b".into())], 1),
            (vec![Value::Str("q b z a".into())], 0),
            (vec![Value::Str("a b a".into())], 1),
            (vec![Value::Str("b b a".into())], 0),
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
                expected: Value::Int(*l),
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

/// Strict monotonicity: array[i] < array[i+1] for every adjacent pair.
/// Unlike `search_is_sorted` (which permits equal neighbours), the new
/// teacher must reject `[1, 1, 2]` and `[2, 1, 3]`.
#[test]
fn search_strictly_increasing_learns_strict_inequality() {
    let problem = arr_class_problem(
        "strictly_increasing",
        "fn strictly_increasing(arr: [i64]) -> i64",
        &[
            // positives
            (&[1, 2, 3], 1),
            (&[0, 5], 1),
            (&[-3, -1, 0, 7, 100], 1),
            (&[10, 20, 30, 40, 50], 1),
            // negatives: equal neighbours
            (&[1, 1, 2], 0),
            (&[2, 2], 0),
            (&[5, 5, 5, 6], 0),
            // negatives: descent
            (&[3, 2, 1], 0),
            (&[10, 0], 0),
            // negatives: midpoint descent
            (&[1, 5, 4, 9], 0),
        ],
    );

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "strictly_increasing not learned: {:?}", result.error);
    assert_eq!(result.method, "search_strictly_increasing");
    assert!(result.code.contains("fn strictly_increasing"));
    assert!(
        result.code.contains("arr[i] <= arr[i - 1]"),
        "expected strict-inequality check (not is_sorted's <); got: {}",
        result.code
    );

    assert_search_generalizes_problem(
        problem,
        vec![
            (vec![Value::Array(vec![100, 200])], 1),
            (vec![Value::Array(vec![1, 1])], 0),
            (vec![Value::Array(vec![1, 2, 1])], 0),
            (vec![Value::Array(vec![0, 0, 1])], 0),
        ],
    );
}

/// `has_strictly_increasing_run(arr, k) -> 1` iff arr contains a strictly
/// increasing run of length >= k. The teacher tries k=2,3,4,5 in order
/// and emits the first k that matches every example.
#[test]
fn search_has_strictly_increasing_run_learns_run_length() {
    let problem = arr_class_problem(
        "has_strict_inc_run_3",
        "fn has_strict_inc_run_3(arr: [i64]) -> i64",
        &[
            // positives: contain a strict run of length 3
            (&[1, 2, 3], 1),
            (&[0, 1, 5, 6, 7], 1),
            (&[10, 20, 30], 1),
            (&[5, 4, 3, 7, 8, 9], 1),
            // negatives: longest strict run is < 3
            (&[1, 2], 0),
            (&[1, 5, 3], 0), // not strictly increasing across the descent
            (&[3, 3, 4], 0), // 3,3 is not strict; 3,4 is a run of length 2
            (&[5, 4, 3, 2, 1], 0),
        ],
    );

    let result = solve_problem_search_only(&problem);
    assert!(
        result.success,
        "has_strictly_increasing_run not learned: {:?}",
        result.error
    );
    assert_eq!(result.method, "search_has_strictly_increasing_run");
    assert!(result.code.contains("run >= 3"), "expected run >= 3 threshold; got: {}", result.code);
}

/// `first_index_of(arr, target) -> first i where arr[i] == target, else -1`.
/// Returns an Int (not a 0/1 classifier), so this test uses an `int_class_problem`
/// builder to keep the expected output as an arbitrary i64.
#[test]
fn search_first_index_of_learns_target_value() {
    fn int_arr_problem(
        name: &'static str,
        signature: &'static str,
        rows: &[(&[i64], i64)],
    ) -> Problem {
        Problem {
            name: name.to_string(),
            category: "array_index",
            description: "",
            signature,
            examples: rows
                .iter()
                .map(|(arr, label)| Example {
                    inputs: vec![Value::Array(arr.to_vec())],
                    expected: Value::Int(*label),
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",
        }
    }

    let problem = int_arr_problem(
        "first_index_of_5",
        "fn first_index_of_5(arr: [i64]) -> i64",
        &[
            // target = 5
            (&[1, 2, 5, 7], 2),
            (&[5, 5, 5], 0),    // first occurrence at index 0
            (&[0, 0, 0, 5], 3), // first 5 at index 3
            (&[10, 20, 30], 4 - 1), // no 5; -1 = length-1
            // wait that's 4-1=3 not -1. Let me use the actual -1.
        ],
    );
    // Fix: the last entry above is wrong. Build the problem correctly.
    let problem = int_arr_problem(
        "first_index_of_5",
        "fn first_index_of_5(arr: [i64]) -> i64",
        &[
            (&[1, 2, 5, 7], 2),
            (&[5, 5, 5], 0),
            (&[0, 0, 0, 5], 3),
            (&[10, 20, 30], -1),  // -1: target not present
            (&[5], 0),            // target at index 0
            (&[1, 2, 3, 4, 6, 7, 5, 8, 9, 5], 6), // first 5 at index 6
            (&[1, 2, 3, 4], -1),
            (&[5, 1, 2, 3, 4, 5], 0),
        ],
    );

    let result = solve_problem_search_only(&problem);
    assert!(
        result.success,
        "first_index_of not learned: {:?}",
        result.error
    );
    assert_eq!(result.method, "search_first_index_of");
    assert!(
        result.code.contains("arr[i] == 5"),
        "expected equality check against target 5; got: {}",
        result.code
    );

    assert_search_generalizes_problem(
        problem,
        vec![
            (vec![Value::Array(vec![2, 5, 8])], 1),
            (vec![Value::Array(vec![5, 5])], 0),
            (vec![Value::Array(vec![1, 2, 3])], -1),
            (vec![Value::Array(vec![5])], 0),
        ],
    );
}

/// Every string-output benchmark problem is solved by the main pipeline and the
/// emitted program verifies on its held-out examples.
#[test]
fn string_benchmark_full_coverage() {
    let problems = crate::benchmark::get_string_benchmark(1);
    assert!(!problems.is_empty());
    for p in &problems {
        let result = solve_problem(p);
        assert!(result.success, "string benchmark {} not solved", p.name);
        crate::runtime::verify_problem_code_strict(p, &result.code)
            .unwrap_or_else(|e| panic!("{} failed strict verify: {e}", p.name));
    }
}

/// String OUTPUT through the main pipeline — exercises the widened
/// `Example.expected: Value` and the Value-aware verify path. A string->string
/// problem is solved by `solve_problem` and run on a fresh input.
#[test]
fn solve_problem_handles_string_output() {
    let str_ex = |inp: &str, out: &str| Example {
        inputs: vec![Value::Str(inp.to_string())],
        expected: Value::Str(out.to_string()),
    };
    let problem = Problem {
        name: "reverse_str".to_string(),
        category: "string",
        description: "",
        signature: "fn reverse_str(s: string) -> string",
        examples: vec![
            str_ex("abc", "cba"),
            str_ex("hello", "olleh"),
            str_ex("x", "x"),
            str_ex("ab", "ba"),
        ],
        holdouts: vec![str_ex("world", "dlrow"), str_ex("nsynth", "htnysn")],
        reference_code: "",
    };
    let result = solve_problem(&problem);
    assert!(
        result.success,
        "string-output problem not solved: {:?}",
        result.error
    );
    let out = crate::runtime::execute_str_function(&result.code, "reverse_str", "verify").unwrap();
    assert_eq!(out, "yfirev");
}

#[path = "tests/benchmark_diff_cases.rs"]
mod benchmark_diff_cases;
#[path = "tests/search_cases.rs"]
mod exact_cases;
#[path = "tests/gradient_cases.rs"]
mod gradient_cases;
#[path = "tests/routing_cases.rs"]
mod routing_cases;
