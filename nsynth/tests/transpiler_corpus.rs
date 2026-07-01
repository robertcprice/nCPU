//! Transpiler/normalizer hardening (no server, no LLM): feed a corpus of diverse
//! but KNOWN-GOOD Mog shapes (the shapes the synthesizer actually emits) through
//! `write_synthesized_project` (to_rust -> normalize_component -> crate -> cargo
//! check). Asserts the generated multi-file crate COMPILES. Locks transpiler +
//! gencode_normalize correctness across shapes and guards regressions (e.g. the
//! Rule 6 `for x in arr` move fix). Tolerates Unverified if cargo can't run.
use mog_synth::agent::repo::nl_fixture_harness::{write_synthesized_project, CompileStatus};

/// (unique module/fn name, Mog source). Names must be DISTINCT — the writer
/// re-exports every module with `pub use m::*`, so two `fn f` would collide.
fn corpus() -> Vec<(String, String)> {
    let mk = |n: &str, c: &str| (n.to_string(), c.to_string());
    vec![
        // array -> scalar fold (sum)
        mk("sum_fold", "fn sum_fold(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for item in arr {\n        acc = acc + item;\n    }\n    return acc;\n}\n"),
        // array -> scalar with arr[0] seed + REUSE (the Rule 6 borrow case: min)
        mk("min_fold", "fn min_fold(arr: [i64]) -> i64 {\n    acc: i64 = arr[0];\n    for item in arr {\n        if item < acc {\n            acc = item;\n        }\n    }\n    return acc;\n}\n"),
        // filter + reduce (guarded accumulation)
        mk("sum_pos", "fn sum_pos(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for x in arr {\n        if x > 0 {\n            acc = acc + x;\n        }\n    }\n    return acc;\n}\n"),
        // length via .len (Rule 2: .len -> .len())
        mk("count", "fn count(arr: [i64]) -> i64 {\n    n: i64 = arr.len;\n    return n;\n}\n"),
        // scalar affine
        mk("affine", "fn affine(x: i64) -> i64 {\n    return ((3 * x) + 5);\n}\n"),
        // top-level scalar branch, non-affine sides
        mk("sgnsq", "fn sgnsq(n: i64) -> i64 {\n    if n < 0 {\n        return (n * n);\n    }\n    return (n + 1);\n}\n"),
        // multi-arg
        mk("add2", "fn add2(a: i64, b: i64) -> i64 {\n    return (a + b);\n}\n"),
        // predicate returning 0/1
        mk("is_pos", "fn is_pos(n: i64) -> i64 {\n    if n > 0 {\n        return 1;\n    }\n    return 0;\n}\n"),
    ]
}

#[test]
fn transpiled_corpus_compiles_as_multifile_crate() {
    let root = std::env::temp_dir().join(format!("nsynth_transpiler_corpus_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("mkdir");

    let components = corpus();
    let outcome = write_synthesized_project(&root, "transpiler_corpus", &components)
        .unwrap_or_else(|e| panic!("write failed: {e}"));
    eprintln!("[corpus] wrote {} files, compile = {:?}", outcome.written.len(), outcome.compile);

    match &outcome.compile {
        CompileStatus::Ok => {}
        CompileStatus::Unverified(why) => eprintln!("[corpus] cargo unavailable ({why}) — tolerated"),
        CompileStatus::Failed(err) => panic!(
            "transpiled corpus does NOT compile (transpiler/normalizer bug):\n{err}"
        ),
    }
    let _ = std::fs::remove_dir_all(&root);
}
