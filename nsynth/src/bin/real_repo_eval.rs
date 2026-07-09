//! HONEST Axis-B measurement: run the repo agent on a REAL third-party crate
//! (not a scaffolded fixture). For each target it injects a bug into a real function,
//! then runs RepoAgent (index the whole crate -> localize the fn among 100+ files ->
//! synthesize a fix from the test's mined I/O -> real `cargo test` oracle -> promote) and
//! reports whether the crate's own tests pass again. The file is restored afterwards.
//!
//!   real_repo_eval /tmp/algorithms_rust
use mog_synth::agent::coding_intent::CodingIntent;
use mog_synth::agent::repo::{GuardrailPolicy, RepoAgent};
use mog_synth::agent::runtime::CodeTaskSpec;

/// (relative path, correct snippet to break, buggy replacement, NL task, cargo-test filter).
const TARGETS: &[(&str, &str, &str, &str, &str)] = &[
    (
        "src/math/factorial.rs",
        "(2..=number).product()",
        "number",
        "the factorial of a number",
        "test_factorial",
    ),
    (
        "src/math/aliquot_sum.rs",
        "(1..=number / 2).filter(|&d| number.is_multiple_of(d)).sum()",
        "number",
        "the sum of the proper divisors of a number",
        "aliquot",
    ),
    (
        "src/math/greatest_common_divisor.rs",
        "let remainder = b % a;",
        "let remainder = b - a;",
        "the greatest common divisor of two numbers",
        "greatest_common",
    ),
    (
        "src/math/sum_of_digits.rs",
        "result += num % 10;",
        "result += num;",
        "the sum of the digits of a number",
        "sum_of_digits",
    ),
];

fn main() {
    let root = std::env::args().nth(1).unwrap_or_else(|| "/tmp/algorithms_rust".to_string());
    let budget_ms: u64 = std::env::var("NSYNTH_QUERY_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30000);

    let (mut total, mut baseline_broke, mut fixed) = (0, 0, 0);
    println!("{:<38} {:>8} {:>7} {:>7}   {}", "TARGET", "baseline", "final", "fixed", "note");
    for &(rel, find, replace, nl, filter) in TARGETS {
        let path = std::path::Path::new(&root).join(rel);
        let Ok(orig) = std::fs::read_to_string(&path) else {
            println!("{rel:<38} SKIP (unreadable)");
            continue;
        };
        if !orig.contains(find) {
            println!("{rel:<38} SKIP (bug marker not found)");
            continue;
        }
        total += 1;
        // Inject the bug (first occurrence).
        let buggy = orig.replacen(find, replace, 1);
        if std::fs::write(&path, &buggy).is_err() {
            println!("{rel:<38} SKIP (unwritable)");
            continue;
        }

        let Ok(intent) = CodingIntent::from_nl(nl) else {
            let _ = std::fs::write(&path, &orig);
            println!("{rel:<38} intent-refused (\"{nl}\")");
            continue;
        };
        let cmd = format!("cargo test --lib {filter}");
        let mut spec = CodeTaskSpec::from_nl(root.clone(), nl, intent, cmd, vec!["src/**".into()], 3);
        spec.budget.max_wall_ms = budget_ms.max(60000); // real cargo compiles are slow

        let mut agent = RepoAgent::new(&root, GuardrailPolicy::default());
        let result = agent.run(&spec);
        // ALWAYS restore the real file, whatever the agent did.
        let _ = std::fs::write(&path, &orig);

        if !result.baseline_passed {
            baseline_broke += 1;
        }
        if result.success && result.final_passed {
            fixed += 1;
        }
        let note = result.error.clone().unwrap_or_default();
        println!(
            "{rel:<38} {:>8} {:>7} {:>7}   {}",
            result.baseline_passed,
            result.final_passed,
            result.success,
            &note[..note.len().min(64)]
        );
    }
    println!(
        "\nREAL-REPO (cargo-test oracle on a real third-party crate): {fixed}/{total} fixed | \
         {baseline_broke}/{total} bugs correctly detected at baseline"
    );
}
