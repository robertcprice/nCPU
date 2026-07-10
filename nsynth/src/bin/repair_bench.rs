//! MODEL-FREE bug-repair bench: plant a realistic subtle bug in a small crate, run the
//! agent with a GENERIC "fix the failing tests" query (no NL that hands over the fix, so
//! the repair ladder — not NL synthesis — must find it), then score with a real `cargo
//! test` oracle. Measures where the deterministic (non-model) engine actually stands on
//! bug repair, and exercises the failing-function localization end-to-end.
//!
//! Two failure shapes, deliberately:
//!   * PANIC bugs (underflow / div-by-zero / unwrap / index) panic INSIDE the production
//!     fn, so the failure reports that fn's `file:line` -> the mutation search localizes to
//!     it (the win: it stays cheap even when the file is large, `--pad` below).
//!   * ASSERTION bugs (wrong operator / method / value) fail in the TEST, so there is no
//!     production line -> the search degrades to whole-file mutation (localization no-op).
//!
//! HONESTY: passing the SHOWN tests is not correctness — with only a few asserts as the
//! oracle, a re-synthesis can OVERFIT (e.g. "parse_flag" collapsing to "has any uppercase
//! char", which satisfies 3 asserts but is not parsing anything). So every fixture also
//! carries HELD-OUT asserts the agent never sees; the crate is re-tested with them appended
//! AFTER the agent finishes. The real score is held-out-green (generalizes), not shown-green.
//!
//! Run model-free (leave NSYNTH_LOCAL_LLM_URL unset):
//!   cargo run --release --bin repair_bench
use mog_synth::agent::repo::GuardrailPolicy;
use mog_synth::agent::session::CodingAgentSession;
use std::path::Path;
use std::process::Command;

/// A padding block of decoy functions, so a fixture's buggy fn sits deep in a large file
/// (whole-file mutation would bury the fix past the search cap; localization should not).
fn decoys(n: usize) -> String {
    (0..n)
        .map(|i| format!("pub fn decoy_{i}(x: i64) -> i64 {{ x.wrapping_mul({i}).wrapping_add({i}) }}\n"))
        .collect()
}

/// One bench case. `shown` = crate source the agent sees (buggy fn + a small failing test).
/// `held_out` = extra assert statements appended AFTER the agent finishes, to detect an
/// overfit that merely satisfies the shown asserts. `expect_generalizes` = whether the
/// model-free engine is expected to produce a fix that survives the held-out asserts.
struct Case {
    id: &'static str,
    class: &'static str,
    expect_generalizes: bool,
    shown: String,
    held_out: &'static str,
}

fn c(id: &'static str, class: &'static str, expect_generalizes: bool, shown: String, held_out: &'static str) -> Case {
    Case { id, class, expect_generalizes, shown, held_out }
}

fn fixtures() -> Vec<Case> {
    vec![
        // --- ASSERTION bugs (fail in the test; whole-file mutation of correct-structure code) ---
        c(
            "wrong_compare",
            "comparison swap: returns smaller not larger",
            true,
            format!(
                "{}pub fn larger(a: i64, b: i64) -> i64 {{ if a < b {{ a }} else {{ b }} }}\n\n\
                 #[cfg(test)]\nmod tests {{\n use super::*;\n #[test]\n fn t() {{\n  \
                 assert_eq!(larger(3, 7), 7);\n  assert_eq!(larger(9, 2), 9);\n }}\n}}\n",
                decoys(6)
            ),
            "assert_eq!(larger(100, -5), 100); assert_eq!(larger(-3, -9), -3); assert_eq!(larger(0, 0), 0);",
        ),
        c(
            "wrong_arith",
            "operator swap: adds a fee that should be subtracted",
            true,
            "pub fn net(gross: i64, fee: i64) -> i64 { gross + fee }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(net(100, 30), 70);\n  assert_eq!(net(50, 5), 45);\n }\n}\n"
                .to_string(),
            "assert_eq!(net(20, 7), 13); assert_eq!(net(0, 0), 0); assert_eq!(net(9, 9), 0);",
        ),
        c(
            "wrong_method",
            "method swap: min where max is meant",
            true,
            "pub fn peak(v: &[i64]) -> i64 { *v.iter().min().unwrap() }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(peak(&[3, 9, 2]), 9);\n  assert_eq!(peak(&[1, 4, 1]), 4);\n }\n}\n"
                .to_string(),
            "assert_eq!(peak(&[5, 1, 5]), 5); assert_eq!(peak(&[-2, -9]), -2); assert_eq!(peak(&[7]), 7);",
        ),
        c(
            "off_by_one_literal",
            "off-by-one: an extra +1 in the size",
            true,
            "pub fn buffer(n: i64) -> i64 { n + n + 1 }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(buffer(5), 10);\n  assert_eq!(buffer(3), 6);\n }\n}\n"
                .to_string(),
            "assert_eq!(buffer(10), 20); assert_eq!(buffer(0), 0); assert_eq!(buffer(1), 2);",
        ),
        // --- PANIC bugs (fail INSIDE the production fn; localization targets it) ---
        c(
            "underflow_guard_padded",
            "usize underflow on VAR-1 when VAR==0, buried deep in a large file",
            true,
            format!(
                "{}pub fn tier(bytes: usize) -> usize {{ let e = bytes / 1000; e - 1 }}\n\n\
                 #[cfg(test)]\nmod tests {{\n use super::*;\n #[test]\n fn t() {{\n  \
                 assert_eq!(tier(500), 0);\n  assert_eq!(tier(2000), 1);\n }}\n}}\n",
                decoys(18)
            ),
            "assert_eq!(tier(0), 0); assert_eq!(tier(5000), 4); assert_eq!(tier(999), 0);",
        ),
        c(
            "missing_zero_guard",
            "divide-by-zero: no guard on the denominator",
            true,
            "pub fn safe_div(a: i64, b: i64) -> i64 { a / b }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(safe_div(6, 2), 3);\n  assert_eq!(safe_div(5, 0), 0);\n }\n}\n"
                .to_string(),
            "assert_eq!(safe_div(8, 0), 0); assert_eq!(safe_div(8, 4), 2); assert_eq!(safe_div(0, 3), 0);",
        ),
        // --- MORE CLASSES (frontier map: which general mutation does the engine actually lack?) ---
        c(
            "bool_negation",
            "boolean flip: missing `!` on the predicate",
            true,
            "pub fn accepts(v: &[i64]) -> bool { v.is_empty() }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(accepts(&[1]), true);\n  assert_eq!(accepts(&[]), false);\n }\n}\n"
                .to_string(),
            "assert_eq!(accepts(&[9, 9]), true); assert_eq!(accepts(&[]), false); assert_eq!(accepts(&[0]), true);",
        ),
        c(
            "two_token",
            "two wrong operators in one expression (exercises the two-edit search)",
            true,
            "pub fn total3(a: i64, b: i64, c: i64) -> i64 { a - b - c }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(total3(1, 2, 3), 6);\n  assert_eq!(total3(10, 5, 5), 20);\n }\n}\n"
                .to_string(),
            "assert_eq!(total3(0, 0, 0), 0); assert_eq!(total3(7, 8, 9), 24); assert_eq!(total3(1, 1, 1), 3);",
        ),
        c(
            "arg_swap",
            "swapped operands: b - a where a - b is meant",
            true,
            "pub fn diff(a: i64, b: i64) -> i64 { b - a }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(diff(10, 3), 7);\n  assert_eq!(diff(5, 2), 3);\n }\n}\n"
                .to_string(),
            "assert_eq!(diff(9, 4), 5); assert_eq!(diff(0, 0), 0); assert_eq!(diff(20, 1), 19);",
        ),
        c(
            "wrong_constant_1024",
            "base confusion: multiplies by 1000 where 1024 is meant",
            true,
            "pub fn kib(n: i64) -> i64 { n * 1000 }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(kib(1), 1024);\n  assert_eq!(kib(2), 2048);\n }\n}\n"
                .to_string(),
            "assert_eq!(kib(3), 3072); assert_eq!(kib(0), 0); assert_eq!(kib(10), 10240);",
        ),
        c(
            "boundary_inclusive",
            "boundary: `<` where `<=` is meant",
            true,
            "pub fn fits(x: i64, cap: i64) -> bool { x < cap }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(fits(5, 5), true);\n  assert_eq!(fits(6, 5), false);\n }\n}\n"
                .to_string(),
            "assert_eq!(fits(0, 0), true); assert_eq!(fits(3, 5), true); assert_eq!(fits(5, 4), false);",
        ),
        // --- HARDER classes (deeper frontier map) ---
        c(
            "struct_field_swap",
            "struct method reads the wrong field",
            true,
            "pub struct P { pub x: i64, pub y: i64 }\n\
             impl P { pub fn first(&self) -> i64 { self.y } }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(P { x: 3, y: 7 }.first(), 3);\n  assert_eq!(P { x: 9, y: 1 }.first(), 9);\n }\n}\n"
                .to_string(),
            "assert_eq!(P { x: 100, y: -5 }.first(), 100); assert_eq!(P { x: 0, y: 8 }.first(), 0);",
        ),
        c(
            "loop_range_bound",
            "exclusive range where inclusive is meant (sum 1..n vs 1..=n)",
            true,
            "pub fn tri(n: i64) -> i64 { let mut s = 0; let mut i = 1; while i < n { s += i; i += 1; } s }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(tri(4), 10);\n  assert_eq!(tri(5), 15);\n }\n}\n"
                .to_string(),
            "assert_eq!(tri(1), 1); assert_eq!(tri(3), 6); assert_eq!(tri(6), 21);",
        ),
        c(
            "helper_two_fn",
            "the bug is in a helper the tested fn calls",
            true,
            "fn scale(v: i64) -> i64 { v * 3 }\n\
             pub fn total(a: i64, b: i64) -> i64 { scale(a) + scale(b) }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(total(1, 2), 6);\n  assert_eq!(total(4, 0), 8);\n }\n}\n"
                .to_string(),
            "assert_eq!(total(3, 3), 12); assert_eq!(total(0, 0), 0); assert_eq!(total(5, 5), 20);",
        ),
        // --- CEILING: a coordinated multi-token rewrite. Model-free RE-SYNTHESIS from the 3 shown
        // asserts OVERFITS (it found "has any uppercase char", green on shown, wrong in general).
        // Held-out asserts expose it -> the honest ceiling where the model lane is actually needed.
        c(
            "complex_rewrite_CEILING",
            "trim + case-fold + parse together — model-free re-synth overfits the shown asserts",
            false,
            "pub fn parse_flag(s: &str) -> bool { s == \"on\" }\n\n\
             #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
             assert_eq!(parse_flag(\"  ON \"), true);\n  assert_eq!(parse_flag(\"off\"), false);\n \
             assert_eq!(parse_flag(\"On\"), true);\n }\n}\n"
                .to_string(),
            "assert_eq!(parse_flag(\"YES\"), false); assert_eq!(parse_flag(\"on\"), true); assert_eq!(parse_flag(\" on \"), true);",
        ),
    ]
}

fn write_crate(root: &Path, id: &str, lib_rs: &str) -> Result<(), String> {
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    let toml = format!(
        "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        id.replace('_', "-")
    );
    std::fs::write(root.join("Cargo.toml"), toml).map_err(|e| e.to_string())?;
    std::fs::write(root.join("src/lib.rs"), lib_rs).map_err(|e| e.to_string())?;
    Ok(())
}

/// Ground-truth oracle: does `cargo test` pass in `root`?
fn cargo_green(root: &Path) -> bool {
    Command::new("cargo")
        .args(["test", "--quiet"])
        .current_dir(root)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn main() {
    if std::env::var_os("NSYNTH_QUERY_BUDGET_MS").is_none() {
        // Bound each solve so a doomed fixture degrades to a refusal instead of spinning.
        unsafe { std::env::set_var("NSYNTH_QUERY_BUDGET_MS", "60000") };
    }
    let model = std::env::var("NSYNTH_LOCAL_LLM_URL").ok();
    println!(
        "MODEL-FREE bug-repair bench (model lane: {})\n",
        model.as_deref().unwrap_or("OFF — deterministic engine only")
    );
    let base = std::env::temp_dir().join(format!("nsynth_repair_bench_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);

    println!(
        "{:<26} {:>7} {:>7} {:>8}   {}",
        "FIXTURE", "shown", "heldout", "expect", "class"
    );
    let (mut shown_ok, mut generalized, mut total, mut as_expected) = (0usize, 0usize, 0usize, 0usize);
    for case in fixtures() {
        total += 1;
        let root = base.join(case.id);
        if let Err(e) = write_crate(&root, case.id, &case.shown) {
            println!("{:<26} write error: {e}", case.id);
            continue;
        }
        // The agent sees ONLY `shown` (buggy fn + the small failing test).
        let policy = GuardrailPolicy::default();
        match CodingAgentSession::load(&root, policy, "bench") {
            Ok(mut session) => {
                let _ = session.handle_query("fix the failing tests");
            }
            Err(e) => {
                println!("{:<26} session load error: {e}", case.id);
                continue;
            }
        }
        let shown_green = cargo_green(&root); // did the agent make the SHOWN tests pass?
        // Now append the HELD-OUT asserts the agent never saw and re-test: passing these means
        // the fix GENERALIZES (not just an overfit of the shown asserts).
        let src = std::fs::read_to_string(root.join("src/lib.rs")).unwrap_or_default();
        let with_held = format!(
            "{src}\n#[cfg(test)]\nmod heldout {{\n use super::*;\n #[test]\n fn h() {{\n  {}\n }}\n}}\n",
            case.held_out
        );
        let _ = std::fs::write(root.join("src/lib.rs"), with_held);
        let held_green = shown_green && cargo_green(&root);
        if shown_green {
            shown_ok += 1;
        }
        if held_green {
            generalized += 1;
        }
        if held_green == case.expect_generalizes {
            as_expected += 1;
        }
        let flag = if shown_green && !held_green { "  <- OVERFIT" } else { "" };
        println!(
            "{:<26} {:>7} {:>7} {:>8}   {}{}",
            case.id,
            if shown_green { "green" } else { "red" },
            if held_green { "GREEN" } else { "red" },
            if case.expect_generalizes { "general" } else { "ceiling" },
            case.class,
            flag
        );
        let _ = std::fs::remove_dir_all(&root);
    }
    let _ = std::fs::remove_dir_all(&base);
    println!(
        "\nMODEL-FREE REPAIR (honest): {generalized}/{total} GENERALIZE (pass held-out asserts) | \
         {shown_ok}/{total} passed the shown tests | {as_expected}/{total} matched expectation.\n\
         Gap between shown-pass and held-out-pass = overfit to the given tests: exactly where the \
         given oracle is too weak and a stronger spec (or the model lane) is needed."
    );
}
