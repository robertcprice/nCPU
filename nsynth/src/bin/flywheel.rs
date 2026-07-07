//! Learn-from-real-code flywheel: `flywheel <git-url | local-path>` mines
//! integer `(function, examples)` from a repo's tests/doctests and its
//! doc-comments, then feeds each to `learn_nl::teach_by_examples` — which
//! SYNTHESIZES a matching op, regression-gates it, and PERSISTS it as a named,
//! verified library op. The doc-comment supplies its NL vocabulary. Turns a real
//! repo into library capability + grounded comprehension, soundly (only ops the
//! verifier accepts are kept).
//!
//!     cargo run --release --bin flywheel -- ./some/repo

use mog_synth::comprehension::Engine;
use mog_synth::doc_ingest::{ingest_dir, ingest_fn_sources_dir, ingest_multiarg_examples_dir};
use mog_synth::foreign_op::{eval_foreign_rust, verify_foreign_rust, ForeignVerdict};
use mog_synth::learn_nl::teach_by_examples_n;
use std::path::PathBuf;
use std::process::Command;

/// Deterministic held-out inputs for the oracle probe: a spread matched to
/// `arity`, excluding anything already among the `mined` example inputs (so the
/// oracle genuinely tests fresh points). Empty for arities we don't spread.
fn fresh_inputs(arity: usize, mined: &[(Vec<i64>, i64)]) -> Vec<Vec<i64>> {
    let seen: std::collections::HashSet<&Vec<i64>> = mined.iter().map(|(i, _)| i).collect();
    let cands: Vec<Vec<i64>> = match arity {
        1 => [4, 5, 8, 11, 13, 17, 23, 50, 64, 77, 101, 128, 255, 500, 999]
            .iter()
            .map(|&n| vec![n])
            .collect(),
        2 => [
            (3, 5), (7, 4), (12, 8), (9, 9), (20, 15), (100, 7), (48, 36), (17, 5), (64, 24), (13, 29),
        ]
        .iter()
        .map(|&(a, b)| vec![a, b])
        .collect(),
        3 => [(2, 3, 4), (7, 1, 5), (10, 10, 10), (9, 4, 6), (15, 20, 25)]
            .iter()
            .map(|&(a, b, c)| vec![a, b, c])
            .collect(),
        _ => Vec::new(),
    };
    cands.into_iter().filter(|c| !seen.contains(c)).collect()
}

fn main() {
    let arg = match std::env::args().nth(1) {
        Some(a) => a,
        None => {
            eprintln!("usage: flywheel <git-url | local-path>");
            std::process::exit(2);
        }
    };
    let (dir, cloned) = if arg.starts_with("http") || arg.ends_with(".git") {
        let tmp = std::env::temp_dir().join(format!("flywheel_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        eprintln!("cloning {arg} ...");
        let ok = Command::new("git")
            .args(["clone", "--depth", "1", &arg, tmp.to_str().unwrap()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            eprintln!("git clone failed");
            std::process::exit(1);
        }
        (tmp, true)
    } else {
        (PathBuf::from(&arg), false)
    };

    // MIN examples per function (arg 2, default 3). >=3 cuts the 2-point overfit
    // (e.g. 7->8, 0->1 matching both increment and next_power_of_2): more examples
    // pin the intended op.
    let min_ex: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let examples: Vec<_> = ingest_multiarg_examples_dir(&dir)
        .into_iter()
        .filter(|(_, v)| v.len() >= min_ex)
        .collect();
    let forms = ingest_dir(&dir);
    let sources = ingest_fn_sources_dir(&dir);
    eprintln!(
        "mined {} functions (any arity) with >= {min_ex} integer examples from {}",
        examples.len(),
        dir.display()
    );

    let engine = Engine::new();
    let mut learned = 0usize;
    let mut foreign = 0usize;
    for (name, rows) in &examples {
        let arity = rows.first().map(|(i, _)| i.len()).unwrap_or(1);
        let vocab: Vec<String> = forms
            .iter()
            .find(|f| &f.lemma == name)
            .map(|f| f.terms.iter().take(6).cloned().collect())
            .unwrap_or_default();

        // ORACLE-AMPLIFY: when the repo's real source is available, probe it on
        // fresh held-out inputs the mined examples never covered and add those as
        // extra teaching examples. A synthesized closed form that merely overfits
        // the handful of mined points (the specification wall — e.g. an LCG's
        // affine form missing its final mod) disagrees with the real source on a
        // fresh input, so it fails the amplified regression gate and never
        // persists. The real source is a differential oracle, for free.
        let src = sources.get(name);
        let mut teach_rows = rows.clone();
        let mut amp = 0usize;
        if let Some(src) = src {
            let fresh = fresh_inputs(arity, rows);
            if !fresh.is_empty() {
                if let Ok(outs) = eval_foreign_rust(name, src, &fresh) {
                    for (ins, out) in fresh.into_iter().zip(outs) {
                        teach_rows.push((ins, out));
                    }
                    amp = teach_rows.len() - rows.len();
                }
            }
        }

        let outcome = teach_by_examples_n(&engine, name, &teach_rows);
        if outcome.success {
            learned += 1;
            println!(
                "LEARNED   {name:<16} arity={arity} ex={:<2} (+{amp} oracle) method={:<24} vocab={:?}",
                rows.len(),
                outcome.method.clone().unwrap_or_default(),
                vocab
            );
        } else if let Some(src) = src {
            // Engine can't SYNTHESIZE it (or its best fit failed the oracle-amplified
            // gate) — import the repo's ACTUAL code, vetted (compile+run against the
            // mined examples). Lands TENTATIVE: the frontier crosser.
            match verify_foreign_rust(name, src, rows).verdict {
                ForeignVerdict::Tentative => {
                    foreign += 1;
                    println!(
                        "FOREIGN   {name:<16} arity={arity} ex={:<2} TENTATIVE (verified foreign source, beyond/over synthesis) vocab={:?}",
                        rows.len(),
                        vocab
                    );
                }
                ForeignVerdict::Refused(why) => {
                    println!("refused   {name:<16} arity={arity} (synth + foreign both failed: {why})")
                }
            }
        } else {
            println!("refused   {name:<16} arity={arity} ex={} (engine could not verify, no source)", rows.len());
        }
    }
    eprintln!(
        "flywheel: {learned} SYNTHESIZED + {foreign} FOREIGN-TENTATIVE of {} mined",
        examples.len()
    );

    if cloned {
        let _ = std::fs::remove_dir_all(&dir);
    }
}
