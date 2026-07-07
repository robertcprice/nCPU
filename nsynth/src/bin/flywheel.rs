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
use mog_synth::foreign_op::{verify_foreign_rust, ForeignVerdict};
use mog_synth::learn_nl::teach_by_examples_n;
use std::path::PathBuf;
use std::process::Command;

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
        let outcome = teach_by_examples_n(&engine, name, rows);
        if outcome.success {
            learned += 1;
            println!(
                "LEARNED   {name:<16} arity={arity} ex={:<2} method={:<24} vocab={:?}",
                rows.len(),
                outcome.method.clone().unwrap_or_default(),
                vocab
            );
        } else if let Some(src) = sources.get(name) {
            // Engine can't SYNTHESIZE it — import the repo's ACTUAL code, vetted
            // (compile+run against the examples). Lands TENTATIVE: the frontier crosser.
            match verify_foreign_rust(name, src, rows).verdict {
                ForeignVerdict::Tentative => {
                    foreign += 1;
                    println!(
                        "FOREIGN   {name:<16} arity={arity} ex={:<2} TENTATIVE (verified foreign source, beyond synthesis) vocab={:?}",
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
