//! Doc-ground a repo: `ingest_repo <git-url | local-path>` — clones (shallow) or
//! reads a directory, extracts every documented symbol (Rust/Python/Go) + the
//! README, and prints one JSON surface-form per line (lemma, terms, gloss).
//! These are UNTRUSTED RECALL candidates for the resolver; verification stays the
//! truth gate.
//!
//!     cargo run --release --bin ingest_repo -- https://github.com/user/repo.git
//!     cargo run --release --bin ingest_repo -- ./some/local/checkout

use mog_synth::doc_ingest::{filter_surface_forms, ingest_dir, ingest_readme, SurfaceForm};
use std::path::PathBuf;
use std::process::Command;

fn json_line(f: &SurfaceForm) -> String {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let terms = f
        .terms
        .iter()
        .map(|t| format!("\"{}\"", esc(t)))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"lemma\":\"{}\",\"gloss\":\"{}\",\"terms\":[{}]}}",
        esc(&f.lemma),
        esc(&f.gloss),
        terms
    )
}

fn main() {
    let arg = match std::env::args().nth(1) {
        Some(a) => a,
        None => {
            eprintln!("usage: ingest_repo <git-url | local-path>");
            std::process::exit(2);
        }
    };

    let (dir, cloned) = if arg.starts_with("http") || arg.ends_with(".git") {
        let tmp = std::env::temp_dir().join(format!("ingest_repo_{}", std::process::id()));
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

    let mut forms = ingest_dir(&dir);

    // README (project-level "what / when") as a <project> surface form.
    for name in ["README.md", "readme.md", "README", "Readme.md"] {
        let p = dir.join(name);
        if let Ok(md) = std::fs::read_to_string(&p) {
            forms.push(ingest_readme(&md));
            break;
        }
    }

    let symbols = forms.iter().filter(|f| f.lemma != "<project>").count();
    let raw_vocab: std::collections::HashSet<&str> =
        forms.iter().flat_map(|f| f.terms.iter().map(String::as_str)).collect();

    // GATE: keep only discriminating terms (document-frequency <= ~2% of symbols,
    // min 3), so a noisy corpus is safe to merge as recall vocabulary.
    let max_df = (symbols / 50).max(3);
    let gated = filter_surface_forms(&forms, max_df);
    let gated_vocab: std::collections::HashSet<&str> =
        gated.iter().flat_map(|f| f.terms.iter().map(String::as_str)).collect();

    eprintln!(
        "ingested {symbols} documented symbols (+{} README) from {}",
        forms.len() - symbols,
        dir.display()
    );
    eprintln!(
        "vocab: {} raw terms -> {} discriminating (max_df={max_df}); {} forms after gating",
        raw_vocab.len(),
        gated_vocab.len(),
        gated.len()
    );
    for f in &gated {
        println!("{}", json_line(f));
    }

    // Close the resolver-merge loop: if NSYNTH_DOC_SURFACE_FORMS points somewhere,
    // append the gated overlay there so the bridge enriches matching ops on its
    // next load (decorate-existing-only; unknown lemmas are dropped at merge time).
    if let Ok(overlay) = std::env::var("NSYNTH_DOC_SURFACE_FORMS") {
        let path = std::path::Path::new(&overlay);
        let mut all = mog_synth::doc_ingest::read_surface_forms_jsonl(path);
        all.extend(gated);
        if let Err(e) = mog_synth::doc_ingest::write_surface_forms_jsonl(path, &all) {
            eprintln!("warning: could not write overlay {overlay}: {e}");
        } else {
            eprintln!("appended overlay -> {overlay} ({} total forms)", all.len());
        }
    }

    if cloned {
        let _ = std::fs::remove_dir_all(&dir);
    }
}
