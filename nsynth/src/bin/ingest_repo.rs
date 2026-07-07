//! Doc-ground a repo: `ingest_repo <git-url | local-path>` — clones (shallow) or
//! reads a directory, extracts every documented symbol (Rust/Python/Go) + the
//! README, and prints one JSON surface-form per line (lemma, terms, gloss).
//! These are UNTRUSTED RECALL candidates for the resolver; verification stays the
//! truth gate.
//!
//!     cargo run --release --bin ingest_repo -- https://github.com/user/repo.git
//!     cargo run --release --bin ingest_repo -- ./some/local/checkout

use mog_synth::doc_ingest::{ingest_dir, ingest_readme, SurfaceForm};
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
    eprintln!(
        "ingested {symbols} documented symbols (+{} README) from {}",
        forms.len() - symbols,
        dir.display()
    );
    for f in &forms {
        println!("{}", json_line(f));
    }

    if cloned {
        let _ = std::fs::remove_dir_all(&dir);
    }
}
