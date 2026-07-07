//! W2 — the PRODUCER for the doc-ingest resolver overlay. Crawls a source tree,
//! derives NL surface forms from `///`/docstring comments (noise-gated by document
//! frequency), and writes the JSONL overlay the resolver already knows how to merge
//! (`LinguigenesisBridge::merge_doc_surface_forms`, gated by `NSYNTH_DOC_SURFACE_FORMS`).
//!
//! This closes the "doc_ingest is built but never invoked" gap: the consume side was
//! already wired into registry init; nothing generated the overlay. Run this over the
//! op-library source (or any user repo) to auto-derive routable vocabulary for every
//! DOCUMENTED op — replacing the hand-authored surface-form bottleneck. The merge is
//! DECORATE-EXISTING-ONLY, so a noisy overlay can only enrich real ops' recall, never
//! inject spurious entities; the verify gate keeps correctness regardless.
//!
//! Usage: ingest_docs [SRC_DIR] [OUT_JSONL] [MAX_DF]
//!   SRC_DIR   default: src            (the crate's own op-library docs)
//!   OUT_JSONL default: .mog_synth_nl_overlay.jsonl
//!   MAX_DF    default: 8  (drop a term appearing as a discriminator in > MAX_DF ops)
//! Then: export NSYNTH_DOC_SURFACE_FORMS=<OUT_JSONL> to activate enrichment.
use mog_synth::doc_ingest::{filter_surface_forms, ingest_dir, write_surface_forms_jsonl};
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let src = args.next().unwrap_or_else(|| "src".to_string());
    let out = args.next().unwrap_or_else(|| ".mog_synth_nl_overlay.jsonl".to_string());
    let max_df: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    let raw = ingest_dir(Path::new(&src));
    let forms = filter_surface_forms(&raw, max_df);
    let total_terms: usize = forms.iter().map(|f| f.terms.len()).sum();

    if let Err(e) = write_surface_forms_jsonl(Path::new(&out), &forms) {
        eprintln!("ingest_docs: failed to write {out}: {e}");
        std::process::exit(1);
    }

    eprintln!(
        "ingest_docs: {} docs -> {} surface forms ({} terms, df<={max_df}) -> {out}",
        raw.len(),
        forms.len(),
        total_terms
    );
    // A few samples so the caller can eyeball the derived vocabulary.
    for sf in forms.iter().take(8) {
        eprintln!("  {} <- [{}]", sf.lemma, sf.terms.join(", "));
    }
    eprintln!("Activate: export NSYNTH_DOC_SURFACE_FORMS={out}");
}
