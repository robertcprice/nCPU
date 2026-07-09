//! PROOF-OF-CONCEPT: does linguigenesis's semantic resolver (already public on the
//! bridge, but only used as an error-handler fallback) ground NOVEL phrasings —
//! ones NOT in the ~30-op hand synonym table — to the right operation?
//!
//! If yes, the whack-a-mole fix is real: feed these candidates into the never-wrong
//! PROPOSER (verified_nl_router ranked_candidates) so the strict verify gate can
//! confirm them, WITHOUT hand-adding a synonym per phrasing.
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

fn main() {
    let bridge = LinguigenesisBridge::new();
    if let Some(e) = bridge.registry_load_error() {
        eprintln!("registry load error: {e}");
        return;
    }
    // Arg mode: probe each supplied WORD directly, showing all KG candidates + scores.
    // `probe_semantic_op topmost least bottommost peak first` — diagnoses whether a
    // superlative/positional synonym has a KG edge (tunable) or none (cross-project data).
    let words: Vec<String> = std::env::args().skip(1).collect();
    if !words.is_empty() {
        for w in &words {
            let cands = bridge.probe_op_candidates(w);
            let s: String = cands
                .iter()
                .take(8)
                .map(|(op, sc, m)| format!("{op}@{sc:.2}({m})"))
                .collect::<Vec<_>>()
                .join(", ");
            println!("{w:<16} -> [{s}]");
        }
        return;
    }
    // Phrasings deliberately OUTSIDE the hand synonym table (capability_miner nl_surface):
    // synonyms / paraphrases the token-only matcher would REFUSE today.
    let phrases = [
        "invert an array",              // invert ~ reverse -> reverse_list
        "flip a list around",           // flip ~ reverse
        "combine two numbers",          // combine ~ add
        "the total of a list",          // total ~ sum
        "the biggest of three numbers", // biggest ~ max_of_three
        "how many items are even",      // count evens
        "make the text uppercase",      // uppercase -> to_upper
        "is the number even",           // is_even
        "the remainder after division", // modulo
        "smallest value in the list",   // min
        "absolute value of a number",   // abs
        "number of characters",         // string length
    ];
    println!("{:<34} -> resolved op            (score)   [top probe candidates]", "PHRASE");
    for p in phrases {
        let resolved = bridge.resolve_phrase_op(p);
        // probe the head content word too, to show the candidate spread
        let head = p.split_whitespace().find(|w| w.len() > 3).unwrap_or(p);
        let cands = bridge.probe_op_candidates(head);
        let cand_str: String = cands
            .iter()
            .take(3)
            .map(|(op, s, _m)| format!("{op}@{s:.2}"))
            .collect::<Vec<_>>()
            .join(", ");
        match resolved {
            Some((op, score)) => println!("{p:<34} -> {op:<22} ({score:.2})   [{cand_str}]"),
            None => println!("{p:<34} -> (none)                          [{cand_str}]"),
        }
    }
}
