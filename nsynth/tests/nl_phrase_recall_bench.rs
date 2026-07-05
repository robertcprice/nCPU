//! PHRASE-RECALL BENCHMARK — registry-derived, no hand table. For EVERY
//! synthesizable op whose lemma is multi-word (reverse_string, running_total,
//! double_each, ...), generate natural phrasings from the lemma's own words +
//! connectives and measure that phrase-level resolution reaches the op. Grows
//! automatically as the registry grows (the universal agent's NL-breadth
//! mining adds compound lemmas; this bench measures them the day they land).
//! Precision guard: unrelated prose must resolve to nothing.

use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

fn multi_word_ops(b: &LinguigenesisBridge) -> Vec<(String, String)> {
    b.op_lemmas()
        .into_iter()
        .filter(|(lemma, _)| lemma.contains('_') && lemma.splitn(2, '_').all(|w| w.len() >= 2))
        .collect()
}

/// Natural templates built from the lemma's own words (w1_w2 -> "w1 w2" order).
fn phrasings(lemma: &str) -> Vec<String> {
    let words: Vec<&str> = lemma.split('_').filter(|w| !w.is_empty()).collect();
    let joined = words.join(" ");
    let of_join = words.join(" of the ");
    vec![
        format!("please {joined} now"),
        format!("the {of_join}"),
        format!("compute the {joined} for me"),
    ]
}

#[test]
fn phrase_recall_over_live_registry() {
    let b = LinguigenesisBridge::new();
    let ops = multi_word_ops(&b);
    assert!(
        ops.len() >= 8,
        "registry should expose multi-word ops (got {}): {ops:?}",
        ops.len()
    );
    let mut hits = 0usize;
    let mut total = 0usize;
    let mut misses: Vec<String> = Vec::new();
    for (lemma, fnn) in &ops {
        for phrase in phrasings(lemma) {
            total += 1;
            match b.resolve_phrase_op(&phrase) {
                Some((got, _)) if &got == fnn || got == *lemma => hits += 1,
                other => misses.push(format!("{phrase:?} -> {other:?} (want {fnn})")),
            }
        }
    }
    let recall = hits as f64 / total as f64;
    eprintln!("[phrase-recall] {hits}/{total} = {:.1}% over {} ops", recall * 100.0, ops.len());
    for m in misses.iter().take(6) {
        eprintln!("[phrase-recall] miss: {m}");
    }
    // Floor: the straight-order templates must overwhelmingly resolve. The
    // "of the" template can legitimately lose on some shapes; 75% floor keeps
    // the bench honest without brittleness.
    assert!(
        recall >= 0.75,
        "phrase recall {:.1}% below floor; misses:\n{}",
        recall * 100.0,
        misses.join("\n")
    );
}

#[test]
fn phrase_precision_unrelated_prose_resolves_nothing() {
    let b = LinguigenesisBridge::new();
    for prose in [
        "sort a list of names",
        "open the file and count the lines",
        "hello world program",
        "make it faster please",
        "the quick brown fox jumps",
    ] {
        assert!(
            b.resolve_phrase_op(prose).is_none(),
            "{prose:?} must not phrase-resolve"
        );
    }
}
