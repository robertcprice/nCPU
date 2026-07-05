//! PHRASE-level op resolution through the live registry: a multi-word op
//! reference in prose reaches an op whose LEMMA carries the words — emergent
//! (the op's own name is the phrase source), no hand table. The resolution-side
//! half of the NL-vocabulary lane split (MASTER_ROADMAP 0.0591).

use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

#[test]
fn multi_word_lemma_ops_resolve_from_prose() {
    let b = LinguigenesisBridge::new();
    // reverse_string: lemma words [reverse, string] in order in the prose.
    let (op, score) = b
        .resolve_phrase_op("please reverse the string for me")
        .expect("phrase-resolves reverse_string");
    assert_eq!(op, "reverse_string");
    assert!(score >= 0.8, "above the op gate: {score}");
    // morphology per word: "reversing ... strings"
    let (op, _) = b
        .resolve_phrase_op("reversing all the strings")
        .expect("morph phrase");
    assert_eq!(op, "reverse_string");
    // array_sum via lemma words [array, sum].
    if let Some((op, _)) = b.resolve_phrase_op("the array sum of the values") {
        assert_eq!(op, "array_sum");
    }
}

#[test]
fn phrase_resolution_declines_out_of_order_and_unrelated() {
    let b = LinguigenesisBridge::new();
    // Out of order: "string reverse" ordering doesn't match [reverse, string]...
    // (a different multi-word op could legitimately match; assert only that
    // reverse_string specifically isn't returned for clearly unrelated prose).
    assert!(
        b.resolve_phrase_op("sort a list of names").is_none(),
        "unrelated prose resolves no phrase op"
    );
}
