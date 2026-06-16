//! ADVERSARIAL verification: explain_self must be REAL, not theater.
//!
//! Skeptic's claim under test: the reflection layer's `explain_self` prints the
//! ACTUAL synthesized Mog program behind the named topic, attributed to the REAL
//! teacher that recovered it — never a fabricated or empty explanation.
//!
//! These tests are written by an external adversary, not the module author. They
//! cross-check the explanation string against the ground truth held by the engine
//! itself: `engine.program()` (the composed synthesized source) and
//! `engine.methods` (the per-component teacher provenance). Anything the engine
//! cannot corroborate is treated as a fabrication and FAILS the test.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::mind::Mind;

/// Pull every `fn <name>` token out of a body of Mog source, returning the bare
/// names (so we can compare what `explain_self` quoted against what the engine
/// actually composed).
fn fn_names(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = src;
    while let Some(pos) = rest.find("fn ") {
        let after = &rest[pos + 3..];
        let name: String = after.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if !name.is_empty() {
            out.push(name);
        }
        rest = &rest[pos + 3..];
    }
    out
}

#[test]
fn explain_self_verb_inflection_quotes_a_real_fn_present_in_the_engine_program() {
    let mind = Mind::new();
    let out = mind.explain_self("verb inflection");

    // (0) Non-empty, non-degenerate: an honest explanation, not a stub.
    assert!(!out.trim().is_empty(), "explanation is empty");
    assert!(
        !out.contains("(source unavailable)"),
        "source was unavailable — explain_self could not surface real code: {out}"
    );
    assert!(
        !out.contains("I don't have a learned program"),
        "verb inflection should map to a learned program: {out}"
    );

    // (1) The explanation must QUOTE at least one real Mog `fn ` definition.
    let quoted = fn_names(&out);
    assert!(
        !quoted.is_empty(),
        "explanation contains no `fn ` definition at all: {out}"
    );

    // (2) GROUND TRUTH: build the engine independently and confirm the SAME source
    // is what the engine actually composes. Every `fn ` the explanation quotes must
    // literally appear, verbatim signature, in `engine.program()`.
    let engine = mind.engine();
    let program = engine.program();
    assert!(
        out.contains("fn verb_3sg("),
        "explanation must quote the runnable inflector `fn verb_3sg(`: {out}"
    );
    assert!(
        program.contains("fn verb_3sg("),
        "the engine program must actually contain `fn verb_3sg(` (else it's fabricated)"
    );

    // The whole quoted body of verb_3sg must be a verbatim substring of the
    // program — proving the explanation SLICED real source, not reconstructed it.
    let needle = "fn verb_3sg(";
    let q_start = out.find(needle).expect("quoted fn present");
    let q_open = out[q_start..].find('{').expect("body opens") + q_start;
    let snippet = &out[q_start..=q_open]; // signature through first brace
    assert!(
        program.contains(snippet),
        "the quoted verb_3sg signature is not a verbatim slice of engine.program(): {snippet:?}"
    );

    // Every quoted fn name must be a real composed function.
    let program_fns = fn_names(program);
    for q in &quoted {
        assert!(
            program_fns.contains(q),
            "explanation quoted `fn {q}` which does NOT exist in engine.program() — fabrication"
        );
    }

    // (3) PROVENANCE: the explanation must name the REAL teacher that recovered the
    // component, exactly as recorded in `engine.methods` (no invented attribution).
    let teacher = engine
        .method_for("regular_3sg")
        .expect("regular_3sg has a recorded teacher in engine.methods");
    assert!(!teacher.trim().is_empty(), "teacher name is empty");
    assert!(
        out.contains(teacher),
        "explanation must name the real teacher {teacher:?} from engine.methods: {out}"
    );

    // The teacher string is genuinely one of the methods the engine holds.
    assert!(
        engine.methods.iter().any(|(name, t)| *name == "regular_3sg" && t == teacher),
        "the named teacher is not the recorded provenance for regular_3sg"
    );

    // (4) The explanation must also surface the RECOVERED RULE's source
    // (fn regular_3sg) — the learned algorithm, not just the dispatch shim.
    assert!(
        out.contains("fn regular_3sg("),
        "explanation must also quote the recovered rule `fn regular_3sg(`: {out}"
    );
    assert!(
        program.contains("fn regular_3sg("),
        "engine.program() must contain the recovered rule `fn regular_3sg(`"
    );

    eprintln!("--- explain_self(\"verb inflection\") ---\n{out}\n--- teacher: {teacher} ---");
}

#[test]
fn explain_self_refuses_to_fabricate_for_an_unknown_topic() {
    // HONESTY trap: a topic with no learned program must produce an honest refusal,
    // never an invented `fn ` definition.
    let mind = Mind::new();
    let out = mind.explain_self("quantum chromodynamics");
    assert!(
        out.contains("I don't have a learned program"),
        "unknown topic must be refused honestly: {out}"
    );
    assert!(
        !out.contains("fn "),
        "unknown topic must NOT fabricate any `fn ` source: {out}"
    );
}

#[test]
fn explain_self_quoted_source_is_not_independently_fabricated_by_the_method() {
    // Independent corroboration WITHOUT trusting explain_self's own slicing: take
    // the teacher + source straight from a freshly built engine and confirm the
    // explanation is consistent with that ground truth for ALL mapped topics.
    let engine = Engine::new();
    let mind = Mind::new();

    for (topic, component, fn_name) in [
        ("verb inflection", "regular_3sg", "verb_3sg"),
        ("past tense", "regular_past", "verb_past"),
        ("noun animacy", "noun_animacy", "noun_animacy"),
        ("agreement", "valid_agreement", "valid_agreement"),
    ] {
        let out = mind.explain_self(topic);
        let sig = format!("fn {fn_name}(");
        assert!(
            engine.program().contains(&sig),
            "ground-truth engine program is missing {sig:?}"
        );
        assert!(out.contains(&sig), "explain_self({topic:?}) failed to quote {sig:?}: {out}");
        let teacher = engine.method_for(component).expect("component has a teacher");
        assert!(
            out.contains(teacher),
            "explain_self({topic:?}) failed to name real teacher {teacher:?}: {out}"
        );
    }
}
