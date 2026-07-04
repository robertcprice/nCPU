//! Gated e2e: the local-LLM front door turns phrasing the symbolic comprehension
//! CAN'T parse into a VERIFIED program over a known op. Skips unless
//! NSYNTH_LOCAL_LLM_URL points at a running OpenAI-compatible endpoint
//! (mlx_lm.server / LM Studio) — CI without a model is unaffected.
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

/// Mode D (verify-and-repair): a task the LLM-free engine CANNOT synthesize
/// (contains-duplicate needs a nested scan / set, not an example-inducible fold).
/// The LLM writes a whole Mog program, verified against every example with repair
/// retries. Gated by NSYNTH_LOCAL_LLM_URL + set NSYNTH_LOCAL_LLM_REPAIR.
#[test]
fn local_llm_repair_loop_solves_engine_miss() {
    if std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty()).is_none() {
        eprintln!("[REPAIR] skipped (no url)");
        return;
    }
    std::env::set_var("NSYNTH_LOCAL_LLM_REPAIR", "1");
    use mog_synth::benchmark::{Example, Value};
    let ex = |a: &[i64], b: bool| Example { inputs: vec![Value::int_array(a)], expected: Value::Bool(b) };
    let exs = vec![
        ex(&[1, 2, 3, 4, 5], false),
        ex(&[1, 2, 3, 2, 5], true),
        ex(&[7, 7], true),
        ex(&[1, 2, 3], false),
        ex(&[4, 5, 6, 4], true),
    ];
    let r = LinguigenesisBridge::synthesize_via_repair_loop(
        "whether a given array of integers contains any duplicate",
        &exs,
    );
    let r = r.unwrap_or_else(|| panic!("[REPAIR] repair loop returned None (server up + model capable?)"));
    eprintln!("[REPAIR] method={}\n{}", r.method, r.code);
    assert!(r.success, "must be a verified result");
    assert!(
        mog_synth::runtime::code_reproduces_examples(&r.code, &exs),
        "the accepted program must reproduce EVERY example"
    );
}

#[test]
fn local_llm_translates_failing_phrasing_to_verified_program() {
    if std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty()).is_none() {
        eprintln!("[LLM] NSYNTH_LOCAL_LLM_URL unset — skipping (no local model served)");
        return;
    }
    let bridge = LinguigenesisBridge::new();
    assert!(bridge.registry_load_error().is_none(), "registry must load");

    // The EXACT phrasing the symbolic path mis-resolves to scalar `add` (see the
    // operand_typed_component_resolution guard: "adds the elements" → add). The
    // untrusted local LLM maps it to the known op array_sum; the op's TRUSTED
    // registry examples then drive synthesis + strict-verify.
    let req = "add up all the elements of an array";
    let res = bridge
        .synthesize_via_local_llm(req)
        .unwrap_or_else(|| panic!("[LLM] {req:?} → None (is mlx_lm.server up at NSYNTH_LOCAL_LLM_URL?)"));
    eprintln!("[LLM] {req:?} → method={}\n{}", res.method, res.code);
    assert!(res.success, "must be a strict-verified result");
    // Must be the ARRAY sum (a fold over arr), NOT scalar add: it loops the array.
    assert!(
        res.code.contains("for ") && res.code.contains("arr"),
        "expected an array fold (array_sum); got method={} code={}",
        res.method,
        res.code
    );
}

/// Mode A' (composition): a request the symbolic path mis-parses → the LLM
/// rephrases to canonical NL → the filter+reduce composition path → verified.
#[test]
fn local_llm_composition_via_rephrase() {
    if std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty()).is_none() {
        eprintln!("[LLM-COMP] skipped (no url)");
        return;
    }
    let bridge = LinguigenesisBridge::new();
    let req = "add up only the positive numbers in the list";
    let res = bridge
        .synthesize_via_local_llm(req)
        .unwrap_or_else(|| panic!("[LLM-COMP] {req:?} → None"));
    eprintln!("[LLM-COMP] {req:?} → method={}\n{}", res.method, res.code);
    assert!(res.success, "composition must strict-verify");
    // sum-of-positives = reduce ∘ filter(>0): a guarded accumulation over arr.
    assert!(
        res.code.contains("for ") && (res.code.contains("if") || res.method.contains("filter")),
        "expected a filter+reduce composition; got method={} code={}",
        res.method,
        res.code
    );
}

/// Mode B (out-of-vocab, RISKIER): no known op fits, so the LLM proposes I/O
/// EXAMPLES; the engine synthesizes from them with a held-out generalization
/// probe + strict-verify. Gated by NSYNTH_LOCAL_LLM_EXAMPLES *and* a served model.
/// Tested in isolation (Mode A can pre-empt composites with a partial op).
#[test]
fn local_llm_mode_b_out_of_vocab_examples() {
    if std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty()).is_none() {
        eprintln!("[MODE-B] skipped (no url)");
        return;
    }
    std::env::set_var("NSYNTH_LOCAL_LLM_EXAMPLES", "1");
    let bridge = LinguigenesisBridge::new();
    // Composite affine (3n+5) — no single registry op computes it, so only the
    // example-driven lane can. The held-out LLM examples guard against an
    // inconsistent spec; strict-verify guards against an unverifiable program.
    let req = "triple a number then add five";
    let res = bridge.synthesize_via_llm_examples(req);
    eprintln!(
        "[MODE-B] {req:?} → {:?}",
        res.as_ref().map(|r| (r.success, r.method.clone(), r.code.clone()))
    );
    let res = res.unwrap_or_else(|| panic!("[MODE-B] {req:?} → None (server up + examples valid?)"));
    assert!(res.success, "Mode B must yield a strict-verified program");
}

/// The agent's NL entry (synthesize_from_description) AUTO-falls-back to the LLM
/// lane when the symbolic path fails — proving the lane is actually used, not just
/// callable. Gated (inert without NSYNTH_LOCAL_LLM_URL).
#[test]
fn synthesize_from_description_auto_falls_back_to_llm() {
    if std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty()).is_none() {
        eprintln!("[AUTO] skipped (no url)");
        return;
    }
    let bridge = LinguigenesisBridge::new();
    // Symbolic mis-resolves "add ... elements" to scalar add → no array program;
    // the wrapper must auto-fall-back to the LLM lane and return a verified fold.
    let r = bridge.synthesize_from_description("add up all the elements of an array", None);
    eprintln!("[AUTO] → {:?}", r.as_ref().map(|x| (x.success, x.method.clone())));
    let r = r.expect("must produce a result");
    assert!(r.success, "auto-fallback must yield a verified program");
    assert!(r.code.contains("for ") && r.code.contains("arr"), "expected array fold: {}", r.code);
}
