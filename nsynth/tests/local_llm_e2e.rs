//! Gated e2e: the local-LLM front door turns phrasing the symbolic comprehension
//! CAN'T parse into a VERIFIED program over a known op. Skips unless
//! NSYNTH_LOCAL_LLM_URL points at a running OpenAI-compatible endpoint
//! (mlx_lm.server / LM Studio) — CI without a model is unaffected.
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

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
