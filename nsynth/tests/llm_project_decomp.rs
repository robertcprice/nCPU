//! Gated e2e for Mode C (LLM project decomposition): an open-ended request is
//! broken by the untrusted LLM into named sub-functions, each synthesized + STRICT-
//! VERIFIED through the normal door. Asserts >=2 components verify. Demonstrates the
//! verified-multi-function capability driven from a single prompt.
//!
//! TRUST scope (asserted by construction): every returned component is strict-
//! verified; the WHOLE-artifact behavior is NOT (no example oracle for the assembly).
//!
//! Skips unless NSYNTH_LOCAL_LLM_URL is served AND NSYNTH_LOCAL_LLM_PROJECT is set.
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

#[test]
fn llm_decomposes_into_verified_components() {
    if std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty()).is_none() {
        eprintln!("[MODE-C] skipped (no url)");
        return;
    }
    std::env::set_var("NSYNTH_LOCAL_LLM_PROJECT", "1");
    let bridge = LinguigenesisBridge::new();
    assert!(bridge.registry_load_error().is_none(), "registry must load");

    // A request that naturally decomposes into several pure numeric/array helpers,
    // each individually synthesizable + verifiable (sum, max, count-positive, …).
    let req = "build helpers to analyze a list of numbers: their total, the largest one, \
               and how many are positive";
    let out = bridge.synthesize_project_via_llm(req);
    let (verified, failed) = out.unwrap_or_else(|| panic!("[MODE-C] {req:?} → None (server + plan valid?)"));
    eprintln!("[MODE-C] verified {} components, {} failed:", verified.len(), failed.len());
    for (name, r) in &verified {
        eprintln!("  ✓ {name}  [{}]", r.method);
    }
    for f in &failed {
        eprintln!("  · {f}");
    }
    // The LLM's decomposition is untrusted, but the engine must verify at least a
    // meaningful subset — a single prompt yielding >=2 strict-verified functions.
    assert!(
        verified.len() >= 2,
        "expected >=2 verified components from the decomposition, got {}",
        verified.len()
    );
    // Every returned component is a real strict-verified result.
    assert!(verified.iter().all(|(_, r)| r.success), "all returned components must be verified");
}
