//! LIVE RLVR loop: the untrusted local model PROPOSES components; the compile +
//! behavior gates dispose of them. `#[ignore]` because it needs a running local
//! model (mlx_lm.server). Run explicitly:
//!
//!   NSYNTH_LOCAL_LLM_URL=http://localhost:8765/v1/chat/completions \
//!   NSYNTH_LOCAL_LLM_MODEL=lmstudio-community/gemma-4-E4B-it-MLX-8bit \
//!   cargo test --test llm_proposer_live -- --ignored --nocapture
//!
//! Observed: Gemma 4 E4B proposes verified components for negate/double/square and
//! COMPOSES multiply+square to cube a number — a novel component the seed registry
//! never had, confirmed correct by the gates. Every hallucination it emits is
//! rejected; only Accepted proposals are real.

use mog_synth::component::{propose_and_verify, ProposalVerdict};
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

#[test]
#[ignore = "needs a running local model (mlx_lm.server)"]
fn live_model_proposes_verified_components() {
    if std::env::var("NSYNTH_LOCAL_LLM_URL").is_err() {
        std::env::set_var(
            "NSYNTH_LOCAL_LLM_URL",
            "http://localhost:8765/v1/chat/completions",
        );
    }
    if std::env::var("NSYNTH_LOCAL_LLM_MODEL").is_err() {
        std::env::set_var(
            "NSYNTH_LOCAL_LLM_MODEL",
            "lmstudio-community/gemma-4-E4B-it-MLX-8bit",
        );
    }
    let bridge = LinguigenesisBridge::new();
    let mut accepted = 0;
    let mut proposed = 0;
    for req in [
        "a thing that negates a number",
        "a doubler that doubles a number",
        "a gadget that cubes a number",
    ] {
        let root = std::env::temp_dir().join(format!("nsynth_prop_live_{}_{}", std::process::id(), req.len()));
        let _ = std::fs::remove_dir_all(&root);
        match propose_and_verify(&bridge, req, &root) {
            None => {
                eprintln!("SKIP (model unreachable): {req}");
                let _ = std::fs::remove_dir_all(&root);
                return; // no server -> nothing to assert
            }
            Some(v) => {
                proposed += 1;
                eprintln!("{req} -> {v:?}");
                if matches!(v, ProposalVerdict::Accepted { .. }) {
                    accepted += 1;
                }
            }
        }
        let _ = std::fs::remove_dir_all(&root);
    }
    // The loop is PRODUCTIVE: at least one untrusted proposal survived every gate.
    assert!(proposed > 0, "model produced no proposals");
    assert!(accepted > 0, "no proposal was Accepted — loop not productive");
}
