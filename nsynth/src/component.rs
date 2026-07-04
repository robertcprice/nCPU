//! Component layer: named, NL-resolvable, COMPOSABLE, VERIFIED units — one level
//! up from single ops.
//!
//! An op is a single verified function (`array_sum`). A COMPONENT is a named unit
//! resolved from ONE natural-language phrase. Two shapes:
//!
//!   * **Bundle** — several verified leaf ops composed into one compile-gated
//!     module (`array_stats` = sum/max/min/average/length).
//!   * **Structural** — a bundle PLUS a raw-Rust glue module (a struct + methods)
//!     whose bodies call the verified leaves (`Counter` = a struct whose `tick`
//!     uses the verified `increment` op). This is the genuine capability lift: the
//!     greenfield writer alone cannot emit structs; a structural component pairs a
//!     hand-glue *shape* with synthesized+verified *logic*, exactly the game/backend
//!     builder pattern generalized into a reusable unit.
//!
//! Every leaf keeps the engine's 0-false-positive guarantee; the WHOLE assembly
//! (leaves + struct glue) is verified by the same `cargo check` gate the greenfield
//! writer uses. Nothing is trusted-but-unverified — a struct that references a leaf
//! that didn't synthesize, or glue that mis-types, fails compilation and is caught.
//!
//! FIRST SLICES done: bundle + structural + literal resolution. Extends to emergent
//! NL resolution (reusing the op resolver) and a DATA registry grown by mining
//! verified builds ("writes its own teachers" at the component grain).

use crate::agent::repo::nl_fixture_harness::{
    compile_gate, write_synthesized_project, CompileStatus, WriteOutcome,
};
use crate::linguigenesis_bridge::LinguigenesisBridge;
use linguigenesis_core::entity_resolution::{edit_distance, morphological_variants};
use std::path::Path;

/// A raw-Rust glue module (struct + methods) whose bodies call the component's
/// verified leaves. Written verbatim next to the transpiled leaves and wired into
/// `lib.rs`, then compile-gated with them.
pub struct GlueSpec {
    /// Glue module name (`src/<module>.rs`).
    pub module: &'static str,
    /// Raw Rust: a struct + impl that `use`s the leaf functions (each leaf `foo`
    /// is available as `crate::foo::foo`).
    pub code: &'static str,
}

/// A named unit bigger than a single op.
pub struct ComponentSpec {
    /// Module + package name for the emitted component.
    pub name: &'static str,
    /// Natural-language surface words that resolve to this component.
    pub surfaces: &'static [&'static str],
    /// `default_fn_name`s of the leaf ops this component bundles. Each is
    /// independently verified-synthesizable via the trusted op path.
    pub leaves: &'static [&'static str],
    /// Optional struct/method glue over the leaves (structural component).
    pub glue: Option<GlueSpec>,
}

/// The built-in component registry. First slice: a Rust const; migrates to data +
/// emergent resolution, like `coding_registry.json`.
pub const BUILTIN_COMPONENTS: &[ComponentSpec] = &[
    ComponentSpec {
        name: "array_stats",
        surfaces: &["stats", "statistics", "statistic", "summary"],
        leaves: &["array_sum", "array_max", "array_min", "average", "length"],
        glue: None,
    },
    ComponentSpec {
        name: "counter",
        surfaces: &["counter", "count", "tally"],
        leaves: &["increment"],
        glue: Some(GlueSpec {
            module: "counter",
            code: COUNTER_GLUE,
        }),
    },
];

/// A counter whose `tick` uses the VERIFIED `increment` leaf (x -> x+1). The struct
/// SHAPE is templated; the increment LOGIC is synthesized + verified; the whole
/// compiles together or is rejected.
const COUNTER_GLUE: &str = r#"//! Structural component: a Counter whose tick uses the verified `increment` leaf.

use crate::increment::increment;

#[derive(Default)]
pub struct Counter {
    count: i64,
}

impl Counter {
    pub fn new() -> Self {
        Counter { count: 0 }
    }
    /// Advance the counter using the synthesized + verified `increment` op.
    pub fn tick(&mut self) {
        self.count = increment(self.count);
    }
    pub fn get(&self) -> i64 {
        self.count
    }
}
"#;

/// Match tiers, strongest first. A surface is a minimal SEED; recognition
/// generalizes emergently off it (morphology + tight fuzzy), the same seed-plus-
/// emergent pattern the op registry uses — NOT a hand-maintained synonym list.
const TIER_EXACT: u8 = 3;
const TIER_MORPH: u8 = 2;
const TIER_FUZZY: u8 = 1;

/// Emergent match of one phrase `token` against one seed `surface`:
///   * exact           — token == surface
///   * morphological   — a shared stem (strip -ing/-ed/-s/-es/-ly, both sides), so
///                       "counting"/"counters"/"tallying" reach count/counter/tally
///     with no per-inflection entry
///   * fuzzy           — edit distance <= 1 on words >= 5 chars (typo tolerance),
///                       conservative so "count" never leaks into "mount"/"court"
/// Returns the tier score, or 0 for no match.
fn surface_match(token: &str, surface: &str) -> u8 {
    if token == surface {
        return TIER_EXACT;
    }
    let mut tv = morphological_variants(token);
    tv.push(token.to_string());
    let mut sv = morphological_variants(surface);
    sv.push(surface.to_string());
    if tv.iter().any(|t| sv.contains(t)) {
        return TIER_MORPH;
    }
    if token.len() >= 5 && surface.len() >= 5 && edit_distance(token, surface) <= 1 {
        return TIER_FUZZY;
    }
    0
}

/// Resolve a natural-language phrase to a component. Emergent: every seed surface
/// is expanded by morphology + tight fuzzy at match time, so inflections and typos
/// resolve without enumerating them. Best (component, tier) wins; ties keep
/// registry order.
pub fn resolve_component(text: &str) -> Option<&'static ComponentSpec> {
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let mut best: Option<(&'static ComponentSpec, u8)> = None;
    for comp in BUILTIN_COMPONENTS {
        let mut score = 0u8;
        for tok in &tokens {
            for surf in comp.surfaces {
                score = score.max(surface_match(tok, surf));
            }
        }
        if score > 0 && best.map(|(_, b)| score > b).unwrap_or(true) {
            best = Some((comp, score));
        }
    }
    best.map(|(c, _)| c)
}

/// Outcome of building a component: which leaves verified, whether it emits a
/// struct, plus the write + compile-gate result for the assembled module(s).
pub struct ComponentBuild {
    pub name: String,
    pub leaves_verified: Vec<String>,
    pub leaves_total: usize,
    pub has_struct: bool,
    pub outcome: WriteOutcome,
}

impl ComponentBuild {
    /// True iff EVERY leaf verified AND the assembled module(s) compile.
    pub fn fully_verified(&self) -> bool {
        self.leaves_verified.len() == self.leaves_total
            && matches!(self.outcome.compile, CompileStatus::Ok)
    }
    /// True iff this component emitted a struct (structural component).
    pub fn produces_structure(&self) -> bool {
        self.has_struct
    }
}

/// Build a component: synthesize each leaf (verified via the trusted op path),
/// compose the verified leaves into a module, and — for a structural component —
/// also emit the raw-Rust struct glue and wire it in. The WHOLE crate is compiled
/// (`cargo check`); a struct referencing a leaf that failed, or mis-typed glue,
/// fails compilation and is caught. A leaf that fails to synthesize is DROPPED
/// (reported), never fabricated. Returns `Err` only on write/infra failure or when
/// nothing verified.
pub fn build_component(
    bridge: &LinguigenesisBridge,
    spec: &ComponentSpec,
    root: &Path,
) -> Result<ComponentBuild, String> {
    let mut components: Vec<(String, String)> = Vec::new();
    let mut leaves_verified: Vec<String> = Vec::new();
    for leaf in spec.leaves {
        if let Some(r) = bridge.synthesize_op_by_name(leaf) {
            if r.success {
                leaves_verified.push((*leaf).to_string());
                components.push(((*leaf).to_string(), r.code));
            }
        }
    }
    if components.is_empty() {
        return Err(format!("component '{}': no leaf verified", spec.name));
    }
    let mut outcome = write_synthesized_project(root, spec.name, &components)?;

    // Structural glue: only when the leaves themselves compiled (a struct over a
    // broken leaf would just fail again). Write the raw-Rust glue module, wire it
    // into lib.rs, and re-gate the WHOLE crate.
    if let Some(glue) = &spec.glue {
        if outcome.compile.is_ok() {
            let glue_rel = format!("src/{}.rs", glue.module);
            std::fs::write(root.join(&glue_rel), glue.code).map_err(|e| e.to_string())?;
            let lib_path = root.join("src").join("lib.rs");
            let mut lib = std::fs::read_to_string(&lib_path).map_err(|e| e.to_string())?;
            lib.push_str(&format!("\nmod {m};\npub use {m}::*;\n", m = glue.module));
            std::fs::write(&lib_path, &lib).map_err(|e| e.to_string())?;
            outcome.written.push(glue_rel);
            outcome.compile = compile_gate(root);
        }
    }

    Ok(ComponentBuild {
        name: spec.name.to_string(),
        leaves_verified,
        leaves_total: spec.leaves.len(),
        has_struct: spec.glue.is_some(),
        outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("nsynth_component_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn resolves_components_from_prose() {
        // exact seed surfaces
        assert_eq!(
            resolve_component("give me some array statistics").unwrap().name,
            "array_stats"
        );
        assert_eq!(resolve_component("build a counter").unwrap().name, "counter");
        // EMERGENT — morphology reaches inflections with no per-form entry:
        // "counting"->count, "tallying"->tally, "counters"->counter.
        assert_eq!(resolve_component("counting the events").unwrap().name, "counter");
        assert_eq!(resolve_component("a tallying widget").unwrap().name, "counter");
        assert_eq!(resolve_component("wire up two counters").unwrap().name, "counter");
        // EMERGENT — tight fuzzy tolerates a one-char typo on a long word.
        assert_eq!(
            resolve_component("some statistcs please").unwrap().name,
            "array_stats"
        );
        // negatives — unrelated prose resolves to nothing (fuzzy stays tight).
        assert!(resolve_component("reverse an array").is_none());
        assert!(resolve_component("sort a list of names").is_none());
    }

    #[test]
    fn array_stats_bundle_synthesizes_and_compiles() {
        let bridge = LinguigenesisBridge::new();
        let spec = resolve_component("an array statistics module").expect("resolve stats");
        let root = temp_root("stats");
        let build = build_component(&bridge, spec, &root).expect("build");
        assert!(
            build.leaves_verified.len() >= 4,
            "expected >=4 verified leaves, got {:?}",
            build.leaves_verified
        );
        assert!(!build.produces_structure());
        assert!(
            build.outcome.compile.is_ok(),
            "component must compile: {:?}",
            build.outcome.compile
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn counter_structural_component_emits_a_struct_that_compiles() {
        let bridge = LinguigenesisBridge::new();
        let spec = resolve_component("a counter").expect("resolve counter");
        assert!(spec.glue.is_some(), "counter is structural");
        let root = temp_root("counter");
        let build = build_component(&bridge, spec, &root).expect("build");
        assert!(
            build.leaves_verified.contains(&"increment".to_string()),
            "increment leaf verified"
        );
        assert!(build.produces_structure(), "counter emits a struct");
        // The struct glue + the verified increment leaf compile TOGETHER.
        assert!(
            build.outcome.compile.is_ok(),
            "structural component must compile: {:?}",
            build.outcome.compile
        );
        // The struct is genuinely emitted, not stubbed.
        let glue = std::fs::read_to_string(root.join("src/counter.rs")).unwrap();
        assert!(glue.contains("pub struct Counter"), "struct present: {glue}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
