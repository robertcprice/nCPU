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

/// Synthesize the verified `(name, code)` pairs for a set of leaves via the
/// TRUSTED op path, de-duplicated by name (sibling components may share a leaf).
/// A leaf that fails to synthesize is DROPPED, never fabricated. Also returns the
/// verified leaf names in encounter order.
fn synth_leaves(
    bridge: &LinguigenesisBridge,
    leaf_sets: &[&[&'static str]],
) -> (Vec<(String, String)>, Vec<String>) {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut components: Vec<(String, String)> = Vec::new();
    let mut verified: Vec<String> = Vec::new();
    for leaves in leaf_sets {
        for leaf in *leaves {
            if !seen.insert((*leaf).to_string()) {
                continue;
            }
            if let Some(r) = bridge.synthesize_op_by_name(leaf) {
                if r.success {
                    verified.push((*leaf).to_string());
                    components.push(((*leaf).to_string(), r.code));
                }
            }
        }
    }
    (components, verified)
}

/// Write a glue module verbatim and wire it into the crate's lib.rs. Idempotent on
/// the module name (a repeated glue module is skipped). Returns the written rel path
/// if it was newly wired.
fn write_and_wire_glue(root: &Path, glue: &GlueSpec) -> Result<Option<String>, String> {
    let glue_rel = format!("src/{}.rs", glue.module);
    let lib_path = root.join("src").join("lib.rs");
    let mut lib = std::fs::read_to_string(&lib_path).map_err(|e| e.to_string())?;
    let decl = format!("mod {};", glue.module);
    if lib.contains(&decl) {
        return Ok(None); // already wired
    }
    std::fs::write(root.join(&glue_rel), glue.code).map_err(|e| e.to_string())?;
    lib.push_str(&format!("\nmod {m};\npub use {m}::*;\n", m = glue.module));
    std::fs::write(&lib_path, &lib).map_err(|e| e.to_string())?;
    Ok(Some(glue_rel))
}

/// Build ONE component: synthesize its leaves, compose them into a module, and —
/// for a structural component — also emit the raw-Rust struct glue and wire it in.
/// The WHOLE crate is compiled (`cargo check`); a struct referencing a leaf that
/// failed, or mis-typed glue, fails compilation and is caught. Returns `Err` only
/// on write/infra failure or when nothing verified.
pub fn build_component(
    bridge: &LinguigenesisBridge,
    spec: &ComponentSpec,
    root: &Path,
) -> Result<ComponentBuild, String> {
    let (components, leaves_verified) = synth_leaves(bridge, &[spec.leaves]);
    if components.is_empty() {
        return Err(format!("component '{}': no leaf verified", spec.name));
    }
    let mut outcome = write_synthesized_project(root, spec.name, &components)?;

    // Structural glue: only when the leaves themselves compiled (a struct over a
    // broken leaf would just fail again). Re-gate the WHOLE crate after wiring.
    if let Some(glue) = &spec.glue {
        if outcome.compile.is_ok() {
            if let Some(rel) = write_and_wire_glue(root, glue)? {
                outcome.written.push(rel);
                outcome.compile = compile_gate(root);
            }
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

/// Resolve ALL components a phrase mentions (each with a positive emergent match),
/// in registry order — the multi-component front door. "a counter and array
/// statistics" -> [counter, array_stats].
pub fn resolve_components(text: &str) -> Vec<&'static ComponentSpec> {
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    BUILTIN_COMPONENTS
        .iter()
        .filter(|comp| {
            tokens
                .iter()
                .any(|tok| comp.surfaces.iter().any(|s| surface_match(tok, s) > 0))
        })
        .collect()
}

/// Outcome of a multi-component project build.
pub struct ProjectBuild {
    pub components: Vec<String>,
    pub leaves_verified: Vec<String>,
    /// Glue module names emitted (structural components in the project).
    pub structs: Vec<String>,
    pub outcome: WriteOutcome,
}

impl ProjectBuild {
    pub fn compiles(&self) -> bool {
        matches!(self.outcome.compile, CompileStatus::Ok)
    }
}

/// Build a MULTI-component project into ONE crate: the union of all components'
/// verified leaves plus each structural component's struct glue, wired into a
/// single lib.rs and compile-gated together. This is the planner's first symbolic
/// form — a prompt naming several concepts becomes one verified crate. Leaves are
/// synthesized once even when shared; glue modules are de-duplicated. Returns `Err`
/// only on write/infra failure or when no leaf across any component verified.
pub fn build_project(
    bridge: &LinguigenesisBridge,
    specs: &[&ComponentSpec],
    root: &Path,
) -> Result<ProjectBuild, String> {
    if specs.is_empty() {
        return Err("build_project: no components".to_string());
    }
    let leaf_sets: Vec<&[&'static str]> = specs.iter().map(|s| s.leaves).collect();
    let (components, leaves_verified) = synth_leaves(bridge, &leaf_sets);
    if components.is_empty() {
        return Err("build_project: no leaf verified across any component".to_string());
    }
    let pkg = specs
        .iter()
        .map(|s| s.name)
        .collect::<Vec<_>>()
        .join("_");
    let mut outcome = write_synthesized_project(root, &pkg, &components)?;

    let mut structs: Vec<String> = Vec::new();
    if outcome.compile.is_ok() {
        let mut wired_any = false;
        for spec in specs {
            if let Some(glue) = &spec.glue {
                if write_and_wire_glue(root, glue)?.is_some() {
                    outcome.written.push(format!("src/{}.rs", glue.module));
                    structs.push(glue.module.to_string());
                    wired_any = true;
                }
            }
        }
        if wired_any {
            outcome.compile = compile_gate(root);
        }
    }

    Ok(ProjectBuild {
        components: specs.iter().map(|s| s.name.to_string()).collect(),
        leaves_verified,
        structs,
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

    #[test]
    fn multi_component_project_wires_struct_and_bundle_into_one_crate() {
        // One prompt naming two concepts resolves to two components...
        let specs = resolve_components("a counter and some array statistics");
        let names: Vec<_> = specs.iter().map(|s| s.name).collect();
        assert!(
            names.contains(&"counter") && names.contains(&"array_stats"),
            "resolved both components: {names:?}"
        );
        // ...and builds into ONE verified crate.
        let bridge = LinguigenesisBridge::new();
        let root = temp_root("project");
        let build = build_project(&bridge, &specs, &root).expect("build");
        // union of leaves: increment (counter) + array reducers (stats)
        assert!(build.leaves_verified.contains(&"increment".to_string()), "{:?}", build.leaves_verified);
        assert!(
            build.leaves_verified.iter().any(|l| l == "array_sum"),
            "stats leaves present: {:?}",
            build.leaves_verified
        );
        // the structural component contributed its struct
        assert!(
            build.structs.contains(&"counter".to_string()),
            "counter struct emitted: {:?}",
            build.structs
        );
        // one crate, compiles together
        assert!(build.compiles(), "project compiles: {:?}", build.outcome.compile);
        // struct + a bundle leaf share the SAME lib.rs
        let lib = std::fs::read_to_string(root.join("src/lib.rs")).unwrap();
        assert!(
            lib.contains("mod counter;") && lib.contains("mod increment;"),
            "one lib wires both: {lib}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
