//! Component layer: named, NL-resolvable, COMPOSABLE, VERIFIED units — one level
//! up from single ops.
//!
//! An op is a single verified function (`array_sum`). A COMPONENT is a named unit
//! that bundles several verified leaf ops into ONE compile-gated module, resolved
//! from ONE natural-language phrase ("array statistics"). This is the foundation
//! for the planner (symbolic + neural): a prompt resolves to component(s), each
//! component's leaves are synthesized + verified via the trusted op path, the
//! verified leaves are composed into a module, and the whole is compile-verified by
//! the same `cargo check` gate the greenfield writer uses. Every leaf keeps the
//! engine's 0-false-positive guarantee; the assembly is verified by compilation.
//! Nothing is trusted-but-unverified.
//!
//! FIRST SLICE: leaves-only bundles + literal surface resolution. It extends to
//! struct defs, glue templates, and emergent NL resolution (reusing the op
//! resolver's morphology/graph/WordNet machinery) — and the registry migrates from
//! this Rust const to data so it can be GROWN by mining verified builds ("writes
//! its own teachers" at the component grain).

use crate::agent::repo::nl_fixture_harness::{write_synthesized_project, CompileStatus, WriteOutcome};
use crate::linguigenesis_bridge::LinguigenesisBridge;
use std::path::Path;

/// A named unit bigger than a single op: a curated bundle of leaf ops composed into
/// one module. (First slice: leaves only.)
pub struct ComponentSpec {
    /// Module + package name for the emitted component.
    pub name: &'static str,
    /// Natural-language surface words that resolve to this component.
    pub surfaces: &'static [&'static str],
    /// `default_fn_name`s of the leaf ops this component bundles. Each is
    /// independently verified-synthesizable via the trusted op path.
    pub leaves: &'static [&'static str],
}

/// The built-in component registry. First slice: a Rust const; migrates to data +
/// emergent resolution, like `coding_registry.json`.
pub const BUILTIN_COMPONENTS: &[ComponentSpec] = &[ComponentSpec {
    name: "array_stats",
    surfaces: &["stats", "statistics", "statistic", "summary"],
    leaves: &["array_sum", "array_max", "array_min", "average", "length"],
}];

/// Resolve a natural-language phrase to a component by surface-word match. First
/// slice: literal, case-insensitive token match; graduates to the emergent op
/// resolver (morphology / graph / WordNet), the same machinery `EntityResolver`
/// uses for ops.
pub fn resolve_component(text: &str) -> Option<&'static ComponentSpec> {
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).collect();
    BUILTIN_COMPONENTS
        .iter()
        .find(|c| c.surfaces.iter().any(|s| tokens.contains(s)))
}

/// Outcome of building a component: which leaves verified, plus the write +
/// compile-gate result for the assembled module.
pub struct ComponentBuild {
    pub name: String,
    pub leaves_verified: Vec<String>,
    pub leaves_total: usize,
    pub outcome: WriteOutcome,
}

impl ComponentBuild {
    /// True iff EVERY leaf verified AND the assembled module compiles — the
    /// component is fully verified end-to-end.
    pub fn fully_verified(&self) -> bool {
        self.leaves_verified.len() == self.leaves_total
            && matches!(self.outcome.compile, CompileStatus::Ok)
    }
}

/// Build a component: synthesize each leaf (verified via the trusted op path),
/// compose the verified leaves into one module, and compile-gate the whole. A leaf
/// that fails to synthesize is DROPPED (reported via `leaves_verified`) — never
/// fabricated — and the component still assembles from what verified. Returns `Err`
/// only on write/infra failure or when nothing verified.
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
    let outcome = write_synthesized_project(root, spec.name, &components)?;
    Ok(ComponentBuild {
        name: spec.name.to_string(),
        leaves_verified,
        leaves_total: spec.leaves.len(),
        outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> std::path::PathBuf {
        // Unique per (process, tag); cleaned before use.
        let mut p = std::env::temp_dir();
        p.push(format!("nsynth_component_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn resolves_array_stats_from_prose() {
        let c = resolve_component("give me some array statistics please").expect("resolve");
        assert_eq!(c.name, "array_stats");
        assert!(resolve_component("reverse an array").is_none());
    }

    #[test]
    fn array_stats_component_synthesizes_and_compiles() {
        let bridge = LinguigenesisBridge::new();
        let spec = resolve_component("an array statistics module").expect("resolve stats");
        let root = temp_root("stats");
        let build = build_component(&bridge, spec, &root).expect("build");
        // The array-reducer leaves all carry >=2 example_cases → most verify.
        assert!(
            build.leaves_verified.len() >= 4,
            "expected >=4 verified leaves, got {:?}",
            build.leaves_verified
        );
        // The assembled multi-leaf component compiles (cargo-check gate) — the
        // whole is verified, not just the leaves.
        assert!(
            build.outcome.compile.is_ok(),
            "component must compile: {:?}",
            build.outcome.compile
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
