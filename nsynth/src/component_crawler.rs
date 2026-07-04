//! Component crawler — a NON-LLM discovery engine that CONTINUOUSLY enumerates
//! compositions of verified leaf ops, verifies each end to end (compile + a
//! differential property gate vs the native composition), dedups by BEHAVIOR, and
//! documents every genuinely novel verified discovery to an append-only log.
//!
//! This is the "writes its own teachers" flywheel made continuous + purely
//! symbolic: no model in the loop. It sweeps the composition space `f ∘ g` over the
//! scalar leaves, and each composition that (a) behaves unlike any known op or a
//! prior discovery AND (b) actually compiles + matches its native oracle across a
//! grid is recorded as a new op — e.g. `double ∘ double = quadruple`,
//! `square ∘ increment = (x+1)^2`. Trivial/identity collapses (`negate ∘ negate`,
//! `increment ∘ decrement`) are skipped by the behavior signature.
//!
//! Run one bounded pass with `crawl_once` (resumable via the log); loop it for a
//! constant crawl (see `src/bin/discover.rs`). Novelty is decided by the op's
//! output vector on a fixed probe grid, so re-runs never re-log the same behavior.

use crate::component::{build_component, ComponentSpec, GlueSpec};
use crate::linguigenesis_bridge::LinguigenesisBridge;
use std::collections::HashSet;
use std::path::Path;

/// Probe grid the behavior signature is sampled on. Small magnitudes keep
/// square-heavy compositions inside i64 comfortably.
const GRID: [i64; 8] = [-3, -2, -1, 0, 1, 2, 3, 4];

/// The verified scalar leaves the crawler composes (each synthesizes with >=2
/// examples, confirmed by the leaf probe).
const SCALARS: [&str; 6] = ["increment", "decrement", "double", "triple", "negate", "square"];

/// Native semantics of each scalar op — the differential ORACLE used only to
/// generate the property assertions + dedup by behavior. The synthesized leaves are
/// still independently verified; this just encodes each op's known spec.
fn native(name: &str) -> fn(i64) -> i64 {
    match name {
        "increment" => |x| x + 1,
        "decrement" => |x| x - 1,
        "double" => |x| 2 * x,
        "triple" => |x| 3 * x,
        "negate" => |x| -x,
        "square" => |x| x * x,
        _ => |x| x,
    }
}

/// Behavior signature: the op's outputs across the probe grid. Two ops with the
/// same signature are behaviorally indistinguishable here.
fn signature_of(f: impl Fn(i64) -> i64) -> Vec<i64> {
    GRID.iter().map(|&x| f(x)).collect()
}

/// A novel verified composition the crawler discovered.
#[derive(Debug, Clone)]
pub struct Discovery {
    /// `compose_<outer>_<inner>`.
    pub name: String,
    pub outer: String,
    pub inner: String,
    /// Output vector on the probe grid (its behavioral identity).
    pub signature: Vec<i64>,
}

/// Build the component that computes `outer(inner(x))`, with a property smoke
/// asserting it equals the native composition across the grid (values baked as
/// literals — the differential oracle).
fn composition_spec(outer: &str, inner: &str) -> ComponentSpec {
    let name = format!("compose_{outer}_{inner}");
    let no = native(outer);
    let ni = native(inner);
    let cases: Vec<String> = GRID
        .iter()
        .map(|&x| format!("({x}, {})", no(ni(x))))
        .collect();
    let mut imports = format!("use crate::{inner}::{inner};\n");
    if outer != inner {
        imports.push_str(&format!("use crate::{outer}::{outer};\n"));
    }
    let code = format!(
        "//! Discovered composition: {outer} of {inner}.\n\n{imports}\npub fn apply(x: i64) -> i64 {{\n    {outer}({inner}(x))\n}}\n"
    );
    let smoke = format!(
        "\n#[cfg(test)]\nmod composed_behaves {{\n    use super::apply;\n    #[test]\n    fn matches_native_composition() {{\n        for (x, expected) in [{}] {{\n            assert_eq!(apply(x), expected, \"at x={{}}\", x);\n        }}\n    }}\n}}\n",
        cases.join(", ")
    );
    let mut leaves = vec![inner.to_string()];
    if outer != inner {
        leaves.push(outer.to_string());
    }
    ComponentSpec {
        name: name.clone(),
        surfaces: vec![name.clone()],
        leaves,
        glue: Some(GlueSpec {
            module: name,
            code,
            smoke: Some(smoke),
        }),
    }
}

/// Load the behavior signatures already recorded in the log (so a continuous crawl
/// never re-logs the same behavior).
fn known_from_log(log_path: &Path) -> HashSet<Vec<i64>> {
    let mut set = HashSet::new();
    if let Ok(text) = std::fs::read_to_string(log_path) {
        for line in text.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(sig) = v["signature"].as_array() {
                    set.insert(sig.iter().filter_map(|n| n.as_i64()).collect());
                }
            }
        }
    }
    set
}

/// Append one discovery to the JSONL log.
fn append_discovery(log_path: &Path, d: &Discovery) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let line = serde_json::json!({
        "name": d.name,
        "outer": d.outer,
        "inner": d.inner,
        "signature": d.signature,
        "grid": GRID,
    })
    .to_string();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    writeln!(f, "{line}")
}

/// One bounded discovery pass: sweep `outer ∘ inner` over the scalar leaves, skip
/// any whose behavior matches a known op / identity / a prior discovery, and for
/// each genuinely novel signature BUILD + VERIFY the composition (compile + the
/// differential property gate). Every verified-novel one is appended to `log_path`
/// and returned. Stops after `budget` new discoveries. Resumable: re-running
/// continues where the log left off.
pub fn crawl_once(
    bridge: &LinguigenesisBridge,
    log_path: &Path,
    work_root: &Path,
    budget: usize,
) -> Vec<Discovery> {
    // Seed the known-behavior set with identity + every single leaf, then everything
    // already in the log.
    let mut known: HashSet<Vec<i64>> = HashSet::new();
    known.insert(signature_of(|x| x));
    for s in SCALARS {
        known.insert(signature_of(native(s)));
    }
    known.extend(known_from_log(log_path));

    let mut found = Vec::new();
    'sweep: for outer in SCALARS {
        for inner in SCALARS {
            if found.len() >= budget {
                break 'sweep;
            }
            let no = native(outer);
            let ni = native(inner);
            let sig = signature_of(|x| no(ni(x)));
            if known.contains(&sig) {
                continue; // behavior already known — not novel
            }
            let spec = composition_spec(outer, inner);
            let croot = work_root.join(&spec.name);
            let _ = std::fs::remove_dir_all(&croot);
            let verified = matches!(
                build_component(bridge, &spec, &croot),
                Ok(ref b) if b.outcome.compile.is_ok() && b.behaves()
            );
            let _ = std::fs::remove_dir_all(&croot);
            if verified {
                known.insert(sig.clone());
                let d = Discovery {
                    name: spec.name,
                    outer: outer.to_string(),
                    inner: inner.to_string(),
                    signature: sig,
                };
                let _ = append_discovery(log_path, &d);
                found.push(d);
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_distinguishes_and_collapses_correctly() {
        // negate ∘ negate == identity (collapses); double ∘ double != any single op.
        let id = signature_of(|x| x);
        let neg = native("negate");
        assert_eq!(signature_of(|x| neg(neg(x))), id, "negate∘negate = identity");
        let dbl = native("double");
        assert_ne!(signature_of(|x| dbl(dbl(x))), signature_of(native("double")));
    }

    #[test]
    fn crawl_finds_and_logs_novel_verified_compositions() {
        let bridge = LinguigenesisBridge::new();
        let tag = std::process::id();
        let log = std::env::temp_dir().join(format!("nsynth_discoveries_{tag}.jsonl"));
        let work = std::env::temp_dir().join(format!("nsynth_crawl_work_{tag}"));
        let _ = std::fs::remove_file(&log);
        let _ = std::fs::remove_dir_all(&work);

        let found = crawl_once(&bridge, &log, &work, 3);
        assert!(!found.is_empty(), "crawler should discover >=1 novel composition");
        // Each discovery is a real, distinct behavior.
        for d in &found {
            assert_eq!(d.signature.len(), GRID.len());
            assert!(d.name.starts_with("compose_"), "{}", d.name);
        }
        // Persisted to the log.
        let logged = std::fs::read_to_string(&log).unwrap();
        assert_eq!(logged.lines().count(), found.len(), "log has one line per discovery");

        // RESUMABLE: a second pass with the same log does not re-log the same
        // behaviors (it only appends genuinely new ones).
        let before = std::fs::read_to_string(&log).unwrap();
        let again = crawl_once(&bridge, &log, &work, 3);
        let after = std::fs::read_to_string(&log).unwrap();
        let prior_sigs: std::collections::HashSet<_> =
            found.iter().map(|d| d.signature.clone()).collect();
        for d in &again {
            assert!(!prior_sigs.contains(&d.signature), "must not re-discover {}", d.name);
        }
        assert_eq!(
            after.lines().count(),
            before.lines().count() + again.len(),
            "log grows only by the new discoveries"
        );

        let _ = std::fs::remove_file(&log);
        let _ = std::fs::remove_dir_all(&work);
    }
}
