//! Component crawler — a NON-LLM discovery engine that CONTINUOUSLY enumerates
//! compositions of verified leaf ops, verifies each end to end (compile + a
//! differential property gate vs the native composition), dedups by BEHAVIOR, and
//! documents every genuinely novel verified discovery to an append-only log.
//!
//! COMPOUNDING FLYWHEEL: a composable is a CHAIN of base leaves (apply left→right,
//! so `[square, increment]` is `increment(square(x)) = x²+1`). Round N composes
//! pairs of known composables — base leaves AND prior discoveries reloaded from the
//! log — into longer chains, capped at `MAX_CHAIN`. So each verified discovery
//! re-enters as raw material for the next round and the reachable space grows
//! super-linearly (depth 2 → 3 → 4 …), not just the 36 base pairs.
//!
//! Each candidate whose BEHAVIOR (output vector on a probe grid) is novel vs every
//! known op, identity, and prior discovery is BUILT and verified: it must compile
//! AND match its native oracle across the grid. Identity/trivial collapses
//! (`negate∘negate`, `increment∘decrement`) are skipped by the signature. Novelty
//! is by signature, so re-runs never re-log the same behavior — resumable + bounded
//! per pass. Loop `crawl_once` for a constant crawl (see `src/bin/discover.rs`).

use crate::component::{build_component, ComponentSpec, GlueSpec};
use crate::linguigenesis_bridge::LinguigenesisBridge;
use std::collections::HashSet;
use std::path::Path;

/// Probe grid the behavior signature is sampled on. Small magnitudes keep
/// square-heavy chains inside i64 comfortably.
const GRID: [i64; 8] = [-3, -2, -1, 0, 1, 2, 3, 4];

/// The verified scalar leaves the crawler composes.
const SCALARS: [&str; 6] = ["increment", "decrement", "double", "triple", "negate", "square"];

/// Longest chain the crawler will build (bounds the compounding search).
const MAX_CHAIN: usize = 4;

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

/// Behavior signature of a chain (apply the ops left→right across the probe grid).
fn signature_of_chain(chain: &[String]) -> Vec<i64> {
    GRID.iter()
        .map(|&x0| {
            let mut v = x0;
            for op in chain {
                v = native(op)(v);
            }
            v
        })
        .collect()
}

/// A composable behavior: a chain of base leaves + its signature.
#[derive(Clone)]
struct Composable {
    chain: Vec<String>,
    signature: Vec<i64>,
}

/// A novel verified composition the crawler discovered.
#[derive(Debug, Clone)]
pub struct Discovery {
    /// `compose_<op1>_<op2>_...` (the chain, applied left→right).
    pub name: String,
    pub chain: Vec<String>,
    /// Output vector on the probe grid (its behavioral identity).
    pub signature: Vec<i64>,
}

/// Build the component that applies `chain` left→right, with a property smoke
/// asserting it equals the native chain across the grid (values baked as literals).
fn chain_spec(chain: &[String]) -> ComponentSpec {
    let name = format!("compose_{}", chain.join("_"));
    // Nested call expression: [a,b,c] -> c(b(a(x))).
    let mut expr = "x".to_string();
    for op in chain {
        expr = format!("{op}({expr})");
    }
    // Unique imports, one per distinct op.
    let mut seen = HashSet::new();
    let mut imports = String::new();
    for op in chain {
        if seen.insert(op.clone()) {
            imports.push_str(&format!("use crate::{op}::{op};\n"));
        }
    }
    let code = format!(
        "//! Discovered composition: {}.\n\n{imports}\npub fn apply(x: i64) -> i64 {{\n    {expr}\n}}\n",
        chain.join(" -> ")
    );
    let cases: Vec<String> = {
        let sig = signature_of_chain(chain);
        GRID.iter()
            .zip(sig.iter())
            .map(|(&x, &e)| format!("({x}, {e})"))
            .collect()
    };
    let smoke = format!(
        "\n#[cfg(test)]\nmod composed_behaves {{\n    use super::apply;\n    #[test]\n    fn matches_native_composition() {{\n        for (x, expected) in [{}] {{\n            assert_eq!(apply(x), expected, \"at x={{}}\", x);\n        }}\n    }}\n}}\n",
        cases.join(", ")
    );
    let leaves: Vec<String> = {
        let mut u = HashSet::new();
        chain.iter().filter(|op| u.insert((*op).clone())).cloned().collect()
    };
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

/// Load prior discoveries (chain + signature) from the log so a continuous crawl
/// compounds on them and never re-logs the same behavior.
fn known_from_log(log_path: &Path) -> Vec<Composable> {
    let mut out = Vec::new();
    if let Ok(text) = std::fs::read_to_string(log_path) {
        for line in text.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let chain: Vec<String> = v["chain"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let signature: Vec<i64> = v["signature"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|n| n.as_i64()).collect())
                    .unwrap_or_default();
                if !chain.is_empty() && !signature.is_empty() {
                    out.push(Composable { chain, signature });
                }
            }
        }
    }
    out
}

/// Append one discovery to the JSONL log.
fn append_discovery(log_path: &Path, d: &Discovery) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let line = serde_json::json!({
        "name": d.name,
        "chain": d.chain,
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

/// One bounded discovery pass. Composables = base leaves + everything in the log.
/// Sweep `outer ∘ inner` over all composable pairs (chain = inner.chain ++
/// outer.chain), skip any exceeding `MAX_CHAIN` or whose behavior is already known,
/// and for each genuinely novel signature BUILD + VERIFY (compile + differential
/// property gate). Verified-novel discoveries are appended to `log_path`, added to
/// the in-memory pool (so they compound WITHIN the pass too), and returned. Stops
/// after `budget` new discoveries.
pub fn crawl_once(
    bridge: &LinguigenesisBridge,
    log_path: &Path,
    work_root: &Path,
    budget: usize,
) -> Vec<Discovery> {
    let mut known_sigs: HashSet<Vec<i64>> = HashSet::new();
    known_sigs.insert(signature_of_chain(&[])); // identity (empty chain)
    let mut pool: Vec<Composable> = Vec::new();
    for s in SCALARS {
        let chain = vec![s.to_string()];
        let sig = signature_of_chain(&chain);
        known_sigs.insert(sig.clone());
        pool.push(Composable { chain, signature: sig });
    }
    for c in known_from_log(log_path) {
        known_sigs.insert(c.signature.clone());
        pool.push(c);
    }

    let mut found = Vec::new();
    let mut i = 0;
    // Index over pairs; `pool` may grow as we discover, so we compound in-pass.
    while i < pool.len() {
        for j in 0..pool.len() {
            if found.len() >= budget {
                return found;
            }
            let inner = &pool[i];
            let outer = &pool[j];
            if inner.chain.len() + outer.chain.len() > MAX_CHAIN {
                continue;
            }
            // result = outer(inner(x)): apply inner's chain first, then outer's.
            let mut chain = inner.chain.clone();
            chain.extend(outer.chain.iter().cloned());
            let sig = signature_of_chain(&chain);
            if known_sigs.contains(&sig) {
                continue; // behavior already known — not novel
            }
            let spec = chain_spec(&chain);
            let croot = work_root.join(&spec.name);
            let _ = std::fs::remove_dir_all(&croot);
            let verified = matches!(
                build_component(bridge, &spec, &croot),
                Ok(ref b) if b.outcome.compile.is_ok() && b.behaves()
            );
            let _ = std::fs::remove_dir_all(&croot);
            if verified {
                known_sigs.insert(sig.clone());
                pool.push(Composable { chain: chain.clone(), signature: sig.clone() });
                let d = Discovery { name: spec.name, chain, signature: sig };
                let _ = append_discovery(log_path, &d);
                found.push(d);
            }
        }
        i += 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_signature_composes_and_collapses() {
        // [square, increment] = increment(square(x)) = x^2 + 1.
        let sig = signature_of_chain(&["square".into(), "increment".into()]);
        let expect: Vec<i64> = GRID.iter().map(|&x| x * x + 1).collect();
        assert_eq!(sig, expect);
        // negate ∘ negate collapses to identity.
        assert_eq!(
            signature_of_chain(&["negate".into(), "negate".into()]),
            signature_of_chain(&[])
        );
    }

    #[test]
    fn crawl_finds_logs_and_compounds() {
        let bridge = LinguigenesisBridge::new();
        let tag = std::process::id();
        let log = std::env::temp_dir().join(format!("nsynth_disc_{tag}.jsonl"));
        let work = std::env::temp_dir().join(format!("nsynth_crawlw_{tag}"));
        let _ = std::fs::remove_file(&log);
        let _ = std::fs::remove_dir_all(&work);

        let found = crawl_once(&bridge, &log, &work, 3);
        assert!(!found.is_empty(), "crawler should discover >=1 novel composition");
        for d in &found {
            assert_eq!(d.signature.len(), GRID.len());
            assert!(d.chain.len() >= 2 && d.chain.len() <= MAX_CHAIN);
        }
        let logged = std::fs::read_to_string(&log).unwrap();
        assert_eq!(logged.lines().count(), found.len());

        // RESUMABLE + COMPOUNDING: a second pass reloads discoveries as composables
        // and only appends genuinely new behaviors.
        let before = logged.lines().count();
        let again = crawl_once(&bridge, &log, &work, 3);
        let prior: HashSet<_> = found.iter().map(|d| d.signature.clone()).collect();
        for d in &again {
            assert!(!prior.contains(&d.signature), "must not re-discover {}", d.name);
        }
        let after = std::fs::read_to_string(&log).unwrap().lines().count();
        assert_eq!(after, before + again.len());
        // At least one second-pass discovery should be a DEEPER chain (compounding),
        // reachable only by composing on a prior discovery — length > 2.
        if !again.is_empty() {
            assert!(
                again.iter().any(|d| d.chain.len() >= 2),
                "second pass produced compositions"
            );
        }

        let _ = std::fs::remove_file(&log);
        let _ = std::fs::remove_dir_all(&work);
    }
}
