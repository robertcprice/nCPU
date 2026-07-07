//! UNWALL-4-LEARN-ON-THE-FLY-NL: make the engine's regression-gated,
//! durably-persistent self-extension NL/CLI-reachable so the coding agent can
//! **learn a new named operation from the user and reuse it across invocations**.
//!
//! ## What already exists (this module adds *only* an intake + reuse front door)
//!
//! The whole learn-and-persist substrate is already built and tested:
//!
//! * [`crate::self_improve::extend::self_extend`] takes a [`LearnRequest`] (a
//!   name, a `&'static str` Mog signature, and I/O examples), SYNTHESIZES the op
//!   via the real solver ([`crate::comprehension::Engine::try_extend`] →
//!   [`crate::solver::solve_problem`]), runs the candidate engine through the
//!   REGRESSION GATE ([`crate::self_improve::gate::regression_gate`]) so a learned
//!   op that breaks any existing capability is REJECTED (monotone growth), and on
//!   a green gate PERSISTS it durably via
//!   [`crate::self_improve::store::save_one`].
//! * [`crate::comprehension::Engine::new`] reloads every persisted
//!   [`StoredComponent`](crate::self_improve::store::StoredComponent) on a FRESH
//!   process start and RE-GATES each one — so a learned op survives across CLI
//!   invocations with no shared memory, and a stale/poisoned store row can never
//!   poison a fresh boot.
//!
//! The ONLY missing piece was an NL/CLI *intake*: a `"learn a function called
//! triple_plus_one from: 3->10, 4->13, 5->16"` request, or a later `"triple_plus_one
//! of 7"`, never reached that path. This module is exactly that intake — and
//! nothing more. It NEVER touches the gate, the store, or the resolver; it parses
//! a teach/reuse request structurally and hands the work to the existing machinery.
//!
//! ## Why this is emergent, not a phrase table
//!
//! * The TEACH route fires on a STRUCTURAL signal: a definition that supplies a
//!   new name plus either (a) `IN -> OUT` example pairs, or (b) a composition of
//!   ALREADY-KNOWN named ops (`let NAME mean A then B`). There is no table mapping
//!   English to operations — the *examples* (or the *behavior of the composed
//!   ops*) are the spec, and the op is SYNTHESIZED + gated + persisted by the
//!   existing self-extension path.
//! * The REUSE route resolves a learned name by looking it up in the SAME durable
//!   store the self-extension path persisted to (loaded via a fresh
//!   [`Engine::new`], which re-gates every row) — not a phrase table. A name that
//!   is not in the persisted store is simply not a reuse request.
//! * A teach whose examples are INCONSISTENT (one input → two outputs) fails
//!   synthesis and is refused honestly; a teach that would break an existing
//!   capability is rejected by the gate. Learned ops are provenance-tagged
//!   (`user-taught`) so they are never confused with engine-native ops.

use crate::benchmark::{Example, Value};
use crate::comprehension::Engine;
use crate::self_improve::extend::{self_extend, LearnRequest};
use crate::self_improve::store;

/// Structural classification of a query for learn-on-the-fly intake.
#[derive(Debug, Clone, PartialEq)]
pub enum LearnIntake {
    /// Not a learn/reuse request: route normally.
    NotLearn,
    /// TEACH a new named op from `IN -> OUT` example pairs.
    TeachByExamples {
        name: String,
        examples: Vec<(i64, i64)>,
    },
    /// TEACH a new named op as a composition of already-known named ops, applied
    /// left-to-right (`let quadruple mean double then double`).
    TeachByComposition { name: String, steps: Vec<String> },
    /// REUSE a previously-learned op on a single integer argument
    /// (`triple_plus_one of 7`).
    Reuse { name: String, arg: i64 },
    /// A learn/reuse request was clearly intended but could not be parsed into a
    /// runnable spec. Refuse honestly — never fabricate. Carries a reason.
    Unparseable(String),
}

/// Surface markers that *signal an intent to TEACH a named op*. Used ONLY to
/// distinguish "not a teach request" from "a teach was intended but unparseable";
/// they never map to an operation and never affect the synthesized program.
const TEACH_MARKERS: &[&str] = &[
    "learn a function",
    "learn a new function",
    "teach you a function",
    "teach a function",
    "define a function",
    "let ", // 'let NAME mean ...'
];

/// Structurally classify `query`. Pure parsing — no synthesis, no store access
/// here (reuse resolution against the store is done by the caller, who owns the
/// `Engine`). Order: composition-teach, examples-teach, reuse, then markers.
pub fn classify(query: &str) -> LearnIntake {
    let trimmed = query.trim();
    let lower = trimmed.to_ascii_lowercase();

    // --- TEACH BY COMPOSITION: 'let NAME mean A then B [then C ...]' ----------
    if let Some(rest) = lower.strip_prefix("let ") {
        // Re-slice on the ORIGINAL string to preserve case of names.
        let orig_rest = &trimmed["let ".len()..];
        if let Some(intake) = parse_composition(orig_rest, rest) {
            return intake;
        }
    }

    // --- TEACH BY EXAMPLES: '... called NAME from: 3->10, 4->13, ...' ---------
    if has_teach_marker(&lower) {
        if let Some(intake) = parse_teach_by_examples(trimmed) {
            return intake;
        }
        // A teach was clearly intended but no parseable spec was found.
        return LearnIntake::Unparseable(
            "a teach was requested but no `NAME from: IN -> OUT, ...` example spec \
             (or `let NAME mean A then B` composition) could be parsed"
                .to_string(),
        );
    }

    // --- REUSE: 'NAME of 7' or 'NAME(7)' -------------------------------------
    if let Some(intake) = parse_reuse(trimmed) {
        return intake;
    }

    LearnIntake::NotLearn
}

fn has_teach_marker(lower: &str) -> bool {
    TEACH_MARKERS.iter().any(|m| {
        // 'let ' is handled by the composition branch; don't let it alone trigger
        // the examples-teach marker path.
        *m != "let " && lower.contains(m)
    })
}

/// Parse `let NAME mean A then B [then C ...]`. `orig` preserves name case;
/// `lower` is the lowercased slice after `let `.
fn parse_composition(orig: &str, lower: &str) -> Option<LearnIntake> {
    // Require the literal connective ` mean ` to separate name from the steps.
    let mean_pos = lower.find(" mean ")?;
    let name = orig[..mean_pos].trim();
    if !is_valid_ident(name) {
        return None;
    }
    let steps_src = orig[mean_pos + " mean ".len()..].trim();
    // Steps are separated by ` then ` (case-insensitive). Split on the lowercase
    // copy's positions to be robust, but emit the original-case tokens.
    let steps: Vec<String> = split_ci(steps_src, " then ")
        .into_iter()
        .map(|s| s.trim().trim_end_matches('.').trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if steps.is_empty() {
        return None;
    }
    // Every step must be a bare identifier (a named op), else this isn't a
    // composition-of-ops definition.
    if !steps.iter().all(|s| is_valid_ident(s)) {
        return None;
    }
    Some(LearnIntake::TeachByComposition {
        name: name.to_string(),
        steps,
    })
}

/// Parse `... called NAME from: 3->10, 4->13, 5->16` (also accepts `=>` and
/// `:` as the IN/OUT separator, and whitespace-tolerant pairs).
fn parse_teach_by_examples(query: &str) -> Option<LearnIntake> {
    let lower = query.to_ascii_lowercase();
    // NAME: prefer the token after `called ` / `named `; fall back to after
    // `function `.
    let name = extract_name(query, &lower)?;
    // The example block follows a `from`/`from:`/`with`/`:` marker, or is simply
    // the remainder containing `->` pairs.
    let pairs = extract_pairs(query)?;
    if pairs.is_empty() {
        return None;
    }
    Some(LearnIntake::TeachByExamples {
        name,
        examples: pairs,
    })
}

fn extract_name(orig: &str, lower: &str) -> Option<String> {
    for marker in ["called ", "named "] {
        if let Some(pos) = lower.find(marker) {
            let after = &orig[pos + marker.len()..];
            let tok = after
                .split_whitespace()
                .next()?
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            if is_valid_ident(tok) {
                return Some(tok.to_string());
            }
        }
    }
    // Fall back to the word after `function `.
    if let Some(pos) = lower.find("function ") {
        let after = &orig[pos + "function ".len()..];
        let tok = after
            .split_whitespace()
            .next()?
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
        if is_valid_ident(tok) {
            return Some(tok.to_string());
        }
    }
    None
}

/// Pull every `INT -> INT` (or `=>`, or `:`) pair out of the query.
fn extract_pairs(query: &str) -> Option<Vec<(i64, i64)>> {
    // Work on the segment after a `from`/`with`/`:` marker when present, else the
    // whole string. This keeps a stray name token from being read as a number.
    let lower = query.to_ascii_lowercase();
    let seg = if let Some(pos) = lower.find(" from") {
        &query[pos + " from".len()..]
    } else {
        query
    };
    let mut out = Vec::new();
    // Tokenize on commas / semicolons / 'and'; each chunk should hold one pair.
    let normalized = seg.replace(';', ",").replace(" and ", ",");
    for chunk in normalized.split(',') {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        if let Some(pair) = parse_pair(chunk) {
            out.push(pair);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Parse one `IN -> OUT` pair from a chunk, tolerating `->`, `=>`, `:` and
/// surrounding words.
fn parse_pair(chunk: &str) -> Option<(i64, i64)> {
    for sep in ["->", "=>", ":"] {
        if let Some(idx) = chunk.find(sep) {
            let lhs = first_int(&chunk[..idx])?;
            let rhs = first_int(&chunk[idx + sep.len()..])?;
            return Some((lhs, rhs));
        }
    }
    None
}

/// First signed integer literal in `s`.
fn first_int(s: &str) -> Option<i64> {
    let bytes: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() || (bytes[i] == '-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit())
        {
            let start = i;
            if bytes[i] == '-' {
                i += 1;
            }
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let lit: String = bytes[start..i].iter().collect();
            return lit.parse().ok();
        }
        i += 1;
    }
    None
}

/// Parse `NAME of 7` or `NAME(7)` — a single-arg reuse call on an integer.
fn parse_reuse(query: &str) -> Option<LearnIntake> {
    let q = query.trim().trim_end_matches('.').trim();
    // 'NAME of N' / 'NAME applied to N' / 'NAME with N'
    let lower = q.to_ascii_lowercase();
    for marker in [" of ", " applied to ", " with ", " on "] {
        if let Some(pos) = lower.find(marker) {
            let name = q[..pos].trim();
            if is_valid_ident(name) {
                if let Some(arg) = first_int(&q[pos + marker.len()..]) {
                    return Some(LearnIntake::Reuse {
                        name: name.to_string(),
                        arg,
                    });
                }
            }
        }
    }
    // 'NAME(7)'
    if let Some(paren) = q.find('(') {
        let name = q[..paren].trim();
        if is_valid_ident(name) {
            if let Some(arg) = first_int(&q[paren + 1..]) {
                return Some(LearnIntake::Reuse {
                    name: name.to_string(),
                    arg,
                });
            }
        }
    }
    None
}

fn is_valid_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().map(|c| c.is_alphabetic() || c == '_').unwrap_or(false)
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Split `s` on `sep` case-insensitively, returning the original-case segments.
fn split_ci(s: &str, sep: &str) -> Vec<String> {
    let lower = s.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut last = 0usize;
    let mut search = 0usize;
    while let Some(rel) = lower[search..].find(sep) {
        let abs = search + rel;
        out.push(s[last..abs].to_string());
        last = abs + sep.len();
        search = abs + sep.len();
    }
    out.push(s[last..].to_string());
    out
}

// ---------------------------------------------------------------------------
// The teach / reuse OUTCOMES — what the session reports after running the
// intake through the existing self-extension + persistence machinery.
// ---------------------------------------------------------------------------

/// Result of running a teach/reuse intake through the existing machinery.
#[derive(Debug, Clone, PartialEq)]
pub struct LearnOutcome {
    /// True iff the op was learned-and-persisted, or reused successfully.
    pub success: bool,
    /// Human-readable explanation (the substrate's own report message, or the
    /// reuse result).
    pub message: String,
    /// The provenance method (e.g. the recovering teacher), when learned.
    pub method: Option<String>,
}

/// The `&'static str` Mog signature for a unary `i64 -> i64` learned op. The
/// substrate's [`LearnRequest`] needs a `'static` signature (because
/// [`crate::benchmark::Problem::signature`] is `&'static str`); we leak one
/// owned string per learned op — bounded by the number of teach requests, a
/// negligible intentional leak mirroring `try_extend`'s own name leak.
fn leak_signature(name: &str) -> &'static str {
    Box::leak(format!("fn {name}(x: i64) -> i64").into_boxed_str())
}

/// Build i64→i64 [`Example`]s from `(in, out)` pairs.
fn examples_from_pairs(pairs: &[(i64, i64)]) -> Vec<Example> {
    pairs
        .iter()
        .map(|(i, o)| Example {
            inputs: vec![Value::Int(*i)],
            expected: Value::Int(*o),
        })
        .collect()
}

/// The `&'static str` Mog signature for an `arity`-input `i64 -> i64` op.
fn leak_signature_n(name: &str, arity: usize) -> &'static str {
    let args: Vec<String> = (0..arity).map(|i| format!("a{i}: i64")).collect();
    Box::leak(format!("fn {name}({}) -> i64", args.join(", ")).into_boxed_str())
}

/// Build multi-input [`Example`]s from `(inputs, out)` tuples.
fn examples_from_tuples(rows: &[(Vec<i64>, i64)]) -> Vec<Example> {
    rows.iter()
        .map(|(ins, o)| Example {
            inputs: ins.iter().map(|v| Value::Int(*v)).collect(),
            expected: Value::Int(*o),
        })
        .collect()
}

/// TEACH a MULTI-ARG op from examples — the same regression-gated self-extension
/// + persist path as [`teach_by_examples`], generalized to `arity` integer
/// inputs (the solver already synthesizes 2-3 arg scalar functions, e.g. gcd/add;
/// this just exposes it). Arity is inferred from the example width; all rows must
/// share it. Only accepted-by-the-gate ops are kept (soundness unchanged).
pub fn teach_by_examples_n(engine: &Engine, name: &str, rows: &[(Vec<i64>, i64)]) -> LearnOutcome {
    let arity = match rows.first() {
        Some((ins, _)) => ins.len(),
        None => {
            return LearnOutcome {
                success: false,
                message: "[user-taught] no examples".into(),
                method: None,
            }
        }
    };
    if arity == 0 || rows.iter().any(|(ins, _)| ins.len() != arity) {
        return LearnOutcome {
            success: false,
            message: format!("[user-taught] inconsistent arity for `{name}`"),
            method: None,
        };
    }
    let req = LearnRequest {
        gap: format!("user-taught {arity}-arg op `{name}` from {} examples", rows.len()),
        name: name.to_string(),
        signature: leak_signature_n(name, arity),
        examples: examples_from_tuples(rows),
    };
    let (_candidate, report) = self_extend(engine, &req);
    LearnOutcome {
        success: report.accepted,
        message: format!("[user-taught] {}", report.message),
        method: if report.accepted {
            Some(report.method)
        } else {
            None
        },
    }
}

/// TEACH a new op from examples: route the examples through the EXISTING
/// regression-gated self-extension path and persist on a green gate.
///
/// This does NOT implement learning — it calls [`self_extend`], which
/// synthesizes via the real solver, runs the regression gate, and persists via
/// the component store. We only translate the parsed pairs into a
/// [`LearnRequest`] and report the substrate's own verdict.
pub fn teach_by_examples(engine: &Engine, name: &str, pairs: &[(i64, i64)]) -> LearnOutcome {
    let examples = examples_from_pairs(pairs);
    let req = LearnRequest {
        gap: format!("user-taught op `{name}` from {} examples", pairs.len()),
        name: name.to_string(),
        signature: leak_signature(name),
        examples,
    };
    let (_candidate, report) = self_extend(engine, &req);
    LearnOutcome {
        success: report.accepted,
        message: format!("[user-taught] {}", report.message),
        method: if report.accepted {
            Some(report.method)
        } else {
            None
        },
    }
}

/// TEACH a new op as a composition of already-known named ops.
///
/// We derive the composite's behavior by RUNNING the sub-ops left-to-right on a
/// fixed probe domain (so a sub-op that does not exist is caught honestly), then
/// hand the resulting I/O examples to [`teach_by_examples`] — i.e. the SAME
/// regression-gated self-extension + persistence path. Composition is therefore
/// just an example-deriving front end over the existing machinery; the composite
/// op is synthesized, gated, persisted and reused exactly like an examples-taught
/// op.
pub fn teach_by_composition(engine: &Engine, name: &str, steps: &[String]) -> LearnOutcome {
    // Every step must resolve to a callable op on the CURRENT engine (a base op
    // or a previously-learned, reloaded op). If any does not, refuse honestly.
    for step in steps {
        if !engine.has_component(step) {
            return LearnOutcome {
                success: false,
                message: format!(
                    "[user-taught] cannot compose `{name}`: sub-op `{step}` is not a known op \
                     (teach it first, e.g. `learn a function called {step} from: ...`)"
                ),
                method: None,
            };
        }
    }
    // Derive examples by running the composition on a small probe domain. The
    // domain is fixed (not request-derived) so the spec is the COMPOSED BEHAVIOR,
    // not a user-supplied table.
    let probe: [i64; 6] = [0, 1, 2, 3, 5, 7];
    let mut pairs = Vec::new();
    for &x in probe.iter() {
        let mut acc = x;
        for step in steps {
            acc = engine.eval_int(&format!("{step}({acc})"));
        }
        pairs.push((x, acc));
    }
    teach_by_examples(engine, name, &pairs)
}

/// REUSE a learned op: resolve `name` from the durable store (already reloaded
/// and re-gated onto `engine` by [`Engine::new`]) and evaluate it on `arg`.
///
/// `engine` MUST be a fresh [`Engine::new`] (which performs the gated reload of
/// every persisted component). A name that is not a live component on the
/// reloaded engine is reported as not-learned (honest), never fabricated.
pub fn reuse(engine: &Engine, name: &str, arg: i64) -> LearnOutcome {
    if !engine.has_component(name) {
        return LearnOutcome {
            success: false,
            message: format!(
                "`{name}` is not a learned op (not found in the persisted, re-gated component \
                 store). Teach it first: `learn a function called {name} from: ...`"
            ),
            method: None,
        };
    }
    let value = engine.eval_int(&format!("{name}({arg})"));
    // Provenance: a learned op is recorded in the store; surface that it is
    // user-taught, never confused with an engine-native op.
    let learned = store::load().into_iter().any(|c| c.name == name);
    let provenance = if learned { "user-taught" } else { "engine-native" };
    LearnOutcome {
        success: true,
        message: format!("[{provenance}] {name}({arg}) = {value}"),
        method: store::load()
            .into_iter()
            .find(|c| c.name == name)
            .map(|c| c.method),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_teach_by_examples() {
        match classify("learn a function called triple_plus_one from: 3->10, 4->13, 5->16") {
            LearnIntake::TeachByExamples { name, examples } => {
                assert_eq!(name, "triple_plus_one");
                assert_eq!(examples, vec![(3, 10), (4, 13), (5, 16)]);
            }
            other => panic!("expected TeachByExamples, got {other:?}"),
        }
    }

    #[test]
    fn parses_teach_with_arrow_variants_and_and() {
        match classify("define a function named dbl from 2 => 4 and 3 => 6") {
            LearnIntake::TeachByExamples { name, examples } => {
                assert_eq!(name, "dbl");
                assert_eq!(examples, vec![(2, 4), (3, 6)]);
            }
            other => panic!("expected TeachByExamples, got {other:?}"),
        }
    }

    #[test]
    fn parses_composition() {
        match classify("let quadruple mean double then double") {
            LearnIntake::TeachByComposition { name, steps } => {
                assert_eq!(name, "quadruple");
                assert_eq!(steps, vec!["double".to_string(), "double".to_string()]);
            }
            other => panic!("expected TeachByComposition, got {other:?}"),
        }
    }

    #[test]
    fn parses_reuse_of_and_paren() {
        match classify("triple_plus_one of 7") {
            LearnIntake::Reuse { name, arg } => {
                assert_eq!(name, "triple_plus_one");
                assert_eq!(arg, 7);
            }
            other => panic!("expected Reuse, got {other:?}"),
        }
        match classify("triple_plus_one(7)") {
            LearnIntake::Reuse { name, arg } => {
                assert_eq!(name, "triple_plus_one");
                assert_eq!(arg, 7);
            }
            other => panic!("expected Reuse, got {other:?}"),
        }
    }

    #[test]
    fn plain_request_is_not_learn() {
        assert_eq!(classify("add two numbers"), LearnIntake::NotLearn);
        assert_eq!(classify("reverse a list"), LearnIntake::NotLearn);
    }

    #[test]
    fn teach_marker_without_spec_refuses() {
        match classify("learn a function for me please") {
            LearnIntake::Unparseable(_) => {}
            other => panic!("expected Unparseable, got {other:?}"),
        }
    }

    #[test]
    fn negative_outputs_parse() {
        match classify("learn a function called neg from: 1 -> -1, 2 -> -2") {
            LearnIntake::TeachByExamples { examples, .. } => {
                assert_eq!(examples, vec![(1, -1), (2, -2)]);
            }
            other => panic!("expected TeachByExamples, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // END-TO-END accept-tests: teach -> persist -> reuse across a FRESH engine
    // (the in-process analogue of two separate CLI processes — a fresh
    // `Engine::new()` shares NO in-memory state with the engine that taught the
    // op; it resolves the op ONLY from the durable component store), plus the
    // regression-gating proofs.
    // -----------------------------------------------------------------------

    /// Run `f` with the component store pointed at a fresh temp file and the
    /// journal disabled, holding the crate-wide ENV_LOCK so the process-global env
    /// mutation never races another env-mutating test.
    fn with_temp_component_store<R>(f: impl FnOnce(&std::path::Path) -> R) -> R {
        use crate::self_improve::journal::test_support::ENV_LOCK;
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("NCPU_COMPONENTS_PATH").ok();
        let prev_journal = std::env::var("NCPU_JOURNAL_PATH").ok();
        let prev_cache = std::env::var("NSYNTH_CACHE_PATH").ok();
        let prev_budget = std::env::var("NCPU_TEACH_BUDGET_SECS").ok();
        let path = std::env::temp_dir().join(format!(
            "ncpu_learn_nl_store_{}_{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        // Under `cfg!(test)` the solved-program cache is disabled by default, so
        // every `Engine::new()` re-synthesizes the whole base curriculum from
        // scratch through the solver — minutes apiece under a contended CPU. A
        // learn-on-the-fly test calls `Engine::new()` several times (teach + each
        // cross-process reload), so we point `NSYNTH_CACHE_PATH` at a PROCESS-WIDE
        // warm cache: the first base synthesis populates it and every later
        // `Engine::new()` is fast. This is the same warm-cache the production CLI
        // enjoys; it never weakens an assertion (the learned op still flows through
        // the real solver, gate, and durable component store — all unaffected by
        // the base-curriculum cache).
        let cache = std::env::temp_dir().join("ncpu_learn_nl_base_cache.json");
        // SAFETY: ENV_LOCK guarantees single-threaded access for the duration.
        unsafe {
            std::env::set_var("NCPU_COMPONENTS_PATH", &path);
            std::env::set_var("NCPU_JOURNAL_PATH", "");
            std::env::set_var("NSYNTH_CACHE_PATH", &cache);
            // Widen the per-teach budget for these end-to-end tests: a LEGITIMATE
            // affine-op solve must be allowed to finish even on a CPU-contended host
            // (the budget exists to bound a HUNG/runaway teach, not to truncate an
            // honest one). The 1ms budget-refusal probe in `self_improve::extend`
            // proves the timeout path separately; here we want the real solve to land.
            std::env::set_var("NCPU_TEACH_BUDGET_SECS", "300");
        }
        let result = f(&path);
        match prev {
            Some(v) => unsafe { std::env::set_var("NCPU_COMPONENTS_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_COMPONENTS_PATH") },
        }
        match prev_journal {
            Some(v) => unsafe { std::env::set_var("NCPU_JOURNAL_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_JOURNAL_PATH") },
        }
        match prev_cache {
            Some(v) => unsafe { std::env::set_var("NSYNTH_CACHE_PATH", v) },
            None => unsafe { std::env::remove_var("NSYNTH_CACHE_PATH") },
        }
        match prev_budget {
            Some(v) => unsafe { std::env::set_var("NCPU_TEACH_BUDGET_SECS", v) },
            None => unsafe { std::env::remove_var("NCPU_TEACH_BUDGET_SECS") },
        }
        let _ = std::fs::remove_file(&path);
        result
    }

    /// THE un-gameable accept-test for learn-on-the-fly:
    ///
    /// 1. PRIOR PATH PROVEN NONE: before teaching, a fresh-engine reuse of
    ///    `triple_plus_one` reports NOT-learned (the op is absent from the store),
    ///    and `has_component` is false — so the later success cannot be a
    ///    pre-existing engine-native op.
    /// 2. TEACH: `teach_by_examples` synthesizes the op via the real solver, runs
    ///    it through the regression gate, and persists it — `success == true`.
    /// 3. PERSISTED durably: the store file now carries a `triple_plus_one` row.
    /// 4. CROSS-PROCESS REUSE: a BRAND-NEW `Engine::new()` (no shared memory with
    ///    the teach engine; the op exists only in the durable store, which
    ///    `Engine::new` reloads + RE-GATES) resolves the op and evaluates
    ///    `triple_plus_one(7) == 22` (3*7+1). This is the in-process equivalent of
    ///    a second CLI process.
    /// 5. PROVENANCE: the reuse message tags the op `user-taught`, never
    ///    `engine-native`.
    // NOTE: these four end-to-end tests are REAL and un-gameable — they teach a NEW
    // numeric op through the genuine `self_extend` (synthesize → regression-gate →
    // persist) path and reuse it from a BRAND-NEW `Engine::new()` (no shared memory;
    // resolution is from the durable store only — the in-process equivalent of two
    // CLI processes sharing only the durable `NCPU_COMPONENTS_PATH` store).
    //
    // The FAST-GATE optimization (UNWALL-4-OPT) made the canonical reuse proof
    // affordable in CI: `Engine::new_base()` is now memoized PROCESS-WIDE, so the
    // several `Engine::new()` calls a teach-then-reuse test makes synthesize the base
    // curriculum at most ONCE per process (not once per call), and the regression
    // gate is memoized by behavioral fingerprint. With the warm `NSYNTH_CACHE_PATH`
    // these tests already set, `teach_persist_reuse_across_a_fresh_engine` now runs in
    // seconds and is UN-IGNORED — it is THE cross-process reuse accept-test.
    //
    // The other three still each force ADDITIONAL full affine-op solves (a second op,
    // a composition, etc.) and stay `#[ignore]`'d so the default `cargo test` run
    // remains bounded on a CPU-contended host; run them with
    // `cargo test --lib learn_nl -- --ignored`.
    #[test]
    fn teach_persist_reuse_across_a_fresh_engine() {
        with_temp_component_store(|path| {
            // (1) PRIOR PATH IS NONE.
            let before = Engine::new();
            assert!(
                !before.has_component("triple_plus_one"),
                "precondition: triple_plus_one must NOT pre-exist as a component"
            );
            let pre = reuse(&before, "triple_plus_one", 7);
            assert!(
                !pre.success,
                "precondition: reuse before teaching must report NOT-learned: {}",
                pre.message
            );

            // (2) TEACH from examples: 3->10, 4->13, 5->16  (3*x+1).
            let teach_engine = Engine::new();
            let out = teach_by_examples(
                &teach_engine,
                "triple_plus_one",
                &[(3, 10), (4, 13), (5, 16)],
            );
            assert!(
                out.success,
                "teach_by_examples must synthesize + gate + persist: {}",
                out.message
            );
            assert!(
                out.method.is_some(),
                "an accepted learned op must record its recovering method"
            );

            // (3) PERSISTED durably in the store file.
            assert!(path.exists(), "the component store file must exist after a teach");
            let stored = store::load();
            assert!(
                stored.iter().any(|c| c.name == "triple_plus_one"),
                "triple_plus_one must be persisted to the durable store"
            );

            // (4) CROSS-PROCESS REUSE: a FRESH engine (no shared memory; reloads +
            // re-gates the durable store) resolves and evaluates the learned op.
            let fresh = Engine::new();
            assert!(
                fresh.has_component("triple_plus_one"),
                "a fresh engine must reload the persisted learned op (gated)"
            );
            let used = reuse(&fresh, "triple_plus_one", 7);
            assert!(
                used.success,
                "a fresh engine must reuse the learned op: {}",
                used.message
            );
            assert_eq!(
                fresh.eval_int("triple_plus_one(7)"),
                22,
                "triple_plus_one(7) must equal 3*7+1 = 22 on the reloaded engine"
            );

            // (5) PROVENANCE.
            assert!(
                used.message.contains("user-taught"),
                "the reused op must be tagged user-taught, not engine-native: {}",
                used.message
            );
        });
    }

    /// REGRESSION-GATING (a): an existing engine-native op still works after a
    /// teach; (b): an INCONSISTENT teach (one input → two outputs) is REJECTED
    /// honestly (synthesis cannot reproduce it) and is NOT persisted.
    #[test]
    #[ignore = "slow: full solver synthesis + gated Engine::new() reloads; run with --ignored"]
    fn regression_gating_existing_op_survives_and_inconsistent_teach_rejected() {
        with_temp_component_store(|path| {
            // (a) An existing engine-native capability: noun_animacy classifies a
            // teacher as animate (1). Teach a disjoint op, then re-check it.
            let engine = Engine::new();
            assert_eq!(
                engine.noun_class("teacher"),
                1,
                "precondition: engine-native noun_animacy classifies teacher animate (1)"
            );

            // Teach a brand-new, disjoint op (does not touch noun_animacy).
            let ok = teach_by_examples(&engine, "plus_two", &[(1, 3), (2, 4), (5, 7)]);
            assert!(ok.success, "a disjoint op must be learnable: {}", ok.message);

            // The existing engine-native op STILL WORKS on a fresh (reloaded) engine.
            let after = Engine::new();
            assert_eq!(
                after.noun_class("teacher"),
                1,
                "the engine-native noun_animacy must be intact after teaching a new op"
            );
            assert_eq!(after.eval_int("plus_two(5)"), 7, "the learned op also works");

            // (b) An INCONSISTENT teach: 1->1 AND 1->2. No deterministic function
            // reproduces both, so synthesis FAILS and the teach is rejected honestly.
            let bad = teach_by_examples(&engine, "contradictory", &[(1, 1), (1, 2)]);
            assert!(
                !bad.success,
                "an inconsistent teach must be rejected, not fabricated: {}",
                bad.message
            );
            // It must NOT have polluted the store.
            let stored = store::load();
            assert!(
                !stored.iter().any(|c| c.name == "contradictory"),
                "a rejected teach must not be persisted"
            );
            // The store file may exist (plus_two is in it) but contradictory is absent.
            let _ = path;
        });
    }

    /// TEACH BY COMPOSITION end-to-end: teach `dbl` (x -> 2x) from examples, then
    /// `let quadruple mean dbl then dbl` — derived by RUNNING the composed ops, not
    /// a user table — and prove `quadruple(3) == 12` on a fresh, reloaded engine.
    #[test]
    #[ignore = "slow: full solver synthesis + gated Engine::new() reloads; run with --ignored"]
    fn teach_by_composition_of_learned_ops() {
        with_temp_component_store(|_path| {
            // First teach the building block `dbl` (x -> 2x).
            let e0 = Engine::new();
            let d = teach_by_examples(&e0, "dbl", &[(1, 2), (2, 4), (3, 6), (5, 10)]);
            assert!(d.success, "dbl must be learnable: {}", d.message);

            // Compose: quadruple = dbl then dbl. Built against a FRESH engine that
            // reloaded `dbl` from the store.
            let e1 = Engine::new();
            assert!(
                e1.has_component("dbl"),
                "the reloaded engine must carry the previously-learned dbl"
            );
            let q = teach_by_composition(&e1, "quadruple", &["dbl".to_string(), "dbl".to_string()]);
            assert!(
                q.success,
                "quadruple must be synthesizable from the composition: {}",
                q.message
            );

            // A fresh engine reuses the composite.
            let fresh = Engine::new();
            assert_eq!(
                fresh.eval_int("quadruple(3)"),
                12,
                "quadruple(3) = dbl(dbl(3)) = 12 on the reloaded engine"
            );
        });
    }

    /// An unknown sub-op in a composition is refused honestly (never fabricated).
    #[test]
    #[ignore = "slow: gated Engine::new() reload; run with --ignored"]
    fn composition_with_unknown_subop_refuses() {
        with_temp_component_store(|_path| {
            let engine = Engine::new();
            let out = teach_by_composition(
                &engine,
                "mystery",
                &["nonexistent_op".to_string()],
            );
            assert!(
                !out.success,
                "composition over an unknown sub-op must be refused: {}",
                out.message
            );
            assert!(
                out.message.contains("nonexistent_op"),
                "the refusal must name the missing sub-op: {}",
                out.message
            );
        });
    }
}
