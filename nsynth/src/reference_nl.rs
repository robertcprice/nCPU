//! UNWALL-3-REFERENCE-INTAKE-NL: make reference-implementation intake
//! NL/CLI-reachable.
//!
//! The engine already has a *reference* spec front door:
//! [`crate::benchmark::problem_from_reference`] builds a solvable
//! [`crate::benchmark::Problem`] from a runnable reference implementation alone —
//! it samples fresh inputs, RUNS the reference to manufacture seed I/O examples,
//! and keeps `reference_code` set so the strict verifier
//! ([`crate::runtime::verify_problem_code_strict`]) does differential testing via
//! [`crate::benchmark::generated_holdouts`]. [`crate::agent::coding_intent::Spec::Reference`]
//! already reduces that to a `Problem`. The ONLY missing piece was an NL/CLI
//! *intake*: a `"make a function that behaves like THIS: <reference>"` request
//! never reached that path. This module is that intake — and nothing more.
//!
//! ## Why this is emergent structural intake, not a phrase table
//! The route fires on a STRUCTURAL signal: the request *carries a runnable
//! reference implementation* — a Mog `fn NAME(params) -> RET { ... }` definition
//! (optionally inside a fenced ```` ``` ```` block or after a `behaves like:` /
//! `equivalent to:` marker). We extract that `fn` block by brace-balancing, derive
//! the signature from its header and the name from the header, and hand the code
//! itself to the existing reference path as the spec. There is no table mapping
//! English phrases to operations: the reference's *behavior* is the spec, and the
//! synthesized program is SEARCHED and strict-verified against fresh inputs run
//! through that reference (see [`crate::benchmark::generated_holdouts`]). A request
//! with no embedded runnable `fn` simply is not a reference request
//! ([`ReferenceIntake::NotReference`]); a request that clearly points at a
//! reference but whose code does not parse into a `fn` is refused honestly
//! ([`ReferenceIntake::Unparseable`]) rather than fabricating a spec.

/// Surface markers that *signal an intent to supply a reference*. These are used
/// ONLY to decide between "this is not a reference request at all"
/// ([`ReferenceIntake::NotReference`]) and "the user pointed at a reference but it
/// could not be parsed" ([`ReferenceIntake::Unparseable`]) — they never map to an
/// operation and never affect the synthesized program. The actual spec is the
/// extracted `fn` body's behavior. A fenced code block also counts as such a
/// marker on its own.
const REFERENCE_MARKERS: &[&str] = &[
    "behaves like",
    "behave like",
    "equivalent to",
    "same as this",
    "same as the",
    "acts like",
    "works like",
    "matches this",
    "like this",
    "like this:",
    "this function",
    "this code",
    "this reference",
    "reference implementation",
];

/// Outcome of structurally inspecting a query for an embedded reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceIntake {
    /// No reference signal: route normally (examples / NL / pipeline / tensor …).
    NotReference,
    /// A runnable reference `fn` was extracted. `signature` is the `fn`-header
    /// slice (`fn name(a: i64) -> i64`) and `code` is the full balanced `fn`
    /// block. `name` is parsed from the header.
    Reference {
        name: String,
        signature: String,
        code: String,
    },
    /// The request points at a reference (fenced block or a `behaves like`-style
    /// marker) but no runnable `fn NAME(params) -> RET { ... }` could be
    /// extracted. Refuse honestly — never fabricate a spec. Carries a reason.
    Unparseable(String),
}

/// Structurally classify a query for reference-implementation intake.
///
/// Emergent rule (no phrase→op table):
///   1. Try to extract a balanced `fn NAME(params) -> RET { ... }` block.
///   2. If one is found AND its header parses into a name + signature → it is the
///      spec ([`ReferenceIntake::Reference`]).
///   3. If the request otherwise *signals* a reference (a fenced code block or a
///      `behaves like`-style marker) but no `fn` block parses → refuse honestly
///      ([`ReferenceIntake::Unparseable`]).
///   4. Otherwise it is not a reference request ([`ReferenceIntake::NotReference`]).
pub fn classify(query: &str) -> ReferenceIntake {
    let fenced = extract_fenced(query);
    let signals_reference = fenced.is_some() || has_reference_marker(query);

    // Prefer code inside a fenced block (the user delimited it), then the whole
    // query. Either way we look for a brace-balanced `fn ... { ... }`.
    let search_spaces: Vec<&str> = match fenced.as_deref() {
        Some(inner) => vec![inner, query],
        None => vec![query],
    };

    for space in search_spaces {
        if let Some(block) = extract_fn_block(space) {
            match parse_header(&block) {
                Some((name, signature)) => {
                    return ReferenceIntake::Reference {
                        name,
                        signature,
                        code: block,
                    };
                }
                None => {
                    // A `fn ... {` was present but the header did not parse into a
                    // name + sampleable-shaped signature. If the user clearly
                    // intended a reference, refuse honestly.
                    if signals_reference {
                        return ReferenceIntake::Unparseable(
                            "a reference was supplied but its `fn` header could not be parsed \
                             (expected `fn NAME(params) -> RET { ... }`)"
                                .to_string(),
                        );
                    }
                }
            }
        }
    }

    if signals_reference {
        // The user pointed at a reference (fenced block / `behaves like` marker)
        // but no runnable `fn` block was found. Honest refusal — never fabricate.
        return ReferenceIntake::Unparseable(
            "a reference was requested (`behaves like`/fenced code) but no runnable \
             `fn NAME(params) -> RET { ... }` could be extracted from the request"
                .to_string(),
        );
    }

    ReferenceIntake::NotReference
}

fn has_reference_marker(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    REFERENCE_MARKERS.iter().any(|m| lower.contains(m))
}

/// Pull the contents of the first fenced ```` ``` ```` block, if any. An optional
/// language tag on the opening fence (```` ```rust ````) is stripped.
fn extract_fenced(query: &str) -> Option<String> {
    let start = query.find("```")?;
    let after = &query[start + 3..];
    let end_rel = after.find("```")?;
    let mut inner = &after[..end_rel];
    // Drop an optional language tag line (e.g. "rust\n").
    if let Some(nl) = inner.find('\n') {
        let first_line = inner[..nl].trim();
        // A language tag is a single bare word with no `fn`/parens/braces.
        if !first_line.is_empty()
            && !first_line.contains(' ')
            && !first_line.contains('(')
            && !first_line.contains('{')
        {
            inner = &inner[nl + 1..];
        }
    }
    Some(inner.to_string())
}

/// Extract the first brace-balanced `fn NAME(...) -> ... { ... }` block from
/// `text`. Returns the slice from `fn` through the matching closing `}`.
fn extract_fn_block(text: &str) -> Option<String> {
    let fn_pos = find_fn_keyword(text)?;
    let rest = &text[fn_pos..];
    // Find the first `{` after the header.
    let brace_open = rest.find('{')?;
    // Balance braces from there.
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    let mut i = brace_open;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(rest[..=i].trim().to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Find a `fn ` keyword at a word boundary (so `transform` does not match).
fn find_fn_keyword(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("fn ") {
        let abs = search_from + rel;
        let before_ok = abs == 0 || !is_ident_byte(bytes[abs - 1]);
        if before_ok {
            return Some(abs);
        }
        search_from = abs + 3;
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Parse the `fn NAME(params) -> RET` header out of a `fn ... { ... }` block.
/// Returns `(name, signature)` where `signature` is the header slice up to (but
/// excluding) the opening `{`, trimmed. Returns `None` if the name is empty or
/// there is no parameter list.
fn parse_header(block: &str) -> Option<(String, String)> {
    let brace = block.find('{')?;
    let header = block[..brace].trim();
    // Must start with `fn ` and contain a parameter list.
    let after_fn = header.strip_prefix("fn ")?.trim_start();
    let paren = after_fn.find('(')?;
    let name = after_fn[..paren].trim();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    // Require a closing paren so the signature is well-formed.
    if !after_fn.contains(')') {
        return None;
    }
    Some((name.to_string(), header.to_string()))
}

// ───────────────────────────────────────────────────────────────────────────
// P2C-PROMPT-TO-CONTRACT: bare NL behaviour DESCRIPTION → ordered primitive chain
// ───────────────────────────────────────────────────────────────────────────
//
// `classify` (above) needs an embedded runnable `fn`. `classify_compositional`
// is the next door down: a *prose* description of a behaviour as a SEQUENCE of
// known operations ("the larger of two numbers, **then** triple it") whose each
// clause resolves — via the SAME emergent resolver the single-op gate uses
// ([`EntityResolver::resolve_operation_surface`]) — to a registry primitive that
// has an emittable body. There is NO phrase→op table: each clause's operation is
// whatever the registry resolver returns at/above the op-resolution floor, and a
// clause that resolves to NO scalar-`i64` primitive is refused, never fabricated.
//
// Structural signal (mirrors `classify`'s `fn`-block / `behaves like` markers):
// the explicit SEQUENTIAL connector **"then"** splitting the description into ≥2
// ordered step-clauses. This is a language-level sequencing word, not a domain
// phrase. v1 scope is a single LINEAR SCALAR composition over `i64`: a head op
// (arity 1 or 2 — it sets the signature) followed by unary `i64→i64` ops applied
// to the running scalar. Anything else (a non-scalar head, a non-unary tail, an
// unresolvable atom) is reported honestly so the caller refuses rather than
// guessing.

use linguigenesis_core::coding_requirements::OP_RESOLVE_FLOOR;
use linguigenesis_core::entity::EntityType;
use linguigenesis_core::entity_resolution::EntityResolver;
use linguigenesis_core::nl_tokens::tokenize_lower;
use linguigenesis_core::registry::Registry;

/// One resolved atomic step of a compositional description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionalStep {
    /// The clause surface word that resolved to the op (for reporting).
    pub surface: String,
    /// The registry op's canonical fn name (`default_fn_name`).
    pub fn_name: String,
    /// The op's arity (1 or 2). The HEAD step's arity sets the signature; every
    /// non-head step must be arity 1 (applied to the running scalar).
    pub arity: u32,
}

/// Outcome of structurally inspecting a query for a SCALAR compositional
/// description (a `"X, then Y, then Z"` sequence of registry primitives).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionalIntake {
    /// Not our shape (no `then` connector, <2 clauses, or the HEAD clause does
    /// not resolve to a scalar-`i64` primitive). Route normally — the array
    /// pipeline / single-op / clarify doors still apply.
    NotCompositional,
    /// A linear scalar chain: an ordered list of resolved primitives, an inferred
    /// `i64` signature, and a collision-safe composed fn name. Downstream emits a
    /// runnable reference body and feeds it to `problem_from_reference`.
    Compositional {
        name: String,
        signature: String,
        chain: Vec<CompositionalStep>,
    },
    /// The description IS a scalar `then`-composition (its HEAD resolves to a
    /// scalar primitive) but a later atomic step does NOT resolve to a unary
    /// scalar primitive with an emittable body. Refuse (clarify) — never
    /// fabricate a contract from a half-understood description. Carries a reason.
    Unresolvable(String),
}

/// Classify a query as a linear scalar composition of registry primitives.
///
/// Emergent rule (no phrase→op table):
///   1. Require the SEQUENTIAL connector `then`, splitting into ≥2 step-clauses.
///   2. Resolve the HEAD clause to a scalar-`i64` primitive (arity 1 or 2). If it
///      does not resolve to one, this is NOT our shape → [`CompositionalIntake::NotCompositional`].
///   3. Resolve every TAIL clause to a UNARY scalar-`i64` primitive. If any tail
///      fails, the description is a scalar composition with an unresolvable atom →
///      [`CompositionalIntake::Unresolvable`] (refuse, never fabricate).
///   4. Otherwise return the ordered chain + inferred signature + composed name.
pub fn classify_compositional(query: &str, registry: &Registry) -> CompositionalIntake {
    let clauses = split_then_clauses(query);
    if clauses.len() < 2 {
        return CompositionalIntake::NotCompositional;
    }

    let resolver = EntityResolver::new(registry.clone());

    // HEAD: must resolve to a scalar-i64 op (arity 1 or 2). If not, we are not
    // confident this is a scalar composition — let the other doors handle it.
    let head = match resolve_scalar_op(&clauses[0], &resolver) {
        Some(step) if step.arity == 1 || step.arity == 2 => step,
        _ => return CompositionalIntake::NotCompositional,
    };

    // TAILS: each must resolve to a UNARY scalar-i64 op. A tail that fails is an
    // unresolvable atom in a confirmed scalar composition → honest refusal.
    let mut chain = vec![head];
    for clause in &clauses[1..] {
        match resolve_scalar_op(clause, &resolver) {
            Some(step) if step.arity == 1 => chain.push(step),
            Some(step) => {
                return CompositionalIntake::Unresolvable(format!(
                    "step {clause:?} resolved to a non-unary op {:?} (arity {}); v1 only \
                     threads UNARY i64→i64 ops after the head",
                    step.fn_name, step.arity
                ));
            }
            None => {
                return CompositionalIntake::Unresolvable(format!(
                    "step {clause:?} does not resolve to any scalar-i64 primitive with an \
                     emittable body — refusing rather than fabricating a contract"
                ));
            }
        }
    }

    let name = compose_name(&chain);
    let signature = if chain[0].arity == 2 {
        format!("fn {name}(a: i64, b: i64) -> i64")
    } else {
        format!("fn {name}(x: i64) -> i64")
    };
    CompositionalIntake::Compositional {
        name,
        signature,
        chain,
    }
}

/// Split a description into ordered step-clauses on the sequential connector
/// `then` (a language-level sequencing word, the structural signal). Empty
/// clauses are dropped. A description without `then` yields a single clause.
fn split_then_clauses(query: &str) -> Vec<String> {
    let mut clauses: Vec<String> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for word in query.split_whitespace() {
        if word.trim_matches(|c: char| !c.is_alphanumeric()).eq_ignore_ascii_case("then") {
            if !current.is_empty() {
                clauses.push(current.join(" "));
                current.clear();
            }
        } else {
            current.push(word);
        }
    }
    if !current.is_empty() {
        clauses.push(current.join(" "));
    }
    clauses
}

/// Resolve a clause to its best SCALAR-`i64` primitive: the highest-confidence
/// (≥ [`OP_RESOLVE_FLOOR`]) Function/Operator op among the clause's tokens whose
/// signature is purely `i64` (every input `i64`, output `i64`, arity 1 or 2).
/// Reads the resolver + registry entity properties — never a hardcoded op list.
fn resolve_scalar_op(clause: &str, resolver: &EntityResolver) -> Option<CompositionalStep> {
    let mut best: Option<(CompositionalStep, f32)> = None;
    for tok in tokenize_lower(clause) {
      // Try the surface token AND a de-inflected (-s/-es stripped) variant, so a
      // 3rd-person/plural verb ("triples", "negates") reaches the SAME op its base
      // form does. This is generic English inflection applied uniformly to every
      // token — NOT a per-op phrase table; the resolver + the scalar-i64 gate below
      // still decide what (if anything) each variant resolves to.
      for surf in surface_variants(&tok) {
        let Some(resolved) = resolver.resolve_operation_surface(&surf) else {
            continue;
        };
        if resolved.evidence.score < OP_RESOLVE_FLOOR {
            continue;
        }
        if !matches!(
            resolved.entity.entity_type,
            EntityType::Function | EntityType::Operator
        ) {
            continue;
        }
        let input_types = resolved
            .entity
            .get_property("input_types")
            .cloned()
            .unwrap_or_default()
            .to_lowercase();
        let output_type = resolved
            .entity
            .get_property("output_type")
            .cloned()
            .unwrap_or_default()
            .to_lowercase();
        // SCALAR-i64 gate: read the resolved entity's own declared signature.
        if output_type != "i64" {
            continue;
        }
        let params: Vec<&str> = input_types.split(',').map(|s| s.trim()).collect();
        if params.is_empty() || params.iter().any(|p| *p != "i64") {
            continue;
        }
        let arity = params.len() as u32;
        let fn_name = resolved
            .entity
            .get_property("default_fn_name")
            .cloned()
            .unwrap_or_else(|| resolved.entity.lemma.clone());
        let step = CompositionalStep {
            surface: tok.clone(),
            fn_name,
            arity,
        };
        let score = resolved.evidence.score;
        match &best {
            Some((_, b)) if *b >= score => {}
            _ => best = Some((step, score)),
        }
      }
    }
    best.map(|(step, _)| step)
}

/// Surface variants of a token for resolution: the token itself, plus a
/// de-inflected form with a trailing English `-s`/`-es` removed (so a
/// 3rd-person/plural verb resolves to the same op as its base form). Generic
/// morphology — no per-word table; only emitted when stripping leaves a word of
/// at least 3 chars, so short stop-tokens (`is`, `as`, `its`) are untouched.
fn surface_variants(tok: &str) -> Vec<String> {
    let mut out = vec![tok.to_string()];
    if tok.len() >= 4 && tok.ends_with('s') && !tok.ends_with("ss") {
        // "-es" after a sibilant ("boxes", "passes") strips to the base; the plain
        // "-s" case ("triples", "negates") strips a single char.
        let base = if tok.ends_with("es")
            && tok.len() >= 5
            && matches!(
                &tok[tok.len() - 4..tok.len() - 2],
                "ch" | "sh" | "ss" | "zz"
            ) {
            &tok[..tok.len() - 2]
        } else {
            &tok[..tok.len() - 1]
        };
        if base.len() >= 3 && base != tok {
            out.push(base.to_string());
        }
    }
    out
}

/// A collision-safe composed fn name: the chain's op names joined by `_then_`.
/// Because the chain has ≥2 steps the name always contains `_then_`, so it can
/// never collide with a bare primitive's name (the helper bodies emitted beside
/// it).
fn compose_name(chain: &[CompositionalStep]) -> String {
    chain
        .iter()
        .map(|s| s.fn_name.as_str())
        .collect::<Vec<_>>()
        .join("_then_")
}

#[cfg(test)]
mod compositional_tests {
    use super::*;

    fn coding_registry() -> Registry {
        // Mirror the bridge's loader so the test runs against the real op surface.
        crate::linguigenesis_bridge::LinguigenesisBridge::new()
            .registry_clone()
            .expect("coding registry must load for compositional tests")
    }

    #[test]
    fn binary_head_then_unary_tail_resolves() {
        let reg = coding_registry();
        match classify_compositional("the larger of two numbers, then triple it", &reg) {
            CompositionalIntake::Compositional {
                name,
                signature,
                chain,
            } => {
                assert_eq!(chain.len(), 2, "head + one tail");
                assert_eq!(chain[0].fn_name, "max");
                assert_eq!(chain[0].arity, 2, "max is the binary head");
                assert_eq!(chain[1].fn_name, "triple");
                assert_eq!(chain[1].arity, 1);
                assert_eq!(signature, "fn max_then_triple(a: i64, b: i64) -> i64");
                assert_eq!(name, "max_then_triple");
            }
            other => panic!("expected Compositional, got {other:?}"),
        }
    }

    #[test]
    fn unary_head_then_unary_tail_resolves() {
        let reg = coding_registry();
        match classify_compositional(
            "the absolute value of a number, then increment it",
            &reg,
        ) {
            CompositionalIntake::Compositional {
                signature, chain, ..
            } => {
                assert_eq!(chain[0].fn_name, "abs");
                assert_eq!(chain[0].arity, 1, "abs is the unary head");
                assert_eq!(chain[1].fn_name, "increment");
                assert_eq!(signature, "fn abs_then_increment(x: i64) -> i64");
            }
            other => panic!("expected Compositional, got {other:?}"),
        }
    }

    #[test]
    fn unresolvable_tail_refuses_not_fabricates() {
        let reg = coding_registry();
        // Head resolves (max) but the tail atom does not resolve to any primitive.
        match classify_compositional(
            "the larger of two numbers, then frobnicate it",
            &reg,
        ) {
            CompositionalIntake::Unresolvable(_) => {}
            other => panic!("expected Unresolvable (refuse), got {other:?}"),
        }
    }

    /// Run a synthesized scalar fn on hand inputs, asserting an `i64` result.
    fn run_scalar(code: &str, fn_name: &str, inputs: &[i64]) -> i64 {
        use crate::benchmark::Value;
        let args: Vec<Value> = inputs.iter().map(|&n| Value::Int(n)).collect();
        match crate::runtime::execute_function(code, fn_name, &args, "p2c-grader") {
            Ok(crate::runtime::Value::Int(v)) => v,
            other => panic!("synthesized {fn_name} returned non-int {other:?}"),
        }
    }

    /// Drive a compositional description ALL THE WAY: comprehend → emit reference
    /// → `problem_from_reference` auto-generates examples + holdout oracle (NO
    /// human examples) → solve → strict-verify → then the UN-GAMEABLE probe: run
    /// the synthesized program on HAND-specified inputs and compare to outputs the
    /// GRADER computes independently (not from the reference). Returns the solved
    /// code + the auto-generated example count for reporting.
    fn drive_end_to_end(description: &str) -> (String, String, usize) {
        use crate::linguigenesis_bridge::LinguigenesisBridge;
        let reg = coding_registry();
        let bridge = LinguigenesisBridge::new();
        let (name, signature, chain) = match classify_compositional(description, &reg) {
            CompositionalIntake::Compositional {
                name,
                signature,
                chain,
            } => (name, signature, chain),
            other => panic!("{description:?} did not comprehend as Compositional: {other:?}"),
        };
        let reference = bridge
            .emit_scalar_reference(&name, &chain)
            .expect("emit reference");
        let sig_static: &'static str = Box::leak(signature.into_boxed_str());
        let ref_static: &'static str = Box::leak(reference.into_boxed_str());
        let problem = crate::benchmark::problem_from_reference(&name, sig_static, ref_static)
            .expect("problem_from_reference must manufacture examples from the emitted reference");
        let n_examples = problem.examples.len();
        assert!(
            n_examples >= 3,
            "auto-generated >=3 seed examples, got {n_examples}"
        );
        let solved = crate::solver::solve_problem(&problem);
        assert!(
            solved.success,
            "solver must synthesize {name}: method={}, err={:?}",
            solved.method, solved.error
        );
        assert!(
            crate::runtime::verify_problem_code_strict(&problem, &solved.code).is_ok(),
            "synthesized {name} must strict-verify against reference-labelled holdouts"
        );
        (name, solved.code, n_examples)
    }

    #[test]
    fn e2e_larger_then_triple_correct_by_hand() {
        let (name, code, _) =
            drive_end_to_end("the larger of two numbers, then triple it");
        // GRADER computes expected INDEPENDENTLY as max(a,b)*3 — NOT reference-
        // derived. Proves the program does what the USER MEANT.
        for (a, b, expected) in [(3i64, 7i64, 21i64), (9, 2, 27), (5, 1, 15)] {
            let got = run_scalar(&code, &name, &[a, b]);
            assert_eq!(got, expected, "max({a},{b})*3 must equal {expected}");
        }
    }

    #[test]
    fn e2e_absolute_then_increment_correct_by_hand() {
        let (name, code, _) =
            drive_end_to_end("the absolute value of a number, then increment it");
        // GRADER computes expected INDEPENDENTLY as |x|+1.
        for (x, expected) in [(-5i64, 6i64), (3, 4), (-10, 11)] {
            let got = run_scalar(&code, &name, &[x]);
            assert_eq!(got, expected, "|{x}|+1 must equal {expected}");
        }
    }

    #[test]
    fn no_then_connector_is_not_compositional() {
        let reg = coding_registry();
        // A single op with no sequential connector is NOT our shape — it must fall
        // through to the existing single-op door.
        assert_eq!(
            classify_compositional("add two numbers", &reg),
            CompositionalIntake::NotCompositional
        );
    }

    #[test]
    fn pure_unresolvable_single_clause_is_not_compositional() {
        let reg = coding_registry();
        // No connector → NotCompositional (the single-op door then refuses the
        // unknown op itself); we do not steal it.
        assert_eq!(
            classify_compositional("a function that frobnicates a number", &reg),
            CompositionalIntake::NotCompositional
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_inline_reference_polynomial() {
        let q = "make a function that behaves like THIS: fn f(x: i64) -> i64 { return x * x - x; }";
        match classify(q) {
            ReferenceIntake::Reference {
                name,
                signature,
                code,
            } => {
                assert_eq!(name, "f");
                assert_eq!(signature, "fn f(x: i64) -> i64");
                assert!(code.starts_with("fn f"));
                assert!(code.trim_end().ends_with('}'));
            }
            other => panic!("expected Reference, got {other:?}"),
        }
    }

    #[test]
    fn extracts_fenced_reference_with_lang_tag() {
        let q = "build me one equivalent to:\n```rust\nfn g(a: i64) -> i64 {\n    if a < 0 { return -a; }\n    return a;\n}\n```";
        match classify(q) {
            ReferenceIntake::Reference {
                name, signature, ..
            } => {
                assert_eq!(name, "g");
                assert_eq!(signature, "fn g(a: i64) -> i64");
            }
            other => panic!("expected Reference, got {other:?}"),
        }
    }

    #[test]
    fn balances_nested_braces() {
        let q = "behaves like: fn h(x: i64) -> i64 { if x > 0 { return 1; } else { return 0; } }";
        match classify(q) {
            ReferenceIntake::Reference { code, .. } => {
                // The full block including the trailing brace.
                assert!(code.trim_end().ends_with("} }") || code.trim_end().ends_with("}}"));
                assert!(code.contains("else"));
            }
            other => panic!("expected Reference, got {other:?}"),
        }
    }

    #[test]
    fn plain_nl_request_is_not_reference() {
        assert_eq!(classify("add two numbers"), ReferenceIntake::NotReference);
        assert_eq!(
            classify("reverse a list of integers"),
            ReferenceIntake::NotReference
        );
    }

    #[test]
    fn reference_marker_without_code_refuses() {
        // Signals a reference but supplies no runnable fn → honest refusal.
        match classify("make a function that behaves like this reference") {
            ReferenceIntake::Unparseable(_) => {}
            other => panic!("expected Unparseable, got {other:?}"),
        }
    }

    #[test]
    fn fenced_block_without_fn_refuses() {
        let q = "equivalent to:\n```\nx * 2\n```";
        match classify(q) {
            ReferenceIntake::Unparseable(_) => {}
            other => panic!("expected Unparseable, got {other:?}"),
        }
    }
}
