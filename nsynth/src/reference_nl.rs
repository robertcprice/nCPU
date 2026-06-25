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
