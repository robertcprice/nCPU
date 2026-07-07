//! Doc-grounded NL ingestion — auto-derive resolver surface forms from the
//! READMEs, doc-comments, and code of real repos, instead of hand-authoring them
//! (the `capability_miner::nl_surface` bottleneck).
//!
//! A doc-comment is a human saying "this symbol does X" — exactly the NL->code
//! mapping the resolver lacks. This pass extracts `(symbol, docstring)` pairs and
//! turns each into a SurfaceForm (candidate NL terms + a gloss) in a schema the
//! resolver can merge. It is UNTRUSTED RECALL signal only: ingested docs raise a
//! symbol's recall for matching prose; the synthesizer/verifier still proves
//! correctness. So noisy/stale docs can widen understanding but never lower
//! soundness. (Rust source for now — the language of the engine + its artifacts.)

/// A documented code symbol: its name, kind, and the doc-comment prose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocSymbol {
    pub name: String,
    pub kind: String, // fn | struct | enum | trait
    pub doc: String,
}

/// Auto-derived NL surface form for a symbol: candidate terms a prompt might use
/// to mean this symbol, plus a one-line gloss. Merges into the resolver as recall
/// vocabulary (same role as capability_miner's hand-authored nl_surface).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceForm {
    pub lemma: String,
    pub terms: Vec<String>,
    pub gloss: String,
}

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "that", "this", "with", "from", "into", "its", "are", "was", "will",
    "each", "all", "any", "not", "but", "you", "your", "our", "use", "used", "uses", "using",
    "given", "return", "returns", "returning", "function", "value", "values", "input", "output",
    "when", "then", "else", "which", "what", "how", "via", "per", "get", "set", "one", "two",
    "over", "under", "only", "also", "same", "such", "does", "can", "may", "must", "should",
];

/// Extract every documented `///` symbol from Rust source.
pub fn extract_doc_symbols(source: &str) -> Vec<DocSymbol> {
    let mut out = Vec::new();
    let mut doc = String::new();
    for raw in source.lines() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("///") {
            if !doc.is_empty() {
                doc.push(' ');
            }
            doc.push_str(rest.trim());
            continue;
        }
        // Attributes sit between the doc and the item — keep the pending doc.
        if line.starts_with("#[") || line.starts_with("#!") {
            continue;
        }
        if doc.is_empty() {
            continue; // ordinary code with no pending doc
        }
        if let Some((kind, name)) = item_kind_name(line) {
            out.push(DocSymbol { name, kind, doc: doc.clone() });
        }
        doc.clear(); // doc attaches only to the immediately-following item
    }
    out
}

/// If `line` declares a `fn`/`struct`/`enum`/`trait`, return (kind, name).
fn item_kind_name(line: &str) -> Option<(String, String)> {
    let mut s = line.trim();
    // Strip visibility + fn modifiers (order matters: pub(...) before pub).
    loop {
        let before = s;
        for pfx in ["pub(crate)", "pub(super)", "pub", "async", "const", "unsafe", "default"] {
            if let Some(r) = s.strip_prefix(pfx) {
                if r.starts_with(char::is_whitespace) || r.starts_with('(') {
                    s = r.trim_start();
                    break;
                }
            }
        }
        if s == before {
            break;
        }
    }
    for (kw, kind) in [("fn ", "fn"), ("struct ", "struct"), ("enum ", "enum"), ("trait ", "trait")] {
        if let Some(rest) = s.strip_prefix(kw) {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some((kind.to_string(), name));
            }
        }
    }
    None
}

/// Extract documented symbols from Python: a `def`/`class` followed by a
/// `"""..."""` (or `'''`) docstring.
pub fn extract_python_symbols(source: &str) -> Vec<DocSymbol> {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim_start();
        let item = t
            .strip_prefix("async def ")
            .map(|r| ("fn", r))
            .or_else(|| t.strip_prefix("def ").map(|r| ("fn", r)))
            .or_else(|| t.strip_prefix("class ").map(|r| ("struct", r)));
        if let Some((kind, rest)) = item {
            let name = leading_ident(rest);
            if !name.is_empty() {
                // Walk to the end of the (possibly multi-line) signature, then the
                // next non-blank line is where a docstring would start.
                let mut j = i;
                while j < lines.len() && !lines[j].trim_end().ends_with(':') {
                    j += 1;
                }
                if let Some((doc, next)) = py_docstring(&lines, j + 1) {
                    out.push(DocSymbol { name, kind: kind.to_string(), doc });
                    i = next;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// If a `"""`/`'''` docstring begins at/after `start` (skipping blanks), return
/// its text + the line index after it.
fn py_docstring(lines: &[&str], start: usize) -> Option<(String, usize)> {
    let mut k = start;
    while k < lines.len() && lines[k].trim().is_empty() {
        k += 1;
    }
    let first = lines.get(k)?.trim();
    let quote = if first.starts_with("\"\"\"") {
        "\"\"\""
    } else if first.starts_with("'''") {
        "'''"
    } else {
        return None;
    };
    let after_open = &first[quote.len()..];
    if let Some(end) = after_open.find(quote) {
        return Some((after_open[..end].trim().to_string(), k + 1));
    }
    let mut doc = String::from(after_open.trim());
    let mut m = k + 1;
    while m < lines.len() {
        let line = lines[m];
        if let Some(end) = line.find(quote) {
            if !line[..end].trim().is_empty() {
                doc.push(' ');
                doc.push_str(line[..end].trim());
            }
            return Some((doc.trim().to_string(), m + 1));
        }
        if !line.trim().is_empty() {
            doc.push(' ');
            doc.push_str(line.trim());
        }
        m += 1;
    }
    Some((doc.trim().to_string(), m))
}

/// Extract documented symbols from Go: contiguous `//` comment lines directly
/// above a `func`/`type` declaration.
pub fn extract_go_symbols(source: &str) -> Vec<DocSymbol> {
    let mut out = Vec::new();
    let mut doc = String::new();
    for raw in source.lines() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("//") {
            if !doc.is_empty() {
                doc.push(' ');
            }
            doc.push_str(rest.trim());
            continue;
        }
        if doc.is_empty() {
            continue;
        }
        let item = line
            .strip_prefix("func ")
            .map(|r| ("fn", r))
            .or_else(|| line.strip_prefix("type ").map(|r| ("struct", r)));
        if let Some((kind, rest)) = item {
            // "func (r *Recv) Name(" — skip a receiver group.
            let rest = rest.trim_start().strip_prefix('(').map_or(rest, |after| {
                after.split_once(')').map(|(_, r)| r.trim_start()).unwrap_or(rest)
            });
            let name = leading_ident(rest.trim_start());
            if !name.is_empty() {
                out.push(DocSymbol { name, kind: kind.to_string(), doc: doc.clone() });
            }
        }
        doc.clear();
    }
    out
}

/// The leading identifier of `s` (letters/digits/underscore).
fn leading_ident(s: &str) -> String {
    s.trim_start()
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

/// Extract documented symbols by file extension (rs/py/go supported).
pub fn extract_by_ext(source: &str, ext: &str) -> Vec<DocSymbol> {
    match ext {
        "py" => extract_python_symbols(source),
        "go" => extract_go_symbols(source),
        _ => extract_doc_symbols(source),
    }
}

/// Turn a documented symbol into a SurfaceForm: the snake_case name words + the
/// doc's content words become candidate NL terms; the first sentence is the gloss.
pub fn derive_surface_form(sym: &DocSymbol) -> SurfaceForm {
    let mut terms: Vec<String> = Vec::new();
    for w in sym.name.split('_') {
        if w.len() >= 2 {
            terms.push(w.to_ascii_lowercase());
        }
    }
    for w in sym.doc.to_ascii_lowercase().split(|c: char| !c.is_ascii_alphanumeric()) {
        if w.len() >= 3 && !STOPWORDS.contains(&w) && w.chars().any(|c| c.is_ascii_alphabetic()) {
            terms.push(w.to_string());
        }
    }
    terms.sort();
    terms.dedup();
    let gloss = sym
        .doc
        .split(['.', '\n'])
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    SurfaceForm { lemma: sym.name.clone(), terms, gloss }
}

/// Ingest one Rust source string into surface-form candidates.
pub fn ingest_source(source: &str) -> Vec<SurfaceForm> {
    extract_doc_symbols(source)
        .iter()
        .map(derive_surface_form)
        .collect()
}

/// Ingest every `.rs` file under `dir` (recursively) — "point it at a repo".
pub fn ingest_dir(dir: &std::path::Path) -> Vec<SurfaceForm> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(p) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&p) else { continue };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                if path.file_name().map(|n| n == "target").unwrap_or(false) {
                    continue;
                }
                stack.push(path);
            } else if let Some(ext) = path.extension().and_then(|x| x.to_str()) {
                if matches!(ext, "rs" | "py" | "go") {
                    if let Ok(src) = std::fs::read_to_string(&path) {
                        out.extend(extract_by_ext(&src, ext).iter().map(derive_surface_form));
                    }
                }
            }
        }
    }
    out
}

/// Ingest a README (markdown) into a project-level SurfaceForm: a summary (the
/// "what / when to use it") + recall terms. Distinct from symbol docs — this is
/// the project-level key for "build me a thing that does <Y>".
pub fn ingest_readme(md: &str) -> SurfaceForm {
    let mut summary = String::new();
    for para in md.split("\n\n") {
        let p = para.trim();
        if p.is_empty() || p.starts_with('#') || p.starts_with("```") || p.starts_with('!') {
            continue;
        }
        summary = p.lines().next().unwrap_or("").trim().to_string();
        break;
    }
    let mut terms: Vec<String> = Vec::new();
    for w in md.to_ascii_lowercase().split(|c: char| !c.is_ascii_alphanumeric()) {
        if w.len() >= 3 && !STOPWORDS.contains(&w) && w.chars().any(|c| c.is_ascii_alphabetic()) {
            terms.push(w.to_string());
        }
    }
    terms.sort();
    terms.dedup();
    SurfaceForm { lemma: "<project>".to_string(), terms, gloss: summary }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
/// Reverse a string in place, preserving unicode scalar order.
pub fn reverse_string(s: &str) -> String { s.chars().rev().collect() }

// not a doc comment — ignored
fn helper() {}

/// Compute the greatest common divisor of two integers.
#[inline]
fn gcd(a: i64, b: i64) -> i64 { if b == 0 { a } else { gcd(b, a % b) } }

/// A bounded ring buffer with fixed capacity.
pub struct RingBuffer { cap: usize }
"#;

    #[test]
    fn extracts_documented_symbols() {
        let syms = extract_doc_symbols(SAMPLE);
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["reverse_string", "gcd", "RingBuffer"], "only /// items, in order");
        assert!(!names.contains(&"helper"), "undocumented item skipped");
        // Attribute between doc and item does not detach the doc.
        let g = syms.iter().find(|s| s.name == "gcd").unwrap();
        assert!(g.doc.contains("greatest common divisor"));
        assert_eq!(g.kind, "fn");
        assert_eq!(syms.iter().find(|s| s.name == "RingBuffer").unwrap().kind, "struct");
    }

    #[test]
    fn derives_surface_forms() {
        let forms = ingest_source(SAMPLE);
        let rev = forms.iter().find(|f| f.lemma == "reverse_string").unwrap();
        // Name words + doc content words become recall terms; stopwords dropped.
        for t in ["reverse", "string", "unicode", "preserving", "scalar", "order"] {
            assert!(rev.terms.contains(&t.to_string()), "term '{t}' in {:?}", rev.terms);
        }
        assert!(!rev.terms.iter().any(|t| t == "the" || t == "in"), "stopwords dropped");
        assert_eq!(rev.gloss, "Reverse a string in place, preserving unicode scalar order");
        // gcd's name + doc both contribute.
        let gcd = forms.iter().find(|f| f.lemma == "gcd").unwrap();
        for t in ["greatest", "common", "divisor", "integers"] {
            assert!(gcd.terms.contains(&t.to_string()), "term '{t}' in {:?}", gcd.terms);
        }
    }

    #[test]
    fn extracts_python_docstrings() {
        let src = "\
def reverse_string(s):
    \"\"\"Reverse the given string and return it.\"\"\"
    return s[::-1]

class Cache:
    '''An LRU cache with time-based eviction.'''
    pass

def no_doc(x):
    return x
";
        let syms = extract_python_symbols(src);
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"reverse_string") && names.contains(&"Cache"), "{names:?}");
        assert!(!names.contains(&"no_doc"), "undocumented python def skipped");
        let rev = syms.iter().find(|s| s.name == "reverse_string").unwrap();
        assert!(rev.doc.contains("Reverse the given string"));
        assert!(syms.iter().find(|s| s.name == "Cache").unwrap().doc.contains("LRU cache"));
    }

    #[test]
    fn extracts_go_doc_comments() {
        let src = "\
// GreatestCommonDivisor returns the gcd of a and b.
func GreatestCommonDivisor(a, b int) int { return a }

// RingBuffer is a fixed-capacity circular buffer.
type RingBuffer struct { cap int }
";
        let syms = extract_go_symbols(src);
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"GreatestCommonDivisor") && names.contains(&"RingBuffer"), "{names:?}");
        let g = syms.iter().find(|s| s.name == "GreatestCommonDivisor").unwrap();
        assert!(g.doc.contains("gcd of a and b"));
    }

    #[test]
    fn ingests_readme_summary_and_terms() {
        let md = "# ripgrep\n\nripgrep recursively searches directories for a regex pattern.\n\n## Install\n\n```\ncargo install ripgrep\n```\n";
        let pd = ingest_readme(md);
        assert_eq!(pd.gloss, "ripgrep recursively searches directories for a regex pattern.");
        for t in ["ripgrep", "recursively", "searches", "directories", "regex", "pattern"] {
            assert!(pd.terms.contains(&t.to_string()), "project term '{t}' in {:?}", pd.terms);
        }
    }
}
