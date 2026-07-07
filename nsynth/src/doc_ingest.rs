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
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SurfaceForm {
    pub lemma: String,
    pub terms: Vec<String>,
    pub gloss: String,
}

/// Write surface forms as line-delimited JSON (the resolver-merge overlay file).
pub fn write_surface_forms_jsonl(path: &std::path::Path, forms: &[SurfaceForm]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    for sf in forms {
        writeln!(f, "{}", serde_json::to_string(sf).unwrap_or_default())?;
    }
    Ok(())
}

/// Read surface forms from a JSONL overlay file (skips malformed lines).
pub fn read_surface_forms_jsonl(path: &std::path::Path) -> Vec<SurfaceForm> {
    std::fs::read_to_string(path)
        .map(|s| {
            s.lines()
                .filter_map(|l| serde_json::from_str::<SurfaceForm>(l.trim()).ok())
                .collect()
        })
        .unwrap_or_default()
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
    let prose = strip_code(&sym.doc);
    for w in prose.to_ascii_lowercase().split(|c: char| !c.is_ascii_alphanumeric()) {
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

/// Drop embedded code from a doc so its tokens don't leak into recall terms:
/// fenced ```code``` blocks, inline `code`, and doctest lines (>>> / assert...).
fn strip_code(doc: &str) -> String {
    // Fenced blocks: keep only segments OUTSIDE ``` pairs.
    let outside: String = doc.split("```").step_by(2).collect::<Vec<_>>().join(" ");
    // Inline `code` spans.
    let no_inline: String = outside.split('`').step_by(2).collect::<Vec<_>>().join(" ");
    // Doctest / assertion lines.
    no_inline
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.starts_with(">>>") && !t.starts_with("assert") && !t.starts_with("let ")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// GATE the raw surface forms into a resolver-ready overlay: keep only
/// DISCRIMINATING doc terms (those appearing in at most `max_df` symbols — a
/// term in many symbols, like "value" or "returns", can't disambiguate). The
/// symbol's own name words are always kept. Forms left with no terms are dropped.
/// This is what makes a noisy corpus safe to merge as recall vocabulary.
pub fn filter_surface_forms(forms: &[SurfaceForm], max_df: usize) -> Vec<SurfaceForm> {
    let mut df: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for f in forms {
        for t in &f.terms {
            *df.entry(t.as_str()).or_default() += 1;
        }
    }
    let mut out = Vec::new();
    for f in forms {
        let name_words: std::collections::HashSet<String> =
            f.lemma.split(|c: char| !c.is_ascii_alphanumeric()).flat_map(|p| {
                let mut v = vec![p.to_ascii_lowercase()];
                // split camelCase / snake handled by the non-alnum split already
                v.retain(|s| s.len() >= 2);
                v
            }).collect();
        let terms: Vec<String> = f
            .terms
            .iter()
            .filter(|t| name_words.contains(t.as_str()) || df.get(t.as_str()).copied().unwrap_or(0) <= max_df)
            .cloned()
            .collect();
        if !terms.is_empty() {
            out.push(SurfaceForm { lemma: f.lemma.clone(), terms, gloss: f.gloss.clone() });
        }
    }
    out
}

/// Mine `(function, [(input, output)])` integer examples from a source/test file:
/// `NAME(IN)` immediately followed by a comparator/comma and an integer `OUT`
/// (covers `assert_eq!(f(2), 4)`, `assert f(2) == 4`, `f(2) => 4`, `f(2) -> 4`).
/// Only names with >= 2 examples are returned. This feeds the verify+register
/// path (`learn_nl::teach_by_examples`) — the flywheel that turns a real repo's
/// tests into named, verified library ops.
pub fn mine_int_examples(source: &str) -> Vec<(String, Vec<(i64, i64)>)> {
    let lines: Vec<&str> = source.lines().collect();
    let mut map: std::collections::BTreeMap<String, Vec<(i64, i64)>> = std::collections::BTreeMap::new();
    let mut push = |map: &mut std::collections::BTreeMap<String, Vec<(i64, i64)>>, name: String, io: (i64, i64)| {
        let v = map.entry(name).or_default();
        if !v.contains(&io) {
            v.push(io);
        }
    };
    for (idx, line) in lines.iter().enumerate() {
        for (name, inp, out) in scan_int_examples(line) {
            push(&mut map, name, (inp, out));
        }
        // Python-style doctest: `>>> f(2)` on one line, the expected int on the next.
        if let Some(rest) = line.trim_start().strip_prefix(">>>") {
            if let Some((name, inp)) = single_int_call(rest.trim()) {
                let mut k = idx + 1;
                while k < lines.len() && lines[k].trim().is_empty() {
                    k += 1;
                }
                if let Some(out) = lines.get(k).and_then(|l| l.trim().parse::<i64>().ok()) {
                    push(&mut map, name, (inp, out));
                }
            }
        }
    }
    map.into_iter().filter(|(_, v)| v.len() >= 2).collect()
}

/// Mine `(function, [(inputs, output)])` for functions of ANY integer arity from
/// a source/test file: `NAME(a, b, ...)` followed by a comparator/comma and an
/// integer `OUT` (covers `assert_eq!(add(1,2), 3)`, `gcd(12,8) == 4`). Subsumes
/// the unary case (arity 1). Feeds learn_nl::teach_by_examples_n.
pub fn mine_multiarg_examples(source: &str) -> Vec<(String, Vec<(Vec<i64>, i64)>)> {
    let mut map: std::collections::BTreeMap<String, Vec<(Vec<i64>, i64)>> = std::collections::BTreeMap::new();
    for line in source.lines() {
        for (name, args, out) in scan_multiarg_examples(line) {
            let v = map.entry(name).or_default();
            let row = (args, out);
            if !v.contains(&row) {
                v.push(row);
            }
        }
    }
    map.into_iter().filter(|(_, v)| v.len() >= 2).collect()
}

fn scan_multiarg_examples(line: &str) -> Vec<(String, Vec<i64>, i64)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !(b[i].is_ascii_alphabetic() || b[i] == b'_') {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        if i >= b.len() || b[i] != b'(' {
            continue;
        }
        let name = line[start..i].to_string();
        // Parse comma-separated signed integer args until ')'.
        let mut j = i + 1;
        let mut args: Vec<i64> = Vec::new();
        let mut ok = true;
        loop {
            while j < b.len() && b[j] == b' ' {
                j += 1;
            }
            let mut k = j;
            if k < b.len() && (b[k] == b'-' || b[k] == b'+') {
                k += 1;
            }
            let ns = k;
            while k < b.len() && b[k].is_ascii_digit() {
                k += 1;
            }
            if k == ns {
                ok = false;
                break;
            }
            let Ok(v) = line[j..k].parse::<i64>() else {
                ok = false;
                break;
            };
            args.push(v);
            while k < b.len() && b[k] == b' ' {
                k += 1;
            }
            if k < b.len() && b[k] == b',' {
                j = k + 1;
                continue;
            }
            if k < b.len() && b[k] == b')' {
                j = k + 1;
                break;
            }
            ok = false;
            break;
        }
        if !ok || args.is_empty() {
            continue;
        }
        let mut k = j;
        while k < b.len() && b[k] == b' ' {
            k += 1;
        }
        let mut matched = false;
        for op in ["==", "=>", "->", ",", "="] {
            if line[k..].starts_with(op) {
                k += op.len();
                matched = true;
                break;
            }
        }
        if !matched {
            continue;
        }
        while k < b.len() && b[k] == b' ' {
            k += 1;
        }
        let mut m = k;
        if m < b.len() && (b[m] == b'-' || b[m] == b'+') {
            m += 1;
        }
        let os = m;
        while m < b.len() && b[m].is_ascii_digit() {
            m += 1;
        }
        if m > os {
            if let Ok(outp) = line[k..m].parse::<i64>() {
                out.push((name, args, outp));
            }
        }
    }
    out
}

/// Mine multi-arg `(function, examples)` from every `.rs`/`.py` file under `dir`.
pub fn ingest_multiarg_examples_dir(dir: &std::path::Path) -> Vec<(String, Vec<(Vec<i64>, i64)>)> {
    let mut map: std::collections::BTreeMap<String, Vec<(Vec<i64>, i64)>> = std::collections::BTreeMap::new();
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
            } else if path.extension().and_then(|x| x.to_str()).map(|x| x == "rs" || x == "py").unwrap_or(false) {
                if let Ok(src) = std::fs::read_to_string(&path) {
                    for (name, rows) in mine_multiarg_examples(&src) {
                        let v = map.entry(name).or_default();
                        for row in rows {
                            if !v.contains(&row) {
                                v.push(row);
                            }
                        }
                    }
                }
            }
        }
    }
    map.into_iter().filter(|(_, v)| v.len() >= 2).collect()
}

/// Parse a bare `NAME(INT)` call (the whole string), returning (name, arg).
fn single_int_call(s: &str) -> Option<(String, i64)> {
    let open = s.find('(')?;
    let name = &s[..open];
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let inner = s[open + 1..].strip_suffix(')')?;
    let arg = inner.trim().parse::<i64>().ok()?;
    Some((name.to_string(), arg))
}

fn scan_int_examples(line: &str) -> Vec<(String, i64, i64)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !(b[i].is_ascii_alphabetic() || b[i] == b'_') {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        if i >= b.len() || b[i] != b'(' {
            continue;
        }
        let name = line[start..i].to_string();
        // Parse a single signed-integer argument: (IN)
        let arg_start = i + 1;
        let mut j = arg_start;
        if j < b.len() && (b[j] == b'-' || b[j] == b'+') {
            j += 1;
        }
        let ns = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == ns || j >= b.len() || b[j] != b')' {
            continue;
        }
        let Ok(inp) = line[arg_start..j].parse::<i64>() else { continue };
        // After ')', require a comparator/comma, then a signed integer OUT.
        let mut k = j + 1;
        while k < b.len() && b[k] == b' ' {
            k += 1;
        }
        let mut matched = false;
        for op in ["==", "=>", "->", ",", "="] {
            if line[k..].starts_with(op) {
                k += op.len();
                matched = true;
                break;
            }
        }
        if !matched {
            continue;
        }
        while k < b.len() && b[k] == b' ' {
            k += 1;
        }
        let mut m = k;
        if m < b.len() && (b[m] == b'-' || b[m] == b'+') {
            m += 1;
        }
        let os = m;
        while m < b.len() && b[m].is_ascii_digit() {
            m += 1;
        }
        if m > os {
            if let Ok(outp) = line[k..m].parse::<i64>() {
                out.push((name, inp, outp));
            }
        }
    }
    out
}

/// Ingest one Rust source string into surface-form candidates.
pub fn ingest_source(source: &str) -> Vec<SurfaceForm> {
    extract_doc_symbols(source)
        .iter()
        .map(derive_surface_form)
        .collect()
}

/// Mine integer `(function, examples)` from every `.rs`/`.py` file under `dir` —
/// the flywheel's example source (feed each to learn_nl::teach_by_examples).
pub fn ingest_examples_dir(dir: &std::path::Path) -> Vec<(String, Vec<(i64, i64)>)> {
    let mut map: std::collections::BTreeMap<String, Vec<(i64, i64)>> = std::collections::BTreeMap::new();
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
            } else if path.extension().and_then(|x| x.to_str()).map(|x| x == "rs" || x == "py").unwrap_or(false) {
                if let Ok(src) = std::fs::read_to_string(&path) {
                    for (name, pairs) in mine_int_examples(&src) {
                        let v = map.entry(name).or_default();
                        for pr in pairs {
                            if !v.contains(&pr) {
                                v.push(pr);
                            }
                        }
                    }
                }
            }
        }
    }
    map.into_iter().filter(|(_, v)| v.len() >= 2).collect()
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
    fn mines_int_examples_from_tests() {
        let src = "\
// double(5) => 10
assert_eq!(double(2), 4);
assert double(3) == 6
let _ = triple(4);            // no expected value -> ignored
assert_eq!(add(1, 2), 3);     // two-arg -> not a unary example
";
        let mined: std::collections::BTreeMap<_, _> = mine_int_examples(src).into_iter().collect();
        let dbl = mined.get("double").expect("double mined");
        assert!(dbl.contains(&(2, 4)) && dbl.contains(&(3, 6)) && dbl.contains(&(5, 10)), "{dbl:?}");
        assert!(!mined.contains_key("triple"), "call with no expected value ignored");
        assert!(!mined.contains_key("add"), "two-arg call is not a unary (in,out)");
    }

    #[test]
    fn mines_multiarg_examples() {
        let src = "assert_eq!(add(1, 2), 3);\nassert_eq!(add(10, 5), 15);\ngcd(12, 8) == 4\ngcd(9, 6) == 3\nassert_eq!(double(2), 4); assert_eq!(double(5), 10);\n";
        let m: std::collections::BTreeMap<_, _> = mine_multiarg_examples(src).into_iter().collect();
        assert_eq!(m.get("add").unwrap(), &vec![(vec![1, 2], 3), (vec![10, 5], 15)]);
        assert_eq!(m.get("gcd").unwrap(), &vec![(vec![12, 8], 4), (vec![9, 6], 3)]);
        // Unary is subsumed (arity 1).
        assert_eq!(m.get("double").unwrap(), &vec![(vec![2], 4), (vec![5], 10)]);
    }

    #[test]
    fn mines_python_doctests() {
        let src = "def square(n):\n    \"\"\"\n    >>> square(2)\n    4\n    >>> square(3)\n    9\n    \"\"\"\n    return n * n\n";
        let mined: std::collections::BTreeMap<_, _> = mine_int_examples(src).into_iter().collect();
        let sq = mined.get("square").expect("square doctest mined");
        assert!(sq.contains(&(2, 4)) && sq.contains(&(3, 9)), "{sq:?}");
    }

    #[test]
    fn gates_noise_into_discriminating_overlay() {
        // Code inside a docstring does not leak into recall terms.
        let sym = DocSymbol {
            name: "read_u8".into(),
            kind: "fn".into(),
            doc: "Reads an unsigned 8 bit integer.\n```\nlet x = assert_eq!(rdr.read_u8(), 5);\n```".into(),
        };
        let f = derive_surface_form(&sym);
        assert!(f.terms.contains(&"reads".to_string()) && f.terms.contains(&"integer".to_string()));
        assert!(
            !f.terms.iter().any(|t| t == "assert_eq" || t == "rdr"),
            "fenced code tokens stripped: {:?}",
            f.terms
        );

        // A generic term (high document-frequency) is dropped, discriminating
        // terms + the symbol's own name words are kept.
        let forms = vec![
            SurfaceForm { lemma: "reverse_string".into(), terms: vec!["reverse".into(), "string".into(), "value".into()], gloss: String::new() },
            SurfaceForm { lemma: "sort_list".into(), terms: vec!["sort".into(), "list".into(), "value".into()], gloss: String::new() },
            SurfaceForm { lemma: "min_value".into(), terms: vec!["min".into(), "value".into()], gloss: String::new() },
        ];
        let gated = filter_surface_forms(&forms, 2); // "value" DF=3 > 2 -> dropped
        let rev = gated.iter().find(|f| f.lemma == "reverse_string").unwrap();
        assert!(rev.terms.contains(&"reverse".to_string()), "discriminating kept");
        assert!(!rev.terms.contains(&"value".to_string()), "generic high-DF term dropped: {:?}", rev.terms);
        let mv = gated.iter().find(|f| f.lemma == "min_value").unwrap();
        assert!(mv.terms.contains(&"value".to_string()), "name word kept even when generic");
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
