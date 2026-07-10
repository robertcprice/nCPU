//! WP2 — Characterization / oracle bootstrap.
//!
//! When a repair request has examples (or a reference body) but the repo has no
//! failing test oracle, manufacture a checkable `#[test]` module from those
//! examples so the hole-filler / mutation ladder has something to verify against.
//! Never invents expected outputs — only encodes what the caller already stated.

use std::path::Path;

/// One concrete I/O row used to emit a characterization assert.
#[derive(Debug, Clone)]
pub struct CharExample {
    pub inputs: Vec<CharValue>,
    pub expected: CharValue,
}

#[derive(Debug, Clone)]
pub enum CharValue {
    Int(i64),
    Bool(bool),
    Str(String),
    IntList(Vec<i64>),
}

impl CharValue {
    fn rust_lit(&self) -> String {
        match self {
            CharValue::Int(n) => n.to_string(),
            CharValue::Bool(b) => b.to_string(),
            CharValue::Str(s) => format!("\"{}\".to_string()", s.replace('\\', "\\\\").replace('"', "\\\"")),
            CharValue::IntList(xs) => {
                let inner = xs
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("vec![{inner}]")
            }
        }
    }
}

/// Parse simple inline examples from prose: `f(1,2)=3`, `f([1,2]) -> 3`, `name: 2->4, 3->6`.
pub fn parse_inline_char_examples(prose: &str) -> Vec<CharExample> {
    let mut out = Vec::new();
    let lower = prose.replace("→", "->");
    // Prefer the arrow-example region after the first `:` when present
    // ("double a number: 2->4, 3->6"), so we don't need the full NL router.
    let body = match lower.split_once(':') {
        Some((_head, tail)) if tail.contains("->") => tail.trim(),
        _ => lower.as_str(),
    };
    // Split on top-level commas/semicolons (not inside () / [] / "").
    for chunk in split_top_level(body) {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let (lhs, rhs) = if let Some(i) = chunk.find("->") {
            (&chunk[..i], &chunk[i + 2..])
        } else if let Some(i) = chunk.find('=') {
            if chunk.contains("==") {
                continue;
            }
            (&chunk[..i], &chunk[i + 1..])
        } else {
            continue;
        };
        let lhs = lhs.trim();
        let rhs = rhs.trim();
        // Call-shaped: name(args)
        if let (Some(open), Some(close)) = (lhs.find('('), lhs.rfind(')')) {
            if close > open {
                let args_src = &lhs[open + 1..close];
                if let (Some(inputs), Some(expected)) =
                    (parse_arg_list(args_src), parse_value(rhs))
                {
                    out.push(CharExample { inputs, expected });
                    continue;
                }
            }
        }
        // Bare arrow row: `2->4` or multi-arg `2,3->5` (lhs has no parens).
        if let (Some(inputs), Some(expected)) = (parse_arg_list(lhs), parse_value(rhs)) {
            if !inputs.is_empty() {
                out.push(CharExample { inputs, expected });
            }
        }
    }
    // Fallback: verified_nl_router form for nested/array literals the light parser misses.
    if out.len() < 2 && prose.contains("->") {
        let (_, bench) = crate::verified_nl_router::split_prompt_examples(prose);
        if let Some(converted) = char_examples_from_bench(&bench) {
            if converted.len() >= 2 {
                return converted;
            }
        }
    }
    out
}

/// Split on `,` / `;` at depth 0 (respecting `()` / `[]` / `"..."`).
fn split_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut last = 0usize;
    let bytes = s.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            b'"' => in_str = !in_str,
            b'(' | b'[' if !in_str => depth += 1,
            b')' | b']' if !in_str => depth -= 1,
            b',' | b';' if !in_str && depth == 0 => {
                parts.push(&s[last..i]);
                last = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[last..]);
    parts
}

fn parse_arg_list(src: &str) -> Option<Vec<CharValue>> {
    let s = src.trim();
    if s.is_empty() {
        return Some(Vec::new());
    }
    // Single list arg: [1,2,3]
    if s.starts_with('[') && s.ends_with(']') {
        return Some(vec![parse_value(s)?]);
    }
    let mut args = Vec::new();
    for part in s.split(',') {
        args.push(parse_value(part.trim())?);
    }
    Some(args)
}

fn parse_value(src: &str) -> Option<CharValue> {
    let s = src.trim();
    if s == "true" {
        return Some(CharValue::Bool(true));
    }
    if s == "false" {
        return Some(CharValue::Bool(false));
    }
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        return Some(CharValue::Str(s[1..s.len() - 1].to_string()));
    }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        if inner.trim().is_empty() {
            return Some(CharValue::IntList(vec![]));
        }
        let mut xs = Vec::new();
        for p in inner.split(',') {
            xs.push(p.trim().parse::<i64>().ok()?);
        }
        return Some(CharValue::IntList(xs));
    }
    if let Ok(n) = s.parse::<i64>() {
        return Some(CharValue::Int(n));
    }
    None
}

/// Infer a Rust fn signature from examples (i64 / bool / String / Vec<i64>).
pub fn infer_rust_signature(fn_name: &str, examples: &[CharExample]) -> Option<String> {
    let ex = examples.first()?;
    let mut params = Vec::new();
    for (i, v) in ex.inputs.iter().enumerate() {
        let ty = match v {
            CharValue::Int(_) => "i64",
            CharValue::Bool(_) => "bool",
            CharValue::Str(_) => "String",
            CharValue::IntList(_) => "Vec<i64>",
        };
        params.push(format!("a{i}: {ty}"));
    }
    let ret = match &ex.expected {
        CharValue::Int(_) => "i64",
        CharValue::Bool(_) => "bool",
        CharValue::Str(_) => "String",
        CharValue::IntList(_) => "Vec<i64>",
    };
    Some(format!(
        "pub fn {fn_name}({}) -> {ret}",
        params.join(", ")
    ))
}

/// Type-correct default body so the scaffold COMPILES (tests fail → hole-filler
/// has a gradient). Empty `{}` is illegal for non-() returns in Rust.
pub fn default_body_for_examples(examples: &[CharExample]) -> &'static str {
    match examples.first().map(|e| &e.expected) {
        Some(CharValue::Int(_)) => "0",
        Some(CharValue::Bool(_)) => "false",
        Some(CharValue::Str(_)) => "String::new()",
        Some(CharValue::IntList(_)) => "Vec::new()",
        None => "()",
    }
}

/// Emit a `#[cfg(test)]` module that pins the given examples (one test per row).
pub fn emit_characterization_tests(fn_name: &str, examples: &[CharExample]) -> String {
    let mut tests = String::new();
    for (i, ex) in examples.iter().enumerate() {
        let args = ex
            .inputs
            .iter()
            .map(|v| v.rust_lit())
            .collect::<Vec<_>>()
            .join(", ");
        let expected = ex.expected.rust_lit();
        tests.push_str(&format!(
            "    #[test]\n    fn char_{i}() {{\n        assert_eq!({fn_name}({args}), {expected});\n    }}\n"
        ));
    }
    format!("#[cfg(test)]\nmod characterization {{\n    use super::*;\n{tests}}}\n")
}

/// Result of writing a characterization scaffold into a crate root.
#[derive(Debug, Clone)]
pub struct CharacterizationScaffold {
    pub fn_name: String,
    pub n_tests: usize,
    pub method: &'static str,
}

/// Write (or overwrite) `src/lib.rs` with an empty-body fn + characterization tests.
/// Used when the session root has no oracle yet but the query carries examples.
pub fn write_characterization_crate(
    root: &Path,
    fn_name: &str,
    examples: &[CharExample],
) -> Result<CharacterizationScaffold, String> {
    if examples.len() < 2 {
        return Err("need >=2 examples to bootstrap a characterization oracle".into());
    }
    let sig = infer_rust_signature(fn_name, examples)
        .ok_or_else(|| "could not infer signature from examples".to_string())?;
    let tests = emit_characterization_tests(fn_name, examples);
    let lib = format!("{sig} {{}}\n\n{tests}");
    crate::schema_component::write_lib_crate(root, "char_crate", &lib)?;
    Ok(CharacterizationScaffold {
        fn_name: fn_name.to_string(),
        n_tests: examples.len(),
        method: "whole-software:characterization",
    })
}

/// Append characterization tests to an existing `src/lib.rs` that already defines `fn_name`.
pub fn append_characterization_tests(
    root: &Path,
    fn_name: &str,
    examples: &[CharExample],
) -> Result<usize, String> {
    if examples.is_empty() {
        return Err("no examples".into());
    }
    let path = root.join("src/lib.rs");
    let mut lib = std::fs::read_to_string(&path).map_err(|e| format!("read lib.rs: {e}"))?;
    if lib.contains("mod characterization") {
        return Ok(0); // already bootstrapped
    }
    lib.push('\n');
    lib.push_str(&emit_characterization_tests(fn_name, examples));
    std::fs::write(&path, lib).map_err(|e| format!("write lib.rs: {e}"))?;
    Ok(examples.len())
}

/// Convert verified_nl_router / benchmark examples into characterization rows.
/// Returns `None` when any value is outside the Rust scaffold type set
/// (i64 / bool / String / Vec<i64>).
pub fn char_examples_from_bench(
    examples: &[crate::benchmark::Example],
) -> Option<Vec<CharExample>> {
    let mut out = Vec::with_capacity(examples.len());
    for ex in examples {
        let mut inputs = Vec::with_capacity(ex.inputs.len());
        for v in &ex.inputs {
            inputs.push(bench_value_to_char(v)?);
        }
        let expected = bench_value_to_char(&ex.expected)?;
        out.push(CharExample { inputs, expected });
    }
    Some(out)
}

fn bench_value_to_char(v: &crate::benchmark::Value) -> Option<CharValue> {
    use crate::benchmark::Value;
    match v {
        Value::Int(n) => Some(CharValue::Int(*n)),
        Value::Bool(b) => Some(CharValue::Bool(*b)),
        Value::Str(s) => Some(CharValue::Str(s.clone())),
        Value::Array(xs) => {
            let mut ints = Vec::with_capacity(xs.len());
            for x in xs {
                match x {
                    Value::Int(n) => ints.push(*n),
                    _ => return None,
                }
            }
            Some(CharValue::IntList(ints))
        }
        _ => None,
    }
}

/// Write a characterization crate from benchmark examples (the 58% Rust-lane path).
pub fn write_characterization_from_bench(
    root: &Path,
    fn_name: &str,
    examples: &[crate::benchmark::Example],
) -> Result<CharacterizationScaffold, String> {
    let char_ex = char_examples_from_bench(examples)
        .ok_or_else(|| "examples contain unsupported value shapes for Rust scaffold".to_string())?;
    write_characterization_crate(root, fn_name, &char_ex)
}

/// Extract a plausible fn name from prose (`fix f`, `function add`, `fn foo`, else `f`).
pub fn fn_name_from_prose(prose: &str) -> String {
    let lower = prose.to_lowercase();
    for marker in ["function ", "fn ", "fix ", "implement "] {
        if let Some(i) = lower.find(marker) {
            let rest = &prose[i + marker.len()..];
            let tok = rest
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .find(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
            if let Some(t) = tok {
                let id: String = t
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect::<String>()
                    .to_lowercase();
                if !id.is_empty() && id != "the" && id != "a" && id != "an" {
                    return id;
                }
            }
        }
    }
    // Prefer name from first call-shaped example: add(1,2)=3
    if let Some(ex_chunk) = prose.split(|c| c == ',' || c == ';').next() {
        if let Some(open) = ex_chunk.find('(') {
            let name = ex_chunk[..open]
                .trim()
                .rsplit(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .next()
                .unwrap_or("")
                .trim();
            let id: String = name
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect::<String>()
                .to_lowercase();
            if !id.is_empty() {
                return id;
            }
        }
    }
    "f".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_call_equals_examples() {
        let ex = parse_inline_char_examples("add(1, 2)=3, add(4, 5)=9");
        assert_eq!(ex.len(), 2);
        assert!(matches!(ex[0].inputs[0], CharValue::Int(1)));
        assert!(matches!(ex[0].expected, CharValue::Int(3)));
    }

    #[test]
    fn parses_arrow_and_list() {
        let ex = parse_inline_char_examples("sum([1,2,3]) -> 6; sum([4]) -> 4");
        assert_eq!(ex.len(), 2);
        assert!(matches!(ex[0].inputs[0], CharValue::IntList(ref v) if v == &[1, 2, 3]));
    }

    #[test]
    fn writes_characterization_crate() {
        let root = std::env::temp_dir().join(format!("nsynth_char_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let ex = parse_inline_char_examples("double(2)=4, double(3)=6, double(0)=0");
        let s = write_characterization_crate(&root, "double", &ex).expect("write");
        assert_eq!(s.n_tests, 3);
        let lib = std::fs::read_to_string(root.join("src/lib.rs")).unwrap();
        assert!(lib.contains("pub fn double(a0: i64) -> i64 {}"));
        assert!(lib.contains("assert_eq!(double(2), 4)"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn fn_name_from_call_site() {
        assert_eq!(fn_name_from_prose("add(1,2)=3, add(4,5)=9"), "add");
    }

    #[test]
    fn converts_bench_examples_to_char() {
        use crate::benchmark::{Example, Value};
        let ex = vec![
            Example {
                inputs: vec![Value::Int(2)],
                expected: Value::Int(4),
            },
            Example {
                inputs: vec![Value::Int(3)],
                expected: Value::Int(6),
            },
        ];
        let c = char_examples_from_bench(&ex).expect("convert");
        assert_eq!(c.len(), 2);
        assert!(matches!(c[0].expected, CharValue::Int(4)));
    }

    #[test]
    fn parses_arrow_prompt_examples() {
        let ex = parse_inline_char_examples("double a number: 2->4, 3->6, 0->0");
        assert!(ex.len() >= 2, "got {ex:?}");
        assert!(matches!(ex[0].expected, CharValue::Int(4)));
    }
}
