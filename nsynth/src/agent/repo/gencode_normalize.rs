//! Robust Rust-normalization for transpiled NL components (HARDEN-1).
//!
//! The Mog transpiler ([`crate::mog_transpile::to_rust`]) emits Rust that is
//! *close* but not *compiling*: it leaves bare `.len` property access (Go-style),
//! `i64`-typed slice indices, and by-value `Vec` params that the body mutates.
//! Rather than touch the transpiler (owned elsewhere), this module rewrites the
//! GENERATED file with a focused, token/line-aware pass so the produced crate
//! actually `cargo build`s.
//!
//! Each rule maps to a concrete compiler error the audit observed:
//!   1. top-level `fn` -> `pub fn`        (so `pub use module::*` re-exports it)
//!   2. `IDENT.len` -> `IDENT.len()`      (E0616: field `len` is private)
//!   3. `IDENT[expr]` -> `IDENT[(expr) as usize]` (E0277: `[i64]` indexed by i64)
//!   4. mutated by-value `Vec` param -> `mut IDENT: Vec<...>` (E0596)
//!   5. `: Vec<i64> = []` -> `: Vec<i64> = Vec::new()` (E0308: empty array literal)
//!
//! This is deliberately NOT an AST — it is a line/token rewriter scoped to the
//! shapes the transpiler actually emits.

/// Apply the full normalization pass to one transpiled component's Rust source.
pub fn normalize_component(rust: &str) -> String {
    // Rule 7 FIRST: FLOAT fns — a Mog body may combine an f64 with a bare integer
    // literal (`(a0 + a1) / 2`), fine in the interpreter but not in Rust
    // (E0277 `no implementation for f64 / {integer}`; observed post-merge when the
    // universal lane started winning `average`). Inside fns whose signature
    // mentions f64, suffix `.0` to standalone integer literals (float scalar fns
    // have no indices to corrupt).
    let rust = &floatize_int_literals(rust);
    // Determine which top-level params are mutated and need `mut` (rule 4),
    // BEFORE rewriting `.len`/index (those don't affect param-mutation detection
    // but we want a stable scan of the original body for the param signature).
    let mutated = mutated_value_params(rust);
    // Rule 6 inputs: every by-value Vec param. `for x in arr` MOVES such a param,
    // so any later `arr[..]`/`arr.len()` fails to borrow a moved value (E0382).
    let vec_params = value_vec_params(rust);

    let mut out_lines: Vec<String> = Vec::with_capacity(rust.lines().count());
    for line in rust.lines() {
        let trimmed = line.trim_start();
        let indent_len = line.len() - trimmed.len();
        let indent = &line[..indent_len];

        // Rule 5 first: empty array literal initializer -> Vec::new().
        let line_owned = rewrite_empty_vec_literal(line);
        let trimmed = line_owned.trim_start();
        let indent_len = line_owned.len() - trimmed.len();
        let indent = &line_owned[..indent_len];

        // Rule 1 + 4: publicize top-level fns and add `mut` to mutated value params.
        let rewritten = if trimmed.starts_with("fn ") {
            let head = format!("{indent}pub {trimmed}");
            add_mut_to_params(&head, &mutated)
        } else {
            line_owned.clone()
        };

        // Rules 2 + 3: `.len` -> `.len()`, `IDENT[expr]` -> `IDENT[(expr) as usize]`.
        let rewritten = rewrite_len_property(&rewritten);
        let rewritten = rewrite_slice_index(&rewritten);
        // Rule 6: borrow-iterate a by-value Vec param so it isn't moved.
        let rewritten = rewrite_for_in_vec_param(&rewritten, &vec_params);
        out_lines.push(rewritten);
    }
    out_lines.join("\n")
}

/// Rule 6: `for VAR in PARAM` over a by-value `Vec` param MOVES the param, so any
/// later use of it (`PARAM[..]`, `PARAM.len()`, a second loop) fails with E0382.
/// Borrow instead: `for VAR in PARAM.iter().copied()` — elements are `Copy` (i64),
/// so VAR keeps its type and the body is unchanged. Only fires when the iterand is
/// the bare param identifier (not `PARAM.iter()` / `&PARAM`), so it never double-
/// rewrites.
fn rewrite_for_in_vec_param(line: &str, vec_params: &[String]) -> String {
    if !line.trim_start().starts_with("for ") {
        return line.to_string();
    }
    let Some(in_pos) = line.find(" in ") else {
        return line.to_string();
    };
    let after = &line[in_pos + 4..];
    let iterand_end = after
        .find(|c: char| c.is_whitespace() || c == '{')
        .unwrap_or(after.len());
    let iterand = after[..iterand_end].trim();
    if vec_params.iter().any(|p| p == iterand) {
        return format!(
            "{}{}.iter().copied(){}",
            &line[..in_pos + 4],
            iterand,
            &after[iterand_end..]
        );
    }
    line.to_string()
}

/// Rule 5: `: Vec<i64> = []` (any spacing) at the end of an assignment becomes
/// `: Vec<i64> = Vec::new()`. Works on the `let mut x: Vec<i64> = [];` shape too.
fn rewrite_empty_vec_literal(line: &str) -> String {
    // Find `= []` possibly followed by `;`, where the preceding type is a Vec.
    if !line.contains("Vec<") {
        return line.to_string();
    }
    // Normalize the spacing variants of the empty-array literal.
    // Match `= []` with optional inner whitespace.
    let mut result = line.to_string();
    for pat in ["= [ ]", "= []", "=[]", "=  []"] {
        if let Some(pos) = result.find(pat) {
            // Only rewrite if a Vec type precedes the `=`.
            if result[..pos].contains("Vec<") {
                result = format!("{}= Vec::new(){}", &result[..pos], &result[pos + pat.len()..]);
                break;
            }
        }
    }
    result
}

/// Rule 2: rewrite `IDENT.len` (Rust field access, illegal) to `IDENT.len()`,
/// but leave an already-called `IDENT.len()` untouched. Operates char-by-char so
/// it only fires on a `.len` token boundary (not e.g. `.length`).
fn rewrite_len_property(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len() + 8);
    let mut i = 0;
    while i < bytes.len() {
        // Look for `.len` preceded by an identifier char and NOT followed by `(`
        // or another identifier char.
        if i + 4 <= bytes.len() && &bytes[i..i + 4] == b".len" {
            let prev_is_ident = i > 0
                && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            let next = bytes.get(i + 4).copied();
            let next_is_call = next == Some(b'(');
            let next_is_ident = next
                .map(|c| c.is_ascii_alphanumeric() || c == b'_')
                .unwrap_or(false);
            if prev_is_ident && !next_is_call && !next_is_ident {
                // `.len()` yields `usize`, but the transpiler uses `.len` inside
                // i64 arithmetic (Mog's array length is i64). Cast back to i64 so
                // the surrounding expression keeps its type (E0308 otherwise).
                out.push_str(".len() as i64");
                i += 4;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Rule 3: rewrite `IDENT[expr]` slice/Vec index to `IDENT[(expr) as usize]` so an
/// `i64` index compiles. Skips `vec![` (macro), already-cast indices, and indices
/// that are bare integer literals (a `usize` literal indexes fine — but the
/// transpiler emits variables, so we cast any non-trivial expr). Char-scanned so
/// it only fires when an identifier char immediately precedes `[`.
fn rewrite_slice_index(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len() + 16);
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'[' && i > 0 {
            let prev = bytes[i - 1];
            // `IDENT[` (not `!` -> macro, not whitespace -> array literal).
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                // Find matching `]` accounting for nesting.
                if let Some(close) = matching_bracket(bytes, i) {
                    let inner = &line[i + 1..close];
                    let inner_trimmed = inner.trim();
                    let already_cast = inner_trimmed.ends_with("as usize");
                    let is_empty = inner_trimmed.is_empty();
                    if !already_cast && !is_empty {
                        out.push('[');
                        out.push('(');
                        out.push_str(inner);
                        out.push_str(") as usize]");
                        i = close + 1;
                        continue;
                    }
                }
            }
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// Index of the `]` matching the `[` at `open`, accounting for nested brackets.
fn matching_bracket(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Collect the names of by-value `Vec<...>` params that the function body mutates
/// (via `.sort()`, `.push(`, `.reverse()`, `.insert(`, `.pop(`, or assignment
/// `IDENT = ...`). These need a `mut` binding (E0596).
fn mutated_value_params(rust: &str) -> Vec<String> {
    let params = value_vec_params(rust);
    params
        .into_iter()
        .filter(|p| param_is_mutated(rust, p))
        .collect()
}

/// All param names declared with a by-value `Vec<...>` type across top-level fns.
fn value_vec_params(rust: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in rust.lines() {
        let t = line.trim_start();
        if !(t.starts_with("fn ") || t.starts_with("pub fn ")) {
            continue;
        }
        // Extract the parenthesized param list.
        let (Some(lp), Some(rp)) = (line.find('('), line.rfind(')')) else {
            continue;
        };
        if lp >= rp {
            continue;
        }
        for raw in line[lp + 1..rp].split(',') {
            let part = raw.trim();
            if part.is_empty() {
                continue;
            }
            // `name: Vec<...>` — by reference (`&`) is fine, only by-value mutated.
            if let Some((name, ty)) = part.split_once(':') {
                let name = name.trim();
                let ty = ty.trim();
                if ty.starts_with("Vec<") && is_ident(name) {
                    names.push(name.to_string());
                }
            }
        }
    }
    names
}

/// Does the body mutate `name` (method-mutate or rebind)?
fn param_is_mutated(rust: &str, name: &str) -> bool {
    for line in rust.lines() {
        let t = line.trim();
        // Skip the signature line itself.
        if t.starts_with("fn ") || t.starts_with("pub fn ") {
            continue;
        }
        for m in [
            format!("{name}.sort("),
            format!("{name}.push("),
            format!("{name}.reverse("),
            format!("{name}.insert("),
            format!("{name}.pop("),
            format!("{name}.remove("),
            format!("{name}.clear("),
            format!("{name}.truncate("),
        ] {
            if line_contains_token(line, &m) {
                return true;
            }
        }
        // Rebinding assignment: `name = ...` (not `==`, not `name.field =`).
        if let Some(pos) = find_token(line, name) {
            let after = line[pos + name.len()..].trim_start();
            if let Some(rest) = after.strip_prefix('=') {
                if !rest.starts_with('=') {
                    return true;
                }
            }
        }
    }
    false
}

/// True if `needle` occurs in `line` with a non-identifier char (or start) before
/// it, so `arr.push` doesn't match inside `myarr.push`.
fn line_contains_token(line: &str, needle: &str) -> bool {
    find_token(line, needle).is_some()
}

fn find_token(line: &str, needle: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let nb = needle.as_bytes();
    if nb.is_empty() {
        return None;
    }
    let mut i = 0;
    while i + nb.len() <= bytes.len() {
        if &bytes[i..i + nb.len()] == nb {
            let prev_ok = i == 0
                || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            if prev_ok {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Rewrite a `fn` signature line, prefixing `mut ` to any param in `mutated`.
fn add_mut_to_params(sig_line: &str, mutated: &[String]) -> String {
    if mutated.is_empty() {
        return sig_line.to_string();
    }
    let (Some(lp), Some(rp)) = (sig_line.find('('), sig_line.rfind(')')) else {
        return sig_line.to_string();
    };
    if lp >= rp {
        return sig_line.to_string();
    }
    let head = &sig_line[..lp + 1];
    let tail = &sig_line[rp..];
    let params = &sig_line[lp + 1..rp];
    let rewritten: Vec<String> = params
        .split(',')
        .map(|raw| {
            let leading_ws: String = raw.chars().take_while(|c| c.is_whitespace()).collect();
            let part = raw.trim();
            if part.is_empty() {
                return raw.to_string();
            }
            if let Some((name, _)) = part.split_once(':') {
                let name = name.trim();
                if mutated.iter().any(|m| m == name) && !part.starts_with("mut ") {
                    return format!("{leading_ws}mut {part}");
                }
            }
            raw.to_string()
        })
        .collect();
    format!("{head}{}{tail}", rewritten.join(","))
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !s.chars().next().unwrap().is_ascii_digit()
}

/// Rust keywords that cannot be used as module names; escape via `r#` raw ident.
const RUST_KEYWORDS: &[&str] = &[
    "as", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "false", "fn",
    "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
    "return", "self", "static", "struct", "super", "trait", "true", "type", "unsafe", "use",
    "where", "while", "async", "await", "box", "do", "final", "macro", "override", "priv",
    "typeof", "unsized", "virtual", "yield", "abstract", "become",
];

/// True if `name` is a Rust keyword that needs escaping when used as a module name.
pub fn is_rust_keyword(name: &str) -> bool {
    RUST_KEYWORDS.contains(&name)
}

/// Escape a sanitized module name so it is a legal `mod`/file/`pub use` identifier.
/// Keywords become `<name>_m` (a stable, file-system-friendly suffix rather than a
/// raw `r#` identifier, which cannot name a file `r#loop.rs`).
pub fn escape_module_name(name: &str) -> String {
    if is_rust_keyword(name) {
        format!("{name}_m")
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_len_property_to_call() {
        let src = "pub fn f(arr: Vec<i64>) -> i64 {\n    let i: i64 = arr.len - 1;\n    return i;\n}\n";
        let out = normalize_component(src);
        assert!(out.contains("arr.len() as i64"), "got: {out}");
        assert!(!out.contains("arr.len -"), "got: {out}");
    }

    #[test]
    fn borrow_iterates_reused_vec_param() {
        // The exact shape that failed the compile gate: `for x in arr` then `arr[..]`
        // again -> E0382 (arr moved). Rule 6 must borrow-iterate.
        let src = "fn find_maximum(arr: Vec<i64>) -> i64 {\n    let mut r0: i64 = arr[0];\n    \
                   for x0 in arr {\n        r0 = max(r0, x0);\n    }\n    let mut lo1: i64 = arr[0];\n    return r0;\n}\n";
        let out = normalize_component(src);
        assert!(out.contains("for x0 in arr.iter().copied()"), "got: {out}");
        assert!(!out.contains("for x0 in arr {"), "still moves arr: {out}");
    }

    #[test]
    fn does_not_double_borrow_iter() {
        // An already-borrowed loop must not be rewritten again.
        let src = "fn f(arr: Vec<i64>) -> i64 {\n    let mut s: i64 = 0;\n    for x in arr.iter().copied() {\n        s = s + x;\n    }\n    return s;\n}\n";
        let out = normalize_component(src);
        assert!(!out.contains(".iter().copied().iter()"), "double-rewrote: {out}");
    }

    #[test]
    fn does_not_double_call_len() {
        let src = "fn f(a: Vec<i64>) -> i64 { return a.len() as i64; }";
        let out = normalize_component(src);
        assert!(out.contains("a.len()"));
        assert!(!out.contains("a.len()()"), "got: {out}");
    }

    #[test]
    fn casts_i64_index_to_usize() {
        let src = "fn f(arr: Vec<i64>) -> i64 {\n    return arr[i];\n}";
        let out = normalize_component(src);
        assert!(out.contains("arr[(i) as usize]"), "got: {out}");
    }

    #[test]
    fn does_not_double_cast_index() {
        let src = "fn f(arr: Vec<i64>) -> i64 { return arr[(i) as usize]; }";
        let out = normalize_component(src);
        assert!(out.contains("arr[(i) as usize]"));
        assert!(!out.contains("as usize) as usize"), "got: {out}");
    }

    #[test]
    fn does_not_touch_vec_macro_or_array_literal() {
        let src = "fn f() -> Vec<i64> { let v = vec![1, 2, 3]; return v; }";
        let out = normalize_component(src);
        assert!(out.contains("vec![1, 2, 3]"), "got: {out}");
    }

    #[test]
    fn adds_mut_to_sorted_param() {
        let src = "fn sort(arr: Vec<i64>) -> Vec<i64> {\n    arr.sort();\n    return arr;\n}";
        let out = normalize_component(src);
        assert!(out.contains("mut arr: Vec<i64>"), "got: {out}");
    }

    #[test]
    fn does_not_add_mut_to_unmutated_param() {
        let src = "fn first(arr: Vec<i64>) -> i64 {\n    return arr[0];\n}";
        let out = normalize_component(src);
        assert!(!out.contains("mut arr"), "got: {out}");
    }

    #[test]
    fn publicizes_top_level_fn() {
        let src = "fn add(a: i64, b: i64) -> i64 { return a + b; }";
        let out = normalize_component(src);
        assert!(out.starts_with("pub fn add"), "got: {out}");
    }

    #[test]
    fn rewrites_empty_vec_literal() {
        let src = "fn f() -> Vec<i64> {\n    let mut r: Vec<i64> = [];\n    return r;\n}";
        let out = normalize_component(src);
        assert!(out.contains("= Vec::new()"), "got: {out}");
        assert!(!out.contains("= [];"), "got: {out}");
    }

    #[test]
    fn keyword_module_names_escaped() {
        assert!(is_rust_keyword("loop"));
        assert!(is_rust_keyword("match"));
        assert!(is_rust_keyword("type"));
        assert_eq!(escape_module_name("loop"), "loop_m");
        assert_eq!(escape_module_name("negate"), "negate");
    }
}

/// Rule 7: inside fns whose SIGNATURE mentions `f64`, rewrite standalone integer
/// literals to float literals (`2` -> `2.0`) so mixed f64/int arithmetic compiles.
/// Brace-scoped to f64 fns only; literals already carrying a `.` are untouched.
fn floatize_int_literals(rust: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut in_f64_fn = false;
    let mut depth: i32 = 0;
    for line in rust.lines() {
        let t = line.trim_start();
        if t.starts_with("fn ") || t.starts_with("pub fn ") {
            in_f64_fn = line.contains("f64");
        }
        let rewritten = if in_f64_fn && !t.starts_with("fn ") && !t.starts_with("pub fn ") {
            let chars: Vec<char> = line.chars().collect();
            let mut s = String::with_capacity(line.len() + 4);
            let mut i = 0;
            while i < chars.len() {
                let c = chars[i];
                if c.is_ascii_digit()
                    && (i == 0 || !(chars[i - 1].is_alphanumeric() || chars[i - 1] == '_' || chars[i - 1] == '.'))
                {
                    let start = i;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                    let lit: String = chars[start..i].iter().collect();
                    // Already a float, or part of an identifier? handled by guards.
                    if i < chars.len() && (chars[i] == '.' || chars[i].is_alphanumeric() || chars[i] == '_') {
                        s.push_str(&lit);
                    } else {
                        s.push_str(&lit);
                        s.push_str(".0");
                    }
                    continue;
                }
                s.push(c);
                i += 1;
            }
            s
        } else {
            line.to_string()
        };
        out.push(rewritten);
        for c in line.chars() {
            if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth <= 0 {
                    in_f64_fn = false;
                }
            }
        }
    }
    let mut s = out.join("\n");
    if rust.ends_with('\n') {
        s.push('\n');
    }
    s
}
