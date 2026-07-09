//! Mog → {Python, Rust, TypeScript, Go, Java} transpiler.
//!
//! The Mog programs emitted by the synthesizer are already C-like. Each
//! target language needs only a small set of rewrites:
//!   - Function headers: type syntax, arrow-return vs colon-return
//!   - Variable declarations: strip vs keep types, Python has no `let`
//!   - Block syntax: braces vs indentation
//!   - Type names: `i64` → {`int`, `i64`, `number`}, `[i64]` → {`list[int]`, `Vec<i64>`, `number[]`}
//!
//! This isn't a full parser — Mog's output grammar is regular enough for
//! a line-based rewrite. If synthesized programs ever grow to multi-line
//! expressions or nested closures we'll need a real parser; today every
//! cached program fits the line-based model.
//!
//! The three public entry points are `to_python`, `to_rust`, `to_typescript`.
//! They share `parse_function_header` + indent tracking; the per-language
//! differences are body rewrites.

/// Language target. Controls header syntax, type rewrites, and block-
/// delimiter emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Python,
    Rust,
    TypeScript,
    Go,
    Java,
}

/// Transpile a Mog source string to Python. Returns idiomatic,
/// runnable Python code.
pub fn to_python(mog: &str) -> String {
    transpile(mog, Target::Python)
}

/// Transpile a Mog source string to Rust. The output is a complete
/// `fn NAME(...) -> RET { ... }` block; embed in a crate for execution.
pub fn to_rust(mog: &str) -> String {
    transpile(mog, Target::Rust)
}

/// Transpile a Mog source string to TypeScript. Output is a
/// `function NAME(...): RET { ... }` block, drop into a `.ts` file.
pub fn to_typescript(mog: &str) -> String {
    transpile(mog, Target::TypeScript)
}

/// Transpile a Mog source string to Go. Output is a complete
/// `func NAME(args) RET { ... }` block; drop into a `_test.go` file
/// or a `package main` and call it from `main()`. The integer type
/// is Go's `int` (matches Mog's `i64` on 64-bit targets; using `int`
/// rather than `int64` lets the emitted Go interoperate with
/// `len()` and indexing without explicit casts).
pub fn to_go(mog: &str) -> String {
    transpile(mog, Target::Go)
}

/// Transpile a Mog source string to Java. Output is a complete
/// `public static RET NAME(args) { ... }` block, ready to drop into a
/// class. Integer type is `long` (matches Mog's `i64`); arrays are
/// `long[]`; `bool` round-trips; structs (`Point`, `Rectangle`) keep
/// their original type spelling for the user to declare alongside.
pub fn to_java(mog: &str) -> String {
    transpile(mog, Target::Java)
}

/// Single parametric transpiler; the three public entries are thin
/// wrappers.
fn transpile(mog: &str, target: Target) -> String {
    let mut out = String::new();
    let mut depth: usize = 0;
    let mut in_fn_header = false;

    // Detect whether the Mog function returns an integer. Mog uses `i64`
    // as its integer return type, matching its closed-form-expression
    // synthesis model. When true, the Python target must rewrite bare `/`
    // to `//` so Python's float-division semantics don't silently produce
    // wrong outputs (observed: sum_abs_diffs returned 0.3 instead of 15).
    let returns_int = mog.contains("-> i64");

    for raw_line in mog.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            // Preserve blank lines for readability; don't emit extra
            // indentation on an empty row.
            out.push('\n');
            continue;
        }

        // Function header: `fn NAME(args) -> RET {`. We detect it by the
        // `fn ` prefix; everything else is a body line.
        if line.starts_with("fn ") {
            if let Some(header) = rewrite_fn_header(line, target) {
                // Python dedents the body via colons + indent; Rust/TS use
                // braces which survive in the output. All three targets
                // start the body at depth+1.
                for _ in 0..depth {
                    out.push_str(indent_unit(target));
                }
                out.push_str(&header);
                out.push('\n');
                depth += 1;
                in_fn_header = true;
                continue;
            }
        }

        // Closing brace: `}` on its own line decreases depth. Python
        // elides the brace entirely (blocks close by dedent).
        if line == "}" {
            if depth > 0 {
                depth -= 1;
            }
            match target {
                Target::Python => { /* no-op; dedent is implicit */ }
                Target::Rust | Target::TypeScript | Target::Go | Target::Java => {
                    for _ in 0..depth {
                        out.push_str(indent_unit(target));
                    }
                    out.push_str("}\n");
                }
            }
            continue;
        }

        // Continuation line starting with `}` — e.g. `} else {`. Python's
        // else is just `else:` at one dedent level. Rust/TS keep the
        // literal line (close-brace + else + open-brace).
        if line.starts_with("}") {
            // The close-brace pops one depth level; the opens_block logic
            // below will push one back when the line ends with `{`.
            if depth > 0 {
                depth -= 1;
            }
            let rest = line[1..].trim();
            match target {
                Target::Python => {
                    // For `} else {` → emit `else:` at the *outer* depth
                    // (which is now `depth`), then the body-depth increase
                    // comes from opens_block below.
                    if rest.starts_with("else") {
                        for _ in 0..depth {
                            out.push_str(indent_unit(target));
                        }
                        // Python has no `else if` — render `else if` as
                        // `elif`. `rest` at this point is e.g. `else {` or
                        // `else if X {`.
                        let stripped = rest.trim_end_matches('{').trim();
                        if let Some(cond) = stripped.strip_prefix("else if ") {
                            out.push_str(&format!("elif {}:\n", cond.trim()));
                        } else {
                            out.push_str("else:\n");
                        }
                        depth += 1; // body block opens
                        continue;
                    }
                    // Other `} ...` shapes fall through to body handling.
                }
                Target::Rust | Target::TypeScript | Target::Go | Target::Java => {
                    for _ in 0..depth {
                        out.push_str(indent_unit(target));
                    }
                    // For TS/Java, `} else if (X) {` needs parens around the
                    // condition just like bare `if`. Handle with a small
                    // rewrite; Rust/Go keep the line as-is.
                    let rewritten = match target {
                        Target::TypeScript | Target::Java => {
                            if let Some(cond_plus) = rest.strip_prefix("else if ") {
                                let cond = cond_plus.trim_end_matches('{').trim();
                                format!("}} else if ({}) {{", cond)
                            } else {
                                format!("}} {}", rest)
                            }
                        }
                        _ => format!("}} {}", rest),
                    };
                    out.push_str(rewritten.trim_end());
                    out.push('\n');
                    if rewritten.trim_end().ends_with('{') {
                        depth += 1;
                    }
                    continue;
                }
            }
        }

        // Body lines: dispatch on prefix to a specific rewriter, otherwise
        // fall through to generic expression rewriting (type-name swaps
        // + statement-end semicolons).
        let mut body = rewrite_body_line(line, target);

        // Python integer-division fix. The Mog synthesizer emits `/` with
        // integer-division intent. Python 3's `/` is float division; for
        // integer-returning functions we rewrite the operator so
        // `5 / 2 == 2`, not `2.5`. Only applies at the top-level body of
        // `-> i64` functions — doesn't touch other targets.
        if target == Target::Python {
            body = rewrite_logical_python(&body);
        }
        if target == Target::Python && returns_int {
            body = rewrite_int_div_python(&body);
        }

        // TypeScript integer-division fix (same intent as the Python one
        // above). JS `/` is float division; Mog `/` is truncating i64
        // division. Wrap each division as `Math.trunc(A / B)` so e.g.
        // `399 / 400 == 0`, not `0.9975`. Without this, synthesized
        // programs that are correct under Mog semantics ship as subtly
        // wrong JS (observed: hit_bottom's `a / b` CEGIS-livelocked
        // because the verifier evaluated it as float division).
        if target == Target::TypeScript && returns_int {
            body = rewrite_int_div_typescript(&body);
        }

        // Go length-call rewrite: Mog's `arr.len` is a property access,
        // but Go uses `len(arr)` as a built-in. Rewrite every `X.len`
        // (where X is an identifier) to `len(X)`. The Python-style
        // indentation pass already ran; this is the only Go-specific
        // syntactic transformation on top of the brace handler.
        // Rust index-style array rewrite: hand-written library ops (list_max,
        // reverse_list, list_length, *_except_*) index arrays with an i64
        // counter and read `arr.len` as a property — neither is valid Rust.
        // The synthesizer's own output is iterator-style (`for x in arr`) and
        // has none of these, so this pass only touches the library-op shapes.
        if target == Target::Rust {
            body = rewrite_arrays_rust(&body);
        }

        if target == Target::Go {
            body = rewrite_dot_len_go(&body);
            body = rewrite_dot_sort_go(&body);
            // Go `var` declaration drops the Mog-style colon:
            // `var X: T = V;` -> `var X T = V;`.
            body = body
                .lines()
                .map(rewrite_var_decl_go)
                .collect::<Vec<_>>()
                .join("\n");
            // Mog's implicit-typed declaration `X: T = V;` -> Go's
            // `var X T = V;` (where T is Go's type, e.g. `int64`).
            body = body
                .lines()
                .map(rewrite_implicit_typed_decl_go)
                .collect::<Vec<_>>()
                .join("\n");
            // Mog's `while` -> Go's `for`.
            body = body
                .lines()
                .map(rewrite_while_to_for_go)
                .collect::<Vec<_>>()
                .join("\n");
        }

        // Opening-brace blocks (`if ... {`, `for ... {`, `while ... {`).
        // Python rewrites the trailing `{` to `:`; Rust/TS keep the brace.
        // In all targets, depth increases by one.
        let opens_block = body.trim_end().ends_with('{');
        if opens_block {
            match target {
                Target::Python => {
                    // Strip the trailing `{` and replace with `:`.
                    body = body.trim_end().trim_end_matches('{').trim_end().to_string();
                    body.push(':');
                }
                Target::Rust | Target::TypeScript | Target::Go | Target::Java => {
                    // Keep the brace; no rewrite.
                }
            }
        }

        for _ in 0..depth {
            out.push_str(indent_unit(target));
        }
        out.push_str(&body);
        out.push('\n');

        if opens_block {
            depth += 1;
        }

        let _ = in_fn_header; // suppress unused-warning; reserved for future
    }

    // Python body-only blocks need a `pass` when empty (rare for
    // synthesizer output but safe guard).
    if target == Target::Python && out.trim().is_empty() {
        return String::from("pass\n");
    }
    if target == Target::Rust {
        return add_mut_to_mutated_params(&infer_rust_vec_element_types(&lower_rust_string_building(
            &lower_rust_char_ops(&rewrite_rust_string_splits(&lower_rust_string_methods(&out))),
        )));
    }
    out
}

/// `X.split(SEP)` in Mog produces a LIST of strings, but Rust's `str::split` yields a lazy iterator
/// of `&str`. Materialize it: `X.split(SEP)` -> `X.split(SEP).map(|w| w.to_string()).collect::<Vec<
/// String>>()`, so it type-checks against the `Vec<String>` the surrounding word-list op expects.
fn rewrite_rust_string_splits(rust: &str) -> String {
    let mut out = String::with_capacity(rust.len() + 64);
    let mut i = 0;
    while i < rust.len() {
        if rust[i..].starts_with(".split(") {
            if let Some((_, close)) = balanced_region(rust, i + 6, b'(', b')') {
                out.push_str(&rust[i..=close]);
                out.push_str(".map(|w| w.to_string()).collect::<Vec<String>>()");
                i = close + 1;
                continue;
            }
        }
        let ch = rust[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Per-function element-type INFERENCE for `Vec<i64>` locals that actually hold non-i64 elements.
/// Mog uses `[i64]` as an untyped "list", so a list of chars or of strings is emitted as
/// `Vec<i64>` and then push/compare a `char`/`String` — a type mismatch Rust rejects. Infer the real
/// element type from the pushes: a Vec pushed a char (loop var over `.chars()`, a `'c'` literal, or
/// `s.chars().nth(..)`) is `Vec<char>`; a Vec pushed a `String` value is `Vec<String>`. Only the
/// declaration is retyped — indexing/compare/return then type-check against the real element.
fn infer_rust_vec_element_types(rust: &str) -> String {
    let had_nl = rust.ends_with('\n');
    let lines: Vec<&str> = rust.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim_start();
        if t.starts_with("fn ") || t.starts_with("pub fn ") {
            // Gather the whole function body (balanced braces) and rewrite its Vec decls together.
            let start = i;
            let mut depth = 0i32;
            let mut opened = false;
            let mut j = i;
            while j < lines.len() {
                for c in lines[j].chars() {
                    if c == '{' {
                        depth += 1;
                        opened = true;
                    } else if c == '}' {
                        depth -= 1;
                    }
                }
                if opened && depth <= 0 {
                    break;
                }
                j += 1;
            }
            let end = j.min(lines.len() - 1);
            out.extend(rewrite_fn_vec_element_types(&lines[start..=end]));
            i = end + 1;
        } else {
            out.push(lines[i].to_string());
            i += 1;
        }
    }
    let mut s = out.join("\n");
    if had_nl {
        s.push('\n');
    }
    s
}

fn rewrite_fn_vec_element_types(fn_lines: &[&str]) -> Vec<String> {
    use std::collections::{HashMap, HashSet};
    // char loop vars: `for VAR in <expr>.chars()`.
    let mut char_vars: HashSet<String> = HashSet::new();
    for l in fn_lines {
        if let Some(fp) = l.find("for ") {
            if let Some(ip) = l[fp + 4..].find(" in ") {
                let var = l[fp + 4..fp + 4 + ip].trim().trim_start_matches('&').trim();
                if l[fp + 4 + ip..].contains(".chars()") && !var.is_empty() {
                    char_vars.insert(var.to_string());
                }
            }
        }
    }
    // String-typed locals (already spelled `: String`) to recognise String pushes.
    let mut string_vars: HashSet<String> = HashSet::new();
    for l in fn_lines {
        let t = l.trim_start();
        if let Some(rest) = t.strip_prefix("let mut ").or_else(|| t.strip_prefix("let ")) {
            if let Some((nm, ty)) = rest.split_once(':') {
                if ty.trim_start().starts_with("String") {
                    string_vars.insert(nm.trim().to_string());
                }
            }
        }
    }
    // Infer each Vec's element type from a push whose argument is a char or a String.
    let mut vec_elem: HashMap<String, &'static str> = HashMap::new();
    for l in fn_lines {
        let mut from = 0;
        while let Some(rel) = l[from..].find(".push(") {
            let at = from + rel;
            let before = &l[..at];
            let vec: String = before
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            from = at + 6;
            if let Some((_, close)) = balanced_region(l, at + 5, b'(', b')') {
                let arg = l[at + 6..close].trim();
                let elem = if arg.starts_with('\'')
                    || char_vars.contains(arg)
                    || arg.contains(".chars().nth(")
                {
                    Some("char")
                } else if string_vars.contains(arg)
                    || arg.ends_with(".to_string()")
                    || arg.starts_with("format!(")
                {
                    Some("String")
                } else {
                    None
                };
                if let (false, Some(e)) = (vec.is_empty(), elem) {
                    vec_elem.insert(vec, e);
                }
            }
        }
    }
    fn_lines
        .iter()
        .map(|l| {
            let mut s = l.to_string();
            for (v, e) in &vec_elem {
                s = s
                    .replace(&format!("let mut {v}: Vec<i64>"), &format!("let mut {v}: Vec<{e}>"))
                    .replace(&format!("let {v}: Vec<i64>"), &format!("let {v}: Vec<{e}>"));
            }
            s
        })
        .collect()
}

/// Lower Mog string BUILDING to Rust, scoped to String-typed operands. Mog accumulates a string
/// with `out = out + ch` (or `+ ch.upper()`, `+ other_string`, `+ " "`) and inits/returns bare
/// `""` literals — none of which are valid Rust (`String + char`, `String = &str`). The whole
/// char-string-building op family (~39 ops: reverse-ish, filters, case maps) depends on it. Two
/// rewrites: `A = A + t1 + t2 ..` (A a String) -> `A = format!("{}{}..", A, t1, t2 ..)` (char, str,
/// String and ToUppercase all impl Display); and a bare string LITERAL assigned to a String
/// binding or returned from a `-> String` fn -> `"lit".to_string()` (or `String::new()` for `""`).
fn lower_rust_string_building(rust: &str) -> String {
    use std::collections::HashSet;
    let had_nl = rust.ends_with('\n');
    let mut lines: Vec<String> = Vec::new();
    let mut string_idents: HashSet<String> = HashSet::new();
    let mut returns_string = false;
    for line in rust.lines() {
        let t = line.trim_start();
        let indent = &line[..line.len() - t.len()];
        if t.starts_with("fn ") || t.starts_with("pub fn ") {
            string_idents.clear();
            returns_string = line.contains("-> String");
            if let Some(op) = line.find('(') {
                if let Some(cp) = line[op..].find(')') {
                    for p in line[op + 1..op + cp].split(',') {
                        if let Some((nm, ty)) = p.split_once(':') {
                            if ty.trim().starts_with("String") {
                                string_idents
                                    .insert(nm.trim().trim_start_matches("mut ").trim().to_string());
                            }
                        }
                    }
                }
            }
        }
        if let Some(rest) = t.strip_prefix("let mut ").or_else(|| t.strip_prefix("let ")) {
            if let Some((nm, ty)) = rest.split_once(':') {
                if ty.trim_start().starts_with("String") {
                    string_idents.insert(nm.trim().to_string());
                }
            }
        }
        let stmt = t.trim_end();
        let rewritten = rewrite_string_accum(stmt, &string_idents)
            .or_else(|| rewrite_string_literal_stmt(stmt, &string_idents, returns_string));
        lines.push(match rewritten {
            Some(r) => format!("{indent}{r}"),
            None => line.to_string(),
        });
    }
    let mut out = lines.join("\n");
    if had_nl {
        out.push('\n');
    }
    out
}

/// `A = A + t1 + t2 ..;` (A a String ident) -> `A = format!("{}{}..", A, t1, t2 ..);`.
fn rewrite_string_accum(stmt: &str, idents: &std::collections::HashSet<String>) -> Option<String> {
    let s = stmt.strip_suffix(';')?.trim();
    let (lhs, rhs) = s.split_once('=')?;
    let lhs = lhs.trim();
    if !idents.contains(lhs) || lhs.contains(char::is_whitespace) {
        return None;
    }
    let terms = split_plus_depth0(rhs.trim());
    if terms.len() < 2 || terms[0] != lhs {
        return None; // only rewrite genuine accumulation `A = A + ..`
    }
    let fmt = "{}".repeat(terms.len());
    Some(format!("{lhs} = format!(\"{fmt}\", {});", terms.join(", ")))
}

/// A bare string literal assigned to a String binding or returned from a `-> String` fn needs
/// `.to_string()` (or `String::new()` for `""`): `let x: String = "";`, `x = "a";`, `return "b";`.
fn rewrite_string_literal_stmt(
    stmt: &str,
    idents: &std::collections::HashSet<String>,
    returns_string: bool,
) -> Option<String> {
    let s = stmt.strip_suffix(';')?.trim();
    if let Some(ret) = s.strip_prefix("return ") {
        if returns_string {
            let w = wrap_string_literal(ret)?;
            return Some(format!("return {w};"));
        }
        return None;
    }
    let (lhs, rhs) = s.split_once('=')?;
    let w = wrap_string_literal(rhs.trim())?;
    let lhs_t = lhs.trim();
    // `let [mut] NAME: String = ..` OR `NAME = ..` where NAME is a known String ident.
    let is_string_decl = lhs_t.starts_with("let ") && lhs.contains(": String");
    let name = lhs_t.rsplit(':').next().unwrap_or("").trim();
    let bare_name = name.rsplit(' ').next().unwrap_or("").trim();
    if is_string_decl || idents.contains(bare_name) || idents.contains(lhs_t) {
        return Some(format!("{lhs}= {w};"));
    }
    None
}

/// A pure double-quoted string literal -> its owned-String form; `None` if not a bare literal.
fn wrap_string_literal(rhs: &str) -> Option<String> {
    let r = rhs.trim();
    if r.len() >= 2 && r.starts_with('"') && r.ends_with('"') && r[1..r.len() - 1].find('"').is_none()
    {
        return Some(if r == "\"\"" {
            "String::new()".to_string()
        } else {
            format!("{r}.to_string()")
        });
    }
    None
}

/// Split on `+` at bracket/paren depth 0 (so `arr[i + 1]` stays one term).
fn split_plus_depth0(s: &str) -> Vec<String> {
    let b = s.as_bytes();
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut last = 0usize;
    for (i, &c) in b.iter().enumerate() {
        match c {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'+' if depth == 0 => {
                parts.push(s[last..i].trim().to_string());
                last = i + 1;
            }
            _ => {}
        }
    }
    parts.push(s[last..].trim().to_string());
    parts
}

/// Lower Mog per-CHARACTER string ops to Rust, scoped to String-typed operands so array code is
/// untouched. Mog iterates a string with `for ch in s` and tests chars with `.is_vowel()` /
/// `.is_alpha()` / `.is_digit()`; Rust needs `for ch in s.chars()` and real predicates. This
/// unlocks the char-counting op family (count_vowels/count_consonants/count_letters/
/// count_non_letters/contains_vowel) which returns i64 — outside lower_rust_string_methods' `->
/// String` scope. Ops that also concat chars or index the string (uppercase_vowels, middle_char)
/// need more machinery and are left for the cargo oracle to reject.
fn lower_rust_char_ops(rust: &str) -> String {
    use std::collections::HashSet;
    let had_nl = rust.ends_with('\n');
    let mut lines: Vec<String> = Vec::new();
    let mut string_idents: HashSet<String> = HashSet::new();
    let mut vec_idents: HashSet<String> = HashSet::new();
    let mut char_vars: HashSet<String> = HashSet::new();
    for line in rust.lines() {
        let t = line.trim_start();
        if t.starts_with("fn ") || t.starts_with("pub fn ") {
            string_idents.clear();
            vec_idents.clear();
            char_vars.clear();
            if let Some(op) = line.find('(') {
                if let Some(cp) = line[op..].find(')') {
                    for p in line[op + 1..op + cp].split(',') {
                        if let Some((nm, ty)) = p.split_once(':') {
                            let name = nm.trim().trim_start_matches("mut ").trim().to_string();
                            let ty = ty.trim();
                            if ty.starts_with("String") {
                                string_idents.insert(name);
                            } else if ty.starts_with("Vec<") {
                                vec_idents.insert(name);
                            }
                        }
                    }
                }
            }
        }
        // Local `let [mut] NAME: TYPE` declares a String / Vec binding.
        if let Some(rest) = t.strip_prefix("let mut ").or_else(|| t.strip_prefix("let ")) {
            if let Some((nm, ty)) = rest.split_once(':') {
                let name = nm.trim().to_string();
                let ty = ty.trim_start();
                if ty.starts_with("String") {
                    string_idents.insert(name);
                } else if ty.starts_with("Vec<") {
                    vec_idents.insert(name);
                }
            }
        }
        let mut l = line.to_string();
        // `for VAR in IDENT {` over a String -> iterate `.chars()`; VAR is then a `char`.
        if let Some(fp) = l.find("for ") {
            let after = &l[fp + 4..];
            if let Some(ip) = after.find(" in ") {
                let var = after[..ip].trim().to_string();
                let ident: String = after[ip + 4..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !ident.is_empty() && string_idents.contains(&ident) {
                    l = l.replacen(&format!("in {ident}"), &format!("in {ident}.chars()"), 1);
                    char_vars.insert(var);
                } else if !ident.is_empty() && vec_idents.contains(&ident) {
                    // `for X in VEC` moves VEC — a nested/subsequent use then fails to compile
                    // (`use of moved value`). Iterate a borrow and clone each element so VEC stays
                    // usable and X is owned (no deref rewrites needed). Cheap for the i64/bool
                    // element types these ops carry; correctness-first for the rest.
                    let _ = &var;
                    l = l.replacen(
                        &format!("in {ident}"),
                        &format!("in {ident}.iter().cloned()"),
                        1,
                    );
                }
            }
        }
        // `.is_vowel()` maps to `"aeiouAEIOU".contains(OPERAND)`, so it needs the operand — handle
        // the loop-var form here (the common one). The remaining char methods map to a Rust method
        // NAME (or a cast), independent of the operand, so they lower GLOBALLY below.
        for cv in &char_vars {
            l = l
                .replace(&format!("{cv}.is_vowel()"), &format!("\"aeiouAEIOU\".contains({cv})"))
                // `return ch;` returns a char, but a Mog fn that returns a char is spelled
                // `-> string`, so the Rust return type is String — materialize the char.
                .replace(&format!("return {cv};"), &format!("return {cv}.to_string();"));
        }
        // GLOBAL char-method lowering: these Mog methods only ever apply to a `char`, so rewriting
        // the method token wherever it appears is safe — including chars produced by string INDEXING
        // (`s.chars().nth(i).unwrap().is_upper()`), which the loop-var pass above cannot see. `.ord()`
        // (char code point) becomes an `as i64` cast.
        l = l
            .replace(".is_upper()", ".is_uppercase()")
            .replace(".is_lower()", ".is_lowercase()")
            .replace(".is_alpha()", ".is_alphabetic()")
            .replace(".is_alnum()", ".is_alphanumeric()")
            .replace(".is_digit()", ".is_ascii_digit()")
            .replace(".ord()", " as i64");
        // WHOLE-string methods on a String operand, in ANY fn (lower_rust_string_methods only fires
        // for `-> String` fns, so `s.reverse()` inside a `-> bool` palindrome check was left raw).
        // Only the forms that REMOVE the original method token — re-running over already-lowered
        // `-> String` output is then a no-op. (`.trim()` keeps its token, so it stays with the
        // `-> String`-scoped pass to avoid a double `.to_string()`.)
        for si in &string_idents {
            l = l
                .replace(
                    &format!("{si}.reverse()"),
                    &format!("{si}.chars().rev().collect::<String>()"),
                )
                .replace(&format!("{si}.upper()"), &format!("{si}.to_uppercase()"))
                .replace(&format!("{si}.lower()"), &format!("{si}.to_lowercase()"));
        }
        // String INDEXING: Rust `String` is not `Index<usize>`. The array pass already rewrote
        // `s[i]` -> `s[(i) as usize]`; turn that (for String idents) into a char lookup
        // `s.chars().nth((i) as usize).unwrap()`. Enables first==last / char-at ops.
        if !string_idents.is_empty() {
            l = lower_string_index(&l, &string_idents);
        }
        lines.push(l);
    }
    let mut out = lines.join("\n");
    if had_nl {
        out.push('\n');
    }
    out
}

/// A Rust fn parameter that the body MUTATES (index-assign `a[i] = ..`, reassign `a = ..`,
/// compound-assign `a += ..`, or an in-place method `a.push(..)/.sort()/..`) must be declared
/// `mut a: T`. Mog is untyped-mutability (every binding is assignable), so its hand-written
/// in-place ops (sort, in-place reverse) mutate their array parameter directly — without this the
/// emitted Rust fails to compile (`cannot borrow as mutable`). The synthesizer's own output never
/// mutates a parameter, so this only lifts the library-op shapes.
fn add_mut_to_mutated_params(rust: &str) -> String {
    let bytes = rust.as_bytes();
    let mut out = String::with_capacity(rust.len() + 16);
    let mut i = 0;
    while i < rust.len() {
        let at_fn = rust[i..].starts_with("fn ")
            && (i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_'));
        if at_fn {
            if let Some((ps, pe)) = rust[i..]
                .find('(')
                .map(|o| i + o)
                .and_then(|open| balanced_region(rust, open, b'(', b')'))
            {
                if let Some((bs, be)) = rust[pe..]
                    .find('{')
                    .map(|o| pe + o)
                    .and_then(|brace| balanced_region(rust, brace, b'{', b'}'))
                {
                    let body = &rust[bs..be];
                    let new_params = rust[ps..pe]
                        .split(',')
                        .map(|p| {
                            let pt = p.trim();
                            let name = pt.split(':').next().unwrap_or("").trim();
                            if !pt.is_empty()
                                && !pt.starts_with("mut ")
                                && !name.is_empty()
                                && param_is_mutated(body, name)
                            {
                                let lead = &p[..p.len() - p.trim_start().len()];
                                format!("{lead}mut {pt}")
                            } else {
                                p.to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    out.push_str(&rust[i..ps]);
                    out.push_str(&new_params);
                    i = pe;
                    continue;
                }
            }
        }
        let ch = rust[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Rewrite `IDENT[EXPR]` -> `IDENT.chars().nth(EXPR).unwrap()` for each IDENT in `idents` (a set of
/// String-typed bindings). The array pass has already wrapped the index as `(..) as usize`, so EXPR
/// is a valid `.nth` argument. Bounded by a word-boundary check so `IDENT` is not a suffix of a
/// longer identifier.
fn lower_string_index(line: &str, idents: &std::collections::HashSet<String>) -> String {
    let b = line.as_bytes();
    let mut out = String::with_capacity(line.len() + 24);
    let mut i = 0;
    while i < line.len() {
        if b[i] == b'[' {
            let ob = out.as_bytes();
            let mut s = out.len();
            while s > 0 && (ob[s - 1].is_ascii_alphanumeric() || ob[s - 1] == b'_') {
                s -= 1;
            }
            let ident = out[s..].to_string();
            if !ident.is_empty() && idents.contains(&ident) {
                if let Some((_, close)) = balanced_region(line, i, b'[', b']') {
                    let inner = &line[i + 1..close];
                    out.push_str(&format!(".chars().nth({inner}).unwrap()"));
                    i = close + 1;
                    continue;
                }
            }
        }
        let ch = line[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// `(inner_start, close_index)` of the balanced region beginning at `at` (where `s[at] == open`).
fn balanced_region(s: &str, at: usize, open: u8, close: u8) -> Option<(usize, usize)> {
    let b = s.as_bytes();
    if b.get(at) != Some(&open) {
        return None;
    }
    let mut depth = 0i32;
    let mut i = at;
    while i < b.len() {
        if b[i] == open {
            depth += 1;
        } else if b[i] == close {
            depth -= 1;
            if depth == 0 {
                return Some((at + 1, i));
            }
        }
        i += 1;
    }
    None
}

/// Whether `body` mutates the binding `name`: `name = ..` (single `=`, not `==`), `name[..] = ..`,
/// `name += ..` (and `-= *= /= %=`), or an in-place method call `name.push(/.sort(/...`.
pub(crate) fn param_is_mutated(body: &str, name: &str) -> bool {
    let b = body.as_bytes();
    let mut i = 0;
    while let Some(rel) = body[i..].find(name) {
        let at = i + rel;
        let end = at + name.len();
        i = end;
        // word boundary on both sides
        if at > 0 && (b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_') {
            continue;
        }
        if end < b.len() && (b[end].is_ascii_alphanumeric() || b[end] == b'_') {
            continue;
        }
        let mut j = end;
        while j < b.len() && b[j] == b' ' {
            j += 1;
        }
        // skip a balanced index `[..]`
        if j < b.len() && b[j] == b'[' {
            if let Some((_, close)) = balanced_region(body, j, b'[', b']') {
                j = close + 1;
                while j < b.len() && b[j] == b' ' {
                    j += 1;
                }
            }
        }
        if j >= b.len() {
            continue;
        }
        // plain assignment `=` (not `==`)
        if b[j] == b'=' && b.get(j + 1) != Some(&b'=') {
            return true;
        }
        // compound assignment `+= -= *= /= %=`
        if matches!(b[j], b'+' | b'-' | b'*' | b'/' | b'%') && b.get(j + 1) == Some(&b'=') {
            return true;
        }
        // in-place method
        if b[j] == b'.' {
            let rest = &body[j + 1..];
            const MUTATORS: [&str; 11] = [
                "push(", "sort(", "sort_by", "insert(", "remove(", "pop(", "clear(", "reverse(",
                "swap(", "truncate(", "retain(",
            ];
            if MUTATORS.iter().any(|m| rest.starts_with(m)) {
                return true;
            }
        }
    }
    false
}

/// Lower Mog string methods to their Rust equivalents, SCOPED to `String`-returning
/// functions so array code (whose `.reverse()` is a loop, not a method) is never
/// touched. Mog spells string ops `.upper()/.lower()/.trim()/.reverse()`; Rust needs
/// `.to_uppercase()/.to_lowercase()/.trim().to_string()/.chars().rev().collect()`.
/// Without this, synthesized string leaves compile-fail (`no method named upper`).
fn lower_rust_string_methods(rust: &str) -> String {
    let had_trailing_nl = rust.ends_with('\n');
    let mut lines: Vec<String> = Vec::new();
    let mut in_string_fn = false;
    let mut depth: i32 = 0;
    for line in rust.lines() {
        let t = line.trim_start();
        if t.starts_with("fn ") || t.starts_with("pub fn ") {
            in_string_fn = line.contains("-> String") || line.contains("-> string");
        }
        let mapped = if in_string_fn {
            line.replace(".upper()", ".to_uppercase()")
                .replace(".lower()", ".to_lowercase()")
                .replace(".reverse()", ".chars().rev().collect::<String>()")
                .replace(".trim()", ".trim().to_string()")
        } else {
            line.to_string()
        };
        lines.push(mapped);
        for c in line.chars() {
            if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth <= 0 {
                    in_string_fn = false;
                }
            }
        }
    }
    let mut out = lines.join("\n");
    if had_trailing_nl {
        out.push('\n');
    }
    out
}

/// Replace bare `/` with `//` in a Python line. Skips cases where the
/// slash is already part of `//` (unchanged), or embedded in a string
/// literal (best-effort: we avoid rewriting inside quotes). Strings in
/// the synthesizer's output are rare; a conservative scan is enough.
/// Rewrite Mog's C-style logical operators to Python keywords on a body line:
/// `&&` -> `and`, `||` -> `or`, a standalone `!` -> `not ` (but never `!=`).
/// `~` (bit-not) needs no change — Python spells it the same. Replacements skip
/// the interior of string literals so a `"&&"` inside a string is left alone.
fn rewrite_logical_python(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len() + 4);
    let mut i = 0;
    let mut in_str = false;
    while i < chars.len() {
        let c = chars[i];
        if c == '"' {
            in_str = !in_str;
            out.push(c);
            i += 1;
            continue;
        }
        if in_str {
            out.push(c);
            i += 1;
            continue;
        }
        let next = chars.get(i + 1).copied();
        if c == '&' && next == Some('&') {
            out.push_str(" and ");
            i += 2;
        } else if c == '|' && next == Some('|') {
            out.push_str(" or ");
            i += 2;
        } else if c == '!' && next != Some('=') {
            out.push_str("not ");
            i += 1;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

fn rewrite_int_div_python(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len() + 4);
    let mut in_single = false;
    let mut in_double = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            _ => {}
        }
        if !in_single && !in_double && c == b'/' {
            // Already `//` (integer division) or `/=` or part of a comment.
            let next = bytes.get(i + 1).copied();
            let prev = if i > 0 { bytes[i - 1] } else { 0 };
            if next == Some(b'/') || next == Some(b'=') || prev == b'/' {
                out.push(c as char);
                i += 1;
                continue;
            }
            out.push_str("//");
            i += 1;
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// Rewrite `A / B` to `Math.trunc(A / B)` in a TypeScript line so JS
/// float division matches Mog's truncating i64 division. Operands are
/// the synthesizer's atoms: identifiers, integer literals (optionally
/// negative), or balanced parenthesized expressions. Scans left to
/// right; chained divisions fold left-associatively.
/// Rewrite `X.len` (Mog property access) to `len(X)` for Go.
/// Operates on the full body string; the caller applies this to
/// the post-brace body.
fn rewrite_dot_len_go(body: &str) -> String {
    // Match an identifier (letters/digits/underscore) followed by `.len`
    // and rewrite to `len(<id>)`. We don't try to handle chained
    // accesses (e.g. `a.b.len`) because Mog's array types don't have
    // struct nesting in the surfaced syntax.
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len() + 8);
    let mut i = 0;
    while i < bytes.len() {
        if i + 4 <= bytes.len()
            && bytes[i] == b'.'
            && bytes[i + 1] == b'l'
            && bytes[i + 2] == b'e'
            && bytes[i + 3] == b'n'
        {
            // Walk backwards from `i` to find the identifier start.
            let mut start = i;
            while start > 0 {
                let prev = bytes[start - 1];
                if prev.is_ascii_alphanumeric() || prev == b'_' {
                    start -= 1;
                } else {
                    break;
                }
            }
            if start < i {
                // Truncate the output by the identifier length we
                // already pushed, then push the rewrite.
                let ident_len = i - start;
                let new_len = out.len() - ident_len;
                out.truncate(new_len);
                let ident = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
                out.push_str(&format!("len({})", ident));
                i += 4; // skip past `.len`
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Lower Mog's index-style array idioms to valid Rust. Mog is an untyped-index
/// language (every value is `i64`); Rust indices are `usize` and `.len` is a
/// method returning `usize`. Three fixes on a post-body-rewrite line:
///   * `X.len` (bare property) -> `(X.len() as i64)` so it composes with the
///     surrounding i64 arithmetic (`arr.len - 1`, `i < arr.len`);
///   * `IDENT[expr]` (indexing) -> `IDENT[(expr) as usize]`;
///   * an array LITERAL used as a value (`[]`, `[1, 2, 3]`) -> `vec![...]`.
/// The synthesizer emits iterator-style code (`for x in arr`) with none of
/// these tokens, so existing synthesized output is unaffected; this only makes
/// the hand-written index-style library ops transpile to compiling Rust.
fn rewrite_arrays_rust(body: &str) -> String {
    rewrite_index_brackets_rust(&rewrite_dot_len_rust(body))
}

/// `X.len` (property access) -> `(X.len() as i64)`. Skips `.len(` (already a
/// call) and `.len<alnum>` (e.g. `.length`) so only the bare Mog property is
/// rewritten.
fn rewrite_dot_len_rust(body: &str) -> String {
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len() + 16);
    let mut i = 0;
    while i < bytes.len() {
        if i + 4 <= bytes.len()
            && &bytes[i..i + 4] == b".len"
            && (i + 4 == bytes.len()
                || !(bytes[i + 4].is_ascii_alphanumeric()
                    || bytes[i + 4] == b'('
                    || bytes[i + 4] == b'_'))
        {
            let mut start = i;
            while start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_')
            {
                start -= 1;
            }
            if start < i {
                let ident_len = i - start;
                out.truncate(out.len() - ident_len);
                let ident = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
                out.push_str(&format!("({ident}.len() as i64)"));
                i += 4;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// `IDENT[expr]` -> `IDENT[(expr) as usize]`; a bracket NOT preceded by an
/// identifier/`)`/`]` is an array literal, rewritten `[..]` -> `vec![..]`.
fn rewrite_index_brackets_rust(body: &str) -> String {
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len() + 16);
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            // Find the matching close bracket.
            let mut depth = 0i32;
            let mut j = i;
            let mut end = None;
            while j < bytes.len() {
                match bytes[j] {
                    b'[' => depth += 1,
                    b']' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            if let Some(end) = end {
                let inner = &body[i + 1..end];
                let prev = out.trim_end().chars().last();
                let is_index =
                    matches!(prev, Some(c) if c.is_alphanumeric() || c == '_' || c == ')' || c == ']');
                if is_index {
                    if inner.trim().is_empty() || inner.contains("as usize") {
                        out.push('[');
                        out.push_str(inner);
                        out.push(']');
                    } else {
                        out.push_str(&format!("[({}) as usize]", inner.trim()));
                    }
                } else {
                    out.push_str(&format!("vec![{inner}]"));
                }
                i = end + 1;
                continue;
            }
        }
        let ch = body[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Rewrite `var X: T = V;` (Mog `var` declaration) to `var X T = V`
/// (Go `var` declaration). Drops the colon and the trailing
/// semicolon. Operates per-line.
fn rewrite_var_decl_go(line: &str) -> String {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("var ") {
        return line.to_string();
    }
    // Find the `:` separator. The shape is `var NAME: TYPE = VALUE;`.
    let bytes = line.as_bytes();
    // Skip "var " prefix and the identifier.
    let mut i = 4;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return line.to_string();
    }
    // Construct the rewrite: "var NAME TYPE = VALUE;" (drop the colon).
    let prefix = &line[..4]; // "var "
    let ident = &line[4..i];
    let rest = &line[i + 1..];
    format!("{prefix}{ident} {rest}")
}

/// Rewrite Mog's implicit-typed declaration `X: T = V;` to Go's
/// `var X T = V;`. The Mog codegen emits `i: i64 = 1;` (no `var`
/// keyword) when the type is inferred from a later use; for Go we
/// need the explicit type. Operates per-line.
fn rewrite_implicit_typed_decl_go(line: &str) -> String {
    // Skip leading whitespace.
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    if i >= bytes.len() {
        return line.to_string();
    }
    // Must start with an identifier.
    if !bytes[i].is_ascii_alphabetic() && bytes[i] != b'_' {
        return line.to_string();
    }
    let ident_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return line.to_string();
    }
    // Skip the `:` and any whitespace, then expect a type.
    i += 1;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'i' {
        return line.to_string();
    }
    // The Mog codegen uses `i64` for int. We convert it to Go's `int64`.
    if i + 3 > bytes.len() || &bytes[i..i + 3] != b"i64" {
        return line.to_string();
    }
    // Rewrite: insert "var " before the identifier, drop the `:` and
    // the `i64` literal, and add a Go type name in its place.
    let prefix_ws = &line[..ident_start];
    let ident = &line[ident_start..line.find(':').unwrap()];
    let rest = &line[line.find(':').unwrap() + 1..];
    // Skip the type and the `=` and the value; just emit the Go
    // declaration.
    format!("{prefix_ws}var {ident} int{rest}")
}

/// Rewrite Mog's `while COND { BODY }` to Go's `for COND { BODY }`.
/// Operates per-line: matches a `while`-prefixed line and rewrites
/// the keyword.
fn rewrite_while_to_for_go(line: &str) -> String {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("while ") && trimmed != "while" {
        return line.to_string();
    }
    // Replace the first 5 characters ("while") with "for ".
    let leading_ws: String = line.chars().take_while(|c| c.is_whitespace()).collect();
    let rest_start = leading_ws.len() + "while".len();
    let rest = &line[rest_start..];
    format!("{leading_ws}for {rest}")
}

/// Rewrite Mog's `X.sort()` (array method call) to Go's
/// `sort.Ints(X)`. The wrapper adds the `import "sort"` line if it
/// sees `sort.Ints` in the body. We assume the array is `[]int`
/// (the nsynth transpile's chosen Go type for `i64`).
fn rewrite_dot_sort_go(body: &str) -> String {
    // Find every `IDENT.sort()` (where IDENT is an identifier) and
    // replace it with `sort.Ints(IDENT)`. The two-pass approach
    // (find the call, then emit the rewrite) is needed because we
    // have to truncate the output to drop the `.sort()` suffix
    // without losing the identifier.
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len() + 16);
    let mut i = 0;
    while i < bytes.len() {
        // Look for `.sort()`.
        if i + 7 <= bytes.len()
            && bytes[i] == b'.'
            && bytes[i + 1] == b's'
            && bytes[i + 2] == b'o'
            && bytes[i + 3] == b'r'
            && bytes[i + 4] == b't'
            && bytes[i + 5] == b'('
            && bytes[i + 6] == b')'
        {
            // Walk backwards to find the identifier start.
            let mut start = i;
            while start > 0 {
                let prev = bytes[start - 1];
                if prev.is_ascii_alphanumeric() || prev == b'_' {
                    start -= 1;
                } else {
                    break;
                }
            }
            if start < i {
                let ident_len = i - start;
                let new_len = out.len() - ident_len;
                out.truncate(new_len);
                let ident = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
                out.push_str(&format!("sort.Ints({})", ident));
                i += 7; // skip past `.sort()`
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn rewrite_int_div_typescript(line: &str) -> String {
    let mut s = line.to_string();
    // Cursor past which we look for the next unwrapped `/`. After each
    // wrap it is moved past the wrapped group's closing paren so the `/`
    // inside `Math.trunc(A / B)` is never re-wrapped; an outer chained
    // `/` to its right still sees the whole wrapped group as its left
    // operand (callee name included), so `a / b / c` folds correctly to
    // `Math.trunc(Math.trunc(a / b) / c)`.
    let mut search_from = 0usize;
    loop {
        let bytes = s.as_bytes();
        // Next `/` not part of `//` or `/=`.
        let mut slash = None;
        let mut i = search_from;
        while i < bytes.len() {
            if bytes[i] == b'/' {
                let next = bytes.get(i + 1).copied();
                let prev = if i > 0 { bytes[i - 1] } else { 0 };
                if next != Some(b'/') && next != Some(b'=') && prev != b'/' {
                    slash = Some(i);
                    break;
                }
            }
            i += 1;
        }
        let Some(slash) = slash else { return s };

        // Left operand: walk back over whitespace, then either a balanced
        // `(...)` group (plus any callee name) or an identifier/number run.
        let mut l_end = slash;
        while l_end > 0 && bytes[l_end - 1] == b' ' {
            l_end -= 1;
        }
        let mut l_start = l_end;
        if l_end > 0 && bytes[l_end - 1] == b')' {
            let mut depth = 0i32;
            let mut j = l_end;
            while j > 0 {
                j -= 1;
                match bytes[j] {
                    b')' => depth += 1,
                    b'(' => {
                        depth -= 1;
                        if depth == 0 {
                            l_start = j;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            while l_start > 0
                && (bytes[l_start - 1].is_ascii_alphanumeric()
                    || bytes[l_start - 1] == b'_'
                    || bytes[l_start - 1] == b'.')
            {
                l_start -= 1;
            }
        } else {
            while l_start > 0
                && (bytes[l_start - 1].is_ascii_alphanumeric()
                    || bytes[l_start - 1] == b'_'
                    || bytes[l_start - 1] == b'.')
            {
                l_start -= 1;
            }
            // Unary minus: only when preceded by an operator or open paren.
            if l_start > 0 && bytes[l_start - 1] == b'-' {
                let mut k = l_start - 1;
                while k > 0 && bytes[k - 1] == b' ' {
                    k -= 1;
                }
                if k == 0
                    || matches!(
                        bytes[k - 1],
                        b'(' | b'+' | b'-' | b'*' | b'%' | b'=' | b',' | b'<' | b'>'
                    )
                {
                    l_start -= 1;
                }
            }
        }
        if l_start == l_end {
            // Malformed/unexpected shape — leave this slash alone.
            search_from = slash + 1;
            continue;
        }

        // Right operand: whitespace, then a balanced group or an
        // identifier/number run (optionally negative).
        let mut r_start = slash + 1;
        while r_start < bytes.len() && bytes[r_start] == b' ' {
            r_start += 1;
        }
        let mut r_end = r_start;
        if r_end < bytes.len() && bytes[r_end] == b'-' {
            r_end += 1;
        }
        if r_end < bytes.len() && bytes[r_end] == b'(' {
            let mut depth = 0i32;
            while r_end < bytes.len() {
                match bytes[r_end] {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            r_end += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                r_end += 1;
            }
        } else {
            while r_end < bytes.len()
                && (bytes[r_end].is_ascii_alphanumeric()
                    || bytes[r_end] == b'_'
                    || bytes[r_end] == b'.')
            {
                r_end += 1;
            }
        }
        if r_start == r_end {
            search_from = slash + 1;
            continue;
        }

        let wrapped_len =
            "Math.trunc(".len() + (l_end - l_start) + " / ".len() + (r_end - r_start) + ")".len();
        s = format!(
            "{}Math.trunc({} / {}){}",
            &s[..l_start],
            &s[l_start..l_end],
            &s[r_start..r_end],
            &s[r_end..]
        );
        search_from = l_start + wrapped_len;
    }
}

fn indent_unit(target: Target) -> &'static str {
    match target {
        Target::Python | Target::TypeScript | Target::Rust => "    ",
        Target::Go => "\t",
        Target::Java => "    ",
    }
}

/// Rewrite the `fn NAME(args) -> RET {` header. Returns `Some(new_line)`
/// when the rewrite succeeds, `None` when the header doesn't match the
/// expected shape (caller falls through to generic body handling).
fn rewrite_fn_header(line: &str, target: Target) -> Option<String> {
    // Parse: `fn NAME(args) -> RET {`
    let body = line.strip_prefix("fn ")?.trim();
    let (name, rest) = body.split_once('(')?;
    let (args, rest) = rest.split_once(')')?;
    let name = name.trim();
    let args = args.trim();
    // Optional return type; if not present, body already contains `{`.
    let (ret_type, tail) = if let Some(stripped) = rest.trim_start().strip_prefix("->") {
        let stripped = stripped.trim();
        let (ret, tail) = stripped.split_once('{').unwrap_or((stripped, ""));
        (ret.trim(), tail.trim())
    } else {
        ("", rest.trim().trim_start_matches('{').trim())
    };
    let _ = tail;
    Some(match target {
        Target::Python => format!("def {}({}):", name, rewrite_args_python(args)),
        Target::Rust => format!(
            "fn {}({}) -> {} {{",
            name,
            rewrite_args_rust(args),
            rewrite_type_rust(ret_type)
        ),
        Target::TypeScript => format!(
            "function {}({}): {} {{",
            name,
            rewrite_args_ts(args),
            rewrite_type_ts(ret_type)
        ),
        Target::Go => format!(
            "func {}({}) {} {{",
            name,
            rewrite_args_go(args),
            rewrite_type_go(ret_type)
        ),
        Target::Java => format!(
            "public static {} {}({}) {{",
            rewrite_type_java(ret_type),
            name,
            rewrite_args_java(args),
        ),
    })
}

fn rewrite_args_python(args: &str) -> String {
    // Python: `a: i64, b: i64` → `a: int, b: int`. Drop types entirely if
    // you want even looser code, but type hints are cheap and useful.
    args.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|arg| {
            if let Some((n, t)) = arg.split_once(':') {
                format!("{}: {}", n.trim(), rewrite_type_python(t.trim()))
            } else {
                arg.trim().to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn rewrite_args_rust(args: &str) -> String {
    args.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|arg| {
            if let Some((n, t)) = arg.split_once(':') {
                format!("{}: {}", n.trim(), rewrite_type_rust(t.trim()))
            } else {
                arg.trim().to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn rewrite_args_ts(args: &str) -> String {
    args.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|arg| {
            if let Some((n, t)) = arg.split_once(':') {
                format!("{}: {}", n.trim(), rewrite_type_ts(t.trim()))
            } else {
                arg.trim().to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn rewrite_type_python(t: &str) -> String {
    match t {
        "i64" => "int".into(),
        "[i64]" => "list[int]".into(),
        "string" => "str".into(),
        other => other.to_string(),
    }
}

fn rewrite_type_rust(t: &str) -> String {
    let t = t.trim();
    match t {
        "i64" | "bool" | "f64" => t.to_string(),
        "string" => "String".into(),
        // RECURSIVE array lowering: `[T]` -> `Vec<rewrite(T)>`, so nested `[[i64]]` (pair lists /
        // maps-as-pair-lists) becomes `Vec<Vec<i64>>` and `[string]` becomes `Vec<String>`. A flat
        // match only handled `[i64]`, leaving every map/nested op as uncompilable `[[i64]]`.
        _ => {
            if let Some(inner) = t.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                format!("Vec<{}>", rewrite_type_rust(inner))
            } else {
                t.to_string()
            }
        }
    }
}

fn rewrite_type_ts(t: &str) -> String {
    match t {
        "i64" => "number".into(),
        "[i64]" => "number[]".into(),
        "string" => "string".into(),
        other => other.to_string(),
    }
}

fn rewrite_args_go(args: &str) -> String {
    // Go: `a: i64, b: i64` → `a int64, b int64`. Type names live in
    // `rewrite_type_go`. Point/range arguments keep their original Mog
    // type spelling so a `(p: Point)` signature round-trips intact
    // for users that need it; the user can then define a `Point` struct
    // alongside the generated function.
    args.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|arg| {
            if let Some((n, t)) = arg.split_once(':') {
                format!("{} {}", n.trim(), rewrite_type_go(t.trim()))
            } else {
                arg.trim().to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn rewrite_type_go(t: &str) -> String {
    match t {
        "i64" => "int".into(),
        "[i64]" => "[]int".into(),
        "string" => "string".into(),
        "bool" => "bool".into(),
        other => other.to_string(),
    }
}

fn rewrite_args_java(args: &str) -> String {
    // Java: `a: i64, b: i64` → `long a, long b`. Same as Go with `long`
    // instead of `int64`. Point/Rectangle structs keep their original
    // Mog spelling for the user to declare alongside.
    args.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|arg| {
            if let Some((n, t)) = arg.split_once(':') {
                format!("{} {}", rewrite_type_java(t.trim()), n.trim())
            } else {
                arg.trim().to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn rewrite_type_java(t: &str) -> String {
    match t {
        "i64" => "long".into(),
        "[i64]" => "long[]".into(),
        "string" => "String".into(),
        "bool" => "boolean".into(),
        other => other.to_string(),
    }
}

/// Body-line rewrites shared across all three targets:
///   - `VAR: TYPE = EXPR;` → declaration in the target's syntax
///   - `return EXPR;` → target-specific return
///   - `VAR = EXPR;` → assignment
///   - bare expression lines stay as-is (modulo semicolon handling)
fn rewrite_body_line(line: &str, target: Target) -> String {
    let line = line.trim_end_matches(';');

    // Pattern: `VAR: TYPE = EXPR`. Matches declarations like
    // `acc: i64 = 0`.
    if let Some((lhs, rhs)) = line.split_once('=') {
        let lhs = lhs.trim();
        let rhs = rhs.trim();
        if let Some((var, ty)) = lhs.split_once(':') {
            let var = var.trim();
            let ty = ty.trim();
            return match target {
                Target::Python => format!("{} = {}", var, rhs),
                Target::Rust => format!("let mut {}: {} = {};", var, rewrite_type_rust(ty), rhs),
                Target::TypeScript => {
                    format!("let {}: {} = {};", var, rewrite_type_ts(ty), rhs)
                }
                Target::Go => format!("var {} {} = {}", var, rewrite_type_go(ty), rhs),
                Target::Java => format!("{} {} = {};", rewrite_type_java(ty), var, rhs),
            };
        }
        // Plain assignment (no type): same var on both sides means
        // reassignment; we emit language-appropriate syntax.
        if !lhs.contains(' ') && !lhs.contains('(') {
            return match target {
                Target::Python => format!("{} = {}", lhs, rhs),
                Target::Rust | Target::TypeScript | Target::Go | Target::Java => {
                    format!("{} = {};", lhs, rhs)
                }
            };
        }
    }

    // `return EXPR`.
    if let Some(expr) = line.strip_prefix("return ") {
        return match target {
            Target::Python => format!("return {}", expr.trim()),
            Target::Rust | Target::TypeScript | Target::Go | Target::Java => {
                format!("return {};", expr.trim())
            }
        };
    }

    // Block headers (`if`, `while`) — Rust/Python accept the bare form;
    // TypeScript requires parens around the condition.
    if line.starts_with("if ") {
        let cond_plus_brace = line.strip_prefix("if ").unwrap_or(line);
        // Inline single-line form `if COND { STMT; }` (the gradient backend
        // emits these). The previous logic treated the whole tail as the
        // condition, producing `if (COND { STMT; })`. Split at the first
        // `{` instead and rewrite the inner statement(s) recursively.
        if let Some(brace) = cond_plus_brace.find('{') {
            if cond_plus_brace.trim_end().ends_with('}') {
                let cond = cond_plus_brace[..brace].trim();
                let inner = cond_plus_brace[brace + 1..]
                    .trim_end()
                    .trim_end_matches('}')
                    .trim();
                let stmts: Vec<String> = inner
                    .split(';')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| rewrite_body_line(s, target))
                    .collect();
                return match target {
                    Target::Python => format!("if {}: {}", cond, stmts.join("; ")),
                    Target::Rust => format!("if {} {{ {} }}", cond, stmts.join(" ")),
                    Target::TypeScript => format!("if ({}) {{ {} }}", cond, stmts.join(" ")),
                    Target::Go => format!("if {} {{ {} }}", cond, stmts.join(" ")),
                    Target::Java => format!("if ({}) {{ {} }}", cond, stmts.join(" ")),
                };
            }
        }
        let cond = cond_plus_brace.trim_end_matches('{').trim();
        return match target {
            Target::Python | Target::Rust | Target::Go => line.to_string(),
            Target::TypeScript | Target::Java => format!("if ({}) {{", cond),
        };
    }
    if line.starts_with("while ") {
        let cond_plus_brace = line.strip_prefix("while ").unwrap_or(line);
        let cond = cond_plus_brace.trim_end_matches('{').trim();
        return match target {
            Target::Python | Target::Rust | Target::Go => line.to_string(),
            Target::TypeScript | Target::Java => format!("while ({}) {{", cond),
        };
    }
    if line == "else" || line.starts_with("else ") {
        return line.to_string();
    }

    // `for VAR in EXPR { ... }` requires a per-target rewrite because
    // TypeScript needs `for (const VAR of EXPR) {` instead of the Mog-
    // native shape. Python + Rust already accept `for VAR in EXPR`.
    // Go's analog is `for _, VAR := range EXPR {` — we translate the
    // Mog form into that here so the output compiles.
    if let Some(rest) = line.strip_prefix("for ") {
        // `VAR in EXPR [{]` — the trailing `{` is stripped by the caller
        // when it detects opens_block. Split off the `in` keyword.
        let trimmed = rest.trim_end_matches('{').trim();
        if let Some((var, iter_expr)) = trimmed.split_once(" in ") {
            let var = var.trim();
            let iter_expr = iter_expr.trim();
            return match target {
                Target::Python | Target::Rust => format!("for {} in {} {{", var, iter_expr),
                Target::TypeScript => format!("for (const {} of {}) {{", var, iter_expr),
                Target::Go => format!("for _, {} := range {} {{", var, iter_expr),
                // Java: enhanced-for uses `for (TYPE var : iterable) { ... }`.
                // The Mog element type defaults to `long` for `i64` arrays;
                // for struct iterables the user declares a wrapper class.
                // We emit `long` here; if the user iterates a `long[]` this
                // is correct, and a `Point[]` case is out of scope.
                Target::Java => format!("for (long {} : (long[]) {}) {{", var, iter_expr),
            };
        }
        // Fallback: pass through.
        return line.to_string();
    }

    // Fallback: emit the line with a target-appropriate terminator.
    match target {
        Target::Python => line.to_string(),
        Target::Rust | Target::TypeScript | Target::Go | Target::Java => format!("{};", line),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative synthesized Mog program: array sum with a
    /// negative-filter. Exercises function header, typed declaration,
    /// for-loop, if-branch, accumulation, and return.
    const SAMPLE: &str = "fn sum_negatives(arr: [i64]) -> i64 {\n\
        acc: i64 = 0;\n\
        for item in arr {\n\
            if item < 0 {\n\
                acc = acc + item;\n\
            }\n\
        }\n\
        return acc;\n\
        }\n";

    #[test]
    fn python_logical_operators_rewrite() {
        // C-style `&&`/`||`/`!` become Python `and`/`or`/`not`; `~` is unchanged.
        let mog = "fn f(x: i64) -> i64 {\n\
            if x > 0 && x < 10 {\n\
            return 1;\n\
            }\n\
            if x < 0 || !(x == 5) {\n\
            return ~x;\n\
            }\n\
            return 0;\n\
            }\n";
        let out = to_python(mog);
        assert!(out.contains("x > 0  and  x < 10"), "&& -> and: {out}");
        assert!(out.contains(" or "), "|| -> or: {out}");
        assert!(out.contains("not (x == 5)"), "! -> not: {out}");
        assert!(out.contains("~x"), "~ unchanged: {out}");
        assert!(
            !out.contains("&&") && !out.contains("||"),
            "no C ops left: {out}"
        );
    }

    #[test]
    fn python_transpile_shape() {
        let out = to_python(SAMPLE);
        assert!(out.contains("def sum_negatives(arr: list[int]):"));
        assert!(out.contains("acc = 0"));
        assert!(out.contains("for item in arr:"));
        assert!(out.contains("if item < 0:"));
        assert!(out.contains("acc = acc + item"));
        assert!(out.contains("return acc"));
        // Python has no `{` / `}` in the output.
        assert!(!out.contains('{'));
        assert!(!out.contains('}'));
    }

    #[test]
    fn rust_transpile_shape() {
        let out = to_rust(SAMPLE);
        assert!(out.contains("fn sum_negatives(arr: Vec<i64>) -> i64 {"));
        assert!(out.contains("let mut acc: i64 = 0;"));
        // Vec iteration borrows + clones each element so the collection is not moved (E0382).
        assert!(out.contains("for item in arr.iter().cloned() {"));
        assert!(out.contains("if item < 0 {"));
        assert!(out.contains("return acc;"));
        assert!(out.ends_with("}\n"));
    }

    #[test]
    fn typescript_int_div_truncates() {
        // Plain division in an i64-returning function must become
        // Math.trunc so JS float division matches Mog semantics.
        let mog = "fn hit_bottom(a: i64, b: i64) -> i64 {\n\
            return (a / b);\n\
            }\n";
        let out = to_typescript(mog);
        assert!(
            out.contains("return (Math.trunc(a / b));"),
            "expected Math.trunc wrap, got {out}"
        );
        // Unparenthesized form too (e.g. flag_or's `return b / 1;`).
        let mog2 = "fn f(b: i64) -> i64 {\n    return b / 1;\n}\n";
        let out2 = to_typescript(mog2);
        assert!(
            out2.contains("return Math.trunc(b / 1);"),
            "expected Math.trunc wrap, got {out2}"
        );
        // Chained division folds left-associatively.
        let mog3 = "fn g(a: i64, b: i64, c: i64) -> i64 {\n    return a / b / c;\n}\n";
        let out3 = to_typescript(mog3);
        assert!(
            out3.contains("return Math.trunc(Math.trunc(a / b) / c);"),
            "expected nested Math.trunc, got {out3}"
        );
        // Nested parenthesized division.
        let mog4 = "fn h(a: i64, b: i64, c: i64) -> i64 {\n    return ((a / b) / c);\n}\n";
        let out4 = to_typescript(mog4);
        assert!(
            out4.contains("return ((Math.trunc((Math.trunc(a / b)) / c)));")
                || out4.contains("Math.trunc((Math.trunc(a / b)) / c)"),
            "expected nested Math.trunc, got {out4}"
        );
        // Division mixed with other ops keeps surrounding expression intact.
        let mog5 = "fn k(a: i64, b: i64) -> i64 {\n    return (a / 2) + b;\n}\n";
        let out5 = to_typescript(mog5);
        assert!(
            out5.contains("return (Math.trunc(a / 2)) + b;")
                || out5.contains("return Math.trunc(a / 2) + b;"),
            "expected trunc on div only, got {out5}"
        );
    }

    #[test]
    fn typescript_transpile_shape() {
        let out = to_typescript(SAMPLE);
        assert!(out.contains("function sum_negatives(arr: number[]): number {"));
        assert!(out.contains("let acc: number = 0;"));
        assert!(
            out.contains("for (const item of arr) {"),
            "TypeScript for-loop must use `for (const X of Y)` shape, got {out}"
        );
        assert!(
            out.contains("if (item < 0) {"),
            "TypeScript if must have parens, got {out}"
        );
        assert!(out.contains("return acc;"));
    }

    #[test]
    fn else_branch_transpiles_cleanly() {
        // Synthesizer emits `} else {` on one continuation line. Python
        // must produce `else:` at the outer indent; Rust/TS must keep the
        // braced form.
        let mog = "fn f(x: i64) -> i64 {\n\
            if x > 0 {\n\
                return 1;\n\
            } else {\n\
                return 0;\n\
            }\n\
            }\n";
        let py = to_python(mog);
        assert!(py.contains("else:"), "Python must emit `else:`, got: {py}");
        assert!(
            !py.contains("} else:"),
            "Python must not keep the close-brace: {py}"
        );
        assert!(!py.contains('{'), "Python must have no braces: {py}");

        let rs = to_rust(mog);
        assert!(rs.contains("} else {"), "Rust keeps braced else: {rs}");

        let ts = to_typescript(mog);
        assert!(ts.contains("} else {"), "TS keeps braced else: {ts}");
    }

    /// Integer-returning Python functions must use `//` for division so
    /// Python 3 doesn't silently float-divide. Rust/TS are unaffected —
    /// `/` on integers is already integer division in both.
    #[test]
    fn python_integer_division_rewrite() {
        let mog = "fn average_two(a: i64, b: i64) -> i64 {\n    return (a + b) / 2;\n}\n";
        let py = to_python(mog);
        assert!(
            py.contains("// 2"),
            "Python must rewrite `/` → `//` for integer-returning fn: {py}"
        );
        // Verify the generated Python actually computes integer division.
        // `(4 + 6) // 2 == 5`; with float division it would be `5.0`.
        assert!(
            !py.contains(" / "),
            "no bare `/` should survive in Python int output: {py}"
        );
    }

    #[test]
    fn simple_scalar_fn_all_targets() {
        let mog = "fn double(a: i64) -> i64 {\n    return a * 2;\n}\n";
        assert!(to_python(mog).contains("def double(a: int):"));
        assert!(to_python(mog).contains("return a * 2"));
        assert!(to_rust(mog).contains("fn double(a: i64) -> i64"));
        assert!(to_rust(mog).contains("return a * 2;"));
        assert!(to_typescript(mog).contains("function double(a: number): number"));
    }
}

#[cfg(test)]
mod inline_if_tests {
    use super::*;

    const MOG: &str = "fn min2(a: i64, b: i64) -> i64 {\n    v0: i64 = b;\n    if b >= a { v0 = a; }\n    return v0;\n}\n";

    #[test]
    fn inline_if_typescript() {
        let ts = to_typescript(MOG);
        assert!(ts.contains("if (b >= a) { v0 = a; }"), "got: {ts}");
        assert!(
            !ts.contains("if (b >= a {"),
            "broken paren wrap survived: {ts}"
        );
    }

    #[test]
    fn inline_if_python() {
        let py = to_python(MOG);
        assert!(py.contains("if b >= a: v0 = a"), "got: {py}");
    }

    #[test]
    fn inline_if_rust() {
        let rs = to_rust(MOG);
        assert!(rs.contains("if b >= a { v0 = a; }"), "got: {rs}");
    }

    #[test]
    fn rust_adds_mut_to_index_assigned_param() {
        // In-place bubble sort mutates its array parameter via `a[j] = ..` — the emitted Rust
        // param must be `mut a` or it fails to compile (`cannot borrow as mutable`).
        let mog = "fn srt(a: [i64]) -> [i64] {\n    i: i64 = 0;\n    while i < a.len {\n        a[i] = a[i];\n        i = i + 1;\n    }\n    return a;\n}\n";
        let rs = to_rust(mog);
        assert!(rs.contains("fn srt(mut a: Vec<i64>)"), "param not made mut: {rs}");
    }

    #[test]
    fn rust_lowers_string_char_iteration_and_predicates() {
        // count_vowels returns i64 (outside the `-> String` string-method scope) and iterates a
        // string char-by-char with `.is_vowel()` — both must be lowered or it fails to compile.
        let mog = "fn count_vowels(s: string) -> i64 {\n    c: i64 = 0;\n    for ch in s {\n        if ch.is_vowel() {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n";
        let rs = to_rust(mog);
        assert!(rs.contains("for ch in s.chars()"), "string not iterated by chars: {rs}");
        assert!(rs.contains("\"aeiouAEIOU\".contains(ch)"), "is_vowel not lowered: {rs}");
        assert!(!rs.contains(".is_vowel()"), "raw .is_vowel() survived: {rs}");
        // An ARRAY loop `for e in arr` must NOT get `.chars()`.
        let arr = "fn sum(arr: [i64]) -> i64 {\n    t: i64 = 0;\n    for e in arr {\n        t = t + e;\n    }\n    return t;\n}\n";
        assert!(!to_rust(arr).contains(".chars()"), "array loop wrongly got .chars()");
    }

    #[test]
    fn rust_lowers_string_building_concat_and_literals() {
        // A char-by-char string builder: `out = ""`, `out = out + ch`, `+ ch.upper()`, and a bare
        // literal return — all invalid Rust as emitted, all lowered here.
        let mog = "fn cap(s: string) -> string {\n    out: string = \"\";\n    for ch in s {\n        out = out + ch.upper();\n    }\n    if s == \"\" {\n        return \"empty\";\n    }\n    return out;\n}\n";
        let rs = to_rust(mog);
        assert!(rs.contains("out: String = String::new();"), "empty init not lowered: {rs}");
        assert!(rs.contains("out = format!(\"{}{}\", out, ch.to_uppercase());"), "concat not lowered: {rs}");
        assert!(rs.contains("return \"empty\".to_string();"), "literal return not lowered: {rs}");
        assert!(!rs.contains("out + ch"), "raw String+char survived: {rs}");
        // Integer accumulation `c = c + 1` must NOT be turned into a format!.
        let ints = "fn cnt(s: string) -> i64 {\n    c: i64 = 0;\n    for ch in s {\n        c = c + 1;\n    }\n    return c;\n}\n";
        assert!(!to_rust(ints).contains("format!"), "int accumulation wrongly formatted");
    }

    #[test]
    fn rust_lowers_case_predicates_and_whole_string_reverse_in_any_fn() {
        // swap-case: `ch.is_upper()/.is_lower()` char predicates.
        let sc = "fn f(s: string) -> string {\n    out: string = \"\";\n    for ch in s {\n        if ch.is_upper() {\n            out = out + ch;\n        }\n    }\n    return out;\n}\n";
        let rs = to_rust(sc);
        assert!(rs.contains("ch.is_uppercase()"), "is_upper not lowered: {rs}");
        assert!(!rs.contains(".is_upper()"), "raw .is_upper() survived: {rs}");
        // palindrome: `s.reverse()` on a String inside a NON-String fn (returns i64/bool).
        let pal = "fn f(s: string) -> i64 {\n    if s == s.reverse() {\n        return 1;\n    }\n    return 0;\n}\n";
        let rp = to_rust(pal);
        assert!(rp.contains("s.chars().rev().collect::<String>()"), "s.reverse() not lowered: {rp}");
        assert!(!rp.contains("s.reverse()"), "raw s.reverse() survived: {rp}");
    }

    #[test]
    fn rust_borrows_vec_in_for_loops_and_lowers_char_methods_globally() {
        // A Vec iterated then reused (nested loop) would `use of moved value`; iterate a borrow.
        let dedup = "fn f(arr: [i64]) -> i64 {\n    seen: [i64] = [];\n    for e in arr {\n        for u in seen {\n            if u == e {\n            }\n        }\n        seen.push(e);\n    }\n    return seen.len;\n}\n";
        let rs = to_rust(dedup);
        assert!(rs.contains("for e in arr.iter().cloned()"), "outer not borrowed: {rs}");
        assert!(rs.contains("for u in seen.iter().cloned()"), "inner (reused) not borrowed: {rs}");
        // char methods lower even on a char from INDEXING, not just a `for ch in s` loop var.
        let idx = "fn g(s: string) -> i64 {\n    if s[0].is_upper() {\n        return s[1].ord();\n    }\n    return 0;\n}\n";
        let rg = to_rust(idx);
        assert!(rg.contains(".is_uppercase()") && !rg.contains(".is_upper()"), "is_upper: {rg}");
        assert!(rg.contains(" as i64") && !rg.contains(".ord()"), "ord not cast: {rg}");
    }

    #[test]
    fn rust_materializes_string_split_into_vec_string() {
        let mog = "fn split_words(s: string) -> [string] {\n    return s.split(\" \");\n}\n";
        let rs = to_rust(mog);
        assert!(
            rs.contains("s.split(\" \").map(|w| w.to_string()).collect::<Vec<String>>()"),
            "split not materialized: {rs}"
        );
    }

    #[test]
    fn rust_infers_vec_char_element_type_from_char_pushes() {
        // A Mog `[i64]` list that actually accumulates chars must become `Vec<char>`, else the
        // `push(ch)` / `keys[i] == ch` type-mismatch.
        let mog = "fn f(s: string) -> i64 {\n    seen: [i64] = [];\n    for ch in s {\n        seen.push(ch);\n    }\n    return seen.len;\n}\n";
        let rs = to_rust(mog);
        assert!(rs.contains("seen: Vec<char>"), "char-holding vec not retyped: {rs}");
        // A parallel i64 list in the same fn stays Vec<i64>.
        let mog2 = "fn g(s: string) -> i64 {\n    keys: [i64] = [];\n    counts: [i64] = [];\n    for ch in s {\n        keys.push(ch);\n        counts.push(1);\n    }\n    return counts.len;\n}\n";
        let rs2 = to_rust(mog2);
        assert!(rs2.contains("keys: Vec<char>"), "keys not char: {rs2}");
        assert!(rs2.contains("counts: Vec<i64>"), "counts wrongly retyped: {rs2}");
    }

    #[test]
    fn rust_type_lowering_is_recursive_for_nested_arrays() {
        // pair-list / map ops carry `[[i64]]`; a flat match left it uncompilable. Vec must nest.
        let mog = "fn map_keys(pairs: [[i64]]) -> [i64] {\n    out: [i64] = [];\n    for p in pairs {\n        out.push(p[0]);\n    }\n    return out;\n}\n";
        let rs = to_rust(mog);
        assert!(rs.contains("pairs: Vec<Vec<i64>>"), "nested array not lowered: {rs}");
        assert!(rs.contains("-> Vec<i64>"), "return array not lowered: {rs}");
        // array-of-string too
        let ms = "fn f(ws: [string]) -> i64 {\n    return ws.len;\n}\n";
        assert!(to_rust(ms).contains("ws: Vec<String>"), "[string] not lowered");
    }

    #[test]
    fn rust_lowers_string_indexing_to_chars_nth() {
        // first == last: `s[0] == s[s.len - 1]` on a String needs char lookups, not `s[usize]`.
        let mog = "fn f(s: string) -> i64 {\n    if s[0] == s[s.len - 1] {\n        return 1;\n    }\n    return 0;\n}\n";
        let rs = to_rust(mog);
        assert!(rs.contains("s.chars().nth((0) as usize).unwrap()"), "s[0] not lowered: {rs}");
        assert!(rs.contains(".chars().nth(") && rs.contains(".unwrap()"), "index not lowered: {rs}");
        assert!(!rs.contains("s[("), "raw String index survived: {rs}");
        // ARRAY indexing must stay `arr[(i) as usize]` (Vec IS Index<usize>).
        let arr = "fn g(arr: [i64]) -> i64 {\n    return arr[0];\n}\n";
        let rg = to_rust(arr);
        assert!(rg.contains("arr[(0) as usize]"), "array index wrongly changed: {rg}");
        assert!(!rg.contains(".chars()"), "array index got .chars(): {rg}");
    }

    #[test]
    fn rust_leaves_unmutated_and_iterated_params_immutable() {
        // `double_each` iterates `arr` (moved) and builds a fresh `out`; `arr` is NOT mutated and
        // must stay immutable (a spurious `mut` would be wrong and warn).
        let mog = "fn double_each(arr: [i64]) -> [i64] {\n    out: [i64] = [];\n    for e in arr {\n        out.push(e * 2);\n    }\n    return out;\n}\n";
        let rs = to_rust(mog);
        assert!(rs.contains("fn double_each(arr: Vec<i64>)"), "arr wrongly made mut: {rs}");
        assert!(!rs.contains("mut arr"), "arr wrongly made mut: {rs}");
    }

    #[test]
    fn inline_if_go() {
        let go = to_go(MOG);
        assert!(go.contains("if b >= a { v0 = a; }"), "got: {go}");
    }

    #[test]
    fn go_emits_func_int64_array() {
        let mog = "fn sum_array(arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for x in arr {\n        s = s + x;\n    }\n    return s;\n}\n";
        let go = to_go(mog);
        assert!(go.contains("func sum_array(arr []int) int"), "got: {go}");
        assert!(go.contains("var s int = 0"), "got: {go}");
        assert!(go.contains("for _, x := range arr {"), "got: {go}");
        assert!(go.contains("return s;"), "got: {go}");
    }

    #[test]
    fn inline_if_java() {
        let ja = to_java(MOG);
        assert!(ja.contains("if (b >= a) { v0 = a; }"), "got: {ja}");
    }

    #[test]
    fn java_emits_static_long_array_enhanced_for() {
        let mog = "fn sum_array(arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for x in arr {\n        s = s + x;\n    }\n    return s;\n}\n";
        let ja = to_java(mog);
        assert!(
            ja.contains("public static long sum_array(long[] arr)"),
            "got: {ja}"
        );
        assert!(ja.contains("long s = 0;"), "got: {ja}");
        assert!(ja.contains("for (long x : (long[]) arr) {"), "got: {ja}");
        assert!(ja.contains("return s;"), "got: {ja}");
    }
}
