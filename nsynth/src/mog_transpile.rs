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
/// is Go's `int64` (matches Mog's `i64`) and arrays are `[]int64`.
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
    match t {
        "i64" => "i64".into(),
        "[i64]" => "Vec<i64>".into(),
        "string" => "String".into(),
        other => other.to_string(),
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
        "i64" => "int64".into(),
        "[i64]" => "[]int64".into(),
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
        assert!(out.contains("for item in arr {"));
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
    fn inline_if_go() {
        let go = to_go(MOG);
        assert!(go.contains("if b >= a { v0 = a; }"), "got: {go}");
    }

    #[test]
    fn go_emits_func_int64_array() {
        let mog = "fn sum_array(arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for x in arr {\n        s = s + x;\n    }\n    return s;\n}\n";
        let go = to_go(mog);
        assert!(go.contains("func sum_array(arr []int64) int64"), "got: {go}");
        assert!(go.contains("var s int64 = 0"), "got: {go}");
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
        assert!(
            ja.contains("for (long x : (long[]) arr) {"),
            "got: {ja}"
        );
        assert!(ja.contains("return s;"), "got: {ja}");
    }
}
