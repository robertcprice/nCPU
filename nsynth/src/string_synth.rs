//! General string-program synthesizer.
//!
//! Makes nSynth fully capable of string-driven programs: given input→output
//! string examples (1+ string args), it bottom-up enumerates a string-expression
//! grammar and returns a verified Mog program. This is the string analog of the
//! scalar enumerative synthesizer — FlashFill-style, deduped by evaluation
//! signature — and covers reverse, case transforms, slicing, affixing, literal
//! replacement, field concatenation, initials, and their compositions.
//!
//! Output strings live entirely in this additive path (the numeric `Example`
//! pipeline is untouched); `morph_transduce` is the fast specialist for
//! suffix-conditioned morphology, this is the general fallback.

use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct StrSynthExample {
    pub inputs: Vec<String>,
    pub expected: String,
}

#[derive(Clone, Debug)]
pub struct StrSynthResult {
    pub success: bool,
    pub code: String,
    pub method: String,
    pub error: Option<String>,
}

// ── Expression grammar ───────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, Hash)]
enum IExpr {
    Const(i64),
    Len(Box<SExpr>, i64), // E.len - k  (k >= 0; k == 0 is E.len)
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum SExpr {
    Var(usize),
    Lit(String),
    Upper(Box<SExpr>),
    Lower(Box<SExpr>),
    Trim(Box<SExpr>),
    Reverse(Box<SExpr>),
    Concat(Box<SExpr>, Box<SExpr>),
    Slice(Box<SExpr>, IExpr, IExpr),
    Replace(Box<SExpr>, String, String),
}

fn clamp(i: i64, n: usize) -> usize {
    if i < 0 {
        0
    } else if i as usize > n {
        n
    } else {
        i as usize
    }
}

fn eval_i(e: &IExpr, inputs: &[String]) -> Option<i64> {
    match e {
        IExpr::Const(k) => Some(*k),
        IExpr::Len(base, k) => {
            let s = eval_s(base, inputs)?;
            Some(s.chars().count() as i64 - *k)
        }
    }
}

fn eval_s(e: &SExpr, inputs: &[String]) -> Option<String> {
    Some(match e {
        SExpr::Var(i) => inputs.get(*i)?.clone(),
        SExpr::Lit(s) => s.clone(),
        SExpr::Upper(a) => eval_s(a, inputs)?.to_uppercase(),
        SExpr::Lower(a) => eval_s(a, inputs)?.to_lowercase(),
        SExpr::Trim(a) => eval_s(a, inputs)?.trim().to_string(),
        SExpr::Reverse(a) => eval_s(a, inputs)?.chars().rev().collect(),
        SExpr::Concat(a, b) => eval_s(a, inputs)? + &eval_s(b, inputs)?,
        SExpr::Slice(a, lo, hi) => {
            let s = eval_s(a, inputs)?;
            let chars: Vec<char> = s.chars().collect();
            let lo = clamp(eval_i(lo, inputs)?, chars.len());
            let hi = clamp(eval_i(hi, inputs)?, chars.len());
            if lo >= hi {
                String::new()
            } else {
                chars[lo..hi].iter().collect()
            }
        }
        SExpr::Replace(a, old, new) => {
            if old.is_empty() {
                return None;
            }
            eval_s(a, inputs)?.replace(old, new)
        }
    })
}

// ── Codegen to Mog ───────────────────────────────────────────────────────────

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn code_i(e: &IExpr, params: &[String]) -> String {
    match e {
        IExpr::Const(k) => k.to_string(),
        IExpr::Len(base, 0) => format!("{}.len", code_s(base, params)),
        IExpr::Len(base, k) => format!("{}.len - {k}", code_s(base, params)),
    }
}

fn code_s(e: &SExpr, params: &[String]) -> String {
    match e {
        SExpr::Var(i) => params[*i].clone(),
        SExpr::Lit(s) => format!("\"{}\"", esc(s)),
        SExpr::Upper(a) => format!("{}.upper()", code_s(a, params)),
        SExpr::Lower(a) => format!("{}.lower()", code_s(a, params)),
        SExpr::Trim(a) => format!("{}.trim()", code_s(a, params)),
        SExpr::Reverse(a) => format!("{}.reverse()", code_s(a, params)),
        SExpr::Concat(a, b) => format!("({} + {})", code_s(a, params), code_s(b, params)),
        SExpr::Slice(a, lo, hi) => format!(
            "{}.slice({}, {})",
            code_s(a, params),
            code_i(lo, params),
            code_i(hi, params)
        ),
        SExpr::Replace(a, old, new) => {
            format!("{}.replace(\"{}\", \"{}\")", code_s(a, params), esc(old), esc(new))
        }
    }
}

// ── Synthesis (bottom-up enumeration, deduped by signature) ──────────────────

/// Evaluate an expression on every example's inputs; None if any eval fails.
fn signature(e: &SExpr, examples: &[StrSynthExample]) -> Option<Vec<String>> {
    examples.iter().map(|ex| eval_s(e, &ex.inputs)).collect()
}

/// Mine literal fragments from the outputs (and a few common separators).
fn mined_literals(examples: &[StrSynthExample]) -> Vec<String> {
    let mut set: HashSet<String> = HashSet::new();
    for sep in [" ", "_", "-", ".", ",", "", "/", ":"] {
        set.insert(sep.to_string());
    }
    for ex in examples {
        let chars: Vec<char> = ex.expected.chars().collect();
        // prefixes/suffixes up to len 4
        for len in 1..=4usize.min(chars.len()) {
            set.insert(chars[..len].iter().collect());
            set.insert(chars[chars.len() - len..].iter().collect());
        }
        // short infix substrings (len <= 3) — captures internal separators like ", "
        for start in 0..chars.len() {
            for len in 1..=3usize.min(chars.len() - start) {
                set.insert(chars[start..start + len].iter().collect());
            }
        }
    }
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}

pub fn synthesize_string_program(
    params: &[String],
    examples: &[StrSynthExample],
) -> StrSynthResult {
    const MAX_SIZE: usize = 7;
    const MAX_BANK: usize = 40_000;
    let n_args = params.len();
    if examples.is_empty() || n_args == 0 {
        return fail("no examples / no args");
    }
    let target: Vec<String> = examples.iter().map(|e| e.expected.clone()).collect();
    let lits = mined_literals(examples);
    let ixs: Vec<IExpr> = {
        let mut ix = vec![IExpr::Const(0), IExpr::Const(1), IExpr::Const(2), IExpr::Const(3)];
        for i in 0..n_args {
            for k in 0..=2 {
                ix.push(IExpr::Len(Box::new(SExpr::Var(i)), k));
            }
        }
        ix
    };

    // Size-ordered bottom-up enumeration, deduped by evaluation signature. by_size[k]
    // holds the distinct expressions of size k; we always find the smallest program.
    let mut seen: HashSet<Vec<String>> = HashSet::new();
    let mut by_size: Vec<Vec<(SExpr, Vec<String>)>> = vec![Vec::new(); MAX_SIZE + 1];

    macro_rules! try_push {
        ($e:expr, $size:expr) => {{
            let e = $e;
            if let Some(sig) = signature(&e, examples) {
                if sig == target {
                    return emit(e, params);
                }
                if seen.insert(sig.clone()) {
                    by_size[$size].push((e, sig));
                }
            }
        }};
    }

    // Size 1: variables + literals.
    for i in 0..n_args {
        try_push!(SExpr::Var(i), 1);
    }
    for l in &lits {
        try_push!(SExpr::Lit(l.clone()), 1);
    }

    for size in 2..=MAX_SIZE {
        // 1. Concat first (field assembly) — the most useful op; never starve it.
        for left in 1..size {
            let right = size - 1 - left;
            if right < 1 {
                continue;
            }
            let las: Vec<SExpr> = by_size[left].iter().map(|(e, _)| e.clone()).collect();
            let ras: Vec<SExpr> = by_size[right].iter().map(|(e, _)| e.clone()).collect();
            for a in &las {
                for b in &ras {
                    try_push!(SExpr::Concat(Box::new(a.clone()), Box::new(b.clone())), size);
                }
            }
        }
        // 2. Unary case/reverse/trim over size-1 children.
        let children: Vec<SExpr> = by_size[size - 1].iter().map(|(e, _)| e.clone()).collect();
        for e in &children {
            try_push!(SExpr::Upper(Box::new(e.clone())), size);
            try_push!(SExpr::Lower(Box::new(e.clone())), size);
            try_push!(SExpr::Trim(Box::new(e.clone())), size);
            try_push!(SExpr::Reverse(Box::new(e.clone())), size);
        }
        // 3. Slices over size-1 children — bounded to prefix/suffix/char windows.
        for e in &children {
            for lo in &ixs {
                for hi in &ixs {
                    try_push!(SExpr::Slice(Box::new(e.clone()), lo.clone(), hi.clone()), size);
                }
            }
            if by_size[size].len() > MAX_BANK {
                break;
            }
        }
        // 4. Replace — only directly on variables, with mined non-separator
        //    literals, since it is the most combinatorially explosive op.
        if size == 2 {
            for i in 0..n_args {
                for old in &lits {
                    if old.chars().count() < 2 {
                        continue;
                    }
                    for new in &lits {
                        try_push!(
                            SExpr::Replace(Box::new(SExpr::Var(i)), old.clone(), new.clone()),
                            size
                        );
                    }
                }
            }
        }
    }

    if std::env::var("STRSYNTH_DEBUG").is_ok() {
        let sizes: Vec<usize> = by_size.iter().map(|v| v.len()).collect();
        eprintln!("[strsynth] by_size lens = {sizes:?}; lits={}; target={target:?}", lits.len());
    }
    fail("no string program found within the enumeration bound")
}

fn emit(e: SExpr, params: &[String]) -> StrSynthResult {
    let body = code_s(&e, params);
    let decls = params
        .iter()
        .map(|p| format!("{p}: string"))
        .collect::<Vec<_>>()
        .join(", ");
    let fn_name = "transform";
    let code = format!("fn {fn_name}({decls}) -> string {{\n    return {body};\n}}\n");
    StrSynthResult {
        success: true,
        code,
        method: "string_synth".to_string(),
        error: None,
    }
}

fn fail(msg: &str) -> StrSynthResult {
    StrSynthResult {
        success: false,
        code: String::new(),
        method: "string_synth_unsupported".to_string(),
        error: Some(msg.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(inputs: &[&str], expected: &str) -> StrSynthExample {
        StrSynthExample {
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            expected: expected.to_string(),
        }
    }

    fn solve(params: &[&str], exs: &[StrSynthExample]) -> StrSynthResult {
        synthesize_string_program(
            &params.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            exs,
        )
    }

    #[test]
    fn learns_reverse() {
        let r = solve(
            &["s"],
            &[ex(&["abc"], "cba"), ex(&["hello"], "olleh"), ex(&["xy"], "yx")],
        );
        assert!(r.success, "{:?}", r.error);
        assert!(r.code.contains("reverse"), "{}", r.code);
    }

    #[test]
    fn learns_capitalize() {
        // first char upper, rest unchanged: cat -> Cat
        let r = solve(
            &["s"],
            &[ex(&["cat"], "Cat"), ex(&["dog"], "Dog"), ex(&["fox"], "Fox")],
        );
        assert!(r.success, "{:?}", r.error);
    }

    #[test]
    fn learns_full_name_concat() {
        // first + " " + last
        let r = solve(
            &["a", "b"],
            &[
                ex(&["john", "smith"], "john smith"),
                ex(&["jane", "doe"], "jane doe"),
                ex(&["amy", "lee"], "amy lee"),
            ],
        );
        assert!(r.success, "{:?}", r.error);
    }

    #[test]
    fn learns_initials() {
        // first letter of each, uppercased: john smith -> JS
        let r = solve(
            &["a", "b"],
            &[
                ex(&["john", "smith"], "JS"),
                ex(&["jane", "doe"], "JD"),
                ex(&["amy", "lee"], "AL"),
            ],
        );
        assert!(r.success, "{:?}", r.error);
    }

    #[test]
    fn learns_upper() {
        let r = solve(&["s"], &[ex(&["abc"], "ABC"), ex(&["heLLo"], "HELLO")]);
        assert!(r.success, "{:?}", r.error);
        assert!(r.code.contains("upper"));
    }

    #[test]
    fn learns_last_first_with_infix_separator() {
        // last, first — needs the internal ", " separator (an infix literal).
        let r = solve(
            &["a", "b"],
            &[
                ex(&["john", "smith"], "smith, john"),
                ex(&["jane", "doe"], "doe, jane"),
                ex(&["amy", "lee"], "lee, amy"),
            ],
        );
        assert!(r.success, "{:?}", r.error);
    }

    #[test]
    fn declines_when_unseparable() {
        // Same input mapped to two different outputs — no function can fit.
        let r = solve(&["s"], &[ex(&["x"], "a"), ex(&["x"], "b")]);
        assert!(!r.success);
    }
}
