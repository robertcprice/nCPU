//! CLI tool emission: wrap a verified function in a runnable command-line tool.
//!
//! The synthesis engine produces verified Rust functions (integer args -> i64).
//! This turns one into a self-contained CLI artifact: parse integer argv, call
//! the function, print the result. Compile-and-run verified, dependency-free.
//! This is the smallest "app" wrapper around the engine's strongest capability
//! (function synthesis) — a new artifact type distinct from sites and backends.

/// The type of a CLI argument, so the wrapper parses argv correctly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliArg {
    /// Parsed as an i64.
    Int,
    /// Used as a String verbatim.
    Str,
}

/// Emit a self-contained Rust CLI wrapping `fn_source` (a `fn NAME(...) -> ...`).
/// One argv value is read per entry in `args`, parsed to its type, and passed to
/// `fn_name`; the result is printed. Supports integer and string tools.
pub fn emit_cli_rust(fn_name: &str, fn_source: &str, args: &[CliArg]) -> String {
    let arity = args.len();
    let mut parse = String::new();
    for (i, ty) in args.iter().enumerate() {
        match ty {
            CliArg::Int => parse.push_str(&format!(
                "    let a{i}: i64 = args.get({i}).and_then(|s| s.parse().ok()).unwrap_or_else(|| {{ eprintln!(\"usage: {fn_name} expects {arity} argument(s)\"); std::process::exit(2); }});\n"
            )),
            CliArg::Str => parse.push_str(&format!(
                "    let a{i}: String = args.get({i}).cloned().unwrap_or_else(|| {{ eprintln!(\"usage: {fn_name} expects {arity} argument(s)\"); std::process::exit(2); }});\n"
            )),
        }
    }
    let call_args: Vec<String> = (0..arity).map(|i| format!("a{i}")).collect();
    format!(
        "// Auto-generated CLI wrapping a verified function.\n{}\n\nfn main() {{\n    let args: Vec<String> = std::env::args().skip(1).collect();\n{}    println!(\"{{}}\", {fn_name}({}));\n}}\n",
        fn_source.trim(),
        parse,
        call_args.join(", ")
    )
}

/// Compile the emitted CLI, run it with `args`, and require stdout equals
/// `expected` (trimmed). Proves the generated tool actually computes the result.
pub fn verify_cli(source: &str, args: &[&str], expected: &str) -> Result<(), String> {
    let (src, bin) = crate::backend_http::compile_to_temp_bin(source, false)?;
    let out = std::process::Command::new(&bin).args(args).output();
    crate::backend_http::cleanup_temp_artifacts(&src, &bin);
    let out = out.map_err(|e| format!("run cli: {e}"))?;
    let got = String::from_utf8_lossy(&out.stdout);
    let got = got.trim();
    if got != expected {
        return Err(format!(
            "cli {args:?} produced {got:?}, expected {expected:?} (stderr: {})",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(())
}

/// A comprehended request to build a command-line tool: the function to wrap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CliAsk {
    pub name: String,
}

/// Comprehend "build a CLI tool that ... where NAME(x)=y ..." — a construction
/// cue + a CLI noun + a function name that carries inline examples. Returns None
/// (falls through to the next intake) when it's not a CLI ask or has no examples.
pub fn comprehend_cli_request(text: &str) -> Option<CliAsk> {
    let lower = text.to_lowercase();
    const CUES: [&str; 6] = ["build", "make", "create", "write", "generate", "want"];
    const CLI_NOUNS: [&str; 6] = ["cli", "command-line", "command line", "tool", "utility", "program"];
    let has_cue = CUES.iter().any(|c| lower.split(|ch: char| !ch.is_alphanumeric()).any(|t| t == *c));
    let has_cli = CLI_NOUNS.iter().any(|n| lower.contains(n));
    if !has_cue || !has_cli {
        return None;
    }
    let name = extract_fn_name(&lower)?;
    Some(CliAsk { name })
}

/// Find the function name that carries inline examples: an explicit "function
/// NAME", else the identifier before a `NAME(` that is followed by an `=`.
fn extract_fn_name(lower: &str) -> Option<String> {
    if let Some(rest) = lower.split("function ").nth(1) {
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() && lower.contains(&format!("{name}(")) {
            return Some(name);
        }
    }
    let bytes = lower.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'(' {
            let mut j = i;
            while j > 0 && (bytes[j - 1].is_ascii_alphanumeric() || bytes[j - 1] == b'_') {
                j -= 1;
            }
            if j < i && lower[i..].contains('=') {
                return Some(lower[j..i].to_string());
            }
        }
    }
    None
}

/// Build a CLI ask: synthesize the verified function from the prose, wrap it in a
/// CLI, VERIFY the CLI compiles + computes an example, and write `cli/main.rs`.
/// Fail-closed — a wrong or unsynthesizable function fails the ask.
pub fn build_cli_ask(
    root: &std::path::Path,
    english: &str,
    ask: &CliAsk,
) -> Result<Vec<String>, String> {
    let (rust, fn_name, examples) =
        crate::backend_intake::synthesize_rust_fn_from_prose(english, &ask.name)?;
    let cli = emit_cli_rust(&fn_name, &rust, &[CliArg::Int]);
    if let Some((inp, out)) = examples.first() {
        verify_cli(&cli, &[&inp.to_string()], &out.to_string())?;
    }
    let out_path = root.join("cli/main.rs");
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&out_path, cli).map_err(|e| e.to_string())?;
    Ok(vec!["cli/main.rs".to_string()])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rustc_available() -> bool {
        std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn comprehends_cli_requests() {
        assert_eq!(
            comprehend_cli_request(
                "build a CLI tool for a function double where double(2)=4 and double(5)=10"
            ),
            Some(CliAsk { name: "double".into() })
        );
        assert_eq!(
            comprehend_cli_request("make a command-line tool for triple where triple(3)=9"),
            Some(CliAsk { name: "triple".into() })
        );
        // No inline examples -> falls through (can't synthesize).
        assert!(comprehend_cli_request("build a CLI tool that doubles a number").is_none());
        // Not a CLI ask -> falls through.
        assert!(comprehend_cli_request("build a website for my bakery").is_none());
        assert!(comprehend_cli_request("a function double where double(2)=4").is_none());
    }

    #[test]
    fn emits_wrapper_shape() {
        let cli = emit_cli_rust("dbl", "fn dbl(x: i64) -> i64 { x * 2 }", &[CliArg::Int]);
        assert!(cli.contains("fn dbl(x: i64) -> i64"), "includes the function");
        assert!(cli.contains("fn main()"), "has a main");
        assert!(cli.contains("dbl(a0)"), "calls the function with parsed arg");
        assert!(cli.contains("args.get(0)"), "parses one argv");
    }

    #[test]
    fn unary_cli_computes() {
        if !rustc_available() {
            eprintln!("skipping CLI run test: rustc unavailable");
            return;
        }
        let cli = emit_cli_rust("dbl", "fn dbl(x: i64) -> i64 { x * 2 }", &[CliArg::Int]);
        verify_cli(&cli, &["5"], "10").expect("dbl 5 -> 10");
        verify_cli(&cli, &["-3"], "-6").expect("dbl -3 -> -6");
    }

    #[test]
    fn binary_cli_computes() {
        if !rustc_available() {
            eprintln!("skipping CLI run test: rustc unavailable");
            return;
        }
        let cli = emit_cli_rust(
            "add",
            "fn add(a: i64, b: i64) -> i64 { a + b }",
            &[CliArg::Int, CliArg::Int],
        );
        verify_cli(&cli, &["3", "4"], "7").expect("add 3 4 -> 7");
    }

    #[test]
    fn string_cli_computes() {
        if !rustc_available() {
            eprintln!("skipping CLI run test: rustc unavailable");
            return;
        }
        // A string tool: reverse a string. Proves CLIs wrap string functions too.
        let cli = emit_cli_rust(
            "rev",
            "fn rev(s: String) -> String { s.chars().rev().collect() }",
            &[CliArg::Str],
        );
        verify_cli(&cli, &["hello"], "olleh").expect("rev hello -> olleh");
    }
}
