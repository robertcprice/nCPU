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
