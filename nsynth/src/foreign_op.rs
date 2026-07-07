//! Foreign-op verifier — the sound first piece of FOREIGN-SOURCE IMPORT (the
//! capability-frontier crosser). A foreign function taken verbatim from a repo is
//! compiled into a harness, run against its mined examples, and accepted ONLY if
//! it reproduces every one. Crucially it lands as TENTATIVE, never Verified:
//!
//!   The engine's own ops are examples-verified, not proven. A foreign op that
//!   passes the SAME examples is the SAME soundness level — EXCEPT it has no
//!   independent corroborator (the engine can't synthesize an equivalent for a
//!   behavior it can't build; that's why we import it). No corroborator =>
//!   TENTATIVE, which the engine already handles honestly. So importing a foreign
//!   op is soundness-CONSISTENT, not soundness-violating.
//!
//! This module is the verify half — it captures functions synthesis can't reach
//! (control flow, loops) and vets them soundly. Making a verified foreign op
//! COMPOSABLE/EMITTABLE (StoredComponent variant + runtime dispatch) is the
//! engine-integration step, specced separately.
//!
//! SECURITY: compiling + running foreign code is arbitrary code execution. This
//! prototype runs a bounded harness with a wall-clock timeout; a PRODUCTION path
//! must sandbox (container / seccomp / no-net) before ingesting untrusted repos.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The vetting verdict for a foreign op. Foreign ops are never `Verified` — the
/// best a borrowed, uncorroborated implementation earns is `Tentative`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForeignVerdict {
    /// Reproduced every mined example in our sandbox — usable, but uncorroborated.
    Tentative,
    /// Did not compile, did not run, or disagreed with an example.
    Refused(String),
}

/// A foreign function vetted for import.
#[derive(Clone, Debug)]
pub struct VerifiedForeignOp {
    pub name: String,
    pub source: String,
    pub arity: usize,
    pub verdict: ForeignVerdict,
}

/// Build a self-contained Rust harness: the foreign `source` + a `main` that runs
/// `name(args)` for each example and exits non-zero on the first mismatch.
fn build_harness(name: &str, source: &str, examples: &[(Vec<i64>, i64)]) -> String {
    let mut checks = String::new();
    for (ins, out) in examples {
        let args = ins.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
        checks.push_str(&format!("    if {name}({args}) != {out} {{ std::process::exit(1); }}\n"));
    }
    format!("#![allow(warnings)]\n{}\n\nfn main() {{\n{checks}    std::process::exit(0);\n}}\n", source.trim())
}

/// Verify a foreign Rust `fn name(a: i64, ..) -> i64` reproduces its mined
/// examples. Returns TENTATIVE on a clean pass (uncorroborated), REFUSED otherwise.
pub fn verify_foreign_rust(name: &str, source: &str, examples: &[(Vec<i64>, i64)]) -> VerifiedForeignOp {
    let arity = examples.first().map(|(i, _)| i.len()).unwrap_or(0);
    let verdict = if examples.len() < 2 {
        ForeignVerdict::Refused("need >= 2 examples".into())
    } else if arity == 0 || examples.iter().any(|(i, _)| i.len() != arity) {
        ForeignVerdict::Refused("inconsistent arity".into())
    } else {
        match compile_and_run(&build_harness(name, source, examples)) {
            Ok(true) => ForeignVerdict::Tentative,
            Ok(false) => ForeignVerdict::Refused("foreign code did not reproduce an example".into()),
            Err(e) => ForeignVerdict::Refused(e),
        }
    };
    VerifiedForeignOp {
        name: name.to_string(),
        source: source.trim().to_string(),
        arity,
        verdict,
    }
}

/// Compile the harness and run it with a wall-clock timeout; Ok(true) iff it
/// exited 0 (every example reproduced).
fn compile_and_run(harness: &str) -> Result<bool, String> {
    let (src, bin) = crate::backend_http::compile_to_temp_bin(harness, false)?;
    let out = run_with_timeout(&bin, Duration::from_secs(5));
    crate::backend_http::cleanup_temp_artifacts(&src, &bin);
    out
}

fn run_with_timeout(bin: &Path, timeout: Duration) -> Result<bool, String> {
    let mut child = Command::new(bin)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("spawn foreign harness: {e}"))?;
    let start = Instant::now();
    loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => return Ok(status.success()),
            None => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err("foreign harness exceeded time budget".into());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rustc_available() -> bool {
        Command::new("rustc").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
    }

    #[test]
    fn foreign_op_beyond_the_synthesis_frontier_is_tentative() {
        if !rustc_available() {
            eprintln!("skipping foreign-op test: rustc unavailable");
            return;
        }
        // Collatz step count: an unbounded WHILE loop the example-synthesizer can't
        // build — exactly the beyond-frontier behavior foreign-import captures.
        let src = "fn collatz_steps(mut n: i64) -> i64 {\n    let mut c = 0;\n    while n != 1 { n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 }; c += 1; }\n    c\n}";
        let ex = vec![(vec![1i64], 0i64), (vec![6], 8), (vec![7], 16), (vec![27], 111)];
        let v = verify_foreign_rust("collatz_steps", src, &ex);
        assert_eq!(v.verdict, ForeignVerdict::Tentative, "correct foreign op -> Tentative (uncorroborated), not Verified");
        assert_eq!(v.arity, 1);

        // A WRONG foreign implementation is REFUSED (doesn't reproduce examples).
        let bad = "fn collatz_steps(n: i64) -> i64 { n }";
        let vb = verify_foreign_rust("collatz_steps", bad, &ex);
        assert!(matches!(vb.verdict, ForeignVerdict::Refused(_)), "wrong foreign op refused");
    }
}
