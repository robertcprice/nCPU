//! Seeded mini-crate harness for NL synthesis fixtures (Package H).
//!
//! Each fixture is a real `cargo test` oracle instead of grep-on-source.

use crate::agent::repo::gencode_normalize::{escape_module_name, normalize_component};
use crate::agent::repo::gencode_tests::{emit_main_demo, emit_tests_module};
use crate::agent::repo::nl_fixture_wrong_stub;
use crate::agent::repo::GuardrailPolicy;
use crate::agent::tools::{FsTool, SecureToolRuntime, Tool, ToolCall};
use crate::mog_transpile::to_rust;
use std::fs;
use std::path::Path;

/// Result of the post-write compile gate on a generated crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileStatus {
    /// `cargo check` ran and the crate compiled clean.
    Ok,
    /// `cargo check` ran and the crate FAILED to compile; carries the compiler error.
    Failed(String),
    /// cargo was unavailable / could not be run — NOT a success.
    Unverified(String),
}

impl CompileStatus {
    pub fn is_ok(&self) -> bool {
        matches!(self, CompileStatus::Ok)
    }
}

/// Outcome of writing a synthesized multi-file project.
#[derive(Debug, Clone)]
pub struct WriteOutcome {
    /// Relative paths written, in write order.
    pub written: Vec<String>,
    /// Whether the generated crate compiles (the gate).
    pub compile: CompileStatus,
}

impl WriteOutcome {
    /// The writer reports overall success ONLY when the compile gate is clean.
    pub fn succeeded(&self) -> bool {
        self.compile.is_ok()
    }
}

/// Result of the post-compile `cargo test` gate on a verified crate (PIECE 4).
///
/// Distinct from [`CompileStatus`]: a crate can compile clean (`CompileStatus::Ok`)
/// yet its generated reproduction tests can FAIL (the fn does not reproduce its
/// own examples). That failure is exactly what this gate exists to catch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestStatus {
    /// `cargo test` ran and every generated test passed.
    Ok,
    /// `cargo test` ran and at least one test FAILED; carries the test output.
    Failed(String),
    /// cargo was unavailable / could not be run, or the test gate was skipped
    /// because compilation failed — NOT a success.
    Unverified(String),
}

impl TestStatus {
    pub fn is_ok(&self) -> bool {
        matches!(self, TestStatus::Ok)
    }
}

/// Outcome of writing a *verified* multi-file project: the same write +
/// compile-gate result as [`WriteOutcome`], PLUS the result of running the
/// generated reproduction tests (`cargo test`).
///
/// SOUNDNESS CAVEAT: `test == Ok` proves only that each fn reproduces its OWN
/// proposed examples (a self-consistency oracle). It does NOT prove the
/// whole-artifact behaves correctly — the components were already strict-verified
/// upstream by the solver; this gate is a defense-in-depth reproduction check on
/// the emitted Rust, not an independent correctness proof.
#[derive(Debug, Clone)]
pub struct VerifiedOutcome {
    /// Relative paths written, in write order.
    pub written: Vec<String>,
    /// Whether the generated crate compiles (the first gate).
    pub compile: CompileStatus,
    /// Whether the generated reproduction tests pass (the second gate; only run
    /// when `compile` is `Ok`).
    pub test: TestStatus,
    /// Components that got a real execution test (>=1 renderable example) — their
    /// behavior is checked by `cargo test`, not just compilation.
    pub tested: Vec<String>,
    /// Components with NO renderable example, so they are COMPILE-ONLY (no
    /// behavioral test). Surfaced so a green `test` gate never falsely implies
    /// every component is execution-verified ("no silent caps").
    pub compile_only: Vec<String>,
}

const CARGO_TOML_TEMPLATE: &str = r#"[package]
name = "{package_name}"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"
"#;

fn fixture_package_name(fixture_id: &str) -> String {
    fixture_id.replace('_', "-")
}

/// `cargo test` command for a fixture id (test name matches fixture id).
pub fn nl_fixture_cargo_test_command(fixture_id: &str) -> Option<String> {
    if !is_nl_fixture_id(fixture_id) {
        return None;
    }
    Some(format!("cargo test {fixture_id} --lib"))
}

fn is_nl_fixture_id(fixture_id: &str) -> bool {
    nl_fixture_wrong_stub(fixture_id).is_some()
        || fixture_id == "nl_fixture_multifile_multiply"
        || fixture_id == "nl_fixture_gcd"
}

/// Write a minimal Cargo project with a wrong implementation and embedded unit test.
pub fn write_nl_fixture_crate(root: &Path, fixture_id: &str) -> Result<(), String> {
    if fixture_id == "nl_fixture_multifile_multiply" {
        return write_multifile_multiply_fixture(root);
    }
    if fixture_id == "nl_fixture_gcd" {
        return write_gcd_fixture(root);
    }
    let stub = nl_fixture_wrong_stub(fixture_id)
        .ok_or_else(|| format!("unknown nl fixture id: {fixture_id}"))?;
    let tests = nl_fixture_test_module(fixture_id)?;
    let cargo_toml =
        CARGO_TOML_TEMPLATE.replace("{package_name}", &fixture_package_name(fixture_id));
    fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    fs::write(root.join("Cargo.toml"), cargo_toml).map_err(|e| e.to_string())?;
    fs::write(root.join("src/lib.rs"), format!("{stub}{tests}")).map_err(|e| e.to_string())?;
    Ok(())
}

/// PRODUCT writer: given solved components `(module_name, mog_code)` and a sandbox
/// `root`, transpile each component's Mog `code` to Rust via [`to_rust`] and write
/// a minimal multi-file crate:
///   * `src/<module>.rs` — the component's synthesized fn(s),
///   * `src/lib.rs`      — `mod <module>; pub use <module>::*;` per component,
///   * `Cargo.toml`      — a minimal library manifest.
///
/// All writes go through the sandboxed, traversal-guarded [`FsTool`] (paths are
/// relative to `root`; `..`/absolute paths are rejected). Module names are
/// sanitized to valid Rust identifiers. Returns the relative paths written, in
/// write order. Sibling components are INDEPENDENT (no inter-fn wiring).
pub fn write_synthesized_project(
    root: &Path,
    package_name: &str,
    components: &[(String, String)],
) -> Result<WriteOutcome, String> {
    if components.is_empty() {
        return Err("no synthesized components to write".to_string());
    }
    let fs_tool = FsTool::new(root.to_path_buf());
    let write = |rel: &str, content: &str| -> Result<(), String> {
        fs_tool
            .invoke(&ToolCall::new("write").arg("path", rel).arg("content", content))
            .map(|_| ())
            .map_err(|e| e.to_string())
    };

    let mut written = Vec::new();
    let mut modules: Vec<String> = Vec::with_capacity(components.len());
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();

    // First pass: assign each component its final module name and record the
    // (fn_name -> module) map so a CONSUMER that calls a SIBLING producer can
    // get a precise `use crate::<producer_module>::<producer_fn>;` injected.
    // (B) MODULE NAMING: sanitize -> keyword-escape -> dedup on the FINAL module
    // name (so addTwo/add_two don't both become `mod add_two`, and a component
    // named `loop` doesn't emit an illegal `mod loop;`).
    let mut fn_to_module: Vec<(String, String)> = Vec::with_capacity(components.len());
    let mut module_for: Vec<String> = Vec::with_capacity(components.len());
    for (name, _) in components {
        let base = escape_module_name(&sanitize_module_name(name));
        let module = dedup_module_name(base, &mut used);
        // The emitted Mog/Rust fn name is the original component name (the
        // transpiler keeps `fn <name>`), so map THAT to the module.
        fn_to_module.push((name.clone(), module.clone()));
        module_for.push(module);
    }

    for (idx, (name, mog_code)) in components.iter().enumerate() {
        let module = module_for[idx].clone();

        let rust = to_rust(mog_code);
        // (A) Robust Rust-normalization pass (replaces brittle publicize_fns):
        // pub-fns, `.len`->`.len()`, i64 index cast, mutated-Vec-param `mut`,
        // empty-array literal -> Vec::new(). GENERATED file only; transpiler untouched.
        let rust = normalize_component(&rust);

        // (STEP7) USE-INJECTION: if this component's body CALLS a sibling
        // producer's fn (true inter-function data flow discovered by the search),
        // inject `use crate::<producer_module>::<producer_fn>;` so the generated
        // module compiles. No transpiler edit — purely a post-transpile prelude
        // on the GENERATED file, mirroring `normalize_component`.
        let mut uses: Vec<String> = Vec::new();
        for (sib_fn, sib_mod) in &fn_to_module {
            if sib_fn == name {
                continue; // never import self
            }
            if body_calls_fn(&rust, sib_fn) {
                uses.push(format!("use crate::{sib_mod}::{sib_fn};"));
            }
        }
        let prelude = if uses.is_empty() {
            String::new()
        } else {
            format!("{}\n\n", uses.join("\n"))
        };

        let body = format!(
            "//! Synthesized component `{module}`.\n\n{prelude}{}\n",
            rust.trim_end()
        );
        let rel = format!("src/{module}.rs");
        write(&rel, &body)?;
        written.push(rel);
        modules.push(module);
    }

    let mut lib = String::from("//! Generated multi-file program (independent sibling components).\n\n");
    for m in &modules {
        lib.push_str(&format!("mod {m};\npub use {m}::*;\n"));
    }
    write("src/lib.rs", &lib)?;
    written.push("src/lib.rs".to_string());

    let pkg = if package_name.trim().is_empty() {
        "generated".to_string()
    } else {
        fixture_package_name(&sanitize_module_name(package_name))
    };
    let cargo_toml = CARGO_TOML_TEMPLATE.replace("{package_name}", &pkg);
    write("Cargo.toml", &cargo_toml)?;
    written.push("Cargo.toml".to_string());

    // (C) COMPILE GATE: `cargo check` the generated crate via the secure runtime.
    let compile = compile_gate(root);
    Ok(WriteOutcome { written, compile })
}

/// (PIECE 4) VERIFIED PRODUCT writer: like [`write_synthesized_project`] but each
/// component carries the verifying [`crate::benchmark::Example`]s it was solved
/// against, so the emitted crate ALSO carries:
///   * a `#[cfg(test)] mod tests` (from [`emit_tests_module`]) appended to each
///     `src/<module>.rs`, asserting the fn reproduces its own examples, and
///   * a `src/main.rs` demo (from [`emit_main_demo`]) calling each fn on its
///     first example, plus a `[[bin]]` target so cargo actually builds it.
///
/// After writing, it runs TWO gates: `cargo check` (compile) and — only if that
/// passes — `cargo test` (reproduction). Both go through the same traversal-
/// guarded [`FsTool`] / secure runtime as [`write_synthesized_project`], which is
/// left UNCHANGED (this is purely additive).
///
/// Emission guards (no false-green): if [`emit_tests_module`] returns `""` for a
/// component, no test module is appended; if [`emit_main_demo`] returns `""`, no
/// `src/main.rs` is written AND no `[[bin]]` is emitted (a dangling bin path
/// would fail the build).
pub fn write_verified_project(
    root: &Path,
    package_name: &str,
    components: &[(String, String, Vec<crate::benchmark::Example>)],
) -> Result<VerifiedOutcome, String> {
    if components.is_empty() {
        return Err("no verified components to write".to_string());
    }
    let fs_tool = FsTool::new(root.to_path_buf());
    let write = |rel: &str, content: &str| -> Result<(), String> {
        fs_tool
            .invoke(&ToolCall::new("write").arg("path", rel).arg("content", content))
            .map(|_| ())
            .map_err(|e| e.to_string())
    };

    let mut written = Vec::new();
    let mut modules: Vec<String> = Vec::with_capacity(components.len());
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut tested: Vec<String> = Vec::new();
    let mut compile_only: Vec<String> = Vec::new();

    // Same module-naming passes as `write_synthesized_project`: sanitize ->
    // keyword-escape -> dedup on the FINAL module name; record fn -> module for
    // sibling use-injection.
    let mut fn_to_module: Vec<(String, String)> = Vec::with_capacity(components.len());
    let mut module_for: Vec<String> = Vec::with_capacity(components.len());
    for (name, _, _) in components {
        let base = escape_module_name(&sanitize_module_name(name));
        let module = dedup_module_name(base, &mut used);
        fn_to_module.push((name.clone(), module.clone()));
        module_for.push(module);
    }

    for (idx, (name, mog_code, examples)) in components.iter().enumerate() {
        let module = module_for[idx].clone();

        let rust = to_rust(mog_code);
        let rust = normalize_component(&rust);

        // Sibling use-injection (identical to `write_synthesized_project`).
        let mut uses: Vec<String> = Vec::new();
        for (sib_fn, sib_mod) in &fn_to_module {
            if sib_fn == name {
                continue;
            }
            if body_calls_fn(&rust, sib_fn) {
                uses.push(format!("use crate::{sib_mod}::{sib_fn};"));
            }
        }
        let prelude = if uses.is_empty() {
            String::new()
        } else {
            format!("{}\n\n", uses.join("\n"))
        };

        let mut body = format!(
            "//! Synthesized component `{module}`.\n\n{prelude}{}\n",
            rust.trim_end()
        );

        // ADDITIVE: append the reproduction-test module next to the fn. The fn is
        // now `pub` (normalize_component), so `use super::*;` re-exports it.
        // Guard (D7): empty string means nothing rendered -> append nothing
        // (never an empty `mod tests {}` or an always-true test).
        let tests = emit_tests_module(name, examples);
        if tests.is_empty() {
            compile_only.push(name.clone());
        } else {
            body.push_str(&tests);
            tested.push(name.clone());
        }

        let rel = format!("src/{module}.rs");
        write(&rel, &body)?;
        written.push(rel);
        modules.push(module);
    }

    let mut lib =
        String::from("//! Generated multi-file program (independent sibling components).\n\n");
    for m in &modules {
        lib.push_str(&format!("mod {m};\npub use {m}::*;\n"));
    }
    write("src/lib.rs", &lib)?;
    written.push("src/lib.rs".to_string());

    // Package name computed EXACTLY as `write_synthesized_project` does (dashes).
    let pkg = if package_name.trim().is_empty() {
        "generated".to_string()
    } else {
        fixture_package_name(&sanitize_module_name(package_name))
    };
    // The Rust crate identifier for `use <crate>::*;` needs UNDERSCORES (cargo
    // maps a `foo-bar` package to the `foo_bar` crate ident).
    let crate_ident = pkg.replace('-', "_");

    // src/main.rs demo + [[bin]] target — ONLY if the demo renders.
    let main_components: Vec<(String, Vec<crate::benchmark::Example>)> = components
        .iter()
        .map(|(n, _, ex)| (n.clone(), ex.clone()))
        .collect();
    let main_body = emit_main_demo(&main_components);
    let has_bin = !main_body.is_empty();
    if has_bin {
        // emit_main_demo yields only `fn main() {...}`; inject the lib `use`.
        let main_src = format!("use {crate_ident}::*;\n\n{main_body}");
        write("src/main.rs", &main_src)?;
        written.push("src/main.rs".to_string());
    }

    // Cargo.toml: always a [lib]; add a [[bin]] ONLY when a main.rs was written.
    // Built locally so the shared CARGO_TOML_TEMPLATE stays untouched.
    let mut cargo_toml = format!(
        "[package]\nname = \"{pkg}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n"
    );
    if has_bin {
        cargo_toml.push_str(&format!(
            "\n[[bin]]\nname = \"{crate_ident}_demo\"\npath = \"src/main.rs\"\n"
        ));
    }
    write("Cargo.toml", &cargo_toml)?;
    written.push("Cargo.toml".to_string());

    // GATE 1: compile. GATE 2: tests — only if compile passed.
    let compile = compile_gate(root);
    let test = if compile.is_ok() {
        test_gate(root)
    } else {
        TestStatus::Unverified("skipped: compile failed".into())
    };
    Ok(VerifiedOutcome {
        written,
        compile,
        test,
        tested,
        compile_only,
    })
}

/// TENSOR REACH (NL-BRIDGE-3B-TENSOR-FORWARD): write a self-contained tensor
/// crate (its `Cargo.toml` + `src/lib.rs` come from
/// [`crate::tensor_nl::tensor_crate_files`], including a path dep on the
/// canonical `mog_synth` crate) to `root`, then run the SAME `cargo check`
/// compile gate used for synthesized projects. The emitted `src/lib.rs` calls
/// real `crate::tensor` forward ops, so a clean gate proves the engine op
/// genuinely type-checks + links — codegen-verified-by-compile, not example
/// search. All writes go through the traversal-guarded [`FsTool`].
pub fn write_tensor_program(
    root: &Path,
    files: &[(String, String)],
) -> Result<WriteOutcome, String> {
    if files.is_empty() {
        return Err("no tensor program files to write".to_string());
    }
    let fs_tool = FsTool::new(root.to_path_buf());
    let mut written = Vec::new();
    for (rel, content) in files {
        fs_tool
            .invoke(&ToolCall::new("write").arg("path", rel).arg("content", content))
            .map_err(|e| e.to_string())?;
        written.push(rel.clone());
    }
    let compile = compile_gate(root);
    Ok(WriteOutcome { written, compile })
}

/// True when `body` contains a CALL to `fn_name` (`fn_name(` with a
/// non-identifier char immediately before, so `mydouble(` does not match
/// `double`). Used to decide whether to inject a `use` for a sibling producer.
pub fn body_calls_fn(body: &str, fn_name: &str) -> bool {
    let pat = format!("{fn_name}(");
    let bytes = body.as_bytes();
    let mut from = 0usize;
    while let Some(off) = body[from..].find(&pat) {
        let at = from + off;
        let prev_ok = if at == 0 {
            true
        } else {
            let pc = bytes[at - 1] as char;
            !(pc.is_alphanumeric() || pc == '_')
        };
        if prev_ok {
            return true;
        }
        from = at + 1;
    }
    false
}

/// Pick a unique module name: if `base` is already used, suffix `_2`, `_3`, …
fn dedup_module_name(base: String, used: &mut std::collections::HashSet<String>) -> String {
    if used.insert(base.clone()) {
        return base;
    }
    let mut n = 2;
    loop {
        let candidate = format!("{base}_{n}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        n += 1;
    }
}

/// (C) Run a sandboxed `cargo check` on the generated crate. Reuses the
/// secure_runtime cargo-check capability (allowlist + guardrails). Returns
/// [`CompileStatus::Unverified`] (NOT success) if cargo cannot be run at all.
/// The behavioral rung above `compile_gate`: run the generated crate's tests
/// (`cargo test`) via the same allowlisted secure runner. A crate that type-checks
/// but whose synthesized logic misbehaves fails HERE. Reuses `CompileStatus`
/// (Ok/Failed/Unverified) as a generic pass/fail-with-error.
pub fn behavior_gate(root: &Path) -> CompileStatus {
    let runtime = SecureToolRuntime::for_repo_repair(root.to_path_buf(), GuardrailPolicy::default());
    match runtime.run_verification_command("cargo test") {
        Ok(v) => {
            if v.success {
                CompileStatus::Ok
            } else {
                let err = if v.stderr.trim().is_empty() {
                    v.stdout
                } else {
                    v.stderr
                };
                CompileStatus::Failed(err)
            }
        }
        Err(e) => CompileStatus::Unverified(e),
    }
}

pub fn compile_gate(root: &Path) -> CompileStatus {
    let runtime = SecureToolRuntime::for_repo_repair(root.to_path_buf(), GuardrailPolicy::default());
    match runtime.run_verification_command("cargo check") {
        Ok(v) => {
            if v.success {
                CompileStatus::Ok
            } else {
                // Surface the compiler error (stderr carries the E-codes).
                let err = if v.stderr.trim().is_empty() {
                    v.stdout
                } else {
                    v.stderr
                };
                CompileStatus::Failed(err)
            }
        }
        Err(e) => CompileStatus::Unverified(e),
    }
}

/// (PIECE 4) Run a sandboxed `cargo test` on the generated crate. Mirrors
/// [`compile_gate`] but runs the test oracle (`cargo test` is on the SAME
/// secure-runtime allowlist as `cargo check`, see `secure_runtime.rs`). Returns
/// [`TestStatus::Unverified`] (NOT success) if cargo cannot be run at all; the
/// caller only invokes this when the compile gate already passed.
fn test_gate(root: &Path) -> TestStatus {
    let runtime = SecureToolRuntime::for_repo_repair(root.to_path_buf(), GuardrailPolicy::default());
    match runtime.run_verification_command("cargo test") {
        Ok(v) => {
            if v.success {
                TestStatus::Ok
            } else {
                // Surface the failing-test output (cargo prints failures to
                // stdout; stderr carries compiler/link errors if any).
                let err = if v.stderr.trim().is_empty() {
                    v.stdout
                } else {
                    v.stderr
                };
                TestStatus::Failed(err)
            }
        }
        Err(e) => TestStatus::Unverified(e),
    }
}

/// Reduce an arbitrary fn/module name to a valid Rust identifier (alnum +
/// underscore, non-leading-digit). Empty ⇒ `component`.
fn sanitize_module_name(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    s = s.trim_matches('_').to_string();
    if s.is_empty() {
        return "component".to_string();
    }
    if s.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        s.insert(0, '_');
    }
    s.to_lowercase()
}

fn write_multifile_multiply_fixture(root: &Path) -> Result<(), String> {
    let cargo_toml = CARGO_TOML_TEMPLATE.replace(
        "{package_name}",
        &fixture_package_name("nl_fixture_multifile_multiply"),
    );
    fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    fs::write(root.join("Cargo.toml"), cargo_toml).map_err(|e| e.to_string())?;
    fs::write(
        root.join("src/ops.rs"),
        "pub fn multiply(a: i64, b: i64) -> i64 { a / b }\n",
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        root.join("src/lib.rs"),
        r#"mod ops;
pub use ops::multiply;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_multifile_multiply() {
        assert_eq!(multiply(3, 4), 12);
    }
}
"#,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn write_gcd_fixture(root: &Path) -> Result<(), String> {
    let cargo_toml =
        CARGO_TOML_TEMPLATE.replace("{package_name}", &fixture_package_name("nl_fixture_gcd"));
    fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    fs::write(root.join("Cargo.toml"), cargo_toml).map_err(|e| e.to_string())?;
    fs::write(
        root.join("src/lib.rs"),
        r#"pub fn gcd(a: i64, b: i64) -> i64 {
    if a < b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1);
    }
}
"#,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn nl_fixture_test_module(fixture_id: &str) -> Result<&'static str, String> {
    match fixture_id {
        "nl_fixture_add" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_add() {
        assert_eq!(add_two(2, 3), 5);
        assert_eq!(add_two(-1, 1), 0);
    }
}
"#),
        "nl_fixture_subtract" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_subtract() {
        assert_eq!(subtract(5, 3), 2);
    }
}
"#),
        "nl_fixture_multiply" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_multiply() {
        assert_eq!(multiply(3, 4), 12);
    }
}
"#),
        "nl_fixture_divide" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_divide() {
        assert_eq!(divide(12, 4), 3);
    }
}
"#),
        "nl_fixture_max" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_max() {
        assert_eq!(max_of(3, 7), 7);
        assert_eq!(max_of(9, 2), 9);
    }
}
"#),
        "nl_fixture_reverse" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_reverse() {
        assert_eq!(reverse(&[1, 2, 3]), vec![3, 2, 1]);
    }
}
"#),
        "nl_fixture_triple" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_triple() {
        assert_eq!(triple(2), 6);
        assert_eq!(triple(5), 15);
        assert_eq!(triple(3), 9);
    }
}
"#),
        // Holdout assertions below use inputs NOT in the synthesis examples, so a
        // passing cargo-test proves the repair generalizes (no example overfit).
        "nl_fixture_square" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_square() {
        assert_eq!(square(7), 49);
        assert_eq!(square(9), 81);
        assert_eq!(square(10), 100);
    }
}
"#),
        "nl_fixture_negate" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_negate() {
        assert_eq!(negate(100), -100);
        assert_eq!(negate(-50), 50);
    }
}
"#),
        "nl_fixture_abs" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_abs() {
        assert_eq!(absval(-100), 100);
        assert_eq!(absval(55), 55);
        assert_eq!(absval(-1), 1);
    }
}
"#),
        "nl_fixture_sum3" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_sum3() {
        assert_eq!(add3(5, 5, 5), 15);
        assert_eq!(add3(100, 1, 1), 102);
    }
}
"#),
        "nl_fixture_arrsum" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_arrsum() {
        assert_eq!(total(vec![10, 20, 30]), 60);
        assert_eq!(total(vec![]), 0);
    }
}
"#),
        "nl_fixture_arrmax" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_arrmax() {
        assert_eq!(biggest(vec![100, 2, 50]), 100);
        assert_eq!(biggest(vec![-3, -1, -2]), -1);
    }
}
"#),
        "nl_fixture_arrlen" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_arrlen() {
        assert_eq!(howmany(vec![10, 20, 30, 40]), 4);
        assert_eq!(howmany(vec![9]), 1);
    }
}
"#),
        "nl_fixture_min3" => Ok(r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_fixture_min3() {
        assert_eq!(smallest(10, 5, 20), 5);
        assert_eq!(smallest(100, 2, 50), 2);
        assert_eq!(smallest(7, 7, 3), 3);
    }
}
"#),
        other => Err(format!("no test module for fixture {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::repo::{GuardrailPolicy, RepairVerifier};

    #[test]
    fn fast_patch_repairs_add_fixture() {
        let root = std::env::temp_dir().join(format!("nsynth_nl_fast_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_add").expect("write");
        let context = crate::agent::repo::RepairContext::build(&root, &GuardrailPolicy::default())
            .expect("context");
        let task = crate::agent::repo::benchmark::nl_synthesis_fixture_suite()
            .into_iter()
            .find(|t| t.id == "nl_fixture_add")
            .expect("fixture")
            .to_task_spec(&root);
        let verification = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify");
        assert!(!verification.success);
        let analysis =
            crate::agent::repo::FailureParser::default().parse(&verification.failure_output());
        let patch = crate::agent::synthesis_proposer::try_nl_repo_fast_patch(
            &task,
            &context,
            "add two numbers",
            Some(&analysis),
        );
        assert!(patch.is_some(), "fast patch should be produced");
        let patch = patch.unwrap();
        fs::write(
            root.join(patch.edits[0].path.clone()),
            patch.edits[0].new_text.clone(),
        )
        .expect("write patch");
        let verification = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(verification.success, "stderr: {}", verification.stderr);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn harness_cargo_test_fails_before_repair() {
        let root = std::env::temp_dir().join(format!("nsynth_nl_harness_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_multiply").expect("write");
        let cmd = nl_fixture_cargo_test_command("nl_fixture_multiply").expect("cmd");
        let verification = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&cmd)
            .expect("verify");
        assert!(!verification.success);
        let _ = fs::remove_dir_all(root);
    }

    fn fresh(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "nsynth_gate_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    /// Two components whose names SANITIZE to the same module must NOT both
    /// become `mod add_two;` (E0428). Dedup is on the FINAL module name, so the
    /// second gets a `_2` suffix; the crate compiles clean through the gate.
    #[test]
    fn name_collision_dedups_module_and_compiles() {
        let root = fresh("collide");
        // `add-two` (non-alnum `-` -> `_`) and `add_two` BOTH sanitize to the
        // module name `add_two`; without dedup that is two `mod add_two;` (E0428).
        let components = vec![
            ("add-two".to_string(), "fn addtwoa(a: i64, b: i64) -> i64 {\n    return (a + b);\n}\n".to_string()),
            ("add_two".to_string(), "fn addtwob(a: i64, b: i64) -> i64 {\n    return (a + b);\n}\n".to_string()),
        ];
        let outcome = write_synthesized_project(&root, "collide", &components).expect("write");
        // Both sanitize to `add_two`; second must be renamed (no duplicate `mod`).
        let lib = fs::read_to_string(root.join("src/lib.rs")).unwrap();
        assert!(lib.contains("mod add_two;"), "lib: {lib}");
        assert!(lib.contains("mod add_two_2;"), "second module deduped: {lib}");
        assert!(
            outcome.compile.is_ok(),
            "collision crate must compile clean: {:?}",
            outcome.compile
        );
        let _ = fs::remove_dir_all(root);
    }

    /// A component whose sanitized name is a Rust keyword must NOT emit
    /// `mod loop;` (a syntax error). The module name is keyword-escaped and the
    /// crate compiles through the gate.
    #[test]
    fn keyword_named_component_escaped_and_compiles() {
        let root = fresh("keyword");
        let components = vec![(
            "loop".to_string(),
            "fn loopfn(x: i64) -> i64 {\n    return (x + 1);\n}\n".to_string(),
        )];
        let outcome = write_synthesized_project(&root, "kw", &components).expect("write");
        let lib = fs::read_to_string(root.join("src/lib.rs")).unwrap();
        assert!(!lib.contains("mod loop;"), "must not emit `mod loop;`: {lib}");
        assert!(lib.contains("mod loop_m;"), "keyword escaped to loop_m: {lib}");
        assert!(root.join("src/loop_m.rs").is_file(), "file name matches module");
        assert!(
            outcome.compile.is_ok(),
            "keyword crate must compile clean: {:?}",
            outcome.compile
        );
        let _ = fs::remove_dir_all(root);
    }

    /// PROVE THE GATE GATES: feed a component whose transpiled body does NOT
    /// compile (references an undefined symbol). The writer must return
    /// CompileStatus::Failed and surface the compiler error — NOT success.
    #[test]
    fn compile_gate_rejects_broken_component() {
        let root = fresh("broken");
        // Deliberately-broken Mog body: references an undefined variable `nope`.
        let bad = "fn broken(x: i64) -> i64 {\n    return (x + nope);\n}\n".to_string();
        let components = vec![("broken".to_string(), bad)];
        let outcome = write_synthesized_project(&root, "broken", &components).expect("write");
        match &outcome.compile {
            CompileStatus::Failed(err) => {
                // Compiler error must be surfaced (E0425: cannot find value `nope`).
                assert!(
                    err.contains("cannot find value") || err.contains("E0425"),
                    "compiler error surfaced: {err}"
                );
            }
            other => panic!("gate must FAIL on broken component, got {other:?}"),
        }
        assert!(!outcome.succeeded(), "writer must NOT report success on broken crate");
        let _ = fs::remove_dir_all(root);
    }

    /// Positive control for the gate: a well-formed two-component request
    /// compiles clean and the outcome reports success.
    #[test]
    fn compile_gate_accepts_good_components() {
        let root = fresh("good");
        let components = vec![
            ("negate".to_string(), "fn negate(x: i64) -> i64 {\n    return (-1 * x);\n}\n".to_string()),
            ("triple".to_string(), "fn triple(x: i64) -> i64 {\n    return (3 * x);\n}\n".to_string()),
        ];
        let outcome = write_synthesized_project(&root, "good", &components).expect("write");
        assert!(
            outcome.succeeded(),
            "good crate must pass the gate: {:?}",
            outcome.compile
        );
        let _ = fs::remove_dir_all(root);
    }

    // ---- PIECE 4: write_verified_project (compile + test gates) ----

    use crate::benchmark::{Example as BenchExample, Value as BenchValue};

    /// A well-formed two-component verified project compiles clean AND its
    /// generated reproduction tests pass (tolerate cargo-absent -> Unverified).
    #[test]
    fn write_verified_project_good_components_compile_ok() {
        let root = fresh("verified_good");
        let components = vec![
            (
                "negate".to_string(),
                "fn negate(x: i64) -> i64 {\n    return (-1 * x);\n}\n".to_string(),
                vec![
                    BenchExample { inputs: vec![BenchValue::Int(5)], expected: BenchValue::Int(-5) },
                    BenchExample { inputs: vec![BenchValue::Int(-3)], expected: BenchValue::Int(3) },
                ],
            ),
            (
                "triple".to_string(),
                "fn triple(x: i64) -> i64 {\n    return (3 * x);\n}\n".to_string(),
                vec![
                    BenchExample { inputs: vec![BenchValue::Int(2)], expected: BenchValue::Int(6) },
                    BenchExample { inputs: vec![BenchValue::Int(5)], expected: BenchValue::Int(15) },
                ],
            ),
        ];
        let outcome = write_verified_project(&root, "verified_good", &components).expect("write");
        // HARD: must compile.
        assert!(
            outcome.compile.is_ok(),
            "verified crate must compile clean: {:?}",
            outcome.compile
        );
        // A main.rs + [[bin]] must have been emitted (both fns render).
        assert!(
            outcome.written.iter().any(|p| p == "src/main.rs"),
            "main.rs should be written: {:?}",
            outcome.written
        );
        // Tests must pass OR cargo be absent (Unverified) — never Failed for good code.
        assert!(
            outcome.test.is_ok() || matches!(outcome.test, TestStatus::Unverified(_)),
            "good code must not FAIL its own reproduction tests: {:?}",
            outcome.test
        );
        // COVERAGE HONESTY: both components render -> both execution-tested, none compile-only.
        assert_eq!(outcome.tested.len(), 2, "both components should be execution-tested");
        assert!(outcome.compile_only.is_empty(), "no compile-only component here");
        let _ = fs::remove_dir_all(root);
    }

    /// THE GATE MUST BITE: identical (correct) code, but one component is handed a
    /// deliberately WRONG example, so `emit_tests_module` writes a false
    /// `assert_eq!`. The crate still type-checks (compile Ok) but `cargo test`
    /// must FAIL — proving the test gate catches a non-reproducing claim.
    /// (If cargo cannot run here, tolerate Unverified — but on any real cargo run
    /// the wrong assert makes it Failed.)
    #[test]
    fn write_verified_project_wrong_example_makes_test_fail() {
        let root = fresh("verified_wrong");
        let components = vec![(
            "triple".to_string(),
            "fn triple(x: i64) -> i64 {\n    return (3 * x);\n}\n".to_string(),
            vec![
                // triple(2) == 6, but we assert 999 -> the generated test must fail.
                BenchExample { inputs: vec![BenchValue::Int(2)], expected: BenchValue::Int(999) },
            ],
        )];
        let outcome = write_verified_project(&root, "verified_wrong", &components).expect("write");
        // It still type-checks (the assert is valid Rust, just false at runtime).
        assert!(
            outcome.compile.is_ok(),
            "crate with a false assert still compiles: {:?}",
            outcome.compile
        );
        match &outcome.test {
            TestStatus::Failed(_) => { /* the gate BIT — exactly what we want */ }
            TestStatus::Unverified(_) => { /* cargo unavailable in this env: tolerated */ }
            TestStatus::Ok => panic!(
                "test gate FAILED TO BITE: a wrong assert_eq! must not pass cargo test"
            ),
        }
        let _ = fs::remove_dir_all(root);
    }

    /// A component whose examples are all unrenderable (e.g. `Str`) yields NO test
    /// module (no always-true test), the crate still compiles, and the test gate
    /// is Ok/Unverified (never Failed) — proving we never emit a false-green test.
    #[test]
    fn write_verified_project_skips_unrenderable_examples() {
        let root = fresh("verified_skip");
        let components = vec![(
            "echo".to_string(),
            "fn echo(x: i64) -> i64 {\n    return x;\n}\n".to_string(),
            vec![
                // Str examples don't render -> emit_tests_module returns "" ->
                // no test module appended, no main demo for this component.
                BenchExample {
                    inputs: vec![BenchValue::Str("a".to_string())],
                    expected: BenchValue::Str("a".to_string()),
                },
            ],
        )];
        let outcome = write_verified_project(&root, "verified_skip", &components).expect("write");
        assert!(
            outcome.compile.is_ok(),
            "crate must compile without a test module: {:?}",
            outcome.compile
        );
        // The module file must NOT contain a `mod tests` (nothing rendered).
        let module_src = fs::read_to_string(root.join("src/echo.rs")).unwrap();
        assert!(
            !module_src.contains("mod tests"),
            "no test module must be emitted for unrenderable examples: {module_src}"
        );
        // No main.rs (no component's first example renders) -> no [[bin]].
        assert!(
            !outcome.written.iter().any(|p| p == "src/main.rs"),
            "no main.rs when nothing renders: {:?}",
            outcome.written
        );
        let cargo = fs::read_to_string(root.join("Cargo.toml")).unwrap();
        assert!(!cargo.contains("[[bin]]"), "no dangling bin target: {cargo}");
        // No tests to fail -> Ok or Unverified, never Failed.
        assert!(
            !matches!(outcome.test, TestStatus::Failed(_)),
            "no false-green/failing test when nothing renders: {:?}",
            outcome.test
        );
        // COVERAGE HONESTY: `echo` had no renderable example, so it is COMPILE-ONLY,
        // not execution-tested — a green `test` gate must not imply otherwise.
        assert_eq!(outcome.compile_only, vec!["echo".to_string()], "must be flagged compile-only");
        assert!(outcome.tested.is_empty(), "no component is execution-tested here");
        let _ = fs::remove_dir_all(root);
    }

    /// STEP7 END-TO-END: solve `double` from examples, register it as a callable
    /// PRIMITIVE, then SEARCH `quadruple` so its body DISCOVERS a call to
    /// `double` (no compose template). Assert the solved quadruple AST contains
    /// a `Call(double)` node (structural, not string), write the 2-module crate
    /// via the real writer (which injects `use crate::double::double;` and runs
    /// the cargo-check gate), then append a generated `assert_eq!(quadruple(3),
    /// 12)` unit test and run `cargo test` — the consumer genuinely calling the
    /// producer, compiled and executed.
    #[test]
    fn step7_quadruple_calls_double_endtoend() {
        use crate::benchmark::{Example, Problem, Value};
        use crate::enumerative::{
            solve_scalar_expr_with_callees, synthesize_scalar_with_callees, NamedCallable,
        };

        // 1. Solve `double(x) = 2*x` from examples (plain base-op search).
        let double_problem = Problem {
            name: "double".to_string(),
            category: "step7",
            description: "double a number",
            signature: "fn double(a: i64) -> i64",
            examples: vec![
                Example { inputs: vec![Value::Int(1)], expected: Value::Int(2) },
                Example { inputs: vec![Value::Int(3)], expected: Value::Int(6) },
                Example { inputs: vec![Value::Int(5)], expected: Value::Int(10) },
                Example { inputs: vec![Value::Int(-2)], expected: Value::Int(-4) },
            ],
            holdouts: vec![],
            reference_code: "",
            ..Default::default()
        };
        let double_res = crate::solver::solve_problem(&double_problem);
        assert!(double_res.success, "double must solve: {:?}", double_res.error);
        let double_code = double_res.code.clone();

        // 2. Register `double` as a callable PRIMITIVE (eval runs its Mog source).
        let dc = double_code.clone();
        let registry: Vec<NamedCallable> = vec![NamedCallable {
            name: "double".to_string(),
            n_args: 1,
            source: double_code.clone(),
            eval: Box::new(move |xs: &[i64]| {
                if xs.len() != 1 {
                    return None;
                }
                let args = vec![Value::Int(xs[0])];
                match crate::runtime::execute_function(&dc, "double", &args, "double") {
                    Ok(crate::runtime::Value::Int(v)) => Some(v),
                    _ => None,
                }
            }),
        }];

        // 3. quadruple(x) = 4*x. SEARCH it WITH `double` registered. Inspect the
        // AST: it MUST contain a Call to double (the searched inter-fn edge).
        let quad_problem = Problem {
            name: "quadruple".to_string(),
            category: "step7",
            description: "quadruple a number using double",
            signature: "fn quadruple(a: i64) -> i64",
            examples: vec![
                Example { inputs: vec![Value::Int(1)], expected: Value::Int(4) },
                Example { inputs: vec![Value::Int(3)], expected: Value::Int(12) },
                Example { inputs: vec![Value::Int(5)], expected: Value::Int(20) },
                Example { inputs: vec![Value::Int(-2)], expected: Value::Int(-8) },
            ],
            holdouts: vec![],
            reference_code: "",
            ..Default::default()
        };

        // AST inspection (no library, controlled): prove a Call IS searched.
        let quad_ast = solve_scalar_expr_with_callees(&quad_problem, &registry, 8_000)
            .expect("quadruple must be solvable with double registered");
        fn contains_call(e: &crate::enumerative::Expr) -> bool {
            use crate::enumerative::Expr;
            match e {
                Expr::Call(..) => true,
                Expr::UnaryOp(_, c) => contains_call(c),
                Expr::BinOp(_, l, r) => contains_call(l) || contains_call(r),
                Expr::IfExpr(_, a, b, c, d) => {
                    contains_call(a) || contains_call(b) || contains_call(c) || contains_call(d)
                }
                _ => false,
            }
        }
        assert!(
            contains_call(&quad_ast),
            "searched quadruple AST must contain a Call(double) node, got: {quad_ast:?}"
        );

        // The emitted quadruple code (with library available) — must call double.
        let quad_res = synthesize_scalar_with_callees(&quad_problem, &registry)
            .expect("quadruple must emit code calling double");
        assert!(
            quad_res.code.contains("double("),
            "emitted quadruple must call double(...): {}",
            quad_res.code
        );

        // 4. Write the 2-module crate via the REAL writer (use-injection + gate).
        let root = fresh("step7");
        let components = vec![
            ("double".to_string(), double_code),
            ("quadruple".to_string(), quad_res.code.clone()),
        ];
        let outcome = write_synthesized_project(&root, "step7", &components).expect("write");
        assert!(
            outcome.compile.is_ok(),
            "2-module crate must compile clean (use-injection working): {:?}\nquad code:\n{}",
            outcome.compile,
            quad_res.code
        );
        // The quadruple module must carry the injected `use crate::double::double;`.
        let quad_mod = fs::read_to_string(root.join("src/quadruple.rs")).unwrap();
        assert!(
            quad_mod.contains("use crate::double::double;"),
            "quadruple module must import double: {quad_mod}"
        );

        // 5. Append a GENERATED unit test and run `cargo test`.
        let mut lib = fs::read_to_string(root.join("src/lib.rs")).unwrap();
        lib.push_str(
            "\n#[cfg(test)]\nmod step7_tests {\n    use super::*;\n    #[test]\n    fn quadruple_calls_double() {\n        assert_eq!(quadruple(3), 12);\n        assert_eq!(quadruple(-2), -8);\n    }\n}\n",
        );
        fs::write(root.join("src/lib.rs"), &lib).unwrap();
        let runtime = SecureToolRuntime::for_repo_repair(root.clone(), GuardrailPolicy::default());
        let test_run = runtime
            .run_verification_command("cargo test")
            .expect("cargo test must run");
        assert!(
            test_run.success,
            "generated unit test must pass:\nstdout:\n{}\nstderr:\n{}",
            test_run.stdout, test_run.stderr
        );
        eprintln!("[STEP7-E2E] crate root = {}", root.display());
        eprintln!("[STEP7-E2E] cargo test stdout:\n{}", test_run.stdout);
        // Keep the crate for inspection when NSYNTH_KEEP_CRATE is set.
        if std::env::var("NSYNTH_KEEP_CRATE").is_err() {
            let _ = fs::remove_dir_all(root);
        }
    }
}
