//! Seeded mini-crate harness for NL synthesis fixtures (Package H).
//!
//! Each fixture is a real `cargo test` oracle instead of grep-on-source.

use crate::agent::repo::gencode_normalize::{escape_module_name, normalize_component};
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
    for (name, mog_code) in components {
        // (B) MODULE NAMING: sanitize -> keyword-escape -> dedup on the FINAL
        // module name (so addTwo/add_two don't both become `mod add_two`, and a
        // component named `loop` doesn't emit an illegal `mod loop;`). The file
        // name and the `mod`/`pub use` name all stay in sync via `module`.
        let base = escape_module_name(&sanitize_module_name(name));
        let module = dedup_module_name(base, &mut used);

        let rust = to_rust(mog_code);
        // (A) Robust Rust-normalization pass (replaces brittle publicize_fns):
        // pub-fns, `.len`->`.len()`, i64 index cast, mutated-Vec-param `mut`,
        // empty-array literal -> Vec::new(). GENERATED file only; transpiler untouched.
        let rust = normalize_component(&rust);
        let body = format!("//! Synthesized component `{module}`.\n\n{}\n", rust.trim_end());
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
fn compile_gate(root: &Path) -> CompileStatus {
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
}
