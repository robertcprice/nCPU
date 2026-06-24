//! Seeded mini-crate harness for NL synthesis fixtures (Package H).
//!
//! Each fixture is a real `cargo test` oracle instead of grep-on-source.

use crate::agent::repo::nl_fixture_wrong_stub;
use crate::agent::tools::{FsTool, Tool, ToolCall};
use crate::mog_transpile::to_rust;
use std::fs;
use std::path::Path;

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
) -> Result<Vec<String>, String> {
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
    let mut modules = Vec::with_capacity(components.len());
    for (name, mog_code) in components {
        let module = sanitize_module_name(name);
        let rust = to_rust(mog_code);
        // Make every top-level fn `pub` so `pub use module::*` in lib.rs actually
        // re-exports it (the transpiler emits bare `fn`, which is private and would
        // make the crate fail to compile / re-export nothing).
        let rust = publicize_fns(&rust);
        // Each module re-exports its own fn(s) via lib.rs `pub use module::*`.
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

    Ok(written)
}

/// Make every top-level `fn` declaration `pub` so it is re-exportable via
/// `pub use module::*`. Also normalizes the transpiler's empty-array literal
/// `: Vec<i64> = [];` to `: Vec<i64> = Vec::new();` so array-output components
/// (e.g. an element-doubling map) compile as a real crate. These are fixups to
/// the GENERATED file only — the transpiler is untouched.
fn publicize_fns(rust: &str) -> String {
    rust.lines()
        .map(|line| {
            let trimmed = line.trim_start();
            let indent = &line[..line.len() - trimmed.len()];
            if trimmed.starts_with("fn ") {
                format!("{indent}pub {trimmed}")
            } else if let Some(pos) = line.find(": Vec<i64> = [];") {
                format!("{}: Vec<i64> = Vec::new();", &line[..pos])
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
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
}
