//! Generated backend compile repair loop (LOOP-3E).
//!
//! Runs `rustc` on emitted backend sources and applies deterministic repairs
//! when known emission bugs surface in compiler diagnostics.

use crate::backend_ir::StoreKind;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn compile_with_repair(source: &str, store: StoreKind, max_attempts: usize) -> Result<String, String> {
    let mut current = source.to_string();
    let mut last_err = String::new();
    for attempt in 0..max_attempts {
        match try_compile(&current, store) {
            Ok(()) => return Ok(current),
            Err(err) => {
                last_err = err;
                match repair_source(&current, &last_err) {
                    Some(next) if next != current => current = next,
                    _ if attempt + 1 < max_attempts => continue,
                    _ => break,
                }
            }
        }
    }
    Err(format!(
        "generated backend compile repair failed after {max_attempts} attempts\n{last_err}\nsource:\n{current}"
    ))
}

pub fn try_compile(source: &str, store: StoreKind) -> Result<(), String> {
    let src = unique_path("ncpu_backend_repair", "rs");
    let bin = unique_path("ncpu_backend_repair", "bin");
    std::fs::write(&src, source).map_err(|e| format!("write {}: {e}", src.display()))?;
    let mut cmd = Command::new("rustc");
    cmd.arg("--edition=2021").arg(&src).arg("-o").arg(&bin);
    if store == StoreKind::Sqlite {
        cmd.arg("-l").arg("sqlite3");
    }
    let output = cmd.output().map_err(|e| format!("run rustc: {e}"))?;
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&bin);
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

pub fn repair_source(source: &str, stderr: &str) -> Option<String> {
    if stderr.contains("/rules") || stderr.contains("expected `,`, found") {
        if let Some(fixed) = repair_rules_response_literal(source) {
            return Some(fixed);
        }
    }
    None
}

fn repair_rules_response_literal(source: &str) -> Option<String> {
    let needle = "(\"GET\", \"/rules\") => write_response(&mut stream, 200, \"";
    let start = source.find(needle)? + needle.len();
    let rest = &source[start..];
    let end = rest.find("\"),")? + start;
    let payload = &source[start..end];
    if !payload.contains("{\"rules\"") {
        return None;
    }
    let escaped = payload.replace('"', "\\\"");
    let mut out = String::with_capacity(source.len() + escaped.len());
    out.push_str(&source[..start]);
    out.push_str(&escaped);
    out.push_str(&source[end..]);
    Some(out)
}

fn unique_path(stem: &str, ext: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("{}_{}_{}.{}", stem, std::process::id(), nanos, ext))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_ir::{BackendApp, RuleModel, StoreKind};
    use crate::backend_mvp::default_rule_specs;
    use crate::backend_mvp::synthesize_backend_app;

    fn rustc_available() -> bool {
        Command::new("rustc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn break_rules_json_quoting(source: &str) -> String {
        let needle = "(\"GET\", \"/rules\") => write_response(&mut stream, 200, \"";
        let start = source.find(needle).expect("rules line") + needle.len();
        let rest = &source[start..];
        let end = rest.find("\"),").expect("rules end") + start;
        let payload = &source[start..end];
        let broken = payload.replace("\\\"", "\"");
        format!("{}{}{}", &source[..start], broken, &source[end..])
    }

    #[test]
    fn compile_repair_fixes_broken_rules_json_literal() {
        if !rustc_available() {
            eprintln!("skipping compile repair test: rustc unavailable");
            return;
        }

        let generated =
            synthesize_backend_app(&default_rule_specs(), StoreKind::Memory).expect("synthesize");
        let broken = break_rules_json_quoting(&generated.source);
        assert!(try_compile(&broken, StoreKind::Memory).is_err());

        let repaired = compile_with_repair(&broken, StoreKind::Memory, 3).expect("repair compile");
        assert!(try_compile(&repaired, StoreKind::Memory).is_ok());
    }

    #[test]
    fn fresh_ir_render_passes_compile_gate_without_repair() {
        if !rustc_available() {
            eprintln!("skipping compile gate test: rustc unavailable");
            return;
        }

        let app = BackendApp::from_rules(
            "repair gate",
            vec![RuleModel {
                name: "score_bonus".to_string(),
                synthesis_method: "search_polynomial_multi".to_string(),
                rule_code: "fn score_bonus(x: i64) -> i64 { 10 * x + 5 }".to_string(),
            }],
            StoreKind::Memory,
        );
        let source = app.render_rust();
        compile_with_repair(&source, StoreKind::Memory, 1).expect("compile gate");
    }
}
