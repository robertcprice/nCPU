//! HTTP verification helpers for generated backends (LOOP-5).
//!
//! Hermetic localhost probes used as an un-gameable accept gate after compile.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpRuleCheck {
    pub rule: String,
    pub input: i64,
    pub output: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpOutputMismatch {
    pub rule: String,
    pub input: i64,
    pub expected: i64,
}

/// Parse an HTTP verification error that reports an output mismatch.
pub fn parse_output_mismatch(err: &str) -> Option<HttpOutputMismatch> {
    if !err.contains("expected \"output\":") {
        return None;
    }
    let rule = err
        .split("/rules/")
        .nth(1)?
        .split("/evaluate")
        .next()?
        .to_string();
    let input = err.split("\"input\":").nth(1)?.split('}').next()?.trim().parse().ok()?;
    let expected = err
        .split("expected \"output\":")
        .nth(1)?
        .split(|c: char| !c.is_ascii_digit() && c != '-')
        .next()?
        .trim()
        .parse()
        .ok()?;
    Some(HttpOutputMismatch {
        rule,
        input,
        expected,
    })
}

pub fn verify_backend_http(
    bin: &Path,
    checks: &[HttpRuleCheck],
    max_attempts: usize,
) -> Result<(), String> {
    let mut last_err = String::new();
    for attempt in 0..max_attempts {
        match probe_backend_once(bin, checks) {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_err = err;
                if attempt + 1 < max_attempts {
                    thread::sleep(Duration::from_millis(50 * (attempt as u64 + 1)));
                }
            }
        }
    }
    Err(format!(
        "generated backend HTTP verification failed after {max_attempts} attempts: {last_err}"
    ))
}

fn probe_backend_once(bin: &Path, checks: &[HttpRuleCheck]) -> Result<(), String> {
    let mut child = Command::new(bin)
        .arg("--port")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn generated backend: {e}"))?;

    let addr = read_ready_addr(&mut child)?;
    thread::sleep(Duration::from_millis(25));

    let result = (|| {
        let health = http_get(&addr, "/health")?;
        if !health.contains("200 OK") || !health.contains("\"ok\":true") {
            return Err(format!("GET /health unexpected response: {health}"));
        }
        for check in checks {
            let path = format!("/rules/{}/evaluate", check.rule);
            let body = format!("{{\"input\":{}}}", check.input);
            let resp = http_post(&addr, &path, &body)?;
            if !resp.contains("200 OK") {
                return Err(format!("POST {path} failed: {resp}"));
            }
            let needle = format!("\"output\":{}", check.output);
            if !resp.contains(&needle) {
                return Err(format!(
                    "POST {path} body {body} expected {needle} in: {resp}"
                ));
            }
        }
        Ok(())
    })();

    stop_child(&mut child);
    result
}

/// SUBMISSION-INTAKE smoke for the site+backend closed loop: boot the
/// generated backend, POST a form-shaped body to /events (the target every
/// api-wired site form carries), require 201 + ok, then require the stored
/// submission to be visible via GET /events. This is the "form has a REAL,
/// live target" verification — end to end, no mocks.
pub fn verify_submission_intake(bin: &Path, max_attempts: usize) -> Result<(), String> {
    let mut last_err = String::new();
    for attempt in 0..max_attempts {
        match probe_submission_once(bin) {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_err = err;
                if attempt + 1 < max_attempts {
                    thread::sleep(Duration::from_millis(50 * (attempt as u64 + 1)));
                }
            }
        }
    }
    Err(format!(
        "generated backend submission intake failed after {max_attempts} attempts: {last_err}"
    ))
}

fn probe_submission_once(bin: &Path) -> Result<(), String> {
    let mut child = Command::new(bin)
        .arg("--port")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn generated backend: {e}"))?;

    let addr = read_ready_addr(&mut child)?;
    thread::sleep(Duration::from_millis(25));

    let result = (|| {
        let resp = http_post(&addr, "/events", "name=ada&message=hello")?;
        if !resp.contains("201") || !resp.contains("\"ok\":true") {
            return Err(format!("POST /events unexpected response: {resp}"));
        }
        let listed = http_get(&addr, "/events")?;
        if !listed.contains("\"rule\":\"submission\"") {
            return Err(format!("submission not visible via GET /events: {listed}"));
        }
        Ok(())
    })();

    stop_child(&mut child);
    result
}

fn read_ready_addr(child: &mut Child) -> Result<String, String> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "generated backend missing stdout".to_string())?;
    let mut reader = BufReader::new(stdout);
    let mut ready = String::new();
    reader
        .read_line(&mut ready)
        .map_err(|e| format!("read BACKEND_READY line: {e}"))?;
    ready
        .trim()
        .strip_prefix("BACKEND_READY http://")
        .map(str::to_string)
        .ok_or_else(|| format!("unexpected ready line: {ready}"))
}

fn http_get(addr: &str, path: &str) -> Result<String, String> {
    request(addr, "GET", path, "")
}

fn http_post(addr: &str, path: &str, body: &str) -> Result<String, String> {
    request(addr, "POST", path, body)
}

fn request(addr: &str, method: &str, path: &str, body: &str) -> Result<String, String> {
    let mut stream =
        TcpStream::connect(addr).map_err(|e| format!("connect {addr}: {e}"))?;
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("write request: {e}"))?;
    let mut resp = String::new();
    stream
        .read_to_string(&mut resp)
        .map_err(|e| format!("read response: {e}"))?;
    Ok(resp)
}

fn stop_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

pub fn compile_to_temp_bin(source: &str, sqlite: bool) -> Result<(std::path::PathBuf, std::path::PathBuf), String> {
    let src = unique_path("ncpu_backend_http", "rs");
    let bin = unique_path("ncpu_backend_http", "bin");
    std::fs::write(&src, source).map_err(|e| format!("write {}: {e}", src.display()))?;
    let mut cmd = Command::new("rustc");
    cmd.arg("--edition=2021").arg(&src).arg("-o").arg(&bin);
    if sqlite {
        cmd.arg("-l").arg("sqlite3");
    }
    let output = cmd.output().map_err(|e| format!("run rustc: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "rustc failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok((src, bin))
}

fn unique_path(stem: &str, ext: &str) -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("{}_{}_{}.{}", stem, std::process::id(), nanos, ext))
}

pub fn cleanup_temp_artifacts(src: &Path, bin: &Path) {
    let _ = std::fs::remove_file(src);
    let _ = std::fs::remove_file(bin);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_mvp::synthesize_backend_app;
    use crate::backend_ir::StoreKind;
    use crate::backend_mvp::default_rule_specs;

    fn rustc_available() -> bool {
        Command::new("rustc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn parse_output_mismatch_extracts_rule_input_expected() {
        let err = r#"POST /rules/damage_penalty/evaluate body {"input":5} expected "output":7 in: HTTP/1.1 200"#;
        let parsed = parse_output_mismatch(err).expect("parse");
        assert_eq!(parsed.rule, "damage_penalty");
        assert_eq!(parsed.input, 5);
        assert_eq!(parsed.expected, 7);
    }

    #[test]
    fn http_gate_accepts_synthesized_multi_rule_backend() {
        if !rustc_available() {
            eprintln!("skipping HTTP gate test: rustc unavailable");
            return;
        }

        let generated =
            synthesize_backend_app(&default_rule_specs(), StoreKind::Memory).expect("synthesize");
        let (src, bin) = compile_to_temp_bin(&generated.source, false).expect("compile");
        let checks = vec![
            HttpRuleCheck {
                rule: "score_bonus".to_string(),
                input: 3,
                output: 35,
            },
            HttpRuleCheck {
                rule: "damage_penalty".to_string(),
                input: 5,
                output: 7,
            },
        ];
        verify_backend_http(&bin, &checks, 2).expect("http verify");
        cleanup_temp_artifacts(&src, &bin);
    }
}
