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

/// RESOURCE CRUD smoke: boot the generated backend, POST a record to
/// `/<resource>` (require 201), then GET `/<resource>` and require the posted
/// record is listed. Proves the in-memory collection round-trips over HTTP.
pub fn verify_resource_crud(bin: &Path, resource: &str, max_attempts: usize) -> Result<(), String> {
    let workdir = unique_path("ncpu_resource_wd", "d");
    let _ = std::fs::create_dir_all(&workdir);
    let mut last_err = String::new();
    let mut out = Err(String::new());
    for attempt in 0..max_attempts {
        match probe_resource_once(bin, resource, &workdir) {
            Ok(()) => {
                out = Ok(());
                break;
            }
            Err(err) => {
                last_err = err;
                if attempt + 1 < max_attempts {
                    thread::sleep(Duration::from_millis(50 * (attempt as u64 + 1)));
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&workdir);
    out.map_err(|_| {
        format!("resource CRUD for /{resource} failed after {max_attempts} attempts: {last_err}")
    })
}

/// PERSISTENCE smoke: boot the backend in a fresh workdir, POST a record, kill
/// it; boot a SECOND process in the SAME workdir and require the record is still
/// there via GET — proving resources survive a restart (file-backed).
pub fn verify_resource_persists(bin: &Path, resource: &str) -> Result<(), String> {
    let workdir = unique_path("ncpu_resource_persist", "d");
    std::fs::create_dir_all(&workdir).map_err(|e| format!("mkdir workdir: {e}"))?;
    let run = |post: bool| -> Result<String, String> {
        let mut child = Command::new(bin)
            .arg("--port")
            .arg("0")
            .current_dir(&workdir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn: {e}"))?;
        let addr = read_ready_addr(&mut child)?;
        thread::sleep(Duration::from_millis(25));
        let r = (|| {
            if post {
                let resp = http_post(&addr, &format!("/{resource}"), "{\"name\":\"ada\"}")?;
                if !resp.contains("201") {
                    return Err(format!("POST not 201: {resp}"));
                }
            }
            http_get(&addr, &format!("/{resource}"))
        })();
        stop_child(&mut child);
        r
    };
    let result = (|| {
        run(true)?; // process 1: create a record, then dies
        let after = run(false)?; // process 2 (restart): read the collection
        if !after.contains("ada") {
            return Err(format!("record did NOT survive restart: {after}"));
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&workdir);
    result
}

fn probe_resource_once(bin: &Path, resource: &str, workdir: &Path) -> Result<(), String> {
    // Run in an isolated workdir: resources persist to <name>.jsonl in CWD, so
    // each probe gets a clean directory (no repo pollution, no parallel clash).
    let mut child = Command::new(bin)
        .arg("--port")
        .arg("0")
        .current_dir(workdir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn generated backend: {e}"))?;
    let addr = read_ready_addr(&mut child)?;
    thread::sleep(Duration::from_millis(25));
    let result = (|| {
        let record = "{\"name\":\"ada\"}";
        // CREATE
        let resp = http_post(&addr, &format!("/{resource}"), record)?;
        if !resp.contains("201") {
            return Err(format!("POST /{resource} not 201: {resp}"));
        }
        // READ-ALL
        let listed = http_get(&addr, &format!("/{resource}"))?;
        if !listed.contains("ada") {
            return Err(format!("posted record not visible via GET /{resource}: {listed}"));
        }
        // READ-ONE
        let one = request(&addr, "GET", &format!("/{resource}/0"), "")?;
        if !one.contains("ada") {
            return Err(format!("GET /{resource}/0 missing record: {one}"));
        }
        // UPDATE
        let upd = request(&addr, "PUT", &format!("/{resource}/0"), "{\"name\":\"bob\"}")?;
        if !upd.contains("updated") {
            return Err(format!("PUT /{resource}/0 not confirmed: {upd}"));
        }
        let one2 = request(&addr, "GET", &format!("/{resource}/0"), "")?;
        if !one2.contains("bob") || one2.contains("ada") {
            return Err(format!("record not updated: {one2}"));
        }
        // DELETE
        let del = request(&addr, "DELETE", &format!("/{resource}/0"), "")?;
        if !del.contains("deleted") {
            return Err(format!("DELETE /{resource}/0 not confirmed: {del}"));
        }
        // GONE
        let after = http_get(&addr, &format!("/{resource}"))?;
        if after.contains("ada") {
            return Err(format!("record still present after delete: {after}"));
        }
        Ok(())
    })();
    stop_child(&mut child);
    result
}

/// SINGLE-ARTIFACT STACK smoke: boot the generated backend with `--static
/// <dir>`, GET `/` and require 200 text/html containing `expect` (a page
/// marker, e.g. the site title). Proves one binary serves the generated site
/// AND its api. Also confirms the api still answers (`/health` 200).
pub fn verify_static_serving(
    bin: &Path,
    static_dir: &Path,
    expect: &str,
    max_attempts: usize,
) -> Result<(), String> {
    let mut last_err = String::new();
    for attempt in 0..max_attempts {
        match probe_static_once(bin, static_dir, expect) {
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
        "generated backend static serving failed after {max_attempts} attempts: {last_err}"
    ))
}

/// MULTI-PAGE serving: boot once with `--static <dir>` and GET every
/// (path, marker) pair, each requiring 200 text/html containing its marker.
/// Proves a multi-page site is fully served (inter-page nav targets resolve
/// over HTTP), not just the index.
pub fn verify_static_pages(
    bin: &Path,
    static_dir: &Path,
    pages: &[(&str, &str)],
    max_attempts: usize,
) -> Result<(), String> {
    let mut last_err = String::new();
    for attempt in 0..max_attempts {
        match probe_static_pages_once(bin, static_dir, pages) {
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
        "generated backend multi-page serving failed after {max_attempts} attempts: {last_err}"
    ))
}

fn probe_static_pages_once(
    bin: &Path,
    static_dir: &Path,
    pages: &[(&str, &str)],
) -> Result<(), String> {
    let mut child = Command::new(bin)
        .arg("--port")
        .arg("0")
        .arg("--static")
        .arg(static_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn generated backend: {e}"))?;

    let addr = read_ready_addr(&mut child)?;
    thread::sleep(Duration::from_millis(25));

    let result = (|| {
        for (path, marker) in pages {
            let resp = http_get(&addr, path)?;
            if !resp.contains("200 OK") {
                return Err(format!("GET {path} not 200: {resp}"));
            }
            if !resp.contains("text/html") {
                return Err(format!("GET {path} missing text/html: {resp}"));
            }
            if !resp.contains(*marker) {
                return Err(format!("GET {path} missing marker {marker:?}: {resp}"));
            }
        }
        Ok(())
    })();

    stop_child(&mut child);
    result
}

fn probe_static_once(bin: &Path, static_dir: &Path, expect: &str) -> Result<(), String> {
    let mut child = Command::new(bin)
        .arg("--port")
        .arg("0")
        .arg("--static")
        .arg(static_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn generated backend: {e}"))?;

    let addr = read_ready_addr(&mut child)?;
    thread::sleep(Duration::from_millis(25));

    let result = (|| {
        let page = http_get(&addr, "/")?;
        if !page.contains("200 OK") {
            return Err(format!("GET / not 200: {page}"));
        }
        if !page.contains("text/html") {
            return Err(format!("GET / missing text/html content-type: {page}"));
        }
        if !page.contains(expect) {
            return Err(format!("GET / missing expected marker {expect:?}: {page}"));
        }
        // The api must still answer alongside the served site.
        let health = http_get(&addr, "/health")?;
        if !health.contains("200 OK") || !health.contains("\"ok\":true") {
            return Err(format!("GET /health unexpected while serving static: {health}"));
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
