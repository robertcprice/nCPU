//! Minimal HTTP API for nsynth.
//!
//! A tiny stdlib-only HTTP/1.1 server exposing one endpoint:
//!
//!   POST /synthesize
//!   Content-Type: application/json
//!   {
//!     "name": "double",
//!     "signature": "fn double(a: i64) -> i64",
//!     "examples": [{"inputs": [1], "expected": 2}, ...],
//!     "lang": "python"
//!   }
//!
//!   → 200 OK
//!   {
//!     "code": "def double(a: int): ...",
//!     "method": "search_polynomial_quadratic",
//!     "solve_ms": 4,
//!     "success": true
//!   }
//!
//! Single-threaded, thread-per-connection. No async runtime dependency —
//! keeps the crate's dep surface tight. Fine for a demo or low-traffic
//! service; wrap with a real ingress (nginx, axum, etc.) when you need
//! production load.
//!
//! Usage:
//!     cargo run --release --bin nsynth_serve -- [--port 7800] [--host 127.0.0.1]
//!
//!     curl -X POST http://localhost:7800/synthesize \
//!         -H 'Content-Type: application/json' \
//!         -d '{"name":"double","signature":"fn double(a: i64) -> i64",
//!              "examples":[{"inputs":[1],"expected":2},{"inputs":[5],"expected":10}],
//!              "lang":"python"}'

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

use serde::Deserialize;

use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::mog_transpile::{to_python, to_rust, to_typescript};
use mog_synth::solver::solve_problem;

#[derive(Deserialize, Debug)]
struct SynthesizeRequest {
    name: String,
    signature: String,
    examples: Vec<RequestExample>,
    #[serde(default = "default_lang")]
    lang: String,
}

#[derive(Deserialize, Debug)]
struct RequestExample {
    inputs: Vec<serde_json::Value>,
    expected: i64,
}

fn default_lang() -> String {
    "python".to_string()
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn value_from_json(v: &serde_json::Value) -> Option<Value> {
    if let Some(i) = v.as_i64() {
        return Some(Value::Int(i));
    }
    if let Some(arr) = v.as_array() {
        let ints: Option<Vec<i64>> = arr.iter().map(|x| x.as_i64()).collect();
        if let Some(ints) = ints {
            return Some(Value::Array(ints));
        }
    }
    if let Some(obj) = v.as_object() {
        if let Some(n) = obj.get("Int").and_then(|x| x.as_i64()) {
            return Some(Value::Int(n));
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let host = arg_value(&args, "--host").unwrap_or_else(|| "127.0.0.1".to_string());
    let port: u16 = arg_value(&args, "--port")
        .and_then(|v| v.parse().ok())
        .unwrap_or(7800);
    let addr = format!("{}:{}", host, port);
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(err) => {
            eprintln!("[nsynth_serve] cannot bind {addr}: {err}");
            std::process::exit(1);
        }
    };

    eprintln!("[nsynth_serve] listening on http://{}/", addr);
    eprintln!("[nsynth_serve] POST /synthesize with JSON body — see header comment");

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                // Thread-per-connection. Cheap at synthesis cadence
                // (one request ≈ ms-to-seconds of work); no need for
                // async runtime or connection pooling.
                std::thread::spawn(move || {
                    if let Err(err) = handle_connection(s) {
                        eprintln!("[nsynth_serve] conn error: {err}");
                    }
                });
            }
            Err(err) => eprintln!("[nsynth_serve] accept error: {err}"),
        }
    }
}

fn handle_connection(mut stream: TcpStream) -> std::io::Result<()> {
    let peer = stream.peer_addr().ok();
    let mut reader = BufReader::new(stream.try_clone()?);

    // Read request line.
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    // Headers.
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }

    // Route. We only implement POST /synthesize + GET /health;
    // everything else gets a plain 404.
    match (method.as_str(), path.as_str()) {
        ("POST", "/synthesize") => {
            let mut body = vec![0u8; content_length];
            reader.read_exact(&mut body)?;
            let body_str = String::from_utf8_lossy(&body);
            let resp = handle_synthesize(&body_str);
            eprintln!(
                "[nsynth_serve] {} → /synthesize {} bytes",
                peer.map(|p| p.to_string()).unwrap_or_else(|| "-".into()),
                content_length,
            );
            write_response(&mut stream, 200, "application/json", &resp)?;
        }
        ("GET", "/health") => {
            write_response(&mut stream, 200, "application/json", r#"{"ok":true}"#)?;
        }
        _ => {
            write_response(
                &mut stream,
                404,
                "application/json",
                r#"{"error":"unknown route — try POST /synthesize or GET /health"}"#,
            )?;
        }
    }
    Ok(())
}

fn handle_synthesize(body: &str) -> String {
    let parsed: SynthesizeRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(err) => {
            return error_response(&format!("parse: {err}"));
        }
    };
    // Build Problem from request.
    let mut examples = Vec::with_capacity(parsed.examples.len());
    for (i, ex) in parsed.examples.into_iter().enumerate() {
        let mut ins = Vec::with_capacity(ex.inputs.len());
        for v in ex.inputs {
            let Some(val) = value_from_json(&v) else {
                return error_response(&format!("example {i}: bad input value"));
            };
            ins.push(val);
        }
        examples.push(Example {
            inputs: ins,
            expected: ex.expected,
        });
    }
    if examples.is_empty() {
        return error_response("no examples provided");
    }
    let signature: &'static str = Box::leak(parsed.signature.into_boxed_str());
    let problem = Problem {
        name: parsed.name,
        category: "serve",
        description: "",
        signature,
        examples,
        holdouts: vec![],
        reference_code: "",
    };

    let t0 = Instant::now();
    let result = solve_problem(&problem);
    let solve_ms = t0.elapsed().as_millis() as u64;
    if !result.success {
        return format!(
            r#"{{"success":false,"solve_ms":{},"method":"{}","error":"{}"}}"#,
            solve_ms,
            escape_json(&result.method),
            escape_json(
                result
                    .error
                    .unwrap_or_else(|| "synthesis failed".to_string())
                    .as_str()
            ),
        );
    }

    let emitted = match parsed.lang.as_str() {
        "python" | "py" => to_python(&result.code),
        "rust" | "rs" => to_rust(&result.code),
        "typescript" | "ts" => to_typescript(&result.code),
        "mog" => result.code.clone(),
        other => {
            return error_response(&format!("unknown lang {other:?}"));
        }
    };

    format!(
        r#"{{"success":true,"solve_ms":{},"method":"{}","lang":"{}","code":"{}"}}"#,
        solve_ms,
        escape_json(&result.method),
        escape_json(&parsed.lang),
        escape_json(&emitted),
    )
}

fn error_response(msg: &str) -> String {
    format!(r#"{{"success":false,"error":"{}"}}"#, escape_json(msg))
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Server Error",
    };
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        status_text,
        content_type,
        body.len(),
        body,
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}
