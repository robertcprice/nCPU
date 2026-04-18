//! Hall of fame: the cached teachers that have transferred the most.
//!
//! Reads the persistent solved cache via `solved_cache::snapshot_solutions_with_meta`,
//! sorts by `(success_count desc, last_used_at desc)`, and prints the top-N
//! entries. This is a readable window into *which programs the system has
//! discovered generalize* — a teacher solving one problem whose
//! `success_count` has climbed to double-digits is encoding a pattern that
//! matched many other problems' I/O shapes.
//!
//! Two output modes:
//!   - Default: human-readable table (rank, success, method, code preview)
//!   - `--json`: one JSONL row per teacher for downstream processing
//!
//! Usage:
//!     cargo run --release --bin top_teachers -- \
//!         [--top N]                 default 20
//!         [--json]                  JSONL instead of human-readable
//!         [--min-success K]         hide entries with success_count < K

use mog_synth::solved_cache;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
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

/// Single-line preview: first 80 chars of the first non-blank code line.
/// Good enough for a human scan — users who want the full body can diff the
/// cache file directly.
fn code_preview(code: &str, limit: usize) -> String {
    let first_real_line = code
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim_start();
    if first_real_line.len() <= limit {
        first_real_line.to_string()
    } else {
        format!("{}…", &first_real_line[..limit])
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let top: usize = arg_value(&args, "--top")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let min_success: u32 = arg_value(&args, "--min-success")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let json_mode = has_flag(&args, "--json");

    let snapshot = solved_cache::snapshot_solutions_with_meta();
    if snapshot.is_empty() {
        eprintln!("[top_teachers] cache is empty — no teachers to rank");
        return;
    }

    // Sort by (success_count desc, last_used_at desc). BTreeMap snapshot
    // order is stable but not score-ordered; explicit sort here.
    let mut ranked: Vec<(String, String, u32, u64)> = snapshot
        .into_iter()
        .filter(|(_, _, sc, _)| *sc >= min_success)
        .collect();
    ranked.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.3.cmp(&a.3)));

    let total = ranked.len();
    let shown = ranked.iter().take(top).count();

    if json_mode {
        for (rank, (method, code, success_count, last_used_at)) in
            ranked.into_iter().take(top).enumerate()
        {
            println!(
                r#"{{"rank":{},"method":"{}","success_count":{},"last_used_at":{},"code":"{}"}}"#,
                rank + 1,
                json_escape(&method),
                success_count,
                last_used_at,
                json_escape(&code),
            );
        }
        eprintln!(
            "[top_teachers] emitted {}/{} teachers (min_success={})",
            shown, total, min_success
        );
    } else {
        eprintln!(
            "[top_teachers] showing top {} of {} cached teachers (min_success={})",
            shown, total, min_success
        );
        println!("{:<4}  {:<6}  {:<28}  {}", "rank", "wins", "method", "code");
        println!("{}", "─".repeat(100));
        for (rank, (method, code, success_count, _)) in ranked.into_iter().take(top).enumerate() {
            println!(
                "{:<4}  {:<6}  {:<28}  {}",
                rank + 1,
                success_count,
                if method.len() > 28 {
                    format!("{}…", &method[..27])
                } else {
                    method
                },
                code_preview(&code, 60),
            );
        }
    }
}
