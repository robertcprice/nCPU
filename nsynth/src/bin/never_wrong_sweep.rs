//! NEVER-WRONG SWEEP — empirically hunt confident-wrong LEAKS + liveness HANGS across
//! ALL product categories, not just function synthesis.
//!
//! The never-wrong guarantee (verified-or-refused against examples + a distinguishing
//! gate) protects the function-synthesis tiers. The PROSE-ROUTED categories
//! (backend / cli / site / structure / component / tensor) instead commit on
//! keyword-match + a compile-gate — so a mis-classified prompt can ship a
//! compiling-but-wrong artifact CONFIDENTLY. (Measured: MBPP id158 `min_Ops` shipped
//! an empty server via backend-intake before the score-floor fix.)
//!
//! Feeds a corpus of ALGORITHMIC prompts (text only, no examples) to `handle_query`
//! and flags any that a NON-function category answers CONFIDENTLY (not `:tentative`).
//! Each handle_query runs in a worker thread with a wall-clock timeout so a single
//! hanging prompt (a liveness bug — also production-relevant) is flagged, not fatal.
//!
//! Usage: never_wrong_sweep <prompts.jsonl> [timeout_secs]
//! Exit 0 iff zero leaks AND zero hangs (usable as a CI gate).
use mog_synth::agent::{CodingAgentSession, GuardrailPolicy};
use std::sync::mpsc;
use std::time::Duration;

const APP_CATEGORIES: &[&str] = &[
    "backend-intake",
    "cli-intake",
    "site-domain",
    "structure-scaffold",
    "component",
    "tensor",
];

fn classify(prompt: &str) -> (bool, String) {
    let root = std::env::temp_dir().join(format!("nwsweep_{:x}", prompt.as_ptr() as usize));
    let _ = std::fs::create_dir_all(&root);
    let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
    // Per-query wall-clock bound (this runs in the worker thread; the guard is
    // thread-local). Matches the product path so the sweep measures the same
    // bounded behavior. Overridable via NSYNTH_QUERY_BUDGET_MS.
    let query_budget_ms: u64 = std::env::var("NSYNTH_QUERY_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20000);
    let _query_budget = mog_synth::synthesis::QuerySolveBudget::millis(query_budget_ms);
    let r = session.handle_query(prompt);
    let _ = std::fs::remove_dir_all(&root);
    let method = r.synthesis_method.clone().unwrap_or_else(|| format!("{:?}", r.route));
    (r.success, method)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: never_wrong_sweep <prompts.jsonl> [timeout_secs]");
    let timeout = Duration::from_secs(
        std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(20),
    );
    let text = std::fs::read_to_string(&path).expect("read prompts");
    let prompts: Vec<String> = text
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_string))
        .collect();
    eprintln!("never-wrong sweep: {} algorithmic prompts, {}s/prompt timeout", prompts.len(), timeout.as_secs());

    let (mut refused, mut tentative, mut fn_confident, mut leaks, mut hangs) = (0, 0, 0, 0, 0);
    let mut leak_list: Vec<(String, String)> = Vec::new();
    let mut hang_list: Vec<String> = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        if i % 50 == 0 {
            eprintln!("  ..{i}/{} (leaks={leaks} hangs={hangs})", prompts.len());
        }
        let p = prompt.clone();
        let (tx, rx) = mpsc::channel();
        // Detached worker: if it hangs, we abandon it and move on (it holds no shared state).
        std::thread::spawn(move || {
            let out = classify(&p);
            let _ = tx.send(out);
        });
        match rx.recv_timeout(timeout) {
            Ok((success, method)) => {
                let confident = success && !method.contains(":tentative");
                if !success {
                    refused += 1;
                } else if !confident {
                    tentative += 1;
                } else if APP_CATEGORIES.iter().any(|c| method.starts_with(c)) {
                    leaks += 1;
                    leak_list.push((prompt.clone(), method));
                } else {
                    fn_confident += 1;
                }
            }
            Err(_) => {
                hangs += 1;
                hang_list.push(prompt.clone());
            }
        }
    }

    eprintln!("──────────────────────────────────────");
    for (p, m) in &leak_list {
        eprintln!("LEAK [{m}] <- {:?}", p.chars().take(90).collect::<String>());
    }
    for p in &hang_list {
        eprintln!("HANG <- {:?}", p.chars().take(90).collect::<String>());
    }
    eprintln!(
        "prompts={} refused={refused} tentative={tentative} fn-confident={fn_confident} \
         APP-CATEGORY-CONFIDENT(leaks)={leaks} HANGS={hangs}",
        prompts.len()
    );
    std::process::exit(if leaks == 0 && hangs == 0 { 0 } else { 1 });
}
