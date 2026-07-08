//! NEVER-WRONG SWEEP — empirically hunt confident-wrong LEAKS across ALL product
//! categories, not just function synthesis.
//!
//! The never-wrong guarantee (verified-or-refused against examples + a distinguishing
//! gate) protects the function-synthesis tiers. The PROSE-ROUTED categories
//! (backend / cli / site / structure / component / tensor) instead commit on
//! keyword-match + a compile-gate — so a mis-classified prompt can ship a
//! compiling-but-wrong artifact CONFIDENTLY. (Measured: MBPP id158 `min_Ops` shipped
//! an empty server via backend-intake before the score-floor fix.)
//!
//! This bin feeds a corpus of ALGORITHMIC prompts (text only, no examples) to
//! `handle_query` and flags any that a NON-function category answers CONFIDENTLY
//! (not `:tentative`). Every such hit is a candidate never-wrong leak: an algorithm
//! prompt has a real answer that is NOT a web app / server / CLI scaffold, so a
//! confident app-category answer is a mis-classification.
//!
//! Usage: never_wrong_sweep <prompts.jsonl>   (each line {text: "..."} or {"text","fn"})
//! Exit code 0 iff zero leaks (usable as a CI gate).
use mog_synth::agent::{CodingAgentSession, GuardrailPolicy};

const APP_CATEGORIES: &[&str] = &[
    "backend-intake",
    "cli-intake",
    "site-domain",
    "structure-scaffold",
    "component",
    "tensor",
];

fn main() {
    let path = std::env::args().nth(1).expect("usage: never_wrong_sweep <prompts.jsonl>");
    let text = std::fs::read_to_string(&path).expect("read prompts");
    let prompts: Vec<String> = text
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_string))
        .collect();
    eprintln!("never-wrong sweep: {} algorithmic prompts from {path}", prompts.len());

    let (mut refused, mut tentative, mut fn_confident, mut leaks) = (0, 0, 0, 0);
    let mut leak_list: Vec<(String, String)> = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let root = std::env::temp_dir().join(format!("nwsweep_{i}_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&root);
        let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        let r = session.handle_query(prompt);
        let _ = std::fs::remove_dir_all(&root);

        let method = r.synthesis_method.clone().unwrap_or_else(|| format!("{:?}", r.route));
        let confident = r.success && !method.contains(":tentative");
        if !r.success {
            refused += 1;
        } else if !confident {
            tentative += 1;
        } else if APP_CATEGORIES.iter().any(|c| method.starts_with(c)) {
            // A confident APP-category answer to an algorithmic prompt = leak candidate.
            leaks += 1;
            leak_list.push((prompt.clone(), method));
        } else {
            fn_confident += 1;
        }
    }

    eprintln!("──────────────────────────────────────");
    for (p, m) in &leak_list {
        eprintln!("LEAK [{m}] <- {:?}", p.chars().take(90).collect::<String>());
    }
    eprintln!(
        "prompts={} refused={refused} tentative={tentative} fn-confident={fn_confident} \
         APP-CATEGORY-CONFIDENT(leaks)={leaks}",
        prompts.len()
    );
    // CI gate: any confident app-category answer to an algorithmic prompt is a leak.
    std::process::exit(if leaks == 0 { 0 } else { 1 });
}
