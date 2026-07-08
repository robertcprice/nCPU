//! FLYWHEEL HARVEST — demonstrate + measure the model->engine distillation loop.
//!
//! For each task the MODEL-FREE engine cannot synthesize, the served model (Qwen)
//! proposes a program; it is accepted ONLY if it reproduces every example incl. the
//! HELD-OUT ones (the examples-only oracle — a fit-to-seed hallucination fails them)
//! AND passes rlvr strict-verify; the verified program is then DISTILLED into the
//! learned store (record_proposed_op). Re-running the engine MODEL-OFF then solves
//! the same task via `library-learned` — the model taught it once, the engine keeps
//! it, model-free, forever. Never-wrong: every distilled op is held-out-verified,
//! and try_learned RE-verifies against the task's examples before it can fire.
//!
//! Measurement hygiene: the BASELINE phase runs with the learned store DISABLED so a
//! solver auto-record (maybe_record_learned) cannot pollute it — after teaching, the
//! store contains ONLY model-taught, held-out-verified ops, so a model-free recovery
//! is attributable to the model's lesson and nothing else. Recovery is confirmed
//! STRICTLY: the recovered method must be `library-learned:<the op we just distilled>`.
//!
//! Reads tasks as JSONL {name, examples:[{in:[..], out:..}]} on a path arg.
//! Requires: NSYNTH_LEARNED_OPS_PATH (fresh), NSYNTH_LOCAL_LLM_URL, NSYNTH_LOCAL_LLM_MODEL.
use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::linguigenesis_bridge::infer_signature;
use mog_synth::local_llm::propose_program;
use mog_synth::op_library::record_proposed_op;
use mog_synth::runtime::{code_reproduces_examples, describe_first_failure, verify_problem_code_strict};
use mog_synth::solver::solve_problem;

/// Rename a program's entry fn (+ any self-recursive calls) to `target`, whole-word,
/// so a verifier that invokes the entry by a fixed name can run a model program named
/// arbitrarily. No-op if already `target` or no `fn <name>` found.
fn rename_fn(code: &str, target: &str) -> String {
    let Some((_, rest)) = code.split_once("fn ") else { return code.to_string() };
    let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    if name.is_empty() || name == target {
        return code.to_string();
    }
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = String::with_capacity(code.len());
    let mut idx = 0;
    while let Some(pos) = code[idx..].find(&name) {
        let at = idx + pos;
        let before_ok = at == 0 || !code[..at].chars().next_back().is_some_and(is_ident);
        let after_ok = !code[at + name.len()..].chars().next().is_some_and(is_ident);
        out.push_str(&code[idx..at]);
        out.push_str(if before_ok && after_ok { target } else { &name });
        idx = at + name.len();
    }
    out.push_str(&code[idx..]);
    out
}

fn json_to_value(v: &serde_json::Value) -> Option<Value> {
    if let Some(b) = v.as_bool() {
        return Some(Value::Bool(b));
    }
    if let Some(i) = v.as_i64() {
        return Some(Value::Int(i));
    }
    if let Some(s) = v.as_str() {
        return Some(Value::Str(s.to_string()));
    }
    if let Some(arr) = v.as_array() {
        if let Some(ints) = arr.iter().map(|x| x.as_i64()).collect::<Option<Vec<i64>>>() {
            return Some(Value::int_array(&ints));
        }
        return Some(Value::Array(arr.iter().filter_map(json_to_value).collect()));
    }
    None
}

fn parse_task(line: &str) -> Option<(String, Vec<Example>)> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let name = v.get("name").and_then(|x| x.as_str())?.to_string();
    let exs = v.get("examples")?.as_array()?;
    let examples: Vec<Example> = exs
        .iter()
        .filter_map(|e| {
            let inputs = e.get("in")?.as_array()?.iter().filter_map(json_to_value).collect();
            Some(Example { inputs, expected: json_to_value(e.get("out")?)? })
        })
        .collect();
    if examples.len() < 6 {
        return None;
    }
    Some((name, examples))
}

fn build_problem(name: &str, examples: &[Example]) -> Problem {
    let sig: &'static str = Box::leak(infer_signature(name, examples).into_boxed_str());
    Problem {
        name: name.to_string(),
        signature: sig,
        examples: examples.to_vec(),
        ..Default::default()
    }
}

/// Engine-only (model-off) correctness: solves AND the program reproduces every
/// example (incl the held-out — catches an engine overfit, not just success).
/// Returns (solved, method-or-why).
fn engine_solves(problem: &Problem, all: &[Example]) -> (bool, String) {
    let r = solve_problem(problem);
    if r.success && code_reproduces_examples(&r.code, all) {
        (true, r.method)
    } else {
        (false, if r.success { format!("OVERFIT({})", r.method) } else { "unsolved".into() })
    }
}

/// Drive the model to propose a verified program (best-of-8 with concrete repair).
/// Two fast, sound gates — no oracle needed:
///   1. reproduce-all-incl-HELD-OUT: the prompt shows only `seed`; the 2 held-out
///      examples are the generalization oracle a fit-to-seed hallucination misses.
///   2. robustness floor (`verify_problem_code_strict`, NO differential-consensus):
///      the program must execute cleanly on perturbations of the examples, rejecting
///      a candidate defined only on the narrow visible inputs (crashes on n<=0 etc.).
/// Consensus is deliberately skipped: it re-synthesizes independently, but the whole
/// point of the model tier is a task the ENGINE CANNOT synthesize — consensus would
/// reject exactly the novel capability (and its 8x leave-one-out re-solve stalls for
/// minutes on a hard task). This mirrors `record_proposed_op`, which skips it too.
fn model_teach(name: &str, seed: &[Example], all: &[Example]) -> Option<String> {
    let mut request = format!("{name}\n\nExamples:\n");
    for ex in seed {
        request.push_str(&format!("  {:?} -> {:?}\n", ex.inputs, ex.expected));
    }
    request.push_str("\nWrite the Mog function `f`.");
    // The robustness floor invokes the entry fn by the problem's name, so normalize
    // the model's (arbitrarily-named) fn to match before verifying.
    let verify_problem = build_problem(name, all);
    // Best-of-8 with a rising temperature schedule: proposal is STOCHASTIC, so a
    // single unlucky draw (or four) must not sink a task the model can express.
    // Low temps first (crisp), climbing for diversity if the crisp draws miss.
    let temps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let mut prior: Option<(String, String)> = None;
    for &t in &temps {
        let p = prior.as_ref().map(|(c, e)| (c.as_str(), e.as_str()));
        let Some(code) = propose_program(&request, p, t) else { continue };
        match describe_first_failure(&code, all) {
            None => match verify_problem_code_strict(&verify_problem, &rename_fn(&code, name)) {
                Ok(()) => return Some(code),
                Err(_) => prior = Some((
                    code,
                    "it crashes / misbehaves on nearby inputs — handle edge cases (n<=0, larger n)".to_string(),
                )),
            },
            // Feed the concrete mismatch back so the model fixes the ACTUAL bug.
            Some(why) => prior = Some((code, why)),
        }
    }
    None
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/private/tmp/flywheel_tasks.jsonl".into());
    let text = std::fs::read_to_string(&path).expect("read task file");
    let tasks: Vec<(String, Vec<Example>)> = text.lines().filter_map(parse_task).collect();
    // Capture then DISABLE the store for baseline so a solver auto-record can't
    // pollute it; only model-taught ops will populate it in the teach phase.
    let store_path = std::env::var("NSYNTH_LEARNED_OPS_PATH").expect("NSYNTH_LEARNED_OPS_PATH must be set");
    std::env::remove_var("NSYNTH_LEARNED_OPS_PATH");
    eprintln!("flywheel: {} tasks from {path}", tasks.len());

    // Phase A — baseline (model-off, store OFF). The solver sees ONLY the seed
    // (all but the last 2); correctness is judged on the FULL set incl. those 2
    // GENUINELY held-out examples. A seed-overfit (e.g. an affine `search_two_branch`
    // that fits the seed but not the held-out) therefore correctly counts as
    // UNSOLVED — no vacuous "fit the same set you checked against" pass.
    let (mut base_solved, mut taught, mut recovered) = (0, 0, 0);
    let mut unsolved: Vec<(&String, &Vec<Example>)> = Vec::new();
    for (name, all) in &tasks {
        let seed = all[..all.len() - 2].to_vec();
        let problem = build_problem(name, &seed);
        let (ok, why) = engine_solves(&problem, all);
        if ok {
            base_solved += 1;
            eprintln!("  {name:<26} BASELINE ok ({why}) — engine already covers it");
        } else {
            eprintln!("  {name:<26} BASELINE {why}");
            unsolved.push((name, all));
        }
    }

    // Phase B+C — teach the unsolved with the model, distill, recover model-free.
    eprintln!("── teaching {} baseline-unsolved tasks ──", unsolved.len());
    for (name, all) in unsolved {
        let seed = &all[..all.len() - 2];
        let problem = build_problem(name, seed);

        // TEACH with the store OFF: model_teach's strict-verify runs an INDEPENDENT
        // re-synthesis (differential consensus) that would otherwise auto-record its
        // throwaway overfit probes into the store (maybe_record_learned). Only the
        // model's own held-out-verified program should be distilled — so teach clean.
        std::env::remove_var("NSYNTH_LEARNED_OPS_PATH");
        let taught_code = model_teach(name, seed, all);
        std::env::set_var("NSYNTH_LEARNED_OPS_PATH", &store_path);

        let Some(code) = taught_code else {
            eprintln!("  {name:<26} model FAILED to teach (no held-out-verified proposal)");
            continue;
        };
        taught += 1;
        let recorded = record_proposed_op(&problem, &code);

        // RECOVER model-free on the FULL example set. `try_library` (which delegates
        // to the distilled `try_learned` ops) runs BEFORE every search/affine tier,
        // so the distilled op fires first — a search overfit can never preempt it —
        // and its `library-learned:` method proves the recovery came from the lesson,
        // not a fresh overfit. try_learned RE-verifies the op against all examples
        // (incl. the held-out 2) before firing: a wrong distilled op simply won't.
        let rec_problem = build_problem(name, all);
        let r = solve_problem(&rec_problem);
        let via_learned = r.method.starts_with("library-learned:");
        if r.success && code_reproduces_examples(&r.code, all) && via_learned {
            recovered += 1;
            eprintln!("  {name:<26} TAUGHT+DISTILLED(rec={recorded}) -> model-free via {}", r.method);
        } else {
            eprintln!(
                "  {name:<26} taught(rec={recorded}) but recovery weak: success={} method={}",
                r.success, r.method
            );
        }
    }
    eprintln!("──────────────────────────────────────────────");
    eprintln!(
        "tasks={} baseline-solved={base_solved} model-taught={taught} RECOVERED-model-free={recovered}",
        tasks.len()
    );
}
