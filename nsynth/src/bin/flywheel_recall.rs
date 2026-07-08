//! FLYWHEEL RECALL — prove a harvested learned store persists ACROSS PROCESSES.
//!
//! `flywheel_harvest` distils model-taught programs into a JSONL store, then
//! recovers them model-free IN THE SAME RUN. This bin closes the "…forever" half:
//! in a FRESH process, with NO model configured, load that store and solve each
//! task MODEL-OFF — a `library-learned:` method proves the engine permanently kept
//! the lesson on disk. try_learned RE-verifies each op against the task's own
//! examples before firing, so a hit is also a correctness check, never a blind load.
//!
//! Usage: NSYNTH_LEARNED_OPS_PATH=<store> flywheel_recall <tasks.jsonl>
//! (do NOT set NSYNTH_LOCAL_LLM_URL — the point is model-free recall.)
use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::linguigenesis_bridge::infer_signature;
use mog_synth::runtime::code_reproduces_examples;
use mog_synth::solver::solve_problem;

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
    (!examples.is_empty()).then_some((name, examples))
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: flywheel_recall <tasks.jsonl>");
    assert!(
        std::env::var("NSYNTH_LEARNED_OPS_PATH").is_ok(),
        "set NSYNTH_LEARNED_OPS_PATH to the harvested store"
    );
    assert!(
        std::env::var("NSYNTH_LOCAL_LLM_URL").is_err(),
        "unset NSYNTH_LOCAL_LLM_URL — recall must be MODEL-FREE"
    );
    let text = std::fs::read_to_string(&path).expect("read tasks");
    let tasks: Vec<(String, Vec<Example>)> = text.lines().filter_map(parse_task).collect();

    let (mut recalled, mut other, mut miss) = (0, 0, 0);
    for (name, all) in &tasks {
        let sig: &'static str = Box::leak(infer_signature(name, all).into_boxed_str());
        let problem = Problem { name: name.clone(), signature: sig, examples: all.clone(), ..Default::default() };
        let r = solve_problem(&problem);
        let ok = r.success && code_reproduces_examples(&r.code, all);
        if ok && r.method.starts_with("library-learned:") {
            recalled += 1;
            println!("  {name:<24} RECALLED model-free via {}", r.method);
        } else if ok {
            other += 1;
            println!("  {name:<24} solved but via {} (not the distilled op)", r.method);
        } else {
            miss += 1;
            println!("  {name:<24} NOT solved model-free ({})", r.method);
        }
    }
    println!("──────────────────────────────────────");
    println!("tasks={} RECALLED-from-store={recalled} solved-otherwise={other} missed={miss}", tasks.len());
}
