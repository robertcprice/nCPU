//! Phase C — harvest a VERIFIED (task -> Mog) SFT corpus to teach a proposer the Mog
//! dialect (the thing Gemma-2-2b provably can't do untuned). Emits mlx_lm.lora chat
//! records (JSONL) to stdout. Every `assistant` program is real, proven Mog — the
//! library reference implementations plus engine self-play solves — so the corpus is
//! guaranteed-correct by construction.
//!
//! Sources:
//!   1. Every library op: (its name as an NL request) -> its proven Mog program.
//!   2. Self-play: generate a task by RUNNING a random op on random inputs, solve it
//!      with the engine, and if the solve reproduces every example, harvest
//!      (task -> the engine's verified Mog). Adds phrasing/solution diversity.
//!
//! Usage: nl_harvest [selfplay_iters] > mog_sft.jsonl
use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::local_llm::training_record;
use mog_synth::op_library::OPS;
use mog_synth::runtime::{benchmark_value_from_runtime, code_reproduces_examples, execute_function};
use mog_synth::solver::solve_problem;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 16
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo).max(1) as u64) as i64
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[(self.next() as usize) % xs.len()]
    }
}

fn param_types(mog: &str) -> Vec<String> {
    let (Some(o), Some(c)) = (mog.find('('), mog.find(')')) else { return vec![] };
    let inner = mog[o + 1..c].trim();
    if inner.is_empty() {
        return vec![];
    }
    inner.split(',').filter_map(|p| p.split(':').nth(1).map(|t| t.trim().to_string())).collect()
}

fn gen_value(ty: &str, rng: &mut Lcg) -> Option<Value> {
    match ty {
        "i64" => Some(Value::Int(rng.range(-9, 40))),
        "bool" => Some(Value::Bool(rng.next() % 2 == 0)),
        "string" => Some(Value::Str(
            (0..rng.range(1, 6)).map(|_| (b'a' + (rng.next() % 26) as u8) as char).collect(),
        )),
        "[i64]" => Some(Value::int_array(&(0..rng.range(1, 6)).map(|_| rng.range(-9, 20)).collect::<Vec<_>>())),
        _ => None,
    }
}

fn entry_name(mog: &str) -> &str {
    mog.split("fn ").nth(1).and_then(|s| s.split('(').next()).map(str::trim).unwrap_or("f")
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let mut seen = std::collections::HashSet::new();
    let mut emitted = 0usize;
    let mut emit = |request: &str, code: &str| {
        // Dedup by the exact program so the corpus isn't dominated by one op.
        if seen.insert(format!("{request}\u{1f}{code}")) {
            println!("{}", training_record(request, code));
            true
        } else {
            false
        }
    };

    // 1. Library ops — clean (name -> proven Mog) pairs.
    let mut lib = 0;
    for op in OPS {
        let request = format!("compute {}", op.name.replace('_', " "));
        if emit(&request, op.mog) {
            lib += 1;
            emitted += 1;
        }
    }

    // 2. Self-play: generate -> solve -> harvest the engine's verified program.
    let mut rng = Lcg(0xD1B54A32D192ED03);
    let mut selfplay = 0;
    for _ in 0..iters {
        let op = rng.pick(OPS);
        let tys = param_types(op.mog);
        if tys.is_empty() || tys.iter().any(|t| gen_value(t, &mut Lcg(1)).is_none()) {
            continue;
        }
        let name = entry_name(op.mog);
        let mut examples: Vec<Example> = Vec::new();
        let mut att = 0;
        while examples.len() < 5 && att < 30 {
            att += 1;
            let inputs: Vec<Value> = tys.iter().filter_map(|t| gen_value(t, &mut rng)).collect();
            if inputs.len() != tys.len() {
                break;
            }
            if let Ok(o) = execute_function(op.mog, name, &inputs, "harvest") {
                if let Ok(e) = benchmark_value_from_runtime(&o) {
                    examples.push(Example { inputs, expected: e });
                }
            }
        }
        if examples.len() < 4 {
            continue;
        }
        let sig: &'static str = Box::leak(
            mog_synth::linguigenesis_bridge::infer_signature("f", &examples).into_boxed_str(),
        );
        let problem = Problem {
            name: "f".to_string(),
            signature: sig,
            examples: examples.clone(),
            ..Default::default()
        };
        let res = solve_problem(&problem);
        if res.success && code_reproduces_examples(&res.code, &examples) {
            let request = format!("compute {}", op.name.replace('_', " "));
            if emit(&request, &res.code) {
                selfplay += 1;
                emitted += 1;
            }
        }
    }

    eprintln!("nl_harvest: {emitted} verified (task -> Mog) pairs  (library {lib}, self-play {selfplay})");
}
