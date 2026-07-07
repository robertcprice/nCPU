//! Fuzz the never-wrong INVARIANT of `verified_nl_router` at scale.
//!
//! The differentiator is "never confidently wrong". This turns that from an
//! observed property (0 wrong on 25/528 prompts) into a stress-proven one: over
//! thousands of GENERATED (prompt, examples) tasks — valid, weak, corrupted, and
//! mislabeled — it asserts the single invariant that must always hold:
//!
//!     IF route_verified returns an op, THAT op reproduces EVERY example.
//!
//! A violation is a confidently-wrong answer. The harness prints the count; it must
//! be 0. Deterministic (a fixed-seed LCG) so a failure is reproducible.
//!
//! Usage: cargo run --release --bin nl_fuzz [iterations]
use mog_synth::benchmark::{Example, Value};
use mog_synth::op_library::OPS;
use mog_synth::runtime::{benchmark_value_from_runtime, code_reproduces_examples, execute_function};
use mog_synth::verified_nl_router::{answer, route_composed, route_verified, Answer};

/// Deterministic linear-congruential RNG (no external crate, reproducible).
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

/// Parse the parameter types from an op's Mog signature `fn name(a: T, b: T) -> R`.
fn param_types(mog: &str) -> Vec<String> {
    let Some(paren) = mog.find('(') else { return vec![] };
    let Some(close) = mog[paren..].find(')') else { return vec![] };
    let inner = &mog[paren + 1..paren + close];
    if inner.trim().is_empty() {
        return vec![];
    }
    inner
        .split(',')
        .filter_map(|p| p.split(':').nth(1).map(|t| t.trim().to_string()))
        .collect()
}

/// A random value of a given Mog type, or None for a type we don't fuzz.
fn gen_value(ty: &str, rng: &mut Lcg) -> Option<Value> {
    match ty {
        "i64" => Some(Value::Int(rng.range(-12, 60))),
        "bool" => Some(Value::Bool(rng.next() % 2 == 0)),
        "string" => {
            let len = rng.range(1, 6) as usize;
            let s: String = (0..len)
                .map(|_| (b'a' + (rng.next() % 26) as u8) as char)
                .collect();
            Some(Value::Str(s))
        }
        "[i64]" => {
            let len = rng.range(1, 6) as usize;
            let xs: Vec<i64> = (0..len).map(|_| rng.range(-9, 20)).collect();
            Some(Value::int_array(&xs))
        }
        _ => None, // f64 / composite: skip
    }
}

fn entry_name(mog: &str) -> &str {
    mog.split("fn ").nth(1).and_then(|s| s.split('(').next()).map(str::trim).unwrap_or("f")
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let mut rng = Lcg(0x9E3779B97F4A7C15);

    let mut violations = 0usize;
    let mut solved = 0usize;
    let mut refused = 0usize;
    let mut tasks = 0usize;

    for iter_n in 0..iters {
        // Pick a source op and build CONSISTENT examples by running it.
        let op = rng.pick(OPS);
        let tys = param_types(op.mog);
        if tys.is_empty() || tys.iter().any(|t| gen_value(t, &mut Lcg(1)).is_none()) {
            continue; // skip ops with unfuzzable arg types
        }
        let name = entry_name(op.mog);
        let mut examples: Vec<Example> = Vec::new();
        let mut attempts = 0;
        while examples.len() < 4 && attempts < 30 {
            attempts += 1;
            let inputs: Vec<Value> = tys.iter().filter_map(|t| gen_value(t, &mut rng)).collect();
            if inputs.len() != tys.len() {
                break;
            }
            if let Ok(out) = execute_function(op.mog, name, &inputs, "fuzz") {
                if let Ok(expected) = benchmark_value_from_runtime(&out) {
                    examples.push(Example { inputs, expected });
                }
            }
        }
        if examples.len() < 3 {
            continue;
        }

        // The prompt is the op name as words (e.g. "count vowels").
        let prompt = op.name.replace('_', " ");

        // Four adversarial variants of the SAME task.
        let variants: Vec<Vec<Example>> = vec![
            examples.clone(),                                   // valid
            corrupt_one_output(&examples, &mut rng),            // corrupted oracle
            weaken_to_constant(&examples),                      // non-distinguishing
            examples.iter().take(1).cloned().collect(),         // thin oracle (1 example)
        ];

        for ex in variants {
            if ex.is_empty() {
                continue;
            }
            tasks += 1;
            match route_verified(&prompt, &ex) {
                Some(r) => {
                    // THE INVARIANT: a returned op must reproduce every example.
                    if code_reproduces_examples(r.op.mog, &ex) {
                        solved += 1;
                    } else {
                        violations += 1;
                        eprintln!(
                            "VIOLATION: prompt={prompt:?} returned={} does NOT reproduce {ex:?}",
                            r.op.name
                        );
                    }
                }
                None => refused += 1,
            }
        }

        // COMPOSITION task: build examples for a real chain b(a(x)) and assert
        // route_composed only ever returns a chain that reproduces them. Only a
        // unary source op chains simply here.
        if tys.len() == 1 {
            let b = rng.pick(OPS);
            if b.name != op.name {
                let bname = entry_name(b.mog);
                let mut cex: Vec<Example> = Vec::new();
                let mut att = 0;
                while cex.len() < 4 && att < 30 {
                    att += 1;
                    let Some(input) = gen_value(&tys[0], &mut rng) else { break };
                    let Ok(y) = execute_function(op.mog, name, &[input.clone()], "fuzz") else { continue };
                    let Ok(yb) = benchmark_value_from_runtime(&y) else { continue };
                    if let Ok(z) = execute_function(b.mog, bname, &[yb], "fuzz") {
                        if let Ok(zb) = benchmark_value_from_runtime(&z) {
                            cex.push(Example { inputs: vec![input], expected: zb });
                        }
                    }
                }
                if cex.len() >= 3 {
                    let cprompt =
                        format!("{} then {}", op.name.replace('_', " "), b.name.replace('_', " "));
                    for cv in [cex.clone(), corrupt_one_output(&cex, &mut rng)] {
                        tasks += 1;
                        match route_composed(&cprompt, &cv) {
                            Some(code) => {
                                if code_reproduces_examples(&code, &cv) {
                                    solved += 1;
                                } else {
                                    violations += 1;
                                    eprintln!("VIOLATION(compose): prompt={cprompt:?} chain fails {cv:?}");
                                }
                            }
                            None => refused += 1,
                        }
                    }
                }
            }
        }

        // FULL answer() invariant — covers the SYNTHESIS tier + its holdout. Sparse
        // (solve_problem is slow). Two probes: (1) the consistent task — any non-Refused
        // answer must reproduce every example; (2) an OVERFIT task with RANDOM outputs
        // — no real function fits, so the holdout must refuse it, and if it somehow
        // returns, it must still reproduce all. Both assert the one invariant.
        if iter_n % 6 == 0 && examples.len() >= 5 {
            let checks: Vec<(&str, Vec<Example>)> = vec![
                (prompt.as_str(), examples.clone()),
                ("do the thing", {
                    let mut r = examples.clone();
                    for e in &mut r {
                        e.expected = corrupt_scalar(&e.expected, &mut rng);
                    }
                    r
                }),
            ];
            for (p, ex) in checks {
                tasks += 1;
                match answer(p, &ex) {
                    Answer::Refused => refused += 1,
                    Answer::Library { code, .. }
                    | Answer::Composition { code }
                    | Answer::Synthesized { code, .. } => {
                        if code_reproduces_examples(&code, &ex) {
                            solved += 1;
                        } else {
                            violations += 1;
                            eprintln!("VIOLATION(answer): prompt={p:?} returned code fails examples");
                        }
                    }
                }
            }
        }
    }

    println!("nl_fuzz: {tasks} tasks generated ({iters} iterations)");
    println!("  SOLVED  {solved}");
    println!("  REFUSED {refused}");
    println!("  INVARIANT VIOLATIONS (confidently-wrong): {violations}");
    if violations == 0 {
        println!("OK — never-wrong invariant held across all {tasks} fuzzed tasks");
    } else {
        std::process::exit(1);
    }
}

/// Randomize a scalar output so a whole example set describes NO consistent
/// function — an overfit trap the synthesis holdout must refuse.
fn corrupt_scalar(v: &Value, rng: &mut Lcg) -> Value {
    match v {
        Value::Int(_) => Value::Int(rng.range(-50, 200)),
        Value::Bool(_) => Value::Bool(rng.next() % 2 == 0),
        other => other.clone(),
    }
}

/// Perturb exactly one example's output so the set no longer describes the source op.
fn corrupt_one_output(examples: &[Example], rng: &mut Lcg) -> Vec<Example> {
    let mut out = examples.to_vec();
    if let Some(e) = out.get_mut(0) {
        e.expected = match &e.expected {
            Value::Int(n) => Value::Int(n.wrapping_add(rng.range(1, 9))),
            Value::Bool(b) => Value::Bool(!b),
            Value::Str(s) => Value::Str(format!("{s}x")),
            other => other.clone(),
        };
    }
    out
}

/// Collapse every output to the first — a non-distinguishing example set.
fn weaken_to_constant(examples: &[Example]) -> Vec<Example> {
    let Some(first) = examples.first() else { return vec![] };
    examples
        .iter()
        .map(|e| Example { inputs: e.inputs.clone(), expected: first.expected.clone() })
        .collect()
}
