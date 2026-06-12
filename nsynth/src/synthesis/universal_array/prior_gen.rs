//! Prior-net training data generator (ROADMAP Rung 9, Phase A, stage A1).
//!
//! Samples random programs from the universal-array space — both the
//! hand-coded restart shapes the solver actually searches (restarts 1..=25
//! in `synthesize_universal_array_fallback`) and the pure-random emergent
//! sampler (`random_bias_init`) — executes their *discretized* Mog code on
//! random inputs through the exact same runtime the solver's verifier uses,
//! and emits clean `(examples -> discrete program description)` JSONL rows.
//!
//! Semantics guarantee: every row's `expected` values come from
//! `execute_program` on the wrapped Mog code (the `verify_problem_code_via_main`
//! path), so a prior net trained on this data is trained against the exact
//! executor that will later verify its proposals. Degenerate programs
//! (execution errors, constant output across all inputs, runaway magnitudes)
//! are rejected and counted.

use super::*;
use std::collections::HashMap;
use std::io::Write as IoWrite;

/// Candidate values for the three non-anchor constant-pool slots. The pool
/// is always `[0, 1, -1, c3, c4, c5]` with `c3..c5` drawn (distinct) from
/// this list — a closed vocabulary so the prior net can predict constant
/// values with a classification head instead of unbounded regression.
pub const CONST_CANDIDATES: [i64; 17] = [
    2, -2, 3, -3, 4, -4, 5, -5, 6, 7, 8, 9, 10, 12, 15, 16, 20,
];

/// Cap on rows per distinct discrete program. Without it the dataset is
/// dominated by a handful of attractor shapes (sum, max, identity).
const MAX_ROWS_PER_CODE: usize = 100;

/// Reject rows whose outputs explode — they aren't realistic synthesis
/// targets and they wreck the value-encoding of the net's tokenizer.
const MAX_OUTPUT_MAG: i64 = 1_000_000;

/// Input distribution: array lengths 1..=12, element values in
/// [-MAX_ELEM, MAX_ELEM] (mixed sign), scalar args in [-MAX_SCALAR, MAX_SCALAR].
const MAX_ARR_LEN: usize = 12;
const MAX_ELEM: i64 = 20;
const MAX_SCALAR: i64 = 10;

#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub attempts: usize,
    pub written: usize,
    pub rejected_exec_error: usize,
    pub rejected_constant: usize,
    pub rejected_magnitude: usize,
    pub rejected_dup_cap: usize,
    pub distinct_codes: usize,
    pub hand_rows: usize,
    pub random_rows: usize,
}

impl GenStats {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "attempts": self.attempts,
            "written": self.written,
            "rejected_exec_error": self.rejected_exec_error,
            "rejected_constant": self.rejected_constant,
            "rejected_magnitude": self.rejected_magnitude,
            "rejected_dup_cap": self.rejected_dup_cap,
            "distinct_codes": self.distinct_codes,
            "hand_rows": self.hand_rows,
            "random_rows": self.random_rows,
        })
    }
}

/// SplitMix64 — small deterministic RNG so generation is reproducible from
/// a single seed without external deps.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform integer in `[lo, hi]` inclusive.
    fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next_u64() % ((hi - lo + 1) as u64)) as i64
    }

    fn chance(&mut self, p: f64) -> bool {
        (self.next_u64() as f64 / u64::MAX as f64) < p
    }
}

fn scalar_names_for(n_scalar: usize) -> Vec<&'static str> {
    match n_scalar {
        0 => vec![],
        1 => vec!["k"],
        _ => vec!["a", "b", "c", "d", "e", "f"]
            .into_iter()
            .take(n_scalar)
            .collect(),
    }
}

/// Sample the constant pool: anchors [0, 1, -1] plus three distinct values
/// from [`CONST_CANDIDATES`]. With probability 0.4 use the historical
/// default tail (2, -2, 10) so the distribution stays close to what the
/// solver's `DEFAULT_CONSTS`-era biases produced.
fn sample_consts(rng: &mut SplitMix64) -> [i64; N_CONSTS] {
    let mut out = [0i64, 1, -1, 2, -2, 10];
    if rng.chance(0.6) {
        let mut picked: Vec<i64> = Vec::with_capacity(3);
        while picked.len() < 3 {
            let c = CONST_CANDIDATES[rng.below(CONST_CANDIDATES.len())];
            if !picked.contains(&c) {
                picked.push(c);
            }
        }
        out[3] = picked[0];
        out[4] = picked[1];
        out[5] = picked[2];
    }
    out
}

/// Apply 1..=2 random extra spikes on top of a base shape — mutates a slot
/// field, a body-init pointer, or the return pointer. Spike 5.0 outranks the
/// shape spikes (4.0) so mutations actually change the argmax structure,
/// giving the dataset diversity *around* the realistic shapes.
fn mutate(prog: &mut SoftUniversalArrayProgram, rng: &mut SplitMix64) {
    let pool = prog.pool();
    let lip = prog.lip();
    let n_mut = 1 + rng.below(2);
    for _ in 0..n_mut {
        match rng.below(9) {
            7 => {
                let bs = rng.below(N_ARR_BODY);
                let io = prog.body_init_off(bs);
                let idx = rng.below(lip);
                prog.params[io + idx] = 5.0;
            }
            8 => {
                let ro = prog.return_off();
                prog.params[ro + rng.below(pool)] = 5.0;
            }
            field => {
                let slot = rng.below(N_ARR_SLOTS);
                let off = prog.slot_off(slot);
                let cb = off + N_OPS + 1 + 2 * pool;
                match field {
                    0 => prog.params[off + rng.below(N_OPS + 1)] = 5.0,
                    1 => prog.params[off + N_OPS + 1 + rng.below(pool)] = 5.0,
                    2 => prog.params[off + N_OPS + 1 + pool + rng.below(pool)] = 5.0,
                    3 => prog.params[cb + rng.below(N_CMPS)] = 5.0,
                    4 => prog.params[cb + N_CMPS + rng.below(pool)] = 5.0,
                    5 => prog.params[cb + N_CMPS + pool + rng.below(pool)] = 5.0,
                    _ => prog.params[cb + N_CMPS + 2 * pool + rng.below(pool)] = 5.0,
                }
            }
        }
    }
}

/// Build a Mog program with a `main()` that prints the function's output for
/// every input set, then execute it once and parse the printed lines. This
/// is the identical execution path to `verify_problem_code_via_main` — one
/// parse + one run per row.
fn run_examples(
    code: &str,
    fn_name: &str,
    inputs: &[(Vec<i64>, Vec<i64>)],
) -> Result<Vec<i64>, String> {
    let mut src = String::from(code.trim_end());
    src.push_str("\n\nfn main() -> i64 {\n");
    for (arr, scalars) in inputs {
        let arr_lit = format!(
            "[{}]",
            arr.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut args = vec![arr_lit];
        for s in scalars {
            args.push(s.to_string());
        }
        src.push_str(&format!("    println_i64({fn_name}({}));\n", args.join(", ")));
    }
    src.push_str("    return 0;\n}\n");

    let result = crate::runtime::execute_program(&src)?;
    let mut outs = Vec::with_capacity(inputs.len());
    for line in result.output.lines() {
        outs.push(
            line.trim()
                .parse::<i64>()
                .map_err(|e| format!("bad output line {line:?}: {e}"))?,
        );
    }
    if outs.len() != inputs.len() {
        return Err(format!(
            "expected {} outputs, got {}",
            inputs.len(),
            outs.len()
        ));
    }
    Ok(outs)
}

fn desc_to_json(desc: &UArrDescription, origin: &str, examples_json: Vec<serde_json::Value>, mog: &str) -> serde_json::Value {
    serde_json::json!({
        "n_scalar": desc.n_scalar,
        "origin": origin,
        "examples": examples_json,
        // The "desc" object is exactly the proposal shape propose.py emits
        // (see `description_from_proposal`), so train/propose share a schema.
        "desc": {
            "consts": desc.consts.to_vec(),
            "slots": desc
                .slots
                .iter()
                .map(|s| vec![s.op, s.s1, s.s2, s.cmp, s.gl, s.gr, s.el])
                .collect::<Vec<_>>(),
            "body_init": desc.body_init.clone(),
            "ret": desc.ret,
        },
        "mog": mog,
    })
}

/// Generate `target_rows` clean rows to `out_path` (JSONL). Returns stats.
/// Deterministic in `seed`.
pub fn generate_prior_data(
    target_rows: usize,
    seed: u64,
    out_path: &str,
) -> Result<GenStats, String> {
    let mut rng = SplitMix64::new(seed);
    let mut stats = GenStats::default();
    let mut code_counts: HashMap<String, usize> = HashMap::new();
    let file = std::fs::File::create(out_path)
        .map_err(|e| format!("cannot create {out_path}: {e}"))?;
    let mut w = std::io::BufWriter::new(file);

    let max_attempts = target_rows.saturating_mul(60).max(10_000);
    while stats.written < target_rows && stats.attempts < max_attempts {
        stats.attempts += 1;

        let n_scalar = rng.below(3);
        let consts = sample_consts(&mut rng);

        // Shape: 50% hand-coded restart bias, 50% pure random bias.
        let hand = rng.chance(0.5);
        let (mut prog, origin) = if hand {
            let mut restart = 1 + rng.below(25);
            while (restart == 6 || restart == 14) && n_scalar == 0 {
                restart = 1 + rng.below(25);
            }
            let mut p = SoftUniversalArrayProgram::new_with_consts(n_scalar, &consts);
            apply_handcoded_restart_bias(&mut p, restart);
            (p, format!("hand:{restart}"))
        } else {
            let p = random_bias_init(n_scalar, rng.next_u64(), &consts);
            (p, "random".to_string())
        };

        // 40% of rows get extra random mutations for diversity around the
        // base shapes.
        if rng.chance(0.4) {
            mutate(&mut prog, &mut rng);
        }

        let desc = prog.describe();
        let scalar_names = scalar_names_for(n_scalar);
        let code = prog.discretize_and_emit("f", &scalar_names);

        // Frequency cap per distinct discrete program.
        let count = code_counts.get(&code).copied().unwrap_or(0);
        if count >= MAX_ROWS_PER_CODE {
            stats.rejected_dup_cap += 1;
            continue;
        }

        // Random inputs: 4..=6 examples.
        let n_ex = 4 + rng.below(3);
        let mut inputs: Vec<(Vec<i64>, Vec<i64>)> = Vec::with_capacity(n_ex);
        for _ in 0..n_ex {
            let len = 1 + rng.below(MAX_ARR_LEN);
            let arr: Vec<i64> = (0..len).map(|_| rng.range_i64(-MAX_ELEM, MAX_ELEM)).collect();
            let scalars: Vec<i64> = (0..n_scalar)
                .map(|_| rng.range_i64(-MAX_SCALAR, MAX_SCALAR))
                .collect();
            inputs.push((arr, scalars));
        }

        // Exact execution through the verifier's runtime.
        let outputs = match run_examples(&code, "f", &inputs) {
            Ok(o) => o,
            Err(_) => {
                stats.rejected_exec_error += 1;
                continue;
            }
        };

        if outputs.iter().any(|o| o.abs() > MAX_OUTPUT_MAG) {
            stats.rejected_magnitude += 1;
            continue;
        }

        // Degenerate: identical output on every (random) input.
        if outputs.windows(2).all(|w| w[0] == w[1]) {
            stats.rejected_constant += 1;
            continue;
        }

        let examples_json: Vec<serde_json::Value> = inputs
            .iter()
            .zip(outputs.iter())
            .map(|((arr, scalars), out)| {
                serde_json::json!({"array": arr, "scalars": scalars, "expected": out})
            })
            .collect();
        let row = desc_to_json(&desc, &origin, examples_json, &code);
        writeln!(w, "{row}").map_err(|e| format!("write failed: {e}"))?;

        *code_counts.entry(code).or_insert(0) += 1;
        stats.written += 1;
        if hand {
            stats.hand_rows += 1;
        } else {
            stats.random_rows += 1;
        }
    }
    w.flush().map_err(|e| format!("flush failed: {e}"))?;
    stats.distinct_codes = code_counts.len();
    Ok(stats)
}

/// Parse a prior-net proposal (the JSON shape emitted by
/// `nsynth/scripts/prior_net/propose.py`) into a [`UArrDescription`].
/// Returns `None` on any malformed field — callers fail soft.
pub(in crate::synthesis) fn description_from_proposal(
    v: &serde_json::Value,
    n_scalar: usize,
) -> Option<UArrDescription> {
    let consts_v = v.get("consts")?.as_array()?;
    let mut consts = [0i64; N_CONSTS];
    for (i, c) in consts_v.iter().take(N_CONSTS).enumerate() {
        consts[i] = c.as_i64()?;
    }
    let slots_v = v.get("slots")?.as_array()?;
    if slots_v.len() != N_ARR_SLOTS {
        return None;
    }
    let mut slots = Vec::with_capacity(N_ARR_SLOTS);
    for s in slots_v {
        let f = s.as_array()?;
        if f.len() != 7 {
            return None;
        }
        let g = |i: usize| -> Option<usize> { f[i].as_u64().map(|x| x as usize) };
        slots.push(UArrSlotDesc {
            op: g(0)?,
            s1: g(1)?,
            s2: g(2)?,
            cmp: g(3)?,
            gl: g(4)?,
            gr: g(5)?,
            el: g(6)?,
        });
    }
    let body_init: Vec<usize> = v
        .get("body_init")?
        .as_array()?
        .iter()
        .filter_map(|x| x.as_u64().map(|u| u as usize))
        .collect();
    if body_init.len() != N_ARR_BODY {
        return None;
    }
    let ret = v.get("ret")?.as_u64()? as usize;
    Some(UArrDescription {
        n_scalar,
        consts,
        slots,
        body_init,
        ret,
    })
}

/// Build the soft program for a proposal description. Exposed to the
/// synthesis module for the tier-0 wiring in
/// `synthesize_universal_array_fallback`.
pub(in crate::synthesis) fn program_from_description(
    desc: &UArrDescription,
) -> SoftUniversalArrayProgram {
    SoftUniversalArrayProgram::from_description(desc)
}

// ── Tier-0 proposer wiring (stage A3) ────────────────────────────────────────

/// Master switch for the prior-net tier-0 proposer. Anything other than
/// `NSYNTH_PRIOR_NET=1` leaves the solver byte-identical to before.
pub(in crate::synthesis) fn prior_net_enabled() -> bool {
    std::env::var("NSYNTH_PRIOR_NET").map(|v| v == "1").unwrap_or(false)
}

/// Locate the propose.py bridge script and the trained model checkpoint.
/// Overridable via `NSYNTH_PRIOR_NET_SCRIPT` / `NSYNTH_PRIOR_NET_MODEL`;
/// otherwise probes upward from the executable and the cwd for the nsynth
/// root (`scripts/prior_net/propose.py`) and expects the model at
/// `<project root>/training/prior_net/prior_net_v0.pt`.
fn find_prior_net_assets() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let locate_script = || -> Option<std::path::PathBuf> {
        const REL: &str = "scripts/prior_net/propose.py";
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            let mut dir = exe;
            for _ in 0..6 {
                dir = dir.parent()?.to_path_buf();
                candidates.push(dir.join(REL));
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join(REL));
            candidates.push(cwd.join("nsynth").join(REL));
        }
        candidates.into_iter().find(|p| p.exists())
    };

    let script = match std::env::var("NSYNTH_PRIOR_NET_SCRIPT") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => locate_script()?,
    };
    let model = match std::env::var("NSYNTH_PRIOR_NET_MODEL") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => {
            // script = <nsynth>/scripts/prior_net/propose.py
            let nsynth_root = script.parent()?.parent()?.parent()?;
            nsynth_root
                .parent()?
                .join("training/prior_net/prior_net_v0.pt")
        }
    };
    if script.exists() && model.exists() {
        Some((script, model))
    } else {
        None
    }
}

/// Tier-0 of the universal-array cascade: ask the Program Prior Net for K
/// proposed discrete programs, try each with a zero-step discretize+verify,
/// then a cheap warm-refine. Verified-or-discarded — a miss falls through
/// to the existing cascade, so coverage can never regress. Any bridge
/// failure (missing python, malformed JSON, model errors) logs to stderr
/// and returns `None` (fail-soft).
pub(in crate::synthesis) fn try_prior_net_proposals(
    problem: &Problem,
    examples: &[ArrExample],
    n_scalar: usize,
    fn_name: &str,
    scalar_names: &[&str],
    rejected_codes: &mut std::collections::HashSet<String>,
    neg: &mut crate::rejected_cache::RejectionRecorder,
) -> Option<SolveResult> {
    use std::process::{Command, Stdio};

    let (script, model) = find_prior_net_assets()?;

    let exs: Vec<serde_json::Value> = examples
        .iter()
        .map(|ex| {
            let len = (ex.arr_len.round() as usize).min(ex.arr.len());
            let arr: Vec<i64> = ex.arr.iter().take(len).map(|v| *v as i64).collect();
            let scalars: Vec<i64> = ex.scalar_args.iter().map(|v| *v as i64).collect();
            serde_json::json!({"array": arr, "scalars": scalars, "expected": ex.expected as i64})
        })
        .collect();
    let req = serde_json::json!({"n_scalar": n_scalar, "examples": exs});

    let child = Command::new("python3")
        .arg(&script)
        .arg("--model")
        .arg(&model)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[prior_net] spawn failed: {e}");
            return None;
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write as _;
        if let Err(e) = stdin.write_all(req.to_string().as_bytes()) {
            eprintln!("[prior_net] stdin write failed: {e}");
        }
    }
    let output = match child.wait_with_output() {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            eprintln!("[prior_net] propose.py exited with {}", o.status);
            return None;
        }
        Err(e) => {
            eprintln!("[prior_net] wait failed: {e}");
            return None;
        }
    };
    let resp: serde_json::Value = match serde_json::from_slice(&output.stdout) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[prior_net] bad response JSON: {e}");
            return None;
        }
    };
    let proposals = match resp.get("proposals").and_then(|p| p.as_array()) {
        Some(p) if !p.is_empty() => p.clone(),
        _ => return None,
    };

    for (k, prop) in proposals.iter().enumerate() {
        let Some(desc) = description_from_proposal(prop, n_scalar) else {
            eprintln!("[prior_net] proposal {k} malformed — skipped");
            continue;
        };
        let prog = SoftUniversalArrayProgram::from_description(&desc);
        let code = prog.discretize_and_emit(fn_name, scalar_names);

        // Zero-step: verify the proposal's discrete program verbatim.
        if !rejected_codes.contains(&code) && !neg.known_bad(&code) {
            if verify_problem_code_strict(problem, &code).is_ok() {
                eprintln!("[prior_net] {fn_name}: proposal {k} verified ZERO-SEARCH");
                crate::learned_biases::record_success(
                    n_scalar,
                    prog.params.clone(),
                    format!("prior_net:{k}:zero"),
                );
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "prior_net".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
            neg.note_rejection(&code);
            rejected_codes.insert(code);
        }

        // Warm refine: a few Adam steps from the proposal as init.
        if let Some(mut result) = super::warm_refine_from_bias(
            &prog.params,
            problem,
            examples,
            n_scalar,
            fn_name,
            scalar_names,
        ) {
            eprintln!("[prior_net] {fn_name}: proposal {k} verified after warm refine");
            crate::learned_biases::record_success(
                n_scalar,
                prog.params.clone(),
                format!("prior_net:{k}:warm"),
            );
            result.method = "prior_net_warm".to_string();
            return Some(result);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_from_description_roundtrip_random() {
        // Round trip: random program -> describe -> from_description must
        // reproduce the identical discrete description AND emit identical
        // Mog code (the property the tier-0 proposer relies on).
        for n_scalar in 0..=2 {
            for seed in 0..20u64 {
                let consts = [0, 1, -1, 3, 7, 12];
                let prog = random_bias_init(n_scalar, seed.wrapping_mul(977), &consts);
                let desc = prog.describe();
                let rebuilt = SoftUniversalArrayProgram::from_description(&desc);
                assert_eq!(rebuilt.describe(), desc, "desc roundtrip n={n_scalar} s={seed}");
                let names = scalar_names_for(n_scalar);
                assert_eq!(
                    rebuilt.discretize_and_emit("f", &names),
                    prog.discretize_and_emit("f", &names),
                    "code roundtrip n={n_scalar} s={seed}"
                );
            }
        }
    }

    #[test]
    fn describe_from_description_roundtrip_hand_shapes() {
        for n_scalar in 0..=2 {
            for restart in 1..=25usize {
                if (restart == 6 || restart == 14) && n_scalar == 0 {
                    continue;
                }
                let consts = [0, 1, -1, 2, -2, 10];
                let mut prog = SoftUniversalArrayProgram::new_with_consts(n_scalar, &consts);
                apply_handcoded_restart_bias(&mut prog, restart);
                let desc = prog.describe();
                let rebuilt = SoftUniversalArrayProgram::from_description(&desc);
                let names = scalar_names_for(n_scalar);
                assert_eq!(
                    rebuilt.discretize_and_emit("f", &names),
                    prog.discretize_and_emit("f", &names),
                    "code roundtrip n={n_scalar} restart={restart}"
                );
            }
        }
    }

    #[test]
    fn tier0_stub_bridge_zero_search_solves_sum() {
        // End-to-end A3 wiring test with a stub propose.py: the stub echoes
        // a proposal that is exactly the describe() of the hand-coded sum
        // shape (restart 1). try_prior_net_proposals must subprocess the
        // stub, rebuild the program, verify it zero-step, and return
        // method == "prior_net".
        let consts = [0i64, 1, -1, 2, -2, 10];
        let mut prog = SoftUniversalArrayProgram::new_with_consts(0, &consts);
        apply_handcoded_restart_bias(&mut prog, 1);
        let desc = prog.describe();
        let proposal = serde_json::json!({
            "consts": desc.consts.to_vec(),
            "slots": desc
                .slots
                .iter()
                .map(|s| vec![s.op, s.s1, s.s2, s.cmp, s.gl, s.gr, s.el])
                .collect::<Vec<_>>(),
            "body_init": desc.body_init.clone(),
            "ret": desc.ret,
        });
        let stub = format!(
            "#!/usr/bin/env python3\nimport sys, json\nsys.stdin.read()\nprint(json.dumps({{\"proposals\": [{proposal}]}}))\n"
        );
        let script_path = std::env::temp_dir().join(format!(
            "pn_stub_{}_{}.py",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&script_path, stub).unwrap();
        // The model path is only checked for existence by the Rust side; the
        // stub ignores --model entirely.
        std::env::set_var("NSYNTH_PRIOR_NET_SCRIPT", &script_path);
        std::env::set_var("NSYNTH_PRIOR_NET_MODEL", &script_path);

        let mk = |arr: &[i64], expected: i64| crate::benchmark::Example {
            inputs: vec![crate::benchmark::Value::Array(arr.to_vec())],
            expected,
        };
        let problem = Problem {
            name: "prior_net_stub_sum".to_string(),
            category: "arrays",
            description: "sum of elements (prior-net stub test)",
            signature: "fn f(arr: [i64]) -> i64",
            examples: vec![
                mk(&[1, 2, 3], 6),
                mk(&[5], 5),
                mk(&[-2, 4], 2),
                mk(&[10, -3, 2, 1], 10),
            ],
            holdouts: vec![mk(&[7, 7], 14), mk(&[0, 1, -1], 0)],
            reference_code: "",
        };
        let examples: Vec<ArrExample> = problem
            .examples
            .iter()
            .map(|ex| {
                let arr = match &ex.inputs[0] {
                    crate::benchmark::Value::Array(a) => a.clone(),
                    _ => unreachable!(),
                };
                let mut padded = vec![0f32; MAX_ARR];
                for (i, v) in arr.iter().enumerate() {
                    padded[i] = *v as f32;
                }
                ArrExample {
                    arr: padded,
                    arr_len: arr.len() as f32,
                    scalar_args: vec![],
                    expected: ex.expected as f32,
                }
            })
            .collect();

        let mut rejected = std::collections::HashSet::new();
        let mut neg = crate::rejected_cache::RejectionRecorder::new(
            crate::solved_cache::examples_fingerprint(&problem.examples),
        );
        let result =
            try_prior_net_proposals(&problem, &examples, 0, "f", &[], &mut rejected, &mut neg);
        std::env::remove_var("NSYNTH_PRIOR_NET_SCRIPT");
        std::env::remove_var("NSYNTH_PRIOR_NET_MODEL");
        let _ = std::fs::remove_file(&script_path);

        let result = result.expect("stub proposal should zero-search solve sum");
        assert_eq!(result.method, "prior_net");
        assert!(
            crate::runtime::verify_problem_code_strict(&problem, &result.code).is_ok(),
            "returned code must verify"
        );
    }

    #[test]
    fn generator_smoke_produces_clean_rows() {
        let tmp = std::env::temp_dir().join(format!("prior_gen_smoke_{}.jsonl", std::process::id()));
        let stats = generate_prior_data(40, 42, tmp.to_str().unwrap()).unwrap();
        assert_eq!(stats.written, 40);
        let text = std::fs::read_to_string(&tmp).unwrap();
        let rows: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(rows.len(), 40);
        for row in &rows {
            let n_scalar = row["n_scalar"].as_u64().unwrap() as usize;
            assert!(n_scalar <= 2);
            let examples = row["examples"].as_array().unwrap();
            assert!(examples.len() >= 4);
            // Not constant across examples.
            let outs: Vec<i64> = examples
                .iter()
                .map(|e| e["expected"].as_i64().unwrap())
                .collect();
            assert!(outs.windows(2).any(|w| w[0] != w[1]), "constant row leaked");
            // Proposal parse + rebuild emits the row's own Mog code.
            let desc =
                description_from_proposal(&row["desc"], n_scalar).expect("desc parse");
            let prog = program_from_description(&desc);
            let names = scalar_names_for(n_scalar);
            assert_eq!(
                prog.discretize_and_emit("f", &names),
                row["mog"].as_str().unwrap(),
                "row desc does not rebuild row mog"
            );
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
