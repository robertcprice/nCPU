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
pub const CONST_CANDIDATES: [i64; 17] =
    [2, -2, 3, -3, 4, -4, 5, -5, 6, 7, 8, 9, 10, 12, 15, 16, 20];

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
        src.push_str(&format!(
            "    println_i64({fn_name}({}));\n",
            args.join(", ")
        ));
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

fn desc_to_json(
    desc: &UArrDescription,
    origin: &str,
    examples_json: Vec<serde_json::Value>,
    mog: &str,
) -> serde_json::Value {
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
    let file =
        std::fs::File::create(out_path).map_err(|e| format!("cannot create {out_path}: {e}"))?;
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
            let arr: Vec<i64> = (0..len)
                .map(|_| rng.range_i64(-MAX_ELEM, MAX_ELEM))
                .collect();
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
fn description_from_proposal(v: &serde_json::Value, n_scalar: usize) -> Option<UArrDescription> {
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

/// Build the soft program for a proposal description (test seam for the
/// tier-0 wiring; production path calls `from_description` directly).
fn program_from_description(desc: &UArrDescription) -> SoftUniversalArrayProgram {
    SoftUniversalArrayProgram::from_description(desc)
}

// ── Tier-0 proposer wiring (stage A3; v1 persistent server + gate) ───────────

/// Master switch for the prior-net tier-0 proposer. The proposer is OPT-IN to
/// honor the Rust-only invariant (MASTER_ROADMAP 2.3): it stays OFF (never
/// spawns `python3`) unless `NSYNTH_PRIOR_NET == "1"`, in which case it
/// warm-starts the universal-array solver when the script/model assets are
/// present. With the flag unset, behavior is byte-identical to the old
/// no-prior path.
pub(in crate::synthesis) fn prior_net_enabled() -> bool {
    if std::env::var("NSYNTH_PRIOR_NET")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return find_prior_net_assets().is_some();
    }
    false
}

/// Default confidence gate (Phase A v1). Calibrated on the 10k held-out
/// rows of the 300k generated split via `training/prior_net/calibrate.py`
/// (see `training/prior_net/confidence_calibration.json`): the gate signal
/// is `mean_logp` — the argmax proposal's mean log max-softmax across the
/// 60 heads (best exact-vs-miss AUC, 0.743) — and tau is the hit-recall
/// rule's choice (largest tau keeping >= 90% of held-out exact hits
/// firing). Below tau the server returns no proposals and the solver pays
/// only the ~ms round-trip before falling through to its cascade.
/// Override via `NSYNTH_PRIOR_NET_TAU`.
const DEFAULT_PRIOR_TAU: f64 = -0.2473;

/// Gate signal name passed to propose.py --signal. Must match the signal the
/// calibration JSON chose tau for. Override via `NSYNTH_PRIOR_NET_SIGNAL`.
const DEFAULT_PRIOR_SIGNAL: &str = "mean_logp";

fn prior_tau() -> f64 {
    std::env::var("NSYNTH_PRIOR_NET_TAU")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(DEFAULT_PRIOR_TAU)
}

fn prior_signal() -> String {
    std::env::var("NSYNTH_PRIOR_NET_SIGNAL")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_PRIOR_SIGNAL.to_string())
}

/// Persistent proposer process (v1 overhead cut). v0 spawned
/// `python3 propose.py` per problem, paying the torch import + model load
/// (~1-3 s) on every call — that overhead alone exceeded the prior's
/// zero-search savings. v1 keeps one `propose.py --serve` child alive for
/// the process lifetime: line-buffered JSON request/response over
/// stdin/stdout, model loaded once behind a `{"ready": true}` handshake.
struct PriorServer {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::io::BufReader<std::process::ChildStdout>,
}

enum ServerState {
    /// No spawn attempted yet this process.
    Untried,
    /// Background thread is doing the spawn + ready handshake. Requests
    /// arriving in this window return `None` immediately (fall through to
    /// the cascade) so the ~10 s torch-import startup overlaps the first
    /// problems' search instead of blocking inside the first tier-0 call.
    Spawning,
    /// Spawn or protocol failed — never retried (fail-soft, no respawn storm).
    Failed,
    Running(Box<PriorServer>),
}

fn prior_server() -> &'static std::sync::Mutex<ServerState> {
    static SERVER: std::sync::OnceLock<std::sync::Mutex<ServerState>> = std::sync::OnceLock::new();
    SERVER.get_or_init(|| std::sync::Mutex::new(ServerState::Untried))
}

/// Test seam: wait until the async spawn settles (Running or Failed).
/// Returns true if the server is Running.
#[cfg(test)]
fn wait_for_prior_server(timeout: std::time::Duration) -> bool {
    let t0 = std::time::Instant::now();
    loop {
        if let Ok(guard) = prior_server().lock() {
            match &*guard {
                ServerState::Running(_) => return true,
                // Untried and Failed are both settled (no spawn in flight).
                ServerState::Failed | ServerState::Untried => return false,
                ServerState::Spawning => {}
            }
        }
        if t0.elapsed() > timeout {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Test seam: kill any live server child and reset to `Untried` so tests
/// that redirect `NSYNTH_PRIOR_NET_SCRIPT` get a fresh spawn. Waits out an
/// in-flight async spawn first so a stale child can't overwrite the reset.
#[cfg(test)]
fn reset_prior_server() {
    let _ = wait_for_prior_server(std::time::Duration::from_secs(30));
    if let Ok(mut guard) = prior_server().lock() {
        if let ServerState::Running(server) = &mut *guard {
            let _ = server.child.kill();
            let _ = server.child.wait();
        }
        *guard = ServerState::Untried;
    }
}

fn spawn_prior_server() -> Option<PriorServer> {
    use std::io::BufRead as _;
    use std::process::{Command, Stdio};
    let (script, model) = find_prior_net_assets()?;
    let mut child = match Command::new("python3")
        .arg(&script)
        .arg("--serve")
        .arg("--model")
        .arg(&model)
        .arg("--tau")
        .arg(format!("{}", prior_tau()))
        .arg("--signal")
        .arg(prior_signal())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[prior_net] serve spawn failed: {e}");
            return None;
        }
    };
    let stdin = child.stdin.take()?;
    let stdout = std::io::BufReader::new(child.stdout.take()?);
    let mut server = PriorServer {
        child,
        stdin,
        stdout,
    };
    // One-time ready handshake — covers the torch import + checkpoint load,
    // paid once per process lifetime instead of once per problem.
    let mut line = String::new();
    match server.stdout.read_line(&mut line) {
        Ok(n) if n > 0 => {}
        _ => {
            eprintln!("[prior_net] server exited before ready line");
            let _ = server.child.kill();
            return None;
        }
    }
    let ready: serde_json::Value = match serde_json::from_str(line.trim()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[prior_net] bad ready line: {e}");
            let _ = server.child.kill();
            return None;
        }
    };
    if ready.get("ready").and_then(|r| r.as_bool()) != Some(true) {
        eprintln!("[prior_net] server not ready: {}", line.trim());
        let _ = server.child.kill();
        return None;
    }
    Some(server)
}

/// One request/response round-trip on the live server. `None` on any
/// protocol failure — the caller demotes the server to `Failed`.
fn server_request(server: &mut PriorServer, req: &serde_json::Value) -> Option<serde_json::Value> {
    use std::io::{BufRead as _, Write as _};
    let mut line = req.to_string();
    line.push('\n');
    server.stdin.write_all(line.as_bytes()).ok()?;
    server.stdin.flush().ok()?;
    let mut resp = String::new();
    let n = server.stdout.read_line(&mut resp).ok()?;
    if n == 0 {
        return None;
    }
    serde_json::from_str(resp.trim()).ok()
}

/// Send `req` through the persistent server, kicking off an async spawn on
/// first use. While the spawn + ready handshake runs in the background the
/// caller gets `None` (cascade proceeds normally), so server startup costs
/// ~0 wall time. Fail-soft: any failure marks the server `Failed` for the
/// rest of the process and returns `None`.
fn prior_server_propose(req: &serde_json::Value) -> Option<serde_json::Value> {
    let mut guard = prior_server().lock().ok()?;
    if matches!(*guard, ServerState::Untried) {
        *guard = ServerState::Spawning;
        drop(guard);
        std::thread::spawn(|| {
            let state = match spawn_prior_server() {
                Some(s) => ServerState::Running(Box::new(s)),
                None => ServerState::Failed,
            };
            if let Ok(mut g) = prior_server().lock() {
                *g = state;
            }
        });
        return None;
    }
    let server = match &mut *guard {
        ServerState::Running(s) => s,
        _ => return None,
    };
    match server_request(server, req) {
        Some(resp) => Some(resp),
        None => {
            eprintln!("[prior_net] server protocol failure — disabling for this run");
            let _ = server.child.kill();
            let _ = server.child.wait();
            *guard = ServerState::Failed;
            None
        }
    }
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
            let dir = nsynth_root.parent()?.join("training/prior_net");
            // Prefer the newest trained checkpoint generation.
            let v1 = dir.join("prior_net_v1.pt");
            if v1.exists() {
                v1
            } else {
                dir.join("prior_net_v0.pt")
            }
        }
    };
    if script.exists() && model.exists() {
        Some((script, model))
    } else {
        None
    }
}

/// Tier-0 of the universal-array cascade (v1): ask the persistent Program
/// Prior Net server for K proposed discrete programs. The server applies
/// the calibrated confidence gate — below tau it returns no proposals, so
/// a gated miss costs only the ~ms round-trip. Gate-open proposals are each
/// tried zero-step (discretize+verify, ~65 ms total for K=4). There is no
/// warm-refine pass in v1: v0 measured 0 conversions from 64 warm-refine
/// attempts (4 per miss x 16 problems) at ~0.4-1 s each — every measured
/// win came from zero-step verification, so warm refine was pure overhead.
/// Verified-or-discarded — a miss falls through to the existing cascade, so
/// coverage can never regress. Any bridge failure (missing python, malformed
/// JSON, model errors) logs to stderr and returns `None` (fail-soft).
pub(in crate::synthesis) fn try_prior_net_proposals(
    problem: &Problem,
    examples: &[ArrExample],
    n_scalar: usize,
    fn_name: &str,
    scalar_names: &[&str],
    rejected_codes: &mut std::collections::HashSet<String>,
    neg: &mut crate::rejected_cache::RejectionRecorder,
) -> Option<SolveResult> {
    let t0 = std::time::Instant::now();
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

    let resp = prior_server_propose(&req)?;
    let confidence = resp
        .get("confidence")
        .and_then(|c| c.as_f64())
        .unwrap_or(0.0);
    if resp.get("gated").and_then(|g| g.as_bool()) == Some(true) {
        eprintln!(
            "[prior_net] {fn_name}: GATED (conf {confidence:.3} < tau {:.3}) in {:.0}ms",
            prior_tau(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
        return None;
    }
    let proposals = match resp.get("proposals").and_then(|p| p.as_array()) {
        Some(p) if !p.is_empty() => p.clone(),
        _ => return None,
    };

    // Zero-step: verify each proposal's discrete program verbatim
    // (cheap: one parse + one run of a tiny Mog program per proposal).
    for (k, prop) in proposals.iter().enumerate() {
        let Some(desc) = description_from_proposal(prop, n_scalar) else {
            eprintln!("[prior_net] proposal {k} malformed — skipped");
            continue;
        };
        let prog = SoftUniversalArrayProgram::from_description(&desc);
        let code = prog.discretize_and_emit(fn_name, scalar_names);

        if !rejected_codes.contains(&code) && !neg.known_bad(&code) {
            if verify_problem_code_strict(problem, &code).is_ok() {
                eprintln!(
                    "[prior_net] {fn_name}: proposal {k} verified ZERO-SEARCH \
                     (conf {confidence:.3}) in {:.0}ms",
                    t0.elapsed().as_secs_f64() * 1000.0
                );
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
    }
    eprintln!(
        "[prior_net] {fn_name}: {} proposals (conf {confidence:.3}), none verified \
         in {:.0}ms — falling through to cascade",
        proposals.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    None
}

// ── Direct-fallback eval harness (stage A4) ──────────────────────────────────

/// Run `synthesize_universal_array_fallback` in isolation on named benchmark
/// problems (bypassing the search-teacher stages that normally pre-empt it)
/// and report per-problem method + wall time. This measures the tier-0
/// prior against the 26-restart cascade on the program space the prior was
/// trained for — the metric ROADMAP Rung 9 Phase A actually asks for.
/// Honors `NSYNTH_PRIOR_NET` like production, so callers can diff OFF/ON.
pub fn eval_fallback_direct(names: &[String]) -> Vec<serde_json::Value> {
    let problems = crate::benchmark::get_benchmark(1);
    let mut rows = Vec::new();
    for problem in &problems {
        if !names.is_empty() && !names.iter().any(|n| &problem.name == n) {
            continue;
        }
        // Same extraction the native_array entry point performs.
        let first = match problem.examples.first() {
            Some(f) => f,
            None => continue,
        };
        if !matches!(
            first.inputs.first(),
            Some(crate::benchmark::Value::Array(_))
        ) {
            continue;
        }
        let n_scalar = first.inputs.len().saturating_sub(1);
        let mut examples = Vec::new();
        let mut ok = true;
        for ex in &problem.examples {
            let arr = match ex.inputs[0].as_i64_slice() {
                Some(a) => a,
                None => {
                    ok = false;
                    break;
                }
            };
            let mut padded = vec![0f32; MAX_ARR];
            for (i, v) in arr.iter().enumerate() {
                if i < MAX_ARR {
                    padded[i] = *v as f32;
                }
            }
            let mut scalar_args = Vec::with_capacity(n_scalar);
            for v in &ex.inputs[1..] {
                match v {
                    crate::benchmark::Value::Int(iv) => scalar_args.push(*iv as f32),
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            examples.push(ArrExample {
                arr: padded,
                arr_len: arr.len() as f32,
                scalar_args,
                expected: ex.expected_int() as f32,
            });
        }
        if !ok {
            continue;
        }
        let scalar_names: Vec<&str> = match n_scalar {
            0 => vec![],
            1 => vec!["k"],
            n => ["a", "b", "c", "d", "e", "f"]
                .iter()
                .take(n)
                .copied()
                .collect(),
        };
        let fn_name = problem.function_name();
        let t0 = std::time::Instant::now();
        let result = super::synthesize_universal_array_fallback(
            problem,
            &examples,
            n_scalar,
            fn_name,
            &scalar_names,
        );
        let secs = t0.elapsed().as_secs_f64();
        rows.push(serde_json::json!({
            "name": problem.name,
            "n_scalar": n_scalar,
            "solved": result.is_some(),
            "method": result.as_ref().map(|r| r.method.clone()),
            "seconds": (secs * 1000.0).round() / 1000.0,
        }));
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rust-only invariant: with NO `NSYNTH_PRIOR_NET` env set, the prior-net
    /// tier-0 proposer must be OFF (never spawns `python3`) even when the
    /// script/model assets are faked. Opt-in only via `NSYNTH_PRIOR_NET=1`.
    /// Env is process-global and cargo runs tests multi-threaded, so we
    /// save/clear the relevant vars on entry and restore on exit.
    #[test]
    fn prior_net_disabled_by_default() {
        let saved = std::env::var("NSYNTH_PRIOR_NET").ok();
        let saved_script = std::env::var("NSYNTH_PRIOR_NET_SCRIPT").ok();
        let saved_model = std::env::var("NSYNTH_PRIOR_NET_MODEL").ok();

        std::env::remove_var("NSYNTH_PRIOR_NET");
        // Fake the assets so we prove the gate, not asset-absence, is what
        // keeps the proposer off.
        std::env::set_var("NSYNTH_PRIOR_NET_SCRIPT", "/tmp/fake_propose.py");
        std::env::set_var("NSYNTH_PRIOR_NET_MODEL", "/tmp/fake_model.pt");

        assert!(
            !prior_net_enabled(),
            "prior_net_enabled() must be false with NSYNTH_PRIOR_NET unset (Rust-only default)"
        );

        // Restore env to avoid cross-test contamination.
        std::env::remove_var("NSYNTH_PRIOR_NET_SCRIPT");
        std::env::remove_var("NSYNTH_PRIOR_NET_MODEL");
        if let Some(v) = saved {
            std::env::set_var("NSYNTH_PRIOR_NET", v);
        }
        if let Some(v) = saved_script {
            std::env::set_var("NSYNTH_PRIOR_NET_SCRIPT", v);
        }
        if let Some(v) = saved_model {
            std::env::set_var("NSYNTH_PRIOR_NET_MODEL", v);
        }
    }

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
                assert_eq!(
                    rebuilt.describe(),
                    desc,
                    "desc roundtrip n={n_scalar} s={seed}"
                );
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
        // End-to-end wiring test with a stub propose.py speaking the v1
        // persistent line protocol: ready handshake, then one response per
        // request line. The stub echoes a proposal that is exactly the
        // describe() of the hand-coded sum shape (restart 1).
        // try_prior_net_proposals must spawn the server, round-trip the
        // request, rebuild the program, verify it zero-step, and return
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
            "confidence": 0.99,
        });
        let stub = format!(
            "#!/usr/bin/env python3\nimport sys, json\n\
             print(json.dumps({{\"ready\": True}}), flush=True)\n\
             for line in sys.stdin:\n\
            \x20    json.loads(line)\n\
            \x20    print(json.dumps({{\"proposals\": [{proposal}], \
             \"confidence\": 0.99, \"gated\": False}}), flush=True)\n"
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
        // The server is process-global; force a fresh spawn so the stub env
        // vars take effect even if another test already started a server.
        reset_prior_server();

        let mk = |arr: &[i64], expected: i64| crate::benchmark::Example {
            inputs: vec![crate::benchmark::Value::int_array(arr)],
            expected: crate::benchmark::Value::Int(expected),
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

            synthetic_args: Vec::new(),

            synthetic_values: Vec::new(),

            recursive_allowed: false,

            tree_input: false,

            explicit_stack: false,
            functions: vec![],
        };
        let examples: Vec<ArrExample> = problem
            .examples
            .iter()
            .map(|ex| {
                let arr = match ex.inputs[0].as_i64_slice() {
                    Some(a) => a,
                    None => unreachable!(),
                };
                let mut padded = vec![0f32; MAX_ARR];
                for (i, v) in arr.iter().enumerate() {
                    padded[i] = *v as f32;
                }
                ArrExample {
                    arr: padded,
                    arr_len: arr.len() as f32,
                    scalar_args: vec![],
                    expected: ex.expected_int() as f32,
                }
            })
            .collect();

        let mut rejected = std::collections::HashSet::new();
        let mut neg = crate::rejected_cache::RejectionRecorder::new(
            crate::solved_cache::examples_fingerprint(&problem.examples),
        );
        // First call kicks the async spawn and returns None (the production
        // behavior that makes server startup cost ~0 wall time).
        let first =
            try_prior_net_proposals(&problem, &examples, 0, "f", &[], &mut rejected, &mut neg);
        assert!(
            first.is_none(),
            "first call must return None while spawning"
        );
        assert!(
            wait_for_prior_server(std::time::Duration::from_secs(20)),
            "stub server failed to become ready"
        );
        let result =
            try_prior_net_proposals(&problem, &examples, 0, "f", &[], &mut rejected, &mut neg);
        reset_prior_server();
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

    /// Utility (ignored): dump one prior-net request JSON per array-input
    /// benchmark problem to /tmp/prior_bench_requests.jsonl so the gate
    /// threshold can be sanity-checked offline against the model:
    ///   cargo test -p mog_synth --release dump_bench_fallback_requests \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_bench_fallback_requests() {
        let problems = crate::benchmark::get_benchmark(1);
        let mut out = String::new();
        for problem in &problems {
            let first = match problem.examples.first() {
                Some(f) => f,
                None => continue,
            };
            if !matches!(
                first.inputs.first(),
                Some(crate::benchmark::Value::Array(_))
            ) {
                continue;
            }
            let n_scalar = first.inputs.len().saturating_sub(1);
            let mut exs = Vec::new();
            let mut ok = true;
            for ex in &problem.examples {
                let arr = match &ex.inputs[0] {
                    crate::benchmark::Value::Array(a) => a.clone(),
                    _ => {
                        ok = false;
                        break;
                    }
                };
                let mut scalars = Vec::new();
                for v in &ex.inputs[1..] {
                    match v {
                        crate::benchmark::Value::Int(iv) => scalars.push(*iv),
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                exs.push(serde_json::json!({
                    "array": arr, "scalars": scalars, "expected": ex.expected,
                }));
            }
            if !ok {
                continue;
            }
            out.push_str(
                &serde_json::json!({
                    "name": problem.name, "n_scalar": n_scalar, "examples": exs,
                })
                .to_string(),
            );
            out.push('\n');
        }
        std::fs::write("/tmp/prior_bench_requests.jsonl", out).unwrap();
    }

    #[test]
    fn generator_smoke_produces_clean_rows() {
        let tmp =
            std::env::temp_dir().join(format!("prior_gen_smoke_{}.jsonl", std::process::id()));
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
            let desc = description_from_proposal(&row["desc"], n_scalar).expect("desc parse");
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
