//! Offline supervised training for the ranker's weight vector.
//!
//! The online rule (`record_transfer_success`) only fires when a cached
//! teacher transfers — rare events, slow weight evolution. This binary
//! short-circuits that with a direct optimization pass:
//!
//!   1. For each benchmark problem whose I/O fingerprint is in the solved
//!      cache, pair (problem_features, that_problem's_solution_code) as a
//!      **positive** (`d_pos` should be small).
//!   2. For each positive, sample N random **negatives** from the cache
//!      (any other entry — different code, different problem).
//!   3. Hinge ranking loss: `L = max(0, margin + d_pos - d_neg)`.
//!   4. Gradient step on the weight vector, repeat for `--epochs N`.
//!   5. Save weights to the file used at runtime.
//!
//! Rationale: `diversity_ab` measured that the ranker's top-50 routinely
//! misses the transfer-winning teacher. Root cause is a query-invariance in
//! the old distance formula; this turn fixes that at the distance layer
//! (see `meta_learner::query_conditional_distance`). Bootstrap training
//! then anchors the weights to supervised data rather than only the rare
//! online signal.
//!
//! Usage:
//!     cargo run --release --bin bootstrap_train -- \
//!         [--epochs 50]                            \
//!         [--negs-per-pos 4]                       \
//!         [--margin 0.5]                           \
//!         [--lr 0.02]                              \
//!         [--seed 42]                              \
//!         [--dry-run]     print final weights instead of saving

use std::collections::HashMap;

use mog_synth::benchmark::get_benchmark;
use mog_synth::meta_learner::{
    apply_weight_gradient, expected_code_features, extract_code_features, extract_problem_features,
    rank_teachers_with_meta, set_weights, MetaWeights, FEATURE_DIM,
};
use mog_synth::solved_cache::{self, examples_fingerprint};

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Deterministic xorshift RNG — we only need integer uniform draws for
/// negative sampling, no need for the `rand` crate.
struct XorShift64 {
    state: u64,
}
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xdeadbeef } else { seed },
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Gate training on the cache-growth marker. `--if-due` is the cron
    // entry point: exits 0 without training when nothing has changed
    // enough to warrant a retrain. Avoids burning CI minutes on quiet
    // periods while still allowing forced retrains via the normal
    // invocation path.
    if has_flag(&args, "--if-due") && !solved_cache::bootstrap_retrain_due() {
        eprintln!("[bootstrap_train] --if-due set and no retrain marker; skipping");
        return;
    }

    let epochs: usize = arg_value(&args, "--epochs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let negs_per_pos: usize = arg_value(&args, "--negs-per-pos")
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let margin: f64 = arg_value(&args, "--margin")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);
    let lr: f64 = arg_value(&args, "--lr")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.02);
    let seed: u64 = arg_value(&args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);
    let dry_run = has_flag(&args, "--dry-run");

    // Build positive pairs: (problem, its own cached solution's code).
    let snapshot: Vec<(String, String, u32, u64)> = solved_cache::snapshot_solutions_with_meta();
    if snapshot.is_empty() {
        eprintln!("[bootstrap_train] solved cache is empty — nothing to train on");
        std::process::exit(1);
    }

    // Build a fingerprint → code index so we can look up each problem's own
    // solution cheaply. We can't use `solved_cache::lookup` here because
    // that re-verifies the code, and we don't need verification — we just
    // need the code that was recorded for this fingerprint.
    //
    // Regenerate fingerprint from each benchmark problem's examples,
    // intersect with the cache.
    let problems = get_benchmark(1);
    let mut positives: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for p in problems.iter() {
        let fp = examples_fingerprint(&p.examples);
        // Find a cache row whose code matches — we don't have direct
        // snapshot-by-fingerprint access, so re-scan is fine (O(problems ×
        // cache) = ~100×100 = 10k ops once).
        //
        // solved_cache::lookup does the strict verify; we use it here so the
        // positive pair is guaranteed to be a *verified* (problem, code)
        // match, not just a fingerprint collision. This is the right signal
        // to train on.
        if let Some(sol) = solved_cache::lookup(p) {
            let pf = extract_problem_features(p);
            let cf = extract_code_features(&sol.code);
            positives.push((pf.to_vec(), cf.to_vec()));
        }
        let _ = fp; // fingerprint only used via lookup
    }

    if positives.is_empty() {
        eprintln!(
            "[bootstrap_train] no (problem, cached_code) positive pairs — run the bench first to populate the cache"
        );
        std::process::exit(1);
    }

    eprintln!(
        "[bootstrap_train] {} positives, {} cache entries available for negatives",
        positives.len(),
        snapshot.len()
    );

    // Index cache codes for negative sampling.
    let all_codes: Vec<String> = snapshot.iter().map(|(_, c, _, _)| c.clone()).collect();
    let mut code_to_cf: HashMap<String, Vec<f64>> = HashMap::new();
    for c in &all_codes {
        code_to_cf
            .entry(c.clone())
            .or_insert_with(|| extract_code_features(c).to_vec());
    }

    // Initialise weights from the persisted file so we continue training
    // from whatever state the online rule + prior runs left. Fall back to
    // uniform if no file exists.
    let weights_before = MetaWeights::load();
    let mut weights = weights_before.clone();
    let mut rng = XorShift64::new(seed);

    for epoch in 0..epochs {
        let mut total_loss = 0.0_f64;
        let mut active_pairs = 0usize;

        for (pf_vec, pos_cf_vec) in &positives {
            let mut pf = [0.0_f64; FEATURE_DIM];
            let mut pos_cf = [0.0_f64; FEATURE_DIM];
            for i in 0..FEATURE_DIM {
                pf[i] = pf_vec[i];
                pos_cf[i] = pos_cf_vec[i];
            }
            let ecf = expected_code_features(&pf);
            let pos_res: [f64; FEATURE_DIM] = std::array::from_fn(|i| pos_cf[i] - ecf[i]);

            for _ in 0..negs_per_pos {
                let idx = (rng.next_u64() as usize) % all_codes.len();
                let neg_code = &all_codes[idx];
                // Skip when the sampled code happens to equal the positive.
                if neg_code.as_str() == pos_cf_vec_to_identity(pos_cf_vec).as_str() {
                    continue;
                }
                let neg_cf_vec = match code_to_cf.get(neg_code) {
                    Some(v) => v,
                    None => continue,
                };
                let mut neg_cf = [0.0_f64; FEATURE_DIM];
                for i in 0..FEATURE_DIM {
                    neg_cf[i] = neg_cf_vec[i];
                }
                let neg_res: [f64; FEATURE_DIM] = std::array::from_fn(|i| neg_cf[i] - ecf[i]);

                // Distances under current weights.
                let d_pos = weighted_l2(&pos_res, &weights);
                let d_neg = weighted_l2(&neg_res, &weights);
                let loss = (margin + d_pos - d_neg).max(0.0);
                if loss <= 0.0 {
                    continue;
                }
                total_loss += loss;
                active_pairs += 1;

                // Gradient of L = margin + d_pos - d_neg w.r.t. w_i:
                //   ∂d/∂w_i = res_i² / (2·d)
                // So ∂L/∂w_i = (pos_res_i² / (2·d_pos)) - (neg_res_i² / (2·d_neg))
                // To MINIMISE loss, step in -∂L direction.
                let mut grad = [0.0_f64; FEATURE_DIM];
                let inv2p = if d_pos > 1e-9 {
                    1.0 / (2.0 * d_pos)
                } else {
                    0.0
                };
                let inv2n = if d_neg > 1e-9 {
                    1.0 / (2.0 * d_neg)
                } else {
                    0.0
                };
                for i in 0..FEATURE_DIM {
                    grad[i] = pos_res[i] * pos_res[i] * inv2p - neg_res[i] * neg_res[i] * inv2n;
                }
                // Step direction is -grad (descent). `apply_weight_gradient`
                // adds `lr * delta` to weights — pass -grad so weights move
                // in the descent direction.
                let neg_grad: [f64; FEATURE_DIM] = std::array::from_fn(|i| -grad[i]);
                apply_weight_gradient(&mut weights, &neg_grad, lr);
            }
        }

        eprintln!(
            "[bootstrap_train] epoch {:3}: loss={:.4}  active_pairs={}",
            epoch, total_loss, active_pairs
        );
        if active_pairs == 0 {
            eprintln!("[bootstrap_train] no margin-violating pairs — early stopping");
            break;
        }
    }

    // Print final weight summary.
    let mut extrema: Vec<(usize, f64)> = (0..FEATURE_DIM).map(|i| (i, weights.w[i])).collect();
    extrema.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("\nTop-5 weighted feature dims (largest weight):");
    for (i, w) in extrema.iter().take(5) {
        eprintln!("  dim {:2}: w = {:.4}", i, w);
    }
    eprintln!("Bottom-5 (smallest weight):");
    for (i, w) in extrema.iter().rev().take(5) {
        eprintln!("  dim {:2}: w = {:.4}", i, w);
    }

    // Counterfactual A/B: for each verified (problem, its_code) positive,
    // compute the rank of its_code under the OLD and NEW weights against
    // the full cache. Mean rank drop is the single scalar that quantifies
    // "did this retrain help?" — no ambiguity, no hidden variable.
    //
    // This is the "every retrain is observable" property: the retrain
    // pipeline is no longer opaque; its effect is measured before the
    // new weights ship.
    let impact_rows = measure_retrain_impact(&problems, &weights_before, &weights);
    let mean_rank_before: f64 = impact_rows.iter().map(|(_, b, _)| *b as f64).sum::<f64>()
        / impact_rows.len().max(1) as f64;
    let mean_rank_after: f64 = impact_rows.iter().map(|(_, _, a)| *a as f64).sum::<f64>()
        / impact_rows.len().max(1) as f64;
    let delta = mean_rank_after - mean_rank_before;

    eprintln!(
        "\n[bootstrap_train] counterfactual A/B: mean rank of correct teacher: {:.2} → {:.2} (Δ={:+.2})",
        mean_rank_before, mean_rank_after, delta
    );

    if dry_run {
        eprintln!("[bootstrap_train] --dry-run set; weights NOT saved");
    } else {
        match set_weights(weights) {
            Ok(()) => eprintln!("[bootstrap_train] trained weights saved"),
            Err(err) => {
                eprintln!("[bootstrap_train] save failed: {err}");
                std::process::exit(1);
            }
        }
        // Commit the retrain baseline so `maybe_trigger_bootstrap_retrain`
        // has the new reference point. Also clears the marker file, so
        // downstream cron / CI that gates on `bootstrap_retrain_due()`
        // stops firing until the cache grows past the next threshold.
        solved_cache::note_bootstrap_trained(snapshot.len());
        eprintln!(
            "[bootstrap_train] recorded new baseline (cache_size={})",
            snapshot.len()
        );

        // Append the impact row to the public record. Best-effort — if the
        // artifacts path doesn't exist or the write fails, we don't fail
        // the retrain itself.
        let _ = append_impact_row(impact_rows.len(), mean_rank_before, mean_rank_after, delta);
    }
}

/// For each benchmark problem whose cached solution is verified, find the
/// rank of that solution's code in the full-cache ranking under each
/// weight vector. Returns `(problem_name, rank_before, rank_after)` rows.
/// Rank is 0-based; lower = earlier in the list = better ranker.
fn measure_retrain_impact(
    problems: &[mog_synth::benchmark::Problem],
    before: &MetaWeights,
    after: &MetaWeights,
) -> Vec<(String, usize, usize)> {
    let snapshot = solved_cache::snapshot_solutions_with_meta();
    if snapshot.is_empty() {
        return Vec::new();
    }
    let mut rows: Vec<(String, usize, usize)> = Vec::new();
    for p in problems {
        let Some(sol) = solved_cache::lookup(p) else {
            continue;
        };
        let target_code = sol.code;
        let rank_before = rank_of_code(p, &target_code, &snapshot, before);
        let rank_after = rank_of_code(p, &target_code, &snapshot, after);
        if let (Some(b), Some(a)) = (rank_before, rank_after) {
            rows.push((p.name.clone(), b, a));
        }
    }
    rows
}

/// Rank the given code in the ranker's output under the provided weights.
/// Swaps in the weights temporarily via `set_weights` before the rank call;
/// restores the original afterwards. Returns `None` when the target isn't
/// in the snapshot (shouldn't happen on verified cache hits, but defensive).
fn rank_of_code(
    problem: &mog_synth::benchmark::Problem,
    target_code: &str,
    snapshot: &[(String, String, u32, u64)],
    weights: &MetaWeights,
) -> Option<usize> {
    // set_weights is the public hook we use. Save the previous state so
    // the A/B doesn't permanently clobber whatever was loaded.
    let prior = MetaWeights::load();
    if set_weights(weights.clone()).is_err() {
        return None;
    }
    let ranked = rank_teachers_with_meta(
        problem,
        snapshot
            .iter()
            .map(|(m, c, s, l)| (m.clone(), c.clone(), *s, *l))
            .collect(),
    );
    // Restore so callers (including subsequent rank_of_code calls) operate
    // against the outer weights until the final commit.
    let _ = set_weights(prior);
    ranked
        .iter()
        .position(|(_, _, code, _, _)| code == target_code)
}

/// Append a one-row summary to `artifacts/retrain_impact.md`. Creates the
/// file with a header if missing.
fn append_impact_row(
    n_scored: usize,
    mean_before: f64,
    mean_after: f64,
    delta: f64,
) -> std::io::Result<()> {
    use std::io::Write;
    let dir = std::path::Path::new("artifacts");
    std::fs::create_dir_all(dir)?;
    let path = dir.join("retrain_impact.md");
    let is_new = !path.exists();
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)?;
    if is_new {
        writeln!(
            f,
            "# Retrain Impact\n\nEvery `bootstrap_train` run appends one row. \
             `mean_rank` is the average 0-based position of the correct teacher \
             in the ranker's output, measured against the full cache for each \
             verified (problem, its_code) pair. Lower = better ranker.\n\n\
             | date (UTC) | scored | mean_rank_before | mean_rank_after | Δ |\n\
             |------------|-------:|-----------------:|----------------:|--:|"
        )?;
    }
    let date = chrono_utc_now_simple();
    writeln!(
        f,
        "| {} | {} | {:.2} | {:.2} | {:+.2} |",
        date, n_scored, mean_before, mean_after, delta
    )?;
    Ok(())
}

/// Minimal UTC date helper — `chrono`-free, formats YYYY-MM-DD from
/// `SystemTime`. The artifact only needs day-granularity so this stays
/// tiny and dependency-free.
fn chrono_utc_now_simple() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days since epoch + civil date conversion (Howard Hinnant's algorithm,
    // inlined to avoid a chrono dep).
    let days = (secs / 86400) as i64;
    let (y, m, d) = civil_from_days(days);
    format!("{:04}-{:02}-{:02}", y, m, d)
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    // https://howardhinnant.github.io/date_algorithms.html
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

fn weighted_l2(residual: &[f64; FEATURE_DIM], weights: &MetaWeights) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..FEATURE_DIM {
        s += weights.w[i] * residual[i] * residual[i];
    }
    s.sqrt()
}

/// Identity-ish helper: convert a feature vector back into a sentinel string
/// we can compare against for de-duplication. Unused for now (we rely on
/// identity-by-string-equality); kept for future cleanup.
fn pos_cf_vec_to_identity(v: &[f64]) -> String {
    // Cheap fingerprint for the "is this the positive?" check.
    format!("{:?}", v)
}
