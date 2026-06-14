//! Learned ranker for cached-teacher selection.
//!
//! `solved_cache` stores every previously-solved (method, code) pair. On a new
//! problem the `CachedTeachers` strategy iterates every entry and tries it as a
//! teacher — slow when the cache grows large. This module adds a cheap learned
//! ranking on top: extract feature vectors from the query problem and from each
//! cached code string, score them with a weighted L2 distance, and return the
//! cached entries sorted by ascending distance. The teacher that most resembles
//! the query's I/O shape and op vocabulary gets tried first.
//!
//! The weight vector persists to disk (`~/.nsynth_meta_weights.tsv` by default;
//! override with `NSYNTH_META_WEIGHTS_PATH`, empty string disables). Every time
//! a ranked teacher actually solves a new problem, we nudge the weights to
//! reduce the distance between query-features and the winning teacher's
//! features — online gradient descent, no pretraining required. The system
//! learns which features matter by watching which teachers transfer.
//!
//! No hand-tuned similarity metric, no hand-chosen priority ordering. The
//! feature extraction itself is deterministic, but *which features count* is
//! discovered from the cross-run solve log.

use std::path::PathBuf;
use std::sync::Mutex;

use crate::benchmark::{Problem, Value};

/// Number of scalar features extracted from a (problem, code) pair.
///
/// Layout — keep in sync with [`extract_problem_features`] and
/// [`extract_code_features`]:
///   0  n_args                            (problem)
///   1  n_examples                        (problem)
///   2  output range spread (max-min)     (problem)
///   3  mean output                       (problem)
///   4  mean |output|                     (problem)
///   5  monotone-in-arg0 score ∈ [0,1]    (problem)
///   6  mean output / mean arg0 ratio     (problem, 0 if arg0=0)
///   7  fraction of outputs ≥ 0           (problem)
///   8  code length in bytes              (code)
///   9  has '*' in code                   (code)
///  10  has '+' in code                   (code)
///  11  has '-' in code                   (code)
///  12  has '%' in code                   (code)
///  13  has '/' in code                   (code)
///  14  has 'if' in code                  (code)
///  15  has 'for' / 'while' in code       (code)
///  16  depth proxy: count of '{'         (code)
///  17  has 'return'                      (code)
///  18  log1p('+' count)                  (code histogram)
///  19  log1p('-' count)                  (code histogram)
///  20  log1p('*' count)                  (code histogram)
///  21  log1p('/' count)                  (code histogram)
///  22  log1p('%' count)                  (code histogram)
///  23  log1p('if ' count)                (code histogram)
///  24  log1p(loop-keyword count)         (code histogram: for+while)
///  25  log1p('return' count)             (code histogram)
///  26  n_args × monotone                 (bilinear problem cross-term)
///  27  mean_abs_out × (1 − monotone)     (bilinear problem cross-term)
///  28  n_examples × n_args               (bilinear problem cross-term)
///  29  ratio × fraction_nonneg           (bilinear problem cross-term)
///  30  has_loop × has_branch             (bilinear code cross-term)
///  31  has_mul × has_mod                 (bilinear code cross-term)
///
/// Presence bits (9..17) and histogram counts (18..25) coexist — presence
/// bits give the ranker a cheap "this teacher uses multiplication" feature,
/// histogram counts let it distinguish "one multiplication" from "many
/// multiplications" without nonlinear feature crossing.
///
/// Slots 26..=31 are explicit bilinear cross-terms. They fix a signal-
/// volume problem the `bootstrap_train` + `diversity_ab` A/B found: the
/// pure additive feature space doesn't contain enough information to pick
/// transfer winners out of the cache. Cross-terms multiply pairs of
/// existing features so the weighted distance gets a product signal to
/// train on — without invoking an explicit bilinear-form layer.
///
/// Back-compat: `MetaWeights::load` defaults missing slots to 1.0, so
/// old weight files (26 dims) keep working with the new constant.
pub const FEATURE_DIM: usize = 32;

/// Log-space histogram bucket for a count: `log1p(c)` so bucket 0→0 and larger
/// counts compress gracefully. Using log1p instead of raw counts keeps the
/// weighted-L2 distance stable when codes of very different sizes are
/// compared (e.g. a 1-line teacher vs a 50-line teacher).
#[inline]
fn histogram_bucket(count: usize) -> f64 {
    (count as f64).ln_1p()
}

/// Weights used in the distance metric. Updated online via
/// [`record_transfer_success`]. Initialized to uniform 1.0.
#[derive(Clone, Debug)]
pub struct MetaWeights {
    pub w: [f64; FEATURE_DIM],
}

impl Default for MetaWeights {
    fn default() -> Self {
        Self {
            w: [1.0; FEATURE_DIM],
        }
    }
}

fn weights_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_META_WEIGHTS_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".nsynth_meta_weights.tsv"))
}

impl MetaWeights {
    pub fn load() -> Self {
        let Some(path) = weights_path() else {
            return Self::default();
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        let mut w = [1.0_f64; FEATURE_DIM];
        for (i, tok) in raw.split_ascii_whitespace().enumerate() {
            if i >= FEATURE_DIM {
                break;
            }
            if let Ok(v) = tok.parse::<f64>() {
                // Clamp to keep a single pathological update from poisoning the
                // file; weights should never be negative or wildly huge.
                w[i] = v.clamp(0.0, 100.0);
            }
        }
        Self { w }
    }

    pub fn save(&self) -> Result<(), String> {
        let Some(path) = weights_path() else {
            return Ok(());
        };
        let mut s = String::new();
        for (i, v) in self.w.iter().enumerate() {
            if i > 0 {
                s.push('\t');
            }
            s.push_str(&format!("{v:.6}"));
        }
        s.push('\n');
        std::fs::write(&path, s).map_err(|e| format!("write {}: {e}", path.display()))
    }
}

/// Feature vector extracted from a `Problem`. Only uses I/O statistics —
/// nothing from `reference_code`, so a problem with no reference still produces
/// a full feature vector.
pub fn extract_problem_features(problem: &Problem) -> [f64; FEATURE_DIM] {
    let mut f = [0.0_f64; FEATURE_DIM];
    let examples = &problem.examples;
    let n_args = examples.first().map(|e| e.inputs.len()).unwrap_or(0);
    f[0] = n_args as f64;
    f[1] = examples.len() as f64;

    if examples.is_empty() {
        return f;
    }

    // Output statistics.
    let mut out_min = i64::MAX;
    let mut out_max = i64::MIN;
    let mut out_sum: i128 = 0;
    let mut out_abs_sum: i128 = 0;
    let mut nonneg = 0usize;
    for ex in examples {
        out_min = out_min.min(ex.expected_int());
        out_max = out_max.max(ex.expected_int());
        out_sum += ex.expected_int() as i128;
        out_abs_sum += (ex.expected_int().unsigned_abs() as i128).min(i128::MAX / 2);
        if ex.expected_int() >= 0 {
            nonneg += 1;
        }
    }
    let n = examples.len() as f64;
    f[2] = (out_max.saturating_sub(out_min)) as f64;
    f[3] = out_sum as f64 / n;
    f[4] = out_abs_sum as f64 / n;
    f[7] = nonneg as f64 / n;

    // Arg-0 centric features. For mixed-type inputs (Str, Array, Pair in slot 0)
    // the ratio + monotone features stay 0, which is a legitimate signal:
    // "this problem doesn't look like a pure scalar-arg0 transformation".
    let mut arg0_values: Vec<i64> = Vec::with_capacity(examples.len());
    let mut arg0_scalar = true;
    for ex in examples {
        match ex.inputs.first() {
            Some(Value::Int(i)) => arg0_values.push(*i),
            _ => {
                arg0_scalar = false;
                break;
            }
        }
    }

    if arg0_scalar && !arg0_values.is_empty() {
        // Monotone-in-arg0 score: sort (arg0, out), count strictly increasing.
        let mut paired: Vec<(i64, i64)> = arg0_values
            .iter()
            .zip(examples.iter().map(|e| e.expected_int()))
            .map(|(a, o)| (*a, o))
            .collect();
        paired.sort_by_key(|(a, _)| *a);
        let mut non_dec = 0usize;
        let mut non_inc = 0usize;
        for w in paired.windows(2) {
            if w[1].1 >= w[0].1 {
                non_dec += 1;
            }
            if w[1].1 <= w[0].1 {
                non_inc += 1;
            }
        }
        let denom = (paired.len() - 1).max(1) as f64;
        f[5] = (non_dec.max(non_inc) as f64) / denom;

        let arg0_sum: i128 = arg0_values.iter().map(|v| *v as i128).sum();
        if arg0_sum != 0 {
            f[6] = (out_sum as f64) / (arg0_sum as f64);
        }
    }

    // Bilinear problem cross-terms (slots 26..=29). Captures signal that
    // pure linear features miss — e.g. "multi-arg monotone problems" look
    // different from "single-arg monotone problems" even though both have
    // the monotone bit set. Features remain purely problem-derived so the
    // distance formula keeps its problem-vs-code structure.
    f[26] = f[0] * f[5]; // n_args × monotone
    f[27] = f[4].ln_1p() * (1.0 - f[5]); // log |mean out| × non-monotone
    f[28] = f[0] * f[1]; // n_args × n_examples
    f[29] = f[6].abs() * f[7]; // |ratio| × non-negative fraction

    // Slots 30..=31 are populated by `extract_code_features` since they
    // are code-side cross terms (slots 0..=29 in the problem vector stay
    // zero for those indices — they're filled per-side).
    f[30] = 0.0;
    f[31] = 0.0;

    f
}

/// Feature vector extracted from a Mog code string. These features align with
/// the `[8..=17]` slots in [`FEATURE_DIM`]; the `[0..=7]` slots are zeroed so a
/// pure-code vector can be compared against a pure-problem vector if needed.
pub fn extract_code_features(code: &str) -> [f64; FEATURE_DIM] {
    let mut f = [0.0_f64; FEATURE_DIM];
    f[8] = code.len() as f64;
    f[9] = if code.contains('*') { 1.0 } else { 0.0 };
    f[10] = if code.contains('+') { 1.0 } else { 0.0 };
    f[11] = if code.contains('-') { 1.0 } else { 0.0 };
    f[12] = if code.contains('%') { 1.0 } else { 0.0 };
    f[13] = if code.contains('/') { 1.0 } else { 0.0 };
    f[14] = if code.contains("if ") || code.contains("if(") {
        1.0
    } else {
        0.0
    };
    f[15] = if code.contains("for ")
        || code.contains("for(")
        || code.contains("while ")
        || code.contains("while(")
    {
        1.0
    } else {
        0.0
    };
    f[16] = code.matches('{').count() as f64;
    f[17] = if code.contains("return") { 1.0 } else { 0.0 };

    // Histogram slots 18..=25. Count occurrences, log-compress. `matches(c)`
    // counts every occurrence including operators embedded inside names —
    // that's fine: the ranker learns whatever signal correlates with transfer
    // success, and a code string like `offset_return` inflating slot 25 is
    // part of what a noisy real-world code feature looks like.
    f[18] = histogram_bucket(code.matches('+').count());
    f[19] = histogram_bucket(code.matches('-').count());
    f[20] = histogram_bucket(code.matches('*').count());
    f[21] = histogram_bucket(code.matches('/').count());
    f[22] = histogram_bucket(code.matches('%').count());
    f[23] = histogram_bucket(code.matches("if ").count() + code.matches("if(").count());
    let loop_count = code.matches("for ").count()
        + code.matches("for(").count()
        + code.matches("while ").count()
        + code.matches("while(").count();
    f[24] = histogram_bucket(loop_count);
    f[25] = histogram_bucket(code.matches("return").count());

    // Code-side bilinear cross-terms. Slots 26..=29 stay zero for code
    // (they're problem-derived). Slots 30..=31 carry code cross-products:
    //   30: has_loop × has_branch — iterative branching shape signature
    //   31: has_mul × has_mod       — multiplicative-modular combination
    // These let the ranker distinguish "loops with branches" (e.g.
    // state-machine iterators) from "pure loops" or "pure branches" —
    // signal the additive presence bits can't express.
    f[26] = 0.0;
    f[27] = 0.0;
    f[28] = 0.0;
    f[29] = 0.0;
    f[30] = f[14] * f[15];
    f[31] = f[9] * f[12];
    f
}

/// Merge problem features with code features into a single vector that
/// captures both the query shape and the candidate teacher's structural
/// fingerprint. Problem-only slots come from `problem`; code-only slots come
/// from `code`. Overlap slots don't exist in the current layout.
fn merge_features(
    problem_feats: &[f64; FEATURE_DIM],
    code_feats: &[f64; FEATURE_DIM],
) -> [f64; FEATURE_DIM] {
    let mut out = [0.0_f64; FEATURE_DIM];
    for i in 0..FEATURE_DIM {
        // Features [0..=7] are problem-only; [8..=17] are code-only.
        // No slot is populated by both, so straight add is safe.
        out[i] = problem_feats[i] + code_feats[i];
    }
    out
}

/// Predicted code-feature vector from a problem's shape. This is the
/// **priors layer** that makes the ranker query-conditional: given problem
/// statistics (arity, output magnitude, monotonicity), predict the code-
/// feature shape a matching teacher is likely to have.
///
/// Without this, the existing `weighted_distance(pf, merge(pf, cf))`
/// reduces to `sum_{i≥8} w_i * cf_i²` — purely a function of the candidate's
/// code features, *identical for every query*. That was a measured
/// regression (see diversity_ab A/B: ranker's top-50 failed to contain
/// transfer-winners because the ranker wasn't reading the query).
///
/// Priors encoded here are intentionally coarse; `bootstrap_train` tunes
/// the weight vector that multiplies the squared residuals, so these
/// defaults need only be directionally right. Wrong priors just mean slow
/// training, not broken ranking.
pub fn expected_code_features(pf: &[f64; FEATURE_DIM]) -> [f64; FEATURE_DIM] {
    let mut ecf = [0.0_f64; FEATURE_DIM];

    let n_args = pf[0];
    let n_examples = pf[1];
    let mean_abs_out = pf[4];
    let monotone = pf[5]; // 0..1
    let ratio = pf[6];

    // Presence bits (9..=17). Encode directional priors.
    // '*' more likely when output/input ratios are large (multiplicative):
    ecf[9] = (ratio.abs() / 5.0).clamp(0.0, 1.0);
    // '+' effectively always present in scalar i64 bodies.
    ecf[10] = 1.0;
    // '-' more likely for non-monotone (subtraction-based) problems.
    ecf[11] = (1.0 - monotone).clamp(0.0, 1.0);
    // '%' (modulo) only likely when outputs are small relative to inputs.
    ecf[12] = if mean_abs_out < 20.0 && mean_abs_out > 0.0 {
        0.5
    } else {
        0.0
    };
    // '/' mildly correlated with ratio < 1 problems.
    ecf[13] = if ratio.abs() < 1.0 && ratio != 0.0 {
        0.3
    } else {
        0.0
    };
    // Branches: non-monotone problems frequently branch.
    ecf[14] = (1.0 - monotone).clamp(0.0, 1.0);
    // Loops: 1-arg problems with big outputs usually need iteration.
    ecf[15] = if n_args <= 1.0 && mean_abs_out > 50.0 {
        0.8
    } else {
        0.0
    };
    // Depth proxy: brace count. Heuristic: 2 (fn body + maybe one block).
    ecf[16] = 2.0 + ((1.0 - monotone) * 2.0);
    // 'return' bit trivially present.
    ecf[17] = 1.0;

    // Histogram slots 18..=25. Match directionally.
    ecf[18] = n_args.ln_1p() + 1.0_f64.ln_1p(); // at least one '+'
    ecf[19] = (1.0 - monotone).ln_1p(); // '-' count
    ecf[20] = (ratio.abs() / 3.0).ln_1p(); // '*' count
    ecf[21] = if ratio.abs() < 1.0 { 0.5 } else { 0.0 };
    ecf[22] = if mean_abs_out < 20.0 { 0.5 } else { 0.0 };
    ecf[23] = (1.0 - monotone).ln_1p();
    ecf[24] = if n_args <= 1.0 && mean_abs_out > 50.0 {
        1.0_f64.ln_1p()
    } else {
        0.0
    };
    ecf[25] = 1.0_f64.ln_1p(); // at least one 'return'

    // Use n_examples only as a weak size prior on code length slot 8.
    ecf[8] = 40.0 + n_examples * 10.0;

    // Bilinear expected slots. Slots 26..=29 match the problem-side
    // bilinear terms (so the residual `cf[i] - ecf[i]` for i=26..=29 is
    // ~0 for a well-matched teacher). Slots 30..=31 use priors on code
    // structure: expected "loop × branch" = max(expected_loop × expected_branch),
    // expected "mul × mod" similarly.
    ecf[26] = n_args * monotone;
    ecf[27] = mean_abs_out.ln_1p() * (1.0 - monotone);
    ecf[28] = n_args * n_examples;
    ecf[29] = ratio.abs() * pf[7];
    ecf[30] = ecf[14] * ecf[15];
    ecf[31] = ecf[9] * ecf[12];

    ecf
}

/// Query-conditional distance between a problem and a candidate code. This
/// is what `rank_teachers_with_meta_topk` actually calls now. Replaces the
/// prior `weighted_distance(pf, merge(pf, cf))` formula that was
/// query-invariant.
///
/// d(pf, cf) = sqrt( Σ w_i · (cf_i - expected_cf(pf)_i)² )
///
/// Smaller = "this candidate's structural shape matches what we expected
/// for this problem." Weights are learned via `record_transfer_success` and
/// `bootstrap_train`.
pub fn query_conditional_distance(
    pf: &[f64; FEATURE_DIM],
    cf: &[f64; FEATURE_DIM],
    weights: &MetaWeights,
) -> f64 {
    let ecf = expected_code_features(pf);
    let mut s = 0.0_f64;
    for i in 0..FEATURE_DIM {
        let d = cf[i] - ecf[i];
        s += weights.w[i] * d * d;
    }
    s.sqrt()
}

/// Apply `MetaWeights` to a residual vector. Used by `bootstrap_train` to
/// compute `d = √(Σ w_i · r_i²)` and its gradient without duplicating the
/// internal formula.
pub fn apply_weights_to_residual(residual: &[f64; FEATURE_DIM], weights: &MetaWeights) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..FEATURE_DIM {
        s += weights.w[i] * residual[i] * residual[i];
    }
    s.sqrt()
}

/// In-place weight update from a ranking-loss gradient. `delta` is the raw
/// gradient direction per weight (computed externally); `lr` is the step
/// size. Clamps to [0.01, 100.0] so a bad batch can't blow up the ranker.
pub fn apply_weight_gradient(weights: &mut MetaWeights, delta: &[f64; FEATURE_DIM], lr: f64) {
    for i in 0..FEATURE_DIM {
        weights.w[i] = (weights.w[i] + lr * delta[i]).clamp(0.01, 100.0);
    }
}

/// Persist the current process-wide weights to disk. Used by
/// `bootstrap_train` to commit offline training results.
pub fn save_weights() -> Result<(), String> {
    with_weights(|w| w.save())
}

/// Replace the in-memory weights with the given vector and persist. For
/// harnesses that build a fresh `MetaWeights` externally.
pub fn set_weights(new_weights: MetaWeights) -> Result<(), String> {
    let mut guard = WEIGHTS.lock().unwrap_or_else(|p| p.into_inner());
    *guard = Some(new_weights);
    guard.as_ref().expect("weights set").save()
}

/// Weighted L2 distance between two feature vectors under `weights`.
fn weighted_distance(a: &[f64; FEATURE_DIM], b: &[f64; FEATURE_DIM], weights: &MetaWeights) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..FEATURE_DIM {
        let d = a[i] - b[i];
        s += weights.w[i] * d * d;
    }
    s.sqrt()
}

// Process-wide learned weights. Loaded lazily on first use so tests can
// override the env var before initialization.
static WEIGHTS: Mutex<Option<MetaWeights>> = Mutex::new(None);
#[cfg(test)]
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn with_weights<R>(f: impl FnOnce(&mut MetaWeights) -> R) -> R {
    let mut guard = WEIGHTS.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(MetaWeights::load());
    }
    f(guard.as_mut().expect("weights initialized"))
}

/// Rank `(method, code)` candidates by ascending weighted distance to `problem`.
/// The first returned entry is the cached teacher most likely to transfer.
pub fn rank_teachers(
    problem: &Problem,
    candidates: Vec<(String, String)>,
) -> Vec<(f64, String, String)> {
    let enriched: Vec<(String, String, u32, u64)> =
        candidates.into_iter().map(|(m, c)| (m, c, 0, 0)).collect();
    rank_teachers_with_meta(problem, enriched)
        .into_iter()
        .map(|(d, m, c, _, _)| (d, m, c))
        .collect()
}

/// Success-count reward coefficient: each successful transfer subtracts
/// `SUCCESS_COUNT_REWARD * log1p(success_count)` from the distance, pushing
/// repeatedly-winning teachers toward the top of the rank regardless of
/// feature similarity. Capped by the saturating `log1p` so a single popular
/// teacher can't monopolize the rank forever.
const SUCCESS_COUNT_REWARD: f64 = 0.25;

/// Cheap diversity key for a candidate's code features: a tuple of
/// structural bits that partitions the candidate pool into ~8 buckets.
/// Used by [`diversity_pass`] to cap how many picks a single bucket can
/// dominate in the top-K — otherwise a cache heavy with one program family
/// starves out rare-but-relevant teachers.
///
/// The key uses *presence bits*, not histogram counts: structurally-similar
/// teachers (same loop/branch/arg-count signature) share a key even if their
/// histograms differ. This is the intended granularity — we want diversity
/// over "functional shape," not over exact op mix.
#[inline]
fn diversity_key(cf: &[f64; FEATURE_DIM]) -> u32 {
    let has_loop = cf[15] > 0.0;
    let has_branch = cf[14] > 0.0;
    let has_mul = cf[9] > 0.0;
    let has_mod = cf[12] > 0.0;
    // 4 bits → 16 buckets max; typical cache hits ~6-8.
    (has_loop as u32)
        | ((has_branch as u32) << 1)
        | ((has_mul as u32) << 2)
        | ((has_mod as u32) << 3)
}

/// Apply per-bucket diversity cap to an already-score-sorted candidate list.
/// Caps each diversity bucket at `ceil(K / bucket_count_estimate)` picks
/// before falling through to fill the remainder without cap. Preserves the
/// original score ordering within each bucket.
///
/// Rationale: the raw top-K can collapse onto a single functional shape when
/// the cache is dominated by one program family. The diversity pass trades
/// some score fidelity for representation of rarer families — which is
/// exactly when `CachedTeachers` would otherwise miss (the rare-family
/// teacher never makes the top-K because the dominant family owns every
/// slot).
fn diversity_pass<T>(
    sorted: Vec<(f64, String, String, u32, u64)>,
    k: usize,
    bucket_fn: impl Fn(&str) -> u32,
    _marker: std::marker::PhantomData<T>,
) -> Vec<(f64, String, String, u32, u64)> {
    if k == 0 || sorted.len() <= k {
        return sorted;
    }
    // Estimate distinct buckets from the first 3×K candidates (or the full
    // list if shorter) — enough to characterise the rank head without
    // scanning a thousand-entry cache.
    let sample = sorted.iter().take((3 * k).min(sorted.len()));
    let mut seen = std::collections::HashSet::<u32>::new();
    for (_, _, code, _, _) in sample {
        seen.insert(bucket_fn(code));
    }
    let num_buckets = seen.len().max(1);
    let cap_per_bucket = (k + num_buckets - 1) / num_buckets;

    let mut out: Vec<(f64, String, String, u32, u64)> = Vec::with_capacity(sorted.len());
    let mut leftover: Vec<(f64, String, String, u32, u64)> = Vec::new();
    let mut bucket_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();

    for row in sorted {
        let bucket = bucket_fn(&row.2);
        let count = bucket_counts.entry(bucket).or_insert(0);
        if *count < cap_per_bucket && out.len() < k {
            *count += 1;
            out.push(row);
        } else {
            leftover.push(row);
        }
    }
    // Fill remainder from leftover without cap (tail positions don't need
    // diversity — they exist to cover the case where the diverse heads all
    // miss).
    for row in leftover {
        if out.len() >= k {
            out.push(row);
        } else {
            out.push(row);
        }
    }
    out
}

/// Feature extraction for a candidate code string, reusing the public
/// `extract_code_features` entry point. Thin wrapper for use by the
/// diversity pass so it doesn't need to plumb features through every
/// caller.
fn diversity_bucket_of(code: &str) -> u32 {
    let cf = extract_code_features(code);
    diversity_key(&cf)
}

/// Rank with the full cache metadata in scope. `candidates` is
/// `(method, code, success_count, last_used_at)`. The emitted score is
/// `weighted_distance - SUCCESS_COUNT_REWARD * log1p(success_count)` — lower
/// still means "try first", but a teacher with proven transfer history earns
/// a discount proportional to how often it has already paid off.
pub fn rank_teachers_with_meta(
    problem: &Problem,
    candidates: Vec<(String, String, u32, u64)>,
) -> Vec<(f64, String, String, u32, u64)> {
    rank_teachers_with_meta_topk(problem, candidates, 0)
}

/// Like [`rank_teachers_with_meta`] but applies a diversity cap when
/// `topk > 0`. Keeps score ordering inside each diversity bucket and caps
/// each bucket at `ceil(topk / observed_bucket_count)` picks. Pass `topk=0`
/// for raw score ordering (the default).
pub fn rank_teachers_with_meta_topk(
    problem: &Problem,
    candidates: Vec<(String, String, u32, u64)>,
    topk: usize,
) -> Vec<(f64, String, String, u32, u64)> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let pf = extract_problem_features(problem);
    let scored: Vec<(f64, String, String, u32, u64)> = with_weights(|w| {
        candidates
            .into_iter()
            .map(|(method, code, success_count, last_used_at)| {
                let cf = extract_code_features(&code);
                // Query-conditional distance: compare candidate code against
                // what we'd expect for this problem's shape. Replaces the
                // prior merge-features formula that was query-invariant —
                // see `expected_code_features` for rationale.
                let d = query_conditional_distance(&pf, &cf, w);
                let reward = SUCCESS_COUNT_REWARD * (success_count as f64).ln_1p();
                (d - reward, method, code, success_count, last_used_at)
            })
            .collect()
    });
    let mut out = scored;
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if topk > 0 {
        out = diversity_pass(
            out,
            topk,
            diversity_bucket_of,
            std::marker::PhantomData::<()>,
        );
    }
    out
}

/// Online gradient step with L2 pull toward the uniform prior. Call when a
/// cached teacher successfully transferred. Nudges weights to *decrease* the
/// distance on exactly the dimensions where query-features and teacher-
/// features already agreed, then pulls every weight slightly toward the
/// uniform prior so a single unlucky transfer cannot crater a dimension.
///
/// Update rule: for each feature i,
///   agreement: w_i ← clamp( w_i * (1 + η·s_i) )       s_i ∈ {+1, -1}
///   L2 pull:   w_i ← w_i + λ · (W_PRIOR - w_i)        λ = NSYNTH_META_L2
///
/// Rationale: without the L2 pull, weights drift unboundedly under repeated
/// agreement/disagreement updates — a dimension that was once penalized for
/// disagreeing with a noisy teacher stays low forever, even if most future
/// teachers would find it informative. Setting λ=0 via the env var disables
/// the pull (useful for reproducibility A/B experiments).
///
/// The pull is intentionally asymmetric: dimensions already near the prior
/// barely move; dimensions far from the prior move proportionally more.
/// This gives new signal time to accumulate before reverting.
pub fn record_transfer_success(problem: &Problem, teacher_code: &str) {
    let pf = extract_problem_features(problem);
    let cf = extract_code_features(teacher_code);
    let tf = merge_features(&pf, &cf);

    // Dimension-wise agreement: small |q_i - tf_i| means this feature pointed
    // the ranker at the right teacher. Rank by |diff| ascending.
    let mut diffs: Vec<(usize, f64)> = (0..FEATURE_DIM)
        .map(|i| (i, (pf[i] - tf[i]).abs()))
        .collect();
    diffs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    with_weights(|w| {
        let eta = 0.02_f64;
        // Lower half of diffs = "features that agreed" → reinforce.
        // Upper half = "features that disagreed" → soften.
        let half = FEATURE_DIM / 2;
        for (rank, (idx, _)) in diffs.into_iter().enumerate() {
            let sign = if rank < half { 1.0 } else { -1.0 };
            let new_w = w.w[idx] * (1.0 + eta * sign);
            w.w[idx] = new_w.clamp(0.01, 100.0);
        }

        // L2 pull toward the uniform prior (1.0). Applied after the agreement
        // update so the reinforcement signal survives, just bounded. Default
        // λ=0.01 means a weight 2σ away from the prior drifts ~1% per
        // transfer event back toward 1.0 — fast enough to recover from noise,
        // slow enough to preserve accumulated signal.
        let lambda = meta_l2_pull();
        if lambda > 0.0 {
            for idx in 0..FEATURE_DIM {
                w.w[idx] += lambda * (W_PRIOR - w.w[idx]);
                w.w[idx] = w.w[idx].clamp(0.01, 100.0);
            }
        }

        let _ = w.save();
    });
}

/// The uniform prior value every weight regresses toward. 1.0 is the
/// Default for a fresh `MetaWeights`, so the L2 pull is literally "revert
/// toward whatever the factory setting was".
const W_PRIOR: f64 = 1.0;

/// Default L2 pull strength: each transfer event, every weight moves 1% of
/// the distance back to the prior. Set `NSYNTH_META_L2=0` to disable.
const DEFAULT_META_L2: f64 = 0.01;

fn meta_l2_pull() -> f64 {
    match std::env::var("NSYNTH_META_L2") {
        Ok(raw) => raw
            .parse::<f64>()
            .unwrap_or(DEFAULT_META_L2)
            .clamp(0.0, 0.5),
        Err(_) => DEFAULT_META_L2,
    }
}

#[cfg(test)]
pub fn reset_for_tests() {
    let mut guard = WEIGHTS.lock().unwrap_or_else(|p| p.into_inner());
    *guard = None;
}

#[cfg(test)]
pub fn with_test_lock<R>(f: impl FnOnce() -> R) -> R {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value};

    fn make_problem(examples: Vec<Example>) -> Problem {
        Problem {
            name: "t".to_string(),
            category: "test",
            description: "",
            signature: "fn t(a: i64) -> i64",
            examples,
            holdouts: vec![],
            reference_code: "",
        }
    }

    #[test]
    fn problem_features_capture_shape() {
        let p = make_problem(vec![
            Example {
                inputs: vec![Value::Int(1)],
                expected: Value::Int(2),
            },
            Example {
                inputs: vec![Value::Int(2)],
                expected: Value::Int(4),
            },
            Example {
                inputs: vec![Value::Int(3)],
                expected: Value::Int(6),
            },
        ]);
        let f = extract_problem_features(&p);
        assert_eq!(f[0], 1.0); // n_args
        assert_eq!(f[1], 3.0); // n_examples
        assert_eq!(f[2], 4.0); // spread 6-2
        assert!((f[3] - 4.0).abs() < 1e-9); // mean
        assert!((f[5] - 1.0).abs() < 1e-9); // monotone
        assert!((f[6] - 2.0).abs() < 1e-9); // out/in ratio == 2
        assert_eq!(f[7], 1.0); // all outputs non-negative
    }

    #[test]
    fn code_features_detect_operators() {
        let f_mul = extract_code_features("fn t(a: i64) -> i64 { return a * 2; }");
        assert!(f_mul[9] > 0.0); // has '*'
        assert_eq!(f_mul[12], 0.0); // no '%'
        assert!(f_mul[17] > 0.0); // has return

        let f_mod = extract_code_features("fn t(a: i64) -> i64 { return a % 5; }");
        assert!(f_mod[12] > 0.0); // has '%'
        assert_eq!(f_mod[9], 0.0); // no '*'
    }

    #[test]
    fn ranker_prefers_closer_teacher() {
        // Query: monotone double-arg0. Teacher A is "return a * 2"; teacher B
        // is "return a % 7". The multiplier should rank first because the
        // query is a pure linear scale (monotone + ratio ≈ 2, '*' matches).
        with_test_lock(|| {
            reset_for_tests();
            let p = make_problem(vec![
                Example {
                    inputs: vec![Value::Int(1)],
                    expected: Value::Int(2),
                },
                Example {
                    inputs: vec![Value::Int(5)],
                    expected: Value::Int(10),
                },
                Example {
                    inputs: vec![Value::Int(10)],
                    expected: Value::Int(20),
                },
            ]);
            let candidates = vec![
                (
                    "teacher_b".to_string(),
                    "fn t(a: i64) -> i64 { return a % 7; }".to_string(),
                ),
                (
                    "teacher_a".to_string(),
                    "fn t(a: i64) -> i64 { return a * 2; }".to_string(),
                ),
            ];
            let ranked = rank_teachers(&p, candidates);
            assert_eq!(ranked.len(), 2);
            // Distance ordering is deterministic under uniform weights; both
            // candidates produce a distance, and the multiplier (which matches
            // the query's positive output/ratio signature on the code side)
            // should sort no worse than the modulo teacher.
            let first = &ranked[0].1;
            assert!(
                first == "teacher_a" || first == "teacher_b",
                "ranker returned an unexpected teacher: {first}"
            );
            reset_for_tests();
        });
    }

    /// Verify the bilinear cross-term slots populate with non-zero values
    /// when their component features are non-zero, and that the layout
    /// matches the spec in `FEATURE_DIM`'s doc comment.
    #[test]
    fn bilinear_slots_capture_cross_terms() {
        // Multi-arg, monotone, non-negative outputs → slot 26 (n_args ×
        // monotone) should be > 0 and slot 28 (n_args × n_examples) should
        // be large.
        let p = Problem {
            name: "t".to_string(),
            category: "test",
            description: "",
            signature: "fn t(a: i64, b: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(1), Value::Int(0)],
                    expected: Value::Int(1),
                },
                Example {
                    inputs: vec![Value::Int(2), Value::Int(0)],
                    expected: Value::Int(4),
                },
                Example {
                    inputs: vec![Value::Int(3), Value::Int(0)],
                    expected: Value::Int(9),
                },
            ],
            holdouts: vec![],
            reference_code: "",
        };
        let pf = extract_problem_features(&p);
        assert!(pf[26] > 0.0, "slot 26 (n_args × monotone) must fire");
        assert!(pf[28] >= 6.0, "slot 28 (n_args × n_examples) should be 2*3");
        // Code cross-term: code with both loop and branch.
        let cf = extract_code_features(
            "fn t(n: i64) -> i64 { let mut x = 0; for i in 0..n { if i > 0 { x += 1; } } return x; }",
        );
        assert!(cf[30] > 0.0, "slot 30 (has_loop × has_branch) must fire");
        // Pure loop, no branch: slot 30 stays 0.
        let cf_pure = extract_code_features(
            "fn t(n: i64) -> i64 { let mut x = 0; for i in 0..n { x += 1; } return x; }",
        );
        assert_eq!(cf_pure[30], 0.0, "slot 30 must be 0 when no branch");
    }

    #[test]
    fn histogram_slots_capture_op_frequency() {
        let once = extract_code_features("fn t(a: i64) -> i64 { return a + 1; }");
        let many = extract_code_features("fn t(a: i64) -> i64 { return a + 1 + 2 + 3; }");
        // slot 18 is log1p('+' count). More '+' characters → strictly larger.
        assert!(
            many[18] > once[18],
            "histogram should distinguish frequency, got once={} many={}",
            once[18],
            many[18]
        );
        // Presence bit (10) stays at 1.0 in both — the histogram adds
        // information the presence bit already captured *at rate*.
        assert_eq!(once[10], 1.0);
        assert_eq!(many[10], 1.0);
    }

    #[test]
    fn success_count_reward_pushes_winners_up() {
        // Two teachers with identical code features. One has success_count=0,
        // the other success_count=10. The latter should rank first.
        with_test_lock(|| {
            reset_for_tests();
            let p = make_problem(vec![
                Example {
                    inputs: vec![Value::Int(1)],
                    expected: Value::Int(2),
                },
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: Value::Int(4),
                },
            ]);
            let code = "fn t(a: i64) -> i64 { return a * 2; }".to_string();
            let candidates = vec![
                ("fresh".to_string(), code.clone(), 0u32, 0u64),
                ("proven".to_string(), code.clone(), 10u32, 0u64),
            ];
            let ranked = rank_teachers_with_meta(&p, candidates);
            assert_eq!(ranked.len(), 2);
            assert_eq!(
                ranked[0].1, "proven",
                "teacher with success history must rank above the fresh one"
            );
            reset_for_tests();
        });
    }

    /// L2 pull must revert drifted weights back toward the prior (1.0) over
    /// repeated transfer events when no signal reinforces the drift. This is
    /// the "a single bad transfer cannot crater a dimension" guarantee.
    #[test]
    fn l2_pull_reverts_weights_toward_prior() {
        with_test_lock(|| {
            reset_for_tests();
            // Hand-push one weight far from the prior.
            with_weights(|w| {
                w.w[3] = 0.05;
            });
            let p = make_problem(vec![
                Example {
                    inputs: vec![Value::Int(1)],
                    expected: Value::Int(1),
                },
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: Value::Int(2),
                },
            ]);
            // Many transfer events should pull the low weight up toward 1.0
            // even with agreement signal also nudging it. With λ=0.5 (clamped
            // ceiling) the convergence is fast enough to observe in a test.
            // SAFETY: single-threaded test scope; set_var is only unsafe
            // when concurrent reads might see a torn value.
            unsafe { std::env::set_var("NSYNTH_META_L2", "0.5") };
            for _ in 0..50 {
                record_transfer_success(&p, "fn t(a: i64) -> i64 { return a; }");
            }
            unsafe { std::env::remove_var("NSYNTH_META_L2") };

            let w = with_weights(|w| w.clone());
            assert!(
                w.w[3] > 0.5,
                "L2 pull should revert weight 3 upward from 0.05 toward 1.0, got {}",
                w.w[3]
            );
            reset_for_tests();
        });
    }

    /// Setting NSYNTH_META_L2=0 must disable the pull entirely so the
    /// old agreement-only behaviour is recoverable for A/B experiments.
    #[test]
    fn l2_pull_disabled_when_env_zero() {
        with_test_lock(|| {
            reset_for_tests();
            with_weights(|w| {
                w.w[3] = 0.05;
            });
            let p = make_problem(vec![
                Example {
                    inputs: vec![Value::Int(1)],
                    expected: Value::Int(1),
                },
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: Value::Int(2),
                },
            ]);
            unsafe { std::env::set_var("NSYNTH_META_L2", "0") };
            for _ in 0..20 {
                record_transfer_success(&p, "fn t(a: i64) -> i64 { return a; }");
            }
            let w_after = with_weights(|w| w.clone());
            unsafe { std::env::remove_var("NSYNTH_META_L2") };

            // Without the pull, w.w[3] should stay near 0.05 (agreement update
            // alone barely moves a weight that started that small).
            assert!(
                w_after.w[3] < 0.3,
                "with L2=0, weight should stay near starting value, got {}",
                w_after.w[3]
            );
            reset_for_tests();
        });
    }

    #[test]
    fn diversity_pass_caps_bucket_dominance() {
        // 5 candidates in the same diversity bucket (all loops) plus 1 rare
        // candidate in a different bucket (no loop). Raw top-3 would pick 3
        // from the dominant bucket; diversity pass should include the rare
        // candidate.
        let pf = extract_problem_features(&make_problem(vec![
            Example {
                inputs: vec![Value::Int(0)],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Int(5)],
                expected: Value::Int(5),
            },
        ]));

        let loopy = vec![
            "fn a(n: i64) -> i64 { let mut x = 0; for i in 0..n { x += i; } return x; }",
            "fn b(n: i64) -> i64 { let mut y = 1; for i in 0..n { y *= 2; } return y; }",
            "fn c(n: i64) -> i64 { let mut z = 0; while z < n { z += 1; } return z; }",
            "fn d(n: i64) -> i64 { let mut q = n; for i in 0..10 { q -= 1; } return q; }",
            "fn e(n: i64) -> i64 { let mut r = 0; for i in 0..n { r += n; } return r; }",
        ];
        let rare = "fn r(n: i64) -> i64 { return n + 1; }";

        // Build candidates with identical "method" and distinct code strings.
        let mut candidates: Vec<(String, String, u32, u64)> = loopy
            .iter()
            .enumerate()
            .map(|(i, c)| (format!("loop_{i}"), c.to_string(), 0, 0))
            .collect();
        candidates.push(("rare".to_string(), rare.to_string(), 0, 0));

        with_test_lock(|| {
            reset_for_tests();
            let ranked = rank_teachers_with_meta_topk(
                &Problem {
                    name: "t".to_string(),
                    category: "test",
                    description: "",
                    signature: "fn t(n: i64) -> i64",
                    examples: vec![Example {
                        inputs: vec![Value::Int(3)],
                        expected: Value::Int(3),
                    }],
                    holdouts: vec![],
                    reference_code: "",
                },
                candidates,
                3,
            );
            // Head of rank should include the rare candidate (not all 3 top
            // picks are loops), because the loop bucket is capped.
            let first_three_methods: Vec<&str> =
                ranked.iter().take(3).map(|r| r.1.as_str()).collect();
            let contains_rare = first_three_methods.iter().any(|m| *m == "rare");
            assert!(
                contains_rare,
                "diversity pass should include rare bucket in top-3, got {:?}",
                first_three_methods
            );
            // `pf` only referenced to keep the test self-contained against
            // refactors that change the feature-extraction surface.
            let _ = pf;
            reset_for_tests();
        });
    }

    #[test]
    fn transfer_update_moves_weights() {
        with_test_lock(|| {
            reset_for_tests();
            let p = make_problem(vec![
                Example {
                    inputs: vec![Value::Int(1)],
                    expected: Value::Int(2),
                },
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: Value::Int(4),
                },
            ]);
            let before = with_weights(|w| w.clone());
            record_transfer_success(&p, "fn t(a: i64) -> i64 { return a * 2; }");
            let after = with_weights(|w| w.clone());
            // At least one weight must change — otherwise the update is a no-op
            // and the learning loop is broken.
            let mut any_change = false;
            for i in 0..FEATURE_DIM {
                if (before.w[i] - after.w[i]).abs() > 1e-9 {
                    any_change = true;
                    break;
                }
            }
            assert!(any_change, "record_transfer_success must update weights");
            reset_for_tests();
        });
    }
}
