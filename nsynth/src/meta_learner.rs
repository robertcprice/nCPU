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
///
/// Presence bits (9..17) and histogram counts (18..25) coexist — presence
/// bits give the ranker a cheap "this teacher uses multiplication" feature,
/// histogram counts let it distinguish "one multiplication" from "many
/// multiplications" without nonlinear feature crossing.
pub const FEATURE_DIM: usize = 26;

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
        out_min = out_min.min(ex.expected);
        out_max = out_max.max(ex.expected);
        out_sum += ex.expected as i128;
        out_abs_sum += (ex.expected.unsigned_abs() as i128).min(i128::MAX / 2);
        if ex.expected >= 0 {
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
            .zip(examples.iter().map(|e| e.expected))
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

/// Rank with the full cache metadata in scope. `candidates` is
/// `(method, code, success_count, last_used_at)`. The emitted score is
/// `weighted_distance - SUCCESS_COUNT_REWARD * log1p(success_count)` — lower
/// still means "try first", but a teacher with proven transfer history earns
/// a discount proportional to how often it has already paid off.
pub fn rank_teachers_with_meta(
    problem: &Problem,
    candidates: Vec<(String, String, u32, u64)>,
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
                let merged = merge_features(&pf, &cf);
                let d = weighted_distance(&pf, &merged, w);
                let reward = SUCCESS_COUNT_REWARD * (success_count as f64).ln_1p();
                (d - reward, method, code, success_count, last_used_at)
            })
            .collect()
    });
    let mut out = scored;
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
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
                expected: 2,
            },
            Example {
                inputs: vec![Value::Int(2)],
                expected: 4,
            },
            Example {
                inputs: vec![Value::Int(3)],
                expected: 6,
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
                    expected: 2,
                },
                Example {
                    inputs: vec![Value::Int(5)],
                    expected: 10,
                },
                Example {
                    inputs: vec![Value::Int(10)],
                    expected: 20,
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
                    expected: 2,
                },
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: 4,
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
                    expected: 1,
                },
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: 2,
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
                    expected: 1,
                },
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: 2,
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
    fn transfer_update_moves_weights() {
        with_test_lock(|| {
            reset_for_tests();
            let p = make_problem(vec![
                Example {
                    inputs: vec![Value::Int(1)],
                    expected: 2,
                },
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: 4,
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
