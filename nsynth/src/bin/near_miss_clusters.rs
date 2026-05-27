//! Cluster the records in `artifacts/transfer_failures.jsonl` by the
//! structural shape of the problem the solver *wanted* to solve but couldn't.
//! The attempted-teacher method strings + problem shape are cheap proxies
//! for "what program family was the system nearly able to express?"
//!
//! Output: k clusters, each with a sample of problem names and the most
//! common methods among their attempted (failed) teachers. Tight clusters
//! are concrete solver-work prioritization signals — they're the shapes the
//! system *keeps* trying and *keeps* failing on.
//!
//! Input schema (emitted by strategy.rs::log_teacher_miss):
//!   {
//!     "problem":"fibonacci_v0",
//!     "n_args":1,
//!     "n_examples":4,
//!     "n_attempted":3,
//!     "reason":"all_teachers_missed",
//!     "attempted":[{"method":"synth_gradient","preview":"fn f(...){...}"},...]
//!   }
//!
//! Usage:
//!     cargo run --release --bin near_miss_clusters -- \
//!         [--in artifacts/transfer_failures.jsonl]    default input
//!         [--k 4]                                     cluster count
//!         [--min-cluster-size 2]                      suppress singletons
//!         [--seed 42]                                 RNG seed
//!         [--json]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use serde::Deserialize;

// ─── CLI ─────────────────────────────────────────────────────────────────────

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

// ─── Input schema ────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct AttemptedTeacher {
    method: String,
    #[serde(default)]
    #[allow(dead_code)]
    preview: String,
}

#[derive(Deserialize, Debug)]
struct MissRow {
    problem: String,
    n_args: usize,
    n_examples: usize,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    attempted: Vec<AttemptedTeacher>,
}

// ─── Feature extraction ──────────────────────────────────────────────────────

/// Per-miss feature vector. Structural + attempted-teacher method fingerprint.
/// Fixed layout; keep in sync with the display below.
///
/// Slots 0..=2  — problem shape (n_args, n_examples, bucket-of-reason)
/// Slots 3..=6 — attempted-teacher method fingerprints (one-hot per family)
/// Slot 7 — n_attempted (how many teachers fired before giving up)
const MISS_FEATURE_DIM: usize = 8;
type MissFeatures = [f64; MISS_FEATURE_DIM];

fn method_family(method: &str) -> usize {
    // Coarse family buckets — matches the dominant method prefixes that show
    // up in solved_cache. Unknown methods fall into "other" (bucket 6).
    if method.starts_with("synth_gradient") || method.starts_with("cached_teachers") {
        3
    } else if method.starts_with("univ_arr") || method.starts_with("arr_gradient") {
        4
    } else if method.starts_with("search_") {
        5
    } else if method.starts_with("enumerative") || method.starts_with("expr_") {
        6
    } else {
        // "other" — falls into slot 6 so unfamiliar methods don't get dropped.
        6
    }
}

fn extract_miss_features(row: &MissRow) -> MissFeatures {
    let mut f = [0.0_f64; MISS_FEATURE_DIM];
    f[0] = row.n_args as f64;
    f[1] = (row.n_examples as f64).ln_1p();
    f[2] = if row.reason == "budget_exceeded" {
        1.0
    } else {
        0.0
    };
    for t in &row.attempted {
        let slot = method_family(&t.method);
        if slot < MISS_FEATURE_DIM {
            f[slot] += 1.0;
        }
    }
    f[7] = (row.attempted.len() as f64).ln_1p();
    f
}

// ─── K-means (same shape as cluster_drift; kept self-contained) ─────────────

fn squared_distance(a: &MissFeatures, b: &MissFeatures) -> f64 {
    let mut s = 0.0;
    for i in 0..MISS_FEATURE_DIM {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

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
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn kmeans_plus_plus_init(
    points: &[MissFeatures],
    k: usize,
    rng: &mut XorShift64,
) -> Vec<MissFeatures> {
    let n = points.len();
    assert!(n >= k);
    let mut centroids = Vec::with_capacity(k);
    centroids.push(points[(rng.next_u64() as usize) % n]);
    while centroids.len() < k {
        let mut dists = Vec::with_capacity(n);
        let mut total = 0.0;
        for p in points {
            let mut best = f64::MAX;
            for c in &centroids {
                best = best.min(squared_distance(p, c));
            }
            dists.push(best);
            total += best;
        }
        if total <= f64::MIN_POSITIVE {
            centroids.push(points[(rng.next_u64() as usize) % n]);
            continue;
        }
        let target = rng.next_f64() * total;
        let mut acc = 0.0;
        for (i, d) in dists.iter().enumerate() {
            acc += d;
            if acc >= target {
                centroids.push(points[i]);
                break;
            }
        }
    }
    centroids
}

fn assign_clusters(points: &[MissFeatures], centroids: &[MissFeatures]) -> Vec<usize> {
    points
        .iter()
        .map(|p| {
            let mut best = 0usize;
            let mut best_d = f64::MAX;
            for (i, c) in centroids.iter().enumerate() {
                let d = squared_distance(p, c);
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            best
        })
        .collect()
}

fn recompute_centroids(
    points: &[MissFeatures],
    assignments: &[usize],
    k: usize,
) -> Vec<MissFeatures> {
    let mut sums: Vec<MissFeatures> = vec![[0.0_f64; MISS_FEATURE_DIM]; k];
    let mut counts = vec![0usize; k];
    for (p, &a) in points.iter().zip(assignments.iter()) {
        for i in 0..MISS_FEATURE_DIM {
            sums[a][i] += p[i];
        }
        counts[a] += 1;
    }
    for (c, n) in sums.iter_mut().zip(counts.iter()) {
        if *n > 0 {
            for v in c.iter_mut() {
                *v /= *n as f64;
            }
        }
    }
    sums
}

fn kmeans(points: &[MissFeatures], k: usize, max_iter: usize, seed: u64) -> Vec<usize> {
    let mut rng = XorShift64::new(seed);
    let mut centroids = kmeans_plus_plus_init(points, k, &mut rng);
    let mut assignments = assign_clusters(points, &centroids);
    for _ in 0..max_iter {
        centroids = recompute_centroids(points, &assignments, k);
        let new_assignments = assign_clusters(points, &centroids);
        if new_assignments == assignments {
            break;
        }
        assignments = new_assignments;
    }
    assignments
}

// ─── Entry ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let in_path =
        arg_value(&args, "--in").unwrap_or_else(|| "artifacts/transfer_failures.jsonl".to_string());
    let k: usize = arg_value(&args, "--k")
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let min_size: usize = arg_value(&args, "--min-cluster-size")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let seed: u64 = arg_value(&args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);
    let json_mode = has_flag(&args, "--json");

    let file = match File::open(&in_path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("[near_miss_clusters] cannot open {}: {err}", in_path);
            eprintln!("[near_miss_clusters] hint: set NSYNTH_LOG_TEACHER_FAILURES=1 during a bench run to populate it");
            std::process::exit(1);
        }
    };

    let mut rows: Vec<MissRow> = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(row) = serde_json::from_str::<MissRow>(&line) {
            rows.push(row);
        }
    }

    if rows.len() < k {
        eprintln!(
            "[near_miss_clusters] {} miss rows in {}, k={} — need more misses for clustering",
            rows.len(),
            in_path,
            k
        );
        return;
    }

    let points: Vec<MissFeatures> = rows.iter().map(extract_miss_features).collect();
    let assignments = kmeans(&points, k, 50, seed);

    // Group + report.
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (idx, &a) in assignments.iter().enumerate() {
        clusters[a].push(idx);
    }

    if json_mode {
        for (cluster_id, members) in clusters.iter().enumerate() {
            if members.len() < min_size {
                continue;
            }
            for &idx in members {
                let row = &rows[idx];
                let methods: Vec<&str> = row.attempted.iter().map(|a| a.method.as_str()).collect();
                println!(
                    r#"{{"cluster":{},"problem":"{}","n_args":{},"n_examples":{},"n_attempted":{},"attempted_methods":{:?}}}"#,
                    cluster_id,
                    json_escape(&row.problem),
                    row.n_args,
                    row.n_examples,
                    row.attempted.len(),
                    methods,
                );
            }
        }
        eprintln!(
            "[near_miss_clusters] {} misses → k={} clusters (min_size={})",
            rows.len(),
            k,
            min_size
        );
    } else {
        eprintln!(
            "[near_miss_clusters] {} misses → k={} clusters (seed={}, min_size={})",
            rows.len(),
            k,
            seed,
            min_size
        );
        for (cluster_id, members) in clusters.iter().enumerate() {
            if members.len() < min_size {
                continue;
            }
            println!(
                "\n── cluster {} — {} near-misses ──",
                cluster_id,
                members.len()
            );
            // Method frequency across the cluster's attempted teachers.
            let mut method_counts: BTreeMap<&str, usize> = BTreeMap::new();
            for &idx in members {
                for t in &rows[idx].attempted {
                    *method_counts.entry(t.method.as_str()).or_insert(0) += 1;
                }
            }
            let mut method_rows: Vec<(&&str, &usize)> = method_counts.iter().collect();
            method_rows.sort_by(|a, b| b.1.cmp(a.1));
            print!("  top attempted teachers:");
            for (m, n) in method_rows.iter().take(5) {
                print!("  {}({})", m, n);
            }
            println!();
            // Sample problem names.
            print!("  problems:");
            for &idx in members.iter().take(8) {
                print!(" {}", rows[idx].problem);
            }
            if members.len() > 8 {
                print!(" ... (+{} more)", members.len() - 8);
            }
            println!();
        }
    }
}
