//! Cluster the persistent solved cache into program families via k-means on
//! the 26-dim code-feature space. Surfaces "which teachers belong to the
//! same functional shape" without any name-based grouping — so the fibonacci,
//! lucas, and tribonacci teachers end up in the same cluster because their
//! feature histograms agree, even though their method names share no tokens.
//!
//! This is the visualization the question "has the system discovered
//! abstractions?" has been missing. A cache of 94 entries turns into 5-8
//! clusters; each cluster preview reveals the discovered program family.
//!
//! Pure Rust, no external deps. K-means uses k-means++ init for reproducible
//! convergence from a fixed seed.
//!
//! Usage:
//!     cargo run --release --bin teacher_clusters -- \
//!         [--k 5]                     number of clusters (default 5)
//!         [--max-iter 50]             k-means iteration cap
//!         [--seed 42]                 RNG seed for k-means++ init
//!         [--json]                    JSONL instead of human-readable

use mog_synth::meta_learner::{extract_code_features, FEATURE_DIM};
use mog_synth::solved_cache;

// ─── CLI ─────────────────────────────────────────────────────────────────────

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
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

fn code_preview(code: &str, limit: usize) -> String {
    let first = code
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim_start();
    if first.len() <= limit {
        first.to_string()
    } else {
        format!("{}…", &first[..limit])
    }
}

// ─── K-means with k-means++ seeding ──────────────────────────────────────────

type Point = [f64; FEATURE_DIM];

fn squared_distance(a: &Point, b: &Point) -> f64 {
    let mut s = 0.0;
    for i in 0..FEATURE_DIM {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Deterministic xorshift RNG so cluster assignments are reproducible given
/// a seed. Avoids the `rand` dependency; we only need uniform f64s for
/// k-means++ selection.
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
        // 53-bit mantissa in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// k-means++ initial centroid selection. First centroid is chosen uniformly;
/// each subsequent centroid is chosen from the remaining points with
/// probability proportional to D(x)² where D(x) is the distance to the
/// nearest already-chosen centroid. This gives strictly better convergence
/// than random init on clusters with unequal variance.
fn kmeans_plus_plus_init(points: &[Point], k: usize, rng: &mut XorShift64) -> Vec<Point> {
    let n = points.len();
    assert!(n >= k, "need at least k points to initialise k centroids");
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
            // All points coincide with existing centroids — fall back to
            // uniform random pick to avoid div-by-zero. Shouldn't happen on
            // real cache data, but keeps k-means++ robust.
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

fn assign_clusters(points: &[Point], centroids: &[Point]) -> Vec<usize> {
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

fn recompute_centroids(points: &[Point], assignments: &[usize], k: usize) -> Vec<Point> {
    let mut sums: Vec<Point> = vec![[0.0_f64; FEATURE_DIM]; k];
    let mut counts = vec![0usize; k];
    for (p, &a) in points.iter().zip(assignments.iter()) {
        for i in 0..FEATURE_DIM {
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

fn kmeans(points: &[Point], k: usize, max_iter: usize, seed: u64) -> Vec<usize> {
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
    let k: usize = arg_value(&args, "--k")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let max_iter: usize = arg_value(&args, "--max-iter")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let seed: u64 = arg_value(&args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);
    let json_mode = has_flag(&args, "--json");

    let snapshot = solved_cache::snapshot_solutions_with_meta();
    if snapshot.is_empty() {
        eprintln!("[teacher_clusters] cache is empty");
        return;
    }
    if snapshot.len() < k {
        eprintln!(
            "[teacher_clusters] cache has {} entries, k={} — need at least {} for meaningful clustering",
            snapshot.len(),
            k,
            k
        );
        return;
    }

    let points: Vec<Point> = snapshot
        .iter()
        .map(|(_, code, _, _)| extract_code_features(code))
        .collect();

    let assignments = kmeans(&points, k, max_iter, seed);

    // Group teachers by cluster.
    let mut clusters: Vec<Vec<(String, String, u32, u64)>> = vec![Vec::new(); k];
    for (row, &a) in snapshot.into_iter().zip(assignments.iter()) {
        clusters[a].push(row);
    }

    if json_mode {
        for (cluster_id, members) in clusters.iter().enumerate() {
            for (method, code, success_count, last_used_at) in members {
                println!(
                    r#"{{"cluster":{},"method":"{}","success_count":{},"last_used_at":{},"code":"{}"}}"#,
                    cluster_id,
                    json_escape(method),
                    success_count,
                    last_used_at,
                    json_escape(code),
                );
            }
        }
    } else {
        eprintln!(
            "[teacher_clusters] clustering {} teachers into k={} groups (seed={})",
            points.len(),
            k,
            seed
        );
        for (cluster_id, members) in clusters.iter().enumerate() {
            if members.is_empty() {
                println!(
                    "\n── cluster {} (empty — centroid without assignments) ──",
                    cluster_id
                );
                continue;
            }
            println!(
                "\n── cluster {} — {} teachers ──",
                cluster_id,
                members.len()
            );
            // Sort by success_count desc for a more informative preview.
            let mut sorted = members.clone();
            sorted.sort_by(|a, b| b.2.cmp(&a.2));
            for (method, code, sc, _) in sorted.iter().take(8) {
                let method_fmt = if method.len() > 30 {
                    format!("{}…", &method[..29])
                } else {
                    method.clone()
                };
                println!(
                    "  {:<6}  {:<30}  {}",
                    sc,
                    method_fmt,
                    code_preview(code, 50)
                );
            }
            if members.len() > 8 {
                println!("  ... {} more", members.len() - 8);
            }
        }
    }
}
