//! Track how cached teachers move between discovered program families over
//! time. Two modes:
//!
//! 1. `--snapshot [--out path.jsonl]`
//!    Clusters the current solved cache with k-means, writes
//!    `{fingerprint_hash, cluster_id, method, code_preview}` rows to a
//!    timestamped JSONL under `artifacts/cluster_history/` by default.
//!
//! 2. `--diff old.jsonl new.jsonl`
//!    Joins two snapshot files by fingerprint_hash, reports entries whose
//!    `cluster_id` changed. A teacher that moved clusters is evidence the
//!    system's representation of "what this program does" has shifted —
//!    usually from weight-update drift or from new cache entries pulling
//!    cluster centroids around.
//!
//! Pure Rust, no external deps. Fingerprint is a cheap 64-bit hash of the
//! (method, code) pair — deterministic, stable across runs.
//!
//! Usage:
//!     cargo run --release --bin cluster_drift -- --snapshot
//!     cargo run --release --bin cluster_drift -- --diff \
//!         artifacts/cluster_history/1700000000.jsonl   \
//!         artifacts/cluster_history/1700500000.jsonl

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

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

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Deterministic 64-bit hash of a (method, code) pair. We avoid a crypto
/// hash dep by composing FNV-1a over both strings — collision-resistant
/// enough at the cache scale (~100s of entries).
fn fingerprint(method: &str, code: &str) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    for b in method.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= 0xff; // separator
    h = h.wrapping_mul(0x100000001b3);
    for b in code.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ─── K-means (duplicated from teacher_clusters — keep binary self-contained)

type Point = [f64; FEATURE_DIM];

fn squared_distance(a: &Point, b: &Point) -> f64 {
    let mut s = 0.0;
    for i in 0..FEATURE_DIM {
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

fn kmeans_plus_plus_init(points: &[Point], k: usize, rng: &mut XorShift64) -> Vec<Point> {
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

// ─── Snapshot mode ──────────────────────────────────────────────────────────

fn do_snapshot(k: usize, seed: u64, out_path: Option<String>) {
    let snapshot = solved_cache::snapshot_solutions_with_meta();
    if snapshot.len() < k {
        eprintln!(
            "[cluster_drift] cache has {} entries, k={} — need more for snapshot",
            snapshot.len(),
            k
        );
        return;
    }
    let points: Vec<Point> = snapshot
        .iter()
        .map(|(_, code, _, _)| extract_code_features(code))
        .collect();
    let assignments = kmeans(&points, k, 50, seed);

    let ts = now_epoch();
    let path = out_path.unwrap_or_else(|| format!("artifacts/cluster_history/{}.jsonl", ts));
    if let Some(parent) = Path::new(&path).parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            eprintln!("[cluster_drift] cannot create {:?}: {err}", parent);
            std::process::exit(1);
        }
    }
    let mut file = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(err) => {
            eprintln!("[cluster_drift] cannot open {}: {err}", path);
            std::process::exit(1);
        }
    };

    for ((method, code, sc, _), cluster_id) in snapshot.iter().zip(assignments.iter()) {
        let fp = fingerprint(method, code);
        let preview: String = code
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .chars()
            .take(60)
            .collect();
        let row = format!(
            r#"{{"fingerprint":{},"cluster":{},"method":"{}","success_count":{},"preview":"{}"}}"#,
            fp,
            cluster_id,
            json_escape(method),
            sc,
            json_escape(&preview),
        );
        let _ = writeln!(file, "{}", row);
    }
    eprintln!(
        "[cluster_drift] snapshot: {} teachers → k={} clusters → {}",
        points.len(),
        k,
        path
    );
}

// ─── Diff mode ──────────────────────────────────────────────────────────────

fn load_snapshot(path: &str) -> BTreeMap<u64, (usize, String, String)> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("[cluster_drift] cannot open {}: {err}", path);
            std::process::exit(1);
        }
    };
    let mut out = BTreeMap::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if line.is_empty() {
            continue;
        }
        let parsed: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let fp = parsed.get("fingerprint").and_then(|v| v.as_u64());
        let cluster = parsed
            .get("cluster")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let method = parsed
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let preview = parsed
            .get("preview")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if let (Some(fp), Some(cluster)) = (fp, cluster) {
            out.insert(fp, (cluster, method, preview));
        }
    }
    out
}

fn do_diff(old_path: &str, new_path: &str) {
    let old = load_snapshot(old_path);
    let new = load_snapshot(new_path);
    let mut moved: Vec<(u64, usize, usize, String, String)> = Vec::new();
    let mut added = 0usize;
    let mut removed = 0usize;

    for (fp, (new_cluster, method, preview)) in &new {
        match old.get(fp) {
            Some((old_cluster, _, _)) if old_cluster != new_cluster => {
                moved.push((
                    *fp,
                    *old_cluster,
                    *new_cluster,
                    method.clone(),
                    preview.clone(),
                ));
            }
            None => added += 1,
            _ => {}
        }
    }
    for fp in old.keys() {
        if !new.contains_key(fp) {
            removed += 1;
        }
    }

    println!("── cluster_drift: {} → {} ──", old_path, new_path);
    println!("old entries:   {}", old.len());
    println!("new entries:   {}", new.len());
    println!("added:         {}", added);
    println!("removed:       {}", removed);
    println!("moved cluster: {}", moved.len());
    if !moved.is_empty() {
        println!("\nTeachers whose cluster changed:");
        for (_, from, to, method, preview) in moved.iter().take(20) {
            println!("  cluster {} → {}  {:<28}  {}", from, to, method, preview);
        }
        if moved.len() > 20 {
            println!("  ... {} more", moved.len() - 20);
        }
    }
}

// ─── Entry ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode_snapshot = has_flag(&args, "--snapshot");
    let mode_diff = has_flag(&args, "--diff");

    if mode_snapshot == mode_diff {
        eprintln!("[cluster_drift] exactly one of --snapshot or --diff required");
        std::process::exit(2);
    }

    if mode_snapshot {
        let k: usize = arg_value(&args, "--k")
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        let seed: u64 = arg_value(&args, "--seed")
            .and_then(|v| v.parse().ok())
            .unwrap_or(42);
        let out = arg_value(&args, "--out");
        do_snapshot(k, seed, out);
    } else {
        // --diff old new (positional after the flag)
        let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
        if positional.len() < 2 {
            eprintln!("[cluster_drift] --diff requires two paths: old_snapshot new_snapshot");
            std::process::exit(2);
        }
        do_diff(positional[0], positional[1]);
    }
}
