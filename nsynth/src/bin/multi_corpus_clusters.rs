//! Cross-source cluster analysis: reads multiple labelled JSONL corpora
//! (one per "source" — e.g. arm64.jsonl, riscv.jsonl, x86.jsonl), extracts
//! features for each record, clusters the combined set, and reports which
//! clusters contain entries from multiple sources (source-invariant program
//! families) vs clusters dominated by a single source (source-specific).
//!
//! This is the cross-ISA scaffolding. Plug in JSONL from different
//! architecture emulators and the tool answers:
//!   - Which program shapes are the same across ISAs?
//!     (clusters with ≥ 2 source labels)
//!   - Which shapes only appear in one ISA?
//!     (clusters dominated by a single label → likely ISA-idiom)
//!
//! Input JSONL schema — matches `jsonl_harvest`'s expected input:
//!   {"name": "...", "signature": "...", "examples": [{"inputs": [...], "expected": N}]}
//!
//! Features extracted from each record use the same 26-dim layout as
//! `meta_learner::extract_problem_features + extract_code_features` so
//! multi-source rankings remain compatible with the rest of the system.
//!
//! Usage:
//!     cargo run --release --bin multi_corpus_clusters -- \
//!         --source arm64:path/to/arm64.jsonl              \
//!         --source riscv:path/to/riscv.jsonl              \
//!         [--k 5] [--seed 42] [--json]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use serde::Deserialize;

use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::meta_learner::{extract_problem_features, FEATURE_DIM};

// ─── CLI ─────────────────────────────────────────────────────────────────────

fn arg_values(args: &[String], flag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == flag {
            out.push(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

// ─── Input schema (matches jsonl_harvest) ────────────────────────────────────

#[derive(Deserialize, Debug)]
struct InputExample {
    inputs: Vec<serde_json::Value>,
    expected: i64,
}

#[derive(Deserialize, Debug)]
struct InputProblem {
    name: String,
    signature: String,
    examples: Vec<InputExample>,
}

fn value_from_json(v: &serde_json::Value) -> Option<Value> {
    if let Some(i) = v.as_i64() {
        return Some(Value::Int(i));
    }
    if let Some(obj) = v.as_object() {
        if let Some(n) = obj.get("Int").and_then(|x| x.as_i64()) {
            return Some(Value::Int(n));
        }
    }
    None
}

fn to_problem(input: InputProblem) -> Option<Problem> {
    let mut examples = Vec::with_capacity(input.examples.len());
    for ex in input.examples {
        let mut ins = Vec::with_capacity(ex.inputs.len());
        for v in ex.inputs {
            let Some(val) = value_from_json(&v) else {
                return None;
            };
            ins.push(val);
        }
        examples.push(Example {
            inputs: ins,
            expected: Value::Int(ex.expected),
        });
    }
    let signature: &'static str = Box::leak(input.signature.into_boxed_str());
    Some(Problem {
        name: input.name,
        category: "multi_corpus",
        description: "",
        signature,
        examples,
        holdouts: vec![],
        reference_code: "",
    })
}

fn load_source(label: &str, path: &str) -> Vec<(String, String, Problem)> {
    let Ok(file) = File::open(path) else {
        eprintln!(
            "[multi_corpus_clusters] cannot open {}: skipping source '{}'",
            path, label
        );
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: InputProblem = match serde_json::from_str(&line) {
            Ok(p) => p,
            Err(_) => continue,
        };
        if let Some(p) = to_problem(parsed) {
            out.push((label.to_string(), p.name.clone(), p));
        }
    }
    out
}

// ─── K-means (same idiom as the other cluster binaries) ────────────────────

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

// ─── Entry ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sources = arg_values(&args, "--source");
    let k: usize = arg_value(&args, "--k")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let seed: u64 = arg_value(&args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);
    let json_mode = has_flag(&args, "--json");

    if sources.len() < 2 {
        eprintln!(
            "[multi_corpus_clusters] at least two --source LABEL:PATH required; got {}",
            sources.len()
        );
        std::process::exit(2);
    }

    let mut records: Vec<(String, String, Problem)> = Vec::new();
    for spec in sources {
        let (label, path) = match spec.split_once(':') {
            Some((l, p)) => (l.to_string(), p.to_string()),
            None => {
                eprintln!(
                    "[multi_corpus_clusters] bad --source '{}': expected LABEL:PATH",
                    spec
                );
                continue;
            }
        };
        let loaded = load_source(&label, &path);
        eprintln!(
            "[multi_corpus_clusters] {} ← {} records",
            label,
            loaded.len()
        );
        records.extend(loaded);
    }

    if records.len() < k {
        eprintln!(
            "[multi_corpus_clusters] only {} records total, k={} — need more for clustering",
            records.len(),
            k
        );
        return;
    }

    let points: Vec<Point> = records
        .iter()
        .map(|(_, _, p)| extract_problem_features(p))
        .collect();

    let assignments = kmeans(&points, k, 50, seed);

    // Per-cluster: source-label histogram.
    let mut cluster_sources: Vec<BTreeMap<String, Vec<String>>> = vec![BTreeMap::new(); k];
    for ((label, name, _), &c) in records.iter().zip(assignments.iter()) {
        cluster_sources[c]
            .entry(label.clone())
            .or_default()
            .push(name.clone());
    }

    if json_mode {
        for (cluster_id, by_source) in cluster_sources.iter().enumerate() {
            let total: usize = by_source.values().map(|v| v.len()).sum();
            let num_sources = by_source.len();
            println!(
                r#"{{"cluster":{},"total":{},"num_sources":{},"sources":{:?}}}"#,
                cluster_id, total, num_sources, by_source,
            );
        }
    } else {
        eprintln!(
            "[multi_corpus_clusters] {} total records → k={} clusters (seed={})",
            records.len(),
            k,
            seed
        );
        println!(
            "\n{:<10}  {:<9}  {:<25}  {}",
            "cluster", "total", "sources", "signature"
        );
        println!("{}", "─".repeat(100));
        for (cluster_id, by_source) in cluster_sources.iter().enumerate() {
            let total: usize = by_source.values().map(|v| v.len()).sum();
            let num_sources = by_source.len();
            let marker = if num_sources >= 2 {
                "★ cross"
            } else {
                "  same"
            };
            let src_list: Vec<String> = by_source
                .iter()
                .map(|(src, names)| format!("{}({})", src, names.len()))
                .collect();
            println!(
                "{}  {:<10}  {:<9}  {:<25}  {}",
                marker,
                cluster_id,
                total,
                src_list.join(","),
                if total == 0 {
                    "(empty)".to_string()
                } else {
                    // sample a name from the first source
                    by_source
                        .values()
                        .next()
                        .and_then(|names| names.first())
                        .cloned()
                        .unwrap_or_default()
                },
            );
        }
        println!("\n★ clusters: program families appearing in ≥2 sources (cross-source invariant)");
        println!("  same clusters: single-source families (likely source-specific idioms)");
    }
}
