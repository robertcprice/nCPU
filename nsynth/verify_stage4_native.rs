/// Stage 4 Time-Parameterized Synthesis Verification
///
/// Directly tests synthesis of representative benchmarks without CLI overhead.
/// Reports: pass_count/total, mean_solve_time, method_distribution, failures.

use std::time::Instant;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct VerificationResult {
    problem_name: String,
    variant: usize,
    success: bool,
    solve_time_secs: f64,
    method: String,
    code_length: usize,
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct VerificationReport {
    pass_count: usize,
    total: usize,
    mean_solve_time_secs: f64,
    method_distribution: HashMap<String, usize>,
    results: Vec<VerificationResult>,
    any_failures: bool,
}

fn verify_stage4_benchmarks() -> VerificationReport {
    use mog_synth::benchmark::{get_benchmark, Example, Value};
    use mog_synth::solver::solve_problem;

    // Representative Stage 4 time-parameterized benchmarks
    let test_cases = vec![
        ("fibonacci", 0),
        ("factorial", 0),
        ("triangular_number", 0),
        ("linear_series", 0),
        ("polynomial_eval", 0),
    ];

    let mut results = Vec::new();
    let mut method_counts: HashMap<String, usize> = HashMap::new();

    println!("============================================================");
    println!("Stage 4 Time-Parameterized Synthesis Verification");
    println!("============================================================");

    for (problem_name, _variant) in &test_cases {
        println!("\nSynthesizing {}...", problem_name);

        // Get all problems from the benchmark factory
        let all_problems = get_benchmark(1);
        let problem = all_problems
            .iter()
            .find(|p| &p.name == problem_name);

        let result = match problem {
            Some(prob) => {
                let start = Instant::now();
                match solve_problem(prob, true) {
                    Some((method, code)) => {
                        let elapsed = start.elapsed().as_secs_f64();
                        let success = !code.is_empty();

                        method_counts
                            .entry(if success { method.clone() } else { "empty_code".to_string() })
                            .or_insert(0)
                            += 1;

                        println!(
                            "  ✓ Success | Time: {:.2}s | Method: {}",
                            elapsed, method
                        );

                        VerificationResult {
                            problem_name: problem_name.to_string(),
                            variant: 0,
                            success,
                            solve_time_secs: elapsed,
                            method,
                            code_length: code.len(),
                            error: None,
                        }
                    }
                    None => {
                        let elapsed = start.elapsed().as_secs_f64();
                        method_counts.entry("no_method_found".to_string()).or_insert(0) += 1;

                        println!("  ✗ Failed | Time: {:.2}s | No method found", elapsed);

                        VerificationResult {
                            problem_name: problem_name.to_string(),
                            variant: 0,
                            success: false,
                            solve_time_secs: elapsed,
                            method: "none".to_string(),
                            code_length: 0,
                            error: Some("Solver returned None".to_string()),
                        }
                    }
                }
            }
            None => {
                method_counts.entry("not_found".to_string()).or_insert(0) += 1;
                println!("  ✗ Not Found in benchmark set");

                VerificationResult {
                    problem_name: problem_name.to_string(),
                    variant: 0,
                    success: false,
                    solve_time_secs: 0.0,
                    method: "not_found".to_string(),
                    code_length: 0,
                    error: Some(format!("Problem '{}' not found in benchmark set", problem_name)),
                }
            }
        };

        results.push(result);
    }

    // Summary
    let passed = results.iter().filter(|r| r.success).count();
    let total = results.len();
    let mean_time = if total > 0 {
        results.iter().map(|r| r.solve_time_secs).sum::<f64>() / total as f64
    } else {
        0.0
    };

    println!("\n============================================================");
    println!("SUMMARY");
    println!("============================================================");
    println!("\nPass Rate: {}/{} ({:3}%)", passed, total, if total > 0 { 100 * passed / total } else { 0 });
    println!("Mean Solve Time: {:.2}s", mean_time);

    println!("\nMethod Distribution:");
    let mut sorted_methods: Vec<_> = method_counts.iter().collect();
    sorted_methods.sort_by(|a, b| b.1.cmp(a.1));
    for (method, count) in sorted_methods {
        println!("  {:20} : {:3} problems", method, count);
    }

    println!("\nDetailed Results:");
    for result in &results {
        let status = if result.success { "✓" } else { "✗" };
        println!(
            "  {} {:20} | {:7.2}s | {:20} | {:6} bytes",
            status, result.problem_name, result.solve_time_secs, result.method, result.code_length
        );
    }

    let failures: Vec<_> = results.iter().filter(|r| !r.success).collect();
    if !failures.is_empty() {
        println!("\nFailures ({}):", failures.len());
        for f in failures {
            println!(
                "  - {}: {}",
                f.problem_name,
                f.error.as_deref().unwrap_or("unknown error")
            );
        }
    }

    let any_failures = !failures.is_empty();

    VerificationReport {
        pass_count: passed,
        total,
        mean_solve_time_secs: mean_time,
        method_distribution: method_counts,
        results,
        any_failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage4_synthesis_verification() {
        let report = verify_stage4_benchmarks();

        println!("\n{}", serde_json::to_string_pretty(&report).unwrap());

        // Verify at least some problems solved
        assert!(
            report.pass_count > 0,
            "Expected at least 1 successful synthesis, got {}",
            report.pass_count
        );

        // Verify all results are accounted for
        assert_eq!(
            report.results.len(),
            report.total,
            "Result count mismatch"
        );

        // Verify method distribution sums to total (successful solves)
        let total_methods: usize = report.method_distribution.values().sum();
        assert_eq!(
            total_methods,
            report.total,
            "Method count distribution doesn't match total"
        );

        println!("✓ Verification passed: {}/{} problems solved", report.pass_count, report.total);
    }
}

fn main() {
    let report = verify_stage4_benchmarks();

    // Output JSON for automated parsing
    match serde_json::to_string_pretty(&report) {
        Ok(json) => {
            println!("\nJSON Output:");
            println!("{}", json);

            // Write to file
            use std::fs;
            let path = "verify_stage4_results.json";
            match fs::write(path, json) {
                Ok(_) => println!("\nResults written to {}", path),
                Err(e) => eprintln!("Failed to write results: {}", e),
            }
        }
        Err(e) => eprintln!("Failed to serialize results: {}", e),
    }

    // Exit code based on success
    std::process::exit(if report.any_failures { 1 } else { 0 });
}
