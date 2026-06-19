use mog_synth::benchmark::get_benchmark;
use mog_synth::solver::solve_problem;
use std::collections::HashMap;
/// Stage 4 Time-Parameterized Synthesis Verification
///
/// End-to-end verification for representative benchmarks (fibonacci, factorial, triangular, linear, poly).
/// Reports: (pass_count / total, mean_solve_time, method_distribution, any_failures).
use std::time::Instant;

#[derive(Clone, Debug)]
struct VerificationResult {
    problem_name: String,
    success: bool,
    solve_time_secs: f64,
    method: String,
    code_length: usize,
    error: Option<String>,
}

fn verify_stage4_benchmarks() {
    // Representative Stage 4 time-parameterized benchmarks
    let test_cases = vec![
        "fibonacci_v0",
        "factorial_v0",
        "triangular_check_v0",
        "polynomial_v0",
        "collatz_steps_v0",
    ];

    let mut results = Vec::new();
    let mut method_counts: HashMap<String, usize> = HashMap::new();

    println!("============================================================");
    println!("Stage 4 Time-Parameterized Synthesis Verification");
    println!("============================================================\n");

    // Get all problems from the benchmark factory
    let all_problems = get_benchmark(1);

    // Debug: Find and print matching problems
    eprintln!("Looking for target problems...");
    for target in &[
        "fibonacci",
        "factorial",
        "triangular",
        "polynomial",
        "collatz",
    ] {
        let matches: Vec<&str> = all_problems
            .iter()
            .filter(|p| p.name.contains(target))
            .map(|p| p.name.as_str())
            .collect();
        if !matches.is_empty() {
            eprintln!("  Matching '{}': {:?}", target, matches);
        }
    }
    eprintln!("Total problems: {}\n", all_problems.len());

    for problem_name in &test_cases {
        println!("Synthesizing {}...", problem_name);

        let problem = all_problems.iter().find(|p| &p.name == *problem_name);

        let result = match problem {
            Some(prob) => {
                let start = Instant::now();
                let solve_result = solve_problem(prob);
                let elapsed = start.elapsed().as_secs_f64();

                let success = solve_result.success;
                let method = solve_result.method.clone();
                let code = solve_result.code.clone();

                {
                    let key = if success {
                        method.clone()
                    } else {
                        "failed".to_string()
                    };
                    *method_counts.entry(key).or_insert(0) += 1;
                }

                let status = if success {
                    "✓ Success"
                } else {
                    "✗ Failed "
                };
                println!(
                    "  {} | Time: {:.2}s | Method: {} | Code: {} bytes",
                    status,
                    elapsed,
                    method,
                    code.len()
                );

                if let Some(ref err) = solve_result.error {
                    println!("    Error: {}", err);
                }

                VerificationResult {
                    problem_name: problem_name.to_string(),
                    success,
                    solve_time_secs: elapsed,
                    method,
                    code_length: code.len(),
                    error: solve_result.error,
                }
            }
            None => {
                method_counts
                    .entry("not_found".to_string())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                println!("  ✗ Not Found in benchmark set");

                VerificationResult {
                    problem_name: problem_name.to_string(),
                    success: false,
                    solve_time_secs: 0.0,
                    method: "not_found".to_string(),
                    code_length: 0,
                    error: Some(format!(
                        "Problem '{}' not found in benchmark set",
                        problem_name
                    )),
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
    println!("\nResults:");
    println!(
        "  Pass Rate:       {}/{} ({:3}%)",
        passed,
        total,
        if total > 0 { 100 * passed / total } else { 0 }
    );
    println!("  Mean Solve Time: {:.2}s", mean_time);

    println!("\nMethod Distribution:");
    let mut sorted_methods: Vec<_> = method_counts.iter().collect();
    sorted_methods.sort_by(|a, b| b.1.cmp(a.1));
    for (method, count) in sorted_methods {
        println!("  {:20}: {:3} problems", method, count);
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
    let has_failures = !failures.is_empty();
    if has_failures {
        println!("\nFailures ({}):", failures.len());
        for f in &failures {
            println!(
                "  - {}: {}",
                f.problem_name,
                f.error.as_deref().unwrap_or("unknown error")
            );
        }
    }

    println!("\n============================================================");
    println!("FINAL RESULT");
    println!("============================================================");
    println!("pass_count / total:        {}/{}", passed, total);
    println!("mean_solve_time:           {:.2}s", mean_time);
    println!("any_failures:              {}", has_failures);
    println!("method_distribution:");
    for (method, count) in method_counts.iter() {
        println!("  {:20}: {}", method, count);
    }

    std::process::exit(if has_failures { 1 } else { 0 });
}

fn main() {
    verify_stage4_benchmarks();
}
