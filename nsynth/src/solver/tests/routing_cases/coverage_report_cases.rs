use super::*;

/// Show which benchmark problems the gradient+template solver can discover
/// without any hardcoded search fallback.
#[test]
fn gradient_synth_coverage_report() {
    let problems = get_benchmark(1);
    let mut solved = 0;
    let mut total = 0;
    let mut solved_names = vec![];
    let mut failed_names = vec![];
    for p in &problems {
        total += 1;
        if let Some(r) = crate::synthesis::synthesize_scalar(p) {
            if r.success {
                solved += 1;
                solved_names.push(p.name.clone());
            } else {
                failed_names.push(p.name.clone());
            }
        } else {
            failed_names.push(p.name.clone());
        }
    }
    println!(
        "\n=== Gradient Synthesis Coverage: {}/{} ===",
        solved, total
    );
    println!("SOLVED: {}", solved_names.join(", "));
    println!("FAILED: {}", failed_names.join(", "));
}

/// Show full pipeline coverage (gradient + template + search).
#[test]
fn full_pipeline_coverage_report() {
    let problems = get_benchmark(1);
    let mut solved = 0;
    let mut total = 0;
    let mut by_method: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut failed_names = vec![];
    let total_problems = problems.len();
    for (idx, p) in problems.iter().enumerate() {
        total += 1;
        println!("[pipeline] {}/{} {}", idx + 1, total_problems, p.name);
        let r = solve_problem(p);
        if r.success {
            solved += 1;
            *by_method.entry(r.method.clone()).or_insert(0) += 1;
            println!("[pipeline] {} -> {}", p.name, r.method);
        } else {
            failed_names.push(p.name.clone());
            println!("[pipeline] {} -> FAILED", p.name);
        }
    }
    let mut method_summary: Vec<_> = by_method.iter().collect();
    method_summary.sort_by_key(|(k, _)| k.as_str());
    println!("\n=== Full Pipeline Coverage: {}/{} ===", solved, total);
    for (method, count) in &method_summary {
        println!("  {}: {}", method, count);
    }
    if !failed_names.is_empty() {
        println!("FAILED: {}", failed_names.join(", "));
    }
}
