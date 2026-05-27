use super::*;

#[test]
fn gradient_only_coverage_report() {
    let problems = get_benchmark(1);
    let mut solved = 0;
    let mut total = 0;
    let mut solved_names = vec![];
    let mut failed_names = vec![];
    for p in &problems {
        if !p.examples.iter().all(|ex| {
            ex.inputs
                .iter()
                .all(|v| matches!(v, crate::benchmark::Value::Int(_)))
        }) {
            continue;
        }
        total += 1;
        let ok = crate::synthesis::synthesize_gradient_only(p)
            .map(|r| r.success)
            .unwrap_or(false);
        if ok {
            solved += 1;
            solved_names.push(p.name.clone());
        } else {
            failed_names.push(p.name.clone());
        }
        println!(
            "  [{}/47] {} {}",
            total,
            p.name,
            if ok { "SOLVED" } else { "failed" }
        );
    }
    println!(
        "\n=== Pure Gradient Coverage (scalar only): {}/{} ===",
        solved, total
    );
    println!("SOLVED: {}", solved_names.join(", "));
    println!("FAILED: {}", failed_names.join(", "));
}

/// Quick smoke-test: biased restarts solve GCD, leading_digit, next_power_of_2,
/// safe_div_or_neg1, digit_count, digit_product, digital_root, polynomial, harmonic_sum,
/// count_divisors, sum_of_divisors, sum_odd_digits, popcount, max_digit
#[test]
fn predicate_loop_quick_test() {
    let problems = get_benchmark(1);
    let targets = [
        "gcd_v0",
        "gcd_extended_v0",
        "leading_digit_v0",
        "next_power_of_2_v0",
        "safe_div_or_neg1_v0",
        "digit_count_v0",
        "digit_product_v0",
        "digital_root_v0",
        "polynomial_v0",
        "harmonic_sum_v0",
        "count_divisors_v0",
        "sum_of_divisors_v0",
        "sum_odd_digits_v0",
        "popcount_v0",
        "max_digit_v0",
    ];
    for name in &targets {
        let Some(p) = problems.iter().find(|p| p.name == *name) else {
            println!("{}: NOT FOUND", name);
            continue;
        };
        let r = crate::synthesis::synthesize_gradient_only(p);
        let solved = r.map(|r| r.success).unwrap_or(false);
        println!("{}: {}", name, if solved { "SOLVED" } else { "failed" });
    }
}
