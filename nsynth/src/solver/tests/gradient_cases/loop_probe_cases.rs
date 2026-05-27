use super::*;

#[test]
fn digital_root_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "digital_root_v0")
        .expect("digital_root_v0 not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!(
        "digital_root_v0: {}",
        if solved { "SOLVED" } else { "failed" }
    );
    assert!(
        solved,
        "digital_root_v0 should be solved by biased gradient restart"
    );
}

/// Targeted smoke-test for the count_divisors biased SoftCondAccumLoop restart.
#[test]
fn count_divisors_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "count_divisors_v0")
        .expect("count_divisors_v0 not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!(
        "count_divisors_v0: {}",
        if solved { "SOLVED" } else { "failed" }
    );
    assert!(
        solved,
        "count_divisors_v0 should be solved by SoftCondAccumLoop biased restart"
    );
}

/// Targeted smoke-test for the sum_of_divisors biased SoftCondAccumLoop restart.
#[test]
fn sum_of_divisors_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "sum_of_divisors_v0")
        .expect("sum_of_divisors_v0 not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!(
        "sum_of_divisors_v0: {}",
        if solved { "SOLVED" } else { "failed" }
    );
    assert!(
        solved,
        "sum_of_divisors_v0 should be solved by SoftCondAccumLoop biased restart"
    );
}

/// count_even_digits has f(0)=1 edge case incompatible with our loop-based approach
/// (loop exits immediately for n=0, returning init=0, but expected=1).
/// This test is informational only (no assert) — tracks if it ever becomes solvable.
#[test]
fn count_even_digits_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "count_even_digits_v0")
        .expect("count_even_digits_v0 not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!(
        "count_even_digits_v0: {}",
        if solved {
            "SOLVED"
        } else {
            "failed (expected — f(0)=1 edge case)"
        }
    );
    // Not asserting: f(0)=1 requires special handling outside our loop program type
}

/// Targeted smoke-test for the sum_odd_digits biased SoftCondDigitLoop restart.
#[test]
fn sum_odd_digits_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "sum_odd_digits_v0")
        .expect("sum_odd_digits_v0 not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!(
        "sum_odd_digits_v0: {}",
        if solved { "SOLVED" } else { "failed" }
    );
    assert!(
        solved,
        "sum_odd_digits_v0 should be solved by SoftCondDigitLoop biased restart"
    );
}

/// Targeted smoke-test for the popcount biased SoftCondDigitLoop restart.
#[test]
fn popcount_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "popcount_v0")
        .expect("popcount_v0 not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!("popcount_v0: {}", if solved { "SOLVED" } else { "failed" });
    assert!(
        solved,
        "popcount_v0 should be solved by SoftCondDigitLoop biased restart"
    );
}

/// Targeted smoke-test for the max_digit biased SoftCondDigitLoop restart.
#[test]
fn max_digit_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "max_digit_v0")
        .expect("max_digit_v0 not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!("max_digit_v0: {}", if solved { "SOLVED" } else { "failed" });
    assert!(
        solved,
        "max_digit_v0 should be solved by SoftCondDigitLoop biased restart"
    );
}

/// count_even_digits with zero_return=1 for n=0 edge case.
#[test]
fn count_even_digits_gradient_only_v2_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "count_even_digits_v0")
        .expect("not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!(
        "count_even_digits_v0: {}",
        if solved { "SOLVED" } else { "failed" }
    );
    assert!(
        solved,
        "count_even_digits_v0 should be solved by SoftCondDigitLoop with zero_return=1"
    );
}

/// is_perfect_square: loop i in [0..n], count where i*i==n, returns 1 for perfect squares.
#[test]
fn is_perfect_square_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "is_perfect_square_v0")
        .expect("not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!(
        "is_perfect_square_v0: {}",
        if solved { "SOLVED" } else { "failed" }
    );
    assert!(
        solved,
        "is_perfect_square_v0 should be solved by SoftCondAccumLoop biased restart"
    );
}

/// min3: two-stage chained ternary — v0=min(a,b), return min(v0,c).
#[test]
fn min3_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "min3_v0")
        .expect("not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!("min3_v0: {}", if solved { "SOLVED" } else { "failed" });
    assert!(
        solved,
        "min3_v0 should be solved by SoftChainedBranch biased restart"
    );
}

/// is_prime: SoftCondAccumCmpReturnLoop — count divisors then return acc==2
#[test]
#[ignore]
fn is_prime_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "is_prime_v0")
        .expect("not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!("is_prime_v0: {}", if solved { "SOLVED" } else { "failed" });
    assert!(
        solved,
        "is_prime_v0 should be solved by SoftCondAccumCmpReturnLoop biased restart"
    );
}

/// triangular_check: SoftPredicateLoopRetCmp — two-acc loop x0=k,x1=tri; if x1==n return 1
#[test]
#[ignore]
fn triangular_check_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "triangular_check_v0")
        .expect("not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!(
        "triangular_check_v0: {}",
        if solved { "SOLVED" } else { "failed" }
    );
    assert!(
        solved,
        "triangular_check_v0 should be solved by SoftPredicateLoopRetCmp biased restart"
    );
}

/// collatz_steps: SoftCondMutateLoop — while x!=1 { if x%2==0 { x=x/2 } else { x=x*3+1 }; acc++ }
#[test]
#[ignore]
fn collatz_steps_gradient_only_test() {
    let problems = get_benchmark(1);
    let p = problems
        .iter()
        .find(|p| p.name == "collatz_steps_v0")
        .expect("not found");
    let r = crate::synthesis::synthesize_gradient_only(p);
    let solved = r.map(|r| r.success).unwrap_or(false);
    println!(
        "collatz_steps_v0: {}",
        if solved { "SOLVED" } else { "failed" }
    );
    assert!(
        solved,
        "collatz_steps_v0 should be solved by SoftCondMutateLoop biased restart"
    );
}
