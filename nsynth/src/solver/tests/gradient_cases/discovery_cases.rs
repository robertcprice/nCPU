use super::*;

#[test]
fn gradient_synth_discovers_add_two() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "add_two_v0")
        .unwrap();
    let result = crate::synthesis::synthesize_scalar(&problem);
    assert!(result.is_some(), "synthesis returned None");
    let result = result.unwrap();
    println!("[add_two] code:\n{}", result.code);
    assert!(result.success, "synthesis failed: {:?}", result.error);
    assert!(
        result.method == "synth_gradient" || result.method == "template",
        "unexpected method: {}",
        result.method
    );
}

#[test]
fn gradient_synth_discovers_max2() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "max2_v0")
        .unwrap();
    let result = crate::synthesis::synthesize_scalar(&problem);
    assert!(result.is_some(), "synthesis returned None");
    let result = result.unwrap();
    println!("[max2] code:\n{}", result.code);
    assert!(result.success, "synthesis failed: {:?}", result.error);
    assert!(
        result.method == "synth_gradient" || result.method == "template",
        "unexpected method: {}",
        result.method
    );
}

#[test]
fn gradient_synth_discovers_sum_to_n() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "sum_to_n_v0")
        .unwrap();
    let result = crate::synthesis::synthesize_scalar(&problem);
    assert!(result.is_some(), "synthesis returned None");
    let result = result.unwrap();
    println!("[sum_to_n] code:\n{}", result.code);
    assert!(result.success, "synthesis failed: {:?}", result.error);
    assert!(
        result.method == "synth_gradient" || result.method == "template",
        "unexpected method: {}",
        result.method
    );
}

#[test]
fn gradient_synth_discovers_abs_diff() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "abs_diff_v0")
        .unwrap();
    let result = crate::synthesis::synthesize_scalar(&problem);
    assert!(result.is_some(), "synthesis returned None");
    let result = result.unwrap();
    println!("[abs_diff] code:\n{}", result.code);
    assert!(result.success, "synthesis failed: {:?}", result.error);
    assert!(
        result.method == "synth_gradient" || result.method == "template",
        "unexpected method: {}",
        result.method
    );
}

#[test]
fn gradient_synth_discovers_is_even() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "is_even_v0")
        .unwrap();
    let result = crate::synthesis::synthesize_scalar(&problem);
    assert!(result.is_some(), "synthesis returned None");
    let result = result.unwrap();
    println!("[is_even] code:\n{}", result.code);
    assert!(result.success, "synthesis failed: {:?}", result.error);
    assert!(
        result.method == "synth_gradient" || result.method == "template",
        "unexpected method: {}",
        result.method
    );
}

fn gradient_synth_discovers_one(problem_name: &str) {
    let p = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == format!("{}_v0", problem_name))
        .unwrap_or_else(|| panic!("problem {} not found", problem_name));
    let r = crate::synthesis::synthesize_scalar(&p);
    assert!(r.is_some(), "{}: synthesis returned None", problem_name);
    let r = r.unwrap();
    println!("[{}] code:\n{}", problem_name, r.code);
    assert!(
        r.success,
        "{}: not verified. code:\n{}",
        problem_name, r.code
    );
    assert!(
        r.method == "synth_gradient" || r.method == "template",
        "unexpected method: {}",
        r.method
    );
}

#[test]
fn gradient_synth_discovers_factorial() {
    gradient_synth_discovers_one("factorial");
}
#[test]
fn gradient_synth_discovers_cube() {
    gradient_synth_discovers_one("cube");
}
#[test]
fn gradient_synth_discovers_square_plus_n() {
    gradient_synth_discovers_one("square_plus_n");
}
#[test]
fn gradient_synth_discovers_product_1_to_n() {
    gradient_synth_discovers_one("product_1_to_n");
}
#[test]
fn gradient_synth_discovers_bilinear3() {
    gradient_synth_discovers_one("bilinear3");
}
#[test]
fn gradient_synth_discovers_sign() {
    gradient_synth_discovers_one("sign");
}
#[test]
fn gradient_synth_discovers_clamp() {
    gradient_synth_discovers_one("clamp_0_100");
}
#[test]
fn gradient_synth_discovers_power() {
    gradient_synth_discovers_one("power");
}
#[test]
fn gradient_synth_discovers_fibonacci() {
    gradient_synth_discovers_one("fibonacci");
}
#[test]
fn gradient_synth_discovers_fib_iter() {
    gradient_synth_discovers_one("fib_iter");
}
#[test]
fn gradient_synth_discovers_lucas() {
    gradient_synth_discovers_one("lucas_number");
}
#[test]
fn gradient_synth_discovers_sum_squares() {
    gradient_synth_discovers_one("sum_squares");
}
#[test]
fn gradient_synth_discovers_celsius() {
    gradient_synth_discovers_one("celsius_to_fahrenheit");
}
#[test]
fn gradient_synth_discovers_product_offset() {
    gradient_synth_discovers_one("product_offset");
}
