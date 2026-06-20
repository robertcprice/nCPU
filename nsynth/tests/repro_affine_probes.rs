use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::solver::solve_problem;
use std::time::Instant;

fn ex1(x: i64, y: i64) -> Example {
    Example {
        inputs: vec![Value::Int(x)],
        expected: Value::Int(y),
    }
}
fn ex2(a: i64, b: i64, y: i64) -> Example {
    Example {
        inputs: vec![Value::Int(a), Value::Int(b)],
        expected: Value::Int(y),
    }
}

fn prob(sig: &'static str, examples: Vec<Example>) -> Problem {
    Problem {
        name: "probe".to_string(),
        category: "serve",
        description: "",
        signature: sig,
        examples,
        holdouts: vec![],
        reference_code: "",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    }
}

fn run1(name: &str, sig: &'static str, f: impl Fn(i64) -> i64) {
    let xs = [0i64, 1, 2, 3, 4, 5, 6, 7];
    let examples: Vec<Example> = xs.iter().map(|&x| ex1(x, f(x))).collect();
    let p = prob(sig, examples);
    let t = Instant::now();
    let r = solve_problem(&p);
    let ms = t.elapsed().as_millis();
    println!(
        "[{name}] success={} ms={} method={}",
        r.success, ms, r.method
    );
}

fn run2(name: &str, sig: &'static str, f: impl Fn(i64, i64) -> i64) {
    let pairs = [
        (0i64, 0i64),
        (1, 2),
        (2, 1),
        (3, 4),
        (4, 3),
        (5, 6),
        (6, 5),
        (7, 2),
        (2, 7),
        (3, 9),
    ];
    let examples: Vec<Example> = pairs.iter().map(|&(a, b)| ex2(a, b, f(a, b))).collect();
    let p = prob(sig, examples);
    let t = Instant::now();
    let r = solve_problem(&p);
    let ms = t.elapsed().as_millis();
    println!(
        "[{name}] success={} ms={} method={}",
        r.success, ms, r.method
    );
}

#[test]
fn repro_failing_probes() {
    run1("2x*x", "fn f(x:i64)->i64", |x| 2 * x * x);
    run1("x*x+5", "fn f(x:i64)->i64", |x| x * x + 5);
    run1("x*x+2x+1", "fn f(x:i64)->i64", |x| x * x + 2 * x + 1);
    run1("x*x-x", "fn f(x:i64)->i64", |x| x * x - x);
    run1("x*x-1", "fn f(x:i64)->i64", |x| x * x - 1);
    run1("3x-7", "fn f(x:i64)->i64", |x| 3 * x - 7);
    run1("xxx-x", "fn f(x:i64)->i64", |x| x * x * x - x);
    run1("2xxx+1", "fn f(x:i64)->i64", |x| 2 * x * x * x + 1);
    run2("a+b", "fn f(a:i64,b:i64)->i64", |a, b| a + b);
    run2("a-b", "fn f(a:i64,b:i64)->i64", |a, b| a - b);
    run2("a*a-b*b", "fn f(a:i64,b:i64)->i64", |a, b| a * a - b * b);
    run2("a*a+b*b", "fn f(a:i64,b:i64)->i64", |a, b| a * a + b * b);
    run2("a+2b", "fn f(a:i64,b:i64)->i64", |a, b| a + 2 * b);
}
