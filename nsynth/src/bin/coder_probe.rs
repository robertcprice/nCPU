//! Validate propose_rust_fn against the live coder model on real repair-shaped tasks.
fn main() {
    let tasks = [
        "a function named twice that doubles its integer argument",
        "a function has_dup that returns whether a Vec<i64> contains any duplicate value",
        "a function gcd that returns the greatest common divisor of two i64 values",
    ];
    for t in tasks {
        match mog_synth::local_llm::propose_rust_fn(t, None, 0.0) {
            Some(c) => println!("=== {t}\n{c}\n"),
            None => println!("=== {t}\nNONE\n"),
        }
    }
}
