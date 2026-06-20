use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

fn main() {
    let bridge = LinguigenesisBridge::new();

    println!("=== Testing 'reverse an array' ===");
    let examples = bridge.nl_to_examples("reverse an array").unwrap();
    for (i, ex) in examples.iter().enumerate() {
        println!(
            "  Example {}: inputs={:?}, expected={:?}",
            i, ex.inputs, ex.expected
        );
    }

    println!("\n=== Testing 'add two numbers' ===");
    let examples2 = bridge.nl_to_examples("add two numbers").unwrap();
    for (i, ex) in examples2.iter().enumerate() {
        println!(
            "  Example {}: inputs={:?}, expected={:?}",
            i, ex.inputs, ex.expected
        );
    }
}
