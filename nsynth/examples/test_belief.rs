use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

fn main() {
    let bridge = LinguigenesisBridge::new();

    println!("=== Belief for 'add two numbers' ===");
    let belief = bridge.get_belief_state("add two numbers").unwrap();
    println!("Entities: {:?}", belief.comprehension.entities);
    println!("Intent: {:?}", belief.intent.intent_type);

    println!("\n=== Belief for 'reverse an array' ===");
    let belief2 = bridge.get_belief_state("reverse an array").unwrap();
    println!("Entities: {:?}", belief2.comprehension.entities);
    println!("Intent: {:?}", belief2.intent.intent_type);
}
