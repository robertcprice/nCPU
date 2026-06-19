//! Multi-turn Dialogue System Demo for nCPU/nSynth
//!
//! This example demonstrates the interactive clarification system
//! for ambiguous requirements using the DialogueManager.

use mog_synth::nl::{
    dialogue::{Ambiguity, Answer, DialogueManager, Question, QuestionType},
    OutputSpec, ParsedRequirements,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-turn Dialogue System Demo ===\n");

    // Create a dialogue manager
    let mut dialogue_mgr = DialogueManager::new();
    println!("✓ Dialogue Manager initialized (max 5 questions per round)\n");

    // Simulate ambiguous requirements
    let ambiguous_req = ParsedRequirements {
        function_name: "process_data".to_string(),
        inputs: vec![],
        output: OutputSpec {
            type_: "unknown".to_string(),
            description: None,
        },
        description: "Process some data and return result".to_string(),
        examples: vec![],
        constraints: vec![],
    };

    println!("Ambiguous Request: \"{}\"\n", ambiguous_req.description);

    // Step 1: Detect ambiguities
    println!("Step 1: Detecting ambiguities...");
    let ambiguities = dialogue_mgr.detect_ambiguity(&ambiguous_req);
    println!("Found {} ambiguities:\n", ambiguities.len());

    for (i, ambiguity) in ambiguities.iter().enumerate() {
        println!(
            "  {}. {} ({:?})",
            i + 1,
            ambiguity.description,
            ambiguity.qtype
        );
        for question in &ambiguity.questions {
            println!("     - {}", question);
        }
        println!();
    }

    // Step 2: Generate follow-up questions
    println!("Step 2: Generating follow-up questions...");
    let questions = dialogue_mgr.generate_followup_questions(&ambiguities);
    println!("Generated {} questions:\n", questions.len());

    for (i, question) in questions.iter().enumerate() {
        println!(
            "  Q{}. {} (Required: {})",
            i + 1,
            question.text,
            question.required
        );
        if !question.options.is_empty() {
            println!("     Options:");
            for (j, option) in question.options.iter().enumerate() {
                println!("       {}. {}", j + 1, option);
            }
        }
        println!();
    }

    // Step 3: Simulate user answering questions
    println!("Step 3: Processing user answers...\n");

    let user_answers = vec![
        ("Return error code -1 for empty input", "empty_input"),
        ("Return integer type", "data_type"),
    ];

    for (i, (answer_text, question_context)) in user_answers.iter().enumerate() {
        println!("  User Answer {} to Q{}: \"{}\"", i + 1, i + 1, answer_text);

        let answer = Answer {
            question_id: question_context.to_string(),
            text: answer_text.to_string(),
            provided: true,
        };

        dialogue_mgr.process_answer(question_context.to_string(), answer);
        println!();
    }

    // Step 4: Refine requirements
    println!("Step 4: Refining requirements based on answers...");
    let refined_req =
        dialogue_mgr.refine_requirements(&ambiguous_req, &dialogue_mgr.state().answers)?;

    println!("✓ Requirements refined:");
    println!("  Function: {}", refined_req.function_name);
    println!("  Output type: {}", refined_req.output.type_);
    if !refined_req.constraints.is_empty() {
        println!("  Constraints:");
        for constraint in &refined_req.constraints {
            println!("    - {}", constraint);
        }
    }
    println!();

    // Step 5: Confirm specification
    println!("Step 5: Confirming specification...");
    let is_complete = dialogue_mgr.confirm_specification(&refined_req)?;
    println!("Specification complete: {}", is_complete);

    if is_complete {
        println!("✓ Ready for synthesis!");
    } else {
        println!("⚠ Additional clarification needed");
    }

    // Step 6: Multi-turn scenario
    println!("\n=== Multi-turn Scenario ===\n");
    println!("Simulating additional clarification round...\n");

    let _continue = dialogue_mgr.next_round()?;
    println!("  Round: {}", dialogue_mgr.state().round);
    println!("  Answers received: {}", dialogue_mgr.state().answers.len());
    println!("  Is complete: {}", dialogue_mgr.is_complete());

    println!("\n=== Demo Complete ===");
    println!("\nDialogue System Features:");
    println!("  ✓ Ambiguity detection in requirements");
    println!("  ✓ Automatic follow-up question generation");
    println!("  ✓ Requirement refinement based on user input");
    println!("  ✓ Specification validation");
    println!("  ✓ Multi-turn dialogue support");

    Ok(())
}
