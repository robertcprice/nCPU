//! Natural language generator for code → NL
//!
//! Converts semantic analysis into human-readable descriptions.

use crate::bidirectional::analyzer::CodeSemantics;

/// Generate natural language description from semantics
pub fn generate_nl(semantics: &CodeSemantics) -> String {
    let mut description = String::new();

    // Start with function purpose
    description.push_str(&generate_function_purpose(semantics));

    // Add operations description
    if !semantics.operations.is_empty() {
        description.push_str(&generate_operations_description(semantics));
    }

    // Add algorithms description
    if !semantics.algorithms.is_empty() {
        description.push_str(&generate_algorithms_description(semantics));
    }

    // Add data structures description
    if !semantics.data_structures.is_empty() {
        description.push_str(&generate_data_structures_description(semantics));
    }

    // Add complexity information
    description.push_str(&generate_complexity_description(semantics));

    description
}

/// Generate function purpose statement
fn generate_function_purpose(semantics: &CodeSemantics) -> String {
    let mut purpose = String::new();

    // Determine main operation
    let main_op = semantics.operations.first().map(|s| s.as_str()).unwrap_or("process");

    match main_op {
        "add" | "addition" => purpose.push_str("Function that adds two values together"),
        "subtract" | "subtraction" => purpose.push_str("Function that subtracts one value from another"),
        "multiply" | "multiplication" => purpose.push_str("Function that multiplies two values"),
        "divide" | "division" => purpose.push_str("Function that divides one value by another"),
        "map" => purpose.push_str("Function that transforms each element in a collection"),
        "filter" => purpose.push_str("Function that selects elements meeting a condition"),
        "reduce" | "fold" => purpose.push_str("Function that combines all elements into a single value"),
        "reverse" => purpose.push_str("Function that reverses the order of elements"),
        "sort" => purpose.push_str("Function that sorts elements in ascending order"),
        "search" | "binary_search" => purpose.push_str("Function that finds an element in a collection"),
        "split" => purpose.push_str("Function that divides a string into parts"),
        "join" => purpose.push_str("Function that concatenates elements with a separator"),
        _ => {
            if !semantics.algorithms.is_empty() {
                purpose.push_str(&format!("Function that implements {}", semantics.algorithms[0]));
            } else {
                purpose.push_str("Function that processes input data");
            }
        }
    }

    // Add input/output info
    purpose.push_str(&format!(", taking {} input(s)", semantics.io_types.inputs.len()));

    if !semantics.io_types.inputs.is_empty() {
        purpose.push_str(&format!(" of type{}", if semantics.io_types.inputs.len() > 1 { "s" } else { "" }));
        for (i, input_type) in semantics.io_types.inputs.iter().enumerate() {
            if i == 0 {
                purpose.push_str(&format!(" {}", input_type));
            } else if i == semantics.io_types.inputs.len() - 1 {
                purpose.push_str(&format!(" and {}", input_type));
            } else {
                purpose.push_str(&format!(", {}", input_type));
            }
        }
    }

    purpose.push_str(&format!(" and returning {}", semantics.io_types.output));
    purpose.push_str(".\n");
    purpose
}

/// Generate operations description
fn generate_operations_description(semantics: &CodeSemantics) -> String {
    let mut desc = String::new();

    if semantics.operations.len() == 1 {
        desc.push_str(&format!("Performs {} operation", semantics.operations[0]));
    } else if semantics.operations.len() == 2 {
        desc.push_str(&format!("Performs {} and {} operations", semantics.operations[0], semantics.operations[1]));
    } else {
        desc.push_str(&format!("Performs multiple operations: "));
        for (i, op) in semantics.operations.iter().enumerate() {
            if i == semantics.operations.len() - 1 {
                desc.push_str(&format!("and {}", op));
            } else {
                desc.push_str(&format!("{}, ", op));
            }
        }
    }

    desc.push_str(".\n");
    desc
}

/// Generate algorithms description
fn generate_algorithms_description(semantics: &CodeSemantics) -> String {
    let mut desc = String::new();

    for algo in &semantics.algorithms {
        match algo.as_str() {
            "binary_search" => desc.push_str("Uses binary search algorithm for O(log n) lookup in sorted data. "),
            "merge_sort" => desc.push_str("Implements merge sort for stable O(n log n) sorting. "),
            "quick_sort" => desc.push_str("Implements quick sort for efficient O(n log n) average-case sorting. "),
            "heap_sort" => desc.push_str("Implements heap sort for O(n log n) in-place sorting. "),
            "dijkstra" => desc.push_str("Implements Dijkstra's algorithm for shortest path finding. "),
            "bfs" => desc.push_str("Uses breadth-first search for level-order traversal. "),
            "dfs" => desc.push_str("Uses depth-first search for graph traversal. "),
            "fibonacci" => desc.push_str("Generates Fibonacci sequence using recursion or iteration. "),
            "factorial" => desc.push_str("Computes factorial of a number. "),
            "gcd" => desc.push_str("Computes greatest common divisor using Euclidean algorithm. "),
            _ => desc.push_str(&format!("Implements {} algorithm. ", algo)),
        }
    }

    if !desc.is_empty() {
        desc.push('\n');
    }

    desc
}

/// Generate data structures description
fn generate_data_structures_description(semantics: &CodeSemantics) -> String {
    let mut desc = String::new();

    if semantics.data_structures.len() == 1 {
        desc.push_str(&format!("Works with {} data structure", semantics.data_structures[0]));
    } else {
        desc.push_str(&format!("Works with multiple data structures: "));
        for (i, ds) in semantics.data_structures.iter().enumerate() {
            if i == semantics.data_structures.len() - 1 {
                desc.push_str(&format!("and {}", ds));
            } else {
                desc.push_str(&format!("{}, ", ds));
            }
        }
    }

    desc.push_str(".\n");
    desc
}

/// Generate complexity description
fn generate_complexity_description(semantics: &CodeSemantics) -> String {
    let mut desc = String::new();

    desc.push_str(&format!("Time complexity: {}, Space complexity: {}.",
        semantics.complexity, semantics.space_complexity));

    desc
}

/// Generate concise single-line description
pub fn generate_summary(semantics: &CodeSemantics) -> String {
    let mut summary = String::new();

    // Main operation
    if let Some(op) = semantics.operations.first() {
        summary.push_str(op);
    } else if let Some(algo) = semantics.algorithms.first() {
        summary.push_str(algo);
    } else {
        summary.push_str("compute");
    }

    // Input types
    summary.push_str(" (");
    for (i, input) in semantics.io_types.inputs.iter().enumerate() {
        if i > 0 {
            summary.push_str(", ");
        }
        summary.push_str(input);
    }
    summary.push_str(") → ");
    summary.push_str(&semantics.io_types.output);

    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidirectional::analyzer::IOTypes;

    #[test]
    fn test_generate_add_function() {
        let semantics = CodeSemantics {
            operations: vec!["addition".to_string()],
            data_structures: vec![],
            algorithms: vec![],
            complexity: "O(1)".to_string(),
            space_complexity: "O(1)".to_string(),
            patterns: vec![],
            io_types: IOTypes {
                inputs: vec!["i64".to_string(), "i64".to_string()],
                output: "i64".to_string(),
            },
            side_effects: vec![],
            control_flow: vec![],
        };

        let desc = generate_nl(&semantics);
        assert!(desc.contains("adds two values"));
        assert!(desc.contains("O(1)"));
    }

    #[test]
    fn test_generate_summary() {
        let semantics = CodeSemantics {
            operations: vec!["addition".to_string()],
            data_structures: vec![],
            algorithms: vec![],
            complexity: "O(1)".to_string(),
            space_complexity: "O(1)".to_string(),
            patterns: vec![],
            io_types: IOTypes {
                inputs: vec!["i64".to_string(), "i64".to_string()],
                output: "i64".to_string(),
            },
            side_effects: vec![],
            control_flow: vec![],
        };

        let summary = generate_summary(&semantics);
        assert_eq!(summary, "addition (i64, i64) → i64");
    }
}
