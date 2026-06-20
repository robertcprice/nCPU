//! Semantic analyzer for code → NL
//!
//! Extracts operations, patterns, algorithms from AST for NL generation.

use crate::bidirectional::parser::{BinOp, Expression, Statement, AST};

/// Semantic information extracted from code
#[derive(Debug, Clone)]
pub struct CodeSemantics {
    /// Operations detected (map, filter, reduce, etc.)
    pub operations: Vec<String>,
    /// Data structures used (array, stack, queue, etc.)
    pub data_structures: Vec<String>,
    /// Algorithms detected (binary search, merge sort, etc.)
    pub algorithms: Vec<String>,
    /// Time complexity
    pub complexity: String,
    /// Space complexity
    pub space_complexity: String,
    /// Key patterns identified
    pub patterns: Vec<String>,
    /// Input/output types
    pub io_types: IOTypes,
    /// Side effects
    pub side_effects: Vec<String>,
    /// Control flow patterns
    pub control_flow: Vec<String>,
}

/// Input/output type information
#[derive(Debug, Clone)]
pub struct IOTypes {
    pub inputs: Vec<String>,
    pub output: String,
}

/// Analyze AST to extract semantic information
pub fn analyze_semantics(ast: &AST) -> CodeSemantics {
    let mut operations = Vec::new();
    let mut data_structures = Vec::new();
    let mut algorithms = Vec::new();
    let mut patterns = Vec::new();
    let mut control_flow = Vec::new();
    let side_effects = Vec::new();

    for func in &ast.functions {
        // Analyze function name
        analyze_function_name(&func.name, &mut operations, &mut algorithms);

        // Analyze parameters for data structures
        for param in &func.params {
            analyze_type(&param.type_, &mut data_structures);
        }

        // Analyze body
        for stmt in &func.body {
            analyze_statement(stmt, &mut operations, &mut control_flow, &mut patterns);
        }
    }

    // Infer complexity from algorithms and operations
    let complexity = infer_complexity(&algorithms, &operations);
    let space_complexity = infer_space_complexity(&data_structures, &algorithms);

    // Extract I/O types from first function
    let io_types = extract_io_types(ast);

    CodeSemantics {
        operations,
        data_structures,
        algorithms,
        complexity,
        space_complexity,
        patterns,
        io_types,
        side_effects,
        control_flow,
    }
}

/// Analyze function name for hints about operations
fn analyze_function_name(name: &str, operations: &mut Vec<String>, algorithms: &mut Vec<String>) {
    let name_lower = name.to_lowercase();

    // Map common name patterns to operations
    let op_keywords = [
        ("map", "map"),
        ("filter", "filter"),
        ("reduce", "reduce"),
        ("fold", "fold"),
        ("scan", "scan"),
        ("zip", "zip"),
        ("split", "split"),
        ("join", "join"),
        ("reverse", "reverse"),
        ("sort", "sort"),
        ("search", "search"),
        ("find", "search"),
        ("contains", "contains"),
        ("add", "add"),
        ("subtract", "subtract"),
        ("multiply", "multiply"),
        ("divide", "divide"),
    ];

    for (keyword, op) in op_keywords {
        if name_lower.contains(keyword) {
            if !operations.contains(&op.to_string()) {
                operations.push(op.to_string());
            }
        }
    }

    // Algorithm-specific names
    let algo_keywords = [
        ("binary", "binary_search"),
        ("merge", "merge_sort"),
        ("quick", "quick_sort"),
        ("heap", "heap_sort"),
        ("bubble", "bubble_sort"),
        ("insert", "insertion_sort"),
        ("selection", "selection_sort"),
        ("dijkstra", "dijkstra"),
        ("astar", "astar"),
        ("dfs", "depth_first_search"),
        ("bfs", "breadth_first_search"),
        ("fibonacci", "fibonacci"),
        ("factorial", "factorial"),
        ("gcd", "gcd"),
        ("lcm", "lcm"),
    ];

    for (keyword, algo) in algo_keywords {
        if name_lower.contains(keyword) {
            if !algorithms.contains(&algo.to_string()) {
                algorithms.push(algo.to_string());
            }
        }
    }
}

/// Analyze type for data structure hints
fn analyze_type(type_: &str, data_structures: &mut Vec<String>) {
    let type_lower = type_.to_lowercase();

    let type_keywords = [
        ("vec", "vector"),
        ("array", "array"),
        ("slice", "array"),
        ("list", "list"),
        ("stack", "stack"),
        ("queue", "queue"),
        ("deque", "deque"),
        ("heap", "heap"),
        ("hashmap", "hash_map"),
        ("map", "map"),
        ("set", "set"),
        ("treeset", "tree"),
        ("treemap", "tree"),
        ("string", "string"),
        ("str", "string"),
        ("option", "option"),
        ("result", "result"),
    ];

    for (keyword, ds) in type_keywords {
        if type_lower.contains(keyword) {
            if !data_structures.contains(&ds.to_string()) {
                data_structures.push(ds.to_string());
            }
        }
    }
}

/// Analyze statement for operations and patterns
fn analyze_statement(
    stmt: &Statement,
    operations: &mut Vec<String>,
    control_flow: &mut Vec<String>,
    patterns: &mut Vec<String>,
) {
    match stmt {
        Statement::Let { .. } => patterns.push("variable_declaration".to_string()),
        Statement::Assign { .. } => patterns.push("assignment".to_string()),
        Statement::Return(_) => patterns.push("return".to_string()),
        Statement::If {
            else_block: Some(_),
            ..
        } => control_flow.push("conditional_with_else".to_string()),
        Statement::If { .. } => control_flow.push("conditional".to_string()),
        Statement::Loop { .. } => control_flow.push("infinite_loop".to_string()),
        Statement::While { .. } => control_flow.push("while_loop".to_string()),
        Statement::For { .. } => control_flow.push("for_loop".to_string()),
        Statement::Block(stmts) => {
            for s in stmts {
                analyze_statement(s, operations, control_flow, patterns);
            }
        }
        Statement::Expr(expr) => analyze_expression(expr, operations, patterns),
    }
}

/// Analyze expression for operations
fn analyze_expression(expr: &Expression, operations: &mut Vec<String>, patterns: &mut Vec<String>) {
    match expr {
        Expression::BinOp { op, .. } => match op {
            BinOp::Add => operations.push("addition".to_string()),
            BinOp::Sub => operations.push("subtraction".to_string()),
            BinOp::Mul => operations.push("multiplication".to_string()),
            BinOp::Div => operations.push("division".to_string()),
            BinOp::Mod => operations.push("modulo".to_string()),
            BinOp::And => operations.push("logical_and".to_string()),
            BinOp::Or => operations.push("logical_or".to_string()),
            BinOp::Eq => operations.push("equality_check".to_string()),
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                operations.push("comparison".to_string())
            }
            _ => {}
        },
        Expression::Call { func, .. } => {
            // Track method calls as operations
            if !operations.contains(func) {
                operations.push(func.clone());
            }
        }
        Expression::MethodCall { method, .. } => {
            if !operations.contains(method) {
                operations.push(method.clone());
            }
        }
        Expression::Array(_) => patterns.push("array_literal".to_string()),
        _ => {}
    }
}

/// Infer time complexity from algorithms and operations
fn infer_complexity(algorithms: &[String], operations: &[String]) -> String {
    // Check for specific algorithms first
    for algo in algorithms {
        match algo.as_str() {
            "binary_search" | "binary" => return "O(log n)".to_string(),
            "merge_sort" | "quick_sort" | "heap_sort" => return "O(n log n)".to_string(),
            "bubble_sort" | "insertion_sort" | "selection_sort" => return "O(n²)".to_string(),
            "dijkstra" => return "O((V + E) log V)".to_string(),
            "bfs" | "dfs" => return "O(V + E)".to_string(),
            "fibonacci" | "factorial" => return "O(2ⁿ)".to_string(),
            _ => {}
        }
    }

    // Check for operation patterns
    if operations
        .iter()
        .any(|o| o.contains("nested") || o.contains("double"))
    {
        return "O(n²)".to_string();
    }

    if operations.iter().any(|o| o == "sort" || o == "search") {
        return "O(n log n)".to_string();
    }

    if operations
        .iter()
        .any(|o| o.contains("loop") || o.contains("iteration"))
    {
        return "O(n)".to_string();
    }

    "O(1)".to_string()
}

/// Infer space complexity
fn infer_space_complexity(data_structures: &[String], algorithms: &[String]) -> String {
    for algo in algorithms {
        match algo.as_str() {
            "merge_sort" | "quick_sort" => return "O(n)".to_string(),
            "recursive" => return "O(n)".to_string(),
            _ => {}
        }
    }

    if data_structures
        .iter()
        .any(|ds| ds == "vector" || ds == "array")
    {
        return "O(n)".to_string();
    }

    "O(1)".to_string()
}

/// Extract I/O types from AST
fn extract_io_types(ast: &AST) -> IOTypes {
    if let Some(func) = ast.functions.first() {
        let inputs = func.params.iter().map(|p| p.type_.clone()).collect();
        let output = func.return_type.clone();
        IOTypes { inputs, output }
    } else {
        IOTypes {
            inputs: vec!["unknown".to_string()],
            output: "unknown".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidirectional::parser::parse_code;

    #[test]
    fn test_analyze_simple_function() {
        let code = r#"
            fn add(a: i64, b: i64) -> i64 {
                return a + b;
            }
        "#;

        let ast = parse_code(code).unwrap();
        let semantics = analyze_semantics(&ast);

        assert!(!semantics.operations.is_empty());
        assert_eq!(semantics.complexity, "O(1)");
    }

    #[test]
    fn test_analyze_array_function() {
        let code = r#"
            fn process_array(arr: Vec<i64>) -> Vec<i64> {
                return arr;
            }
        "#;

        let ast = parse_code(code).unwrap();
        let semantics = analyze_semantics(&ast);

        assert!(semantics.data_structures.contains(&"vector".to_string()));
        assert_eq!(semantics.space_complexity, "O(n)");
    }
}
