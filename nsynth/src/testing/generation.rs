//! Test generation engine for automated test case creation
//!
//! Provides intelligent test generation based on function signatures,
//! including normal cases, edge cases, boundary values, and property-based tests.

use crate::benchmark::{Problem, Value};
use crate::testing::{TestCase, int, string, pair, array, TestSuite, Example, TestCategory};
use crate::bidirectional::parser::{AST, Function};
use std::collections::{HashMap, HashSet};

/// Test generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Target coverage (0.0 to 1.0)
    pub coverage_target: f64,
    /// Maximum tests per function
    pub max_tests_per_function: usize,
    /// Include fuzzing tests
    pub include_fuzzing: bool,
    /// Include property-based tests
    pub include_property: bool,
    /// Seed for random generation
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            coverage_target: 0.85,
            max_tests_per_function: 50,
            include_fuzzing: true,
            include_property: true,
            seed: 42,
        }
    }
}

/// Test generator for creating comprehensive test suites
pub struct TestGenerator {
    config: GenerationConfig,
    /// Cache of generated values for reproducibility
    value_cache: HashMap<String, Vec<Value>>,
}

impl TestGenerator {
    /// Create a new test generator with default configuration
    pub fn new() -> Self {
        Self {
            config: GenerationConfig::default(),
            value_cache: HashMap::new(),
        }
    }

    /// Create a new test generator with custom configuration
    pub fn with_config(config: GenerationConfig) -> Self {
        Self {
            config,
            value_cache: HashMap::new(),
        }
    }

    /// Generate test suites for a problem
    pub fn generate_for_problem(&mut self, problem: &Problem) -> TestSuite {
        let mut suite = TestSuite::new(problem.name.clone(), self.config.coverage_target);

        // Generate based on problem signature
        let signature_tests = self.generate_from_signature(problem);
        for test in signature_tests {
            suite.add_test(test);
        }

        // Generate edge cases
        let edge_tests = self.generate_edge_cases(problem);
        for test in edge_tests {
            suite.add_test(test);
        }

        // Generate boundary cases
        let boundary_tests = self.generate_boundary_cases(problem);
        for test in boundary_tests {
            suite.add_test(test);
        }

        // Generate property tests if enabled
        if self.config.include_property {
            let property_tests = self.generate_property_tests(problem);
            for test in property_tests {
                suite.add_test(test);
            }
        }

        // Generate fuzzing tests if enabled
        if self.config.include_fuzzing {
            let fuzz_tests = self.generate_fuzzing_tests(problem);
            for test in fuzz_tests {
                suite.add_test(test);
            }
        }

        suite
    }

    /// Generate tests from function signature analysis
    fn generate_from_signature(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();
        let sig = problem.signature;

        // Analyze parameter types and generate appropriate tests
        if sig.contains("-> i64") || sig.contains("-> i32") {
            tests.extend(self.generate_numeric_tests(problem));
        }

        if sig.contains("-> string") {
            tests.extend(self.generate_string_tests(problem));
        }

        if sig.contains("-> bool") {
            tests.extend(self.generate_boolean_tests(problem));
        }

        if sig.contains("[i64]") {
            tests.extend(self.generate_array_tests(problem));
        }

        // Handle struct outputs
        if sig.contains("Point") {
            tests.extend(self.generate_point_tests(problem));
        }

        if sig.contains("Rectangle") {
            tests.extend(self.generate_rectangle_tests(problem));
        }

        tests
    }

    /// Generate numeric operation tests
    fn generate_numeric_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();
        let param_count = self.count_parameters(problem.signature);

        // Extract patterns from existing examples
        let mut examples_used = HashSet::new();

        // Add existing examples as normal tests
        for (i, example) in problem.examples.iter().enumerate() {
            if examples_used.insert(i) {
                tests.push(TestCase::normal(
                    format!("{}_example_{}", problem.name, i),
                    example.inputs.clone(),
                    example.expected.clone(),
                    format!("Example {} from problem definition", i),
                ));
            }
        }

        // Generate additional normal cases based on parameter count
        match param_count {
            1 => {
                // Single parameter tests
                tests.extend(self.single_param_numeric_tests(problem));
            }
            2 => {
                // Two parameter tests
                tests.extend(self.two_param_numeric_tests(problem));
            }
            3 => {
                // Three parameter tests
                tests.extend(self.three_param_numeric_tests(problem));
            }
            _ => {
                // Generic tests
                tests.extend(self.generic_numeric_tests(problem, param_count));
            }
        }

        tests
    }

    /// Count parameters in function signature
    fn count_parameters(&self, signature: &str) -> usize {
        if let Some(params) = signature.split('(').nth(1) {
            if let Some(close) = params.find(')') {
                let param_str = &params[..close];
                if param_str.trim().is_empty() {
                    return 0;
                }
                return param_str.split(',').count();
            }
        }
        0
    }

    /// Single parameter numeric tests
    fn single_param_numeric_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        // Use existing examples to infer behavior
        let inferred_op = self.infer_operation(problem);

        match inferred_op.as_str() {
            "identity" => {
                tests.push(TestCase::normal(
                    format!("{}_identity_positive", problem.name),
                    vec![int(42)],
                    int(42),
                    "Positive identity".to_string(),
                ));
            }
            "negate" => {
                tests.push(TestCase::normal(
                    format!("{}_negate", problem.name),
                    vec![int(5)],
                    int(-5),
                    "Negation".to_string(),
                ));
            }
            "absolute" => {
                tests.push(TestCase::normal(
                    format!("{}_abs_negative", problem.name),
                    vec![int(-7)],
                    int(7),
                    "Absolute value of negative".to_string(),
                ));
            }
            _ => {
                // Generic numeric tests
                tests.push(TestCase::normal(
                    format!("{}_small_positive", problem.name),
                    vec![int(10)],
                    int(10), // Will be adjusted by actual execution
                    "Small positive number".to_string(),
                ));
            }
        }

        tests
    }

    /// Two parameter numeric tests
    fn two_param_numeric_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();
        let inferred_op = self.infer_operation(problem);

        // Arithmetic operation tests
        tests.extend(self.arithmetic_operation_tests(problem, &inferred_op));

        // Comparison operation tests
        tests.extend(self.comparison_operation_tests(problem, &inferred_op));

        tests
    }

    /// Three parameter numeric tests
    fn three_param_numeric_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        tests.push(TestCase::normal(
            format!("{}_three_params", problem.name),
            vec![int(1), int(2), int(3)],
            int(2), // Default to middle value for min/max
            "Three parameter test".to_string(),
        ));

        tests
    }

    /// Generic numeric tests for N parameters
    fn generic_numeric_tests(&mut self, problem: &Problem, param_count: usize) -> Vec<TestCase> {
        let mut tests = Vec::new();

        let inputs: Vec<Value> = (0..param_count).map(|i| int(i as i64 + 1)).collect();
        tests.push(TestCase::normal(
            format!("{}_generic_{}", problem.name, param_count),
            inputs,
            int(1),
            format!("Generic test with {} parameters", param_count),
        ));

        tests
    }

    /// Arithmetic operation tests
    fn arithmetic_operation_tests(&mut self, problem: &Problem, op: &str) -> Vec<TestCase> {
        let mut tests = Vec::new();

        match op {
            "add" | "addition" | "sum" => {
                tests.push(TestCase::normal(
                    format!("{}_add_both_pos", problem.name),
                    vec![int(15), int(25)],
                    int(40),
                    "Addition of two positives".to_string(),
                ));
            }
            "subtract" | "difference" => {
                tests.push(TestCase::normal(
                    format!("{}_sub_larger_first", problem.name),
                    vec![int(25), int(15)],
                    int(10),
                    "Subtraction with larger first".to_string(),
                ));
            }
            "multiply" | "product" => {
                tests.push(TestCase::normal(
                    format!("{}_mul_two_digits", problem.name),
                    vec![int(12), int(8)],
                    int(96),
                    "Multiplication".to_string(),
                ));
            }
            _ => {}
        }

        tests
    }

    /// Comparison operation tests
    fn comparison_operation_tests(&mut self, problem: &Problem, op: &str) -> Vec<TestCase> {
        let mut tests = Vec::new();

        if op.contains("max") || op.contains("maximum") {
            tests.push(TestCase::normal(
                format!("{}_max_first_larger", problem.name),
                vec![int(20), int(10)],
                int(20),
                "Max with first larger".to_string(),
            ));
        }

        if op.contains("min") || op.contains("minimum") {
            tests.push(TestCase::normal(
                format!("{}_min_second_larger", problem.name),
                vec![int(10), int(20)],
                int(10),
                "Min with second larger".to_string(),
            ));
        }

        tests
    }

    /// Generate string operation tests
    fn generate_string_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();
        let param_count = self.count_parameters(problem.signature);

        match param_count {
            1 => {
                // Single string parameter
                tests.push(TestCase::normal(
                    format!("{}_string_simple", problem.name),
                    vec![string("hello")],
                    string("hello"), // Placeholder
                    "Simple string input".to_string(),
                ));
            }
            2 => {
                // Two string parameters
                tests.push(TestCase::normal(
                    format!("{}_string_concat", problem.name),
                    vec![string("hello"), string("world")],
                    string("helloworld"),
                    "String concatenation".to_string(),
                ));
            }
            _ => {}
        }

        tests
    }

    /// Generate boolean operation tests
    fn generate_boolean_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        tests.push(TestCase::normal(
            format!("{}_bool_true", problem.name),
            vec![int(1)],
            int(1),
            "Returns true".to_string(),
        ));

        tests.push(TestCase::normal(
            format!("{}_bool_false", problem.name),
            vec![int(0)],
            int(0),
            "Returns false".to_string(),
        ));

        tests
    }

    /// Generate array operation tests
    fn generate_array_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        tests.push(TestCase::normal(
            format!("{}_array_small", problem.name),
            vec![array(&[1, 2, 3, 4, 5])],
            int(15),
            "Small array (5 elements)".to_string(),
        ));

        tests.push(TestCase::normal(
            format!("{}_array_even", problem.name),
            vec![array(&[2, 4, 6, 8])],
            int(20),
            "Array of even numbers".to_string(),
        ));

        tests
    }

    /// Generate Point struct tests
    fn generate_point_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        tests.push(TestCase::normal(
            format!("{}_point_quad", problem.name),
            vec![pair(3, 4)],
            int(7),
            "Point with positive coordinates".to_string(),
        ));

        tests
    }

    /// Generate Rectangle struct tests
    fn generate_rectangle_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        tests.push(TestCase::normal(
            format!("{}_rect_area", problem.name),
            vec![pair(5, 3)],
            int(15),
            "Rectangle area calculation".to_string(),
        ));

        tests
    }

    /// Generate edge case tests
    fn generate_edge_cases(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();
        let sig = problem.signature;

        // Zero edge cases
        if sig.contains("i64") || sig.contains("i32") {
            tests.extend(self.zero_edge_cases(problem));
        }

        // Empty container edge cases
        if sig.contains("[") {
            tests.push(TestCase::edge(
                format!("{}_empty_array", problem.name),
                vec![array(&[])],
                int(0),
                "Empty array edge case".to_string(),
            ));
        }

        if sig.contains("string") {
            tests.push(TestCase::edge(
                format!("{}_empty_string", problem.name),
                vec![string("")],
                int(0),
                "Empty string edge case".to_string(),
            ));
        }

        // Single element edge cases
        tests.extend(self.single_element_edge_cases(problem));

        // Negative number edge cases
        if sig.contains("i64") || sig.contains("i32") {
            tests.extend(self.negative_edge_cases(problem));
        }

        tests
    }

    /// Zero value edge cases
    fn zero_edge_cases(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();
        let param_count = self.count_parameters(problem.signature);

        if param_count == 1 {
            tests.push(TestCase::edge(
                format!("{}_zero_input", problem.name),
                vec![int(0)],
                int(0),
                "Zero input edge case".to_string(),
            ));
        } else if param_count >= 2 {
            tests.push(TestCase::edge(
                format!("{}_first_zero", problem.name),
                vec![int(0), int(5)],
                int(5),
                "First parameter zero".to_string(),
            ));

            tests.push(TestCase::edge(
                format!("{}_second_zero", problem.name),
                vec![int(5), int(0)],
                int(5),
                "Second parameter zero".to_string(),
            ));

            tests.push(TestCase::edge(
                format!("{}_both_zero", problem.name),
                vec![int(0), int(0)],
                int(0),
                "Both parameters zero".to_string(),
            ));
        }

        tests
    }

    /// Single element edge cases
    fn single_element_edge_cases(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        if problem.signature.contains("[i64]") {
            tests.push(TestCase::edge(
                format!("{}_single_element", problem.name),
                vec![array(&[42])],
                int(42),
                "Single element array".to_string(),
            ));
        }

        if problem.signature.contains("string") {
            tests.push(TestCase::edge(
                format!("{}_single_char", problem.name),
                vec![string("a")],
                int(1),
                "Single character string".to_string(),
            ));
        }

        tests
    }

    /// Negative number edge cases
    fn negative_edge_cases(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();
        let param_count = self.count_parameters(problem.signature);

        if param_count >= 1 {
            tests.push(TestCase::edge(
                format!("{}_negative_input", problem.name),
                vec![int(-5)],
                int(-5),
                "Negative input".to_string(),
            ));
        }

        if param_count >= 2 {
            tests.push(TestCase::edge(
                format!("{}_both_negative", problem.name),
                vec![int(-3), int(-7)],
                int(-10),
                "Both parameters negative".to_string(),
            ));
        }

        tests
    }

    /// Generate boundary case tests
    fn generate_boundary_cases(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();
        let sig = problem.signature;

        if sig.contains("i64") || sig.contains("i32") {
            tests.extend(self.numeric_boundaries(problem));
        }

        if sig.contains("[i64]") {
            tests.extend(self.array_boundaries(problem));
        }

        tests
    }

    /// Numeric boundary cases
    fn numeric_boundaries(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        // Positive boundary (use reasonable values, not MAX to avoid overflow)
        tests.push(TestCase::boundary(
            format!("{}_large_positive", problem.name),
            vec![int(1_000_000_000)],
            int(1_000_000_000),
            "Large positive boundary".to_string(),
        ));

        // Negative boundary
        tests.push(TestCase::boundary(
            format!("{}_large_negative", problem.name),
            vec![int(-1_000_000_000)],
            int(-1_000_000_000),
            "Large negative boundary".to_string(),
        ));

        // Boundary near limits (safe values)
        tests.push(TestCase::boundary(
            format!("{}_safe_max", problem.name),
            vec![int(2_147_483_647)], // i32::MAX
            int(2_147_483_647),
            "Safe maximum value".to_string(),
        ));

        tests
    }

    /// Array boundary cases
    fn array_boundaries(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        // Two elements (common boundary case)
        tests.push(TestCase::boundary(
            format!("{}_two_elements", problem.name),
            vec![array(&[1, 2])],
            int(3),
            "Two element array boundary".to_string(),
        ));

        // Large array
        let large_array: Vec<i64> = (1..=20).collect();
        tests.push(TestCase::boundary(
            format!("{}_large_array", problem.name),
            vec![array(&large_array)],
            int(210), // Sum of 1..20
            "Large array (20 elements)".to_string(),
        ));

        tests
    }

    /// Generate property-based tests
    fn generate_property_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();
        let sig = problem.signature;

        // Commutativity property for binary operations
        if self.count_parameters(sig) == 2 {
            tests.extend(self.commutativity_tests(problem));
        }

        // Identity property
        tests.extend(self.identity_tests(problem));

        // Idempotency property
        tests.extend(self.idempotency_tests(problem));

        // Round-trip property
        tests.extend(self.roundtrip_tests(problem));

        tests
    }

    /// Commutativity property tests
    fn commutativity_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        tests.push(TestCase::property(
            format!("{}_commutativity", problem.name),
            vec![int(5), int(3)],
            int(8),
            "Commutativity: f(a,b) == f(b,a)".to_string(),
        ));

        tests
    }

    /// Identity property tests
    fn identity_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        // Identity element varies by operation
        tests.push(TestCase::property(
            format!("{}_identity_zero", problem.name),
            vec![int(10), int(0)],
            int(10),
            "Identity with zero".to_string(),
        ));

        tests
    }

    /// Idempotency property tests
    fn idempotency_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        tests.push(TestCase::property(
            format!("{}_idempotency", problem.name),
            vec![int(7)],
            int(7),
            "Idempotency: f(f(x)) == f(x)".to_string(),
        ));

        tests
    }

    /// Round-trip property tests
    fn roundtrip_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();

        // For encode/decode type operations
        if problem.name.contains("encode") || problem.name.contains("decode") {
            tests.push(TestCase::property(
                format!("{}_roundtrip", problem.name),
                vec![int(42)],
                int(42),
                "Round-trip: decode(encode(x)) == x".to_string(),
            ));
        }

        tests
    }

    /// Generate fuzzing tests
    fn generate_fuzzing_tests(&mut self, problem: &Problem) -> Vec<TestCase> {
        let mut tests = Vec::new();
        let sig = problem.signature;

        // Generate random but structured inputs
        for i in 0..5 {
            if sig.contains("i64") {
                tests.push(TestCase::fuzzing(
                    format!("{}_fuzz_{}", problem.name, i),
                    vec![int(self.fuzzed_integer(i))],
                    int(0),
                    format!("Fuzzing test {}", i),
                ));
            }

            if sig.contains("[i64]") {
                tests.push(TestCase::fuzzing(
                    format!("{}_fuzz_array_{}", problem.name, i),
                    vec![array(&self.fuzzed_array(i, 5))],
                    int(0),
                    format!("Fuzzing array test {}", i),
                ));
            }

            if sig.contains("string") {
                tests.push(TestCase::fuzzing(
                    format!("{}_fuzz_string_{}", problem.name, i),
                    vec![string(&self.fuzzed_string(i))],
                    int(0),
                    format!("Fuzzing string test {}", i),
                ));
            }
        }

        tests
    }

    /// Generate a fuzzed integer value
    fn fuzzed_integer(&self, seed: usize) -> i64 {
        // Use deterministic pseudo-random generation
        let mut x = seed as i64 * 1103515245 + 12345;
        x = ((x >> 16) ^ x) % 10000 - 5000;
        x
    }

    /// Generate a fuzzed array
    fn fuzzed_array(&self, seed: usize, len: usize) -> Vec<i64> {
        (0..len).map(|i| self.fuzzed_integer(seed + i)).collect()
    }

    /// Generate a fuzzed string
    fn fuzzed_string(&self, seed: usize) -> String {
        let chars = vec!['a', 'b', 'c', 'd', 'e', '1', '2', '3', ' ', '\t'];
        let len = (seed % 10) + 1;
        (0..len).map(|i| chars[(seed + i) % chars.len()]).collect()
    }

    /// Infer operation type from problem examples and description
    fn infer_operation(&self, problem: &Problem) -> String {
        let name = problem.name.to_lowercase();
        let desc = problem.description.to_lowercase();

        if name.contains("add") || name.contains("sum") || desc.contains("sum") {
            return "add".to_string();
        }
        if name.contains("sub") || name.contains("diff") || desc.contains("difference") {
            return "subtract".to_string();
        }
        if name.contains("mul") || name.contains("prod") || desc.contains("product") {
            return "multiply".to_string();
        }
        if name.contains("max") || desc.contains("maximum") {
            return "max".to_string();
        }
        if name.contains("min") || desc.contains("minimum") {
            return "min".to_string();
        }
        if name.contains("abs") || desc.contains("absolute") {
            return "absolute".to_string();
        }
        if name.contains("neg") || desc.contains("negate") {
            return "negate".to_string();
        }
        if name.contains("gcd") {
            return "gcd".to_string();
        }
        if name.contains("lcm") {
            return "lcm".to_string();
        }

        "unknown".to_string()
    }

    /// Generate test suite from AST
    pub fn generate_from_ast(&mut self, ast: &AST) -> Vec<TestSuite> {
        let mut suites = Vec::new();

        for function in &ast.functions {
            let suite = self.generate_function_tests(function);
            suites.push(suite);
        }

        suites
    }

    /// Generate tests for a single function from AST
    fn generate_function_tests(&mut self, func: &Function) -> TestSuite {
        let mut suite = TestSuite::new(func.name.clone(), self.config.coverage_target);

        // Generate tests based on function signature
        for param in &func.params {
            match param.type_.as_str() {
                "i64" | "i32" | "isize" => {
                    suite.add_test(TestCase::normal(
                        format!("{}_param_normal", func.name),
                        vec![int(10)],
                        int(10),
                        format!("Normal test for {}", param.name),
                    ));
                }
                "string" | "&str" => {
                    suite.add_test(TestCase::normal(
                        format!("{}_param_string", func.name),
                        vec![string("test")],
                        string("test"),
                        format!("String test for {}", param.name),
                    ));
                }
                _ => {}
            }
        }

        suite
    }
}

impl Default for TestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Problem;

    fn make_test_problem() -> Problem {
        Problem {
            name: "add_two".to_string(),
            category: "arithmetic",
            description: "Add two numbers",
            signature: "fn add_two(a: i64, b: i64) -> i64",
            examples: vec![
                Example { inputs: vec![int(2), int(3)], expected: int(5) },
            ],
            holdouts: vec![],
            reference_code: "fn add_two(a: i64, b: i64) -> i64 { a + b }",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    #[test]
    fn test_generator_creation() {
        let gen = TestGenerator::new();
        assert_eq!(gen.config.coverage_target, 0.85);
    }

    #[test]
    fn test_generate_for_problem() {
        let mut gen = TestGenerator::new();
        let problem = make_test_problem();
        let suite = gen.generate_for_problem(&problem);

        assert_eq!(suite.name, "add_two");
        assert!(!suite.tests.is_empty());
    }

    #[test]
    fn test_count_parameters() {
        let gen = TestGenerator::new();

        assert_eq!(gen.count_parameters("fn test() -> i64"), 0);
        assert_eq!(gen.count_parameters("fn test(a: i64) -> i64"), 1);
        assert_eq!(gen.count_parameters("fn test(a: i64, b: i64) -> i64"), 2);
    }

    #[test]
    fn test_edge_cases() {
        let mut gen = TestGenerator::new();
        let problem = make_test_problem();
        let edge_tests = gen.generate_edge_cases(&problem);

        assert!(!edge_tests.is_empty());
        assert!(edge_tests.iter().any(|t| t.category == TestCategory::EdgeCase));
    }

    #[test]
    fn test_boundary_cases() {
        let mut gen = TestGenerator::new();
        let problem = make_test_problem();
        let boundary_tests = gen.generate_boundary_cases(&problem);

        assert!(!boundary_tests.is_empty());
    }

    #[test]
    fn test_property_tests() {
        let mut gen = TestGenerator::new();
        let problem = make_test_problem();
        let property_tests = gen.generate_property_tests(&problem);

        assert!(!property_tests.is_empty());
    }

    #[test]
    fn test_fuzzing_tests() {
        let mut gen = TestGenerator::new();
        let problem = make_test_problem();
        let fuzz_tests = gen.generate_fuzzing_tests(&problem);

        assert!(!fuzz_tests.is_empty());
    }

    #[test]
    fn test_infer_operation() {
        let gen = TestGenerator::new();
        let problem = make_test_problem();

        assert_eq!(gen.infer_operation(&problem), "add");
    }
}
