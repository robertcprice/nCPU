//! QUARANTINED legacy LLM NL module.
//!
//! Production NL routing is `linguigenesis_bridge` + `solve_from_description`.
//! This module delegates synthesis to the registry-driven bridge; the old
//! `ExampleSynthesizer` keyword path is not used.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// API key for Anthropic Claude
/// In production, this should come from environment variables
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// Maximum number of cached examples to maintain
const MAX_CACHE_SIZE: usize = 1000;

/// Default model to use for parsing
const DEFAULT_MODEL: &str = "claude-3-5-sonnet-20241022";

/// Error types for natural language processing
#[derive(Debug, thiserror::Error)]
pub enum NLError {
    /// API key not found or invalid
    #[error("Anthropic API key not found. Set {0} environment variable.")]
    MissingApiKey(String),

    /// API request failed
    #[error("Anthropic API request failed: {0}")]
    ApiError(String),

    /// Response parsing failed
    #[error("Failed to parse API response: {0}")]
    ParseError(String),

    /// Invalid input format
    #[error("Invalid input format: {0}")]
    InvalidInput(String),

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Not implemented
    #[error("Feature not yet implemented")]
    NotImplemented,
}

/// Parsed requirements from natural language input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedRequirements {
    /// Function name to synthesize
    pub function_name: String,

    /// Input parameter specifications
    pub inputs: Vec<InputSpec>,

    /// Output type specification
    pub output: OutputSpec,

    /// Natural language description of the function
    pub description: String,

    /// Extracted examples from the input
    pub examples: Vec<Example>,

    /// Any additional constraints or hints
    pub constraints: Vec<String>,
}

/// Input parameter specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    /// Parameter name
    pub name: String,

    /// Parameter type (for now: "int", "list", "string")
    pub type_: String,

    /// Optional description
    pub description: Option<String>,
}

/// Output specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    /// Output type (for now: "int")
    pub type_: String,

    /// Optional description
    pub description: Option<String>,
}

/// Input/output example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Input values
    pub inputs: Vec<serde_json::Value>,

    /// Expected output value
    pub expected: serde_json::Value,

    /// Optional explanation
    pub explanation: Option<String>,
}

/// Cached parsing result
#[derive(Debug, Clone)]
struct CacheEntry {
    requirements: ParsedRequirements,
    timestamp: std::time::Instant,
}

/// Natural Language Processing Pipeline
pub struct NLPipeline {
    /// API key for Anthropic (stored when available)
    api_key: Option<String>,

    /// Model identifier to use
    model: String,

    /// Example cache for repeated queries
    example_cache: Arc<RwLock<HashMap<String, CacheEntry>>>,

    /// Maximum cache size
    max_cache_size: usize,

    /// Whether to use the Anthropic API
    use_api: bool,
}

impl NLPipeline {
    /// Create a new NLP pipeline
    pub fn new() -> Self {
        let api_key = std::env::var(ANTHROPIC_API_KEY_ENV).ok();
        let use_api = api_key.is_some();

        Self {
            api_key,
            model: DEFAULT_MODEL.to_string(),
            example_cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size: MAX_CACHE_SIZE,
            use_api,
        }
    }

    /// Create with custom model
    pub fn with_model(model: String) -> Self {
        let mut pipeline = Self::new();
        pipeline.model = model;
        pipeline
    }

    /// Create with API key
    pub fn with_api_key(api_key: String) -> Self {
        Self {
            api_key: Some(api_key),
            model: DEFAULT_MODEL.to_string(),
            example_cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size: MAX_CACHE_SIZE,
            use_api: true,
        }
    }

    /// Check if the pipeline is ready (has API key)
    pub fn is_ready(&self) -> bool {
        self.api_key.is_some()
    }

    /// Parse natural language requirements into structured format
    pub async fn parse_requirements(&self, input: &str) -> Result<ParsedRequirements, NLError> {
        // Check cache first
        let cache_key = Self::compute_cache_key(input);
        {
            let cache = self.example_cache.read().await;
            if let Some(entry) = cache.get(&cache_key) {
                // Cache hit - return cached result
                return Ok(entry.requirements.clone());
            }
        }

        // If API is not available, return error
        if !self.use_api {
            return Err(NLError::NotImplemented);
        }

        // TODO: Implement actual API call using reqwest directly
        // For now, return a placeholder error
        let _ = (input, &self.model, &self.api_key);
        Err(NLError::NotImplemented)
    }

    /// Parse requirements from pre-extracted examples
    pub fn parse_from_examples(&self, examples: Vec<Example>) -> ParsedRequirements {
        // Simple extraction when examples are already provided
        ParsedRequirements {
            function_name: "synthesized_function".to_string(),
            inputs: self.infer_input_specs(&examples),
            output: OutputSpec {
                type_: "int".to_string(),
                description: None,
            },
            description: "Function synthesized from examples".to_string(),
            examples,
            constraints: vec![],
        }
    }

    /// Infer input specifications from examples
    fn infer_input_specs(&self, examples: &[Example]) -> Vec<InputSpec> {
        if examples.is_empty() {
            return vec![];
        }

        let first_example = &examples[0];
        first_example
            .inputs
            .iter()
            .enumerate()
            .map(|(i, val)| InputSpec {
                name: format!("arg_{}", i),
                type_: self.infer_type(val),
                description: None,
            })
            .collect()
    }

    /// Infer type from JSON value
    fn infer_type(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Number(_) => "int".to_string(),
            serde_json::Value::Array(_) => "list".to_string(),
            serde_json::Value::String(_) => "string".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Compute cache key from input
    fn compute_cache_key(input: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        input.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Cache a parsing result
    async fn cache_result(&self, key: String, requirements: ParsedRequirements) {
        let mut cache = self.example_cache.write().await;

        // Evict old entries if cache is too large
        if cache.len() >= self.max_cache_size {
            // Simple eviction: remove oldest entry by timestamp
            if let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.timestamp)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest_key);
            }
        }

        cache.insert(
            key,
            CacheEntry {
                requirements,
                timestamp: std::time::Instant::now(),
            },
        );
    }

    /// Clear the example cache
    pub async fn clear_cache(&self) {
        let mut cache = self.example_cache.write().await;
        cache.clear();
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> (usize, usize) {
        let cache = self.example_cache.read().await;
        (cache.len(), self.max_cache_size)
    }
}

impl Default for NLPipeline {
    fn default() -> Self {
        Self::new()
    }
}

pub mod synthesizer;

pub mod dialogue;

/// Natural language synthesis result
#[derive(Debug, Clone)]
pub struct NLSynthesisResult {
    /// Generated code
    pub code: String,

    /// Method used for synthesis
    pub method: String,

    /// Whether synthesis was successful
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,
}

impl NLPipeline {
    /// Get the model identifier
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Synthesize a program from natural language input
    ///
    /// This is the main entry point for the NL → code pipeline.
    /// It takes natural language input, generates examples, and
    /// synthesizes a program using the nCPU/nSynth solver.
    pub fn synthesize_from_nl(&self, input: &str) -> NLSynthesisResult {
        let bridge = crate::linguigenesis_bridge::LinguigenesisBridge::new();
        match bridge.synthesize_from_description(input, None) {
            Ok(result) => NLSynthesisResult {
                code: result.code,
                method: result.method,
                success: result.success,
                error: result.error,
            },
            Err(message) => NLSynthesisResult {
                code: String::new(),
                method: "linguigenesis_bridge".to_string(),
                success: false,
                error: Some(message),
            },
        }
    }

    /// Convert JSON value to nCPU/nSynth Value
    fn json_to_value(&self, json: serde_json::Value) -> Option<crate::benchmark::Value> {
        use crate::benchmark::Value;
        match json {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(Value::Int(i))
                } else if let Some(f) = n.as_f64() {
                    Some(Value::Float(f64::to_bits(f)))
                } else {
                    None
                }
            }
            serde_json::Value::Array(arr) => {
                // Recurse on EVERY element instead of the old lossy
                // `filter_map(as_i64)` flatten (which turned `[[1,2],[3]]` into
                // `[]`, `["a","b"]` into `[]`, and silently dropped non-int
                // elements). Each element converts through `json_to_value`, so
                // nesting and element types survive; if any element is
                // unrepresentable, the whole array is `None` (all-or-nothing,
                // never a silent drop). All-int arrays are byte-identical to the
                // old `int_array` path since each number becomes `Value::Int`.
                let elems: Vec<Value> = arr
                    .into_iter()
                    .map(|v| self.json_to_value(v))
                    .collect::<Option<Vec<_>>>()?;
                Some(Value::array_of(elems))
            }
            serde_json::Value::String(s) => Some(Value::Str(s)),
            serde_json::Value::Bool(b) => Some(Value::Bool(b)),
            _ => None,
        }
    }

    /// Infer function signature from examples
    fn infer_signature(&self, examples: &[crate::benchmark::Example]) -> String {
        use crate::benchmark::Value;
        if examples.is_empty() {
            return "fn f() -> i64".to_string();
        }

        let first = &examples[0];

        // Infer parameter types from the ACTUAL values (recursing into arrays so
        // `[[i64]]`/`[string]` survive instead of always collapsing to `[i64]`).
        let mut params = Vec::new();
        for (i, input) in first.inputs.iter().enumerate() {
            params.push(format!("x{}: {}", i, Self::value_type_str(input)));
        }

        // Infer return type the same way.
        let return_type = Self::value_type_str(&first.expected);

        format!("fn f({}) -> {}", params.join(", "), return_type)
    }

    /// Derive the signature type string for a value, recursing into arrays so the
    /// element/nested type is reflected accurately (`[i64]`, `[[i64]]`,
    /// `[string]`, …) rather than the old hard-coded `[i64]`. An empty array
    /// defaults its element type to `i64` (the historical default). Scalars and
    /// the structural shapes (`Pair`/`Quad`/`Tree`) keep their existing strings.
    fn value_type_str(value: &crate::benchmark::Value) -> String {
        use crate::benchmark::Value;
        match value {
            Value::Int(_) => "i64".to_string(),
            Value::Float(_) => "f64".to_string(),
            Value::Str(_) => "string".to_string(),
            Value::Bool(_) => "bool".to_string(),
            Value::Array(elems) => {
                let inner = elems
                    .first()
                    .map(Self::value_type_str)
                    .unwrap_or_else(|| "i64".to_string());
                format!("[{inner}]")
            }
            Value::Pair(_, _) => "(i64, i64)".to_string(),
            Value::Quad(_, _, _, _) => "{a: i64, b: i64, c: i64, d: i64}".to_string(),
            Value::Tree(_) => "Tree".to_string(),
            // A positional tuple renders its element types: `(T0, T1, ...)`.
            Value::Tuple(elems) => format!(
                "({})",
                elems
                    .iter()
                    .map(Self::value_type_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            // A named struct renders `{name: T, ...}` from its own field names.
            Value::Struct(fields) => format!(
                "{{{}}}",
                fields
                    .iter()
                    .map(|(k, v)| format!("{k}: {}", Self::value_type_str(v)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    /// Generate a function name from input text
    fn function_name_from_input(&self, input: &str) -> String {
        input
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .take(3)
            .collect::<Vec<_>>()
            .join("_")
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect()
    }
}

#[cfg(test)]
mod tests;
