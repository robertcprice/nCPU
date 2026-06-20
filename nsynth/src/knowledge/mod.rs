//! Knowledge module for API and library information
//!
//! This module provides comprehensive knowledge about APIs, libraries,
//! and frameworks across multiple programming languages, enabling
//! intelligent NL→API mapping and code generation.

pub mod api_graph;
pub mod graph;

pub use api_graph::{
    populate_default_graph, APICategory, APIGraph, APINode, Alternative, Language, MigrationPath,
    SharedAPIGraph, UsagePattern, API_ID,
};

pub use graph::{CodeKnowledgeGraph, DonorNode};

/// Natural language query processing for API discovery
pub fn query_apis(query: &str, language_hint: Option<Language>) -> Vec<String> {
    let graph = populate_default_graph();
    let mut results = Vec::new();

    let query_lower = query.to_lowercase();

    for node in graph.nodes.values() {
        if language_hint.map_or(true, |lang| node.language == lang) {
            if node.name.to_lowercase().contains(&query_lower)
                || node
                    .tags
                    .iter()
                    .any(|t| t.to_lowercase().contains(&query_lower))
            {
                results.push(node.name.clone());
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_apis() {
        let results = query_apis("react", None);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_query_with_language() {
        let results = query_apis("web", Some(Language::Rust));
        // Should return Rust web frameworks
        assert!(!results.is_empty());
    }
}
