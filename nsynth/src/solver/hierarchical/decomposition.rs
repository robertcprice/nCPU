//! Hierarchical decomposition for modular synthesis
//!
//! Breaks down large specifications into manageable modules
//! with clear interfaces and dependencies.

use crate::benchmark::{Example, Problem};
use std::collections::{HashMap, HashSet};

/// Module specification for hierarchical synthesis
#[derive(Debug, Clone)]
pub struct ModuleSpec {
    /// Module name
    pub name: String,
    /// Interface (imports/exports)
    pub interface: Interface,
    /// Examples for this module
    pub examples: Vec<Example>,
    /// Dependencies on other modules
    pub dependencies: Vec<String>,
    /// Type annotations
    pub types: HashMap<String, String>,
}

/// Interface definition
#[derive(Debug, Clone)]
pub struct Interface {
    /// Imported functions/types
    pub imports: Vec<Import>,
    /// Exported functions
    pub exports: Vec<Export>,
    /// Type declarations
    pub types: Vec<TypeDecl>,
}

/// Import declaration
#[derive(Debug, Clone)]
pub struct Import {
    pub name: String,
    pub module: String,
    pub type_: ImportType,
}

/// Import type
#[derive(Debug, Clone, PartialEq)]
pub enum ImportType {
    Function,
    Type,
    Constant,
}

/// Export declaration
#[derive(Debug, Clone)]
pub struct Export {
    pub name: String,
    pub signature: String,
    pub visibility: Visibility,
}

/// Visibility modifier
#[derive(Debug, Clone, PartialEq)]
pub enum Visibility {
    Public,
    Private,
    Protected,
}

/// Type declaration
#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub name: String,
    pub kind: TypeKind,
    pub generics: Vec<String>,
}

/// Type kind
#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    Struct,
    Enum,
    Trait,
    Alias(String),
    Primitive,
}

/// Problem specification for decomposition
#[derive(Debug, Clone)]
pub struct ProblemSpec {
    pub name: String,
    pub description: String,
    pub requirements: Vec<Requirement>,
    pub constraints: Vec<Constraint>,
}

/// Requirement
#[derive(Debug, Clone)]
pub struct Requirement {
    pub description: String,
    pub priority: Priority,
}

/// Priority level
#[derive(Debug, Clone, PartialEq)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// Constraint
#[derive(Debug, Clone)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub description: String,
}

/// Constraint kind
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    Performance,
    Memory,
    Correctness,
    Security,
}

/// Decompose problem spec into modules
pub fn decompose(spec: &ProblemSpec) -> Vec<ModuleSpec> {
    let mut modules = Vec::new();
    let mut module_names = HashSet::new();

    // Analyze requirements to identify natural boundaries
    for req in &spec.requirements {
        if let Some(module_name) = extract_module_from_requirement(req) {
            if !module_names.contains(&module_name) {
                module_names.insert(module_name.clone());
                modules.push(create_module_from_requirement(&module_name, req));
            }
        }
    }

    // If no modules found, create default single module
    if modules.is_empty() {
        modules.push(create_default_module(spec));
    }

    // Analyze dependencies between modules
    resolve_dependencies(&mut modules);

    modules
}

/// Extract module name from requirement
fn extract_module_from_requirement(req: &Requirement) -> Option<String> {
    let desc = req.description.to_lowercase();

    // Common patterns indicating modules
    let patterns = [
        ("authentication", "auth"),
        ("authorization", "authz"),
        ("database", "db"),
        ("storage", "storage"),
        ("api", "api"),
        ("http", "http"),
        ("websocket", "websocket"),
        ("validation", "validation"),
        ("logging", "logging"),
        ("config", "config"),
        ("crypto", "crypto"),
        ("compression", "compression"),
        ("serialization", "serde"),
    ];

    for (keyword, module) in patterns {
        if desc.contains(keyword) {
            return Some(module.to_string());
        }
    }

    None
}

/// Create module from requirement
fn create_module_from_requirement(name: &str, req: &Requirement) -> ModuleSpec {
    ModuleSpec {
        name: name.to_string(),
        interface: Interface {
            imports: Vec::new(),
            exports: Vec::new(),
            types: Vec::new(),
        },
        examples: Vec::new(), // Would be populated from examples
        dependencies: Vec::new(),
        types: HashMap::new(),
    }
}

/// Create default module
fn create_default_module(spec: &ProblemSpec) -> ModuleSpec {
    ModuleSpec {
        name: spec.name.clone(),
        interface: Interface {
            imports: Vec::new(),
            exports: Vec::new(),
            types: Vec::new(),
        },
        examples: Vec::new(),
        dependencies: Vec::new(),
        types: HashMap::new(),
    }
}

/// Resolve dependencies between modules
fn resolve_dependencies(modules: &mut [ModuleSpec]) {
    // Analyze type references and function calls to determine dependencies
    for i in 0..modules.len() {
        for j in 0..modules.len() {
            if i != j {
                if has_dependency(&modules[i], &modules[j]) {
                    modules[i].dependencies.push(modules[j].name.clone());
                }
            }
        }
    }

    // Remove duplicates
    for module in modules.iter_mut() {
        module.dependencies.sort();
        module.dependencies.dedup();
    }
}

/// Check if module A depends on module B
fn has_dependency(a: &ModuleSpec, b: &ModuleSpec) -> bool {
    // Check if A imports types from B
    for import in &a.interface.imports {
        if import.module == b.name {
            return true;
        }
    }

    // Check if A uses types from B
    for (type_name, type_def) in &a.types {
        if type_def.contains(&format!("{}::", b.name)) {
            return true;
        }
    }

    false
}

/// Create module spec from problem (legacy compatibility)
pub fn problem_to_module(problem: &Problem) -> ModuleSpec {
    ModuleSpec {
        name: problem.name.clone(),
        interface: Interface {
            imports: Vec::new(),
            exports: vec![Export {
                name: problem.function_name().to_string(),
                signature: problem.signature.to_string(),
                visibility: Visibility::Public,
            }],
            types: Vec::new(),
        },
        examples: problem.examples.clone(),
        dependencies: Vec::new(),
        types: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_simple() {
        let spec = ProblemSpec {
            name: "test".to_string(),
            description: "Simple test".to_string(),
            requirements: vec![],
            constraints: vec![],
        };

        let modules = decompose(&spec);
        assert_eq!(modules.len(), 1);
        assert_eq!(modules[0].name, "test");
    }

    #[test]
    fn test_decompose_with_requirements() {
        let spec = ProblemSpec {
            name: "webapp".to_string(),
            description: "Web application".to_string(),
            requirements: vec![
                Requirement {
                    description: "User authentication system".to_string(),
                    priority: Priority::High,
                },
                Requirement {
                    description: "Database storage".to_string(),
                    priority: Priority::High,
                },
            ],
            constraints: vec![],
        };

        let modules = decompose(&spec);
        assert!(modules.len() >= 2);
    }
}
