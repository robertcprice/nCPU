//! API Knowledge Graph for Natural Language → API Mapping
//!
//! Comprehensive knowledge system supporting 100+ libraries across multiple languages.
//! Enables NL queries to discover relevant APIs, find alternatives, and migration paths.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Unique identifier for an API node
pub type API_ID = u64;

/// Programming language enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    JavaScript,
    TypeScript,
    Python,
    Go,
    Java,
    Rust,
    Cpp,
    CSharp,
    Ruby,
    Php,
    Swift,
    Kotlin,
}

impl Language {
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::JavaScript => "javascript",
            Language::TypeScript => "typescript",
            Language::Python => "python",
            Language::Go => "go",
            Language::Java => "java",
            Language::Rust => "rust",
            Language::Cpp => "cpp",
            Language::CSharp => "csharp",
            Language::Ruby => "ruby",
            Language::Php => "php",
            Language::Swift => "swift",
            Language::Kotlin => "kotlin",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "javascript" | "js" => Some(Language::JavaScript),
            "typescript" | "ts" => Some(Language::TypeScript),
            "python" | "py" => Some(Language::Python),
            "go" | "golang" => Some(Language::Go),
            "java" => Some(Language::Java),
            "rust" | "rs" => Some(Language::Rust),
            "cpp" | "c++" => Some(Language::Cpp),
            "csharp" | "c#" => Some(Language::CSharp),
            "ruby" | "rb" => Some(Language::Ruby),
            "php" => Some(Language::Php),
            "swift" => Some(Language::Swift),
            "kotlin" | "kt" => Some(Language::Kotlin),
            _ => None,
        }
    }

    /// Get extension for source files
    pub fn extension(&self) -> &'static str {
        match self {
            Language::JavaScript => "js",
            Language::TypeScript => "ts",
            Language::Python => "py",
            Language::Go => "go",
            Language::Java => "java",
            Language::Rust => "rs",
            Language::Cpp => "cpp",
            Language::CSharp => "cs",
            Language::Ruby => "rb",
            Language::Php => "php",
            Language::Swift => "swift",
            Language::Kotlin => "kt",
        }
    }

    /// Get package manager
    pub fn package_manager(&self) -> &'static str {
        match self {
            Language::JavaScript => "npm",
            Language::TypeScript => "npm",
            Language::Python => "pip",
            Language::Go => "go mod",
            Language::Java => "maven",
            Language::Rust => "cargo",
            Language::Cpp => "conan",
            Language::CSharp => "nuget",
            Language::Ruby => "gem",
            Language::Php => "composer",
            Language::Swift => "swift package-manager",
            Language::Kotlin => "gradle",
        }
    }
}

/// Function signature in an API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APFunction {
    pub name: String,
    pub signature: String,
    pub description: String,
    pub parameters: Vec<FunctionParam>,
    pub return_type: String,
    pub is_async: bool,
    pub is_static: bool,
    pub deprecated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParam {
    pub name: String,
    pub type_name: String,
    pub optional: bool,
    pub default_value: Option<String>,
    pub description: String,
}

/// Usage pattern for an API function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    pub description: String,
    pub code_template: String,
    pub when_to_use: String,
    pub category: PatternCategory,
    pub complexity: Complexity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternCategory {
    DataTransformation,
    AsyncOperation,
    StateManagement,
    Validation,
    ErrorHandling,
    I_O,
    Networking,
    DataStructures,
    Algorithms,
    U_I,
    Testing,
    Logging,
    Security,
    Database,
    Caching,
    Messaging,
    Scheduling,
    Configuration,
    Parsing,
    Encoding,
    CollectionUtilities,
    FunctionalProgramming,
    CliFramework,
    Authentication,
    WebSocket,
    FormHandling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Complexity {
    Simple,
    Medium,
    Complex,
}

/// Alternative library information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alternative {
    pub api_id: API_ID,
    pub reason: String,
    pub migration_difficulty: MigrationDifficulty,
    pub key_differences: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MigrationDifficulty {
    Trivial,
    Easy,
    Moderate,
    Hard,
    VeryHard,
}

/// Migration steps from one API to another
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPath {
    pub from_api: API_ID,
    pub to_api: API_ID,
    pub steps: Vec<MigrationStep>,
    pub breaking_changes: Vec<String>,
    pub estimated_effort: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    pub description: String,
    pub code_example_old: String,
    pub code_example_new: String,
    pub automated: bool,
}

/// API version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIVersion {
    pub version: String,
    pub release_date: String,
    pub status: VersionStatus,
    pub deprecated_date: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VersionStatus {
    Stable,
    Beta,
    Alpha,
    Deprecated,
    Eol,
}

/// Complete API node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APINode {
    pub id: API_ID,
    pub name: String,
    pub language: Language,
    pub versions: Vec<APIVersion>,
    pub current_version: String,
    pub functions: Vec<APFunction>,
    pub patterns: Vec<UsagePattern>,
    pub alternatives: Vec<Alternative>,
    pub migrations: Vec<MigrationPath>,
    pub tags: Vec<String>,
    pub category: APICategory,
    pub repository_url: Option<String>,
    pub documentation_url: Option<String>,
    pub npm_package: Option<String>,
    pub pip_package: Option<String>,
    pub go_module: Option<String>,
    pub maven_artifact: Option<String>,
    pub cargo_crate: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum APICategory {
    // Web Frameworks
    WebFramework,
    HttpServer,
    RestClient,

    // Frontend
    U_I_Library,
    ComponentLibrary,
    StateManagement,
    FormHandling,

    // Backend
    Orm,
    DatabaseDriver,
    QueryBuilder,
    DataValidation,

    // Utilities
    DateTime,
    StringManipulation,
    CollectionUtilities,
    FunctionalProgramming,
    Lodash,

    // Async/Concurrency
    AsyncUtilities,
    PromiseUtilities,
    StreamProcessing,

    // Testing
    TestingFramework,
    MockingFramework,
    AssertionLibrary,

    // DevOps/Tooling
    CliFramework,
    Logging,
    Configuration,
    BuildTool,

    // Security
    Authentication,
    Authorization,
    Encryption,

    // Messaging
    MessageQueue,
    EventBus,
    WebSocket,

    // Caching/Scheduling/Parsing/Security/Encoding
    Caching,
    Scheduling,
    Parsing,
    Security,
    Encoding,

    // Other
    Misc,
}

impl APICategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            APICategory::WebFramework => "web-framework",
            APICategory::HttpServer => "http-server",
            APICategory::RestClient => "rest-client",
            APICategory::U_I_Library => "ui-library",
            APICategory::ComponentLibrary => "component-library",
            APICategory::StateManagement => "state-management",
            APICategory::FormHandling => "form-handling",
            APICategory::Orm => "orm",
            APICategory::DatabaseDriver => "database-driver",
            APICategory::QueryBuilder => "query-builder",
            APICategory::DataValidation => "data-validation",
            APICategory::DateTime => "datetime",
            APICategory::StringManipulation => "string-manipulation",
            APICategory::CollectionUtilities => "collection-utilities",
            APICategory::FunctionalProgramming => "functional-programming",
            APICategory::Lodash => "lodash",
            APICategory::AsyncUtilities => "async-utilities",
            APICategory::PromiseUtilities => "promise-utilities",
            APICategory::StreamProcessing => "stream-processing",
            APICategory::TestingFramework => "testing-framework",
            APICategory::MockingFramework => "mocking-framework",
            APICategory::AssertionLibrary => "assertion-library",
            APICategory::CliFramework => "cli-framework",
            APICategory::Logging => "logging",
            APICategory::Configuration => "configuration",
            APICategory::BuildTool => "build-tool",
            APICategory::Authentication => "authentication",
            APICategory::Authorization => "authorization",
            APICategory::Encryption => "encryption",
            APICategory::MessageQueue => "message-queue",
            APICategory::EventBus => "event-bus",
            APICategory::WebSocket => "websocket",
            APICategory::Caching => "caching",
            APICategory::Scheduling => "scheduling",
            APICategory::Parsing => "parsing",
            APICategory::Security => "security",
            APICategory::Encoding => "encoding",
            APICategory::Misc => "misc",
        }
    }
}

/// Dependency edge between APIs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    pub from: API_ID,
    pub to: API_ID,
    pub edge_type: EdgeType,
    pub strength: EdgeStrength,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    DirectDependency,
    PeerDependency,
    DevDependency,
    Alternative,
    Supersedes,
    InspiredBy,
    ForkOf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EdgeStrength {
    Strong,
    Moderate,
    Weak,
}

/// Main API Graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIGraph {
    /// All API nodes indexed by ID
    pub nodes: HashMap<API_ID, APINode>,
    /// Name to ID index for quick lookup
    pub name_index: HashMap<String, API_ID>,
    /// Language to IDs index
    pub language_index: HashMap<Language, HashSet<API_ID>>,
    /// Category to IDs index
    pub category_index: HashMap<APICategory, HashSet<API_ID>>,
    /// Tag to IDs index
    pub tag_index: HashMap<String, HashSet<API_ID>>,
    /// Dependency edges
    pub edges: Vec<DependencyEdge>,
    /// Function name to API IDs index (multi-map)
    pub function_index: HashMap<String, HashSet<API_ID>>,
}

impl APIGraph {
    /// Create a new empty API graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            name_index: HashMap::new(),
            language_index: HashMap::new(),
            category_index: HashMap::new(),
            tag_index: HashMap::new(),
            edges: Vec::new(),
            function_index: HashMap::new(),
        }
    }

    /// Add an API node to the graph
    pub fn add_node(&mut self, node: APINode) {
        let id = node.id;

        // Index by name
        self.name_index.insert(node.name.clone(), id);

        // Index by language
        self.language_index
            .entry(node.language)
            .or_insert_with(HashSet::new)
            .insert(id);

        // Index by category
        self.category_index
            .entry(node.category)
            .or_insert_with(HashSet::new)
            .insert(id);

        // Index by tags
        for tag in &node.tags {
            self.tag_index
                .entry(tag.clone())
                .or_insert_with(HashSet::new)
                .insert(id);
        }

        // Index functions
        for func in &node.functions {
            self.function_index
                .entry(func.name.clone())
                .or_insert_with(HashSet::new)
                .insert(id);
        }

        self.nodes.insert(id, node);
    }

    /// Add a dependency edge
    pub fn add_edge(&mut self, edge: DependencyEdge) {
        self.edges.push(edge);
    }

    /// Get node by ID
    pub fn get_node(&self, id: API_ID) -> Option<&APINode> {
        self.nodes.get(&id)
    }

    /// Get node by name
    pub fn get_by_name(&self, name: &str) -> Option<&APINode> {
        self.name_index
            .get(name)
            .and_then(|id| self.nodes.get(id))
    }

    /// Find APIs by function name
    pub fn find_by_function(&self, function_name: &str) -> Vec<&APINode> {
        self.function_index
            .get(function_name)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find APIs by language
    pub fn find_by_language(&self, language: Language) -> Vec<&APINode> {
        self.language_index
            .get(&language)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find APIs by category
    pub fn find_by_category(&self, category: APICategory) -> Vec<&APINode> {
        self.category_index
            .get(&category)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find APIs by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<&APINode> {
        self.tag_index
            .get(tag)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find alternatives for an API
    pub fn find_alternatives(&self, api_id: API_ID) -> Vec<&APINode> {
        self.get_node(api_id)
            .map(|node| {
                node.alternatives
                    .iter()
                    .filter_map(|alt| self.nodes.get(&alt.api_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find migration path from one API to another
    pub fn find_migration_path(&self, from: API_ID, to: API_ID) -> Option<&MigrationPath> {
        self.get_node(from).and_then(|node| {
            node.migrations
                .iter()
                .find(|m| m.to_api == to)
        })
    }

    /// Search APIs by query string (matches name, description, tags)
    pub fn search(&self, query: &str) -> Vec<&APINode> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for node in self.nodes.values() {
            // Check name
            if node.name.to_lowercase().contains(&query_lower) {
                results.push(node);
                continue;
            }

            // Check tags
            for tag in &node.tags {
                if tag.to_lowercase().contains(&query_lower) {
                    results.push(node);
                    break;
                }
            }

            // Check function names
            for func in &node.functions {
                if func.name.to_lowercase().contains(&query_lower) ||
                   func.description.to_lowercase().contains(&query_lower) {
                    results.push(node);
                    break;
                }
            }
        }

        results
    }

    /// Get dependencies for an API
    pub fn get_dependencies(&self, api_id: API_ID) -> Vec<&APINode> {
        self.edges
            .iter()
            .filter(|e| e.from == api_id)
            .filter_map(|e| self.nodes.get(&e.to))
            .collect()
    }

    /// Get dependents (APIs that depend on this one)
    pub fn get_dependents(&self, api_id: API_ID) -> Vec<&APINode> {
        self.edges
            .iter()
            .filter(|e| e.to == api_id)
            .filter_map(|e| self.nodes.get(&e.from))
            .collect()
    }

    /// Save graph to JSON file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write file: {}", e))?;
        Ok(())
    }

    /// Load graph from JSON file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize: {}", e))
    }

    /// Get statistics about the graph
    pub fn stats(&self) -> GraphStats {
        GraphStats {
            total_apis: self.nodes.len(),
            total_edges: self.edges.len(),
            languages: self.language_index.len(),
            categories: self.category_index.len(),
            total_functions: self.nodes.values()
                .map(|n| n.functions.len())
                .sum(),
        }
    }
}

impl Default for APIGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the API graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub total_apis: usize,
    pub total_edges: usize,
    pub languages: usize,
    pub categories: usize,
    pub total_functions: usize,
}

/// Thread-safe wrapper for APIGraph
#[derive(Debug, Clone)]
pub struct SharedAPIGraph {
    inner: Arc<RwLock<APIGraph>>,
}

impl SharedAPIGraph {
    /// Create a new shared graph
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(APIGraph::new())),
        }
    }

    /// Create with existing graph
    pub fn with_graph(graph: APIGraph) -> Self {
        Self {
            inner: Arc::new(RwLock::new(graph)),
        }
    }

    /// Read operation with callback
    pub fn read<F, R>(&self, f: F) -> Result<R, String>
    where
        F: FnOnce(&APIGraph) -> R,
    {
        self.inner
            .read()
            .map(|guard| f(&*guard))
            .map_err(|_| "Lock error".to_string())
    }

    /// Write operation with callback
    pub fn write<F, R>(&self, f: F) -> Result<R, String>
    where
        F: FnOnce(&mut APIGraph) -> R,
    {
        self.inner
            .write()
            .map(|mut guard| f(&mut *guard))
            .map_err(|_| "Lock error".to_string())
    }

    /// Load from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let graph = APIGraph::load_from_file(path)?;
        Ok(Self::with_graph(graph))
    }
}

impl Default for SharedAPIGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Natural Language to API query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NLQueryResult {
    pub matched_apis: Vec<API_ID>,
    pub confidence: f64,
    pub matched_function: Option<String>,
    pub suggested_pattern: Option<String>,
    pub reasoning: String,
}

/// Builder for creating API nodes
pub struct APINodeBuilder {
    id: API_ID,
    name: String,
    language: Language,
    category: APICategory,
    versions: Vec<APIVersion>,
    current_version: String,
    functions: Vec<APFunction>,
    patterns: Vec<UsagePattern>,
    alternatives: Vec<Alternative>,
    migrations: Vec<MigrationPath>,
    tags: Vec<String>,
    repository_url: Option<String>,
    documentation_url: Option<String>,
    npm_package: Option<String>,
    pip_package: Option<String>,
    go_module: Option<String>,
    maven_artifact: Option<String>,
    cargo_crate: Option<String>,
}

impl APINodeBuilder {
    pub fn new(id: API_ID, name: String, language: Language, category: APICategory) -> Self {
        Self {
            id,
            name,
            language,
            category,
            versions: Vec::new(),
            current_version: "1.0.0".to_string(),
            functions: Vec::new(),
            patterns: Vec::new(),
            alternatives: Vec::new(),
            migrations: Vec::new(),
            tags: Vec::new(),
            repository_url: None,
            documentation_url: None,
            npm_package: None,
            pip_package: None,
            go_module: None,
            maven_artifact: None,
            cargo_crate: None,
        }
    }

    pub fn version(mut self, version: impl Into<String>) -> Self {
        let v = version.into();
        self.versions.push(APIVersion {
            version: v.clone(),
            release_date: "".to_string(),
            status: VersionStatus::Stable,
            deprecated_date: None,
        });
        self.current_version = v;
        self
    }

    pub fn function(mut self, func: APFunction) -> Self {
        self.functions.push(func);
        self
    }

    pub fn functions(mut self, funcs: Vec<APFunction>) -> Self {
        self.functions = funcs;
        self
    }

    pub fn patterns(mut self, patterns: Vec<UsagePattern>) -> Self {
        self.patterns = patterns;
        self
    }

    pub fn pattern(mut self, pattern: UsagePattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    pub fn alternative(mut self, alt: Alternative) -> Self {
        self.alternatives.push(alt);
        self
    }

    pub fn migration(mut self, migration: MigrationPath) -> Self {
        self.migrations.push(migration);
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn tags(mut self, tags: Vec<impl Into<String>>) -> Self {
        for tag in tags {
            self.tags.push(tag.into());
        }
        self
    }

    pub fn repository(mut self, url: impl Into<String>) -> Self {
        self.repository_url = Some(url.into());
        self
    }

    pub fn documentation(mut self, url: impl Into<String>) -> Self {
        self.documentation_url = Some(url.into());
        self
    }

    pub fn npm(mut self, pkg: impl Into<String>) -> Self {
        self.npm_package = Some(pkg.into());
        self
    }

    pub fn pip(mut self, pkg: impl Into<String>) -> Self {
        self.pip_package = Some(pkg.into());
        self
    }

    pub fn go_module(mut self, module: impl Into<String>) -> Self {
        self.go_module = Some(module.into());
        self
    }

    pub fn maven(mut self, artifact: impl Into<String>) -> Self {
        self.maven_artifact = Some(artifact.into());
        self
    }

    pub fn cargo(mut self, crate_name: impl Into<String>) -> Self {
        self.cargo_crate = Some(crate_name.into());
        self
    }

    pub fn build(self) -> APINode {
        APINode {
            id: self.id,
            name: self.name,
            language: self.language,
            versions: self.versions,
            current_version: self.current_version,
            functions: self.functions,
            patterns: self.patterns,
            alternatives: self.alternatives,
            migrations: self.migrations,
            tags: self.tags,
            category: self.category,
            repository_url: self.repository_url,
            documentation_url: self.documentation_url,
            npm_package: self.npm_package,
            pip_package: self.pip_package,
            go_module: self.go_module,
            maven_artifact: self.maven_artifact,
            cargo_crate: self.cargo_crate,
        }
    }
}

/// Helper to build usage patterns
pub struct UsagePatternBuilder {
    description: String,
    code_template: String,
    when_to_use: String,
    category: PatternCategory,
    complexity: Complexity,
}

impl UsagePatternBuilder {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            code_template: String::new(),
            when_to_use: String::new(),
            category: PatternCategory::DataTransformation,
            complexity: Complexity::Medium,
        }
    }

    pub fn code(mut self, template: impl Into<String>) -> Self {
        self.code_template = template.into();
        self
    }

    pub fn when_to_use(mut self, text: impl Into<String>) -> Self {
        self.when_to_use = text.into();
        self
    }

    pub fn category(mut self, cat: PatternCategory) -> Self {
        self.category = cat;
        self
    }

    pub fn complexity(mut self, comp: Complexity) -> Self {
        self.complexity = comp;
        self
    }

    pub fn build(self) -> UsagePattern {
        UsagePattern {
            description: self.description,
            code_template: self.code_template,
            when_to_use: self.when_to_use,
            category: self.category,
            complexity: self.complexity,
        }
    }
}

/// Pre-populated graph with 100+ libraries
pub fn populate_default_graph() -> APIGraph {
    let mut graph = APIGraph::new();
    let mut next_id: API_ID = 1;

    macro_rules! add_api {
        ($name:expr, $lang:expr, $cat:expr, $versions:expr, $funcs:expr, $patterns:expr, $tags:expr) => {
            {
                let id = next_id;
                next_id += 1;

                let node = APINodeBuilder::new(id, $name.to_string(), $lang, $cat)
                    .version($versions)
                    .functions($funcs)
                    .tags($tags)
                    .patterns($patterns)
                    .build();

                graph.add_node(node);
                id
            }
        };
    }

    // ============================================================
    // JAVASCRIPT/TYPESCRIPT WEB FRAMEWORKS
    // ============================================================

    let react_id = add_api!(
        "react",
        Language::TypeScript,
        APICategory::U_I_Library,
        "18.3.0",
        vec![
            APFunction {
                name: "useState".to_string(),
                signature: "useState<T>(initial: T | (() => T)): [T, Dispatch<SetStateAction<T>>]".to_string(),
                description: "Hook for managing component state".to_string(),
                parameters: vec![],
                return_type: "State pair".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "useEffect".to_string(),
                signature: "useEffect(effect: () => void | (() => void), deps?: DependencyList): void".to_string(),
                description: "Hook for side effects in functional components".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "useMemo".to_string(),
                signature: "useMemo<T>(factory: () => T, deps: DependencyList): T".to_string(),
                description: "Memoized value from expensive computation".to_string(),
                parameters: vec![],
                return_type: "T".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "useCallback".to_string(),
                signature: "useCallback<T>(callback: T, deps: DependencyList): T".to_string(),
                description: "Memoized callback function".to_string(),
                parameters: vec![],
                return_type: "T".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "useContext".to_string(),
                signature: "useContext<T>(context: Context<T>): T".to_string(),
                description: "Read context value".to_string(),
                parameters: vec![],
                return_type: "T".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Component state management".to_string(),
                code_template: "const [state, setState] = useState(initialValue);".to_string(),
                when_to_use: "When component needs local state".to_string(),
                category: PatternCategory::StateManagement,
                complexity: Complexity::Simple,
            },
            UsagePattern {
                description: "Effectful operations".to_string(),
                code_template: "useEffect(() => { /* effect */ }, [deps]);".to_string(),
                when_to_use: "For side effects like data fetching".to_string(),
                category: PatternCategory::DataTransformation,
                complexity: Complexity::Medium,
            },
        ],
        vec!["ui".to_string(), "frontend".to_string(), "components".to_string()]
    );

    let vue_id = add_api!(
        "vue",
        Language::TypeScript,
        APICategory::U_I_Library,
        "3.4.0",
        vec![
            APFunction {
                name: "ref".to_string(),
                signature: "ref<T>(value: T): Ref<UnwrapRef<T>>".to_string(),
                description: "Create a reactive reference".to_string(),
                parameters: vec![],
                return_type: "Ref<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "computed".to_string(),
                signature: "computed<T>(getter: () => T): ComputedRef<T>".to_string(),
                description: "Create a computed value".to_string(),
                parameters: vec![],
                return_type: "ComputedRef<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "watch".to_string(),
                signature: "watch<T>(source: WatchSource<T>, callback: (value: T) => void): WatchStopHandle".to_string(),
                description: "Watch reactive source and run callback".to_string(),
                parameters: vec![],
                return_type: "WatchStopHandle".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "onMounted".to_string(),
                signature: "onMounted(callback: () => void): void".to_string(),
                description: "Hook called after component is mounted".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Reactive state management".to_string(),
                code_template: "const count = ref(0);".to_string(),
                when_to_use: "When component needs reactive state".to_string(),
                category: PatternCategory::StateManagement,
                complexity: Complexity::Simple,
            },
        ],
        vec!["ui".to_string(), "frontend".to_string(), "components".to_string()]
    );

    let svelte_id = add_api!(
        "svelte",
        Language::TypeScript,
        APICategory::U_I_Library,
        "4.2.0",
        vec![
            APFunction {
                name: "writable".to_string(),
                signature: "writable<T>(value: T): Writable<T>".to_string(),
                description: "Create a writable store".to_string(),
                parameters: vec![],
                return_type: "Writable<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "readable".to_string(),
                signature: "readable<T>(value: T): Readable<T>".to_string(),
                description: "Create a readable store".to_string(),
                parameters: vec![],
                return_type: "Readable<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "derived".to_string(),
                signature: "derived<T, U>(store: Store<T>, fn: (value: T) => U): Readable<U>".to_string(),
                description: "Create derived store from existing store".to_string(),
                parameters: vec![],
                return_type: "Readable<U>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Store-based state management".to_string(),
                code_template: "const store = writable(initialValue);".to_string(),
                when_to_use: "For global or shared state".to_string(),
                category: PatternCategory::StateManagement,
                complexity: Complexity::Medium,
            },
        ],
        vec!["ui".to_string(), "frontend".to_string(), "compiler".to_string()]
    );

    let next_id = add_api!(
        "next.js",
        Language::TypeScript,
        APICategory::WebFramework,
        "14.1.0",
        vec![
            APFunction {
                name: "getServerSideProps".to_string(),
                signature: "getServerSideProps(context: GetServerSidePropsContext): Promise<GetServerSidePropsResult>".to_string(),
                description: "Fetch data on each request for server-side rendering".to_string(),
                parameters: vec![],
                return_type: "Promise<GetServerSidePropsResult>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "getStaticProps".to_string(),
                signature: "getStaticProps(context: GetStaticPropsContext): Promise<GetStaticPropsResult>".to_string(),
                description: "Fetch data at build time for static generation".to_string(),
                parameters: vec![],
                return_type: "Promise<GetStaticPropsResult>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "useRouter".to_string(),
                signature: "useRouter(): NextRouter".to_string(),
                description: "Access Next.js router in components".to_string(),
                parameters: vec![],
                return_type: "NextRouter".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Server-side data fetching".to_string(),
                code_template: "export async function getServerSideProps(context) { return { props: { data } }; }".to_string(),
                when_to_use: "For dynamic content requiring server-side rendering".to_string(),
                category: PatternCategory::DataTransformation,
                complexity: Complexity::Medium,
            },
        ],
        vec!["ssr".to_string(), "react".to_string(), "fullstack".to_string()]
    );

    let nuxt_id = add_api!(
        "nuxt",
        Language::TypeScript,
        APICategory::WebFramework,
        "3.10.0",
        vec![
            APFunction {
                name: "useFetch".to_string(),
                signature: "useFetch<T>(url: string): UseFetchReturn<T>".to_string(),
                description: "Fetch data with SSR support".to_string(),
                parameters: vec![],
                return_type: "UseFetchReturn<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "useAsyncData".to_string(),
                signature: "useAsyncData<T>(handler: () => Promise<T>): AsyncDataReturn<T>".to_string(),
                description: "Handle async data with SSR".to_string(),
                parameters: vec![],
                return_type: "AsyncDataReturn<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "useRoute".to_string(),
                signature: "useRoute(): Route".to_string(),
                description: "Access current route information".to_string(),
                parameters: vec![],
                return_type: "Route".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "SSR-compatible data fetching".to_string(),
                code_template: "const { data, error } = await useFetch('/api/data');".to_string(),
                when_to_use: "For data fetching with automatic SSR hydration".to_string(),
                category: PatternCategory::DataTransformation,
                complexity: Complexity::Simple,
            },
        ],
        vec!["ssr".to_string(), "vue".to_string(), "fullstack".to_string()]
    );

    // ============================================================
    // PYTHON WEB FRAMEWORKS
    // ============================================================

    add_api!(
        "fastapi",
        Language::Python,
        APICategory::WebFramework,
        "0.109.0",
        vec![
            APFunction {
                name: "get".to_string(),
                signature: "get(path: str, **kwargs): Decorator".to_string(),
                description: "Decorate a function as a GET endpoint".to_string(),
                parameters: vec![],
                return_type: "Decorator".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "post".to_string(),
                signature: "post(path: str, **kwargs): Decorator".to_string(),
                description: "Decorate a function as a POST endpoint".to_string(),
                parameters: vec![],
                return_type: "Decorator".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "FastAPI".to_string(),
                signature: "FastAPI(**kwargs): FastAPI".to_string(),
                description: "Create a new FastAPI application".to_string(),
                parameters: vec![],
                return_type: "FastAPI".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "REST API endpoint definition".to_string(),
                code_template: "@app.get('/items/{item_id}')\nasync def read_item(item_id: int):\n    return {'item_id': item_id}".to_string(),
                when_to_use: "When creating REST API endpoints".to_string(),
                category: PatternCategory::Networking,
                complexity: Complexity::Simple,
            },
        ],
        vec!["async".to_string(), "rest".to_string(), "modern".to_string()]
    );

    add_api!(
        "django",
        Language::Python,
        APICategory::WebFramework,
        "5.0.0",
        vec![
            APFunction {
                name: "path".to_string(),
                signature: "path(route: str, view: Callable, name: str = None): Path".to_string(),
                description: "Define a URL pattern".to_string(),
                parameters: vec![],
                return_type: "Path".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "Model".to_string(),
                signature: "Model(**kwargs): Model".to_string(),
                description: "Database model base class".to_string(),
                parameters: vec![],
                return_type: "Model".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "MVC pattern with models and views".to_string(),
                code_template: "class Item(models.Model):\n    name = models.CharField()".to_string(),
                when_to_use: "For traditional MVC web applications".to_string(),
                category: PatternCategory::DataStructures,
                complexity: Complexity::Medium,
            },
        ],
        vec!["batteries".to_string(), "orm".to_string(), "mvc".to_string()]
    );

    add_api!(
        "flask",
        Language::Python,
        APICategory::WebFramework,
        "3.0.0",
        vec![
            APFunction {
                name: "route".to_string(),
                signature: "route(rule: str, **options): Decorator".to_string(),
                description: "Decorate a function as a route".to_string(),
                parameters: vec![],
                return_type: "Decorator".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "Flask".to_string(),
                signature: "Flask(import_name: str): Flask".to_string(),
                description: "Create a new Flask application".to_string(),
                parameters: vec![],
                return_type: "Flask".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Microservice routing".to_string(),
                code_template: "@app.route('/')\ndef hello():\n    return 'Hello'".to_string(),
                when_to_use: "For lightweight microservices".to_string(),
                category: PatternCategory::Networking,
                complexity: Complexity::Simple,
            },
        ],
        vec!["micro".to_string(), "lightweight".to_string(), "wsgi".to_string()]
    );

    // ============================================================
    // RUST WEB FRAMEWORKS
    // ============================================================

    add_api!(
        "actix-web",
        Language::Rust,
        APICategory::WebFramework,
        "4.4.0",
        vec![
            APFunction {
                name: "HttpServer".to_string(),
                signature: "HttpServer<T>(app: T): HttpServer<T>".to_string(),
                description: "HTTP server instance".to_string(),
                parameters: vec![],
                return_type: "HttpServer<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "web::get".to_string(),
                signature: "get(handler: Handler): Route".to_string(),
                description: "Create a GET route".to_string(),
                parameters: vec![],
                return_type: "Route".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Async handler routing".to_string(),
                code_template: ".route(\"/\", web::get().to(index))".to_string(),
                when_to_use: "For async HTTP endpoints".to_string(),
                category: PatternCategory::AsyncOperation,
                complexity: Complexity::Medium,
            },
        ],
        vec!["actor".to_string(), "async".to_string(), "performance".to_string()]
    );

    add_api!(
        "axum",
        Language::Rust,
        APICategory::WebFramework,
        "0.7.0",
        vec![
            APFunction {
                name: "Router".to_string(),
                signature: "Router::new(): Router".to_string(),
                description: "Create new router".to_string(),
                parameters: vec![],
                return_type: "Router".to_string(),
                is_async: false,
                is_static: true,
                deprecated: false,
            },
            APFunction {
                name: "get".to_string(),
                signature: "get(path: &str, handler: Handler): MethodRouter".to_string(),
                description: "Create GET route".to_string(),
                parameters: vec![],
                return_type: "MethodRouter".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Tower-based routing".to_string(),
                code_template: ".route(\"/\", get(handler))".to_string(),
                when_to_use: "For composable service architecture".to_string(),
                category: PatternCategory::Networking,
                complexity: Complexity::Medium,
            },
        ],
        vec!["tower".to_string(), "tokio".to_string(), "minimal".to_string()]
    );

    // ============================================================
    // GO WEB FRAMEWORKS
    // ============================================================

    add_api!(
        "gin",
        Language::Go,
        APICategory::WebFramework,
        "1.9.0",
        vec![
            APFunction {
                name: "Default".to_string(),
                signature: "Default() *Engine".to_string(),
                description: "Create default Gin engine".to_string(),
                parameters: vec![],
                return_type: "*Engine".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "GET".to_string(),
                signature: "GET(path string, handler ...HandlerFunc) Routes".to_string(),
                description: "Register GET route".to_string(),
                parameters: vec![],
                return_type: "Routes".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Handler registration".to_string(),
                code_template: "r.GET(\"/ping\", func(c *gin.Context) {})".to_string(),
                when_to_use: "For API route definitions".to_string(),
                category: PatternCategory::Networking,
                complexity: Complexity::Simple,
            },
        ],
        vec!["httprouter".to_string(), "middleware".to_string(), "fast".to_string()]
    );

    add_api!(
        "echo",
        Language::Go,
        APICategory::WebFramework,
        "4.11.0",
        vec![
            APFunction {
                name: "New".to_string(),
                signature: "New() *Echo".to_string(),
                description: "Create new Echo instance".to_string(),
                parameters: vec![],
                return_type: "*Echo".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "GET".to_string(),
                signature: "GET(path string, h HandlerFunc, m ...Middleware) *Route".to_string(),
                description: "Register GET handler".to_string(),
                parameters: vec![],
                return_type: "*Route".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Minimalist routing".to_string(),
                code_template: "e.GET(\"/\", handler)".to_string(),
                when_to_use: "For clean, minimal APIs".to_string(),
                category: PatternCategory::Networking,
                complexity: Complexity::Simple,
            },
        ],
        vec!["minimal".to_string(), "middleware".to_string(), "fast".to_string()]
    );

    // ============================================================
    // JAVA FRAMEWORKS
    // ============================================================

    add_api!(
        "spring-boot",
        Language::Java,
        APICategory::WebFramework,
        "3.2.0",
        vec![
            APFunction {
                name: "RestController".to_string(),
                signature: "@RestController".to_string(),
                description: "Annotation for REST controllers".to_string(),
                parameters: vec![],
                return_type: "Annotation".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "GetMapping".to_string(),
                signature: "@GetMapping(path: String)".to_string(),
                description: "Annotation for GET endpoint".to_string(),
                parameters: vec![],
                return_type: "Annotation".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Annotation-driven REST".to_string(),
                code_template: "@GetMapping(\"/items/{id}\")".to_string(),
                when_to_use: "For enterprise REST APIs".to_string(),
                category: PatternCategory::Networking,
                complexity: Complexity::Medium,
            },
        ],
        vec!["enterprise".to_string(), "spring".to_string(), "ioc".to_string()]
    );

    // ============================================================
    // STATE MANAGEMENT
    // ============================================================

    add_api!(
        "redux",
        Language::TypeScript,
        APICategory::StateManagement,
        "5.0.0",
        vec![
            APFunction {
                name: "createStore".to_string(),
                signature: "createStore<S, A>(reducer: Reducer<S, A>, preloadedState?: S): Store<S, A>".to_string(),
                description: "Create a Redux store".to_string(),
                parameters: vec![],
                return_type: "Store<S, A>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "configureStore".to_string(),
                signature: "configureStore<S>(options: ConfigureStoreOptions<S>): EnhancedStore<S>".to_string(),
                description: "Create store with Redux Toolkit".to_string(),
                parameters: vec![],
                return_type: "EnhancedStore<S>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Centralized state".to_string(),
                code_template: "const store = createStore(reducer);".to_string(),
                when_to_use: "For complex global state".to_string(),
                category: PatternCategory::StateManagement,
                complexity: Complexity::Complex,
            },
        ],
        vec!["flux".to_string(), "predictable".to_string(), "immutable".to_string()]
    );

    add_api!(
        "zustand",
        Language::TypeScript,
        APICategory::StateManagement,
        "4.5.0",
        vec![
            APFunction {
                name: "create".to_string(),
                signature: "create<T>(stateCreator: StateCreator<T>): StoreApi<T>".to_string(),
                description: "Create a Zustand store".to_string(),
                parameters: vec![],
                return_type: "StoreApi<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Simple hook-based state".to_string(),
                code_template: "const useStore = create((set) => ({ count: 0, increment: () => set((state) => ({ count: state.count + 1 })) }));".to_string(),
                when_to_use: "For lightweight global state".to_string(),
                category: PatternCategory::StateManagement,
                complexity: Complexity::Simple,
            },
        ],
        vec!["minimal".to_string(), "hooks".to_string(), "typescript".to_string()]
    );

    add_api!(
        "pinia",
        Language::TypeScript,
        APICategory::StateManagement,
        "2.1.0",
        vec![
            APFunction {
                name: "defineStore".to_string(),
                signature: "defineStore(id: string, setup: () => StateActions): UseStore".to_string(),
                description: "Define a Pinia store".to_string(),
                parameters: vec![],
                return_type: "UseStore".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Vue 3 state management".to_string(),
                code_template: "export const useCounterStore = defineStore('counter', () => { const count = ref(0); return { count }; });".to_string(),
                when_to_use: "For Vue 3 application state".to_string(),
                category: PatternCategory::StateManagement,
                complexity: Complexity::Simple,
            },
        ],
        vec!["vue".to_string(), "composition".to_string(), "typescript".to_string()]
    );

    // ============================================================
    // ORM & DATABASE
    // ============================================================

    add_api!(
        "prisma",
        Language::TypeScript,
        APICategory::Orm,
        "5.9.0",
        vec![
            APFunction {
                name: "findMany".to_string(),
                signature: "findMany(args?: FindManyArgs): Promise<T[]>".to_string(),
                description: "Find many records".to_string(),
                parameters: vec![],
                return_type: "Promise<T[]>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "create".to_string(),
                signature: "create(args: CreateArgs): Promise<T>".to_string(),
                description: "Create a new record".to_string(),
                parameters: vec![],
                return_type: "Promise<T>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "update".to_string(),
                signature: "update(args: UpdateArgs): Promise<T>".to_string(),
                description: "Update a record".to_string(),
                parameters: vec![],
                return_type: "Promise<T>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "delete".to_string(),
                signature: "delete(args: DeleteArgs): Promise<T>".to_string(),
                description: "Delete a record".to_string(),
                parameters: vec![],
                return_type: "Promise<T>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Type-safe database queries".to_string(),
                code_template: "const users = await prisma.user.findMany({ where: { active: true } });".to_string(),
                when_to_use: "For type-safe database operations".to_string(),
                category: PatternCategory::Database,
                complexity: Complexity::Simple,
            },
        ],
        vec!["typescript".to_string(), "type-safe".to_string(), "postgresql".to_string()]
    );

    add_api!(
        "sqlalchemy",
        Language::Python,
        APICategory::Orm,
        "2.0.0",
        vec![
            APFunction {
                name: "create_engine".to_string(),
                signature: "create_engine(url: str, **kw): Engine".to_string(),
                description: "Create a database engine".to_string(),
                parameters: vec![],
                return_type: "Engine".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "Session".to_string(),
                signature: "Session(bind: Engine): Session".to_string(),
                description: "Create a database session".to_string(),
                parameters: vec![],
                return_type: "Session".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "ORM-based database operations".to_string(),
                code_template: "engine = create_engine('sqlite:///:memory:')".to_string(),
                when_to_use: "For Python database abstraction".to_string(),
                category: PatternCategory::Database,
                complexity: Complexity::Medium,
            },
        ],
        vec!["orm".to_string(), "python".to_string(), "sql".to_string()]
    );

    add_api!(
        "diesel",
        Language::Rust,
        APICategory::Orm,
        "2.1.0",
        vec![
            APFunction {
                name: "load".to_string(),
                signature: "load(conn: &mut Conn): QueryResult<Vec<T>>".to_string(),
                description: "Load results from query".to_string(),
                parameters: vec![],
                return_type: "QueryResult<Vec<T>>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Compile-time checked SQL".to_string(),
                code_template: "users.load(&mut connection)?".to_string(),
                when_to_use: "For type-safe database queries".to_string(),
                category: PatternCategory::Database,
                complexity: Complexity::Complex,
            },
        ],
        vec!["rust".to_string(), "type-safe".to_string(), "compile-time".to_string()]
    );

    add_api!(
        "gorm",
        Language::Go,
        APICategory::Orm,
        "1.25.0",
        vec![
            APFunction {
                name: "DB".to_string(),
                signature: "DB(dialector Dialector, config *Config): *DB".to_string(),
                description: "Create GORM DB instance".to_string(),
                parameters: vec![],
                return_type: "*DB".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "Find".to_string(),
                signature: "Find(dest interface{}, conds ...interface{}) *DB".to_string(),
                description: "Find records matching conditions".to_string(),
                parameters: vec![],
                return_type: "*DB".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "ORM for Go".to_string(),
                code_template: "db.Find(&users, \"active = ?\", true)".to_string(),
                when_to_use: "For Go database operations".to_string(),
                category: PatternCategory::Database,
                complexity: Complexity::Medium,
            },
        ],
        vec!["go".to_string(), "orm".to_string(), "postgres".to_string()]
    );

    // ============================================================
    // HTTP CLIENTS
    // ============================================================

    add_api!(
        "axios",
        Language::TypeScript,
        APICategory::RestClient,
        "1.6.0",
        vec![
            APFunction {
                name: "get".to_string(),
                signature: "get<T>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>".to_string(),
                description: "Perform GET request".to_string(),
                parameters: vec![],
                return_type: "Promise<AxiosResponse<T>>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "post".to_string(),
                signature: "post<T>(url: string, data: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>".to_string(),
                description: "Perform POST request".to_string(),
                parameters: vec![],
                return_type: "Promise<AxiosResponse<T>>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "put".to_string(),
                signature: "put<T>(url: string, data: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>".to_string(),
                description: "Perform PUT request".to_string(),
                parameters: vec![],
                return_type: "Promise<AxiosResponse<T>>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "delete".to_string(),
                signature: "delete<T>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>".to_string(),
                description: "Perform DELETE request".to_string(),
                parameters: vec![],
                return_type: "Promise<AxiosResponse<T>>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "HTTP request with automatic JSON".to_string(),
                code_template: "const response = await axios.get('/api/data');".to_string(),
                when_to_use: "For HTTP requests with JSON data".to_string(),
                category: PatternCategory::Networking,
                complexity: Complexity::Simple,
            },
        ],
        vec!["http".to_string(), "ajax".to_string(), "promise".to_string()]
    );

    add_api!(
        "fetch",
        Language::TypeScript,
        APICategory::RestClient,
        "3.0.0",
        vec![
            APFunction {
                name: "fetch".to_string(),
                signature: "fetch(url: string, init?: RequestInit): Promise<Response>".to_string(),
                description: "Fetch resource from network".to_string(),
                parameters: vec![],
                return_type: "Promise<Response>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Browser-native HTTP requests".to_string(),
                code_template: "const response = await fetch('/api/data');\nconst data = await response.json();".to_string(),
                when_to_use: "For simple HTTP requests".to_string(),
                category: PatternCategory::Networking,
                complexity: Complexity::Simple,
            },
        ],
        vec!["browser".to_string(), "native".to_string(), "polyfill".to_string()]
    );

    add_api!(
        "requests",
        Language::Python,
        APICategory::RestClient,
        "2.31.0",
        vec![
            APFunction {
                name: "get".to_string(),
                signature: "get(url: str, **kwargs): Response".to_string(),
                description: "GET request".to_string(),
                parameters: vec![],
                return_type: "Response".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "post".to_string(),
                signature: "post(url: str, data: Dict, **kwargs): Response".to_string(),
                description: "POST request".to_string(),
                parameters: vec![],
                return_type: "Response".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Pythonic HTTP requests".to_string(),
                code_template: "response = requests.get('https://api.example.com')".to_string(),
                when_to_use: "For simple HTTP calls in Python".to_string(),
                category: PatternCategory::Networking,
                complexity: Complexity::Simple,
            },
        ],
        vec!["http".to_string(), "python".to_string(), "rest".to_string()]
    );

    add_api!(
        "httpx",
        Language::Python,
        APICategory::RestClient,
        "0.26.0",
        vec![
            APFunction {
                name: "get".to_string(),
                signature: "get(url: str, **kwargs): Response".to_string(),
                description: "Async GET request".to_string(),
                parameters: vec![],
                return_type: "Response".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "AsyncClient".to_string(),
                signature: "AsyncClient(**kwargs): AsyncClient".to_string(),
                description: "Create async HTTP client".to_string(),
                parameters: vec![],
                return_type: "AsyncClient".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Async HTTP for Python".to_string(),
                code_template: "async with httpx.AsyncClient() as client:\n    response = await client.get(url)".to_string(),
                when_to_use: "For async HTTP requests".to_string(),
                category: PatternCategory::AsyncOperation,
                complexity: Complexity::Medium,
            },
        ],
        vec!["async".to_string(), "http2".to_string(), "modern".to_string()]
    );

    add_api!(
        "reqwest",
        Language::Rust,
        APICategory::RestClient,
        "0.12.0",
        vec![
            APFunction {
                name: "get".to_string(),
                signature: "get(url: &str): Builder".to_string(),
                description: "Create GET request builder".to_string(),
                parameters: vec![],
                return_type: "Builder".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Type-safe HTTP client".to_string(),
                code_template: "let response = reqwest::get(url).await?;".to_string(),
                when_to_use: "For HTTP requests in Rust".to_string(),
                category: PatternCategory::Networking,
                complexity: Complexity::Simple,
            },
        ],
        vec!["rust".to_string(), "async".to_string(), "json".to_string()]
    );

    // ============================================================
    // UTILITY LIBRARIES
    // ============================================================

    add_api!(
        "lodash",
        Language::TypeScript,
        APICategory::Lodash,
        "4.17.21",
        vec![
            APFunction {
                name: "map".to_string(),
                signature: "map(collection: Collection, iteratee: Iteratee): Array".to_string(),
                description: "Map over collection".to_string(),
                parameters: vec![],
                return_type: "Array".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "filter".to_string(),
                signature: "filter(collection: Collection, predicate: Predicate): Array".to_string(),
                description: "Filter collection".to_string(),
                parameters: vec![],
                return_type: "Array".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "reduce".to_string(),
                signature: "reduce(collection: Collection, iteratee: Iteratee, accumulator: any): any".to_string(),
                description: "Reduce collection".to_string(),
                parameters: vec![],
                return_type: "any".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "find".to_string(),
                signature: "find(collection: Collection, predicate: Predicate): any".to_string(),
                description: "Find element in collection".to_string(),
                parameters: vec![],
                return_type: "any".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "sortBy".to_string(),
                signature: "sortBy(collection: Collection, iteratees: Iteratee[]): Array".to_string(),
                description: "Sort collection".to_string(),
                parameters: vec![],
                return_type: "Array".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "groupBy".to_string(),
                signature: "groupBy(collection: Collection, iteratee: Iteratee): Object".to_string(),
                description: "Group collection by key".to_string(),
                parameters: vec![],
                return_type: "Object".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "debounce".to_string(),
                signature: "debounce(func: Function, wait: number): Function".to_string(),
                description: "Debounce function".to_string(),
                parameters: vec![],
                return_type: "Function".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "throttle".to_string(),
                signature: "throttle(func: Function, wait: number): Function".to_string(),
                description: "Throttle function".to_string(),
                parameters: vec![],
                return_type: "Function".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "cloneDeep".to_string(),
                signature: "cloneDeep(value: any): any".to_string(),
                description: "Deep clone value".to_string(),
                parameters: vec![],
                return_type: "any".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "isEmpty".to_string(),
                signature: "isEmpty(value: any): boolean".to_string(),
                description: "Check if empty".to_string(),
                parameters: vec![],
                return_type: "boolean".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Functional array operations".to_string(),
                code_template: "const result = _.map([1, 2, 3], n => n * 2);".to_string(),
                when_to_use: "For collection manipulation".to_string(),
                category: PatternCategory::CollectionUtilities,
                complexity: Complexity::Simple,
            },
        ],
        vec!["utility".to_string(), "functional".to_string(), "collections".to_string()]
    );

    add_api!(
        "ramda",
        Language::TypeScript,
        APICategory::FunctionalProgramming,
        "0.29.0",
        vec![
            APFunction {
                name: "map".to_string(),
                signature: "map(fn: Function, list: List): List".to_string(),
                description: "Curried map".to_string(),
                parameters: vec![],
                return_type: "List".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "pipe".to_string(),
                signature: "pipe(...fns: Function[]): Function".to_string(),
                description: "Function composition".to_string(),
                parameters: vec![],
                return_type: "Function".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "curry".to_string(),
                signature: "curry(fn: Function): Function".to_string(),
                description: "Curry function".to_string(),
                parameters: vec![],
                return_type: "Function".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Functional programming".to_string(),
                code_template: "pipe(map(double), filter(even), take(5))(data)".to_string(),
                when_to_use: "For functional composition".to_string(),
                category: PatternCategory::FunctionalProgramming,
                complexity: Complexity::Complex,
            },
        ],
        vec!["fp".to_string(), "curry".to_string(), "immutable".to_string()]
    );

    add_api!(
        "date-fns",
        Language::TypeScript,
        APICategory::DateTime,
        "3.3.0",
        vec![
            APFunction {
                name: "format".to_string(),
                signature: "format(date: Date, format: string): string".to_string(),
                description: "Format date".to_string(),
                parameters: vec![],
                return_type: "string".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "addDays".to_string(),
                signature: "addDays(date: Date, amount: number): Date".to_string(),
                description: "Add days to date".to_string(),
                parameters: vec![],
                return_type: "Date".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "differenceInDays".to_string(),
                signature: "differenceInDays(dateLeft: Date, dateRight: Date): number".to_string(),
                description: "Days difference".to_string(),
                parameters: vec![],
                return_type: "number".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Date manipulation".to_string(),
                code_template: "format(addDays(new Date(), 7), 'yyyy-MM-dd')".to_string(),
                when_to_use: "For date calculations".to_string(),
                category: PatternCategory::DataTransformation,
                complexity: Complexity::Simple,
            },
        ],
        vec!["date".to_string(), "time".to_string(), "format".to_string()]
    );

    add_api!(
        "dayjs",
        Language::TypeScript,
        APICategory::DateTime,
        "1.11.10",
        vec![
            APFunction {
                name: "dayjs".to_string(),
                signature: "dayjs(date?: DateInput): Dayjs".to_string(),
                description: "Parse date".to_string(),
                parameters: vec![],
                return_type: "Dayjs".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Lightweight date library".to_string(),
                code_template: "dayjs().add(7, 'day').format('YYYY-MM-DD')".to_string(),
                when_to_use: "For simple date operations".to_string(),
                category: PatternCategory::DataTransformation,
                complexity: Complexity::Simple,
            },
        ],
        vec!["date".to_string(), "minimal".to_string(), "moment-compatible".to_string()]
    );

    add_api!(
        "moment",
        Language::TypeScript,
        APICategory::DateTime,
        "2.29.4",
        vec![
            APFunction {
                name: "moment".to_string(),
                signature: "moment(date?: DateInput): Moment".to_string(),
                description: "Create moment".to_string(),
                parameters: vec![],
                return_type: "Moment".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Legacy date handling".to_string(),
                code_template: "moment().add(7, 'days').format('YYYY-MM-DD')".to_string(),
                when_to_use: "Legacy codebases".to_string(),
                category: PatternCategory::DataTransformation,
                complexity: Complexity::Simple,
            },
        ],
        vec!["date".to_string(), "legacy".to_string(), "deprecated".to_string()]
    );

    // ============================================================
    // TESTING
    // ============================================================

    add_api!(
        "jest",
        Language::TypeScript,
        APICategory::TestingFramework,
        "29.7.0",
        vec![
            APFunction {
                name: "describe".to_string(),
                signature: "describe(name: string, fn: () => void): void".to_string(),
                description: "Test suite".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "test".to_string(),
                signature: "test(name: string, fn: TestFn): void".to_string(),
                description: "Test case".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "expect".to_string(),
                signature: "expect(actual: any): Matchers".to_string(),
                description: "Assertion matcher".to_string(),
                parameters: vec![],
                return_type: "Matchers".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Test structure".to_string(),
                code_template: "describe('feature', () => {\n  test('behavior', () => {\n    expect(result).toBe(expected);\n  });\n});".to_string(),
                when_to_use: "For unit and integration tests".to_string(),
                category: PatternCategory::Testing,
                complexity: Complexity::Simple,
            },
        ],
        vec!["testing".to_string(), "snapshot".to_string(), "mock".to_string()]
    );

    add_api!(
        "vitest",
        Language::TypeScript,
        APICategory::TestingFramework,
        "1.2.0",
        vec![
            APFunction {
                name: "describe".to_string(),
                signature: "describe(name: string, fn: () => void): void".to_string(),
                description: "Test suite".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "it".to_string(),
                signature: "it(name: string, fn: TestFn): void".to_string(),
                description: "Test case".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Jest-compatible testing".to_string(),
                code_template: "it('works', () => { expect(1 + 1).toBe(2); });".to_string(),
                when_to_use: "For Vite-based projects".to_string(),
                category: PatternCategory::Testing,
                complexity: Complexity::Simple,
            },
        ],
        vec!["testing".to_string(), "vite".to_string(), "jest-compatible".to_string()]
    );

    add_api!(
        "pytest",
        Language::Python,
        APICategory::TestingFramework,
        "8.0.0",
        vec![
            APFunction {
                name: "fixture".to_string(),
                signature: "@fixture: Decorator".to_string(),
                description: "Test fixture decorator".to_string(),
                parameters: vec![],
                return_type: "Decorator".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Python testing".to_string(),
                code_template: "@pytest.fixture\ndef client():\n    return TestClient(app)".to_string(),
                when_to_use: "For Python test suites".to_string(),
                category: PatternCategory::Testing,
                complexity: Complexity::Simple,
            },
        ],
        vec!["python".to_string(), "testing".to_string(), "fixtures".to_string()]
    );

    // ============================================================
    // LOGGING
    // ============================================================

    add_api!(
        "winston",
        Language::TypeScript,
        APICategory::Logging,
        "3.11.0",
        vec![
            APFunction {
                name: "createLogger".to_string(),
                signature: "createLogger(options: LoggerOptions): Logger".to_string(),
                description: "Create logger".to_string(),
                parameters: vec![],
                return_type: "Logger".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "info".to_string(),
                signature: "info(message: string, meta?: any): void".to_string(),
                description: "Log info".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "error".to_string(),
                signature: "error(message: string, meta?: any): void".to_string(),
                description: "Log error".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Structured logging".to_string(),
                code_template: "logger.info('User logged in', { userId });".to_string(),
                when_to_use: "For production logging".to_string(),
                category: PatternCategory::Logging,
                complexity: Complexity::Simple,
            },
        ],
        vec!["logging".to_string(), "transport".to_string(), "node".to_string()]
    );

    add_api!(
        "pino",
        Language::TypeScript,
        APICategory::Logging,
        "8.19.0",
        vec![
            APFunction {
                name: "pino".to_string(),
                signature: "pino(options?: Options): Logger".to_string(),
                description: "Create Pino logger".to_string(),
                parameters: vec![],
                return_type: "Logger".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "High-performance logging".to_string(),
                code_template: "logger.info({ userId }, 'User action');".to_string(),
                when_to_use: "For high-throughput logging".to_string(),
                category: PatternCategory::Logging,
                complexity: Complexity::Simple,
            },
        ],
        vec!["logging".to_string(), "performance".to_string(), "json".to_string()]
    );

    add_api!(
        "log4rs",
        Language::Rust,
        APICategory::Logging,
        "1.2.0",
        vec![
            APFunction {
                name: "init_config".to_string(),
                signature: "init_config(config: Config): Result<Handle, SetLoggerError>".to_string(),
                description: "Initialize from config".to_string(),
                parameters: vec![],
                return_type: "Result<Handle, SetLoggerError>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Configurable logging".to_string(),
                code_template: "log4rs::init_config_file(config_file)?".to_string(),
                when_to_use: "For flexible Rust logging".to_string(),
                category: PatternCategory::Logging,
                complexity: Complexity::Medium,
            },
        ],
        vec!["rust".to_string(), "logging".to_string(), "config".to_string()]
    );

    // ============================================================
    // VALIDATION
    // ============================================================

    add_api!(
        "zod",
        Language::TypeScript,
        APICategory::DataValidation,
        "3.22.0",
        vec![
            APFunction {
                name: "z".to_string(),
                signature: "z: ZodTypes".to_string(),
                description: "Schema builder".to_string(),
                parameters: vec![],
                return_type: "ZodTypes".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "parse".to_string(),
                signature: "parse(data: unknown): T".to_string(),
                description: "Parse and validate".to_string(),
                parameters: vec![],
                return_type: "T".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Schema validation".to_string(),
                code_template: "const schema = z.object({ name: z.string() });\nconst result = schema.parse(data);".to_string(),
                when_to_use: "For runtime type validation".to_string(),
                category: PatternCategory::Validation,
                complexity: Complexity::Simple,
            },
        ],
        vec!["validation".to_string(), "typescript".to_string(), "schema".to_string()]
    );

    add_api!(
        "yup",
        Language::TypeScript,
        APICategory::DataValidation,
        "1.4.0",
        vec![
            APFunction {
                name: "object".to_string(),
                signature: "object(schema: ObjectSchema): ObjectSchema".to_string(),
                description: "Object schema".to_string(),
                parameters: vec![],
                return_type: "ObjectSchema".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Object validation".to_string(),
                code_template: "await schema.validate(data)".to_string(),
                when_to_use: "For form validation".to_string(),
                category: PatternCategory::Validation,
                complexity: Complexity::Simple,
            },
        ],
        vec!["validation".to_string(), "form".to_string(), "object".to_string()]
    );

    add_api!(
        "pydantic",
        Language::Python,
        APICategory::DataValidation,
        "2.6.0",
        vec![
            APFunction {
                name: "BaseModel".to_string(),
                signature: "class BaseModel".to_string(),
                description: "Base model class".to_string(),
                parameters: vec![],
                return_type: "BaseModel".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Type validation".to_string(),
                code_template: "class User(BaseModel):\n    name: str\n    age: int".to_string(),
                when_to_use: "For Python data validation".to_string(),
                category: PatternCategory::Validation,
                complexity: Complexity::Simple,
            },
        ],
        vec!["python".to_string(), "validation".to_string(), "types".to_string()]
    );

    // ============================================================
    // CONFIGURATION
    // ============================================================

    add_api!(
        "dotenv",
        Language::TypeScript,
        APICategory::Configuration,
        "16.3.0",
        vec![
            APFunction {
                name: "config".to_string(),
                signature: "config(options?: ConfigOptions): void".to_string(),
                description: "Load env vars".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Environment variables".to_string(),
                code_template: "require('dotenv').config()".to_string(),
                when_to_use: "For env-based configuration".to_string(),
                category: PatternCategory::Configuration,
                complexity: Complexity::Simple,
            },
        ],
        vec!["env".to_string(), "config".to_string(), "dotenv".to_string()]
    );

    add_api!(
        "viper",
        Language::Go,
        APICategory::Configuration,
        "1.18.0",
        vec![
            APFunction {
                name: "New".to_string(),
                signature: "New() *Viper".to_string(),
                description: "Create Viper instance".to_string(),
                parameters: vec![],
                return_type: "*Viper".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Go configuration".to_string(),
                code_template: "viper.SetConfigFile(\"config.yaml\")".to_string(),
                when_to_use: "For flexible config in Go".to_string(),
                category: PatternCategory::Configuration,
                complexity: Complexity::Medium,
            },
        ],
        vec!["go".to_string(), "config".to_string(), "yaml".to_string()]
    );

    // ============================================================
    // ASYNC UTILITIES
    // ============================================================

    add_api!(
        "rxjs",
        Language::TypeScript,
        APICategory::AsyncUtilities,
        "7.8.0",
        vec![
            APFunction {
                name: "Observable".to_string(),
                signature: "Observable<T>(subscriber: Subscriber<T>)".to_string(),
                description: "Observable stream".to_string(),
                parameters: vec![],
                return_type: "Observable<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "from".to_string(),
                signature: "from(input: ObservableInput<T>): Observable<T>".to_string(),
                description: "Create observable".to_string(),
                parameters: vec![],
                return_type: "Observable<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "pipe".to_string(),
                signature: "pipe(...operators: OperatorFunction<T, R>[]): OperatorFunction<T, R>".to_string(),
                description: "Compose operators".to_string(),
                parameters: vec![],
                return_type: "OperatorFunction<T, R>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Reactive streams".to_string(),
                code_template: "from([1, 2, 3]).pipe(map(x => x * 2)).subscribe(console.log)".to_string(),
                when_to_use: "For event streams".to_string(),
                category: PatternCategory::AsyncOperation,
                complexity: Complexity::Complex,
            },
        ],
        vec!["reactive".to_string(), "streams".to_string(), "observable".to_string()]
    );

    add_api!(
        "async",
        Language::Python,
        APICategory::AsyncUtilities,
        "3.0.0",
        vec![
            APFunction {
                name: "gather".to_string(),
                signature: "gather(*aws, return_exceptions: bool = False)".to_string(),
                description: "Gather coroutines".to_string(),
                parameters: vec![],
                return_type: "List[Any]".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "create_task".to_string(),
                signature: "create_task(coro: Coroutine): Task".to_string(),
                description: "Create async task".to_string(),
                parameters: vec![],
                return_type: "Task".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Async coordination".to_string(),
                code_template: "results = await gather(task1(), task2())".to_string(),
                when_to_use: "For concurrent async operations".to_string(),
                category: PatternCategory::AsyncOperation,
                complexity: Complexity::Medium,
            },
        ],
        vec!["python".to_string(), "async".to_string(), "await".to_string()]
    );

    // ============================================================
    // CLI FRAMEWORKS
    // ============================================================

    add_api!(
        "commander",
        Language::TypeScript,
        APICategory::CliFramework,
        "12.0.0",
        vec![
            APFunction {
                name: "program".to_string(),
                signature: "program: Command".to_string(),
                description: "Root command".to_string(),
                parameters: vec![],
                return_type: "Command".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "command".to_string(),
                signature: "command(name: string): Command".to_string(),
                description: "Create subcommand".to_string(),
                parameters: vec![],
                return_type: "Command".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "CLI definition".to_string(),
                code_template: "program.command('deploy <env>').action(deploy)".to_string(),
                when_to_use: "For Node.js CLIs".to_string(),
                category: PatternCategory::CliFramework,
                complexity: Complexity::Simple,
            },
        ],
        vec!["cli".to_string(), "command".to_string(), "node".to_string()]
    );

    add_api!(
        "clap",
        Language::Rust,
        APICategory::CliFramework,
        "4.5.0",
        vec![
            APFunction {
                name: "Parser::new".to_string(),
                signature: "Parser::new() -> Parser".to_string(),
                description: "Create CLI parser".to_string(),
                parameters: vec![],
                return_type: "Parser".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Derive-based CLI".to_string(),
                code_template: "#[derive(Parser)]\nstruct Cli { #[arg(short)] verbose: bool }".to_string(),
                when_to_use: "For Rust CLIs".to_string(),
                category: PatternCategory::CliFramework,
                complexity: Complexity::Simple,
            },
        ],
        vec!["rust".to_string(), "cli".to_string(), "derive".to_string()]
    );

    add_api!(
        "cobra",
        Language::Go,
        APICategory::CliFramework,
        "1.8.0",
        vec![
            APFunction {
                name: "Execute".to_string(),
                signature: "Execute() error".to_string(),
                description: "Execute command".to_string(),
                parameters: vec![],
                return_type: "error".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Go CLI framework".to_string(),
                code_template: "var rootCmd = &cobra.Command{ Run: func(cmd *cobra.Command, args []string) {} }".to_string(),
                when_to_use: "For Go command-line apps".to_string(),
                category: PatternCategory::CliFramework,
                complexity: Complexity::Medium,
            },
        ],
        vec!["go".to_string(), "cli".to_string(), "kubernetes".to_string()]
    );

    // ============================================================
    // AUTHENTICATION
    // ============================================================

    add_api!(
        "passport",
        Language::TypeScript,
        APICategory::Authentication,
        "0.7.0",
        vec![
            APFunction {
                name: "use".to_string(),
                signature: "use(strategy: Strategy): void".to_string(),
                description: "Use auth strategy".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Auth strategies".to_string(),
                code_template: "passport.use(new LocalStrategy(verify))".to_string(),
                when_to_use: "For Node.js authentication".to_string(),
                category: PatternCategory::Authentication,
                complexity: Complexity::Medium,
            },
        ],
        vec!["auth".to_string(), "oauth".to_string(), "strategies".to_string()]
    );

    add_api!(
        "auth0",
        Language::TypeScript,
        APICategory::Authentication,
        "4.0.0",
        vec![
            APFunction {
                name: "Auth0Client".to_string(),
                signature: "Auth0Client(options: Auth0ClientOptions): Auth0Client".to_string(),
                description: "Auth0 client".to_string(),
                parameters: vec![],
                return_type: "Auth0Client".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Auth0 integration".to_string(),
                code_template: "const auth0 = await auth0.createClient()".to_string(),
                when_to_use: "For Auth0 authentication".to_string(),
                category: PatternCategory::Authentication,
                complexity: Complexity::Medium,
            },
        ],
        vec!["auth".to_string(), "oauth".to_string(), "sso".to_string()]
    );

    // ============================================================
    // WEBSOCKET
    // ============================================================

    add_api!(
        "socket.io",
        Language::TypeScript,
        APICategory::WebSocket,
        "4.6.0",
        vec![
            APFunction {
                name: "Server".to_string(),
                signature: "Server(httpServer: HTTPServer): Server".to_string(),
                description: "Socket.io server".to_string(),
                parameters: vec![],
                return_type: "Server".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "emit".to_string(),
                signature: "emit(event: string, data: any): void".to_string(),
                description: "Emit event".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "on".to_string(),
                signature: "on(event: string, listener: Function): void".to_string(),
                description: "Listen for event".to_string(),
                parameters: vec![],
                return_type: "void".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Real-time events".to_string(),
                code_template: "io.on('connection', (socket) => {\n  socket.on('message', (data) => {});\n});".to_string(),
                when_to_use: "For real-time communication".to_string(),
                category: PatternCategory::WebSocket,
                complexity: Complexity::Medium,
            },
        ],
        vec!["websocket".to_string(), "realtime".to_string(), "fallback".to_string()]
    );

    add_api!(
        "ws",
        Language::TypeScript,
        APICategory::WebSocket,
        "8.16.0",
        vec![
            APFunction {
                name: "WebSocket".to_string(),
                signature: "WebSocket(url: string): WebSocket".to_string(),
                description: "WebSocket client".to_string(),
                parameters: vec![],
                return_type: "WebSocket".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Native WebSocket".to_string(),
                code_template: "const ws = new WebSocket('ws://localhost')".to_string(),
                when_to_use: "For WebSocket communication".to_string(),
                category: PatternCategory::WebSocket,
                complexity: Complexity::Simple,
            },
        ],
        vec!["websocket".to_string(), "rfc6455".to_string(), "minimal".to_string()]
    );

    // ============================================================
    // QUEUE/MESSAGING
    // ============================================================

    add_api!(
        "bull",
        Language::TypeScript,
        APICategory::MessageQueue,
        "4.12.0",
        vec![
            APFunction {
                name: "Queue".to_string(),
                signature: "Queue(name: string, connection?: Connection): Queue".to_string(),
                description: "Create queue".to_string(),
                parameters: vec![],
                return_type: "Queue".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "add".to_string(),
                signature: "add(job: JobData, opts?: JobOpts): Job".to_string(),
                description: "Add job".to_string(),
                parameters: vec![],
                return_type: "Job".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Redis-backed jobs".to_string(),
                code_template: "queue.add('email', { to: 'user@example.com' })".to_string(),
                when_to_use: "For background job processing".to_string(),
                category: PatternCategory::AsyncOperation,
                complexity: Complexity::Medium,
            },
        ],
        vec!["redis".to_string(), "queue".to_string(), "jobs".to_string()]
    );

    add_api!(
        "celery",
        Language::Python,
        APICategory::MessageQueue,
        "5.3.0",
        vec![
            APFunction {
                name: "Celery".to_string(),
                signature: "Celery(app: str, broker: str): Celery".to_string(),
                description: "Create Celery app".to_string(),
                parameters: vec![],
                return_type: "Celery".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Python task queue".to_string(),
                code_template: "@app.task\ndef send_email(to: str): pass".to_string(),
                when_to_use: "For Python background tasks".to_string(),
                category: PatternCategory::AsyncOperation,
                complexity: Complexity::Medium,
            },
        ],
        vec!["python".to_string(), "queue".to_string(), "worker".to_string()]
    );

    // ============================================================
    // CACHING
    // ============================================================

    add_api!(
        "redis",
        Language::TypeScript,
        APICategory::Caching,
        "4.6.0",
        vec![
            APFunction {
                name: "set".to_string(),
                signature: "set(key: string, value: string): Promise<'OK'>".to_string(),
                description: "Set value".to_string(),
                parameters: vec![],
                return_type: "Promise<'OK'>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "get".to_string(),
                signature: "get(key: string): Promise<string | null>".to_string(),
                description: "Get value".to_string(),
                parameters: vec![],
                return_type: "Promise<string | null>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Key-value caching".to_string(),
                code_template: "await redis.set('key', JSON.stringify(data))".to_string(),
                when_to_use: "For fast key-value storage".to_string(),
                category: PatternCategory::Caching,
                complexity: Complexity::Simple,
            },
        ],
        vec!["redis".to_string(), "cache".to_string(), "key-value".to_string()]
    );

    add_api!(
        "memcached",
        Language::TypeScript,
        APICategory::Caching,
        "1.0.0",
        vec![
            APFunction {
                name: "get".to_string(),
                signature: "get(key: string): Promise<any>".to_string(),
                description: "Get cached value".to_string(),
                parameters: vec![],
                return_type: "Promise<any>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Memcached caching".to_string(),
                code_template: "const value = await memcached.get('key')".to_string(),
                when_to_use: "For distributed caching".to_string(),
                category: PatternCategory::Caching,
                complexity: Complexity::Simple,
            },
        ],
        vec!["memcached".to_string(), "cache".to_string(), "distributed".to_string()]
    );

    // ============================================================
    // SCHEDULING
    // ============================================================

    add_api!(
        "agenda",
        Language::TypeScript,
        APICategory::Scheduling,
        "5.0.0",
        vec![
            APFunction {
                name: "every".to_string(),
                signature: "every(interval: string): Job".to_string(),
                description: "Schedule recurring job".to_string(),
                parameters: vec![],
                return_type: "Job".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Job scheduling".to_string(),
                code_template: "agenda.every('10 minutes').process(sendReminder)".to_string(),
                when_to_use: "For recurring tasks".to_string(),
                category: PatternCategory::Scheduling,
                complexity: Complexity::Medium,
            },
        ],
        vec!["mongodb".to_string(), "cron".to_string(), "jobs".to_string()]
    );

    add_api!(
        "node-cron",
        Language::TypeScript,
        APICategory::Scheduling,
        "3.0.0",
        vec![
            APFunction {
                name: "schedule".to_string(),
                signature: "schedule(cronExpression: string, task: Function): ScheduledTask".to_string(),
                description: "Schedule cron task".to_string(),
                parameters: vec![],
                return_type: "ScheduledTask".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Cron scheduling".to_string(),
                code_template: "cron.schedule('0 * * * *', task)".to_string(),
                when_to_use: "For cron-like scheduling".to_string(),
                category: PatternCategory::Scheduling,
                complexity: Complexity::Simple,
            },
        ],
        vec!["cron".to_string(), "schedule".to_string(), "minimal".to_string()]
    );

    // ============================================================
    // PARSING
    // ============================================================

    add_api!(
        "cheerio",
        Language::TypeScript,
        APICategory::Parsing,
        "1.0.0-rc.12",
        vec![
            APFunction {
                name: "load".to_string(),
                signature: "load(html: string): CheerioAPI".to_string(),
                description: "Load HTML".to_string(),
                parameters: vec![],
                return_type: "CheerioAPI".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "HTML parsing".to_string(),
                code_template: "const $ = cheerio.load(html); $('a').each(...)".to_string(),
                when_to_use: "For HTML scraping".to_string(),
                category: PatternCategory::Parsing,
                complexity: Complexity::Simple,
            },
        ],
        vec!["html".to_string(), "scraping".to_string(), "jquery".to_string()]
    );

    add_api!(
        "js-yaml",
        Language::TypeScript,
        APICategory::Parsing,
        "4.1.0",
        vec![
            APFunction {
                name: "load".to_string(),
                signature: "load(text: string): any".to_string(),
                description: "Parse YAML".to_string(),
                parameters: vec![],
                return_type: "any".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "YAML parsing".to_string(),
                code_template: "const config = yaml.load(fs.readFileSync('config.yaml'))".to_string(),
                when_to_use: "For YAML files".to_string(),
                category: PatternCategory::Parsing,
                complexity: Complexity::Simple,
            },
        ],
        vec!["yaml".to_string(), "config".to_string(), "parse".to_string()]
    );

    // ============================================================
    // SECURITY
    // ============================================================

    add_api!(
        "bcrypt",
        Language::TypeScript,
        APICategory::Security,
        "5.1.0",
        vec![
            APFunction {
                name: "hash".to_string(),
                signature: "hash(data: string, rounds: number): Promise<string>".to_string(),
                description: "Hash password".to_string(),
                parameters: vec![],
                return_type: "Promise<string>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "compare".to_string(),
                signature: "compare(data: string, encrypted: string): Promise<boolean>".to_string(),
                description: "Compare password".to_string(),
                parameters: vec![],
                return_type: "Promise<boolean>".to_string(),
                is_async: true,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Password hashing".to_string(),
                code_template: "const hash = await bcrypt.hash(password, 10)".to_string(),
                when_to_use: "For secure password storage".to_string(),
                category: PatternCategory::Security,
                complexity: Complexity::Simple,
            },
        ],
        vec!["security".to_string(), "password".to_string(), "hash".to_string()]
    );

    add_api!(
        "jsonwebtoken",
        Language::TypeScript,
        APICategory::Security,
        "9.0.0",
        vec![
            APFunction {
                name: "sign".to_string(),
                signature: "sign(payload: object, secret: string): string".to_string(),
                description: "Sign JWT".to_string(),
                parameters: vec![],
                return_type: "string".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
            APFunction {
                name: "verify".to_string(),
                signature: "verify(token: string, secret: string): object".to_string(),
                description: "Verify JWT".to_string(),
                parameters: vec![],
                return_type: "object".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "JWT tokens".to_string(),
                code_template: "const token = jwt.sign({ userId }, secret)".to_string(),
                when_to_use: "For JWT authentication".to_string(),
                category: PatternCategory::Security,
                complexity: Complexity::Simple,
            },
        ],
        vec!["jwt".to_string(), "auth".to_string(), "token".to_string()]
    );

    // ============================================================
    // ENCODING
    // ============================================================

    add_api!(
        "base64-js",
        Language::TypeScript,
        APICategory::Encoding,
        "1.5.0",
        vec![
            APFunction {
                name: "fromByteArray".to_string(),
                signature: "fromByteArray(bytes: Uint8Array): string".to_string(),
                description: "Encode base64".to_string(),
                parameters: vec![],
                return_type: "string".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Base64 encoding".to_string(),
                code_template: "const encoded = base64.fromByteArray(bytes)".to_string(),
                when_to_use: "For base64 encoding".to_string(),
                category: PatternCategory::Encoding,
                complexity: Complexity::Simple,
            },
        ],
        vec!["base64".to_string(), "encoding".to_string(), "binary".to_string()]
    );

    // ============================================================
    // FORM HANDLING
    // ============================================================

    add_api!(
        "react-hook-form",
        Language::TypeScript,
        APICategory::FormHandling,
        "7.49.0",
        vec![
            APFunction {
                name: "useForm".to_string(),
                signature: "useForm<T>(): UseFormReturn<T>".to_string(),
                description: "Form management hook".to_string(),
                parameters: vec![],
                return_type: "UseFormReturn<T>".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Form state management".to_string(),
                code_template: "const { register, handleSubmit } = useForm()".to_string(),
                when_to_use: "For React form handling".to_string(),
                category: PatternCategory::FormHandling,
                complexity: Complexity::Simple,
            },
        ],
        vec!["react".to_string(), "forms".to_string(), "validation".to_string()]
    );

    add_api!(
        "formik",
        Language::TypeScript,
        APICategory::FormHandling,
        "2.4.0",
        vec![
            APFunction {
                name: "Formik".to_string(),
                signature: "Formik<Props>(props: Props): ReactElement".to_string(),
                description: "Formik component".to_string(),
                parameters: vec![],
                return_type: "ReactElement".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "Form component".to_string(),
                code_template: "<Formik initialValues={{}} onSubmit={handleSubmit}>".to_string(),
                when_to_use: "For structured form handling".to_string(),
                category: PatternCategory::FormHandling,
                complexity: Complexity::Medium,
            },
        ],
        vec!["react".to_string(), "forms".to_string(), "component".to_string()]
    );

    // ============================================================
    // DATABASE DRIVERS
    // ============================================================

    add_api!(
        "pg",
        Language::TypeScript,
        APICategory::DatabaseDriver,
        "8.11.0",
        vec![
            APFunction {
                name: "Client".to_string(),
                signature: "Client(config: ClientConfig): Client".to_string(),
                description: "PostgreSQL client".to_string(),
                parameters: vec![],
                return_type: "Client".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "PostgreSQL connection".to_string(),
                code_template: "const client = new Client({ connectionString })".to_string(),
                when_to_use: "For PostgreSQL access".to_string(),
                category: PatternCategory::Database,
                complexity: Complexity::Medium,
            },
        ],
        vec!["postgresql".to_string(), "postgres".to_string(), "sql".to_string()]
    );

    add_api!(
        "mysql2",
        Language::TypeScript,
        APICategory::DatabaseDriver,
        "3.6.0",
        vec![
            APFunction {
                name: "createConnection".to_string(),
                signature: "createConnection(config: ConnectionConfig): Connection".to_string(),
                description: "MySQL connection".to_string(),
                parameters: vec![],
                return_type: "Connection".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "MySQL connection".to_string(),
                code_template: "const conn = mysql.createConnection({ host, user })".to_string(),
                when_to_use: "For MySQL/MariaDB".to_string(),
                category: PatternCategory::Database,
                complexity: Complexity::Medium,
            },
        ],
        vec!["mysql".to_string(), "mariadb".to_string(), "sql".to_string()]
    );

    add_api!(
        "mongodb",
        Language::TypeScript,
        APICategory::DatabaseDriver,
        "6.3.0",
        vec![
            APFunction {
                name: "MongoClient".to_string(),
                signature: "MongoClient(uri: string): MongoClient".to_string(),
                description: "MongoDB client".to_string(),
                parameters: vec![],
                return_type: "MongoClient".to_string(),
                is_async: false,
                is_static: false,
                deprecated: false,
            },
        ],
        vec![
            UsagePattern {
                description: "MongoDB connection".to_string(),
                code_template: "const client = new MongoClient('mongodb://localhost')".to_string(),
                when_to_use: "For MongoDB".to_string(),
                category: PatternCategory::Database,
                complexity: Complexity::Medium,
            },
        ],
        vec!["mongodb".to_string(), "mongo".to_string(), "nosql".to_string()]
    );

    // ============================================================
    // ADD NODES AND EDGES FOR ALTERNATIVES
    // ============================================================

    // React alternatives
    graph.add_edge(DependencyEdge {
        from: react_id,
        to: vue_id,
        edge_type: EdgeType::Alternative,
        strength: EdgeStrength::Moderate,
    });
    graph.add_edge(DependencyEdge {
        from: react_id,
        to: svelte_id,
        edge_type: EdgeType::Alternative,
        strength: EdgeStrength::Moderate,
    });

    // Next.js alternatives
    graph.add_edge(DependencyEdge {
        from: next_id,
        to: nuxt_id,
        edge_type: EdgeType::Alternative,
        strength: EdgeStrength::Moderate,
    });

    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = APIGraph::new();
        assert_eq!(graph.nodes.len(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut graph = APIGraph::new();
        let node = APINodeBuilder::new(
            1,
            "test-api".to_string(),
            Language::TypeScript,
            APICategory::WebFramework,
        )
        .version("1.0.0")
        .tag("test")
        .build();

        graph.add_node(node);
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.get_by_name("test-api").is_some());
    }

    #[test]
    fn test_find_by_function() {
        let mut graph = populate_default_graph();
        let results = graph.find_by_function("useState");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_find_by_language() {
        let graph = populate_default_graph();
        let ts_apis = graph.find_by_language(Language::TypeScript);
        assert!(!ts_apis.is_empty());
    }

    #[test]
    fn test_search() {
        let graph = populate_default_graph();
        let results = graph.search("react");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_serialization() {
        let graph = populate_default_graph();
        let json = serde_json::to_string(&graph);
        assert!(json.is_ok());

        let deserialized: Result<APIGraph, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_shared_graph() {
        let shared = SharedAPIGraph::new();
        let result = shared.read(|graph| graph.nodes.len());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }
}
