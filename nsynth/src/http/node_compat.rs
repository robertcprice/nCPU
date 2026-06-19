//! Node.js Compatibility for nCPU/nSynth
//!
//! Node.js runtime APIs, CommonJS modules, and npm integration.

use serde_json::Value;
use std::collections::HashMap;

/// Node.js runtime API
#[derive(Debug, Clone)]
pub enum NodeApi {
    /// fs - file system
    Fs { operation: FsOperation },
    /// http - HTTP server/client
    Http { operation: HttpOperation },
    /// path - file paths
    Path { operation: PathOperation },
    /// events - EventEmitter
    Events { operation: EventOperation },
    /// stream - streams
    Stream { operation: StreamOperation },
    /// buffer - Buffer
    Buffer { operation: BufferOperation },
    /// crypto - cryptography
    Crypto { operation: CryptoOperation },
    /// process - process info
    Process { field: ProcessField },
    /// module - module system
    Module { operation: ModuleOperation },
    /// setTimeout/setInterval
    Timer { duration: u32, repeat: bool },
}

/// File system operations
#[derive(Debug, Clone)]
pub enum FsOperation {
    ReadFile {
        path: String,
        encoding: Option<String>,
    },
    WriteFile {
        path: String,
        data: String,
        encoding: Option<String>,
    },
    Unlink {
        path: String,
    },
    Mkdir {
        path: String,
        recursive: bool,
    },
    Readdir {
        path: String,
    },
    Stat {
        path: String,
    },
    Exists {
        path: String,
    },
}

/// HTTP operations
#[derive(Debug, Clone)]
pub enum HttpOperation {
    CreateServer { port: u16 },
    Request { url: String, method: String },
}

/// Path operations
#[derive(Debug, Clone)]
pub enum PathOperation {
    Join { paths: Vec<String> },
    Basename { path: String },
    Dirname { path: String },
    Resolve { path: String },
    Normalize { path: String },
}

/// Event operations
#[derive(Debug, Clone)]
pub enum EventOperation {
    On { event: String, handler: String },
    Emit { event: String, data: Value },
    Once { event: String, handler: String },
    RemoveListener { event: String, handler: String },
}

/// Stream operations
#[derive(Debug, Clone)]
pub enum StreamOperation {
    Read { chunk_size: usize },
    Write { data: String },
    Pipe { destination: String },
    OnData { handler: String },
    OnEnd { handler: String },
}

/// Buffer operations
#[derive(Debug, Clone)]
pub enum BufferOperation {
    From { data: String, encoding: String },
    Alloc { size: usize },
    Concat { buffers: Vec<String> },
    ToString { buffer: String, encoding: String },
}

/// Crypto operations
#[derive(Debug, Clone)]
pub enum CryptoOperation {
    Hash {
        algorithm: String,
        data: String,
    },
    Hmac {
        algorithm: String,
        key: String,
        data: String,
    },
    RandomBytes {
        size: usize,
    },
    Cipher {
        algorithm: String,
        key: String,
        iv: String,
        data: String,
    },
}

/// Process fields
#[derive(Debug, Clone)]
pub enum ProcessField {
    Argv,
    Env,
    Cwd,
    Platform,
    Version,
    Uptime,
    MemoryUsage,
}

/// Module operations
#[derive(Debug, Clone)]
pub enum ModuleOperation {
    Require { module: String },
    Export { name: String, value: Value },
    Imports { source: String },
}

/// CommonJS module
#[derive(Debug, Clone)]
pub struct CommonJsModule {
    /// Module name/path
    pub name: String,
    /// Module exports
    pub exports: HashMap<String, Value>,
    /// Required dependencies
    pub dependencies: Vec<String>,
    /// Module source code
    pub source: String,
}

impl CommonJsModule {
    /// Create new module
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            exports: HashMap::new(),
            dependencies: Vec::new(),
            source: String::new(),
        }
    }

    /// Add export
    pub fn export(mut self, name: impl Into<String>, value: Value) -> Self {
        self.exports.insert(name.into(), value);
        self
    }

    /// Add dependency
    pub fn require(mut self, module: impl Into<String>) -> Self {
        self.dependencies.push(module.into());
        self
    }

    /// Set source code
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Generate CommonJS code
    pub fn to_commonjs(&self) -> String {
        let mut code = String::new();

        // Requires
        for dep in &self.dependencies {
            code.push_str(&format!("const {} = require('{}');\n", dep, dep));
        }

        // Exports
        if self.exports.len() == 1 {
            let (name, _) = self.exports.iter().next().unwrap();
            code.push_str(&format!("module.exports = {};\n", name));
        } else {
            code.push_str("module.exports = {\n");
            for (name, value) in &self.exports {
                code.push_str(&format!(
                    "  {}, {},\n",
                    name,
                    serde_json::to_string(value).unwrap_or_default()
                ));
            }
            code.push_str("};\n");
        }

        code
    }

    /// Generate ES module (import/export)
    pub fn to_es_module(&self) -> String {
        let mut code = String::new();

        // Imports
        for dep in &self.dependencies {
            code.push_str(&format!("import {} from '{}';\n", dep, dep));
        }

        // Exports
        for (name, _) in &self.exports {
            code.push_str(&format!("export const {} = /* value */;\n", name));
        }

        code
    }
}

/// NPM package
#[derive(Debug, Clone)]
pub struct NpmPackage {
    /// Package name
    pub name: String,
    /// Version
    pub version: String,
    /// Dependencies
    pub dependencies: Vec<(String, String)>,
    /// Main entry point
    pub main: String,
    /// Scripts
    pub scripts: HashMap<String, String>,
}

impl NpmPackage {
    /// Create new package
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            dependencies: Vec::new(),
            main: "index.js".to_string(),
            scripts: HashMap::new(),
        }
    }

    /// Add dependency
    pub fn dependency(mut self, package: impl Into<String>, version: impl Into<String>) -> Self {
        self.dependencies.push((package.into(), version.into()));
        self
    }

    /// Add script
    pub fn script(mut self, name: impl Into<String>, command: impl Into<String>) -> Self {
        self.scripts.insert(name.into(), command.into());
        self
    }

    /// Set main entry point
    pub fn with_main(mut self, main: impl Into<String>) -> Self {
        self.main = main.into();
        self
    }

    /// Generate package.json
    pub fn to_package_json(&self) -> String {
        let mut json = String::new();

        json.push_str("{\n");
        json.push_str(&format!("  \"name\": \"{}\",\n", self.name));
        json.push_str(&format!("  \"version\": \"{}\",\n", self.version));
        json.push_str(&format!("  \"main\": \"{}\",\n", self.main));

        if !self.dependencies.is_empty() {
            json.push_str("  \"dependencies\": {\n");
            for (i, (pkg, ver)) in self.dependencies.iter().enumerate() {
                let comma = if i < self.dependencies.len() - 1 {
                    ","
                } else {
                    ""
                };
                json.push_str(&format!("    \"{}\": \"{}\"{}\n", pkg, ver, comma));
            }
            json.push_str("  },\n");
        }

        if !self.scripts.is_empty() {
            json.push_str("  \"scripts\": {\n");
            for (i, (name, cmd)) in self.scripts.iter().enumerate() {
                let comma = if i < self.scripts.len() - 1 { "," } else { "" };
                json.push_str(&format!("    \"{}\": \"{}\"{}\n", name, cmd, comma));
            }
            json.push_str("  }\n");
        }

        json.push_str("}\n");
        json
    }
}

/// Express.js route
#[derive(Debug, Clone)]
pub struct ExpressRoute {
    /// HTTP method
    pub method: String,
    /// Path pattern
    pub path: String,
    /// Handler function body
    pub handler: String,
    /// Middleware
    pub middleware: Vec<String>,
}

impl ExpressRoute {
    /// Create new route
    pub fn new(
        method: impl Into<String>,
        path: impl Into<String>,
        handler: impl Into<String>,
    ) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
            handler: handler.into(),
            middleware: Vec::new(),
        }
    }

    /// Add middleware
    pub fn use_(mut self, middleware: impl Into<String>) -> Self {
        self.middleware.push(middleware.into());
        self
    }

    /// Generate Express code
    pub fn to_express(&self) -> String {
        let mut code = String::new();

        // Middleware
        for mw in &self.middleware {
            code.push_str(&format!("app.use({});\n", mw));
        }

        // Route
        code.push_str(&format!(
            "app.{}('{}', (req, res) => {{\n  {}\n}});\n",
            self.method.to_lowercase(),
            self.path,
            self.handler
        ));

        code
    }
}

/// Node.js script generator
#[derive(Debug, Clone)]
pub struct NodeScript {
    /// Script content
    pub content: String,
    /// Shebang
    pub shebang: Option<String>,
    /// Strict mode
    pub strict: bool,
}

impl NodeScript {
    /// Create new script
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            shebang: Some("#!/usr/bin/env node".to_string()),
            strict: true,
        }
    }

    /// Generate complete script
    pub fn to_script(&self) -> String {
        let mut code = String::new();

        // Shebang
        if let Some(ref shebang) = self.shebang {
            code.push_str(shebang);
            code.push('\n');
        }

        // Strict mode
        if self.strict {
            code.push_str("'use strict';\n\n");
        }

        code.push_str(&self.content);

        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commonjs_module() {
        let module = CommonJsModule::new("myModule")
            .export("add", Value::from("function"))
            .require("lodash");

        let code = module.to_commonjs();
        assert!(code.contains("require"));
        assert!(code.contains("module.exports"));
    }

    #[test]
    fn test_npm_package() {
        let pkg = NpmPackage::new("my-app", "1.0.0")
            .dependency("express", "4.18.0")
            .script("start", "node index.js");

        let json = pkg.to_package_json();
        assert!(json.contains("\"name\": \"my-app\""));
        assert!(json.contains("\"express\": \"4.18.0\""));
    }

    #[test]
    fn test_express_route() {
        let route = ExpressRoute::new("GET", "/users", "return res.json(users);");
        let code = route.to_express();
        assert!(code.contains("app.get"));
        assert!(code.contains("/users"));
    }

    #[test]
    fn test_node_script() {
        let script = NodeScript::new("console.log('Hello');");
        let code = script.to_script();
        assert!(code.contains("#!/usr/bin/env node"));
        assert!(code.contains("'use strict'"));
    }
}
