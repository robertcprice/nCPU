//! React/Next.js Framework Support for nCPU/nSynth
//!
//! React component generation and Next.js-style routing.

use serde_json::Value;
use std::collections::HashMap;

/// React component
#[derive(Debug, Clone)]
pub struct Component {
    /// Component name
    pub name: String,
    /// Props
    pub props: Vec<Prop>,
    /// State
    pub state: Vec<StateVar>,
    /// Children
    pub children: Vec<Node>,
    /// Hooks
    pub hooks: Vec<Hook>,
}

/// React prop
#[derive(Debug, Clone)]
pub struct Prop {
    /// Prop name
    pub name: String,
    /// Prop type
    pub prop_type: PropType,
    /// Optional
    pub optional: bool,
    /// Default value
    pub default: Option<Value>,
}

/// Prop type
#[derive(Debug, Clone)]
pub enum PropType {
    String,
    Number,
    Boolean,
    Array(Box<PropType>),
    Object(Vec<(String, PropType)>),
    Custom(String),
}

/// React state variable
#[derive(Debug, Clone)]
pub struct StateVar {
    /// Variable name
    pub name: String,
    /// Type
    pub var_type: PropType,
    /// Initial value
    pub initial: Option<Value>,
}

/// React node
#[derive(Debug, Clone)]
pub enum Node {
    Element {
        tag: String,
        props: HashMap<String, Value>,
        children: Vec<Node>,
    },
    Component {
        name: String,
        props: HashMap<String, Value>,
    },
    Text(String),
    Fragment(Vec<Node>),
}

/// React hook
#[derive(Debug, Clone)]
pub enum Hook {
    /// useState hook
    UseState { var: String, initial: Value },
    /// useEffect hook
    UseEffect { deps: Vec<String>, body: String },
    /// useContext hook
    UseContext { context: String },
    /// useMemo hook
    UseMemo { deps: Vec<String>, compute: String },
    /// useCallback hook
    UseCallback { deps: Vec<String>, func: String },
    /// useRef hook
    UseRef { initial: Option<Value> },
    /// Custom hook
    Custom { name: String, args: Vec<String> },
}

impl Component {
    /// Create new component
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            props: Vec::new(),
            state: Vec::new(),
            children: Vec::new(),
            hooks: Vec::new(),
        }
    }

    /// Add prop
    pub fn prop(mut self, name: impl Into<String>, prop_type: PropType) -> Self {
        self.props.push(Prop {
            name: name.into(),
            prop_type,
            optional: false,
            default: None,
        });
        self
    }

    /// Add optional prop with default
    pub fn prop_opt(
        mut self,
        name: impl Into<String>,
        prop_type: PropType,
        default: Value,
    ) -> Self {
        self.props.push(Prop {
            name: name.into(),
            prop_type,
            optional: true,
            default: Some(default),
        });
        self
    }

    /// Add state variable
    pub fn state(mut self, name: impl Into<String>, var_type: PropType, initial: Value) -> Self {
        self.state.push(StateVar {
            name: name.into(),
            var_type,
            initial: Some(initial),
        });
        self
    }

    /// Add hook
    pub fn hook(mut self, hook: Hook) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Generate React code
    pub fn to_react(&self) -> String {
        let mut code = String::new();

        // Component signature
        code.push_str(&format!(
            "function {}({} ",
            self.name,
            self.props
                .iter()
                .map(|p| format!("{{{}}}", p.name))
                .collect::<Vec<_>>()
                .join(", ")
        ));

        // Props destructuring with defaults
        if self.props.iter().any(|p| p.optional) {
            code.push_str(" {\n");
            for prop in &self.props {
                if prop.optional {
                    if let Some(default) = &prop.default {
                        code.push_str(&format!(
                            "  {} = {},\n",
                            prop.name,
                            serde_json::to_string(default).unwrap_or_default()
                        ));
                    }
                }
            }
            code.push_str("}");
        }
        code.push_str(") {\n");

        // Hooks
        for hook in &self.hooks {
            code.push_str("  ");
            match hook {
                Hook::UseState { var, initial } => {
                    code.push_str(&format!(
                        "const [{}, set{}] = useState({});\n",
                        var,
                        var,
                        serde_json::to_string(initial).unwrap_or_default()
                    ));
                }
                Hook::UseEffect { deps, body } => {
                    code.push_str(&format!(
                        "useEffect(() => {{ {} }}, [{}]);\n",
                        body,
                        deps.join(", ")
                    ));
                }
                Hook::UseMemo { deps, compute } => {
                    code.push_str(&format!(
                        "const memo = useMemo(() => {{ {} }}, [{}]);\n",
                        compute,
                        deps.join(", ")
                    ));
                }
                _ => {
                    code.push_str("// hook\n");
                }
            }
        }

        // Render
        code.push_str("  return (\n");
        code.push_str("    <>\n");
        for child in &self.children {
            code.push_str(&self.render_node(child, 6));
        }
        code.push_str("    </>\n");
        code.push_str("  );\n");
        code.push_str("}\n");

        // Export
        code.push_str(&format!("export default {};\n", self.name));

        code
    }

    /// Render node to JSX
    fn render_node(&self, node: &Node, indent: usize) -> String {
        let spaces = " ".repeat(indent);
        match node {
            Node::Element {
                tag,
                props,
                children,
            } => {
                let mut result = format!("{}<{}", spaces, tag);
                for (k, v) in props {
                    result.push_str(&format!(
                        " {}={}",
                        k,
                        serde_json::to_string(v).unwrap_or_default()
                    ));
                }
                result.push_str(">\n");
                for child in children {
                    result.push_str(&self.render_node(child, indent + 2));
                }
                result.push_str(&format!("{}</{}>\n", spaces, tag));
                result
            }
            Node::Component { name, props } => {
                let mut result = format!("{}<{}", spaces, name);
                for (k, v) in props {
                    result.push_str(&format!(
                        " {}={}",
                        k,
                        serde_json::to_string(v).unwrap_or_default()
                    ));
                }
                result.push_str(" />\n");
                result
            }
            Node::Text(text) => {
                format!("{}{}\n", spaces, text)
            }
            Node::Fragment(nodes) => {
                let mut result = format!("{}<>\n", spaces);
                for node in nodes {
                    result.push_str(&self.render_node(node, indent + 2));
                }
                result.push_str("{}</>\n");
                result
            }
        }
    }
}

/// Next.js page route
#[derive(Debug, Clone)]
pub struct PageRoute {
    /// Route path (e.g., "/users/[id]")
    pub path: String,
    /// Page component
    pub component: Component,
    /// Static generation vs server-side rendering
    pub render_strategy: RenderStrategy,
    /// GetStaticProps or getServerSideProps
    pub data_fetch: Option<DataFetch>,
}

/// Render strategy
#[derive(Debug, Clone, Copy)]
pub enum RenderStrategy {
    /// Static generation (build-time)
    Static,
    /// Server-side rendering (per-request)
    Server,
    /// Client-side rendering
    Client,
}

/// Data fetching method
#[derive(Debug, Clone)]
pub enum DataFetch {
    /// getStaticProps
    Static { props: String },
    /// getServerSideProps
    Server { props: String },
    /// getInitialProps (legacy)
    Initial { props: String },
}

impl PageRoute {
    /// Create new page route
    pub fn new(path: impl Into<String>, component: Component) -> Self {
        Self {
            path: path.into(),
            component,
            render_strategy: RenderStrategy::Static,
            data_fetch: None,
        }
    }

    /// Set render strategy
    pub fn with_strategy(mut self, strategy: RenderStrategy) -> Self {
        self.render_strategy = strategy;
        self
    }

    /// Add data fetching
    pub fn with_data_fetch(mut self, fetch: DataFetch) -> Self {
        self.data_fetch = Some(fetch);
        self
    }

    /// Generate Next.js page code
    pub fn to_nextjs(&self) -> String {
        let mut code = String::new();

        // Imports
        code.push_str("import React from 'react';\n\n");

        // Data fetching function
        if let Some(ref fetch) = self.data_fetch {
            match fetch {
                DataFetch::Static { props } => {
                    code.push_str(&format!(
                        "export async function getStaticProps() {{\n  return {{ {} }};\n}}\n\n",
                        props
                    ));
                }
                DataFetch::Server { props } => {
                    code.push_str(&format!("export async function getServerSideProps(context) {{\n  return {{ {} }};\n}}\n\n", props));
                }
                DataFetch::Initial { props } => {
                    code.push_str(&format!(
                        "{}.getInitialProps = async () => {{ return {{ {} }}; }};\n\n",
                        self.component.name, props
                    ));
                }
            }
        }

        // Component
        code.push_str(&self.component.to_react());

        // Export for routing
        code.push_str(&format!("export default {};\n", self.component.name));

        code
    }
}

/// API route for Next.js
#[derive(Debug, Clone)]
pub struct ApiRoute {
    /// Route path (e.g., "/api/users" or "/api/users/[id]")
    pub path: String,
    /// HTTP methods
    pub methods: Vec<String>,
    /// Handler function body
    pub handler: String,
}

impl ApiRoute {
    /// Create new API route
    pub fn new(
        path: impl Into<String>,
        methods: Vec<impl Into<String>>,
        handler: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            methods: methods.into_iter().map(|m| m.into()).collect(),
            handler: handler.into(),
        }
    }

    /// Generate Next.js API route code
    pub fn to_nextjs(&self) -> String {
        format!(
            "// API Route: {}\nexport default function handler(req, res) {{\n  {}\n}}\n",
            self.path, self.handler
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_creation() {
        let comp = Component::new("Button")
            .prop("label", PropType::String)
            .prop("onClick", PropType::Custom("() => void".to_string()));

        assert_eq!(comp.name, "Button");
        assert_eq!(comp.props.len(), 2);
    }

    #[test]
    fn test_component_with_state() {
        let comp = Component::new("Counter")
            .state("count", PropType::Number, Value::from(0))
            .hook(Hook::UseState {
                var: "count".to_string(),
                initial: Value::from(0),
            });

        assert_eq!(comp.state.len(), 1);
        assert_eq!(comp.hooks.len(), 1);
    }

    #[test]
    fn test_page_route() {
        let page = PageRoute::new("/users/[id]", Component::new("UserPage"))
            .with_strategy(RenderStrategy::Server);

        assert_eq!(page.path, "/users/[id]");
    }

    #[test]
    fn test_api_route() {
        let api = ApiRoute::new(
            "/api/users",
            vec!["GET", "POST"],
            "return res.json({users});",
        );

        assert_eq!(api.path, "/api/users");
        assert_eq!(api.methods.len(), 2);
    }

    #[test]
    fn test_react_generation() {
        let comp = Component::new("Hello").hook(Hook::UseState {
            var: "name".to_string(),
            initial: Value::from("World"),
        });

        let code = comp.to_react();
        assert!(code.contains("function Hello"));
        assert!(code.contains("useState"));
    }
}
