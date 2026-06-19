//! Modern Framework Support for nCPU/nSynth
//!
//! Complete Svelte and Solid.js framework support with runes, signals, memos, effects, and routing.

use serde_json::Value;
use std::collections::HashMap;

/// Svelte component with runes mode support
#[derive(Debug, Clone)]
pub struct SvelteComponent {
    /// Component name
    pub name: String,
    /// Props
    pub props: Vec<SvelteProp>,
    /// State using runes
    pub state: Vec<SvelteState>,
    /// Reactive statements
    pub reactive: Vec<ReactiveStatement>,
    /// Template blocks
    pub blocks: Vec<TemplateBlock>,
    /// Lifecycle handlers
    pub lifecycle: Vec<LifecycleHandler>,
}

/// Svelte prop with TypeScript typing
#[derive(Debug, Clone)]
pub struct SvelteProp {
    /// Prop name
    pub name: String,
    /// Prop type (TypeScript)
    pub prop_type: String,
    /// Optional
    pub optional: bool,
    /// Default value
    pub default: Option<Value>,
    /// Reactive with $props runes
    pub reactive: bool,
}

/// Svelte state using runes
#[derive(Debug, Clone)]
pub struct SvelteState {
    /// Variable name
    pub name: String,
    /// Type (TypeScript)
    pub var_type: String,
    /// Initial value
    pub initial: Value,
    /// Rune type
    pub rune: RuneType,
}

/// Rune type for Svelte 5
#[derive(Debug, Clone, Copy)]
pub enum RuneType {
    /// let state = $state(...)
    State,
    /// let derived = $derived(...)
    Derived,
    /// let store = $store(...)
    Store,
}

/// Reactive statement ($: or $effect)
#[derive(Debug, Clone)]
pub enum ReactiveStatement {
    /// Legacy $: statement
    Legacy { code: String },
    /// Modern $effect rune
    Effect { code: String, deps: Vec<String> },
    /// $derived by
    Derived { name: String, computation: String },
}

/// Template block
#[derive(Debug, Clone)]
pub enum TemplateBlock {
    /// Each block
    Each {
        variable: String,
        collection: String,
        key: Option<String>,
        body: Vec<TemplateNode>,
    },
    /// If block
    If {
        condition: String,
        then_body: Vec<TemplateNode>,
        else_body: Option<Vec<TemplateNode>>,
    },
    /// Await block
    Await {
        promise: String,
        pending: Option<Vec<TemplateNode>>,
        then: Option<Vec<TemplateNode>>,
        catch: Option<Vec<TemplateNode>>,
    },
    /// Key block
    Key {
        expression: String,
        body: Vec<TemplateNode>,
    },
}

/// Template node
#[derive(Debug, Clone)]
pub enum TemplateNode {
    Element {
        tag: String,
        attributes: HashMap<String, String>,
        children: Vec<TemplateNode>,
    },
    Text(String),
    Expression(String),
    Component {
        name: String,
        props: HashMap<String, String>,
    },
}

/// Lifecycle handler
#[derive(Debug, Clone)]
pub enum LifecycleHandler {
    /// onMount
    OnMount { code: String },
    /// onDestroy
    OnDestroy { code: String },
    /// beforeUpdate
    BeforeUpdate { code: String },
    /// afterUpdate
    AfterUpdate { code: String },
    /// onTick (new in Svelte 5)
    OnTick { code: String },
}

impl SvelteComponent {
    /// Create new Svelte component
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            props: Vec::new(),
            state: Vec::new(),
            reactive: Vec::new(),
            blocks: Vec::new(),
            lifecycle: Vec::new(),
        }
    }

    /// Add prop
    pub fn prop(mut self, name: impl Into<String>, prop_type: impl Into<String>) -> Self {
        self.props.push(SvelteProp {
            name: name.into(),
            prop_type: prop_type.into(),
            optional: false,
            default: None,
            reactive: false,
        });
        self
    }

    /// Add reactive prop with $props rune
    pub fn prop_reactive(mut self, name: impl Into<String>, prop_type: impl Into<String>) -> Self {
        self.props.push(SvelteProp {
            name: name.into(),
            prop_type: prop_type.into(),
            optional: false,
            default: None,
            reactive: true,
        });
        self
    }

    /// Add optional prop with default
    pub fn prop_opt(
        mut self,
        name: impl Into<String>,
        prop_type: impl Into<String>,
        default: Value,
    ) -> Self {
        self.props.push(SvelteProp {
            name: name.into(),
            prop_type: prop_type.into(),
            optional: true,
            default: Some(default),
            reactive: false,
        });
        self
    }

    /// Add state with $state rune
    pub fn state(
        mut self,
        name: impl Into<String>,
        var_type: impl Into<String>,
        initial: Value,
    ) -> Self {
        self.state.push(SvelteState {
            name: name.into(),
            var_type: var_type.into(),
            initial,
            rune: RuneType::State,
        });
        self
    }

    /// Add derived state with $derived rune
    pub fn derived(mut self, name: impl Into<String>, computation: impl Into<String>) -> Self {
        self.reactive.push(ReactiveStatement::Derived {
            name: name.into(),
            computation: computation.into(),
        });
        self
    }

    /// Add effect
    pub fn effect(mut self, code: impl Into<String>, deps: Vec<impl Into<String>>) -> Self {
        self.reactive.push(ReactiveStatement::Effect {
            code: code.into(),
            deps: deps.into_iter().map(|s| s.into()).collect(),
        });
        self
    }

    /// Add each block
    pub fn each_block(
        mut self,
        variable: impl Into<String>,
        collection: impl Into<String>,
        body: Vec<TemplateNode>,
    ) -> Self {
        self.blocks.push(TemplateBlock::Each {
            variable: variable.into(),
            collection: collection.into(),
            key: None,
            body,
        });
        self
    }

    /// Add if block
    pub fn if_block(mut self, condition: impl Into<String>, then_body: Vec<TemplateNode>) -> Self {
        self.blocks.push(TemplateBlock::If {
            condition: condition.into(),
            then_body,
            else_body: None,
        });
        self
    }

    /// Add lifecycle handler
    pub fn on_mount(mut self, code: impl Into<String>) -> Self {
        self.lifecycle
            .push(LifecycleHandler::OnMount { code: code.into() });
        self
    }

    /// Generate Svelte 5 code with runes
    pub fn to_svelte(&self) -> String {
        let mut code = String::new();

        // Script tag with TypeScript
        code.push_str("<script lang=\"ts\">\n");

        // Props with $props rune (if reactive)
        if self.props.iter().any(|p| p.reactive) {
            code.push_str("  let { ");
            for (i, prop) in self.props.iter().filter(|p| p.reactive).enumerate() {
                if i > 0 {
                    code.push_str(", ");
                }
                code.push_str(&prop.name);
            }
            code.push_str(" } = $props();\n\n");
        } else if !self.props.is_empty() {
            // Legacy export let props
            for prop in &self.props {
                code.push_str(&format!(
                    "  export let {}: {};\n",
                    prop.name, prop.prop_type
                ));
                if let Some(default) = &prop.default {
                    code.push_str(&format!(
                        "  {} = {};\n",
                        prop.name,
                        serde_json::to_string(default).unwrap_or_default()
                    ));
                }
            }
            code.push('\n');
        }

        // State runes
        for state in &self.state {
            match state.rune {
                RuneType::State => {
                    code.push_str(&format!(
                        "  let {}: {} = $state({});\n",
                        state.name,
                        state.var_type,
                        serde_json::to_string(&state.initial).unwrap_or_default()
                    ));
                }
                RuneType::Derived => {
                    code.push_str(&format!(
                        "  let {} = $derived(() => {});\n",
                        state.name, state.var_type
                    ));
                }
                RuneType::Store => {
                    code.push_str(&format!(
                        "  let {} = $store({});\n",
                        state.name,
                        serde_json::to_string(&state.initial).unwrap_or_default()
                    ));
                }
            }
        }

        // Reactive statements
        for stmt in &self.reactive {
            match stmt {
                ReactiveStatement::Legacy { code: c } => {
                    code.push_str(&format!("  $: {}\n", c));
                }
                ReactiveStatement::Effect { code: c, deps } => {
                    code.push_str("  $effect(() => {\n");
                    code.push_str(&format!("    {}\n", c));
                    if !deps.is_empty() {
                        code.push_str(&format!("  }}); // deps: [{}]\n", deps.join(", ")));
                    } else {
                        code.push_str("  });\n");
                    }
                }
                ReactiveStatement::Derived { name, computation } => {
                    code.push_str(&format!("  let {} = $derived({});\n", name, computation));
                }
            }
        }

        // Lifecycle
        for handler in &self.lifecycle {
            match handler {
                LifecycleHandler::OnMount { code: c } => {
                    code.push_str(&format!("  onMount(() => {{\n    {}\n  }});\n", c));
                }
                LifecycleHandler::OnDestroy { code: c } => {
                    code.push_str(&format!("  onDestroy(() => {{\n    {}\n  }});\n", c));
                }
                LifecycleHandler::BeforeUpdate { code: c } => {
                    code.push_str(&format!("  beforeUpdate(() => {{\n    {}\n  }});\n", c));
                }
                LifecycleHandler::AfterUpdate { code: c } => {
                    code.push_str(&format!("  afterUpdate(() => {{\n    {}\n  }});\n", c));
                }
                LifecycleHandler::OnTick { code: c } => {
                    code.push_str(&format!("  $tick().then(() => {{\n    {}\n  }});\n", c));
                }
            }
        }

        code.push_str("</script>\n\n");

        // Template
        for block in &self.blocks {
            code.push_str(&self.render_block(block, 0));
        }

        code
    }

    fn render_block(&self, block: &TemplateBlock, indent: usize) -> String {
        let spaces = " ".repeat(indent);
        match block {
            TemplateBlock::Each {
                variable,
                collection,
                key,
                body,
            } => {
                let mut result = format!("{}{{#each {} as {}}}\n", spaces, collection, variable);
                if let Some(k) = key {
                    result = format!(
                        "{}{{#each {} as {} ({})}}\n",
                        spaces, collection, variable, k
                    );
                }
                for node in body {
                    result.push_str(&self.render_node(node, indent + 2));
                }
                result.push_str(&format!("{}{{/each}}\n", spaces));
                result
            }
            TemplateBlock::If {
                condition,
                then_body,
                else_body,
            } => {
                let mut result = format!("{}{{#if {}}}\n", spaces, condition);
                for node in then_body {
                    result.push_str(&self.render_node(node, indent + 2));
                }
                if let Some(else_nodes) = else_body {
                    result.push_str(&format!("{}{{else}}\n", spaces));
                    for node in else_nodes {
                        result.push_str(&self.render_node(node, indent + 2));
                    }
                }
                result.push_str(&format!("{}{{/if}}\n", spaces));
                result
            }
            TemplateBlock::Await {
                promise,
                pending,
                then,
                catch,
            } => {
                let mut result = String::new();
                result.push_str(&format!("{}{{#await {}}}\n", spaces, promise));
                if let Some(nodes) = pending {
                    for node in nodes {
                        result.push_str(&self.render_node(node, indent + 2));
                    }
                }
                if let Some(nodes) = then {
                    result.push_str(&format!("{}{{then}}\n", spaces));
                    for node in nodes {
                        result.push_str(&self.render_node(node, indent + 2));
                    }
                }
                if let Some(nodes) = catch {
                    result.push_str(&format!("{}{{catch}}\n", spaces));
                    for node in nodes {
                        result.push_str(&self.render_node(node, indent + 2));
                    }
                }
                result.push_str(&format!("{}{{/await}}\n", spaces));
                result
            }
            TemplateBlock::Key { expression, body } => {
                let mut result = format!("{}{{#key {}}}\n", spaces, expression);
                for node in body {
                    result.push_str(&self.render_node(node, indent + 2));
                }
                result.push_str(&format!("{}{{/key}}\n", spaces));
                result
            }
        }
    }

    fn render_node(&self, node: &TemplateNode, indent: usize) -> String {
        let spaces = " ".repeat(indent);
        match node {
            TemplateNode::Element {
                tag,
                attributes,
                children,
            } => {
                let mut result = format!("{}<{}", spaces, tag);
                for (k, v) in attributes {
                    if v.is_empty() {
                        result.push_str(&format!(" {}", k));
                    } else {
                        result.push_str(&format!(" {}=\"{}\"", k, v));
                    }
                }
                if children.is_empty() {
                    result.push_str(" />\n");
                } else {
                    result.push_str(">\n");
                    for child in children {
                        result.push_str(&self.render_node(child, indent + 2));
                    }
                    result.push_str(&format!("{}</{}>\n", spaces, tag));
                }
                result
            }
            TemplateNode::Text(text) => format!("{}{}\n", spaces, text),
            TemplateNode::Expression(expr) => format!("{}{{{}}}\n", spaces, expr),
            TemplateNode::Component { name, props } => {
                let mut result = format!("{}<{}", spaces, name);
                for (k, v) in props {
                    result.push_str(&format!(" {}={}", k, v));
                }
                result.push_str(" />\n");
                result
            }
        }
    }
}

/// Svelte store for state management
#[derive(Debug, Clone)]
pub struct SvelteStore {
    /// Store name
    pub name: String,
    /// Store type
    pub store_type: StoreType,
    /// Initial value
    pub initial: Option<Value>,
    /// Derived from other stores
    pub derived: Option<DerivedStore>,
}

/// Store type
#[derive(Debug, Clone, Copy)]
pub enum StoreType {
    /// writable
    Writable,
    /// readable
    Readable,
    /// derived
    Derived,
    /// readable with custom start/stop
    Custom,
}

/// Derived store specification
#[derive(Debug, Clone)]
pub struct DerivedStore {
    /// Source stores
    pub sources: Vec<String>,
    /// Derivation function
    pub derive_fn: String,
}

impl SvelteStore {
    /// Create writable store
    pub fn writable(name: impl Into<String>, initial: Value) -> Self {
        Self {
            name: name.into(),
            store_type: StoreType::Writable,
            initial: Some(initial),
            derived: None,
        }
    }

    /// Create readable store
    pub fn readable(name: impl Into<String>, initial: Value) -> Self {
        Self {
            name: name.into(),
            store_type: StoreType::Readable,
            initial: Some(initial),
            derived: None,
        }
    }

    /// Create derived store
    pub fn derived(
        name: impl Into<String>,
        sources: Vec<impl Into<String>>,
        derive_fn: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            store_type: StoreType::Derived,
            initial: None,
            derived: Some(DerivedStore {
                sources: sources.into_iter().map(|s| s.into()).collect(),
                derive_fn: derive_fn.into(),
            }),
        }
    }

    /// Generate store code
    pub fn to_store(&self) -> String {
        match self.store_type {
            StoreType::Writable => {
                format!(
                    "export const {} = writable({});",
                    self.name,
                    serde_json::to_string(self.initial.as_ref().unwrap_or(&Value::Null))
                        .unwrap_or_default()
                )
            }
            StoreType::Readable => {
                format!(
                    "export const {} = readable({}, () => {{ /* start/stop */ }});",
                    self.name,
                    serde_json::to_string(self.initial.as_ref().unwrap_or(&Value::Null))
                        .unwrap_or_default()
                )
            }
            StoreType::Derived => {
                if let Some(derived) = &self.derived {
                    format!(
                        "export const {} = derived([{}], ([${}]) => {});",
                        self.name,
                        derived.sources.join(", "),
                        derived.sources.join(", $"),
                        derived.derive_fn
                    )
                } else {
                    String::new()
                }
            }
            StoreType::Custom => {
                format!("export const {} = customStore();", self.name)
            }
        }
    }
}

/// Solid.js component with signals
#[derive(Debug, Clone)]
pub struct SolidComponent {
    /// Component name
    pub name: String,
    /// Props
    pub props: Vec<SolidProp>,
    /// Signals
    pub signals: Vec<SolidSignal>,
    /// Memos
    pub memos: Vec<SolidMemo>,
    /// Effects
    pub effects: Vec<SolidEffect>,
    /// Context
    pub context: Vec<SolidContext>,
    /// Children
    pub children: Vec<SolidNode>,
}

/// Solid prop
#[derive(Debug, Clone)]
pub struct SolidProp {
    /// Prop name
    pub name: String,
    /// Prop type (TypeScript)
    pub prop_type: String,
    /// Optional
    pub optional: bool,
    /// Default value
    pub default: Option<Value>,
}

/// Solid signal
#[derive(Debug, Clone)]
pub struct SolidSignal {
    /// Signal name
    pub name: String,
    /// Value type
    pub value_type: String,
    /// Initial value
    pub initial: Value,
}

/// Solid memo
#[derive(Debug, Clone)]
pub struct SolidMemo {
    /// Memo name
    pub name: String,
    /// Computation function
    pub computation: String,
    /// Dependencies
    pub deps: Vec<String>,
}

/// Solid effect
#[derive(Debug, Clone)]
pub struct SolidEffect {
    /// Effect function
    pub effect_fn: String,
    /// Dependencies
    pub deps: Vec<String>,
    /// On mount only
    pub on_mount: bool,
}

/// Solid context
#[derive(Debug, Clone)]
pub struct SolidContext {
    /// Context name
    pub name: String,
    /// Context type
    pub context_type: String,
    /// Default value
    pub default: Option<Value>,
}

/// Solid JSX node
#[derive(Debug, Clone)]
pub enum SolidNode {
    Element {
        tag: String,
        props: HashMap<String, String>,
        children: Vec<SolidNode>,
    },
    Component {
        name: String,
        props: HashMap<String, String>,
    },
    Text(String),
    Expression(String),
    Show {
        condition: String,
        then: Vec<SolidNode>,
        fallback: Option<Vec<SolidNode>>,
    },
    For {
        each: String,
        index: Option<String>,
        children: Vec<SolidNode>,
    },
}

impl SolidComponent {
    /// Create new Solid component
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            props: Vec::new(),
            signals: Vec::new(),
            memos: Vec::new(),
            effects: Vec::new(),
            context: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Add prop
    pub fn prop(mut self, name: impl Into<String>, prop_type: impl Into<String>) -> Self {
        self.props.push(SolidProp {
            name: name.into(),
            prop_type: prop_type.into(),
            optional: false,
            default: None,
        });
        self
    }

    /// Add signal
    pub fn signal(
        mut self,
        name: impl Into<String>,
        value_type: impl Into<String>,
        initial: Value,
    ) -> Self {
        self.signals.push(SolidSignal {
            name: name.into(),
            value_type: value_type.into(),
            initial,
        });
        self
    }

    /// Add memo
    pub fn memo(
        mut self,
        name: impl Into<String>,
        computation: impl Into<String>,
        deps: Vec<impl Into<String>>,
    ) -> Self {
        self.memos.push(SolidMemo {
            name: name.into(),
            computation: computation.into(),
            deps: deps.into_iter().map(|s| s.into()).collect(),
        });
        self
    }

    /// Add effect
    pub fn effect(mut self, effect_fn: impl Into<String>, deps: Vec<impl Into<String>>) -> Self {
        self.effects.push(SolidEffect {
            effect_fn: effect_fn.into(),
            deps: deps.into_iter().map(|s| s.into()).collect(),
            on_mount: false,
        });
        self
    }

    /// Add onMount effect
    pub fn on_mount(mut self, effect_fn: impl Into<String>) -> Self {
        self.effects.push(SolidEffect {
            effect_fn: effect_fn.into(),
            deps: Vec::new(),
            on_mount: true,
        });
        self
    }

    /// Add context
    pub fn context(mut self, name: impl Into<String>, context_type: impl Into<String>) -> Self {
        self.context.push(SolidContext {
            name: name.into(),
            context_type: context_type.into(),
            default: None,
        });
        self
    }

    /// Add child node
    pub fn child(mut self, node: SolidNode) -> Self {
        self.children.push(node);
        self
    }

    /// Generate Solid.js code
    pub fn to_solid(&self) -> String {
        let mut code = String::new();

        // Imports
        code.push_str("import { createSignal, createEffect, createMemo, onMount, Show, For } from 'solid-js';\n\n");

        // Props interface
        if !self.props.is_empty() {
            code.push_str(&format!("interface {}Props {{\n", self.name));
            for prop in &self.props {
                code.push_str(&format!(
                    "  {}{}: {};\n",
                    prop.name,
                    if prop.optional { "?" } else { "" },
                    prop.prop_type
                ));
                if let Some(default) = &prop.default {
                    code.push_str(&format!(
                        "  /* default: {} */\n",
                        serde_json::to_string(default).unwrap_or_default()
                    ));
                }
            }
            code.push_str("}\n\n");
        }

        // Component function
        code.push_str(&format!(
            "function {}({}: {}Props) {{\n",
            self.name,
            if self.props.is_empty() { "" } else { "props" },
            self.name
        ));

        // Destructure props
        if !self.props.is_empty() {
            code.push_str("  const { ");
            for (i, prop) in self.props.iter().enumerate() {
                if i > 0 {
                    code.push_str(", ");
                }
                code.push_str(&prop.name);
            }
            code.push_str(" } = props;\n\n");
        }

        // Signals
        for signal in &self.signals {
            code.push_str(&format!(
                "  const [{}{}, set{}] = createSignal<{}>({});\n",
                signal.name,
                if !self.signals.is_empty()
                    && self
                        .signals
                        .iter()
                        .filter(|s| s.name == signal.name)
                        .count()
                        > 1
                {
                    "1"
                } else {
                    ""
                },
                signal.name,
                signal.value_type,
                serde_json::to_string(&signal.initial).unwrap_or_default()
            ));
        }

        // Memos
        for memo in &self.memos {
            code.push_str(&format!(
                "  const {} = createMemo(() => {});\n",
                memo.name, memo.computation
            ));
        }

        // Effects
        for effect in &self.effects {
            if effect.on_mount {
                code.push_str(&format!("  onMount(() => {{ {} }});\n", effect.effect_fn));
            } else {
                code.push_str(&format!(
                    "  createEffect(() => {{ {} }});\n",
                    effect.effect_fn
                ));
            }
        }

        // Context usage
        for ctx in &self.context {
            code.push_str(&format!("  const {} = use{}();\n", ctx.name, ctx.name));
        }

        // Render return
        code.push_str("\n  return (\n");
        for child in &self.children {
            code.push_str(&self.render_node(child, 4));
        }
        code.push_str("  );\n");
        code.push_str("}\n\n");

        // Export
        code.push_str(&format!("export default {};\n", self.name));

        code
    }

    fn render_node(&self, node: &SolidNode, indent: usize) -> String {
        let spaces = " ".repeat(indent);
        match node {
            SolidNode::Element {
                tag,
                props,
                children,
            } => {
                let mut result = format!("{}<{}", spaces, tag);
                for (k, v) in props {
                    if v == "true" {
                        result.push_str(&format!(" {}", k));
                    } else {
                        result.push_str(&format!(" {}={}", k, v));
                    }
                }
                if children.is_empty() {
                    result.push_str(" />\n");
                } else {
                    result.push_str(">\n");
                    for child in children {
                        result.push_str(&self.render_node(child, indent + 2));
                    }
                    result.push_str(&format!("{}</{}>\n", spaces, tag));
                }
                result
            }
            SolidNode::Component { name, props } => {
                let mut result = format!("{}<{}", spaces, name);
                for (k, v) in props {
                    result.push_str(&format!(" {}={}", k, v));
                }
                result.push_str(" />\n");
                result
            }
            SolidNode::Text(text) => format!("{}{}\n", spaces, text),
            SolidNode::Expression(expr) => format!("{}{{{}}}\n", spaces, expr),
            SolidNode::Show {
                condition,
                then,
                fallback,
            } => {
                let mut result = format!("{}<Show when={{{} }}>\n", spaces, condition);
                for child in then {
                    result.push_str(&self.render_node(child, indent + 2));
                }
                result.push_str(&format!("{}</Show>\n", spaces));
                if let Some(else_nodes) = fallback {
                    result.push_str(&format!("{}<Show when={{false}} fallback={{\n", spaces));
                    for child in else_nodes {
                        result.push_str(&self.render_node(child, indent + 2));
                    }
                    result.push_str(&format!("{}}}>\n", spaces));
                }
                result
            }
            SolidNode::For {
                each,
                index,
                children: childs,
            } => {
                let mut result = if index.is_some() {
                    format!(
                        "{}<For each={{{} }} fallback={{<div>Loading...</div>}}>\n",
                        spaces, each
                    )
                } else {
                    format!("{}<For each={{{} }}>\n", spaces, each)
                };
                if let Some(idx) = index {
                    result.push_str(&format!("{{({}, {}) =>\n", each.replace("()", ""), idx));
                }
                for child in childs {
                    result.push_str(&self.render_node(child, indent + 2));
                }
                if index.is_some() {
                    result.push_str(&format!("{}}}\n", spaces));
                }
                result.push_str(&format!("{}</For>\n", spaces));
                result
            }
        }
    }
}

/// Solid.js router configuration
#[derive(Debug, Clone)]
pub struct SolidRouter {
    /// Routes
    pub routes: Vec<Route>,
    /// Router type
    pub router_type: RouterType,
    /// Base path
    pub base: Option<String>,
}

/// Router type
#[derive(Debug, Clone, Copy)]
pub enum RouterType {
    /// @solidjs/router (standard)
    Standard,
    /// @solidjs/start (file-based)
    Start,
    /// TanStack Router (experimental)
    TanStack,
}

/// Route definition
#[derive(Debug, Clone)]
pub struct Route {
    /// Path (e.g., "/users/:id")
    pub path: String,
    /// Component name
    pub component: String,
    /// Lazy loaded
    pub lazy: bool,
    /// Nested routes
    pub children: Vec<Route>,
    /// Data loading function
    pub loader: Option<String>,
}

impl SolidRouter {
    /// Create new router
    pub fn new(router_type: RouterType) -> Self {
        Self {
            routes: Vec::new(),
            router_type,
            base: None,
        }
    }

    /// Add route
    pub fn route(mut self, route: Route) -> Self {
        self.routes.push(route);
        self
    }

    /// Set base path
    pub fn base(mut self, base: impl Into<String>) -> Self {
        self.base = Some(base.into());
        self
    }

    /// Generate router code
    pub fn to_router(&self) -> String {
        let mut code = String::new();

        match self.router_type {
            RouterType::Standard => {
                code.push_str("import { Router, Route, Routes } from '@solidjs/router';\n\n");
                code.push_str("function App() {\n");
                code.push_str("  return (\n");
                code.push_str("    <Router");
                if let Some(base) = &self.base {
                    code.push_str(&format!(" base=\"{}\"", base));
                }
                code.push_str(">\n");
                code.push_str("      <Routes>\n");
                for route in &self.routes {
                    code.push_str(&self.render_route(route, 8));
                }
                code.push_str("      </Routes>\n");
                code.push_str("    </Router>\n");
                code.push_str("  );\n");
                code.push_str("}\n\nexport default App;\n");
            }
            RouterType::Start => {
                code.push_str("// File-based routing for @solidjs/start\n");
                code.push_str("// Place components in routes/ directory:\n");
                for route in &self.routes {
                    code.push_str(&format!(
                        "// routes{}.tsx -> {}\n",
                        route.path.replace("/", "").replace(":", "[$"),
                        route.component
                    ));
                }
            }
            RouterType::TanStack => {
                code.push_str("import { createRouter } from '@tanstack/solid-router';\n\n");
                code.push_str("const router = createRouter({\n");
                code.push_str("  routeTree: [\n");
                for route in &self.routes {
                    code.push_str(&format!(
                        "    {{ path: '{}', component: {} }},\n",
                        route.path, route.component
                    ));
                }
                code.push_str("  ],\n");
                code.push_str("});\n\nexport default router;\n");
            }
        }

        code
    }

    fn render_route(&self, route: &Route, indent: usize) -> String {
        let spaces = " ".repeat(indent);
        let mut result = if route.lazy {
            format!(
                "{}<Route path=\"{}\" component={{lazy(() => import('./{}.tsx'))}}>\n",
                spaces, route.path, route.component
            )
        } else {
            format!(
                "{}<Route path=\"{}\" component={{{}}} />\n",
                spaces, route.path, route.component
            )
        };

        if !route.children.is_empty() {
            for child in &route.children {
                result.push_str(&self.render_route(child, indent + 2));
            }
            result.push_str(&format!("{}</Route>\n", spaces));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svelte_component_creation() {
        let comp = SvelteComponent::new("Counter")
            .prop("count", "number")
            .state("count", "number", Value::from(0));

        assert_eq!(comp.name, "Counter");
        assert_eq!(comp.props.len(), 1);
        assert_eq!(comp.state.len(), 1);
    }

    #[test]
    fn test_svelte_reactive_props() {
        let comp = SvelteComponent::new("Button")
            .prop_reactive("label", "string")
            .prop_reactive("onClick", "() => void");

        assert_eq!(comp.props.len(), 2);
        assert!(comp.props[0].reactive);
        assert!(comp.props[1].reactive);
    }

    #[test]
    fn test_svelte_store() {
        let store = SvelteStore::writable("count", Value::from(0));

        let code = store.to_store();
        assert!(code.contains("writable"));
        assert!(code.contains("count"));
    }

    #[test]
    fn test_svelte_derived_store() {
        let store = SvelteStore::derived("doubled", vec!["count"], "$count * 2");

        let code = store.to_store();
        assert!(code.contains("derived"));
        assert!(code.contains("count"));
    }

    #[test]
    fn test_svelte_generation() {
        let comp = SvelteComponent::new("Hello")
            .state("name", "string", Value::from("World"))
            .effect("console.log(name)", vec!["name"]);

        let code = comp.to_svelte();
        assert!(code.contains("$state"));
        assert!(code.contains("$effect"));
    }

    #[test]
    fn test_solid_component_creation() {
        let comp = SolidComponent::new("Counter")
            .prop("initial", "number")
            .signal("count", "number", Value::from(0));

        assert_eq!(comp.name, "Counter");
        assert_eq!(comp.props.len(), 1);
        assert_eq!(comp.signals.len(), 1);
    }

    #[test]
    fn test_solid_signals() {
        let comp = SolidComponent::new("Form")
            .signal("username", "string", Value::from(""))
            .signal("password", "string", Value::from(""));

        assert_eq!(comp.signals.len(), 2);
        assert_eq!(comp.signals[0].name, "username");
    }

    #[test]
    fn test_solid_memo() {
        let comp = SolidComponent::new("Computed").memo("doubled", "count() * 2", vec!["count"]);

        assert_eq!(comp.memos.len(), 1);
    }

    #[test]
    fn test_solid_effect() {
        let comp = SolidComponent::new("Logger").effect("console.log(count())", vec!["count"]);

        assert_eq!(comp.effects.len(), 1);
        assert!(!comp.effects[0].on_mount);
    }

    #[test]
    fn test_solid_on_mount() {
        let comp = SolidComponent::new("Mounted").on_mount("console.log('mounted')");

        assert_eq!(comp.effects.len(), 1);
        assert!(comp.effects[0].on_mount);
    }

    #[test]
    fn test_solid_generation() {
        let comp = SolidComponent::new("Hello")
            .signal("name", "string", Value::from("World"))
            .child(SolidNode::Element {
                tag: "div".to_string(),
                props: HashMap::new(),
                children: vec![SolidNode::Expression("name()".to_string())],
            });

        let code = comp.to_solid();
        assert!(code.contains("createSignal"));
        assert!(code.contains("function Hello"));
    }

    #[test]
    fn test_solid_router() {
        let router = SolidRouter::new(RouterType::Standard)
            .route(Route {
                path: "/".to_string(),
                component: "Home".to_string(),
                lazy: false,
                children: Vec::new(),
                loader: None,
            })
            .route(Route {
                path: "/about".to_string(),
                component: "About".to_string(),
                lazy: true,
                children: Vec::new(),
                loader: None,
            });

        let code = router.to_router();
        assert!(code.contains("Router"));
        assert!(code.contains("Home"));
        assert!(code.contains("lazy"));
    }

    #[test]
    fn test_svelte_each_block() {
        let comp = SvelteComponent::new("ItemList").each_block(
            "item",
            "items",
            vec![TemplateNode::Text("Item: ".to_string())],
        );

        assert_eq!(comp.blocks.len(), 1);
    }

    #[test]
    fn test_svelte_if_block() {
        let comp = SvelteComponent::new("Conditional")
            .if_block("show", vec![TemplateNode::Text("Shown".to_string())]);

        assert_eq!(comp.blocks.len(), 1);
    }

    #[test]
    fn test_solid_show_node() {
        let node = SolidNode::Show {
            condition: "count() > 0".to_string(),
            then: vec![SolidNode::Text("Count is positive".to_string())],
            fallback: None,
        };

        let comp = SolidComponent::new("Test").child(node);
        assert_eq!(comp.children.len(), 1);
    }

    #[test]
    fn test_solid_for_node() {
        let node = SolidNode::For {
            each: "items()".to_string(),
            index: Some("index".to_string()),
            children: vec![SolidNode::Text("Item".to_string())],
        };

        let comp = SolidComponent::new("Test").child(node);
        assert_eq!(comp.children.len(), 1);
    }
}
