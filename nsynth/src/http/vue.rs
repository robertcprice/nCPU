//! Vue.js Framework Support for nCPU/nSynth
//!
//! Vue 3 component generation with Composition API, Vue Router, and Pinia state management.
//! Supports template compilation, reactive state, computed properties, watchers, and emits.

use serde_json::Value;
use std::collections::HashMap;

/// Vue 3 component with Composition API
#[derive(Debug, Clone)]
pub struct VueComponent {
    /// Component name
    pub name: String,
    /// Props definition
    pub props: Vec<VueProp>,
    /// Reactive state (ref/reactive)
    pub state: Vec<ReactiveState>,
    /// Computed properties
    pub computed: Vec<ComputedProperty>,
    /// Watchers
    pub watchers: Vec<Watcher>,
    /// Methods
    pub methods: Vec<Method>,
    /// Lifecycle hooks
    pub lifecycle: Vec<LifecycleHook>,
    /// Emits (events)
    pub emits: Vec<Event>,
    /// Template
    pub template: Option<VueTemplate>,
    /// Component setup
    pub setup_body: Vec<String>,
}

/// Vue prop definition
#[derive(Debug, Clone)]
pub struct VueProp {
    /// Prop name
    pub name: String,
    /// Prop type
    pub prop_type: PropType,
    /// Optional
    pub optional: bool,
    /// Default value
    pub default: Option<Value>,
    /// Validator
    pub validator: Option<String>,
}

/// Prop type for Vue
#[derive(Debug, Clone)]
pub enum PropType {
    String,
    Number,
    Boolean,
    Array,
    Object,
    Function,
    Symbol,
    Any,
    Union(Vec<PropType>),
    Custom(String),
}

/// Reactive state variable
#[derive(Debug, Clone)]
pub enum ReactiveState {
    /// Single reactive value (ref)
    Ref {
        name: String,
        value_type: PropType,
        initial: Value,
    },
    /// Reactive object (reactive)
    Reactive {
        name: String,
        properties: HashMap<String, PropType>,
        initial: Value,
    },
    /// Computed ref
    ComputedRef {
        name: String,
        value_type: PropType,
        getter: String,
    },
}

/// Computed property
#[derive(Debug, Clone)]
pub struct ComputedProperty {
    /// Property name
    pub name: String,
    /// Return type
    pub return_type: PropType,
    /// Getter function body
    pub getter: String,
    /// Dependencies (for automatic tracking)
    pub dependencies: Vec<String>,
}

/// Watcher
#[derive(Debug, Clone)]
pub struct Watcher {
    /// Source to watch
    pub source: WatchSource,
    /// Callback body
    pub callback: String,
    /// Options (immediate, deep)
    pub options: WatcherOptions,
}

/// Watch source
#[derive(Debug, Clone)]
pub enum WatchSource {
    /// Single ref
    Ref(String),
    /// Multiple refs
    MultipleRefs(Vec<String>),
    /// Getter function
    Getter(String),
}

/// Watcher options
#[derive(Debug, Clone)]
pub struct WatcherOptions {
    /// Run callback immediately on creation
    pub immediate: bool,
    /// Deep watch for objects
    pub deep: bool,
    /// Flush timing ('pre', 'post', 'sync')
    pub flush: String,
}

impl Default for WatcherOptions {
    fn default() -> Self {
        Self {
            immediate: false,
            deep: false,
            flush: "pre".to_string(),
        }
    }
}

/// Method
#[derive(Debug, Clone)]
pub struct Method {
    /// Method name
    pub name: String,
    /// Parameters (name, type)
    pub params: Vec<(String, PropType)>,
    /// Return type
    pub return_type: Option<PropType>,
    /// Function body
    pub body: String,
}

/// Lifecycle hook
#[derive(Debug, Clone)]
pub enum LifecycleHook {
    /// onBeforeMount
    BeforeMount(String),
    /// onMounted
    Mounted(String),
    /// onBeforeUpdate
    BeforeUpdate(String),
    /// onUpdated
    Updated(String),
    /// onBeforeUnmount
    BeforeUnmount(String),
    /// onUnmounted
    Unmounted(String),
    /// onErrorCaptured
    ErrorCaptured(String),
    /// onRenderTracked
    RenderTracked(String),
    /// onRenderTriggered
    RenderTriggered(String),
    /// onActivated (keep-alive)
    Activated(String),
    /// onDeactivated (keep-alive)
    Deactivated(String),
    /// onServerPrefetch (SSR)
    ServerPrefetch(String),
}

/// Event emission definition
#[derive(Debug, Clone)]
pub struct Event {
    /// Event name
    pub name: String,
    /// Payload type
    pub payload_type: Option<PropType>,
}

/// Vue template
#[derive(Debug, Clone)]
pub struct VueTemplate {
    /// Template content
    pub content: String,
    /// Directives used
    pub directives: Vec<TemplateDirective>,
}

/// Template directive
#[derive(Debug, Clone)]
pub enum TemplateDirective {
    /// v-if / v-else-if / v-else
    If { condition: String },
    /// v-for
    For { item: String, source: String },
    /// v-model
    Model { binding: String },
    /// v-bind / :
    Bind { attr: String, value: String },
    /// v-on / @
    On { event: String, handler: String },
    /// v-show
    Show { condition: String },
    /// v-slot
    Slot { name: String, props: Option<String> },
    /// v-html
    Html { expression: String },
    /// v-text
    Text { expression: String },
    /// v-once
    Once,
    /// v-memo
    Memo { condition: String },
    /// v-cloak
    Cloak,
    /// Custom directive
    Custom { name: String, value: String },
}

impl VueComponent {
    /// Create new Vue component
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            props: Vec::new(),
            state: Vec::new(),
            computed: Vec::new(),
            watchers: Vec::new(),
            methods: Vec::new(),
            lifecycle: Vec::new(),
            emits: Vec::new(),
            template: None,
            setup_body: Vec::new(),
        }
    }

    /// Add prop
    pub fn prop(mut self, name: impl Into<String>, prop_type: PropType) -> Self {
        self.props.push(VueProp {
            name: name.into(),
            prop_type,
            optional: false,
            default: None,
            validator: None,
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
        self.props.push(VueProp {
            name: name.into(),
            prop_type,
            optional: true,
            default: Some(default),
            validator: None,
        });
        self
    }

    /// Add prop with validator
    pub fn prop_with_validator(
        mut self,
        name: impl Into<String>,
        prop_type: PropType,
        validator: impl Into<String>,
    ) -> Self {
        self.props.push(VueProp {
            name: name.into(),
            prop_type,
            optional: false,
            default: None,
            validator: Some(validator.into()),
        });
        self
    }

    /// Add reactive ref
    pub fn add_ref(
        mut self,
        name: impl Into<String>,
        value_type: PropType,
        initial: Value,
    ) -> Self {
        self.state.push(ReactiveState::Ref {
            name: name.into(),
            value_type,
            initial,
        });
        self
    }

    /// Add reactive object
    pub fn reactive(
        mut self,
        name: impl Into<String>,
        properties: HashMap<String, PropType>,
        initial: Value,
    ) -> Self {
        self.state.push(ReactiveState::Reactive {
            name: name.into(),
            properties,
            initial,
        });
        self
    }

    /// Add computed property
    pub fn computed(
        mut self,
        name: impl Into<String>,
        return_type: PropType,
        getter: impl Into<String>,
    ) -> Self {
        self.computed.push(ComputedProperty {
            name: name.into(),
            return_type,
            getter: getter.into(),
            dependencies: Vec::new(),
        });
        self
    }

    /// Add computed with dependencies
    pub fn computed_with_deps(
        mut self,
        name: impl Into<String>,
        return_type: PropType,
        getter: impl Into<String>,
        dependencies: Vec<String>,
    ) -> Self {
        self.computed.push(ComputedProperty {
            name: name.into(),
            return_type,
            getter: getter.into(),
            dependencies,
        });
        self
    }

    /// Add watcher
    pub fn watch(mut self, source: WatchSource, callback: impl Into<String>) -> Self {
        self.watchers.push(Watcher {
            source,
            callback: callback.into(),
            options: WatcherOptions::default(),
        });
        self
    }

    /// Add watcher with options
    pub fn watch_with_options(
        mut self,
        source: WatchSource,
        callback: impl Into<String>,
        options: WatcherOptions,
    ) -> Self {
        self.watchers.push(Watcher {
            source,
            callback: callback.into(),
            options,
        });
        self
    }

    /// Add method
    pub fn method(
        mut self,
        name: impl Into<String>,
        params: Vec<(String, PropType)>,
        body: impl Into<String>,
    ) -> Self {
        self.methods.push(Method {
            name: name.into(),
            params,
            return_type: None,
            body: body.into(),
        });
        self
    }

    /// Add lifecycle hook
    pub fn lifecycle(mut self, hook: LifecycleHook) -> Self {
        self.lifecycle.push(hook);
        self
    }

    /// Add event emission
    pub fn emit(mut self, name: impl Into<String>, payload_type: Option<PropType>) -> Self {
        self.emits.push(Event {
            name: name.into(),
            payload_type,
        });
        self
    }

    /// Set template
    pub fn template(mut self, template: VueTemplate) -> Self {
        self.template = Some(template);
        self
    }

    /// Add custom setup code
    pub fn setup_code(mut self, code: impl Into<String>) -> Self {
        self.setup_body.push(code.into());
        self
    }

    /// Generate Vue 3 Composition API code
    pub fn to_vue3(&self) -> String {
        let mut code = String::new();

        // Script setup import
        code.push_str("<script setup lang=\"ts\">\n");

        // Imports
        code.push_str("import { ref, reactive, computed, watch, ");
        for (i, hook) in self.lifecycle.iter().enumerate() {
            code.push_str(&match hook {
                LifecycleHook::BeforeMount(_) => "onBeforeMount",
                LifecycleHook::Mounted(_) => "onMounted",
                LifecycleHook::BeforeUpdate(_) => "onBeforeUpdate",
                LifecycleHook::Updated(_) => "onUpdated",
                LifecycleHook::BeforeUnmount(_) => "onBeforeUnmount",
                LifecycleHook::Unmounted(_) => "onUnmounted",
                LifecycleHook::ErrorCaptured(_) => "onErrorCaptured",
                LifecycleHook::Activated(_) => "onActivated",
                LifecycleHook::Deactivated(_) => "onDeactivated",
                LifecycleHook::ServerPrefetch(_) => "onServerPrefetch",
                LifecycleHook::RenderTracked(_) | LifecycleHook::RenderTriggered(_) => continue,
            });
            if i < self.lifecycle.len() - 1 {
                code.push_str(", ");
            }
        }
        code.push_str(" } from 'vue';\n\n");

        // Props definition
        if !self.props.is_empty() || !self.emits.is_empty() {
            code.push_str("interface Props ");
            if !self.emits.is_empty() {
                code.push_str("{\n");
                if !self.props.is_empty() {
                    for prop in &self.props {
                        code.push_str(&format!(
                            "  {}?: {};\n",
                            prop.name,
                            self.prop_type_to_ts(&prop.prop_type)
                        ));
                    }
                }
                if !self.emits.is_empty() {
                    code.push_str("\n  // Events\n");
                }
                for event in &self.emits {
                    if let Some(pt) = &event.payload_type {
                        code.push_str(&format!(
                            "  on{}?: (payload: {}) => void;\n",
                            to_camel_case(&event.name),
                            self.prop_type_to_ts(pt)
                        ));
                    } else {
                        code.push_str(&format!(
                            "  on{}?: () => void;\n",
                            to_camel_case(&event.name)
                        ));
                    }
                }
                code.push_str("}\n");
            } else {
                code.push_str("{}\n");
            }

            code.push_str("const props = defineProps<Props>();\n\n");
        }

        // Emits
        if !self.emits.is_empty() {
            code.push_str("const emit = defineEmits<{\n");
            for event in &self.emits {
                if let Some(pt) = &event.payload_type {
                    code.push_str(&format!(
                        "  (event: '{}', payload: {}): void;\n",
                        event.name,
                        self.prop_type_to_ts(pt)
                    ));
                } else {
                    code.push_str(&format!("  (event: '{}'): void;\n", event.name));
                }
            }
            code.push_str("}>();\n\n");
        }

        // Reactive state
        for state in &self.state {
            match state {
                ReactiveState::Ref {
                    name,
                    value_type,
                    initial,
                } => {
                    code.push_str(&format!(
                        "const {} = ref<{}>({});\n",
                        name,
                        self.prop_type_to_ts(value_type),
                        serde_json::to_string(initial).unwrap_or_default()
                    ));
                }
                ReactiveState::Reactive {
                    name,
                    properties,
                    initial,
                } => {
                    let props_str = properties
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k, self.prop_type_to_ts(v)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    code.push_str(&format!(
                        "interface {}State {{ {} }}\n",
                        to_pascal_case(name),
                        props_str
                    ));
                    code.push_str(&format!(
                        "const {} = reactive<{}State>({});\n",
                        name,
                        to_pascal_case(name),
                        serde_json::to_string(initial).unwrap_or_default()
                    ));
                }
                ReactiveState::ComputedRef {
                    name,
                    value_type,
                    getter,
                } => {
                    code.push_str(&format!(
                        "const {} = computed<{}>(() => {{ {} }});\n",
                        name,
                        self.prop_type_to_ts(value_type),
                        getter
                    ));
                }
            }
        }

        if !self.state.is_empty() {
            code.push('\n');
        }

        // Computed properties
        for comp in &self.computed {
            code.push_str(&format!(
                "const {} = computed<{}>(() => {{\n  return {};\n}});\n",
                comp.name,
                self.prop_type_to_ts(&comp.return_type),
                comp.getter
            ));
        }

        if !self.computed.is_empty() {
            code.push('\n');
        }

        // Watchers
        for watcher in &self.watchers {
            let source_str = match &watcher.source {
                WatchSource::Ref(name) => name.clone(),
                WatchSource::MultipleRefs(names) => format!("[{}]", names.join(", ")),
                WatchSource::Getter(getter) => format!("() => {{ {} }}", getter),
            };

            let params = match &watcher.source {
                WatchSource::Ref(_) | WatchSource::MultipleRefs(_) => "newVal, oldVal",
                WatchSource::Getter(_) => "",
            };

            code.push_str(&format!(
                "watch({}, ({}) => {{\n  {}\n",
                source_str, params, watcher.callback
            ));

            if watcher.options.immediate || watcher.options.deep {
                code.push_str(", {{ ");
                let mut opts = Vec::new();
                if watcher.options.immediate {
                    opts.push("immediate: true".to_string());
                }
                if watcher.options.deep {
                    opts.push("deep: true".to_string());
                }
                if !watcher.options.flush.is_empty() && watcher.options.flush != "pre" {
                    opts.push(format!("flush: '{}'", watcher.options.flush));
                }
                code.push_str(&opts.join(", "));
                code.push_str(" }");
            }

            code.push_str(");\n");
        }

        if !self.watchers.is_empty() {
            code.push('\n');
        }

        // Methods
        for method in &self.methods {
            let params_str = method
                .params
                .iter()
                .map(|(n, t)| format!("{}: {}", n, self.prop_type_to_ts(t)))
                .collect::<Vec<_>>()
                .join(", ");

            let return_str = method
                .return_type
                .as_ref()
                .map(|t| format!(": {}", self.prop_type_to_ts(t)))
                .unwrap_or_default();

            code.push_str(&format!(
                "const {} = ({}): {} => {{\n  {};\n}};\n",
                method.name, params_str, return_str, method.body
            ));
        }

        if !self.methods.is_empty() {
            code.push('\n');
        }

        // Lifecycle hooks
        for hook in &self.lifecycle {
            match hook {
                LifecycleHook::BeforeMount(body) => {
                    code.push_str(&format!("onBeforeMount(() => {{\n  {};\n}});\n", body));
                }
                LifecycleHook::Mounted(body) => {
                    code.push_str(&format!("onMounted(() => {{\n  {};\n}});\n", body));
                }
                LifecycleHook::BeforeUpdate(body) => {
                    code.push_str(&format!("onBeforeUpdate(() => {{\n  {};\n}});\n", body));
                }
                LifecycleHook::Updated(body) => {
                    code.push_str(&format!("onUpdated(() => {{\n  {};\n}});\n", body));
                }
                LifecycleHook::BeforeUnmount(body) => {
                    code.push_str(&format!("onBeforeUnmount(() => {{\n  {};\n}});\n", body));
                }
                LifecycleHook::Unmounted(body) => {
                    code.push_str(&format!("onUnmounted(() => {{\n  {};\n}});\n", body));
                }
                LifecycleHook::ErrorCaptured(body) => {
                    code.push_str(&format!(
                        "onErrorCaptured((err, instance, info) => {{\n  {};\n}});\n",
                        body
                    ));
                }
                LifecycleHook::Activated(body) => {
                    code.push_str(&format!("onActivated(() => {{\n  {};\n}});\n", body));
                }
                LifecycleHook::Deactivated(body) => {
                    code.push_str(&format!("onDeactivated(() => {{\n  {};\n}});\n", body));
                }
                LifecycleHook::ServerPrefetch(body) => {
                    code.push_str(&format!(
                        "onServerPrefetch(async () => {{\n  {};\n}});\n",
                        body
                    ));
                }
                _ => {}
            }
        }

        if !self.lifecycle.is_empty() {
            code.push('\n');
        }

        // Custom setup code
        for line in &self.setup_body {
            code.push_str(line);
            code.push('\n');
        }

        code.push_str("</script>\n\n");

        // Template
        if let Some(template) = &self.template {
            code.push_str("<template>\n");
            code.push_str(&template.content);
            code.push_str("\n</template>\n");
        }

        // Style (placeholder)
        code.push_str("\n<style scoped>\n/* Component styles */\n</style>\n");

        code
    }

    /// Convert prop type to TypeScript
    fn prop_type_to_ts(&self, prop_type: &PropType) -> String {
        match prop_type {
            PropType::String => "string".to_string(),
            PropType::Number => "number".to_string(),
            PropType::Boolean => "boolean".to_string(),
            PropType::Array => "any[]".to_string(),
            PropType::Object => "Record<string, any>".to_string(),
            PropType::Function => "(...args: any[]) => any".to_string(),
            PropType::Symbol => "symbol".to_string(),
            PropType::Any => "any".to_string(),
            PropType::Union(types) => types
                .iter()
                .map(|t| self.prop_type_to_ts(t))
                .collect::<Vec<_>>()
                .join(" | "),
            PropType::Custom(s) => s.clone(),
        }
    }
}

/// Vue Router route
#[derive(Debug, Clone)]
pub struct VueRoute {
    /// Route path
    pub path: String,
    /// Route name
    pub name: Option<String>,
    /// Component to render
    pub component: String,
    /// Lazy loaded component
    pub lazy_component: Option<String>,
    /// Children routes
    pub children: Vec<VueRoute>,
    /// Route meta information
    pub meta: HashMap<String, Value>,
    /// Redirect target
    pub redirect: Option<String>,
    /// Route alias
    pub alias: Option<String>,
    /// Props pass to component
    pub props: bool,
}

impl VueRoute {
    /// Create new route
    pub fn new(path: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            name: None,
            component: component.into(),
            lazy_component: None,
            children: Vec::new(),
            meta: HashMap::new(),
            redirect: None,
            alias: None,
            props: false,
        }
    }

    /// Set route name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set lazy loaded component
    pub fn with_lazy(mut self, lazy: impl Into<String>) -> Self {
        self.lazy_component = Some(lazy.into());
        self
    }

    /// Add child route
    pub fn child(mut self, route: VueRoute) -> Self {
        self.children.push(route);
        self
    }

    /// Add meta information
    pub fn meta(mut self, key: impl Into<String>, value: Value) -> Self {
        self.meta.insert(key.into(), value);
        self
    }

    /// Set redirect
    pub fn redirect(mut self, to: impl Into<String>) -> Self {
        self.redirect = Some(to.into());
        self
    }

    /// Set alias
    pub fn alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// Enable props passing
    pub fn with_props(mut self) -> Self {
        self.props = true;
        self
    }

    /// Generate route configuration
    pub fn to_route_config(&self) -> String {
        let mut code = String::new();

        code.push_str("{\n");
        code.push_str(&format!("  path: '{}',\n", self.path));

        if let Some(name) = &self.name {
            code.push_str(&format!("  name: '{}',\n", name));
        }

        if let Some(lazy) = &self.lazy_component {
            code.push_str(&format!("  component: () => import('{}'),\n", lazy));
        } else {
            code.push_str(&format!("  component: {},\n", self.component));
        }

        if self.props {
            code.push_str("  props: true,\n");
        }

        if let Some(redirect) = &self.redirect {
            code.push_str(&format!("  redirect: '{}',\n", redirect));
        }

        if let Some(alias) = &self.alias {
            code.push_str(&format!("  alias: '{}',\n", alias));
        }

        if !self.meta.is_empty() {
            code.push_str("  meta: {\n");
            for (key, value) in &self.meta {
                code.push_str(&format!(
                    "    {}: {},\n",
                    key,
                    serde_json::to_string(value).unwrap_or_default()
                ));
            }
            code.push_str("  },\n");
        }

        if !self.children.is_empty() {
            code.push_str("  children: [\n");
            for child in &self.children {
                let child_config = child.to_route_config();
                for line in child_config.lines() {
                    code.push_str(&format!("    {}\n", line));
                }
            }
            code.push_str("  ],\n");
        }

        code.push_str("}");

        code
    }
}

/// Vue Router configuration
#[derive(Debug, Clone)]
pub struct VueRouter {
    /// Routes
    pub routes: Vec<VueRoute>,
    /// Router mode (history, hash)
    pub mode: RouterMode,
    /// Base URL
    pub base: String,
    /// Link active class
    pub link_active_class: Option<String>,
    /// Scroll behavior
    pub scroll_behavior: Option<String>,
}

/// Router mode
#[derive(Debug, Clone, Copy)]
pub enum RouterMode {
    /// HTML5 History mode
    History,
    /// Hash mode
    Hash,
    /// Memory mode (abstract)
    Memory,
}

impl VueRouter {
    /// Create new router configuration
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            mode: RouterMode::History,
            base: "/".to_string(),
            link_active_class: None,
            scroll_behavior: None,
        }
    }

    /// Add route
    pub fn route(mut self, route: VueRoute) -> Self {
        self.routes.push(route);
        self
    }

    /// Set router mode
    pub fn with_mode(mut self, mode: RouterMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set base URL
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }

    /// Generate router configuration
    pub fn to_router(&self) -> String {
        let mut code = String::new();

        code.push_str("import { createRouter, createWebHistory, createWebHashHistory } from 'vue-router';\n\n");

        code.push_str("const routes = [\n");
        for route in &self.routes {
            let route_config = route.to_route_config();
            for line in route_config.lines() {
                code.push_str(&format!("  {}\n", line));
            }
            code.push_str(",\n");
        }
        code.push_str("];\n\n");

        let history_fn = match self.mode {
            RouterMode::History => "createWebHistory",
            RouterMode::Hash => "createWebHashHistory",
            RouterMode::Memory => "createMemoryHistory",
        };

        code.push_str(&format!("const router = createRouter({{\n"));
        code.push_str(&format!("  history: {}('{}'),\n", history_fn, self.base));

        if let Some(active_class) = &self.link_active_class {
            code.push_str(&format!("  linkActiveClass: '{}',\n", active_class));
        }

        if let Some(scroll) = &self.scroll_behavior {
            code.push_str(&format!(
                "  scrollBehavior(to, from, savedPosition) {{\n    {}\n  }},\n",
                scroll
            ));
        }

        code.push_str("  routes,\n");
        code.push_str("});\n\n");
        code.push_str("export default router;\n");

        code
    }
}

impl Default for VueRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Pinia store
#[derive(Debug, Clone)]
pub struct PiniaStore {
    /// Store name (id)
    pub id: String,
    /// State
    pub state: HashMap<String, (PropType, Value)>,
    /// Getters
    pub getters: Vec<Getter>,
    /// Actions
    pub actions: Vec<Action>,
}

/// Getter in Pinia store
#[derive(Debug, Clone)]
pub struct Getter {
    /// Getter name
    pub name: String,
    /// Return type
    pub return_type: PropType,
    /// Function body
    pub body: String,
}

/// Action in Pinia store
#[derive(Debug, Clone)]
pub struct Action {
    /// Action name
    pub name: String,
    /// Parameters (name, type)
    pub params: Vec<(String, PropType)>,
    /// Return type
    pub return_type: Option<PropType>,
    /// Function body
    pub body: String,
    /// Async
    pub is_async: bool,
}

impl PiniaStore {
    /// Create new Pinia store
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            state: HashMap::new(),
            getters: Vec::new(),
            actions: Vec::new(),
        }
    }

    /// Add state property
    pub fn state(mut self, name: impl Into<String>, prop_type: PropType, initial: Value) -> Self {
        self.state.insert(name.into(), (prop_type, initial));
        self
    }

    /// Add getter
    pub fn getter(
        mut self,
        name: impl Into<String>,
        return_type: PropType,
        body: impl Into<String>,
    ) -> Self {
        self.getters.push(Getter {
            name: name.into(),
            return_type,
            body: body.into(),
        });
        self
    }

    /// Add action
    pub fn action(
        mut self,
        name: impl Into<String>,
        params: Vec<(String, PropType)>,
        body: impl Into<String>,
    ) -> Self {
        self.actions.push(Action {
            name: name.into(),
            params,
            return_type: None,
            body: body.into(),
            is_async: false,
        });
        self
    }

    /// Add async action
    pub fn async_action(
        mut self,
        name: impl Into<String>,
        params: Vec<(String, PropType)>,
        body: impl Into<String>,
    ) -> Self {
        self.actions.push(Action {
            name: name.into(),
            params,
            return_type: None,
            body: body.into(),
            is_async: true,
        });
        self
    }

    /// Generate Pinia store code
    pub fn to_store(&self) -> String {
        let mut code = String::new();

        code.push_str("import { defineStore } from 'pinia';\n");
        code.push_str(&format!("import {{ ref, computed }} from 'vue';\n\n"));

        code.push_str(&format!(
            "export const use{}Store = defineStore('{}', () => {{\n",
            to_pascal_case(&self.id),
            self.id
        ));

        // State
        if !self.state.is_empty() {
            code.push_str("  // State\n");
            for (name, (prop_type, initial)) in &self.state {
                code.push_str(&format!(
                    "  const {} = ref<{}>({});\n",
                    name,
                    self.prop_type_to_ts(prop_type),
                    serde_json::to_string(initial).unwrap_or_default()
                ));
            }
            code.push('\n');
        }

        // Getters
        if !self.getters.is_empty() {
            code.push_str("  // Getters\n");
            for getter in &self.getters {
                code.push_str(&format!(
                    "  const {} = computed<{}>(() => {{\n    {};\n  }});\n",
                    getter.name,
                    self.prop_type_to_ts(&getter.return_type),
                    getter.body
                ));
            }
            code.push('\n');
        }

        // Actions
        if !self.actions.is_empty() {
            code.push_str("  // Actions\n");
            for action in &self.actions {
                let async_kw = if action.is_async { "async " } else { "" };
                let params_str = action
                    .params
                    .iter()
                    .map(|(n, t)| format!("{}: {}", n, self.prop_type_to_ts(t)))
                    .collect::<Vec<_>>()
                    .join(", ");
                code.push_str(&format!(
                    "  const {} = {}({}): void => {{\n    {};\n  }};\n",
                    action.name, async_kw, params_str, action.body
                ));
            }
            code.push('\n');
        }

        // Return
        code.push_str("  return {\n");
        for name in self.state.keys() {
            code.push_str(&format!("    {},\n", name));
        }
        for getter in &self.getters {
            code.push_str(&format!("    {},\n", getter.name));
        }
        for action in &self.actions {
            code.push_str(&format!("    {},\n", action.name));
        }
        code.push_str("  };\n");
        code.push_str("});\n");

        code
    }

    /// Convert prop type to TypeScript
    fn prop_type_to_ts(&self, prop_type: &PropType) -> String {
        match prop_type {
            PropType::String => "string".to_string(),
            PropType::Number => "number".to_string(),
            PropType::Boolean => "boolean".to_string(),
            PropType::Array => "any[]".to_string(),
            PropType::Object => "Record<string, any>".to_string(),
            PropType::Function => "(...args: any[]) => any".to_string(),
            PropType::Symbol => "symbol".to_string(),
            PropType::Any => "any".to_string(),
            PropType::Union(types) => types
                .iter()
                .map(|t| self.prop_type_to_ts(t))
                .collect::<Vec<_>>()
                .join(" | "),
            PropType::Custom(s) => s.clone(),
        }
    }
}

/// Vue template compiler
#[derive(Debug, Clone)]
pub struct VueTemplateCompiler {
    /// Template content
    pub content: String,
    /// Compilation options
    pub options: CompilerOptions,
}

/// Compiler options
#[derive(Debug, Clone)]
pub struct CompilerOptions {
    /// Enable scope ID
    pub scope_id: bool,
    /// Mode (module, function)
    pub mode: CompilerMode,
    /// Source map
    pub source_map: bool,
}

/// Compiler mode
#[derive(Debug, Clone, Copy)]
pub enum CompilerMode {
    Module,
    Function,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            scope_id: true,
            mode: CompilerMode::Module,
            source_map: false,
        }
    }
}

impl VueTemplateCompiler {
    /// Create new template compiler
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            options: CompilerOptions::default(),
        }
    }

    /// Set compiler options
    pub fn with_options(mut self, options: CompilerOptions) -> Self {
        self.options = options;
        self
    }

    /// Compile template to render function
    pub fn compile(&self) -> Result<CompiledTemplate, String> {
        let mut code = String::new();

        // Parse and compile directives
        let directives = self.parse_directives();

        code.push_str("// Compiled render function\n");
        code.push_str(
            "import { createElementVNode, toDisplayString, normalizeStyle } from 'vue';\n\n",
        );

        code.push_str("export function render(_ctx, _cache) {\n");
        code.push_str("  return ");

        // Simple template compilation (basic implementation)
        let compiled = self.compile_template(&self.content, &directives);
        code.push_str(&compiled);

        code.push_str("\n}\n");

        Ok(CompiledTemplate { code, directives })
    }

    /// Parse template directives
    fn parse_directives(&self) -> Vec<TemplateDirective> {
        let mut directives = Vec::new();

        // Basic directive parsing (regex-based)
        for line in self.content.lines() {
            if line.contains("v-if") {
                let condition = line
                    .split("v-if=\"")
                    .nth(1)
                    .and_then(|s| s.split('"').next())
                    .unwrap_or("true");
                directives.push(TemplateDirective::If {
                    condition: condition.to_string(),
                });
            }
            if line.contains("v-for") {
                let parts = line
                    .split("v-for=\"")
                    .nth(1)
                    .and_then(|s| s.split('"').next())
                    .unwrap_or("");
                let mut iter = parts.split(" in ");
                if let (Some(item), Some(source)) = (iter.next(), iter.next()) {
                    directives.push(TemplateDirective::For {
                        item: item.trim().to_string(),
                        source: source.trim().to_string(),
                    });
                }
            }
            if line.contains("v-model=") {
                let binding = line
                    .split("v-model=\"")
                    .nth(1)
                    .and_then(|s| s.split('"').next())
                    .unwrap_or("");
                directives.push(TemplateDirective::Model {
                    binding: binding.to_string(),
                });
            }
            if line.contains("@") || line.contains("v-on:") {
                let event = if let Some(pos) = line.find('@') {
                    line[pos + 1..].split('=').next().unwrap_or("")
                } else {
                    line.split("v-on:")
                        .nth(1)
                        .and_then(|s| s.split('=').next())
                        .unwrap_or("")
                };
                if !event.is_empty() {
                    let handler = line
                        .split('=')
                        .nth(1)
                        .and_then(|s| s.split('"').next())
                        .unwrap_or("");
                    directives.push(TemplateDirective::On {
                        event: event.to_string(),
                        handler: handler.to_string(),
                    });
                }
            }
        }

        directives
    }

    /// Compile template content
    fn compile_template(&self, content: &str, _directives: &[TemplateDirective]) -> String {
        let mut result = String::new();
        let lines: Vec<&str> = content.lines().map(|l| l.trim()).collect();

        for (i, line) in lines.iter().enumerate() {
            if line.is_empty() || line.starts_with("<!--") {
                continue;
            }

            if line.starts_with('<') && !line.starts_with("<!--") {
                // Element
                if let Some(tag_end) = line.find('>') {
                    let tag_part = &line[1..tag_end];
                    let tag_name = tag_part.split_whitespace().next().unwrap_or("div");

                    let attrs: Vec<&str> = tag_part
                        .split_whitespace()
                        .filter(|s| {
                            !s.starts_with('@') && !s.starts_with("v-") && !s.starts_with(':')
                        })
                        .collect();

                    let mut attrs_str = attrs.join(" ");

                    // Check for v-bind or shorthand
                    for part in tag_part.split_whitespace() {
                        if part.starts_with(':') || part.starts_with("v-bind:") {
                            attrs_str = format!("{} {}", attrs_str, part);
                        }
                    }

                    result.push_str(&format!(
                        "createElementVNode(\"{}\", {{ {} }}",
                        tag_name, attrs_str
                    ));

                    // Check for content
                    let content_start = tag_end + 1;
                    if let Some(closing) = line[content_start..].find("</") {
                        let content = &line[content_start..content_start + closing];
                        if !content.trim().is_empty() {
                            result.push_str(&format!(", \"{}\"", content));
                        }
                    }

                    result.push_str(")");
                }
            } else {
                // Text content
                if line.contains("{{") {
                    let expr = line
                        .split("{{")
                        .nth(1)
                        .and_then(|s| s.split("}}").next())
                        .unwrap_or("");
                    if !expr.is_empty() {
                        result.push_str(&format!("toDisplayString(_ctx.{})", expr.trim()));
                    }
                } else {
                    result.push_str(&format!("\"{}\"", line));
                }
            }

            // Add comma for all but last non-empty, non-comment line
            let has_more_content = lines[i + 1..]
                .iter()
                .any(|l| !l.is_empty() && !l.starts_with("<!--"));
            if has_more_content {
                result.push_str(",\n  ");
            }
        }

        // Remove trailing comma
        if result.ends_with(",\n  ") {
            result.truncate(result.len() - 3);
        }

        result
    }
}

/// Compiled template result
#[derive(Debug, Clone)]
pub struct CompiledTemplate {
    /// Generated code
    pub code: String,
    /// Found directives
    pub directives: Vec<TemplateDirective>,
}

/// Helper functions
fn to_camel_case(s: &str) -> String {
    s.split('-')
        .enumerate()
        .map(|(i, part)| {
            if i == 0 {
                part.to_string()
            } else {
                let mut chars = part.chars();
                match chars.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                    None => String::new(),
                }
            }
        })
        .collect()
}

fn to_pascal_case(s: &str) -> String {
    s.split(|c: char| c == '_' || c == '-' || c == ' ')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

impl Default for VueTemplateCompiler {
    fn default() -> Self {
        Self::new("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_creation() {
        let comp = VueComponent::new("Button")
            .prop("label", PropType::String)
            .prop("disabled", PropType::Boolean);

        assert_eq!(comp.name, "Button");
        assert_eq!(comp.props.len(), 2);
    }

    #[test]
    fn test_component_with_state() {
        let comp = VueComponent::new("Counter")
            .add_ref("count", PropType::Number, Value::from(0))
            .computed("doubled", PropType::Number, "count.value * 2");

        assert_eq!(comp.state.len(), 1);
        assert_eq!(comp.computed.len(), 1);
    }

    #[test]
    fn test_component_with_watchers() {
        let comp = VueComponent::new("UserForm").watch(
            WatchSource::Ref("username".to_string()),
            "console.log('Username changed:', newVal)",
        );

        assert_eq!(comp.watchers.len(), 1);
    }

    #[test]
    fn test_component_with_emits() {
        let comp = VueComponent::new("Modal")
            .emit("close", None)
            .emit("submit", Some(PropType::Object));

        assert_eq!(comp.emits.len(), 2);
    }

    #[test]
    fn test_reactive_state() {
        let mut props = HashMap::new();
        props.insert("x".to_string(), PropType::Number);
        props.insert("y".to_string(), PropType::Number);

        let mut initial = serde_json::Map::new();
        initial.insert("x".to_string(), Value::from(0));
        initial.insert("y".to_string(), Value::from(0));

        let comp = VueComponent::new("Point").reactive("position", props, Value::Object(initial));

        assert_eq!(comp.state.len(), 1);
    }

    #[test]
    fn test_vue_route_creation() {
        let route = VueRoute::new("/users/:id", "UserView")
            .with_name("user")
            .meta("requiresAuth", Value::Bool(true))
            .with_props();

        assert_eq!(route.path, "/users/:id");
        assert_eq!(route.name, Some("user".to_string()));
        assert!(route.props);
    }

    #[test]
    fn test_nested_routes() {
        let parent = VueRoute::new("/admin", "AdminLayout")
            .child(VueRoute::new("users", "AdminUsers"))
            .child(VueRoute::new("settings", "AdminSettings"));

        assert_eq!(parent.children.len(), 2);
    }

    #[test]
    fn test_router_config() {
        let router = VueRouter::new()
            .route(VueRoute::new("/", "HomeView"))
            .route(VueRoute::new("/about", "AboutView"))
            .with_mode(RouterMode::History);

        assert_eq!(router.routes.len(), 2);
    }

    #[test]
    fn test_pinia_store_creation() {
        let store = PiniaStore::new("counter")
            .state("count", PropType::Number, Value::from(0))
            .getter("double", PropType::Number, "count.value * 2")
            .action("increment", vec![], "count.value++");

        assert_eq!(store.state.len(), 1);
        assert_eq!(store.getters.len(), 1);
        assert_eq!(store.actions.len(), 1);
    }

    #[test]
    fn test_async_action() {
        let store = PiniaStore::new("user").async_action(
            "fetchUser",
            vec![("id".to_string(), PropType::Number)],
            "const response = await fetch(`/api/users/${id}`); user.value = await response.json();",
        );

        assert_eq!(store.actions.len(), 1);
        assert!(store.actions[0].is_async);
    }

    #[test]
    fn test_vue_generation() {
        let comp = VueComponent::new("Hello")
            .prop("name", PropType::String)
            .add_ref("greeting", PropType::String, Value::from("Hello"))
            .computed(
                "message",
                PropType::String,
                "greeting.value + ', ' + props.name",
            );

        let code = comp.to_vue3();
        assert!(code.contains("<script setup"));
        assert!(code.contains("const greeting = ref<string>"));
        assert!(code.contains("const message = computed<string>"));
    }

    #[test]
    fn test_template_compiler() {
        let compiler = VueTemplateCompiler::new("<div v-if=\"visible\">{{ message }}</div>");
        let compiled = compiler.compile().unwrap();

        assert!(!compiled.code.is_empty());
        assert!(!compiled.directives.is_empty());
    }

    #[test]
    fn test_prop_type_conversion() {
        let comp = VueComponent::new("Test");

        assert_eq!(comp.prop_type_to_ts(&PropType::String), "string");
        assert_eq!(comp.prop_type_to_ts(&PropType::Number), "number");
        assert_eq!(comp.prop_type_to_ts(&PropType::Boolean), "boolean");
        assert_eq!(comp.prop_type_to_ts(&PropType::Array), "any[]");
    }

    #[test]
    fn test_watcher_options() {
        let opts = WatcherOptions {
            immediate: true,
            deep: true,
            flush: "post".to_string(),
        };

        assert!(opts.immediate);
        assert!(opts.deep);
    }

    #[test]
    fn test_lifecycle_hooks() {
        let comp = VueComponent::new("Test")
            .lifecycle(LifecycleHook::Mounted("console.log('mounted')".to_string()))
            .lifecycle(LifecycleHook::Unmounted("cleanup()".to_string()));

        assert_eq!(comp.lifecycle.len(), 2);
    }

    #[test]
    fn test_template_directive_parsing() {
        let compiler = VueTemplateCompiler::new(
            "<div v-for=\"item in items\" @click=\"handleClick\">{{ item.name }}</div>",
        );
        let directives = compiler.parse_directives();

        assert!(directives
            .iter()
            .any(|d| matches!(d, TemplateDirective::For { .. })));
    }

    #[test]
    fn test_route_config_generation() {
        let route = VueRoute::new("/test", "TestView")
            .with_name("test")
            .meta("title", Value::from("Test Page"));

        let config = route.to_route_config();
        assert!(config.contains("path: '/test'"));
        assert!(config.contains("name: 'test'"));
    }

    #[test]
    fn test_router_modes() {
        let router = VueRouter::new()
            .with_mode(RouterMode::Hash)
            .with_base("/app");

        assert!(matches!(router.mode, RouterMode::Hash));
        assert_eq!(router.base, "/app");
    }

    #[test]
    fn test_method_generation() {
        let comp = VueComponent::new("Form").method(
            "submit",
            vec![("data".to_string(), PropType::Object)],
            "console.log(data);",
        );

        assert_eq!(comp.methods.len(), 1);
        assert_eq!(comp.methods[0].name, "submit");
    }

    #[test]
    fn test_reactive_object_generation() {
        let mut props = HashMap::new();
        props.insert("count".to_string(), PropType::Number);
        props.insert("name".to_string(), PropType::String);

        let mut initial = serde_json::Map::new();
        initial.insert("count".to_string(), Value::from(0));
        initial.insert("name".to_string(), Value::from("test"));

        let comp = VueComponent::new("State").reactive("state", props, Value::Object(initial));

        let code = comp.to_vue3();
        assert!(code.contains("reactive"));
    }

    #[test]
    fn test_event_emission_with_payload() {
        let comp = VueComponent::new("Emitter")
            .emit("data", Some(PropType::Object))
            .emit("simple", None);

        assert_eq!(comp.emits.len(), 2);
        assert!(comp.emits[0].payload_type.is_some());
        assert!(comp.emits[1].payload_type.is_none());
    }

    #[test]
    fn test_lazy_route_loading() {
        let route = VueRoute::new("/lazy", "LazyComponent").with_lazy("./views/LazyView.vue");

        assert!(route.lazy_component.is_some());
    }

    #[test]
    fn test_route_with_redirect() {
        let route = VueRoute::new("/old-path", "OldView").redirect("/new-path");

        assert_eq!(route.redirect, Some("/new-path".to_string()));
    }

    #[test]
    fn test_route_with_alias() {
        let route = VueRoute::new("/main", "MainView").alias("/alternative");

        assert_eq!(route.alias, Some("/alternative".to_string()));
    }

    #[test]
    fn test_computed_with_dependencies() {
        let comp = VueComponent::new("Dependent").computed_with_deps(
            "total",
            PropType::Number,
            "price.value * quantity.value",
            vec!["price".to_string(), "quantity".to_string()],
        );

        assert_eq!(comp.computed[0].dependencies.len(), 2);
    }

    #[test]
    fn test_multiple_watchers() {
        let comp = VueComponent::new("MultiWatch").watch(
            WatchSource::MultipleRefs(vec!["a".to_string(), "b".to_string()]),
            "console.log('a or b changed')",
        );

        assert!(matches!(
            comp.watchers[0].source,
            WatchSource::MultipleRefs(_)
        ));
    }

    #[test]
    fn test_watcher_with_getter() {
        let comp = VueComponent::new("GetterWatch").watch(
            WatchSource::Getter("items.value.filter(i => i.active)".to_string()),
            "console.log('active items changed')",
        );

        assert!(matches!(comp.watchers[0].source, WatchSource::Getter(_)));
    }

    #[test]
    fn test_prop_union_types() {
        let comp = VueComponent::new("UnionProp").prop(
            "value",
            PropType::Union(vec![PropType::String, PropType::Number]),
        );

        let code = comp.to_vue3();
        assert!(code.contains("string | number"));
    }

    #[test]
    fn test_custom_prop_types() {
        let comp = VueComponent::new("Custom").prop("user", PropType::Custom("User".to_string()));

        let code = comp.to_vue3();
        assert!(code.contains("User"));
    }

    #[test]
    fn test_prop_with_validator() {
        let comp = VueComponent::new("Validated").prop_with_validator(
            "age",
            PropType::Number,
            "value >= 0 && value <= 120",
        );

        assert_eq!(
            comp.props[0].validator,
            Some("value >= 0 && value <= 120".to_string())
        );
    }

    #[test]
    fn test_template_directive_variety() {
        let template = VueTemplate {
            content: "<div v-show=\"visible\" v-once v-memo=\"expr\">Content</div>".to_string(),
            directives: vec![
                TemplateDirective::Show {
                    condition: "visible".to_string(),
                },
                TemplateDirective::Once,
                TemplateDirective::Memo {
                    condition: "expr".to_string(),
                },
            ],
        };

        assert_eq!(template.directives.len(), 3);
    }

    #[test]
    fn test_vue_template_integration() {
        let template = VueTemplate {
            content: "<div>{{ greeting }}</div>".to_string(),
            directives: vec![],
        };

        let comp = VueComponent::new("Greeting").template(template);

        assert!(comp.template.is_some());
        let code = comp.to_vue3();
        assert!(code.contains("<template>"));
        assert!(code.contains("</template>"));
    }
}
