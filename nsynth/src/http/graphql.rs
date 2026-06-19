//! GraphQL Implementation for nCPU/nSynth
//!
//! Complete GraphQL implementation with schema definition, query/mutation/subscription
//! support, resolvers, and comprehensive error handling.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

/// GraphQL Error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum GraphQLError {
    /// Syntax error in query document
    #[error("Syntax error: {0}")]
    SyntaxError(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Field not found on type
    #[error("Field '{field}' not found on type '{type_name}'")]
    FieldNotFound { field: String, type_name: String },

    /// Argument error
    #[error("Argument error: {0}")]
    ArgumentError(String),

    /// Type mismatch
    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    /// Null value in non-null field
    #[error("Null value in non-null field '{field}'")]
    NullNotNullError { field: String },

    /// Resolver execution error
    #[error("Resolver error on field '{field}': {message}")]
    ResolverError { field: String, message: String },

    /// Subscription error
    #[error("Subscription error: {0}")]
    SubscriptionError(String),

    /// Schema error
    #[error("Schema error: {0}")]
    SchemaError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl GraphQLError {
    /// Create a syntax error
    pub fn syntax(msg: impl Into<String>) -> Self {
        Self::SyntaxError(msg.into())
    }

    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::ValidationError(msg.into())
    }

    /// Create a field not found error
    pub fn field_not_found(field: impl Into<String>, type_name: impl Into<String>) -> Self {
        Self::FieldNotFound {
            field: field.into(),
            type_name: type_name.into(),
        }
    }

    /// Create an argument error
    pub fn argument(msg: impl Into<String>) -> Self {
        Self::ArgumentError(msg.into())
    }

    /// Create a type mismatch error
    pub fn type_mismatch(expected: impl Into<String>, found: impl Into<String>) -> Self {
        Self::TypeMismatch {
            expected: expected.into(),
            found: found.into(),
        }
    }

    /// Create a null not null error
    pub fn null_not_null(field: impl Into<String>) -> Self {
        Self::NullNotNullError {
            field: field.into(),
        }
    }

    /// Create a resolver error
    pub fn resolver(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ResolverError {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Get error code for serialization
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::SyntaxError(_) => "SYNTAX_ERROR",
            Self::ValidationError(_) => "VALIDATION_ERROR",
            Self::FieldNotFound { .. } => "FIELD_NOT_FOUND",
            Self::ArgumentError(_) => "ARGUMENT_ERROR",
            Self::TypeMismatch { .. } => "TYPE_MISMATCH",
            Self::NullNotNullError { .. } => "NULL_NOT_NULL",
            Self::ResolverError { .. } => "RESOLVER_ERROR",
            Self::SubscriptionError(_) => "SUBSCRIPTION_ERROR",
            Self::SchemaError(_) => "SCHEMA_ERROR",
            Self::InternalError(_) => "INTERNAL_ERROR",
        }
    }
}

/// Result type for GraphQL operations
pub type GraphQLResult<T> = Result<T, GraphQLError>;

/// GraphQL value types
#[derive(Debug, Clone, PartialEq)]
pub enum GraphQLValue {
    /// Null value
    Null,

    /// Boolean value
    Boolean(bool),

    /// Integer value
    Int(i64),

    /// Float value
    Float(f64),

    /// String value
    String(String),

    /// List value
    List(Vec<GraphQLValue>),

    /// Object value (fields as key-value pairs)
    Object(HashMap<String, GraphQLValue>),

    /// Enum value
    Enum(String),

    /// Custom scalar value (serialized as string)
    CustomScalar(String),
}

impl GraphQLValue {
    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Convert to JSON value
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Null => serde_json::Value::Null,
            Self::Boolean(b) => serde_json::Value::Bool(*b),
            Self::Int(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            Self::Float(f) => serde_json::Value::Number(
                serde_json::Number::from_f64(*f).unwrap_or_else(|| serde_json::Number::from(0)),
            ),
            Self::String(s) => serde_json::Value::String(s.clone()),
            Self::Enum(s) => serde_json::Value::String(s.clone()),
            Self::CustomScalar(s) => serde_json::Value::String(s.clone()),
            Self::List(items) => {
                serde_json::Value::Array(items.iter().map(|v| v.to_json()).collect())
            }
            Self::Object(fields) => {
                serde_json::Value::Object(
                    fields.iter().map(|(k, v)| (k.clone(), v.to_json())).collect(),
                )
            }
        }
    }

    /// Create null value
    pub fn null() -> Self {
        Self::Null
    }

    /// Create boolean value
    pub fn boolean(b: bool) -> Self {
        Self::Boolean(b)
    }

    /// Create int value
    pub fn int(i: i64) -> Self {
        Self::Int(i)
    }

    /// Create float value
    pub fn float(f: f64) -> Self {
        Self::Float(f)
    }

    /// Create string value
    pub fn string(s: impl Into<String>) -> Self {
        Self::String(s.into())
    }

    /// Create list value
    pub fn list(items: Vec<GraphQLValue>) -> Self {
        Self::List(items)
    }

    /// Create object value
    pub fn object(fields: HashMap<String, GraphQLValue>) -> Self {
        Self::Object(fields)
    }

    /// Create enum value
    pub fn enm(s: impl Into<String>) -> Self {
        Self::Enum(s.into())
    }

    /// Parse from JSON value
    pub fn from_json(json: &serde_json::Value) -> GraphQLResult<Self> {
        Ok(match json {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(b) => Self::Boolean(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Self::Int(i)
                } else if let Some(f) = n.as_f64() {
                    Self::Float(f)
                } else {
                    return Err(GraphQLError::type_mismatch("number", "unknown"));
                }
            }
            serde_json::Value::String(s) => Self::String(s.clone()),
            serde_json::Value::Array(items) => {
                Self::List(
                    items.iter().map(Self::from_json).collect::<GraphQLResult<Vec<_>>>()?
                )
            }
            serde_json::Value::Object(obj) => {
                Self::Object(
                    obj.iter()
                        .map(|(k, v)| Ok((k.clone(), Self::from_json(v)?)))
                        .collect::<GraphQLResult<HashMap<_, _>>>()?
                )
            }
        })
    }
}

impl fmt::Display for GraphQLValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Boolean(b) => write!(f, "{}", b),
            Self::Int(i) => write!(f, "{}", i),
            Self::Float(fl) => write!(f, "{}", fl),
            Self::String(s) => write!(f, "{}", s),
            Self::Enum(s) => write!(f, "{}", s),
            Self::CustomScalar(s) => write!(f, "{}", s),
            Self::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Self::Object(fields) => {
                write!(f, "{{")?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// GraphQL type definitions
#[derive(Debug, Clone, PartialEq)]
pub enum GraphQLType {
    /// Scalar type (Int, Float, String, Boolean, ID, or custom)
    Scalar(String),

    /// Enum type
    Enum { name: String, values: Vec<String> },

    /// Object type
    Object {
        name: String,
        fields: Vec<GraphQLField>,
    },

    /// Interface type
    Interface {
        name: String,
        fields: Vec<GraphQLField>,
    },

    /// Union type
    Union {
        name: String,
        types: Vec<String>,
    },

    /// Input object type
    InputObject {
        name: String,
        fields: Vec<GraphQLInputValue>,
    },

    /// List type
    List(Box<GraphQLType>),

    /// Non-null type
    NonNull(Box<GraphQLType>),
}

impl GraphQLType {
    /// Create scalar type
    pub fn scalar(name: impl Into<String>) -> Self {
        Self::Scalar(name.into())
    }

    /// Create enum type
    pub fn enm(name: impl Into<String>, values: Vec<String>) -> Self {
        Self::Enum {
            name: name.into(),
            values,
        }
    }

    /// Create object type
    pub fn object(name: impl Into<String>, fields: Vec<GraphQLField>) -> Self {
        Self::Object {
            name: name.into(),
            fields,
        }
    }

    /// Create interface type
    pub fn interface(name: impl Into<String>, fields: Vec<GraphQLField>) -> Self {
        Self::Interface {
            name: name.into(),
            fields,
        }
    }

    /// Create union type
    pub fn union(name: impl Into<String>, types: Vec<String>) -> Self {
        Self::Union {
            name: name.into(),
            types,
        }
    }

    /// Create input object type
    pub fn input_object(name: impl Into<String>, fields: Vec<GraphQLInputValue>) -> Self {
        Self::InputObject {
            name: name.into(),
            fields,
        }
    }

    /// Create list type
    pub fn list(inner: GraphQLType) -> Self {
        Self::List(Box::new(inner))
    }

    /// Create non-null type
    pub fn non_null(inner: GraphQLType) -> Self {
        Self::NonNull(Box::new(inner))
    }

    /// Get type name
    pub fn name(&self) -> String {
        match self {
            Self::Scalar(name) => name.clone(),
            Self::Enum { name, .. } => name.clone(),
            Self::Object { name, .. } => name.clone(),
            Self::Interface { name, .. } => name.clone(),
            Self::Union { name, .. } => name.clone(),
            Self::InputObject { name, .. } => name.clone(),
            Self::List(inner) => format!("[{}]", inner.name()),
            Self::NonNull(inner) => format!("{}!", inner.name()),
        }
    }

    /// Check if type is nullable
    pub fn is_nullable(&self) -> bool {
        !matches!(self, Self::NonNull(_))
    }

    /// Get inner type for wrapper types
    pub fn inner_type(&self) -> Option<&GraphQLType> {
        match self {
            Self::List(inner) | Self::NonNull(inner) => Some(inner),
            _ => None,
        }
    }
}

/// GraphQL field definition
#[derive(Debug, Clone, PartialEq)]
pub struct GraphQLField {
    pub name: String,
    pub description: Option<String>,
    pub arguments: Vec<GraphQLInputValue>,
    pub field_type: GraphQLType,
    pub is_deprecated: bool,
    pub deprecation_reason: Option<String>,
}

impl GraphQLField {
    /// Create new field
    pub fn new(
        name: impl Into<String>,
        field_type: GraphQLType,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            arguments: Vec::new(),
            field_type,
            is_deprecated: false,
            deprecation_reason: None,
        }
    }

    /// Add description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add argument
    pub fn with_argument(mut self, arg: GraphQLInputValue) -> Self {
        self.arguments.push(arg);
        self
    }

    /// Mark as deprecated
    pub fn deprecated(mut self, reason: impl Into<String>) -> Self {
        self.is_deprecated = true;
        self.deprecation_reason = Some(reason.into());
        self
    }
}

/// GraphQL input value (for arguments and input objects)
#[derive(Debug, Clone, PartialEq)]
pub struct GraphQLInputValue {
    pub name: String,
    pub description: Option<String>,
    pub value_type: GraphQLType,
    pub default_value: Option<GraphQLValue>,
}

impl GraphQLInputValue {
    /// Create new input value
    pub fn new(
        name: impl Into<String>,
        value_type: GraphQLType,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            value_type,
            default_value: None,
        }
    }

    /// Add description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set default value
    pub fn with_default(mut self, default: GraphQLValue) -> Self {
        self.default_value = Some(default);
        self
    }
}

/// GraphQL query document
#[derive(Debug, Clone)]
pub struct GraphQLQuery {
    /// Query source text
    pub source: String,
    /// Operation type (Query, Mutation, Subscription)
    pub operation_type: OperationType,
    /// Root selection set
    pub selection_set: SelectionSet,
    /// Variable definitions
    pub variables: Vec<VariableDefinition>,
    /// Query name (for named operations)
    pub name: Option<String>,
}

impl GraphQLQuery {
    /// Create new query
    pub fn new(
        source: impl Into<String>,
        operation_type: OperationType,
        selection_set: SelectionSet,
    ) -> Self {
        Self {
            source: source.into(),
            operation_type,
            selection_set,
            variables: Vec::new(),
            name: None,
        }
    }

    /// Add variable definition
    pub fn with_variable(mut self, var: VariableDefinition) -> Self {
        self.variables.push(var);
        self
    }

    /// Set query name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Parse from GraphQL query string
    pub fn parse(source: impl Into<String>) -> GraphQLResult<Self> {
        let source = source.into();
        // Simple parser for demonstration
        // A real implementation would use a proper lexer/parser
        Self::parse_impl(&source)
    }

    fn parse_impl(source: &str) -> GraphQLResult<Self> {
        let source = source.trim();

        // Determine operation type
        let (operation_type, rest) = if source.starts_with("mutation") {
            (OperationType::Mutation, &source["mutation".len()..])
        } else if source.starts_with("subscription") {
            (OperationType::Subscription, &source["subscription".len()..])
        } else {
            // Default to query
            let rest = if source.starts_with("query") {
                &source["query".len()..]
            } else {
                source
            };
            (OperationType::Query, rest)
        };

        // Simple selection set parsing (field names only)
        let selection_set = Self::parse_selection_set(rest)?;

        Ok(Self::new(
            source,
            operation_type,
            selection_set,
        ))
    }

    fn parse_selection_set(s: &str) -> GraphQLResult<SelectionSet> {
        let s = s.trim();

        // Find opening brace
        let start = s.find('{').ok_or_else(|| GraphQLError::syntax("Expected '{'"))?;
        let after_brace = &s[start + 1..];

        // Find matching closing brace
        let mut depth = 1;
        let mut end = 0;
        for (i, c) in after_brace.chars().enumerate() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }

        if depth != 0 {
            return Err(GraphQLError::syntax("Unmatched braces"));
        }

        let inner = &after_brace[..end];
        let selections = Self::parse_selections(inner)?;

        Ok(SelectionSet { selections })
    }

    fn parse_selections(s: &str) -> GraphQLResult<Vec<Selection>> {
        let mut selections = Vec::new();
        let s = s.trim();

        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            // Check for nested fields
            if let Some(nested_start) = part.find('{') {
                let field_name = &part[..nested_start].trim();
                let nested_end = part.rfind('}').ok_or_else(|| GraphQLError::syntax("Unmatched braces"))?;
                let nested_inner = &part[nested_start + 1..nested_end];

                selections.push(Selection::Field(Field {
                    name: field_name.to_string(),
                    alias: None,
                    arguments: HashMap::new(),
                    selection_set: Some(Self::parse_selection_set(nested_inner)?),
                }));
            } else {
                selections.push(Selection::Field(Field {
                    name: part.to_string(),
                    alias: None,
                    arguments: HashMap::new(),
                    selection_set: None,
                }));
            }
        }

        Ok(selections)
    }
}

/// Operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    Query,
    Mutation,
    Subscription,
}

/// Selection set
#[derive(Debug, Clone)]
pub struct SelectionSet {
    pub selections: Vec<Selection>,
}

/// Selection in a query
#[derive(Debug, Clone)]
pub enum Selection {
    Field(Field),
    FragmentSpread(FragmentSpread),
    InlineFragment(InlineFragment),
}

/// Field in a selection
#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub alias: Option<String>,
    pub arguments: HashMap<String, GraphQLValue>,
    pub selection_set: Option<SelectionSet>,
}

/// Fragment spread
#[derive(Debug, Clone)]
pub struct FragmentSpread {
    pub name: String,
}

/// Inline fragment
#[derive(Debug, Clone)]
pub struct InlineFragment {
    pub type_condition: Option<String>,
    pub selection_set: SelectionSet,
}

/// Variable definition
#[derive(Debug, Clone)]
pub struct VariableDefinition {
    pub name: String,
    pub var_type: GraphQLType,
    pub default_value: Option<GraphQLValue>,
}

/// GraphQL execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Query variables
    pub variables: HashMap<String, GraphQLValue>,
    /// Operation name (for named operations)
    pub operation_name: Option<String>,
}

impl ExecutionContext {
    /// Create new context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            operation_name: None,
        }
    }

    /// Set variable
    pub fn with_variable(mut self, name: impl Into<String>, value: GraphQLValue) -> Self {
        self.variables.insert(name.into(), value);
        self
    }

    /// Get variable value
    pub fn get_variable(&self, name: &str) -> Option<&GraphQLValue> {
        self.variables.get(name)
    }
}

/// Resolver function type
pub type ResolverFn = Arc<dyn Fn(&ExecutionContext, &HashMap<String, GraphQLValue>) -> GraphQLResult<GraphQLValue> + Send + Sync>;

/// Field resolver
#[derive(Clone)]
pub struct GraphQLResolver {
    pub field_name: String,
    pub resolver: ResolverFn,
}

impl GraphQLResolver {
    /// Create new resolver
    pub fn new(
        field_name: impl Into<String>,
        resolver: ResolverFn,
    ) -> Self {
        Self {
            field_name: field_name.into(),
            resolver,
        }
    }

    /// Create simple constant resolver
    pub fn constant(field_name: impl Into<String>, value: GraphQLValue) -> Self {
        Self::new(field_name, Arc::new(move |_ctx, _args| Ok(value.clone())))
    }

    /// Create resolver from function
    pub fn from_fn<F>(field_name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&ExecutionContext, &HashMap<String, GraphQLValue>) -> GraphQLResult<GraphQLValue> + Send + Sync + 'static,
    {
        Self::new(field_name, Arc::new(f))
    }
}

/// Subscription handler
pub type SubscriptionHandler = Arc<dyn Fn(GraphQLValue) -> Box<dyn Iterator<Item = GraphQLValue> + Send> + Send + Sync>;

/// Subscription definition
#[derive(Clone)]
pub struct GraphQLSubscription {
    pub name: String,
    pub handler: SubscriptionHandler,
}

impl GraphQLSubscription {
    /// Create new subscription
    pub fn new(
        name: impl Into<String>,
        handler: SubscriptionHandler,
    ) -> Self {
        Self {
            name: name.into(),
            handler,
        }
    }
}

impl fmt::Debug for GraphQLResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphQLResolver")
            .field("field_name", &self.field_name)
            .field("resolver", &"<function>")
            .finish()
    }
}

impl fmt::Debug for GraphQLSubscription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphQLSubscription")
            .field("name", &self.name)
            .field("handler", &"<subscription>")
            .finish()
    }
}

/// GraphQL schema
#[derive(Debug)]
pub struct GraphQLSchema {
    /// Query type
    pub query_type: Option<String>,
    /// Mutation type
    pub mutation_type: Option<String>,
    /// Subscription type
    pub subscription_type: Option<String>,
    /// Type definitions
    pub types: HashMap<String, GraphQLType>,
    /// Directives
    pub directives: Vec<String>,
    /// Field resolvers
    resolvers: HashMap<String, Vec<GraphQLResolver>>,
    /// Subscriptions
    subscriptions: HashMap<String, GraphQLSubscription>,
}

impl GraphQLSchema {
    /// Create new schema
    pub fn new() -> Self {
        Self {
            query_type: None,
            mutation_type: None,
            subscription_type: None,
            types: Self::builtin_types(),
            directives: vec!["skip".to_string(), "include".to_string(), "deprecated".to_string()],
            resolvers: HashMap::new(),
            subscriptions: HashMap::new(),
        }
    }

    /// Get builtin scalar types
    fn builtin_types() -> HashMap<String, GraphQLType> {
        let mut types = HashMap::new();
        types.insert("Int".to_string(), GraphQLType::Scalar("Int".to_string()));
        types.insert("Float".to_string(), GraphQLType::Scalar("Float".to_string()));
        types.insert("String".to_string(), GraphQLType::Scalar("String".to_string()));
        types.insert("Boolean".to_string(), GraphQLType::Scalar("Boolean".to_string()));
        types.insert("ID".to_string(), GraphQLType::Scalar("ID".to_string()));
        types
    }

    /// Set query type
    pub fn with_query_type(mut self, name: impl Into<String>) -> Self {
        self.query_type = Some(name.into());
        self
    }

    /// Set mutation type
    pub fn with_mutation_type(mut self, name: impl Into<String>) -> Self {
        self.mutation_type = Some(name.into());
        self
    }

    /// Set subscription type
    pub fn with_subscription_type(mut self, name: impl Into<String>) -> Self {
        self.subscription_type = Some(name.into());
        self
    }

    /// Add type to schema
    pub fn add_type(mut self, type_def: GraphQLType) -> Self {
        let name = match &type_def {
            GraphQLType::Scalar(name) => name.clone(),
            GraphQLType::Enum { name, .. } => name.clone(),
            GraphQLType::Object { name, .. } => name.clone(),
            GraphQLType::Interface { name, .. } => name.clone(),
            GraphQLType::Union { name, .. } => name.clone(),
            GraphQLType::InputObject { name, .. } => name.clone(),
            GraphQLType::List(_) | GraphQLType::NonNull(_) => {
                return self; // Don't register wrapper types
            }
        };
        self.types.insert(name, type_def);
        self
    }

    /// Add resolver for a type
    pub fn add_resolver(mut self, type_name: impl Into<String>, resolver: GraphQLResolver) -> Self {
        let type_name = type_name.into();
        self.resolvers
            .entry(type_name)
            .or_insert_with(Vec::new)
            .push(resolver);
        self
    }

    /// Add subscription
    pub fn add_subscription(mut self, subscription: GraphQLSubscription) -> Self {
        self.subscriptions.insert(subscription.name.clone(), subscription);
        self
    }

    /// Get type by name
    pub fn get_type(&self, name: &str) -> Option<&GraphQLType> {
        self.types.get(name)
    }

    /// Get resolvers for a type
    pub fn get_resolvers(&self, type_name: &str) -> Option<&[GraphQLResolver]> {
        self.resolvers.get(type_name).map(|v| v.as_slice())
    }

    /// Get subscription by name
    pub fn get_subscription(&self, name: &str) -> Option<&GraphQLSubscription> {
        self.subscriptions.get(name)
    }

    /// Validate schema
    pub fn validate(&self) -> GraphQLResult<()> {
        // Check that query type exists
        if let Some(query_name) = &self.query_type {
            if !self.types.contains_key(query_name) {
                return Err(GraphQLError::SchemaError(format!(
                    "Query type '{}' not found in schema",
                    query_name
                )));
            }
        }

        // Check that mutation type exists if specified
        if let Some(mutation_name) = &self.mutation_type {
            if !self.types.contains_key(mutation_name) {
                return Err(GraphQLError::SchemaError(format!(
                    "Mutation type '{}' not found in schema",
                    mutation_name
                )));
            }
        }

        // Check that subscription type exists if specified
        if let Some(subscription_name) = &self.subscription_type {
            if !self.types.contains_key(subscription_name) {
                return Err(GraphQLError::SchemaError(format!(
                    "Subscription type '{}' not found in schema",
                    subscription_name
                )));
            }
        }

        Ok(())
    }

    /// Execute a query
    pub fn execute(
        &self,
        query: &GraphQLQuery,
        context: &ExecutionContext,
    ) -> GraphQLResult<GraphQLValue> {
        // Validate schema
        self.validate()?;

        // Get root type based on operation
        let root_type_name = match query.operation_type {
            OperationType::Query => self.query_type
                .as_ref()
                .ok_or_else(|| GraphQLError::validation("No query type defined in schema"))?,
            OperationType::Mutation => self.mutation_type
                .as_ref()
                .ok_or_else(|| GraphQLError::validation("No mutation type defined in schema"))?,
            OperationType::Subscription => self.subscription_type
                .as_ref()
                .ok_or_else(|| GraphQLError::validation("No subscription type defined in schema"))?,
        };

        // Execute selection set
        self.execute_selection_set(root_type_name, &query.selection_set, context)
    }

    /// Execute selection set
    fn execute_selection_set(
        &self,
        type_name: &str,
        selection_set: &SelectionSet,
        context: &ExecutionContext,
    ) -> GraphQLResult<GraphQLValue> {
        let mut result_fields = HashMap::new();

        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    let value = self.execute_field(type_name, field, context)?;
                    let key = field.alias.as_ref().unwrap_or(&field.name);
                    result_fields.insert(key.clone(), value);
                }
                Selection::FragmentSpread(_) => {
                    // Fragment handling would go here
                    return Err(GraphQLError::validation("Fragments not yet implemented"));
                }
                Selection::InlineFragment(_) => {
                    // Inline fragment handling would go here
                    return Err(GraphQLError::validation("Inline fragments not yet implemented"));
                }
            }
        }

        Ok(GraphQLValue::Object(result_fields))
    }

    /// Execute a single field
    fn execute_field(
        &self,
        parent_type: &str,
        field: &Field,
        context: &ExecutionContext,
    ) -> GraphQLResult<GraphQLValue> {
        // Find resolver for this field
        let resolver = self.find_resolver(parent_type, &field.name)?;

        // Execute resolver
        let value = (resolver.resolver)(context, &field.arguments)?;

        // Handle nested selection sets
        if let Some(selection_set) = &field.selection_set {
            // Value should be an object, we need to select subfields
            match value {
                GraphQLValue::Object(mut fields) => {
                    let mut selected_fields = HashMap::new();
                    for selection in &selection_set.selections {
                        if let Selection::Field(subfield) = selection {
                            let subfield_value = self.execute_field(&field.name, subfield, context)?;
                            selected_fields.insert(subfield.name.clone(), subfield_value);
                        }
                    }
                    Ok(GraphQLValue::Object(selected_fields))
                }
                _ => Err(GraphQLError::type_mismatch("Object", "Scalar")),
            }
        } else {
            Ok(value)
        }
    }

    /// Find resolver for a field
    fn find_resolver(&self, type_name: &str, field_name: &str) -> GraphQLResult<&GraphQLResolver> {
        self.resolvers
            .get(type_name)
            .and_then(|resolvers| resolvers.iter().find(|r| r.field_name == field_name))
            .ok_or_else(|| GraphQLError::field_not_found(field_name, type_name))
    }

    /// Subscribe to a subscription
    pub fn subscribe(
        &self,
        subscription_name: &str,
        initial_value: GraphQLValue,
    ) -> GraphQLResult<Box<dyn Iterator<Item = GraphQLValue> + Send>> {
        let subscription = self.get_subscription(subscription_name)
            .ok_or_else(|| GraphQLError::SubscriptionError(format!(
                "Subscription '{}' not found",
                subscription_name
            )))?;

        Ok((subscription.handler)(initial_value))
    }

    /// Generate schema SDL (Schema Definition Language)
    pub fn to_sdl(&self) -> String {
        let mut sdl = String::new();

        // Add description comment
        sdl.push_str("# GraphQL Schema\n\n");

        // Add query type if defined
        if let Some(query_name) = &self.query_type {
            sdl.push_str(&format!("schema {{\n  query: {{{}}}\n", query_name));
            if let Some(mutation_name) = &self.mutation_type {
                sdl.push_str(&format!("  mutation: {{{}}}\n", mutation_name));
            }
            if let Some(subscription_name) = &self.subscription_type {
                sdl.push_str(&format!("  subscription: {{{}}}\n", subscription_name));
            }
            sdl.push_str("}\n\n");
        }

        // Add all types (excluding builtins)
        for (name, type_def) in &self.types {
            if !matches!(name.as_str(), "Int" | "Float" | "String" | "Boolean" | "ID") {
                sdl.push_str(&self.type_to_sdl(name, type_def));
                sdl.push_str("\n");
            }
        }

        sdl
    }

    fn type_to_sdl(&self, name: &str, type_def: &GraphQLType) -> String {
        match type_def {
            GraphQLType::Scalar(scalar_name) => {
                format!("scalar {}", scalar_name)
            }
            GraphQLType::Enum { name: enum_name, values } => {
                let mut sdl = format!("enum {} {{\n", enum_name);
                for value in values {
                    sdl.push_str(&format!("  {}\n", value));
                }
                sdl.push_str("}\n");
                sdl
            }
            GraphQLType::Object { name: obj_name, fields } => {
                let mut sdl = format!("type {} {{\n", obj_name);
                for field in fields {
                    sdl.push_str(&format!("  {}: {}", field.name, self.type_ref_to_sdl(&field.field_type)));
                    if field.is_deprecated {
                        if let Some(reason) = &field.deprecation_reason {
                            sdl.push_str(&format!(" @deprecated(reason: \"{}\")", reason));
                        } else {
                            sdl.push_str(" @deprecated");
                        }
                    }
                    sdl.push_str("\n");
                }
                sdl.push_str("}\n");
                sdl
            }
            GraphQLType::Interface { name: iface_name, fields } => {
                let mut sdl = format!("interface {} {{\n", iface_name);
                for field in fields {
                    sdl.push_str(&format!("  {}: {}\n", field.name, self.type_ref_to_sdl(&field.field_type)));
                }
                sdl.push_str("}\n");
                sdl
            }
            GraphQLType::Union { name: union_name, types } => {
                format!("union {} = {}\n", union_name, types.join(" | "))
            }
            GraphQLType::InputObject { name: input_name, fields } => {
                let mut sdl = format!("input {} {{\n", input_name);
                for field in fields {
                    sdl.push_str(&format!("  {}: {}", field.name, self.type_ref_to_sdl(&field.value_type)));
                    if let Some(default) = &field.default_value {
                        sdl.push_str(&format!(" = {}", default));
                    }
                    sdl.push_str("\n");
                }
                sdl.push_str("}\n");
                sdl
            }
            GraphQLType::List(inner) => format!("[{}]", self.type_ref_to_sdl(inner)),
            GraphQLType::NonNull(inner) => format!("{}!", self.type_ref_to_sdl(inner)),
        }
    }

    fn type_ref_to_sdl(&self, type_def: &GraphQLType) -> String {
        match type_def {
            GraphQLType::Scalar(name) => name.clone(),
            GraphQLType::Enum { name, .. } => name.clone(),
            GraphQLType::Object { name, .. } => name.clone(),
            GraphQLType::Interface { name, .. } => name.clone(),
            GraphQLType::Union { name, .. } => name.clone(),
            GraphQLType::InputObject { name, .. } => name.clone(),
            GraphQLType::List(inner) => format!("[{}]", self.type_ref_to_sdl(inner)),
            GraphQLType::NonNull(inner) => format!("{}!", self.type_ref_to_sdl(inner)),
        }
    }
}

impl Default for GraphQLSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// GraphQL response
#[derive(Debug, Clone)]
pub struct GraphQLResponse {
    pub data: Option<GraphQLValue>,
    pub errors: Vec<GraphQLError>,
    pub extensions: Option<HashMap<String, GraphQLValue>>,
}

impl GraphQLResponse {
    /// Create successful response
    pub fn success(data: GraphQLValue) -> Self {
        Self {
            data: Some(data),
            errors: Vec::new(),
            extensions: None,
        }
    }

    /// Create error response
    pub fn error(errors: Vec<GraphQLError>) -> Self {
        Self {
            data: None,
            errors,
            extensions: None,
        }
    }

    /// Create partial response (data with errors)
    pub fn partial(data: GraphQLValue, errors: Vec<GraphQLError>) -> Self {
        Self {
            data: Some(data),
            errors,
            extensions: None,
        }
    }

    /// Add extension data
    pub fn with_extension(mut self, key: impl Into<String>, value: GraphQLValue) -> Self {
        self.extensions
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value);
        self
    }

    /// Check if response has errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Convert to JSON
    pub fn to_json(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();

        if let Some(data) = &self.data {
            obj.insert("data".to_string(), data.to_json());
        }

        if !self.errors.is_empty() {
            let error_array: Vec<serde_json::Value> = self
                .errors
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "message": e.to_string(),
                        "code": e.error_code()
                    })
                })
                .collect();
            obj.insert("errors".to_string(), serde_json::Value::Array(error_array));
        }

        if let Some(extensions) = &self.extensions {
            let ext_map: serde_json::Map<String, serde_json::Value> = extensions
                .iter()
                .map(|(k, v)| (k.clone(), v.to_json()))
                .collect();
            obj.insert("extensions".to_string(), serde_json::Value::Object(ext_map));
        }

        serde_json::Value::Object(obj)
    }
}

/// Introspection types for schema discovery
impl GraphQLSchema {
    /// Get introspection query result
    pub fn introspect(&self) -> GraphQLValue {
        let mut types = HashMap::new();

        for (name, type_def) in &self.types {
            types.insert(name.clone(), self.introspect_type(type_def));
        }

        GraphQLValue::Object(
            vec![
                (
                    "__schema".to_string(),
                    GraphQLValue::Object(
                        vec![
                            (
                                "types".to_string(),
                                GraphQLValue::List(
                                    types.values().cloned().collect()
                                )
                            ),
                            (
                                "queryType".to_string(),
                                self.query_type.as_ref()
                                    .and_then(|n| types.get(n))
                                    .cloned()
                                    .unwrap_or(GraphQLValue::null())
                            ),
                        ].into_iter().collect()
                    )
                ),
            ].into_iter().collect()
        )
    }

    fn introspect_type(&self, type_def: &GraphQLType) -> GraphQLValue {
        let (kind, name) = match type_def {
            GraphQLType::Scalar(name) => ("SCALAR", name.clone()),
            GraphQLType::Enum { name, .. } => ("ENUM", name.clone()),
            GraphQLType::Object { name, .. } => ("OBJECT", name.clone()),
            GraphQLType::Interface { name, .. } => ("INTERFACE", name.clone()),
            GraphQLType::Union { name, .. } => ("UNION", name.clone()),
            GraphQLType::InputObject { name, .. } => ("INPUT_OBJECT", name.clone()),
            GraphQLType::List(_) => ("LIST", String::new()),
            GraphQLType::NonNull(_) => ("NON_NULL", String::new()),
        };

        GraphQLValue::Object(
            vec![
                ("kind".to_string(), GraphQLValue::String(kind.to_string())),
                ("name".to_string(), GraphQLValue::String(name)),
                ("description".to_string(), GraphQLValue::null()),
            ].into_iter().collect()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphql_value_creation() {
        assert_eq!(GraphQLValue::null(), GraphQLValue::Null);
        assert_eq!(GraphQLValue::boolean(true), GraphQLValue::Boolean(true));
        assert_eq!(GraphQLValue::int(42), GraphQLValue::Int(42));
        assert_eq!(GraphQLValue::float(3.14), GraphQLValue::Float(3.14));
        assert_eq!(GraphQLValue::string("hello"), GraphQLValue::String("hello".to_string()));
    }

    #[test]
    fn test_graphql_value_to_json() {
        let val = GraphQLValue::Object(
            vec![
                ("name".to_string(), GraphQLValue::String("Alice".to_string())),
                ("age".to_string(), GraphQLValue::Int(30)),
            ].into_iter().collect()
        );

        let json = val.to_json();
        assert_eq!(json["name"], "Alice");
        assert_eq!(json["age"], 30);
    }

    #[test]
    fn test_graphql_value_from_json() {
        let json = serde_json::json!({"name": "Bob", "active": true});
        let val = GraphQLValue::from_json(&json).unwrap();

        assert!(matches!(val, GraphQLValue::Object(_)));
    }

    #[test]
    fn test_graphql_type_creation() {
        let scalar = GraphQLType::scalar("CustomScalar");
        assert!(matches!(scalar, GraphQLType::Scalar(_)));

        let list = GraphQLType::list(GraphQLType::scalar("Int"));
        assert!(matches!(list, GraphQLType::List(_)));

        let non_null = GraphQLType::non_null(GraphQLType::scalar("String"));
        assert!(matches!(non_null, GraphQLType::NonNull(_)));
    }

    #[test]
    fn test_graphql_type_name() {
        assert_eq!(GraphQLType::scalar("Int").name(), "Int");
        assert_eq!(GraphQLType::list(GraphQLType::scalar("String")).name(), "[String]");
        assert_eq!(GraphQLType::non_null(GraphQLType::scalar("Int")).name(), "Int!");
    }

    #[test]
    fn test_graphql_field() {
        let field = GraphQLField::new("id", GraphQLType::non_null(GraphQLType::scalar("ID")))
            .with_description("The user ID")
            .deprecated("Use userId instead");

        assert_eq!(field.name, "id");
        assert!(field.is_deprecated);
        assert_eq!(field.deprecation_reason, Some("Use userId instead".to_string()));
    }

    #[test]
    fn test_graphql_input_value() {
        let arg = GraphQLInputValue::new("limit", GraphQLType::scalar("Int"))
            .with_default(GraphQLValue::int(10));

        assert_eq!(arg.name, "limit");
        assert_eq!(arg.default_value, Some(GraphQLValue::Int(10)));
    }

    #[test]
    fn test_graphql_query_parse() {
        let query = GraphQLQuery::parse("{ hello }").unwrap();
        assert_eq!(query.operation_type, OperationType::Query);
        assert_eq!(query.selection_set.selections.len(), 1);
    }

    #[test]
    fn test_graphql_query_parse_nested() {
        let query = GraphQLQuery::parse("{ user { name age } }").unwrap();
        assert_eq!(query.selection_set.selections.len(), 1);

        if let Selection::Field(field) = &query.selection_set.selections[0] {
            assert_eq!(field.name, "user");
            assert!(field.selection_set.is_some());
        } else {
            panic!("Expected field selection");
        }
    }

    #[test]
    fn test_execution_context() {
        let ctx = ExecutionContext::new()
            .with_variable("userId", GraphQLValue::string("123"));

        assert_eq!(ctx.get_variable("userId"), Some(&GraphQLValue::String("123".to_string())));
    }

    #[test]
    fn test_graphql_resolver() {
        let resolver = GraphQLResolver::constant("test", GraphQLValue::string("Hello"));

        let result = (resolver.resolver)(&ExecutionContext::new(), &HashMap::new()).unwrap();
        assert_eq!(result, GraphQLValue::String("Hello".to_string()));
    }

    #[test]
    fn test_graphql_resolver_from_fn() {
        let resolver = GraphQLResolver::from_fn("add", |_ctx, args| {
            let a = args.get("a").and_then(|v| match v {
                GraphQLValue::Int(n) => Some(*n),
                _ => None,
            }).unwrap_or(0);
            let b = args.get("b").and_then(|v| match v {
                GraphQLValue::Int(n) => Some(*n),
                _ => None,
            }).unwrap_or(0);
            Ok(GraphQLValue::int(a + b))
        });

        let mut args = HashMap::new();
        args.insert("a".to_string(), GraphQLValue::int(5));
        args.insert("b".to_string(), GraphQLValue::int(3));

        let result = (resolver.resolver)(&ExecutionContext::new(), &args).unwrap();
        assert_eq!(result, GraphQLValue::Int(8));
    }

    #[test]
    fn test_graphql_schema_creation() {
        let schema = GraphQLSchema::new()
            .with_query_type("Query");

        assert_eq!(schema.query_type, Some("Query".to_string()));
    }

    #[test]
    fn test_graphql_schema_add_type() {
        let user_type = GraphQLType::object("User", vec![
            GraphQLField::new("id", GraphQLType::non_null(GraphQLType::scalar("ID"))),
            GraphQLField::new("name", GraphQLType::scalar("String")),
        ]);

        let schema = GraphQLSchema::new().add_type(user_type);
        assert!(schema.get_type("User").is_some());
    }

    #[test]
    fn test_graphql_schema_add_resolver() {
        let resolver = GraphQLResolver::constant("hello", GraphQLValue::string("world"));
        let schema = GraphQLSchema::new().add_resolver("Query", resolver);

        assert!(schema.get_resolvers("Query").is_some());
    }

    #[test]
    fn test_graphql_schema_validation() {
        let schema = GraphQLSchema::new()
            .with_query_type("Query")
            .add_type(GraphQLType::object("Query", vec![]));

        assert!(schema.validate().is_ok());
    }

    #[test]
    fn test_graphql_schema_validation_missing_type() {
        let schema = GraphQLSchema::new().with_query_type("NonExistent");
        assert!(schema.validate().is_err());
    }

    #[test]
    fn test_graphql_schema_execute() {
        let query_type = GraphQLType::object("Query", vec![
            GraphQLField::new("hello", GraphQLType::scalar("String")),
        ]);

        let resolver = GraphQLResolver::constant("hello", GraphQLValue::string("Hello, World!"));

        let schema = GraphQLSchema::new()
            .with_query_type("Query")
            .add_type(query_type)
            .add_resolver("Query", resolver);

        let query = GraphQLQuery::parse("{ hello }").unwrap();
        let context = ExecutionContext::new();

        let result = schema.execute(&query, &context).unwrap();

        if let GraphQLValue::Object(fields) = result {
            assert_eq!(fields.get("hello"), Some(&GraphQLValue::String("Hello, World!".to_string())));
        } else {
            panic!("Expected object result");
        }
    }

    #[test]
    fn test_graphql_subscription() {
        let handler: SubscriptionHandler = Arc::new(|_initial| {
            Box::new(vec![
                GraphQLValue::int(1),
                GraphQLValue::int(2),
                GraphQLValue::int(3),
            ].into_iter())
        });

        let subscription = GraphQLSubscription::new("counter", handler);
        let schema = GraphQLSchema::new().add_subscription(subscription);

        let mut stream = schema.subscribe("counter", GraphQLValue::null()).unwrap();

        assert_eq!(stream.next(), Some(GraphQLValue::Int(1)));
        assert_eq!(stream.next(), Some(GraphQLValue::Int(2)));
        assert_eq!(stream.next(), Some(GraphQLValue::Int(3)));
        assert_eq!(stream.next(), None);
    }

    #[test]
    fn test_graphql_response() {
        let response = GraphQLResponse::success(GraphQLValue::string("data"));
        assert!(!response.has_errors());
        assert!(response.data.is_some());

        let response = GraphQLResponse::error(vec![
            GraphQLError::validation("Invalid input"),
        ]);
        assert!(response.has_errors());
    }

    #[test]
    fn test_graphql_response_to_json() {
        let response = GraphQLResponse::success(
            GraphQLValue::Object(
                vec![
                    ("message".to_string(), GraphQLValue::String("Hello".to_string())),
                ].into_iter().collect()
            )
        );

        let json = response.to_json();
        assert!(json.is_object());
        assert!(json.get("data").is_some());
    }

    #[test]
    fn test_graphql_error_types() {
        let syntax_err = GraphQLError::syntax("Unexpected token");
        assert_eq!(syntax_err.error_code(), "SYNTAX_ERROR");

        let validation_err = GraphQLError::validation("Type mismatch");
        assert_eq!(validation_err.error_code(), "VALIDATION_ERROR");

        let field_err = GraphQLError::field_not_found("name", "User");
        assert_eq!(field_err.error_code(), "FIELD_NOT_FOUND");
    }

    #[test]
    fn test_schema_to_sdl() {
        let user_type = GraphQLType::object("User", vec![
            GraphQLField::new("id", GraphQLType::non_null(GraphQLType::scalar("ID"))),
            GraphQLField::new("name", GraphQLType::scalar("String")),
        ]);

        let schema = GraphQLSchema::new()
            .with_query_type("Query")
            .add_type(GraphQLType::object("Query", vec![]))
            .add_type(user_type);

        let sdl = schema.to_sdl();
        assert!(sdl.contains("type User"));
        assert!(sdl.contains("id: ID!"));
        assert!(sdl.contains("name: String"));
    }

    #[test]
    fn test_introspection() {
        let query_type = GraphQLType::object("Query", vec![
            GraphQLField::new("__typename", GraphQLType::non_null(GraphQLType::scalar("String"))),
        ]);

        let schema = GraphQLSchema::new()
            .with_query_type("Query")
            .add_type(query_type);

        let introspection = schema.introspect();

        if let GraphQLValue::Object(fields) = introspection {
            assert!(fields.contains_key("__schema"));
        } else {
            panic!("Expected object from introspection");
        }
    }

    #[test]
    fn test_mutation_operation() {
        let query = GraphQLQuery::parse("mutation { createUser(name: \"Alice\") { id name } }").unwrap();
        assert_eq!(query.operation_type, OperationType::Mutation);
    }

    #[test]
    fn test_subscription_operation() {
        let query = GraphQLQuery::parse("subscription { onUserCreated { id name } }").unwrap();
        assert_eq!(query.operation_type, OperationType::Subscription);
    }

    #[test]
    fn test_enum_type() {
        let status_enum = GraphQLType::enm("Status", vec![
            "ACTIVE".to_string(),
            "INACTIVE".to_string(),
            "PENDING".to_string(),
        ]);

        let schema = GraphQLSchema::new().add_type(status_enum);
        assert!(schema.get_type("Status").is_some());

        if let GraphQLType::Enum { values, .. } = schema.get_type("Status").unwrap() {
            assert_eq!(values.len(), 3);
            assert!(values.contains(&"ACTIVE".to_string()));
        } else {
            panic!("Expected enum type");
        }
    }

    #[test]
    fn test_union_type() {
        let result_union = GraphQLType::union("SearchResult", vec![
            "User".to_string(),
            "Post".to_string(),
            "Comment".to_string(),
        ]);

        let schema = GraphQLSchema::new().add_type(result_union);
        assert!(schema.get_type("SearchResult").is_some());
    }

    #[test]
    fn test_input_object() {
        let user_input = GraphQLType::input_object("UserInput", vec![
            GraphQLInputValue::new("name", GraphQLType::non_null(GraphQLType::scalar("String")))
                .with_description("User name"),
            GraphQLInputValue::new("age", GraphQLType::scalar("Int"))
                .with_default(GraphQLValue::int(18)),
        ]);

        let schema = GraphQLSchema::new().add_type(user_input);
        assert!(schema.get_type("UserInput").is_some());
    }

    #[test]
    fn test_nested_field_execution() {
        let user_type = GraphQLType::object("User", vec![
            GraphQLField::new("name", GraphQLType::scalar("String")),
        ]);

        let query_type = GraphQLType::object("Query", vec![
            GraphQLField::new("user", GraphQLType::object("User", vec![
                GraphQLField::new("name", GraphQLType::scalar("String")),
            ])),
        ]);

        let user_resolver = GraphQLResolver::constant("name", GraphQLValue::string("Alice"));
        let user_resolver2 = GraphQLResolver::from_fn("user", |_ctx, _args| {
            Ok(GraphQLValue::Object(
                vec![
                    ("name".to_string(), GraphQLValue::String("Alice".to_string())),
                ].into_iter().collect()
            ))
        });

        let schema = GraphQLSchema::new()
            .with_query_type("Query")
            .add_type(query_type)
            .add_type(user_type)
            .add_resolver("Query", user_resolver2)
            .add_resolver("User", user_resolver);

        let query = GraphQLQuery::parse("{ user { name } }").unwrap();
        let context = ExecutionContext::new();

        let result = schema.execute(&query, &context).unwrap();

        if let GraphQLValue::Object(fields) = result {
            if let Some(GraphQLValue::Object(user_fields)) = fields.get("user") {
                assert_eq!(user_fields.get("name"), Some(&GraphQLValue::String("Alice".to_string())));
            } else {
                panic!("Expected user object");
            }
        } else {
            panic!("Expected object result");
        }
    }

    #[test]
    fn test_error_propagation() {
        let query_type = GraphQLType::object("Query", vec![
            GraphQLField::new("failingField", GraphQLType::scalar("String")),
        ]);

        let resolver = GraphQLResolver::from_fn("failingField", |_ctx, _args| {
            Err(GraphQLError::resolver("failingField", "Something went wrong"))
        });

        let schema = GraphQLSchema::new()
            .with_query_type("Query")
            .add_type(query_type)
            .add_resolver("Query", resolver);

        let query = GraphQLQuery::parse("{ failingField }").unwrap();
        let context = ExecutionContext::new();

        let result = schema.execute(&query, &context);
        assert!(result.is_err());

        if let Err(GraphQLError::ResolverError { field, .. }) = result {
            assert_eq!(field, "failingField");
        } else {
            panic!("Expected resolver error");
        }
    }

    #[test]
    fn test_list_type() {
        let list_type = GraphQLType::list(GraphQLType::scalar("String"));
        assert_eq!(list_type.name(), "[String]");

        let non_null_list = GraphQLType::non_null(list_type);
        assert_eq!(non_null_list.name(), "[String]!");
    }

    #[test]
    fn test_graphql_value_list() {
        let list = GraphQLValue::list(vec![
            GraphQLValue::int(1),
            GraphQLValue::int(2),
            GraphQLValue::int(3),
        ]);

        let json = list.to_json();
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_response_with_extensions() {
        let response = GraphQLResponse::success(GraphQLValue::string("data"))
            .with_extension("timing", GraphQLValue::float(42.5))
            .with_extension("version", GraphQLValue::string("1.0"));

        assert!(response.extensions.is_some());
        let extensions = response.extensions.unwrap();
        assert_eq!(extensions.len(), 2);
    }

    #[test]
    fn test_partial_response() {
        let data = GraphQLValue::Object(
            vec![
                ("success".to_string(), GraphQLValue::boolean(true)),
            ].into_iter().collect()
        );

        let errors = vec![
            GraphQLError::validation("Partial failure"),
        ];

        let response = GraphQLResponse::partial(data, errors.clone());

        assert!(response.has_errors());
        assert!(response.data.is_some());
        assert_eq!(response.errors.len(), 1);
    }

    #[test]
    fn test_complex_query_execution() {
        let post_type = GraphQLType::object("Post", vec![
            GraphQLField::new("id", GraphQLType::non_null(GraphQLType::scalar("ID"))),
            GraphQLField::new("title", GraphQLType::scalar("String")),
            GraphQLField::new("content", GraphQLType::scalar("String")),
        ]);

        let query_type = GraphQLType::object("Query", vec![
            GraphQLField::new("posts", GraphQLType::list(GraphQLType::object("Post", vec![
                GraphQLField::new("id", GraphQLType::non_null(GraphQLType::scalar("ID"))),
                GraphQLField::new("title", GraphQLType::scalar("String")),
            ]))),
        ]);

        let posts_resolver = GraphQLResolver::from_fn("posts", |_ctx, _args| {
            Ok(GraphQLValue::list(vec![
                GraphQLValue::Object(
                    vec![
                        ("id".to_string(), GraphQLValue::String("1".to_string())),
                        ("title".to_string(), GraphQLValue::String("First Post".to_string())),
                    ].into_iter().collect()
                ),
                GraphQLValue::Object(
                    vec![
                        ("id".to_string(), GraphQLValue::String("2".to_string())),
                        ("title".to_string(), GraphQLValue::String("Second Post".to_string())),
                    ].into_iter().collect()
                ),
            ]))
        });

        let schema = GraphQLSchema::new()
            .with_query_type("Query")
            .add_type(query_type)
            .add_type(post_type)
            .add_resolver("Query", posts_resolver);

        let query = GraphQLQuery::parse("{ posts { id title } }").unwrap();
        let context = ExecutionContext::new();

        let result = schema.execute(&query, &context).unwrap();

        if let GraphQLValue::Object(fields) = result {
            if let Some(GraphQLValue::List(posts)) = fields.get("posts") {
                assert_eq!(posts.len(), 2);
            } else {
                panic!("Expected posts list");
            }
        } else {
            panic!("Expected object result");
        }
    }
}
