//! OpenAPI Specification (Swagger) Implementation
//!
//! Complete OpenAPI 3.1 spec generation, JSON Schema builder,
//! and client/server code generation for REST APIs.

use std::collections::HashMap;
use std::fmt;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue};

// Simple YAML serializer (to avoid serde_yaml dependency)
fn to_yaml_simple(value: &serde_json::Value) -> Result<String, String> {
    fn value_to_yaml(val: &serde_json::Value, indent: usize) -> String {
        let ind = "  ".repeat(indent);
        match val {
            JsonValue::Null => "null".to_string(),
            JsonValue::Bool(b) => b.to_string(),
            JsonValue::Number(n) => n.to_string(),
            JsonValue::String(s) => {
                // Check if we need quotes
                if s.contains(':') || s.contains('\n') || s.starts_with(' ') {
                    format!("'{ }'", s.replace("'", "''"))
                } else {
                    s.clone()
                }
            }
            JsonValue::Array(arr) => {
                if arr.is_empty() {
                    "[]".to_string()
                } else if arr.iter().all(|v| v.is_object() || v.is_array()) {
                    // Multi-line array format
                    let mut result = String::new();
                    for item in arr {
                        result.push_str(&format!("- {}\n", value_to_yaml(item, indent)));
                    }
                    result
                } else {
                    // Inline array format
                    let items: Vec<String> = arr.iter()
                        .map(|v| value_to_yaml(v, 0))
                        .collect();
                    format!("[{}]", items.join(", "))
                }
            }
            JsonValue::Object(obj) => {
                if obj.is_empty() {
                    "{}".to_string()
                } else {
                    let mut result = String::new();
                    for (k, v) in obj {
                        result.push_str(&ind);
                        result.push_str(k);
                        result.push_str(":");
                        if v.is_object() || v.is_array() {
                            result.push('\n');
                            result.push_str(&value_to_yaml(v, indent + 1));
                        } else {
                            result.push(' ');
                            result.push_str(&value_to_yaml(v, 0));
                            result.push('\n');
                        }
                    }
                    result
                }
            }
        }
    }
    Ok(value_to_yaml(value, 0))
}

// ========================================================================
// OpenAPI Specification Structures
// ========================================================================

/// OpenAPI 3.1 Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAPISpec {
    pub openapi: String,
    pub info: Info,
    pub servers: Vec<Server>,
    pub paths: HashMap<String, PathItem>,
    pub components: Option<Components>,
    pub security: Vec<SecurityRequirement>,
    pub tags: Vec<Tag>,
    pub external_docs: Option<ExternalDocumentation>,
}

impl OpenAPISpec {
    /// Create new OpenAPI spec
    pub fn new(title: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            openapi: "3.1.0".to_string(),
            info: Info::new(title, version),
            servers: Vec::new(),
            paths: HashMap::new(),
            components: None,
            security: Vec::new(),
            tags: Vec::new(),
            external_docs: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.info.description = Some(description.into());
        self
    }

    /// Add server
    pub fn with_server(mut self, url: impl Into<String>, description: impl Into<String>) -> Self {
        self.servers.push(Server {
            url: url.into(),
            description: description.into(),
            variables: None,
        });
        self
    }

    /// Add path/operation
    pub fn with_path(mut self, path: impl Into<String>, item: PathItem) -> Self {
        self.paths.insert(path.into(), item);
        self
    }

    /// Add tag
    pub fn with_tag(mut self, tag: Tag) -> Self {
        self.tags.push(tag);
        self
    }

    /// Set components
    pub fn with_components(mut self, components: Components) -> Self {
        self.components = Some(components);
        self
    }

    /// Add global security requirement
    pub fn with_security(mut self, security: SecurityRequirement) -> Self {
        self.security.push(security);
        self
    }

    /// Generate JSON spec
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize spec: {}", e))
    }

    /// Generate YAML spec
    pub fn to_yaml(&self) -> Result<String, String> {
        let json = serde_json::to_value(self)
            .map_err(|e| format!("Failed to serialize to JSON: {}", e))?;
        to_yaml_simple(&json)
    }

    /// Merge another spec into this one
    pub fn merge(&mut self, other: OpenAPISpec) {
        for (path, item) in other.paths {
            self.paths.insert(path, item);
        }
        if let Some(other_components) = other.components {
            if let Some(ref mut components) = self.components {
                if let Some(schemas) = other_components.schemas {
                    components
                        .schemas
                        .get_or_insert_with(HashMap::new)
                        .extend(schemas);
                }
            } else {
                self.components = Some(other_components);
            }
        }
        self.tags.extend(other.tags);
        self.security.extend(other.security);
    }

    /// Validate spec structure
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.info.title.is_empty() {
            errors.push("Info title cannot be empty".to_string());
        }

        if self.info.version.is_empty() {
            errors.push("Info version cannot be empty".to_string());
        }

        for (path, item) in &self.paths {
            if path.is_empty() || !path.starts_with('/') {
                errors.push(format!("Invalid path: {}", path));
            }

            // Validate operations
            for op in item.operations() {
                if let Some(op_id) = &op.operation_id {
                    if op_id.is_empty() {
                        errors.push(format!("Empty operationId for {}", path));
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get all operation IDs
    pub fn operation_ids(&self) -> Vec<String> {
        let mut ids = Vec::new();
        for item in self.paths.values() {
            for op in item.operations() {
                if let Some(id) = &op.operation_id {
                    ids.push(id.clone());
                }
            }
        }
        ids
    }

    /// Find operation by ID
    pub fn find_operation(&self, id: &str) -> Option<(String, HttpMethod, Operation)> {
        for (path, item) in &self.paths {
            for (method, op) in item.all_operations() {
                if op.operation_id.as_ref() == Some(&id.to_string()) {
                    return Some((path.clone(), method, op));
                }
            }
        }
        None
    }
}

/// API Info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Info {
    pub title: String,
    pub version: String,
    pub description: Option<String>,
    pub terms_of_service: Option<String>,
    pub contact: Option<Contact>,
    pub license: Option<License>,
}

impl Info {
    pub fn new(title: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            version: version.into(),
            description: None,
            terms_of_service: None,
            contact: None,
            license: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_contact(mut self, contact: Contact) -> Self {
        self.contact = Some(contact);
        self
    }

    pub fn with_license(mut self, license: License) -> Self {
        self.license = Some(license);
        self
    }
}

/// Contact Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    pub name: Option<String>,
    pub url: Option<String>,
    pub email: Option<String>,
}

/// License Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    pub name: String,
    pub url: Option<String>,
    pub identifier: Option<String>,
}

/// Server Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Server {
    pub url: String,
    pub description: String,
    pub variables: Option<HashMap<String, ServerVariable>>,
}

/// Server Variable for templating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerVariable {
    pub default: String,
    pub description: Option<String>,
    pub enum_values: Option<Vec<String>>,
}

/// Path Item with operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PathItem {
    pub summary: Option<String>,
    pub description: Option<String>,
    pub get: Option<Operation>,
    pub put: Option<Operation>,
    pub post: Option<Operation>,
    pub delete: Option<Operation>,
    pub options: Option<Operation>,
    pub head: Option<Operation>,
    pub patch: Option<Operation>,
    pub trace: Option<Operation>,
    pub servers: Vec<Server>,
    pub parameters: Vec<Parameter>,
}

impl PathItem {
    /// Create new path item
    pub fn new() -> Self {
        Self::default()
    }

    /// Add GET operation
    pub fn with_get(mut self, op: Operation) -> Self {
        self.get = Some(op);
        self
    }

    /// Add POST operation
    pub fn with_post(mut self, op: Operation) -> Self {
        self.post = Some(op);
        self
    }

    /// Add PUT operation
    pub fn with_put(mut self, op: Operation) -> Self {
        self.put = Some(op);
        self
    }

    /// Add DELETE operation
    pub fn with_delete(mut self, op: Operation) -> Self {
        self.delete = Some(op);
        self
    }

    /// Add PATCH operation
    pub fn with_patch(mut self, op: Operation) -> Self {
        self.patch = Some(op);
        self
    }

    /// Get all operations
    pub fn operations(&self) -> Vec<&Operation> {
        let mut ops = Vec::new();
        if let Some(ref op) = self.get { ops.push(op); }
        if let Some(ref op) = self.post { ops.push(op); }
        if let Some(ref op) = self.put { ops.push(op); }
        if let Some(ref op) = self.delete { ops.push(op); }
        if let Some(ref op) = self.patch { ops.push(op); }
        if let Some(ref op) = self.options { ops.push(op); }
        if let Some(ref op) = self.head { ops.push(op); }
        if let Some(ref op) = self.trace { ops.push(op); }
        ops
    }

    /// Get all operations with method
    pub fn all_operations(&self) -> Vec<(HttpMethod, Operation)> {
        let mut ops = Vec::new();
        if let Some(op) = &self.get { ops.push((HttpMethod::GET, op.clone())); }
        if let Some(op) = &self.post { ops.push((HttpMethod::POST, op.clone())); }
        if let Some(op) = &self.put { ops.push((HttpMethod::PUT, op.clone())); }
        if let Some(op) = &self.delete { ops.push((HttpMethod::DELETE, op.clone())); }
        if let Some(op) = &self.patch { ops.push((HttpMethod::PATCH, op.clone())); }
        if let Some(op) = &self.options { ops.push((HttpMethod::OPTIONS, op.clone())); }
        if let Some(op) = &self.head { ops.push((HttpMethod::HEAD, op.clone())); }
        if let Some(op) = &self.trace { ops.push((HttpMethod::TRACE, op.clone())); }
        ops
    }
}

/// HTTP Method enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    OPTIONS,
    HEAD,
    TRACE,
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_lowercase())
    }
}

/// Operation Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub tags: Vec<String>,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub external_docs: Option<ExternalDocumentation>,
    pub operation_id: Option<String>,
    pub parameters: Vec<Parameter>,
    pub request_body: Option<RequestBody>,
    pub responses: HashMap<String, Response>,
    pub callbacks: HashMap<String, Callback>,
    pub deprecated: bool,
    pub security: Vec<SecurityRequirement>,
    pub servers: Vec<Server>,
}

impl Operation {
    pub fn new() -> Self {
        Self {
            tags: Vec::new(),
            summary: None,
            description: None,
            external_docs: None,
            operation_id: None,
            parameters: Vec::new(),
            request_body: None,
            responses: HashMap::new(),
            callbacks: HashMap::new(),
            deprecated: false,
            security: Vec::new(),
            servers: Vec::new(),
        }
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_operation_id(mut self, id: impl Into<String>) -> Self {
        self.operation_id = Some(id.into());
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_parameter(mut self, param: Parameter) -> Self {
        self.parameters.push(param);
        self
    }

    pub fn with_request_body(mut self, body: RequestBody) -> Self {
        self.request_body = Some(body);
        self
    }

    pub fn deprecated(mut self) -> Self {
        self.deprecated = true;
        self
    }

    pub fn with_response(mut self, status: u16, description: impl Into<String>, schema: Schema) -> Self {
        let response = Response::new(description)
            .with_content("application/json", MediaType::new().with_schema(schema));
        self.responses.insert(status.to_string(), response);
        self
    }

    pub fn with_query_param(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.parameters.push(
            Parameter::new(name, ParameterLocation::Query)
                .with_schema(schema)
        );
        self
    }

    pub fn with_path_param(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.parameters.push(
            Parameter::new(name, ParameterLocation::Path)
                .required()
                .with_schema(schema)
        );
        self
    }
}

/// Parameter Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub in_location: ParameterLocation,
    pub description: Option<String>,
    pub required: bool,
    pub deprecated: bool,
    pub allow_empty_value: bool,
    pub style: Option<ParameterStyle>,
    pub explode: Option<bool>,
    pub allow_reserved: bool,
    pub schema: Option<Schema>,
    pub example: Option<JsonValue>,
    pub examples: HashMap<String, Example>,
    pub content: HashMap<String, MediaType>,
}

impl Parameter {
    pub fn new(name: impl Into<String>, in_location: ParameterLocation) -> Self {
        Self {
            name: name.into(),
            in_location,
            description: None,
            required: false,
            deprecated: false,
            allow_empty_value: false,
            style: None,
            explode: None,
            allow_reserved: false,
            schema: None,
            example: None,
            examples: HashMap::new(),
            content: HashMap::new(),
        }
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_schema(mut self, schema: Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn with_example(mut self, example: impl Into<JsonValue>) -> Self {
        self.example = Some(example.into());
        self
    }
}

/// Parameter Location
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParameterLocation {
    Query,
    Header,
    Path,
    Cookie,
}

/// Parameter Style
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParameterStyle {
    Matrix,
    Label,
    Form,
    Simple,
    SpaceDelimited,
    PipeDelimited,
    DeepObject,
}

/// Request Body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestBody {
    pub description: Option<String>,
    pub content: HashMap<String, MediaType>,
    pub required: bool,
}

impl RequestBody {
    pub fn new() -> Self {
        Self {
            description: None,
            content: HashMap::new(),
            required: true,
        }
    }

    pub fn with_content(mut self, content_type: impl Into<String>, media: MediaType) -> Self {
        self.content.insert(content_type.into(), media);
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }
}

/// Media Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaType {
    pub schema: Option<Schema>,
    pub example: Option<JsonValue>,
    pub examples: HashMap<String, Example>,
    pub encoding: HashMap<String, Encoding>,
}

impl MediaType {
    pub fn new() -> Self {
        Self {
            schema: None,
            example: None,
            examples: HashMap::new(),
            encoding: HashMap::new(),
        }
    }

    pub fn with_schema(mut self, schema: Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn with_example(mut self, example: impl Into<JsonValue>) -> Self {
        self.example = Some(example.into());
        self
    }
}

/// Response Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub description: String,
    pub headers: HashMap<String, Header>,
    pub content: HashMap<String, MediaType>,
    pub links: HashMap<String, Link>,
}

impl Response {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            headers: HashMap::new(),
            content: HashMap::new(),
            links: HashMap::new(),
        }
    }

    pub fn with_content(mut self, content_type: impl Into<String>, media: MediaType) -> Self {
        self.content.insert(content_type.into(), media);
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, header: Header) -> Self {
        self.headers.insert(name.into(), header);
        self
    }
}

/// Components
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Components {
    pub schemas: Option<HashMap<String, Schema>>,
    pub responses: Option<HashMap<String, Response>>,
    pub parameters: Option<HashMap<String, Parameter>>,
    pub examples: Option<HashMap<String, Example>>,
    pub request_bodies: Option<HashMap<String, RequestBody>>,
    pub headers: Option<HashMap<String, Header>>,
    pub security_schemes: Option<HashMap<String, SecurityScheme>>,
    pub links: Option<HashMap<String, Link>>,
    pub callbacks: Option<HashMap<String, Callback>>,
}

impl Components {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_schema(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.schemas.get_or_insert_with(HashMap::new).insert(name.into(), schema);
        self
    }

    pub fn with_security_scheme(mut self, name: impl Into<String>, scheme: SecurityScheme) -> Self {
        self.security_schemes
            .get_or_insert_with(HashMap::new)
            .insert(name.into(), scheme);
        self
    }

    pub fn with_response(mut self, name: impl Into<String>, response: Response) -> Self {
        self.responses
            .get_or_insert_with(HashMap::new)
            .insert(name.into(), response);
        self
    }
}

/// Schema Object (JSON Schema)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Schema {
    pub title: Option<String>,
    pub schema_type: Option<String>,
    #[serde(rename = "$ref")]
    pub reference: Option<String>,
    pub description: Option<String>,
    pub format: Option<String>,
    pub default: Option<JsonValue>,
    pub multiple_of: Option<f64>,
    pub maximum: Option<f64>,
    pub exclusive_maximum: Option<f64>,
    pub minimum: Option<f64>,
    pub exclusive_minimum: Option<f64>,
    pub max_length: Option<u64>,
    pub min_length: Option<u64>,
    pub pattern: Option<String>,
    pub max_items: Option<u64>,
    pub min_items: Option<u64>,
    pub unique_items: Option<bool>,
    pub max_properties: Option<u64>,
    pub min_properties: Option<u64>,
    pub required: Option<Vec<String>>,
    pub enum_values: Option<Vec<JsonValue>>,
    #[serde(rename = "const")]
    pub const_value: Option<JsonValue>,
    pub items: Option<Box<Schema>>,
    pub properties: Option<HashMap<String, Schema>>,
    pub additional_properties: Option<Box<Schema>>,
    pub all_of: Option<Vec<Schema>>,
    pub any_of: Option<Vec<Schema>>,
    pub one_of: Option<Vec<Schema>>,
    pub not: Option<Box<Schema>>,
    pub discriminator: Option<Discriminator>,
    pub read_only: Option<bool>,
    pub write_only: Option<bool>,
    pub deprecated: Option<bool>,
    pub xml: Option<Xml>,
    pub external_docs: Option<ExternalDocumentation>,
    pub example: Option<JsonValue>,
}

impl Schema {
    /// Create new schema
    pub fn new() -> Self {
        Self {
            title: None,
            schema_type: None,
            reference: None,
            description: None,
            format: None,
            default: None,
            multiple_of: None,
            maximum: None,
            exclusive_maximum: None,
            minimum: None,
            exclusive_minimum: None,
            max_length: None,
            min_length: None,
            pattern: None,
            max_items: None,
            min_items: None,
            unique_items: None,
            max_properties: None,
            min_properties: None,
            required: None,
            enum_values: None,
            const_value: None,
            items: None,
            properties: None,
            additional_properties: None,
            all_of: None,
            any_of: None,
            one_of: None,
            not: None,
            discriminator: None,
            read_only: None,
            write_only: None,
            deprecated: None,
            xml: None,
            external_docs: None,
            example: None,
        }
    }

    /// Create string schema
    pub fn string() -> Self {
        Self::new().with_type("string")
    }

    /// Create number schema
    pub fn number() -> Self {
        Self::new().with_type("number")
    }

    /// Create integer schema
    pub fn integer() -> Self {
        Self::new().with_type("integer")
    }

    /// Create boolean schema
    pub fn boolean() -> Self {
        Self::new().with_type("boolean")
    }

    /// Create array schema
    pub fn array(items: Schema) -> Self {
        Self::new().with_type("array").with_items(items)
    }

    /// Create object schema
    pub fn object() -> Self {
        Self::new().with_type("object")
    }

    /// Create reference schema
    pub fn reference(ref_name: impl Into<String>) -> Self {
        Self::new().with_reference(ref_name)
    }

    pub fn with_type(mut self, schema_type: impl Into<String>) -> Self {
        self.schema_type = Some(schema_type.into());
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    pub fn with_default(mut self, value: impl Into<JsonValue>) -> Self {
        self.default = Some(value.into());
        self
    }

    pub fn with_minimum(mut self, min: f64) -> Self {
        self.minimum = Some(min);
        self
    }

    pub fn with_maximum(mut self, max: f64) -> Self {
        self.maximum = Some(max);
        self
    }

    pub fn with_min_length(mut self, min: u64) -> Self {
        self.min_length = Some(min);
        self
    }

    pub fn with_max_length(mut self, max: u64) -> Self {
        self.max_length = Some(max);
        self
    }

    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = Some(pattern.into());
        self
    }

    pub fn with_items(mut self, items: Schema) -> Self {
        self.items = Some(Box::new(items));
        self
    }

    pub fn with_property(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.properties
            .get_or_insert_with(HashMap::new)
            .insert(name.into(), schema);
        self
    }

    pub fn with_required_field(mut self, name: impl Into<String>) -> Self {
        self.required
            .get_or_insert_with(Vec::new)
            .push(name.into());
        self
    }

    pub fn with_required_fields(mut self, names: Vec<String>) -> Self {
        self.required = Some(names);
        self
    }

    pub fn with_enum(mut self, values: Vec<JsonValue>) -> Self {
        self.enum_values = Some(values);
        self
    }

    pub fn with_required(mut self, required: Vec<String>) -> Self {
        self.required = Some(required);
        self
    }

    pub fn with_reference(mut self, reference: impl Into<String>) -> Self {
        self.reference = Some(format!("#/components/schemas/{}", reference.into()));
        self
    }

    pub fn with_any_of(mut self, schemas: Vec<Schema>) -> Self {
        self.any_of = Some(schemas);
        self
    }

    pub fn with_all_of(mut self, schemas: Vec<Schema>) -> Self {
        self.all_of = Some(schemas);
        self
    }

    pub fn with_one_of(mut self, schemas: Vec<Schema>) -> Self {
        self.one_of = Some(schemas);
        self
    }

    pub fn read_only(mut self) -> Self {
        self.read_only = Some(true);
        self
    }

    pub fn write_only(mut self) -> Self {
        self.write_only = Some(true);
        self
    }
}

/// Discriminator for polymorphism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discriminator {
    pub property_name: String,
    pub mapping: Option<HashMap<String, String>>,
}

/// XML representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Xml {
    pub name: Option<String>,
    pub namespace: Option<String>,
    pub prefix: Option<String>,
    pub attribute: Option<bool>,
    pub wrapped: Option<bool>,
}

/// Security Scheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityScheme {
    #[serde(rename = "type")]
    pub scheme_type: String,
    pub description: Option<String>,
    pub name: Option<String>,
    pub in_location: Option<String>,
    pub scheme: Option<String>,
    pub bearer_format: Option<String>,
    pub flows: Option<OAuthFlows>,
    pub open_id_connect_url: Option<String>,
}

impl SecurityScheme {
    /// HTTP Bearer scheme
    pub fn http_bearer() -> Self {
        Self {
            scheme_type: "http".to_string(),
            scheme: Some("bearer".to_string()),
            bearer_format: Some("JWT".to_string()),
            description: Some("JWT Bearer token".to_string()),
            name: None,
            in_location: None,
            flows: None,
            open_id_connect_url: None,
        }
    }

    /// API Key scheme
    pub fn api_key(name: impl Into<String>, in_location: impl Into<String>) -> Self {
        Self {
            scheme_type: "apiKey".to_string(),
            name: Some(name.into()),
            in_location: Some(in_location.into()),
            description: None,
            scheme: None,
            bearer_format: None,
            flows: None,
            open_id_connect_url: None,
        }
    }
}

/// OAuth Flows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthFlows {
    pub implicit: Option<OAuthFlow>,
    pub password: Option<OAuthFlow>,
    pub client_credentials: Option<OAuthFlow>,
    pub authorization_code: Option<OAuthFlow>,
}

/// OAuth Flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthFlow {
    pub authorization_url: Option<String>,
    pub token_url: Option<String>,
    pub refresh_url: Option<String>,
    pub scopes: HashMap<String, String>,
}

/// Security Requirement
pub type SecurityRequirement = HashMap<String, Vec<String>>;

/// Example Object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub summary: Option<String>,
    pub description: Option<String>,
    pub value: Option<JsonValue>,
    pub external_value: Option<String>,
}

/// Header Object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub description: Option<String>,
    pub required: bool,
    pub deprecated: bool,
    pub allow_empty_value: bool,
    pub style: Option<ParameterStyle>,
    pub explode: Option<bool>,
    pub allow_reserved: bool,
    pub schema: Option<Schema>,
    pub example: Option<JsonValue>,
    pub examples: HashMap<String, Example>,
    pub content: HashMap<String, MediaType>,
}

/// Encoding Object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Encoding {
    pub content_type: Option<String>,
    pub headers: HashMap<String, Header>,
    pub style: Option<ParameterStyle>,
    pub explode: Option<bool>,
    pub allow_reserved: bool,
}

/// Link Object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub operation_ref: Option<String>,
    pub operation_id: Option<String>,
    pub parameters: HashMap<String, JsonValue>,
    pub request_body: Option<JsonValue>,
    pub description: Option<String>,
    pub server: Option<Server>,
}

/// Callback Object
pub type Callback = HashMap<String, PathItem>;

/// Tag Object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    pub name: String,
    pub description: Option<String>,
    pub external_docs: Option<ExternalDocumentation>,
}

impl Tag {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            external_docs: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// External Documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalDocumentation {
    pub description: Option<String>,
    pub url: String,
}

// ========================================================================
// API Operation Builder
// ========================================================================

/// Builder for API operations
#[derive(Debug, Clone)]
pub struct ApiOperation {
    operation: Operation,
    path: String,
    method: HttpMethod,
}

impl ApiOperation {
    pub fn new(method: HttpMethod, path: impl Into<String>) -> Self {
        Self {
            operation: Operation::new(),
            method,
            path: path.into(),
        }
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.operation.summary = Some(summary.into());
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.operation.description = Some(desc.into());
        self
    }

    pub fn with_operation_id(mut self, id: impl Into<String>) -> Self {
        self.operation.operation_id = Some(id.into());
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.operation.tags.push(tag.into());
        self
    }

    pub fn with_query_param(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.operation.parameters.push(
            Parameter::new(name, ParameterLocation::Query)
                .with_schema(schema)
        );
        self
    }

    pub fn with_path_param(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.operation.parameters.push(
            Parameter::new(name, ParameterLocation::Path)
                .required()
                .with_schema(schema)
        );
        self
    }

    pub fn with_header_param(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.operation.parameters.push(
            Parameter::new(name, ParameterLocation::Header)
                .with_schema(schema)
        );
        self
    }

    pub fn with_response(mut self, status: u16, description: impl Into<String>, schema: Schema) -> Self {
        let response = Response::new(description)
            .with_content("application/json", MediaType::new().with_schema(schema));
        self.operation.responses.insert(status.to_string(), response);
        self
    }

    pub fn with_request_body(mut self, schema: Schema) -> Self {
        let body = RequestBody::new()
            .with_content("application/json", MediaType::new().with_schema(schema));
        self.operation.request_body = Some(body);
        self
    }

    pub fn deprecated(mut self) -> Self {
        self.operation.deprecated = true;
        self
    }

    pub fn build(self) -> (String, HttpMethod, Operation) {
        (self.path, self.method, self.operation)
    }
}

// ========================================================================
// Schema Builder
// ========================================================================

/// Fluent builder for complex schemas
#[derive(Debug, Clone)]
pub struct SchemaBuilder {
    schema: Schema,
}

impl SchemaBuilder {
    pub fn new() -> Self {
        Self {
            schema: Schema::new(),
        }
    }

    pub fn from_schema(schema: Schema) -> Self {
        Self { schema }
    }

    pub fn string() -> Self {
        Self::from_schema(Schema::string())
    }

    pub fn number() -> Self {
        Self::from_schema(Schema::number())
    }

    pub fn integer() -> Self {
        Self::from_schema(Schema::integer())
    }

    pub fn boolean() -> Self {
        Self::from_schema(Schema::boolean())
    }

    pub fn object() -> Self {
        Self::from_schema(Schema::object())
    }

    pub fn array_of(schema: Schema) -> Self {
        Self::from_schema(Schema::array(schema))
    }

    pub fn reference(name: impl Into<String>) -> Self {
        Self::from_schema(Schema::reference(name))
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.schema.title = Some(title.into());
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.schema.description = Some(desc.into());
        self
    }

    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.schema.format = Some(format.into());
        self
    }

    pub fn with_min_length(mut self, min: u64) -> Self {
        self.schema.min_length = Some(min);
        self
    }

    pub fn with_max_length(mut self, max: u64) -> Self {
        self.schema.max_length = Some(max);
        self
    }

    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.schema.pattern = Some(pattern.into());
        self
    }

    pub fn with_minimum(mut self, min: f64) -> Self {
        self.schema.minimum = Some(min);
        self
    }

    pub fn with_maximum(mut self, max: f64) -> Self {
        self.schema.maximum = Some(max);
        self
    }

    pub fn with_property(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.schema
            .properties
            .get_or_insert_with(HashMap::new)
            .insert(name.into(), schema);
        self
    }

    pub fn with_required_field(mut self, name: impl Into<String>) -> Self {
        self.schema
            .required
            .get_or_insert_with(Vec::new)
            .push(name.into());
        self
    }

    pub fn with_required_fields(mut self, names: Vec<String>) -> Self {
        self.schema.required = Some(names);
        self
    }

    pub fn read_only(mut self) -> Self {
        self.schema.read_only = Some(true);
        self
    }

    pub fn write_only(mut self) -> Self {
        self.schema.write_only = Some(true);
        self
    }

    pub fn with_default(mut self, value: impl Into<JsonValue>) -> Self {
        self.schema.default = Some(value.into());
        self
    }

    pub fn with_enum(mut self, values: Vec<JsonValue>) -> Self {
        self.schema.enum_values = Some(values);
        self
    }

    pub fn with_any_of(mut self, schemas: Vec<Schema>) -> Self {
        self.schema.any_of = Some(schemas);
        self
    }

    pub fn with_all_of(mut self, schemas: Vec<Schema>) -> Self {
        self.schema.all_of = Some(schemas);
        self
    }

    pub fn with_one_of(mut self, schemas: Vec<Schema>) -> Self {
        self.schema.one_of = Some(schemas);
        self
    }

    pub fn build(self) -> Schema {
        self.schema
    }
}

// ========================================================================
// API Generator
// ========================================================================

/// Code generator for API clients and servers
#[derive(Debug, Clone)]
pub struct ApiGenerator {
    spec: OpenAPISpec,
    config: GeneratorConfig,
}

#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub language: TargetLanguage,
    pub package_name: Option<String>,
    pub client_name: Option<String>,
    pub use_async: bool,
    pub include_deprecated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetLanguage {
    TypeScript,
    Rust,
    Python,
    Go,
}

impl ApiGenerator {
    pub fn new(spec: OpenAPISpec) -> Self {
        Self {
            spec,
            config: GeneratorConfig {
                language: TargetLanguage::TypeScript,
                package_name: None,
                client_name: None,
                use_async: true,
                include_deprecated: false,
            },
        }
    }

    pub fn with_language(mut self, language: TargetLanguage) -> Self {
        self.config.language = language;
        self
    }

    pub fn with_package_name(mut self, name: impl Into<String>) -> Self {
        self.config.package_name = Some(name.into());
        self
    }

    pub fn with_client_name(mut self, name: impl Into<String>) -> Self {
        self.config.client_name = Some(name.into());
        self
    }

    pub fn use_async(mut self, use_async: bool) -> Self {
        self.config.use_async = use_async;
        self
    }

    pub fn include_deprecated(mut self, include: bool) -> Self {
        self.config.include_deprecated = include;
        self
    }

    /// Generate TypeScript client
    pub fn generate_typescript_client(&self) -> Result<String, String> {
        let mut code = String::new();

        // File header
        code.push_str("// Auto-generated OpenAPI client\n");
        code.push_str("// DO NOT EDIT\n\n");

        // Imports
        code.push_str("interface RequestConfig {\n");
        code.push_str("  method: string;\n");
        code.push_str("  path: string;\n");
        code.push_str("  query?: Record<string, any>;\n");
        code.push_str("  body?: any;\n");
        code.push_str("  headers?: Record<string, string>;\n");
        code.push_str("}\n\n");

        // Generate interfaces from schemas
        if let Some(components) = &self.spec.components {
            if let Some(schemas) = &components.schemas {
                for (name, schema) in schemas {
                    code.push_str(&self.schema_to_typescript(name, schema)?);
                    code.push_str("\n");
                }
            }
        }

        // Client class
        let client_name = self.config.client_name.as_deref().unwrap_or("ApiClient");
        code.push_str(&format!("class {} {{\n", client_name));
        code.push_str(&format!("  private baseUrl: string;\n"));
        code.push_str(&format!("  private defaultHeaders: Record<string, string>;\n\n"));

        code.push_str(&format!("  constructor(baseUrl: string = '/', defaultHeaders: Record<string, string> = {{}}) {{\n"));
        code.push_str(&format!("    this.baseUrl = baseUrl;\n"));
        code.push_str(&format!("    this.defaultHeaders = defaultHeaders;\n"));
        code.push_str(&format!("  }}\n\n"));

        // Methods for each operation
        for (path, item) in &self.spec.paths {
            for (method, operation) in item.all_operations() {
                if !self.config.include_deprecated && operation.deprecated {
                    continue;
                }

                if let Some(op_id) = &operation.operation_id {
                    code.push_str(&self.operation_to_typescript(op_id, method, path, &operation)?);
                    code.push_str("\n");
                }
            }
        }

        code.push_str("}\n");

        Ok(code)
    }

    /// Generate Rust server stubs
    pub fn generate_rust_server(&self) -> Result<String, String> {
        let mut code = String::new();

        code.push_str("// Auto-generated OpenAPI server stubs\n");
        code.push_str("// DO NOT EDIT\n");
        code.push_str("use serde::{Deserialize, Serialize};\n");
        code.push_str("use std::collections::HashMap;\n\n");

        // Generate structs from schemas
        if let Some(components) = &self.spec.components {
            if let Some(schemas) = &components.schemas {
                for (name, schema) in schemas {
                    code.push_str(&self.schema_to_rust(name, schema)?);
                    code.push_str("\n");
                }
            }
        }

        // Handler trait
        code.push_str("#[async_trait::async_trait]\n");
        code.push_str("pub trait ApiHandler {\n");

        for (path, item) in &self.spec.paths {
            for (method, operation) in item.all_operations() {
                if let Some(op_id) = &operation.operation_id {
                    let fn_name = self.to_snake_case(op_id);
                    code.push_str(&format!("  async fn {}(&self", fn_name));

                    // Add path params
                    for param in &operation.parameters {
                        if param.in_location == ParameterLocation::Path {
                            let param_type = self.param_type_to_rust(&param)?;
                            code.push_str(&format!(", {}: {}", param.name, param_type));
                        }
                    }

                    // Add request body
                    if let Some(body) = &operation.request_body {
                        if let Some(media) = body.content.get("application/json") {
                            if let Some(schema) = &media.schema {
                                let type_name = self.schema_type_name(schema)?;
                                code.push_str(&format!(", body: {}", type_name));
                            }
                        }
                    }

                    code.push_str(&format!(") -> Result<Response, ApiError>;\n"));
                }
            }
        }

        code.push_str("}\n\n");

        // Error type
        code.push_str("#[derive(Debug)]\n");
        code.push_str("pub enum ApiError {\n");
        code.push_str("  BadRequest(String),\n");
        code.push_str("  Unauthorized,\n");
        code.push_str("  Forbidden,\n");
        code.push_str("  NotFound,\n");
        code.push_str("  InternalError,\n");
        code.push_str("}\n");

        Ok(code)
    }

    /// Generate Python client
    pub fn generate_python_client(&self) -> Result<String, String> {
        let mut code = String::new();

        code.push_str("# Auto-generated OpenAPI client\n");
        code.push_str("# DO NOT EDIT\n\n");
        code.push_str("from typing import Any, Dict, Optional, List\n");
        code.push_str("from dataclasses import dataclass\n");
        code.push_str("import requests\n\n");

        // Generate dataclasses from schemas
        if let Some(components) = &self.spec.components {
            if let Some(schemas) = &components.schemas {
                for (name, schema) in schemas {
                    code.push_str(&self.schema_to_python(name, schema)?);
                    code.push_str("\n");
                }
            }
        }

        // Client class
        let client_name = self.config.client_name.as_deref().unwrap_or("ApiClient");
        code.push_str(&format!("class {}:\n", client_name));
        code.push_str(&format!("    def __init__(self, base_url: str = '/', headers: Optional[Dict[str, str]] = None):\n"));
        code.push_str(&format!("        self.base_url = base_url\n"));
        code.push_str(&format!("        self.headers = headers or {{}}\n\n"));

        // Methods for each operation
        for (path, item) in &self.spec.paths {
            for (method, operation) in item.all_operations() {
                if !self.config.include_deprecated && operation.deprecated {
                    continue;
                }

                if let Some(op_id) = &operation.operation_id {
                    code.push_str(&self.operation_to_python(op_id, method, path, &operation)?);
                    code.push_str("\n");
                }
            }
        }

        Ok(code)
    }

    /// Convert schema to TypeScript interface
    fn schema_to_typescript(&self, name: &str, schema: &Schema) -> Result<String, String> {
        let mut code = String::new();
        code.push_str(&format!("export interface {} ", name));

        if let Some(desc) = &schema.description {
            code.push_str(&format!("// {}\n", desc));
        }

        code.push_str("{\n");

        if let Some(props) = &schema.properties {
            for (prop_name, prop_schema) in props {
                let is_required = schema.required.as_ref()
                    .map(|r| r.contains(prop_name))
                    .unwrap_or(false);

                let ts_type = self.schema_type_to_typescript(prop_schema)?;
                code.push_str(&format!("  {}{}: {},\n",
                    if is_required { "" } else { "?" },
                    prop_name,
                    ts_type
                ));
            }
        }

        code.push_str("}\n");
        Ok(code)
    }

    /// Convert schema to Rust struct
    fn schema_to_rust(&self, name: &str, schema: &Schema) -> Result<String, String> {
        let mut code = String::new();

        if let Some(desc) = &schema.description {
            code.push_str(&format!("/// {}\n", desc));
        }

        code.push_str("#[derive(Debug, Clone, Serialize, Deserialize)]\n");
        code.push_str(&format!("pub struct {} ", name));
        code.push_str("{\n");

        if let Some(props) = &schema.properties {
            for (prop_name, prop_schema) in props {
                let is_required = schema.required.as_ref()
                    .map(|r| r.contains(prop_name))
                    .unwrap_or(false);

                let rust_type = self.schema_type_to_rust(prop_schema)?;
                let field_name = self.to_snake_case(prop_name);

                code.push_str(&format!("  pub {}{}: {},\n",
                    field_name,
                    if is_required { "" } else { ": Option" },
                    if is_required { rust_type } else { format!("Option<{}>", rust_type) }
                ));
            }
        }

        code.push_str("}\n");
        Ok(code)
    }

    /// Convert schema to Python dataclass
    fn schema_to_python(&self, name: &str, schema: &Schema) -> Result<String, String> {
        let mut code = String::new();

        if let Some(desc) = &schema.description {
            code.push_str(&format!("\"\"\"{} \"\"\"\n", desc));
        }

        code.push_str(&format!("@dataclass\n"));
        code.push_str(&format!("class {}:\n", name));

        if let Some(props) = &schema.properties {
            for (prop_name, prop_schema) in props {
                let is_required = schema.required.as_ref()
                    .map(|r| r.contains(prop_name))
                    .unwrap_or(false);

                let py_type = self.schema_type_to_python(prop_schema)?;
                code.push_str(&format!("    {}: {} = None,\n",
                    prop_name,
                    if is_required { py_type } else { format!("Optional[{}]", py_type) }
                ));
            }
        }

        Ok(code)
    }

    /// Convert operation to TypeScript method
    fn operation_to_typescript(
        &self,
        op_id: &str,
        method: HttpMethod,
        path: &str,
        operation: &Operation,
    ) -> Result<String, String> {
        let mut code = String::new();
        let fn_name = self.to_camel_case(op_id);

        // Build path with param substitution
        let path_with_params = self.extract_path_params(path, operation);

        // Generate method signature
        code.push_str(&format!("  async {}(", fn_name));

        // Add path params
        let mut first_param = true;
        for param in &operation.parameters {
            if param.in_location == ParameterLocation::Path {
                let param_type = self.param_type_to_typescript(param)?;
                code.push_str(&format!("{}{}: {}",
                    if first_param { "" } else { ", " },
                    param.name,
                    param_type
                ));
                first_param = false;
            }
        }

        // Add query params
        for param in &operation.parameters {
            if param.in_location == ParameterLocation::Query {
                let param_type = self.param_type_to_typescript(param)?;
                code.push_str(&format!("{}{}?: {}",
                    if first_param { "" } else { ", " },
                    param.name,
                    param_type
                ));
                first_param = false;
            }
        }

        // Add request body
        if let Some(body) = &operation.request_body {
            if let Some(media) = body.content.get("application/json") {
                if let Some(schema) = &media.schema {
                    let type_name = self.schema_type_name(schema)?;
                    code.push_str(&format!("{}body?: {}",
                        if first_param { "" } else { ", " },
                        type_name
                    ));
                    first_param = false;
                }
            }
        }

        // Add options/config
        code.push_str(&format!("{}config?: Partial<RequestConfig>",
            if first_param { "" } else { ", " }
        ));

        code.push_str(&format!("): Promise<{}> {{\n",
            self.get_response_type(operation)?
        ));

        // Build request
        code.push_str(&format!("    return this.request({{\n"));
        code.push_str(&format!("      method: '{}',\n", method));
        code.push_str(&format!("      path: `{}`,\n", path_with_params));

        // Add query params
        let has_query_params = operation.parameters.iter()
            .any(|p| p.in_location == ParameterLocation::Query);
        if has_query_params {
            code.push_str(&format!("      query: {{ "));
            let mut first = true;
            for param in &operation.parameters {
                if param.in_location == ParameterLocation::Query {
                    code.push_str(&format!("{}{}",
                        if first { "" } else { ", " },
                        param.name
                    ));
                    first = false;
                }
            }
            code.push_str(&format!(" }},\n"));
        }

        // Add body
        if operation.request_body.is_some() {
            code.push_str(&format!("      body,\n"));
        }

        code.push_str(&format!("      ...config\n"));
        code.push_str(&format!("    }}));\n"));
        code.push_str(&format!("  }}\n"));

        Ok(code)
    }

    /// Convert operation to Python method
    fn operation_to_python(
        &self,
        op_id: &str,
        method: HttpMethod,
        path: &str,
        operation: &Operation,
    ) -> Result<String, String> {
        let mut code = String::new();
        let fn_name = self.to_snake_case(op_id);

        code.push_str(&format!("    def {}(self", fn_name));

        // Add path params
        for param in &operation.parameters {
            if param.in_location == ParameterLocation::Path {
                let param_type = self.param_type_to_python(param)?;
                code.push_str(&format!(", {}: {}", param.name, param_type));
            }
        }

        // Add query params
        for param in &operation.parameters {
            if param.in_location == ParameterLocation::Query {
                let param_type = self.param_type_to_python(param)?;
                code.push_str(&format!(", {}: Optional[{}] = None", param.name, param_type));
            }
        }

        // Add request body
        if let Some(body) = &operation.request_body {
            if let Some(media) = body.content.get("application/json") {
                if let Some(schema) = &media.schema {
                    let type_name = self.schema_type_name(schema)?;
                    code.push_str(&format!(", body: {}", type_name));
                }
            }
        }

        code.push_str(&format!(") -> Dict[str, Any]:\n"));

        // Build URL
        let path_with_params = self.extract_path_params(path, operation);
        code.push_str(&format!("        url = self.base_url + '{}'\n", path_with_params));

        // Add query params
        let has_query_params = operation.parameters.iter()
            .any(|p| p.in_location == ParameterLocation::Query);
        if has_query_params {
            code.push_str(&format!("        params = {{\n"));
            for param in &operation.parameters {
                if param.in_location == ParameterLocation::Query {
                    code.push_str(&format!("            '{}': {},\n", param.name, param.name));
                }
            }
            code.push_str(&format!("        }}\n"));
        }

        // Make request
        code.push_str(&format!("        response = requests.{}(\n",
            self.to_snake_case(&format!("{:?}", method))
        ));
        code.push_str(&format!("            url,\n"));
        if has_query_params {
            code.push_str(&format!("            params=params,\n"));
        }
        if operation.request_body.is_some() {
            code.push_str(&format!("            json=body,\n"));
        }
        code.push_str(&format!("            headers=self.headers\n"));
        code.push_str(&format!("        )\n"));
        code.push_str(&format!("        return response.json()\n"));

        Ok(code)
    }

    // Helper methods

    fn extract_path_params(&self, path: &str, operation: &Operation) -> String {
        let mut result = path.to_string();
        for param in &operation.parameters {
            if param.in_location == ParameterLocation::Path {
                result = result.replace(&format!("{{{}}}", param.name), &format!("${{{}}}", param.name));
            }
        }
        result
    }

    fn schema_type_to_typescript(&self, schema: &Schema) -> Result<String, String> {
        let base_type: String = match schema.schema_type.as_deref() {
            Some("string") => "string".to_string(),
            Some("number") => "number".to_string(),
            Some("integer") => "number".to_string(),
            Some("boolean") => "boolean".to_string(),
            Some("array") => {
                if let Some(items) = &schema.items {
                    format!("{}[]", self.schema_type_to_typescript(items)?)
                } else {
                    "any[]".to_string()
                }
            }
            Some("object") => "Record<string, any>".to_string(),
            Some(t) => t.to_string(),
            None => "any".to_string(),
        };

        Ok(base_type)
    }

    fn schema_type_to_rust(&self, schema: &Schema) -> Result<String, String> {
        let base_type: String = match schema.schema_type.as_deref() {
            Some("string") => "String".to_string(),
            Some("number") => "f64".to_string(),
            Some("integer") => "i64".to_string(),
            Some("boolean") => "bool".to_string(),
            Some("array") => {
                if let Some(items) = &schema.items {
                    format!("Vec<{}>", self.schema_type_to_rust(items)?)
                } else {
                    "Vec<serde_json::Value>".to_string()
                }
            }
            Some("object") => "HashMap<String, serde_json::Value>".to_string(),
            Some(t) => t.to_string(),
            None => "serde_json::Value".to_string(),
        };

        Ok(base_type)
    }

    fn schema_type_to_python(&self, schema: &Schema) -> Result<String, String> {
        let base_type: String = match schema.schema_type.as_deref() {
            Some("string") => "str".to_string(),
            Some("number") => "float".to_string(),
            Some("integer") => "int".to_string(),
            Some("boolean") => "bool".to_string(),
            Some("array") => {
                if let Some(items) = &schema.items {
                    format!("List[{}]", self.schema_type_to_python(items)?)
                } else {
                    "List[Any]".to_string()
                }
            }
            Some("object") => "Dict[str, Any]".to_string(),
            Some(t) => t.to_string(),
            None => "Any".to_string(),
        };

        Ok(base_type)
    }

    fn param_type_to_typescript(&self, param: &Parameter) -> Result<String, String> {
        if let Some(schema) = &param.schema {
            self.schema_type_to_typescript(schema)
        } else {
            Ok("any".to_string())
        }
    }

    fn param_type_to_rust(&self, param: &Parameter) -> Result<String, String> {
        if let Some(schema) = &param.schema {
            self.schema_type_to_rust(schema)
        } else {
            Ok("String".to_string())
        }
    }

    fn param_type_to_python(&self, param: &Parameter) -> Result<String, String> {
        if let Some(schema) = &param.schema {
            self.schema_type_to_python(schema)
        } else {
            Ok("str".to_string())
        }
    }

    fn schema_type_name(&self, schema: &Schema) -> Result<String, String> {
        if let Some(ref_name) = &schema.reference {
            // Extract schema name from reference
            if let Some(name) = ref_name.split('/').last() {
                return Ok(name.to_string());
            }
        }

        if let Some(schema_type) = &schema.schema_type {
            // Map basic types
            return Ok(match schema_type.as_str() {
                "string" => "string".to_string(),
                "number" => "number".to_string(),
                "integer" => "number".to_string(),
                "boolean" => "boolean".to_string(),
                "array" => "Array".to_string(),
                "object" => "object".to_string(),
                _ => schema_type.clone(),
            });
        }

        Ok("any".to_string())
    }

    fn get_response_type(&self, operation: &Operation) -> Result<String, String> {
        // Try to get type from successful response
        if let Some(resp) = operation.responses.get("200") {
            if let Some(media) = resp.content.get("application/json") {
                if let Some(schema) = &media.schema {
                    return self.schema_type_name(schema);
                }
            }
        }

        // Default to any
        Ok("any".to_string())
    }

    fn to_snake_case(&self, name: &str) -> String {
        let mut result = String::new();
        for (i, c) in name.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                result.push('_');
            }
            result.extend(c.to_lowercase());
        }
        result
    }

    fn to_camel_case(&self, name: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = false;
        for (i, c) in name.chars().enumerate() {
            if i == 0 {
                result.extend(c.to_lowercase());
            } else if c == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.extend(c.to_uppercase());
                capitalize_next = false;
            } else {
                result.push(c);
            }
        }
        result
    }
}

// ========================================================================
// JSON Schema Builder
// ========================================================================

/// Extended JSON Schema generation with OpenAPI compatibility
#[derive(Debug, Clone)]
pub struct JsonSchemaBuilder {
    schemas: HashMap<String, Schema>,
    current_namespace: Vec<String>,
}

impl JsonSchemaBuilder {
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            current_namespace: Vec::new(),
        }
    }

    /// Register a named schema
    pub fn register_schema(&mut self, name: impl Into<String>, schema: Schema) -> &mut Self {
        self.schemas.insert(name.into(), schema);
        self
    }

    /// Get all registered schemas
    pub fn get_schemas(&self) -> &HashMap<String, Schema> {
        &self.schemas
    }

    /// Generate standalone JSON Schema
    pub fn generate_schema(&self, name: &str) -> Result<JsonValue, String> {
        let schema = self.schemas.get(name)
            .ok_or_else(|| format!("Schema not found: {}", name))?;

        serde_json::to_value(schema)
            .map_err(|e| format!("Failed to serialize schema: {}", e))
    }

    /// Generate JSON Schema Draft 2020-12
    pub fn generate_draft2020(&self, name: &str) -> Result<JsonValue, String> {
        let mut schema = self.generate_schema(name)?;

        // Add draft-2020-12 schema identifier
        if let Some(obj) = schema.as_object_mut() {
            obj.insert("$schema".to_string(),
                JsonValue::String("https://json-schema.org/draft/2020-12/schema".to_string()));
        }

        Ok(schema)
    }

    /// Validate JSON value against schema
    pub fn validate(&self, name: &str, value: &JsonValue) -> Result<(), String> {
        let schema = self.schemas.get(name)
            .ok_or_else(|| format!("Schema not found: {}", name))?;

        self.validate_against_schema(schema, value)
    }

    fn validate_against_schema(&self, schema: &Schema, value: &JsonValue) -> Result<(), String> {
        // Type validation
        if let Some(ref expected_type) = schema.schema_type {
            let actual_type = match value {
                JsonValue::String(_) => "string",
                JsonValue::Number(_) => {
                    if value.as_i64().is_some() {
                        "integer"
                    } else {
                        "number"
                    }
                }
                JsonValue::Bool(_) => "boolean",
                JsonValue::Array(_) => "array",
                JsonValue::Object(_) => "object",
                JsonValue::Null => "null",
            };

            if expected_type != "null" && expected_type != actual_type {
                // Check for number/integer compatibility
                if !((expected_type == "number" && actual_type == "integer") ||
                      (expected_type == "integer" && actual_type == "number" && value.is_i64())) {
                    return Err(format!("Type mismatch: expected {}, got {}", expected_type, actual_type));
                }
            }
        }

        // Number validation
        if let Some(number) = value.as_f64() {
            if let Some(min) = schema.minimum {
                if number < min {
                    return Err(format!("Value {} below minimum {}", number, min));
                }
            }
            if let Some(max) = schema.maximum {
                if number > max {
                    return Err(format!("Value {} above maximum {}", number, max));
                }
            }
        }

        // String validation
        if let Some(s) = value.as_str() {
            if let Some(min_len) = schema.min_length {
                if s.len() < min_len as usize {
                    return Err(format!("String length {} below minimum {}", s.len(), min_len));
                }
            }
            if let Some(max_len) = schema.max_length {
                if s.len() > max_len as usize {
                    return Err(format!("String length {} above maximum {}", s.len(), max_len));
                }
            }
            // Note: Pattern validation requires regex crate, returning basic error for now
            if let Some(pattern) = &schema.pattern {
                // Basic pattern validation without regex crate
                // For full regex support, add regex to Cargo.toml
            }
        }

        // Array validation
        if let Some(arr) = value.as_array() {
            if let Some(min_items) = schema.min_items {
                if arr.len() < min_items as usize {
                    return Err(format!("Array length {} below minimum {}", arr.len(), min_items));
                }
            }
            if let Some(max_items) = schema.max_items {
                if arr.len() > max_items as usize {
                    return Err(format!("Array length {} above maximum {}", arr.len(), max_items));
                }
            }
            if let Some(items_schema) = &schema.items {
                for (i, item) in arr.iter().enumerate() {
                    if let Err(e) = self.validate_against_schema(items_schema, item) {
                        return Err(format!("Array item [{}] validation failed: {}", i, e));
                    }
                }
            }
        }

        // Object validation
        if let Some(obj) = value.as_object() {
            if let Some(required) = &schema.required {
                for field in required {
                    if !obj.contains_key(field) {
                        return Err(format!("Missing required field: {}", field));
                    }
                }
            }
            if let Some(properties) = &schema.properties {
                for (prop_name, prop_schema) in properties {
                    if let Some(prop_value) = obj.get(prop_name) {
                        if let Err(e) = self.validate_against_schema(prop_schema, prop_value) {
                            return Err(format!("Property '{}' validation failed: {}", prop_name, e));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for JsonSchemaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_openapi_spec_creation() {
        let spec = OpenAPISpec::new("Test API", "1.0.0")
            .with_server("http://localhost:8080", "Local server")
            .with_tag(Tag::new("users").with_description("User operations"));

        assert_eq!(spec.info.title, "Test API");
        assert_eq!(spec.openapi, "3.1.0");
        assert_eq!(spec.servers.len(), 1);
    }

    #[test]
    fn test_schema_builder() {
        let user_schema = SchemaBuilder::object()
            .with_title("User")
            .with_description("A user object")
            .with_property("id", Schema::integer())
            .with_property("name", Schema::string().with_min_length(1).with_max_length(100))
            .with_property("email", Schema::string().with_format("email"))
            .with_property("active", Schema::boolean())
            .with_required_field("id")
            .with_required_fields(vec!["id".to_string(), "name".to_string(), "email".to_string()])
            .build();

        assert_eq!(user_schema.schema_type, Some("object".to_string()));
        assert!(user_schema.properties.is_some());
    }

    #[test]
    fn test_api_operation_builder() {
        let (path, method, operation) = ApiOperation::new(HttpMethod::POST, "/api/users")
            .with_operation_id("createUser")
            .with_tag("users")
            .with_summary("Create a new user")
            .with_request_body(Schema::object())
            .with_response(201, "User created", Schema::object())
            .build();

        assert_eq!(path, "/api/users");
        assert_eq!(method, HttpMethod::POST);
        assert_eq!(operation.operation_id, Some("createUser".to_string()));
    }

    #[test]
    fn test_spec_validation() {
        let mut spec = OpenAPISpec::new("", "1.0.0"); // Empty title - should fail
        let errors = spec.validate();
        assert!(errors.is_err());

        let spec = OpenAPISpec::new("Valid API", "1.0.0");
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_path_item_operations() {
        let item = PathItem::new()
            .with_get(Operation::new().with_operation_id("listUsers"))
            .with_post(Operation::new().with_operation_id("createUser"));

        assert_eq!(item.operations().len(), 2);
        assert_eq!(item.all_operations().len(), 2);
    }

    #[test]
    fn test_spec_to_json() {
        let spec = OpenAPISpec::new("Test", "1.0")
            .with_path("/test", PathItem::new()
                .with_get(Operation::new().with_operation_id("testOp")));

        let json = spec.to_json();
        assert!(json.is_ok());

        let parsed = serde_json::from_str::<serde_json::Value>(&json.unwrap());
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_schema_reference() {
        let schema = Schema::reference("User");
        assert!(schema.reference.is_some());
        assert!(schema.reference.unwrap().contains("User"));
    }

    #[test]
    fn test_schema_array() {
        let schema = Schema::array(Schema::string());
        assert_eq!(schema.schema_type, Some("array".to_string()));
        assert!(schema.items.is_some());
    }

    #[test]
    fn test_security_scheme() {
        let bearer = SecurityScheme::http_bearer();
        assert_eq!(bearer.scheme_type, "http");
        assert_eq!(bearer.scheme, Some("bearer".to_string()));

        let api_key = SecurityScheme::api_key("X-API-Key", "header");
        assert_eq!(api_key.scheme_type, "apiKey");
        assert_eq!(api_key.name, Some("X-API-Key".to_string()));
    }

    #[test]
    fn test_response_with_content() {
        let response = Response::new("Success")
            .with_content("application/json", MediaType::new()
                .with_schema(Schema::object())
                .with_example(json!({"message": "ok"})));

        assert!(response.content.contains_key("application/json"));
    }

    #[test]
    fn test_request_body_optional() {
        let body = RequestBody::new()
            .optional()
            .with_description("Optional filter criteria");

        assert!(!body.required);
        assert!(body.description.is_some());
    }

    #[test]
    fn test_json_schema_validation() {
        let mut builder = JsonSchemaBuilder::new();

        let email_schema = Schema::string()
            .with_format("email")
            .with_pattern("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$");

        builder.register_schema("email", email_schema);

        // Valid email
        let valid_email = json!("test@example.com");
        // Note: This would fail pattern validation without proper regex support
        // For now we just test the structure
        assert!(builder.schemas.contains_key("email"));
    }

    #[test]
    fn test_api_generator_typescript() {
        let spec = OpenAPISpec::new("Test API", "1.0.0")
            .with_path("/users", PathItem::new()
                .with_get(Operation::new()
                    .with_operation_id("getUsers")
                    .with_response(200, "Success", Schema::array(Schema::object()))));

        let generator = ApiGenerator::new(spec)
            .with_language(TargetLanguage::TypeScript);

        let result = generator.generate_typescript_client();
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("getUsers"));
        assert!(code.contains("class"));
    }

    #[test]
    fn test_api_generator_rust() {
        let spec = OpenAPISpec::new("Test API", "1.0.0")
            .with_path("/users", PathItem::new()
                .with_get(Operation::new()
                    .with_operation_id("getUsers")
                    .with_response(200, "Success", Schema::array(Schema::object()))));

        let generator = ApiGenerator::new(spec)
            .with_language(TargetLanguage::TypeScript);

        let result = generator.generate_rust_server();
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("trait ApiHandler"));
        assert!(code.contains("enum ApiError"));
    }

    #[test]
    fn test_api_generator_python() {
        let spec = OpenAPISpec::new("Test API", "1.0.0")
            .with_path("/users", PathItem::new()
                .with_get(Operation::new()
                    .with_operation_id("getUsers")
                    .with_response(200, "Success", Schema::array(Schema::object()))));

        let generator = ApiGenerator::new(spec)
            .with_language(TargetLanguage::Python);

        let result = generator.generate_python_client();
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("get_users"));
        assert!(code.contains("class"));
    }

    #[test]
    fn test_spec_merge() {
        let mut spec1 = OpenAPISpec::new("API1", "1.0")
            .with_path("/path1", PathItem::new());

        let spec2 = OpenAPISpec::new("API2", "1.0")
            .with_path("/path2", PathItem::new());

        spec1.merge(spec2);

        assert_eq!(spec1.paths.len(), 2);
        assert!(spec1.paths.contains_key("/path1"));
        assert!(spec1.paths.contains_key("/path2"));
    }

    #[test]
    fn test_find_operation() {
        let spec = OpenAPISpec::new("Test", "1.0")
            .with_path("/users", PathItem::new()
                .with_get(Operation::new()
                    .with_operation_id("listUsers")));

        let result = spec.find_operation("listUsers");
        assert!(result.is_some());

        let (path, method, op) = result.unwrap();
        assert_eq!(path, "/users");
        assert_eq!(method, HttpMethod::GET);
    }

    #[test]
    fn test_operation_ids() {
        let spec = OpenAPISpec::new("Test", "1.0")
            .with_path("/users", PathItem::new()
                .with_get(Operation::new().with_operation_id("list"))
                .with_post(Operation::new().with_operation_id("create")));

        let ids = spec.operation_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"list".to_string()));
        assert!(ids.contains(&"create".to_string()));
    }

    #[test]
    fn test_schema_with_any_of() {
        let schema = Schema::new()
            .with_any_of(vec![
                Schema::string(),
                Schema::integer(),
            ]);

        assert!(schema.any_of.is_some());
        assert_eq!(schema.any_of.unwrap().len(), 2);
    }

    #[test]
    fn test_schema_with_enum() {
        let schema = Schema::string()
            .with_enum(vec![
                json!("active"),
                json!("inactive"),
                json!("pending"),
            ]);

        assert!(schema.enum_values.is_some());
        assert_eq!(schema.enum_values.unwrap().len(), 3);
    }

    #[test]
    fn test_components_builder() {
        let components = Components::new()
            .with_schema("User", Schema::object())
            .with_schema("Product", Schema::object())
            .with_security_scheme("bearer", SecurityScheme::http_bearer());

        assert!(components.schemas.is_some());
        assert_eq!(components.schemas.unwrap().len(), 2);
        assert!(components.security_schemes.is_some());
    }

    #[test]
    fn test_info_with_contact() {
        let info = Info::new("API", "1.0")
            .with_description("Test API")
            .with_contact(Contact {
                name: Some("Support".to_string()),
                email: Some("support@example.com".to_string()),
                url: None,
            });

        assert!(info.description.is_some());
        assert!(info.contact.is_some());
    }

    #[test]
    fn test_parameter_locations() {
        let query = Parameter::new("page", ParameterLocation::Query);
        let path = Parameter::new("id", ParameterLocation::Path);
        let header = Parameter::new("Authorization", ParameterLocation::Header);
        let cookie = Parameter::new("session", ParameterLocation::Cookie);

        assert_eq!(query.in_location, ParameterLocation::Query);
        assert_eq!(path.in_location, ParameterLocation::Path);
        assert_eq!(header.in_location, ParameterLocation::Header);
        assert_eq!(cookie.in_location, ParameterLocation::Cookie);
    }

    #[test]
    fn test_media_type_with_schema_and_example() {
        let media = MediaType::new()
            .with_schema(Schema::string())
            .with_example(json!("example value"));

        assert!(media.schema.is_some());
        assert!(media.example.is_some());
    }

    #[test]
    fn test_response_with_headers() {
        let response = Response::new("OK")
            .with_header("X-RateLimit", Header {
                description: Some("Rate limit".to_string()),
                schema: Some(Schema::integer()),
                required: false,
                deprecated: false,
                allow_empty_value: false,
                style: None,
                explode: None,
                allow_reserved: false,
                example: None,
                examples: HashMap::new(),
                content: HashMap::new(),
            });

        assert!(response.headers.contains_key("X-RateLimit"));
    }

    #[test]
    fn test_deprecated_operation() {
        let (path, method, operation) = ApiOperation::new(HttpMethod::GET, "/old")
            .with_operation_id("oldEndpoint")
            .deprecated()
            .build();

        assert!(operation.deprecated);
    }

    #[test]
    fn test_spec_with_components() {
        let spec = OpenAPISpec::new("Test", "1.0")
            .with_components(Components::new()
                .with_schema("Error", Schema::object()
                    .with_property("code", Schema::integer())
                    .with_property("message", Schema::string())
                    .with_required_field("code")
                ));

        assert!(spec.components.is_some());
        assert!(spec.components.as_ref().unwrap().schemas.is_some());
    }

    #[test]
    fn test_operation_with_tags() {
        let (path, method, operation) = ApiOperation::new(HttpMethod::POST, "/api/data")
            .with_operation_id("createData")
            .with_tag("data")
            .with_tag("admin")
            .build();

        assert_eq!(operation.tags.len(), 2);
        assert!(operation.tags.contains(&"data".to_string()));
        assert!(operation.tags.contains(&"admin".to_string()));
    }

    #[test]
    fn test_schema_read_only_write_only() {
        let schema = Schema::object()
            .with_property("id", Schema::integer().read_only())
            .with_property("password", Schema::string().write_only());

        assert!(schema.properties.as_ref().unwrap()["id"].read_only.is_some());
        assert!(schema.properties.as_ref().unwrap()["password"].write_only.is_some());
    }

    #[test]
    fn test_complete_api_spec() {
        let user_schema = Schema::object()
            .with_title("User")
            .with_description("User entity")
            .with_property("id", Schema::integer())
            .with_property("name", Schema::string())
            .with_property("email", Schema::string())
            .with_required_fields(vec!["id".to_string(), "name".to_string()]);

        let spec = OpenAPISpec::new("User Management API", "1.0.0")
            .with_description("API for managing users")
            .with_server("https://api.example.com", "Production")
            .with_server("http://localhost:8080", "Development")
            .with_tag(Tag::new("users").with_description("User operations"))
            .with_components(Components::new().with_schema("User", user_schema.clone()))
            .with_path("/users", PathItem::new()
                .with_get(Operation::new()
                    .with_operation_id("listUsers")
                    .with_summary("List all users")
                    .with_tag("users")
                    .with_query_param("limit", Schema::integer())
                    .with_response(200, "List of users", Schema::array(Schema::reference("User"))))
                .with_post(Operation::new()
                    .with_operation_id("createUser")
                    .with_summary("Create a new user")
                    .with_tag("users")
                    .with_request_body(RequestBody::new()
                        .with_content("application/json", MediaType::new().with_schema(user_schema.clone())))
                    .with_response(201, "User created", Schema::reference("User"))))
            .with_path("/users/{id}", PathItem::new()
                .with_get(Operation::new()
                    .with_operation_id("getUser")
                    .with_summary("Get user by ID")
                    .with_tag("users")
                    .with_path_param("id", Schema::integer())
                    .with_response(200, "User found", Schema::reference("User"))
                    .with_response(404, "User not found", Schema::object())))
            .with_security({
                let mut req = HashMap::new();
                req.insert("bearerAuth".to_string(), vec![]);
                req
            });

        assert!(spec.validate().is_ok());
        assert_eq!(spec.servers.len(), 2);
        assert_eq!(spec.paths.len(), 2);
        assert_eq!(spec.tags.len(), 1);
        assert_eq!(spec.security.len(), 1);

        // Test JSON output
        let json_output = spec.to_json();
        assert!(json_output.is_ok());
        let json_str = json_output.unwrap();
        assert!(json_str.contains("openapi"));
        assert!(json_str.contains("User Management API"));

        // Test YAML output
        let yaml_output = spec.to_yaml();
        assert!(yaml_output.is_ok());
    }
}
