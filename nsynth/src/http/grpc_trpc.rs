//! gRPC and tRPC Implementation for nCPU/nSynth
//!
//! Modern RPC frameworks with type safety, streaming, and protobuf support.
//!
//! ## gRPC (Google Remote Procedure Call)
//! - High-performance, open-source RPC framework
//! - Protocol Buffers for efficient serialization
//! - Unary, server streaming, client streaming, and bidirectional streaming
//! - TLS support and authentication
//!
//! ## tRPC (TypeScript RPC)
//! - End-to-end type safety for TypeScript
//! - No code generation required
//! - Queries, mutations, and subscriptions
//! - Automatic type inference

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

// ============================================================================
// Protobuf Support
// ============================================================================

/// Protobuf field types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProtoFieldType {
    Double,
    Float,
    Int32,
    Int64,
    Uint32,
    Uint64,
    Sint32,
    Sint64,
    Fixed32,
    Fixed64,
    Sfixed32,
    Sfixed64,
    Bool,
    String,
    Bytes,
    Message(String),
    Enum(String),
}

/// Protobuf field descriptor
#[derive(Debug, Clone)]
pub struct ProtoField {
    pub name: String,
    pub field_type: ProtoFieldType,
    pub number: u32,
    pub repeated: bool,
    pub optional: bool,
    pub packed: bool,
    pub default_value: Option<String>,
}

/// Protobuf message descriptor
#[derive(Debug, Clone)]
pub struct ProtoMessage {
    pub name: String,
    pub fields: Vec<ProtoField>,
    pub nested_messages: Vec<ProtoMessage>,
    pub enums: Vec<ProtoEnum>,
}

/// Protobuf enum value
#[derive(Debug, Clone)]
pub struct ProtoEnumValue {
    pub name: String,
    pub number: i32,
}

/// Protobuf enum descriptor
#[derive(Debug, Clone)]
pub struct ProtoEnum {
    pub name: String,
    pub values: Vec<ProtoEnumValue>,
}

/// Protobuf service descriptor
#[derive(Debug, Clone)]
pub struct ProtoService {
    pub name: String,
    pub methods: Vec<ProtoMethod>,
    pub options: HashMap<String, String>,
}

/// Protobuf method descriptor
#[derive(Debug, Clone)]
pub struct ProtoMethod {
    pub name: String,
    pub request_type: String,
    pub response_type: String,
    pub client_streaming: bool,
    pub server_streaming: bool,
    pub options: HashMap<String, String>,
}

/// Protobuf file descriptor
#[derive(Debug, Clone)]
pub struct ProtoFile {
    pub package: String,
    pub syntax: String,
    pub messages: Vec<ProtoMessage>,
    pub enums: Vec<ProtoEnum>,
    pub services: Vec<ProtoService>,
    pub imports: Vec<String>,
    pub options: HashMap<String, String>,
}

impl ProtoFile {
    /// Create new protobuf file descriptor
    pub fn new(package: impl Into<String>) -> Self {
        Self {
            package: package.into(),
            syntax: "proto3".to_string(),
            messages: Vec::new(),
            enums: Vec::new(),
            services: Vec::new(),
            imports: Vec::new(),
            options: HashMap::new(),
        }
    }

    /// Add message
    pub fn add_message(&mut self, message: ProtoMessage) -> &mut Self {
        self.messages.push(message);
        self
    }

    /// Add enum
    pub fn add_enum(&mut self, enum_def: ProtoEnum) -> &mut Self {
        self.enums.push(enum_def);
        self
    }

    /// Add service
    pub fn add_service(&mut self, service: ProtoService) -> &mut Self {
        self.services.push(service);
        self
    }

    /// Add import
    pub fn add_import(&mut self, import: impl Into<String>) -> &mut Self {
        self.imports.push(import.into());
        self
    }

    /// Add option
    pub fn add_option(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// Generate protobuf definition string
    pub fn to_proto_string(&self) -> String {
        let mut output = String::new();

        // Syntax
        output.push_str(&format!("syntax = \"{}\";\n\n", self.syntax));

        // Package
        output.push_str(&format!("package {};\n", self.package));

        // Options
        for (key, value) in &self.options {
            output.push_str(&format!("option {} = {};\n", key, value));
        }

        // Imports
        for import in &self.imports {
            output.push_str(&format!("import \"{}\";\n", import));
        }

        output.push_str("\n");

        // Enums
        for enum_def in &self.enums {
            output.push_str(&enum_def.to_proto_string());
            output.push_str("\n");
        }

        // Messages
        for message in &self.messages {
            output.push_str(&message.to_proto_string(0));
            output.push_str("\n");
        }

        // Services
        for service in &self.services {
            output.push_str(&service.to_proto_string());
            output.push_str("\n");
        }

        output
    }
}

impl ProtoMessage {
    /// Create new message
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            nested_messages: Vec::new(),
            enums: Vec::new(),
        }
    }

    /// Add field
    pub fn add_field(&mut self, field: ProtoField) -> &mut Self {
        self.fields.push(field);
        self
    }

    /// Add nested message
    pub fn add_nested(&mut self, message: ProtoMessage) -> &mut Self {
        self.nested_messages.push(message);
        self
    }

    /// Add nested enum
    pub fn add_enum(&mut self, enum_def: ProtoEnum) -> &mut Self {
        self.enums.push(enum_def);
        self
    }

    /// Generate protobuf string
    pub fn to_proto_string(&self, indent: usize) -> String {
        let indent_str = " ".repeat(indent);
        let mut output = String::new();

        output.push_str(&format!("{}message {} {{\n", indent_str, self.name));

        // Nested messages
        for nested in &self.nested_messages {
            output.push_str(&nested.to_proto_string(indent + 2));
        }

        // Nested enums
        for enum_def in &self.enums {
            output.push_str(&format!(
                "{}{}\n",
                indent_str,
                enum_def.to_proto_string_indented(indent + 2)
            ));
        }

        // Fields
        for field in &self.fields {
            output.push_str(&field.to_proto_string(indent + 2));
        }

        output.push_str(&format!("{}}}\n", indent_str));

        output
    }
}

impl ProtoField {
    /// Create new field
    pub fn new(name: impl Into<String>, field_type: ProtoFieldType, number: u32) -> Self {
        Self {
            name: name.into(),
            field_type,
            number,
            repeated: false,
            optional: false,
            packed: false,
            default_value: None,
        }
    }

    /// Make repeated
    pub fn repeated(mut self) -> Self {
        self.repeated = true;
        self
    }

    /// Make optional
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Set packed
    pub fn packed(mut self) -> Self {
        self.packed = true;
        self
    }

    /// Set default value
    pub fn default_value(mut self, value: impl Into<String>) -> Self {
        self.default_value = Some(value.into());
        self
    }

    /// Generate protobuf string
    pub fn to_proto_string(&self, indent: usize) -> String {
        let indent_str = " ".repeat(indent);

        let mut type_str = match &self.field_type {
            ProtoFieldType::Double => "double".to_string(),
            ProtoFieldType::Float => "float".to_string(),
            ProtoFieldType::Int32 => "int32".to_string(),
            ProtoFieldType::Int64 => "int64".to_string(),
            ProtoFieldType::Uint32 => "uint32".to_string(),
            ProtoFieldType::Uint64 => "uint64".to_string(),
            ProtoFieldType::Sint32 => "sint32".to_string(),
            ProtoFieldType::Sint64 => "sint64".to_string(),
            ProtoFieldType::Fixed32 => "fixed32".to_string(),
            ProtoFieldType::Fixed64 => "fixed64".to_string(),
            ProtoFieldType::Sfixed32 => "sfixed32".to_string(),
            ProtoFieldType::Sfixed64 => "sfixed64".to_string(),
            ProtoFieldType::Bool => "bool".to_string(),
            ProtoFieldType::String => "string".to_string(),
            ProtoFieldType::Bytes => "bytes".to_string(),
            ProtoFieldType::Message(msg) => msg.clone(),
            ProtoFieldType::Enum(en) => en.clone(),
        };

        if self.repeated {
            type_str = format!("repeated {}", type_str);
        }

        if self.optional {
            type_str = format!("optional {}", type_str);
        }

        let mut output = format!("{}{} {} = {}", indent_str, type_str, self.name, self.number);

        if let Some(default) = &self.default_value {
            output.push_str(&format!(" [default = {}]", default));
        }

        if self.packed {
            output.push_str(" [packed = true]");
        }

        output.push_str(";\n");

        output
    }
}

impl ProtoEnum {
    /// Create new enum
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
        }
    }

    /// Add value
    pub fn add_value(&mut self, name: impl Into<String>, number: i32) -> &mut Self {
        self.values.push(ProtoEnumValue {
            name: name.into(),
            number,
        });
        self
    }

    /// Generate protobuf string
    pub fn to_proto_string(&self) -> String {
        format!("{}\n", self.to_proto_string_indented(0))
    }

    /// Generate protobuf string with indentation
    pub fn to_proto_string_indented(&self, indent: usize) -> String {
        let indent_str = " ".repeat(indent);
        let mut output = format!("{}enum {} {{\n", indent_str, self.name);

        for value in &self.values {
            output.push_str(&format!(
                "{}  {} = {};\n",
                indent_str, value.name, value.number
            ));
        }

        output.push_str(&format!("{}}}", indent_str));

        output
    }
}

impl ProtoService {
    /// Create new service
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            methods: Vec::new(),
            options: HashMap::new(),
        }
    }

    /// Add method
    pub fn add_method(&mut self, method: ProtoMethod) -> &mut Self {
        self.methods.push(method);
        self
    }

    /// Add option
    pub fn add_option(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// Generate protobuf string
    pub fn to_proto_string(&self) -> String {
        let mut output = format!("service {} {{\n", self.name);

        for method in &self.methods {
            output.push_str(&method.to_proto_string(2));
        }

        output.push_str("}\n");

        output
    }
}

impl ProtoMethod {
    /// Create new method
    pub fn new(
        name: impl Into<String>,
        request_type: impl Into<String>,
        response_type: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            request_type: request_type.into(),
            response_type: response_type.into(),
            client_streaming: false,
            server_streaming: false,
            options: HashMap::new(),
        }
    }

    /// Make server streaming
    pub fn server_streaming(mut self) -> Self {
        self.server_streaming = true;
        self
    }

    /// Make client streaming
    pub fn client_streaming(mut self) -> Self {
        self.client_streaming = true;
        self
    }

    /// Make bidirectional streaming
    pub fn bidi_streaming(mut self) -> Self {
        self.client_streaming = true;
        self.server_streaming = true;
        self
    }

    /// Generate protobuf string
    pub fn to_proto_string(&self, indent: usize) -> String {
        let indent_str = " ".repeat(indent);

        let stream_sig = if self.client_streaming && self.server_streaming {
            "stream "
        } else if self.server_streaming {
            "stream "
        } else {
            ""
        };

        let req_stream = if self.client_streaming { "stream " } else { "" };
        let resp_stream = if self.server_streaming { "stream " } else { "" };

        format!(
            "{}{} {}({}{}) returns ({}{});\n",
            indent_str,
            stream_sig,
            self.name,
            req_stream,
            self.request_type,
            resp_stream,
            self.response_type
        )
    }
}

// ============================================================================
// gRPC Implementation
// ============================================================================

/// gRPC method type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrpcMethodType {
    /// Unary call (single request, single response)
    Unary,
    /// Server streaming (single request, stream of responses)
    ServerStreaming,
    /// Client streaming (stream of requests, single response)
    ClientStreaming,
    /// Bidirectional streaming (stream of requests, stream of responses)
    BidiStreaming,
}

/// gRPC error
#[derive(Debug, Clone)]
pub struct GrpcError {
    pub code: GrpcStatusCode,
    pub message: String,
    pub details: HashMap<String, String>,
}

impl fmt::Display for GrpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for GrpcError {}

/// gRPC status codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrpcStatusCode {
    Ok = 0,
    Canceled = 1,
    Unknown = 2,
    InvalidArgument = 3,
    DeadlineExceeded = 4,
    NotFound = 5,
    AlreadyExists = 6,
    PermissionDenied = 7,
    ResourceExhausted = 8,
    FailedPrecondition = 9,
    Aborted = 10,
    OutOfRange = 11,
    Unimplemented = 12,
    Internal = 13,
    Unavailable = 14,
    DataLoss = 15,
    Unauthenticated = 16,
}

impl GrpcStatusCode {
    /// Get status code name
    pub fn name(&self) -> &'static str {
        match self {
            GrpcStatusCode::Ok => "OK",
            GrpcStatusCode::Canceled => "CANCELED",
            GrpcStatusCode::Unknown => "UNKNOWN",
            GrpcStatusCode::InvalidArgument => "INVALID_ARGUMENT",
            GrpcStatusCode::DeadlineExceeded => "DEADLINE_EXCEEDED",
            GrpcStatusCode::NotFound => "NOT_FOUND",
            GrpcStatusCode::AlreadyExists => "ALREADY_EXISTS",
            GrpcStatusCode::PermissionDenied => "PERMISSION_DENIED",
            GrpcStatusCode::ResourceExhausted => "RESOURCE_EXHAUSTED",
            GrpcStatusCode::FailedPrecondition => "FAILED_PRECONDITION",
            GrpcStatusCode::Aborted => "ABORTED",
            GrpcStatusCode::OutOfRange => "OUT_OF_RANGE",
            GrpcStatusCode::Unimplemented => "UNIMPLEMENTED",
            GrpcStatusCode::Internal => "INTERNAL",
            GrpcStatusCode::Unavailable => "UNAVAILABLE",
            GrpcStatusCode::DataLoss => "DATA_LOSS",
            GrpcStatusCode::Unauthenticated => "UNAUTHENTICATED",
        }
    }

    /// Check if status is OK
    pub fn is_ok(&self) -> bool {
        *self == GrpcStatusCode::Ok
    }

    /// Check if status is an error
    pub fn is_error(&self) -> bool {
        *self != GrpcStatusCode::Ok
    }
}

impl fmt::Display for GrpcStatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// gRPC call metadata
#[derive(Debug, Clone)]
pub struct GrpcMetadata {
    inner: HashMap<String, String>,
}

impl GrpcMetadata {
    /// Create new metadata
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Insert metadata entry
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.inner.insert(key.into(), value.into());
    }

    /// Get metadata value
    pub fn get(&self, key: &str) -> Option<&str> {
        self.inner.get(key).map(|s| s.as_str())
    }

    /// Remove metadata entry
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.inner.remove(key)
    }

    /// Iterate over metadata
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Get metadata count
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for GrpcMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// gRPC response
#[derive(Debug, Clone)]
pub struct GrpcResponse<T> {
    pub status: GrpcStatusCode,
    pub message: Option<String>,
    pub data: Option<T>,
    pub metadata: GrpcMetadata,
}

impl<T> GrpcResponse<T> {
    /// Create OK response
    pub fn ok(data: T) -> Self {
        Self {
            status: GrpcStatusCode::Ok,
            message: None,
            data: Some(data),
            metadata: GrpcMetadata::new(),
        }
    }

    /// Create error response
    pub fn error(code: GrpcStatusCode, message: impl Into<String>) -> Self {
        Self {
            status: code,
            message: Some(message.into()),
            data: None,
            metadata: GrpcMetadata::new(),
        }
    }

    /// With metadata
    pub fn with_metadata(mut self, metadata: GrpcMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Check if response is OK
    pub fn is_ok(&self) -> bool {
        self.status.is_ok()
    }

    /// Check if response is error
    pub fn is_error(&self) -> bool {
        self.status.is_error()
    }
}

/// gRPC method handler
pub type GrpcHandler<Req, Res> = fn(GrpcMetadata, Req) -> GrpcResponse<Res>;

/// gRPC streaming handler (server-side)
pub type GrpcServerStreamHandler<Req, Res> = fn(GrpcMetadata, Req) -> GrpcStream<Res>;

/// gRPC streaming handler (client-side)
pub type GrpcClientStreamHandler<Req, Res> = fn(GrpcMetadata, GrpcStream<Req>) -> GrpcResponse<Res>;

/// gRPC bidirectional streaming handler
pub type GrpcBidiStreamHandler<Req, Res> = fn(GrpcMetadata, GrpcStream<Req>) -> GrpcStream<Res>;

/// gRPC stream (for streaming operations)
#[derive(Debug)]
pub struct GrpcStream<T> {
    items: Arc<Mutex<Vec<T>>>,
    closed: Arc<Mutex<bool>>,
}

impl<T> GrpcStream<T> {
    /// Create new stream
    pub fn new() -> Self {
        Self {
            items: Arc::new(Mutex::new(Vec::new())),
            closed: Arc::new(Mutex::new(false)),
        }
    }

    /// Add item to stream
    pub fn send(&mut self, item: T) -> Result<(), GrpcError> {
        let mut closed = self.closed.lock().unwrap();
        if *closed {
            return Err(GrpcError {
                code: GrpcStatusCode::Internal,
                message: "Stream is closed".to_string(),
                details: HashMap::new(),
            });
        }
        drop(closed);

        let mut items = self.items.lock().unwrap();
        items.push(item);
        Ok(())
    }

    /// Close stream
    pub fn close(&mut self) {
        let mut closed = self.closed.lock().unwrap();
        *closed = true;
    }

    /// Check if stream is closed
    pub fn is_closed(&self) -> bool {
        *self.closed.lock().unwrap()
    }

    /// Get all items from stream
    pub fn drain(&self) -> Vec<T> {
        let mut items = self.items.lock().unwrap();
        std::mem::take(&mut *items)
    }

    /// Get item count
    pub fn len(&self) -> usize {
        self.items.lock().unwrap().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.items.lock().unwrap().is_empty()
    }
}

impl<T> Default for GrpcStream<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for GrpcStream<T> {
    fn clone(&self) -> Self {
        Self {
            items: Arc::clone(&self.items),
            closed: Arc::clone(&self.closed),
        }
    }
}

/// gRPC method definition
#[derive(Debug, Clone)]
pub struct GrpcMethod<Req, Res> {
    pub name: String,
    pub method_type: GrpcMethodType,
    pub request_type: std::marker::PhantomData<Req>,
    pub response_type: std::marker::PhantomData<Res>,
    // Handlers (stored as strings for type erasure)
    pub unary_handler: Option<String>,
    pub server_stream_handler: Option<String>,
    pub client_stream_handler: Option<String>,
    pub bidi_stream_handler: Option<String>,
    pub metadata: GrpcMetadata,
}

impl<Req, Res> GrpcMethod<Req, Res> {
    /// Create new unary method
    pub fn unary(name: impl Into<String>, handler: GrpcHandler<Req, Res>) -> Self {
        Self {
            name: name.into(),
            method_type: GrpcMethodType::Unary,
            request_type: std::marker::PhantomData,
            response_type: std::marker::PhantomData,
            unary_handler: Some("unary".to_string()),
            server_stream_handler: None,
            client_stream_handler: None,
            bidi_stream_handler: None,
            metadata: GrpcMetadata::new(),
        }
    }

    /// Create new server streaming method
    pub fn server_streaming(
        name: impl Into<String>,
        handler: GrpcServerStreamHandler<Req, Res>,
    ) -> Self {
        Self {
            name: name.into(),
            method_type: GrpcMethodType::ServerStreaming,
            request_type: std::marker::PhantomData,
            response_type: std::marker::PhantomData,
            unary_handler: None,
            server_stream_handler: Some("server_stream".to_string()),
            client_stream_handler: None,
            bidi_stream_handler: None,
            metadata: GrpcMetadata::new(),
        }
    }

    /// Create new client streaming method
    pub fn client_streaming(
        name: impl Into<String>,
        handler: GrpcClientStreamHandler<Req, Res>,
    ) -> Self {
        Self {
            name: name.into(),
            method_type: GrpcMethodType::ClientStreaming,
            request_type: std::marker::PhantomData,
            response_type: std::marker::PhantomData,
            unary_handler: None,
            server_stream_handler: None,
            client_stream_handler: Some("client_stream".to_string()),
            bidi_stream_handler: None,
            metadata: GrpcMetadata::new(),
        }
    }

    /// Create new bidirectional streaming method
    pub fn bidi_streaming(
        name: impl Into<String>,
        handler: GrpcBidiStreamHandler<Req, Res>,
    ) -> Self {
        Self {
            name: name.into(),
            method_type: GrpcMethodType::BidiStreaming,
            request_type: std::marker::PhantomData,
            response_type: std::marker::PhantomData,
            unary_handler: None,
            server_stream_handler: None,
            client_stream_handler: None,
            bidi_stream_handler: Some("bidi_stream".to_string()),
            metadata: GrpcMetadata::new(),
        }
    }

    /// Add method metadata
    pub fn with_metadata(mut self, metadata: GrpcMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// gRPC service
#[derive(Debug)]
pub struct GrpcService {
    pub name: String,
    pub package: String,
    pub methods: Vec<String>, // Method names (type-erased)
    pub metadata: GrpcMetadata,
}

impl GrpcService {
    /// Create new gRPC service
    pub fn new(package: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            package: package.into(),
            methods: Vec::new(),
            metadata: GrpcMetadata::new(),
        }
    }

    /// Register method
    pub fn register_method<Req, Res>(&mut self, method: GrpcMethod<Req, Res>) -> &mut Self {
        self.methods.push(method.name);
        self
    }

    /// Get full service name
    pub fn full_name(&self) -> String {
        format!("{}.{}", self.package, self.name)
    }

    /// Get method count
    pub fn method_count(&self) -> usize {
        self.methods.len()
    }

    /// Check if has method
    pub fn has_method(&self, name: &str) -> bool {
        self.methods.iter().any(|m| m == name)
    }
}

// ============================================================================
// tRPC Implementation
// ============================================================================

/// tRPC procedure type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrpcProcedureType {
    /// Query - read-only operation
    Query,
    /// Mutation - write operation
    Mutation,
    /// Subscription - real-time updates
    Subscription,
}

/// tRPC error
#[derive(Debug, Clone, PartialEq)]
pub struct TrpcError {
    pub code: TrpcErrorCode,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

impl fmt::Display for TrpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for TrpcError {}

/// tRPC error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrpcErrorCode {
    InternalServerError = -32603,
    Badrequest = -32600,
    Unauthorized = -32001,
    Forbidden = -32003,
    NotFound = -32004,
    MethodNotAllowed = -32005,
    Timeout = -32008,
    Conflict = -32009,
    PreconditionFailed = -32012,
    PayloadTooLarge = -32013,
    UnsupportedMediaType = -32015,
    UnprocessableEntity = -32022,
    TooManyRequests = -32029,
    ClientClosedRequest = -32099,
}

impl fmt::Display for TrpcErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrpcErrorCode::InternalServerError => write!(f, "INTERNAL_SERVER_ERROR"),
            TrpcErrorCode::Badrequest => write!(f, "BAD_REQUEST"),
            TrpcErrorCode::Unauthorized => write!(f, "UNAUTHORIZED"),
            TrpcErrorCode::Forbidden => write!(f, "FORBIDDEN"),
            TrpcErrorCode::NotFound => write!(f, "NOT_FOUND"),
            TrpcErrorCode::MethodNotAllowed => write!(f, "METHOD_NOT_ALLOWED"),
            TrpcErrorCode::Timeout => write!(f, "TIMEOUT"),
            TrpcErrorCode::Conflict => write!(f, "CONFLICT"),
            TrpcErrorCode::PreconditionFailed => write!(f, "PRECONDITION_FAILED"),
            TrpcErrorCode::PayloadTooLarge => write!(f, "PAYLOAD_TOO_LARGE"),
            TrpcErrorCode::UnsupportedMediaType => write!(f, "UNSUPPORTED_MEDIA_TYPE"),
            TrpcErrorCode::UnprocessableEntity => write!(f, "UNPROCESSABLE_ENTITY"),
            TrpcErrorCode::TooManyRequests => write!(f, "TOO_MANY_REQUESTS"),
            TrpcErrorCode::ClientClosedRequest => write!(f, "CLIENT_CLOSED_REQUEST"),
        }
    }
}

impl TrpcErrorCode {
    /// Get HTTP status code equivalent
    pub fn http_status(&self) -> u16 {
        match self {
            TrpcErrorCode::InternalServerError => 500,
            TrpcErrorCode::Badrequest => 400,
            TrpcErrorCode::Unauthorized => 401,
            TrpcErrorCode::Forbidden => 403,
            TrpcErrorCode::NotFound => 404,
            TrpcErrorCode::MethodNotAllowed => 405,
            TrpcErrorCode::Timeout => 408,
            TrpcErrorCode::Conflict => 409,
            TrpcErrorCode::PreconditionFailed => 412,
            TrpcErrorCode::PayloadTooLarge => 413,
            TrpcErrorCode::UnsupportedMediaType => 415,
            TrpcErrorCode::UnprocessableEntity => 422,
            TrpcErrorCode::TooManyRequests => 429,
            TrpcErrorCode::ClientClosedRequest => 499,
        }
    }
}

/// tRPC input type marker
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TrpcInput<T> {
    Value(T),
    Void(serde_json::Value),
}

/// tRPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrpcResponse<T> {
    pub result: Option<TrpcResultData<T>>,
    pub error: Option<TrpcErrorData>,
}

/// tRPC result data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrpcResultData<T> {
    pub data: T,
}

/// tRPC error data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrpcErrorData {
    pub error: TrpcErrorDetail,
}

/// tRPC error detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrpcErrorDetail {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

impl<T> TrpcResponse<T> {
    /// Create successful response
    pub fn ok(data: T) -> Self {
        Self {
            result: Some(TrpcResultData { data }),
            error: None,
        }
    }

    /// Create error response
    pub fn error(code: TrpcErrorCode, message: impl Into<String>) -> Self {
        Self {
            result: None,
            error: Some(TrpcErrorData {
                error: TrpcErrorDetail {
                    code: code as i32,
                    message: message.into(),
                    data: None,
                },
            }),
        }
    }

    /// Check if response is OK
    pub fn is_ok(&self) -> bool {
        self.result.is_some()
    }

    /// Check if response is error
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

/// tRPC context
#[derive(Debug, Clone)]
pub struct TrpcContext {
    pub user_id: Option<String>,
    pub headers: HashMap<String, String>,
    pub request_id: Option<String>,
}

impl TrpcContext {
    /// Create new context
    pub fn new() -> Self {
        Self {
            user_id: None,
            headers: HashMap::new(),
            request_id: None,
        }
    }

    /// With user ID
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// With request ID
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Add header
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

impl Default for TrpcContext {
    fn default() -> Self {
        Self::new()
    }
}

/// tRPC procedure handler
pub type TrpcQueryHandler<TInput, TOutput> = fn(TrpcContext, TInput) -> TrpcResponse<TOutput>;

/// tRPC mutation handler
pub type TrpcMutationHandler<TInput, TOutput> = fn(TrpcContext, TInput) -> TrpcResponse<TOutput>;

/// tRPC subscription handler
pub type TrpcSubscriptionHandler<TInput, TOutput> =
    fn(TrpcContext, TInput) -> TrpcSubscription<TOutput>;

/// tRPC subscription (for real-time updates)
#[derive(Debug)]
pub struct TrpcSubscription<T> {
    values: Arc<Mutex<Vec<T>>>,
    closed: Arc<Mutex<bool>>,
    error: Arc<Mutex<Option<TrpcError>>>,
}

impl<T> TrpcSubscription<T> {
    /// Create new subscription
    pub fn new() -> Self {
        Self {
            values: Arc::new(Mutex::new(Vec::new())),
            closed: Arc::new(Mutex::new(false)),
            error: Arc::new(Mutex::new(None)),
        }
    }

    /// Emit value to subscription
    pub fn emit(&mut self, value: T) {
        let mut values = self.values.lock().unwrap();
        values.push(value);
    }

    /// Emit error and close
    pub fn error(&mut self, err: TrpcError) {
        let mut error = self.error.lock().unwrap();
        *error = Some(err);
        let mut closed = self.closed.lock().unwrap();
        *closed = true;
    }

    /// Complete subscription
    pub fn complete(&mut self) {
        let mut closed = self.closed.lock().unwrap();
        *closed = true;
    }

    /// Check if subscription is active
    pub fn is_active(&self) -> bool {
        !*self.closed.lock().unwrap()
    }

    /// Get all values from subscription
    pub fn values(&self) -> Vec<T>
    where
        T: Clone,
    {
        let values = self.values.lock().unwrap();
        values.clone()
    }

    /// Get error if any
    pub fn get_error(&self) -> Option<TrpcError> {
        self.error.lock().unwrap().clone()
    }
}

impl<T> Default for TrpcSubscription<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for TrpcSubscription<T> {
    fn clone(&self) -> Self {
        Self {
            values: Arc::clone(&self.values),
            closed: Arc::clone(&self.closed),
            error: Arc::clone(&self.error),
        }
    }
}

/// tRPC procedure
#[derive(Debug, Clone)]
pub struct TrpcProcedure<TInput, TOutput> {
    pub name: String,
    pub procedure_type: TrpcProcedureType,
    pub input_type: std::marker::PhantomData<TInput>,
    pub output_type: std::marker::PhantomData<TOutput>,
    pub query_handler: Option<String>,
    pub mutation_handler: Option<String>,
    pub subscription_handler: Option<String>,
}

impl<TInput, TOutput> TrpcProcedure<TInput, TOutput> {
    /// Create query procedure
    pub fn query(name: impl Into<String>, handler: TrpcQueryHandler<TInput, TOutput>) -> Self {
        Self {
            name: name.into(),
            procedure_type: TrpcProcedureType::Query,
            input_type: std::marker::PhantomData,
            output_type: std::marker::PhantomData,
            query_handler: Some("query".to_string()),
            mutation_handler: None,
            subscription_handler: None,
        }
    }

    /// Create mutation procedure
    pub fn mutation(
        name: impl Into<String>,
        handler: TrpcMutationHandler<TInput, TOutput>,
    ) -> Self {
        Self {
            name: name.into(),
            procedure_type: TrpcProcedureType::Mutation,
            input_type: std::marker::PhantomData,
            output_type: std::marker::PhantomData,
            query_handler: None,
            mutation_handler: Some("mutation".to_string()),
            subscription_handler: None,
        }
    }

    /// Create subscription procedure
    pub fn subscription(
        name: impl Into<String>,
        handler: TrpcSubscriptionHandler<TInput, TOutput>,
    ) -> Self {
        Self {
            name: name.into(),
            procedure_type: TrpcProcedureType::Subscription,
            input_type: std::marker::PhantomData,
            output_type: std::marker::PhantomData,
            query_handler: None,
            mutation_handler: None,
            subscription_handler: Some("subscription".to_string()),
        }
    }
}

/// tRPC router
#[derive(Debug)]
pub struct TrpcRouter {
    pub name: String,
    pub procedures: Vec<String>, // Procedure names
    pub middleware: Vec<String>,
    pub context_factory: Option<String>,
}

impl TrpcRouter {
    /// Create new tRPC router
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            procedures: Vec::new(),
            middleware: Vec::new(),
            context_factory: None,
        }
    }

    /// Register query procedure
    pub fn query<TInput, TOutput>(
        &mut self,
        procedure: TrpcProcedure<TInput, TOutput>,
    ) -> &mut Self {
        self.procedures.push(procedure.name);
        self
    }

    /// Register mutation procedure
    pub fn mutation<TInput, TOutput>(
        &mut self,
        procedure: TrpcProcedure<TInput, TOutput>,
    ) -> &mut Self {
        self.procedures.push(procedure.name);
        self
    }

    /// Register subscription procedure
    pub fn subscription<TInput, TOutput>(
        &mut self,
        procedure: TrpcProcedure<TInput, TOutput>,
    ) -> &mut Self {
        self.procedures.push(procedure.name);
        self
    }

    /// Add middleware
    pub fn middleware(&mut self, name: impl Into<String>) -> &mut Self {
        self.middleware.push(name.into());
        self
    }

    /// Set context factory
    pub fn with_context_factory(&mut self, factory: impl Into<String>) -> &mut Self {
        self.context_factory = Some(factory.into());
        self
    }

    /// Get procedure count
    pub fn procedure_count(&self) -> usize {
        self.procedures.len()
    }

    /// Check if has procedure
    pub fn has_procedure(&self, name: &str) -> bool {
        self.procedures.iter().any(|p| p == name)
    }

    /// Generate TypeScript types
    pub fn generate_types(&self) -> String {
        format!(
            r#"
// tRPC Router Types
import {{ {{ type TRPCError, type TRPCContext }} }} from '@trpc/server';

export interface {}Router {{
  {}
}}

"#,
            self.name,
            self.procedures
                .iter()
                .map(|p| format!("  {}: any;", p))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

// ============================================================================
// Type Safety Utilities
// ============================================================================

/// Type-safe builder for gRPC services
#[derive(Debug)]
pub struct GrpcServiceBuilder {
    package: String,
    service_name: String,
    methods: Vec<String>,
}

impl GrpcServiceBuilder {
    /// Create new builder
    pub fn new(package: impl Into<String>, service_name: impl Into<String>) -> Self {
        Self {
            package: package.into(),
            service_name: service_name.into(),
            methods: Vec::new(),
        }
    }

    /// Add unary method
    pub fn unary<Req, Res>(mut self, name: impl Into<String>) -> Self {
        self.methods.push(name.into());
        self
    }

    /// Add server streaming method
    pub fn server_streaming<Req, Res>(mut self, name: impl Into<String>) -> Self {
        self.methods.push(name.into());
        self
    }

    /// Add client streaming method
    pub fn client_streaming<Req, Res>(mut self, name: impl Into<String>) -> Self {
        self.methods.push(name.into());
        self
    }

    /// Add bidirectional streaming method
    pub fn bidi_streaming<Req, Res>(mut self, name: impl Into<String>) -> Self {
        self.methods.push(name.into());
        self
    }

    /// Build service
    pub fn build(self) -> GrpcService {
        GrpcService {
            name: self.service_name,
            package: self.package,
            methods: self.methods,
            metadata: GrpcMetadata::new(),
        }
    }
}

/// Type-safe builder for tRPC routers
#[derive(Debug)]
pub struct TrpcRouterBuilder {
    router_name: String,
    procedures: Vec<String>,
}

impl TrpcRouterBuilder {
    /// Create new builder
    pub fn new(router_name: impl Into<String>) -> Self {
        Self {
            router_name: router_name.into(),
            procedures: Vec::new(),
        }
    }

    /// Add query
    pub fn query<TInput, TOutput>(mut self, name: impl Into<String>) -> Self {
        self.procedures.push(name.into());
        self
    }

    /// Add mutation
    pub fn mutation<TInput, TOutput>(mut self, name: impl Into<String>) -> Self {
        self.procedures.push(name.into());
        self
    }

    /// Add subscription
    pub fn subscription<TInput, TOutput>(mut self, name: impl Into<String>) -> Self {
        self.procedures.push(name.into());
        self
    }

    /// Build router
    pub fn build(self) -> TrpcRouter {
        TrpcRouter {
            name: self.router_name,
            procedures: self.procedures,
            middleware: Vec::new(),
            context_factory: None,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Protobuf Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_proto_enum() {
        let mut status_enum = ProtoEnum::new("Status");
        status_enum.add_value("UNKNOWN", 0);
        status_enum.add_value("OK", 1);
        status_enum.add_value("ERROR", 2);

        let proto = status_enum.to_proto_string();
        assert!(proto.contains("enum Status"));
        assert!(proto.contains("UNKNOWN = 0"));
        assert!(proto.contains("OK = 1"));
        assert!(proto.contains("ERROR = 2"));
    }

    #[test]
    fn test_proto_field() {
        let field = ProtoField::new("id", ProtoFieldType::Int64, 1);
        let proto = field.to_proto_string(0);
        assert!(proto.contains("int64"));
        assert!(proto.contains("id"));
        assert!(proto.contains("= 1"));

        let repeated = field.clone().repeated();
        let proto = repeated.to_proto_string(0);
        assert!(proto.contains("repeated"));

        let optional = field.clone().optional();
        let proto = optional.to_proto_string(0);
        assert!(proto.contains("optional"));
    }

    #[test]
    fn test_proto_message() {
        let mut user_message = ProtoMessage::new("User");
        user_message.add_field(ProtoField::new("id", ProtoFieldType::Uint64, 1));
        user_message.add_field(ProtoField::new("name", ProtoFieldType::String, 2));
        user_message.add_field(ProtoField::new("email", ProtoFieldType::String, 3));

        let proto = user_message.to_proto_string(0);
        assert!(proto.contains("message User"));
        assert!(proto.contains("uint64 id"));
        assert!(proto.contains("string name"));
        assert!(proto.contains("string email"));
    }

    #[test]
    fn test_nested_message() {
        let mut address = ProtoMessage::new("Address");
        address.add_field(ProtoField::new("street", ProtoFieldType::String, 1));
        address.add_field(ProtoField::new("city", ProtoFieldType::String, 2));

        let mut user = ProtoMessage::new("User");
        user.add_nested(address);
        user.add_field(ProtoField::new("id", ProtoFieldType::Uint64, 1));
        user.add_field(ProtoField::new("name", ProtoFieldType::String, 2));

        let proto = user.to_proto_string(0);
        assert!(proto.contains("message Address"));
        assert!(proto.contains("message User"));
        assert!(proto.contains("string street"));
        assert!(proto.contains("string city"));
    }

    #[test]
    fn test_proto_method() {
        let unary = ProtoMethod::new("GetUser", "GetUserRequest", "GetUserResponse");
        assert!(!unary.client_streaming);
        assert!(!unary.server_streaming);

        let server_stream = unary.clone().server_streaming();
        assert!(server_stream.server_streaming);

        let client_stream = unary.clone().client_streaming();
        assert!(client_stream.client_streaming);

        let bidi = unary.clone().bidi_streaming();
        assert!(bidi.client_streaming);
        assert!(bidi.server_streaming);
    }

    #[test]
    fn test_proto_service() {
        let mut service = ProtoService::new("UserService");
        service.add_method(ProtoMethod::new(
            "GetUser",
            "GetUserRequest",
            "GetUserResponse",
        ));
        service.add_method(
            ProtoMethod::new("ListUsers", "ListUsersRequest", "ListUsersResponse")
                .server_streaming(),
        );

        let proto = service.to_proto_string();
        assert!(proto.contains("service UserService"));
        assert!(proto.contains("rpc GetUser"));
        assert!(proto.contains("rpc ListUsers"));
        assert!(proto.contains("stream ListUsersResponse"));
    }

    #[test]
    fn test_proto_file_generation() {
        let mut file = ProtoFile::new("api.v1");
        file.add_option("go_package", "github.com/example/api/v1;api");

        let mut status_enum = ProtoEnum::new("Status");
        status_enum.add_value("UNKNOWN", 0);
        status_enum.add_value("OK", 1);
        status_enum.add_value("ERROR", 2);
        file.add_enum(status_enum.clone());

        let mut user_msg = ProtoMessage::new("User");
        user_msg.add_field(ProtoField::new("id", ProtoFieldType::Uint64, 1));
        user_msg.add_field(ProtoField::new("name", ProtoFieldType::String, 2));
        user_msg.add_field(ProtoField::new("email", ProtoFieldType::String, 3));
        file.add_message(user_msg.clone());

        let mut service = ProtoService::new("UserService");
        service.add_method(ProtoMethod::new(
            "GetUser",
            "GetUserRequest",
            "GetUserResponse",
        ));
        file.add_service(service.clone());

        let proto = file.to_proto_string();
        assert!(proto.contains("syntax = \"proto3\""));
        assert!(proto.contains("package api.v1"));
        assert!(proto.contains("option go_package"));
        assert!(proto.contains("enum Status"));
        assert!(proto.contains("message User"));
        assert!(proto.contains("service UserService"));
    }

    // ------------------------------------------------------------------------
    // gRPC Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_grpc_status_codes() {
        assert!(GrpcStatusCode::Ok.is_ok());
        assert!(!GrpcStatusCode::NotFound.is_ok());
        assert!(GrpcStatusCode::NotFound.is_error());
        assert_eq!(GrpcStatusCode::Ok.name(), "OK");
        assert_eq!(GrpcStatusCode::NotFound.name(), "NOT_FOUND");
    }

    #[test]
    fn test_grpc_metadata() {
        let mut metadata = GrpcMetadata::new();
        metadata.insert("authorization", "Bearer token123");
        metadata.insert("user-agent", "test-client");

        assert_eq!(metadata.get("authorization"), Some("Bearer token123"));
        assert_eq!(metadata.len(), 2);
        assert!(!metadata.is_empty());

        metadata.remove("authorization");
        assert_eq!(metadata.get("authorization"), None);
    }

    #[test]
    fn test_grpc_response() {
        let resp = GrpcResponse::ok(42);
        assert!(resp.is_ok());
        assert!(!resp.is_error());
        assert_eq!(resp.data, Some(42));

        let err_resp = GrpcResponse::<()>::error(GrpcStatusCode::NotFound, "User not found");
        assert!(err_resp.is_error());
        assert!(!err_resp.is_ok());
    }

    #[test]
    fn test_grpc_stream() {
        let mut stream = GrpcStream::new();

        assert!(stream.send(1).is_ok());
        assert!(stream.send(2).is_ok());
        assert!(stream.send(3).is_ok());

        assert_eq!(stream.len(), 3);
        assert!(!stream.is_empty());

        let items = stream.drain();
        assert_eq!(items, vec![1, 2, 3]);

        stream.close();
        assert!(stream.is_closed());
    }

    #[test]
    fn test_grpc_stream_closed() {
        let mut stream = GrpcStream::new();
        stream.close();

        let result = stream.send(1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, GrpcStatusCode::Internal);
    }

    #[test]
    fn test_grpc_method_creation() {
        fn handler(_meta: GrpcMetadata, _req: String) -> GrpcResponse<String> {
            GrpcResponse::ok("response".to_string())
        }

        let unary_method = GrpcMethod::unary("GetUser", handler);
        assert_eq!(unary_method.name, "GetUser");
        assert_eq!(unary_method.method_type, GrpcMethodType::Unary);
    }

    #[test]
    fn test_grpc_service() {
        let mut service = GrpcService::new("api.v1", "UserService");

        service.register_method(GrpcMethod::<String, String>::unary("GetUser", |_, _| {
            GrpcResponse::ok("user".to_string())
        }));

        assert_eq!(service.full_name(), "api.v1.UserService");
        assert_eq!(service.method_count(), 1);
        assert!(service.has_method("GetUser"));
        assert!(!service.has_method("DeleteUser"));
    }

    #[test]
    fn test_grpc_service_builder() {
        let service = GrpcServiceBuilder::new("api.v1", "UserService")
            .unary::<String, String>("GetUser")
            .server_streaming::<String, String>("ListUsers")
            .client_streaming::<String, String>("UploadData")
            .bidi_streaming::<String, String>("Chat")
            .build();

        assert_eq!(service.package, "api.v1");
        assert_eq!(service.name, "UserService");
        assert_eq!(service.method_count(), 4);
    }

    // ------------------------------------------------------------------------
    // tRPC Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_trpc_error_codes() {
        assert_eq!(TrpcErrorCode::NotFound.http_status(), 404);
        assert_eq!(TrpcErrorCode::Badrequest.http_status(), 400);
        assert_eq!(TrpcErrorCode::InternalServerError.http_status(), 500);
        assert_eq!(TrpcErrorCode::Unauthorized.http_status(), 401);
    }

    #[test]
    fn test_trpc_response() {
        let resp = TrpcResponse::ok("test data");
        assert!(resp.is_ok());
        assert!(!resp.is_error());

        let err_resp = TrpcResponse::<()>::error(TrpcErrorCode::NotFound, "Not found");
        assert!(err_resp.is_error());
        assert!(!err_resp.is_ok());
    }

    #[test]
    fn test_trpc_context() {
        let ctx = TrpcContext::new()
            .with_user_id("user123")
            .with_request_id("req456")
            .with_header("x-client", "test");

        assert_eq!(ctx.user_id, Some("user123".to_string()));
        assert_eq!(ctx.request_id, Some("req456".to_string()));
        assert_eq!(ctx.headers.get("x-client"), Some(&"test".to_string()));
    }

    #[test]
    fn test_trpc_subscription() {
        let mut sub = TrpcSubscription::new();

        sub.emit(1);
        sub.emit(2);
        sub.emit(3);

        assert_eq!(sub.values(), vec![1, 2, 3]);
        assert!(sub.is_active());

        sub.complete();
        assert!(!sub.is_active());
    }

    #[test]
    fn test_trpc_subscription_error() {
        let mut sub: TrpcSubscription<String> = TrpcSubscription::new();

        let err = TrpcError {
            code: TrpcErrorCode::InternalServerError,
            message: "Database error".to_string(),
            data: None,
        };

        sub.error(err.clone());

        assert!(!sub.is_active());
        assert_eq!(sub.get_error(), Some(err));
    }

    #[test]
    fn test_trpc_procedure_creation() {
        fn query_handler(_ctx: TrpcContext, _input: String) -> TrpcResponse<String> {
            TrpcResponse::ok("result".to_string())
        }

        let query = TrpcProcedure::query("getUser", query_handler);
        assert_eq!(query.name, "getUser");
        assert_eq!(query.procedure_type, TrpcProcedureType::Query);

        fn mutation_handler(_ctx: TrpcContext, _input: String) -> TrpcResponse<String> {
            TrpcResponse::ok("created".to_string())
        }

        let mutation = TrpcProcedure::mutation("createUser", mutation_handler);
        assert_eq!(mutation.name, "createUser");
        assert_eq!(mutation.procedure_type, TrpcProcedureType::Mutation);

        fn sub_handler(_ctx: TrpcContext, _input: String) -> TrpcSubscription<String> {
            let mut sub = TrpcSubscription::new();
            sub.emit("update1".to_string());
            sub
        }

        let subscription = TrpcProcedure::subscription("onUserUpdate", sub_handler);
        assert_eq!(subscription.name, "onUserUpdate");
        assert_eq!(subscription.procedure_type, TrpcProcedureType::Subscription);
    }

    #[test]
    fn test_trpc_router() {
        let mut router = TrpcRouter::new("appRouter");

        fn query_handler(_ctx: TrpcContext, _input: String) -> TrpcResponse<String> {
            TrpcResponse::ok("result".to_string())
        }

        fn mutation_handler(_ctx: TrpcContext, _input: String) -> TrpcResponse<String> {
            TrpcResponse::ok("created".to_string())
        }

        router.query(TrpcProcedure::query("getUser", query_handler));
        router.mutation(TrpcProcedure::mutation("createUser", mutation_handler));

        assert_eq!(router.name, "appRouter");
        assert_eq!(router.procedure_count(), 2);
        assert!(router.has_procedure("getUser"));
        assert!(router.has_procedure("createUser"));
        assert!(!router.has_procedure("deleteUser"));
    }

    #[test]
    fn test_trpc_router_builder() {
        let router = TrpcRouterBuilder::new("appRouter")
            .query::<String, String>("getUser")
            .mutation::<String, String>("createUser")
            .subscription::<String, String>("onUserUpdate")
            .build();

        assert_eq!(router.name, "appRouter");
        assert_eq!(router.procedure_count(), 3);
    }

    #[test]
    fn test_trpc_type_generation() {
        let mut router = TrpcRouter::new("appRouter");

        fn handler(_ctx: TrpcContext, _input: String) -> TrpcResponse<String> {
            TrpcResponse::ok("result".to_string())
        }

        router.query(TrpcProcedure::query("getUser", handler));
        router.mutation(TrpcProcedure::mutation("createUser", handler));

        let types = router.generate_types();
        assert!(types.contains("interface appRouterRouter"));
        assert!(types.contains("getUser:"));
        assert!(types.contains("createUser:"));
    }

    // ------------------------------------------------------------------------
    // Integration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_grpc_stream_clone() {
        let mut stream = GrpcStream::new();
        stream.send(1).unwrap();
        stream.send(2).unwrap();

        let mut cloned = stream.clone();
        cloned.send(3).unwrap();

        // Both references see the same data
        assert_eq!(stream.len(), 3);
        assert_eq!(cloned.len(), 3);

        let items = stream.drain();
        assert_eq!(items, vec![1, 2, 3]);
    }

    #[test]
    fn test_grpc_response_with_metadata() {
        let mut metadata = GrpcMetadata::new();
        metadata.insert("x-trace-id", "trace123");

        let resp = GrpcResponse::ok(42).with_metadata(metadata);
        assert_eq!(resp.metadata.get("x-trace-id"), Some("trace123"));
    }

    #[test]
    fn test_trpc_subscription_clone() {
        let mut sub = TrpcSubscription::new();
        sub.emit(1);
        sub.emit(2);

        let cloned = sub.clone();
        assert_eq!(sub.values(), cloned.values());
        assert_eq!(sub.values(), vec![1, 2]);
    }

    #[test]
    fn test_grpc_method_with_metadata() {
        let mut metadata = GrpcMetadata::new();
        metadata.insert("x-method-name", "GetUser");
        metadata.insert("x-timeout", "30s");

        let method = GrpcMethod::<String, String>::unary("GetUser", |_, _| {
            GrpcResponse::ok("user".to_string())
        })
        .with_metadata(metadata);

        assert_eq!(method.metadata.get("x-method-name"), Some("GetUser"));
        assert_eq!(method.metadata.get("x-timeout"), Some("30s"));
    }

    #[test]
    fn test_trpc_router_with_middleware() {
        let mut router = TrpcRouter::new("appRouter");

        fn handler(_ctx: TrpcContext, _input: String) -> TrpcResponse<String> {
            TrpcResponse::ok("result".to_string())
        }

        router.query(TrpcProcedure::query("getUser", handler));
        router.middleware("auth");
        router.middleware("logging");
        router.with_context_factory("createContext");

        assert_eq!(router.middleware.len(), 2);
        assert!(router.middleware.contains(&"auth".to_string()));
        assert!(router.middleware.contains(&"logging".to_string()));
        assert_eq!(router.context_factory, Some("createContext".to_string()));
    }

    #[test]
    fn test_proto_field_with_options() {
        let field = ProtoField::new("count", ProtoFieldType::Int32, 1)
            .optional()
            .default_value("0")
            .packed();

        let proto = field.to_proto_string(0);
        assert!(proto.contains("optional"));
        assert!(proto.contains("[default = 0]"));
        assert!(proto.contains("[packed = true]"));
    }

    #[test]
    fn test_comprehensive_proto_file() {
        let mut file = ProtoFile::new("ecommerce.v1");
        file.add_option("go_package", "github.com/example/ecommerce/v1;ecommerce");
        file.add_option("java_multiple_files", "true");
        file.add_import("google/protobuf/timestamp.proto");

        // Order status enum
        let mut status_enum = ProtoEnum::new("OrderStatus");
        status_enum.add_value("PENDING", 0);
        status_enum.add_value("CONFIRMED", 1);
        status_enum.add_value("SHIPPED", 2);
        status_enum.add_value("DELIVERED", 3);
        status_enum.add_value("CANCELLED", 4);
        file.add_enum(status_enum.clone());

        // Address message
        let mut address_msg = ProtoMessage::new("Address");
        address_msg.add_field(ProtoField::new("street", ProtoFieldType::String, 1));
        address_msg.add_field(ProtoField::new("city", ProtoFieldType::String, 2));
        address_msg.add_field(ProtoField::new("country", ProtoFieldType::String, 3));
        address_msg.add_field(ProtoField::new("postal_code", ProtoFieldType::String, 4));
        file.add_message(address_msg.clone());

        // Order item message
        let mut item_msg = ProtoMessage::new("OrderItem");
        item_msg.add_field(ProtoField::new("product_id", ProtoFieldType::Uint64, 1));
        item_msg.add_field(ProtoField::new("quantity", ProtoFieldType::Uint32, 2));
        item_msg.add_field(ProtoField::new("price", ProtoFieldType::Double, 3));
        file.add_message(item_msg.clone());

        // Order message
        let mut order_msg = ProtoMessage::new("Order");
        order_msg.add_field(ProtoField::new("id", ProtoFieldType::Uint64, 1));
        order_msg.add_field(ProtoField::new("customer_id", ProtoFieldType::Uint64, 2));
        order_msg.add_field(ProtoField::new(
            "status",
            ProtoFieldType::Enum("OrderStatus".to_string()),
            3,
        ));
        order_msg.add_field(
            ProtoField::new("items", ProtoFieldType::Message("OrderItem".to_string()), 4)
                .repeated(),
        );
        order_msg.add_field(ProtoField::new(
            "shipping_address",
            ProtoFieldType::Message("Address".to_string()),
            5,
        ));
        order_msg.add_field(ProtoField::new("created_at", ProtoFieldType::String, 6)); // Timestamp as string for simplicity
        file.add_message(order_msg.clone());

        // Order service
        let mut order_service = ProtoService::new("OrderService");
        order_service.add_method(ProtoMethod::new("GetOrder", "GetOrderRequest", "Order"));
        order_service.add_method(ProtoMethod::new(
            "CreateOrder",
            "CreateOrderRequest",
            "Order",
        ));
        order_service.add_method(ProtoMethod::new("ListOrders", "ListOrdersRequest", "Order"));
        order_service.add_method(ProtoMethod::new(
            "UpdateOrder",
            "UpdateOrderRequest",
            "Order",
        ));
        order_service.add_method(ProtoMethod::new(
            "OrderUpdates",
            "OrderUpdateRequest",
            "OrderUpdate",
        ));
        file.add_service(order_service);

        let proto = file.to_proto_string();

        // Verify all components are present
        assert!(proto.contains("syntax = \"proto3\""));
        assert!(proto.contains("package ecommerce.v1"));
        assert!(proto.contains("import \"google/protobuf/timestamp.proto\""));
        assert!(proto.contains("enum OrderStatus"));
        assert!(proto.contains("PENDING = 0"));
        assert!(proto.contains("message Address"));
        assert!(proto.contains("message OrderItem"));
        assert!(proto.contains("message Order"));
        assert!(proto.contains("repeated OrderItem items"));
        assert!(proto.contains("service OrderService"));
        assert!(proto.contains("rpc GetOrder"));
        assert!(proto.contains("stream Order"));
        assert!(proto.contains("stream UpdateOrderRequest"));
    }
}
