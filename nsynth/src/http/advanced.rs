//! Advanced HTTP Features for nCPU/nSynth
//!
//! HTTP/2, proxy, compression, range requests, SSE, GraphQL, gRPC, WebDAV.

use std::collections::HashMap;

/// HTTP version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpVersion {
    Http1_0,
    Http1_1,
    Http2,
    Http3,
}

impl HttpVersion {
    pub fn as_str(&self) -> &str {
        match self {
            HttpVersion::Http1_0 => "HTTP/1.0",
            HttpVersion::Http1_1 => "HTTP/1.1",
            HttpVersion::Http2 => "HTTP/2",
            HttpVersion::Http3 => "HTTP/3",
        }
    }
}

/// HTTP/2 frame header
#[derive(Debug, Clone)]
pub struct Http2Frame {
    pub length: u32,
    pub frame_type: u8,
    pub flags: u8,
    pub stream_id: u32,
    pub payload: Vec<u8>,
}

impl Http2Frame {
    /// Create HTTP/2 DATA frame
    pub fn data(stream_id: u32, data: &[u8]) -> Self {
        Self {
            length: data.len() as u32,
            frame_type: 0x0,
            flags: 0x0,
            stream_id,
            payload: data.to_vec(),
        }
    }

    /// Create HTTP/2 HEADERS frame
    pub fn headers(stream_id: u32, headers: &HashMap<String, String>) -> Self {
        // Simplified - would use HPACK compression in production
        let mut payload = Vec::new();
        for (k, v) in headers {
            payload.extend_from_slice(k.as_bytes());
            payload.push(b':');
            payload.extend_from_slice(v.as_bytes());
            payload.push(b'\r');
            payload.push(b'\n');
        }

        Self {
            length: payload.len() as u32,
            frame_type: 0x1,
            flags: 0x4, // END_HEADERS
            stream_id,
            payload,
        }
    }

    /// Create HTTP/2 SETTINGS frame
    pub fn settings() -> Self {
        Self {
            length: 0,
            frame_type: 0x4,
            flags: 0x0,
            stream_id: 0,
            payload: Vec::new(),
        }
    }

    /// Encode to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Frame length (3 bytes)
        bytes.push((self.length >> 16) as u8);
        bytes.push((self.length >> 8) as u8);
        bytes.push(self.length as u8);

        // Frame type
        bytes.push(self.frame_type);

        // Flags
        bytes.push(self.flags);

        // Stream identifier (4 bytes)
        bytes.push((self.stream_id >> 24) as u8);
        bytes.push((self.stream_id >> 16) as u8);
        bytes.push((self.stream_id >> 8) as u8);
        bytes.push(self.stream_id as u8);

        // Payload
        bytes.extend_from_slice(&self.payload);

        bytes
    }
}

/// Proxy configuration
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub proxy_url: String,
    pub auth: Option<(String, String)>,
    pub whitelist: Vec<String>,
}

impl ProxyConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            proxy_url: url.into(),
            auth: None,
            whitelist: Vec::new(),
        }
    }

    pub fn with_auth(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.auth = Some((username.into(), password.into()));
        self
    }

    pub fn should_proxy(&self, url: &str) -> bool {
        if self.whitelist.is_empty() {
            return true;
        }

        for pattern in &self.whitelist {
            if url.contains(pattern) {
                return true;
            }
        }
        false
    }
}

/// Compression algorithm
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Compression {
    Gzip,
    Deflate,
    Brotli,
    None,
}

impl Compression {
    pub fn as_str(&self) -> &str {
        match self {
            Compression::Gzip => "gzip",
            Compression::Deflate => "deflate",
            Compression::Brotli => "br",
            Compression::None => "identity",
        }
    }

    pub fn from_header(value: &str) -> Self {
        if value.contains("gzip") {
            Compression::Gzip
        } else if value.contains("br") {
            Compression::Brotli
        } else if value.contains("deflate") {
            Compression::Deflate
        } else {
            Compression::None
        }
    }

    /// Compress data
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        match self {
            Compression::Gzip => self.gzip_compress(data),
            Compression::Deflate => self.deflate_compress(data),
            Compression::Brotli => self.brotli_compress(data),
            Compression::None => Ok(data.to_vec()),
        }
    }

    /// Decompress data
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        match self {
            Compression::Gzip => self.gzip_decompress(data),
            Compression::Deflate => self.deflate_decompress(data),
            Compression::Brotli => self.brotli_decompress(data),
            Compression::None => Ok(data.to_vec()),
        }
    }

    fn gzip_compress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // Placeholder - would use real compression library
        let mut result = Vec::new();
        result.extend_from_slice(b"\x1f\x8b"); // Gzip magic
        result.extend_from_slice(data);
        Ok(result)
    }

    fn gzip_decompress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // Placeholder
        if data.starts_with(b"\x1f\x8b") {
            Ok(data[2..].to_vec())
        } else {
            Err("Invalid gzip data".to_string())
        }
    }

    fn deflate_compress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // Placeholder
        Ok(data.to_vec())
    }

    fn deflate_decompress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // Placeholder
        Ok(data.to_vec())
    }

    fn brotli_compress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // Placeholder
        Ok(data.to_vec())
    }

    fn brotli_decompress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // Placeholder
        Ok(data.to_vec())
    }
}

/// Range request
#[derive(Debug, Clone)]
pub struct Range {
    pub start: u64,
    pub end: Option<u64>,
}

impl Range {
    /// Parse Range header
    pub fn parse(header: &str) -> Result<Vec<Range>, String> {
        if !header.starts_with("bytes=") {
            return Err("Invalid range header".to_string());
        }

        let ranges_str = &header[6..];
        let mut ranges = Vec::new();

        for part in ranges_str.split(',') {
            let part = part.trim();
            if let Some(idx) = part.find('-') {
                let start_str = &part[..idx];
                let end_str = &part[idx + 1..];

                let start = if start_str.is_empty() {
                    0
                } else {
                    start_str.parse().map_err(|_| "Invalid start")?
                };

                let end = if end_str.is_empty() {
                    None
                } else {
                    Some(end_str.parse().map_err(|_| "Invalid end")?)
                };

                ranges.push(Range { start, end });
            }
        }

        Ok(ranges)
    }

    /// Format as Range header value
    pub fn to_header(&self) -> String {
        if let Some(end) = self.end {
            format!("bytes={}-{}", self.start, end)
        } else {
            format!("bytes={}-", self.start)
        }
    }
}

/// Range response
#[derive(Debug, Clone)]
pub struct RangeResponse {
    pub ranges: Vec<Range>,
    pub total_length: u64,
    pub boundary: String,
}

impl RangeResponse {
    /// Create multipart/byteranges response
    pub fn to_multipart(&self, data: &[u8]) -> Vec<u8> {
        let mut result = Vec::new();

        for range in &self.ranges {
            result.extend_from_slice(format!(
                "--{}\r\nContent-Type: application/octet-stream\r\nContent-Range: bytes {}-{}/{}\r\n\r\n",
                self.boundary,
                range.start,
                range.end.unwrap_or(self.total_length - 1),
                self.total_length
            ).as_bytes());

            let end = range
                .end
                .unwrap_or(self.total_length)
                .min(data.len() as u64) as usize;
            let start = range.start as usize;
            result.extend_from_slice(&data[start..end.min(data.len())]);
            result.extend_from_slice(b"\r\n");
        }

        result.extend_from_slice(format!("--{}--\r\n", self.boundary).as_bytes());
        result
    }
}

/// Server-Sent Event
#[derive(Debug, Clone)]
pub struct ServerSentEvent {
    pub id: Option<String>,
    pub event: Option<String>,
    pub data: String,
    pub retry: Option<u32>,
}

impl ServerSentEvent {
    pub fn new(data: impl Into<String>) -> Self {
        Self {
            id: None,
            event: None,
            data: data.into(),
            retry: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_event(mut self, event: impl Into<String>) -> Self {
        self.event = Some(event.into());
        self
    }

    pub fn with_retry(mut self, retry: u32) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Format SSE message
    pub fn format(&self) -> String {
        let mut output = String::new();

        if let Some(ref id) = self.id {
            output.push_str(&format!("id: {}\n", id));
        }

        if let Some(ref event) = self.event {
            output.push_str(&format!("event: {}\n", event));
        }

        if let Some(retry) = self.retry {
            output.push_str(&format!("retry: {}\n", retry));
        }

        for line in self.data.lines() {
            output.push_str(&format!("data: {}\n", line));
        }

        output.push_str("\n");
        output
    }
}

/// GraphQL query
#[derive(Debug, Clone)]
pub struct GraphQLQuery {
    pub query: String,
    pub operation_name: Option<String>,
    pub variables: Option<serde_json::Value>,
}

impl GraphQLQuery {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            operation_name: None,
            variables: None,
        }
    }

    pub fn with_operation(mut self, name: impl Into<String>) -> Self {
        self.operation_name = Some(name.into());
        self
    }

    pub fn with_variables(mut self, variables: serde_json::Value) -> Self {
        self.variables = Some(variables);
        self
    }

    /// Format as JSON for request body
    pub fn to_json(&self) -> String {
        let mut obj = serde_json::Map::new();
        obj.insert(
            "query".to_string(),
            serde_json::Value::String(self.query.clone()),
        );

        if let Some(ref name) = self.operation_name {
            obj.insert(
                "operationName".to_string(),
                serde_json::Value::String(name.clone()),
            );
        }

        if let Some(ref vars) = self.variables {
            obj.insert("variables".to_string(), vars.clone());
        }

        serde_json::to_string(&obj).unwrap()
    }
}

/// GraphQL response
#[derive(Debug, Clone)]
pub struct GraphQLResponse {
    pub data: Option<serde_json::Value>,
    pub errors: Option<Vec<GraphQL>>,
}

#[derive(Debug, Clone)]
pub struct GraphQL {
    pub message: String,
    pub path: Option<Vec<serde_json::Value>>,
}

/// gRPC method definition
#[derive(Debug, Clone)]
pub struct GrpcMethod {
    pub service: String,
    pub method: String,
    pub input_type: String,
    pub output_type: String,
}

impl GrpcMethod {
    pub fn new(service: impl Into<String>, method: impl Into<String>) -> Self {
        Self {
            service: service.into(),
            method: method.into(),
            input_type: "Request".to_string(),
            output_type: "Response".to_string(),
        }
    }

    /// Generate gRPC proto definition
    pub fn to_proto(&self) -> String {
        format!(
            r#"syntax = "proto3";

service {} {{
  rpc {}({}) returns ({});
}}

message {} {{
  // fields
}}

message {} {{
  // fields
}}
"#,
            self.service,
            self.method,
            self.input_type,
            self.output_type,
            self.input_type,
            self.output_type
        )
    }
}

/// WebDAV method
#[derive(Debug, Clone)]
pub enum WebDavMethod {
    PropFind,
    PropPatch,
    MkCol,
    Copy,
    Move,
    Lock,
    Unlock,
}

impl WebDavMethod {
    pub fn as_str(&self) -> &str {
        match self {
            WebDavMethod::PropFind => "PROPFIND",
            WebDavMethod::PropPatch => "PROPPATCH",
            WebDavMethod::MkCol => "MKCOL",
            WebDavMethod::Copy => "COPY",
            WebDavMethod::Move => "MOVE",
            WebDavMethod::Lock => "LOCK",
            WebDavMethod::Unlock => "UNLOCK",
        }
    }
}

/// WebDAV property
#[derive(Debug, Clone)]
pub struct WebDavProperty {
    pub name: String,
    pub namespace: String,
    pub value: String,
}

/// Cache directive
#[derive(Debug, Clone)]
pub enum CacheDirective {
    NoCache,
    NoStore,
    MaxAge(u32),
    MustRevalidate,
    Public,
    Private,
}

impl CacheDirective {
    pub fn as_str(&self) -> String {
        match self {
            CacheDirective::NoCache => "no-cache".to_string(),
            CacheDirective::NoStore => "no-store".to_string(),
            CacheDirective::MaxAge(secs) => format!("max-age={}", secs),
            CacheDirective::MustRevalidate => "must-revalidate".to_string(),
            CacheDirective::Public => "public".to_string(),
            CacheDirective::Private => "private".to_string(),
        }
    }
}

/// ETag for cache validation
#[derive(Debug, Clone)]
pub struct ETag {
    pub value: String,
    pub weak: bool,
}

impl ETag {
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            weak: false,
        }
    }

    pub fn weak(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            weak: true,
        }
    }

    /// Generate ETag from content
    pub fn from_content(content: &[u8]) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        Self::new(format!(r#""{}""#, hasher.finish()))
    }

    pub fn to_header(&self) -> String {
        let prefix = if self.weak { "W/" } else { "" };
        format!("{}{}", prefix, self.value)
    }
}

/// API version
#[derive(Debug, Clone)]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ApiVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn as_str(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }

    pub fn from_header(value: &str) -> Result<Self, String> {
        let parts: Vec<&str> = value.split('.').collect();
        if parts.len() != 3 {
            return Err("Invalid version format".to_string());
        }

        Ok(Self {
            major: parts[0].parse().map_err(|_| "Invalid major")?,
            minor: parts[1].parse().map_err(|_| "Invalid minor")?,
            patch: parts[2].parse().map_err(|_| "Invalid patch")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_version() {
        assert_eq!(HttpVersion::Http2.as_str(), "HTTP/2");
    }

    #[test]
    fn test_http2_frame() {
        let data: &[u8] = b"hello";
        let frame = Http2Frame::data(1, data);
        let encoded = frame.encode();
        assert!(encoded.len() > 9); // Header + payload
    }

    #[test]
    fn test_range_parse() {
        let ranges = Range::parse("bytes=0-499").unwrap();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[0].end, Some(499));
    }

    #[test]
    fn test_compression_from_header() {
        assert_eq!(Compression::from_header("gzip, deflate"), Compression::Gzip);
        assert_eq!(Compression::from_header("br"), Compression::Brotli);
        assert_eq!(Compression::from_header("identity"), Compression::None);
    }

    #[test]
    fn test_sse_format() {
        let event = ServerSentEvent::new("Hello world")
            .with_id("1")
            .with_event("message");

        let formatted = event.format();
        assert!(formatted.contains("id: 1"));
        assert!(formatted.contains("event: message"));
        assert!(formatted.contains("data: Hello world"));
    }

    #[test]
    fn test_graphql_query() {
        let query = GraphQLQuery::new("query { user { name } }")
            .with_variables(serde_json::json!({"id": 123}));

        let json = query.to_json();
        assert!(json.contains("query"));
        assert!(json.contains("variables"));
    }

    #[test]
    fn test_grpc_method() {
        let method = GrpcMethod::new("UserService", "GetUser");
        let proto = method.to_proto();
        assert!(proto.contains("service UserService"));
        assert!(proto.contains("rpc GetUser"));
    }

    #[test]
    fn test_etag_from_content() {
        let etag = ETag::from_content(b"hello");
        assert!(etag.to_header().starts_with('"'));
    }

    #[test]
    fn test_webdav_methods() {
        assert_eq!(WebDavMethod::PropFind.as_str(), "PROPFIND");
        assert_eq!(WebDavMethod::Lock.as_str(), "LOCK");
    }
}
