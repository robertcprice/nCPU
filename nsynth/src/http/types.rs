//! HTTP Types for nCPU/nSynth
//!
//! Core HTTP data structures: Request, Response, Method, StatusCode

use std::collections::HashMap;
use std::fmt;

/// HTTP Method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
    CONNECT,
    TRACE,
}

impl Method {
    /// Parse method from string
    pub fn from_str(s: &str) -> Option<Method> {
        match s.to_uppercase().as_str() {
            "GET" => Some(Method::GET),
            "POST" => Some(Method::POST),
            "PUT" => Some(Method::PUT),
            "DELETE" => Some(Method::DELETE),
            "PATCH" => Some(Method::PATCH),
            "HEAD" => Some(Method::HEAD),
            "OPTIONS" => Some(Method::OPTIONS),
            "CONNECT" => Some(Method::CONNECT),
            "TRACE" => Some(Method::TRACE),
            _ => None,
        }
    }

    /// Convert method to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Method::GET => "GET",
            Method::POST => "POST",
            Method::PUT => "PUT",
            Method::DELETE => "DELETE",
            Method::PATCH => "PATCH",
            Method::HEAD => "HEAD",
            Method::OPTIONS => "OPTIONS",
            Method::CONNECT => "CONNECT",
            Method::TRACE => "TRACE",
        }
    }

    /// Check if method is safe (read-only)
    pub fn is_safe(&self) -> bool {
        matches!(
            self,
            Method::GET | Method::HEAD | Method::OPTIONS | Method::TRACE
        )
    }

    /// Check if method allows body
    pub fn allows_body(&self) -> bool {
        matches!(
            self,
            Method::POST | Method::PUT | Method::PATCH | Method::DELETE
        )
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// HTTP Status Code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusCode {
    Continue = 100,
    SwitchingProtocols = 101,
    OK = 200,
    Created = 201,
    Accepted = 202,
    NoContent = 204,
    MovedPermanently = 301,
    Found = 302,
    NotModified = 304,
    BadRequest = 400,
    Unauthorized = 401,
    Forbidden = 403,
    NotFound = 404,
    MethodNotAllowed = 405,
    RequestTimeout = 408,
    PayloadTooLarge = 413,
    TooManyRequests = 429,
    InternalServerError = 500,
    NotImplemented = 501,
    BadGateway = 502,
    ServiceUnavailable = 503,
    GatewayTimeout = 504,
}

impl StatusCode {
    /// Get status code as number
    pub fn as_u16(&self) -> u16 {
        *self as u16
    }

    /// Get status reason phrase
    pub fn reason_phrase(&self) -> &'static str {
        match self {
            StatusCode::Continue => "Continue",
            StatusCode::SwitchingProtocols => "Switching Protocols",
            StatusCode::OK => "OK",
            StatusCode::Created => "Created",
            StatusCode::Accepted => "Accepted",
            StatusCode::NoContent => "No Content",
            StatusCode::MovedPermanently => "Moved Permanently",
            StatusCode::Found => "Found",
            StatusCode::NotModified => "Not Modified",
            StatusCode::BadRequest => "Bad Request",
            StatusCode::Unauthorized => "Unauthorized",
            StatusCode::Forbidden => "Forbidden",
            StatusCode::NotFound => "Not Found",
            StatusCode::MethodNotAllowed => "Method Not Allowed",
            StatusCode::RequestTimeout => "Request Timeout",
            StatusCode::PayloadTooLarge => "Payload Too Large",
            StatusCode::TooManyRequests => "Too Many Requests",
            StatusCode::InternalServerError => "Internal Server Error",
            StatusCode::NotImplemented => "Not Implemented",
            StatusCode::BadGateway => "Bad Gateway",
            StatusCode::ServiceUnavailable => "Service Unavailable",
            StatusCode::GatewayTimeout => "Gateway Timeout",
        }
    }

    /// Check if status is informational (1xx)
    pub fn is_informational(&self) -> bool {
        self.as_u16() >= 100 && self.as_u16() < 200
    }

    /// Check if status is success (2xx)
    pub fn is_success(&self) -> bool {
        self.as_u16() >= 200 && self.as_u16() < 300
    }

    /// Check if status is redirect (3xx)
    pub fn is_redirect(&self) -> bool {
        self.as_u16() >= 300 && self.as_u16() < 400
    }

    /// Check if status is client error (4xx)
    pub fn is_client_error(&self) -> bool {
        self.as_u16() >= 400 && self.as_u16() < 500
    }

    /// Check if status is server error (5xx)
    pub fn is_server_error(&self) -> bool {
        self.as_u16() >= 500 && self.as_u16() < 600
    }

    /// Check if status is error (4xx or 5xx)
    pub fn is_error(&self) -> bool {
        self.is_client_error() || self.is_server_error()
    }
}

impl fmt::Display for StatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.as_u16(), self.reason_phrase())
    }
}

/// HTTP Header Map
#[derive(Debug, Clone, Default)]
pub struct HeaderMap {
    inner: HashMap<String, String>,
}

impl HeaderMap {
    /// Create new empty header map
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Insert header (normalizes name to Title-Case)
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let name = Self::normalize_name(name.into());
        self.inner.insert(name, value.into());
    }

    /// Get header value (case-insensitive)
    pub fn get(&self, name: &str) -> Option<&str> {
        self.inner
            .get(&Self::normalize_name(name.to_string()))
            .map(|s| s.as_str())
    }

    /// Remove header
    pub fn remove(&mut self, name: &str) -> Option<String> {
        self.inner.remove(&Self::normalize_name(name.to_string()))
    }

    /// Check if header exists
    pub fn contains(&self, name: &str) -> bool {
        self.inner
            .contains_key(&Self::normalize_name(name.to_string()))
    }

    /// Get all headers as iterator
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Get header count
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Normalize header name to Title-Case
    fn normalize_name(name: String) -> String {
        let mut result = String::new();
        let mut capitalize_next = true;
        for c in name.chars() {
            if c == '-' {
                result.push(c);
                capitalize_next = true;
            } else if capitalize_next {
                result.extend(c.to_uppercase());
                capitalize_next = false;
            } else {
                result.extend(c.to_lowercase());
            }
        }
        result
    }

    /// Parse from raw header lines
    pub fn parse_from_lines(lines: &[&str]) -> Self {
        let mut headers = Self::new();
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim(), value.trim());
            }
        }
        headers
    }

    /// Convert to HTTP header string
    pub fn to_http_string(&self) -> String {
        self.iter()
            .map(|(name, value)| format!("{}: {}\r\n", name, value))
            .collect()
    }
}

/// HTTP Request
#[derive(Debug, Clone)]
pub struct Request {
    pub method: Method,
    pub path: String,
    pub query: Option<String>,
    pub version: String,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl Request {
    /// Create new request
    pub fn new(method: Method, path: impl Into<String>) -> Self {
        Self {
            method,
            path: path.into(),
            query: None,
            version: "HTTP/1.1".to_string(),
            headers: HeaderMap::new(),
            body: Vec::new(),
        }
    }

    /// Set query string
    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    /// Set header
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name, value);
        self
    }

    /// Set body
    pub fn with_body(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = body.into();
        self
    }

    /// Get content-type header
    pub fn content_type(&self) -> Option<&str> {
        self.headers.get("content-type")
    }

    /// Get content-length header
    pub fn content_length(&self) -> Option<usize> {
        self.headers.get("content-length")?.parse().ok()
    }

    /// Get user-agent header
    pub fn user_agent(&self) -> Option<&str> {
        self.headers.get("user-agent")
    }

    /// Parse from raw HTTP bytes
    pub fn parse_from_bytes(data: &[u8]) -> Result<Self, String> {
        let text =
            String::from_utf8(data.to_vec()).map_err(|_| "Invalid UTF-8 in request".to_string())?;

        let mut lines = text.lines();
        let request_line = lines.next().ok_or("Empty request")?;

        // Parse request line: METHOD /path HTTP/1.1
        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err("Invalid request line".to_string());
        }

        let method = Method::from_str(parts[0]).ok_or(format!("Unknown method: {}", parts[0]))?;

        let full_path = parts[1];
        let (path, query) = if let Some(idx) = full_path.find('?') {
            (
                full_path[..idx].to_string(),
                Some(full_path[idx + 1..].to_string()),
            )
        } else {
            (full_path.to_string(), None)
        };

        let version = if parts.len() > 2 {
            parts[2].to_string()
        } else {
            "HTTP/1.1".to_string()
        };

        // Parse headers
        let mut headers = HeaderMap::new();
        let mut header_lines = Vec::new();
        for line in lines {
            if line.is_empty() {
                break;
            }
            header_lines.push(line);
        }
        headers = HeaderMap::parse_from_lines(&header_lines);

        // Get body
        let body_start = text
            .find("\r\n\r\n")
            .ok_or("No body separator".to_string())?
            + 4;

        let body = data.get(body_start..).unwrap_or(&[]).to_vec();

        Ok(Self {
            method,
            path,
            query,
            version,
            headers,
            body,
        })
    }

    /// Convert to HTTP bytes
    pub fn to_http_bytes(&self) -> Vec<u8> {
        let mut result = String::new();

        // Request line
        result.push_str(&format!("{} {}", self.method, self.path));
        if let Some(query) = &self.query {
            result.push_str("?query=");
            result.push_str(query);
        }
        result.push_str(&format!(" {}\r\n", self.version));

        // Headers
        result.push_str(&self.headers.to_http_string());

        // Body separator
        result.push_str("\r\n");

        let mut bytes = result.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}

/// HTTP Response
#[derive(Debug, Clone)]
pub struct Response {
    pub version: String,
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl Response {
    /// Create new response
    pub fn new(status: StatusCode) -> Self {
        Self {
            version: "HTTP/1.1".to_string(),
            status,
            headers: HeaderMap::new(),
            body: Vec::new(),
        }
    }

    /// Set header
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name, value);
        self
    }

    /// Set body
    pub fn with_body(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = body.into();
        self.headers
            .insert("content-length", self.body.len().to_string());
        self
    }

    /// Set content-type
    pub fn with_content_type(mut self, ct: impl Into<String>) -> Self {
        self.headers.insert("content-type", ct.into());
        self
    }

    /// Get content-type header
    pub fn content_type(&self) -> Option<&str> {
        self.headers.get("content-type")
    }

    /// Create OK response with body
    pub fn ok(body: impl Into<Vec<u8>>) -> Self {
        Self::new(StatusCode::OK).with_body(body)
    }

    /// Create 201 Created response
    pub fn created(body: impl Into<Vec<u8>>) -> Self {
        Self::new(StatusCode::Created).with_body(body)
    }

    /// Create 404 response
    pub fn not_found() -> Self {
        Self::new(StatusCode::NotFound).with_body(b"404 Not Found")
    }

    /// Create 500 error response
    pub fn internal_error() -> Self {
        Self::new(StatusCode::InternalServerError).with_body(b"500 Internal Server Error")
    }

    /// Convert to HTTP bytes
    pub fn to_http_bytes(&self) -> Vec<u8> {
        let mut result = format!("{} {}\r\n", self.version, self.status);

        // Headers
        result.push_str(&self.headers.to_http_string());

        // Body separator
        result.push_str("\r\n");

        let mut bytes = result.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }

    /// Parse from raw HTTP bytes
    pub fn parse_from_bytes(data: &[u8]) -> Result<Self, String> {
        let text = String::from_utf8(data.to_vec())
            .map_err(|_| "Invalid UTF-8 in response".to_string())?;

        let mut lines = text.lines();
        let status_line = lines.next().ok_or("Empty response")?;

        // Parse status line: HTTP/1.1 200 OK
        let parts: Vec<&str> = status_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err("Invalid status line".to_string());
        }

        let version = parts[0].to_string();
        let status_code: u16 = parts[1]
            .parse()
            .map_err(|_| "Invalid status code".to_string())?;

        let status = match status_code {
            100 => StatusCode::Continue,
            101 => StatusCode::SwitchingProtocols,
            200 => StatusCode::OK,
            201 => StatusCode::Created,
            202 => StatusCode::Accepted,
            204 => StatusCode::NoContent,
            301 => StatusCode::MovedPermanently,
            302 => StatusCode::Found,
            304 => StatusCode::NotModified,
            400 => StatusCode::BadRequest,
            401 => StatusCode::Unauthorized,
            403 => StatusCode::Forbidden,
            404 => StatusCode::NotFound,
            405 => StatusCode::MethodNotAllowed,
            408 => StatusCode::RequestTimeout,
            413 => StatusCode::PayloadTooLarge,
            429 => StatusCode::TooManyRequests,
            500 => StatusCode::InternalServerError,
            501 => StatusCode::NotImplemented,
            502 => StatusCode::BadGateway,
            503 => StatusCode::ServiceUnavailable,
            504 => StatusCode::GatewayTimeout,
            _ => return Err(format!("Unknown status code: {}", status_code)),
        };

        // Parse headers
        let mut headers = HeaderMap::new();
        let mut header_lines = Vec::new();
        for line in lines {
            if line.is_empty() {
                break;
            }
            header_lines.push(line);
        }
        headers = HeaderMap::parse_from_lines(&header_lines);

        // Get body
        let body_start = text
            .find("\r\n\r\n")
            .ok_or("No body separator".to_string())?
            + 4;

        let body = data.get(body_start..).unwrap_or(&[]).to_vec();

        Ok(Self {
            version,
            status,
            headers,
            body,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_from_str() {
        assert_eq!(Method::from_str("GET"), Some(Method::GET));
        assert_eq!(Method::from_str("post"), Some(Method::POST));
        assert_eq!(Method::from_str("INVALID"), None);
    }

    #[test]
    fn test_method_properties() {
        assert!(Method::GET.is_safe());
        assert!(!Method::POST.is_safe());
        assert!(Method::POST.allows_body());
        assert!(!Method::GET.allows_body());
    }

    #[test]
    fn test_status_code_properties() {
        assert!(StatusCode::OK.is_success());
        assert!(StatusCode::NotFound.is_client_error());
        assert!(StatusCode::InternalServerError.is_server_error());
        assert!(StatusCode::MovedPermanently.is_redirect());
        assert!(StatusCode::Continue.is_informational());
    }

    #[test]
    fn test_header_map() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json");
        headers.insert("Content-Length", "123");

        assert_eq!(headers.get("Content-Type"), Some("application/json"));
        assert_eq!(headers.get("content-length"), Some("123"));
        assert!(headers.contains("CONTENT-TYPE"));
    }

    #[test]
    fn test_request_creation() {
        let req = Request::new(Method::POST, "/api/users")
            .with_header("content-type", "application/json")
            .with_body(r#"{"name":"Alice"}"#);

        assert_eq!(req.method, Method::POST);
        assert_eq!(req.path, "/api/users");
        assert_eq!(req.content_type(), Some("application/json"));
    }

    #[test]
    fn test_response_creation() {
        let resp = Response::ok(b"Hello, World!").with_content_type("text/plain");

        assert_eq!(resp.status, StatusCode::OK);
        assert_eq!(resp.body, b"Hello, World!");
        assert_eq!(resp.content_type(), Some("text/plain"));
    }

    #[test]
    fn test_response_helpers() {
        assert_eq!(Response::not_found().status, StatusCode::NotFound);
        assert_eq!(
            Response::internal_error().status,
            StatusCode::InternalServerError
        );
        assert_eq!(Response::created(b"test").status, StatusCode::Created);
    }

    #[test]
    fn test_request_to_bytes() {
        let req = Request::new(Method::GET, "/test").with_header("User-Agent", "Test");

        let bytes = req.to_http_bytes();
        let text = String::from_utf8(bytes).unwrap();

        assert!(text.contains("GET /test"));
        assert!(text.contains("User-Agent: Test"));
    }

    #[test]
    fn test_response_to_bytes() {
        let resp = Response::ok(b"test body").with_content_type("text/plain");

        let bytes = resp.to_http_bytes();
        let text = String::from_utf8(bytes).unwrap();

        assert!(text.contains("200 OK"));
        assert!(text.contains("test body"));
    }

    #[test]
    fn test_parse_request() {
        let raw = b"GET /api/users?page=1 HTTP/1.1\r\n\
            Host: example.com\r\n\
            User-Agent: Test\r\n\
            \r\n\
            body content";

        let req = Request::parse_from_bytes(raw).unwrap();
        assert_eq!(req.method, Method::GET);
        assert_eq!(req.path, "/api/users");
        assert_eq!(req.query, Some("page=1".to_string()));
        assert_eq!(req.headers.get("host"), Some("example.com"));
        assert_eq!(req.body, b"body content");
    }

    #[test]
    fn test_parse_response() {
        let raw = b"HTTP/1.1 404 Not Found\r\n\
            Content-Type: text/plain\r\n\
            Content-Length: 13\r\n\
            \r\n\
            404 Not Found";

        let resp = Response::parse_from_bytes(raw).unwrap();
        assert_eq!(resp.status, StatusCode::NotFound);
        assert_eq!(resp.headers.get("content-type"), Some("text/plain"));
        assert_eq!(resp.body, b"404 Not Found");
    }
}
