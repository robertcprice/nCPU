//! HTTP Client for nCPU/nSynth
//!
//! HTTP client implementation using the FFI layer from Phase 12.

use crate::http::types::{HeaderMap, Method, Request, Response, StatusCode};
use crate::runtime::syscall::{self, open_flags, seek, sock_type, socket};
use crate::runtime::{Errno, FfiResult, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// HTTP Client
#[derive(Debug)]
pub struct Client {
    /// Base URL for requests
    base_url: String,
    /// Default headers
    default_headers: HeaderMap,
    /// Request timeout
    timeout: Duration,
    /// Connection pool (simplified - single connection for now)
    conn_fd: Arc<Mutex<Option<i32>>>,
}

impl Client {
    /// Create new client
    pub fn new() -> Self {
        Self {
            base_url: String::new(),
            default_headers: HeaderMap::new(),
            timeout: Duration::from_secs(30),
            conn_fd: Arc::new(Mutex::new(None)),
        }
    }

    /// Create with base URL
    pub fn with_base_url(url: impl Into<String>) -> Self {
        Self {
            base_url: url.into(),
            default_headers: HeaderMap::new(),
            timeout: Duration::from_secs(30),
            conn_fd: Arc::new(Mutex::new(None)),
        }
    }

    /// Set default header
    pub fn with_default_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.default_headers.insert(name, value);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Make a GET request
    pub fn get(&self, path: &str) -> Result<Response, String> {
        let url = if self.base_url.is_empty() {
            path.to_string()
        } else {
            format!("{}{}", self.base_url.trim_end_matches('/'), path)
        };
        let req = self.build_request(Method::GET, &url, None);
        self.send_request(req)
    }

    /// Make a POST request
    pub fn post(&self, path: &str, body: impl Into<Vec<u8>>) -> Result<Response, String> {
        let url = if self.base_url.is_empty() {
            path.to_string()
        } else {
            format!("{}{}", self.base_url.trim_end_matches('/'), path)
        };
        let req = self.build_request(Method::POST, &url, Some(body.into()));
        self.send_request(req)
    }

    /// Make a PUT request
    pub fn put(&self, path: &str, body: impl Into<Vec<u8>>) -> Result<Response, String> {
        let url = if self.base_url.is_empty() {
            path.to_string()
        } else {
            format!("{}{}", self.base_url.trim_end_matches('/'), path)
        };
        let req = self.build_request(Method::PUT, &url, Some(body.into()));
        self.send_request(req)
    }

    /// Make a DELETE request
    pub fn delete(&self, path: &str) -> Result<Response, String> {
        let url = if self.base_url.is_empty() {
            path.to_string()
        } else {
            format!("{}{}", self.base_url.trim_end_matches('/'), path)
        };
        let req = self.build_request(Method::DELETE, &url, None);
        self.send_request(req)
    }

    /// Build request with default headers
    fn build_request(&self, method: Method, url: &str, body: Option<Vec<u8>>) -> Request {
        let (host, path) = self.parse_url(url);

        let mut req = Request::new(method, path);

        // Add host header
        if let Some(host) = host {
            req = req.with_header("Host", host);
        }

        // Add default headers
        for (name, value) in self.default_headers.iter() {
            req = req.with_header(name, value);
        }

        // Add body if present
        if let Some(body) = body {
            req = req.with_header("Content-Length", body.len().to_string());
            req = req.with_body(body);
        }

        // Add connection header
        if !self.default_headers.contains("connection") {
            req = req.with_header("Connection", "close");
        }

        req
    }

    /// Parse URL into (host, path)
    fn parse_url(&self, url: &str) -> (Option<String>, String) {
        // Simple parser for http://host/path URLs
        if url.starts_with("http://") {
            let rest = &url[7..];
            if let Some(idx) = rest.find('/') {
                let host = rest[..idx].to_string();
                let path = rest[idx..].to_string();
                (Some(host), path)
            } else {
                (Some(rest.to_string()), "/".to_string())
            }
        } else if url.starts_with("https://") {
            let rest = &url[8..];
            if let Some(idx) = rest.find('/') {
                let host = rest[..idx].to_string();
                let path = rest[idx..].to_string();
                (Some(host), path)
            } else {
                (Some(rest.to_string()), "/".to_string())
            }
        } else {
            (None, url.to_string())
        }
    }

    /// Send request and read response
    fn send_request(&self, req: Request) -> Result<Response, String> {
        // For now, this is a stub - full implementation would connect via TCP
        // using the syscalls from Phase 12
        Err("HTTP client not fully implemented - requires TCP connection".to_string())
    }

    /// Get base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get timeout
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience trait for making HTTP requests
pub trait HttpClient {
    fn get(&self, path: &str) -> Result<Response, String>;
    fn post(&self, path: &str, body: impl Into<Vec<u8>>) -> Result<Response, String>;
}

impl HttpClient for Client {
    fn get(&self, path: &str) -> Result<Response, String> {
        self.get(path)
    }

    fn post(&self, path: &str, body: impl Into<Vec<u8>>) -> Result<Response, String> {
        self.post(path, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = Client::new();
        assert_eq!(client.base_url(), "");
        assert_eq!(client.timeout(), Duration::from_secs(30));
    }

    #[test]
    fn test_client_with_base_url() {
        let client = Client::with_base_url("http://example.com/api");
        assert_eq!(client.base_url(), "http://example.com/api");
    }

    #[test]
    fn test_client_with_headers() {
        let client = Client::new()
            .with_default_header("User-Agent", "TestClient")
            .with_default_header("Accept", "application/json");

        assert!(client.default_headers.contains("user-agent"));
        assert!(client.default_headers.contains("accept"));
    }

    #[test]
    fn test_parse_url() {
        let client = Client::new();

        let (host, path) = client.parse_url("http://example.com/api/users");
        assert_eq!(host, Some("example.com".to_string()));
        assert_eq!(path, "/api/users");

        let (host, path) = client.parse_url("http://example.com");
        assert_eq!(host, Some("example.com".to_string()));
        assert_eq!(path, "/");

        let (host, path) = client.parse_url("/api/users");
        assert_eq!(host, None);
        assert_eq!(path, "/api/users");
    }

    #[test]
    fn test_build_request() {
        let client = Client::with_base_url("http://example.com")
            .with_default_header("User-Agent", "TestClient");

        let req = client.build_request(Method::GET, "http://example.com/api/users", None);

        assert_eq!(req.method, Method::GET);
        assert_eq!(req.path, "/api/users");
        assert_eq!(req.headers.get("host"), Some("example.com"));
        assert_eq!(req.headers.get("user-agent"), Some("TestClient"));
    }

    #[test]
    fn test_build_request_with_body() {
        let client = Client::new();

        let req = client.build_request(Method::POST, "/api/users", Some(b"test body".to_vec()));

        assert_eq!(req.method, Method::POST);
        assert_eq!(req.body, b"test body");
        assert_eq!(req.headers.get("content-length"), Some("9"));
    }

    #[test]
    fn test_get_post_methods() {
        let client = Client::with_base_url("http://example.com/api");

        // These should return errors since client is stubbed
        let result = client.get("/users");
        assert!(result.is_err());

        let result = client.post("/users", b"data");
        assert!(result.is_err());
    }
}
