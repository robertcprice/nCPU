//! HTTP Server for nCPU/nSynth
//!
//! HTTP server implementation using the FFI layer from Phase 12.

use crate::http::types::{Method, Request, Response, StatusCode};
use crate::runtime::syscall::{self, sock_type, socket};
use crate::runtime::{Errno, FfiResult, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Route definition
#[derive(Debug, Clone)]
pub struct Route {
    pub method: Method,
    pub path: String,
    pub handler: Handler,
}

/// Handler function type
pub type Handler = fn(Request) -> Response;

/// HTTP Server
#[derive(Debug)]
pub struct Server {
    routes: Vec<Route>,
    addr: String,
    port: u16,
    /// Socket file descriptor (when running)
    socket_fd: Option<i32>,
    /// Running state
    running: Arc<Mutex<bool>>,
}

impl Server {
    /// Create new server
    pub fn new(addr: impl Into<String>, port: u16) -> Self {
        Self {
            routes: Vec::new(),
            addr: addr.into(),
            port,
            socket_fd: None,
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Add a route
    pub fn route(
        &mut self,
        method: Method,
        path: impl Into<String>,
        handler: Handler,
    ) -> &mut Self {
        self.routes.push(Route {
            method,
            path: path.into(),
            handler,
        });
        self
    }

    /// Add GET route
    pub fn get(&mut self, path: impl Into<String>, handler: Handler) -> &mut Self {
        self.route(Method::GET, path, handler)
    }

    /// Add POST route
    pub fn post(&mut self, path: impl Into<String>, handler: Handler) -> &mut Self {
        self.route(Method::POST, path, handler)
    }

    /// Add PUT route
    pub fn put(&mut self, path: impl Into<String>, handler: Handler) -> &mut Self {
        self.route(Method::PUT, path, handler)
    }

    /// Add DELETE route
    pub fn delete(&mut self, path: impl Into<String>, handler: Handler) -> &mut Self {
        self.route(Method::DELETE, path, handler)
    }

    /// Find matching route
    fn find_route(&self, method: Method, path: &str) -> Option<&Route> {
        self.routes
            .iter()
            .find(|r| r.method == method && self.path_matches(&r.path, path))
    }

    /// Check if path pattern matches (simple exact match for now)
    fn path_matches(&self, pattern: &str, path: &str) -> bool {
        // Simple exact match
        if pattern == path {
            return true;
        }

        // Handle :param patterns
        if pattern.contains(':') {
            let pattern_parts: Vec<&str> = pattern.split('/').collect();
            let path_parts: Vec<&str> = path.split('/').collect();

            if pattern_parts.len() != path_parts.len() {
                return false;
            }

            for (p, actual) in pattern_parts.iter().zip(path_parts.iter()) {
                if p.starts_with(':') {
                    // Parameter, matches anything
                    continue;
                }
                if p != actual {
                    return false;
                }
            }
            return true;
        }

        false
    }

    /// Extract path parameters
    fn extract_params(&self, pattern: &str, path: &str) -> HashMap<String, String> {
        let mut params = HashMap::new();

        let pattern_parts: Vec<&str> = pattern.split('/').collect();
        let path_parts: Vec<&str> = path.split('/').collect();

        for (p, actual) in pattern_parts.iter().zip(path_parts.iter()) {
            if p.starts_with(':') {
                let name = p[1..].to_string();
                params.insert(name, actual.to_string());
            }
        }

        params
    }

    /// Start the server (requires unsafe operations)
    ///
    /// # Safety
    /// This function performs socket operations and is marked unsafe
    pub unsafe fn listen(&mut self) -> FfiResult<()> {
        // Create socket
        let fd_value = syscall::sys_socket(socket::AF_INET, sock_type::SOCK_STREAM, 0)?;
        let fd = if let Value::Int(n) = fd_value {
            n as i32
        } else {
            return Err(Errno::IOError("socket returned invalid value".to_string()));
        };

        // Parse address
        let addr_bytes: Vec<u8> = self
            .addr
            .parse::<std::net::IpAddr>()
            .map_err(|_| Errno::InvalidArgument("Invalid address".to_string()))?
            .to_string()
            .split('.')
            .map(|p| p.parse::<u8>().unwrap_or(0))
            .collect();

        if addr_bytes.len() != 4 {
            return Err(Errno::InvalidArgument("Invalid IPv4 address".to_string()));
        }

        // Bind socket
        syscall::sys_bind(fd, &addr_bytes, self.port)?;

        // Listen
        syscall::sys_listen(fd, 16)?;

        self.socket_fd = Some(fd);
        *self.running.lock().unwrap() = true;

        Ok(())
    }

    /// Accept one connection and handle it
    ///
    /// # Safety
    /// This function performs socket operations and is marked unsafe
    pub unsafe fn accept_one(&self) -> FfiResult<(Request, i32)> {
        let sockfd = self
            .socket_fd
            .ok_or_else(|| Errno::InvalidArgument("Server not listening".to_string()))?;

        let client_fd_value = syscall::sys_accept(sockfd)?;
        let client_fd = if let Value::Int(n) = client_fd_value {
            n as i32
        } else {
            return Err(Errno::IOError("accept returned invalid value".to_string()));
        };

        // Read request
        let mut buffer = vec![0u8; 4096];
        let bytes_read = libc::read(
            client_fd,
            buffer.as_mut_ptr() as *mut libc::c_void,
            buffer.len(),
        );

        if bytes_read < 0 {
            syscall::sys_close(client_fd)?;
            return Err(Errno::IOError("Failed to read request".to_string()));
        }

        buffer.truncate(bytes_read as usize);

        let request = Request::parse_from_bytes(&buffer)
            .map_err(|e| Errno::InvalidArgument(format!("Invalid request: {}", e)))?;

        Ok((request, client_fd))
    }

    /// Send response to client
    ///
    /// # Safety
    /// This function performs socket operations and is marked unsafe
    pub unsafe fn send_response(&self, client_fd: i32, response: &Response) -> FfiResult<()> {
        let bytes = response.to_http_bytes();
        let _written = syscall::sys_write(client_fd, &bytes)?;
        syscall::sys_close(client_fd)?;
        Ok(())
    }

    /// Handle a single request
    pub fn handle_request(&self, req: Request) -> Response {
        if let Some(route) = self.find_route(req.method, &req.path) {
            let _params = self.extract_params(&route.path, &req.path);
            // For now, ignore params - handler gets original request
            (route.handler)(req)
        } else {
            Response::not_found()
        }
    }

    /// Run server loop (blocking)
    ///
    /// # Safety
    /// This function performs socket operations
    pub unsafe fn run(&mut self) -> FfiResult<()> {
        self.listen()?;

        while *self.running.lock().unwrap() {
            match self.accept_one() {
                Ok((req, client_fd)) => {
                    let response = self.handle_request(req);
                    if let Err(e) = self.send_response(client_fd, &response) {
                        eprintln!("Error sending response: {}", e);
                    }
                }
                Err(e) => {
                    // Accept failed, maybe server is shutting down
                    if !*self.running.lock().unwrap() {
                        break;
                    }
                    eprintln!("Accept error: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Stop the server
    pub fn stop(&mut self) -> FfiResult<()> {
        *self.running.lock().unwrap() = false;

        if let Some(fd) = self.socket_fd {
            unsafe {
                syscall::sys_close(fd)?;
            }
            self.socket_fd = None;
        }

        Ok(())
    }

    /// Check if server is running
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }

    /// Get address
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Get port
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Get route count
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

/// Create a JSON response
pub fn json_response(data: &serde_json::Value) -> Response {
    Response::ok(data.to_string().as_bytes()).with_content_type("application/json")
}

/// Create a text response
pub fn text_response(text: &str) -> Response {
    Response::ok(text.as_bytes()).with_content_type("text/plain")
}

/// Create an HTML response
pub fn html_response(html: &str) -> Response {
    Response::ok(html.as_bytes()).with_content_type("text/html; charset=utf-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = Server::new("127.0.0.1", 8080);
        assert_eq!(server.addr(), "127.0.0.1");
        assert_eq!(server.port(), 8080);
        assert_eq!(server.route_count(), 0);
        assert!(!server.is_running());
    }

    #[test]
    fn test_route_addition() {
        let mut server = Server::new("127.0.0.1", 8080);
        server.get("/test", |_| Response::ok(b"test"));
        server.post("/users", |_| Response::created(b"created"));
        assert_eq!(server.route_count(), 2);
    }

    #[test]
    fn test_path_matching() {
        let server = Server::new("127.0.0.1", 8080);

        // Exact match
        assert!(server.path_matches("/users", "/users"));
        assert!(!server.path_matches("/users", "/posts"));

        // Parameter match
        assert!(server.path_matches("/users/:id", "/users/123"));
        assert!(server.path_matches("/posts/:id/comments/:cid", "/posts/45/comments/67"));
        assert!(!server.path_matches("/users/:id", "/posts/123"));
    }

    #[test]
    fn test_extract_params() {
        let server = Server::new("127.0.0.1", 8080);

        let params = server.extract_params("/users/:id", "/users/123");
        assert_eq!(params.get("id"), Some(&"123".to_string()));

        let params = server.extract_params("/posts/:pid/comments/:cid", "/posts/45/comments/67");
        assert_eq!(params.get("pid"), Some(&"45".to_string()));
        assert_eq!(params.get("cid"), Some(&"67".to_string()));
    }

    #[test]
    fn test_find_route() {
        let mut server = Server::new("127.0.0.1", 8080);
        server.get("/users", |_| Response::ok(b"users list"));
        server.get("/users/:id", |_| Response::ok(b"user detail"));

        assert!(server.find_route(Method::GET, "/users").is_some());
        assert!(server.find_route(Method::GET, "/users/123").is_some());
        assert!(server.find_route(Method::POST, "/users").is_none());
    }

    #[test]
    fn test_handle_request() {
        let mut server = Server::new("127.0.0.1", 8080);
        server.get("/test", |_| Response::ok(b"test response"));

        let req = Request::new(Method::GET, "/test");
        let resp = server.handle_request(req);

        assert_eq!(resp.status, StatusCode::OK);
        assert_eq!(resp.body, b"test response");
    }

    #[test]
    fn test_handle_request_not_found() {
        let server = Server::new("127.0.0.1", 8080);

        let req = Request::new(Method::GET, "/nonexistent");
        let resp = server.handle_request(req);

        assert_eq!(resp.status, StatusCode::NotFound);
    }

    #[test]
    fn test_response_helpers() {
        let data = serde_json::json!({"message": "hello"});
        let resp = json_response(&data);

        assert_eq!(resp.status, StatusCode::OK);
        assert_eq!(resp.content_type(), Some("application/json"));

        let resp = text_response("hello world");
        assert_eq!(resp.content_type(), Some("text/plain"));

        let resp = html_response("<h1>Hello</h1>");
        assert!(resp.content_type().unwrap().contains("text/html"));
    }
}
