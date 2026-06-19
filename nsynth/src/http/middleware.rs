//! Middleware Pipeline for nCPU/nSynth
//!
//! Request/response processing chain with interceptors.

use crate::http::types::{HeaderMap, Method, Request, Response, StatusCode};
use std::sync::{Arc, Mutex};

/// Middleware function type
pub type MiddlewareFn = Arc<dyn Fn(Request) -> Result<Response, MiddlewareError> + Send + Sync>;

/// Middleware error
#[derive(Debug, Clone)]
pub enum MiddlewareError {
    Halted(Response),
    Error(String),
}

impl From<Response> for MiddlewareError {
    fn from(resp: Response) -> Self {
        MiddlewareError::Halted(resp)
    }
}

/// Middleware chain
#[derive(Clone)]
pub struct MiddlewareChain {
    middleware: Vec<MiddlewareFn>,
    index: Arc<Mutex<usize>>,
}

impl MiddlewareChain {
    /// Create new chain
    pub fn new() -> Self {
        Self {
            middleware: Vec::new(),
            index: Arc::new(Mutex::new(0)),
        }
    }

    /// Add middleware to chain
    pub fn add(&mut self, middleware: MiddlewareFn) -> &mut Self {
        self.middleware.push(middleware);
        self
    }

    /// Process request through middleware chain
    pub fn process(&self, request: Request) -> Result<Response, MiddlewareError> {
        for middleware in &self.middleware {
            match middleware(request.clone()) {
                Ok(resp) => return Ok(resp),
                Err(MiddlewareError::Halted(resp)) => return Ok(resp),
                Err(MiddlewareError::Error(e)) if e == "Continue" => {
                    // Continue to next middleware with same request
                }
                Err(MiddlewareError::Error(e)) => return Err(MiddlewareError::Error(e)),
            };
        }
        // No middleware handled the request
        Err(MiddlewareError::Error("No handler".to_string()))
    }

    /// Get middleware count
    pub fn len(&self) -> usize {
        self.middleware.len()
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Common middleware builders
pub struct Middleware;

impl Middleware {
    /// CORS middleware
    pub fn cors(allow_origin: impl Into<String>) -> MiddlewareFn {
        let origin = allow_origin.into();
        Arc::new(
            move |mut req: Request| -> Result<Response, MiddlewareError> {
                // Handle preflight
                if req.method == Method::OPTIONS {
                    return Ok(Response::new(StatusCode::OK)
                        .with_header("Access-Control-Allow-Origin", &origin)
                        .with_header(
                            "Access-Control-Allow-Methods",
                            "GET, POST, PUT, DELETE, OPTIONS",
                        )
                        .with_header(
                            "Access-Control-Allow-Headers",
                            "Content-Type, Authorization",
                        )
                        .with_header("Access-Control-Max-Age", "86400")
                        .with_body(b""));
                }

                // Add CORS headers to request context
                req.headers.insert("x-cors-origin", &origin);
                Err(MiddlewareError::Error("Continue".to_string()))
            },
        )
    }

    /// Rate limiting middleware
    pub fn rate_limit(max_requests: u32, per_seconds: u64) -> MiddlewareFn {
        use std::collections::HashMap;
        use std::sync::Mutex as StdMutex;
        use std::time::{SystemTime, UNIX_EPOCH};

        #[derive(Debug, Clone, Default)]
        struct RateLimitState {
            requests: HashMap<String, Vec<u64>>,
        }

        let state = Arc::new(StdMutex::new(RateLimitState::default()));

        Arc::new(move |req: Request| -> Result<Response, MiddlewareError> {
            let client_id = req
                .headers
                .get("user-agent")
                .unwrap_or("unknown")
                .to_string();

            let mut state = state.lock().unwrap();
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let cutoff = now - per_seconds;

            // Clean old requests
            if let Some(requests) = state.requests.get_mut(&client_id) {
                requests.retain(|&t| t > cutoff);
            }

            // Check limit
            let count = state
                .requests
                .entry(client_id.clone())
                .or_insert_with(Vec::new)
                .len();
            if count >= max_requests as usize {
                return Ok(Response::new(StatusCode::TooManyRequests)
                    .with_header("Retry-After", per_seconds.to_string())
                    .with_body(b"Rate limit exceeded"));
            }

            // Add current request
            state
                .requests
                .entry(client_id)
                .or_insert_with(Vec::new)
                .push(now);

            Err(MiddlewareError::Error("Continue".to_string()))
        })
    }

    /// Logging middleware
    pub fn logger() -> MiddlewareFn {
        Arc::new(|req: Request| -> Result<Response, MiddlewareError> {
            eprintln!("[{}] {} {}", req.method, req.path, req.version);
            Err(MiddlewareError::Error("Continue".to_string()))
        })
    }

    /// Authentication middleware (Bearer token)
    pub fn auth_bearer(validator: Arc<dyn Fn(&str) -> bool + Send + Sync>) -> MiddlewareFn {
        Arc::new(move |req: Request| -> Result<Response, MiddlewareError> {
            let auth_header = req.headers.get("authorization").ok_or_else(|| {
                MiddlewareError::Halted(
                    Response::new(StatusCode::Unauthorized)
                        .with_header("WWW-Authenticate", "Bearer")
                        .with_body(b"Missing authorization header"),
                )
            })?;

            if !auth_header.starts_with("Bearer ") {
                return Ok(Response::new(StatusCode::Unauthorized)
                    .with_header("WWW-Authenticate", "Bearer")
                    .with_body(b"Invalid authorization type"));
            }

            let token = &auth_header[7..];
            if !validator(token) {
                return Ok(Response::new(StatusCode::Unauthorized).with_body(b"Invalid token"));
            }

            Err(MiddlewareError::Error("Continue".to_string()))
        })
    }

    /// Basic auth middleware
    pub fn auth_basic(credentials: (String, String)) -> MiddlewareFn {
        let (username, password) = credentials;
        Arc::new(move |req: Request| -> Result<Response, MiddlewareError> {
            let auth_header = req.headers.get("authorization").ok_or_else(|| {
                MiddlewareError::Halted(
                    Response::new(StatusCode::Unauthorized)
                        .with_header("WWW-Authenticate", "Basic realm=\"Restricted\"")
                        .with_body(b"Authorization required"),
                )
            })?;

            if !auth_header.starts_with("Basic ") {
                return Ok(Response::new(StatusCode::Unauthorized)
                    .with_header("WWW-Authenticate", "Basic realm=\"Restricted\"")
                    .with_body(b"Invalid authorization type"));
            }

            // Decode base64
            let encoded = &auth_header[6..];
            let decoded = match base64_simd::STANDARD.decode_to_vec(encoded.as_bytes()) {
                Ok(d) => String::from_utf8(d).unwrap_or_default(),
                Err(_) => {
                    return Ok(Response::new(StatusCode::Unauthorized)
                        .with_body(b"Invalid credentials encoding"));
                }
            };

            if let Some(creds) = decoded.split_once(':') {
                if creds.0 == username && creds.1 == password {
                    return Err(MiddlewareError::Error("Continue".to_string()));
                }
            }

            Ok(Response::new(StatusCode::Unauthorized)
                .with_header("WWW-Authenticate", "Basic realm=\"Restricted\"")
                .with_body(b"Invalid credentials"))
        })
    }

    /// Request size limit middleware
    pub fn max_body_size(max_bytes: usize) -> MiddlewareFn {
        Arc::new(move |req: Request| -> Result<Response, MiddlewareError> {
            if req.body.len() > max_bytes {
                return Ok(Response::new(StatusCode::PayloadTooLarge)
                    .with_header("Content-Length", "0")
                    .with_body(
                        format!("Request body too large (max {} bytes)", max_bytes).as_bytes(),
                    ));
            }

            Err(MiddlewareError::Error("Continue".to_string()))
        })
    }

    /// Timeout middleware
    pub fn timeout(seconds: u64) -> MiddlewareFn {
        Arc::new(move |_req: Request| -> Result<Response, MiddlewareError> {
            // Timeout would be handled by the executor
            // This just sets context
            Err(MiddlewareError::Error(format!("Timeout: {}s", seconds)))
        })
    }

    /// Compression middleware (response compression)
    pub fn compression() -> MiddlewareFn {
        Arc::new(|req: Request| -> Result<Response, MiddlewareError> {
            // Check Accept-Encoding header
            let accepts_gzip = req
                .headers
                .get("accept-encoding")
                .map(|h| h.contains("gzip"))
                .unwrap_or(false);

            if accepts_gzip {
                // Mark response for compression
                return Err(MiddlewareError::Error("Compress: gzip".to_string()));
            }

            Err(MiddlewareError::Error("Continue".to_string()))
        })
    }

    /// JSON content type enforcement
    pub fn require_json() -> MiddlewareFn {
        Arc::new(|req: Request| -> Result<Response, MiddlewareError> {
            let ct = req.content_type().unwrap_or("");

            if !ct.contains("application/json") {
                return Ok(Response::new(StatusCode::BadRequest)
                    .with_body(b"Content-Type must be application/json"));
            }

            Err(MiddlewareError::Error("Continue".to_string()))
        })
    }

    /// Security headers middleware
    pub fn security_headers() -> MiddlewareFn {
        Arc::new(|req: Request| -> Result<Response, MiddlewareError> {
            // Add security context
            Err(MiddlewareError::Error("SecurityHeaders".to_string()))
        })
    }
}

/// Response modifier middleware
pub struct ResponseModifier;

impl ResponseModifier {
    /// Add security headers to response
    pub fn add_security_headers(mut resp: Response) -> Response {
        resp = resp.with_header("X-Content-Type-Options", "nosniff");
        resp = resp.with_header("X-Frame-Options", "DENY");
        resp = resp.with_header("X-XSS-Protection", "1; mode=block");
        resp = resp.with_header(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        );
        resp
    }

    /// Add CORS headers to response
    pub fn add_cors_headers(mut resp: Response, origin: &str) -> Response {
        resp = resp.with_header("Access-Control-Allow-Origin", origin);
        resp = resp.with_header("Access-Control-Allow-Credentials", "true");
        resp
    }

    /// Compress response body (simplified - would use actual gzip)
    pub fn compress_body(resp: Response) -> Response {
        // In real implementation, would compress body
        resp.with_header("Content-Encoding", "gzip")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_middleware_chain() {
        let mut chain = MiddlewareChain::new();

        let middleware1: MiddlewareFn =
            Arc::new(|req| Err(MiddlewareError::Error("Continue".to_string())));

        let middleware2: MiddlewareFn = Arc::new(|_| Ok(Response::ok(b"handled")));

        chain.add(middleware1);
        chain.add(middleware2);

        let req = Request::new(Method::GET, "/test");
        let result = chain.process(req);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().body, b"handled");
    }

    #[test]
    fn test_cors_middleware() {
        let cors = Middleware::cors("*");

        let req = Request::new(Method::OPTIONS, "/api/data");
        let result = cors(req);

        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.status, StatusCode::OK);
        assert!(resp.headers.get("access-control-allow-origin").is_some());
    }

    #[test]
    fn test_rate_limit_middleware() {
        let rate_limit = Middleware::rate_limit(2, 60);

        // First request - should pass
        let req1 = Request::new(Method::GET, "/test").with_header("User-Agent", "test1");
        let _ = rate_limit(req1.clone());

        // Second request - should pass
        let _ = rate_limit(req1.clone());

        // Third request - should be rate limited
        let result = rate_limit(req1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, StatusCode::TooManyRequests);
    }

    #[test]
    fn test_require_json() {
        let require_json = Middleware::require_json();

        let req1 =
            Request::new(Method::POST, "/api/data").with_header("Content-Type", "application/json");
        let result1 = require_json(req1);
        assert!(matches!(result1, Err(MiddlewareError::Error(_))));

        let req2 =
            Request::new(Method::POST, "/api/data").with_header("Content-Type", "text/plain");
        let result2 = require_json(req2);
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap().status, StatusCode::BadRequest);
    }

    #[test]
    fn test_max_body_size() {
        let max_size = Middleware::max_body_size(100);

        let small_req = Request::new(Method::POST, "/upload").with_body(vec![0; 50]);
        let result1 = max_size(small_req);
        assert!(matches!(result1, Err(MiddlewareError::Error(_))));

        let large_req = Request::new(Method::POST, "/upload").with_body(vec![0; 200]);
        let result2 = max_size(large_req);
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap().status, StatusCode::PayloadTooLarge);
    }
}
