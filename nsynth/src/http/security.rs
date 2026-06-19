//! Cybersecurity/Security Layer for nCPU/nSynth
//!
//! JWT, OAuth2, CSRF, XSS protection, secure headers, rate limiting.

use serde_json::Value;
use std::collections::HashMap;

/// JWT token
#[derive(Debug, Clone)]
pub struct JwtToken {
    /// Header (algorithm, type)
    pub header: HashMap<String, String>,
    /// Payload (claims)
    pub payload: HashMap<String, Value>,
    /// Signature
    pub signature: String,
}

impl JwtToken {
    /// Create new JWT
    pub fn new(payload: HashMap<String, Value>) -> Self {
        let mut header = HashMap::new();
        header.insert("alg".to_string(), "HS256".to_string());
        header.insert("typ".to_string(), "JWT".to_string());

        Self {
            header,
            payload,
            signature: String::new(),
        }
    }

    /// Encode to JWT string
    pub fn encode(&self, secret: &str) -> String {
        // In real implementation: base64url encode header + payload, sign with HMAC
        let header_b64 = base64url_encode(serde_json::to_string(&self.header).unwrap());
        let payload_b64 = base64url_encode(serde_json::to_string(&self.payload).unwrap());
        let signing_input = format!("{}.{}", header_b64, payload_b64);
        let signature = hmac_sha256(secret, &signing_input);
        format!("{}.{}", signing_input, signature)
    }

    /// Decode from JWT string
    pub fn decode(token: &str, secret: &str) -> Result<Self, String> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err("Invalid token format".to_string());
        }

        // Verify signature
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let expected_signature = hmac_sha256(secret, &signing_input);
        if parts[2] != expected_signature {
            return Err("Invalid signature".to_string());
        }

        let header: HashMap<String, String> = serde_json::from_str(&base64url_decode(parts[0]))
            .map_err(|_| "Invalid header".to_string())?;

        let payload: HashMap<String, Value> = serde_json::from_str(&base64url_decode(parts[1]))
            .map_err(|_| "Invalid payload".to_string())?;

        Ok(Self {
            header,
            payload,
            signature: parts[2].to_string(),
        })
    }

    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.payload.get("exp").and_then(|v| v.as_i64()) {
            return exp
                < (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64);
        }
        false
    }
}

/// OAuth2 flow types
#[derive(Debug, Clone)]
pub enum OAuth2Flow {
    /// Authorization code flow
    AuthorizationCode {
        client_id: String,
        redirect_uri: String,
        scope: Vec<String>,
    },
    /// Client credentials flow
    ClientCredentials {
        client_id: String,
        client_secret: String,
        scope: Vec<String>,
    },
    /// Implicit flow (deprecated)
    Implicit {
        client_id: String,
        redirect_uri: String,
    },
    /// Resource owner password flow
    PasswordCredentials {
        client_id: String,
        username: String,
        password: String,
    },
}

impl OAuth2Flow {
    /// Generate authorization URL
    pub fn auth_url(&self) -> String {
        match self {
            OAuth2Flow::AuthorizationCode {
                client_id,
                redirect_uri,
                scope,
            } => {
                format!(
                    "https://auth.provider.com/authorize?client_id={}&redirect_uri={}&response_type=code&scope={}",
                    client_id,
                    redirect_uri,
                    scope.join(" ")
                )
            }
            _ => String::new(),
        }
    }
}

/// CSRF token
#[derive(Debug, Clone)]
pub struct CsrfToken {
    pub token: String,
    pub expires_at: i64,
}

impl CsrfToken {
    /// Generate new CSRF token
    pub fn new() -> Self {
        Self {
            token: generate_random_token(32),
            expires_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
                + 3600,
        }
    }

    /// Verify CSRF token
    pub fn verify(&self, token: &str) -> bool {
        !self.is_expired() && self.token == token
    }

    fn is_expired(&self) -> bool {
        self.expires_at
            < (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64)
    }
}

/// Security headers
#[derive(Debug, Clone)]
pub struct SecurityHeaders {
    /// Content-Security-Policy
    pub csp: Option<String>,
    /// X-Frame-Options
    pub frame_options: Option<String>,
    /// X-Content-Type-Options
    pub content_type_options: Option<String>,
    /// Strict-Transport-Security
    pub hsts: Option<String>,
    /// X-XSS-Protection
    pub xss_protection: Option<String>,
    /// Referrer-Policy
    pub referrer_policy: Option<String>,
    /// Permissions-Policy
    pub permissions_policy: Option<String>,
}

impl SecurityHeaders {
    pub fn new() -> Self {
        Self {
            csp: None,
            frame_options: Some("DENY".to_string()),
            content_type_options: Some("nosniff".to_string()),
            hsts: Some("max-age=31536000; includeSubDomains".to_string()),
            xss_protection: Some("1; mode=block".to_string()),
            referrer_policy: Some("strict-origin-when-cross-origin".to_string()),
            permissions_policy: None,
        }
    }

    /// With CSP
    pub fn with_csp(mut self, csp: impl Into<String>) -> Self {
        self.csp = Some(csp.into());
        self
    }

    /// Convert to header map
    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(ref csp) = self.csp {
            headers.insert("Content-Security-Policy".to_string(), csp.clone());
        }
        if let Some(ref frame) = self.frame_options {
            headers.insert("X-Frame-Options".to_string(), frame.clone());
        }
        if let Some(ref ct) = self.content_type_options {
            headers.insert("X-Content-Type-Options".to_string(), ct.clone());
        }
        if let Some(ref hsts) = self.hsts {
            headers.insert("Strict-Transport-Security".to_string(), hsts.clone());
        }
        if let Some(ref xss) = self.xss_protection {
            headers.insert("X-XSS-Protection".to_string(), xss.clone());
        }
        if let Some(ref referrer) = self.referrer_policy {
            headers.insert("Referrer-Policy".to_string(), referrer.clone());
        }
        if let Some(ref pp) = self.permissions_policy {
            headers.insert("Permissions-Policy".to_string(), pp.clone());
        }

        headers
    }
}

impl Default for SecurityHeaders {
    fn default() -> Self {
        Self::new()
    }
}

/// Rate limiter with token bucket
#[derive(Debug, Clone)]
pub struct RateLimiter {
    /// Tokens per second
    pub rate: f64,
    /// Burst size
    pub burst: usize,
    /// Current tokens
    pub tokens: f64,
    /// Last update
    pub last_update: i64,
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(rate: f64, burst: usize) -> Self {
        Self {
            rate,
            burst,
            tokens: burst as f64,
            last_update: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        }
    }

    /// Check if request is allowed
    pub fn check(&mut self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let elapsed = (now - self.last_update) as f64;
        self.tokens = (self.tokens + elapsed * self.rate).min(self.burst as f64);
        self.last_update = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Input sanitizer
pub struct Sanitizer;

impl Sanitizer {
    /// Sanitize HTML (remove tags and tag content)
    pub fn html(input: &str) -> String {
        let mut result = String::new();
        let mut chars = input.chars().peekable();
        let mut in_element = false; // Inside an HTML element (between opening and closing tags)

        while let Some(c) = chars.next() {
            if c == '<' {
                // Check if this is a closing tag
                let mut temp_chars = chars.clone().peekable();
                let is_closing = if temp_chars.peek() == Some(&'/') {
                    // Skip the '/'
                    temp_chars.next();
                    true
                } else {
                    false
                };

                // Skip entire tag (including the '>')
                while let Some(next) = chars.next() {
                    if next == '>' {
                        break;
                    }
                }

                // Update state
                if is_closing {
                    in_element = false;
                } else {
                    in_element = true;
                }
            } else if !in_element {
                result.push(c);
            }
            // Characters inside elements are skipped
        }
        result
    }

    /// Sanitize SQL (escape quotes)
    pub fn sql(input: &str) -> String {
        input.replace("'", "''").replace("\\", "\\\\")
    }

    /// Sanitize for URL
    pub fn url(input: &str) -> String {
        percent_encode(input.as_bytes())
    }

    /// XSS escape
    pub fn xss(input: &str) -> String {
        input
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
            .replace("'", "&#x27;")
    }
}

/// API key authentication
#[derive(Debug, Clone)]
pub struct ApiKey {
    pub key: String,
    pub scopes: Vec<String>,
}

impl ApiKey {
    /// Verify API key
    pub fn verify(&self, provided_key: &str) -> bool {
        self.key == provided_key
    }

    /// Check if has scope
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.contains(&scope.to_string())
    }
}

/// HMAC signature
#[derive(Debug, Clone)]
pub struct HmacSignature {
    pub algorithm: String,
    pub signature: String,
}

impl HmacSignature {
    /// Sign data with HMAC
    pub fn sign(data: &str, secret: &str) -> String {
        hmac_sha256(secret, data)
    }

    /// Verify signature
    pub fn verify(data: &str, secret: &str, signature: &str) -> bool {
        Self::sign(data, secret) == signature
    }
}

/// IP whitelist/blacklist
#[derive(Debug, Clone)]
pub struct IpFilter {
    pub whitelist: Vec<String>,
    pub blacklist: Vec<String>,
}

impl IpFilter {
    pub fn new() -> Self {
        Self {
            whitelist: Vec::new(),
            blacklist: Vec::new(),
        }
    }

    /// Add to whitelist
    pub fn allow(mut self, ip: impl Into<String>) -> Self {
        self.whitelist.push(ip.into());
        self
    }

    /// Add to blacklist
    pub fn block(mut self, ip: impl Into<String>) -> Self {
        self.blacklist.push(ip.into());
        self
    }

    /// Check if IP is allowed
    pub fn is_allowed(&self, ip: &str) -> bool {
        // Check blacklist first
        for blocked in &self.blacklist {
            if self.ip_matches(ip, blocked) {
                return false;
            }
        }

        // If whitelist is empty, allow all
        if self.whitelist.is_empty() {
            return true;
        }

        // Check whitelist
        for allowed in &self.whitelist {
            if self.ip_matches(ip, allowed) {
                return true;
            }
        }

        false
    }

    fn ip_matches(&self, ip: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('.').collect();
            let ip_parts: Vec<&str> = ip.split('.').collect();

            for (p, ip_part) in parts.iter().zip(ip_parts.iter()) {
                if *p != "*" && *p != *ip_part {
                    return false;
                }
            }
            return true;
        }
        ip == pattern
    }
}

impl Default for IpFilter {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions (simplified - would use real crypto in production)

fn base64url_encode(input: String) -> String {
    base64_simd::STANDARD
        .encode_to_string(input.as_bytes())
        .replace('+', "-")
        .replace('/', "_")
        .trim_end_matches('=')
        .to_string()
}

fn base64url_decode(input: &str) -> String {
    let mut padded = input.replace('-', "+").replace('_', "/");
    while padded.len() % 4 != 0 {
        padded.push('=');
    }
    String::from_utf8(
        base64_simd::STANDARD
            .decode_to_vec(padded.as_bytes())
            .unwrap(),
    )
    .unwrap()
}

fn hmac_sha256(secret: &str, data: &str) -> String {
    // Simplified - would use real HMAC-SHA256 in production
    format!("{:x}", md5_compute(format!("{}{}", secret, data)))
}

fn md5_compute(input: String) -> u128 {
    // Placeholder for actual hash computation
    let mut hash: u128 = 0;
    for b in input.as_bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(*b as u128);
    }
    hash
}

fn generate_random_token(len: usize) -> String {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut result = String::new();
    let mut rng = seed;

    for _ in 0..len {
        result.push(charset.chars().nth((rng % 62) as usize).unwrap());
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
    }
    result
}

fn percent_encode(bytes: &[u8]) -> String {
    let mut result = String::new();
    for &b in bytes {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(b as char);
            }
            _ => {
                result.push_str(&format!("%{:02X}", b));
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jwt_encode_decode() {
        let mut payload = HashMap::new();
        payload.insert("sub".to_string(), Value::from("1234567890"));
        payload.insert("name".to_string(), Value::from("John Doe"));

        let token = JwtToken::new(payload.clone());
        let encoded = token.encode("secret");

        let decoded = JwtToken::decode(&encoded, "secret").unwrap();
        assert_eq!(
            decoded.payload.get("name").unwrap(),
            &Value::from("John Doe")
        );
    }

    #[test]
    fn test_csrf_token() {
        let token = CsrfToken::new();
        assert!(token.verify(&token.token));
        assert!(!token.verify("wrong"));
    }

    #[test]
    fn test_security_headers() {
        let headers = SecurityHeaders::new()
            .with_csp("default-src 'self'".to_string())
            .to_headers();

        assert!(headers.contains_key("Content-Security-Policy"));
        assert!(headers.contains_key("X-Frame-Options"));
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(10.0, 5);

        // First 5 requests should pass
        for _ in 0..5 {
            assert!(limiter.check());
        }

        // 6th should fail (burst exceeded)
        assert!(!limiter.check());
    }

    #[test]
    fn test_sanitizer_html() {
        let input = "<script>alert('xss')</script>Hello";
        let output = Sanitizer::html(input);
        assert_eq!(output, "Hello");
    }

    #[test]
    fn test_sanitizer_xss() {
        let input = "<script>alert('xss')</script>";
        let output = Sanitizer::xss(input);
        assert!(output.contains("&lt;"));
        assert!(!output.contains("<script>"));
    }

    #[test]
    fn test_ip_filter() {
        let filter = IpFilter::new().allow("192.168.1.*").block("10.0.0.1");

        assert!(filter.is_allowed("192.168.1.100"));
        assert!(!filter.is_allowed("10.0.0.1"));
        assert!(!filter.is_allowed("8.8.8.8")); // Not in whitelist
    }
}
