//! Cookie Management for nCPU/nSynth
//!
//! Cookie jar, session management, RFC 6265 compliance.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Cookie representation
#[derive(Debug, Clone)]
pub struct Cookie {
    pub name: String,
    pub value: String,
    pub domain: Option<String>,
    pub path: Option<String>,
    pub expires: Option<u64>, // Unix timestamp
    pub max_age: Option<u64>, // Seconds
    pub secure: bool,
    pub http_only: bool,
    pub same_site: SameSite,
}

/// SameSite attribute
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameSite {
    None,
    Lax,
    Strict,
}

impl Cookie {
    /// Create new cookie
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            domain: None,
            path: None,
            expires: None,
            max_age: None,
            secure: false,
            http_only: false,
            same_site: SameSite::None,
        }
    }

    /// Set domain
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Set path
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set expires (Unix timestamp)
    pub fn with_expires(mut self, expires: u64) -> Self {
        self.expires = Some(expires);
        self
    }

    /// Set max-age (seconds)
    pub fn with_max_age(mut self, max_age: u64) -> Self {
        self.max_age = Some(max_age);
        self
    }

    /// Set secure flag
    pub fn with_secure(mut self, secure: bool) -> Self {
        self.secure = secure;
        self
    }

    /// Set http-only flag
    pub fn with_http_only(mut self, http_only: bool) -> Self {
        self.http_only = http_only;
        self
    }

    /// Set same-site attribute
    pub fn with_same_site(mut self, same_site: SameSite) -> Self {
        self.same_site = same_site;
        self
    }

    /// Check if cookie is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires) = self.expires {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            return now > expires;
        }
        false
    }

    /// Check if cookie matches domain
    pub fn matches_domain(&self, domain: &str) -> bool {
        match &self.domain {
            None => true,
            Some(cookie_domain) => {
                // Strip leading dot for matching
                let effective_domain = cookie_domain.strip_prefix('.').unwrap_or(cookie_domain);

                // Exact match
                if domain == effective_domain || domain == cookie_domain {
                    return true;
                }
                // Subdomain match: domain should be subdomain.effective_domain
                if domain.ends_with(effective_domain) {
                    let prefix_len = domain.len() - effective_domain.len();
                    if prefix_len > 0 {
                        let prefix = &domain[..prefix_len];
                        return prefix.ends_with('.') || prefix.ends_with('.');
                    }
                }
                false
            }
        }
    }

    /// Check if cookie matches path
    pub fn matches_path(&self, path: &str) -> bool {
        match &self.path {
            None => path.starts_with('/'),
            Some(cookie_path) => path.starts_with(cookie_path),
        }
    }

    /// Serialize to Set-Cookie header value
    pub fn to_set_cookie_value(&self) -> String {
        let mut parts = vec![format!("{}={}", self.name, self.value)];

        if let Some(domain) = &self.domain {
            parts.push(format!("Domain={}", domain));
        }
        if let Some(path) = &self.path {
            parts.push(format!("Path={}", path));
        }
        if let Some(expires) = self.expires {
            // Convert to HTTP-date format
            parts.push(format!("Expires={}", Self::format_http_date(expires)));
        }
        if let Some(max_age) = self.max_age {
            parts.push(format!("Max-Age={}", max_age));
        }
        if self.secure {
            parts.push("Secure".to_string());
        }
        if self.http_only {
            parts.push("HttpOnly".to_string());
        }
        if self.same_site != SameSite::None {
            parts.push(format!("SameSite={}", self.same_site.as_str()));
        }

        parts.join("; ")
    }

    /// Parse from Cookie header value
    pub fn parse_from_cookie_value(
        cookie_str: &str,
    ) -> impl Iterator<Item = (String, String)> + use<'_> {
        cookie_str.split(';').filter_map(|pair| {
            let pair = pair.trim();
            pair.split_once('=')
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        })
    }

    /// Format Unix timestamp as HTTP date
    fn format_http_date(timestamp: u64) -> String {
        use std::time::UNIX_EPOCH;
        let datetime = UNIX_EPOCH + std::time::Duration::from_secs(timestamp);

        // Format as RFC 1123: "Wed, 21 Oct 2015 07:28:00 GMT"
        // This is simplified - real implementation would use chrono
        format!("{:?}", datetime)
    }
}

impl SameSite {
    fn as_str(&self) -> &'static str {
        match self {
            SameSite::None => "None",
            SameSite::Lax => "Lax",
            SameSite::Strict => "Strict",
        }
    }
}

/// Cookie jar for storing cookies per domain
#[derive(Debug, Clone, Default)]
pub struct CookieJar {
    cookies: HashMap<String, Vec<Cookie>>,
}

impl CookieJar {
    /// Create new cookie jar
    pub fn new() -> Self {
        Self::default()
    }

    /// Add cookie to jar
    pub fn add(&mut self, cookie: Cookie) {
        let domain = cookie
            .domain
            .clone()
            .unwrap_or_else(|| "default".to_string());
        self.cookies
            .entry(domain)
            .or_insert_with(Vec::new)
            .push(cookie);
    }

    /// Get cookies for a specific domain and path
    pub fn get(&self, domain: &str, path: &str) -> Vec<Cookie> {
        let mut result = Vec::new();

        for (cookie_domain, cookies) in &self.cookies {
            for cookie in cookies {
                if cookie.matches_domain(domain)
                    && cookie.matches_path(path)
                    && !cookie.is_expired()
                {
                    result.push(cookie.clone());
                }
            }
        }

        result
    }

    /// Get all cookie names for domain
    pub fn get_names(&self, domain: &str, path: &str) -> Vec<String> {
        self.get(domain, path).into_iter().map(|c| c.name).collect()
    }

    /// Get cookie value by name
    pub fn get_value(&self, domain: &str, path: &str, name: &str) -> Option<String> {
        self.get(domain, path)
            .into_iter()
            .find(|c| c.name == name)
            .map(|c| c.value.clone())
    }

    /// Remove cookie by name
    pub fn remove(&mut self, domain: &str, name: &str) {
        if let Some(cookies) = self.cookies.get_mut(domain) {
            cookies.retain(|c| c.name != name);
        }
    }

    /// Clear expired cookies
    pub fn clear_expired(&mut self) {
        for cookies in self.cookies.values_mut() {
            cookies.retain(|c| !c.is_expired());
        }
    }

    /// Clear all cookies
    pub fn clear(&mut self) {
        self.cookies.clear();
    }

    /// Generate Cookie header value for request
    pub fn to_cookie_header(&self, domain: &str, path: &str) -> Option<String> {
        let cookies = self.get(domain, path);
        if cookies.is_empty() {
            return None;
        }
        Some(
            cookies
                .into_iter()
                .map(|c| format!("{}={}", c.name, c.value))
                .collect::<Vec<_>>()
                .join("; "),
        )
    }
}

/// Session management helper
#[derive(Debug)]
pub struct Session {
    pub id: String,
    pub data: HashMap<String, String>,
    pub jar: CookieJar,
    pub last_accessed: u64,
}

impl Session {
    /// Create new session
    pub fn new() -> Self {
        Self {
            id: Self::generate_id(),
            data: HashMap::new(),
            jar: CookieJar::new(),
            last_accessed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Generate session ID
    fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("session_{}", timestamp)
    }

    /// Set session data
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.data.insert(key.into(), value.into());
    }

    /// Get session data
    pub fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(|s| s.as_str())
    }

    /// Remove session data
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }

    /// Check if session is expired (timeout)
    pub fn is_expired(&self, timeout_secs: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now - self.last_accessed > timeout_secs
    }

    /// Touch session (update last accessed)
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cookie_creation() {
        let cookie = Cookie::new("session", "abc123")
            .with_domain("example.com")
            .with_path("/")
            .with_secure(true);

        assert_eq!(cookie.name, "session");
        assert_eq!(cookie.value, "abc123");
        assert!(cookie.secure);
    }

    #[test]
    fn test_cookie_domain_matching() {
        let cookie = Cookie::new("test", "value").with_domain(".example.com");

        assert!(cookie.matches_domain("sub.example.com"));
        assert!(cookie.matches_domain("example.com"));
        assert!(!cookie.matches_domain("other.com"));
    }

    #[test]
    fn test_cookie_path_matching() {
        let cookie = Cookie::new("test", "value").with_path("/api");

        assert!(cookie.matches_path("/api/users"));
        assert!(cookie.matches_path("/api/"));
        assert!(!cookie.matches_path("/"));
    }

    #[test]
    fn test_cookie_jar() {
        let mut jar = CookieJar::new();
        jar.add(Cookie::new("session", "abc").with_domain("example.com"));
        jar.add(Cookie::new("user", "alice").with_domain("example.com"));

        let cookies = jar.get("example.com", "/");
        assert_eq!(cookies.len(), 2);
    }

    #[test]
    fn test_cookie_header_value() {
        let mut jar = CookieJar::new();
        jar.add(Cookie::new("a", "1").with_domain("example.com"));
        jar.add(Cookie::new("b", "2").with_domain("example.com"));

        let header = jar.to_cookie_header("example.com", "/");
        assert_eq!(header, Some("a=1; b=2".to_string()));
    }

    #[test]
    fn test_session() {
        let mut session = Session::new();
        session.set("user_id", "123");
        session.set("username", "alice");

        assert_eq!(session.get("user_id"), Some("123"));
        assert_eq!(session.get("username"), Some("alice"));
        assert!(!session.is_expired(3600));
    }

    #[test]
    fn test_cookie_expiration() {
        let mut jar = CookieJar::new();

        // Add expired cookie
        let expired_cookie = Cookie::new("old", "value").with_expires(1000000000); // 2001-09-09
        jar.add(expired_cookie);

        jar.clear_expired();

        let cookies = jar.get("example.com", "/");
        assert!(cookies.is_empty());
    }
}
