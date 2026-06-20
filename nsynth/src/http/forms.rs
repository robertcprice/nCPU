//! Form Data Handling for nCPU/nSynth
//!
//! URL encoding, form parsing, query string building.

use percent_encoding::{percent_decode, percent_encode, AsciiSet, CONTROLS};
use std::collections::HashMap;

/// Character set for URL encoding (RFC 3986)
const URL_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'!')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'`')
    .add(b'{')
    .add(b'}')
    .add(b'/')
    .add(b':')
    .add(b';')
    .add(b'=')
    .add(b'@')
    .add(b'[')
    .add(b']')
    .add(b'\\')
    .add(b'^')
    .add(b'|')
    .add(b'$')
    .add(b'%')
    .add(b'&')
    .add(b'+')
    .add(b',');

/// Form data container
#[derive(Debug, Clone, Default)]
pub struct FormData {
    fields: HashMap<String, String>,
}

impl FormData {
    /// Create new form data
    pub fn new() -> Self {
        Self::default()
    }

    /// Add field
    pub fn add(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.fields.insert(name.into(), value.into());
    }

    /// Get field
    pub fn get(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(|s| s.as_str())
    }

    /// Get all field names
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(|s| s.as_str())
    }

    /// Get field count
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Encode as application/x-www-form-urlencoded
    pub fn encode(&self) -> String {
        self.fields
            .iter()
            .map(|(k, v)| format!("{}={}", Self::form_encode(k), Self::form_encode(v)))
            .collect::<Vec<_>>()
            .join("&")
    }

    /// URL encode a string (RFC 3986)
    pub fn url_encode(s: &str) -> String {
        percent_encode(s.as_bytes(), URL_ENCODE_SET).to_string()
    }

    /// Form encode (spaces become + per application/x-www-form-urlencoded)
    pub fn form_encode(s: &str) -> String {
        percent_encode(s.as_bytes(), URL_ENCODE_SET)
            .to_string()
            .replace("%20", "+")
    }

    /// URL decode a string
    pub fn url_decode(s: &str) -> Result<String, String> {
        // Convert + to space first (application/x-www-form-urlencoded)
        let with_spaces = s.replace('+', " ");
        let decoded = percent_decode(with_spaces.as_bytes())
            .decode_utf8()
            .map_err(|e| format!("Invalid UTF-8: {}", e))?;
        Ok(decoded.to_string())
    }

    /// Parse from application/x-www-form-urlencoded string
    pub fn parse(encoded: &str) -> Result<Self, String> {
        let mut fields = HashMap::new();

        for pair in encoded.split('&') {
            if pair.is_empty() {
                continue;
            }
            let (key, value) = if let Some(idx) = pair.find('=') {
                (&pair[..idx], &pair[idx + 1..])
            } else {
                (pair, "")
            };

            let key = Self::url_decode(key)?;
            let value = Self::url_decode(value)?;
            fields.insert(key, value);
        }

        Ok(Self { fields })
    }

    /// Convert to HashMap
    pub fn to_map(&self) -> HashMap<String, String> {
        self.fields.clone()
    }

    /// Extend with another form data
    pub fn extend(&mut self, other: FormData) {
        self.fields.extend(other.fields);
    }
}

/// Query string builder
#[derive(Debug, Clone, Default)]
pub struct QueryBuilder {
    params: Vec<(String, String)>,
}

impl QueryBuilder {
    /// Create new query builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add parameter
    pub fn add(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.push((key.into(), value.into()));
        self
    }

    /// Add optional parameter (only if Some)
    pub fn add_opt(self, key: impl Into<String>, value: Option<impl Into<String>>) -> Self {
        if let Some(v) = value {
            self.add(key, v.into())
        } else {
            self
        }
    }

    /// Add multiple values for same key
    pub fn add_many(mut self, key: impl Into<String>, values: Vec<impl Into<String>>) -> Self {
        let key = key.into();
        for value in values {
            self.params.push((key.clone(), value.into()));
        }
        self
    }

    /// Build query string (without leading '?')
    pub fn build(&self) -> String {
        if self.params.is_empty() {
            String::new()
        } else {
            self.params
                .iter()
                .map(|(k, v)| format!("{}={}", FormData::url_encode(k), FormData::url_encode(v)))
                .collect::<Vec<_>>()
                .join("&")
        }
    }

    /// Build with leading '?'
    pub fn build_with_prefix(&self) -> String {
        let query = self.build();
        if query.is_empty() {
            String::new()
        } else {
            format!("?{}", query)
        }
    }

    /// Parse query string
    pub fn parse(query: &str) -> Result<Self, String> {
        let query = query.trim_start_matches('?');
        if query.is_empty() {
            return Ok(Self::new());
        }

        let mut params = Vec::new();
        for pair in query.split('&') {
            if pair.is_empty() {
                continue;
            }
            let (key, value) = if let Some(idx) = pair.find('=') {
                (&pair[..idx], &pair[idx + 1..])
            } else {
                (pair, "")
            };

            let key = FormData::url_decode(key)?;
            let value = FormData::url_decode(value)?;
            params.push((key, value));
        }

        Ok(Self { params })
    }

    /// Get parameter value
    pub fn get(&self, key: &str) -> Option<&str> {
        self.params
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Get all values for a key
    pub fn get_all(&self, key: &str) -> Vec<&str> {
        self.params
            .iter()
            .filter(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    /// Get parameter count
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

/// URL builder
#[derive(Debug, Clone)]
pub struct UrlBuilder {
    scheme: String,
    host: String,
    port: Option<u16>,
    path: String,
    query: QueryBuilder,
    fragment: Option<String>,
}

impl UrlBuilder {
    /// Create new URL builder
    pub fn new() -> Self {
        Self {
            scheme: "http".to_string(),
            host: String::new(),
            port: None,
            path: String::new(),
            query: QueryBuilder::new(),
            fragment: None,
        }
    }

    /// Set scheme (http, https)
    pub fn scheme(mut self, scheme: impl Into<String>) -> Self {
        self.scheme = scheme.into();
        self
    }

    /// Set host
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set port
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set path
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }

    /// Set query parameter
    pub fn query_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query = self.query.add(key, value);
        self
    }

    /// Set fragment
    pub fn fragment(mut self, fragment: impl Into<String>) -> Self {
        self.fragment = Some(fragment.into());
        self
    }

    /// Build URL string
    pub fn build(&self) -> String {
        let mut url = format!("{}://{}", self.scheme, self.host);

        if let Some(port) = self.port {
            if (self.scheme == "http" && port != 80) || (self.scheme == "https" && port != 443) {
                url.push(':');
                url.push_str(&port.to_string());
            }
        }

        if !self.path.is_empty() {
            url.push_str(&self.path);
        }

        let query = self.query.build();
        if !query.is_empty() {
            url.push('?');
            url.push_str(&query);
        }

        if let Some(fragment) = &self.fragment {
            url.push('#');
            url.push_str(fragment);
        }

        url
    }

    /// Parse URL string
    pub fn parse(url: &str) -> Result<Self, String> {
        // Simple URL parser
        let (scheme, rest) = if let Some(idx) = url.find("://") {
            (&url[..idx], &url[idx + 3..])
        } else {
            return Err("Invalid URL: missing scheme".to_string());
        };

        let (host_port, rest) = rest.split_once('/').unwrap_or((rest, ""));
        let (host, port) = if let Some(idx) = host_port.find(':') {
            (
                &host_port[..idx],
                Some(
                    host_port[idx + 1..]
                        .parse::<u16>()
                        .map_err(|_| "Invalid port".to_string())?,
                ),
            )
        } else {
            (host_port, None)
        };

        let (path, query_fragment) = if rest.is_empty() {
            ("/", "")
        } else {
            // Split at ? first, then # will be in query_fragment
            // Note: rest doesn't include leading / (it was consumed by split_once)
            let pf = if let Some(idx) = rest.find('?') {
                (&rest[..idx], &rest[idx..])
            } else if let Some(idx) = rest.find('#') {
                (&rest[..idx], &rest[idx..])
            } else {
                (rest, "")
            };
            pf
        };

        let (final_path, query, fragment) = if query_fragment.is_empty() {
            // No query or fragment
            (path, "", None)
        } else if query_fragment.starts_with('?') {
            // query_fragment starts with ?, so path is already set
            let after_q = &query_fragment[1..];
            if let Some(frag_idx) = after_q.find('#') {
                (path, &after_q[..frag_idx], Some(&after_q[frag_idx + 1..]))
            } else {
                (path, after_q, None)
            }
        } else if let Some(idx) = query_fragment.find('?') {
            // ? is somewhere in the middle - path is before ?
            let after_q = &query_fragment[idx + 1..];
            if let Some(frag_idx) = after_q.find('#') {
                (
                    &query_fragment[..idx],
                    &after_q[..frag_idx],
                    Some(&after_q[frag_idx + 1..]),
                )
            } else {
                (&query_fragment[..idx], after_q, None)
            }
        } else if let Some(frag_idx) = query_fragment.find('#') {
            // Only fragment, no query
            (
                &query_fragment[..frag_idx],
                "",
                Some(&query_fragment[frag_idx + 1..]),
            )
        } else {
            (path, "", None)
        };

        let query = QueryBuilder::parse(query)?;

        // Ensure path starts with /
        let final_path_str = if final_path.starts_with('/') {
            final_path.to_string()
        } else {
            format!("/{}", final_path)
        };

        Ok(Self {
            scheme: scheme.to_string(),
            host: host.to_string(),
            port,
            path: final_path_str,
            query,
            fragment: fragment.map(|s| s.to_string()),
        })
    }
}

impl Default for UrlBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_form_data_encode() {
        let mut form = FormData::new();
        form.add("username", "alice@example.com");
        form.add("message", "hello world!");

        let encoded = form.encode();
        assert!(encoded.contains("username=alice%40example.com"));
        assert!(encoded.contains("message=hello+world%21"));
    }

    #[test]
    fn test_form_data_parse() {
        let encoded = "name=Bob+Smith&age=30&city=New+York";
        let form = FormData::parse(encoded).unwrap();

        assert_eq!(form.get("name"), Some("Bob Smith"));
        assert_eq!(form.get("age"), Some("30"));
        assert_eq!(form.get("city"), Some("New York"));
    }

    #[test]
    fn test_url_encode_decode() {
        let original = "hello world!@#$%";
        let encoded = FormData::url_encode(original);
        let decoded = FormData::url_decode(&encoded).unwrap();

        assert_eq!(decoded, original);
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::new()
            .add("page", "1")
            .add("limit", "10")
            .add_opt("filter", Some("active"))
            .build();

        assert_eq!(query, "page=1&limit=10&filter=active");
    }

    #[test]
    fn test_query_builder_parse() {
        let query = QueryBuilder::parse("page=2&sort=name&sort=date").unwrap();

        assert_eq!(query.get("page"), Some("2"));
        assert_eq!(query.get_all("sort").len(), 2);
    }

    #[test]
    fn test_url_builder() {
        let url = UrlBuilder::new()
            .scheme("https")
            .host("example.com")
            .port(8443)
            .path("/api/users")
            .query_param("page", "1")
            .fragment("section")
            .build();

        assert_eq!(url, "https://example.com:8443/api/users?page=1#section");
    }

    #[test]
    fn test_url_parse() {
        let url = UrlBuilder::parse("https://example.com:8080/path?query=value#frag").unwrap();

        assert_eq!(url.scheme, "https");
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, Some(8080));
        assert_eq!(url.path, "/path");
        assert_eq!(url.query.get("query"), Some("value"));
        assert_eq!(url.fragment, Some("frag".to_string()));
    }

    #[test]
    fn test_form_data_multi_value() {
        let query = QueryBuilder::new()
            .add_many("tag", vec!["rust", "http", "web"])
            .build();

        assert!(query.contains("tag=rust"));
        assert!(query.contains("tag=http"));
        assert!(query.contains("tag=web"));
    }
}
