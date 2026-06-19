//! Multipart Form Data Support for nCPU/nSynth
//!
//! RFC 7578 multipart/form-data encoding and parsing.

use crate::http::types::HeaderMap;
use std::collections::HashMap;

/// Multipart form data field
#[derive(Debug, Clone)]
pub enum MultipartField {
    /// Simple text field
    Text(String),
    /// File upload with filename and content
    File {
        filename: String,
        content: Vec<u8>,
        content_type: String,
    },
}

/// Multipart form data
#[derive(Debug, Clone)]
pub struct MultipartData {
    pub boundary: String,
    pub fields: HashMap<String, MultipartField>,
}

impl MultipartData {
    /// Create new multipart data with random boundary
    pub fn new() -> Self {
        Self::with_boundary(Self::generate_boundary())
    }

    /// Create with specific boundary
    pub fn with_boundary(boundary: String) -> Self {
        Self {
            boundary,
            fields: HashMap::new(),
        }
    }

    /// Generate random boundary string
    fn generate_boundary() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("----boundary_{}", timestamp)
    }

    /// Add text field
    pub fn add_text(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.fields
            .insert(name.into(), MultipartField::Text(value.into()));
    }

    /// Add file field
    pub fn add_file(
        &mut self,
        name: impl Into<String>,
        filename: impl Into<String>,
        content: Vec<u8>,
        content_type: impl Into<String>,
    ) {
        self.fields.insert(
            name.into(),
            MultipartField::File {
                filename: filename.into(),
                content,
                content_type: content_type.into(),
            },
        );
    }

    /// Encode to multipart/form-data bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut result = Vec::new();

        for (name, field) in &self.fields {
            // Boundary
            result.extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());

            match field {
                MultipartField::Text(value) => {
                    result.extend_from_slice(
                        format!(
                            "Content-Disposition: form-data; name=\"{}\"\r\n\r\n{}\r\n",
                            name, value
                        )
                        .as_bytes(),
                    );
                }
                MultipartField::File {
                    filename,
                    content,
                    content_type,
                } => {
                    result.extend_from_slice(
                        format!(
                            "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n",
                            name, filename
                        )
                        .as_bytes(),
                    );
                    result.extend_from_slice(
                        format!("Content-Type: {}\r\n\r\n", content_type).as_bytes(),
                    );
                    result.extend_from_slice(content);
                    result.extend_from_slice(b"\r\n");
                }
            }
        }

        // Final boundary
        result.extend_from_slice(format!("--{}--\r\n", self.boundary).as_bytes());

        result
    }

    /// Get content-type header value for this multipart data
    pub fn content_type(&self) -> String {
        format!("multipart/form-data; boundary={}", self.boundary)
    }

    /// Parse multipart data from bytes
    pub fn parse(content_type: &str, body: &[u8]) -> Result<Self, String> {
        // Extract boundary from Content-Type header
        let boundary = if let Some(b) = content_type.strip_prefix("multipart/form-data; boundary=")
        {
            b.to_string()
        } else if let Some(b) = content_type.strip_prefix("multipart/form-data; boundary=") {
            b.to_string()
        } else {
            return Err("Invalid Content-Type for multipart".to_string());
        };

        let mut fields = HashMap::new();
        let body_str = String::from_utf8(body.to_vec())
            .map_err(|_| "Invalid UTF-8 in multipart body".to_string())?;

        // Split by boundary
        let boundary_marker = format!("--{}", boundary);
        let parts: Vec<&str> = body_str.split(&boundary_marker).collect();

        // Skip first (empty) and last (closing marker) parts
        for part in parts.iter().skip(1).take(parts.len() - 2) {
            if let Some(field) = Self::parse_part(part)? {
                fields.insert(field.0, field.1);
            }
        }

        Ok(Self { boundary, fields })
    }

    /// Parse a single multipart part
    fn parse_part(part: &str) -> Result<Option<(String, MultipartField)>, String> {
        let lines: Vec<&str> = part.lines().collect();
        if lines.is_empty() {
            return Ok(None);
        }

        // Parse headers (until empty line)
        let mut headers = HeaderMap::new();
        let mut header_end = 0;
        for (i, line) in lines.iter().enumerate() {
            if line.is_empty() {
                header_end = i;
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim(), value.trim());
            }
        }

        // Get Content-Disposition
        let cd = headers
            .get("content-disposition")
            .ok_or("Missing Content-Disposition")?;

        // Parse name from Content-Disposition
        let name = if let Some(n) = cd.split("name=\"").nth(1) {
            n.split('"').next().unwrap_or("").to_string()
        } else {
            return Err("Missing name in Content-Disposition".to_string());
        };

        // Check for filename
        let filename = if cd.contains("filename=\"") {
            if let Some(f) = cd.split("filename=\"").nth(1) {
                Some(f.split('"').next().unwrap_or("").to_string())
            } else {
                None
            }
        } else {
            None
        };

        // Get content (after headers)
        let content: String = lines
            .iter()
            .skip(header_end + 1)
            .cloned()
            .collect::<Vec<_>>()
            .join("\r\n");

        let field = if let Some(fname) = filename {
            let content_type = headers
                .get("content-type")
                .unwrap_or(&"application/octet-stream".to_string())
                .to_string();
            MultipartField::File {
                filename: fname,
                content: content.into_bytes(),
                content_type,
            }
        } else {
            MultipartField::Text(content.trim().to_string())
        };

        Ok(Some((name, field)))
    }

    /// Get field by name
    pub fn get(&self, name: &str) -> Option<&MultipartField> {
        self.fields.get(name)
    }

    /// Get text field
    pub fn get_text(&self, name: &str) -> Option<&str> {
        match self.get(name) {
            Some(MultipartField::Text(s)) => Some(s),
            _ => None,
        }
    }

    /// Get file field
    pub fn get_file(&self, name: &str) -> Option<&[u8]> {
        match self.get(name) {
            Some(MultipartField::File { content, .. }) => Some(content),
            _ => None,
        }
    }

    /// Get field count
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl Default for MultipartData {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multipart_creation() {
        let mut mp = MultipartData::new();
        mp.add_text("username", "alice");
        mp.add_file("avatar", "avatar.jpg", vec![1, 2, 3], "image/jpeg");

        assert_eq!(mp.len(), 2);
        assert_eq!(mp.get_text("username"), Some("alice"));
    }

    #[test]
    fn test_multipart_encode() {
        let mut mp = MultipartData::with_boundary("----xyz".to_string());
        mp.add_text("field1", "value1");

        let encoded = mp.encode();
        let encoded_str = String::from_utf8(encoded).unwrap();

        assert!(encoded_str.contains("----xyz"));
        assert!(encoded_str.contains("name=\"field1\""));
        assert!(encoded_str.contains("value1"));
    }

    #[test]
    fn test_content_type_header() {
        let mp = MultipartData::with_boundary("----abc123".to_string());
        assert_eq!(
            mp.content_type(),
            "multipart/form-data; boundary=----abc123"
        );
    }

    #[test]
    fn test_multipart_with_file() {
        let mut mp = MultipartData::new();
        mp.add_file(
            "upload",
            "test.txt",
            b"Hello, World!".to_vec(),
            "text/plain",
        );

        if let Some(MultipartField::File {
            filename,
            content,
            content_type,
        }) = mp.get("upload")
        {
            assert_eq!(filename, "test.txt");
            assert_eq!(content, b"Hello, World!");
            assert_eq!(content_type, "text/plain");
        } else {
            panic!("Expected file field");
        }
    }

    #[test]
    fn test_get_file() {
        let mut mp = MultipartData::new();
        mp.add_file(
            "data",
            "file.bin",
            vec![0, 1, 2],
            "application/octet-stream",
        );

        assert_eq!(mp.get_file("data"), Some(&[0, 1, 2][..]));
    }
}
