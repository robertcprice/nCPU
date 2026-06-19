//! JSON Utilities for nCPU/nSynth
//!
//! JSON request/response parsing and generation.

use crate::http::types::{Request, Response, StatusCode};
use serde_json::{from_str, to_string, Value};

/// JSON request extension
pub trait JsonRequest {
    /// Parse request body as JSON
    fn json<T>(&self) -> Result<T, String>
    where
        T: for<'de> serde::Deserialize<'de>;

    /// Get body as serde_json::Value
    fn json_value(&self) -> Result<Value, String>;
}

impl JsonRequest for Request {
    fn json<T>(&self) -> Result<T, String>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        let body_str = std::str::from_utf8(&self.body)
            .map_err(|_| "Request body is not valid UTF-8".to_string())?;
        from_str(body_str).map_err(|e| format!("Failed to parse JSON: {}", e))
    }

    fn json_value(&self) -> Result<Value, String> {
        self.json::<Value>()
    }
}

/// JSON response builder
pub struct JsonBuilder {
    value: Value,
}

impl JsonBuilder {
    /// Create from serde_json::Value
    pub fn new(value: Value) -> Self {
        Self { value }
    }

    /// Create from any serializable type
    pub fn from<T>(value: &T) -> Result<Self, String>
    where
        T: serde::Serialize,
    {
        let json_value = serde_json::to_value(value)
            .map_err(|e| format!("Failed to serialize to JSON: {}", e))?;
        Ok(Self::new(json_value))
    }

    /// Build response
    pub fn response(self) -> Response {
        Response::new(StatusCode::OK)
            .with_content_type("application/json")
            .with_body(to_string(&self.value).unwrap_or_default().into_bytes())
    }

    /// Build with custom status code
    pub fn response_with_status(self, status: StatusCode) -> Response {
        Response::new(status)
            .with_content_type("application/json")
            .with_body(to_string(&self.value).unwrap_or_default().into_bytes())
    }
}

/// JSON extraction helpers
pub struct JsonExtractor;

impl JsonExtractor {
    /// Extract field from JSON body
    pub fn extract_field(req: &Request, field: &str) -> Result<Option<Value>, String> {
        let json = req.json_value()?;
        Ok(json.get(field).cloned())
    }

    /// Extract required field from JSON body
    pub fn extract_required(req: &Request, field: &str) -> Result<Value, String> {
        Self::extract_field(req, field)?.ok_or_else(|| format!("Missing required field: {}", field))
    }

    /// Extract string field
    pub fn extract_string(req: &Request, field: &str) -> Result<Option<String>, String> {
        Ok(Self::extract_field(req, field)?.and_then(|v| v.as_str().map(|s| s.to_string())))
    }

    /// Extract number field
    pub fn extract_number(req: &Request, field: &str) -> Result<Option<f64>, String> {
        Ok(Self::extract_field(req, field)?.and_then(|v| v.as_f64()))
    }

    /// Extract bool field
    pub fn extract_bool(req: &Request, field: &str) -> Result<Option<bool>, String> {
        Ok(Self::extract_field(req, field)?.and_then(|v| v.as_bool()))
    }

    /// Extract array field
    pub fn extract_array(req: &Request, field: &str) -> Result<Option<Vec<Value>>, String> {
        Ok(Self::extract_field(req, field)?.and_then(|v| v.as_array().map(|arr| arr.to_vec())))
    }

    /// Extract object field
    pub fn extract_object(
        req: &Request,
        field: &str,
    ) -> Result<Option<serde_json::Map<String, Value>>, String> {
        Ok(Self::extract_field(req, field)?.and_then(|v| v.as_object().map(|obj| obj.clone())))
    }
}

/// JSON API response helpers
pub struct ApiResponse;

impl ApiResponse {
    /// Success response with data
    pub fn success<T>(data: &T) -> Response
    where
        T: serde::Serialize,
    {
        let json = serde_json::json!({
            "success": true,
            "data": data
        });
        Response::new(StatusCode::OK)
            .with_content_type("application/json")
            .with_body(to_string(&json).unwrap_or_default().into_bytes())
    }

    /// Error response with message
    pub fn error(message: &str, status: StatusCode) -> Response {
        let json = serde_json::json!({
            "success": false,
            "error": message
        });
        Response::new(status)
            .with_content_type("application/json")
            .with_body(to_string(&json).unwrap_or_default().into_bytes())
    }

    /// Validation error response
    pub fn validation_error(errors: Vec<&str>) -> Response {
        let json = serde_json::json!({
            "success": false,
            "error": "Validation failed",
            "errors": errors
        });
        Response::new(StatusCode::BadRequest)
            .with_content_type("application/json")
            .with_body(to_string(&json).unwrap_or_default().into_bytes())
    }

    /// Created response (201)
    pub fn created<T>(data: &T) -> Response
    where
        T: serde::Serialize,
    {
        let json = serde_json::json!({
            "success": true,
            "data": data
        });
        Response::new(StatusCode::Created)
            .with_content_type("application/json")
            .with_body(to_string(&json).unwrap_or_default().into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::types::Request;

    #[test]
    fn test_json_request_parse() {
        let req = Request::new(crate::http::types::Method::POST, "/api/data")
            .with_header("content-type", "application/json")
            .with_body(r#"{"name":"Alice","age":30}"#);

        let json: Value = req.json().unwrap();
        assert_eq!(json["name"], "Alice");
        assert_eq!(json["age"], 30);
    }

    #[test]
    fn test_json_request_value() {
        let req = Request::new(crate::http::types::Method::POST, "/api/data")
            .with_body(r#"{"count":42}"#);

        let json = req.json_value().unwrap();
        assert_eq!(json["count"], 42);
    }

    #[test]
    fn test_json_builder() {
        let value = serde_json::json!({"message": "hello"});
        let resp = JsonBuilder::new(value).response();

        assert_eq!(resp.status, StatusCode::OK);
        assert_eq!(resp.content_type(), Some("application/json"));
    }

    #[test]
    fn test_json_builder_from() {
        let data = serde_json::json!({"user": "Bob"});
        let builder = JsonBuilder::from(&data).unwrap();
        let resp = builder.response();

        assert_eq!(resp.status, StatusCode::OK);
        let body = String::from_utf8(resp.body).unwrap();
        assert!(body.contains("Bob"));
    }

    #[test]
    fn test_json_extract_field() {
        let req = Request::new(crate::http::types::Method::POST, "/api/data")
            .with_body(r#"{"name":"Charlie","email":"charlie@example.com"}"#);

        assert_eq!(
            JsonExtractor::extract_string(&req, "name").unwrap(),
            Some("Charlie".to_string())
        );
        assert_eq!(
            JsonExtractor::extract_string(&req, "email").unwrap(),
            Some("charlie@example.com".to_string())
        );
        assert_eq!(
            JsonExtractor::extract_string(&req, "missing").unwrap(),
            None
        );
    }

    #[test]
    fn test_json_extract_required() {
        let req =
            Request::new(crate::http::types::Method::POST, "/api/data").with_body(r#"{"id":123}"#);

        let id = JsonExtractor::extract_required(&req, "id").unwrap();
        assert_eq!(id, 123);

        let err = JsonExtractor::extract_required(&req, "missing");
        assert!(err.is_err());
    }

    #[test]
    fn test_api_response_success() {
        let data = serde_json::json!({"result": "ok"});
        let resp = ApiResponse::success(&data);

        let body = String::from_utf8(resp.body).unwrap();
        assert!(body.contains("success"));
        assert!(body.contains("data"));
    }

    #[test]
    fn test_api_response_error() {
        let resp = ApiResponse::error("Not found", StatusCode::NotFound);

        let body = String::from_utf8(resp.body).unwrap();
        assert!(body.contains("success"));
        assert!(body.contains("error"));
    }
}
