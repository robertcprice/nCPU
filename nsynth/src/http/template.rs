//! HTML Template Engine for nCPU/nSynth
//!
//! Simple template rendering with variable substitution.

use crate::http::types::Response;
use std::collections::HashMap;

/// Template engine
#[derive(Debug, Clone)]
pub struct Template {
    /// Template content
    content: String,
}

impl Template {
    /// Create new template from string
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    /// Create new template from file
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Ok(Self::new(content))
    }

    /// Render template with variables
    pub fn render(&self, vars: &HashMap<String, String>) -> String {
        let mut result = self.content.clone();

        for (key, value) in vars {
            result = result.replace(&format!("{{{{{}}}}}", key), value);
        }

        result
    }

    /// Render with optional variables (missing keys become empty string)
    pub fn render_opt(&self, vars: &HashMap<String, Option<String>>) -> String {
        let mut result = self.content.clone();

        for (key, value_opt) in vars {
            let replacement = value_opt.as_deref().unwrap_or("");
            result = result.replace(&format!("{{{{{}}}}}", key), replacement);
        }

        result
    }

    /// Render to HTML response
    pub fn html_response(&self, vars: &HashMap<String, String>) -> Response {
        Response::new(crate::http::types::StatusCode::OK)
            .with_content_type("text/html")
            .with_body(self.render(vars).into_bytes())
    }

    /// Parse template file
    pub fn parse(path: impl AsRef<std::path::Path>) -> Result<Self, std::io::Error> {
        Self::from_file(path)
    }
}

/// Template builder for programmatic template creation
#[derive(Debug, Clone)]
pub struct TemplateBuilder {
    /// Buffer for building template
    buffer: String,
}

impl TemplateBuilder {
    /// Create new template builder
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Add HTML header
    pub fn html_header(mut self, title: &str) -> Self {
        self.buffer.push_str(&format!(
            "<!DOCTYPE html>\n<html>\n<head><title>{}</title>\n",
            title
        ));
        self
    }

    /// Add stylesheet link
    pub fn stylesheet(mut self, href: &str) -> Self {
        self.buffer
            .push_str(&format!("<link rel=\"stylesheet\" href=\"{}\">\n", href));
        self
    }

    /// Add inline CSS
    pub fn css(mut self, css: &str) -> Self {
        self.buffer.push_str(&format!("<style>{}</style>\n", css));
        self
    }

    /// Add script tag
    pub fn script(mut self, src: &str) -> Self {
        self.buffer
            .push_str(&format!("<script src=\"{}\"></script>\n", src));
        self
    }

    /// Close head and start body
    pub fn body_start(mut self) -> Self {
        self.buffer.push_str("</head>\n<body>\n");
        self
    }

    /// Add heading
    pub fn heading(mut self, level: u8, text: &str) -> Self {
        self.buffer
            .push_str(&format!("<h{}>{}</h{}>\n", level, text, level));
        self
    }

    /// Add paragraph
    pub fn paragraph(mut self, text: &str) -> Self {
        self.buffer.push_str(&format!("<p>{}</p>\n", text));
        self
    }

    /// Add div
    pub fn div(mut self, class: &str, content: &str) -> Self {
        self.buffer
            .push_str(&format!("<div class=\"{}\">{}</div>\n", class, content));
        self
    }

    /// Add link
    pub fn link(mut self, href: &str, text: &str) -> Self {
        self.buffer
            .push_str(&format!("<a href=\"{}\">{}</a>\n", href, text));
        self
    }

    /// Add form
    pub fn form(mut self, action: &str, method: &str, content: &str) -> Self {
        self.buffer.push_str(&format!(
            "<form action=\"{}\" method=\"{}\">{}</form>\n",
            action, method, content
        ));
        self
    }

    /// Add input field
    pub fn input(mut self, name: &str, input_type: &str) -> Self {
        self.buffer.push_str(&format!(
            "<input type=\"{}\" name=\"{}\">\n",
            input_type, name
        ));
        self
    }

    /// Add button
    pub fn button(mut self, text: &str) -> Self {
        self.buffer
            .push_str(&format!("<button>{}</button>\n", text));
        self
    }

    /// Add table row
    pub fn table_row(mut self, cells: &[&str]) -> Self {
        self.buffer.push_str("<tr>");
        for cell in cells {
            self.buffer.push_str(&format!("<td>{}</td>", cell));
        }
        self.buffer.push_str("</tr>\n");
        self
    }

    /// Close body and html
    pub fn html_footer(mut self) -> Self {
        self.buffer.push_str("</body>\n</html>");
        self
    }

    /// Build template string
    pub fn build(self) -> Template {
        Template::new(self.buffer)
    }
}

impl Default for TemplateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Common HTML templates
pub struct HtmlTemplates;

impl HtmlTemplates {
    /// Basic error page
    pub fn error(title: &str, message: &str) -> String {
        format!(
            "<!DOCTYPE html>\n<html>\n<head><title>{}</title>\n\
             <style>body{{font-family:sans-serif;margin:40px;}}</style>\n\
             </head>\n<body>\n<h1>{}</h1>\n<p>{}</p>\n</body>\n</html>",
            title, title, message
        )
    }

    /// Basic success page
    pub fn success(title: &str, message: &str) -> String {
        format!(
            "<!DOCTYPE html>\n<html>\n<head><title>{}</title>\n\
             <style>body{{font-family:sans-serif;margin:40px;}}</style>\n\
             </head>\n<body>\n<h1>{}</h1>\n<p>{}</p>\n</body>\n</html>",
            title, title, message
        )
    }

    /// Login form template
    pub fn login_form(action: &str) -> String {
        format!(
            "<!DOCTYPE html>\n<html>\n<head><title>Login</title>\n\
             <style>body{{font-family:sans-serif;margin:40px;}}\
             label{{display:block;}} input{{margin-bottom:10px;}}</style>\n\
             </head>\n<body>\n<h1>Login</h1>\n\
             <form action=\"{}\" method=\"POST\">\n\
             <label>Username:</label><input type=\"text\" name=\"username\">\n\
             <label>Password:</label><input type=\"password\" name=\"password\">\n\
             <button type=\"submit\">Login</button>\n\
             </form>\n</body>\n</html>",
            action
        )
    }

    /// Redirect page with meta refresh
    pub fn redirect(title: &str, message: &str, redirect_url: &str, delay: u32) -> String {
        format!(
            "<!DOCTYPE html>\n<html>\n<head><title>{}</title>\n\
             <meta http-equiv=\"refresh\" content=\"{};url={}\">\n\
             <style>body{{font-family:sans-serif;margin:40px;}}</style>\n\
             </head>\n<body>\n<h1>{}</h1>\n<p>{}</p>\n\
             <p>Redirecting to <a href=\"{}\">{}</a>...</p>\n</body>\n</html>",
            title, delay, redirect_url, title, message, redirect_url, redirect_url
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_render() {
        let tpl = Template::new("Hello {{name}}!");
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "World".to_string());

        assert_eq!(tpl.render(&vars), "Hello World!");
    }

    #[test]
    fn test_template_render_multiple_vars() {
        let tpl = Template::new("{{greeting}}, {{name}}!");
        let mut vars = HashMap::new();
        vars.insert("greeting".to_string(), "Hello".to_string());
        vars.insert("name".to_string(), "Alice".to_string());

        assert_eq!(tpl.render(&vars), "Hello, Alice!");
    }

    #[test]
    fn test_template_render_missing_var() {
        let tpl = Template::new("Hello {{name}}!");
        let vars = HashMap::new();

        assert_eq!(tpl.render(&vars), "Hello {{name}}!");
    }

    #[test]
    fn test_template_html_response() {
        let tpl = Template::new("<h1>{{title}}</h1>");
        let mut vars = HashMap::new();
        vars.insert("title".to_string(), "Test".to_string());

        let resp = tpl.html_response(&vars);
        assert_eq!(resp.status, crate::http::types::StatusCode::OK);
        assert_eq!(resp.content_type(), Some("text/html"));
    }

    #[test]
    fn test_template_builder() {
        let tpl = TemplateBuilder::new()
            .html_header("Test")
            .body_start()
            .heading(1, "Welcome")
            .html_footer()
            .build();

        let rendered = tpl.render(&HashMap::new());
        assert!(rendered.contains("<h1>Welcome</h1>"));
        assert!(rendered.contains("<title>Test</title>"));
    }

    #[test]
    fn test_template_builder_input() {
        let tpl = TemplateBuilder::new()
            .html_header("Form")
            .body_start()
            .input("username", "text")
            .input("password", "password")
            .html_footer()
            .build();

        let rendered = tpl.render(&HashMap::new());
        assert!(rendered.contains("type=\"text\""));
        assert!(rendered.contains("type=\"password\""));
    }

    #[test]
    fn test_html_templates_error() {
        let html = HtmlTemplates::error("Error", "Something went wrong");
        assert!(html.contains("Error"));
        assert!(html.contains("Something went wrong"));
    }

    #[test]
    fn test_html_templates_login() {
        let html = HtmlTemplates::login_form("/login");
        assert!(html.contains("<form"));
        assert!(html.contains("username"));
        assert!(html.contains("password"));
    }
}
