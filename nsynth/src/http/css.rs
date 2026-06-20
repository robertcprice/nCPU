//! CSS/Styling Support for nCPU/nSynth
//!
//! CSS generation and inlining for web applications.

use std::collections::HashMap;

/// CSS property value
#[derive(Debug, Clone)]
pub enum CssValue {
    Color(String),
    Length(f32, Unit),
    Percent(f32),
    String(String),
    Integer(i32),
    Auto,
}

/// CSS unit
#[derive(Debug, Clone, Copy)]
pub enum Unit {
    Px,
    Em,
    Rem,
    Vw,
    Vh,
    Percent,
}

impl CssValue {
    /// Create color value
    pub fn color(hex: impl Into<String>) -> Self {
        CssValue::Color(hex.into())
    }

    /// Create pixel length
    pub fn px(value: f32) -> Self {
        CssValue::Length(value, Unit::Px)
    }

    /// Create em length
    pub fn em(value: f32) -> Self {
        CssValue::Length(value, Unit::Em)
    }

    /// Create percentage
    pub fn percent(value: f32) -> Self {
        CssValue::Percent(value)
    }

    /// Create arbitrary string value
    pub fn string(value: impl Into<String>) -> Self {
        CssValue::String(value.into())
    }

    /// Create auto value
    pub fn auto() -> Self {
        CssValue::Auto
    }

    /// Convert to CSS string
    pub fn to_css(&self) -> String {
        match self {
            CssValue::Color(c) => c.clone(),
            CssValue::Length(v, u) => format!(
                "{}{}",
                v,
                match u {
                    Unit::Px => "px",
                    Unit::Em => "em",
                    Unit::Rem => "rem",
                    Unit::Vw => "vw",
                    Unit::Vh => "vh",
                    Unit::Percent => "%",
                }
            ),
            CssValue::Percent(v) => format!("{}%", v),
            CssValue::String(s) => s.clone(),
            CssValue::Integer(i) => i.to_string(),
            CssValue::Auto => "auto".to_string(),
        }
    }
}

/// CSS rule set
#[derive(Debug, Clone)]
pub struct CssRule {
    /// Selector
    pub selector: String,
    /// Properties
    pub properties: HashMap<String, CssValue>,
}

impl CssRule {
    /// Create new CSS rule
    pub fn new(selector: impl Into<String>) -> Self {
        Self {
            selector: selector.into(),
            properties: HashMap::new(),
        }
    }

    /// Add property
    pub fn add(mut self, property: impl Into<String>, value: CssValue) -> Self {
        self.properties.insert(property.into(), value);
        self
    }

    /// Convert to CSS string
    pub fn to_css(&self) -> String {
        let mut props = self
            .properties
            .iter()
            .map(|(k, v)| format!("{}: {};", k, v.to_css()))
            .collect::<Vec<_>>()
            .join(" ");

        format!("{} {{ {} }}", self.selector, props)
    }
}

/// CSS stylesheet
#[derive(Debug, Clone)]
pub struct Stylesheet {
    /// Rules
    pub rules: Vec<CssRule>,
}

impl Stylesheet {
    /// Create new stylesheet
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add rule
    pub fn add_rule(&mut self, rule: CssRule) -> &mut Self {
        self.rules.push(rule);
        self
    }

    /// Add rule with builder pattern
    pub fn rule(mut self, rule: CssRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Convert to CSS string
    pub fn to_css(&self) -> String {
        self.rules
            .iter()
            .map(|r| r.to_css())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Wrap in <style> tag
    pub fn to_html(&self) -> String {
        format!("<style>{}</style>", self.to_css())
    }
}

impl Default for Stylesheet {
    fn default() -> Self {
        Self::new()
    }
}

/// CSS builder for programmatic styles
#[derive(Debug, Clone)]
pub struct CssBuilder {
    /// Current selector being built
    current_selector: Option<String>,
    /// Current properties being built
    current_props: HashMap<String, CssValue>,
    /// All rules
    rules: Vec<CssRule>,
}

impl CssBuilder {
    /// Create new CSS builder
    pub fn new() -> Self {
        Self {
            current_selector: None,
            current_props: HashMap::new(),
            rules: Vec::new(),
        }
    }

    /// Start a new rule
    pub fn select(mut self, selector: impl Into<String>) -> Self {
        // Save previous rule if exists
        if let Some(sel) = &self.current_selector {
            if !self.current_props.is_empty() {
                let mut rule = CssRule::new(sel.clone());
                for (k, v) in self.current_props.clone() {
                    rule = rule.add(k, v);
                }
                self.rules.push(rule);
            }
        }

        self.current_selector = Some(selector.into());
        self.current_props = HashMap::new();
        self
    }

    /// Add property to current rule
    pub fn set(mut self, property: impl Into<String>, value: CssValue) -> Self {
        self.current_props.insert(property.into(), value);
        self
    }

    /// Add width property
    pub fn width(self, value: CssValue) -> Self {
        self.set("width", value)
    }

    /// Add height property
    pub fn height(self, value: CssValue) -> Self {
        self.set("height", value)
    }

    /// Add margin property
    pub fn margin(self, value: CssValue) -> Self {
        self.set("margin", value)
    }

    /// Add padding property
    pub fn padding(self, value: CssValue) -> Self {
        self.set("padding", value)
    }

    /// Add background-color property
    pub fn background_color(self, color: impl Into<String>) -> Self {
        self.set("background-color", CssValue::Color(color.into()))
    }

    /// Add color property
    pub fn color(self, color: impl Into<String>) -> Self {
        self.set("color", CssValue::Color(color.into()))
    }

    /// Add font-size property
    pub fn font_size(self, value: CssValue) -> Self {
        self.set("font-size", value)
    }

    /// Add display property
    pub fn display(self, value: impl Into<String>) -> Self {
        self.set("display", CssValue::String(value.into()))
    }

    /// Add text-align property
    pub fn text_align(self, value: impl Into<String>) -> Self {
        self.set("text-align", CssValue::String(value.into()))
    }

    /// Finish current rule and build stylesheet
    pub fn build(mut self) -> Stylesheet {
        // Add final rule
        if let Some(sel) = &self.current_selector {
            if !self.current_props.is_empty() {
                let mut rule = CssRule::new(sel.clone());
                for (k, v) in self.current_props {
                    rule = rule.add(k, v);
                }
                self.rules.push(rule);
            }
        }

        Stylesheet { rules: self.rules }
    }
}

impl Default for CssBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Common CSS presets
pub struct CssPresets;

impl CssPresets {
    /// Reset CSS
    pub fn reset() -> String {
        String::from("*{margin:0;padding:0;box-sizing:border-box;}")
    }

    /// Flexbox center
    pub fn flex_center() -> String {
        String::from("{display:flex;justify-content:center;align-items:center;}")
    }

    /// Card style
    pub fn card() -> String {
        String::from("{border:1px solid #ddd;border-radius:8px;padding:16px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}")
    }

    /// Button primary style
    pub fn button_primary() -> String {
        String::from("{background:#007bff;color:white;padding:8px 16px;border:none;border-radius:4px;cursor:pointer;}")
    }

    /// Button secondary style
    pub fn button_secondary() -> String {
        String::from("{background:#6c757d;color:white;padding:8px 16px;border:none;border-radius:4px;cursor:pointer;}")
    }

    /// Container with max-width
    pub fn container(max_width: f32) -> String {
        format!(
            "{{max-width:{}px;margin:0 auto;padding:0 20px;}}",
            max_width
        )
    }

    /// Grid layout
    pub fn grid(columns: u32, gap: f32) -> String {
        format!(
            "{{display:grid;grid-template-columns:repeat({}, 1fr);gap:{}px;}}",
            columns, gap
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_css_value_color() {
        let val = CssValue::color("#ff0000");
        assert_eq!(val.to_css(), "#ff0000");
    }

    #[test]
    fn test_css_value_length() {
        let val = CssValue::px(100.0);
        assert_eq!(val.to_css(), "100px");
    }

    #[test]
    fn test_css_value_percent() {
        let val = CssValue::percent(50.0);
        assert_eq!(val.to_css(), "50%");
    }

    #[test]
    fn test_css_rule() {
        let rule = CssRule::new(".button")
            .add("background-color", CssValue::color("#007bff"))
            .add("color", CssValue::color("#ffffff"));

        let css = rule.to_css();
        assert!(css.contains(".button"));
        assert!(css.contains("background-color"));
        assert!(css.contains("#007bff"));
    }

    #[test]
    fn test_stylesheet() {
        let mut sheet = Stylesheet::new();
        sheet.add_rule(
            CssRule::new("body")
                .add("font-family", CssValue::string("sans-serif"))
                .add("margin", CssValue::px(0.0)),
        );

        let css = sheet.to_css();
        assert!(css.contains("body"));
        assert!(css.contains("font-family"));
    }

    #[test]
    fn test_css_builder() {
        let sheet = CssBuilder::new()
            .select(".container")
            .width(CssValue::px(1200.0))
            .margin(CssValue::auto())
            .select(".button")
            .background_color("#007bff")
            .color("#ffffff")
            .build();

        let css = sheet.to_css();
        assert!(css.contains(".container"));
        assert!(css.contains(".button"));
        assert!(css.contains("background-color"));
    }

    #[test]
    fn test_css_presets() {
        assert!(CssPresets::reset().contains("margin:0"));
        assert!(CssPresets::button_primary().contains("#007bff"));
    }

    #[test]
    fn test_css_presets_grid() {
        let grid = CssPresets::grid(3, 16.0);
        assert!(grid.contains("grid-template-columns"));
        assert!(grid.contains("repeat(3"));
    }
}
