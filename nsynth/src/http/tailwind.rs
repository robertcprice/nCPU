//! Tailwind CSS Implementation for nCPU/nSynth
//!
//! Complete Tailwind CSS utility system with configuration, design tokens,
//! responsive variants, and pre-built component primitives.
//!
//! # Examples
//!
//! ```rust
//! use nsynth::http::tailwind::{TailwindConfig, TailwindUtility, DesignSystem};
//!
//! let config = TailwindConfig::default();
//! let utility = TailwindUtility::new(&config);
//!
//! // Generate utility classes
//! let classes = utility.classes()
//!     .padding("p-4")
//!     .margin("m-2")
//!     .text("text-blue-500")
//!     .background("bg-white")
//!     .rounded("rounded-lg")
//!     .shadow("shadow-md")
//!     .build();
//!
//! // Use design tokens
//! let theme = DesignSystem::modern();
//! let primary_color = theme.color("primary");
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

// ============================================================================
// Tailwind Configuration
// ============================================================================

/// Tailwind CSS configuration
#[derive(Debug, Clone)]
pub struct TailwindConfig {
    /// Content paths for CSS purging
    pub content: Vec<String>,
    /// Theme configuration
    pub theme: ThemeConfig,
    /// Plugins configuration
    pub plugins: Vec<String>,
    /// Prefix for all utilities
    pub prefix: Option<String>,
    /// Important selector strategy
    pub important: bool,
    /// Separator for modifiers
    pub separator: String,
    /// Whether to use @apply directives
    pub core_plugins: Vec<String>,
}

/// Theme configuration
#[derive(Debug, Clone)]
pub struct ThemeConfig {
    /// Color palette
    pub colors: ColorPalette,
    /// Spacing scale
    pub spacing: SpacingScale,
    /// Font sizes
    pub font_size: FontSizes,
    /// Border radius
    pub border_radius: BorderRadius,
    /// Shadows
    pub shadows: Shadows,
    /// Screens (breakpoints)
    pub screens: Screens,
    /// Z-index scale
    pub z_index: ZIndex,
    /// Transition timing
    pub transition: TransitionConfig,
}

/// Color palette configuration
#[derive(Debug, Clone)]
pub struct ColorPalette {
    /// Primary colors
    pub primary: HashMap<String, String>,
    /// Secondary colors
    pub secondary: HashMap<String, String>,
    /// Neutral colors
    pub neutral: HashMap<String, String>,
    /// Semantic colors
    pub semantic: SemanticColors,
}

/// Semantic colors for common UI states
#[derive(Debug, Clone)]
pub struct SemanticColors {
    pub info: HashMap<String, String>,
    pub success: HashMap<String, String>,
    pub warning: HashMap<String, String>,
    pub error: HashMap<String, String>,
}

impl Default for ColorPalette {
    fn default() -> Self {
        let mut primary = HashMap::new();
        primary.insert("50".to_string(), "#eff6ff".to_string());
        primary.insert("100".to_string(), "#dbeafe".to_string());
        primary.insert("200".to_string(), "#bfdbfe".to_string());
        primary.insert("300".to_string(), "#93c5fd".to_string());
        primary.insert("400".to_string(), "#60a5fa".to_string());
        primary.insert("500".to_string(), "#3b82f6".to_string());
        primary.insert("600".to_string(), "#2563eb".to_string());
        primary.insert("700".to_string(), "#1d4ed8".to_string());
        primary.insert("800".to_string(), "#1e40af".to_string());
        primary.insert("900".to_string(), "#1e3a8a".to_string());
        primary.insert("950".to_string(), "#172554".to_string());

        let mut neutral = HashMap::new();
        neutral.insert("50".to_string(), "#f9fafb".to_string());
        neutral.insert("100".to_string(), "#f3f4f6".to_string());
        neutral.insert("200".to_string(), "#e5e7eb".to_string());
        neutral.insert("300".to_string(), "#d1d5db".to_string());
        neutral.insert("400".to_string(), "#9ca3af".to_string());
        neutral.insert("500".to_string(), "#6b7280".to_string());
        neutral.insert("600".to_string(), "#4b5563".to_string());
        neutral.insert("700".to_string(), "#374151".to_string());
        neutral.insert("800".to_string(), "#1f2937".to_string());
        neutral.insert("900".to_string(), "#111827".to_string());
        neutral.insert("950".to_string(), "#030712".to_string());

        let mut error = HashMap::new();
        error.insert("50".to_string(), "#fef2f2".to_string());
        error.insert("100".to_string(), "#fee2e2".to_string());
        error.insert("500".to_string(), "#ef4444".to_string());
        error.insert("600".to_string(), "#dc2626".to_string());
        error.insert("700".to_string(), "#b91c1c".to_string());

        let mut success = HashMap::new();
        success.insert("50".to_string(), "#f0fdf4".to_string());
        success.insert("100".to_string(), "#dcfce7".to_string());
        success.insert("500".to_string(), "#22c55e".to_string());
        success.insert("600".to_string(), "#16a34a".to_string());
        success.insert("700".to_string(), "#15803d".to_string());

        let semantic = SemanticColors {
            info: HashMap::new(),
            success,
            warning: HashMap::new(),
            error,
        };

        Self {
            primary,
            secondary: HashMap::new(),
            neutral,
            semantic,
        }
    }
}

/// Spacing scale configuration
#[derive(Debug, Clone)]
pub struct SpacingScale {
    pub scale: HashMap<String, String>,
}

impl Default for SpacingScale {
    fn default() -> Self {
        let mut scale = HashMap::new();
        scale.insert("0".to_string(), "0px".to_string());
        scale.insert("px".to_string(), "1px".to_string());
        scale.insert("0.5".to_string(), "0.125rem".to_string());
        scale.insert("1".to_string(), "0.25rem".to_string());
        scale.insert("2".to_string(), "0.5rem".to_string());
        scale.insert("3".to_string(), "0.75rem".to_string());
        scale.insert("4".to_string(), "1rem".to_string());
        scale.insert("5".to_string(), "1.25rem".to_string());
        scale.insert("6".to_string(), "1.5rem".to_string());
        scale.insert("8".to_string(), "2rem".to_string());
        scale.insert("10".to_string(), "2.5rem".to_string());
        scale.insert("12".to_string(), "3rem".to_string());
        scale.insert("16".to_string(), "4rem".to_string());
        scale.insert("20".to_string(), "5rem".to_string());
        scale.insert("24".to_string(), "6rem".to_string());
        scale.insert("32".to_string(), "8rem".to_string());
        scale.insert("40".to_string(), "10rem".to_string());
        scale.insert("48".to_string(), "12rem".to_string());
        scale.insert("56".to_string(), "14rem".to_string());
        scale.insert("64".to_string(), "16rem".to_string());
        scale.insert("72".to_string(), "18rem".to_string());
        scale.insert("80".to_string(), "20rem".to_string());
        scale.insert("96".to_string(), "24rem".to_string());
        Self { scale }
    }
}

/// Font size configuration
#[derive(Debug, Clone)]
pub struct FontSizes {
    pub scale: HashMap<String, FontSize>,
}

#[derive(Debug, Clone)]
pub struct FontSize {
    pub size: String,
    pub line_height: Option<String>,
}

impl Default for FontSizes {
    fn default() -> Self {
        let mut scale = HashMap::new();
        scale.insert(
            "xs".to_string(),
            FontSize {
                size: "0.75rem".to_string(),
                line_height: Some("1rem".to_string()),
            },
        );
        scale.insert(
            "sm".to_string(),
            FontSize {
                size: "0.875rem".to_string(),
                line_height: Some("1.25rem".to_string()),
            },
        );
        scale.insert(
            "base".to_string(),
            FontSize {
                size: "1rem".to_string(),
                line_height: Some("1.5rem".to_string()),
            },
        );
        scale.insert(
            "lg".to_string(),
            FontSize {
                size: "1.125rem".to_string(),
                line_height: Some("1.75rem".to_string()),
            },
        );
        scale.insert(
            "xl".to_string(),
            FontSize {
                size: "1.25rem".to_string(),
                line_height: Some("1.75rem".to_string()),
            },
        );
        scale.insert(
            "2xl".to_string(),
            FontSize {
                size: "1.5rem".to_string(),
                line_height: Some("2rem".to_string()),
            },
        );
        scale.insert(
            "3xl".to_string(),
            FontSize {
                size: "1.875rem".to_string(),
                line_height: Some("2.25rem".to_string()),
            },
        );
        scale.insert(
            "4xl".to_string(),
            FontSize {
                size: "2.25rem".to_string(),
                line_height: Some("2.5rem".to_string()),
            },
        );
        scale.insert(
            "5xl".to_string(),
            FontSize {
                size: "3rem".to_string(),
                line_height: Some("1".to_string()),
            },
        );
        scale.insert(
            "6xl".to_string(),
            FontSize {
                size: "3.75rem".to_string(),
                line_height: Some("1".to_string()),
            },
        );
        scale.insert(
            "7xl".to_string(),
            FontSize {
                size: "4.5rem".to_string(),
                line_height: Some("1".to_string()),
            },
        );
        scale.insert(
            "8xl".to_string(),
            FontSize {
                size: "6rem".to_string(),
                line_height: Some("1".to_string()),
            },
        );
        scale.insert(
            "9xl".to_string(),
            FontSize {
                size: "8rem".to_string(),
                line_height: Some("1".to_string()),
            },
        );
        Self { scale }
    }
}

/// Border radius configuration
#[derive(Debug, Clone)]
pub struct BorderRadius {
    pub scale: HashMap<String, String>,
}

impl Default for BorderRadius {
    fn default() -> Self {
        let mut scale = HashMap::new();
        scale.insert("none".to_string(), "0px".to_string());
        scale.insert("sm".to_string(), "0.125rem".to_string());
        scale.insert("DEFAULT".to_string(), "0.25rem".to_string());
        scale.insert("md".to_string(), "0.375rem".to_string());
        scale.insert("lg".to_string(), "0.5rem".to_string());
        scale.insert("xl".to_string(), "0.75rem".to_string());
        scale.insert("2xl".to_string(), "1rem".to_string());
        scale.insert("3xl".to_string(), "1.5rem".to_string());
        scale.insert("full".to_string(), "9999px".to_string());
        Self { scale }
    }
}

/// Shadow configuration
#[derive(Debug, Clone)]
pub struct Shadows {
    pub scale: HashMap<String, String>,
}

impl Default for Shadows {
    fn default() -> Self {
        let mut scale = HashMap::new();
        scale.insert(
            "sm".to_string(),
            "0 1px 2px 0 rgb(0 0 0 / 0.05)".to_string(),
        );
        scale.insert(
            "DEFAULT".to_string(),
            "0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)".to_string(),
        );
        scale.insert(
            "md".to_string(),
            "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)".to_string(),
        );
        scale.insert(
            "lg".to_string(),
            "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)".to_string(),
        );
        scale.insert(
            "xl".to_string(),
            "0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)".to_string(),
        );
        scale.insert(
            "2xl".to_string(),
            "0 25px 50px -12px rgb(0 0 0 / 0.25)".to_string(),
        );
        scale.insert(
            "inner".to_string(),
            "inset 0 2px 4px 0 rgb(0 0 0 / 0.05)".to_string(),
        );
        scale.insert("none".to_string(), "0 0 #0000".to_string());
        Self { scale }
    }
}

/// Screen breakpoints
#[derive(Debug, Clone)]
pub struct Screens {
    pub breakpoints: HashMap<String, String>,
}

impl Default for Screens {
    fn default() -> Self {
        let mut breakpoints = HashMap::new();
        breakpoints.insert("sm".to_string(), "640px".to_string());
        breakpoints.insert("md".to_string(), "768px".to_string());
        breakpoints.insert("lg".to_string(), "1024px".to_string());
        breakpoints.insert("xl".to_string(), "1280px".to_string());
        breakpoints.insert("2xl".to_string(), "1536px".to_string());
        Self { breakpoints }
    }
}

/// Z-index scale
#[derive(Debug, Clone)]
pub struct ZIndex {
    pub scale: HashMap<String, String>,
}

impl Default for ZIndex {
    fn default() -> Self {
        let mut scale = HashMap::new();
        scale.insert("0".to_string(), "0".to_string());
        scale.insert("10".to_string(), "10".to_string());
        scale.insert("20".to_string(), "20".to_string());
        scale.insert("30".to_string(), "30".to_string());
        scale.insert("40".to_string(), "40".to_string());
        scale.insert("50".to_string(), "50".to_string());
        scale.insert("auto".to_string(), "auto".to_string());
        Self { scale }
    }
}

/// Transition configuration
#[derive(Debug, Clone)]
pub struct TransitionConfig {
    pub duration: HashMap<String, String>,
    pub timing: HashMap<String, String>,
}

impl Default for TransitionConfig {
    fn default() -> Self {
        let mut duration = HashMap::new();
        duration.insert("75".to_string(), "75ms".to_string());
        duration.insert("100".to_string(), "100ms".to_string());
        duration.insert("150".to_string(), "150ms".to_string());
        duration.insert("200".to_string(), "200ms".to_string());
        duration.insert("300".to_string(), "300ms".to_string());
        duration.insert("500".to_string(), "500ms".to_string());
        duration.insert("700".to_string(), "700ms".to_string());
        duration.insert("1000".to_string(), "1000ms".to_string());

        let mut timing = HashMap::new();
        timing.insert("linear".to_string(), "linear".to_string());
        timing.insert("in".to_string(), "cubic-bezier(0.4, 0, 1, 1)".to_string());
        timing.insert("out".to_string(), "cubic-bezier(0, 0, 0.2, 1)".to_string());
        timing.insert(
            "in-out".to_string(),
            "cubic-bezier(0.4, 0, 0.2, 1)".to_string(),
        );

        Self { duration, timing }
    }
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            colors: ColorPalette::default(),
            spacing: SpacingScale::default(),
            font_size: FontSizes::default(),
            border_radius: BorderRadius::default(),
            shadows: Shadows::default(),
            screens: Screens::default(),
            z_index: ZIndex::default(),
            transition: TransitionConfig::default(),
        }
    }
}

impl Default for TailwindConfig {
    fn default() -> Self {
        Self {
            content: vec!["./src/**/*.html".to_string(), "./src/**/*.rs".to_string()],
            theme: ThemeConfig::default(),
            plugins: vec![
                "@tailwindcss/forms".to_string(),
                "@tailwindcss/typography".to_string(),
            ],
            prefix: None,
            important: false,
            separator: ":".to_string(),
            core_plugins: vec![
                "preflight".to_string(),
                "container".to_string(),
                "accessibility".to_string(),
                "variants".to_string(),
            ],
        }
    }
}

impl TailwindConfig {
    /// Create a new Tailwind configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Add content path
    pub fn add_content(mut self, path: impl Into<String>) -> Self {
        self.content.push(path.into());
        self
    }

    /// Set custom theme
    pub fn theme(mut self, theme: ThemeConfig) -> Self {
        self.theme = theme;
        self
    }

    /// Set prefix for utilities
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Enable important mode
    pub fn important(mut self, important: bool) -> Self {
        self.important = important;
        self
    }

    /// Export configuration to CSS
    pub fn to_css(&self) -> String {
        let mut css =
            String::from("@tailwind base;\n@tailwind components;\n@tailwind utilities;\n\n");

        // Add custom theme configuration as CSS custom properties
        css.push_str(":root {\n");
        for (key, value) in self.theme.colors.primary.iter() {
            css.push_str(&format!("  --color-primary-{}: {};\n", key, value));
        }
        css.push_str("}\n");

        css
    }
}

// ============================================================================
// Tailwind Utility Classes
// ============================================================================

/// Tailwind utility class builder
#[derive(Debug, Clone)]
pub struct TailwindUtility {
    /// Configuration reference
    config: TailwindConfig,
    /// Accumulated utility classes
    classes: HashSet<String>,
    /// Responsive variants
    responsive: HashMap<String, Vec<String>>,
    /// State variants
    states: HashMap<String, Vec<String>>,
}

impl TailwindUtility {
    /// Create new utility builder
    pub fn new(config: &TailwindConfig) -> Self {
        Self {
            config: config.clone(),
            classes: HashSet::new(),
            responsive: HashMap::new(),
            states: HashMap::new(),
        }
    }

    /// Add utility class
    pub fn add(mut self, class: impl Into<String>) -> Self {
        self.classes.insert(class.into());
        self
    }

    /// Add multiple utility classes
    pub fn add_all(mut self, classes: &[impl AsRef<str>]) -> Self {
        for class in classes {
            self.classes.insert(class.as_ref().to_string());
        }
        self
    }

    /// Add padding utility
    pub fn padding(self, class: impl Into<String>) -> Self {
        self.add(class.into())
    }

    /// Add margin utility
    pub fn margin(self, class: impl Into<String>) -> Self {
        self.add(class.into())
    }

    /// Add text utility
    pub fn text(self, class: impl Into<String>) -> Self {
        self.add(class.into())
    }

    /// Add background utility
    pub fn background(self, class: impl Into<String>) -> Self {
        self.add(class.into())
    }

    /// Add rounded utility
    pub fn rounded(self, class: impl Into<String>) -> Self {
        self.add(class.into())
    }

    /// Add shadow utility
    pub fn shadow(self, class: impl Into<String>) -> Self {
        self.add(class.into())
    }

    /// Add flex utility
    pub fn flex(self) -> Self {
        self.add("flex")
    }

    /// Add grid utility
    pub fn grid(self) -> Self {
        self.add("grid")
    }

    /// Add flex direction
    pub fn flex_direction(self, direction: impl Into<String>) -> Self {
        self.add(format!("flex-{}", direction.into()))
    }

    /// Add justify content
    pub fn justify(self, value: impl Into<String>) -> Self {
        self.add(format!("justify-{}", value.into()))
    }

    /// Add align items
    pub fn items(self, value: impl Into<String>) -> Self {
        self.add(format!("items-{}", value.into()))
    }

    /// Add gap utility
    pub fn gap(self, size: impl Into<String>) -> Self {
        self.add(format!("gap-{}", size.into()))
    }

    /// Add width utility
    pub fn width(self, size: impl Into<String>) -> Self {
        self.add(format!("w-{}", size.into()))
    }

    /// Add height utility
    pub fn height(self, size: impl Into<String>) -> Self {
        self.add(format!("h-{}", size.into()))
    }

    /// Add display utility
    pub fn display(self, value: impl Into<String>) -> Self {
        self.add(value.into())
    }

    /// Add position utility
    pub fn position(self, value: impl Into<String>) -> Self {
        self.add(value.into())
    }

    /// Add overflow utility
    pub fn overflow(self, value: impl Into<String>) -> Self {
        self.add(format!("overflow-{}", value.into()))
    }

    /// Add border utility
    pub fn border(self, class: impl Into<String>) -> Self {
        self.add(class.into())
    }

    /// Add cursor utility
    pub fn cursor(self, value: impl Into<String>) -> Self {
        self.add(format!("cursor-{}", value.into()))
    }

    /// Add transition utility
    pub fn transition(self, properties: impl Into<String>) -> Self {
        self.add(format!("transition-{}", properties.into()))
    }

    /// Add transform utility
    pub fn transform(self) -> Self {
        self.add("transform")
    }

    /// Add scale utility
    pub fn scale(self, value: impl Into<String>) -> Self {
        self.add(format!("scale-{}", value.into()))
    }

    /// Add rotate utility
    pub fn rotate(self, value: impl Into<String>) -> Self {
        self.add(format!("rotate-{}", value.into()))
    }

    /// Add translate utility
    pub fn translate(self, axis: impl Into<String>, value: impl Into<String>) -> Self {
        self.add(format!("translate-{}-{}", axis.into(), value.into()))
    }

    /// Add opacity utility
    pub fn opacity(self, value: impl Into<String>) -> Self {
        self.add(format!("opacity-{}", value.into()))
    }

    /// Add responsive variant
    pub fn responsive(mut self, breakpoint: impl Into<String>, classes: Vec<String>) -> Self {
        self.responsive.insert(breakpoint.into(), classes);
        self
    }

    /// Add hover state
    pub fn hover(mut self, classes: Vec<String>) -> Self {
        self.states.insert("hover".to_string(), classes);
        self
    }

    /// Add focus state
    pub fn focus(mut self, classes: Vec<String>) -> Self {
        self.states.insert("focus".to_string(), classes);
        self
    }

    /// Add active state
    pub fn active(mut self, classes: Vec<String>) -> Self {
        self.states.insert("active".to_string(), classes);
        self
    }

    /// Build final class string
    pub fn build(self) -> String {
        let mut all_classes = self.classes.into_iter().collect::<Vec<_>>();

        // Add responsive variants
        for (breakpoint, classes) in self.responsive {
            for class in classes {
                all_classes.push(format!("{}:{}", breakpoint, class));
            }
        }

        // Add state variants
        for (state, classes) in self.states {
            for class in classes {
                all_classes.push(format!("{}:{}", state, class));
            }
        }

        all_classes.sort();
        all_classes.dedup();
        all_classes.join(" ")
    }

    /// Generate CSS for accumulated utilities
    pub fn to_css(&self) -> String {
        let mut css = String::new();

        for class in &self.classes {
            css.push_str(&format!(".{} {{ }}\n", class));
        }

        css
    }
}

impl fmt::Display for TailwindUtility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.clone().build())
    }
}

// ============================================================================
// Design System
// ============================================================================

/// Design system with pre-configured themes and component styles
#[derive(Debug, Clone)]
pub struct DesignSystem {
    /// System name
    pub name: String,
    /// Color scheme
    pub colors: ColorScheme,
    /// Typography scale
    pub typography: TypographyScale,
    /// Spacing scale
    pub spacing: SpacingScheme,
    /// Component primitives
    pub components: ComponentPrimitives,
}

/// Color scheme
#[derive(Debug, Clone)]
pub struct ColorScheme {
    pub primary: String,
    pub secondary: String,
    pub accent: String,
    pub background: String,
    pub surface: String,
    pub text: String,
    pub text_secondary: String,
    pub border: String,
    pub success: String,
    pub warning: String,
    pub error: String,
}

/// Typography scale
#[derive(Debug, Clone)]
pub struct TypographyScale {
    pub font_family: String,
    pub font_family_mono: String,
    pub heading: HashMap<String, TextStyle>,
    pub body: HashMap<String, TextStyle>,
}

#[derive(Debug, Clone)]
pub struct TextStyle {
    pub font_size: String,
    pub font_weight: String,
    pub line_height: String,
    pub letter_spacing: Option<String>,
}

/// Spacing scheme
#[derive(Debug, Clone)]
pub struct SpacingScheme {
    pub unit: String,
    pub scale: Vec<String>,
}

impl DesignSystem {
    /// Create modern design system
    pub fn modern() -> Self {
        let colors = ColorScheme {
            primary: "#3b82f6".to_string(),
            secondary: "#6366f1".to_string(),
            accent: "#8b5cf6".to_string(),
            background: "#ffffff".to_string(),
            surface: "#f9fafb".to_string(),
            text: "#111827".to_string(),
            text_secondary: "#6b7280".to_string(),
            border: "#e5e7eb".to_string(),
            success: "#22c55e".to_string(),
            warning: "#f59e0b".to_string(),
            error: "#ef4444".to_string(),
        };

        let mut heading = HashMap::new();
        heading.insert(
            "h1".to_string(),
            TextStyle {
                font_size: "2.25rem".to_string(),
                font_weight: "700".to_string(),
                line_height: "2.5rem".to_string(),
                letter_spacing: Some("-0.025em".to_string()),
            },
        );
        heading.insert(
            "h2".to_string(),
            TextStyle {
                font_size: "1.875rem".to_string(),
                font_weight: "600".to_string(),
                line_height: "2.25rem".to_string(),
                letter_spacing: Some("-0.025em".to_string()),
            },
        );
        heading.insert(
            "h3".to_string(),
            TextStyle {
                font_size: "1.5rem".to_string(),
                font_weight: "600".to_string(),
                line_height: "2rem".to_string(),
                letter_spacing: None,
            },
        );
        heading.insert(
            "h4".to_string(),
            TextStyle {
                font_size: "1.25rem".to_string(),
                font_weight: "600".to_string(),
                line_height: "1.75rem".to_string(),
                letter_spacing: None,
            },
        );

        let mut body = HashMap::new();
        body.insert(
            "large".to_string(),
            TextStyle {
                font_size: "1.125rem".to_string(),
                font_weight: "400".to_string(),
                line_height: "1.75rem".to_string(),
                letter_spacing: None,
            },
        );
        body.insert(
            "base".to_string(),
            TextStyle {
                font_size: "1rem".to_string(),
                font_weight: "400".to_string(),
                line_height: "1.5rem".to_string(),
                letter_spacing: None,
            },
        );
        body.insert(
            "small".to_string(),
            TextStyle {
                font_size: "0.875rem".to_string(),
                font_weight: "400".to_string(),
                line_height: "1.25rem".to_string(),
                letter_spacing: None,
            },
        );

        let typography = TypographyScale {
            font_family: "Inter, system-ui, -apple-system, sans-serif".to_string(),
            font_family_mono: "'JetBrains Mono', 'Fira Code', monospace".to_string(),
            heading,
            body,
        };

        let spacing = SpacingScheme {
            unit: "0.25rem".to_string(),
            scale: vec![
                "0".to_string(),
                "1".to_string(),
                "2".to_string(),
                "3".to_string(),
                "4".to_string(),
                "5".to_string(),
                "6".to_string(),
                "8".to_string(),
                "10".to_string(),
                "12".to_string(),
                "16".to_string(),
                "20".to_string(),
                "24".to_string(),
                "32".to_string(),
                "40".to_string(),
                "48".to_string(),
            ],
        };

        Self {
            name: "modern".to_string(),
            colors,
            typography,
            spacing,
            components: ComponentPrimitives::default(),
        }
    }

    /// Create dark theme design system
    pub fn dark() -> Self {
        let mut system = Self::modern();
        system.name = "dark".to_string();
        system.colors.background = "#111827".to_string();
        system.colors.surface = "#1f2937".to_string();
        system.colors.text = "#f9fafb".to_string();
        system.colors.text_secondary = "#d1d5db".to_string();
        system.colors.border = "#374151".to_string();
        system
    }

    /// Get color value by name
    pub fn color(&self, name: &str) -> String {
        match name {
            "primary" => self.colors.primary.clone(),
            "secondary" => self.colors.secondary.clone(),
            "accent" => self.colors.accent.clone(),
            "background" => self.colors.background.clone(),
            "surface" => self.colors.surface.clone(),
            "text" => self.colors.text.clone(),
            "text-secondary" => self.colors.text_secondary.clone(),
            "border" => self.colors.border.clone(),
            "success" => self.colors.success.clone(),
            "warning" => self.colors.warning.clone(),
            "error" => self.colors.error.clone(),
            _ => self.colors.text.clone(),
        }
    }

    /// Get spacing value by index
    pub fn spacing(&self, index: usize) -> String {
        self.spacing
            .scale
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("{}rem", (index as f32 * 0.25)))
    }

    /// Generate CSS variables for the design system
    pub fn to_css_vars(&self) -> String {
        format!(
            ":root {{\n\
              --color-primary: {};\n\
              --color-secondary: {};\n\
              --color-accent: {};\n\
              --color-background: {};\n\
              --color-surface: {};\n\
              --color-text: {};\n\
              --color-text-secondary: {};\n\
              --color-border: {};\n\
              --color-success: {};\n\
              --color-warning: {};\n\
              --color-error: {};\n\
              --font-family: {};\n\
              --font-family-mono: {};\n\
            }}",
            self.colors.primary,
            self.colors.secondary,
            self.colors.accent,
            self.colors.background,
            self.colors.surface,
            self.colors.text,
            self.colors.text_secondary,
            self.colors.border,
            self.colors.success,
            self.colors.warning,
            self.colors.error,
            self.typography.font_family,
            self.typography.font_family_mono
        )
    }
}

// ============================================================================
// Component Primitives
// ============================================================================

/// Pre-built component style primitives
#[derive(Debug, Clone)]
pub struct ComponentPrimitives {
    /// Button styles
    pub button: ButtonStyles,
    /// Input styles
    pub input: InputStyles,
    /// Card styles
    pub card: CardStyles,
    /// Badge styles
    pub badge: BadgeStyles,
    /// Modal styles
    pub modal: ModalStyles,
    /// Dropdown styles
    pub dropdown: DropdownStyles,
    /// Navigation styles
    pub navigation: NavigationStyles,
}

/// Button component styles
#[derive(Debug, Clone)]
pub struct ButtonStyles {
    pub base: String,
    pub primary: String,
    pub secondary: String,
    pub danger: String,
    pub ghost: String,
    pub sizes: ButtonSizes,
}

#[derive(Debug, Clone)]
pub struct ButtonSizes {
    pub small: String,
    pub medium: String,
    pub large: String,
}

impl Default for ButtonStyles {
    fn default() -> Self {
        Self {
            base: "inline-flex items-center justify-center rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50".to_string(),
            primary: "bg-blue-600 text-white hover:bg-blue-700 focus-visible:ring-blue-600".to_string(),
            secondary: "bg-gray-200 text-gray-900 hover:bg-gray-300 focus-visible:ring-gray-400".to_string(),
            danger: "bg-red-600 text-white hover:bg-red-700 focus-visible:ring-red-600".to_string(),
            ghost: "hover:bg-gray-100 hover:text-gray-900".to_string(),
            sizes: ButtonSizes {
                small: "h-8 px-3 text-sm".to_string(),
                medium: "h-10 px-4 text-base".to_string(),
                large: "h-12 px-6 text-lg".to_string(),
            },
        }
    }
}

/// Input component styles
#[derive(Debug, Clone)]
pub struct InputStyles {
    pub base: String,
    pub variants: InputVariants,
    pub sizes: InputSizes,
}

#[derive(Debug, Clone)]
pub struct InputVariants {
    pub default: String,
    pub filled: String,
    pub outlined: String,
}

#[derive(Debug, Clone)]
pub struct InputSizes {
    pub small: String,
    pub medium: String,
    pub large: String,
}

impl Default for InputStyles {
    fn default() -> Self {
        Self {
            base: "flex w-full rounded-lg border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50".to_string(),
            variants: InputVariants {
                default: "border-gray-300 bg-white focus-visible:ring-blue-600 focus-visible:border-blue-600".to_string(),
                filled: "border-transparent bg-gray-100 focus-visible:bg-white focus-visible:ring-blue-600".to_string(),
                outlined: "border-gray-300 bg-transparent focus-visible:ring-blue-600 focus-visible:border-blue-600".to_string(),
            },
            sizes: InputSizes {
                small: "h-8 px-3 text-sm".to_string(),
                medium: "h-10 px-4 text-base".to_string(),
                large: "h-12 px-5 text-lg".to_string(),
            },
        }
    }
}

/// Card component styles
#[derive(Debug, Clone)]
pub struct CardStyles {
    pub base: String,
    pub header: String,
    pub body: String,
    pub footer: String,
    pub variants: CardVariants,
}

#[derive(Debug, Clone)]
pub struct CardVariants {
    pub elevated: String,
    pub outlined: String,
    pub flat: String,
}

impl Default for CardStyles {
    fn default() -> Self {
        Self {
            base: "rounded-lg border bg-white shadow-sm".to_string(),
            header: "flex flex-col space-y-1.5 p-6".to_string(),
            body: "p-6 pt-0".to_string(),
            footer: "flex items-center p-6 pt-0".to_string(),
            variants: CardVariants {
                elevated: "shadow-lg border-transparent".to_string(),
                outlined: "border-gray-200 shadow-none".to_string(),
                flat: "border-transparent shadow-none bg-gray-50".to_string(),
            },
        }
    }
}

/// Badge component styles
#[derive(Debug, Clone)]
pub struct BadgeStyles {
    pub base: String,
    pub variants: BadgeVariants,
    pub sizes: BadgeSizes,
}

#[derive(Debug, Clone)]
pub struct BadgeVariants {
    pub default: String,
    pub primary: String,
    pub secondary: String,
    pub success: String,
    pub warning: String,
    pub danger: String,
}

#[derive(Debug, Clone)]
pub struct BadgeSizes {
    pub small: String,
    pub medium: String,
    pub large: String,
}

impl Default for BadgeStyles {
    fn default() -> Self {
        Self {
            base: "inline-flex items-center rounded-full font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2".to_string(),
            variants: BadgeVariants {
                default: "bg-gray-100 text-gray-800".to_string(),
                primary: "bg-blue-100 text-blue-800".to_string(),
                secondary: "bg-purple-100 text-purple-800".to_string(),
                success: "bg-green-100 text-green-800".to_string(),
                warning: "bg-yellow-100 text-yellow-800".to_string(),
                danger: "bg-red-100 text-red-800".to_string(),
            },
            sizes: BadgeSizes {
                small: "px-2 py-0.5 text-xs".to_string(),
                medium: "px-2.5 py-0.5 text-sm".to_string(),
                large: "px-3 py-1 text-base".to_string(),
            },
        }
    }
}

/// Modal component styles
#[derive(Debug, Clone)]
pub struct ModalStyles {
    pub overlay: String,
    pub container: String,
    pub content: String,
    pub header: String,
    pub body: String,
    pub footer: String,
}

impl Default for ModalStyles {
    fn default() -> Self {
        Self {
            overlay: "fixed inset-0 z-50 bg-black/80 backdrop-blur-sm".to_string(),
            container: "fixed inset-0 z-50 flex items-center justify-center p-4".to_string(),
            content: "relative w-full max-w-lg rounded-lg border bg-white shadow-lg".to_string(),
            header: "flex items-center justify-between p-6 border-b".to_string(),
            body: "p-6".to_string(),
            footer: "flex items-center justify-end gap-2 p-6 border-t".to_string(),
        }
    }
}

/// Dropdown component styles
#[derive(Debug, Clone)]
pub struct DropdownStyles {
    pub trigger: String,
    pub content: String,
    pub item: String,
}

impl Default for DropdownStyles {
    fn default() -> Self {
        Self {
            trigger: "inline-flex items-center justify-center rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4".to_string(),
            content: "relative z-50 min-w-[8rem] overflow-hidden rounded-lg border bg-white shadow-lg p-1".to_string(),
            item: "relative flex cursor-pointer select-none items-center rounded-sm px-3 py-2 text-sm outline-none hover:bg-gray-100 focus:bg-gray-100 data-[disabled]:pointer-events-none data-[disabled]:opacity-50".to_string(),
        }
    }
}

/// Navigation component styles
#[derive(Debug, Clone)]
pub struct NavigationStyles {
    pub container: String,
    pub item: String,
    pub active: String,
    pub divider: String,
}

impl Default for NavigationStyles {
    fn default() -> Self {
        Self {
            container: "flex flex-col space-y-1 p-2".to_string(),
            item: "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all hover:bg-gray-100".to_string(),
            active: "bg-gray-100 text-gray-900".to_string(),
            divider: "my-2 border-t border-gray-200".to_string(),
        }
    }
}

impl Default for ComponentPrimitives {
    fn default() -> Self {
        Self {
            button: ButtonStyles::default(),
            input: InputStyles::default(),
            card: CardStyles::default(),
            badge: BadgeStyles::default(),
            modal: ModalStyles::default(),
            dropdown: DropdownStyles::default(),
            navigation: NavigationStyles::default(),
        }
    }
}

impl ComponentPrimitives {
    /// Get button classes
    pub fn button(&self, variant: &str, size: &str) -> String {
        let variant_classes = match variant {
            "primary" => &self.button.primary,
            "secondary" => &self.button.secondary,
            "danger" => &self.button.danger,
            "ghost" => &self.button.ghost,
            _ => &self.button.base,
        };

        let size_classes = match size {
            "small" => &self.button.sizes.small,
            "large" => &self.button.sizes.large,
            _ => &self.button.sizes.medium,
        };

        format!("{} {} {}", self.button.base, variant_classes, size_classes)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get input classes
    pub fn input(&self, variant: &str, size: &str) -> String {
        let variant_classes = match variant {
            "filled" => &self.input.variants.filled,
            "outlined" => &self.input.variants.outlined,
            _ => &self.input.variants.default,
        };

        let size_classes = match size {
            "small" => &self.input.sizes.small,
            "large" => &self.input.sizes.large,
            _ => &self.input.sizes.medium,
        };

        format!("{} {} {}", self.input.base, variant_classes, size_classes)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get card classes
    pub fn card(&self, variant: &str) -> String {
        let variant_classes = match variant {
            "elevated" => &self.card.variants.elevated,
            "outlined" => &self.card.variants.outlined,
            "flat" => &self.card.variants.flat,
            _ => "",
        };

        format!("{} {}", self.card.base, variant_classes)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get badge classes
    pub fn badge(&self, variant: &str, size: &str) -> String {
        let variant_classes = match variant {
            "primary" => &self.badge.variants.primary,
            "secondary" => &self.badge.variants.secondary,
            "success" => &self.badge.variants.success,
            "warning" => &self.badge.variants.warning,
            "danger" => &self.badge.variants.danger,
            _ => &self.badge.variants.default,
        };

        let size_classes = match size {
            "small" => &self.badge.sizes.small,
            "large" => &self.badge.sizes.large,
            _ => &self.badge.sizes.medium,
        };

        format!("{} {} {}", self.badge.base, variant_classes, size_classes)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ============================================================================
// Responsive Variants
// ============================================================================

/// Responsive utility builder
#[derive(Debug, Clone)]
pub struct ResponsiveBuilder {
    /// Base utilities (no breakpoint)
    pub base: Vec<String>,
    /// Small breakpoint (640px)
    pub sm: Vec<String>,
    /// Medium breakpoint (768px)
    pub md: Vec<String>,
    /// Large breakpoint (1024px)
    pub lg: Vec<String>,
    /// Extra large breakpoint (1280px)
    pub xl: Vec<String>,
    /// 2X large breakpoint (1536px)
    pub xl2: Vec<String>,
}

impl Default for ResponsiveBuilder {
    fn default() -> Self {
        Self {
            base: vec![],
            sm: vec![],
            md: vec![],
            lg: vec![],
            xl: vec![],
            xl2: vec![],
        }
    }
}

impl ResponsiveBuilder {
    /// Create new responsive builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add base utilities
    pub fn base(mut self, classes: &[&str]) -> Self {
        self.base.extend(classes.iter().map(|s| s.to_string()));
        self
    }

    /// Add small breakpoint utilities
    pub fn sm(mut self, classes: &[&str]) -> Self {
        self.sm.extend(classes.iter().map(|s| s.to_string()));
        self
    }

    /// Add medium breakpoint utilities
    pub fn md(mut self, classes: &[&str]) -> Self {
        self.md.extend(classes.iter().map(|s| s.to_string()));
        self
    }

    /// Add large breakpoint utilities
    pub fn lg(mut self, classes: &[&str]) -> Self {
        self.lg.extend(classes.iter().map(|s| s.to_string()));
        self
    }

    /// Add extra large breakpoint utilities
    pub fn xl(mut self, classes: &[&str]) -> Self {
        self.xl.extend(classes.iter().map(|s| s.to_string()));
        self
    }

    /// Add 2X large breakpoint utilities
    pub fn xl2(mut self, classes: &[&str]) -> Self {
        self.xl2.extend(classes.iter().map(|s| s.to_string()));
        self
    }

    /// Build responsive class string
    pub fn build(self) -> String {
        let mut all_classes = self.base;

        for class in self.sm {
            all_classes.push(format!("sm:{}", class));
        }
        for class in self.md {
            all_classes.push(format!("md:{}", class));
        }
        for class in self.lg {
            all_classes.push(format!("lg:{}", class));
        }
        for class in self.xl {
            all_classes.push(format!("xl:{}", class));
        }
        for class in self.xl2 {
            all_classes.push(format!("2xl:{}", class));
        }

        all_classes.join(" ")
    }
}

// ============================================================================
// Presets
// ============================================================================

/// Common layout presets
pub struct LayoutPresets;

impl LayoutPresets {
    /// Container preset
    pub fn container() -> String {
        "w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8".to_string()
    }

    /// Center content preset
    pub fn center() -> String {
        "flex items-center justify-center".to_string()
    }

    /// Stack preset (vertical layout)
    pub fn stack(gap: &str) -> String {
        format!("flex flex-col gap-{}", gap)
    }

    /// Row preset (horizontal layout)
    pub fn row(gap: &str) -> String {
        format!("flex flex-row gap-{}", gap)
    }

    /// Grid preset
    pub fn grid(columns: u32, gap: &str) -> String {
        format!("grid grid-cols-{} gap-{}", columns, gap)
    }
}

/// Common spacing presets
pub struct SpacingPresets;

impl SpacingPresets {
    /// No spacing
    pub fn none() -> String {
        "m-0 p-0".to_string()
    }

    /// Compact spacing
    pub fn compact() -> String {
        "p-2 gap-2".to_string()
    }

    /// Normal spacing
    pub fn normal() -> String {
        "p-4 gap-4".to_string()
    }

    /// Comfortable spacing
    pub fn comfortable() -> String {
        "p-6 gap-6".to_string()
    }

    /// Spacious spacing
    pub fn spacious() -> String {
        "p-8 gap-8".to_string()
    }
}

/// Common typography presets
pub struct TypographyPresets;

impl TypographyPresets {
    /// Heading styles
    pub fn heading_1() -> String {
        "text-4xl font-bold tracking-tight".to_string()
    }

    pub fn heading_2() -> String {
        "text-3xl font-bold tracking-tight".to_string()
    }

    pub fn heading_3() -> String {
        "text-2xl font-semibold tracking-tight".to_string()
    }

    pub fn heading_4() -> String {
        "text-xl font-semibold".to_string()
    }

    /// Body text styles
    pub fn body_large() -> String {
        "text-lg leading-relaxed".to_string()
    }

    pub fn body_base() -> String {
        "text-base leading-normal".to_string()
    }

    pub fn body_small() -> String {
        "text-sm leading-relaxed".to_string()
    }

    /// Text alignment
    pub fn text_center() -> String {
        "text-center".to_string()
    }

    pub fn text_justify() -> String {
        "text-justify".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tailwind_config_default() {
        let config = TailwindConfig::default();
        assert_eq!(config.content.len(), 2);
        assert_eq!(config.separator, ":");
        assert!(!config.important);
    }

    #[test]
    fn test_tailwind_config_builder() {
        let config = TailwindConfig::new()
            .add_content("./tests/**/*.rs")
            .prefix("tw-")
            .important(true);

        assert_eq!(config.content.len(), 3);
        assert_eq!(config.prefix, Some("tw-".to_string()));
        assert!(config.important);
    }

    #[test]
    fn test_tailwind_config_to_css() {
        let config = TailwindConfig::default();
        let css = config.to_css();

        assert!(css.contains("@tailwind base"));
        assert!(css.contains("@tailwind components"));
        assert!(css.contains("@tailwind utilities"));
    }

    #[test]
    fn test_tailwind_utility_builder() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config)
            .add("flex")
            .add("items-center")
            .add("justify-center")
            .padding("p-4")
            .margin("m-2")
            .background("bg-white")
            .rounded("rounded-lg")
            .shadow("shadow-md");

        let classes = utility.build();
        assert!(classes.contains("flex"));
        assert!(classes.contains("items-center"));
        assert!(classes.contains("p-4"));
        assert!(classes.contains("bg-white"));
    }

    #[test]
    fn test_tailwind_utility_display() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config)
            .flex()
            .justify("center")
            .items("center");

        let classes = utility.build();
        assert!(classes.contains("flex"));
        assert!(classes.contains("justify-center"));
        assert!(classes.contains("items-center"));
    }

    #[test]
    fn test_tailwind_utility_grid() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config).grid().gap("4");

        let classes = utility.build();
        assert!(classes.contains("grid"));
        assert!(classes.contains("gap-4"));
    }

    #[test]
    fn test_tailwind_utility_width_height() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config).width("full").height("screen");

        let classes = utility.build();
        assert!(classes.contains("w-full"));
        assert!(classes.contains("h-screen"));
    }

    #[test]
    fn test_tailwind_utility_transform() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config)
            .transform()
            .scale("100")
            .rotate("45");

        let classes = utility.build();
        assert!(classes.contains("transform"));
        assert!(classes.contains("scale-100"));
        assert!(classes.contains("rotate-45"));
    }

    #[test]
    fn test_tailwind_utility_opacity() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config).opacity("50");

        let classes = utility.build();
        assert!(classes.contains("opacity-50"));
    }

    #[test]
    fn test_tailwind_utility_responsive() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config)
            .responsive("sm", vec!["text-sm".to_string()])
            .responsive("md", vec!["text-base".to_string()])
            .responsive("lg", vec!["text-lg".to_string()]);

        let classes = utility.build();
        assert!(classes.contains("sm:text-sm"));
        assert!(classes.contains("md:text-base"));
        assert!(classes.contains("lg:text-lg"));
    }

    #[test]
    fn test_tailwind_utility_hover() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config).hover(vec!["bg-blue-600".to_string()]);

        let classes = utility.build();
        assert!(classes.contains("hover:bg-blue-600"));
    }

    #[test]
    fn test_tailwind_utility_focus() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config)
            .focus(vec!["ring-2".to_string(), "ring-blue-500".to_string()]);

        let classes = utility.build();
        assert!(classes.contains("focus:ring-2"));
        assert!(classes.contains("focus:ring-blue-500"));
    }

    #[test]
    fn test_design_system_modern() {
        let system = DesignSystem::modern();
        assert_eq!(system.name, "modern");
        assert_eq!(system.colors.primary, "#3b82f6");
        assert_eq!(system.colors.success, "#22c55e");
        assert_eq!(system.colors.error, "#ef4444");
    }

    #[test]
    fn test_design_system_dark() {
        let system = DesignSystem::dark();
        assert_eq!(system.name, "dark");
        assert_eq!(system.colors.background, "#111827");
        assert_eq!(system.colors.text, "#f9fafb");
    }

    #[test]
    fn test_design_system_color() {
        let system = DesignSystem::modern();
        assert_eq!(system.color("primary"), "#3b82f6");
        assert_eq!(system.color("error"), "#ef4444");
    }

    #[test]
    fn test_design_system_spacing() {
        let system = DesignSystem::modern();
        assert_eq!(system.spacing(0), "0");
        assert_eq!(system.spacing(4), "4");
    }

    #[test]
    fn test_design_system_to_css_vars() {
        let system = DesignSystem::modern();
        let css = system.to_css_vars();

        assert!(css.contains("--color-primary: #3b82f6"));
        assert!(css.contains("--color-error: #ef4444"));
        assert!(css.contains("--font-family:"));
    }

    #[test]
    fn test_component_primitives_button() {
        let components = ComponentPrimitives::default();
        let button = components.button("primary", "medium");

        assert!(button.contains("inline-flex"));
        assert!(button.contains("bg-blue-600"));
        assert!(button.contains("h-10"));
    }

    #[test]
    fn test_component_primitives_button_sizes() {
        let components = ComponentPrimitives::default();
        let small = components.button("primary", "small");
        let large = components.button("primary", "large");

        assert!(small.contains("h-8"));
        assert!(large.contains("h-12"));
    }

    #[test]
    fn test_component_primitives_input() {
        let components = ComponentPrimitives::default();
        let input = components.input("filled", "medium");

        assert!(input.contains("w-full"));
        assert!(input.contains("rounded-lg"));
        assert!(input.contains("h-10"));
    }

    #[test]
    fn test_component_primitives_card() {
        let components = ComponentPrimitives::default();
        let card = components.card("elevated");

        assert!(card.contains("rounded-lg"));
        assert!(card.contains("shadow-lg"));
    }

    #[test]
    fn test_component_primitives_badge() {
        let components = ComponentPrimitives::default();
        let badge = components.badge("success", "medium");

        assert!(badge.contains("inline-flex"));
        assert!(badge.contains("bg-green-100"));
    }

    #[test]
    fn test_responsive_builder() {
        let builder = ResponsiveBuilder::new()
            .base(&["flex"])
            .sm(&["flex-col"])
            .md(&["flex-row"])
            .lg(&["gap-4"]);

        let classes = builder.build();
        assert!(classes.contains("flex"));
        assert!(classes.contains("sm:flex-col"));
        assert!(classes.contains("md:flex-row"));
        assert!(classes.contains("lg:gap-4"));
    }

    #[test]
    fn test_layout_presets() {
        assert!(LayoutPresets::container().contains("max-w-7xl"));
        assert!(LayoutPresets::center().contains("flex"));
        assert!(LayoutPresets::stack("4").contains("flex-col"));
        assert!(LayoutPresets::row("4").contains("flex-row"));
        assert!(LayoutPresets::grid(3, "4").contains("grid-cols-3"));
    }

    #[test]
    fn test_spacing_presets() {
        assert_eq!(SpacingPresets::none(), "m-0 p-0");
        assert!(SpacingPresets::compact().contains("p-2"));
        assert!(SpacingPresets::normal().contains("p-4"));
        assert!(SpacingPresets::comfortable().contains("p-6"));
        assert!(SpacingPresets::spacious().contains("p-8"));
    }

    #[test]
    fn test_typography_presets() {
        assert!(TypographyPresets::heading_1().contains("text-4xl"));
        assert!(TypographyPresets::heading_2().contains("text-3xl"));
        assert!(TypographyPresets::body_large().contains("text-lg"));
        assert!(TypographyPresets::body_base().contains("text-base"));
        assert!(TypographyPresets::body_small().contains("text-sm"));
    }

    #[test]
    fn test_color_palette_default() {
        let palette = ColorPalette::default();
        assert_eq!(palette.primary.get("500"), Some(&"#3b82f6".to_string()));
        assert_eq!(palette.neutral.get("500"), Some(&"#6b7280".to_string()));
    }

    #[test]
    fn test_spacing_scale_default() {
        let scale = SpacingScale::default();
        assert_eq!(scale.scale.get("4"), Some(&"1rem".to_string()));
        assert_eq!(scale.scale.get("8"), Some(&"2rem".to_string()));
    }

    #[test]
    fn test_shadows_default() {
        let shadows = Shadows::default();
        assert!(shadows.scale.get("md").unwrap().contains("rgb(0 0 0"));
        assert!(shadows.scale.get("xl").unwrap().contains("0 20px"));
    }

    #[test]
    fn test_screens_default() {
        let screens = Screens::default();
        assert_eq!(screens.breakpoints.get("md"), Some(&"768px".to_string()));
        assert_eq!(screens.breakpoints.get("lg"), Some(&"1024px".to_string()));
    }

    #[test]
    fn test_transition_config_default() {
        let config = TransitionConfig::default();
        assert_eq!(config.duration.get("300"), Some(&"300ms".to_string()));
        assert!(config.timing.get("ease-in").is_some());
    }

    #[test]
    fn test_utility_builder_complex() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config)
            .flex()
            .flex_direction("col")
            .justify("between")
            .items("center")
            .padding("p-6")
            .background("bg-white")
            .rounded("rounded-xl")
            .shadow("shadow-lg")
            .border("border-gray-200")
            .hover(vec!["shadow-xl".to_string()])
            .responsive("md", vec!["flex-row".to_string()]);

        let classes = utility.build();
        assert!(classes.contains("flex"));
        assert!(classes.contains("flex-col"));
        assert!(classes.contains("p-6"));
        assert!(classes.contains("hover:shadow-xl"));
        assert!(classes.contains("md:flex-row"));
    }

    #[test]
    fn test_utility_add_all() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config).add_all(&[
            "flex",
            "items-center",
            "justify-center",
            "gap-4",
        ]);

        let classes = utility.build();
        assert!(classes.contains("flex"));
        assert!(classes.contains("gap-4"));
    }

    #[test]
    fn test_utility_cursor() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config).cursor("pointer");

        let classes = utility.build();
        assert!(classes.contains("cursor-pointer"));
    }

    #[test]
    fn test_utility_overflow() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config).overflow("hidden");

        let classes = utility.build();
        assert!(classes.contains("overflow-hidden"));
    }

    #[test]
    fn test_utility_position() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config)
            .position("relative")
            .position("absolute");

        let classes = utility.build();
        assert!(classes.contains("relative"));
        assert!(classes.contains("absolute"));
    }

    #[test]
    fn test_utility_transition() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config).transition("all");

        let classes = utility.build();
        assert!(classes.contains("transition-all"));
    }

    #[test]
    fn test_utility_translate() {
        let config = TailwindConfig::default();
        let utility = TailwindUtility::new(&config)
            .translate("x", "4")
            .translate("y", "-2");

        let classes = utility.build();
        assert!(classes.contains("translate-x-4"));
        assert!(classes.contains("translate-y--2"));
    }
}
