//! Responsive UI System for nCPU/nSynth
//!
//! Complete responsive design framework with:
//! - ResponsiveBreakpoints: Mobile-first and desktop-first strategies
//! - DarkMode: Theme switching and color management
//! - AnimationSystem: CSS animations and transitions
//! - GridSystem: CSS Grid layout primitives

use std::collections::HashMap;

/// Responsive design breakpoints
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Breakpoint {
    /// Extra small devices (phones, < 576px)
    XS = 0,
    /// Small devices (landscape phones, >= 576px)
    SM = 576,
    /// Medium devices (tablets, >= 768px)
    MD = 768,
    /// Large devices (desktops, >= 992px)
    LG = 992,
    /// Extra large devices (large desktops, >= 1200px)
    XL = 1200,
    /// Extra extra large devices (>= 1400px)
    XXL = 1400,
}

impl Breakpoint {
    /// Get breakpoint value in pixels
    pub fn px(&self) -> u32 {
        *self as u32
    }

    /// Get min-width media query
    pub fn min_width(&self) -> String {
        match self {
            Breakpoint::XS => String::new(), // No min-width for XS
            _ => format!("(min-width: {}px)", self.px()),
        }
    }

    /// Get max-width media query
    pub fn max_width(&self) -> String {
        match self {
            Breakpoint::XS => "(max-width: 575px)".to_string(),
            Breakpoint::SM => "(max-width: 767px)".to_string(),
            Breakpoint::MD => "(max-width: 991px)".to_string(),
            Breakpoint::LG => "(max-width: 1199px)".to_string(),
            Breakpoint::XL => "(max-width: 1399px)".to_string(),
            Breakpoint::XXL => String::new(), // No max-width for XXL
        }
    }

    /// Get range media query (between breakpoints)
    pub fn range(&self, next: Breakpoint) -> String {
        if *self >= next {
            return String::new();
        }
        format!(
            "(min-width: {}px) and (max-width: {}px)",
            self.px(),
            next.px() - 1
        )
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "xs" => Some(Breakpoint::XS),
            "sm" => Some(Breakpoint::SM),
            "md" => Some(Breakpoint::MD),
            "lg" => Some(Breakpoint::LG),
            "xl" => Some(Breakpoint::XL),
            "xxl" => Some(Breakpoint::XXL),
            _ => None,
        }
    }

    /// Get all breakpoints in order
    pub fn all() -> Vec<Breakpoint> {
        vec![
            Breakpoint::XS,
            Breakpoint::SM,
            Breakpoint::MD,
            Breakpoint::LG,
            Breakpoint::XL,
            Breakpoint::XXL,
        ]
    }
}

/// Responsive design strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsiveStrategy {
    /// Mobile-first: start with XS and add breakpoints
    MobileFirst,
    /// Desktop-first: start with XXL and add breakpoints
    DesktopFirst,
}

impl ResponsiveStrategy {
    /// Get breakpoints in strategy order
    pub fn breakpoints(&self) -> Vec<Breakpoint> {
        match self {
            ResponsiveStrategy::MobileFirst => Breakpoint::all(),
            ResponsiveStrategy::DesktopFirst => {
                let mut all = Breakpoint::all();
                all.reverse();
                all
            }
        }
    }
}

/// Responsive breakpoints configuration
#[derive(Debug, Clone)]
pub struct ResponsiveBreakpoints {
    /// Strategy (mobile-first or desktop-first)
    pub strategy: ResponsiveStrategy,
    /// Custom breakpoints (name -> px)
    pub custom: HashMap<String, u32>,
    /// Container widths per breakpoint
    pub container_widths: HashMap<Breakpoint, u32>,
    /// Gutter width
    pub gutter_width: u32,
    /// Number of columns
    pub columns: u32,
}

impl Default for ResponsiveBreakpoints {
    fn default() -> Self {
        let mut container_widths = HashMap::new();
        container_widths.insert(Breakpoint::XS, 100);
        container_widths.insert(Breakpoint::SM, 540);
        container_widths.insert(Breakpoint::MD, 720);
        container_widths.insert(Breakpoint::LG, 960);
        container_widths.insert(Breakpoint::XL, 1140);
        container_widths.insert(Breakpoint::XXL, 1320);

        Self {
            strategy: ResponsiveStrategy::MobileFirst,
            custom: HashMap::new(),
            container_widths,
            gutter_width: 30,
            columns: 12,
        }
    }
}

impl ResponsiveBreakpoints {
    /// Create new responsive breakpoints
    pub fn new() -> Self {
        Self::default()
    }

    /// Set strategy
    pub fn with_strategy(mut self, strategy: ResponsiveStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add custom breakpoint
    pub fn add_custom(mut self, name: impl Into<String>, px: u32) -> Self {
        self.custom.insert(name.into(), px);
        self
    }

    /// Set container width for breakpoint
    pub fn container_width(mut self, breakpoint: Breakpoint, width: u32) -> Self {
        self.container_widths.insert(breakpoint, width);
        self
    }

    /// Set gutter width
    pub fn gutter(mut self, width: u32) -> Self {
        self.gutter_width = width;
        self
    }

    /// Set number of columns
    pub fn columns(mut self, cols: u32) -> Self {
        self.columns = cols;
        self
    }

    /// Generate media query CSS
    pub fn media_query_css(&self) -> String {
        let mut css = String::new();

        for bp in self.strategy.breakpoints() {
            let min_width = bp.min_width();
            if !min_width.is_empty() {
                css.push_str(&format!("@media {} {{\n", min_width));
            }

            if let Some(&width) = self.container_widths.get(&bp) {
                css.push_str(&format!("  .container {{ max-width: {}px; }}\n", width));
            }

            if !min_width.is_empty() {
                css.push_str("}\n");
            }
        }

        // Custom breakpoints
        for (name, &px) in &self.custom {
            css.push_str(&format!("@media (min-width: {}px) {{\n", px));
            css.push_str(&format!("  /* Custom breakpoint: {} */\n", name));
            css.push_str("}\n");
        }

        css
    }

    /// Generate container CSS
    pub fn container_css(&self) -> String {
        format!(
            ".container {{
    width: 100%;
    margin-right: auto;
    margin-left: auto;
    padding-right: calc({} / 2);
    padding-left: calc({} / 2);
}}",
            self.gutter_width, self.gutter_width
        )
    }

    /// Generate grid CSS
    pub fn grid_css(&self) -> String {
        format!(
            ".row {{
    display: flex;
    flex-wrap: wrap;
    margin-right: calc({} / -2);
    margin-left: calc({} / -2);
}}

.col {{
    flex-basis: 0;
    flex-grow: 1;
    max-width: 100%;
    padding-right: calc({} / 2);
    padding-left: calc({} / 2);
}}",
            self.gutter_width, self.gutter_width, self.gutter_width, self.gutter_width
        )
    }
}

/// Theme mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeMode {
    /// Light theme
    Light,
    /// Dark theme
    Dark,
    /// Auto (system preference)
    Auto,
}

impl ThemeMode {
    /// Get media query for auto theme
    pub fn prefers_color_scheme() -> String {
        "@media (prefers-color-scheme: dark)".to_string()
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "light" => Some(ThemeMode::Light),
            "dark" => Some(ThemeMode::Dark),
            "auto" => Some(ThemeMode::Auto),
            _ => None,
        }
    }
}

/// Color definition
#[derive(Debug, Clone)]
pub struct ThemeColor {
    /// Light mode value
    pub light: String,
    /// Dark mode value
    pub dark: String,
}

impl ThemeColor {
    /// Create new theme color
    pub fn new(light: impl Into<String>, dark: impl Into<String>) -> Self {
        Self {
            light: light.into(),
            dark: dark.into(),
        }
    }

    /// Create single color (same for both modes)
    pub fn single(color: impl Into<String>) -> Self {
        let color = color.into();
        Self {
            light: color.clone(),
            dark: color,
        }
    }
}

/// Dark mode configuration
#[derive(Debug, Clone)]
pub struct DarkMode {
    /// Current theme mode
    pub mode: ThemeMode,
    /// Theme colors
    pub colors: HashMap<String, ThemeColor>,
    /// Use CSS custom properties
    pub use_custom_properties: bool,
    /// Transition duration
    pub transition_duration: String,
}

impl Default for DarkMode {
    fn default() -> Self {
        Self {
            mode: ThemeMode::Auto,
            colors: Self::default_colors(),
            use_custom_properties: true,
            transition_duration: "0.3s".to_string(),
        }
    }
}

impl DarkMode {
    /// Create new dark mode config
    pub fn new() -> Self {
        Self::default()
    }

    /// Get default color palette
    fn default_colors() -> HashMap<String, ThemeColor> {
        let mut colors = HashMap::new();

        // Background colors
        colors.insert(
            "bg-primary".to_string(),
            ThemeColor::new("#ffffff", "#121212"),
        );
        colors.insert(
            "bg-secondary".to_string(),
            ThemeColor::new("#f8f9fa", "#1e1e1e"),
        );
        colors.insert(
            "bg-tertiary".to_string(),
            ThemeColor::new("#e9ecef", "#2d2d2d"),
        );

        // Text colors
        colors.insert(
            "text-primary".to_string(),
            ThemeColor::new("#212529", "#ffffff"),
        );
        colors.insert(
            "text-secondary".to_string(),
            ThemeColor::new("#6c757d", "#b0b0b0"),
        );
        colors.insert(
            "text-tertiary".to_string(),
            ThemeColor::new("#adb5bd", "#808080"),
        );

        // Accent colors (work in both modes)
        colors.insert("accent".to_string(), ThemeColor::single("#007bff"));
        colors.insert("accent-hover".to_string(), ThemeColor::single("#0056b3"));

        // Border colors
        colors.insert("border".to_string(), ThemeColor::new("#dee2e6", "#404040"));

        // Error, warning, success colors
        colors.insert("error".to_string(), ThemeColor::single("#dc3545"));
        colors.insert("warning".to_string(), ThemeColor::single("#ffc107"));
        colors.insert("success".to_string(), ThemeColor::single("#28a745"));

        colors
    }

    /// Set theme mode
    pub fn with_mode(mut self, mode: ThemeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Add color
    pub fn add_color(mut self, name: impl Into<String>, color: ThemeColor) -> Self {
        self.colors.insert(name.into(), color);
        self
    }

    /// Set transition duration
    pub fn transition(mut self, duration: impl Into<String>) -> Self {
        self.transition_duration = duration.into();
        self
    }

    /// Generate CSS variables
    pub fn css_variables(&self) -> String {
        let mut css = String::from(":root {\n");

        for (name, color) in &self.colors {
            css.push_str(&format!("  --color-{}: {};\n", name, color.light));
        }

        css.push_str(&format!(
            "  --theme-transition: color {} ease, background-color {} ease;\n",
            self.transition_duration, self.transition_duration
        ));
        css.push_str("}\n\n");

        // Dark mode
        css.push_str("@media (prefers-color-scheme: dark) {\n");
        css.push_str(":root {\n");

        for (name, color) in &self.colors {
            css.push_str(&format!("  --color-{}: {};\n", name, color.dark));
        }

        css.push_str("}\n");
        css.push_str("}\n\n");

        // Manual dark mode class
        css.push_str(".dark {\n");
        for (name, color) in &self.colors {
            css.push_str(&format!("  --color-{}: {};\n", name, color.dark));
        }
        css.push_str("}\n");

        css
    }

    /// Generate utility CSS classes
    pub fn utility_classes(&self) -> String {
        let mut css = String::new();

        // Background utilities
        for name in self.colors.keys() {
            css.push_str(&format!(
                ".bg-{{}} {{ background-color: var(--color-{}); }}\n",
                name
            ));
        }

        // Text utilities
        for name in self.colors.keys() {
            css.push_str(&format!(".text-{{}} {{ color: var(--color-{}); }}\n", name));
        }

        // Border utilities
        css.push_str(&format!(
            ".border {{ border-color: var(--color-border); }}\n"
        ));

        // Theme transition
        css.push_str(&format!(
            ".theme-transition {{\n  transition: var(--theme-transition);\n}}\n"
        ));

        css
    }

    /// Generate JavaScript for theme switching
    pub fn js_switcher(&self) -> String {
        format!(
            r#"// Theme switching
(function() {{
    const theme = localStorage.getItem('theme') || 'auto';
    document.documentElement.classList.toggle('dark', theme === 'dark' || (theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches));
}})();

function setTheme(mode) {{
    localStorage.setItem('theme', mode);
    document.documentElement.classList.toggle('dark', mode === 'dark' || (mode === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches));
}}
"#
        )
    }
}

/// Animation easing function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EasingFunction {
    Linear,
    Ease,
    EaseIn,
    EaseOut,
    EaseInOut,
    EaseInQuad,
    EaseOutQuad,
    EaseInOutQuad,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    Spring,
    Bounce,
}

impl EasingFunction {
    /// Get CSS timing function
    pub fn to_css(&self) -> &str {
        match self {
            EasingFunction::Linear => "linear",
            EasingFunction::Ease => "ease",
            EasingFunction::EaseIn => "ease-in",
            EasingFunction::EaseOut => "ease-out",
            EasingFunction::EaseInOut => "ease-in-out",
            EasingFunction::EaseInQuad => "cubic-bezier(0.55, 0.085, 0.68, 0.53)",
            EasingFunction::EaseOutQuad => "cubic-bezier(0.25, 0.46, 0.45, 0.94)",
            EasingFunction::EaseInOutQuad => "cubic-bezier(0.455, 0.03, 0.515, 0.955)",
            EasingFunction::EaseInCubic => "cubic-bezier(0.55, 0.055, 0.675, 0.19)",
            EasingFunction::EaseOutCubic => "cubic-bezier(0.215, 0.61, 0.355, 1)",
            EasingFunction::EaseInOutCubic => "cubic-bezier(0.645, 0.045, 0.355, 1)",
            EasingFunction::Spring => "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
            EasingFunction::Bounce => "cubic-bezier(0.68, -0.55, 0.265, 1.55)",
        }
    }
}

/// Animation definition
#[derive(Debug, Clone)]
pub struct Animation {
    /// Animation name
    pub name: String,
    /// Duration (e.g., "300ms", "0.3s")
    pub duration: String,
    /// Timing function
    pub easing: EasingFunction,
    /// Delay
    pub delay: String,
    /// Iteration count
    pub iteration_count: String,
    /// Direction
    pub direction: String,
    /// Fill mode
    pub fill_mode: String,
    /// Keyframes
    pub keyframes: Vec<Keyframe>,
}

/// Animation keyframe
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Percentage (0-100) or "from" / "to"
    pub selector: String,
    /// CSS properties
    pub properties: HashMap<String, String>,
}

impl Keyframe {
    /// Create new keyframe
    pub fn new(selector: impl Into<String>) -> Self {
        Self {
            selector: selector.into(),
            properties: HashMap::new(),
        }
    }

    /// Add property
    pub fn add(mut self, property: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(property.into(), value.into());
        self
    }
}

impl Animation {
    /// Create new animation
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            duration: "300ms".to_string(),
            easing: EasingFunction::Ease,
            delay: "0s".to_string(),
            iteration_count: "1".to_string(),
            direction: "normal".to_string(),
            fill_mode: "both".to_string(),
            keyframes: Vec::new(),
        }
    }

    /// Set duration
    pub fn duration(mut self, duration: impl Into<String>) -> Self {
        self.duration = duration.into();
        self
    }

    /// Set easing
    pub fn easing(mut self, easing: EasingFunction) -> Self {
        self.easing = easing;
        self
    }

    /// Set delay
    pub fn delay(mut self, delay: impl Into<String>) -> Self {
        self.delay = delay.into();
        self
    }

    /// Set iteration count
    pub fn iteration_count(mut self, count: impl Into<String>) -> Self {
        self.iteration_count = count.into();
        self
    }

    /// Set infinite iteration
    pub fn infinite(mut self) -> Self {
        self.iteration_count = "infinite".to_string();
        self
    }

    /// Set direction
    pub fn direction(mut self, direction: impl Into<String>) -> Self {
        self.direction = direction.into();
        self
    }

    /// Set fill mode
    pub fn fill_mode(mut self, mode: impl Into<String>) -> Self {
        self.fill_mode = mode.into();
        self
    }

    /// Add keyframe
    pub fn keyframe(mut self, keyframe: Keyframe) -> Self {
        self.keyframes.push(keyframe);
        self
    }

    /// Generate keyframes CSS
    pub fn keyframes_css(&self) -> String {
        let mut css = format!("@keyframes {} {{\n", self.name);

        for kf in &self.keyframes {
            css.push_str(&format!("  {} {{\n", kf.selector));
            for (prop, value) in &kf.properties {
                css.push_str(&format!("    {}: {};\n", prop, value));
            }
            css.push_str("  }\n");
        }

        css.push_str("}\n");
        css
    }

    /// Generate animation shorthand CSS
    pub fn to_css(&self) -> String {
        format!(
            "{} {} {} {} {}",
            self.name,
            self.duration,
            self.easing.to_css(),
            self.delay,
            self.iteration_count,
        )
    }

    /// Generate utility class
    pub fn utility_class(&self) -> String {
        format!(".animate-{} {{ animation: {}; }}", self.name, self.to_css())
    }
}

/// Animation system
#[derive(Debug, Clone)]
pub struct AnimationSystem {
    /// Animations
    pub animations: HashMap<String, Animation>,
    /// Default duration
    pub default_duration: String,
    /// Default easing
    pub default_easing: EasingFunction,
}

impl Default for AnimationSystem {
    fn default() -> Self {
        let mut animations = HashMap::new();

        // Fade in
        animations.insert(
            "fade-in".to_string(),
            Animation {
                name: "fade-in".to_string(),
                duration: "300ms".to_string(),
                easing: EasingFunction::EaseOut,
                delay: "0s".to_string(),
                iteration_count: "1".to_string(),
                direction: "normal".to_string(),
                fill_mode: "both".to_string(),
                keyframes: vec![
                    Keyframe::new("from").add("opacity", "0"),
                    Keyframe::new("to").add("opacity", "1"),
                ],
            },
        );

        // Fade out
        animations.insert(
            "fade-out".to_string(),
            Animation {
                name: "fade-out".to_string(),
                duration: "300ms".to_string(),
                easing: EasingFunction::EaseIn,
                delay: "0s".to_string(),
                iteration_count: "1".to_string(),
                direction: "normal".to_string(),
                fill_mode: "both".to_string(),
                keyframes: vec![
                    Keyframe::new("from").add("opacity", "1"),
                    Keyframe::new("to").add("opacity", "0"),
                ],
            },
        );

        // Slide up
        animations.insert(
            "slide-up".to_string(),
            Animation {
                name: "slide-up".to_string(),
                duration: "300ms".to_string(),
                easing: EasingFunction::EaseOut,
                delay: "0s".to_string(),
                iteration_count: "1".to_string(),
                direction: "normal".to_string(),
                fill_mode: "both".to_string(),
                keyframes: vec![
                    Keyframe::new("from")
                        .add("transform", "translateY(20px)")
                        .add("opacity", "0"),
                    Keyframe::new("to")
                        .add("transform", "translateY(0)")
                        .add("opacity", "1"),
                ],
            },
        );

        // Slide down
        animations.insert(
            "slide-down".to_string(),
            Animation {
                name: "slide-down".to_string(),
                duration: "300ms".to_string(),
                easing: EasingFunction::EaseOut,
                delay: "0s".to_string(),
                iteration_count: "1".to_string(),
                direction: "normal".to_string(),
                fill_mode: "both".to_string(),
                keyframes: vec![
                    Keyframe::new("from")
                        .add("transform", "translateY(-20px)")
                        .add("opacity", "0"),
                    Keyframe::new("to")
                        .add("transform", "translateY(0)")
                        .add("opacity", "1"),
                ],
            },
        );

        // Scale in
        animations.insert(
            "scale-in".to_string(),
            Animation {
                name: "scale-in".to_string(),
                duration: "300ms".to_string(),
                easing: EasingFunction::Spring,
                delay: "0s".to_string(),
                iteration_count: "1".to_string(),
                direction: "normal".to_string(),
                fill_mode: "both".to_string(),
                keyframes: vec![
                    Keyframe::new("from")
                        .add("transform", "scale(0.9)")
                        .add("opacity", "0"),
                    Keyframe::new("to")
                        .add("transform", "scale(1)")
                        .add("opacity", "1"),
                ],
            },
        );

        // Bounce
        animations.insert(
            "bounce".to_string(),
            Animation {
                name: "bounce".to_string(),
                duration: "500ms".to_string(),
                easing: EasingFunction::Bounce,
                delay: "0s".to_string(),
                iteration_count: "1".to_string(),
                direction: "normal".to_string(),
                fill_mode: "both".to_string(),
                keyframes: vec![
                    Keyframe::new("0%, 100%").add("transform", "translateY(0)"),
                    Keyframe::new("50%").add("transform", "translateY(-25%)"),
                ],
            },
        );

        // Pulse
        animations.insert(
            "pulse".to_string(),
            Animation {
                name: "pulse".to_string(),
                duration: "2s".to_string(),
                easing: EasingFunction::EaseInOut,
                delay: "0s".to_string(),
                iteration_count: "infinite".to_string(),
                direction: "normal".to_string(),
                fill_mode: "both".to_string(),
                keyframes: vec![
                    Keyframe::new("0%, 100%").add("opacity", "1"),
                    Keyframe::new("50%").add("opacity", "0.5"),
                ],
            },
        );

        // Spin
        animations.insert(
            "spin".to_string(),
            Animation {
                name: "spin".to_string(),
                duration: "1s".to_string(),
                easing: EasingFunction::Linear,
                delay: "0s".to_string(),
                iteration_count: "infinite".to_string(),
                direction: "normal".to_string(),
                fill_mode: "both".to_string(),
                keyframes: vec![
                    Keyframe::new("from").add("transform", "rotate(0deg)"),
                    Keyframe::new("to").add("transform", "rotate(360deg)"),
                ],
            },
        );

        Self {
            animations,
            default_duration: "300ms".to_string(),
            default_easing: EasingFunction::Ease,
        }
    }
}

impl AnimationSystem {
    /// Create new animation system
    pub fn new() -> Self {
        Self::default()
    }

    /// Add animation
    pub fn add(mut self, name: impl Into<String>, animation: Animation) -> Self {
        self.animations.insert(name.into(), animation);
        self
    }

    /// Set default duration
    pub fn default_duration(mut self, duration: impl Into<String>) -> Self {
        self.default_duration = duration.into();
        self
    }

    /// Set default easing
    pub fn default_easing(mut self, easing: EasingFunction) -> Self {
        self.default_easing = easing;
        self
    }

    /// Generate all keyframes CSS
    pub fn keyframes_css(&self) -> String {
        self.animations
            .values()
            .map(|a| a.keyframes_css())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Generate utility classes
    pub fn utility_classes(&self) -> String {
        self.animations
            .values()
            .map(|a| a.utility_class())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get animation by name
    pub fn get(&self, name: &str) -> Option<&Animation> {
        self.animations.get(name)
    }
}

/// Grid track size
#[derive(Debug, Clone, PartialEq)]
pub enum TrackSize {
    /// Fixed size (px, em, etc.)
    Fixed(String),
    /// Fraction of available space
    Fr(f32),
    /// Minmax constraint
    Minmax(Box<TrackSize>, Box<TrackSize>),
    /// Auto sizing
    Auto,
    /// Content-based sizing
    MinContent,
    MaxContent,
    /// Fit-content
    FitContent(String),
}

impl TrackSize {
    /// Convert to CSS
    pub fn to_css(&self) -> String {
        match self {
            TrackSize::Fixed(s) => s.clone(),
            TrackSize::Fr(n) => format!("{}fr", n),
            TrackSize::Minmax(min, max) => format!("minmax({}, {})", min.to_css(), max.to_css()),
            TrackSize::Auto => "auto".to_string(),
            TrackSize::MinContent => "min-content".to_string(),
            TrackSize::MaxContent => "max-content".to_string(),
            TrackSize::FitContent(s) => format!("fit-content({})", s),
        }
    }

    /// Create fixed pixel size
    pub fn px(px: u32) -> Self {
        TrackSize::Fixed(format!("{}px", px))
    }

    /// Create percentage size
    pub fn percent(pct: u32) -> Self {
        TrackSize::Fixed(format!("{}%", pct))
    }

    /// Create fraction size
    pub fn fr(n: f32) -> Self {
        TrackSize::Fr(n)
    }

    /// Create minmax
    pub fn minmax(min: TrackSize, max: TrackSize) -> Self {
        TrackSize::Minmax(Box::new(min), Box::new(max))
    }
}

/// Grid placement
#[derive(Debug, Clone, PartialEq)]
pub enum GridPlacement {
    /// Auto placement
    Auto,
    /// Specific line number
    Line(i32),
    /// Span count
    Span(u32),
    /// Named line
    Named(String),
}

impl GridPlacement {
    /// Convert to CSS
    pub fn to_css(&self) -> String {
        match self {
            GridPlacement::Auto => "auto".to_string(),
            GridPlacement::Line(n) => n.to_string(),
            GridPlacement::Span(n) => format!("span {}", n),
            GridPlacement::Named(s) => s.clone(),
        }
    }
}

/// Grid area definition
#[derive(Debug, Clone)]
pub struct GridArea {
    /// Area name
    pub name: String,
    /// Column start
    pub column_start: GridPlacement,
    /// Column end
    pub column_end: GridPlacement,
    /// Row start
    pub row_start: GridPlacement,
    /// Row end
    pub row_end: GridPlacement,
}

impl GridArea {
    /// Create new grid area
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            column_start: GridPlacement::Auto,
            column_end: GridPlacement::Auto,
            row_start: GridPlacement::Auto,
            row_end: GridPlacement::Auto,
        }
    }

    /// Set column placement
    pub fn column(mut self, start: GridPlacement, end: GridPlacement) -> Self {
        self.column_start = start;
        self.column_end = end;
        self
    }

    /// Set row placement
    pub fn row(mut self, start: GridPlacement, end: GridPlacement) -> Self {
        self.row_start = start;
        self.row_end = end;
        self
    }

    /// Generate area CSS for template
    pub fn to_area_css(&self) -> String {
        format!(
            "{}: {} / {} / {} / {}",
            self.name,
            self.row_start.to_css(),
            self.column_start.to_css(),
            self.row_end.to_css(),
            self.column_end.to_css()
        )
    }
}

/// CSS Grid system
#[derive(Debug, Clone)]
pub struct GridSystem {
    /// Column tracks
    pub columns: Vec<TrackSize>,
    /// Row tracks
    pub rows: Vec<TrackSize>,
    /// Gap between columns
    pub column_gap: Option<String>,
    /// Gap between rows
    pub row_gap: Option<String>,
    /// Grid areas
    pub areas: Vec<GridArea>,
    /// Auto-fit configuration
    pub auto_fit: bool,
    /// Auto-fill configuration
    pub auto_fill: bool,
    /// Minimum track size for auto-fit/fill
    pub min_track_size: Option<TrackSize>,
}

impl Default for GridSystem {
    fn default() -> Self {
        Self {
            columns: vec![TrackSize::fr(1.0)],
            rows: vec![TrackSize::Auto],
            column_gap: None,
            row_gap: None,
            areas: Vec::new(),
            auto_fit: false,
            auto_fill: false,
            min_track_size: None,
        }
    }
}

impl GridSystem {
    /// Create new grid system
    pub fn new() -> Self {
        Self::default()
    }

    /// Create fixed column grid
    pub fn fixed_columns(cols: u32) -> Self {
        Self {
            columns: vec![TrackSize::fr(1.0); cols as usize],
            ..Default::default()
        }
    }

    /// Create responsive grid with auto-fit
    pub fn responsive(min_width: u32, max_width: Option<u32>) -> Self {
        Self {
            columns: vec![TrackSize::Minmax(
                Box::new(TrackSize::px(min_width)),
                Box::new(max_width.map(TrackSize::px).unwrap_or(TrackSize::Fr(1.0))),
            )],
            auto_fit: true,
            min_track_size: Some(TrackSize::px(min_width)),
            ..Default::default()
        }
    }

    /// Set columns
    pub fn columns(mut self, columns: Vec<TrackSize>) -> Self {
        self.columns = columns;
        self
    }

    /// Set rows
    pub fn rows(mut self, rows: Vec<TrackSize>) -> Self {
        self.rows = rows;
        self
    }

    /// Set column gap
    pub fn column_gap(mut self, gap: impl Into<String>) -> Self {
        self.column_gap = Some(gap.into());
        self
    }

    /// Set row gap
    pub fn row_gap(mut self, gap: impl Into<String>) -> Self {
        self.row_gap = Some(gap.into());
        self
    }

    /// Set gap (both)
    pub fn gap(mut self, gap: impl Into<String>) -> Self {
        let gap = gap.into();
        self.column_gap = Some(gap.clone());
        self.row_gap = Some(gap);
        self
    }

    /// Add grid area
    pub fn area(mut self, area: GridArea) -> Self {
        self.areas.push(area);
        self
    }

    /// Generate grid template CSS
    pub fn grid_template_css(&self) -> String {
        let mut css = String::new();

        // Grid template columns
        if self.auto_fit || self.auto_fill {
            let repeat_fn = if self.auto_fit {
                "auto-fit"
            } else {
                "auto-fill"
            };
            if let Some(min_size) = &self.min_track_size {
                let max_size = self
                    .columns
                    .first()
                    .map(|c| {
                        if let TrackSize::Minmax(_, max) = c {
                            (**max).clone()
                        } else {
                            TrackSize::Fr(1.0)
                        }
                    })
                    .unwrap_or(TrackSize::Fr(1.0));

                css.push_str(&format!(
                    "  grid-template-columns: repeat({}, minmax({}, {}));\n",
                    repeat_fn,
                    min_size.to_css(),
                    max_size.to_css()
                ));
            }
        } else {
            let cols: String = self
                .columns
                .iter()
                .map(|c| c.to_css())
                .collect::<Vec<_>>()
                .join(" ");
            css.push_str(&format!("  grid-template-columns: {};\n", cols));
        }

        // Grid template rows
        if !self.rows.is_empty() {
            let rows: String = self
                .rows
                .iter()
                .map(|r| r.to_css())
                .collect::<Vec<_>>()
                .join(" ");
            css.push_str(&format!("  grid-template-rows: {};\n", rows));
        }

        // Gaps
        if let Some(ref col_gap) = self.column_gap {
            css.push_str(&format!("  column-gap: {};\n", col_gap));
        }
        if let Some(ref row_gap) = self.row_gap {
            css.push_str(&format!("  row-gap: {};\n", row_gap));
        }

        // Grid template areas (if defined)
        if !self.areas.is_empty() {
            // This would require area name mapping to actual template
            // For now, just placeholder
        }

        css
    }

    /// Generate grid container class
    pub fn container_class(&self, class_name: &str) -> String {
        format!(
            ".{} {{
  display: grid;
{}}}",
            class_name,
            self.grid_template_css()
        )
    }

    /// Generate grid item CSS
    pub fn item_css(&self, column: GridPlacement, row: GridPlacement) -> String {
        format!(
            "grid-column: {}; grid-row: {};",
            column.to_css(),
            row.to_css()
        )
    }
}

/// Responsive UI system - combines all components
#[derive(Debug, Clone)]
pub struct ResponsiveUISystem {
    /// Responsive breakpoints
    pub breakpoints: ResponsiveBreakpoints,
    /// Dark mode
    pub dark_mode: DarkMode,
    /// Animation system
    pub animations: AnimationSystem,
    /// Grid system
    pub grid: GridSystem,
}

impl Default for ResponsiveUISystem {
    fn default() -> Self {
        Self {
            breakpoints: ResponsiveBreakpoints::new(),
            dark_mode: DarkMode::new(),
            animations: AnimationSystem::new(),
            grid: GridSystem::new(),
        }
    }
}

impl ResponsiveUISystem {
    /// Create new responsive UI system
    pub fn new() -> Self {
        Self::default()
    }

    /// Set breakpoints
    pub fn breakpoints(mut self, breakpoints: ResponsiveBreakpoints) -> Self {
        self.breakpoints = breakpoints;
        self
    }

    /// Set dark mode
    pub fn dark_mode(mut self, dark_mode: DarkMode) -> Self {
        self.dark_mode = dark_mode;
        self
    }

    /// Set animations
    pub fn animations(mut self, animations: AnimationSystem) -> Self {
        self.animations = animations;
        self
    }

    /// Set grid
    pub fn grid(mut self, grid: GridSystem) -> Self {
        self.grid = grid;
        self
    }

    /// Generate complete CSS framework
    pub fn to_css(&self) -> String {
        let mut css = String::new();

        // CSS reset
        css.push_str("/* Responsive UI Framework */\n\n");
        css.push_str("/* CSS Variables & Dark Mode */\n");
        css.push_str(&self.dark_mode.css_variables());
        css.push_str("\n");

        // Responsive breakpoints
        css.push_str("/* Responsive Breakpoints */\n");
        css.push_str(&self.breakpoints.container_css());
        css.push_str("\n");
        css.push_str(&self.breakpoints.grid_css());
        css.push_str("\n");
        css.push_str(&self.breakpoints.media_query_css());
        css.push_str("\n");

        // Grid system
        css.push_str("/* Grid System */\n");
        css.push_str(&self.grid.container_class("grid"));
        css.push_str("\n");

        // Animations
        css.push_str("/* Animations */\n");
        css.push_str(&self.animations.keyframes_css());
        css.push_str("\n");
        css.push_str("/* Animation Utility Classes */\n");
        css.push_str(&self.animations.utility_classes());
        css.push_str("\n");

        // Dark mode utilities
        css.push_str("/* Dark Mode Utilities */\n");
        css.push_str(&self.dark_mode.utility_classes());
        css.push_str("\n");

        css
    }

    /// Generate HTML with inline CSS
    pub fn to_html(&self) -> String {
        format!(
            "<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Responsive UI</title>
    <style>
{}
    </style>
</head>
<body class=\"theme-transition\">
    <div class=\"container\">
        <!-- Your content here -->
    </div>
    <script>
{}
    </script>
</body>
</html>",
            self.to_css(),
            self.dark_mode.js_switcher()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breakpoint_px() {
        assert_eq!(Breakpoint::SM.px(), 576);
        assert_eq!(Breakpoint::XL.px(), 1200);
    }

    #[test]
    fn test_breakpoint_min_width() {
        assert!(Breakpoint::XS.min_width().is_empty());
        assert_eq!(Breakpoint::MD.min_width(), "(min-width: 768px)");
    }

    #[test]
    fn test_breakpoint_max_width() {
        assert_eq!(Breakpoint::XS.max_width(), "(max-width: 575px)");
        assert_eq!(Breakpoint::SM.max_width(), "(max-width: 767px)");
        assert!(Breakpoint::XXL.max_width().is_empty());
    }

    #[test]
    fn test_breakpoint_from_str() {
        assert_eq!(Breakpoint::from_str("xs"), Some(Breakpoint::XS));
        assert_eq!(Breakpoint::from_str("XL"), Some(Breakpoint::XL));
        assert_eq!(Breakpoint::from_str("invalid"), None);
    }

    #[test]
    fn test_responsive_strategy_breakpoints() {
        let mobile_first = ResponsiveStrategy::MobileFirst.breakpoints();
        assert_eq!(mobile_first[0], Breakpoint::XS);
        assert_eq!(mobile_first[5], Breakpoint::XXL);

        let desktop_first = ResponsiveStrategy::DesktopFirst.breakpoints();
        assert_eq!(desktop_first[0], Breakpoint::XXL);
        assert_eq!(desktop_first[5], Breakpoint::XS);
    }

    #[test]
    fn test_responsive_breakpoints_default() {
        let bp = ResponsiveBreakpoints::default();
        assert_eq!(bp.strategy, ResponsiveStrategy::MobileFirst);
        assert_eq!(bp.columns, 12);
        assert_eq!(bp.gutter_width, 30);
    }

    #[test]
    fn test_responsive_breakpoints_builder() {
        let bp = ResponsiveBreakpoints::new()
            .with_strategy(ResponsiveStrategy::DesktopFirst)
            .columns(16)
            .gutter(24);

        assert_eq!(bp.strategy, ResponsiveStrategy::DesktopFirst);
        assert_eq!(bp.columns, 16);
        assert_eq!(bp.gutter_width, 24);
    }

    #[test]
    fn test_responsive_breakpoints_media_query() {
        let bp = ResponsiveBreakpoints::new();
        let css = bp.media_query_css();
        assert!(css.contains("@media"));
        assert!(css.contains("max-width"));
    }

    #[test]
    fn test_theme_mode_from_str() {
        assert_eq!(ThemeMode::from_str("light"), Some(ThemeMode::Light));
        assert_eq!(ThemeMode::from_str("DARK"), Some(ThemeMode::Dark));
        assert_eq!(ThemeMode::from_str("auto"), Some(ThemeMode::Auto));
        assert_eq!(ThemeMode::from_str("invalid"), None);
    }

    #[test]
    fn test_theme_color() {
        let color = ThemeColor::new("#ffffff", "#000000");
        assert_eq!(color.light, "#ffffff");
        assert_eq!(color.dark, "#000000");

        let single = ThemeColor::single("#ff0000");
        assert_eq!(single.light, "#ff0000");
        assert_eq!(single.dark, "#ff0000");
    }

    #[test]
    fn test_dark_mode_default() {
        let dm = DarkMode::new();
        assert_eq!(dm.mode, ThemeMode::Auto);
        assert!(dm.colors.contains_key("bg-primary"));
        assert!(dm.colors.contains_key("text-primary"));
    }

    #[test]
    fn test_dark_mode_css_variables() {
        let dm = DarkMode::new();
        let css = dm.css_variables();
        assert!(css.contains(":root"));
        assert!(css.contains("--color-bg-primary"));
        assert!(css.contains("@media (prefers-color-scheme: dark)"));
    }

    #[test]
    fn test_easing_function_to_css() {
        assert_eq!(EasingFunction::Linear.to_css(), "linear");
        assert_eq!(EasingFunction::EaseInOut.to_css(), "ease-in-out");
        assert!(EasingFunction::Spring.to_css().contains("cubic-bezier"));
    }

    #[test]
    fn test_keyframe() {
        let kf = Keyframe::new("0%")
            .add("opacity", "0")
            .add("transform", "scale(0.5)");

        assert_eq!(kf.selector, "0%");
        assert_eq!(kf.properties.len(), 2);
        assert!(kf.properties.contains_key("opacity"));
    }

    #[test]
    fn test_animation() {
        let anim = Animation::new("test")
            .duration("500ms")
            .easing(EasingFunction::EaseOut)
            .infinite();

        assert_eq!(anim.name, "test");
        assert_eq!(anim.duration, "500ms");
        assert_eq!(anim.iteration_count, "infinite");
    }

    #[test]
    fn test_animation_keyframes_css() {
        let anim = Animation::new("fade")
            .keyframe(Keyframe::new("from").add("opacity", "0"))
            .keyframe(Keyframe::new("to").add("opacity", "1"));

        let css = anim.keyframes_css();
        assert!(css.contains("@keyframes fade"));
        assert!(css.contains("opacity: 0"));
        assert!(css.contains("opacity: 1"));
    }

    #[test]
    fn test_animation_system_default() {
        let sys = AnimationSystem::new();
        assert!(sys.animations.contains_key("fade-in"));
        assert!(sys.animations.contains_key("slide-up"));
        assert!(sys.animations.contains_key("bounce"));
        assert!(sys.animations.contains_key("spin"));
    }

    #[test]
    fn test_animation_system_get() {
        let sys = AnimationSystem::new();
        let fade = sys.get("fade-in");
        assert!(fade.is_some());
        assert_eq!(fade.unwrap().name, "fade-in");

        assert!(sys.get("nonexistent").is_none());
    }

    #[test]
    fn test_track_size_to_css() {
        assert_eq!(TrackSize::Auto.to_css(), "auto");
        assert_eq!(TrackSize::px(100).to_css(), "100px");
        assert_eq!(TrackSize::percent(50).to_css(), "50%");
        assert_eq!(TrackSize::fr(1.5).to_css(), "1.5fr");
    }

    #[test]
    fn test_track_size_minmax() {
        let size = TrackSize::minmax(TrackSize::px(200), TrackSize::fr(1.0));
        assert!(size.to_css().contains("minmax"));
        assert!(size.to_css().contains("200px"));
    }

    #[test]
    fn test_grid_placement_to_css() {
        assert_eq!(GridPlacement::Auto.to_css(), "auto");
        assert_eq!(GridPlacement::Line(2).to_css(), "2");
        assert_eq!(GridPlacement::Span(3).to_css(), "span 3");
        assert_eq!(
            GridPlacement::Named("sidebar".to_string()).to_css(),
            "sidebar"
        );
    }

    #[test]
    fn test_grid_area() {
        let area = GridArea::new("header")
            .column(GridPlacement::Line(1), GridPlacement::Line(4))
            .row(GridPlacement::Line(1), GridPlacement::Line(2));

        assert_eq!(area.name, "header");
        assert!(area.to_area_css().contains("header:"));
    }

    #[test]
    fn test_grid_system_default() {
        let grid = GridSystem::new();
        assert_eq!(grid.columns.len(), 1);
        assert_eq!(grid.rows.len(), 1);
        assert!(!grid.auto_fit);
    }

    #[test]
    fn test_grid_system_fixed_columns() {
        let grid = GridSystem::fixed_columns(12);
        assert_eq!(grid.columns.len(), 12);
    }

    #[test]
    fn test_grid_system_responsive() {
        let grid = GridSystem::responsive(250, Some(350));
        assert!(grid.auto_fit);
        assert!(grid.min_track_size.is_some());
    }

    #[test]
    fn test_grid_system_builder() {
        let grid = GridSystem::new()
            .columns(vec![TrackSize::fr(1.0), TrackSize::fr(2.0)])
            .gap("16px");

        assert_eq!(grid.columns.len(), 2);
        assert!(grid.column_gap.is_some());
    }

    #[test]
    fn test_grid_system_template_css() {
        let grid = GridSystem::fixed_columns(3).gap("20px");
        let css = grid.grid_template_css();
        assert!(css.contains("grid-template-columns"));
        assert!(css.contains("column-gap"));
        assert!(css.contains("20px"));
    }

    #[test]
    fn test_responsive_ui_system_default() {
        let sys = ResponsiveUISystem::new();
        assert!(sys
            .breakpoints
            .container_widths
            .contains_key(&Breakpoint::MD));
        assert!(sys.dark_mode.colors.contains_key("bg-primary"));
        assert!(sys.animations.animations.contains_key("fade-in"));
    }

    #[test]
    fn test_responsive_ui_system_to_css() {
        let sys = ResponsiveUISystem::new();
        let css = sys.to_css();

        assert!(css.contains("/* Responsive UI Framework */"));
        assert!(css.contains("--color-"));
        assert!(css.contains("@media"));
        assert!(css.contains("@keyframes"));
        assert!(css.contains(".grid"));
    }

    #[test]
    fn test_responsive_ui_system_to_html() {
        let sys = ResponsiveUISystem::new();
        let html = sys.to_html();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<style>"));
        assert!(html.contains("<script>"));
        assert!(html.contains("setTheme"));
        assert!(html.contains("theme-transition"));
    }

    #[test]
    fn test_complete_integration() {
        let sys = ResponsiveUISystem::new()
            .breakpoints(
                ResponsiveBreakpoints::new()
                    .with_strategy(ResponsiveStrategy::DesktopFirst)
                    .columns(16),
            )
            .dark_mode(
                DarkMode::new()
                    .with_mode(ThemeMode::Dark)
                    .transition("0.5s"),
            )
            .grid(GridSystem::responsive(300, Some(400)).gap("24px"));

        let css = sys.to_css();
        assert!(css.contains("grid-template-columns"));
        assert!(css.contains("--theme-transition: color 0.5s"));

        let html = sys.to_html();
        assert!(html.contains("setTheme"));
    }
}
