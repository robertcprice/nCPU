//! Bundling configuration and optimization for modern web applications.
//!
//! This module provides comprehensive support for:
//! - Vite configuration and plugin development
//! - Webpack configuration and loader system
//! - Esbuild configuration and optimization
//! - Module resolution strategies
//! - Bundle analysis and optimization
//! - Code splitting and lazy loading
//! - Tree shaking and dead code elimination
//! - Asset optimization and caching

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ============================================================================
// Core Bundling Types
// ============================================================================

/// Bundle format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BundleFormat {
    /// ESM format for modern browsers
    Esm,
    /// CommonJS format for Node.js
    Cjs,
    /// IIFE format for direct browser use
    Iife,
    /// UMD format for universal use
    Umd,
    /// SystemJS format
    System,
}

impl Default for BundleFormat {
    fn default() -> Self {
        Self::Esm
    }
}

/// Target environment for bundling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TargetEnvironment {
    /// Modern browsers (ES2020+)
    Modern,
    /// Legacy browsers (ES5)
    Legacy,
    /// Node.js environment
    Node,
    /// Electron environment
    Electron,
    /// Deno runtime
    Deno,
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations
    Basic,
    /// Advanced optimizations
    Advanced,
    /// Aggressive optimizations
    Aggressive,
}

/// Source map configuration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SourceMapType {
    /// No source map
    None,
    /// Inline source map
    Inline,
    /// External source map file
    External,
    /// Hidden source map (not referenced)
    Hidden,
    /// Eval source map
    Eval,
}

impl Default for SourceMapType {
    fn default() -> Self {
        Self::None
    }
}

/// Module resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ResolutionStrategy {
    /// Node.js resolution algorithm
    Node,
    /// Browser resolution with index field
    Browser,
    /// Alloy resolution (Node + Browser)
    Alloy,
}

impl Default for ResolutionStrategy {
    fn default() -> Self {
        Self::Browser
    }
}

// ============================================================================
// Vite Configuration
// ============================================================================

/// Complete Vite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteConfig {
    /// Project root directory
    pub root: Option<PathBuf>,
    /// Base public path
    #[serde(default = "default_base")]
    pub base: String,
    /// Public directory
    #[serde(default = "default_public_dir")]
    pub public_dir: String,
    /// Build configuration
    #[serde(default)]
    pub build: ViteBuildConfig,
    /// Server configuration
    #[serde(default)]
    pub server: ViteServerConfig,
    /// Preview configuration
    #[serde(default)]
    pub preview: VitePreviewConfig,
    /// Dependencies optimization
    #[serde(default)]
    pub optimize_deps: ViteOptimizeDepsConfig,
    /// CSS configuration
    #[serde(default)]
    pub css: ViteCssConfig,
    /// Assets configuration
    #[serde(default)]
    pub assets: ViteAssetsConfig,
    /// Plugins
    #[serde(default)]
    pub plugins: Vec<VitePlugin>,
    /// Resolve configuration
    #[serde(default)]
    pub resolve: ViteResolveConfig,
    /// Environment variables
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Define constants
    #[serde(default)]
    pub define: HashMap<String, String>,
}

fn default_base() -> String {
    "/".to_string()
}

fn default_public_dir() -> String {
    "public".to_string()
}

impl Default for ViteConfig {
    fn default() -> Self {
        Self {
            root: None,
            base: default_base(),
            public_dir: default_public_dir(),
            build: Default::default(),
            server: Default::default(),
            preview: Default::default(),
            optimize_deps: Default::default(),
            css: Default::default(),
            assets: Default::default(),
            plugins: Vec::new(),
            resolve: Default::default(),
            env: HashMap::new(),
            define: HashMap::new(),
        }
    }
}

/// Vite build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteBuildConfig {
    /// Target output directory
    #[serde(default = "default_out_dir")]
    pub out_dir: PathBuf,
    /// Asset output directory
    #[serde(default = "default_assets_dir")]
    pub assets_dir: String,
    /// Emit declaration files
    #[serde(default)]
    pub emit_declare: bool,
    /// Empty output directory before build
    #[serde(default = "default_empty_out_dir")]
    pub empty_out_dir: bool,
    /// Bundle configuration
    #[serde(default)]
    pub bundle: ViteBundleConfig,
    /// Rollup options
    #[serde(default)]
    pub rollup_options: ViteRollupOptions,
    /// Minify settings
    #[serde(default)]
    pub minify: ViteMinifyType,
    /// Target environment
    #[serde(default = "default_target")]
    pub target: String,
    /// Polyfill dynamic import
    #[serde(default)]
    pub polyfill_dynamic_import: bool,
    /// Write to disk
    #[serde(default = "default_write")]
    pub write: bool,
    /// CSS code splitting
    #[serde(default = "default_css_code_split")]
    pub css_code_split: bool,
    /// CSS inline limit
    #[serde(default = "default_css_inline_limit")]
    pub css_inline_limit: usize,
    /// Module pre-loading
    #[serde(default)]
    pub module_preload: bool,
    /// Sourcemap configuration
    #[serde(default)]
    pub sourcemap: SourceMapType,
    /// Manifest file
    #[serde(default)]
    pub manifest: bool,
}

fn default_out_dir() -> PathBuf {
    PathBuf::from("dist")
}

fn default_assets_dir() -> String {
    "assets".to_string()
}

fn default_empty_out_dir() -> bool {
    true
}

fn default_target() -> String {
    "modules".to_string()
}

fn default_write() -> bool {
    true
}

fn default_css_code_split() -> bool {
    true
}

fn default_css_inline_limit() -> usize {
    4096
}

impl Default for ViteBuildConfig {
    fn default() -> Self {
        Self {
            out_dir: default_out_dir(),
            assets_dir: default_assets_dir(),
            emit_declare: false,
            empty_out_dir: default_empty_out_dir(),
            bundle: Default::default(),
            rollup_options: Default::default(),
            minify: Default::default(),
            target: default_target(),
            polyfill_dynamic_import: false,
            write: default_write(),
            css_code_split: default_css_code_split(),
            css_inline_limit: default_css_inline_limit(),
            module_preload: true,
            sourcemap: SourceMapType::None,
            manifest: false,
        }
    }
}

/// Vite bundle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteBundleConfig {
    /// Generate bundle
    #[serde(default = "default_bundle")]
    pub bundle: bool,
}

fn default_bundle() -> bool {
    true
}

impl Default for ViteBundleConfig {
    fn default() -> Self {
        Self {
            bundle: default_bundle(),
        }
    }
}

/// Vite rollup options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteRollupOptions {
    /// Input entries
    #[serde(default)]
    pub input: Vec<String>,
    /// Output options
    #[serde(default)]
    pub output: ViteRollupOutputOptions,
    /// External modules
    #[serde(default)]
    pub external: Vec<String>,
    /// Plugins
    #[serde(default)]
    pub plugins: Vec<String>,
}

impl Default for ViteRollupOptions {
    fn default() -> Self {
        Self {
            input: vec!["index.html".to_string()],
            output: Default::default(),
            external: Vec::new(),
            plugins: Vec::new(),
        }
    }
}

/// Vite rollup output options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteRollupOutputOptions {
    /// Output format
    #[serde(default)]
    pub format: BundleFormat,
    /// Asset file name
    #[serde(default = "default_asset_file_name")]
    pub asset_file_name: String,
    /// Chunk file name
    #[serde(default = "default_chunk_file_name")]
    pub chunk_file_name: String,
    /// Entry file name
    #[serde(default = "default_entry_file_name")]
    pub entry_file_names: String,
}

fn default_asset_file_name() -> String {
    "assets/[name]-[hash][extname]".to_string()
}

fn default_chunk_file_name() -> String {
    "assets/[name]-[hash].js".to_string()
}

fn default_entry_file_name() -> String {
    "assets/[name]-[hash].js".to_string()
}

impl Default for ViteRollupOutputOptions {
    fn default() -> Self {
        Self {
            format: BundleFormat::Esm,
            asset_file_name: default_asset_file_name(),
            chunk_file_name: default_chunk_file_name(),
            entry_file_names: default_entry_file_name(),
        }
    }
}

/// Vite minify type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ViteMinifyType {
    /// No minification
    None,
    /// ESBuild minification
    Esbuild,
    /// Terser minification
    Terser,
}

impl Default for ViteMinifyType {
    fn default() -> Self {
        Self::Esbuild
    }
}

/// Vite server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteServerConfig {
    /// Server host
    #[serde(default = "default_host")]
    pub host: String,
    /// Server port
    #[serde(default = "default_port")]
    pub port: u16,
    /// Strict port
    #[serde(default)]
    pub strict_port: bool,
    /// HTTPS configuration
    #[serde(default)]
    pub https: Option<ViteHttpsConfig>,
    /// Open browser automatically
    #[serde(default)]
    pub open: bool,
    /// Proxy configuration
    #[serde(default)]
    pub proxy: HashMap<String, ViteProxyConfig>,
    /// CORS configuration
    #[serde(default)]
    pub cors: ViteCorsConfig,
    /// HMR configuration
    #[serde(default)]
    pub hmr: ViteHmrConfig,
}

fn default_host() -> String {
    "localhost".to_string()
}

fn default_port() -> u16 {
    5173
}

impl Default for ViteServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            strict_port: false,
            https: None,
            open: false,
            proxy: HashMap::new(),
            cors: Default::default(),
            hmr: Default::default(),
        }
    }
}

/// Vite HTTPS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteHttpsConfig {
    /// Path to cert file
    pub cert: PathBuf,
    /// Path to key file
    pub key: PathBuf,
}

/// Vite proxy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteProxyConfig {
    /// Target URL
    pub target: String,
    /// Change origin
    #[serde(default)]
    pub change_origin: bool,
    /// Secure flag
    #[serde(default = "default_secure")]
    pub secure: bool,
    /// Rewrite path
    #[serde(default)]
    pub rewrite: Option<String>,
    /// Configure webpack
    #[serde(default)]
    pub configure: Option<serde_json::Value>,
}

fn default_secure() -> bool {
    true
}

/// Vite CORS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteCorsConfig {
    /// Enable CORS
    #[serde(default = "default_cors_enabled")]
    pub enabled: bool,
    /// Allowed origins
    #[serde(default)]
    pub origin: String,
}

fn default_cors_enabled() -> bool {
    true
}

impl Default for ViteCorsConfig {
    fn default() -> Self {
        Self {
            enabled: default_cors_enabled(),
            origin: "*".to_string(),
        }
    }
}

/// Vite HMR configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteHmrConfig {
    /// Enable HMR
    #[serde(default = "default_hmr_enabled")]
    pub enabled: bool,
    /// HMR protocol
    #[serde(default = "default_hmr_protocol")]
    pub protocol: String,
    /// HMR host
    #[serde(default)]
    pub host: Option<String>,
    /// HMR port
    #[serde(default)]
    pub port: Option<u16>,
}

fn default_hmr_enabled() -> bool {
    true
}

fn default_hmr_protocol() -> String {
    "ws".to_string()
}

impl Default for ViteHmrConfig {
    fn default() -> Self {
        Self {
            enabled: default_hmr_enabled(),
            protocol: default_hmr_protocol(),
            host: None,
            port: None,
        }
    }
}

/// Vite preview configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VitePreviewConfig {
    /// Preview host
    #[serde(default = "default_host")]
    pub host: String,
    /// Preview port
    #[serde(default = "default_preview_port")]
    pub port: u16,
    /// Open browser
    #[serde(default)]
    pub open: bool,
}

fn default_preview_port() -> u16 {
    4173
}

impl Default for VitePreviewConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_preview_port(),
            open: false,
        }
    }
}

/// Vite dependencies optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteOptimizeDepsConfig {
    /// Include dependencies
    #[serde(default)]
    pub include: Vec<String>,
    /// Exclude dependencies
    #[serde(default)]
    pub exclude: Vec<String>,
    /// Force optimization
    #[serde(default)]
    pub force: Option<bool>,
}

impl Default for ViteOptimizeDepsConfig {
    fn default() -> Self {
        Self {
            include: Vec::new(),
            exclude: Vec::new(),
            force: None,
        }
    }
}

/// Vite CSS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteCssConfig {
    /// CSS modules configuration
    #[serde(default)]
    pub modules: ViteCssModulesConfig,
    /// PostCSS configuration
    #[serde(default)]
    pub postcss: Option<VitePostcssConfig>,
    /// Preprocessor options
    #[serde(default)]
    pub preprocessor_options: HashMap<String, serde_json::Value>,
}

impl Default for ViteCssConfig {
    fn default() -> Self {
        Self {
            modules: Default::default(),
            postcss: None,
            preprocessor_options: HashMap::new(),
        }
    }
}

/// Vite CSS modules configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteCssModulesConfig {
    /// Generate scoped locals
    #[serde(default)]
    pub localsConvention: String,
    /// Scope behaviour
    #[serde(default = "default_scope_behaviour")]
    pub scope_behaviour: String,
}

fn default_scope_behaviour() -> String {
    "global".to_string()
}

impl Default for ViteCssModulesConfig {
    fn default() -> Self {
        Self {
            localsConvention: "camelCase".to_string(),
            scope_behaviour: default_scope_behaviour(),
        }
    }
}

/// Vite PostCSS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VitePostcssConfig {
    /// PostCSS plugins
    #[serde(default)]
    pub plugins: Vec<String>,
}

/// Vite assets configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteAssetsConfig {
    /// Inline asset limit
    #[serde(default = "default_inline_limit")]
    pub inline_limit: usize,
    /// CSS inline limit
    #[serde(default = "default_css_inline_limit")]
    pub css_inline_limit: usize,
    /// Assets include
    #[serde(default)]
    pub assets_include: Vec<String>,
}

fn default_inline_limit() -> usize {
    4096
}

impl Default for ViteAssetsConfig {
    fn default() -> Self {
        Self {
            inline_limit: default_inline_limit(),
            css_inline_limit: default_css_inline_limit(),
            assets_include: vec![
                "*.png".to_string(),
                "*.jpg".to_string(),
                "*.jpeg".to_string(),
                "*.gif".to_string(),
                "*.svg".to_string(),
                "*.ico".to_string(),
                "*.webp".to_string(),
                "*.avif".to_string(),
            ],
        }
    }
}

/// Vite resolve configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViteResolveConfig {
    /// Alias configuration
    #[serde(default)]
    pub alias: HashMap<String, String>,
    /// Extension order
    #[serde(default = "default_extensions")]
    pub extensions: Vec<String>,
    /// Main fields
    #[serde(default = "default_main_fields")]
    pub main_fields: Vec<String>,
    /// Conditions
    #[serde(default)]
    pub conditions: Vec<String>,
}

fn default_extensions() -> Vec<String> {
    vec![
        ".mjs".to_string(),
        ".js".to_string(),
        ".mts".to_string(),
        ".ts".to_string(),
        ".jsx".to_string(),
        ".tsx".to_string(),
        ".json".to_string(),
    ]
}

fn default_main_fields() -> Vec<String> {
    vec!["browser".to_string(), "module".to_string(), "main".to_string()]
}

impl Default for ViteResolveConfig {
    fn default() -> Self {
        Self {
            alias: HashMap::new(),
            extensions: default_extensions(),
            main_fields: default_main_fields(),
            conditions: vec!["module".to_string(), "browser".to_string(), "development".to_string()],
        }
    }
}

// ============================================================================
// Vite Plugin System
// ============================================================================

/// Vite plugin interface
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VitePlugin {
    /// Plugin name
    pub name: String,
    /// Plugin hooks
    #[serde(default)]
    pub hooks: VitePluginHooks,
    /// Plugin configuration
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

/// Vite plugin hooks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VitePluginHooks {
    /// Build start hook
    #[serde(default)]
    pub build_start: Option<String>,
    /// Transform hook
    #[serde(default)]
    pub transform: Option<String>,
    /// Load hook
    #[serde(default)]
    pub load: Option<String>,
    /// Resolve ID hook
    #[serde(default)]
    pub resolve_id: Option<String>,
    /// Generate bundle hook
    #[serde(default)]
    pub generate_bundle: Option<String>,
    /// Write bundle hook
    #[serde(default)]
    pub write_bundle: Option<String>,
    /// Close bundle hook
    #[serde(default)]
    pub close_bundle: Option<String>,
    /// Configure server hook
    #[serde(default)]
    pub configure_server: Option<String>,
    /// Handle hot update hook
    #[serde(default)]
    pub handle_hot_update: Option<String>,
    /// Transform index HTML hook
    #[serde(default)]
    pub transform_index_html: Option<String>,
}

impl Default for VitePluginHooks {
    fn default() -> Self {
        Self {
            build_start: None,
            transform: None,
            load: None,
            resolve_id: None,
            generate_bundle: None,
            write_bundle: None,
            close_bundle: None,
            configure_server: None,
            handle_hot_update: None,
            transform_index_html: None,
        }
    }
}

/// Built-in Vite plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ViteBuiltinPlugin {
    /// Vue plugin
    Vue,
    /// React plugin
    React,
    /// React refresh plugin
    ReactRefresh,
    /// Legacy browser plugin
    Legacy,
    /// SSR plugin
    Ssr,
    /// Manifest plugin
    Manifest,
    /// Visualizer plugin
    Visualizer,
    /// Components plugin
    Components,
    /// Auto-import plugin
    AutoImport,
    /// Markdown plugin
    Markdown,
    /// SVG loader plugin
    SvgLoader,
    /// Webfont loader plugin
    WebfontLoader,
    /// PWA plugin
    Pwa,
    /// ESLint plugin
    Eslint,
    /// TypeScript plugin
    TypeScript,
}

// ============================================================================
// Webpack Configuration
// ============================================================================

/// Complete Webpack configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackConfig {
    /// Entry points
    #[serde(default)]
    pub entry: WebpackEntry,
    /// Output configuration
    #[serde(default)]
    pub output: WebpackOutputConfig,
    /// Module configuration
    #[serde(default)]
    pub module: WebpackModuleConfig,
    /// Resolve configuration
    #[serde(default)]
    pub resolve: WebpackResolveConfig,
    /// Plugins
    #[serde(default)]
    pub plugins: Vec<WebpackPlugin>,
    /// Optimization configuration
    #[serde(default)]
    pub optimization: WebpackOptimizationConfig,
    /// Mode
    #[serde(default = "default_webpack_mode")]
    pub mode: String,
    /// Devtool configuration
    #[serde(default)]
    pub devtool: SourceMapType,
    /// Target environment
    #[serde(default = "default_target")]
    pub target: String,
    /// Externals
    #[serde(default)]
    pub externals: Vec<String>,
    /// Dev server configuration
    #[serde(default)]
    pub dev_server: Option<WebpackDevServerConfig>,
    /// Performance configuration
    #[serde(default)]
    pub performance: WebpackPerformanceConfig,
    /// Stats configuration
    #[serde(default)]
    pub stats: WebpackStatsConfig,
}

fn default_webpack_mode() -> String {
    "production".to_string()
}

impl Default for WebpackConfig {
    fn default() -> Self {
        Self {
            entry: Default::default(),
            output: Default::default(),
            module: Default::default(),
            resolve: Default::default(),
            plugins: Vec::new(),
            optimization: Default::default(),
            mode: default_webpack_mode(),
            devtool: SourceMapType::None,
            target: default_target(),
            externals: Vec::new(),
            dev_server: None,
            performance: Default::default(),
            stats: Default::default(),
        }
    }
}

/// Webpack entry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WebpackEntry {
    /// Single entry
    Single(String),
    /// Multiple entries
    Multiple(HashMap<String, String>),
}

impl Default for WebpackEntry {
    fn default() -> Self {
        Self::Single("./src/index.ts".to_string())
    }
}

/// Webpack output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackOutputConfig {
    /// Output directory
    #[serde(default = "default_out_dir")]
    pub path: PathBuf,
    /// Public path
    #[serde(default = "default_base")]
    pub public_path: String,
    /// Filename
    #[serde(default = "default_webpack_filename")]
    pub filename: String,
    /// Chunk filename
    #[serde(default = "default_chunk_file_name")]
    pub chunk_filename: String,
    /// Asset filename
    #[serde(default = "default_asset_file_name")]
    pub asset_filename: String,
    /// Library name
    #[serde(default)]
    pub library: Option<String>,
    /// Library target
    #[serde(default)]
    pub library_target: Option<BundleFormat>,
    /// Clean before emit
    #[serde(default = "default_clean")]
    pub clean: bool,
}

fn default_webpack_filename() -> String {
    "[name].js".to_string()
}

fn default_clean() -> bool {
    true
}

impl Default for WebpackOutputConfig {
    fn default() -> Self {
        Self {
            path: default_out_dir(),
            public_path: default_base(),
            filename: default_webpack_filename(),
            chunk_filename: default_chunk_file_name(),
            asset_filename: default_asset_file_name(),
            library: None,
            library_target: None,
            clean: default_clean(),
        }
    }
}

/// Webpack module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackModuleConfig {
    /// Rules for processing different file types
    #[serde(default)]
    pub rules: Vec<WebpackRule>,
    /// No parse conditions
    #[serde(default)]
    pub no_parse: Vec<String>,
}

impl Default for WebpackModuleConfig {
    fn default() -> Self {
        Self {
            rules: vec![
                WebpackRule::javascript(),
                WebpackRule::typescript(),
                WebpackRule::css(),
                WebpackRule::images(),
            ],
            no_parse: Vec::new(),
        }
    }
}

/// Webpack rule for file processing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackRule {
    /// Test pattern
    pub test: String,
    /// Rule type
    #[serde(default)]
    pub rule_type: WebpackRuleType,
    /// Use loaders
    #[serde(default = "default_loaders")]
    pub loaders: Vec<WebpackLoader>,
    /// Exclude patterns
    #[serde(default)]
    pub exclude: Vec<String>,
    /// Include patterns
    #[serde(default)]
    pub include: Vec<String>,
    /// Resource query
    #[serde(default)]
    pub resource_query: Option<String>,
    /// Parser options
    #[serde(default)]
    pub parser: HashMap<String, serde_json::Value>,
    /// Generator options
    #[serde(default)]
    pub generator: HashMap<String, serde_json::Value>,
}

fn default_loaders() -> Vec<WebpackLoader> {
    Vec::new()
}

/// Webpack rule type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebpackRuleType {
    /// JavaScript/TypeScript
    Javascript,
    /// Asset
    Asset,
    /// CSS
    Css,
    /// JSON
    Json,
}

impl Default for WebpackRuleType {
    fn default() -> Self {
        Self::Javascript
    }
}

impl WebpackRule {
    /// Create JavaScript rule
    pub fn javascript() -> Self {
        Self {
            test: r"\.(mjs|jsx?|cjs)$".to_string(),
            rule_type: WebpackRuleType::Javascript,
            loaders: vec![WebpackLoader::babel()],
            exclude: vec!["node_modules".to_string()],
            include: Vec::new(),
            resource_query: None,
            parser: HashMap::new(),
            generator: HashMap::new(),
        }
    }

    /// Create TypeScript rule
    pub fn typescript() -> Self {
        Self {
            test: r"\.(mts|ts|tsx)$".to_string(),
            rule_type: WebpackRuleType::Javascript,
            loaders: vec![WebpackLoader::ts_loader()],
            exclude: vec!["node_modules".to_string()],
            include: Vec::new(),
            resource_query: None,
            parser: HashMap::new(),
            generator: HashMap::new(),
        }
    }

    /// Create CSS rule
    pub fn css() -> Self {
        Self {
            test: r"\.(css|less|sass|scss)$".to_string(),
            rule_type: WebpackRuleType::Css,
            loaders: vec![
                WebpackLoader::style_loader(),
                WebpackLoader::css_loader(),
                WebpackLoader::postcss_loader(),
            ],
            exclude: Vec::new(),
            include: Vec::new(),
            resource_query: None,
            parser: HashMap::new(),
            generator: HashMap::new(),
        }
    }

    /// Create images rule
    pub fn images() -> Self {
        Self {
            test: r"\.(png|jpg|jpeg|gif|svg|webp|avif)$".to_string(),
            rule_type: WebpackRuleType::Asset,
            loaders: Vec::new(),
            exclude: Vec::new(),
            include: Vec::new(),
            resource_query: None,
            parser: HashMap::new(),
            generator: HashMap::new(),
        }
    }
}

/// Webpack loader configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackLoader {
    /// Loader name
    pub name: String,
    /// Loader options
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

impl WebpackLoader {
    /// Babel loader
    pub fn babel() -> Self {
        let mut options = HashMap::new();
        options.insert("presets".to_string(), serde_json::json!(["@babel/preset-env", "@babel/preset-react"]));
        options.insert("cacheDirectory".to_string(), serde_json::json!(true));

        Self {
            name: "babel-loader".to_string(),
            options,
        }
    }

    /// TypeScript loader
    pub fn ts_loader() -> Self {
        let mut options = HashMap::new();
        options.insert("transpileOnly".to_string(), serde_json::json!(true));

        Self {
            name: "ts-loader".to_string(),
            options,
        }
    }

    /// Style loader
    pub fn style_loader() -> Self {
        Self {
            name: "style-loader".to_string(),
            options: HashMap::new(),
        }
    }

    /// CSS loader
    pub fn css_loader() -> Self {
        let mut options = HashMap::new();
        options.insert("modules".to_string(), serde_json::json!(true));

        Self {
            name: "css-loader".to_string(),
            options,
        }
    }

    /// PostCSS loader
    pub fn postcss_loader() -> Self {
        Self {
            name: "postcss-loader".to_string(),
            options: HashMap::new(),
        }
    }

    /// File loader
    pub fn file_loader() -> Self {
        let mut options = HashMap::new();
        options.insert("name".to_string(), serde_json::json!("[name].[hash].[ext]"));

        Self {
            name: "file-loader".to_string(),
            options,
        }
    }
}

/// Webpack resolve configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackResolveConfig {
    /// Alias configuration
    #[serde(default)]
    pub alias: HashMap<String, String>,
    /// Extension order
    #[serde(default = "default_extensions")]
    pub extensions: Vec<String>,
    /// Main fields
    #[serde(default = "default_main_fields")]
    pub main_fields: Vec<String>,
    /// Main files
    #[serde(default = "default_main_files")]
    pub main_files: Vec<String>,
    /// Modules directories
    #[serde(default = "default_modules")]
    pub modules: Vec<String>,
    /// Resolution strategy
    #[serde(default)]
    pub strategy: ResolutionStrategy,
}

fn default_main_files() -> Vec<String> {
    vec!["index".to_string()]
}

fn default_modules() -> Vec<String> {
    vec!["node_modules".to_string()]
}

impl Default for WebpackResolveConfig {
    fn default() -> Self {
        Self {
            alias: HashMap::new(),
            extensions: default_extensions(),
            main_fields: default_main_fields(),
            main_files: default_main_files(),
            modules: default_modules(),
            strategy: ResolutionStrategy::Browser,
        }
    }
}

/// Webpack plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackPlugin {
    /// Plugin name
    pub name: String,
    /// Plugin options
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

/// Built-in Webpack plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WebpackBuiltinPlugin {
    /// HTML webpack plugin
    HtmlWebpack,
    /// Mini CSS extract plugin
    MiniCssExtract,
    /// Clean webpack plugin
    Clean,
    /// Copy webpack plugin
    Copy,
    /// Define plugin
    Define,
    /// Environment plugin
    Environment,
    /// Hot module replacement plugin
    HotModuleReplacement,
    /// Bundle analyzer plugin
    BundleAnalyzer,
    /// Progress plugin
    Progress,
    /// Fork ts checker webpack plugin
    ForkTsChecker,
    /// ESLint plugin
    Eslint,
}

/// Webpack optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackOptimizationConfig {
    /// Minimize
    #[serde(default = "default_minimize")]
    pub minimize: bool,
    /// Minimizer
    #[serde(default)]
    pub minimizer: Vec<String>,
    /// Split chunks
    #[serde(default)]
    pub split_chunks: WebpackSplitChunksConfig,
    /// Runtime chunk
    #[serde(default)]
    pub runtime_chunk: WebpackRuntimeChunk,
    /// Module ids
    #[serde(default)]
    pub module_ids: WebpackModuleIds,
    /// Chunk ids
    #[serde(default)]
    pub chunk_ids: WebpackChunkIds,
}

fn default_minimize() -> bool {
    true
}

impl Default for WebpackOptimizationConfig {
    fn default() -> Self {
        Self {
            minimize: default_minimize(),
            minimizer: vec!["terser-webpack-plugin".to_string()],
            split_chunks: Default::default(),
            runtime_chunk: WebpackRuntimeChunk::Single,
            module_ids: WebpackModuleIds::Deterministic,
            chunk_ids: WebpackChunkIds::Deterministic,
        }
    }
}

/// Webpack split chunks configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackSplitChunksConfig {
    /// Chunk selection
    #[serde(default)]
    pub chunks: WebpackChunksSelection,
    /// Minimum size
    #[serde(default = "default_min_size")]
    pub min_size: usize,
    /// Maximum size
    #[serde(default = "default_max_size")]
    pub max_size: usize,
    /// Minimum chunks
    #[serde(default = "default_min_chunks")]
    pub min_chunks: u32,
    /// Maximum async requests
    #[serde(default = "default_max_async_requests")]
    pub max_async_requests: u32,
    /// Maximum initial requests
    #[serde(default = "default_max_initial_requests")]
    pub max_initial_requests: u32,
    /// Automatic name delimiter
    #[serde(default = "default_auto_name_delimiter")]
    pub automatic_name_delimiter: String,
    /// Cache groups
    #[serde(default)]
    pub cache_groups: HashMap<String, WebpackCacheGroup>,
}

fn default_min_size() -> usize {
    20000
}

fn default_max_size() -> usize {
    244000
}

fn default_min_chunks() -> u32 {
    1
}

fn default_max_async_requests() -> u32 {
    30
}

fn default_max_initial_requests() -> u32 {
    30
}

fn default_auto_name_delimiter() -> String {
    "~".to_string()
}

impl Default for WebpackSplitChunksConfig {
    fn default() -> Self {
        let mut cache_groups = HashMap::new();
        cache_groups.insert(
            "default".to_string(),
            WebpackCacheGroup {
                min_chunks: 2,
                priority: -10,
                reuse_existing_chunk: true,
                name: None,
                test: None,
            },
        );
        cache_groups.insert(
            "vendors".to_string(),
            WebpackCacheGroup {
                test: Some("[/\\\\]node_modules[/\\\\]".to_string()),
                priority: -10,
                ..Default::default()
            },
        );

        Self {
            chunks: WebpackChunksSelection::Async,
            min_size: default_min_size(),
            max_size: default_max_size(),
            min_chunks: default_min_chunks(),
            max_async_requests: default_max_async_requests(),
            max_initial_requests: default_max_initial_requests(),
            automatic_name_delimiter: default_auto_name_delimiter(),
            cache_groups,
        }
    }
}

/// Webpack chunks selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebpackChunksSelection {
    /// All chunks
    All,
    /// Async chunks only
    Async,
    /// Initial chunks only
    Initial,
}

impl Default for WebpackChunksSelection {
    fn default() -> Self {
        Self::Async
    }
}

/// Webpack cache group
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackCacheGroup {
    /// Test pattern
    #[serde(default)]
    pub test: Option<String>,
    /// Name
    #[serde(default)]
    pub name: Option<String>,
    /// Priority
    #[serde(default)]
    pub priority: i32,
    /// Minimum chunks
    #[serde(default)]
    pub min_chunks: u32,
    /// Reuse existing chunk
    #[serde(default)]
    pub reuse_existing_chunk: bool,
}

impl Default for WebpackCacheGroup {
    fn default() -> Self {
        Self {
            test: None,
            name: None,
            priority: 0,
            min_chunks: 1,
            reuse_existing_chunk: false,
        }
    }
}

/// Webpack runtime chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebpackRuntimeChunk {
    /// Single runtime chunk
    Single,
    /// Multiple runtime chunks
    Multiple,
    /// No runtime chunk
    False,
}

impl Default for WebpackRuntimeChunk {
    fn default() -> Self {
        Self::Single
    }
}

/// Webpack module IDs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebpackModuleIds {
    /// Named modules
    Named,
    /// Deterministic IDs
    Deterministic,
    /// Natural IDs
    Natural,
}

impl Default for WebpackModuleIds {
    fn default() -> Self {
        Self::Deterministic
    }
}

/// Webpack chunk IDs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebpackChunkIds {
    /// Named chunks
    Named,
    /// Deterministic IDs
    Deterministic,
    /// Natural IDs
    Natural,
}

impl Default for WebpackChunkIds {
    fn default() -> Self {
        Self::Deterministic
    }
}

/// Webpack dev server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackDevServerConfig {
    /// Server host
    #[serde(default = "default_host")]
    pub host: String,
    /// Server port
    #[serde(default = "default_port")]
    pub port: u16,
    /// Hot mode
    #[serde(default)]
    pub hot: bool,
    /// Live reload
    #[serde(default = "default_live_reload")]
    pub live_reload: bool,
    /// Open browser
    #[serde(default)]
    pub open: bool,
    /// Static files
    #[serde(default = "default_static_files")]
    pub static_files: Vec<PathBuf>,
    /// History API fallback
    #[serde(default = "default_history_api_fallback")]
    pub history_api_fallback: bool,
    /// Proxy configuration
    #[serde(default)]
    pub proxy: HashMap<String, WebpackProxyConfig>,
}

fn default_live_reload() -> bool {
    true
}

fn default_history_api_fallback() -> bool {
    true
}

fn default_static_files() -> Vec<PathBuf> {
    vec![PathBuf::from("public")]
}

impl Default for WebpackDevServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            hot: true,
            live_reload: default_live_reload(),
            open: false,
            static_files: default_static_files(),
            history_api_fallback: default_history_api_fallback(),
            proxy: HashMap::new(),
        }
    }
}

/// Webpack proxy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackProxyConfig {
    /// Target URL
    pub target: String,
    /// Change origin
    #[serde(default)]
    pub change_origin: bool,
    /// Secure flag
    #[serde(default = "default_secure")]
    pub secure: bool,
}

/// Webpack performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackPerformanceConfig {
    /// Enable performance hints
    #[serde(default = "default_hints")]
    pub hints: String,
    /// Maximum entrypoint size
    #[serde(default = "default_max_entrypoint_size")]
    pub max_entrypoint_size: usize,
    /// Maximum asset size
    #[serde(default = "default_max_asset_size")]
    pub max_asset_size: usize,
}

fn default_hints() -> String {
    "warning".to_string()
}

fn default_max_entrypoint_size() -> usize {
    250000
}

fn default_max_asset_size() -> usize {
    250000
}

impl Default for WebpackPerformanceConfig {
    fn default() -> Self {
        Self {
            hints: default_hints(),
            max_entrypoint_size: default_max_entrypoint_size(),
            max_asset_size: default_max_asset_size(),
        }
    }
}

/// Webpack stats configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebpackStatsConfig {
    /// Colors
    #[serde(default = "default_colors")]
    pub colors: bool,
    /// Modules
    #[serde(default)]
    pub modules: bool,
    /// Chunks
    #[serde(default)]
    pub chunks: bool,
    /// Assets
    #[serde(default)]
    pub assets: bool,
    /// Errors
    #[serde(default = "default_errors")]
    pub errors: bool,
    /// Warnings
    #[serde(default = "default_warnings")]
    pub warnings: bool,
}

fn default_colors() -> bool {
    true
}

fn default_errors() -> bool {
    true
}

fn default_warnings() -> bool {
    true
}

impl Default for WebpackStatsConfig {
    fn default() -> Self {
        Self {
            colors: default_colors(),
            modules: false,
            chunks: false,
            assets: true,
            errors: default_errors(),
            warnings: default_warnings(),
        }
    }
}

// ============================================================================
// Esbuild Configuration
// ============================================================================

/// Complete Esbuild configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EsbuildConfig {
    /// Entry points
    #[serde(default)]
    pub entry_points: Vec<String>,
    /// Bundle
    #[serde(default = "default_bundle")]
    pub bundle: bool,
    /// Output directory
    #[serde(default)]
    pub outdir: Option<PathBuf>,
    /// Output file
    #[serde(default)]
    pub outfile: Option<PathBuf>,
    /// Output format
    #[serde(default)]
    pub format: BundleFormat,
    /// Target environment
    #[serde(default = "default_target")]
    pub target: String,
    /// Platform
    #[serde(default = "default_platform")]
    pub platform: String,
    /// External modules
    #[serde(default)]
    pub external: Vec<String>,
    /// Inject
    #[serde(default)]
    pub inject: Vec<String>,
    /// Define constants
    #[serde(default)]
    pub define: HashMap<String, String>,
    /// Banner
    #[serde(default)]
    pub banner: HashMap<String, String>,
    /// Footer
    #[serde(default)]
    pub footer: HashMap<String, String>,
    /// Minify
    #[serde(default)]
    pub minify: bool,
    /// Minify whitespace
    #[serde(default)]
    pub minify_whitespace: bool,
    /// Minify identifiers
    #[serde(default)]
    pub minify_identifiers: bool,
    /// Minify syntax
    #[serde(default)]
    pub minify_syntax: bool,
    /// Tree shaking
    #[serde(default = "default_tree_shaking")]
    pub tree_shaking: bool,
    /// Source map
    #[serde(default)]
    pub sourcemap: SourceMapType,
    /// Source root
    #[serde(default)]
    pub sourceroot: Option<String>,
    /// Source file
    #[serde(default)]
    pub sourcefile: Option<String>,
    /// Metafile
    #[serde(default)]
    pub metafile: bool,
    /// Preserve symlinks
    #[serde(default)]
    pub preserve_symlinks: bool,
    /// Working directory
    #[serde(default)]
    pub working_directory: Option<PathBuf>,
    /// Abspath working directory
    #[serde(default)]
    pub abs_working_dir: Option<PathBuf>,
    /// Public path
    #[serde(default)]
    pub public_path: Option<String>,
    /// JSX factory
    #[serde(default)]
    pub jsx_factory: Option<String>,
    /// JSX fragment
    #[serde(default)]
    pub jsx_fragment: Option<String>,
    /// JSX import source
    #[serde(default)]
    pub jsx_import_source: Option<String>,
    /// JSX development
    #[serde(default)]
    pub jsx_dev: bool,
    /// JSX mode
    #[serde(default)]
    pub jsx_mode: Option<String>,
    /// CSS
    #[serde(default)]
    pub css: bool,
    /// Loader configuration
    #[serde(default)]
    pub loader: HashMap<String, EsbuildLoader>,
    /// Plugins
    #[serde(default)]
    pub plugins: Vec<EsbuildPlugin>,
    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Error limit
    #[serde(default)]
    pub error_limit: Option<usize>,
    /// Charset
    #[serde(default = "default_charset")]
    pub charset: String,
}

fn default_platform() -> String {
    "browser".to_string()
}

fn default_tree_shaking() -> bool {
    true
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_charset() -> String {
    "utf8".to_string()
}

impl Default for EsbuildConfig {
    fn default() -> Self {
        Self {
            entry_points: vec!["src/index.ts".to_string()],
            bundle: default_bundle(),
            outdir: Some(default_out_dir()),
            outfile: None,
            format: BundleFormat::Esm,
            target: default_target(),
            platform: default_platform(),
            external: Vec::new(),
            inject: Vec::new(),
            define: HashMap::new(),
            banner: HashMap::new(),
            footer: HashMap::new(),
            minify: false,
            minify_whitespace: false,
            minify_identifiers: false,
            minify_syntax: false,
            tree_shaking: default_tree_shaking(),
            sourcemap: SourceMapType::None,
            sourceroot: None,
            sourcefile: None,
            metafile: false,
            preserve_symlinks: false,
            working_directory: None,
            abs_working_dir: None,
            public_path: None,
            jsx_factory: None,
            jsx_fragment: None,
            jsx_import_source: None,
            jsx_dev: false,
            jsx_mode: None,
            css: true,
            loader: HashMap::new(),
            plugins: Vec::new(),
            log_level: default_log_level(),
            error_limit: None,
            charset: default_charset(),
        }
    }
}

/// Esbuild loader
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EsbuildLoader {
    /// JavaScript loader
    Js,
    /// JSX loader
    Jsx,
    /// TypeScript loader
    Ts,
    /// TSX loader
    Tsx,
    /// CSS loader
    Css,
    /// JSON loader
    Json,
    /// Text loader
    Text,
    /// Base64 loader
    Base64,
    /// Data URL loader
    Dataurl,
    /// File loader
    File,
    /// Binary loader
    Binary,
    /// Copy loader
    Copy,
    /// Empty loader
    Empty,
}

/// Esbuild plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EsbuildPlugin {
    /// Plugin name
    pub name: String,
    /// Setup function
    #[serde(default)]
    pub setup: String,
}

/// Built-in Esbuild plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EsbuildBuiltinPlugin {
    /// Node modules plugin
    NodeModules,
    /// CDN plugin
    Cdn,
    /// Globals plugin
    Globals,
    /// CSS modules plugin
    CssModules,
    /// SVG plugin
    Svg,
    /// Wasm plugin
    Wasm,
}

// ============================================================================
// Bundle Analysis
// ============================================================================

/// Bundle analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BundleAnalysis {
    /// Total bundle size
    pub total_size: usize,
    /// Individual chunks
    pub chunks: Vec<ChunkInfo>,
    /// Duplicate modules
    pub duplicates: Vec<DuplicateModule>,
    /// Large modules
    pub large_modules: Vec<LargeModule>,
    /// Dependencies
    pub dependencies: Vec<DependencyInfo>,
    /// Asset files
    pub assets: Vec<AssetInfo>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Errors
    pub errors: Vec<String>,
}

/// Chunk information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkInfo {
    /// Chunk name
    pub name: String,
    /// Chunk size
    pub size: usize,
    /// Chunk type
    pub chunk_type: ChunkType,
    /// Modules in chunk
    pub modules: Vec<String>,
    /// Entry point
    pub is_entry: bool,
    /// Initial chunk
    pub is_initial: bool,
}

/// Chunk type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkType {
    /// JavaScript chunk
    JavaScript,
    /// CSS chunk
    Css,
    /// Asset chunk
    Asset,
}

/// Duplicate module information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuplicateModule {
    /// Module name
    pub name: String,
    /// Chunks containing this module
    pub chunks: Vec<String>,
    /// Total wasted size
    pub wasted_size: usize,
}

/// Large module information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LargeModule {
    /// Module name
    pub name: String,
    /// Module size
    pub size: usize,
    /// Percentage of bundle
    pub percentage: f64,
}

/// Dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DependencyInfo {
    /// Dependency name
    pub name: String,
    /// Version
    pub version: String,
    /// Size
    pub size: usize,
    /// License
    pub license: Option<String>,
}

/// Asset information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetInfo {
    /// Asset name
    pub name: String,
    /// Asset size
    pub size: usize,
    /// Asset type
    pub asset_type: String,
}

// ============================================================================
// Bundle Optimization
// ============================================================================

/// Bundle optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BundleOptimization {
    /// Code splitting configuration
    #[serde(default)]
    pub code_splitting: CodeSplittingConfig,
    /// Tree shaking configuration
    #[serde(default)]
    pub tree_shaking: TreeShakingConfig,
    /// Minification configuration
    #[serde(default)]
    pub minification: MinificationConfig,
    /// Compression configuration
    #[serde(default)]
    pub compression: CompressionConfig,
    /// Caching configuration
    #[serde(default)]
    pub caching: CachingConfig,
    /// Lazy loading configuration
    #[serde(default)]
    pub lazy_loading: LazyLoadingConfig,
}

/// Code splitting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeSplittingConfig {
    /// Enable code splitting
    #[serde(default = "default_code_splitting")]
    pub enabled: bool,
    /// Split vendor chunks
    #[serde(default = "default_split_vendor")]
    pub split_vendor: bool,
    /// Split common chunks
    #[serde(default = "default_split_common")]
    pub split_common: bool,
    /// Manual split points
    #[serde(default)]
    pub manual_splits: Vec<SplitPoint>,
    /// Automatic chunk size threshold
    #[serde(default = "default_size_threshold")]
    pub size_threshold: usize,
}

fn default_code_splitting() -> bool {
    true
}

fn default_split_vendor() -> bool {
    true
}

fn default_split_common() -> bool {
    true
}

fn default_size_threshold() -> usize {
    50000
}

impl Default for CodeSplittingConfig {
    fn default() -> Self {
        Self {
            enabled: default_code_splitting(),
            split_vendor: default_split_vendor(),
            split_common: default_split_common(),
            manual_splits: Vec::new(),
            size_threshold: default_size_threshold(),
        }
    }
}

/// Split point configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitPoint {
    /// Module path
    pub module: String,
    /// Chunk name
    pub name: String,
    /// Priority
    #[serde(default)]
    pub priority: i32,
}

/// Tree shaking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TreeShakingConfig {
    /// Enable tree shaking
    #[serde(default = "default_tree_shaking")]
    pub enabled: bool,
    /// Shake side effects
    #[serde(default = "default_side_effects")]
    pub side_effects: bool,
    /// Preserve exports
    #[serde(default)]
    pub preserve_exports: Vec<String>,
}

fn default_side_effects() -> bool {
    true
}

impl Default for TreeShakingConfig {
    fn default() -> Self {
        Self {
            enabled: default_tree_shaking(),
            side_effects: default_side_effects(),
            preserve_exports: Vec::new(),
        }
    }
}

/// Minification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MinificationConfig {
    /// Enable minification
    #[serde(default = "default_minify")]
    pub enabled: bool,
    /// Minify whitespace
    #[serde(default)]
    pub whitespace: bool,
    /// Minify identifiers
    #[serde(default)]
    pub identifiers: bool,
    /// Minify syntax
    #[serde(default)]
    pub syntax: bool,
    /// Mangle names
    #[serde(default)]
    pub mangle: bool,
    /// Remove comments
    #[serde(default = "default_remove_comments")]
    pub remove_comments: bool,
    /// Remove console
    #[serde(default)]
    pub remove_console: bool,
    /// Remove debugger
    #[serde(default)]
    pub remove_debugger: bool,
}

fn default_minify() -> bool {
    true
}

fn default_remove_comments() -> bool {
    true
}

impl Default for MinificationConfig {
    fn default() -> Self {
        Self {
            enabled: default_minify(),
            whitespace: true,
            identifiers: true,
            syntax: true,
            mangle: true,
            remove_comments: default_remove_comments(),
            remove_console: false,
            remove_debugger: true,
        }
    }
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompressionConfig {
    /// Enable compression
    #[serde(default = "default_compression")]
    pub enabled: bool,
    /// Gzip compression
    #[serde(default = "default_gzip")]
    pub gzip: bool,
    /// Brotli compression
    #[serde(default = "default_brotli")]
    pub brotli: bool,
    /// Zstd compression
    #[serde(default)]
    pub zstd: bool,
    /// Compression level
    #[serde(default = "default_compression_level")]
    pub level: u8,
}

fn default_compression() -> bool {
    false
}

fn default_gzip() -> bool {
    true
}

fn default_brotli() -> bool {
    true
}

fn default_compression_level() -> u8 {
    6
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: default_compression(),
            gzip: default_gzip(),
            brotli: default_brotli(),
            zstd: false,
            level: default_compression_level(),
        }
    }
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CachingConfig {
    /// Enable caching
    #[serde(default = "default_caching")]
    pub enabled: bool,
    /// Cache strategy
    #[serde(default)]
    pub strategy: CacheStrategy,
    /// Cache directory
    #[serde(default = "default_cache_dir")]
    pub cache_directory: PathBuf,
    /// Cache version
    #[serde(default)]
    pub version: Option<String>,
    /// Invalidate on change
    #[serde(default = "default_invalidate")]
    pub invalidate_on_change: bool,
}

fn default_caching() -> bool {
    true
}

fn default_cache_dir() -> PathBuf {
    PathBuf::from(".cache")
}

fn default_invalidate() -> bool {
    true
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enabled: default_caching(),
            strategy: Default::default(),
            cache_directory: default_cache_dir(),
            version: None,
            invalidate_on_change: default_invalidate(),
        }
    }
}

/// Cache strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CacheStrategy {
    /// Content-based caching
    ContentHash,
    /// File name-based caching
    FileName,
    /// Time-based caching
    TimeBased,
}

impl Default for CacheStrategy {
    fn default() -> Self {
        Self::ContentHash
    }
}

/// Lazy loading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LazyLoadingConfig {
    /// Enable lazy loading
    #[serde(default = "default_lazy_loading")]
    pub enabled: bool,
    /// Prefetch threshold
    #[serde(default = "default_prefetch_threshold")]
    pub prefetch_threshold: usize,
    /// Preload threshold
    #[serde(default = "default_preload_threshold")]
    pub preload_threshold: usize,
    /// Component-based lazy loading
    #[serde(default)]
    pub component_based: bool,
    /// Route-based lazy loading
    #[serde(default)]
    pub route_based: bool,
}

fn default_lazy_loading() -> bool {
    true
}

fn default_prefetch_threshold() -> usize {
    100000
}

fn default_preload_threshold() -> usize {
    50000
}

impl Default for LazyLoadingConfig {
    fn default() -> Self {
        Self {
            enabled: default_lazy_loading(),
            prefetch_threshold: default_prefetch_threshold(),
            preload_threshold: default_preload_threshold(),
            component_based: true,
            route_based: true,
        }
    }
}

impl Default for BundleOptimization {
    fn default() -> Self {
        Self {
            code_splitting: Default::default(),
            tree_shaking: Default::default(),
            minification: Default::default(),
            compression: Default::default(),
            caching: Default::default(),
            lazy_loading: Default::default(),
        }
    }
}

// ============================================================================
// Builder Implementations
// ============================================================================

/// Vite configuration builder
#[derive(Debug, Clone)]
pub struct ViteConfigBuilder {
    config: ViteConfig,
}

impl ViteConfigBuilder {
    /// Create a new Vite configuration builder
    pub fn new() -> Self {
        Self {
            config: ViteConfig::default(),
        }
    }

    /// Set the project root
    pub fn root(mut self, root: PathBuf) -> Self {
        self.config.root = Some(root);
        self
    }

    /// Set the base path
    pub fn base(mut self, base: String) -> Self {
        self.config.base = base;
        self
    }

    /// Set the public directory
    pub fn public_dir(mut self, public_dir: String) -> Self {
        self.config.public_dir = public_dir;
        self
    }

    /// Set build configuration
    pub fn build(mut self, build: ViteBuildConfig) -> Self {
        self.config.build = build;
        self
    }

    /// Set server configuration
    pub fn server(mut self, server: ViteServerConfig) -> Self {
        self.config.server = server;
        self
    }

    /// Add a plugin
    pub fn plugin(mut self, plugin: VitePlugin) -> Self {
        self.config.plugins.push(plugin);
        self
    }

    /// Add an alias
    pub fn alias(mut self, from: String, to: String) -> Self {
        self.config.resolve.alias.insert(from, to);
        self
    }

    /// Add environment variable
    pub fn env(mut self, key: String, value: String) -> Self {
        self.config.env.insert(key, value);
        self
    }

    /// Add define constant
    pub fn define(mut self, key: String, value: String) -> Self {
        self.config.define.insert(key, value);
        self
    }

    /// Build the configuration
    pub fn build_config(self) -> ViteConfig {
        self.config
    }
}

impl Default for ViteConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Webpack configuration builder
#[derive(Debug, Clone)]
pub struct WebpackConfigBuilder {
    config: WebpackConfig,
}

impl WebpackConfigBuilder {
    /// Create a new Webpack configuration builder
    pub fn new() -> Self {
        Self {
            config: WebpackConfig::default(),
        }
    }

    /// Set entry point
    pub fn entry(mut self, entry: WebpackEntry) -> Self {
        self.config.entry = entry;
        self
    }

    /// Set output configuration
    pub fn output(mut self, output: WebpackOutputConfig) -> Self {
        self.config.output = output;
        self
    }

    /// Set mode
    pub fn mode(mut self, mode: String) -> Self {
        self.config.mode = mode;
        self
    }

    /// Add a rule
    pub fn rule(mut self, rule: WebpackRule) -> Self {
        self.config.module.rules.push(rule);
        self
    }

    /// Add a plugin
    pub fn plugin(mut self, plugin: WebpackPlugin) -> Self {
        self.config.plugins.push(plugin);
        self
    }

    /// Add an alias
    pub fn alias(mut self, from: String, to: String) -> Self {
        self.config.resolve.alias.insert(from, to);
        self
    }

    /// Set optimization configuration
    pub fn optimization(mut self, optimization: WebpackOptimizationConfig) -> Self {
        self.config.optimization = optimization;
        self
    }

    /// Set dev server configuration
    pub fn dev_server(mut self, dev_server: WebpackDevServerConfig) -> Self {
        self.config.dev_server = Some(dev_server);
        self
    }

    /// Build the configuration
    pub fn build_config(self) -> WebpackConfig {
        self.config
    }
}

impl Default for WebpackConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Esbuild configuration builder
#[derive(Debug, Clone)]
pub struct EsbuildConfigBuilder {
    config: EsbuildConfig,
}

impl EsbuildConfigBuilder {
    /// Create a new Esbuild configuration builder
    pub fn new() -> Self {
        Self {
            config: EsbuildConfig::default(),
        }
    }

    /// Add entry point
    pub fn entry_point(mut self, entry: String) -> Self {
        self.config.entry_points.push(entry);
        self
    }

    /// Set output directory
    pub fn outdir(mut self, outdir: PathBuf) -> Self {
        self.config.outdir = Some(outdir);
        self
    }

    /// Set output file
    pub fn outfile(mut self, outfile: PathBuf) -> Self {
        self.config.outfile = Some(outfile);
        self
    }

    /// Set bundle format
    pub fn format(mut self, format: BundleFormat) -> Self {
        self.config.format = format;
        self
    }

    /// Set target
    pub fn target(mut self, target: String) -> Self {
        self.config.target = target;
        self
    }

    /// Enable minification
    pub fn minify(mut self, minify: bool) -> Self {
        self.config.minify = minify;
        self
    }

    /// Set source map type
    pub fn sourcemap(mut self, sourcemap: SourceMapType) -> Self {
        self.config.sourcemap = sourcemap;
        self
    }

    /// Add external module
    pub fn external(mut self, external: String) -> Self {
        self.config.external.push(external);
        self
    }

    /// Add define constant
    pub fn define(mut self, key: String, value: String) -> Self {
        self.config.define.insert(key, value);
        self
    }

    /// Add loader
    pub fn loader(mut self, pattern: String, loader: EsbuildLoader) -> Self {
        self.config.loader.insert(pattern, loader);
        self
    }

    /// Enable metafile
    pub fn metafile(mut self, metafile: bool) -> Self {
        self.config.metafile = metafile;
        self
    }

    /// Build the configuration
    pub fn build_config(self) -> EsbuildConfig {
        self.config
    }
}

impl Default for EsbuildConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vite_config_default() {
        let config = ViteConfig::default();
        assert_eq!(config.base, "/");
        assert_eq!(config.public_dir, "public");
        assert_eq!(config.build.out_dir, PathBuf::from("dist"));
    }

    #[test]
    fn test_vite_config_builder() {
        let config = ViteConfigBuilder::new()
            .base("/app/".to_string())
            .public_dir("static".to_string())
            .plugin(VitePlugin {
                name: "test-plugin".to_string(),
                hooks: Default::default(),
                config: HashMap::new(),
            })
            .alias("@".to_string(), "/src".to_string())
            .env("NODE_ENV".to_string(), "production".to_string())
            .build_config();

        assert_eq!(config.base, "/app/");
        assert_eq!(config.public_dir, "static");
        assert_eq!(config.plugins.len(), 1);
        assert_eq!(config.resolve.alias.get("@"), Some(&"/src".to_string()));
        assert_eq!(
            config.env.get("NODE_ENV"),
            Some(&"production".to_string())
        );
    }

    #[test]
    fn test_webpack_config_default() {
        let config = WebpackConfig::default();
        assert_eq!(config.mode, "production");
        assert_eq!(config.target, "modules");
        assert_eq!(config.output.path, PathBuf::from("dist"));
    }

    #[test]
    fn test_webpack_config_builder() {
        let config = WebpackConfigBuilder::new()
            .entry(WebpackEntry::Single("./src/index.ts".to_string()))
            .mode("development".to_string())
            .rule(WebpackRule::javascript())
            .plugin(WebpackPlugin {
                name: "html-webpack-plugin".to_string(),
                options: HashMap::new(),
            })
            .alias("@".to_string(), "/src".to_string())
            .build_config();

        assert_eq!(config.mode, "development");
        assert_eq!(config.module.rules.len(), 5); // 4 default + 1 added
        assert_eq!(config.plugins.len(), 1);
        assert_eq!(config.resolve.alias.get("@"), Some(&"/src".to_string()));
    }

    #[test]
    fn test_esbuild_config_default() {
        let config = EsbuildConfig::default();
        assert_eq!(config.entry_points, vec!["src/index.ts"]);
        assert!(config.bundle);
        assert!(config.tree_shaking);
    }

    #[test]
    fn test_esbuild_config_builder() {
        let config = EsbuildConfigBuilder::new()
            .entry_point("src/main.ts".to_string())
            .outdir(PathBuf::from("dist"))
            .format(BundleFormat::Iife)
            .minify(true)
            .sourcemap(SourceMapType::External)
            .external("lodash".to_string())
            .define("process.env.NODE_ENV".to_string(), "\"production\"".to_string())
            .loader(".png".to_string(), EsbuildLoader::Dataurl)
            .metafile(true)
            .build_config();

        assert!(config.entry_points.contains(&"src/main.ts".to_string()));
        assert!(config.entry_points.contains(&"src/index.ts".to_string()));
        assert_eq!(config.format, BundleFormat::Iife);
        assert!(config.minify);
        assert_eq!(config.sourcemap, SourceMapType::External);
        assert!(config.external.contains(&"lodash".to_string()));
        assert!(config.metafile);
    }

    #[test]
    fn test_bundle_optimization_default() {
        let opt = BundleOptimization::default();
        assert!(opt.code_splitting.enabled);
        assert!(opt.tree_shaking.enabled);
        assert!(opt.minification.enabled);
        assert!(opt.caching.enabled);
        assert!(opt.lazy_loading.enabled);
    }

    #[test]
    fn test_code_splitting_config() {
        let config = CodeSplittingConfig::default();
        assert!(config.enabled);
        assert!(config.split_vendor);
        assert!(config.split_common);
        assert_eq!(config.size_threshold, 50000);
    }

    #[test]
    fn test_minification_config() {
        let config = MinificationConfig::default();
        assert!(config.enabled);
        assert!(config.whitespace);
        assert!(config.identifiers);
        assert!(config.syntax);
        assert!(config.mangle);
        assert!(config.remove_comments);
        assert!(config.remove_debugger);
    }

    #[test]
    fn test_vite_rule_constructors() {
        let js_rule = WebpackRule::javascript();
        assert_eq!(js_rule.test, r"\.(mjs|jsx?|cjs)$");
        assert_eq!(js_rule.rule_type, WebpackRuleType::Javascript);

        let ts_rule = WebpackRule::typescript();
        assert_eq!(ts_rule.test, r"\.(mts|ts|tsx)$");

        let css_rule = WebpackRule::css();
        assert_eq!(css_rule.test, r"\.(css|less|sass|scss)$");
        assert_eq!(css_rule.rule_type, WebpackRuleType::Css);

        let img_rule = WebpackRule::images();
        assert_eq!(img_rule.test, r"\.(png|jpg|jpeg|gif|svg|webp|avif)$");
        assert_eq!(img_rule.rule_type, WebpackRuleType::Asset);
    }

    #[test]
    fn test_webpack_loader_constructors() {
        let babel = WebpackLoader::babel();
        assert_eq!(babel.name, "babel-loader");
        assert!(babel.options.contains_key("presets"));

        let ts = WebpackLoader::ts_loader();
        assert_eq!(ts.name, "ts-loader");

        let css = WebpackLoader::css_loader();
        assert_eq!(css.name, "css-loader");
    }

    #[test]
    fn test_bundle_format_serialization() {
        let formats = vec![
            (BundleFormat::Esm, "esm"),
            (BundleFormat::Cjs, "cjs"),
            (BundleFormat::Iife, "iife"),
            (BundleFormat::Umd, "umd"),
            (BundleFormat::System, "system"),
        ];

        for (format, expected) in formats {
            let serialized = serde_json::to_string(&format).unwrap();
            assert!(serialized.contains(expected));
        }
    }

    #[test]
    fn test_source_map_type() {
        let types = vec![
            SourceMapType::None,
            SourceMapType::Inline,
            SourceMapType::External,
            SourceMapType::Hidden,
            SourceMapType::Eval,
        ];

        for map_type in types {
            let serialized = serde_json::to_string(&map_type).unwrap();
            let deserialized: SourceMapType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(map_type, deserialized);
        }
    }

    #[test]
    fn test_resolution_strategy() {
        let strategies = vec![
            ResolutionStrategy::Node,
            ResolutionStrategy::Browser,
            ResolutionStrategy::Alloy,
        ];

        for strategy in strategies {
            let serialized = serde_json::to_string(&strategy).unwrap();
            let deserialized: ResolutionStrategy =
                serde_json::from_str(&serialized).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_vite_resolve_config() {
        let config = ViteResolveConfig::default();
        assert!(config.extensions.contains(&".ts".to_string()));
        assert!(config.extensions.contains(&".tsx".to_string()));
        assert!(config.main_fields.contains(&"browser".to_string()));
        assert!(config.main_fields.contains(&"module".to_string()));
    }

    #[test]
    fn test_webpack_resolve_config() {
        let config = WebpackResolveConfig::default();
        assert!(config.extensions.contains(&".ts".to_string()));
        assert!(config.modules.contains(&"node_modules".to_string()));
        assert_eq!(config.strategy, ResolutionStrategy::Browser);
    }

    #[test]
    fn test_optimization_level() {
        let levels = vec![
            OptimizationLevel::None,
            OptimizationLevel::Basic,
            OptimizationLevel::Advanced,
            OptimizationLevel::Aggressive,
        ];

        for level in levels {
            let serialized = serde_json::to_string(&level).unwrap();
            let deserialized: OptimizationLevel = serde_json::from_str(&serialized).unwrap();
            assert_eq!(level, deserialized);
        }
    }

    #[test]
    fn test_vite_minify_type() {
        let types = vec![
            ViteMinifyType::None,
            ViteMinifyType::Esbuild,
            ViteMinifyType::Terser,
        ];

        for minify_type in types {
            let serialized = serde_json::to_string(&minify_type).unwrap();
            let deserialized: ViteMinifyType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(minify_type, deserialized);
        }
    }

    #[test]
    fn test_webpack_runtime_chunk() {
        let chunks = vec![
            WebpackRuntimeChunk::Single,
            WebpackRuntimeChunk::Multiple,
            WebpackRuntimeChunk::False,
        ];

        for chunk in chunks {
            let serialized = serde_json::to_string(&chunk).unwrap();
            let deserialized: WebpackRuntimeChunk = serde_json::from_str(&serialized).unwrap();
            assert_eq!(chunk, deserialized);
        }
    }

    #[test]
    fn test_cache_strategy() {
        let strategies = vec![
            CacheStrategy::ContentHash,
            CacheStrategy::FileName,
            CacheStrategy::TimeBased,
        ];

        for strategy in strategies {
            let serialized = serde_json::to_string(&strategy).unwrap();
            let deserialized: CacheStrategy = serde_json::from_str(&serialized).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_chunk_type() {
        let types = vec![
            ChunkType::JavaScript,
            ChunkType::Css,
            ChunkType::Asset,
        ];

        for chunk_type in types {
            let serialized = serde_json::to_string(&chunk_type).unwrap();
            let deserialized: ChunkType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(chunk_type, deserialized);
        }
    }

    #[test]
    fn test_target_environment() {
        let environments = vec![
            TargetEnvironment::Modern,
            TargetEnvironment::Legacy,
            TargetEnvironment::Node,
            TargetEnvironment::Electron,
            TargetEnvironment::Deno,
        ];

        for env in environments {
            let serialized = serde_json::to_string(&env).unwrap();
            let deserialized: TargetEnvironment = serde_json::from_str(&serialized).unwrap();
            assert_eq!(env, deserialized);
        }
    }

    #[test]
    fn test_esbuild_loader() {
        let loaders = vec![
            (EsbuildLoader::Js, "js"),
            (EsbuildLoader::Jsx, "jsx"),
            (EsbuildLoader::Ts, "ts"),
            (EsbuildLoader::Tsx, "tsx"),
            (EsbuildLoader::Css, "css"),
            (EsbuildLoader::Json, "json"),
            (EsbuildLoader::Text, "text"),
            (EsbuildLoader::Base64, "base64"),
            (EsbuildLoader::Dataurl, "dataurl"),
            (EsbuildLoader::File, "file"),
            (EsbuildLoader::Binary, "binary"),
            (EsbuildLoader::Copy, "copy"),
            (EsbuildLoader::Empty, "empty"),
        ];

        for (loader, expected) in loaders {
            let serialized = serde_json::to_string(&loader).unwrap();
            assert!(serialized.contains(expected));
        }
    }

    #[test]
    fn test_vite_server_config() {
        let config = ViteServerConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 5173);
        assert!(config.hmr.enabled);
        assert_eq!(config.hmr.protocol, "ws");
    }

    #[test]
    fn test_vite_preview_config() {
        let config = VitePreviewConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 4173);
    }

    #[test]
    fn test_webpack_dev_server_config() {
        let config = WebpackDevServerConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 5173);
        assert!(config.hot);
        assert!(config.live_reload);
        assert!(config.history_api_fallback);
    }

    #[test]
    fn test_vite_assets_config() {
        let config = ViteAssetsConfig::default();
        assert_eq!(config.inline_limit, 4096);
        assert!(config.assets_include.contains(&"*.png".to_string()));
        assert!(config.assets_include.contains(&"*.svg".to_string()));
    }

    #[test]
    fn test_vite_optimize_deps_config() {
        let config = ViteOptimizeDepsConfig::default();
        assert!(config.include.is_empty());
        assert!(config.exclude.is_empty());
        assert!(config.force.is_none());
    }

    #[test]
    fn test_vite_css_config() {
        let config = ViteCssConfig::default();
        assert_eq!(config.modules.localsConvention, "camelCase");
        assert_eq!(config.modules.scope_behaviour, "global");
    }

    #[test]
    fn test_webpack_split_chunks_config() {
        let config = WebpackSplitChunksConfig::default();
        assert!(matches!(config.chunks, WebpackChunksSelection::Async));
        assert!(config.cache_groups.contains_key("default"));
        assert!(config.cache_groups.contains_key("vendors"));
    }

    #[test]
    fn test_webpack_performance_config() {
        let config = WebpackPerformanceConfig::default();
        assert_eq!(config.hints, "warning");
        assert_eq!(config.max_entrypoint_size, 250000);
        assert_eq!(config.max_asset_size, 250000);
    }

    #[test]
    fn test_webpack_stats_config() {
        let config = WebpackStatsConfig::default();
        assert!(config.colors);
        assert!(config.errors);
        assert!(config.warnings);
        assert!(config.assets);
    }

    #[test]
    fn test_lazy_loading_config() {
        let config = LazyLoadingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.prefetch_threshold, 100000);
        assert_eq!(config.preload_threshold, 50000);
        assert!(config.component_based);
        assert!(config.route_based);
    }

    #[test]
    fn test_compression_config() {
        let config = CompressionConfig::default();
        assert!(!config.enabled);
        assert!(config.gzip);
        assert!(config.brotli);
        assert_eq!(config.level, 6);
    }

    #[test]
    fn test_caching_config() {
        let config = CachingConfig::default();
        assert!(config.enabled);
        assert!(matches!(config.strategy, CacheStrategy::ContentHash));
        assert_eq!(config.cache_directory, PathBuf::from(".cache"));
        assert!(config.invalidate_on_change);
    }
}
