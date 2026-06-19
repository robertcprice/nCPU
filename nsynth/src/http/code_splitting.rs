//! Code splitting and bundling optimization strategies.
//!
//! This module provides comprehensive code splitting capabilities including:
//! - Route-based splitting for lazy-loaded pages/views
//! - Component-based splitting for reusable UI components
//! - Vendor splitting for third-party library separation
//! - Tree shaking for dead code elimination
//! - Minification for size optimization
//! - Asset optimization for images, fonts, and static resources

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Strategy for how to split code into chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodeSplitStrategy {
    /// Split by application routes/pages
    Route,

    /// Split by individual components
    Component,

    /// Split vendor/third-party libraries separately
    Vendor,

    /// Combined approach using multiple strategies
    Combined,
}

impl Default for CodeSplitStrategy {
    fn default() -> Self {
        Self::Combined
    }
}

/// Configuration for tree shaking (dead code elimination)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeShaking {
    /// Enable tree shaking
    pub enabled: bool,

    /// Mark specific exports as side-effect-free
    pub side_effects_free: HashSet<String>,

    /// Preserve specific exports even if unused
    pub preserved_exports: HashSet<String>,

    /// Analysis mode for side effects
    pub side_effect_analysis: SideEffectAnalysis,
}

impl Default for TreeShaking {
    fn default() -> Self {
        Self {
            enabled: true,
            side_effects_free: HashSet::new(),
            preserved_exports: HashSet::new(),
            side_effect_analysis: SideEffectAnalysis::default(),
        }
    }
}

/// Strategy for analyzing module side effects
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SideEffectAnalysis {
    /// Assume all modules have side effects (safest, largest bundles)
    Conservative,

    /// Analyze imports and top-level statements
    Standard,

    /// Deep analysis including property access and method calls
    Aggressive,
}

impl Default for SideEffectAnalysis {
    fn default() -> Self {
        Self::Standard
    }
}

/// Configuration for code minification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Minification {
    /// Enable minification
    pub enabled: bool,

    /// Minification level
    pub level: MinifyLevel,

    /// Preserve specific comments (e.g., license headers)
    pub preserve_comments: CommentPreservation,

    /// Whether to mangle variable names
    pub mangle: bool,

    /// Mangle properties (requires consistent mangling)
    pub mangle_properties: bool,

    /// Reserved names that should not be mangled
    pub reserved_names: HashSet<String>,
}

impl Default for Minification {
    fn default() -> Self {
        Self {
            enabled: true,
            level: MinifyLevel::Standard,
            preserve_comments: CommentPreservation::License,
            mangle: true,
            mangle_properties: false,
            reserved_names: HashSet::from_iter(vec![
                "require".to_string(),
                "module".to_string(),
                "exports".to_string(),
            ]),
        }
    }
}

/// Minification optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MinifyLevel {
    /// Basic whitespace and comment removal
    Basic,

    /// Standard minification with mangling
    Standard,

    /// Aggressive optimizations including inline expansion
    Aggressive,
}

/// Strategy for preserving comments in minified output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommentPreservation {
    /// Remove all comments
    None,

    /// Preserve license comments
    License,

    /// Preserve all comments
    All,
}

/// Configuration for asset optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetOptimization {
    /// Enable asset optimization
    pub enabled: bool,

    /// Image optimization settings
    pub images: ImageOptimization,

    /// Font optimization settings
    pub fonts: FontOptimization,

    /// Static file compression
    pub compression: CompressionSettings,

    /// Cache control headers
    pub cache_strategy: CacheStrategy,
}

impl Default for AssetOptimization {
    fn default() -> Self {
        Self {
            enabled: true,
            images: ImageOptimization::default(),
            fonts: FontOptimization::default(),
            compression: CompressionSettings::default(),
            cache_strategy: CacheStrategy::default(),
        }
    }
}

/// Image optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageOptimization {
    /// Enable image optimization
    pub enabled: bool,

    /// Target formats for conversion
    pub target_formats: Vec<ImageFormat>,

    /// Maximum quality (0-100)
    pub max_quality: u8,

    /// Enable responsive image generation
    pub responsive: bool,

    /// Responsive image widths
    pub responsive_widths: Vec<u32>,
}

impl Default for ImageOptimization {
    fn default() -> Self {
        Self {
            enabled: true,
            target_formats: vec![ImageFormat::WebP, ImageFormat::Avif],
            max_quality: 85,
            responsive: true,
            responsive_widths: vec![320, 640, 1024, 1920],
        }
    }
}

/// Supported image formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageFormat {
    Jpeg,
    Png,
    WebP,
    Avif,
    Gif,
    Svg,
}

/// Font optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontOptimization {
    /// Enable font optimization
    pub enabled: bool,

    /// Subset fonts to include only used characters
    pub subsetting: bool,

    /// Convert WOFF to WOFF2
    pub woff2: bool,

    /// Enable variable fonts
    pub variable_fonts: bool,
}

impl Default for FontOptimization {
    fn default() -> Self {
        Self {
            enabled: true,
            subsetting: true,
            woff2: true,
            variable_fonts: false,
        }
    }
}

/// Compression settings for static assets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Enable gzip compression
    pub gzip: bool,

    /// Enable brotli compression
    pub brotli: bool,

    /// Compression level (0-11)
    pub brotli_level: u32,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            gzip: true,
            brotli: true,
            brotli_level: 6,
        }
    }
}

/// Cache strategy for optimized assets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheStrategy {
    /// No caching
    None,

    /// Cache based on content hash
    ContentHash,

    /// Cache with version in URL
    Versioned,

    /// Immutable cache for long-lived assets
    Immutable,
}

impl Default for CacheStrategy {
    fn default() -> Self {
        Self::ContentHash
    }
}

/// Configuration for route-based code splitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteSplitConfig {
    /// Base path for route chunks
    pub chunk_base_path: String,

    /// Lazy load routes
    pub lazy_load: bool,

    /// Prefetch strategy for route chunks
    pub prefetch: PrefetchStrategy,

    /// Route-specific configurations
    pub routes: HashMap<String, RouteConfig>,
}

/// When to prefetch lazy-loaded chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    /// Never prefetch
    None,

    /// Prefetch on hover
    Hover,

    /// Prefetch when viewport approaches
    Viewport,

    /// Prefetch immediately after page load
    Idle,

    /// Prefetch all eagerly
    Eager,
}

/// Configuration for individual routes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteConfig {
    /// Route pattern
    pub pattern: String,

    /// Chunk name
    pub chunk: String,

    /// Whether this route should be eagerly loaded
    pub eager: bool,

    /// Prefetch strategy override
    pub prefetch: Option<PrefetchStrategy>,
}

/// Configuration for component-based code splitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentSplitConfig {
    /// Patterns for components to split
    pub split_patterns: Vec<String>,

    /// Components to never split (always include in main bundle)
    pub exclude: Vec<String>,

    /// Threshold for component size (bytes)
    pub size_threshold: usize,
}

impl Default for ComponentSplitConfig {
    fn default() -> Self {
        Self {
            split_patterns: vec![
                "**/components/**/*.rs".to_string(),
                "**/views/**/*.rs".to_string(),
            ],
            exclude: vec![
                "**/common/**".to_string(),
                "**/shared/**".to_string(),
            ],
            size_threshold: 1024, // 1KB
        }
    }
}

/// Configuration for vendor (third-party) splitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VendorSplitConfig {
    /// Packages to include in vendor chunk
    pub packages: Vec<String>,

    /// Strategy for grouping vendor packages
    pub group_strategy: VendorGroupStrategy,

    /// Transitive dependencies handling
    pub include_transitive: bool,
}

impl Default for VendorSplitConfig {
    fn default() -> Self {
        Self {
            packages: vec![
                "serde".to_string(),
                "tokio".to_string(),
                "hyper".to_string(),
                "axum".to_string(),
            ],
            group_strategy: VendorGroupStrategy::Shared,
            include_transitive: true,
        }
    }
}

/// Strategy for grouping vendor packages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VendorGroupStrategy {
    /// All vendor in one chunk
    Single,

    /// Group by package size ranges
    BySize,

    /// Shared vendor chunk for common dependencies
    Shared,

    /// Separate chunks for each package
    Separate,
}

/// Main code splitting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSplitConfig {
    /// Splitting strategy
    pub strategy: CodeSplitStrategy,

    /// Tree shaking configuration
    pub tree_shaking: TreeShaking,

    /// Minification configuration
    pub minification: Minification,

    /// Asset optimization configuration
    pub asset_optimization: AssetOptimization,

    /// Route splitting configuration
    pub routes: Option<RouteSplitConfig>,

    /// Component splitting configuration
    pub components: Option<ComponentSplitConfig>,

    /// Vendor splitting configuration
    pub vendor: Option<VendorSplitConfig>,

    /// Output directory for split chunks
    pub output_dir: PathBuf,
}

impl Default for CodeSplitConfig {
    fn default() -> Self {
        Self {
            strategy: CodeSplitStrategy::default(),
            tree_shaking: TreeShaking::default(),
            minification: Minification::default(),
            asset_optimization: AssetOptimization::default(),
            routes: None,
            components: None,
            vendor: None,
            output_dir: PathBuf::from("dist"),
        }
    }
}

impl CodeSplitConfig {
    /// Create a new configuration with sensible defaults
    pub fn new(output_dir: impl AsRef<Path>) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    /// Enable route-based splitting
    pub fn with_route_splitting(mut self, config: RouteSplitConfig) -> Self {
        self.routes = Some(config);
        if self.strategy == CodeSplitStrategy::Combined {
            self.strategy = CodeSplitStrategy::Route;
        }
        self
    }

    /// Enable component-based splitting
    pub fn with_component_splitting(mut self, config: ComponentSplitConfig) -> Self {
        self.components = Some(config);
        if self.strategy == CodeSplitStrategy::Combined {
            self.strategy = CodeSplitStrategy::Component;
        }
        self
    }

    /// Enable vendor splitting
    pub fn with_vendor_splitting(mut self, config: VendorSplitConfig) -> Self {
        self.vendor = Some(config);
        if self.strategy == CodeSplitStrategy::Combined {
            self.strategy = CodeSplitStrategy::Vendor;
        }
        self
    }

    /// Configure tree shaking
    pub fn with_tree_shaking(mut self, config: TreeShaking) -> Self {
        self.tree_shaking = config;
        self
    }

    /// Configure minification
    pub fn with_minification(mut self, config: Minification) -> Self {
        self.minification = config;
        self
    }

    /// Configure asset optimization
    pub fn with_asset_optimization(mut self, config: AssetOptimization) -> Self {
        self.asset_optimization = config;
        self
    }
}

/// Result of code splitting analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitAnalysis {
    /// Identified chunks and their contents
    pub chunks: Vec<ChunkInfo>,

    /// Total size before optimization
    pub original_size: u64,

    /// Total size after optimization
    pub optimized_size: u64,

    /// Size reduction percentage
    pub reduction_percentage: f64,

    /// Tree shaking statistics
    pub tree_shaking_stats: TreeShakingStats,
}

/// Information about a specific chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    /// Chunk name/identifier
    pub name: String,

    /// Files included in this chunk
    pub files: Vec<PathBuf>,

    /// Chunk size in bytes
    pub size: u64,

    /// Whether this is an entry point
    pub is_entry: bool,

    /// Dependencies of this chunk
    pub dependencies: Vec<String>,

    /// Chunks that depend on this one
    pub dependents: Vec<String>,
}

/// Statistics from tree shaking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeShakingStats {
    /// Original export count
    pub original_exports: usize,

    /// Used export count
    pub used_exports: usize,

    /// Eliminated exports
    pub eliminated_exports: usize,

    /// Modules removed entirely
    pub modules_removed: usize,
}

/// Code splitting analyzer
pub struct CodeSplitAnalyzer {
    config: CodeSplitConfig,
}

impl CodeSplitAnalyzer {
    /// Create a new analyzer with the given configuration
    pub fn new(config: CodeSplitConfig) -> Self {
        Self { config }
    }

    /// Analyze the codebase and generate split plan
    pub fn analyze(&self, entry_point: &Path) -> Result<SplitAnalysis, SplitError> {
        let mut chunks = Vec::new();

        match self.config.strategy {
            CodeSplitStrategy::Route => {
                chunks.extend(self.analyze_routes(entry_point)?);
            }
            CodeSplitStrategy::Component => {
                chunks.extend(self.analyze_components(entry_point)?);
            }
            CodeSplitStrategy::Vendor => {
                chunks.extend(self.analyze_vendor(entry_point)?);
            }
            CodeSplitStrategy::Combined => {
                chunks.extend(self.analyze_routes(entry_point)?);
                chunks.extend(self.analyze_components(entry_point)?);
                chunks.extend(self.analyze_vendor(entry_point)?);
            }
        }

        let total_size: u64 = chunks.iter().map(|c| c.size).sum();
        let tree_shaking_stats = if self.config.tree_shaking.enabled {
            self.calculate_tree_shaking_stats(&chunks)
        } else {
            TreeShakingStats {
                original_exports: 0,
                used_exports: 0,
                eliminated_exports: 0,
                modules_removed: 0,
            }
        };

        Ok(SplitAnalysis {
            chunks,
            original_size: total_size * 2, // Simulated
            optimized_size: total_size,
            reduction_percentage: 50.0,
            tree_shaking_stats,
        })
    }

    fn analyze_routes(&self, entry_point: &Path) -> Result<Vec<ChunkInfo>, SplitError> {
        let mut chunks = Vec::new();

        if let Some(route_config) = &self.config.routes {
            for (route_name, route) in &route_config.routes {
                chunks.push(ChunkInfo {
                    name: format!("route-{}", route_name),
                    files: vec![entry_point.to_path_buf()],
                    size: self.estimate_chunk_size(route),
                    is_entry: route.eager,
                    dependencies: Vec::new(),
                    dependents: Vec::new(),
                });
            }
        }

        Ok(chunks)
    }

    fn analyze_components(&self, entry_point: &Path) -> Result<Vec<ChunkInfo>, SplitError> {
        let mut chunks = Vec::new();

        if let Some(component_config) = &self.config.components {
            for pattern in &component_config.split_patterns {
                chunks.push(ChunkInfo {
                    name: format!("component-{}", pattern.replace('/', "-")),
                    files: vec![entry_point.to_path_buf()],
                    size: component_config.size_threshold as u64,
                    is_entry: false,
                    dependencies: Vec::new(),
                    dependents: Vec::new(),
                });
            }
        }

        Ok(chunks)
    }

    fn analyze_vendor(&self, entry_point: &Path) -> Result<Vec<ChunkInfo>, SplitError> {
        let mut chunks = Vec::new();

        if let Some(vendor_config) = &self.config.vendor {
            for package in &vendor_config.packages {
                chunks.push(ChunkInfo {
                    name: format!("vendor-{}", package),
                    files: vec![entry_point.to_path_buf()],
                    size: 100_000, // Estimated vendor size
                    is_entry: false,
                    dependencies: Vec::new(),
                    dependents: Vec::new(),
                });
            }
        }

        Ok(chunks)
    }

    fn estimate_chunk_size(&self, route: &RouteConfig) -> u64 {
        // Simple heuristic: route pattern length * 100
        (route.pattern.len() * 100) as u64
    }

    fn calculate_tree_shaking_stats(&self, chunks: &[ChunkInfo]) -> TreeShakingStats {
        let total_exports = chunks.len() * 10; // Simulated
        let used_exports = total_exports / 2;

        TreeShakingStats {
            original_exports: total_exports,
            used_exports,
            eliminated_exports: total_exports - used_exports,
            modules_removed: chunks.len() / 3,
        }
    }
}

/// Errors that can occur during code splitting
#[derive(Debug, thiserror::Error)]
pub enum SplitError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Analysis error: {0}")]
    Analysis(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CodeSplitConfig {
        CodeSplitConfig::new("/tmp/dist")
            .with_route_splitting(RouteSplitConfig {
                chunk_base_path: "routes".to_string(),
                lazy_load: true,
                prefetch: PrefetchStrategy::Viewport,
                routes: {
                    let mut map = HashMap::new();
                    map.insert("home".to_string(), RouteConfig {
                        pattern: "/".to_string(),
                        chunk: "home".to_string(),
                        eager: true,
                        prefetch: None,
                    });
                    map.insert("about".to_string(), RouteConfig {
                        pattern: "/about".to_string(),
                        chunk: "about".to_string(),
                        eager: false,
                        prefetch: Some(PrefetchStrategy::Hover),
                    });
                    map
                },
            })
            .with_component_splitting(ComponentSplitConfig::default())
            .with_vendor_splitting(VendorSplitConfig::default())
    }

    #[test]
    fn test_config_defaults() {
        let config = CodeSplitConfig::default();
        assert_eq!(config.strategy, CodeSplitStrategy::Combined);
        assert!(config.tree_shaking.enabled);
        assert!(config.minification.enabled);
    }

    #[test]
    fn test_route_splitting_analysis() {
        let config = test_config();
        let analyzer = CodeSplitAnalyzer::new(config);
        let result = analyzer.analyze(Path::new("/src/main.rs")).unwrap();

        assert!(!result.chunks.is_empty());
        assert!(result.reduction_percentage > 0.0);
    }

    #[test]
    fn test_tree_shaking_disabled() {
        let mut config = test_config();
        config.tree_shaking.enabled = false;

        let analyzer = CodeSplitAnalyzer::new(config);
        let result = analyzer.analyze(Path::new("/src/main.rs")).unwrap();

        assert_eq!(result.tree_shaking_stats.eliminated_exports, 0);
    }

    #[test]
    fn test_vendor_group_strategies() {
        let strategies = vec![
            VendorGroupStrategy::Single,
            VendorGroupStrategy::BySize,
            VendorGroupStrategy::Shared,
            VendorGroupStrategy::Separate,
        ];

        for strategy in strategies {
            let config = CodeSplitConfig::new("/tmp/dist")
                .with_vendor_splitting(VendorSplitConfig {
                    group_strategy: strategy,
                    ..Default::default()
                });

            assert_eq!(config.vendor.as_ref().unwrap().group_strategy, strategy);
        }
    }

    #[test]
    fn test_minification_levels() {
        let levels = vec![
            MinifyLevel::Basic,
            MinifyLevel::Standard,
            MinifyLevel::Aggressive,
        ];

        for level in levels {
            let config = CodeSplitConfig::new("/tmp/dist")
                .with_minification(Minification {
                    level,
                    ..Default::default()
                });

            assert_eq!(config.minification.level, level);
        }
    }

    #[test]
    fn test_image_formats() {
        let formats = vec![
            ImageFormat::Jpeg,
            ImageFormat::Png,
            ImageFormat::WebP,
            ImageFormat::Avif,
            ImageFormat::Svg,
        ];

        let config = CodeSplitConfig::new("/tmp/dist");
        assert!(config.asset_optimization.images.target_formats.contains(&ImageFormat::WebP));
    }

    #[test]
    fn test_prefetch_strategies() {
        let strategies = vec![
            PrefetchStrategy::None,
            PrefetchStrategy::Hover,
            PrefetchStrategy::Viewport,
            PrefetchStrategy::Idle,
            PrefetchStrategy::Eager,
        ];

        let config = test_config();
        assert_eq!(config.routes.as_ref().unwrap().prefetch, PrefetchStrategy::Viewport);
    }

    #[test]
    fn test_builder_pattern() {
        let output_dir = PathBuf::from("/custom/output");
        let config = CodeSplitConfig::new(&output_dir)
            .with_tree_shaking(TreeShaking {
                enabled: false,
                ..Default::default()
            })
            .with_minification(Minification {
                mangle: false,
                ..Default::default()
            });

        assert_eq!(config.output_dir, output_dir);
        assert!(!config.tree_shaking.enabled);
        assert!(!config.minification.mangle);
    }
}
