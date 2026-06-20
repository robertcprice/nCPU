//! Progressive Web App (PWA) Support for nCPU/nSynth
//!
//! Comprehensive PWA implementation including:
//! - Service Worker lifecycle and registration
//! - Multiple caching strategies (CacheFirst, NetworkFirst, StaleWhileRevalidate, etc.)
//! - Push notification subscriptions and management
//! - Web App Manifest generation and serving
//! - Background sync capabilities
//! - Periodic background sync
//! - App shortcuts and install prompts

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Service Worker Core Types
// ============================================================================

/// Service Worker lifecycle states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceWorkerState {
    /// Parsing the worker script
    Parsing,
    /// Worker is installing
    Installing,
    /// Worker is waiting to activate
    Waiting,
    /// Worker is active and controlling pages
    Activated,
    /// Worker is redundant (replaced or failed)
    Redundant,
}

/// Service Worker registration
#[derive(Debug, Clone)]
pub struct ServiceWorkerRegistration {
    /// Unique registration ID
    pub id: String,
    /// Scope of URLs this worker controls
    pub scope: String,
    /// URL to the worker script
    pub script_url: String,
    /// Current state of the worker
    pub state: ServiceWorkerState,
    /// When the worker was last updated
    pub last_updated: i64,
    /// Whether the worker is currently controlling clients
    pub is_controlling: bool,
}

impl ServiceWorkerRegistration {
    /// Create a new service worker registration
    pub fn new(scope: impl Into<String>, script_url: impl Into<String>) -> Self {
        Self {
            id: uuid_v4(),
            scope: scope.into(),
            script_url: script_url.into(),
            state: ServiceWorkerState::Parsing,
            last_updated: chrono_timestamp(),
            is_controlling: false,
        }
    }

    /// Update the worker state
    pub fn update_state(&mut self, new_state: ServiceWorkerState) {
        self.state = new_state;
        self.last_updated = chrono_timestamp();
    }

    /// Mark as controlling pages
    pub fn set_controlling(&mut self, controlling: bool) {
        self.is_controlling = controlling;
    }
}

/// Service Worker client (controlled page)
#[derive(Debug, Clone)]
pub struct ServiceWorkerClient {
    /// Unique client ID
    pub id: String,
    /// URL of the controlled page
    pub url: String,
    /// Client visibility state
    pub visibility_state: VisibilityState,
    /// Frame type (top-level, nested, etc.)
    pub frame_type: FrameType,
}

/// Visibility state of a client
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisibilityState {
    Visible,
    Hidden,
}

/// Frame type of the client
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    TopLevel,
    Nested,
    Auxiliary,
    None,
}

// ============================================================================
// Cache Strategies
// ============================================================================

/// Cache strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategyType {
    /// Check cache first, fall back to network
    CacheFirst,
    /// Check network first, fall back to cache
    NetworkFirst,
    /// Serve from cache, update in background
    StaleWhileRevalidate,
    /// Network only, no caching
    NetworkOnly,
    /// Cache only, no network requests
    CacheOnly,
    /// Try cache and network in parallel, use fastest
    RaceNetworkAndCache,
}

/// Cache strategy configuration
#[derive(Debug, Clone)]
pub struct CacheStrategy {
    /// Strategy type
    pub strategy_type: CacheStrategyType,
    /// Cache name to use
    pub cache_name: String,
    /// URL patterns this strategy applies to
    pub patterns: Vec<String>,
    /// Maximum age for cached items (seconds)
    pub max_age_seconds: Option<u64>,
    /// Maximum number of entries
    pub max_entries: Option<usize>,
    /// Whether to query cache in background
    pub background_sync: bool,
}

impl CacheStrategy {
    /// Create a new cache strategy
    pub fn new(strategy_type: CacheStrategyType, cache_name: impl Into<String>) -> Self {
        Self {
            strategy_type,
            cache_name: cache_name.into(),
            patterns: Vec::new(),
            max_age_seconds: None,
            max_entries: None,
            background_sync: false,
        }
    }

    /// Add a URL pattern to this strategy
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.patterns.push(pattern.into());
        self
    }

    /// Set maximum cache age
    pub fn with_max_age(mut self, seconds: u64) -> Self {
        self.max_age_seconds = Some(seconds);
        self
    }

    /// Set maximum entries
    pub fn with_max_entries(mut self, count: usize) -> Self {
        self.max_entries = Some(count);
        self
    }

    /// Enable background sync
    pub fn with_background_sync(mut self, enabled: bool) -> Self {
        self.background_sync = enabled;
        self
    }

    /// Check if a URL matches any pattern
    pub fn matches(&self, url: &str) -> bool {
        if self.patterns.is_empty() {
            return true;
        }
        self.patterns
            .iter()
            .any(|pattern| wildcard_match(pattern, url))
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The cached response
    pub response: CachedResponse,
    /// When this entry was cached
    pub cached_at: i64,
    /// How many times this was accessed
    pub access_count: u32,
    /// Expected ETag for validation
    pub etag: Option<String>,
    /// Last modified header
    pub last_modified: Option<String>,
}

/// Cached response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    /// Status code
    pub status: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response body (base64 encoded)
    pub body: String,
    /// Content type
    pub content_type: String,
}

/// Cache storage interface
#[derive(Debug)]
pub struct CacheStorage {
    /// Named caches
    caches: HashMap<String, HashMap<String, CacheEntry>>,
    /// Maximum size per cache (bytes)
    max_cache_size: usize,
}

impl CacheStorage {
    /// Create new cache storage
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            caches: HashMap::new(),
            max_cache_size,
        }
    }

    /// Open a named cache
    pub fn open(&mut self, name: impl Into<String>) -> &mut HashMap<String, CacheEntry> {
        let name = name.into();
        self.caches.entry(name).or_insert_with(HashMap::new)
    }

    /// Get entry from cache
    pub fn get(&self, cache_name: &str, url: &str) -> Option<&CacheEntry> {
        self.caches.get(cache_name)?.get(url)
    }

    /// Put entry in cache
    pub fn put(&mut self, cache_name: &str, url: impl Into<String>, entry: CacheEntry) {
        let cache = self
            .caches
            .entry(cache_name.to_string())
            .or_insert_with(HashMap::new);
        cache.insert(url.into(), entry);

        // Enforce size limit
        self.enforce_limit(cache_name);
    }

    /// Delete entry from cache
    pub fn delete(&mut self, cache_name: &str, url: &str) -> bool {
        if let Some(cache) = self.caches.get_mut(cache_name) {
            cache.remove(url).is_some()
        } else {
            false
        }
    }

    /// Clear a cache
    pub fn clear(&mut self, cache_name: &str) {
        if let Some(cache) = self.caches.get_mut(cache_name) {
            cache.clear();
        }
    }

    /// Get cache keys
    pub fn keys(&self, cache_name: &str) -> Vec<String> {
        if let Some(cache) = self.caches.get(cache_name) {
            cache.keys().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Enforce cache size limit using LRU eviction
    fn enforce_limit(&mut self, cache_name: &str) {
        if let Some(cache) = self.caches.get_mut(cache_name) {
            while cache.len() > self.max_cache_size {
                // Find least recently used (least access_count, oldest cached_at)
                let lru_key = cache
                    .iter()
                    .min_by_key(|(_, v)| (v.access_count, v.cached_at))
                    .map(|(k, _)| k.clone());

                if let Some(key) = lru_key {
                    cache.remove(&key);
                }
            }
        }
    }

    /// Clean expired entries
    pub fn clean_expired(&mut self, cache_name: &str, max_age_seconds: u64) {
        if let Some(cache) = self.caches.get_mut(cache_name) {
            let now = chrono_timestamp();
            let cutoff = now - (max_age_seconds as i64);

            cache.retain(|_, v| v.cached_at > cutoff);
        }
    }
}

// ============================================================================
// Push Notifications
// ============================================================================

/// Push subscription information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushSubscription {
    /// Unique subscription ID
    pub id: String,
    /// Push endpoint URL
    pub endpoint: String,
    /// Authentication secret
    pub auth: String,
    /// Public key for encryption
    pub p256dh: String,
    /// User agent that subscribed
    pub user_agent: Option<String>,
    /// When subscription was created
    pub created_at: i64,
    /// Whether subscription is active
    pub is_active: bool,
}

impl PushSubscription {
    /// Create new push subscription
    pub fn new(
        endpoint: impl Into<String>,
        auth: impl Into<String>,
        p256dh: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid_v4(),
            endpoint: endpoint.into(),
            auth: auth.into(),
            p256dh: p256dh.into(),
            user_agent: None,
            created_at: chrono_timestamp(),
            is_active: true,
        }
    }

    /// Set user agent
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = Some(ua.into());
        self
    }

    /// Mark subscription as inactive
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    /// Reactivate subscription
    pub fn activate(&mut self) {
        self.is_active = true;
    }
}

/// Push notification payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushMessage {
    /// Notification title
    pub title: String,
    /// Notification body
    pub body: Option<String>,
    /// Notification icon URL
    pub icon: Option<String>,
    /// Notification badge URL
    pub badge: Option<String>,
    /// Notification image URL
    pub image: Option<String>,
    /// Notification data payload
    pub data: Option<serde_json::Value>,
    /// Vibration pattern
    pub vibrate: Option<Vec<u32>>,
    /// Notification tag (for grouping)
    pub tag: Option<String>,
    /// Whether to require interaction
    pub require_interaction: bool,
    /// Notification actions
    pub actions: Vec<NotificationAction>,
}

/// Notification action button
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationAction {
    /// Action identifier
    pub action: String,
    /// Button label
    pub title: String,
    /// Icon URL
    pub icon: Option<String>,
}

impl PushMessage {
    /// Create new push message
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: None,
            icon: None,
            badge: None,
            image: None,
            data: None,
            vibrate: None,
            tag: None,
            require_interaction: false,
            actions: Vec::new(),
        }
    }

    /// Set body text
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Set icon
    pub fn with_icon(mut self, icon: impl Into<String>) -> Self {
        self.icon = Some(icon.into());
        self
    }

    /// Set data
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Add action button
    pub fn with_action(mut self, action: impl Into<String>, title: impl Into<String>) -> Self {
        self.actions.push(NotificationAction {
            action: action.into(),
            title: title.into(),
            icon: None,
        });
        self
    }
}

/// Push notification manager
#[derive(Debug)]
pub struct PushManager {
    /// Active subscriptions
    subscriptions: HashMap<String, PushSubscription>,
    /// VAPID keys for push service authentication
    vapid_keys: Option<VapidKeys>,
}

/// VAPID (Voluntary Application Server Identification) keys
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VapidKeys {
    /// Public key
    pub public_key: String,
    /// Private key
    pub private_key: String,
}

impl PushManager {
    /// Create new push manager
    pub fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            vapid_keys: None,
        }
    }

    /// Set VAPID keys
    pub fn with_vapid_keys(
        mut self,
        public: impl Into<String>,
        private: impl Into<String>,
    ) -> Self {
        self.vapid_keys = Some(VapidKeys {
            public_key: public.into(),
            private_key: private.into(),
        });
        self
    }

    /// Subscribe to push notifications
    pub fn subscribe(&mut self, subscription: PushSubscription) -> &PushSubscription {
        let id = subscription.id.clone();
        self.subscriptions.insert(id.clone(), subscription);
        self.subscriptions.get(&id).unwrap()
    }

    /// Unsubscribe
    pub fn unsubscribe(&mut self, id: &str) -> bool {
        self.subscriptions.remove(id).is_some()
    }

    /// Get subscription by ID
    pub fn get(&self, id: &str) -> Option<&PushSubscription> {
        self.subscriptions.get(id)
    }

    /// Get all active subscriptions
    pub fn active_subscriptions(&self) -> Vec<&PushSubscription> {
        self.subscriptions
            .values()
            .filter(|s| s.is_active)
            .collect()
    }

    /// Send push message to subscription
    pub async fn send(&self, id: &str, _message: &PushMessage) -> Result<(), PushError> {
        let subscription = self.get(id).ok_or(PushError::SubscriptionNotFound)?;

        if !subscription.is_active {
            return Err(PushError::SubscriptionInactive);
        }

        // In a real implementation, this would:
        // 1. Encrypt the payload using the subscription's p256dh and auth
        // 2. Send to the push service endpoint
        // 3. Handle authentication using VAPID keys if available

        Ok(())
    }

    /// Broadcast to all active subscriptions
    pub async fn broadcast(&self, message: &PushMessage) -> Vec<Result<(), PushError>> {
        let mut results = Vec::new();
        for sub in self.active_subscriptions() {
            results.push(self.send(&sub.id, message).await);
        }
        results
    }
}

/// Push error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushError {
    SubscriptionNotFound,
    SubscriptionInactive,
    EncryptionFailed,
    NetworkError,
    InvalidPayload,
}

// ============================================================================
// Web Manifest
// ============================================================================

/// Web App Manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebManifest {
    /// Application name
    pub name: String,
    /// Short name (for homescreen)
    pub short_name: String,
    /// Description
    pub description: Option<String>,
    /// Start URL
    pub start_url: String,
    /// Display mode
    pub display: DisplayMode,
    /// Orientation preference
    pub orientation: Orientation,
    /// Theme color
    pub theme_color: String,
    /// Background color
    pub background_color: String,
    /// Application icons
    pub icons: Vec<ManifestIcon>,
    /// Categories
    pub categories: Vec<String>,
    /// Screenshots
    pub screenshots: Vec<ManifestScreenshot>,
    /// Related applications
    pub related_applications: Vec<RelatedApplication>,
    /// Preferred application (native app)
    pub prefer_related_applications: bool,
    /// Scope
    pub scope: Option<String>,
    /// Shortcuts
    pub shortcuts: Vec<AppShortcut>,
    /// URL protocol handlers
    pub protocol_handlers: Vec<ProtocolHandler>,
    /// File handlers
    pub file_handlers: Vec<FileHandler>,
    /// Share target
    pub share_target: Option<ShareTarget>,
}

/// Display mode for PWA
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DisplayMode {
    Fullscreen,
    Standalone,
    MinimalUi,
    Browser,
}

/// Orientation preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Orientation {
    Any,
    Natural,
    Landscape,
    Portrait,
    PortraitPrimary,
    PortraitSecondary,
    LandscapePrimary,
    LandscapeSecondary,
}

/// Manifest icon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestIcon {
    /// Icon source URL
    pub src: String,
    /// Icon sizes (e.g., "192x192")
    pub sizes: String,
    /// MIME type
    #[serde(rename = "type")]
    pub icon_type: String,
    /// Icon purpose
    pub purpose: Option<String>,
}

/// Manifest screenshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestScreenshot {
    /// Screenshot source URL
    pub src: String,
    /// Screenshot sizes
    pub sizes: String,
    /// Screenshot label
    pub label: Option<String>,
}

/// Related application (native app)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedApplication {
    /// Platform (playstore, appstore, etc.)
    pub platform: String,
    /// App URL
    pub url: Option<String>,
    /// App ID
    pub id: Option<String>,
}

/// App shortcut (quick actions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppShortcut {
    /// Shortcut ID
    pub id: String,
    /// Shortcut name
    pub name: String,
    /// Short description
    pub short_name: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Shortcut icon
    pub icon: Option<String>,
    /// URL to launch
    pub url: String,
}

/// Protocol handler (URL schemes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolHandler {
    /// Protocol scheme (e.g., "web+myapp")
    pub protocol: String,
    /// URL to handle
    pub url: String,
}

/// File handler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHandler {
    /// Handler action
    pub action: String,
    /// File icons
    pub icons: HashMap<String, String>,
    /// Accepted file types
    pub accept: HashMap<String, Vec<String>>,
}

/// Share target configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareTarget {
    /// Action URL template
    pub action: String,
    /// Share method
    pub method: ShareMethod,
    /// Enc type for form data
    pub enctype: Option<String>,
    /// Parameters to accept
    pub params: ShareParams,
}

/// Share method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShareMethod {
    Get,
    Post,
}

/// Share parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareParams {
    /// Title field name
    pub title: Option<String>,
    /// Text field name
    pub text: Option<String>,
    /// URL field name
    pub url: Option<String>,
    /// Files field name
    pub files: Option<String>,
}

impl WebManifest {
    /// Create a minimal manifest
    pub fn new(
        name: impl Into<String>,
        short_name: impl Into<String>,
        start_url: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            short_name: short_name.into(),
            description: None,
            start_url: start_url.into(),
            display: DisplayMode::Standalone,
            orientation: Orientation::Any,
            theme_color: "#ffffff".to_string(),
            background_color: "#ffffff".to_string(),
            icons: Vec::new(),
            categories: Vec::new(),
            screenshots: Vec::new(),
            related_applications: Vec::new(),
            prefer_related_applications: false,
            scope: None,
            shortcuts: Vec::new(),
            protocol_handlers: Vec::new(),
            file_handlers: Vec::new(),
            share_target: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set display mode
    pub fn with_display(mut self, display: DisplayMode) -> Self {
        self.display = display;
        self
    }

    /// Set theme colors
    pub fn with_colors(mut self, theme: impl Into<String>, background: impl Into<String>) -> Self {
        self.theme_color = theme.into();
        self.background_color = background.into();
        self
    }

    /// Add icon
    pub fn with_icon(
        mut self,
        src: impl Into<String>,
        sizes: impl Into<String>,
        icon_type: impl Into<String>,
    ) -> Self {
        self.icons.push(ManifestIcon {
            src: src.into(),
            sizes: sizes.into(),
            icon_type: icon_type.into(),
            purpose: Some("any".to_string()),
        });
        self
    }

    /// Add shortcut
    pub fn with_shortcut(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        self.shortcuts.push(AppShortcut {
            id: id.into(),
            name: name.into(),
            short_name: None,
            description: None,
            icon: None,
            url: url.into(),
        });
        self
    }

    /// Generate manifest JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse manifest from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ============================================================================
// Background Sync
// ============================================================================

/// Background sync registration
#[derive(Debug, Clone)]
pub struct BackgroundSyncRegistration {
    /// Sync tag/identifier
    pub tag: String,
    /// Minimum interval (milliseconds)
    pub min_interval: u32,
    /// Whether sync is one-time or periodic
    pub sync_type: BackgroundSyncType,
    /// Network requirements
    pub network_requirements: NetworkRequirements,
}

/// Background sync type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundSyncType {
    OneShot,
    Periodic,
}

/// Network requirements for sync
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkRequirements {
    Any,
    Online,
    NonCellular,
}

/// Background sync manager
#[derive(Debug)]
pub struct BackgroundSyncManager {
    /// Registered sync tasks
    registrations: HashMap<String, BackgroundSyncRegistration>,
    /// Pending sync tasks (waiting for connectivity)
    pending: Vec<String>,
}

impl BackgroundSyncManager {
    /// Create new background sync manager
    pub fn new() -> Self {
        Self {
            registrations: HashMap::new(),
            pending: Vec::new(),
        }
    }

    /// Register a sync task
    pub fn register(&mut self, registration: BackgroundSyncRegistration) {
        self.registrations
            .insert(registration.tag.clone(), registration);
    }

    /// Unregister sync task
    pub fn unregister(&mut self, tag: &str) -> bool {
        self.registrations.remove(tag).is_some()
    }

    /// Get registration
    pub fn get(&self, tag: &str) -> Option<&BackgroundSyncRegistration> {
        self.registrations.get(tag)
    }

    /// Mark sync as pending
    pub fn mark_pending(&mut self, tag: &str) {
        if !self.pending.contains(&tag.to_string()) {
            self.pending.push(tag.to_string());
        }
    }

    /// Get pending tasks
    pub fn pending_tasks(&self) -> Vec<&str> {
        self.pending.iter().map(|s| s.as_str()).collect()
    }

    /// Complete sync task
    pub fn complete(&mut self, tag: &str) {
        self.pending.retain(|t| t != tag);
    }
}

// ============================================================================
// PWA Primitives - Main Interface
// ============================================================================

/// PWA configuration and management
#[derive(Debug)]
pub struct PWAPrimitives {
    /// Service worker registrations
    service_workers: HashMap<String, ServiceWorkerRegistration>,
    /// Cache strategies
    cache_strategies: Vec<CacheStrategy>,
    /// Cache storage
    cache_storage: CacheStorage,
    /// Push manager
    push_manager: PushManager,
    /// Web manifest
    manifest: Option<WebManifest>,
    /// Background sync manager
    sync_manager: BackgroundSyncManager,
    /// Whether PWA is installable
    is_installable: bool,
}

impl PWAPrimitives {
    /// Create new PWA instance
    pub fn new() -> Self {
        Self {
            service_workers: HashMap::new(),
            cache_strategies: Vec::new(),
            cache_storage: CacheStorage::new(100),
            push_manager: PushManager::new(),
            manifest: None,
            sync_manager: BackgroundSyncManager::new(),
            is_installable: false,
        }
    }

    // -----------------------------------------------------------------------
    // Service Worker Management
    // -----------------------------------------------------------------------

    /// Register a service worker
    pub fn register_service_worker(
        &mut self,
        scope: impl Into<String>,
        script_url: impl Into<String>,
    ) -> &ServiceWorkerRegistration {
        let registration = ServiceWorkerRegistration::new(scope, script_url);
        let id = registration.id.clone();
        self.service_workers.insert(id.clone(), registration);
        self.service_workers.get(&id).unwrap()
    }

    /// Get service worker by scope
    pub fn get_service_worker(&self, scope: &str) -> Option<&ServiceWorkerRegistration> {
        self.service_workers.values().find(|sw| sw.scope == scope)
    }

    /// Update service worker state
    pub fn update_sw_state(&mut self, id: &str, state: ServiceWorkerState) -> bool {
        if let Some(sw) = self.service_workers.get_mut(id) {
            sw.update_state(state);
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // Cache Strategy Management
    // -----------------------------------------------------------------------

    /// Add cache strategy
    pub fn add_cache_strategy(&mut self, strategy: CacheStrategy) {
        self.cache_strategies.push(strategy);
    }

    /// Find matching cache strategy for URL
    pub fn find_cache_strategy(&self, url: &str) -> Option<&CacheStrategy> {
        self.cache_strategies.iter().find(|s| s.matches(url))
    }

    /// Get cache storage
    pub fn cache_storage(&mut self) -> &mut CacheStorage {
        &mut self.cache_storage
    }

    // -----------------------------------------------------------------------
    // Push Notification Management
    // -----------------------------------------------------------------------

    /// Get push manager
    pub fn push_manager(&mut self) -> &mut PushManager {
        &mut self.push_manager
    }

    /// Subscribe to push notifications
    pub fn subscribe_push(&mut self, subscription: PushSubscription) -> &PushSubscription {
        self.push_manager.subscribe(subscription)
    }

    /// Send push notification
    pub async fn send_push(&self, id: &str, message: &PushMessage) -> Result<(), PushError> {
        self.push_manager.send(id, message).await
    }

    // -----------------------------------------------------------------------
    // Manifest Management
    // -----------------------------------------------------------------------

    /// Set web manifest
    pub fn set_manifest(&mut self, manifest: WebManifest) {
        self.manifest = Some(manifest);
        self.is_installable = self.check_installable();
    }

    /// Get manifest
    pub fn manifest(&self) -> Option<&WebManifest> {
        self.manifest.as_ref()
    }

    /// Get manifest as JSON
    pub fn manifest_json(&self) -> Option<String> {
        self.manifest.as_ref().and_then(|m| m.to_json().ok())
    }

    /// Check if PWA meets installability criteria
    fn check_installable(&self) -> bool {
        if let Some(manifest) = &self.manifest {
            // Check for required manifest properties
            manifest.name.len() > 0
                && manifest.short_name.len() > 0
                && manifest.start_url.len() > 0
                && !manifest.icons.is_empty()
                && manifest.display == DisplayMode::Standalone
        } else {
            false
        }
    }

    /// Check if PWA is installable
    pub fn is_installable(&self) -> bool {
        self.is_installable
    }

    // -----------------------------------------------------------------------
    // Background Sync
    // -----------------------------------------------------------------------

    /// Get sync manager
    pub fn sync_manager(&mut self) -> &mut BackgroundSyncManager {
        &mut self.sync_manager
    }

    /// Register background sync
    pub fn register_sync(&mut self, registration: BackgroundSyncRegistration) {
        self.sync_manager.register(registration);
    }
}

impl Default for PWAPrimitives {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate UUID v4
fn uuid_v4() -> String {
    // Simple UUID v4 generation
    format!(
        "{:04x}{:04x}-{:04x}-{:04x}-{:04x}-{:04x}{:04x}{:04x}",
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>()
    )
}

/// Get current timestamp (milliseconds since epoch)
fn chrono_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

/// Wildcard pattern matching
fn wildcard_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            let prefix = parts[0];
            let suffix = parts[1];
            return text.starts_with(prefix) && text.ends_with(suffix);
        }
    }

    pattern == text
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_worker_registration() {
        let sw = ServiceWorkerRegistration::new("/scope", "/sw.js");
        assert_eq!(sw.scope, "/scope");
        assert_eq!(sw.script_url, "/sw.js");
        assert_eq!(sw.state, ServiceWorkerState::Parsing);
        assert!(!sw.is_controlling);
    }

    #[test]
    fn test_service_worker_state_update() {
        let mut sw = ServiceWorkerRegistration::new("/scope", "/sw.js");
        sw.update_state(ServiceWorkerState::Activated);
        assert_eq!(sw.state, ServiceWorkerState::Activated);
    }

    #[test]
    fn test_cache_strategy_creation() {
        let strategy = CacheStrategy::new(CacheStrategyType::CacheFirst, "v1")
            .with_pattern("/api/*")
            .with_max_age(3600)
            .with_max_entries(50);

        assert_eq!(strategy.strategy_type, CacheStrategyType::CacheFirst);
        assert_eq!(strategy.cache_name, "v1");
        assert!(strategy.patterns.contains(&"/api/*".to_string()));
        assert_eq!(strategy.max_age_seconds, Some(3600));
        assert_eq!(strategy.max_entries, Some(50));
    }

    #[test]
    fn test_cache_strategy_matching() {
        let strategy =
            CacheStrategy::new(CacheStrategyType::CacheFirst, "v1").with_pattern("/api/*");

        assert!(strategy.matches("/api/users"));
        assert!(strategy.matches("/api/posts/123"));
        assert!(!strategy.matches("/other/path"));
    }

    #[test]
    fn test_cache_storage() {
        let mut storage = CacheStorage::new(10);

        let entry = CacheEntry {
            response: CachedResponse {
                status: 200,
                headers: {
                    let mut map = HashMap::new();
                    map.insert("content-type".to_string(), "text/html".to_string());
                    map
                },
                body: "SGVsbG8=".to_string(),
                content_type: "text/html".to_string(),
            },
            cached_at: chrono_timestamp(),
            access_count: 0,
            etag: Some("abc123".to_string()),
            last_modified: Some("Wed, 21 Oct 2015 07:28:00 GMT".to_string()),
        };

        storage.put("v1", "/test", entry);
        assert!(storage.get("v1", "/test").is_some());
        assert_eq!(storage.keys("v1").len(), 1);

        storage.delete("v1", "/test");
        assert!(storage.get("v1", "/test").is_none());
    }

    #[test]
    fn test_push_subscription() {
        let sub = PushSubscription::new(
            "https://push.example.com/endpoint",
            "auth_secret",
            "p256dh_key",
        );

        assert!(sub.is_active);
        assert_eq!(sub.endpoint, "https://push.example.com/endpoint");

        let mut sub = sub;
        sub.deactivate();
        assert!(!sub.is_active);

        sub.activate();
        assert!(sub.is_active);
    }

    #[test]
    fn test_push_message() {
        let msg = PushMessage::new("Test Notification")
            .with_body("This is a test")
            .with_icon("/icon.png")
            .with_action("view", "View")
            .with_action("dismiss", "Dismiss");

        assert_eq!(msg.title, "Test Notification");
        assert!(msg.body.is_some());
        assert_eq!(msg.actions.len(), 2);
    }

    #[test]
    fn test_push_manager() {
        let mut manager = PushManager::new().with_vapid_keys("public_key", "private_key");

        let sub = PushSubscription::new("endpoint", "auth", "p256dh");
        let registered = manager.subscribe(sub);
        let registered_id = registered.id.clone();

        assert_eq!(manager.active_subscriptions().len(), 1);
        assert!(manager.get(&registered_id).is_some());

        manager.unsubscribe(&registered_id);
        assert_eq!(manager.active_subscriptions().len(), 0);
    }

    #[test]
    fn test_web_manifest() {
        let manifest = WebManifest::new("My App", "MyApp", "/")
            .with_description("A test application")
            .with_display(DisplayMode::Standalone)
            .with_colors("#000000", "#ffffff")
            .with_icon("/icon-192.png", "192x192", "image/png")
            .with_icon("/icon-512.png", "512x512", "image/png")
            .with_shortcut("new", "Create New", "/new");

        assert_eq!(manifest.name, "My App");
        assert_eq!(manifest.short_name, "MyApp");
        assert_eq!(manifest.display, DisplayMode::Standalone);
        assert_eq!(manifest.icons.len(), 2);
        assert_eq!(manifest.shortcuts.len(), 1);

        let json = manifest.to_json().unwrap();
        assert!(json.contains("My App"));
    }

    #[test]
    fn test_manifest_from_json() {
        let json = r##"{
            "name": "Test App",
            "short_name": "Test",
            "start_url": "/",
            "display": "standalone",
            "orientation": "any",
            "theme_color": "#ffffff",
            "background_color": "#ffffff",
            "icons": []
        }"##;

        let manifest = WebManifest::from_json(json).unwrap();
        assert_eq!(manifest.name, "Test App");
        assert_eq!(manifest.short_name, "Test");
    }

    #[test]
    fn test_background_sync_registration() {
        let registration = BackgroundSyncRegistration {
            tag: "sync-posts".to_string(),
            min_interval: 3600000,
            sync_type: BackgroundSyncType::Periodic,
            network_requirements: NetworkRequirements::Online,
        };

        let mut manager = BackgroundSyncManager::new();
        manager.register(registration);

        assert!(manager.get("sync-posts").is_some());
        assert_eq!(manager.get("sync-posts").unwrap().min_interval, 3600000);
    }

    #[test]
    fn test_pwa_primitives() {
        let mut pwa = PWAPrimitives::new();

        // Register service worker
        let sw = pwa.register_service_worker("/", "/sw.js");
        assert_eq!(sw.state, ServiceWorkerState::Parsing);

        // Add cache strategy
        let strategy = CacheStrategy::new(CacheStrategyType::CacheFirst, "v1");
        pwa.add_cache_strategy(strategy);
        assert!(pwa.find_cache_strategy("/any").is_some());

        // Set manifest
        let manifest =
            WebManifest::new("Test", "T", "/").with_icon("/icon.png", "192x192", "image/png");
        pwa.set_manifest(manifest);
        assert!(pwa.is_installable());
    }

    #[test]
    fn test_wildcard_match() {
        assert!(wildcard_match("*", "/anything"));
        assert!(wildcard_match("/api/*", "/api/users"));
        assert!(wildcard_match("/api/*", "/api/posts/123/comments"));
        assert!(!wildcard_match("/api/*", "/other"));
        assert!(wildcard_match("/exact", "/exact"));
        assert!(!wildcard_match("/exact", "/exact/other"));
    }

    #[test]
    fn test_display_mode_serialization() {
        let display = DisplayMode::Standalone;
        let json = serde_json::to_string(&display).unwrap();
        assert_eq!(json, "\"standalone\"");
    }

    #[test]
    fn test_orientation_serialization() {
        let orientation = Orientation::Portrait;
        let json = serde_json::to_string(&orientation).unwrap();
        assert_eq!(json, "\"portrait\"");
    }

    #[test]
    fn test_cache_storage_enforcement() {
        let mut storage = CacheStorage::new(3);

        for i in 0..5 {
            let entry = CacheEntry {
                response: CachedResponse {
                    status: 200,
                    headers: HashMap::new(),
                    body: "Ym9keQ==".to_string(),
                    content_type: "text/plain".to_string(),
                },
                cached_at: chrono_timestamp() + (i as i64),
                access_count: 0,
                etag: None,
                last_modified: None,
            };
            storage.put("test", format!("/key{}", i), entry);
        }

        // Should only have 3 entries due to size limit
        assert_eq!(storage.keys("test").len(), 3);
    }

    #[test]
    fn test_cache_clean_expired() {
        let mut storage = CacheStorage::new(10);
        let now = chrono_timestamp();

        let old_entry = CacheEntry {
            response: CachedResponse {
                status: 200,
                headers: HashMap::new(),
                body: "old".to_string(),
                content_type: "text/plain".to_string(),
            },
            cached_at: now - 10000,
            access_count: 0,
            etag: None,
            last_modified: None,
        };

        let new_entry = CacheEntry {
            response: CachedResponse {
                status: 200,
                headers: HashMap::new(),
                body: "new".to_string(),
                content_type: "text/plain".to_string(),
            },
            cached_at: now,
            access_count: 0,
            etag: None,
            last_modified: None,
        };

        storage.put("test", "/old", old_entry);
        storage.put("test", "/new", new_entry);

        storage.clean_expired("test", 5); // 5 seconds max age

        // Old entry should be removed
        assert!(storage.get("test", "/old").is_none());
        assert!(storage.get("test", "/new").is_some());
    }

    #[test]
    fn test_service_worker_client() {
        let client = ServiceWorkerClient {
            id: uuid_v4(),
            url: "https://example.com/page".to_string(),
            visibility_state: VisibilityState::Visible,
            frame_type: FrameType::TopLevel,
        };

        assert_eq!(client.visibility_state, VisibilityState::Visible);
        assert_eq!(client.frame_type, FrameType::TopLevel);
    }

    #[test]
    fn test_share_target() {
        let share_target = ShareTarget {
            action: "/share".to_string(),
            method: ShareMethod::Post,
            enctype: Some("multipart/form-data".to_string()),
            params: ShareParams {
                title: Some("title".to_string()),
                text: Some("text".to_string()),
                url: Some("url".to_string()),
                files: Some("files".to_string()),
            },
        };

        assert_eq!(share_target.method, ShareMethod::Post);
        assert_eq!(share_target.params.title, Some("title".to_string()));
    }

    #[test]
    fn test_protocol_handler() {
        let handler = ProtocolHandler {
            protocol: "web+myapp".to_string(),
            url: "/handle/%s".to_string(),
        };

        assert_eq!(handler.protocol, "web+myapp");
    }

    #[test]
    fn test_file_handler() {
        let mut icons = HashMap::new();
        icons.insert("image/png".to_string(), "/file-icon.png".to_string());

        let mut accept = HashMap::new();
        accept.insert("image/png".to_string(), vec![".png".to_string()]);

        let handler = FileHandler {
            action: "/handle-file".to_string(),
            icons,
            accept,
        };

        assert_eq!(handler.action, "/handle-file");
        assert!(handler.accept.contains_key("image/png"));
    }

    #[test]
    fn test_notification_action() {
        let action = NotificationAction {
            action: "view".to_string(),
            title: "View".to_string(),
            icon: Some("/view-icon.png".to_string()),
        };

        assert_eq!(action.action, "view");
        assert!(action.icon.is_some());
    }

    #[test]
    fn test_related_application() {
        let app = RelatedApplication {
            platform: "playstore".to_string(),
            url: Some("https://play.google.com/store/apps/details.id=com.example".to_string()),
            id: Some("com.example".to_string()),
        };

        assert_eq!(app.platform, "playstore");
        assert!(app.url.is_some());
        assert!(app.id.is_some());
    }

    #[test]
    fn test_manifest_screenshot() {
        let screenshot = ManifestScreenshot {
            src: "/screenshot.png".to_string(),
            sizes: "1280x720".to_string(),
            label: Some("Home Screen".to_string()),
        };

        assert_eq!(screenshot.sizes, "1280x720");
        assert!(screenshot.label.is_some());
    }

    #[test]
    fn test_manifest_icon_purpose() {
        let icon = ManifestIcon {
            src: "/maskable-icon.png".to_string(),
            sizes: "192x192".to_string(),
            icon_type: "image/png".to_string(),
            purpose: Some("any maskable".to_string()),
        };

        assert!(icon.purpose.as_ref().unwrap().contains("maskable"));
    }
}
