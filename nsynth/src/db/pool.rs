//! Connection Pool for nCPU/nSynth Database Layer
//!
//! Thread-safe connection pooling with automatic reclamation and reuse.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Database connection pool
#[derive(Debug)]
pub struct Pool {
    connections: Vec<Connection>,
    url: String,
    max_size: usize,
    created: Instant,
}

/// Individual database connection
#[derive(Debug, Clone)]
pub struct Connection {
    /// Connection ID
    pub id: usize,
    /// Database URL
    pub url: String,
    /// Whether connection is in use
    pub in_use: Arc<Mutex<bool>>,
    /// Last activity timestamp
    pub last_activity: Arc<Mutex<Instant>>,
    /// Connection options
    pub options: ConnectionOptions,
}

/// Connection options
#[derive(Debug, Clone, Copy)]
pub struct ConnectionOptions {
    /// Connection timeout in seconds
    pub connect_timeout: u64,
    /// Query timeout in seconds
    pub query_timeout: u64,
    /// Maximum idle time before reaping
    pub max_idle_time: u64,
    /// Whether to use SSL
    pub ssl: bool,
}

impl Default for ConnectionOptions {
    fn default() -> Self {
        Self {
            connect_timeout: 30,
            query_timeout: 30,
            max_idle_time: 600,
            ssl: false,
        }
    }
}

impl Pool {
    /// Create a new connection pool
    pub fn new(url: impl Into<String>, max_size: usize) -> Self {
        Self {
            connections: Vec::with_capacity(max_size),
            url: url.into(),
            max_size,
            created: Instant::now(),
        }
    }

    /// Create with default max size
    pub fn with_url(url: impl Into<String>) -> Self {
        Self::new(url, 10)
    }

    /// Get a connection from the pool
    pub fn get_connection(&mut self) -> Option<Connection> {
        // Try to find an idle connection
        for conn in &mut self.connections {
            let mut in_use = conn.in_use.lock().unwrap();
            if !*in_use {
                *in_use = true;
                *conn.last_activity.lock().unwrap() = Instant::now();
                return Some(conn.clone());
            }
        }

        // If at capacity, return None
        if self.connections.len() >= self.max_size {
            return None;
        }

        // Create new connection
        let id = self.connections.len();
        let conn = Connection {
            id,
            url: self.url.clone(),
            in_use: Arc::new(Mutex::new(true)),
            last_activity: Arc::new(Mutex::new(Instant::now())),
            options: ConnectionOptions::default(),
        };

        self.connections.push(conn.clone());
        Some(conn)
    }

    /// Return a connection to the pool
    pub fn return_connection(&self, conn: &Connection) {
        if let Some(in_use) = conn.in_use.lock().ok() {
            let mut guard = in_use;
            *guard = false;
        }
    }

    /// Get pool size
    pub fn size(&self) -> usize {
        self.connections.len()
    }

    /// Get max pool size
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get number of idle connections
    pub fn idle_count(&self) -> usize {
        self.connections
            .iter()
            .filter(|c| c.in_use.lock().map(|u| !*u).unwrap_or(true))
            .count()
    }

    /// Get number of active connections
    pub fn active_count(&self) -> usize {
        self.connections.len() - self.idle_count()
    }

    /// Reap idle connections older than max_idle_time
    pub fn reap_idle(&mut self, max_idle: Duration) -> usize {
        let now = Instant::now();
        let initial_len = self.connections.len();

        self.connections.retain(|conn| {
            let last_activity = conn.last_activity.lock().unwrap();
            let idle = now.duration_since(*last_activity);
            let in_use = *conn.in_use.lock().unwrap();

            // Keep if in use or not idle for too long
            in_use || idle < max_idle
        });

        initial_len - self.connections.len()
    }

    /// Clear all connections
    pub fn clear(&mut self) {
        self.connections.clear();
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total: self.connections.len(),
            active: self.active_count(),
            idle: self.idle_count(),
            max_size: self.max_size,
            uptime: self.created.elapsed(),
        }
    }
}

impl Clone for Pool {
    fn clone(&self) -> Self {
        Self {
            connections: self.connections.clone(),
            url: self.url.clone(),
            max_size: self.max_size,
            created: self.created,
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    pub total: usize,
    pub active: usize,
    pub idle: usize,
    pub max_size: usize,
    pub uptime: Duration,
}

/// Shared pool type for thread-safe access
pub type SharedPool = Arc<Mutex<Pool>>;

/// Create a new shared pool
pub fn shared_pool(url: impl Into<String>, max_size: usize) -> SharedPool {
    Arc::new(Mutex::new(Pool::new(url, max_size)))
}

/// Create shared pool with defaults
pub fn shared_pool_default(url: impl Into<String>) -> SharedPool {
    shared_pool(url, 10)
}

/// RAII guard for auto-returning connections
#[derive(Debug)]
pub struct PooledConnection {
    pool: SharedPool,
    conn: Option<Connection>,
}

impl PooledConnection {
    /// Create a new pooled connection guard
    pub fn new(pool: SharedPool, conn: Connection) -> Self {
        Self {
            pool,
            conn: Some(conn),
        }
    }

    /// Get the underlying connection
    pub fn connection(&self) -> Option<&Connection> {
        self.conn.as_ref()
    }

    /// Get mutable reference to connection
    pub fn connection_mut(&mut self) -> Option<&mut Connection> {
        self.conn.as_mut()
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            if let Ok(pool) = self.pool.lock() {
                pool.return_connection(&conn);
            }
        }
    }
}

/// Connection manager for managing multiple pools
#[derive(Debug)]
pub struct ConnectionManager {
    pools: HashMap<String, SharedPool>,
}

impl ConnectionManager {
    /// Create a new connection manager
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    /// Register a connection pool
    pub fn register(&mut self, name: impl Into<String>, pool: SharedPool) {
        self.pools.insert(name.into(), pool);
    }

    /// Get a connection pool by name
    pub fn get(&self, name: &str) -> Option<SharedPool> {
        self.pools.get(name).cloned()
    }

    /// Remove a connection pool
    pub fn remove(&mut self, name: &str) -> Option<SharedPool> {
        self.pools.remove(name)
    }

    /// List all registered pool names
    pub fn list(&self) -> Vec<String> {
        self.pools.keys().cloned().collect()
    }

    /// Get statistics for all pools
    pub fn stats(&self) -> Vec<(String, PoolStats)> {
        self.pools
            .iter()
            .filter_map(|(name, pool)| pool.lock().ok().map(|p| (name.clone(), p.stats())))
            .collect()
    }
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = Pool::with_url("sqlite::memory:");
        assert_eq!(pool.size(), 0);
        assert_eq!(pool.max_size(), 10);
    }

    #[test]
    fn test_get_connection() {
        let mut pool = Pool::with_url("sqlite::memory:");
        let conn = pool.get_connection();
        assert!(conn.is_some());
        assert_eq!(pool.size(), 1);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn test_return_connection() {
        let mut pool = Pool::with_url("sqlite::memory:");
        let conn = pool.get_connection().unwrap();
        pool.return_connection(&conn);
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.idle_count(), 1);
    }

    #[test]
    fn test_pool_capacity() {
        let mut pool = Pool::new("sqlite::memory:", 2);
        pool.get_connection();
        pool.get_connection();
        let third = pool.get_connection();
        assert!(third.is_none());
    }

    #[test]
    fn test_pool_stats() {
        let pool = Pool::with_url("sqlite::memory:");
        let stats = pool.stats();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.max_size, 10);
    }

    #[test]
    fn test_connection_manager() {
        let mut manager = ConnectionManager::new();
        let pool = shared_pool_default("sqlite::memory:");
        manager.register("test", pool);

        assert!(manager.get("test").is_some());
        assert_eq!(manager.list().len(), 1);
    }

    #[test]
    fn test_connection_options() {
        let options = ConnectionOptions::default();
        assert_eq!(options.connect_timeout, 30);
        assert_eq!(options.query_timeout, 30);
        assert!(!options.ssl);
    }
}
