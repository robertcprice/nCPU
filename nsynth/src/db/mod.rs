//! Database Layer for nCPU/nSynth
//!
//! Type-safe database operations with SQL query builders, connection pooling,
//! and ORM patterns for synthesizing database programs from examples.

pub mod sql;
pub mod pool;
pub mod orm;

pub use sql::{
    Query, Expr, Op, Value, ColumnDef, ColumnType,
    SelectBuilder, InsertBuilder, UpdateBuilder, DeleteBuilder, to_sql,
};

pub use pool::{
    Pool, Connection, ConnectionOptions, PoolStats,
    shared_pool, shared_pool_default, SharedPool,
    PooledConnection, ConnectionManager,
};

pub use orm::{
    Model, Repository, QueryBuilder, SelectQuery,
    Schema, ColumnDef as OrmColumnDef, Migration, MigrationRunner,
};

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Database URL
    pub url: String,
    /// Maximum pool size
    pub max_pool_size: usize,
    /// Connection timeout in seconds
    pub connect_timeout: u64,
    /// Whether to enable SSL
    pub ssl: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "sqlite::memory:".to_string(),
            max_pool_size: 10,
            connect_timeout: 30,
            ssl: false,
        }
    }
}

impl DatabaseConfig {
    /// Create a new database configuration
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Default::default()
        }
    }

    /// Set maximum pool size
    pub fn with_max_pool_size(mut self, size: usize) -> Self {
        self.max_pool_size = size;
        self
    }

    /// Set connection timeout
    pub fn with_connect_timeout(mut self, timeout: u64) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Enable SSL
    pub fn with_ssl(mut self, ssl: bool) -> Self {
        self.ssl = ssl;
        self
    }

    /// Create a connection pool from this config
    pub fn create_pool(&self) -> SharedPool {
        shared_pool(self.url.clone(), self.max_pool_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_config() {
        let config = DatabaseConfig::new("postgresql://localhost/test");
        assert_eq!(config.url, "postgresql://localhost/test");
        assert_eq!(config.max_pool_size, 10);
    }

    #[test]
    fn test_database_config_builder() {
        let config = DatabaseConfig::new("sqlite::memory:")
            .with_max_pool_size(20)
            .with_connect_timeout(60)
            .with_ssl(true);

        assert_eq!(config.max_pool_size, 20);
        assert_eq!(config.connect_timeout, 60);
        assert!(config.ssl);
    }
}
