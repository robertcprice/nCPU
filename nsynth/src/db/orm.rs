//! ORM Patterns for nCPU/nSynth Database Layer
//!
//! Object-Relational Mapping traits and repository patterns for type-safe database operations.

use std::marker::PhantomData;
use crate::db::pool::SharedPool;
use crate::db::sql::{Value, ColumnType};

/// Trait for database models
pub trait Model {
    /// Get the table name for this model
    fn table_name() -> String;

    /// Create model from database row (hash map of column names to values)
    fn from_row(row: &std::collections::HashMap<String, Value>) -> Option<Self>
    where
        Self: Sized;

    /// Convert model to values for INSERT/UPDATE
    fn to_values(&self) -> std::collections::HashMap<String, Value>;

    /// Get the primary key value
    fn primary_key(&self) -> Value;
}

/// Repository for CRUD operations
pub struct Repository<T>
where
    T: Model,
{
    pool: SharedPool,
    _phantom: PhantomData<T>,
}

impl<T> Repository<T>
where
    T: Model,
{
    /// Create a new repository
    pub fn new(pool: SharedPool) -> Self {
        Self {
            pool,
            _phantom: PhantomData,
        }
    }

    /// Find by primary key
    pub fn find(&self, _id: &Value) -> Result<Option<T>, String> {
        let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
        let _conn = pool.get_connection().ok_or("No connection available")?;

        // In production, would execute SELECT query here
        // For now, return placeholder
        Ok(None)
    }

    /// Find all records
    pub fn find_all(&self) -> Result<Vec<T>, String> {
        let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
        let _conn = pool.get_connection().ok_or("No connection available")?;

        // In production, would execute SELECT * query here
        Ok(Vec::new())
    }

    /// Find records matching WHERE clause
    pub fn find_where(&self, _where_clause: &str) -> Result<Vec<T>, String> {
        let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
        let _conn = pool.get_connection().ok_or("No connection available")?;

        Ok(Vec::new())
    }

    /// Insert a new record
    pub fn insert(&self, _model: &T) -> Result<bool, String> {
        let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
        let _conn = pool.get_connection().ok_or("No connection available")?;

        // In production, would execute INSERT query here
        Ok(true)
    }

    /// Update an existing record
    pub fn update(&self, _model: &T) -> Result<bool, String> {
        let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
        let _conn = pool.get_connection().ok_or("No connection available")?;

        // In production, would execute UPDATE query here
        Ok(true)
    }

    /// Delete a record by primary key
    pub fn delete(&self, _id: &Value) -> Result<bool, String> {
        let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
        let _conn = pool.get_connection().ok_or("No connection available")?;

        // In production, would execute DELETE query here
        Ok(true)
    }

    /// Count all records
    pub fn count(&self) -> Result<usize, String> {
        let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
        let _conn = pool.get_connection().ok_or("No connection available")?;

        // In production, would execute COUNT(*) query here
        Ok(0)
    }
}

/// Query builder for complex queries
pub struct QueryBuilder<T>
where
    T: Model,
{
    pool: SharedPool,
    _phantom: PhantomData<T>,
}

impl<T> QueryBuilder<T>
where
    T: Model,
{
    /// Create a new query builder
    pub fn new(pool: SharedPool) -> Self {
        Self {
            pool,
            _phantom: PhantomData,
        }
    }

    /// Build a SELECT query
    pub fn select(&self) -> SelectQuery<T> {
        SelectQuery {
            pool: self.pool.clone(),
            table: T::table_name(),
            columns: Vec::new(),
            where_clause: None,
            order_by: Vec::new(),
            limit: None,
            offset: None,
            _phantom: PhantomData,
        }
    }
}

/// SELECT query builder
#[derive(Clone)]
pub struct SelectQuery<T>
where
    T: Model,
{
    pool: SharedPool,
    table: String,
    columns: Vec<String>,
    where_clause: Option<String>,
    order_by: Vec<String>,
    limit: Option<usize>,
    offset: Option<usize>,
    _phantom: PhantomData<T>,
}

impl<T> SelectQuery<T>
where
    T: Model,
{
    /// Select specific columns
    pub fn columns(mut self, cols: Vec<impl Into<String>>) -> Self {
        self.columns = cols.into_iter().map(|c| c.into()).collect();
        self
    }

    /// Add WHERE clause
    pub fn where_(mut self, clause: impl Into<String>) -> Self {
        self.where_clause = Some(clause.into());
        self
    }

    /// Add ORDER BY clause
    pub fn order_by(mut self, col: impl Into<String>, asc: bool) -> Self {
        let dir = if asc { "ASC" } else { "DESC" };
        self.order_by.push(format!("{} {}", col.into(), dir));
        self
    }

    /// Add LIMIT
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Add OFFSET
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    /// Execute the query
    pub fn execute(&self) -> Result<Vec<T>, String> {
        let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
        let _conn = pool.get_connection().ok_or("No connection available")?;

        // In production, would build and execute SQL here
        Ok(Vec::new())
    }

    /// Execute and return first result
    pub fn first(&self) -> Result<Option<T>, String> {
        let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
        let _conn = pool.get_connection().ok_or("No connection available")?;

        // In production, would build and execute SELECT ... LIMIT 1 query here
        Ok(None)
    }

    /// Build the SQL string
    pub fn to_sql(&self) -> String {
        let cols = if self.columns.is_empty() {
            "*".to_string()
        } else {
            self.columns.join(", ")
        };

        let mut sql = format!("SELECT {} FROM \"{}\"", cols, self.table);

        if let Some(w) = &self.where_clause {
            sql.push_str(&format!(" WHERE {}", w));
        }

        if !self.order_by.is_empty() {
            sql.push_str(&format!(" ORDER BY {}", self.order_by.join(", ")));
        }

        if let Some(n) = self.limit {
            sql.push_str(&format!(" LIMIT {}", n));
        }

        if let Some(n) = self.offset {
            sql.push_str(&format!(" OFFSET {}", n));
        }

        sql
    }
}

/// Schema definition for automatic table creation
pub struct Schema {
    pub table_name: String,
    pub columns: Vec<ColumnDef>,
}

/// Column definition for schema
#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub type_: ColumnType,
    pub nullable: bool,
    pub primary_key: bool,
    pub auto_increment: bool,
    pub unique: bool,
}

/// Schema migration trait
pub trait Migration {
    /// Get migration version
    fn version(&self) -> u64;

    /// Get migration name
    fn name(&self) -> &str;

    /// Get up SQL (apply migration)
    fn up(&self) -> String;

    /// Get down SQL (rollback migration)
    fn down(&self) -> String;
}

/// Migration runner
pub struct MigrationRunner {
    pool: SharedPool,
    migrations: Vec<Box<dyn Migration>>,
}

impl MigrationRunner {
    /// Create a new migration runner
    pub fn new(pool: SharedPool) -> Self {
        Self {
            pool,
            migrations: Vec::new(),
        }
    }

    /// Add a migration
    pub fn add_migration(&mut self, migration: Box<dyn Migration>) {
        self.migrations.push(migration);
    }

    /// Run all pending migrations
    pub fn run(&self) -> Result<Vec<u64>, String> {
        let mut applied = Vec::new();

        for migration in &self.migrations {
            let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
            let _conn = pool.get_connection().ok_or("No connection available")?;

            // In production, would:
            // 1. Check if migration already applied
            // 2. Execute up() SQL
            // 3. Record migration as applied

            applied.push(migration.version());
        }

        Ok(applied)
    }

    /// Rollback last migration
    pub fn rollback(&self) -> Result<bool, String> {
        if self.migrations.last().is_some() {
            let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
            let _conn = pool.get_connection().ok_or("No connection available")?;

            // In production, would execute down() SQL
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repository_creation() {
        let pool = crate::db::pool::shared_pool_default("sqlite::memory:");
        let repo: Repository<TestModel> = Repository::new(pool);
        assert_eq!(repo.count().unwrap(), 0);
    }

    #[test]
    fn test_query_builder() {
        let pool = crate::db::pool::shared_pool_default("sqlite::memory:");
        let builder: QueryBuilder<TestModel> = QueryBuilder::new(pool);
        let query = builder.select().to_sql();
        assert!(query.contains("SELECT"));
        assert!(query.contains("test_models"));
    }

    #[test]
    fn test_select_query_clauses() {
        let pool = crate::db::pool::shared_pool_default("sqlite::memory:");
        let builder: QueryBuilder<TestModel> = QueryBuilder::new(pool);

        let sql = builder
            .select()
            .where_("age > 18")
            .order_by("name", true)
            .limit(10)
            .to_sql();

        assert!(sql.contains("WHERE age > 18"));
        assert!(sql.contains("ORDER BY name ASC"));
        assert!(sql.contains("LIMIT 10"));
    }
}

// Test model
struct TestModel {
    id: i64,
    name: String,
}

impl Model for TestModel {
    fn table_name() -> String {
        "test_models".to_string()
    }

    fn from_row(_row: &std::collections::HashMap<String, Value>) -> Option<Self> {
        Some(TestModel {
            id: 0,
            name: "test".to_string(),
        })
    }

    fn to_values(&self) -> std::collections::HashMap<String, Value> {
        let mut map = std::collections::HashMap::new();
        map.insert("id".to_string(), Value::Int(self.id));
        map.insert("name".to_string(), Value::Text(self.name.clone()));
        map
    }

    fn primary_key(&self) -> Value {
        Value::Int(self.id)
    }
}
