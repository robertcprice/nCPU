//! SQL Types and Query Builder for nCPU/nSynth
//!
//! Type-safe SQL query construction with parameter binding and validation.

/// SQL query types
#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// SELECT query
    Select {
        table: String,
        cols: Vec<String>,
        where_: Option<Expr>,
        order_by: Option<Vec<OrderBy>>,
        limit: Option<usize>,
        offset: Option<usize>,
    },
    /// INSERT query
    Insert {
        table: String,
        values: Vec<(String, Value)>,
    },
    /// UPDATE query
    Update {
        table: String,
        set: Vec<(String, Value)>,
        where_: Option<Expr>,
    },
    /// DELETE query
    Delete {
        table: String,
        where_: Option<Expr>,
    },
    /// CREATE TABLE query
    Create {
        table: String,
        columns: Vec<ColumnDef>,
    },
    /// DROP TABLE query
    Drop {
        table: String,
        if_exists: bool,
    },
}

impl Query {
    /// Get the query type name
    pub fn type_name(&self) -> &str {
        match self {
            Query::Select { .. } => "SELECT",
            Query::Insert { .. } => "INSERT",
            Query::Update { .. } => "UPDATE",
            Query::Delete { .. } => "DELETE",
            Query::Create { .. } => "CREATE",
            Query::Drop { .. } => "DROP",
        }
    }

    /// Get the target table name
    pub fn table(&self) -> &str {
        match self {
            Query::Select { table, .. } => table,
            Query::Insert { table, .. } => table,
            Query::Update { table, .. } => table,
            Query::Delete { table, .. } => table,
            Query::Create { table, .. } => table,
            Query::Drop { table, .. } => table,
        }
    }
}

/// SQL expression
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Column reference
    Column(String),
    /// Literal value
    Literal(Value),
    /// Binary operation
    BinOp(Box<Expr>, Op, Box<Expr>),
    /// Unary operation
    UnaryOp(Op, Box<Expr>),
    /// Function call
    Function(String, Vec<Expr>),
    /// NULL value
    Null,
    /// Parameter placeholder
    Parameter(usize),
}

/// SQL operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Logical
    And,
    Or,
    Not,
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // String
    Like,
    NotLike,
    // Null checks
    IsNull,
    IsNotNull,
}

/// Order by clause
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderBy {
    pub column: String,
    pub asc: bool,
}

/// SQL value types
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Blob(Vec<u8>),
}

/// Column definition
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    pub type_: ColumnType,
    pub nullable: bool,
    pub primary_key: bool,
    pub default: Option<Value>,
    pub unique: bool,
}

/// Column data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Integer,
    Real,
    Text,
    Blob,
    Boolean,
    Timestamp,
    /// Varchar with max length
    Varchar(usize),
}

/// Build SELECT query
pub struct SelectBuilder {
    table: String,
    cols: Vec<String>,
    where_: Option<Expr>,
    order_by: Vec<OrderBy>,
    limit: Option<usize>,
    offset: Option<usize>,
}

impl SelectBuilder {
    /// Create new SELECT builder
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            cols: Vec::new(),
            where_: None,
            order_by: Vec::new(),
            limit: None,
            offset: None,
        }
    }

    /// Select specific columns
    pub fn columns(mut self, cols: Vec<impl Into<String>>) -> Self {
        self.cols = cols.into_iter().map(|c| c.into()).collect();
        self
    }

    /// Select all columns
    pub fn all(mut self) -> Self {
        self.cols = vec!["*".to_string()];
        self
    }

    /// Add WHERE clause
    pub fn where_(mut self, expr: Expr) -> Self {
        self.where_ = Some(expr);
        self
    }

    /// Add ORDER BY clause
    pub fn order_by(mut self, column: impl Into<String>, asc: bool) -> Self {
        self.order_by.push(OrderBy {
            column: column.into(),
            asc,
        });
        self
    }

    /// Add LIMIT clause
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Add OFFSET clause
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    /// Build the query
    pub fn build(self) -> Query {
        Query::Select {
            table: self.table,
            cols: self.cols,
            where_: self.where_,
            order_by: if self.order_by.is_empty() {
                None
            } else {
                Some(self.order_by)
            },
            limit: self.limit,
            offset: self.offset,
        }
    }
}

/// Build INSERT query
pub struct InsertBuilder {
    table: String,
    values: Vec<(String, Value)>,
}

impl InsertBuilder {
    /// Create new INSERT builder
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            values: Vec::new(),
        }
    }

    /// Set a column value
    pub fn set(mut self, column: impl Into<String>, value: Value) -> Self {
        self.values.push((column.into(), value));
        self
    }

    /// Set multiple values
    pub fn set_all(mut self, values: Vec<(impl Into<String>, Value)>) -> Self {
        self.values = values.into_iter().map(|(c, v)| (c.into(), v)).collect();
        self
    }

    /// Build the query
    pub fn build(self) -> Query {
        Query::Insert {
            table: self.table,
            values: self.values,
        }
    }
}

/// Build UPDATE query
pub struct UpdateBuilder {
    table: String,
    set: Vec<(String, Value)>,
    where_: Option<Expr>,
}

impl UpdateBuilder {
    /// Create new UPDATE builder
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            set: Vec::new(),
            where_: None,
        }
    }

    /// Set a column value
    pub fn set(mut self, column: impl Into<String>, value: Value) -> Self {
        self.set.push((column.into(), value));
        self
    }

    /// Add WHERE clause
    pub fn where_(mut self, expr: Expr) -> Self {
        self.where_ = Some(expr);
        self
    }

    /// Build the query
    pub fn build(self) -> Query {
        Query::Update {
            table: self.table,
            set: self.set,
            where_: self.where_,
        }
    }
}

/// Build DELETE query
pub struct DeleteBuilder {
    table: String,
    where_: Option<Expr>,
}

impl DeleteBuilder {
    /// Create new DELETE builder
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            where_: None,
        }
    }

    /// Add WHERE clause
    pub fn where_(mut self, expr: Expr) -> Self {
        self.where_ = Some(expr);
        self
    }

    /// Build the query
    pub fn build(self) -> Query {
        Query::Delete {
            table: self.table,
            where_: self.where_,
        }
    }
}

/// Generate SQL string from query
pub fn to_sql(query: &Query) -> String {
    match query {
        Query::Select {
            table,
            cols,
            where_,
            order_by,
            limit,
            offset,
        } => {
            let cols_str = if cols.is_empty() { "*" } else { &cols.join(", ") };
            let mut sql = format!("SELECT {} FROM {}", cols_str, quote_table(table));

            if let Some(w) = where_ {
                sql.push_str(&format!(" WHERE {}", expr_to_sql(w)));
            }

            if let Some(ob) = order_by {
                let order_str: Vec<String> = ob
                    .iter()
                    .map(|o| format!("{} {}", quote_ident(&o.column), if o.asc { "ASC" } else { "DESC" }))
                    .collect();
                sql.push_str(&format!(" ORDER BY {}", order_str.join(", ")));
            }

            if let Some(n) = limit {
                sql.push_str(&format!(" LIMIT {}", n));
            }

            if let Some(n) = offset {
                sql.push_str(&format!(" OFFSET {}", n));
            }

            sql
        }
        Query::Insert { table, values } => {
            let cols: Vec<&str> = values.iter().map(|(c, _)| c.as_str()).collect();
            let vals: Vec<String> = values.iter().map(|(_, v)| value_to_sql(v)).collect();
            format!(
                "INSERT INTO {} ({}) VALUES ({})",
                quote_table(table),
                cols.iter().map(|c| quote_ident(c)).collect::<Vec<_>>().join(", "),
                vals.join(", ")
            )
        }
        Query::Update { table, set, where_ } => {
            let set_str: Vec<String> = set
                .iter()
                .map(|(c, v)| format!("{} = {}", quote_ident(c), value_to_sql(v)))
                .collect();
            let mut sql = format!("UPDATE {} SET {}", quote_table(table), set_str.join(", "));

            if let Some(w) = where_ {
                sql.push_str(&format!(" WHERE {}", expr_to_sql(w)));
            }

            sql
        }
        Query::Delete { table, where_ } => {
            let mut sql = format!("DELETE FROM {}", quote_table(table));

            if let Some(w) = where_ {
                sql.push_str(&format!(" WHERE {}", expr_to_sql(w)));
            }

            sql
        }
        Query::Create { table, columns } => {
            let cols: Vec<String> = columns
                .iter()
                    .map(|c| {
                        let mut parts = vec![
                            quote_ident(&c.name),
                            column_type_to_sql(&c.type_).to_string(),
                        ];

                    if c.primary_key {
                        parts.push("PRIMARY KEY".to_string());
                    }

                    if c.unique && !c.primary_key {
                        parts.push("UNIQUE".to_string());
                    }

                    if !c.nullable && !c.primary_key {
                        parts.push("NOT NULL".to_string());
                    }

                    if let Some(default) = &c.default {
                        parts.push(format!("DEFAULT {}", value_to_sql(default)));
                    }

                    parts.join(" ")
                })
                .collect();

            format!("CREATE TABLE {} ({})", quote_table(table), cols.join(", "))
        }
        Query::Drop { table, if_exists } => {
            format!(
                "DROP TABLE{} {}",
                if *if_exists { " IF EXISTS" } else { "" },
                quote_table(table)
            )
        }
    }
}

/// Convert expression to SQL string
fn expr_to_sql(expr: &Expr) -> String {
    match expr {
        Expr::Column(name) => quote_ident(name),
        Expr::Literal(v) => value_to_sql(v),
        Expr::BinOp(left, op, right) => {
            format!("({} {} {})", expr_to_sql(left), op_to_sql(op), expr_to_sql(right))
        }
        Expr::UnaryOp(op, inner) => format!("{} {}", op_to_sql(op), expr_to_sql(inner)),
        Expr::Function(name, args) => {
            let args_str: Vec<String> = args.iter().map(expr_to_sql).collect();
            format!("{}({})", name, args_str.join(", "))
        }
        Expr::Null => "NULL".to_string(),
        Expr::Parameter(n) => format!("${}", n + 1),
    }
}

/// Convert operator to SQL string
fn op_to_sql(op: &Op) -> &str {
    match op {
        Op::Eq => "=",
        Op::Ne => "!=",
        Op::Lt => "<",
        Op::Le => "<=",
        Op::Gt => ">",
        Op::Ge => ">=",
        Op::And => "AND",
        Op::Or => "OR",
        Op::Not => "NOT",
        Op::Add => "+",
        Op::Sub => "-",
        Op::Mul => "*",
        Op::Div => "/",
        Op::Mod => "%",
        Op::Like => "LIKE",
        Op::NotLike => "NOT LIKE",
        Op::IsNull => "IS NULL",
        Op::IsNotNull => "IS NOT NULL",
    }
}

/// Convert value to SQL string
fn value_to_sql(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_string(),
        Value::Bool(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Text(s) => format!("'{}", escape_string(s)),
        Value::Blob(b) => format!("X'{}'", hex_encode(b)),
    }
}

/// Convert column type to SQL string
fn column_type_to_sql(t: &ColumnType) -> &str {
    match t {
        ColumnType::Integer => "INTEGER",
        ColumnType::Real => "REAL",
        ColumnType::Text => "TEXT",
        ColumnType::Blob => "BLOB",
        ColumnType::Boolean => "BOOLEAN",
        ColumnType::Timestamp => "TIMESTAMP",
        ColumnType::Varchar(_n) => "VARCHAR", // Simplified
    }
}

/// Quote table/identifier
fn quote_table(name: &str) -> String {
    format!("\"{}\"", name)
}

/// Quote identifier
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name)
}

/// Escape string literal
fn escape_string(s: &str) -> String {
    s.replace('\'', "''")
}

/// Hex encode blob
fn hex_encode(b: &[u8]) -> String {
    b.iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_builder() {
        let query = SelectBuilder::new("users")
            .columns(vec!["id", "name"])
            .where_(Expr::BinOp(
                Box::new(Expr::Column("age".to_string())),
                Op::Gt,
                Box::new(Expr::Literal(Value::Int(18))),
            ))
            .build();

        assert_eq!(query.table(), "users");
    }

    #[test]
    fn test_insert_builder() {
        let query = InsertBuilder::new("users")
            .set("name", Value::Text("Alice".to_string()))
            .set("age", Value::Int(30))
            .build();

        let sql = to_sql(&query);
        assert!(sql.contains("INSERT INTO"));
        assert!(sql.contains("'Alice'"));
    }

    #[test]
    fn test_value_to_sql() {
        assert_eq!(value_to_sql(&Value::Int(42)), "42");
        assert_eq!(value_to_sql(&Value::Text("test".to_string())), "'test'");
        assert_eq!(value_to_sql(&Value::Null), "NULL");
    }

    #[test]
    fn test_expr_to_sql() {
        let expr = Expr::BinOp(
            Box::new(Expr::Column("x".to_string())),
            Op::Eq,
            Box::new(Expr::Literal(Value::Int(5))),
        );
        assert_eq!(expr_to_sql(&expr), "(\"x\" = 5)");
    }
}
