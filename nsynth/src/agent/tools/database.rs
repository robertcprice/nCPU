//! In-memory database tool.
//!
//! A real, self-contained relational-style store (no external dependency). Rows
//! are stored as string cells keyed by declared columns. Supports table
//! creation, insertion, equality-filtered selection and deletion, counting, and
//! listing. State is shared across invocations via an interior mutex, so the
//! tool can be registered once and reused.

use super::registry::{Tool, ToolCall, ToolError, ToolOutput};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Default)]
struct Table {
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
}

/// In-memory table store tool.
pub struct DbTool {
    tables: Mutex<HashMap<String, Table>>,
}

impl Default for DbTool {
    fn default() -> Self {
        Self::new()
    }
}

impl DbTool {
    pub fn new() -> Self {
        Self {
            tables: Mutex::new(HashMap::new()),
        }
    }

    /// Parse a `col=value` filter into its parts.
    fn parse_filter(spec: &str) -> Result<(String, String), ToolError> {
        spec.split_once('=')
            .map(|(c, v)| (c.trim().to_string(), v.trim().to_string()))
            .ok_or_else(|| {
                ToolError::InvalidArg("where".to_string(), "expected 'col=value'".to_string())
            })
    }
}

impl Tool for DbTool {
    fn name(&self) -> &str {
        "database"
    }

    fn description(&self) -> &str {
        "In-memory table store: create_table, insert, select, delete, count, list_tables"
    }

    fn actions(&self) -> Vec<&'static str> {
        vec![
            "create_table",
            "insert",
            "select",
            "delete",
            "count",
            "list_tables",
        ]
    }

    fn invoke(&self, call: &ToolCall) -> Result<ToolOutput, ToolError> {
        let mut tables = self
            .tables
            .lock()
            .map_err(|e| ToolError::Execution(format!("lock poisoned: {e}")))?;

        match call.action.as_str() {
            "create_table" => {
                let name = call.require("table")?.to_string();
                let columns: Vec<String> = call
                    .require("columns")?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if columns.is_empty() {
                    return Err(ToolError::InvalidArg(
                        "columns".to_string(),
                        "at least one column required".to_string(),
                    ));
                }
                tables.insert(
                    name,
                    Table {
                        columns,
                        rows: Vec::new(),
                    },
                );
                Ok(ToolOutput::new("ok"))
            }
            "insert" => {
                let name = call.require("table")?;
                let table = tables.get_mut(name).ok_or_else(|| {
                    ToolError::InvalidArg("table".to_string(), "no such table".to_string())
                })?;
                let values: Vec<String> = call
                    .require("values")?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
                if values.len() != table.columns.len() {
                    return Err(ToolError::InvalidArg(
                        "values".to_string(),
                        format!(
                            "expected {} values, got {}",
                            table.columns.len(),
                            values.len()
                        ),
                    ));
                }
                table.rows.push(values);
                Ok(ToolOutput::new("ok").with_meta("rows", table.rows.len().to_string()))
            }
            "select" => {
                let name = call.require("table")?;
                let table = tables.get(name).ok_or_else(|| {
                    ToolError::InvalidArg("table".to_string(), "no such table".to_string())
                })?;

                let filter = match call.optional("where") {
                    Some(spec) => {
                        let (col, val) = Self::parse_filter(spec)?;
                        let idx =
                            table
                                .columns
                                .iter()
                                .position(|c| c == &col)
                                .ok_or_else(|| {
                                    ToolError::InvalidArg(
                                        "where".to_string(),
                                        format!("no column '{col}'"),
                                    )
                                })?;
                        Some((idx, val))
                    }
                    None => None,
                };

                let mut out = String::new();
                out.push_str(&table.columns.join(","));
                out.push('\n');
                let mut matched = 0usize;
                for row in &table.rows {
                    let keep = match &filter {
                        Some((idx, val)) => row.get(*idx).map(|c| c == val).unwrap_or(false),
                        None => true,
                    };
                    if keep {
                        out.push_str(&row.join(","));
                        out.push('\n');
                        matched += 1;
                    }
                }
                Ok(ToolOutput::new(out.trim_end().to_string())
                    .with_meta("matched", matched.to_string()))
            }
            "delete" => {
                let name = call.require("table")?;
                let table = tables.get_mut(name).ok_or_else(|| {
                    ToolError::InvalidArg("table".to_string(), "no such table".to_string())
                })?;
                let (col, val) = Self::parse_filter(call.require("where")?)?;
                let idx = table
                    .columns
                    .iter()
                    .position(|c| c == &col)
                    .ok_or_else(|| {
                        ToolError::InvalidArg("where".to_string(), format!("no column '{col}'"))
                    })?;
                let before = table.rows.len();
                table
                    .rows
                    .retain(|row| row.get(idx).map(|c| c != &val).unwrap_or(true));
                let removed = before - table.rows.len();
                Ok(ToolOutput::new("ok").with_meta("removed", removed.to_string()))
            }
            "count" => {
                let name = call.require("table")?;
                let table = tables.get(name).ok_or_else(|| {
                    ToolError::InvalidArg("table".to_string(), "no such table".to_string())
                })?;
                Ok(ToolOutput::new(table.rows.len().to_string()))
            }
            "list_tables" => {
                let mut names: Vec<String> = tables.keys().cloned().collect();
                names.sort();
                Ok(ToolOutput::new(names.join("\n")).with_meta("count", names.len().to_string()))
            }
            other => Err(ToolError::UnknownAction {
                tool: "database".to_string(),
                action: other.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> DbTool {
        let db = DbTool::new();
        db.invoke(
            &ToolCall::new("create_table")
                .arg("table", "users")
                .arg("columns", "id,name"),
        )
        .unwrap();
        db.invoke(
            &ToolCall::new("insert")
                .arg("table", "users")
                .arg("values", "1,alice"),
        )
        .unwrap();
        db.invoke(
            &ToolCall::new("insert")
                .arg("table", "users")
                .arg("values", "2,bob"),
        )
        .unwrap();
        db
    }

    #[test]
    fn test_create_insert_count() {
        let db = setup();
        let count = db
            .invoke(&ToolCall::new("count").arg("table", "users"))
            .unwrap();
        assert_eq!(count.content, "2");
    }

    #[test]
    fn test_select_all_and_filtered() {
        let db = setup();
        let all = db
            .invoke(&ToolCall::new("select").arg("table", "users"))
            .unwrap();
        assert_eq!(all.metadata.get("matched").map(|s| s.as_str()), Some("2"));

        let filtered = db
            .invoke(
                &ToolCall::new("select")
                    .arg("table", "users")
                    .arg("where", "name=alice"),
            )
            .unwrap();
        assert_eq!(
            filtered.metadata.get("matched").map(|s| s.as_str()),
            Some("1")
        );
        assert!(filtered.content.contains("alice"));
        assert!(!filtered.content.contains("bob"));
    }

    #[test]
    fn test_delete() {
        let db = setup();
        let del = db
            .invoke(
                &ToolCall::new("delete")
                    .arg("table", "users")
                    .arg("where", "id=1"),
            )
            .unwrap();
        assert_eq!(del.metadata.get("removed").map(|s| s.as_str()), Some("1"));
        let count = db
            .invoke(&ToolCall::new("count").arg("table", "users"))
            .unwrap();
        assert_eq!(count.content, "1");
    }

    #[test]
    fn test_insert_arity_mismatch() {
        let db = setup();
        let err = db
            .invoke(
                &ToolCall::new("insert")
                    .arg("table", "users")
                    .arg("values", "3"),
            )
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArg(_, _)));
    }

    #[test]
    fn test_list_tables() {
        let db = setup();
        let out = db.invoke(&ToolCall::new("list_tables")).unwrap();
        assert_eq!(out.content, "users");
    }
}
