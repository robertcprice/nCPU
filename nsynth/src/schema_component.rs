//! SCHEMA → VERIFIED TYPED COMPONENT (Phase-2 front door, library form).
//!
//! A schema is a DECIDABLE decomposition: "a todo list where each task has a title and a
//! priority number" fully determines a typed record, a collection with CRUD, and the
//! canonical tests those obey. This module parses the prose into a schema and EMITS a
//! Rust crate = record struct + collection + method STUBS + per-method TESTS. The
//! engine's multi-hole solver then fills the stubs and `cargo test` verifies — model-free.
//!
//! Product path: [`try_write_schema_crate`] from `handle_query` / [`crate::whole_software`].

use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub ty: FieldTy,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FieldTy {
    Int,
    Bool,
    Str,
}

impl FieldTy {
    pub fn rust(self) -> &'static str {
        match self {
            FieldTy::Int => "i64",
            FieldTy::Bool => "bool",
            FieldTy::Str => "String",
        }
    }
    /// A concrete literal for the k-th record in the generated tests. `seed` is the field's
    /// column index so DISTINCT int columns get DISTINCT data.
    fn sample(self, name: &str, seed: usize, k: usize) -> String {
        match self {
            FieldTy::Int => Self::int_value(seed, k).to_string(),
            FieldTy::Bool => {
                if k % 2 == 0 {
                    "true".into()
                } else {
                    "false".into()
                }
            }
            FieldTy::Str => format!("\"{name}{k}\".to_string()"),
        }
    }
    fn sample_value(self, seed: usize, k: usize) -> i64 {
        Self::int_value(seed, k)
    }
    fn int_value(seed: usize, k: usize) -> i64 {
        ((k as i64) * 4) % 7 + 2 + (seed as i64) * 11
    }
}

/// Parsed schema: collection type, record type, typed fields.
#[derive(Clone, Debug)]
pub struct Schema {
    pub collection: String,
    pub record: String,
    pub fields: Vec<Field>,
}

/// Turn a noun phrase into an UpperCamel type name.
fn camel(words: &[&str]) -> String {
    words
        .iter()
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

fn ident(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<String>()
        .to_lowercase()
}

fn field_type(phrase: &str) -> FieldTy {
    let p = phrase.to_lowercase();
    let int_hints = [
        "number", "count", "priority", "amount", "age", "quantity", "price", "score", "int",
        "points", "size", "rank", "level", "stock",
    ];
    let bool_hints = [
        "flag", "bool", "done", "active", "enabled", "complete", "is ", "has ", "boolean",
    ];
    if int_hints.iter().any(|h| p.contains(h)) {
        FieldTy::Int
    } else if bool_hints.iter().any(|h| p.contains(h)) {
        FieldTy::Bool
    } else {
        FieldTy::Str
    }
}

/// The last alphabetic word of a field phrase names the field (`a priority number` → `priority`).
fn field_name(phrase: &str) -> String {
    let stop = ["number", "flag", "bool", "int", "boolean", "value", "field"];
    let words: Vec<&str> = phrase
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the" | "with" | "and"))
        .collect();
    for w in words.iter().rev() {
        let id = ident(w);
        if !id.is_empty() && !stop.contains(&id.as_str()) {
            return id;
        }
    }
    words
        .last()
        .map(|w| ident(w))
        .unwrap_or_else(|| "field".into())
}

/// Parse "<collection> where each <record> has <f>, <f> and <f>" (also accepts "with" for "has").
pub fn parse_schema(prose: &str) -> Option<Schema> {
    let lower = prose.to_lowercase();
    let (head, rest) = if let Some(i) = lower.find("each ") {
        (&prose[..i], &prose[i + 5..])
    } else {
        return None;
    };
    let (record_words, fields_clause) = {
        let r = rest.to_lowercase();
        let cut = r
            .find(" has ")
            .or_else(|| r.find(" with "))
            .or_else(|| r.find(" having "));
        match cut {
            Some(i) => {
                let marker_len = if r[i..].starts_with(" has ") {
                    5
                } else if r[i..].starts_with(" with ") {
                    6
                } else {
                    8
                };
                (&rest[..i], &rest[i + marker_len..])
            }
            None => return None,
        }
    };
    let head_words: Vec<&str> = head
        .split_whitespace()
        .skip_while(|w| {
            matches!(
                w.to_lowercase().as_str(),
                "a" | "an"
                    | "the"
                    | "build"
                    | "me"
                    | "create"
                    | "make"
                    | "please"
                    | "implement"
                    | "write"
                    | "generate"
            )
        })
        .take_while(|w| {
            !matches!(
                w.to_lowercase().as_str(),
                "where" | "that" | "which" | "in" | "whose" | "with" | "having" | "has"
            )
        })
        .take(3)
        .collect();
    let collection = if head_words.is_empty() {
        "Collection".to_string()
    } else {
        camel(&head_words)
    };
    let rec_words: Vec<&str> = record_words.split_whitespace().take(2).collect();
    let record = if rec_words.is_empty() {
        "Record".to_string()
    } else {
        camel(&rec_words[..1])
    };
    let mut fields = Vec::new();
    for chunk in fields_clause.replace(" and ", ", ").split(',') {
        let phrase = chunk.trim();
        if phrase.is_empty() {
            continue;
        }
        let name = field_name(phrase);
        if name.is_empty() || fields.iter().any(|f: &Field| f.name == name) {
            continue;
        }
        fields.push(Field {
            name,
            ty: field_type(phrase),
        });
    }
    if fields.is_empty() {
        return None;
    }
    Some(Schema {
        collection,
        record,
        fields,
    })
}

/// True when prose is a decidable schema (Phase-2 front door can fire).
pub fn is_schema_prose(prose: &str) -> bool {
    parse_schema(prose).is_some()
}

pub fn emit_crate(schema: &Schema) -> String {
    let collection = &schema.collection;
    let record = &schema.record;
    let fields = &schema.fields;
    let rec_fields: String = fields
        .iter()
        .map(|f| format!("pub {}: {}", f.name, f.ty.rust()))
        .collect::<Vec<_>>()
        .join(", ");
    let add_params: String = fields
        .iter()
        .map(|f| format!("{}: {}", f.name, f.ty.rust()))
        .collect::<Vec<_>>()
        .join(", ");
    let int_fields: Vec<(usize, &Field)> = fields
        .iter()
        .enumerate()
        .filter(|(_, f)| f.ty == FieldTy::Int)
        .collect();

    let mut methods = String::new();
    methods.push_str(&format!(
        "    pub fn new() -> Self {{ {collection} {{ items: vec![] }} }}\n"
    ));
    methods.push_str(&format!("    pub fn add(&mut self, {add_params}) {{}}\n"));
    methods.push_str("    pub fn count(&self) -> i64 {}\n");
    methods.push_str("    pub fn is_empty(&self) -> bool {}\n");
    methods.push_str("    pub fn clear(&mut self) {}\n");
    methods.push_str("    pub fn remove_at(&mut self, i: i64) {}\n");
    for (_, f) in &int_fields {
        methods.push_str(&format!("    pub fn total_{}(&self) -> i64 {{}}\n", f.name));
        methods.push_str(&format!("    pub fn max_{}(&self) -> i64 {{}}\n", f.name));
        methods.push_str(&format!(
            "    pub fn {}_at(&self, i: i64) -> i64 {{}}\n",
            f.name
        ));
        methods.push_str(&format!(
            "    pub fn set_{}(&mut self, i: i64, v: i64) {{}}\n",
            f.name
        ));
    }
    let key_field: Option<&Field> = fields.iter().find(|f| f.ty == FieldTy::Str);
    if let Some(kf) = key_field {
        methods.push_str(&format!(
            "    pub fn contains(&self, {}: String) -> bool {{}}\n",
            kf.name
        ));
        for (_, f) in &int_fields {
            methods.push_str(&format!(
                "    pub fn {}_of(&self, {}: String) -> i64 {{}}\n",
                f.name, kf.name
            ));
        }
    }

    let n = 4usize;
    let read_idx = 2usize;
    let add_calls: String = (0..n)
        .map(|k| {
            let args: Vec<String> = fields
                .iter()
                .enumerate()
                .map(|(fi, f)| f.ty.sample(&f.name, fi, k))
                .collect();
            format!("        c.add({});\n", args.join(", "))
        })
        .collect();
    let full = |name: &str, extra: &str, call: &str, expected: &str| -> String {
        format!(
            "    #[test]\n    fn {name}() {{\n        let mut c = {collection}::new();\n{add_calls}{extra}        assert_eq!(c.{call}, {expected});\n    }}\n"
        )
    };
    let mut tests = String::new();
    tests.push_str(&full("t_count", "", "count()", &n.to_string()));
    tests.push_str(&format!(
        "    #[test]\n    fn t_is_empty() {{\n        let c = {collection}::new();\n        assert_eq!(c.is_empty(), true);\n    }}\n"
    ));
    tests.push_str(&full("t_clear", "        c.clear();\n", "count()", "0"));
    tests.push_str(&full(
        "t_remove",
        "        c.remove_at(0);\n",
        "count()",
        &(n - 1).to_string(),
    ));
    for (seed, f) in &int_fields {
        let vals: Vec<i64> = (0..n).map(|k| f.ty.sample_value(*seed, k)).collect();
        let sum: i64 = vals.iter().sum();
        let max: i64 = *vals.iter().max().unwrap();
        let at = vals[read_idx];
        tests.push_str(&full(
            &format!("t_total_{}", f.name),
            "",
            &format!("total_{}()", f.name),
            &sum.to_string(),
        ));
        tests.push_str(&full(
            &format!("t_max_{}", f.name),
            "",
            &format!("max_{}()", f.name),
            &max.to_string(),
        ));
        tests.push_str(&full(
            &format!("t_{}_at", f.name),
            "",
            &format!("{}_at({read_idx})", f.name),
            &at.to_string(),
        ));
        tests.push_str(&full(
            &format!("t_set_{}", f.name),
            &format!("        c.set_{}(0, 12345);\n", f.name),
            &format!("{}_at(0)", f.name),
            "12345",
        ));
    }
    if let Some(kf) = key_field {
        let key = format!("\"{}{read_idx}\".to_string()", kf.name);
        tests.push_str(&full("t_contains", "", &format!("contains({key})"), "true"));
        tests.push_str(&full(
            "t_not_contains",
            "",
            "contains(\"__absent__\".to_string())",
            "false",
        ));
        for (seed, f) in &int_fields {
            let at = f.ty.sample_value(*seed, read_idx);
            tests.push_str(&full(
                &format!("t_{}_of", f.name),
                "",
                &format!("{}_of({key})", f.name),
                &at.to_string(),
            ));
        }
    }

    format!(
        "#[derive(Clone)]\npub struct {record} {{ {rec_fields} }}\n\n\
         pub struct {collection} {{ pub items: Vec<{record}> }}\n\n\
         impl {collection} {{\n{methods}}}\n\n\
         #[cfg(test)]\nmod tests {{\n    use super::*;\n{tests}}}\n"
    )
}

/// Result of writing a schema scaffold crate.
#[derive(Debug, Clone)]
pub struct WrittenSchemaCrate {
    pub root: PathBuf,
    pub collection: String,
    pub record: String,
    pub n_fields: usize,
    pub n_tests: usize,
    pub method: &'static str,
}

/// Parse prose as a schema and write a stub+test crate under `out_dir`.
/// Returns `None` when prose is not schema-shaped.
pub fn try_write_schema_crate(out_dir: &Path, prose: &str) -> Option<WrittenSchemaCrate> {
    let schema = parse_schema(prose)?;
    let lib = emit_crate(&schema);
    let n_tests = lib.matches("#[test]").count();
    write_lib_crate(out_dir, &schema.collection.to_lowercase(), &lib).ok()?;
    Some(WrittenSchemaCrate {
        root: out_dir.to_path_buf(),
        collection: schema.collection,
        record: schema.record,
        n_fields: schema.fields.len(),
        n_tests,
        method: "whole-software:schema",
    })
}

/// Write a Cargo lib crate (`Cargo.toml` + `src/lib.rs`) under `out_dir`.
pub fn write_lib_crate(out_dir: &Path, crate_name: &str, lib_rs: &str) -> Result<(), String> {
    let src = out_dir.join("src");
    std::fs::create_dir_all(&src).map_err(|e| format!("mkdir: {e}"))?;
    let cargo = format!("[package]\nname = \"{crate_name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n");
    std::fs::write(out_dir.join("Cargo.toml"), cargo).map_err(|e| format!("write Cargo.toml: {e}"))?;
    std::fs::write(src.join("lib.rs"), lib_rs).map_err(|e| format!("write lib.rs: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_collection_record_and_typed_fields() {
        let s = parse_schema(
            "a todo list where each task has a title and a priority number and a done flag",
        )
        .unwrap();
        assert_eq!(s.collection, "TodoList");
        assert_eq!(s.record, "Task");
        assert_eq!(s.fields.len(), 3);
        assert_eq!((s.fields[0].name.as_str(), s.fields[0].ty), ("title", FieldTy::Str));
        assert_eq!(
            (s.fields[1].name.as_str(), s.fields[1].ty),
            ("priority", FieldTy::Int)
        );
        assert_eq!((s.fields[2].name.as_str(), s.fields[2].ty), ("done", FieldTy::Bool));
    }

    #[test]
    fn emits_typed_stubs_and_per_method_tests() {
        let s = parse_schema("an inventory where each product has a price number").unwrap();
        let src = emit_crate(&s);
        assert!(src.contains("pub struct Product { pub price: i64 }"));
        assert!(src.contains("pub fn add(&mut self, price: i64) {}"));
        assert!(src.contains("pub fn total_price(&self) -> i64 {}"));
        assert!(src.contains("fn t_count()") && src.contains("fn t_price_at()"));
        assert_eq!(src.matches("#[test]").count(), 8);
    }

    #[test]
    fn try_write_schema_crate_round_trips() {
        let root = std::env::temp_dir().join(format!(
            "nsynth_schema_write_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let written = try_write_schema_crate(
            &root,
            "a shelf where each book has a title and a pages number",
        )
        .expect("schema");
        assert_eq!(written.collection, "Shelf");
        assert!(root.join("Cargo.toml").is_file());
        assert!(root.join("src/lib.rs").is_file());
        let lib = std::fs::read_to_string(root.join("src/lib.rs")).unwrap();
        assert!(lib.contains("pub struct Book"));
        assert!(lib.contains("#[test]"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn non_schema_prose_refuses() {
        assert!(parse_schema("build a snake game with keyboard controls").is_none());
        assert!(parse_schema("add two numbers").is_none());
    }
}
