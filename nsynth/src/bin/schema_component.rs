//! SCHEMA -> VERIFIED TYPED COMPONENT (the novel Phase-2 front door).
//!
//! A schema is a DECIDABLE decomposition: "a todo list where each task has a title and a priority
//! number" fully determines a typed record, a collection with CRUD, and the canonical tests those
//! obey. This bin does the "model writes the spec" half DETERMINISTICALLY: it parses the prose into a
//! schema and EMITS a Rust crate = the record struct + a collection struct + method STUBS + the
//! canonical CRUD/aggregate TESTS (the spec). The engine's multi-hole solver then fills the stubs and
//! `cargo test` verifies -- so the whole component is synthesized model-free, every method carrying a
//! passing test. No model writes code here; the schema writes the spec, the solver writes the code.
//!
//! Usage:
//!   schema_component "<prose>" [out_dir]
//!   e.g. schema_component "a todo list where each task has a title and a priority number and a done flag"
//! Then:  coding_agent --root <out_dir> query "fix the failing tests"   # fills + verifies

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy};
use std::path::PathBuf;

#[derive(Clone)]
struct Field {
    name: String,
    ty: FieldTy,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum FieldTy {
    Int,
    Bool,
    Str,
}

impl FieldTy {
    fn rust(self) -> &'static str {
        match self {
            FieldTy::Int => "i64",
            FieldTy::Bool => "bool",
            FieldTy::Str => "String",
        }
    }
    /// A concrete literal for the k-th record in the generated tests. `seed` is the field's column
    /// index so DISTINCT int columns get DISTINCT data (else `total_quantity` can't be told from
    /// `total_price` and the solver maps the wrong field).
    fn sample(self, name: &str, seed: usize, k: usize) -> String {
        match self {
            FieldTy::Int => Self::int_value(seed, k).to_string(),
            FieldTy::Bool => if k % 2 == 0 { "true".into() } else { "false".into() },
            FieldTy::Str => format!("\"{name}{k}\".to_string()"),
        }
    }
    fn sample_value(self, seed: usize, k: usize) -> i64 {
        Self::int_value(seed, k)
    }
    fn int_value(seed: usize, k: usize) -> i64 {
        // per-column offset (11*seed) keeps columns distinct; the base still puts a non-extreme value
        // at index 2 so an indexed read can't be faked by min/max/first/last.
        ((k as i64) * 4) % 7 + 2 + (seed as i64) * 11
    }
}

/// Turn a noun phrase into an UpperCamel type name and a snake identifier.
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
    word.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_').collect::<String>().to_lowercase()
}

fn field_type(phrase: &str) -> FieldTy {
    let p = phrase.to_lowercase();
    let int_hints = ["number", "count", "priority", "amount", "age", "quantity", "price", "score", "int", "points", "size", "rank", "level"];
    let bool_hints = ["flag", "bool", "done", "active", "enabled", "complete", "is ", "has ", "boolean"];
    if int_hints.iter().any(|h| p.contains(h)) {
        FieldTy::Int
    } else if bool_hints.iter().any(|h| p.contains(h)) {
        FieldTy::Bool
    } else {
        FieldTy::Str
    }
}

/// The last alphabetic word of a field phrase names the field (`a priority number` -> `priority`),
/// skipping the type-hint word when it trails.
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
    words.last().map(|w| ident(w)).unwrap_or_else(|| "field".into())
}

/// Parse "<collection> where each <record> has <f>, <f> and <f>" (also accepts "with" for "has").
fn parse_schema(prose: &str) -> Option<(String, String, Vec<Field>)> {
    let lower = prose.to_lowercase();
    // Split the record/fields clause.
    let (head, rest) = if let Some(i) = lower.find("each ") {
        (&prose[..i], &prose[i + 5..])
    } else {
        (prose, "")
    };
    let (record_words, fields_clause) = {
        let r = rest.to_lowercase();
        let cut = r.find(" has ").or_else(|| r.find(" with ")).or_else(|| r.find(" having "));
        match cut {
            Some(i) => {
                let marker_len = if r[i..].starts_with(" has ") { 5 } else if r[i..].starts_with(" with ") { 6 } else { 8 };
                (&rest[..i], &rest[i + marker_len..])
            }
            None => (rest, ""),
        }
    };
    // Collection name: the head noun phrase, stripping articles/verbs and STOPPING at a connective
    // ("a todo list where ..." -> TodoList, not TodoListWhere).
    let head_words: Vec<&str> = head
        .split_whitespace()
        .skip_while(|w| matches!(w.to_lowercase().as_str(), "a" | "an" | "the" | "build" | "me" | "create" | "make" | "please"))
        .take_while(|w| !matches!(w.to_lowercase().as_str(), "where" | "that" | "which" | "in" | "whose" | "with" | "having" | "has"))
        .take(3)
        .collect();
    let collection = if head_words.is_empty() { "Collection".to_string() } else { camel(&head_words) };
    // Record name: the noun after "each".
    let rec_words: Vec<&str> = record_words.split_whitespace().take(2).collect();
    let record = if rec_words.is_empty() { "Record".to_string() } else { camel(&rec_words[..1]) };
    // Fields: split the clause on commas / "and".
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
        fields.push(Field { name, ty: field_type(phrase) });
    }
    if fields.is_empty() {
        return None;
    }
    Some((collection, record, fields))
}

fn emit_crate(collection: &str, record: &str, fields: &[Field]) -> String {
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
    // (column index, field) for the int fields — the index seeds distinct per-column sample data.
    let int_fields: Vec<(usize, &Field)> = fields.iter().enumerate().filter(|(_, f)| f.ty == FieldTy::Int).collect();

    // impl: full CRUD over the typed collection (STUBS the solver fills). new/add/count/is_empty/
    // clear/remove_at, and per-int-field total_/max_/<field>_at/set_<field>.
    let mut methods = String::new();
    methods.push_str(&format!("    pub fn new() -> Self {{ {collection} {{ items: vec![] }} }}\n"));
    methods.push_str(&format!("    pub fn add(&mut self, {add_params}) {{}}\n"));
    methods.push_str("    pub fn count(&self) -> i64 {}\n");
    methods.push_str("    pub fn is_empty(&self) -> bool {}\n");
    methods.push_str("    pub fn clear(&mut self) {}\n");
    methods.push_str("    pub fn remove_at(&mut self, i: i64) {}\n");
    for (_, f) in &int_fields {
        methods.push_str(&format!("    pub fn total_{}(&self) -> i64 {{}}\n", f.name));
        methods.push_str(&format!("    pub fn max_{}(&self) -> i64 {{}}\n", f.name));
        methods.push_str(&format!("    pub fn {}_at(&self, i: i64) -> i64 {{}}\n", f.name));
        methods.push_str(&format!("    pub fn set_{}(&mut self, i: i64, v: i64) {{}}\n", f.name));
    }
    // Query by KEY: if the record has a String field, add contains + per-int-field lookup-by-key.
    let key_field: Option<&Field> = fields.iter().find(|f| f.ty == FieldTy::Str);
    if let Some(kf) = key_field {
        methods.push_str(&format!("    pub fn contains(&self, {}: String) -> bool {{}}\n", kf.name));
        for (_, f) in &int_fields {
            methods.push_str(&format!("    pub fn {}_of(&self, {}: String) -> i64 {{}}\n", f.name, kf.name));
        }
    }

    // Canonical tests: add N sample records, then one assertion PER METHOD (not a mega-test) so the
    // multi-hole solver gets a gradient. N=4 with the DISTINGUISHING value at index 2 (neither the
    // min/max/sum nor the first/last element) so an INDEXED read can't be faked by an aggregate.
    let n = 4usize;
    let read_idx = 2usize;
    let add_calls: String = (0..n)
        .map(|k| {
            let args: Vec<String> = fields.iter().enumerate().map(|(fi, f)| f.ty.sample(&f.name, fi, k)).collect();
            format!("        c.add({});\n", args.join(", "))
        })
        .collect();
    // A test that fills c with the sample records then asserts one call.
    let full = |name: &str, extra: &str, call: &str, expected: &str| -> String {
        format!(
            "    #[test]\n    fn {name}() {{\n        let mut c = {collection}::new();\n{add_calls}{extra}        assert_eq!(c.{call}, {expected});\n    }}\n"
        )
    };
    let mut tests = String::new();
    tests.push_str(&full("t_count", "", "count()", &n.to_string()));
    // is_empty on a FRESH collection is true (no adds).
    tests.push_str(&format!(
        "    #[test]\n    fn t_is_empty() {{\n        let c = {collection}::new();\n        assert_eq!(c.is_empty(), true);\n    }}\n"
    ));
    tests.push_str(&full("t_clear", "        c.clear();\n", "count()", "0"));
    tests.push_str(&full("t_remove", "        c.remove_at(0);\n", "count()", &(n - 1).to_string()));
    for (seed, f) in &int_fields {
        let vals: Vec<i64> = (0..n).map(|k| f.ty.sample_value(*seed, k)).collect();
        let sum: i64 = vals.iter().sum();
        let max: i64 = *vals.iter().max().unwrap();
        let at = vals[read_idx];
        tests.push_str(&full(&format!("t_total_{}", f.name), "", &format!("total_{}()", f.name), &sum.to_string()));
        tests.push_str(&full(&format!("t_max_{}", f.name), "", &format!("max_{}()", f.name), &max.to_string()));
        tests.push_str(&full(&format!("t_{}_at", f.name), "", &format!("{}_at({read_idx})", f.name), &at.to_string()));
        // set then read back at the SAME index: pins set_<field> once <field>_at is right.
        tests.push_str(&full(
            &format!("t_set_{}", f.name),
            &format!("        c.set_{}(0, 12345);\n", f.name),
            &format!("{}_at(0)", f.name),
            "12345",
        ));
    }
    // Query-by-key tests, looking up the DISTINGUISHING record (index `read_idx`) by its String key.
    if let Some(kf) = key_field {
        let key = format!("\"{}{read_idx}\".to_string()", kf.name);
        tests.push_str(&full("t_contains", "", &format!("contains({key})"), "true"));
        tests.push_str(&full("t_not_contains", "", "contains(\"__absent__\".to_string())", "false"));
        for (seed, f) in &int_fields {
            let at = f.ty.sample_value(*seed, read_idx);
            tests.push_str(&full(&format!("t_{}_of", f.name), "", &format!("{}_of({key})", f.name), &at.to_string()));
        }
    }

    format!(
        "#[derive(Clone)]\npub struct {record} {{ {rec_fields} }}\n\n\
         pub struct {collection} {{ pub items: Vec<{record}> }}\n\n\
         impl {collection} {{\n{methods}}}\n\n\
         #[cfg(test)]\nmod tests {{\n    use super::*;\n{tests}}}\n"
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prose = args.get(1).cloned().unwrap_or_default();
    if prose.trim().is_empty() {
        eprintln!("usage: schema_component \"<prose schema>\" [out_dir] [--no-fill]");
        std::process::exit(2);
    }
    // `--no-fill` keeps the old emit-only behavior; by default this now runs the WHOLE Phase-2 loop
    // (prose -> schema -> crate -> fill -> verify) as ONE command and reports the verified result.
    let no_fill = args.iter().any(|a| a == "--no-fill");
    let out_dir = args
        .iter()
        .skip(2)
        .find(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("schema_component_out"));

    let Some((collection, record, fields)) = parse_schema(&prose) else {
        eprintln!("could not parse a schema (need e.g. \"... where each X has a, b and c\")");
        std::process::exit(1);
    };

    let crate_name = collection.to_lowercase();
    let lib = emit_crate(&collection, &record, &fields);
    let src = out_dir.join("src");
    if let Err(e) = std::fs::create_dir_all(&src) {
        eprintln!("mkdir failed: {e}");
        std::process::exit(1);
    }
    let cargo = format!("[package]\nname = \"{crate_name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n");
    if std::fs::write(out_dir.join("Cargo.toml"), cargo).is_err()
        || std::fs::write(src.join("lib.rs"), &lib).is_err()
    {
        eprintln!("write failed");
        std::process::exit(1);
    }
    let field_list: Vec<String> = fields.iter().map(|f| format!("{}:{}", f.name, f.ty.rust())).collect();
    println!("schema: {collection} {{ items: Vec<{record}> }}  record {record} {{ {} }}", field_list.join(", "));
    println!("wrote crate to {}", out_dir.display());

    if no_fill {
        println!("fill + verify:  coding_agent --root {} query \"fix the failing tests\"", out_dir.display());
        return;
    }

    // FILL + VERIFY in-process — the same path `coding_agent --root <dir> query "fix the failing tests"`
    // drives. The emitted crate is a struct + empty-body method stubs + one test per method; the
    // model-free engine fills every body against those tests, cargo-gated. One command now yields a
    // VERIFIED working crate, not a scaffold plus instructions.
    println!("filling {} method bodies (model-free) ...", count_stub_methods(&lib));
    let mut session = match CodingAgentSession::load(&out_dir, GuardrailPolicy::default(), "schema") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("session load failed: {e}");
            std::process::exit(1);
        }
    };
    let result = session.handle_query("fix the failing tests");
    // Independently re-run the crate's own test suite so the reported result is cargo's, not ours.
    let verified = std::process::Command::new("cargo")
        .arg("test")
        .current_dir(&out_dir)
        .output();
    match verified {
        Ok(out) => {
            // `cargo test` prints one `test result:` line PER binary (lib unit tests, then the empty
            // Doc-tests). Sum across all of them rather than taking one line, so the report reflects
            // the whole suite, not the trailing 0-test doc block.
            let text = String::from_utf8_lossy(&out.stdout);
            let (mut passed, mut failed) = (0u32, 0u32);
            for line in text.lines().filter(|l| l.contains("test result:")) {
                passed += count_before(line, "passed");
                failed += count_before(line, "failed");
            }
            if out.status.success() && failed == 0 {
                println!(
                    "VERIFIED  {passed} tests passing, 0 failing  (repair route: {:?}, success: {})",
                    result.route, result.success
                );
            } else {
                println!(
                    "NOT GREEN  {passed} passing, {failed} failing — some bodies left unfilled (honest refusal, no wrong code shipped)"
                );
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("cargo test could not run: {e}");
            std::process::exit(1);
        }
    }
}

/// The integer immediately preceding `word` in a `cargo` "N passed; M failed" summary line, or 0.
fn count_before(line: &str, word: &str) -> u32 {
    let toks: Vec<&str> = line.split_whitespace().collect();
    toks.windows(2)
        .find(|w| w[1].trim_end_matches(&[';', ','][..]) == word)
        .and_then(|w| w[0].parse().ok())
        .unwrap_or(0)
}

/// Count the empty-body method stubs (`{}` right after a signature) the engine must fill — for a
/// one-line "filling N method bodies" progress note. Heuristic display only.
fn count_stub_methods(lib: &str) -> usize {
    lib.matches(") {}").count() + lib.matches("-> i64 {}").count() + lib.matches("-> bool {}").count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_collection_record_and_typed_fields() {
        let (c, r, f) =
            parse_schema("a todo list where each task has a title and a priority number and a done flag").unwrap();
        assert_eq!(c, "TodoList", "connective 'where' must not leak into the name");
        assert_eq!(r, "Task");
        assert_eq!(f.len(), 3);
        assert_eq!((f[0].name.as_str(), f[0].ty), ("title", FieldTy::Str));
        assert_eq!((f[1].name.as_str(), f[1].ty), ("priority", FieldTy::Int));
        assert_eq!((f[2].name.as_str(), f[2].ty), ("done", FieldTy::Bool));
    }

    #[test]
    fn emits_typed_stubs_and_per_method_tests() {
        let (c, r, f) = parse_schema("an inventory where each product has a price number").unwrap();
        let src = emit_crate(&c, &r, &f);
        assert!(src.contains("pub struct Product { pub price: i64 }"), "record struct");
        assert!(src.contains("pub fn add(&mut self, price: i64) {}"), "add stub");
        assert!(src.contains("pub fn total_price(&self) -> i64 {}"), "sum getter stub");
        assert!(src.contains("pub fn max_price(&self) -> i64 {}"), "max getter stub");
        // full CRUD stubs.
        assert!(src.contains("pub fn is_empty(&self) -> bool {}"), "is_empty stub");
        assert!(src.contains("pub fn remove_at(&mut self, i: i64) {}"), "remove stub");
        assert!(src.contains("pub fn price_at(&self, i: i64) -> i64 {}"), "indexed read stub");
        assert!(src.contains("pub fn set_price(&mut self, i: i64, v: i64) {}"), "field-update stub");
        // per-METHOD tests give the solver a gradient (each fill flips one test), not one mega-test.
        assert!(src.contains("fn t_count()") && src.contains("fn t_price_at()") && src.contains("fn t_set_price()"));
        assert_eq!(src.matches("#[test]").count(), 8, "count/is_empty/clear/remove + 4 per int field");
    }
}
