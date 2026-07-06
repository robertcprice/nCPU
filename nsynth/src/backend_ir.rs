//! Backend intermediate representation (LOOP-3D foundation).
//!
//! Describes a generated local backend as structured data before Rust emission.
//! The renderer turns this IR into a dependency-free stdlib HTTP server artifact.

use crate::backend_mvp::BackendRuleSpec;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StoreKind {
    #[default]
    Memory,
    File,
    Sqlite,
}

impl StoreKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "memory" | "mem" => Some(Self::Memory),
            "file" | "jsonl" | "persistent" => Some(Self::File),
            "sqlite" | "sql" => Some(Self::Sqlite),
            _ => None,
        }
    }

    pub fn cli_name(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::File => "file",
            Self::Sqlite => "sqlite",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteKind {
    Health,
    ListRules,
    ListEvents,
    EvaluateRule,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteSpec {
    pub method: HttpMethod,
    pub path: String,
    pub kind: RouteKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleModel {
    pub name: String,
    pub synthesis_method: String,
    pub rule_code: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreSpec {
    pub kind: StoreKind,
    pub default_path: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendApp {
    pub service_name: &'static str,
    pub description: String,
    pub rules: Vec<RuleModel>,
    pub store: StoreSpec,
    /// REST collection resources ("users", "orders"): each gets an in-memory
    /// list with GET /name (list) and POST /name (append). Empty by default, so
    /// existing rule-only backends are byte-for-byte unchanged.
    pub resources: Vec<String>,
}

impl BackendApp {
    pub fn from_synthesis(
        spec: &BackendRuleSpec,
        rule_code: &str,
        synthesis_method: &str,
        store_kind: StoreKind,
    ) -> Self {
        Self::from_rules(
            spec.english,
            vec![RuleModel {
                name: spec.name.to_string(),
                synthesis_method: synthesis_method.to_string(),
                rule_code: rule_code.trim().to_string(),
            }],
            store_kind,
        )
    }

    pub fn from_rules(description: &str, rules: Vec<RuleModel>, store_kind: StoreKind) -> Self {
        let store = StoreSpec {
            kind: store_kind,
            default_path: match store_kind {
                StoreKind::File => "backend_events.jsonl",
                StoreKind::Sqlite => "backend_events.db",
                StoreKind::Memory => "",
            },
        };
        Self {
            service_name: "generated-rule-backend",
            description: description.to_string(),
            rules,
            store,
            resources: Vec::new(),
        }
    }

    /// Attach REST collection resources (names) to this backend.
    pub fn with_resources(mut self, resources: Vec<String>) -> Self {
        self.resources = resources;
        self
    }

    pub fn render_rust(&self) -> String {
        render_backend_app(self)
    }
}

/// In-memory collection state for the resources (a name -> Vec<String> map).
/// Empty resource list -> empty string (no state, existing backends unchanged).
fn render_resource_state(resources: &[String]) -> String {
    if resources.is_empty() {
        return String::new();
    }
    // Each resource is loaded from its <name>.jsonl file at startup (durable
    // across restarts).
    let loads: String = resources
        .iter()
        .map(|r| format!("        m.insert(\"{r}\".to_string(), load_collection(\"{r}\"));\n"))
        .collect();
    let names_list: String = resources
        .iter()
        .map(|r| format!("\"{r}\""))
        .collect::<Vec<_>>()
        .join(", ");
    RESOURCE_STATE_TEMPLATE
        .replace("NAMES_LOAD", &loads)
        .replace("NAMES_LIST", &names_list)
}

/// Match arms: a single server-rendered admin UI at GET "/" (list + add-form for
/// every resource), then per-resource REST arms (list/create/read/update/delete).
fn render_resource_arms(resources: &[String]) -> String {
    if resources.is_empty() {
        return String::new();
    }
    let mut out = String::from(RESOURCE_INDEX_ARM);
    for r in resources {
        out.push_str(&RESOURCE_ARM_TEMPLATE.replace("RES", r));
    }
    out
}

const RESOURCE_INDEX_ARM: &str = "        (\"GET\", \"/\") => write_bytes(&mut stream, 200, \"text/html; charset=utf-8\", render_index_page().as_bytes()),\n";

const RESOURCE_STATE_TEMPLATE: &str = r#"const RESOURCE_NAMES: &[&str] = &[NAMES_LIST];
static COLLECTIONS: std::sync::OnceLock<Mutex<std::collections::HashMap<String, Vec<String>>>> = std::sync::OnceLock::new();
fn collection_path(name: &str) -> String { format!("{}.jsonl", name) }
fn load_collection(name: &str) -> Vec<String> {
    std::fs::read_to_string(collection_path(name))
        .map(|s| s.lines().filter(|l| !l.is_empty()).map(|l| l.to_string()).collect())
        .unwrap_or_default()
}
fn save_collection(name: &str, items: &[String]) {
    let _ = std::fs::write(collection_path(name), items.join("\n"));
}
fn collections() -> &'static Mutex<std::collections::HashMap<String, Vec<String>>> {
    COLLECTIONS.get_or_init(|| Mutex::new({
        let mut m = std::collections::HashMap::new();
NAMES_LOAD        m
    }))
}
fn form_value(body: &str) -> String {
    body.splitn(2, '=').nth(1).unwrap_or(body).replace('+', " ")
}
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}
fn write_redirect(stream: &mut TcpStream, location: &str) -> io::Result<()> {
    let resp = format!("HTTP/1.1 303 See Other\r\nLocation: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", location);
    stream.write_all(resp.as_bytes())
}
fn render_index_page() -> String {
    let g = collections().lock().unwrap();
    let mut b = String::from("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Admin</title><style>body{font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem;color:#16181d}h2{margin-top:1.5rem}ul{list-style:none;padding:0}li{padding:.45rem 0;border-bottom:1px solid #eee;display:flex;gap:.4rem;align-items:center}.row{flex:1;display:flex;gap:.4rem;margin:0}.del{background:#e5484d}form{display:flex;gap:.5rem;margin:.6rem 0}input{flex:1;padding:.45rem;border:1px solid #ccc;border-radius:6px}button{padding:.45rem 1rem;border:0;border-radius:6px;background:#2b57ff;color:#fff;cursor:pointer}.empty{color:#888}</style></head><body><h1>Admin</h1>");
    for name in RESOURCE_NAMES {
        b.push_str(&format!("<section><h2>{}</h2><ul>", name));
        match g.get(*name) {
            Some(items) if !items.is_empty() => {
                for (i, it) in items.iter().enumerate() {
                    b.push_str(&format!("<li><form method=\"post\" action=\"/{}/{}/edit\" class=\"row\"><input name=\"item\" value=\"{}\"><button>Save</button></form><form method=\"post\" action=\"/{}/{}/delete\"><button class=\"del\">Delete</button></form></li>", name, i, escape_html(it), name, i));
                }
            }
            _ => b.push_str("<li class=\"empty\">No items yet.</li>"),
        }
        b.push_str(&format!("</ul><form method=\"post\" action=\"/{}\"><input name=\"item\" placeholder=\"new {}\" required><button>Add</button></form></section>", name, name));
    }
    b.push_str("</body></html>");
    b
}
"#;

const RESOURCE_ARM_TEMPLATE: &str = r#"        ("GET", "/RES") => {
            let g = collections().lock().unwrap();
            let items = g.get("RES").cloned().unwrap_or_default();
            write_response(&mut stream, 200, &format!("[{}]", items.join(",")))
        }
        ("POST", "/RES") => {
            // A browser FORM submit (urlencoded) stores the field value and
            // redirects back to the admin page; an API client (json/other) gets
            // 201 and the record verbatim.
            let is_form = content_type.starts_with("application/x-www-form-urlencoded");
            let value = if is_form { form_value(&body) } else { body.replace('\n', " ") };
            {
                let mut g = collections().lock().unwrap();
                let v = g.entry("RES".to_string()).or_default();
                v.push(value);
                save_collection("RES", v);
            }
            if is_form {
                write_redirect(&mut stream, "/")
            } else {
                write_response(&mut stream, 201, "{\"ok\":true,\"created\":1}")
            }
        }
        ("GET", p) if p.starts_with("/RES/") => {
            let id: usize = p["/RES/".len()..].parse().unwrap_or(usize::MAX);
            let g = collections().lock().unwrap();
            match g.get("RES").and_then(|v| v.get(id)) {
                Some(item) => write_response(&mut stream, 200, item),
                None => write_response(&mut stream, 404, "{\"error\":\"not found\"}"),
            }
        }
        ("PUT", p) if p.starts_with("/RES/") => {
            let id: usize = p["/RES/".len()..].parse().unwrap_or(usize::MAX);
            let mut g = collections().lock().unwrap();
            match g.get_mut("RES") {
                Some(v) if id < v.len() => {
                    v[id] = body.replace('\n', " ");
                    save_collection("RES", v);
                    write_response(&mut stream, 200, "{\"ok\":true,\"updated\":1}")
                }
                _ => write_response(&mut stream, 404, "{\"error\":\"not found\"}"),
            }
        }
        ("DELETE", p) if p.starts_with("/RES/") => {
            let id: usize = p["/RES/".len()..].parse().unwrap_or(usize::MAX);
            let mut g = collections().lock().unwrap();
            match g.get_mut("RES") {
                Some(v) if id < v.len() => {
                    v.remove(id);
                    save_collection("RES", v);
                    write_response(&mut stream, 200, "{\"ok\":true,\"deleted\":1}")
                }
                _ => write_response(&mut stream, 404, "{\"error\":\"not found\"}"),
            }
        }
        ("POST", p) if p.starts_with("/RES/") && p.ends_with("/delete") => {
            let mid = &p["/RES/".len()..p.len() - "/delete".len()];
            let id: usize = mid.parse().unwrap_or(usize::MAX);
            {
                let mut g = collections().lock().unwrap();
                if let Some(v) = g.get_mut("RES") {
                    if id < v.len() {
                        v.remove(id);
                        save_collection("RES", v);
                    }
                }
            }
            write_redirect(&mut stream, "/")
        }
        ("POST", p) if p.starts_with("/RES/") && p.ends_with("/edit") => {
            let mid = &p["/RES/".len()..p.len() - "/edit".len()];
            let id: usize = mid.parse().unwrap_or(usize::MAX);
            let value = form_value(&body);
            {
                let mut g = collections().lock().unwrap();
                if let Some(v) = g.get_mut("RES") {
                    if id < v.len() {
                        v[id] = value;
                        save_collection("RES", v);
                    }
                }
            }
            write_redirect(&mut stream, "/")
        }
"#;

fn render_backend_app(app: &BackendApp) -> String {
    let store_layer = render_store_layer(app.store.kind);
    let store_init = render_store_init(app);
    let path_import = match app.store.kind {
        StoreKind::Memory => "",
        StoreKind::File | StoreKind::Sqlite => "use std::path::PathBuf;\n",
    };
    let rule_code = app
        .rules
        .iter()
        .map(|r| r.rule_code.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let evaluate_arms = render_evaluate_match_arms(&app.rules);
    let resource_state = render_resource_state(&app.resources);
    let resource_arms = render_resource_arms(&app.resources);
    let rules_json = escape_for_rust_string(&render_rules_list_literal(&app.rules));
    let description = escape_for_rust_string(&app.description);
    let rule_count = app.rules.len();

    format!(
        r###"// Auto-generated by nCPU LOOP-3D generated backend (BackendIR).
// Rules: {rule_count} synthesized handler(s)
// Description: {description}
// Store: {store_kind}
// Compile:
//   memory: rustc --edition=2021 generated_rule_backend.rs -o generated_rule_backend
//   file:   rustc --edition=2021 generated_rule_backend.rs -o generated_rule_backend
//   sqlite: rustc --edition=2021 generated_rule_backend.rs -o generated_rule_backend -l sqlite3

use std::io::{{self, BufRead, BufReader, Read, Write}};
use std::net::{{TcpListener, TcpStream}};
{path_import}use std::sync::{{Arc, Mutex}};
use std::thread;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Event {{
    rule: String,
    input: i64,
    output: i64,
}}

{store_layer}

{resource_state}
{rule_code}

fn main() {{
    let args: Vec<String> = std::env::args().collect();
    let port = arg_value(&args, "--port").unwrap_or_else(|| "7800".to_string());
    let host = arg_value(&args, "--host").unwrap_or_else(|| "127.0.0.1".to_string());
    // Optional static site root: when set, GET requests not matching an API
    // route are served as files from this directory (single-artifact stack).
    let static_dir = arg_value(&args, "--static");
    let bind = format!("{{}}:{{}}", host, port);
    let listener = TcpListener::bind(&bind).unwrap_or_else(|e| {{
        eprintln!("cannot bind {{bind}}: {{e}}");
        std::process::exit(1);
    }});
    let addr = listener.local_addr().expect("local addr");
    {store_init}
    println!("BACKEND_READY http://{{}}", addr);
    std::io::stdout().flush().ok();

    for stream in listener.incoming() {{
        match stream {{
            Ok(stream) => {{
                let store = Arc::clone(&store);
                let static_dir = static_dir.clone();
                thread::spawn(move || {{
                    let _ = handle_connection(stream, store, static_dir);
                }});
            }}
            Err(e) => eprintln!("accept error: {{e}}"),
        }}
    }}
}}

fn arg_value(args: &[String], flag: &str) -> Option<String> {{
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}}

fn handle_connection(
    mut stream: TcpStream,
    store: Arc<dyn EventStore>,
    static_dir: Option<String>,
) -> io::Result<()> {{
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {{
        return Ok(());
    }}
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut content_length = 0usize;
    let mut content_type = String::new();
    loop {{
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 || line == "\r\n" || line == "\n" {{
            break;
        }}
        if let Some((k, v)) = line.split_once(':') {{
            if k.trim().eq_ignore_ascii_case("content-length") {{
                content_length = v.trim().parse().unwrap_or(0);
            }} else if k.trim().eq_ignore_ascii_case("content-type") {{
                content_type = v.trim().to_string();
            }}
        }}
    }}
    let _ = &content_type;
    let mut body = vec![0u8; content_length];
    if content_length > 0 {{
        reader.read_exact(&mut body)?;
    }}
    let body = String::from_utf8_lossy(&body);

    match (method.as_str(), path.as_str()) {{
        ("GET", "/health") => write_response(
            &mut stream,
            200,
            &format!(
                "{{{{\"ok\":true,\"service\":\"{service_name}\",\"store\":\"{store_kind}\",\"rules\":{rule_count}}}}}"
            ),
        ),
        ("GET", "/rules") => write_response(&mut stream, 200, "{rules_json}"),
        ("GET", "/events") => {{
            let snapshot = store.list().unwrap_or_default();
            write_response(&mut stream, 200, &events_json(&snapshot))
        }}
        // SUBMISSION intake (site+backend integration): a POSTed form/event is
        // appended to the store verbatim under the "submission" rule tag —
        // the generated site's contact form has a REAL target.
        ("POST", "/events") => {{
            let event = Event {{
                rule: "submission".to_string(),
                input: body.len() as i64,
                output: 0,
            }};
            match store.append(event) {{
                Ok(()) => write_response(&mut stream, 201, "{{\"ok\":true}}"),
                Err(err) => write_response(
                    &mut stream,
                    500,
                    &format!("{{{{\"error\":\"store append failed: {{}}\"}}}}", err),
                ),
            }}
        }}
{evaluate_arms}
{resource_arms}
        // STATIC FALLBACK: unmatched GETs are served from the site root when
        // one was provided (`--static <dir>`). This is what lets one binary
        // serve the generated site AND its api. Fail-closed to 404.
        ("GET", _) => match static_dir.as_deref().and_then(|d| load_static(d, &path)) {{
            Some((bytes, ctype)) => write_bytes(&mut stream, 200, ctype, &bytes),
            None => write_response(&mut stream, 404, "{{\"error\":\"unknown route\"}}"),
        }},
        _ => write_response(&mut stream, 404, "{{\"error\":\"unknown route\"}}"),
    }}
}}

fn load_static(dir: &str, path: &str) -> Option<(Vec<u8>, &'static str)> {{
    let rel = path.split('?').next().unwrap_or("/");
    let rel = if rel == "/" {{ "index.html" }} else {{ rel.trim_start_matches('/') }};
    // Path-traversal guard: no parent refs, no absolute escape.
    if rel.is_empty() || rel.contains("..") {{
        return None;
    }}
    let full = std::path::Path::new(dir).join(rel);
    let bytes = std::fs::read(&full).ok()?;
    let ctype = match full.extension().and_then(|e| e.to_str()) {{
        Some("html") | Some("htm") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        _ => "application/octet-stream",
    }};
    Some((bytes, ctype))
}}

fn write_bytes(stream: &mut TcpStream, status: u16, ctype: &str, body: &[u8]) -> io::Result<()> {{
    let status_text = match status {{
        200 => "OK",
        404 => "Not Found",
        _ => "Server Error",
    }};
    let header = format!(
        "HTTP/1.1 {{}} {{}}\r\nContent-Type: {{}}\r\nContent-Length: {{}}\r\nConnection: close\r\n\r\n",
        status, status_text, ctype, body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}}

fn extract_i64_field(body: &str, field: &str) -> Option<i64> {{
    let key = format!("\"{{}}\"", field);
    let idx = body.find(&key)?;
    let after_key = &body[idx + key.len()..];
    let colon = after_key.find(':')?;
    let mut s = after_key[colon + 1..].trim_start();
    let mut end = 0usize;
    for (i, ch) in s.char_indices() {{
        if i == 0 && ch == '-' {{
            end = ch.len_utf8();
            continue;
        }}
        if ch.is_ascii_digit() {{
            end = i + ch.len_utf8();
        }} else {{
            break;
        }}
    }}
    if end == 0 {{
        return None;
    }}
    s = &s[..end];
    s.parse().ok()
}}

fn events_json(events: &[Event]) -> String {{
    let rows = events
        .iter()
        .map(|e| format!(
            "{{{{\"rule\":\"{{}}\",\"input\":{{}},\"output\":{{}}}}}}",
            e.rule, e.input, e.output
        ))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{{\"events\":[{{}}],\"count\":{{}}}}}}", rows, events.len())
}}

fn write_response(stream: &mut TcpStream, status: u16, body: &str) -> io::Result<()> {{
    let status_text = match status {{
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Server Error",
    }};
    let response = format!(
        "HTTP/1.1 {{}} {{}}\r\nContent-Type: application/json\r\nContent-Length: {{}}\r\nConnection: close\r\n\r\n{{}}",
        status,
        status_text,
        body.len(),
        body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}}
"###,
        description = description,
        rule_count = rule_count,
        store_kind = app.store.kind.cli_name(),
        path_import = path_import,
        store_layer = store_layer,
        store_init = store_init,
        rule_code = rule_code,
        service_name = app.service_name,
        rules_json = rules_json,
        evaluate_arms = evaluate_arms,
    )
}

fn render_evaluate_match_arms(rules: &[RuleModel]) -> String {
    rules
        .iter()
        .map(|rule| {
            format!(
                r#"        ("POST", "/rules/{name}/evaluate") => match extract_i64_field(&body, "input") {{
            Some(input) => {{
                let output = {name}(input);
                let event = Event {{
                    rule: "{name}".to_string(),
                    input,
                    output,
                }};
                if let Err(err) = store.append(event) {{
                    return write_response(
                        &mut stream,
                        500,
                        &format!("{{{{\"error\":\"store append failed: {{}}\"}}}}", err),
                    );
                }}
                write_response(
                    &mut stream,
                    200,
                    &format!(
                        "{{{{\"rule\":\"{name}\",\"input\":{{}},\"output\":{{}}}}}}",
                        input, output
                    ),
                )
            }}
            None => write_response(
                &mut stream,
                400,
                "{{\"error\":\"expected JSON body with integer field input\"}}",
            ),
        }},"#,
                name = rule.name
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_rules_list_literal(rules: &[RuleModel]) -> String {
    let entries = rules
        .iter()
        .map(|rule| {
            format!(
                "{{\"name\":\"{}\",\"method\":\"{}\"}}",
                escape_for_json_string(&rule.name),
                escape_for_json_string(&rule.synthesis_method)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"rules\":[{entries}]}}")
}

fn escape_for_json_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn render_store_init(app: &BackendApp) -> String {
    let default_path = app.store.default_path;
    match app.store.kind {
        StoreKind::Memory => "    let store: Arc<dyn EventStore> = Arc::new(MemoryStore::new());".to_string(),
        StoreKind::File => format!(
            r#"    let store_path = arg_value(&args, "--store-path")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("{default_path}"));
    let store: Arc<dyn EventStore> = Arc::new(
        FileStore::open(store_path).unwrap_or_else(|e| {{
            eprintln!("cannot open file store: {{e}}");
            std::process::exit(1);
        }}),
    );"#
        ),
        StoreKind::Sqlite => format!(
            r#"    let store_path = arg_value(&args, "--store-path")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("{default_path}"));
    let store: Arc<dyn EventStore> = Arc::new(
        SqliteStore::open(store_path).unwrap_or_else(|e| {{
            eprintln!("cannot open sqlite store: {{e}}");
            std::process::exit(1);
        }}),
    );"#
        ),
    }
}

fn render_store_layer(kind: StoreKind) -> &'static str {
    match kind {
        StoreKind::Memory => MEMORY_STORE_LAYER,
        StoreKind::File => FILE_STORE_LAYER,
        StoreKind::Sqlite => SQLITE_STORE_LAYER,
    }
}

const MEMORY_STORE_LAYER: &str = r###"trait EventStore: Send + Sync {
    fn append(&self, event: Event) -> io::Result<()>;
    fn list(&self) -> io::Result<Vec<Event>>;
}

struct MemoryStore {
    events: Mutex<Vec<Event>>,
}

impl MemoryStore {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }
}

impl EventStore for MemoryStore {
    fn append(&self, event: Event) -> io::Result<()> {
        self.events
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "store lock poisoned"))?
            .push(event);
        Ok(())
    }

    fn list(&self) -> io::Result<Vec<Event>> {
        Ok(self
            .events
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "store lock poisoned"))?
            .clone())
    }
}
"###;

const FILE_STORE_LAYER: &str = r###"trait EventStore: Send + Sync {
    fn append(&self, event: Event) -> io::Result<()>;
    fn list(&self) -> io::Result<Vec<Event>>;
}

struct FileStore {
    path: PathBuf,
    events: Mutex<Vec<Event>>,
}

impl FileStore {
    fn open(path: PathBuf) -> io::Result<Self> {
        let mut events = Vec::new();
        if path.exists() {
            let file = std::fs::File::open(&path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Some(event) = parse_event_line(&line) {
                    events.push(event);
                }
            }
        }
        Ok(Self {
            path,
            events: Mutex::new(events),
        })
    }

    fn persist_line(path: &PathBuf, event: &Event) -> io::Result<()> {
        use std::fs::OpenOptions;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(file, "{{\"rule\":\"{}\",\"input\":{},\"output\":{}}}", event.rule, event.input, event.output)?;
        file.sync_all()?;
        Ok(())
    }
}

impl EventStore for FileStore {
    fn append(&self, event: Event) -> io::Result<()> {
        Self::persist_line(&self.path, &event)?;
        self.events
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "store lock poisoned"))?
            .push(event);
        Ok(())
    }

    fn list(&self) -> io::Result<Vec<Event>> {
        Ok(self
            .events
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "store lock poisoned"))?
            .clone())
    }
}

fn parse_event_line(line: &str) -> Option<Event> {
    let rule = extract_json_string(line, "rule")?;
    let input = extract_json_i64(line, "input")?;
    let output = extract_json_i64(line, "output")?;
    Some(Event { rule, input, output })
}

fn extract_json_string(body: &str, field: &str) -> Option<String> {
    let key = format!("\"{}\"", field);
    let idx = body.find(&key)?;
    let after_key = &body[idx + key.len()..];
    let colon = after_key.find(':')?;
    let mut s = after_key[colon + 1..].trim_start();
    if !s.starts_with('"') {
        return None;
    }
    s = &s[1..];
    let end = s.find('"')?;
    Some(s[..end].to_string())
}

fn extract_json_i64(body: &str, field: &str) -> Option<i64> {
    let key = format!("\"{}\"", field);
    let idx = body.find(&key)?;
    let after_key = &body[idx + key.len()..];
    let colon = after_key.find(':')?;
    let mut s = after_key[colon + 1..].trim_start();
    let mut end = 0usize;
    for (i, ch) in s.char_indices() {
        if i == 0 && ch == '-' {
            end = ch.len_utf8();
            continue;
        }
        if ch.is_ascii_digit() {
            end = i + ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    s = &s[..end];
    s.parse().ok()
}
"###;

const SQLITE_STORE_LAYER: &str = r###"use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

trait EventStore: Send + Sync {
    fn append(&self, event: Event) -> io::Result<()>;
    fn list(&self) -> io::Result<Vec<Event>>;
}

type SqliteHandle = *mut c_void;

#[link(name = "sqlite3")]
extern "C" {
    fn sqlite3_open(filename: *const c_char, pp_db: *mut SqliteHandle) -> c_int;
    fn sqlite3_close(db: SqliteHandle) -> c_int;
    fn sqlite3_exec(
        db: SqliteHandle,
        sql: *const c_char,
        callback: *const c_void,
        arg: *mut c_void,
        errmsg: *mut *mut c_char,
    ) -> c_int;
}

struct SqliteStore {
    db: Mutex<SqliteHandle>,
    path: PathBuf,
}

impl SqliteStore {
    fn open(path: PathBuf) -> io::Result<Self> {
        let c_path = CString::new(path.to_string_lossy().as_ref())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid sqlite path"))?;
        let mut db: SqliteHandle = std::ptr::null_mut();
        let rc = unsafe { sqlite3_open(c_path.as_ptr(), &mut db) };
        if rc != 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("sqlite3_open failed with code {rc}"),
            ));
        }
        let store = Self {
            db: Mutex::new(db),
            path,
        };
        store.exec("CREATE TABLE IF NOT EXISTS events (rule TEXT NOT NULL, input INTEGER NOT NULL, output INTEGER NOT NULL)")?;
        Ok(store)
    }

    fn exec(&self, sql: &str) -> io::Result<()> {
        let c_sql = CString::new(sql)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid sql"))?;
        let db = self
            .db
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "store lock poisoned"))?;
        let rc = unsafe { sqlite3_exec(*db, c_sql.as_ptr(), std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut()) };
        if rc != 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("sqlite exec failed with code {rc}"),
            ));
        }
        Ok(())
    }
}

impl Drop for SqliteStore {
    fn drop(&mut self) {
        if let Ok(db) = self.db.lock() {
            if !db.is_null() {
                unsafe {
                    sqlite3_close(*db);
                }
            }
        }
    }
}

impl EventStore for SqliteStore {
    fn append(&self, event: Event) -> io::Result<()> {
        let sql = format!(
            "INSERT INTO events (rule, input, output) VALUES ('{}', {}, {})",
            event.rule.replace('\'', "''"),
            event.input,
            event.output
        );
        self.exec(&sql)
    }

    fn list(&self) -> io::Result<Vec<Event>> {
        let db = self
            .db
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "store lock poisoned"))?;
        let mut stmt: SqliteHandle = std::ptr::null_mut();
        let sql = CString::new("SELECT rule, input, output FROM events ORDER BY rowid ASC")
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid sql"))?;
        extern "C" {
            fn sqlite3_prepare_v2(
                db: SqliteHandle,
                sql: *const c_char,
                n_byte: c_int,
                pp_stmt: *mut SqliteHandle,
                pp_tail: *mut *const c_char,
            ) -> c_int;
            fn sqlite3_step(stmt: SqliteHandle) -> c_int;
            fn sqlite3_column_text(stmt: SqliteHandle, i_col: c_int) -> *const c_char;
            fn sqlite3_column_int64(stmt: SqliteHandle, i_col: c_int) -> i64;
            fn sqlite3_finalize(stmt: SqliteHandle) -> c_int;
        }
        fn sqlite_text(stmt: SqliteHandle, col: c_int) -> String {
            let ptr = unsafe { sqlite3_column_text(stmt, col) };
            if ptr.is_null() {
                return String::new();
            }
            unsafe {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        }
        const SQLITE_ROW: c_int = 100;
        const SQLITE_DONE: c_int = 101;
        let rc = unsafe {
            sqlite3_prepare_v2(*db, sql.as_ptr(), -1, &mut stmt, std::ptr::null_mut())
        };
        if rc != 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("sqlite prepare failed with code {rc}"),
            ));
        }
        let mut events = Vec::new();
        loop {
            let step = unsafe { sqlite3_step(stmt) };
            if step == SQLITE_ROW {
                let rule = sqlite_text(stmt, 0);
                let input = unsafe { sqlite3_column_int64(stmt, 1) };
                let output = unsafe { sqlite3_column_int64(stmt, 2) };
                events.push(Event { rule, input, output });
            } else if step == SQLITE_DONE {
                break;
            } else {
                unsafe {
                    sqlite3_finalize(stmt);
                }
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("sqlite step failed with code {step}"),
                ));
            }
        }
        unsafe {
            sqlite3_finalize(stmt);
        }
        Ok(events)
    }
}
"###;

fn escape_for_rust_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_ir_renders_multi_rule_routes() {
        let app = BackendApp::from_rules(
            "bonus and penalty rules",
            vec![
                RuleModel {
                    name: "score_bonus".to_string(),
                    synthesis_method: "search_polynomial_multi".to_string(),
                    rule_code: "fn score_bonus(x: i64) -> i64 { 10 * x + 5 }".to_string(),
                },
                RuleModel {
                    name: "damage_penalty".to_string(),
                    synthesis_method: "search_polynomial_multi".to_string(),
                    rule_code: "fn damage_penalty(x: i64) -> i64 { 2 * x - 3 }".to_string(),
                },
            ],
            StoreKind::File,
        );
        let source = app.render_rust();
        assert!(source.contains("/rules/score_bonus/evaluate"));
        assert!(source.contains("/rules/damage_penalty/evaluate"));
        assert!(source.contains("\\\"name\\\":\\\"score_bonus\\\""));
        assert!(source.contains("\\\"name\\\":\\\"damage_penalty\\\""));
        assert!(source.contains("rule: String"));
        assert!(!source.contains(concat!("to", "do!")));
    }

    #[test]
    fn backend_ir_renders_resource_routes() {
        let app = BackendApp::from_rules("users api", vec![], StoreKind::Memory)
            .with_resources(vec!["users".to_string()]);
        let src = app.render_rust();
        assert!(src.contains("(\"GET\", \"/users\")"), "GET /users (list) arm present");
        assert!(src.contains("(\"POST\", \"/users\")"), "POST /users (create) arm present");
        assert!(src.contains("p.starts_with(\"/users/\")"), "item-level /users/ id arms present");
        assert!(src.contains("(\"PUT\", p)"), "PUT (update) arm present");
        assert!(src.contains("(\"DELETE\", p)"), "DELETE arm present");
        assert!(src.contains("fn collections()"), "collection state present");
        assert!(!src.contains("INSERTS"), "state template fully substituted");
        // No resources requested -> no state, existing backends unchanged.
        let plain = BackendApp::from_rules("x", vec![], StoreKind::Memory).render_rust();
        assert!(!plain.contains("fn collections()"), "no resource state when none asked");
    }

    #[test]
    fn backend_ir_resource_crud_round_trips_over_http() {
        let rustc_ok = std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !rustc_ok {
            eprintln!("skipping resource CRUD HTTP test: rustc unavailable");
            return;
        }
        let app = BackendApp::from_rules("users api", vec![], StoreKind::Memory)
            .with_resources(vec!["users".to_string()]);
        let (src_path, bin) =
            crate::backend_http::compile_to_temp_bin(&app.render_rust(), false).expect("compile");
        let res = crate::backend_http::verify_resource_crud(&bin, "users", 6);
        crate::backend_http::cleanup_temp_artifacts(&src_path, &bin);
        res.expect("POST then GET /users round-trips over HTTP");
    }

    #[test]
    fn backend_ir_resource_serves_admin_ui() {
        let rustc_ok = std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !rustc_ok {
            eprintln!("skipping admin UI test: rustc unavailable");
            return;
        }
        let app = BackendApp::from_rules("users api", vec![], StoreKind::Memory)
            .with_resources(vec!["users".to_string()]);
        let (src_path, bin) =
            crate::backend_http::compile_to_temp_bin(&app.render_rust(), false).expect("compile");
        let res = crate::backend_http::verify_resource_ui(&bin, "users");
        crate::backend_http::cleanup_temp_artifacts(&src_path, &bin);
        res.expect("server-rendered admin UI: GET / -> form submit -> item shown");
    }

    #[test]
    fn backend_ir_resource_persists_across_restart() {
        let rustc_ok = std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !rustc_ok {
            eprintln!("skipping resource persistence test: rustc unavailable");
            return;
        }
        let app = BackendApp::from_rules("users api", vec![], StoreKind::Memory)
            .with_resources(vec!["users".to_string()]);
        let (src_path, bin) =
            crate::backend_http::compile_to_temp_bin(&app.render_rust(), false).expect("compile");
        let res = crate::backend_http::verify_resource_persists(&bin, "users");
        crate::backend_http::cleanup_temp_artifacts(&src_path, &bin);
        res.expect("resource survives a process restart (file-backed)");
    }

    #[test]
    fn store_kind_parse_accepts_aliases() {
        assert_eq!(StoreKind::parse("memory"), Some(StoreKind::Memory));
        assert_eq!(StoreKind::parse("jsonl"), Some(StoreKind::File));
        assert_eq!(StoreKind::parse("sqlite"), Some(StoreKind::Sqlite));
        assert_eq!(StoreKind::parse("nosuch"), None);
    }
}



