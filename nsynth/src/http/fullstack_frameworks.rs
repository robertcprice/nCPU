//! Full-Stack Framework Support for nCPU/nSynth
//!
//! Complete Next.js (App Router) and Remix framework support with:
//! - Next.js App Router (app/ directory)
//! - Server Components and Client Components
//! - Data fetching patterns (Server Actions, fetch)
//! - Remix routes, loaders, and actions
//! - Metadata generation
//! - Route configuration

use serde_json::Value;

/// Component type for Next.js App Router
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentType {
    /// Server Component (default)
    Server,
    /// Client Component ("use client")
    Client,
}

/// Next.js App Router structure
#[derive(Debug, Clone)]
pub struct NextAppRouter {
    /// Route path (e.g., "app/dashboard/page.tsx")
    pub path: String,
    /// Layout components
    pub layouts: Vec<NextLayout>,
    /// Pages
    pub pages: Vec<NextPage>,
    /// Server actions
    pub actions: Vec<ServerAction>,
    /// API routes
    pub api_routes: Vec<ApiRoute>,
}

/// Next.js Layout component
#[derive(Debug, Clone)]
pub struct NextLayout {
    /// Layout path
    pub path: String,
    /// Component type
    pub component_type: ComponentType,
    /// Children rendering
    pub children: bool,
    /// Props interface
    pub props: Vec<Prop>,
    /// Metadata
    pub metadata: Option<Metadata>,
    /// Body content
    pub body: String,
}

impl NextLayout {
    /// Create new layout
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            component_type: ComponentType::Server,
            children: true,
            props: Vec::new(),
            metadata: None,
            body: String::new(),
        }
    }

    /// Set component type
    pub fn with_component_type(mut self, ct: ComponentType) -> Self {
        self.component_type = ct;
        self
    }

    /// Add prop
    pub fn prop(mut self, name: impl Into<String>, prop_type: PropType) -> Self {
        self.props.push(Prop {
            name: name.into(),
            prop_type,
            optional: false,
            default: None,
        });
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set body content
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// Generate Next.js layout code
    pub fn to_nextjs(&self) -> String {
        let mut code = String::new();

        // Client directive
        if self.component_type == ComponentType::Client {
            code.push_str("'use client';\n\n");
        }

        // Imports
        code.push_str("import React from 'react';\n");

        // Metadata export
        if let Some(ref metadata) = self.metadata {
            code.push_str(&format!(
                "export const metadata = {};\n\n",
                serde_json::to_string(metadata).unwrap_or_default()
            ));
        }

        // Props interface
        if !self.props.is_empty() {
            code.push_str("interface LayoutProps {\n");
            for prop in &self.props {
                code.push_str(&format!(
                    "  {}{}: {};\n",
                    prop.name,
                    if prop.optional { "?" } else { "" },
                    prop.prop_type.as_ts()
                ));
            }
            code.push_str("  children: React.ReactNode;\n");
            code.push_str("}\n\n");
        }

        // Layout component
        code.push_str(&format!(
            "export default function Layout({}: {} {{\n",
            if self.props.is_empty() {
                "{ children }"
            } else {
                "{ children, ...props }"
            },
            if self.props.is_empty() {
                "React.PropsWithChildren"
            } else {
                "LayoutProps"
            }
        ));

        code.push_str(&self.body);
        code.push_str("\n}\n");

        code
    }
}

/// Next.js Page component (App Router)
#[derive(Debug, Clone)]
pub struct NextPage {
    /// Page path
    pub path: String,
    /// Component type
    pub component_type: ComponentType,
    /// Dynamic params (e.g., ["id"] for [id])
    pub params: Vec<String>,
    /// Props interface
    pub props: Vec<Prop>,
    /// Metadata
    pub metadata: Option<Metadata>,
    /// Data fetching
    pub data_fetching: Option<NextDataFetching>,
    /// Body content
    pub body: String,
}

/// Next.js data fetching strategy
#[derive(Debug, Clone)]
pub enum NextDataFetching {
    /// Direct async function (App Router default)
    Async { body: String },
    /// Server Action call
    ServerAction { action: String, args: Vec<String> },
    /// Direct fetch call
    Fetch { url: String, options: String },
    /// Static generation with revalidation
    Static { revalidate: Option<u64> },
    /// Dynamic rendering (no caching)
    Dynamic,
}

impl NextPage {
    /// Create new page
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            component_type: ComponentType::Server,
            params: Vec::new(),
            props: Vec::new(),
            metadata: None,
            data_fetching: None,
            body: String::new(),
        }
    }

    /// Set component type
    pub fn with_component_type(mut self, ct: ComponentType) -> Self {
        self.component_type = ct;
        self
    }

    /// Add dynamic param
    pub fn param(mut self, name: impl Into<String>) -> Self {
        self.params.push(name.into());
        self
    }

    /// Add prop
    pub fn prop(mut self, name: impl Into<String>, prop_type: PropType) -> Self {
        self.props.push(Prop {
            name: name.into(),
            prop_type,
            optional: false,
            default: None,
        });
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set data fetching
    pub fn with_data_fetching(mut self, fetch: NextDataFetching) -> Self {
        self.data_fetching = Some(fetch);
        self
    }

    /// Set body content
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// Generate Next.js page code
    pub fn to_nextjs(&self) -> String {
        let mut code = String::new();

        // Client directive
        if self.component_type == ComponentType::Client {
            code.push_str("'use client';\n\n");
        }

        // Imports
        code.push_str("import React from 'react';\n");

        // Metadata export
        if let Some(ref metadata) = self.metadata {
            code.push_str(&format!(
                "export const metadata = {};\n\n",
                serde_json::to_string(metadata).unwrap_or_default()
            ));
        }

        // Props interface
        let has_params = !self.params.is_empty();
        if has_params || !self.props.is_empty() {
            code.push_str("interface PageProps {\n");
            for param in &self.params {
                code.push_str(&format!("  params: {{ {}?: string }};\n", param));
            }
            for prop in &self.props {
                code.push_str(&format!(
                    "  {}{}: {};\n",
                    prop.name,
                    if prop.optional { "?" } else { "" },
                    prop.prop_type.as_ts()
                ));
            }
            code.push_str("}\n\n");
        }

        // Page component
        let props_param = if has_params {
            "{ params }".to_string()
        } else if !self.props.is_empty() {
            "props".to_string()
        } else {
            String::new()
        };

        let props_type = if has_params || !self.props.is_empty() {
            "PageProps"
        } else {
            ""
        };

        code.push_str(&format!(
            "export default async function Page({}: {} {{\n",
            props_param, props_type
        ));

        // Data fetching
        if let Some(ref df) = self.data_fetching {
            match df {
                NextDataFetching::Async { body } => {
                    code.push_str(&format!("  // Async data fetching\n  {}\n", body));
                }
                NextDataFetching::ServerAction { action, args } => {
                    code.push_str(&format!(
                        "  const data = await {}({});\n",
                        action,
                        args.join(", ")
                    ));
                }
                NextDataFetching::Fetch { url, options } => {
                    code.push_str(&format!(
                        "  const res = await fetch('{}', {});\n",
                        url, options
                    ));
                    code.push_str("  const data = await res.json();\n");
                }
                NextDataFetching::Static { revalidate } => {
                    if let Some(reval) = revalidate {
                        code.push_str(&format!("export const revalidate = {};\n\n", reval));
                    }
                }
                NextDataFetching::Dynamic => {
                    code.push_str("export const dynamic = 'force-dynamic';\n\n");
                }
            }
        }

        code.push_str(&self.body);
        code.push_str("\n}\n");

        code
    }
}

/// Server Action for Next.js App Router
#[derive(Debug, Clone)]
pub struct ServerAction {
    /// Action name
    pub name: String,
    /// Input type
    pub input_type: Option<String>,
    /// Return type
    pub return_type: Option<String>,
    /// Action body
    pub body: String,
    /// Whether to use 'use server'
    pub use_server: bool,
}

impl ServerAction {
    /// Create new server action
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input_type: None,
            return_type: None,
            body: String::new(),
            use_server: true,
        }
    }

    /// Set input type
    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.input_type = Some(input.into());
        self
    }

    /// Set return type
    pub fn with_return(mut self, ret: impl Into<String>) -> Self {
        self.return_type = Some(ret.into());
        self
    }

    /// Set body
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// Generate Next.js server action code
    pub fn to_nextjs(&self) -> String {
        let mut code = String::new();

        if self.use_server {
            code.push_str("'use server';\n\n");
        }

        let input_param = self
            .input_type
            .as_ref()
            .map(|t| format!("input: {}", t))
            .unwrap_or_else(|| "input".to_string());

        let return_annotation = self
            .return_type
            .as_ref()
            .map(|t| format!(": {}", t))
            .unwrap_or_default();

        code.push_str(&format!(
            "export async function {}({}){} {{\n  {}\n}}\n",
            self.name, input_param, return_annotation, self.body
        ));

        code
    }
}

/// Remix route structure
#[derive(Debug, Clone)]
pub struct RemixRoute {
    /// Route path (e.g., "routes/users.$id.tsx")
    pub path: String,
    /// Loader function
    pub loader: Option<RemixLoader>,
    /// Action function
    pub action: Option<RemixAction>,
    /// Component type (default or "use client")
    pub component_type: ComponentType,
    /// Metadata
    pub metadata: Option<Metadata>,
    /// Body content
    pub body: String,
    /// Error boundary
    pub error_boundary: Option<String>,
    /// Default export (component)
    pub default_export: bool,
}

/// Remix loader function
#[derive(Debug, Clone)]
pub struct RemixLoader {
    /// Args type
    pub args_type: Option<String>,
    /// Return type
    pub return_type: Option<String>,
    /// Body
    pub body: String,
}

impl RemixLoader {
    /// Create new loader
    pub fn new() -> Self {
        Self {
            args_type: None,
            return_type: None,
            body: String::new(),
        }
    }

    /// Set args type
    pub fn with_args(mut self, args: impl Into<String>) -> Self {
        self.args_type = Some(args.into());
        self
    }

    /// Set return type
    pub fn with_return(mut self, ret: impl Into<String>) -> Self {
        self.return_type = Some(ret.into());
        self
    }

    /// Set body
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// Generate Remix loader code
    pub fn to_remix(&self) -> String {
        let mut code = String::new();

        let args_param = self
            .args_type
            .as_ref()
            .map(|t| format!("args: {}", t))
            .unwrap_or_else(|| "args".to_string());

        let return_annotation = self
            .return_type
            .as_ref()
            .map(|t| format!(": {}", t))
            .unwrap_or_default();

        code.push_str(&format!(
            "export async function loader({}){} {{\n  {}\n}}\n\n",
            args_param, return_annotation, self.body
        ));

        code
    }
}

/// Remix action function
#[derive(Debug, Clone)]
pub struct RemixAction {
    /// Args type
    pub args_type: Option<String>,
    /// Return type
    pub return_type: Option<String>,
    /// Body
    pub body: String,
}

impl RemixAction {
    /// Create new action
    pub fn new() -> Self {
        Self {
            args_type: None,
            return_type: None,
            body: String::new(),
        }
    }

    /// Set args type
    pub fn with_args(mut self, args: impl Into<String>) -> Self {
        self.args_type = Some(args.into());
        self
    }

    /// Set return type
    pub fn with_return(mut self, ret: impl Into<String>) -> Self {
        self.return_type = Some(ret.into());
        self
    }

    /// Set body
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// Generate Remix action code
    pub fn to_remix(&self) -> String {
        let mut code = String::new();

        let args_param = self
            .args_type
            .as_ref()
            .map(|t| format!("args: {}", t))
            .unwrap_or_else(|| "args".to_string());

        let return_annotation = self
            .return_type
            .as_ref()
            .map(|t| format!(": {}", t))
            .unwrap_or_default();

        code.push_str(&format!(
            "export async function action({}){} {{\n  {}\n}}\n\n",
            args_param, return_annotation, self.body
        ));

        code
    }
}

impl RemixRoute {
    /// Create new Remix route
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            loader: None,
            action: None,
            component_type: ComponentType::Server,
            metadata: None,
            body: String::new(),
            error_boundary: None,
            default_export: true,
        }
    }

    /// Set loader
    pub fn with_loader(mut self, loader: RemixLoader) -> Self {
        self.loader = Some(loader);
        self
    }

    /// Set action
    pub fn with_action(mut self, action: RemixAction) -> Self {
        self.action = Some(action);
        self
    }

    /// Set component type
    pub fn with_component_type(mut self, ct: ComponentType) -> Self {
        self.component_type = ct;
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set body
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// Set error boundary
    pub fn with_error_boundary(mut self, error: impl Into<String>) -> Self {
        self.error_boundary = Some(error.into());
        self
    }

    /// Generate Remix route code
    pub fn to_remix(&self) -> String {
        let mut code = String::new();

        // Client directive
        if self.component_type == ComponentType::Client {
            code.push_str("'use client';\n\n");
        }

        // Imports
        code.push_str(
            "import type { LoaderFunctionArgs, ActionFunctionArgs } from '@remix-run/node';\n",
        );
        code.push_str("import { useLoaderData, useActionData } from '@remix-run/react';\n\n");

        // Metadata export (Remix v2)
        if let Some(ref metadata) = self.metadata {
            code.push_str(&format!(
                "export const meta = () => {};\n\n",
                serde_json::to_string(metadata).unwrap_or_default()
            ));
        }

        // Loader
        if let Some(ref loader) = self.loader {
            code.push_str(&loader.to_remix());
        }

        // Action
        if let Some(ref action) = self.action {
            code.push_str(&action.to_remix());
        }

        // Error boundary
        if let Some(ref error) = self.error_boundary {
            code.push_str(&format!(
                "export function ErrorBoundary() {{\n  {}\n}}\n\n",
                error
            ));
        }

        // Default component
        if self.default_export {
            code.push_str("export default function Route() {\n");
            if self.loader.is_some() {
                code.push_str("  const data = useLoaderData();\n");
            }
            if self.action.is_some() {
                code.push_str("  const actionData = useActionData();\n");
            }
            code.push_str(&self.body);
            code.push_str("\n}\n");
        }

        code
    }
}

/// Prop type (TypeScript-compatible)
#[derive(Debug, Clone)]
pub struct PropType(String);

impl PropType {
    pub fn as_ts(&self) -> &str {
        &self.0
    }

    pub fn string() -> Self {
        Self("string".to_string())
    }
    pub fn number() -> Self {
        Self("number".to_string())
    }
    pub fn boolean() -> Self {
        Self("boolean".to_string())
    }
    pub fn any() -> Self {
        Self("any".to_string())
    }
    pub fn unknown() -> Self {
        Self("unknown".to_string())
    }
    pub fn void() -> Self {
        Self("void".to_string())
    }
    pub fn array(inner: PropType) -> Self {
        Self(format!("{}[]", inner.0))
    }
    pub fn record(value: PropType) -> Self {
        Self(format!("Record<string, {}>", value.0))
    }
    pub fn promise(inner: PropType) -> Self {
        Self(format!("Promise<{}>", inner.0))
    }
    pub fn optional(inner: PropType) -> Self {
        Self(format!("{} | undefined", inner.0))
    }
    pub fn custom(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// Prop definition
#[derive(Debug, Clone)]
pub struct Prop {
    pub name: String,
    pub prop_type: PropType,
    pub optional: bool,
    pub default: Option<Value>,
}

/// Metadata for Next.js/Remix
#[derive(Debug, Clone, serde::Serialize)]
pub struct Metadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Option<Vec<String>>,
    pub open_graph: Option<OpenGraphMetadata>,
    pub twitter: Option<TwitterMetadata>,
    pub viewport: Option<String>,
    pub robots: Option<String>,
    pub canonical: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct OpenGraphMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub image: Option<String>,
    pub url: Option<String>,
    pub type_: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TwitterMetadata {
    pub card: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub image: Option<String>,
}

impl Metadata {
    /// Create basic metadata
    pub fn new() -> Self {
        Self {
            title: None,
            description: None,
            keywords: None,
            open_graph: None,
            twitter: None,
            viewport: None,
            robots: None,
            canonical: None,
        }
    }

    /// Set title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set keywords
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = Some(keywords);
        self
    }

    /// Set open graph
    pub fn with_open_graph(mut self, og: OpenGraphMetadata) -> Self {
        self.open_graph = Some(og);
        self
    }

    /// Set twitter
    pub fn with_twitter(mut self, tw: TwitterMetadata) -> Self {
        self.twitter = Some(tw);
        self
    }

    /// Set viewport
    pub fn with_viewport(mut self, viewport: impl Into<String>) -> Self {
        self.viewport = Some(viewport.into());
        self
    }

    /// Set canonical URL
    pub fn with_canonical(mut self, url: impl Into<String>) -> Self {
        self.canonical = Some(url.into());
        self
    }
}

/// API route for both frameworks
#[derive(Debug, Clone)]
pub struct ApiRoute {
    /// Route path
    pub path: String,
    /// HTTP methods
    pub methods: Vec<HttpMethod>,
    /// Handler body
    pub handler: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
}

impl HttpMethod {
    pub fn as_str(&self) -> &str {
        match self {
            HttpMethod::GET => "GET",
            HttpMethod::POST => "POST",
            HttpMethod::PUT => "PUT",
            HttpMethod::DELETE => "DELETE",
            HttpMethod::PATCH => "PATCH",
        }
    }
}

impl ApiRoute {
    /// Create new API route
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            methods: vec![HttpMethod::GET],
            handler: String::new(),
        }
    }

    /// Add method
    pub fn method(mut self, method: HttpMethod) -> Self {
        self.methods.push(method);
        self
    }

    /// Set handler
    pub fn with_handler(mut self, handler: impl Into<String>) -> Self {
        self.handler = handler.into();
        self
    }

    /// Generate Next.js API route (App Router)
    pub fn to_nextjs(&self) -> String {
        format!(
            "import {{ NextResponse }} from 'next/server';\n\n\
            export async function {}() {{\n  {}\n}}\n",
            if self.methods.len() == 1 && self.methods[0] == HttpMethod::GET {
                "GET"
            } else {
                "runtime"
            },
            self.handler
        )
    }

    /// Generate Remix resource route
    pub fn to_remix(&self) -> String {
        format!(
            "import {{ json }} from '@remix-run/node';\n\n\
            export async function {}() {{\n  {}\n}}\n",
            if self.methods.len() == 1 && self.methods[0] == HttpMethod::GET {
                "loader"
            } else {
                "action"
            },
            self.handler
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_layout_basic() {
        let layout = NextLayout::new("app/layout.tsx").with_body("return <div>{children}</div>;");

        let code = layout.to_nextjs();
        assert!(code.contains("export default function Layout"));
        // When there are no props, uses React.PropsWithChildren instead of custom interface
        assert!(
            code.contains("children")
                && (code.contains("React.ReactNode") || code.contains("React.PropsWithChildren"))
        );
        assert!(code.contains("<div>{children}</div>"));
    }

    #[test]
    fn test_next_layout_client() {
        let layout = NextLayout::new("app/layout.tsx")
            .with_component_type(ComponentType::Client)
            .with_body("return <div>{children}</div>;");

        let code = layout.to_nextjs();
        assert!(code.contains("'use client'"));
    }

    #[test]
    fn test_next_layout_with_metadata() {
        let metadata = Metadata::new()
            .with_title("My App")
            .with_description("A great application");

        let layout = NextLayout::new("app/layout.tsx")
            .with_metadata(metadata)
            .with_body("return <div>{children}</div>;");

        let code = layout.to_nextjs();
        assert!(code.contains("export const metadata"));
        assert!(code.contains("My App"));
    }

    #[test]
    fn test_next_page_basic() {
        let page = NextPage::new("app/page.tsx").with_body("return <h1>Hello</h1>;");

        let code = page.to_nextjs();
        assert!(code.contains("export default async function Page"));
        assert!(code.contains("<h1>Hello</h1>"));
    }

    #[test]
    fn test_next_page_with_params() {
        let page = NextPage::new("app/users/[id]/page.tsx")
            .param("id")
            .with_body("return <div>User {params.id}</div>;");

        let code = page.to_nextjs();
        assert!(code.contains("params: { id?: string }"));
        assert!(code.contains("{ params }"));
    }

    #[test]
    fn test_next_page_with_data_fetching() {
        let page = NextPage::new("app/page.tsx")
            .with_data_fetching(NextDataFetching::Fetch {
                url: "https://api.example.com/data".to_string(),
                options: "{ cache: 'no-store' }".to_string(),
            })
            .with_body("return <div>{data.title}</div>;");

        let code = page.to_nextjs();
        assert!(code.contains("await fetch"));
        assert!(code.contains("https://api.example.com/data"));
    }

    #[test]
    fn test_next_page_with_revalidation() {
        let page = NextPage::new("app/page.tsx")
            .with_data_fetching(NextDataFetching::Static {
                revalidate: Some(3600),
            })
            .with_body("return <h1>Hello</h1>;");

        let code = page.to_nextjs();
        assert!(code.contains("export const revalidate = 3600"));
    }

    #[test]
    fn test_server_action() {
        let action = ServerAction::new("submitForm")
            .with_input("{ name: string; email: string }")
            .with_return("{ success: boolean }")
            .with_body("console.log(input); return { success: true };");

        let code = action.to_nextjs();
        assert!(code.contains("'use server'"));
        assert!(code.contains("export async function submitForm"));
        assert!(code.contains("input: { name: string; email: string }"));
        assert!(code.contains(": { success: boolean }"));
    }

    #[test]
    fn test_remix_route_basic() {
        let route = RemixRoute::new("routes/users.tsx").with_body("return <div>Users</div>;");

        let code = route.to_remix();
        assert!(code.contains("export default function Route"));
        assert!(code.contains("<div>Users</div>"));
    }

    #[test]
    fn test_remix_route_with_loader() {
        let loader = RemixLoader::new()
            .with_args("LoaderFunctionArgs")
            .with_return("{ users: Array<{ id: string; name: string }> }")
            .with_body("const users = await db.users.findMany(); return json({ users });");

        let route = RemixRoute::new("routes/users.tsx")
            .with_loader(loader)
            .with_body("return <div>{data.users.length} users</div>;");

        let code = route.to_remix();
        assert!(code.contains("export async function loader"));
        assert!(code.contains("LoaderFunctionArgs"));
        assert!(code.contains("useLoaderData"));
    }

    #[test]
    fn test_remix_route_with_action() {
        let action = RemixAction::new()
            .with_args("ActionFunctionArgs")
            .with_body(
                "const formData = await args.request.formData(); return json({ success: true });",
            );

        let route = RemixRoute::new("routes/users.tsx")
            .with_action(action)
            .with_body("return <Form method='post'>...</Form>;");

        let code = route.to_remix();
        assert!(code.contains("export async function action"));
        assert!(code.contains("useActionData"));
    }

    #[test]
    fn test_remix_route_with_error_boundary() {
        let route = RemixRoute::new("routes/users.tsx")
            .with_error_boundary("return <div>Error loading users</div>;")
            .with_body("return <div>Users</div>;");

        let code = route.to_remix();
        assert!(code.contains("export function ErrorBoundary"));
        assert!(code.contains("Error loading users"));
    }

    #[test]
    fn test_metadata() {
        let metadata = Metadata::new()
            .with_title("Test Page")
            .with_description("Test description")
            .with_keywords(vec!["test".to_string(), "example".to_string()]);

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("Test Page"));
        assert!(json.contains("Test description"));
        assert!(json.contains("test"));
        assert!(json.contains("example"));
    }

    #[test]
    fn test_open_graph_metadata() {
        let og = OpenGraphMetadata {
            title: Some("OG Title".to_string()),
            description: Some("OG Description".to_string()),
            image: Some("https://example.com/image.png".to_string()),
            url: Some("https://example.com".to_string()),
            type_: Some("website".to_string()),
        };

        let metadata = Metadata::new().with_title("Page Title").with_open_graph(og);

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("OG Title"));
        assert!(json.contains("https://example.com/image.png"));
    }

    #[test]
    fn test_api_route() {
        let api = ApiRoute::new("app/api/users/route.ts")
            .method(HttpMethod::GET)
            .method(HttpMethod::POST)
            .with_handler("return NextResponse.json({ users: [] });");

        let nextjs_code = api.to_nextjs();
        assert!(nextjs_code.contains("NextResponse"));

        let remix_code = api.to_remix();
        assert!(remix_code.contains("@remix-run/node"));
    }

    #[test]
    fn test_prop_type() {
        assert_eq!(PropType::string().as_ts(), "string");
        assert_eq!(PropType::number().as_ts(), "number");
        assert_eq!(PropType::array(PropType::string()).as_ts(), "string[]");
        assert_eq!(PropType::promise(PropType::any()).as_ts(), "Promise<any>");
    }

    #[test]
    fn test_next_page_client_component() {
        let page = NextPage::new("app/counter/page.tsx")
            .with_component_type(ComponentType::Client)
            .with_body("const [count, setCount] = useState(0); return <button onClick={() => setCount(c => c + 1)}>{count}</button>;");

        let code = page.to_nextjs();
        assert!(code.contains("'use client'"));
    }
}
