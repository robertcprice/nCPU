//! CLI: universal coding agent — any NL query, full secure tools, session resume.

use mog_synth::agent::{
    AgentQueryResult, CodingAgentSession, GuardrailPolicy, QueryRoute, ToolCall,
};
use std::env;
use std::path::PathBuf;
use std::process;

fn usage() {
    eprintln!(
        "usage: coding_agent --root <repo_path> [options] [query]\n\
\n\
Commands (default: query):\n\
  query <text>              Run NL query through registry workflow router\n\
  --tools                   List allowed secure tool capabilities\n\
  --capabilities            Emit runtime engine capability introspection (JSON)\n\
  --tool <name> <action>    Direct tool invoke (e.g. --tool fs read path=src/lib.rs)\n\
  --clarify <answer>        Answer pending clarification for --session\n\
\n\
Options:\n\
  --root <path>             Repository / sandbox root (required)\n\
  --session <id>            Session name for persist/resume (default: main)\n\
  --json                    Emit JSON result\n\
  --allow-http <host>       Add HTTP host allowlist entry\n\
\n\
Examples:\n\
  coding_agent --root . query \"add two numbers\"\n\
  coding_agent --root . --tools\n\
  coding_agent --root . --tool fs list path=src\n\
  coding_agent --root . --session dev query \"fix failing tests\"\n\
  coding_agent --root . --session dev --clarify \"multiply two integers\""
    );
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
        process::exit(2);
    }

    let root = arg_value(&args, "--root").map(PathBuf::from);
    let session_id = arg_value(&args, "--session").unwrap_or_else(|| "main".to_string());
    let json_out = args.iter().any(|a| a == "--json");
    let list_tools = args.iter().any(|a| a == "--tools");
    let capabilities = args.iter().any(|a| a == "--capabilities");
    let clarify = arg_value(&args, "--clarify");

    let Some(root) = root else {
        usage();
        process::exit(2);
    };
    if !root.is_dir() {
        eprintln!("error: --root is not a directory: {}", root.display());
        process::exit(2);
    }

    let mut policy = GuardrailPolicy::default();
    for (i, arg) in args.iter().enumerate() {
        if arg == "--allow-http" {
            if let Some(host) = args.get(i + 1) {
                policy = policy.with_allowed_http_host(host);
            }
        }
    }

    let mut session =
        CodingAgentSession::load(&root, policy.clone(), &session_id).unwrap_or_else(|e| {
            eprintln!("error: load session: {e}");
            process::exit(2);
        });

    if list_tools {
        let caps = session.allowed_tool_capabilities();
        if json_out {
            println!(
                "{}",
                serde_json::to_string_pretty(&caps).unwrap_or_default()
            );
        } else {
            for cap in caps {
                println!("{cap}");
            }
        }
        return;
    }

    if capabilities {
        let doc = mog_synth::agent_introspect::engine_capabilities_json(&root, policy);
        if json_out {
            println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
        } else {
            println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
        }
        return;
    }

    if let Some(tool_idx) = args.iter().position(|a| a == "--tool") {
        let tool = args.get(tool_idx + 1).cloned().unwrap_or_default();
        let action = args.get(tool_idx + 2).cloned().unwrap_or_default();
        if tool.is_empty() || action.is_empty() {
            eprintln!("error: --tool requires <name> <action> [key=value ...]");
            process::exit(2);
        }
        let mut call = ToolCall::new(action);
        for arg in args.iter().skip(tool_idx + 3) {
            if let Some((k, v)) = arg.split_once('=') {
                call = call.arg(k, v);
            }
        }
        match session.invoke_tool(&tool, &call) {
            Ok(out) => {
                if json_out {
                    let body = serde_json::json!({
                        "tool": tool,
                        "content": out.content,
                        "metadata": out.metadata,
                    });
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&body).unwrap_or_default()
                    );
                } else {
                    print!("{}", out.content);
                    if !out.content.ends_with('\n') {
                        println!();
                    }
                }
            }
            Err(error) => {
                eprintln!("tool error: {error}");
                process::exit(1);
            }
        }
        return;
    }

    if let Some(answer) = clarify {
        if !session.has_pending_clarification() {
            eprintln!("error: no pending clarification for session '{session_id}'");
            process::exit(2);
        }
        let result = session.clarify_and_continue(&answer).unwrap_or_else(|e| {
            eprintln!("error: clarify: {e}");
            process::exit(1);
        });
        emit_result(&result, json_out);
        if !result.success && result.route != QueryRoute::Clarification {
            process::exit(1);
        }
        return;
    }

    let query = if let Some(pos) = args.iter().position(|a| a == "query") {
        args.iter()
            .skip(pos + 1)
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        args.iter()
            .skip(1)
            .filter(|a| !a.starts_with("--"))
            .filter(|a| {
                !matches!(
                    a.as_str(),
                    "query" | "read" | "write" | "list" | "get" | "post" | "run"
                )
            })
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };

    if query.trim().is_empty() {
        usage();
        process::exit(2);
    }

    let result = session.handle_query(query.trim());
    emit_result(&result, json_out);
    if !result.success && result.route != QueryRoute::Clarification {
        process::exit(1);
    }
}

fn emit_result(result: &AgentQueryResult, json_out: bool) {
    if json_out {
        let value = serde_json::json!({
            "route": format!("{:?}", result.route),
            "success": result.success,
            "workflow": result.workflow,
            "response": result.response,
            "clarification_questions": result.clarification_questions,
            "synthesis_method": result.synthesis_method,
            "tool_trace": result.tool_trace,
            "repo_result": result.repo_result,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value).unwrap_or_default()
        );
        return;
    }
    println!("route: {:?}", result.route);
    println!("workflow: {}", result.workflow);
    println!("success: {}", result.success);
    if let Some(method) = &result.synthesis_method {
        println!("synthesis: {}", method);
    }
    if !result.clarification_questions.is_empty() {
        println!("clarifications:");
        for q in &result.clarification_questions {
            println!("  - {q}");
        }
    }
    println!("---");
    println!("{}", result.response);
}
