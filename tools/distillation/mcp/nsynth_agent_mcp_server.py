#!/usr/bin/env python3
"""
nsynth agent MCP server — thin stdio adapter over native Rust session API.

Capabilities are NOT hand-maintained here: `agent_capabilities` calls
`coding_agent --capabilities --json`, which introspects the live registry,
miner overlay, prose router, sandbox tools, and tensor surface at runtime.

Config example (Cursor / Claude Desktop):
    "mcpServers": {
      "nsynth-agent": {
        "command": "python3",
        "args": ["/absolute/path/nCPU/tools/distillation/mcp/nsynth_agent_mcp_server.py"],
        "env": {
          "NSYNTH_ROOT": "/absolute/path/nCPU",
          "NSYNTH_USE_RELEASE": "1"
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

SERVER_NAME = "nsynth-agent"
SERVER_VERSION = "0.2.0"


def nsynth_root() -> Path:
    env = os.environ.get("NSYNTH_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def nsynth_crate() -> Path:
    return nsynth_root() / "nsynth"


def _bin_cmd(bin_name: str, args: List[str]) -> tuple[List[str], Path]:
    crate = nsynth_crate()
    if not crate.is_dir():
        raise FileNotFoundError(f"nsynth crate not found at {crate}")
    use_release = os.environ.get("NSYNTH_USE_RELEASE", "").strip() == "1"
    release_bin = crate / "target" / "release" / bin_name
    if use_release and release_bin.is_file():
        return [str(release_bin), *args], crate
    return ["cargo", "run", "--quiet", "--bin", bin_name, "--", *args], crate


def run_bin(
    bin_name: str,
    args: List[str],
    *,
    timeout: int,
) -> Dict[str, Any]:
    cmd, cwd = _bin_cmd(bin_name, args)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "error": f"{bin_name} exceeded {timeout}s",
            "stdout": (exc.stdout or "")[-4000:],
            "stderr": (exc.stderr or "")[-4000:],
        }
    except FileNotFoundError as exc:
        return {"status": "error", "error": str(exc)}

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    body: Dict[str, Any] = {
        "status": "ok" if proc.returncode == 0 else "error",
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr[-8000:] if stderr else "",
        "command": cmd,
    }
    if proc.returncode != 0:
        body["error"] = stderr.strip() or stdout.strip() or f"exit {proc.returncode}"
    return body


def tool_agent_capabilities(args: Dict[str, Any]) -> Dict[str, Any]:
    root = args.get("root") or str(nsynth_root())
    timeout = int(args.get("timeout_s", 120))
    raw = run_bin(
        "coding_agent",
        ["--root", root, "--json", "--capabilities"],
        timeout=timeout,
    )
    if raw.get("status") != "ok":
        return raw
    try:
        return {"status": "ok", "capabilities": json.loads(raw["stdout"])}
    except json.JSONDecodeError:
        return {"status": "error", "error": "invalid capabilities JSON", "raw": raw["stdout"]}


def tool_agent_query(args: Dict[str, Any]) -> Dict[str, Any]:
    root = args.get("root") or str(nsynth_root())
    query = (args.get("query") or "").strip()
    if not query:
        return {"status": "error", "error": "query is required"}
    session = args.get("session", "mcp")
    timeout = int(args.get("timeout_s", os.environ.get("NSYNTH_QUERY_TIMEOUT", "600")))
    cmd_args = ["--root", root, "--session", session, "--json", "query", query]
    for host in args.get("allow_http_hosts") or []:
        cmd_args.extend(["--allow-http", str(host)])
    raw = run_bin("coding_agent", cmd_args, timeout=timeout)
    if raw.get("status") != "ok":
        return raw
    try:
        parsed = json.loads(raw["stdout"])
    except json.JSONDecodeError:
        parsed = {"raw_stdout": raw["stdout"]}
    if isinstance(parsed, dict):
        repo_result = parsed.get("repo_result")
        if isinstance(repo_result, dict):
            parsed["repair_summary"] = {
                "success": repo_result.get("success"),
                "repair_iterations": repo_result.get("repair_iterations"),
                "phases_completed": repo_result.get("phases_completed", []),
                "error": repo_result.get("error"),
            }
    return {"status": "ok", "result": parsed}


def tool_agent_clarify(args: Dict[str, Any]) -> Dict[str, Any]:
    root = args.get("root") or str(nsynth_root())
    answer = (args.get("answer") or "").strip()
    if not answer:
        return {"status": "error", "error": "answer is required"}
    session = args.get("session", "mcp")
    timeout = int(args.get("timeout_s", os.environ.get("NSYNTH_QUERY_TIMEOUT", "600")))
    raw = run_bin(
        "coding_agent",
        ["--root", root, "--session", session, "--json", "--clarify", answer],
        timeout=timeout,
    )
    if raw.get("status") != "ok":
        return raw
    try:
        parsed = json.loads(raw["stdout"])
    except json.JSONDecodeError:
        parsed = {"raw_stdout": raw["stdout"]}
    return {"status": "ok", "result": parsed}


def tool_agent_list_tools(args: Dict[str, Any]) -> Dict[str, Any]:
    root = args.get("root") or str(nsynth_root())
    timeout = int(args.get("timeout_s", 120))
    raw = run_bin(
        "coding_agent",
        ["--root", root, "--json", "--tools"],
        timeout=timeout,
    )
    if raw.get("status") != "ok":
        return raw
    try:
        caps = json.loads(raw["stdout"])
    except json.JSONDecodeError:
        caps = raw["stdout"].strip().splitlines()
    return {"status": "ok", "capabilities": caps}


def tool_agent_invoke_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    root = args.get("root") or str(nsynth_root())
    tool = (args.get("tool") or "").strip()
    action = (args.get("action") or "").strip()
    if not tool or not action:
        return {"status": "error", "error": "tool and action are required"}
    timeout = int(args.get("timeout_s", 300))
    cmd_args = ["--root", root, "--json", "--tool", tool, action]
    for key, value in (args.get("params") or {}).items():
        cmd_args.append(f"{key}={value}")
    raw = run_bin("coding_agent", cmd_args, timeout=timeout)
    if raw.get("status") != "ok":
        return raw
    try:
        parsed = json.loads(raw["stdout"])
    except json.JSONDecodeError:
        parsed = {"content": raw["stdout"]}
    return {"status": "ok", "result": parsed}


def tool_build_rule_backend(args: Dict[str, Any]) -> Dict[str, Any]:
    english = (args.get("english") or "").strip()
    if not english:
        return {"status": "error", "error": "english contract is required"}
    timeout = int(args.get("timeout_s", os.environ.get("NSYNTH_BUILD_TIMEOUT", "900")))
    mode = (args.get("mode") or "unified").strip().lower()
    store = (args.get("store") or "memory").strip().lower()
    out = args.get("output_path") or "demos/synthesized_backend/generated_rule_backend.rs"

    cmd_args: List[str] = []
    if mode in ("unified", "p2c"):
        cmd_args.append("--p2c")
    elif mode == "inline":
        pass
    elif mode == "hand":
        cmd_args.append("--hand-specs")
    else:
        return {"status": "error", "error": f"unknown mode: {mode}"}

    cmd_args.extend(["--store", store, "--out", out, "--text", english])
    raw = run_bin("build_backend_nl", cmd_args, timeout=timeout)
    if raw.get("status") != "ok":
        return raw

    artifact = nsynth_crate().parent / out
    source = artifact.read_text(encoding="utf-8") if artifact.is_file() else ""
    return {
        "status": "ok",
        "output_path": str(artifact),
        "source_bytes": len(source),
        "source_preview": source[:4000],
    }


TOOLS = {
    "agent_capabilities": {
        "impl": tool_agent_capabilities,
        "description": (
            "Runtime engine capability introspection from the live registry, miner "
            "overlay, prose router catalog, sandbox tools, and tensor forward surface. "
            "Call first — not a static hand list."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "string"},
                "timeout_s": {"type": "integer", "default": 120},
            },
        },
    },
    "agent_query": {
        "impl": tool_agent_query,
        "description": (
            "Run NL through the native coding agent. Returns route, success, "
            "response, synthesis_method, clarification questions, and repo_result "
            "(repair_iterations, phases_completed) on repair routes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "root": {"type": "string"},
                "session": {"type": "string", "default": "mcp"},
                "allow_http_hosts": {"type": "array", "items": {"type": "string"}},
                "timeout_s": {"type": "integer", "default": 600},
            },
            "required": ["query"],
        },
    },
    "agent_clarify": {
        "impl": tool_agent_clarify,
        "description": "Answer pending clarification for a session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "root": {"type": "string"},
                "session": {"type": "string", "default": "mcp"},
                "timeout_s": {"type": "integer", "default": 600},
            },
            "required": ["answer"],
        },
    },
    "agent_list_tools": {
        "impl": tool_agent_list_tools,
        "description": "List sandboxed tool capabilities allowed at root.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "string"},
                "timeout_s": {"type": "integer", "default": 120},
            },
        },
    },
    "agent_invoke_tool": {
        "impl": tool_agent_invoke_tool,
        "description": "Invoke one sandboxed agent tool (fs/shell/git/http/db).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "action": {"type": "string"},
                "params": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "root": {"type": "string"},
                "timeout_s": {"type": "integer", "default": 300},
            },
            "required": ["tool", "action"],
        },
    },
    "build_rule_backend": {
        "impl": tool_build_rule_backend,
        "description": (
            "Synthesize a local HTTP backend from an English contract via unified "
            "prose intake (registry doors + HTTP verify). Refuses honestly when "
            "comprehend cannot manufacture examples."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "english": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["unified", "p2c", "inline", "hand"],
                    "default": "unified",
                },
                "store": {
                    "type": "string",
                    "enum": ["memory", "file", "sqlite"],
                    "default": "memory",
                },
                "output_path": {"type": "string"},
                "timeout_s": {"type": "integer", "default": 900},
            },
            "required": ["english"],
        },
    },
}


def rpc_result(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_error(req_id: Any, code: int, msg: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}


def handle_initialize(_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def handle_tools_list() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["inputSchema"],
            }
            for name, spec in TOOLS.items()
        ]
    }


def handle_tools_call(params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name", "")
    args = params.get("arguments", {}) or {}
    spec = TOOLS.get(name)
    if spec is None:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"unknown tool: {name}"}],
        }
    try:
        result = spec["impl"](args)
    except Exception as exc:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"tool error: {exc!r}"}],
        }
    return {
        "content": [{"type": "text", "text": json.dumps(result, default=str)}],
    }


def serve_stdio() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            sys.stderr.write(f"[nsynth-agent-mcp] parse error: {exc}\n")
            sys.stderr.flush()
            continue
        method = req.get("method", "")
        req_id = req.get("id")
        params = req.get("params", {}) or {}

        if method == "initialize":
            resp = rpc_result(req_id, handle_initialize(params))
        elif method == "tools/list":
            resp = rpc_result(req_id, handle_tools_list())
        elif method == "tools/call":
            resp = rpc_result(req_id, handle_tools_call(params))
        elif method.startswith("notifications/"):
            continue
        elif req_id is None:
            continue
        else:
            resp = rpc_error(req_id, -32601, f"method not found: {method}")

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        serve_stdio()
    except (EOFError, KeyboardInterrupt):
        pass
