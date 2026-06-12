"""nCPU MCP server: natural language → verified program (Rung 7).

Stdio Model Context Protocol server exposing the nsynth synthesis
cascade as a coding tool: I/O examples in, proof-carrying code out.
Run with ``python3 -m ncpu.mcp_server``; see ``README.md`` for client
setup and the tool reference.
"""

from ncpu.mcp_server.fingerprint import examples_fingerprint, lookup_solved
from ncpu.mcp_server.server import (
    PROTOCOL_VERSION,
    SERVER_INFO,
    TOOL_DEFINITIONS,
    McpServer,
    main,
)
from ncpu.mcp_server.tools import (
    consult_library,
    library_stats,
    run_program,
    synthesize_from_examples,
    synthesize_from_prompt,
    verify_candidate,
)

__all__ = [
    "McpServer",
    "TOOL_DEFINITIONS",
    "PROTOCOL_VERSION",
    "SERVER_INFO",
    "main",
    "examples_fingerprint",
    "lookup_solved",
    "synthesize_from_examples",
    "synthesize_from_prompt",
    "consult_library",
    "library_stats",
    "verify_candidate",
    "run_program",
]
