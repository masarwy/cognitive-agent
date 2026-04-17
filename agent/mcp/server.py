"""
FastMCP server exposing cognitive-agent tools as MCP-compliant tools.

Each existing :class:`agent.tools.base.Tool` subclass is wrapped with a
thin function decorated by ``@mcp.tool()``. The underlying tool logic and
call signatures are preserved verbatim — the MCP layer only adapts the
invocation interface to the Model Context Protocol so that any MCP-aware
client (the cognitive-agent's own executor, Claude Desktop, Cursor, etc.)
can call them.

Run the server directly::

    python -m agent.mcp.server

By default FastMCP uses stdio transport, which is what most MCP clients
expect. Pass ``--http`` to run a streamable HTTP server instead (useful
for remote or out-of-process executors).
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from fastmcp import FastMCP

from agent.tools.code_analyze_tool import CodeAnalyzeTool
from agent.tools.code_tool import CodeGenTool
from agent.tools.github_clone_tool import GitHubCloneTool
from agent.tools.hardware_tool import HardwareTool
from agent.tools.ingest_tool import IngestTool
from agent.tools.reason_tool import ReasonTool
from agent.tools.retrieve_tool import RetrieveTool
from agent.tools.summarize_tool import SummarizeTool


# Singleton FastMCP server. A single instance is reused so that both the
# standalone ``python -m agent.mcp.server`` entry point and the in-process
# MCP client (see ``agent.mcp.client``) share the exact same tool
# registrations.
mcp: FastMCP = FastMCP("cognitive-agent")


# Shared tool instances — we instantiate each underlying tool once and
# delegate @mcp.tool() calls to them. This preserves the original
# behaviour (including internal state such as LLM clients) while keeping
# the MCP wrappers lightweight.
_github_clone = GitHubCloneTool()
_ingest = IngestTool()
_retrieve = RetrieveTool()
_summarize = SummarizeTool()
_reason = ReasonTool()
_code_gen = CodeGenTool()
_hardware = HardwareTool()
_code_analyze = CodeAnalyzeTool()


@mcp.tool(
    name="github_clone",
    description=(
        "Clone a GitHub repository to the local filesystem. "
        "Input should describe the clone task and include the repo URL "
        "(e.g. 'Clone repo https://github.com/user/repo'). Returns the "
        "local clone path and file count."
    ),
)
def github_clone(step_description: str) -> str:
    """MCP-compliant wrapper around :class:`GitHubCloneTool`."""
    return _github_clone.execute(step_description)


@mcp.tool(
    name="ingest",
    description=(
        "Ingest local files (code, text, YAML, JSON) into the RAG server "
        "so that the `retrieve` and `code_analyze` tools can query them. "
        "Provide a folder path in the description (quoted or absolute)."
    ),
)
def ingest(step_description: str, file_types: Optional[List[str]] = None) -> str:
    """MCP-compliant wrapper around :class:`IngestTool`."""
    return _ingest.execute(step_description, file_types=file_types)


@mcp.tool(
    name="retrieve",
    description=(
        "Semantic search over the ingested corpus via the RAG server. "
        "Returns the concatenated top-k matching snippets."
    ),
)
def retrieve(input_text: str) -> str:
    """MCP-compliant wrapper around :class:`RetrieveTool`."""
    return _retrieve.execute(input_text)


@mcp.tool(
    name="summarize",
    description=(
        "Summarize technical content concisely, emphasising optimizations, "
        "performance improvements, and practical implementation insights."
    ),
)
def summarize(input_text: str) -> str:
    """MCP-compliant wrapper around :class:`SummarizeTool`."""
    return _summarize.execute(input_text)


@mcp.tool(
    name="reason",
    description=(
        "Senior AI-systems-engineer style reasoning: executive summary plus "
        "detailed analysis covering tradeoffs, performance vs accuracy, "
        "engineering implications, and best practices."
    ),
)
def reason(input_text: str) -> str:
    """MCP-compliant wrapper around :class:`ReasonTool`."""
    return _reason.execute(input_text)


@mcp.tool(
    name="code",
    description=(
        "Generate Python code, configuration changes, or technical "
        "suggestions based on the provided context."
    ),
)
def code(input_text: str) -> str:
    """MCP-compliant wrapper around :class:`CodeGenTool`."""
    return _code_gen.execute(input_text)


@mcp.tool(
    name="hardware_analyze",
    description=(
        "Detect local hardware capabilities: OS, CPU, RAM, GPU, CUDA, "
        "TensorRT, Tensor cores, AVX features, Jetson platform flags."
    ),
)
def hardware_analyze(input_text: str = "") -> str:
    """MCP-compliant wrapper around :class:`HardwareTool`."""
    return _hardware.execute(input_text)


@mcp.tool(
    name="code_analyze",
    description=(
        "Analyze an ingested codebase for memory usage, performance "
        "bottlenecks, and optimization opportunities. Requires that "
        "`ingest` has already been run on the target folder."
    ),
)
def code_analyze(step_description: str) -> str:
    """MCP-compliant wrapper around :class:`CodeAnalyzeTool`."""
    return _code_analyze.execute(step_description)


def build_mcp_server() -> FastMCP:
    """
    Return the configured FastMCP server instance.

    Exposed as a factory so external callers (tests, custom runners,
    :class:`agent.mcp.client.MCPToolClient`) can obtain the same
    singleton without importing internals.
    """
    return mcp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cognitive-agent FastMCP server."
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over streamable HTTP instead of the default stdio transport.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind when using --http (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind when using --http (default: 8765).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: ``python -m agent.mcp.server``."""
    args = _parse_args()
    if args.http:
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()  # stdio transport (MCP default)


if __name__ == "__main__":
    main()
