"""
MCP (Model Context Protocol) integration for the cognitive-agent.

This package exposes the cognitive-agent's tool suite as an MCP-compliant
server built on top of `fastmcp`. It preserves the existing tool
implementations (under ``agent.tools``) and simply re-surfaces them via
``@mcp.tool()`` decorators so that the planner/executor loop can invoke
them through the standardized MCP protocol.

Modules
-------
- ``server``: FastMCP server that registers every tool as an MCP tool.
- ``client``: Thin in-process client used by :class:`ToolExecutor` to
  dispatch tool calls through MCP while keeping the legacy abstraction
  layer available as a fallback.
"""

from agent.mcp.server import build_mcp_server, mcp  # noqa: F401
from agent.mcp.client import MCPToolClient  # noqa: F401

__all__ = ["build_mcp_server", "mcp", "MCPToolClient"]
