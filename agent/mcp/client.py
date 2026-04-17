"""
In-process MCP client used by :class:`agent.tools.executor.ToolExecutor`.

The cognitive-agent historically invoked tools through a custom
abstraction layer (``ToolRegistry`` + ``Tool.execute``). With MCP
integration we want the executor to dispatch tool calls through the
Model Context Protocol instead — while keeping the legacy path fully
functional as a fallback.

FastMCP ships with a ``Client`` that can connect to a ``FastMCP`` server
object directly via an in-memory transport. That means we get a real
MCP-protocol round-trip (proper request/response framing, structured
tool results, schema validation) without the operational overhead of
managing a subprocess or HTTP server.

This module wraps that client in a small synchronous API that mirrors
:class:`agent.tools.executor.ToolExecutor.execute` so it can be plugged
in with minimal changes elsewhere in the codebase.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastmcp import Client, FastMCP

from agent.mcp.server import build_mcp_server


class MCPToolClient:
    """
    Thin synchronous facade over ``fastmcp.Client``.

    The client is lazily instantiated on first use and reused across
    calls. Each ``call`` opens a short-lived session against the bound
    FastMCP server (in-memory transport by default) and returns the
    tool's textual result, matching the legacy ``Tool.execute`` contract.
    """

    def __init__(self, server: Optional[FastMCP] = None) -> None:
        # Default to the singleton server defined in agent.mcp.server so
        # that every component in the process talks to the same tool
        # registrations.
        self._server: FastMCP = server or build_mcp_server()
        self._client: Client = Client(self._server)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(self, tool_name: str, input_text: str, **extra: Any) -> str:
        """
        Invoke ``tool_name`` through the MCP protocol and return the
        tool's text output.

        Parameters
        ----------
        tool_name:
            Name of the MCP tool to invoke (matches the ``name=`` kwarg
            of the corresponding ``@mcp.tool()`` decorator).
        input_text:
            Primary string input. Mapped to either ``input_text`` or
            ``step_description`` depending on the target tool's
            signature.
        **extra:
            Additional keyword arguments forwarded to the MCP tool.
        """
        arguments = self._build_arguments(tool_name, input_text, extra)
        return asyncio.run(self._call_async(tool_name, arguments))

    def list_tools(self) -> List[str]:
        """Return the names of all tools exposed by the MCP server."""
        return asyncio.run(self._list_tools_async())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Tools whose primary parameter is called ``step_description`` rather
    # than ``input_text``. Keeping this explicit (instead of
    # introspecting the server schema on every call) avoids an extra
    # round-trip per invocation and keeps behaviour deterministic.
    _STEP_DESCRIPTION_TOOLS = {"github_clone", "ingest", "code_analyze"}

    def _build_arguments(
        self,
        tool_name: str,
        input_text: str,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        if tool_name in self._STEP_DESCRIPTION_TOOLS:
            args: Dict[str, Any] = {"step_description": input_text}
        else:
            args = {"input_text": input_text}
        args.update(extra)
        return args

    async def _call_async(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        async with self._client as client:
            result = await client.call_tool(tool_name, arguments)
        return self._extract_text(result)

    async def _list_tools_async(self) -> List[str]:
        async with self._client as client:
            tools = await client.list_tools()
        return [t.name for t in tools]

    @staticmethod
    def _extract_text(result: Any) -> str:
        """
        Normalise a FastMCP ``CallToolResult`` to a plain string.

        FastMCP ≥ 2 returns a result object that may expose ``data``
        (structured output) or ``content`` (a list of content blocks).
        We prefer structured data when it's already a string and
        otherwise fall back to concatenating any text blocks, mirroring
        what the legacy tools returned directly.
        """
        data = getattr(result, "data", None)
        if isinstance(data, str):
            return data

        content = getattr(result, "content", None)
        if content:
            parts: List[str] = []
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts)

        # Last-resort fallback: stringify whatever came back so the
        # executor never receives ``None``.
        return "" if data is None else str(data)
