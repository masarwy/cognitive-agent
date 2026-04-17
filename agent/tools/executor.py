"""
Tool executor for the cognitive-agent.

Historically the executor dispatched tool calls through the custom
:class:`ToolRegistry` abstraction (``tool.execute(input_text)``). With
the MCP integration it can now route calls through an
:class:`agent.mcp.client.MCPToolClient` instead, speaking the Model
Context Protocol to the FastMCP server defined in ``agent.mcp.server``.

The legacy registry-based path is preserved as an automatic fallback so
that existing callers keep working even if the ``fastmcp`` dependency is
not installed or the MCP client fails at runtime. Select the backend
explicitly via the ``use_mcp`` flag.
"""

from __future__ import annotations

from typing import Optional

from agent.tools.registry import ToolRegistry


class ToolExecutor:
    """
    Execute tools via either the legacy ``ToolRegistry`` abstraction or
    the MCP protocol.

    Parameters
    ----------
    registry:
        The existing tool registry, used as the legacy/fallback path.
    use_mcp:
        If ``True``, route calls through an in-process MCP client. If
        ``False`` (default), use the legacy registry-based dispatch.
    mcp_client:
        Optional pre-built :class:`MCPToolClient`. When omitted and
        ``use_mcp=True``, a default client bound to the shared FastMCP
        server is instantiated lazily on first use.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        use_mcp: bool = False,
        mcp_client=None,
    ):
        self.registry = registry
        self.use_mcp = use_mcp
        self._mcp_client = mcp_client

    def execute(self, tool_name: str, input_text: str):
        if self.use_mcp:
            try:
                return self._execute_via_mcp(tool_name, input_text)
            except Exception as exc:  # pragma: no cover - defensive path
                print(
                    f"[ToolExecutor] MCP dispatch failed for '{tool_name}': "
                    f"{exc}. Falling back to legacy registry."
                )

        return self._execute_via_registry(tool_name, input_text)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _execute_via_registry(self, tool_name: str, input_text: str):
        tool = self.registry.get(tool_name)
        return tool.execute(input_text)

    def _execute_via_mcp(self, tool_name: str, input_text: str):
        client = self._get_mcp_client()
        return client.call(tool_name, input_text)

    def _get_mcp_client(self):
        if self._mcp_client is None:
            # Imported lazily so that users who never enable MCP do not
            # pay the ``fastmcp`` import cost.
            from agent.mcp.client import MCPToolClient

            self._mcp_client = MCPToolClient()
        return self._mcp_client
