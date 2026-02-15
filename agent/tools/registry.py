from typing import Dict
from agent.tools.base import Tool


class ToolRegistry:

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:

        if name not in self.tools:
            raise ValueError(f"Tool '{name}' is not registered.")

        return self.tools[name]

    def list_tools(self):
        return list(self.tools.keys())