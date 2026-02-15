from agent.tools.registry import ToolRegistry


class ToolExecutor:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, tool_name: str, input_text: str):

        tool = self.registry.get(tool_name)

        result = tool.execute(input_text)

        return result
