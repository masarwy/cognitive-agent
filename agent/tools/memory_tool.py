from agent.tools.base import Tool


class MemoryTool(Tool):

    def __init__(self):
        super().__init__("memory")
        self.storage = []

    def execute(self, input_text: str) -> str:

        print(f"[MemoryTool] Storing: {input_text}")

        self.storage.append(input_text)

        return "Stored in memory"
