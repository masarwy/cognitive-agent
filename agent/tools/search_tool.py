from agent.tools.base import Tool


class SearchTool(Tool):

    def __init__(self):
        super().__init__("search")

    def execute(self, input_text: str) -> str:

        print(f"[SearchTool] Searching for: {input_text}")

        # Placeholder for real search
        return f"Search results for: {input_text}"
