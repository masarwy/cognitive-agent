# agent/tools/code_gen_tool.py
from agent.tools.llm_tool import LLMTool  # Assuming your LLMTool is in llm_tool.py

class CodeGenTool(LLMTool):
    def __init__(self):
        system_prompt = (
            "You are an AI assistant that generates Python code, configuration changes, or technical suggestions. "
            "Focus on providing clear, actionable, and correct code snippets or instructions. "
            "Explain briefly why the suggested change is helpful when relevant. "
            "Do not limit yourself to any particular type of optimization."
        )
        super().__init__("code", system_prompt)

    def execute(self, input_text: str) -> str:
        print(f"[{self.name}] Generating memory optimization code...")
        return super().execute(input_text)
