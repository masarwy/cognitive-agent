from agent.tools.llm_tool import LLMTool


class AnalyzeTool(LLMTool):

    def __init__(self):

        super().__init__(
            name="analyze",
            system_prompt="""
You are an expert AI researcher.

Analyze the provided information and extract:

• optimization techniques
• technical mechanisms
• performance implications

Be precise and technical.
"""
        )
