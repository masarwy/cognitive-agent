from agent.tools.llm_tool import LLMTool


class ReasonTool(LLMTool):

    def __init__(self):

        super().__init__(
            name="reason",
            system_prompt="""
You are a senior AI systems engineer.

Reason deeply about:

• tradeoffs
• performance vs accuracy
• engineering implications
• best practices
"""
        )
