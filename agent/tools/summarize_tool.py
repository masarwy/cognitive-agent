from agent.tools.llm_tool import LLMTool


class SummarizeTool(LLMTool):

    def __init__(self):

        super().__init__(
            name="summarize",
            system_prompt="""
You summarize technical content clearly and concisely.

Focus on:

• key optimizations
• performance improvements
• practical implementation insights
"""
        )
