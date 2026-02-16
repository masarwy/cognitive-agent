from agent.planning.llm_planner import LLMPlanner
from agent.tools.registry import ToolRegistry
from agent.tools.executor import ToolExecutor
from agent.tools.search_tool import SearchTool
from agent.tools.memory_tool import MemoryTool
from agent.tools.retrieve_tool import RetrieveTool
from agent.tools.ingest_tool import IngestTool
from agent.tools.analyze_tool import AnalyzeTool
from agent.tools.summarize_tool import SummarizeTool
from agent.tools.reason_tool import ReasonTool
from agent.tools.code_tool import CodeGenTool
from agent.tools.hardware_tool import HardwareTool
from agent.tools.code_analyze_tool import CodeAnalyzeTool
from agent.tools.optimization_tool import OptimizationTool


class Agent:

    def __init__(self, name: str):

        self.name = name

        self.planner = LLMPlanner()

        self.registry = ToolRegistry()

        self._register_tools()

        self.executor = ToolExecutor(self.registry)

    def _register_tools(self):

        # self.registry.register(SearchTool())
        # self.registry.register(MemoryTool())
        self.registry.register(RetrieveTool())
        self.registry.register(IngestTool())
        # self.registry.register(AnalyzeTool())
        self.registry.register(SummarizeTool())
        self.registry.register(ReasonTool())
        self.registry.register(CodeGenTool())
        self.registry.register(HardwareTool())
        self.registry.register(CodeAnalyzeTool())
        # self.registry.register(OptimizationTool())

    def think(self, task: str):

        print(f"\n[{self.name}] Task:")
        print(task)

        plan = self.planner.create_plan(task)

        print(f"\n[{self.name}] Plan:")

        for step in plan:
            print(f"{step.id}. {step.description} (tool={step.tool})")

        return plan

    def execute(self, plan):

        print(f"\n[{self.name}] Executing:")

        context = ""

        for step in plan:

            print(f"\nStep {step.id}: {step.description}")

            tool_input = context if context else step.description

            result = self.executor.execute(step.tool, tool_input)

            context = result

            print(f"Result: {result}")

    def run(self, task: str):

        plan = self.think(task)

        self.execute(plan)