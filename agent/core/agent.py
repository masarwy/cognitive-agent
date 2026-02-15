from agent.planning.llm_planner import LLMPlanner


class Agent:

    def __init__(self, name: str):

        self.name = name

        self.planner = LLMPlanner()

    def think(self, task: str):

        print(f"\n[{self.name}] Task:")
        print(task)

        plan = self.planner.create_plan(task)

        print(f"\n[{self.name}] Plan:")

        for step in plan:
            print(f"{step.id}. {step.description} (tool={step.tool})")

        return plan

    def execute(self, plan):

        print(f"\n[{self.name}] Executing...")

        for step in plan:
            print(f"Executing step {step.id}: {step.description}")

    def run(self, task: str):

        plan = self.think(task)

        self.execute(plan)