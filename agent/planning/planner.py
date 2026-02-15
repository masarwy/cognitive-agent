from typing import List
from dataclasses import dataclass


@dataclass
class PlanStep:
    id: int
    description: str
    tool: str = None


class Planner:
    """
    Converts a high-level task into executable steps.
    """

    def __init__(self):
        pass

    def create_plan(self, task: str) -> List[PlanStep]:
        """
        Simple rule-based planner (we'll replace with LLM later)
        """

        steps = []

        task_lower = task.lower()

        if "research" in task_lower:
            steps = [
                PlanStep(1, "Search for relevant information", tool="search"),
                PlanStep(2, "Retrieve relevant documents", tool="retrieve"),
                PlanStep(3, "Analyze retrieved content", tool="analyze"),
                PlanStep(4, "Summarize findings", tool="summarize"),
            ]

        elif "summarize" in task_lower:
            steps = [
                PlanStep(1, "Retrieve relevant documents", tool="retrieve"),
                PlanStep(2, "Summarize content", tool="summarize"),
            ]

        else:
            steps = [
                PlanStep(1, "Understand the task", tool="reason"),
                PlanStep(2, "Execute appropriate action", tool="act"),
            ]

        return steps
