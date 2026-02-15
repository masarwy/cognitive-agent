from typing import List


class Agent:
    def __init__(self, name: str):
        self.name: str = name
        self.tools: List = []

    def register_tool(self, tool):
        self.tools.append(tool)

    def think(self, task: str):
        print(f"[{self.name}] Thinking about: {task}")

    def act(self, task: str):
        print(f"[{self.name}] Acting on: {task}")

    def run(self, task: str):
        self.think(task)
        self.act(task)
