import json
import requests
from typing import List
from dataclasses import dataclass

from agent.config import config


@dataclass
class PlanStep:
    id: int
    description: str
    tool: str


class LLMPlanner:
    """
    Planner powered by an LLM (NVIDIA NIM, OpenAI-compatible API, etc.)
    """

    def __init__(self):

        self.server_url = config.LLM_SERVER_URL
        self.model_name = config.LLM_MODEL_NAME
        self.api_key = config.NVIDIA_API_KEY

        self._validate_config()

    def _validate_config(self):

        if not self.server_url:
            raise ValueError(
                "APP_LLM_SERVERURL is not set.\n"
                "Example: https://integrate.api.nvidia.com"
            )

        if not self.model_name:
            raise ValueError(
                "APP_LLM_MODELNAME is not set.\n"
                "Example: nvidia/llama-3.3-nemotron-super-49b-v1.5"
            )

        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set.\n"
                "Please set it as an environment variable.\n"
                "Example:\n"
                "  export NVIDIA_API_KEY='your_api_key_here'"
            )

        if not isinstance(self.api_key, str):
            raise ValueError(
                "NVIDIA_API_KEY must be a string."
            )

        if not self.api_key.startswith("nvapi-"):
            raise ValueError(
                "Invalid NVIDIA_API_KEY format.\n"
                "The key must start with the 'nvapi-' prefix."
            )

    def create_plan(self, task: str) -> List[PlanStep]:

        prompt = self._build_prompt(task)

        response = self._call_llm(prompt)

        steps = self._parse_response(response)

        return steps

    def _build_prompt(self, task: str) -> str:
        tools = [
            "search",
            "retrieve",
            "analyze",
            "summarize",
            "reason",
            "code",
            "memory",
            "optimize",
            "ingest",
        ]

        tools_list = "\n".join(f"- {t}" for t in tools)

        return (
            "You are an AI agent planner.\n\n"
            "Your job is to break down the user's task into clear execution steps.\n\n"
            "IMPORTANT PLANNING RULES:\n"
            "- If the task references a local folder, file path, repository, or external data source, "
            "you MUST first use the ingest tool before using search, retrieve, analyze, or summarize.\n"
            "- search, retrieve, analyze, and summarize require indexed data.\n"
            "- ingest prepares external data for downstream tools.\n"
            "- Do NOT retrieve or analyze data that has not been ingested.\n\n"
            f"Available tools:\n{tools_list}\n\n"
            "Return ONLY valid JSON in this format:\n\n"
            "{\n"
            '  "steps": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "description": "...",\n'
            '      "tool": "search"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"User task:\n{task}"
        )

    def _call_llm(self, prompt: str) -> str:

        url = f"{self.server_url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a precise planner."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            print("LLM Error Response:")
            print(response.text)

        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def _parse_response(self, text: str) -> List[PlanStep]:

        try:

            # Find first JSON object in response
            json_start = text.find("{")
            json_end = text.rfind("}") + 1

            if json_start == -1 or json_end == -1:
                raise ValueError("No JSON object found in LLM response")

            json_text = text[json_start:json_end]

            data = json.loads(json_text)

            steps = []

            for step in data["steps"]:
                steps.append(
                    PlanStep(
                        id=step["id"],
                        description=step["description"],
                        tool=step["tool"]
                    )
                )

            return steps

        except Exception as e:

            print("\n=== RAW LLM RESPONSE ===")
            print(text)
            print("\n=== END RESPONSE ===")

            raise e
