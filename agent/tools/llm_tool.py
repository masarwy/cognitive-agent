import requests

from agent.tools.base import Tool
from agent.config import config


class LLMTool(Tool):

    def __init__(self, name, system_prompt):

        super().__init__(name)

        self.system_prompt = system_prompt

        self.api_key = config.NVIDIA_API_KEY
        self.base_url = config.LLM_SERVER_URL
        self.model = config.LLM_MODEL_NAME

    def execute(self, input_text: str) -> str:

        print(f"[{self.name}] Processing...")

        url = f"{self.base_url}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            "temperature": 0.2
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload
        )

        response.raise_for_status()

        data = response.json()

        raw_content = data["choices"][0]["message"]["content"]

        # Clean response automatically for all LLM tools
        return self._clean_response(raw_content)

    def _clean_response(self, text: str) -> str:
        """Remove thinking tags from LLM responses"""
        if '<think>' in text:
            think_start = text.find('<think>')
            think_end = text.find('</think>')
            if think_end != -1:
                think_end += len('</think>')
                text = text[:think_start] + text[think_end:]
        return text.strip()
