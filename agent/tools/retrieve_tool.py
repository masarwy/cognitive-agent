import requests

from agent.tools.base import Tool
from agent.config import config


class RetrieveTool(Tool):

    def __init__(self):

        super().__init__("retrieve")

        self.server_url = config.RAG_SERVER_URL

    def execute(self, input_text: str) -> str:

        print(f"[RetrieveTool] Querying RAG server: {input_text}")

        url = f"{self.server_url}/query"

        payload = {
            "query": input_text,
            "top_k": 5
        }

        response = requests.post(url, json=payload)

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        texts = [doc["text"] for doc in results]

        combined = "\n\n".join(texts)

        return combined if combined else "No results found."
