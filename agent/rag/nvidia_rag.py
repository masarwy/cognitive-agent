import requests
from .base import BaseRetriever

from agent.config import config


class NvidiaRetriever(BaseRetriever):

    def __init__(self):
        self.url = config.RAG_SERVER_URL

    def retrieve(self, query: str, top_k: int = 5):
        print(f"[NvidiaRetriever] Querying: {query}")

        response = requests.post(
            f"{self.url}/query",
            json={
                "query": query,
                "top_k": top_k
            }
        )

        response.raise_for_status()

        data = response.json()

        return data.get("results", [])

    def ingest(self, documents):
        raise NotImplementedError("Use NVIDIA ingestor service")
