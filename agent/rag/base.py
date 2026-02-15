from abc import ABC, abstractmethod
from typing import List, Dict


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        pass

    @abstractmethod
    def ingest(self, documents: List[str]):
        pass
