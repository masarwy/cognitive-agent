from .local_faiss import LocalFaissRetriever
from .nvidia_rag import NvidiaRetriever
from agent.config import config

def create_retriever():

    backend = config.RAG_BACKEND.lower()

    if backend == "local_faiss":
        return LocalFaissRetriever()

    elif backend == "nvidia_rag":
        return NvidiaRetriever()

    elif backend == "none":
        return None

    else:

        raise ValueError(
            f"Unknown RAG_BACKEND: {backend}"
        )