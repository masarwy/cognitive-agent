import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ========= LLM =========

    LLM_SERVER_URL: str = os.getenv(
        "APP_LLM_SERVERURL",
        "https://integrate.api.nvidia.com"
    )

    LLM_MODEL_NAME: str = os.getenv(
        "APP_LLM_MODELNAME",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    )

    NVIDIA_API_KEY: str = os.getenv(
        "NVIDIA_API_KEY"
    )

    # ========= RAG =========

    RAG_BACKEND: str = os.getenv(
        "RAG_BACKEND",
        "local"
    )

    RAG_SERVER_URL: str = os.getenv(
        "RAG_SERVER_URL",
        "http://localhost:8000"
    )

    # ========= Embeddings =========

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5"
    )

    EMBEDDING_DEVICE: str = os.getenv(
        "EMBEDDING_DEVICE",
        "cpu"
    )


# Global config instance
config = Config()
