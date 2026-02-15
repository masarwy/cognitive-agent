from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from agent.rag.factory import create_retriever
from .local_faiss import LocalFaissRetriever

app = FastAPI(title="Local RAG Server")

retriever = create_retriever()

# --- Architecture validation guard ---
if not isinstance(retriever, LocalFaissRetriever):
    raise RuntimeError(
        f"❌ Unsupported retriever type: {type(retriever).__name__}\n"
        "rag/server only supports LocalRetriever (FAISS).\n"
        "Check your RAG_MODE configuration."
    )


# ------------------------
# Request schemas
# ------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class IngestRequest(BaseModel):
    documents: List[str]


# ------------------------
# Endpoints
# ------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest):
    try:
        results = retriever.retrieve(
            request.query,
            top_k=request.top_k
        )

        return {"results": results}

    except Exception as e:
        return {
            "results": [],
            "error": str(e)
        }


@app.post("/ingest")
def ingest(request: IngestRequest):
    retriever.ingest(request.documents)

    return {
        "status": "ingested",
        "count": len(request.documents)
    }
