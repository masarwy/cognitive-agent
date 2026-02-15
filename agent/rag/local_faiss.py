import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer

from agent.config import config
from .base import BaseRetriever


class LocalFaissRetriever(BaseRetriever):

    def __init__(self, path="agent/data/vectorstore"):

        self.path = path
        self.index_path = os.path.join(path, "index.faiss")
        self.docs_path = os.path.join(path, "documents.pkl")

        os.makedirs(path, exist_ok=True)

        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE
        )

        self.index = None
        self.documents = []
        self.document_set = set()

        self._load()

    # ------------------------

    def _load(self):

        if os.path.exists(self.index_path):
            print("[VectorStore] Loading FAISS index...")

            self.index = faiss.read_index(self.index_path)

        else:
            print("[VectorStore] No FAISS index found.")

        if os.path.exists(self.docs_path):

            print("[VectorStore] Loading documents...")

            with open(self.docs_path, "rb") as f:
                self.documents = pickle.load(f)

            self.document_set = set(self.documents)

        else:
            self.documents = []
            self.document_set = set()

        if self.index is None and len(self.documents) > 0:
            print("[VectorStore] Rebuilding FAISS index from documents...")

            embeddings = self.model.encode(
                self.documents,
                convert_to_numpy=True
            ).astype("float32")

            faiss.normalize_L2(embeddings)

            dim = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(dim)

            self.index.add(embeddings)

            self._save()

    # ------------------------

    def _save(self):

        print("[VectorStore] Saving index and documents...")

        faiss.write_index(self.index, self.index_path)

        with open(self.docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    # ------------------------

    def rebuild_index(self):

        print("[VectorStore] Rebuilding index from unique documents...")

        if not self.documents:
            print("[VectorStore] No documents to rebuild.")
            return

        # Remove duplicates
        unique_docs = list(dict.fromkeys(self.documents))

        self.documents = unique_docs
        self.document_set = set(unique_docs)

        embeddings = self.model.encode(
            unique_docs,
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)

        self._save()

        print(f"[VectorStore] Rebuilt with {len(unique_docs)} unique documents.")

    # ------------------------

    def ingest(self, documents):

        if not documents:
            return

        new_docs = [doc for doc in documents if doc not in self.document_set]

        if not new_docs:
            print("[VectorStore] No new documents to ingest.")
            return

        embeddings = self.model.encode(
            new_docs,
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]

        if self.index is None:
            print("[VectorStore] Creating new FAISS index...")

            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)

        self.documents.extend(new_docs)

        self.document_set.update(new_docs)

        self._save()

    # ------------------------

    def retrieve(self, query, top_k=5):

        if self.index is None or len(self.documents) == 0:
            return []

        top_k = min(top_k, len(self.documents))

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(embedding)

        distances, indices = self.index.search(
            embedding,
            top_k
        )

        results = []

        for i, idx in enumerate(indices[0]):

            if idx < 0 or idx >= len(self.documents):
                continue

            results.append({
                "text": self.documents[idx],
                "score": float(distances[0][i])
            })

        return results
