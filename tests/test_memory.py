from agent.rag.local_faiss import LocalFaissRetriever

retriever = LocalFaissRetriever()

retriever.rebuild_index()

retriever.ingest([
    "TensorRT-LLM improves inference performance using kernel fusion.",
    "FAISS enables efficient similarity search.",
    "Quantization reduces model memory usage."
])

results = retriever.retrieve("TensorRT optimization")

print(results)
