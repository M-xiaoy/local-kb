Gravity Knowledge Base — A local RAG that replaces BM25, HyDE, and cross-encoder rerankers with K-Means geometry.

No hybrid search, no cross-encoder, no HyDE generation. Just: cluster routing → query vector drift → FAISS search → term weight fusion → redundancy penalty.

Full pipeline ~2.1s (bottleneck is Ollama embedding), retrieval layer itself ~15ms.

Tech: Ollama + FAISS + sklearn + FastAPI. Single-file frontend. One-click deploy.

Open source, MIT: https://github.com/M-xiaoy/local-kb
