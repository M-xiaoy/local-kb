# I built a local RAG that uses geometry instead of hybrid search — here's what I learned

**GitHub**: https://github.com/M-xiaoy/local-kb

I've been working on a local knowledge base RAG system that replaces several standard components with plain geometry operations on K-Means clusters. No BM25 index, no cross-encoder, no HyDE generation.

**How it works:**

1. Documents are uploaded → chunked → embedded (nomic-embed-text via Ollama locally) → clustered automatically (K-Means with Silhouette score for auto K detection)
2. Query comes in → query is embedded → cluster routing finds top-2 most relevant clusters
3. If initial top-5 results cluster in one cluster, the query vector is drifted 15% toward that cluster's centroid (replaces HyDE — same effect, zero tokens)
4. Term weights are embedded in each sphere on ingestion (pure Python regex + Counter, no BM25 index needed), then blended at 0.7 semantic + 0.3 term score
5. Redundancy penalty — spheres in dense regions get slightly penalized to force diversity

**Current state:**
- 1460 spheres, auto-detected 5 clusters
- Full pipeline latency ~2.1s (bottleneck is Ollama embedding at ~2s, the retrieval layer itself is ~15ms)
- Supports Ollama local LLM or DeepSeek API for answer generation
- Single-page HTML frontend with continuous session support
- One-click deploy on Windows/Linux/macOS

**Tech stack:** Ollama (nomic-embed-text) + FAISS + sklearn + FastAPI

**What I learned:**
- K-Means centroids are surprisingly good as a cluster-level index — equivalent to RAPTOR's tree search in one line of numpy
- Geometry drift (vector toward centroid) achieves what HyDE does without any generation cost
- TF weights embedded in the sphere metadata replaces BM25 without needing a separate inverted index. Good enough for personal KB scale (1000-10000 docs)

Thoughts? I'm especially curious about failure modes at scale — I'm guessing this breaks around 50k+ docs but I haven't tested that yet.

Repo: https://github.com/M-xiaoy/local-kb
