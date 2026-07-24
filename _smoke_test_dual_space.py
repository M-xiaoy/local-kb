"""
快速验证：双空间 Reranker 端到端
"""
import sys; sys.path.insert(0, '.')
import os; os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys; sys.stdout.reconfigure(encoding='utf-8')
import time

from api.main import app
from core.kb import KnowledgeBase
from core.repo.adapter import AdapterRepository
from config import paths
from storage.sphere_store import SphereStore
from storage.faiss_store import FaissStore
from storage.registry import Registry

# Load state
store = SphereStore(paths.spheres_data)
store.load()
faiss_store = FaissStore()
faiss_store.load(paths.faiss_index)
registry = Registry(paths.registry_map)
registry.load()

# Build retriever from shared state (bypass AppState for quick test)
from pipeline.embedder import Embedder
from retrieval.retriever import Retriever

embedder = Embedder()
retriever = Retriever(
    embedder=embedder,
    faiss_store=faiss_store,
    registry=registry,
    sphere_store=store,
)
retriever.build_norms_from_spheres()

# Test queries
tests = [
    "什么是重力空间",
    "Poincaré 球的双曲距离公式",
    "ComfyUI 工作流",
    "Transformer 注意力机制",
    "溪流传感器数据采集",
]

print("=== Dual-Space Reranker Smoke Test ===")
for q in tests:
    t0 = time.time()
    result = retriever.retrieve(q, use_hyperbolic=True)
    dt = time.time() - t0

    print(f"\nQ: {q}")
    print(f"  Time: {dt:.3f}s (embed+search+rerank)")
    print(f"  Timing: {result.timing}")
    print(f"  Top-5:")
    for i, (s, d) in enumerate(zip(result.spheres[:5], result.scores[:5])):
        snippet = s.text[:80].replace('\n', ' ')
        print(f"    {i+1}. [{s.source_type}] dist={d:.4f}  {snippet}...")

print("\nDone.")
