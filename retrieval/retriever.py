"""
retriever.py — 检索编排器
=========================
集成 FAISS + Registry + SphereStore + FieldDetector + DiversitySorter
的完整检索流水线。

执行流程：
  query_text
    → embedder.embed_query() → query_vector
    → faiss_store.search(query_vector, k=100) → faiss_ids
    → registry.sphere_id(faiss_id) → sphere_ids
    → sphere_store.get_many(sphere_ids) → Sphere 列表
    → field_detector.detect(query_vector) → field_scores
    → diversity_sorter.sort(...) → Top-5 sphere_ids + scores
    → sphere_store.get_many(Top-5) → 返回结果
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import retrieval as cfg_retrieval
from pipeline.embedder import Embedder
from storage.faiss_store import FaissStore
from storage.registry import Registry
from storage.sphere_store import SphereStore, Sphere
from retrieval.field_detector import FieldDetector
from retrieval.diversity_sorter import DiversitySorter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 结果类型
# ──────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """一次检索的完整结果"""
    query: str
    spheres: List[Sphere]          # Top-K Sphere 对象
    scores: List[float]            # 多样性排序得分
    field_affinities: Dict[str, float]  # 场域亲和度
    timing: Dict[str, float] = field(default_factory=dict)  # 各阶段耗时(秒)
    total_count: int = 0           # 知识库中总球体数


# ──────────────────────────────────────────────
# 检索编排器
# ──────────────────────────────────────────────

class Retriever:
    """检索编排器——组装各模块的完整流水线"""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        faiss_store: Optional[FaissStore] = None,
        registry: Optional[Registry] = None,
        sphere_store: Optional[SphereStore] = None,
        field_detector: Optional[FieldDetector] = None,
        diversity_sorter: Optional[DiversitySorter] = None,
    ):
        self.embedder = embedder or Embedder()
        self.faiss = faiss_store or FaissStore()
        self.registry = registry or Registry()
        self.spheres = sphere_store or SphereStore()
        self.field_detector = field_detector or FieldDetector()
        self.sorter = diversity_sorter or DiversitySorter(
            lambda_mmr=cfg_retrieval.similarity_weight * 2,  # 映射 [0,1]
            source_penalty=0.15,
            field_bonus_weight=0.1,
        )

    # ── 检索主入口 ───────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 100,
    ) -> RetrievalResult:
        """执行一次完整检索

        Args:
            query: 用户查询文本
            top_k: 最终返回数量
            fetch_k: FAISS 粗搜数量（候选池大小）

        Returns:
            RetrievalResult 包含结果和耗时指标
        """
        timings = {}
        t0 = time.time()

        # 1. 查询向量化
        query_vector = self.embedder.embed_query(query)
        timings["embed"] = time.time() - t0

        # 2. FAISS 粗搜（返回 ID、距离、向量）
        t1 = time.time()
        faiss_ids, faiss_distances, faiss_vectors = self.faiss.search(
            query_vector, top_k=fetch_k
        )
        timings["faiss_search"] = time.time() - t1

        if len(faiss_ids) == 0:
            return RetrievalResult(
                query=query,
                spheres=[],
                scores=[],
                field_affinities={},
                timing=timings,
                total_count=self.spheres.count,
            )

        # 3. Registry 转 sphere_ids（过滤无效映射）
        t2 = time.time()
        valid_indices = []
        sphere_ids = []
        for i, fid in enumerate(faiss_ids):
            sid = self.registry.sphere_id(fid)
            if sid is not None:
                valid_indices.append(i)
                sphere_ids.append(sid)

        if not sphere_ids:
            return RetrievalResult(
                query=query,
                spheres=[],
                scores=[],
                field_affinities={},
                timing=timings,
                total_count=self.spheres.count,
            )

        # 同步过滤向量
        candidate_vectors = faiss_vectors[valid_indices]

        # 4. 查 Sphere 元数据
        candidate_spheres = self.spheres.get_many(sphere_ids)
        timings["lookup"] = time.time() - t2

        # 5. 场域检测
        t3 = time.time()
        field_affinities = self.field_detector.detect(query_vector)
        timings["field_detect"] = time.time() - t3

        if not candidate_spheres:
            return RetrievalResult(
                query=query,
                spheres=[],
                scores=[],
                field_affinities=field_affinities,
                timing=timings,
                total_count=self.spheres.count,
            )

        # 6. 多样性排序（向量来自 FAISS 缓存，无需重新嵌入）
        t4 = time.time()
        candidate_ids = [s.id for s in candidate_spheres]
        candidate_sources = [s.source_file for s in candidate_spheres]
        candidate_types = [s.source_type for s in candidate_spheres]

        sorted_results = self.sorter.sort(
            query_vector=query_vector,
            candidate_vectors=candidate_vectors,
            candidate_ids=candidate_ids,
            source_files=candidate_sources,
            source_types=candidate_types,
            field_affinities=field_affinities,
            top_k=top_k,
        )
        timings["diversity_sort"] = time.time() - t4

        # 7. 组装结果
        final_spheres: List[Sphere] = []
        final_scores: List[float] = []

        for sphere_id, score in sorted_results:
            sphere = self.spheres.get(sphere_id)
            if sphere:
                final_spheres.append(sphere)
                final_scores.append(score)

        timings["total"] = time.time() - t0

        return RetrievalResult(
            query=query,
            spheres=final_spheres,
            scores=final_scores,
            field_affinities=field_affinities,
            timing=timings,
            total_count=self.spheres.count,
        )

    # ── 批量检索（无状态，多次调用 retrieve） ──

    def retrieve_batch(self, queries: List[str], **kwargs) -> List[RetrievalResult]:
        """批量检索多条查询"""
        return [self.retrieve(q, **kwargs) for q in queries]


# ──────────────────────────────────────────────
# 快捷函数
# ──────────────────────────────────────────────

_global_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = Retriever()
    return _global_retriever


def retrieve(query: str, **kwargs) -> RetrievalResult:
    return get_retriever().retrieve(query, **kwargs)
