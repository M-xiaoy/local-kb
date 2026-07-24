"""
retriever.py — 重力知识库·检索编排器（v4 双空间混合）
=================================================
2026-07-24 第四空间诊断确认：

  Embed (BGE-M3) → FAISS ANN (Top-50) → Dual-Space Rerank (Top-10)

  距离公式：d = alpha1 * ball_distance + alpha2 * euclidean_distance
  - 球面子空间（f[:256]）：聚类亲疏
  - 欧氏子空间（f[256:]）：语义远近
  
  诊断实验确认球面分量能有效区分同簇和无关对，α1=2.0, α2=1.0 起步。
  Poincaré 双曲分量因与欧氏高度相关（corr=0.87）暂不启用，
  等待投影层解耦后恢复三空间混合。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from pipeline.embedder import Embedder
from storage.faiss_store import FaissStore
from storage.registry import Registry
from storage.sphere_store import SphereStore, Sphere
from retrieval.dual_space_rerank import dual_space_rerank

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 结果类型
# ──────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """一次检索的完整结果"""
    query: str
    spheres: List[Sphere]
    scores: List[float]
    timing: Dict[str, float] = field(default_factory=dict)
    total_count: int = 0


# ──────────────────────────────────────────────
# 检索编排器（硬化版，无旧管线引用）
# ──────────────────────────────────────────────

class Retriever:
    """检索编排器 —— 硬化版

    新管线（唯一路径）：
      1. Embed: bge-m3 查询向量化
      2. FAISS: 欧氏 ANN 初召 top-K (fetch_k)
      3. Poincaré: 测地线距离重排候选集
      4. Source dedup & 截断输出

    不再支持多模式路由（simple/gravity/deep/poincare/explore）。
    所有旧模式已移除——现在只有一种检索路径。
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        faiss_store: Optional[FaissStore] = None,
        registry: Optional[Registry] = None,
        sphere_store: Optional[SphereStore] = None,
        poincare_norms: Optional[Dict[int, float]] = None,
    ):
        self.embedder = embedder or Embedder()
        self.faiss = faiss_store or FaissStore()
        self.registry = registry or Registry()
        self.spheres = sphere_store or SphereStore()
        self._faiss_to_norm = poincare_norms or {}

    def attach_poincare_norms(self, norms: Dict[int, float]):
        """关联 faiss_id → poincare_norm 映射（可选）"""
        self._faiss_to_norm = norms

    def build_norms_from_spheres(self):
        """从 SphereStore 自动构建范数映射"""
        self._faiss_to_norm = {}
        for sid, sphere in self.spheres._spheres.items():
            if sphere.active:
                fid = self.registry.faiss_id(sid)
                if fid is not None:
                    norm = getattr(sphere, 'poincare_norm', None)
                    if norm is not None and isinstance(norm, (int, float)) and 0 < norm < 1:
                        self._faiss_to_norm[fid] = norm
                    else:
                        self._faiss_to_norm[fid] = 0.5

    # ── 检索主入口 ───────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        fetch_k: int = 50,
        exclude_ids: Optional[List[str]] = None,
        max_source_count: int = 3,
        use_hyperbolic: bool = True,
    ) -> RetrievalResult:
        """执行一次检索

        Args:
            query: 用户查询文本
            top_k: 最终返回数量
            fetch_k: FAISS 粗搜数量（默认 50，别改）
            exclude_ids: 排除的球体 ID 列表
            max_source_count: 同源文件最大允许块数
            use_hyperbolic: 是否使用 Poincaré 测地线距离重排（False = 纯 FAISS 余弦排序）

        Returns:
            RetrievalResult
        """
        timings = {}
        t0 = time.time()

        # ── Step 1: 查询向量化 ──
        t_embed = time.time()
        query_vector = self.embedder.embed_query(query)
        timings["embed"] = time.time() - t_embed

        if query_vector is None or (hasattr(query_vector, 'size') and query_vector.size == 0):
            logger.warning("Query embedding returned empty")
            return RetrievalResult(query=query, spheres=[], scores=[], timing=timings)

        # ── Step 2: FAISS 欧氏初召 ──
        t_faiss = time.time()
        faiss_ids, faiss_distances, faiss_vectors = self.faiss.search(
            query_vector, top_k=fetch_k
        )
        timings["faiss_search"] = time.time() - t_faiss

        if len(faiss_ids) == 0:
            logger.warning("FAISS search returned no results")
            return RetrievalResult(query=query, spheres=[], scores=[], timing=timings)

        # ── Step 3: 解析 FAISS ID → Sphere ID（过滤 exclusions） ──
        t_resolve = time.time()
        valid_sphere_ids = []
        valid_vectors = []
        for i, fid in enumerate(faiss_ids):
            sid = self.registry.sphere_id(int(fid))
            if sid is None:
                continue
            if exclude_ids and sid in exclude_ids:
                continue
            sphere = self.spheres.get(sid)
            if not sphere or not sphere.active:
                continue
            valid_sphere_ids.append(sid)
            valid_vectors.append(faiss_vectors[i])

        if not valid_sphere_ids:
            return RetrievalResult(query=query, spheres=[], scores=[], timing=timings)

        valid_ids_arr = np.array(faiss_ids[:len(valid_sphere_ids)], dtype=np.int64)
        valid_vecs_arr = np.stack(valid_vectors, axis=0)
        timings["resolve"] = time.time() - t_resolve

        # ── Step 4: 重排（双空间混合距离 or FAISS 余弦） ──
        t_rerank = time.time()

        if use_hyperbolic:
            # 双空间混合重排：球面子空间（聚类）+ 欧氏子空间（语义）
            reranked = dual_space_rerank(
                query_vector=query_vector,
                candidate_vectors=valid_vecs_arr,
                candidate_ids=valid_ids_arr.tolist(),
                top_k=None,  # 全量重排，后续截断
            )
            sorted_ids = np.array([sid for sid, _ in reranked], dtype=np.int64)
            sorted_dists = np.array([d for _, d in reranked], dtype=np.float64)
            timings["dual_space_rerank"] = time.time() - t_rerank
        else:
            # 纯 FAISS 余弦距离排序（对照组）
            sorted_dists = faiss_distances[:len(valid_ids_arr)]
            sorted_indices = np.argsort(-sorted_dists)  # 余弦距离越大越相关
            sorted_ids = valid_ids_arr[sorted_indices]
            sorted_dists = sorted_dists[sorted_indices]
            timings["dual_space_rerank"] = time.time() - t_rerank
            timings["mode"] = "cosine"

        # ── Step 5: 查 Sphere 元数据 + 来源去重 ──
        t_dedup = time.time()
        sphere_ids_order = [self.registry.sphere_id(int(fid)) for fid in sorted_ids]

        source_count = {}
        final_spheres = []
        final_scores = []
        for sid, dist in zip(sphere_ids_order, sorted_dists):
            sphere = self.spheres.get(sid)
            if not sphere:
                continue

            src = sphere.source_file or ""
            source_count[src] = source_count.get(src, 0) + 1
            if source_count[src] > max_source_count:
                continue

            final_spheres.append(sphere)
            final_scores.append(float(dist))

            if len(final_spheres) >= top_k:
                break

        timings["dedup"] = time.time() - t_dedup
        timings["total"] = time.time() - t0

        return RetrievalResult(
            query=query,
            spheres=final_spheres,
            scores=final_scores,
            timing=timings,
            total_count=self.spheres.count,
        )

    # ── 兼容别名（旧评估脚本用，逐步弃用） ──

    def _retrieve_simple(self, query, query_vector, fetch_k, top_k,
                         field_focus, exclude_ids, timings, t0):
        """兼容旧评估脚本。内部调用新 retrieve()。"""
        result = self.retrieve(query, top_k=top_k, fetch_k=fetch_k,
                               exclude_ids=exclude_ids, max_source_count=3)
        # 包装回旧结构
        from types import SimpleNamespace
        fake = SimpleNamespace()
        fake.spheres = result.spheres
        fake.scores = result.scores
        fake.timing = result.timing
        timings.update(result.timing)
        return fake

    # ── 状态 ─────────────────────────────────

    @property
    def mode(self) -> str:
        return "hardened_v3"
