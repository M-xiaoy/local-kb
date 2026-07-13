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
from pipeline.keywords import extract_from_query, match_term_gravity
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
            lambda_mmr=cfg_retrieval.similarity_weight,  # [0,1], 0.6=60%相关+40%多样
            source_penalty=0.15,
            field_bonus_weight=0.1,
        )

    # ── 检索主入口 ───────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 200,
        field_focus: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """执行一次完整检索

        Args:
            query: 用户查询文本
            top_k: 最终返回数量
            fetch_k: FAISS 粗搜数量（候选池大小）
            field_focus: 聚焦某场域，只返回该场域内的球体
                         （基于 gravity_field 过滤）
            exclude_ids: 要排除的球体 ID 列表（会话去重）

        Returns:
            RetrievalResult 包含结果和耗时指标
        """
        timings = {}
        t0 = time.time()

        # 1. 查询向量化
        query_vector = self.embedder.embed_query(query)
        timings["embed"] = time.time() - t0

        # ── 2.5 簇级路由 ──────────────────────
        t_cluster = time.time()
        # 提前计算场域亲和度，用于候选优先排序
        full_affinities: dict = {}
        top2_clusters: set = set()
        if self.field_detector.field_count > 0:
            full_affinities = self.field_detector.detect(query_vector)
            if full_affinities:
                sorted_names = sorted(full_affinities, key=full_affinities.get, reverse=True)
                top2_clusters = set(sorted_names[:min(2, len(sorted_names))])
        timings["cluster_routing"] = time.time() - t_cluster

        # ── 2.6 引力漂移（Gravitational Drift） ──
        # 概念：初检发现多数结果指向同一簇 → 查询向量向该簇质心微调。
        # 等价于 HyDE 的「先理解再搜」，但走几何路径，不生成 token。
        t_drift = time.time()
        drift_applied = False
        if (
            full_affinities
            and self.field_detector.field_count > 0
            and self.faiss.is_built
        ):
            try:
                # 快速初检：搜 30 个看分布
                probe_ids, _, _ = self.faiss.search(query_vector, top_k=30)
                if len(probe_ids) >= 5:
                    probe_cluster_counts: dict = {}
                    for fid in probe_ids[:5]:
                        sid = self.registry.sphere_id(fid)
                        if sid:
                            sphere = self.spheres.get(sid)
                            if sphere and sphere.cluster_id >= 0:
                                cname = f"簇{sphere.cluster_id}"
                                probe_cluster_counts[cname] = (
                                    probe_cluster_counts.get(cname, 0) + 1
                                )

                    for cname, count in probe_cluster_counts.items():
                        if count >= 3 and cname in self.field_detector._centroids:
                            centroid = self.field_detector._centroids[cname]
                            query_vector = query_vector + 0.15 * centroid
                            norm = np.linalg.norm(query_vector)
                            if norm > 0:
                                query_vector = query_vector / norm
                            drift_applied = True
                            logger.info(
                                f"Gravitational drift: query toward {cname} "
                                f"({count}/5 in cluster)"
                            )
                            break
            except Exception:
                pass  # 漂移失败不影响主检索
        timings["gravity_drift"] = time.time() - t_drift

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

        # ── 3.5 场域聚焦过滤 ─────────────────
        if field_focus or exclude_ids:
            focused_indices = []
            focused_sphere_ids = []

            for idx_i, sid in enumerate(sphere_ids):
                # 排除已返回的球体
                if exclude_ids and sid in exclude_ids:
                    continue

                # 场域聚焦：只在对应聚簇中检索
                if field_focus:
                    sphere = self.spheres.get(sid)
                    if not sphere:
                        continue
                    # field_focus = "簇0" → 提取簇 ID
                    if field_focus.startswith("簇"):
                        try:
                            expected_id = int(field_focus[1:])
                        except ValueError:
                            continue
                        if sphere.cluster_id != expected_id:
                            continue
                    else:
                        # 兼容旧标签格式（暂保留）
                        if sphere.source_type != field_focus:
                            continue

                focused_indices.append(idx_i)
                focused_sphere_ids.append(sid)

            sphere_ids = focused_sphere_ids
            # 同步过滤候选向量
            if focused_indices:
                candidate_vectors = candidate_vectors[focused_indices]
            else:
                candidate_vectors = candidate_vectors[:0]  # 空

        # ── 3.6 簇级优先排序（无 field_focus 时启用） ──
        # 让属于 top-2 亲和簇的候选球体排在前面，MMR 优先从中选择
        if top2_clusters and not field_focus and len(sphere_ids) > 1:
            prio_indices = []
            non_prio_indices = []
            for i, sid in enumerate(sphere_ids):
                sphere = self.spheres.get(sid)
                cname = f"簇{sphere.cluster_id}" if sphere and sphere.cluster_id >= 0 else ""
                if cname in top2_clusters:
                    prio_indices.append(i)
                else:
                    non_prio_indices.append(i)

            if prio_indices and non_prio_indices:
                reorder = prio_indices + non_prio_indices
                sphere_ids = [sphere_ids[i] for i in reorder]
                candidate_vectors = candidate_vectors[reorder]

        # 4. 查 Sphere 元数据
        candidate_spheres = self.spheres.get_many(sphere_ids)
        timings["lookup"] = time.time() - t2

        # 5. 场域检测（复用路由阶段的结果，避免重复调用）
        t3 = time.time()
        field_affinities = full_affinities
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
        # 用簇名替代旧标签，使 field_bonus 能与 field_affinities 的“簇N”匹配
        candidate_types = [
            f"簇{s.cluster_id}" if s.cluster_id >= 0 else s.source_type
            for s in candidate_spheres
        ]

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

        # 6.5 术语引力融合（质量场）
        t_term = time.time()
        query_keywords = extract_from_query(query)
        if query_keywords and sorted_results:
            # 为每个候选球体计算术语引力分
            sphere_kw_map = {}
            for s in candidate_spheres:
                sphere_kw_map[s.id] = s.term_weights

            fused_results = []
            for sphere_id, div_score in sorted_results:
                tw = sphere_kw_map.get(sphere_id, {})
                term_score = match_term_gravity(query_keywords, tw)
                # 语义 0.7 + 术语 0.3
                fused = 0.7 * div_score + 0.3 * term_score
                fused_results.append((sphere_id, fused))
            # 按融合分降序
            fused_results.sort(key=lambda x: -x[1])
            sorted_results = fused_results
        timings["term_fusion"] = time.time() - t_term

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
