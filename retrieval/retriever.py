"""
retriever.py — 重力检索编排器（v2）
====================================
集成所有检索模块的完整流水线。

检索流程（四种模式可选）：

  simple — 纯 FAISS 相似度（保留原始逻辑，迁移过渡用）
     query → embed → FAISS search → lookup → simple sort → Top-5

  gravity — 默认重力检索
     query → embed → FAISS search → (可选activation) → (可选rerank)
     → gravity_focus → diversity_sort → Top-5

  deep — 完整深度检索（最慢但最准）
     query → rewrite → embed → FAISS → activation → rerank
     → gravity_focus → diversity_sort → Top-5

  poincare — 双曲空间 Poincaré Ball 检索
     query → embed → PoincaréDistance → lookup → gravity_focus
     → diversity_sort → Top-5
     （不依赖 FAISS，纯 NumPy 双曲距离）

  explore — 探索模式（用于内部工具）
     不依赖 query，直接对 sphere_id 或 cluster_id 操作
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import retrieval as cfg_retrieval, activation as cfg_activation, role as cfg_role
from config import traction as cfg_traction
from pipeline.embedder import Embedder
from pipeline.keywords import extract_from_query, match_term_gravity
from pipeline.rewriter import TextRewriter
from storage.faiss_store import FaissStore
from storage.registry import Registry
from storage.sphere_store import SphereStore, Sphere
from retrieval.field_detector import FieldDetector
from retrieval.diversity_sorter import DiversitySorter
from retrieval.activation import ActivationPropagator
from retrieval.reranker import LocalReranker
from retrieval.role_expander import RoleExpander
from retrieval.poincare_search import PoincareSearch
from retrieval.traction_reranker import TractionReranker, TractionConfig as TractionRerankerConfig

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
    field_affinities: Dict[str, float] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    total_count: int = 0
    mode: str = "gravity"
    propagation_stats: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# 检索编排器
# ──────────────────────────────────────────────

class Retriever:
    """重力检索编排器 —— 组装各模块的完整流水线"""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        faiss_store: Optional[FaissStore] = None,
        registry: Optional[Registry] = None,
        sphere_store: Optional[SphereStore] = None,
        field_detector: Optional[FieldDetector] = None,
        diversity_sorter: Optional[DiversitySorter] = None,
        propagator: Optional[ActivationPropagator] = None,
        reranker: Optional[LocalReranker] = None,
        rewriter: Optional[TextRewriter] = None,
        connections_provider: Optional[Callable] = None,
    ):
        self.embedder = embedder or Embedder()
        self.faiss = faiss_store or FaissStore()
        self.poincare = PoincareSearch()
        self.registry = registry or Registry()
        self.spheres = sphere_store or SphereStore()
        self.field_detector = field_detector or FieldDetector()
        self.sorter = diversity_sorter or DiversitySorter(
            lambda_mmr=cfg_retrieval.similarity_weight,
            source_penalty=0.15,
            field_bonus_weight=0.1,
        )
        self.propagator = propagator or ActivationPropagator()
        self.reranker = reranker or LocalReranker()
        self.rewriter = rewriter or TextRewriter()
        self.role_expander = RoleExpander()
        self.traction = TractionReranker(
            connections_provider=connections_provider,
            sphere_store=self.spheres,
            config=TractionRerankerConfig(
                alpha=cfg_traction.alpha,
                min_weight=cfg_traction.min_connection_weight,
                seed_boost=cfg_traction.seed_boost,
            )
        )
        self._conn_provider = connections_provider

        # 如果 propagator 没有 connections_provider，把我们的设给它
        if self.propagator._conn_provider is None and connections_provider:
            self.propagator.attach(connections_provider)

    def attach_connections(self, provider: Callable, type_checker: Optional[Callable] = None):
        """关联连接提供者"""
        self._conn_provider = provider
        self.propagator.attach(provider, type_checker=type_checker)

    def attach_role_table(self, role_table):
        """关联角色共现表"""
        self.role_expander.attach(role_table)

    # ── 检索主入口 ───────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 100,
        field_focus: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
        mode: str = "gravity",
        use_activation: Optional[bool] = None,
        use_reranker: Optional[bool] = None,
        use_rewrite: Optional[bool] = None,
        max_hops: int = 2,
    ) -> RetrievalResult:
        """执行一次完整检索

        Args:
            query: 用户查询
            top_k: 最终返回数量
            fetch_k: FAISS 粗搜数量
            field_focus: 场域聚焦
            exclude_ids: 排除的球体 ID
            mode: "simple" | "gravity" | "deep"
            use_activation: 覆盖 activation 开关
            use_reranker: 覆盖 reranker 开关
            use_rewrite: 覆盖 rewrite 开关
            max_hops: 激活传播跳数

        Returns:
            RetrievalResult
        """
        timings = {}
        t0 = time.time()

        # ── Step 0: Query Rewrite (仅 deep 模式) ──
        actual_query = query
        if mode == "deep" and (use_rewrite is None or use_rewrite):
            t_rewrite = time.time()
            try:
                clean = self.rewriter.rewrite(query, source_type="")
                if clean.cleaned_text and len(clean.cleaned_text) > 10:
                    actual_query = clean.cleaned_text[:500]
            except Exception as e:
                logger.debug(f"Query rewrite failed: {e}")
            timings["rewrite"] = time.time() - t_rewrite
        else:
            timings["rewrite"] = 0

        # ── Step 1: 查询向量化 ──
        t_embed = time.time()
        query_vector = self.embedder.embed_query(actual_query)
        timings["embed"] = time.time() - t_embed

        # ── Step 2: 场域检测 ──
        t_field = time.time()
        full_affinities: dict = {}
        if self.field_detector.field_count > 0:
            full_affinities = self.field_detector.detect(query_vector)
        timings["field_detect"] = time.time() - t_field

        # === 简单模式：纯 FAISS 相似度 ===
        if mode == "simple":
            return self._retrieve_simple(
                query, query_vector, fetch_k, top_k,
                field_focus, exclude_ids, timings, t0
            )

        # === Poincaré 模式：双曲空间检索 ===
        if mode == "poincare":
            if not self.poincare.is_built:
                # 从 FaissStore 的向量缓存构建（首次运行时自动同步）
                self.poincare.build_from_store(self.faiss)
            return self._retrieve_poincare(
                query, query_vector, fetch_k, top_k,
                field_focus, exclude_ids, timings, t0
            )

        # ── Step 3: FAISS 粗搜 ──
        t_faiss = time.time()
        faiss_ids, faiss_distances, faiss_vectors = self.faiss.search(
            query_vector, top_k=fetch_k
        )
        timings["faiss_search"] = time.time() - t_faiss

        if len(faiss_ids) == 0:
            return self._empty_result(query, timings, t0, mode)

        # ── Step 4: Registry → sphere_ids ──
        t_lookup = time.time()
        valid_indices, sphere_ids, candidate_vectors = \
            self._resolve_ids(faiss_ids, faiss_distances, faiss_vectors,
                              field_focus, exclude_ids)
        timings["lookup"] = time.time() - t_lookup

        if not sphere_ids:
            return self._empty_result(query, timings, t0, mode)

        # ── 保存种子球体 ID（牵引力用，在展开/传播之前）──
        seed_sphere_ids = list(sphere_ids)

        # ── Step 5: 查 Sphere 元数据 ──
        candidate_spheres = self.spheres.get_many(sphere_ids)

        if not candidate_spheres:
            return self._empty_result(query, timings, t0, mode)

        # ── Step 5.5: 一级球体展开 ──
        # FAISS 可能命中了一级球体（概念级），需要展开到它的二级子球体
        for sphere in list(candidate_spheres):
            if sphere.level == 1 and sphere.child_ids:
                children = self.spheres.get_children(sphere.id)
                for child in children:
                    if child.id not in sphere_ids:
                        sphere_ids.append(child.id)
                        candidate_spheres.append(child)

        # ── Step 6: Activation Propagation ──
        use_act = cfg_activation.enabled if use_activation is None else use_activation
        propagation_stats = {}
        if mode == "deep" or (use_act and self._conn_provider):
            t_act = time.time()
            seed_scores = dict(zip(sphere_ids,
                                    faiss_distances[valid_indices].tolist()))
            self.propagator.max_hops = max_hops
            propagated = self.propagator.propagate(seed_scores)

            if propagated:
                # 合并传播结果到候选
                propagated_ids = [sid for sid, _ in propagated]
                # 新孢子（不在 FAISS 结果中）
                new_ids = [sid for sid in propagated_ids
                           if sid not in sphere_ids]
                if new_ids:
                    new_spheres = self.spheres.get_many(new_ids)
                    for ns in new_spheres:
                        if ns:
                            sphere_ids.append(ns.id)
                            candidate_spheres.append(ns)

                # 更新排序依据
                prop_map = dict(propagated)
                # 创建一个排序分数：激活值 + 原始FAISS分数(打折)
                combined_scores = []
                for sid in sphere_ids:
                    act_score = prop_map.get(sid, 0)
                    faiss_idx = sphere_ids.index(sid) if sid in sphere_ids[:len(faiss_ids)] else -1
                    faiss_score = faiss_distances[faiss_idx] if faiss_idx >= 0 and faiss_idx < len(faiss_distances) else 0
                    combined = 0.6 * act_score + 0.4 * faiss_score
                    combined_scores.append((sid, combined))

                combined_scores.sort(key=lambda x: -x[1])
                sphere_ids = [sid for sid, _ in combined_scores]

                propagation_stats = self.propagator.activation_stats(propagated)
            timings["activation"] = time.time() - t_act
        else:
            timings["activation"] = 0

        # ── Step 6.5: Role Expansion（角色共现跳转） ──
        role_enabled = cfg_role.enabled and cfg_role.expand_after_faiss
        if role_enabled:
            t_role = time.time()
            faiss_scores = {
                sid: faiss_distances[i]
                for i, sid in enumerate(sphere_ids[:len(faiss_distances)])
                if i < len(faiss_distances)
            }
            role_candidates = self.role_expander.expand(
                faiss_hit_ids=sphere_ids[:len(faiss_distances)],
                faiss_hit_scores=faiss_scores,
                max_expansions_per_hit=cfg_role.max_expansions_per_hit,
                total_max_expansions=cfg_role.total_max_expansions,
                min_confidence=cfg_role.min_confidence,
                decay=cfg_role.decay_factor,
                exclude_ids=set(sphere_ids),
            )
            if role_candidates:
                new_role_ids = [sid for sid in role_candidates
                                if sid not in sphere_ids]
                if new_role_ids:
                    new_spheres = self.spheres.get_many(new_role_ids)
                    for ns in new_spheres:
                        if ns:
                            sphere_ids.append(ns.id)
                            candidate_spheres.append(ns)
                    logger.debug(
                        f"Role expansion added {len(new_role_ids)} candidates"
                    )
            timings["role_expansion"] = time.time() - t_role
        else:
            timings["role_expansion"] = 0

        # ── Step 7: Gravity Focus ──
        t_gravity = time.time()
        multipliers = self.field_detector.gravity_focus(
            query_vector, candidate_spheres, strength=0.2
        )
        timings["gravity_focus"] = time.time() - t_gravity

        # ── Step 8: Cross-encoder Rerank (可选) ──
        use_re = (mode == "deep") if use_reranker is None else use_reranker
        if use_re and len(candidate_spheres) > 3:
            t_rr = time.time()
            rerank_candidates = [(s.id, s.text[:500])
                                  for s in candidate_spheres[:cfg_retrieval.faiss_top_k]]
            reranked = self.reranker.rerank(actual_query, rerank_candidates,
                                             top_k=cfg_retrieval.final_top_k * 3)
            if reranked:
                rerank_ids = [sid for sid, _ in reranked]
                # 重排 sphere_ids
                sphere_ids = rerank_ids + [sid for sid in sphere_ids
                                            if sid not in rerank_ids]
            timings["rerank"] = time.time() - t_rr
        else:
            timings["rerank"] = 0

        # ── Step 9: 多样性排序（五层） ──
        t_sort = time.time()

        # 准备多样性排序的输入（取前 fetch_k 个）
        sort_ids = sphere_ids[:min(len(sphere_ids), cfg_retrieval.faiss_top_k)]
        sort_spheres = [s for s in candidate_spheres if s.id in sort_ids]
        sort_spheres.sort(key=lambda s: sort_ids.index(s.id))

        # 向量
        sort_vectors_list = []
        for s in sort_spheres:
            vec = self.faiss._vectors.get(
                self.registry.faiss_id(s.id)
            ) if self.registry and self.registry.faiss_id(s.id) else None
            if vec is None:
                vec = np.zeros(self.embedder.embed_dim, dtype=np.float32)
            sort_vectors_list.append(vec)

        if sort_vectors_list:
            sort_vectors = np.stack(sort_vectors_list, axis=0)
        else:
            sort_vectors = np.zeros((0, self.embedder.embed_dim), dtype=np.float32)

        sorted_results = self.sorter.sort(
            query_vector=query_vector,
            candidate_vectors=sort_vectors,
            candidate_ids=[s.id for s in sort_spheres],
            source_files=[s.source_file for s in sort_spheres],
            source_types=[
                f"簇{s.cluster_id}" if s.cluster_id >= 0 else s.source_type
                for s in sort_spheres
            ],
            field_affinities=full_affinities,
            top_k=top_k,
            connections_provider=self._conn_provider,
        )
        timings["diversity_sort"] = time.time() - t_sort

        # ── Step 10: 术语引力融合 ──
        t_term = time.time()
        query_keywords = extract_from_query(actual_query)
        if query_keywords and sorted_results:
            fused_results = []
            for sphere_id, div_score in sorted_results:
                sphere = self.spheres.get(sphere_id)
                tw = sphere.term_weights if sphere else {}
                term_score = match_term_gravity(query_keywords, tw)
                fused = 0.7 * div_score + 0.3 * term_score
                fused_results.append((sphere_id, fused))
            fused_results.sort(key=lambda x: -x[1])
            sorted_results = fused_results
        timings["term_fusion"] = time.time() - t_term

        # ── Step 10.5: 牵引力重排序 ──
        use_traction = cfg_traction.enabled and cfg_traction.alpha > 0
        if use_traction and sorted_results and seed_sphere_ids:
            t_tract = time.time()
            sorted_results = self.traction.rerank(
                sorted_results,
                seed_sphere_ids=seed_sphere_ids,
            )
            timings["traction"] = time.time() - t_tract
        else:
            timings["traction"] = 0

        # ── Step 11: Context Assembly ──
        # 检查是否有连续 chunk 可以合并
        t_ctx = time.time()
        seen_files = {}
        assembled = []
        for sphere_id, score in sorted_results:
            sphere = self.spheres.get(sphere_id)
            if not sphere:
                continue
            # 如果前一个球体与当前球体同源文件且时序相邻
            # 标记但不修改（留给生成器处理）
            assembled.append((sphere_id, score))
        timings["context_assembly"] = time.time() - t_ctx

        # ── 组装结果 ──
        final_spheres = []
        final_scores = []
        for sphere_id, score in assembled:
            sphere = self.spheres.get(sphere_id)
            if sphere:
                final_spheres.append(sphere)
                final_scores.append(score)

        timings["total"] = time.time() - t0

        return RetrievalResult(
            query=query,
            spheres=final_spheres[:top_k],
            scores=final_scores[:top_k],
            field_affinities=full_affinities,
            timing=timings,
            total_count=self.spheres.count,
            mode=mode,
            propagation_stats=propagation_stats,
        )

    # ── 简单模式（纯FAISS相似度，保留原始逻辑） ──

    def _retrieve_simple(self, query, query_vector, fetch_k, top_k,
                          field_focus, exclude_ids, timings, t0):
        """纯 FAISS 相似度检索"""
        faiss_ids, faiss_distances, faiss_vectors = self.faiss.search(
            query_vector, top_k=fetch_k
        )

        if len(faiss_ids) == 0:
            return self._empty_result(query, timings, t0, "simple")

        valid_indices, sphere_ids, _ = self._resolve_ids(
            faiss_ids, faiss_distances, faiss_vectors,
            field_focus, exclude_ids
        )

        if not sphere_ids:
            return self._empty_result(query, timings, t0, "simple")

        candidate_spheres = self.spheres.get_many(sphere_ids)

        # 直接按距离排序
        results = []
        for i, (sid, sphere) in enumerate(zip(sphere_ids, candidate_spheres)):
            if i < len(valid_indices) and valid_indices[i] < len(faiss_distances):
                results.append((sid, sphere,
                                float(faiss_distances[valid_indices[i]])))

        results.sort(key=lambda x: -x[2])

        # 来源去重
        source_count = {}
        deduped = []
        for sid, sphere, score in results[:top_k]:
            src = sphere.source_file or ""
            source_count[src] = source_count.get(src, 0) + 1
            if source_count[src] <= 3:
                deduped.append((sid, sphere, score))

        timings["total"] = time.time() - t0
        return RetrievalResult(
            query=query,
            spheres=[s for _, s, _ in deduped],
            scores=[s for _, _, s in deduped],
            field_affinities={},
            timing=timings,
            total_count=self.spheres.count,
            mode="simple",
        )

    # ── Poincaré 双曲检索模式 ─────────────────

    def _retrieve_poincare(self, query, query_vector, fetch_k, top_k,
                            field_focus, exclude_ids, timings, t0):
        """Poincaré Ball 双曲空间检索

        核心改进：方向来自 bge-m3 embedding，半径来自社区覆盖率/层次结构。
        不再对所有向量做自动压缩，而是使用每个球体的 poincare_norm。

        完整继承后续流水线：球体展开 → 重力聚焦 → 多样性排序 → 术语融合。
        """
        # ── 构建 faiss_id → poincare_norm 映射 ──
        faiss_to_norm = {}
        for sid, sphere in self.spheres._spheres.items():
            if sphere.active:
                fid = self.registry.faiss_id(sid)
                if fid is not None:
                    faiss_to_norm[fid] = sphere.poincare_norm

        # ── Step 3: Poincaré 距离检索 ──
        t_search = time.time()
        # 查询范数动态：短查询（词少，偏具体）→ 大范数靠球面；长查询（词多，偏抽象）→ 小范数靠球心
        word_count = len(query.strip().split())
        query_norm = max(0.1, min(0.9, 0.9 - 0.05 * word_count))
        try:
            if not self.poincare.is_built:
                self.poincare.build_from_store(self.faiss)
            faiss_ids, poincare_distances, poincare_vectors = self.poincare.search(
                query_vector, top_k=fetch_k,
                query_norm=query_norm,
                faiss_to_norm=faiss_to_norm,
            )
        except RuntimeError as e:
            logger.warning(f"Poincaré search failed, falling back to FAISS: {e}")
            faiss_ids, poincare_distances, poincare_vectors = self.faiss.search(
                query_vector, top_k=fetch_k
            )
        timings["poincare_search"] = time.time() - t_search

        if len(faiss_ids) == 0:
            return self._empty_result(query, timings, t0, "poincare")

        # ── Step 4: Registry → sphere_ids ──
        t_lookup = time.time()
        valid_indices, sphere_ids, candidate_vectors = \
            self._resolve_ids(faiss_ids, poincare_distances, poincare_vectors,
                              field_focus, exclude_ids)
        timings["lookup"] = time.time() - t_lookup

        if not sphere_ids:
            return self._empty_result(query, timings, t0, "poincare")

        # ── 保存种子球体 ID（牵引力用）──
        seed_sphere_ids = list(sphere_ids)

        # ── Step 5: 查 Sphere 元数据 ──
        candidate_spheres = self.spheres.get_many(sphere_ids)

        if not candidate_spheres:
            return self._empty_result(query, timings, t0, "poincare")

        # ── Step 5.5: 一级球体展开 ──
        for sphere in list(candidate_spheres):
            if sphere.level == 1 and sphere.child_ids:
                children = self.spheres.get_children(sphere.id)
                for child in children:
                    if child.id not in sphere_ids:
                        sphere_ids.append(child.id)
                        candidate_spheres.append(child)

        # ── Step 7: Gravity Focus（双曲模式下跳过了 activation 和 rerank） ──
        t_gravity = time.time()
        multipliers = self.field_detector.gravity_focus(
            query_vector, candidate_spheres, strength=0.2
        )
        timings["gravity_focus"] = time.time() - t_gravity

        # ── Step 9: 多样性排序 ──
        t_sort = time.time()

        sort_ids = sphere_ids[:min(len(sphere_ids), fetch_k)]
        sort_spheres = [s for s in candidate_spheres if s.id in sort_ids]
        sort_spheres.sort(key=lambda s: sort_ids.index(s.id))

        sort_vectors_list = []
        for s in sort_spheres:
            vec = self.faiss._vectors.get(
                self.registry.faiss_id(s.id)
            ) if self.registry and self.registry.faiss_id(s.id) else None
            if vec is None:
                vec = np.zeros(self.embedder.embed_dim, dtype=np.float32)
            sort_vectors_list.append(vec)

        if sort_vectors_list:
            sort_vectors = np.stack(sort_vectors_list, axis=0)
        else:
            sort_vectors = np.zeros((0, self.embedder.embed_dim), dtype=np.float32)

        sorted_results = self.sorter.sort(
            query_vector=query_vector,
            candidate_vectors=sort_vectors,
            candidate_ids=[s.id for s in sort_spheres],
            source_files=[s.source_file for s in sort_spheres],
            source_types=[
                f"簇{s.cluster_id}" if s.cluster_id >= 0 else s.source_type
                for s in sort_spheres
            ],
            field_affinities=self.field_detector.detect(query_vector),
            top_k=top_k,
            connections_provider=self._conn_provider,
        )
        timings["diversity_sort"] = time.time() - t_sort

        # ── Step 10: 术语引力融合 ──
        t_term = time.time()
        query_keywords = extract_from_query(query)
        if query_keywords and sorted_results:
            fused_results = []
            for sphere_id, div_score in sorted_results:
                sphere = self.spheres.get(sphere_id)
                tw = sphere.term_weights if sphere else {}
                term_score = match_term_gravity(query_keywords, tw)
                fused = 0.7 * div_score + 0.3 * term_score
                fused_results.append((sphere_id, fused))
            fused_results.sort(key=lambda x: -x[1])
            sorted_results = fused_results
        timings["term_fusion"] = time.time() - t_term

        # ── Step 10.5: 牵引力重排序 ──
        use_traction = cfg_traction.enabled and cfg_traction.alpha > 0
        if use_traction and sorted_results and seed_sphere_ids:
            t_tract = time.time()
            sorted_results = self.traction.rerank(
                sorted_results,
                seed_sphere_ids=seed_sphere_ids,
            )
            timings["traction"] = time.time() - t_tract
        else:
            timings["traction"] = 0

        # ── Step 11: Context Assembly ──
        t_ctx = time.time()
        assembled = []
        for sphere_id, score in sorted_results:
            sphere = self.spheres.get(sphere_id)
            if not sphere:
                continue
            assembled.append((sphere_id, score))
        timings["context_assembly"] = time.time() - t_ctx

        # ── 组装结果 ──
        final_spheres = []
        final_scores = []
        for sphere_id, score in assembled:
            sphere = self.spheres.get(sphere_id)
            if sphere:
                final_spheres.append(sphere)
                final_scores.append(score)

        timings["total"] = time.time() - t0

        return RetrievalResult(
            query=query,
            spheres=final_spheres[:top_k],
            scores=final_scores[:top_k],
            field_affinities=self.field_detector.detect(query_vector),
            timing=timings,
            total_count=self.spheres.count,
            mode="poincare",
            propagation_stats={},
        )

    # ── 辅助 ─────────────────────────────────

    def _resolve_ids(self, faiss_ids, faiss_distances, faiss_vectors,
                      field_focus, exclude_ids):
        """将 FAISS ID 解析为 sphere_id 列表（含过滤）"""
        valid_indices = []
        sphere_ids = []

        for i, fid in enumerate(faiss_ids):
            sid = self.registry.sphere_id(fid)
            if sid is not None:
                valid_indices.append(i)
                sphere_ids.append(sid)

        # 场域聚焦和排除
        if field_focus or exclude_ids:
            focused_indices = []
            focused_sids = []
            for idx_i, sid in enumerate(sphere_ids):
                if exclude_ids and sid in exclude_ids:
                    continue
                if field_focus and field_focus.startswith("簇"):
                    sphere = self.spheres.get(sid)
                    if not sphere:
                        continue
                    try:
                        expected_id = int(field_focus[1:])
                    except ValueError:
                        continue
                    if sphere.cluster_id != expected_id:
                        continue
                focused_indices.append(idx_i)
                focused_sids.append(sid)

            sphere_ids = focused_sids
            valid_indices = [valid_indices[i] for i in focused_indices]

        # 向量过滤
        if valid_indices:
            candidate_vectors = faiss_vectors[valid_indices]
        else:
            candidate_vectors = faiss_vectors[:0]

        return valid_indices, sphere_ids, candidate_vectors

    def _empty_result(self, query, timings, t0, mode):
        timings["total"] = time.time() - t0
        return RetrievalResult(
            query=query, spheres=[], scores=[],
            field_affinities={}, timing=timings,
            total_count=self.spheres.count, mode=mode,
        )


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
