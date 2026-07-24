"""KnowledgeBase — 重力知识库核心

单一入口，内部不走 HTTP。
外部通过 api/ 层调用，测试直接 import core.kb。

Phase 1 焦点：
  · add_document 路径：入库时确定 Poincaré 半径并落库
  · query 路径：通过 repo 接口做向量检索
  · 结构化日志：每次操作记录状态快照

日志约束（"不将就"规则）：
  · add_document: doc_id、半径值、范数、球体总数
  · query: query 摘要、mode、Top-3 得分与半径
  · 半径推导失败：raise RuntimeError（除非显式降级）
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from core.repo.interfaces import (
    KnowledgeBaseRepository,
    SearchResult,
    SphereData,
)
from pipeline.embedder import Embedder, poincare_project_query

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """重力知识库核心入口

    Usage:
        repo = AdapterRepository(sphere_store, faiss_store, registry)
        kb = KnowledgeBase(repo)
        kb.add_document(doc_id="...", text="...", source="...")
        results = kb.query("...", mode="poincare", top_k=10)
    """

    def __init__(self, repo: KnowledgeBaseRepository,
                 embedder: Optional[Embedder] = None,
                 retriever=None, sessions=None,
                 generator=None, rewriter=None,
                 field_detector=None, cluster_engine=None):
        self._repo = repo
        self._embedder = embedder or Embedder()
        # 外部子系统引用（Phase 4: 门面模式，由 api/main.py 注入）
        self.retriever = retriever
        self.sessions = sessions
        self.generator = generator
        self.rewriter = rewriter
        self.field_detector = field_detector
        self.cluster_engine = cluster_engine

    # ── 属性 ──────────────────────────────────

    @property
    def repo(self) -> KnowledgeBaseRepository:
        return self._repo

    @property
    def sphere_count(self) -> int:
        return self._repo.count()

    # ── 文档入库 ─────────────────────────────

    def add_document(self, text: str, source_file: str,
                     source_type: str = "", chunk: bool = True,
                     radius_override: Optional[float] = None,
                     doc_terms: Optional[List[str]] = None) -> int:
        """添加文档到知识库

        Args:
            text: 文档原文
            source_file: 源文件名
            source_type: 场域标签
            chunk: 是否分块（长文本自动分割）
            radius_override: 可选 — 强制指定 Poincaré 半径
                             （Phase 3 半径推导器实现前使用）
            doc_terms: 文档级关键术语列表

        Returns:
            新增球体数量

        Raises:
            RuntimeError: 半径推导失败且无降级策略时
        """
        from pipeline.chunker import chunk_text
        from storage.sphere_store import make_sphere_id as make_id

        t0 = time.time()

        # ── 分块 ─────────────────────────────
        if chunk:
            raw_chunks = chunk_text(text, source_type=source_type)
            # chunk_text returns list of strings; wrap as dict-like
            chunks = [
                type('Chunk', (), {'text': c, 'source_file': source_file})()
                for i, c in enumerate(raw_chunks) if c.strip()
            ]
        else:
            chunks = [
                type('Chunk', (), {'text': text, 'source_file': source_file})()
            ]

        # ── 构建球体 ─────────────────────────
        new_spheres: List[SphereData] = []
        for c in chunks:
            sid = make_id(c.text, source_file)
            sphere = SphereData(
                id=sid,
                text=c.text,
                source_file=source_file,
                source_type=source_type,
                active=True,
            )
            new_spheres.append(sphere)

        # ── 过滤已存在 ──────────────────────
        to_add = [s for s in new_spheres
                  if self._repo.get(s.id) is None]
        if not to_add:
            logger.info(
                f"add_document: source={source_file} — 0 new (all duplicate)"
            )
            return 0

        # ── 入库（元数据） ──────────────────
        added = self._repo.add_many(to_add)

        # ── 嵌入向量 ────────────────────────
        texts = [s.text for s in to_add]
        vectors = self._embedder.embed_documents(texts)
        sphere_ids = [s.id for s in to_add]
        self._repo.add_vectors(sphere_ids, vectors)

        # ── 半径推导（Phase 3：多信号融合） ──
        if radius_override is not None:
            for sphere in to_add:
                self._repo.set_poincare_norm(
                    sphere.id, radius_override, "explicit"
                )
        else:
            from core.hyperbolic.radius_deriver import RadiusDeriver
            deriver = RadiusDeriver(self._repo)
            norms = deriver.derive_batch([s.id for s in to_add])
            for sid, norm in norms.items():
                self._repo.set_poincare_norm(sid, norm, "radius_deriver_v3")

        # ── 结构化日志（"不将就"规则） ──────
        elapsed = time.time() - t0
        norm_vals = [self._repo.get_poincare_norm(s.id) for s in to_add]
        norm_str = (
            f"min={min(norm_vals):.3f} max={max(norm_vals):.3f}"
            if all(n is not None for n in norm_vals)
            else "N/A"
        )
        logger.info(
            f"add_document: source={source_file} "
            f"type={source_type} "
            f"added={added}/{len(to_add)} chunks "
            f"total_spheres={self._repo.count()} "
            f"poincare_norm=({norm_str}) "
            f"time={elapsed:.2f}s"
        )

        return added

    # ── 查询 ──────────────────────────────────

    def query(self, query_text: str, top_k: int = 10,
              fetch_k: int = 50,
              exclude_ids: Optional[List[str]] = None,
              use_hyperbolic: bool = True) -> SearchResult:
        """统一查询入口

        由 Retriever 执行 FAISS 初召 + (可选) Poincaré 重排。
        无降级路径 —— 未初始化 retriever 时直接返回空结果。

        Args:
            query_text: 自然语言查询
            top_k: 返回结果数
            fetch_k: FAISS 粗搜候选池大小
            exclude_ids: 排除球体列表
            use_hyperbolic: True=Poincaré 重排, False=纯余弦排序

        Returns:
            SearchResult
        """
        t0 = time.time()

        if self.retriever is None:
            logger.error("kb.query called but retriever is None")
            return SearchResult(sphere_ids=[], distances=[], scores=[])

        try:
            raw = self.retriever.retrieve(
                query=query_text,
                top_k=top_k,
                fetch_k=fetch_k,
                exclude_ids=exclude_ids or [],
                use_hyperbolic=use_hyperbolic,
            )
            sphere_ids = [s.id for s in raw.spheres]
            scores = list(raw.scores)
            elapsed = time.time() - t0

            top_norms = []
            for sid in sphere_ids[:3]:
                n = self._repo.get_poincare_norm(sid)
                top_norms.append(f"{n:.3f}" if n else "?")
            mode_label = "hyperbolic" if use_hyperbolic else "cosine"
            logger.info(
                f"kb.query: query='{query_text[:40]}' "
                f"mode={mode_label} results={len(sphere_ids)} "
                f"top_scores={[round(s,4) for s in scores[:3]]} "
                f"top_norms=[{','.join(top_norms)}] "
                f"time={elapsed:.2f}s"
            )
            return SearchResult(
                sphere_ids=sphere_ids,
                distances=[],
                scores=scores,
            )
        except Exception as e:
            logger.error(f"kb.query failed: {e}")
            return SearchResult(sphere_ids=[], distances=[], scores=[])

    # ── 评分策略 ─────────────────────────────

    def _score_poincare(
        self, sphere_ids: List[str], distances: List[float],
        query_vec: np.ndarray,
        sphere_vectors: Optional[np.ndarray] = None
    ) -> List:
        """真实 Poincaré 双曲距离重排

        d(x,y) = arccosh(1 + 2·||x-y||² / ((1-||x||²)(1-||y||²)))

        这个距离有「近球心处放大微观差异」的几何性质：
          · 球心附近（低 norm，抽象概念）：距离放大，精细区分
          · 球面附近（高 norm，具体事实）：距离缩小，宽泛匹配
          · 同一径向方向上向量之间的差异被非线性放大

        Returns:
            [(sphere_id, score, ip_dist)]
            score = -poincare_dist（负值，大=好，与排序公约保持一致）
        """
        from config import poincare as poincare_cfg

        eps = poincare_cfg.eps
        q = query_vec.flatten().astype(np.float64)
        q_norm_sq = float(np.dot(q, q))
        q_denom = max(1.0 - q_norm_sq, eps)

        scored = []
        for i, (sid, ip_dist) in enumerate(zip(sphere_ids, distances)):
            sv = sphere_vectors[i].astype(np.float64) if (sphere_vectors is not None and i < len(sphere_vectors)) else None

            if sv is not None:
                x_norm_sq = float(np.dot(sv, sv))
                x_denom = max(1.0 - x_norm_sq, eps)

                diff = q - sv
                diff_sq = float(np.dot(diff, diff))

                cosh_arg = 1.0 + 2.0 * diff_sq / max(x_denom * q_denom, eps)
                cosh_arg = max(cosh_arg, 1.0 + eps)

                poincare_dist = float(np.arccosh(cosh_arg))
                score = -poincare_dist
            else:
                # 降级：没有向量就用 IP 距离
                score = float(ip_dist)

            scored.append((sid, score, float(ip_dist)))

        return scored

    def _score_gravity(
        self, sphere_ids: List[str], distances: List[float],
        query_vec: np.ndarray
    ) -> List:
        """引力空间评分（现有 gravity mode 逻辑的简化版）

        融合质量 + 多样性 + 余弦相似度
        """
        scored = []
        for sid, ip_dist in zip(sphere_ids, distances):
            sphere = self._repo.get(sid)
            if not sphere:
                continue
            # effective_mass 作为权重
            mass_weight = sphere.effective_mass
            score = ip_dist * mass_weight
            scored.append((sid, float(score), float(ip_dist)))

        return scored

    # ── 状态 ──────────────────────────────────

    def status(self) -> dict:
        """返回运行时状态快照"""
        return {
            "total_spheres": self._repo.total_count(),
            "active_spheres": self._repo.count(),
            "vectors": self._repo.count(),
            "dim": self._repo.dim(),
        }

    # ══════════════════════════════════════════
    # Phase 4: Facade 门面方法
    # ══════════════════════════════════════════

    def get_document(self, doc_id: str) -> Optional[dict]:
        """获取文档元数据 + 当前半径"""
        meta = self._repo.get_metadata(doc_id)
        if not meta:
            return None
        norm = self._repo.get_poincare_norm(doc_id)
        meta["poincare_norm"] = norm
        return meta

    def delete_document(self, doc_id: str) -> bool:
        """删除文档，同步清理连接/半径/向量"""
        logger.info(f"KnowledgeBase.delete_document: id={doc_id[:8]}")
        self._repo.delete_edges(doc_id)
        self._repo.delete_poincare_norm(doc_id)
        return self._repo.delete_sphere(doc_id)

    def list_documents(self, limit: int = 100,
                       offset: int = 0) -> List[dict]:
        """分页列出文档，附带半径"""
        doc_ids = self._repo.list_ids(limit, offset)
        result = []
        for sid in doc_ids:
            meta = self._repo.get_metadata(sid) or {}
            norm = self._repo.get_poincare_norm(sid) or 0.0
            meta["poincare_norm"] = norm
            result.append(meta)
        logger.info(f"KnowledgeBase.list_documents: {len(result)} docs "
                    f"(offset={offset}, limit={limit})")
        return result

    def get_stats(self) -> dict:
        """全局统计（含范数分布）"""
        active_ids = self._repo.list_ids(1000, 0)
        norms = []
        for sid in active_ids:
            n = self._repo.get_poincare_norm(sid)
            if n is not None:
                norms.append(n)
        return {
            "total_spheres": self._repo.total_count(),
            "active_spheres": self._repo.count(),
            "norm_sample_count": len(norms),
            "avg_norm": float(np.mean(norms)) if norms else 0.0,
            "min_norm": float(np.min(norms)) if norms else 0.0,
            "max_norm": float(np.max(norms)) if norms else 0.0,
            "dim": self._repo.dim(),
        }

    def add_connection(self, from_id: str, to_id: str,
                       weight: float = 1.0) -> bool:
        """添加连接边（不触犯半径重算）"""
        logger.info(f"KnowledgeBase.add_connection: "
                    f"{from_id[:8]} → {to_id[:8]} w={weight}")
        return self._repo.add_edge(from_id, to_id, weight)

    def get_connections(self, doc_id: str) -> List[dict]:
        """获取节点的所有连接"""
        edges = self._repo.get_connections(doc_id)
        return [
            {"target": tid, "weight": w}
            for tid, w in edges.items()
        ]

    def get_active_spheres(self) -> List[SphereData]:
        """获取所有活跃球体（给 explore/bridge/trace 用）"""
        return self._repo.get_active()

    def get_sphere(self, sphere_id: str) -> Optional[SphereData]:
        """获取单个球体"""
        return self._repo.get(sphere_id)
