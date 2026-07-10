"""
field_detector.py — 场域检测器
===============================
给定查询向量，计算其对每个场域的亲和度（soft routing）。

核心思想（基于调研）：
  不搞硬路由（LLM 选一个场域），不搞元数据过滤（仅保留匹配的）。
  而是用 **场域质心 + 软评分**：查询向量与每个场域的质心做余弦相似度，
  得到一个 [0, 1] 的亲和度分数。这个分数在后续的多样性排序中
  作为权重因子——不排除任何场域的结果，只是倾向更匹配的场域。

质心路由的好处：
  1. 动态更新——每加一个文档自动重算质心，无需维护示例
  2. 软分配——一个查询可以同时高匹配多个场域（"小说化技术写作"）
  3. 轻量——质心就是均值向量，O(n) 计算，毫秒级

使用场景：
  query → embedder → query_vector
                    → field_detector.detect(query_vector)
                    → {"技术笔记": 0.85, "小说创作": 0.32, ...}
                    → 传给 diversity_sorter 做场域加权
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from config import ollama as cfg_ollama

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 场域检测器
# ──────────────────────────────────────────────

class FieldDetector:
    """给定查询向量，检测其对每个已知场域的亲和度"""

    def __init__(self, dim: Optional[int] = None):
        self.dim = dim or cfg_ollama.embed_dim
        # field_name → centroid vector (L2 归一化后的均值)
        self._centroids: Dict[str, np.ndarray] = {}
        # field_name → 该场域中的球体数量
        self._field_counts: Dict[str, int] = {}

    # ── 属性 ──────────────────────────────────

    @property
    def fields(self) -> List[str]:
        return list(self._centroids.keys())

    @property
    def field_count(self) -> int:
        return len(self._centroids)

    # ── 维护质心 ─────────────────────────────

    def update_centroid(
        self, field: str, vector: np.ndarray
    ):
        """增量更新一个场域的质心

        使用 Welford 在线算法（增量均值）：
          新均值 = 旧均值 + (新值 - 旧均值) / (n + 1)
        避免每次添加都重新计算全部球体的均值。
        """
        if field not in self._centroids:
            # 第一个球体 → 直接设为质心
            self._centroids[field] = vector.copy()
            self._field_counts[field] = 1
            return

        count = self._field_counts[field]
        old_centroid = self._centroids[field]

        # Welford 增量更新
        new_centroid = old_centroid + (vector - old_centroid) / (count + 1)

        # 重新归一化（确保质心在单位球面上）
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm

        self._centroids[field] = new_centroid
        self._field_counts[field] = count + 1

    def remove_centroid_contribution(
        self, field: str, vector: np.ndarray
    ):
        """从质心中移除一个球体的贡献（软删/重建时使用）

        逆向 Welford：
          新均值 = (旧均值 × n - 值) / (n - 1)
        """
        if field not in self._centroids:
            return

        count = self._field_counts[field]
        if count <= 1:
            # 最后一个球体被移除 → 删除该场域质心
            self._centroids.pop(field, None)
            self._field_counts.pop(field, None)
            return

        old_centroid = self._centroids[field]
        new_centroid = (old_centroid * count - vector) / (count - 1)

        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm

        self._centroids[field] = new_centroid
        self._field_counts[field] = count - 1

    def rebuild_centroids(self, field_vectors: Dict[str, List[np.ndarray]]):
        """全量重建所有场域质心

        用于从持久化数据恢复状态时。
        Args:
            field_vectors: {field_name: [vec1, vec2, ...]}
        """
        self._centroids.clear()
        self._field_counts.clear()

        for field, vectors in field_vectors.items():
            if not vectors:
                continue

            centroid = np.mean(vectors, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            self._centroids[field] = centroid
            self._field_counts[field] = len(vectors)

        logger.info(
            f"Rebuilt {len(self._centroids)} field centroids "
            f"from {sum(self._field_counts.values())} vectors"
        )

    # ── 检测 ──────────────────────────────────

    def detect(
        self, query_vector: np.ndarray, threshold: float = 0.0
    ) -> Dict[str, float]:
        """检测查询向量与每个场域的亲和度

        Args:
            query_vector: shape (dim,), float32, 已 L2 归一化
            threshold: 低于此值的场域不进结果（默认 0 = 全部保留）

        Returns:
            {field_name: affinity_score} 按亲和度降序
              affinity ∈ [0, 1]（归一化向量的余弦值经 [0,1] 映射）
        """
        if not self._centroids:
            return {}

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores: Dict[str, float] = {}

        for field, centroid in self._centroids.items():
            # 余弦相似度（已归一化 → dot product）
            raw = float(np.dot(query_vector, centroid.reshape(1, -1).T)[0, 0])

            # 将 [-1, 1] 映射到 [0, 1]（防止负值干扰权重计算）
            score = max(0.0, raw)

            if score >= threshold:
                scores[field] = round(score, 4)

        # 降序排列
        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    def best_field(self, query_vector: np.ndarray) -> tuple:
        """返回最匹配的场域和分数

        Returns:
            (field_name, score) or ("", 0.0) 没有场域时
        """
        scores = self.detect(query_vector)
        if not scores:
            return ("", 0.0)
        best = next(iter(scores.items()))
        return best

    # ── 序列化 ───────────────────────────────

    def get_state(self) -> dict:
        """导出状态用于持久化"""
        return {
            "centroids": {
                name: vec.tolist()
                for name, vec in self._centroids.items()
            },
            "counts": dict(self._field_counts),
        }

    def set_state(self, state: dict):
        """从持久化恢复状态"""
        self._centroids.clear()
        self._field_counts.clear()
        for name, vec_list in state.get("centroids", {}).items():
            vec = np.array(vec_list, dtype=np.float32)
            self._centroids[name] = vec
            self._field_counts[name] = state["counts"].get(name, 0)
