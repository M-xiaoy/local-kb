"""
poincare_search.py — Poincaré Ball 双曲距离检索
=================================================
替代 FAISS 余弦相似度，在双曲空间计算测地线距离。

使用方式：
  在 retriever.py 里 mode="poincare" 时调用此模块。
  外部不直接依赖 FAISS。

核心操作：
  1. to_poincare_ball() — 欧氏向量 → 映射到 Poincaré Ball (||x|| < 1)
  2. poincare_distance() — 两个双曲面向量间的测地线距离
  3. batch_search() — 全量遍历 + 双曲距离排序

性能备注：
  · O(n) 全遍历（无 ANN 索引），适合 < 10 万向量
  · > 10 万时需自行封装索引（如双曲空间 HNSW 变体）
  · 实测在 n=5000, dim=768 时一次约 15-30ms
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 安全边界：Poincaré Ball 的半径 = 1，数值上留 0.001 余量
_EPS = 1e-5
_BALL_RADIUS = 1.0 - 1e-5  # 留余量避免 arccosh 发散


# ──────────────────────────────────────────────
# 核心变换
# ──────────────────────────────────────────────

def to_poincare_ball(
    vectors: np.ndarray,
    norms: Optional[np.ndarray] = None,
    eps: float = _EPS,
) -> np.ndarray:
    """将欧氏向量映射到 Poincaré Ball 内 (||x|| < 1)

    有两种使用方式：
      1. norms=None（默认）：对 L2 归一化向量做范数压缩，保证 ||x|| < 1
         适合 FAISS 已归一化的向量直接塞入球中（但所有向量范数≈1，无层次）

      2. norms 指定每个向量的目标范数：方向来自 vectors，范数来自 norms
         球体的范数从社区覆盖率/层次结构推导而来。
         query 的范数固定为 0.5（中性抽象度）。
         这是正确用法——范数编码层次，方向编码语义。

    Args:
        vectors: shape (n, dim) 或 (dim,) — 欧氏空间向量（L2 归一化方向）
        norms: shape (n,) 或 None — 每个向量在 Poincaré Ball 中的目标范数
        eps: 数值稳定性边界

    Returns:
        shape 相同，满足 ||x|| ≤ 1 - eps 的 Poincaré Ball 向量
    """
    was_1d = vectors.ndim == 1
    if was_1d:
        vectors = vectors.reshape(1, -1)

    if norms is not None:
        # === 模式 2：指定范数 ===
        # 先确保方向向量是 L2 归一化的
        dir_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        dir_norms = np.where(dir_norms == 0, 1.0, dir_norms)
        unit_dirs = vectors / dir_norms

        # 把范数整理成 (n, 1) 形状
        if isinstance(norms, (int, float)):
            norm_array = np.full((vectors.shape[0], 1), float(norms))
        else:
            norms_arr = np.asarray(norms, dtype=np.float32).flatten()
            if norms_arr.shape[0] == 1:
                norm_array = np.full((vectors.shape[0], 1), norms_arr[0])
            else:
                norm_array = norms_arr.reshape(-1, 1)

        # 数值安全截断
        norm_array = np.clip(norm_array, 0.0, 1.0 - eps)

        result = unit_dirs * norm_array

        if was_1d:
            return result[0]
        return result

    # === 模式 1：自动压缩 ===（旧行为）
    was_1d = vectors.ndim == 1
    if was_1d:
        vectors = vectors.reshape(1, -1)

    vec_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    factors = (1.0 - eps) / np.maximum(vec_norms, eps)
    result = vectors * factors

    # 数值兜底：显式截断到 [-1+eps, 1-eps] 范围
    clip_bound = 1.0 - eps
    result = np.clip(result, -clip_bound, clip_bound)

    if was_1d:
        return result[0]
    return result


# ──────────────────────────────────────────────
# 距离计算
# ──────────────────────────────────────────────

def poincare_distance(
    u: np.ndarray,
    v: np.ndarray,
    eps: float = _EPS,
) -> float:
    """两个 Poincaré Ball 向量间的测地线距离

    Args:
        u: (dim,) — 已映射到 Poincaré Ball
        v: (dim,) — 已映射到 Poincaré Ball
        eps: 数值稳定性边界

    Returns:
        测地线距离 d(u, v) ≥ 0

    公式：
      d(u,v) = arccosh(1 + 2 * ||u-v||² / ((1 - ||u||²)(1 - ||v||²)))
    """
    diff_sq = float(np.sum((u - v) ** 2))
    u_norm_sq = float(np.sum(u ** 2))
    v_norm_sq = float(np.sum(v ** 2))

    denom = max((1.0 - u_norm_sq) * (1.0 - v_norm_sq), eps)
    arg = 1.0 + 2.0 * diff_sq / denom
    arg = max(arg, 1.0)  # arccosh 定义域 [1, +∞)

    return float(np.arccosh(arg))


def batch_poincare_distance(
    query: np.ndarray,
    candidates: np.ndarray,
    query_norm: float = 0.5,
    candidate_norms: Optional[np.ndarray] = None,
    eps: float = _EPS,
) -> np.ndarray:
    """向量化批量 Poincaré 距离计算

    Args:
        query: (dim,) — 查询向量（L2 归一化方向向量）
        candidates: (n, dim) — 候选向量（L2 归一化方向向量）
        query_norm: 查询在 Poincaré Ball 中的范数（默认 0.5，中性抽象度）
        candidate_norms: (n,) 或 None — 每个候选球的 Poincaré 范数
                          如果为 None，使用旧行为（自动压缩）
        eps: 数值稳定性边界

    Returns:
        (n,) — 每个候选向量到查询的测地线距离
    """
    n = candidates.shape[0]

    # 1. 映射到 Poincaré Ball
    q = to_poincare_ball(query.reshape(1, -1), norms=np.array([query_norm]), eps=eps)[0]
    c = to_poincare_ball(candidates, norms=candidate_norms, eps=eps)  # (n, dim)

    # 2. 批量距离计算
    q_norm_sq = float(np.sum(q ** 2))

    # ||u - v||² = ||u||² + ||v||² - 2·u·v
    diff_sq = q_norm_sq + np.sum(c ** 2, axis=1) - 2 * (c @ q)

    v_norm_sq = np.sum(c ** 2, axis=1)

    denom = np.maximum((1.0 - q_norm_sq) * (1.0 - v_norm_sq), eps)
    arg = 1.0 + 2.0 * diff_sq / denom
    arg = np.clip(arg, 1.0, None)

    return np.arccosh(arg).astype(np.float32)


# ──────────────────────────────────────────────
# 批量检索
# ──────────────────────────────────────────────

def batch_search(
    query_vector: np.ndarray,
    all_vectors: Dict[int, np.ndarray],
    top_k: int = 100,
    query_norm: float = 0.5,
    faiss_to_norm: Optional[Dict[int, float]] = None,
    eps: float = _EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在全量向量上执行 Poincaré Ball 检索

    Args:
        query_vector: (dim,) — 查询向量（L2 归一化方向向量）
        all_vectors: {faiss_id: vector} — 全量向量缓存（L2 归一化）
        top_k: 返回 top-k 结果
        query_norm: 查询在 Poincaré Ball 中的范数（默认 0.5）
        faiss_to_norm: {faiss_id: poincare_norm} — 每个 FAISS ID 对应的 Poincaré 范数
                        如果为 None，使用旧行为（自动压缩）
        eps: 数值稳定性边界

    Returns:
        (ids, distances, vectors):
          ids:       (top_k,) — 候选 IDs（int64，与原 FaissStore 兼容）
          distances: (top_k,) — 双曲距离（越小越相关）
          vectors:   (top_k, dim) — 对应的原始 L2 归一化向量
    """
    if not all_vectors:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    ids_list = list(all_vectors.keys())
    vecs = np.stack([all_vectors[fid] for fid in ids_list], axis=0)

    # 构造候选的范数数组
    candidate_norms = None
    if faiss_to_norm:
        candidate_norms = np.array(
            [faiss_to_norm.get(fid, 0.5) for fid in ids_list],
            dtype=np.float32,
        )

    distances = batch_poincare_distance(query_vector, vecs,
                                         query_norm=query_norm,
                                         candidate_norms=candidate_norms,
                                         eps=eps)

    # 按距离升序排序（小 = 更近）
    sorted_indices = np.argsort(distances)

    n = min(top_k, len(sorted_indices))
    top_indices = sorted_indices[:n]

    result_ids = np.array([ids_list[i] for i in top_indices], dtype=np.int64)
    result_dists = distances[top_indices]
    result_vecs = vecs[top_indices]

    return result_ids, result_dists, result_vecs


# ──────────────────────────────────────────────
# 快捷函数：与 FaissStore.search() API 兼容
# ──────────────────────────────────────────────

class PoincareSearch:
    """Poincaré Ball 检索器，与 FaissStore 接口对齐

    用法：
        searcher = PoincareSearch()
        ids, distances, vectors = searcher.search(query_vector, top_k=100)
    """

    def __init__(self):
        self._vectors: Dict[int, np.ndarray] = {}

    @property
    def count(self) -> int:
        return len(self._vectors)

    @property
    def is_built(self) -> bool:
        return self.count > 0

    def add_vectors(self, faiss_ids: np.ndarray, vectors: np.ndarray):
        """批量添加向量（与 FaissStore.add() 格式兼容）

        Args:
            faiss_ids: (n,) int64 — FAISS 兼容 ID
            vectors: (n, dim) float32 — 原始欧氏向量
        """
        for i in range(len(faiss_ids)):
            self._vectors[int(faiss_ids[i])] = vectors[i].copy()

    def build_from_store(self, faiss_store) -> int:
        """从 FaissStore 的向量缓存中批量加载

        Args:
            faiss_store: FaissStore 实例（读取其 _vectors 缓存）

        Returns:
            加载的向量数量
        """
        if hasattr(faiss_store, "_vectors"):
            self._vectors = dict(faiss_store._vectors)
        else:
            raise TypeError("faiss_store must have _vectors dict")
        logger.info(
            f"PoincareSearch: loaded {self.count} vectors from FaissStore"
        )
        return self.count

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        query_norm: float = 0.5,
        faiss_to_norm: Optional[Dict[int, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Poincaré Ball 双曲距离检索

        Args:
            query_vector: (dim,) or (1, dim) — L2 归一化方向向量
            top_k: 返回数量
            query_norm: 查询在 Poincaré Ball 中的范数（默认 0.5）
            faiss_to_norm: {faiss_id: poincare_norm} — 每个候选向量的范数

        Returns:
            (ids, distances, vectors)
        """
        if query_vector.ndim == 2:
            query_vector = query_vector[0]

        if not self.is_built:
            raise RuntimeError("PoincareSearch is empty. Add vectors first.")

        return batch_search(
            query_vector, self._vectors, top_k,
            query_norm=query_norm,
            faiss_to_norm=faiss_to_norm,
        )

    def get_projection_stats(self) -> dict:
        """返回 Poincaré Ball 上向量的分布统计

        可用于验证向量的双曲分布是否符合预期：
        - 靠球心 = 宽泛概念（小范数）
        - 靠球面 = 具体概念（大范数）
        """
        if not self._vectors:
            return {}

        norms = [
            float(np.linalg.norm(v)) for v in self._vectors.values()
        ]
        return {
            "count": len(norms),
            "min_norm": min(norms),
            "max_norm": max(norms),
            "mean_norm": float(np.mean(norms)),
            "median_norm": float(np.median(norms)),
            "ball_edge_ratio": sum(1 for n in norms if n > 0.8) / len(norms),
        }
