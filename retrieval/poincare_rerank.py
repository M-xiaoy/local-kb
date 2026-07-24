"""
poincare_rerank.py — Poincaré 测地线距离重排（纯函数，1ms）
=====================================================
检索链路上的**唯一双曲点**。

在 FAISS 欧氏初召 Top-N 之后，对候选集做 Poincaré 测地线距离排序。
不依赖任何其他组件（无 DiversitySorter / FieldDetector / TractionReranker）。

用法：
    from retrieval.poincare_rerank import rerank
    sorted_ids, sorted_dists = rerank(query_vector, ids, vectors, norms)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from retrieval.poincare_search import batch_poincare_distance

logger = logging.getLogger(__name__)


def rerank(
    query_vector: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_vectors: np.ndarray,
    faiss_to_norm: Optional[Dict[int, float]] = None,
    query_norm: float = 0.5,
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS 候选集 → Poincaré 测地线距离重排

    Args:
        query_vector: (dim,) — L2 归一化查询向量
        candidate_ids: (n,) — FAISS ID 列表
        candidate_vectors: (n, dim) — 对应的 L2 归一化向量
        faiss_to_norm: {faiss_id: poincare_norm} — 候选集的 Poincaré 范数（可选）
        query_norm: 查询在 Poincaré Ball 中的范数（默认 0.5，中性抽象度）
        top_k: 返回数量（None = 全部）

    Returns:
        (sorted_ids, sorted_distances):
          sorted_ids:      按测地线距离升序排列的 FAISS ID
          sorted_distances:对应的测地线距离
    """
    if len(candidate_ids) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # 构造候选集范数数组
    candidate_norms = None
    if faiss_to_norm is not None:
        candidate_norms = np.array(
            [faiss_to_norm.get(int(fid), 0.5) for fid in candidate_ids],
            dtype=np.float32,
        )

    # 批量 Poincaré 测地线距离（向量化，< 1ms for 50 items）
    distances = batch_poincare_distance(
        query_vector, candidate_vectors,
        query_norm=query_norm,
        candidate_norms=candidate_norms,
    )

    # 按距离升序排序（小 = 更近）
    sorted_indices = np.argsort(distances)

    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]

    result_ids = candidate_ids[sorted_indices]
    result_dists = distances[sorted_indices]

    return result_ids, result_dists
