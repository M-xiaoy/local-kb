"""
dual_space_rerank.py — 双空间混合重排（球面 + 欧氏）

距离公式：
  d(q, c) = α₁ · 2(1 - a_q·a_c)  +  α₂ · ||b_q - b_c||²

其中：
  a = normalize(f[:256])   → 球面子空间（聚类亲疏）
  b = f[256:]              → 欧氏子空间（语义远近）

α 基于诊断实验确定：
  球面：Cluster 1.048 vs Negative 1.235  → 区分度 0.19
  欧氏：Cluster 0.433 vs Negative 0.509  → 区分度 0.08
  → α₁ 应大于 α₂，建议 α₁=2.0, α₂=1.0
"""

import numpy as np
from typing import Optional

# Default coefficients
ALPHA_BALL_DEFAULT = 2.0
ALPHA_EUC_DEFAULT = 1.0


def dual_space_distance(q_vector: np.ndarray, c_vector: np.ndarray,
                        alpha_ball: float = ALPHA_BALL_DEFAULT,
                        alpha_euc: float = ALPHA_EUC_DEFAULT) -> float:
    """计算查询向量 q 和候选向量 c 之间的双空间距离。

    Args:
        q_vector: 查询嵌入（1024-d）
        c_vector: 候选嵌入（1024-d）
        alpha_ball: 球面子空间权重
        alpha_euc: 欧氏子空间权重

    Returns:
        距离标量（越小越相关）
    """
    # ── 球面分量：前 256 维，归一化后算弦距离 ──
    a_q = q_vector[:256] / (np.linalg.norm(q_vector[:256]) + 1e-8)
    a_c = c_vector[:256] / (np.linalg.norm(c_vector[:256]) + 1e-8)
    chordal = 2.0 * (1.0 - np.dot(a_q, a_c))

    # ── 欧氏分量：后 768 维，L2 平方 ──
    b_q = q_vector[256:]
    b_c = c_vector[256:]
    euclidean = float(np.sum((b_q - b_c) ** 2))

    return alpha_ball * chordal + alpha_euc * euclidean


def dual_space_rerank(query_vector: np.ndarray,
                      candidate_vectors: np.ndarray,
                      candidate_ids: list,
                      alpha_ball: float = ALPHA_BALL_DEFAULT,
                      alpha_euc: float = ALPHA_EUC_DEFAULT,
                      top_k: Optional[int] = None) -> list:
    """批量重排：对候选集按双空间距离重新排序。

    Args:
        query_vector: 查询嵌入（1024-d）
        candidate_vectors: N×1024 候选嵌入矩阵
        candidate_ids: 对应的 sphere_id 列表（长度 N）
        alpha_ball: 球面权重
        alpha_euc: 欧氏权重
        top_k: 返回前 k 个（None = 返回全部）

    Returns:
        [(sphere_id, distance), ...] 按距离升序
    """
    n = len(candidate_ids)
    distances = np.zeros(n, dtype=np.float64)

    for i in range(n):
        distances[i] = dual_space_distance(
            query_vector, candidate_vectors[i],
            alpha_ball, alpha_euc
        )

    # 按距离升序排列
    order = np.argsort(distances)
    results = [(candidate_ids[idx], float(distances[idx])) for idx in order]

    if top_k is not None and top_k < len(results):
        results = results[:top_k]

    return results
