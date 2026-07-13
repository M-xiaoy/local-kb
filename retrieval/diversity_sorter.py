"""
diversity_sorter.py — 多样性排序器
==================================
从 FAISS Top-100 中重排出 Top-5，兼顾相关性和多样性。

算法架构（四层叠加）：

  1. MMR 基础层（行业标准）
     MMR(d) = λ · Sim(d, query) - (1-λ) · max(Sim(d, selected))
     λ=0.5 时均衡相关性与多样性，防止 Top-5 全是同一份文档的切片

  2. 来源多样性惩罚
     同一源文件的切片先选一个，再选第二个时受惩罚
     避免 Top-5 全部来自同一篇文档

  3. 场域亲和度加权
     利用 field_detector 的输出给匹配场域的切片加分
     实现重力空间中的场域偏好

  4. 重力冗余度惩罚（Gravity Redundancy）
     同一簇内与其他候选的平均相似度越高 → 信息冗余越大 → 惩罚越重
     稀有位置（远离簇中心、邻居稀疏）的球体被优先选中
"""

import logging
from typing import Dict, List, Optional, Set

import numpy as np

from config import retrieval as cfg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 多样性排序器
# ──────────────────────────────────────────────

class DiversitySorter:
    """三层多样性排序器

    输入：FAISS Top-100 的 (向量, ID, 来源, 场域)
    输出：Top-5 (ID, 得分) — 平衡相关性 + 多样性 + 场域偏好
    """

    def __init__(
        self,
        lambda_mmr: float = 0.5,
        source_penalty: float = 0.15,
        field_bonus_weight: float = 0.1,
        redundancy_penalty_weight: float = 0.05,
    ):
        """
        Args:
            lambda_mmr: MMR 平衡参数
                0.5 = 均衡  0.7 = 偏相关  0.3 = 偏多样
            source_penalty: 同源文件的额外惩罚（0~1）
                0.15 = 轻微惩罚  0.3 = 明显惩罚
            field_bonus_weight: 场域亲和度的权重
            redundancy_penalty_weight: 簇内冗余惩罚
                0.05 = 轻微惩罚  0.15 = 明显惩罚
        """
        self.lambda_mmr = lambda_mmr
        self.source_penalty = source_penalty
        self.field_bonus_weight = field_bonus_weight
        self.redundancy_penalty_weight = redundancy_penalty_weight

    def sort(
        self,
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        candidate_ids: List[str],
        source_files: List[str],
        source_types: Optional[List[str]] = None,
        field_affinities: Optional[Dict[str, float]] = None,
        top_k: int = 5,
    ) -> List[tuple]:
        """多样性排序主入口

        Args:
            query_vector: shape (dim,), float32, 已 L2 归一化
            candidate_vectors: shape (n, dim), float32, 已 L2 归一化
            candidate_ids: [sphere_id, ...] 与向量一一对应
            source_files: [文件名, ...] 来源文件
            source_types: [场域标签, ...] 或 None（不启用场域加权）
            field_affinities: {场域: 亲和度} 来自 field_detector.detect()
            top_k: 最终返回数量

        Returns:
            [(sphere_id, score), ...] 按得分降序，最多 top_k 条

        Raises:
            ValueError: 输入维度不匹配
        """
        n = len(candidate_ids)
        if n == 0:
            return []

        # 校验输入
        if candidate_vectors.shape[0] != n:
            raise ValueError(
                f"vectors ({candidate_vectors.shape[0]}) and ids ({n}) mismatch"
            )
        if source_files is not None and len(source_files) != n:
            raise ValueError(
                f"source_files ({len(source_files)}) and ids ({n}) mismatch"
            )
        if source_types is not None and len(source_types) != n:
            raise ValueError(
                f"source_types ({len(source_types)}) and ids ({n}) mismatch"
            )

        # 确保一维查询向量
        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()

        # 预计算查询相似度
        query_sims = candidate_vectors @ query_vector  # 归一化后 = cosine

        # 预计算候选向量之间的相似度矩阵（用于冗余度计算）
        sim_matrix = candidate_vectors @ candidate_vectors.T

        # 场域亲和度映射
        field_scores = field_affinities or {}

        # 如果没传入 source_types，所有候选标记为 None
        stypes = source_types if source_types else [None] * n
        sfiles = source_files if source_files else [None] * n

        # ── 预计算簇内冗余度（重力重排序）───────
        # 对每个候选，计算它跟同一簇内其他候选的平均相似度
        # 平均相似度高 = 信息冗余 = 被惩罚
        redundancy_scores = self._precompute_redundancy(
            n, sim_matrix, stypes
        )

        # 已选索引和结果
        selected_indices: List[int] = []
        final_results: List[tuple] = []

        for _ in range(min(top_k, n)):
            best_idx = self._select_next(
                n, selected_indices, query_sims, sim_matrix,
                sfiles, stypes, field_scores,
                redundancy_scores=redundancy_scores,
            )
            if best_idx is None:
                break

            # 计算最终得分（仅用于展示）
            score = self._final_score(
                best_idx, selected_indices, query_sims, sim_matrix,
                sfiles, stypes, field_scores,
                redundancy_scores=redundancy_scores,
            )

            selected_indices.append(best_idx)
            final_results.append((candidate_ids[best_idx], round(score, 4)))

        return final_results

    # ── 内部：选下一个 ────────────────────────

    def _select_next(
        self,
        n: int,
        selected: List[int],
        query_sims: np.ndarray,
        sim_matrix: np.ndarray,
        source_files: List,
        source_types: List,
        field_scores: Dict[str, float],
        redundancy_scores: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """从剩余候选中选得分最高的"""
        remaining = [i for i in range(n) if i not in selected]

        if not remaining:
            return None

        best_score = -float("inf")
        best_idx = None

        for idx in remaining:
            score = self._mmr_score(idx, selected, query_sims, sim_matrix)

            # 来源多样性惩罚
            if len(selected) > 0:
                score += self._source_penalty_score(idx, selected, source_files)

            # 场域亲和度加分
            if field_scores:
                score += self._field_bonus_score(idx, source_types, field_scores)

            # 重力冗余度惩罚（第四层）
            if redundancy_scores is not None and len(selected) > 0:
                score += self._gravity_redundancy_score(
                    idx, redundancy_scores
                )

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    # ── 三层评分 ─────────────────────────────

    def _mmr_score(
        self,
        idx: int,
        selected: List[int],
        query_sims: np.ndarray,
        sim_matrix: np.ndarray,
    ) -> float:
        """MMR 基础分"""
        relevance = query_sims[idx]

        if not selected:
            # 第一个选最相关的
            return relevance

        # 与已选结果的最大相似度（冗余度）
        redundancy = max(sim_matrix[idx, s] for s in selected)

        return (self.lambda_mmr * relevance
                - (1 - self.lambda_mmr) * redundancy)

    def _source_penalty_score(
        self,
        idx: int,
        selected: List[int],
        source_files: List,
    ) -> float:
        """来源多样性惩罚

        如果 idx 的源文件已经在 selected 中出现过，给予负分惩罚。
        每多一个同源，惩罚递增（log scale）。
        """
        current_file = source_files[idx]
        if current_file is None:
            return 0.0

        same_source_count = sum(
            1 for s in selected if source_files[s] == current_file
        )

        if same_source_count == 0:
            return 0.0  # 新来源，不惩罚

        # 首次同源：-0.15, 第二次同源：-0.15*1.5, ...
        return -self.source_penalty * (1.5 ** (same_source_count - 1))

    def _field_bonus_score(
        self,
        idx: int,
        source_types: List,
        field_scores: Dict[str, float],
    ) -> float:
        """场域亲和度加分

        利用 field_detector 的输出给匹配场域的切片加分。
        查询"什么是预测编码"→ field_detector 给"技术笔记"高分 0.85
        → 这个切片的 source_type="技术笔记" → 获得 0.1 × 0.85 = 0.085 加分
        """
        stype = source_types[idx]
        if stype is None or stype not in field_scores:
            return 0.0

        return self.field_bonus_weight * field_scores[stype]

    def _final_score(
        self,
        idx: int,
        selected: List[int],
        query_sims: np.ndarray,
        sim_matrix: np.ndarray,
        source_files: List,
        source_types: List,
        field_scores: Dict[str, float],
        redundancy_scores: Optional[np.ndarray] = None,
    ) -> float:
        """完整得分（仅用于展示）"""
        score = self._mmr_score(idx, selected, query_sims, sim_matrix)
        if selected:
            score += self._source_penalty_score(idx, selected, source_files)
        if field_scores:
            score += self._field_bonus_score(idx, source_types, field_scores)
        if redundancy_scores is not None and selected:
            score += self._gravity_redundancy_score(idx, redundancy_scores)
        return score

    # ── 第四层：重力冗余度 ─────────────────────

    def _precompute_redundancy(
        self,
        n: int,
        sim_matrix: np.ndarray,
        source_types: List,
    ) -> np.ndarray:
        """预计算每个候选在簇内的平均冗余度

        复用 sort() 中已算好的 sim_matrix，不重复计算。
        对每个候选，找出同簇内所有其他候选，计算平均余弦相似度。
        平均相似度高 = 该球体位于簇内的密集区域 = 信息冗余度高。

        Returns:
            shape (n,) float32 — 值域 [0, 1]
            0 = 该簇中唯一的球体/与其他球体正交
            1 = 与其他球体完全一致
        """
        if n <= 1:
            return np.zeros(n, dtype=np.float32)

        redundancy = np.zeros(n, dtype=np.float32)

        for i in range(n):
            same_cluster_sims = []
            for j in range(n):
                if (
                    i != j
                    and source_types[i] is not None
                    and source_types[i] == source_types[j]
                ):
                    same_cluster_sims.append(sim_matrix[i, j])

            if same_cluster_sims:
                redundancy[i] = float(np.mean(same_cluster_sims))

        return redundancy

    def _gravity_redundancy_score(
        self,
        idx: int,
        redundancy_scores: np.ndarray,
    ) -> float:
        """重力冗余度惩罚分

        簇内平均相似度越高 → 信息越冗余 → 负分越大
        Score = -redundancy_penalty_weight × avg_similarity
        """
        return -self.redundancy_penalty_weight * float(redundancy_scores[idx])
