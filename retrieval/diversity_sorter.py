"""
diversity_sorter.py — 多样性排序器（五层）
==========================================
从 FAISS Top-100 / 激活传播后的候选池中重排出 Top-K。

算法架构（五层叠加，从下到上）：

  1. MMR 基础层（行业标准）
     MMR(d) = lambda * Sim(d, query) - (1-lambda) * max(Sim(d, selected))

  2. 来源多样性惩罚
     同一源文件的切片先选一个，再选第二个时受惩罚

  3. 场域亲和度加权
     利用 field_detector 的输出给匹配场域的切片加分

  4. 重力冗余度惩罚
     同一簇内与其他候选的平均相似度越高 → 信息冗余越大 → 惩罚越重

  5. 连接密度惩罚（新增）
     与已选球体有强连接 → 信息冗余 → 惩罚
     实现"空间碰撞"效果
"""

import logging
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from config import retrieval as cfg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 多样性排序器
# ──────────────────────────────────────────────

class DiversitySorter:
    """五层多样性排序器

    输入：候选池 (向量, ID, 来源, 场域)
    输出：Top-K (ID, 得分) — 平衡相关性 + 多样性 + 场域 + 连接
    """

    def __init__(
        self,
        lambda_mmr: float = 0.5,
        source_penalty: float = 0.15,
        field_bonus_weight: float = 0.1,
        redundancy_penalty_weight: float = 0.05,
        connection_penalty_weight: float = 0.1,
    ):
        """
        Args:
            lambda_mmr: MMR 平衡参数
            source_penalty: 同源文件惩罚
            field_bonus_weight: 场域亲和度权重
            redundancy_penalty_weight: 簇内冗余惩罚
            connection_penalty_weight: 连接密度惩罚
        """
        self.lambda_mmr = lambda_mmr
        self.source_penalty = source_penalty
        self.field_bonus_weight = field_bonus_weight
        self.redundancy_penalty_weight = redundancy_penalty_weight
        self.connection_penalty_weight = connection_penalty_weight

    def sort(
        self,
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        candidate_ids: List[str],
        source_files: List[str],
        source_types: Optional[List[str]] = None,
        field_affinities: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        connections_provider: Optional[Callable] = None,
    ) -> List[tuple]:
        """多样性排序主入口

        Args:
            query_vector: shape (dim,), float32, 已 L2 归一化
            candidate_vectors: shape (n, dim), float32, 已 L2 归一化
            candidate_ids: [sphere_id, ...] 与向量一一对应
            source_files: [文件名, ...] 来源文件
            source_types: [场域标签, ...] 或 None
            field_affinities: {场域: 亲和度}
            top_k: 最终返回数量
            connections_provider: 函数(sphere_id) -> {neighbor_id: weight}
        """
        n = len(candidate_ids)
        if n == 0:
            return []

        # 校验
        if candidate_vectors.shape[0] != n:
            raise ValueError(
                f"vectors ({candidate_vectors.shape[0]}) and ids ({n}) mismatch"
            )
        if source_files is not None and len(source_files) != n:
            raise ValueError(
                f"source_files ({len(source_files)}) and ids ({n}) mismatch"
            )

        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()

        # 查询相似度
        query_sims = candidate_vectors @ query_vector

        # 候选间相似度矩阵
        sim_matrix = candidate_vectors @ candidate_vectors.T

        field_scores = field_affinities or {}
        stypes = source_types if source_types else [None] * n
        sfiles = source_files if source_files else [None] * n

        # 预计算冗余度
        redundancy_scores = self._precompute_redundancy(n, sim_matrix, stypes)

        selected_indices: List[int] = []
        final_results: List[tuple] = []

        for _ in range(min(top_k, n)):
            best_idx = self._select_next(
                n, selected_indices, query_sims, sim_matrix,
                sfiles, stypes, field_scores,
                redundancy_scores=redundancy_scores,
                candidate_ids=candidate_ids,
                conn_provider=connections_provider,
            )
            if best_idx is None:
                break

            score = self._final_score(
                best_idx, selected_indices, query_sims, sim_matrix,
                sfiles, stypes, field_scores,
                redundancy_scores=redundancy_scores,
                candidate_ids=candidate_ids,
                conn_provider=connections_provider,
            )

            selected_indices.append(best_idx)
            final_results.append((candidate_ids[best_idx], round(score, 4)))

        return final_results

    # ── 选择下一个 ────────────────────────────

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
        candidate_ids: Optional[List[str]] = None,
        conn_provider: Optional[Callable] = None,
    ) -> Optional[int]:
        remaining = [i for i in range(n) if i not in selected]
        if not remaining:
            return None

        best_score = -float("inf")
        best_idx = None

        for idx in remaining:
            score = self._mmr_score(idx, selected, query_sims, sim_matrix)

            if len(selected) > 0:
                score += self._source_penalty_score(idx, selected, source_files)

            if field_scores:
                score += self._field_bonus_score(idx, source_types, field_scores)

            if redundancy_scores is not None and len(selected) > 0:
                score += self._gravity_redundancy_score(idx, redundancy_scores)

            # 第五层：连接密度惩罚
            if conn_provider is not None and candidate_ids and len(selected) > 0:
                score += self._connection_density_penalty(
                    idx, selected, candidate_ids, conn_provider
                )

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    # ── 评分层 ───────────────────────────────

    def _mmr_score(self, idx, selected, query_sims, sim_matrix) -> float:
        relevance = query_sims[idx]
        if not selected:
            return relevance
        redundancy = max(sim_matrix[idx, s] for s in selected)
        return (self.lambda_mmr * relevance
                - (1 - self.lambda_mmr) * redundancy)

    def _source_penalty_score(self, idx, selected, source_files) -> float:
        current_file = source_files[idx]
        if current_file is None:
            return 0.0
        same_source_count = sum(
            1 for s in selected if source_files[s] == current_file
        )
        if same_source_count == 0:
            return 0.0
        return -self.source_penalty * (1.5 ** (same_source_count - 1))

    def _field_bonus_score(self, idx, source_types, field_scores) -> float:
        stype = source_types[idx]
        if stype is None or stype not in field_scores:
            return 0.0
        return self.field_bonus_weight * field_scores[stype]

    def _gravity_redundancy_score(self, idx, redundancy_scores) -> float:
        return -self.redundancy_penalty_weight * float(redundancy_scores[idx])

    def _connection_density_penalty(self, idx, selected, candidate_ids,
                                     conn_provider) -> float:
        """第五层：连接密度惩罚

        如果候选球体与已选球体有强连接，说明信息冗余，给予惩罚。
        连接权重越高，惩罚越大。实现"空间碰撞"效果。
        """
        if idx >= len(candidate_ids):
            return 0.0
        sphere_id = candidate_ids[idx]
        neighbors = conn_provider(sphere_id)
        if not neighbors:
            return 0.0
        penalty = 0.0
        for sel_idx in selected:
            if sel_idx >= len(candidate_ids):
                continue
            sel_id = candidate_ids[sel_idx]
            weight = neighbors.get(sel_id, 0.0)
            if weight >= 0.15:
                penalty -= weight * self.connection_penalty_weight
        return penalty

    def _final_score(self, idx, selected, query_sims, sim_matrix,
                      source_files, source_types, field_scores,
                      redundancy_scores=None,
                      candidate_ids=None, conn_provider=None) -> float:
        score = self._mmr_score(idx, selected, query_sims, sim_matrix)
        if selected:
            score += self._source_penalty_score(idx, selected, source_files)
        if field_scores:
            score += self._field_bonus_score(idx, source_types, field_scores)
        if redundancy_scores is not None and selected:
            score += self._gravity_redundancy_score(idx, redundancy_scores)
        if conn_provider is not None and candidate_ids and selected:
            score += self._connection_density_penalty(
                idx, selected, candidate_ids, conn_provider
            )
        return score

    # ── 冗余度 ───────────────────────────────

    def _precompute_redundancy(self, n, sim_matrix, source_types) -> np.ndarray:
        """预计算每个候选在簇内的平均冗余度"""
        if n <= 1:
            return np.zeros(n, dtype=np.float32)
        redundancy = np.zeros(n, dtype=np.float32)
        for i in range(n):
            same_cluster_sims = []
            for j in range(n):
                if (i != j and source_types[i] is not None
                        and source_types[i] == source_types[j]):
                    same_cluster_sims.append(sim_matrix[i, j])
            if same_cluster_sims:
                redundancy[i] = float(np.mean(same_cluster_sims))
        return redundancy
