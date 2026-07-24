"""
traction_reranker.py — 牵引力重排序（v1）
=========================================

核心思路：有效距离 = Poincaré 距离 - α × 连接强度

在 diversity_sort + term_fusion 之后，对候选球体增加牵引力奖励：
  候选球体与 FAISS 种子球体之间有连接 → 得分提升（排序前移）

连接来源：
  - Sphere.connections（语义/时序/共现连接，权重 [0.1, 0.6]）
  - 可选：RoleTable 实体共现（间接连接，v2 扩展）

控制参数（alpha）：
  - α ∈ [0, 1]，默认 0.1
  - α=0 时无牵引力效果，退化为纯多样性排序
  - α 过大 → 排序坍塌（全被连接最强的种子球体吸引）
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── 牵引力配置 ──────────────────────────────

@dataclass
class TractionConfig:
    """牵引力参数"""
    alpha: float = 0.1        # 牵引力强度 [0, 1]
    seed_boost: float = 0.0   # 种子球体自身的额外奖励（0=不额外奖励种子）
    min_weight: float = 0.15  # 低于此权重的连接不参与牵引
    use_role_cooccur: bool = False  # v2: 启用角色共现间接连接


# ── 牵引力重排序器 ─────────────────────────

class TractionReranker:
    """牵引力重排序器

    用法：
      reranker = TractionReranker(sphere_store, alpha=0.1)
      boosted = reranker.rerank(sorted_results, seed_sphere_ids)
    """

    def __init__(
        self,
        connections_provider: Optional[Callable] = None,
        sphere_store=None,
        config: Optional[TractionConfig] = None,
    ):
        """
        Args:
            connections_provider: Callable(sphere_id) -> Dict[str, float]
            sphere_store: SphereStore（备选，当 connections_provider 不可用时直接读球体）
            config: 牵引力参数
        """
        self._conn_provider = connections_provider
        self._sphere_store = sphere_store
        self.config = config or TractionConfig()

    def attach(self, provider: Callable, sphere_store=None):
        """关联连接提供者和/或球体库"""
        if provider:
            self._conn_provider = provider
        if sphere_store:
            self._sphere_store = sphere_store

    def rerank(
        self,
        sorted_results: List[Tuple[str, float]],
        seed_sphere_ids: List[str],
    ) -> List[Tuple[str, float]]:
        """应用牵引力奖励（v2 公式：连接密度加法）

        公式演变：
          v1: boosted = score × (1 + α × max_conn_weight)
              → 问题：所有候选都有连接，乘法提升均匀，不产生排序变化
          v2: boosted = score + α × seed_conn_count × avg_base_score
              → 按连接密度差异产生区分度

        其中 seed_conn_count = 该候选连接到的种子球体数量（≥ min_weight）
             avg_base_score = 所有候选的平均分（保证加成与分数尺度匹配）

        Args:
            sorted_results: [(sphere_id, score), ...] 已排序的候选列表
            seed_sphere_ids: FAISS 粗搜命中的种子球体 ID（原始未展开）

        Returns:
            [(sphere_id, boosted_score), ...] 按牵引调整后降序排列
        """
        if not sorted_results or not seed_sphere_ids:
            return sorted_results

        alpha = self.config.alpha
        if alpha <= 0:
            return sorted_results

        min_weight = self.config.min_weight
        seed_set = set(seed_sphere_ids)

        # 计算基准尺度（候选的分数范围）
        scores = [sc for _, sc in sorted_results]
        score_range = max(scores) - min(scores) if max(scores) > min(scores) else 1.0

        # 计算每个候选连接到的种子数量
        candidate_scores = []
        for sphere_id, score in sorted_results:
            conns = self._get_connections(sphere_id)
            # 统计连接到多少种子的数量（加权求和，权重=连接强度）
            seed_conn_strength = 0.0
            for target_id, weight in conns.items():
                if target_id in seed_set and weight >= min_weight:
                    # 加权：连接越强越有价值
                    seed_conn_strength += weight

            # v2 公式：加法，按连接密度产生差异
            # seed_conn_strength 是加权和，范围 [0, seed_count * max_weight]
            # 归一化到 [0, 0.5] 范围
            conn_unit = min(0.5, seed_conn_strength / max(len(seed_set), 1))
            boosted = score + alpha * conn_unit * score_range

            # 对种子球体本身额外奖励（可选）
            if sphere_id in seed_set and self.config.seed_boost > 0:
                boosted += self.config.seed_boost * score_range

            candidate_scores.append((sphere_id, boosted))

        # 重排序
        candidate_scores.sort(key=lambda x: -x[1])

        if alpha > 0 and len(sorted_results) > 0:
            # 统计移动情况
            orig_order = {sid: i for i, (sid, _) in enumerate(sorted_results)}
            movements = []
            for i, (sid, _) in enumerate(candidate_scores):
                orig_pos = orig_order.get(sid, -1)
                if orig_pos >= 0:
                    movements.append(orig_pos - i)  # 正数=前移

            if movements:
                avg_move = sum(movements) / len(movements)
                moved_up = sum(1 for m in movements if m > 0)
                logger.debug(
                    f"Traction rerank: avg_move={avg_move:.1f}, "
                    f"moved_up={moved_up}/{len(movements)}, "
                    f"alpha={alpha}"
                )

        return candidate_scores

    def _get_connections(self, sphere_id: str) -> Dict[str, float]:
        """获取球体的连接表"""
        if self._conn_provider:
            return self._conn_provider(sphere_id) or {}
        if self._sphere_store:
            sphere = self._sphere_store.get(sphere_id)
            if sphere:
                return sphere.connections
        return {}
