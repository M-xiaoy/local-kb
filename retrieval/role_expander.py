"""
role_expander.py — 检索时角色扩展器
=====================================
在 FAISS 粗搜 → 多样性排序 之间插入的扩展阶段。

逻辑：
  1. 对每个 FAISS 命中的球体，查 RoleTable 的宾语→主语跳转
  2. 把跳转命中的新球体加入候选池（加大召回半径）
  3. 新球体的得分 = 源球体的 FAISS 得分 × 跳转置信度 × 衰减系数
  4. 保留源球体的层级标记（一级/二级/三级），排序器可见
"""

import logging
from typing import Dict, List, Optional, Tuple

from pipeline.role_table import RoleTable, JumpCandidate

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 角色扩展器
# ──────────────────────────────────────────────

class RoleExpander:
    """检索时角色扩展器

    不修改 RoleTable，只读查询，返回扩展候选。
    设计为可插拔——enable/disable 通过 config 控制。
    """

    def __init__(self, role_table: Optional[RoleTable] = None):
        self._table = role_table
        self._enabled = True

    def attach(self, role_table: RoleTable):
        """延迟关联角色表"""
        self._table = role_table

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    # ── 扩展入口 ─────────────────────────────

    def expand(
        self,
        faiss_hit_ids: List[str],
        faiss_hit_scores: Dict[str, float],
        max_expansions_per_hit: int = 3,
        total_max_expansions: int = 20,
        min_confidence: float = 0.3,
        decay: float = 0.6,
        exclude_ids: Optional[set] = None,
    ) -> Dict[str, float]:
        """扩展检索候选池

        Args:
            faiss_hit_ids: FAISS 命中的球体 ID 列表
            faiss_hit_scores: {sphere_id: faiss_score}
            max_expansions_per_hit: 每个命中球体最多扩展几个
            total_max_expansions: 总扩展数量上限
            min_confidence: 跳转最低置信度
            decay: 扩展得分衰减系数（0.6=跳转后得分打6折）
            exclude_ids: 要排除的球体 ID

        Returns:
            {expanded_sphere_id: combined_score}
            得分已考虑：源 FAISS 得分 × 置信度 × 衰减
        """
        if not self._enabled or not self._table:
            return {}

        excluded = exclude_ids or set()
        all_expanded: Dict[str, Tuple[float, int]] = {}  # id → (best_score, hop_count)

        for sid in faiss_hit_ids:
            if sid in excluded:
                continue

            source_score = faiss_hit_scores.get(sid, 0.5)

            jumps = self._table.expand_from_sphere(
                sid,
                max_candidates=max_expansions_per_hit,
                min_confidence=min_confidence,
            )

            for jc in jumps:
                if jc.target_sphere_id in excluded:
                    continue
                if jc.target_sphere_id in faiss_hit_ids:
                    # 已在 FAISS 结果中，跳过（不重复）
                    continue

                combined = source_score * jc.confidence * decay
                if jc.target_sphere_id in all_expanded:
                    if combined > all_expanded[jc.target_sphere_id][0]:
                        all_expanded[jc.target_sphere_id] = (combined, 1)
                else:
                    all_expanded[jc.target_sphere_id] = (combined, 1)

            if len(all_expanded) >= total_max_expansions:
                break

        # 截断到上限
        sorted_expanded = sorted(
            all_expanded.items(), key=lambda x: -x[1][0]
        )
        result = {
            sid: score
            for sid, (score, _) in sorted_expanded[:total_max_expansions]
        }

        if result:
            logger.info(
                f"Role expansion: {len(result)} candidates from "
                f"{len(faiss_hit_ids)} hits "
                f"(confidence≥{min_confidence}, decay={decay})"
            )

        return result

    # ── 扩展统计 ─────────────────────────────

    def expansion_stats(
        self,
        faiss_hit_ids: List[str],
    ) -> Dict:
        """预览扩展统计（不实际扩展，只报告数量级）"""
        if not self._table:
            return {"available": False}

        total_possible = 0
        hits_with_jumps = 0

        for sid in faiss_hit_ids:
            jumps = self._table.expand_from_sphere(sid, max_candidates=1)
            if jumps:
                hits_with_jumps += 1
                total_possible += len(jumps)

        return {
            "available": True,
            "hits_with_jumps": hits_with_jumps,
            "total_possible_jumps": total_possible,
            "total_hits": len(faiss_hit_ids),
        }
