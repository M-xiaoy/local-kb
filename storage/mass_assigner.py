"""
mass_assigner.py — 球体质量递归赋值器（v2：hierarchy_mass × TF-IDF）
=====================================================================
质量定义 = hierarchy_mass × (1 + tfidf_factor)

  hierarchy_mass = 1（自身基准）+ Σ child.mass（递归子概念广度）
  tfidf_factor   = 球体内词语的内容密度（归一化到 [0, 1]）

叶子球体不再一律 mass=1。内容密度高的叶子（如包含多个关键词的解释性句子）
比内容稀疏的句子（如"好的"、"知道了"）质量更高 → 范数更小 → 更靠近球心。

TF-IDF 因子计算：
  1. 从每个球体的 term_weights 取 TF 加权和
  2. 取全局中位数归一化到 [0, 1]

质量 → Poincaré 范数的映射（在 NormDeriver 中完成）：
  mass_scale = ln(1 + mass)
  norm = 0.90 - 0.80 × (mass_scale / max_mass_scale)
"""

import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MassAssigner:
    """球体质量赋值器（v2）

    Usage:
        assigner = MassAssigner(sphere_store)
        mass_map = assigner.assign_all()
        # mass_map: {sphere_id: mass_value}
    """

    def __init__(self, sphere_store):
        self._store = sphere_store
        self._cache: Dict[str, float] = {}  # 递归缓存

    def assign_all(self) -> Dict[str, float]:
        """全量递归赋值

        Returns:
            {sphere_id: mass}
        """
        self._cache = {}
        active = self._store.get_active()

        # Step 1: 计算 TF-IDF 因子（所有叶子的内容密度）
        tfidf_map = self._compute_tfidf_factors(active)

        # Step 2: 深度优先递归 hierarchy_mass
        for sphere in active:
            self._compute_hierarchy_mass(sphere.id)

        # Step 3: 融合 hierarchy_mass × (1 + tfidf_factor)
        final_masses = {}
        for sphere in active:
            sid = sphere.id
            hier_mass = self._cache.get(sid, 1.0)
            tfidf = tfidf_map.get(sid, 0.0)
            # 概念球体（有子节点）用纯 hierarchy_mass，避免双倍放大
            if sphere.child_ids:
                final_mass = hier_mass
            else:
                final_mass = hier_mass * (1.0 + tfidf)
            final_masses[sid] = final_mass

        # Step 4: 写回 sphere_store
        for sid, mass in final_masses.items():
            sphere = self._store.get(sid)
            if sphere:
                sphere.mass = mass
                sphere._sync_effective_mass()

        # 统计
        masses = list(final_masses.values())
        if masses:
            hier_only = list(self._cache.values())
            logger.info(
                f"MassAssigner: assigned {len(masses)} spheres, "
                f"range [{min(masses):.2f}, {max(masses):.1f}], "
                f"mean={sum(masses)/len(masses):.2f}, "
                f"(hierarchy range [{min(hier_only):.1f}, {max(hier_only):.1f}])"
            )

            # TF-IDF 效果简报
            tfidf_vals = list(tfidf_map.values())
            leaf_before = sum(1 for m in hier_only if abs(m - 1.0) < 0.01)
            leaf_after = sum(1 for m in masses if abs(m - 1.0) < 0.01)
            logger.info(
                f"  TF-IDF: {sum(1 for v in tfidf_vals if v > 0)}/{len(active)} non-zero, "
                f"max={max(tfidf_vals):.3f}, "
                f"leaves: {leaf_before} -> {leaf_after} (pure hierarchy mass=1)"
            )
        else:
            logger.info("MassAssigner: no spheres to assign")

        return final_masses

    def _compute_hierarchy_mass(self, sphere_id: str) -> float:
        """递归计算 hierarchy_mass（带缓存）

        hierarchy_mass = 1（自身基准）+ Σ child.mass
        """
        if sphere_id in self._cache:
            return self._cache[sphere_id]

        sphere = self._store.get(sphere_id)
        if not sphere or not sphere.active:
            self._cache[sphere_id] = 0.0
            return 0.0

        children_mass = 0.0
        for child_id in sphere.child_ids:
            children_mass += self._compute_hierarchy_mass(child_id)

        total = 1.0 + children_mass
        self._cache[sphere_id] = total
        return total

    def _compute_tfidf_factors(self, spheres) -> Dict[str, float]:
        """计算每个球体的 TF-IDF 因子

        从 term_weights（TF 值）中计算内容密度分数。

        Args:
            spheres: Sphere 列表

        Returns:
            {sphere_id: tfidf_factor} 归一化到 [0, 1]
        """
        # 计算每个球体的 TF 加权和
        raw_scores = {}
        for s in spheres:
            tw = s.term_weights
            if tw:
                raw_scores[s.id] = sum(tw.values())
            else:
                raw_scores[s.id] = 0.0

        # 全局中位数归一化
        all_scores = list(raw_scores.values())
        if not all_scores:
            return {}

        median = float(np.median(all_scores))
        if median <= 0:
            # 中位数为 0，过半球体无 TF → TF-IDF 退化为 0
            return {sid: 0.0 for sid in raw_scores}

        # 归一化到 [0, 1]，使用线性缩放+截断
        max_score = max(all_scores)
        if max_score <= median:
            return {sid: 0.0 for sid in raw_scores}

        factors = {}
        for sid, score in raw_scores.items():
            norm = min(1.0, score / (median * 2))  # 2倍中位数 → 1.0
            factors[sid] = float(norm)

        return factors

    def get_stats(self, mass_map: Dict[str, float]) -> dict:
        """质量分布统计"""
        if not mass_map:
            return {}
        vals = list(mass_map.values())
        return {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "leaves (=1.0)": sum(1 for v in vals if abs(v - 1.0) < 0.01),
            "multi (>1.0)": sum(1 for v in vals if v > 1.0),
            ">2.0": sum(1 for v in vals if v > 2.0),
        }
