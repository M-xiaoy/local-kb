"""
norm_deriver.py — Poincaré 范数推导器（v2，质量驱动）
====================================================
从球体质量推导在 Poincaré Ball 中的径向范数。

核心逻辑：
  mass_scale = ln(1 + mass)
  poincare_r = 0.90 - 0.80 × (mass_scale / max_mass_scale)

  质量越大 → 抽象层次越高 → 范数越小（靠球心）
  质量越小 → 具体事实 → 范数越大（靠球面）

相比 v1（社区覆盖率+层级+hub 三路信号）：
  ✓ 单一信号源，无冲突
  ✓ 质量天然反映抽象层次（子概念越多=越抽象）
  ✓ 无需额外的社区检测依赖
"""

import logging
from typing import Dict, Optional

import numpy as np

from storage.sphere_store import SphereStore

logger = logging.getLogger(__name__)

_NORM_MIN = 0.05
_NORM_MAX = 0.90


class NormDeriver:
    """从质量推导 Poincaré 范数

    Usage:
        deriver = NormDeriver(sphere_store)
        norms = deriver.derive_all(mass_map)
        # 写入 sphere_store: 每个 sphere.poincare_norm = norms[sphere.id]
    """

    def __init__(self, sphere_store: SphereStore):
        self._store = sphere_store

    def derive_all(self, mass_map: Dict[str, float]) -> Dict[str, float]:
        """全量推导活跃球体的 Poincaré 范数

        Args:
            mass_map: {sphere_id: mass} — 来自 MassAssigner.assign_all()

        Returns:
            {sphere_id: poincare_norm} — 范数在 [0.05, 0.90] 范围内
        """
        if not mass_map:
            logger.warning("NormDeriver: empty mass_map, returning defaults")
            return {}

        masses = [m for m in mass_map.values() if m > 0]
        if not masses:
            return {}

        # 对数压缩：幂律 → 线性
        mass_scales = {sid: np.log(1.0 + max(0.0, m))
                       for sid, m in mass_map.items()}
        max_scale = max(mass_scales.values())

        if max_scale <= 0:
            # 边缘情况：所有质量为 0
            return {sid: _NORM_MAX for sid in mass_map}

        # 线性映射到 [0.05, 0.90]
        norms: Dict[str, float] = {}
        for sid, scale in mass_scales.items():
            ratio = scale / max_scale          # [0, 1]
            norm = 0.90 - 0.80 * ratio          # [0.10, 0.90]
            norm = np.clip(norm, _NORM_MIN, _NORM_MAX)
            norms[sid] = float(norm)

        # 统计
        vals = list(norms.values())
        logger.info(
            f"NormDeriver: derived norms for {len(norms)} spheres, "
            f"mass range [{min(masses):.1f}, {max(masses):.1f}], "
            f"norm range [{min(vals):.3f}, {max(vals):.3f}]"
        )
        return norms

    def derive_one(self, sphere_id: str, mass: float,
                   max_scale: float) -> float:
        """为单个球体计算范数（用于增量更新时）

        Args:
            sphere_id: 球体 ID（仅用于日志）
            mass: 球体质量
            max_scale: 当前全局最大 ln(1+mass) 值

        Returns:
            Poincaré 范数
        """
        scale = np.log(1.0 + max(0.0, mass))
        if max_scale <= 0:
            return _NORM_MAX
        ratio = scale / max_scale
        norm = 0.90 - 0.80 * ratio
        return float(np.clip(norm, _NORM_MIN, _NORM_MAX))

    def get_projection_stats(self, norms: Dict[str, float]) -> dict:
        """范数分布统计"""
        if not norms:
            return {}
        vals = list(norms.values())
        return {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "near_core (<0.2)": sum(1 for v in vals if v < 0.2),
            "mid (0.2-0.7)": sum(1 for v in vals if 0.2 <= v <= 0.7),
            "near_surface (>0.7)": sum(1 for v in vals if v > 0.7),
        }


def derive_and_write(sphere_store, mass_map):
    """快捷函数：推导范数并写回球体"""
    deriver = NormDeriver(sphere_store)
    norms = deriver.derive_all(mass_map)

    for sid, norm in norms.items():
        sphere = sphere_store.get(sid)
        if sphere:
            sphere.poincare_norm = norm
            sphere.poincare_norm_source = "mass"

    sphere_store.save()
    return norms
