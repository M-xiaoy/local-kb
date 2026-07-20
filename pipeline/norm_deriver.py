"""
norm_deriver.py — Poincaré 范数推导器
=======================================
从社区覆盖率 + 层次结构 + 连接数 推导每个球体在 Poincaré Ball 中的范数。

核心逻辑：
  coverage ↑ → 越抽象 → 越靠近球心 → 范数 ↓
  level   ↓ → 越具体 → 越靠近球面 → 范数 ↑

三路信号融合（0-1 归一化后加权）：
  norm = 1 - w_coverage × coverage_score
                 + w_level    × level_score
                 + w_hub      × hub_score
                 )

范数被 clamp 到 [0.05, 0.95] 保证数值安全。
"""

import logging
from typing import Dict, Optional

import numpy as np

from storage.sphere_store import SphereStore
from pipeline.role_table import RoleTable

logger = logging.getLogger(__name__)

# 默认权重：覆盖率最重要，层次次之，hub 度最弱
_DEFAULT_WEIGHTS = {
    "coverage": 0.50,   # 社区内实体出现占比
    "level": 0.35,      # 球体在层次结构中的位置
    "hub": 0.15,        # 被 connections 数（hub vs leaf）
}

_NORM_MIN = 0.05
_NORM_MAX = 0.95


class NormDeriver:
    """从结构推导 Poincaré Ball 范数

    Usage:
        deriver = NormDeriver(sphere_store, role_table)
        norms = deriver.derive_all()
        # 写入 sphere_store: 每个 sphere.poincare_norm = norms[sphere.id]
    """

    def __init__(
        self,
        sphere_store: SphereStore,
        role_table: Optional[RoleTable] = None,
        community_map: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self._store = sphere_store
        self._table = role_table
        self._community = community_map  # {sphere_id: community_id}
        self._weights = weights or _DEFAULT_WEIGHTS

    def attach_community(self, community_map: Dict[str, int]):
        """附加社区检测结果（可选，如果不传则缺省走 level 决定范数）"""
        self._community = community_map

    def derive_one(self, sphere_id: str) -> float:
        """为单个球体计算 Poincaré 范数

        只用 sphere_store 中已有信息，不依赖社区检测。
        社区因子在 derive_all() 中额外计算。
        """
        sphere = self._store.get(sphere_id)
        if not sphere or not sphere.active:
            return _NORM_MAX  # 不活跃球体推到边缘

        # --- level 因子：level 越高（越具体）范数越大 ---
        level_factor = self._level_factor(sphere.level)
        # 1→0.3, 2→0.5, 3→0.7, 默认→0.5
        level_to_norm = {1: 0.3, 2: 0.5, 3: 0.7}
        level_norm = level_to_norm.get(sphere.level, 0.5)

        # --- hub 因子：连接越多（hub 节点）范数越小 ---
        conn_count = len(sphere.connections)
        hub_norm = max(0.0, 1.0 - conn_count / max(50, conn_count))

        # 融合（此时不用 coverage，因为单个球体不知道社区信息）
        norm = (
            self._weights["level"] * level_norm +
            self._weights["hub"] * (0.2 + 0.6 * hub_norm) +
            self._weights["coverage"] * 0.5  # 无社区信息，取中值
        )

        return float(np.clip(norm, _NORM_MIN, _NORM_MAX))

    def derive_all(self) -> Dict[str, float]:
        """全量推导所有活跃球体的范数

        如果有 community_map，用社区内覆盖率计算；
        否则回退到 level + hub 因子。
        """
        norms = {}
        spheres = self._store.get_active()

        # 预计算社区覆盖率（如果有 community_map）
        coverage_map = self._compute_coverage() if self._community else {}

        for sphere in spheres:
            sid = sphere.id

            # 1. Level 因子
            level_to_norm = {1: 0.25, 2: 0.50, 3: 0.75}
            level_norm = level_to_norm.get(sphere.level, 0.50)

            # 2. Hub 因子
            conn_count = len(sphere.connections)
            hub_score = 1.0 - min(1.0, conn_count / 50.0)
            hub_norm = 0.2 + 0.6 * hub_score

            # 3. Coverage 因子（核心！）
            coverage_score = coverage_map.get(sid, 0.5)
            # coverage 越高（常见实体），范数越小（靠球心）
            coverage_norm = 0.9 - 0.7 * coverage_score

            # 融合
            norm = (
                self._weights["coverage"] * coverage_norm +
                self._weights["level"] * level_norm +
                self._weights["hub"] * hub_norm
            )

            norms[sid] = float(np.clip(norm, _NORM_MIN, _NORM_MAX))

        logger.info(
            f"NormDeriver: computed norms for {len(norms)} spheres "
            f"(coverage={'loaded' if coverage_map else 'none'})"
        )
        return norms

    def _compute_coverage(self) -> Dict[str, float]:
        """为每个球体计算「社区内实体覆盖率」

        对社区 C 中的球体 s：
          覆盖率 = s 中含有的、在社区 C 中出现频次最高的实体的频次
                  / 社区 C 的总球体数

        覆盖率 ≈ 该球体在多大程度上「代表」该社区的核心实体。
        覆盖率越高 → 该球体越接近社区「概念」→ 范数越小。
        """
        if not self._community or not self._table:
            return {}

        # 按社区分组
        community_spheres: Dict[int, list] = {}
        for sid, cid in self._community.items():
            community_spheres.setdefault(cid, []).append(sid)

        # 对每个社区，统计各实体的覆盖率
        coverage: Dict[str, float] = {}

        for cid, members in community_spheres.items():
            community_size = len(members)
            if community_size < 2:
                for sid in members:
                    coverage[sid] = 0.5
                continue

            # 统计社区内实体的出现次数
            entity_count = {}
            for sid in members:
                entities = self._table._sphere_entities.get(sid, set())
                for eid in entities:
                    entity_count[eid] = entity_count.get(eid, 0) + 1

            if not entity_count:
                for sid in members:
                    coverage[sid] = 0.5
                continue

            # 每个实体的覆盖率
            max_count = max(entity_count.values())
            entity_coverage = {
                eid: cnt / community_size
                for eid, cnt in entity_count.items()
            }

            # 对社区内每个球体，取它含有的实体的最高覆盖率
            for sid in members:
                entities = self._table._sphere_entities.get(sid, set())
                if not entities:
                    coverage[sid] = 0.3  # 不含实体的球体，默认低覆盖率
                    continue
                max_cov = max(
                    entity_coverage.get(eid, 0.0) for eid in entities
                )
                coverage[sid] = float(max_cov)

        return coverage

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
            "per_level": {
                str(level): float(np.mean([
                    norms[sid] for sid in norms
                    if self._store.get(sid) and self._store.get(sid).level == level
                ] or [0]))
                for level in [1, 2, 3]
            },
        }

    # ── 辅助 ─────────────────────────────────

    @staticmethod
    def _level_factor(level: int) -> float:
        """Level → [0, 1] 归一化（level 1=0.0, level 3=1.0）"""
        mapping = {1: 0.0, 2: 0.5, 3: 1.0}
        return mapping.get(level, 0.5)

    @staticmethod
    def _norm_for_level(level: int) -> float:
        """Level → 基准范数"""
        mapping = {1: 0.30, 2: 0.55, 3: 0.80}
        return mapping.get(level, 0.55)


def derive_and_write(sphere_store, role_table=None, community_map=None):
    """快捷函数：推导范数并写回球体"""
    deriver = NormDeriver(sphere_store, role_table, community_map=community_map)
    norms = deriver.derive_all()

    for sid, norm in norms.items():
        sphere = sphere_store.get(sid)
        if sphere:
            sphere.poincare_norm = norm
            sphere.poincare_norm_source = "community"

    sphere_store.save()
    return norms
