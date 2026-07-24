"""
radius_deriver.py — Poincaré 范数推导器（v3，多信号融合）
============================================================
替代 pipeline/norm_deriver.py 的 mass-only 占位逻辑。

核心思想：
  范数 = 球体在 Poincaré Ball 中的径向距离（0=球心=最抽象，1=球面=最具体）

三路信号融合：
  1. hierarchy（层级）：level 字段
     - level=1（概念级）→ 0.15（近球心，抽象）
     - level=2（句子级）→ 0.50（中等）
     - level=3（子概念）→ 0.80（近球面，具体）
     - 无层级信息 → 0.50

  2. hubness（中心度）：连接图中的连接度
     - 高连接度 → 核心节点 → 小范数（近球心）
     - 低/零连接度 → 叶子节点 → 大范数（近球面）
     - 范围 [0, 1] → 映射到 [0.15, 0.80]

  3. density（语义密度）：嵌入空间中的局部密度
     - 高密度（邻居近）→ 处于共识区域 → 小范数
     - 低密度（邻居远）→ 边缘/独特点 → 大范数
     - 范围 [0, 1] → 映射到 [0.15, 0.80]

融合公式：
  norm = w_h × hierarchy + w_u × hubness + w_d × density
  默认权重: w_h=0.25, w_u=0.25, w_d=0.50 （density 权重最高，因为适用范围最广）

使用方式：
  deriver = RadiusDeriver(repo)
  norms = deriver.derive_all()         # 全量推导
  norms = deriver.derive_batch(ids)    # 批量（新球体）
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.repo.interfaces import KnowledgeBaseRepository, SphereData

logger = logging.getLogger(__name__)

# ── 默认权重 ─────────────────────────────────
_DEFAULT_WEIGHTS = {
    "hierarchy": 0.25,
    "hubness": 0.25,
    "density": 0.50,
}

# ── 层级范数映射 ────────────────────────────
_HIERARCHY_NORM = {1: 0.15, 2: 0.50, 3: 0.80}

_NORM_MIN = 0.05
_NORM_MAX = 0.90


class RadiusDeriver:
    """Poincaré 范数推导器（v3，多信号融合）"""

    def __init__(self, repo: KnowledgeBaseRepository,
                 weights: Optional[Dict[str, float]] = None):
        self._repo = repo
        self._w = weights or dict(_DEFAULT_WEIGHTS)

    # ── 全量推导 ─────────────────────────────

    def derive_all(self) -> Dict[str, float]:
        """为所有活跃球体推导 Poincaré 范数

        Returns:
            {sphere_id: poincare_norm}
        """
        spheres = self._repo.get_active()
        if not spheres:
            return {}

        sphere_ids = [s.id for s in spheres]

        # 三个信号
        h_signal = self._batch_hierarchy_signal(spheres)
        u_signal = self._batch_hubness_signal(spheres)
        d_signal = self._batch_density_signal(sphere_ids)

        norms = self._fuse(h_signal, u_signal, d_signal, sphere_ids)

        logger.info(
            f"RadiusDeriver: derived {len(norms)} norms "
            f"range=[{min(norms.values()):.3f}, {max(norms.values()):.3f}] "
            f"weights={self._w}"
        )
        return norms

    def derive_batch(self, sphere_ids: List[str]) -> Dict[str, float]:
        """为指定球体批量推导（用于 add_document 增量写入）

        Args:
            sphere_ids: 新球体的 ID 列表

        Returns:
            {sphere_id: poincare_norm}
        """
        if not sphere_ids:
            return {}

        spheres = [self._repo.get(sid) for sid in sphere_ids]
        spheres = [s for s in spheres if s is not None]

        if not spheres:
            return {}

        h_signal = self._batch_hierarchy_signal(spheres)
        u_signal = self._batch_hubness_signal(spheres)
        # batch 模式下 density 信号使用全局统计（只查这批球体在存量球体中的密度）
        d_signal = self._batch_density_signal(sphere_ids)

        norms = self._fuse(h_signal, u_signal, d_signal, sphere_ids)

        logger.debug(
            f"RadiusDeriver.derive_batch: {len(norms)} spheres "
            f"range=[{min(norms.values()):.3f}, {max(norms.values()):.3f}]"
        )
        return norms

    # ── 三路信号计算 ─────────────────────────

    def _batch_hierarchy_signal(
        self, spheres: List[SphereData]
    ) -> Dict[str, float]:
        """层级信号：从 level 字段映射为范数

        Returns:
            {sphere_id: norm_in_0_1}
        """
        signal = {}
        for s in spheres:
            base = _HIERARCHY_NORM.get(s.level, 0.50)
            # 有 parent_id 的二级球体，如果 parent 已存在则微调
            if s.level == 2 and s.parent_id:
                parent = self._repo.get(s.parent_id)
                if parent:
                    parent_norm = _HIERARCHY_NORM.get(parent.level, 0.50)
                    base = (base + parent_norm) / 2
            signal[s.id] = base
        return signal

    def _batch_hubness_signal(
        self, spheres: List[SphereData]
    ) -> Dict[str, float]:
        """中心度信号：从连接度映射为范数

        高连接度 = 核心节点 → 小范数（近球心）
        零连接度 = 叶子节点 → 大范数（近球面）
        """
        degrees = {s.id: self._repo.degree(s.id) for s in spheres}
        max_deg = max(degrees.values()) if degrees else 1

        signal = {}
        for sid, deg in degrees.items():
            if max_deg > 0:
                norm = 0.80 - 0.65 * (deg / max_deg)
            else:
                norm = 0.80  # 无连接 → 近球面
            signal[sid] = max(0.15, min(0.80, norm))

        return signal

    def _batch_density_signal(
        self, sphere_ids: List[str]
    ) -> Dict[str, float]:
        """语义密度信号：从嵌入空间局部密度映射为范数

        对每个球体，在 FAISS 中找 Top-K 近邻，计算平均距离。
        距离大 = 密度低 = 边缘信息 = 大范数（近球面）
        距离小 = 密度高 = 共识信息 = 小范数（近球心）

        降级策略：当 FAISS 不可用时返回 0.5 中位数。
        """
        k = min(20, self._repo.count())
        if k < 2:
            return {sid: 0.50 for sid in sphere_ids}

        signal = {}
        for sid in sphere_ids:
            vec = self._repo.get_vector(sid)
            if vec is None:
                signal[sid] = 0.50
                continue

            try:
                result = self._repo.search(vec, top_k=k)
                if not result.distances:
                    signal[sid] = 0.50
                    continue

                # 排除自身（最大距离的那个点可能是自身）
                distances = result.distances[:]
                if len(distances) > 1:
                    # 移除最大距离（极可能是自己）
                    distances = sorted(distances)[:k-1]

                avg_dist = float(np.mean(distances)) if distances else 0.5
                # IP 距离映射: [-1, 1] → [0.15, 0.80]
                # 高 IP（相近）→ 密度高 → 小范数
                # 低 IP（相远）→ 密度低 → 大范数
                # IP 通常范围 [0.2, 0.8]
                norm = 0.80 - 0.65 * max(0.0, min(1.0, (avg_dist + 1.0) / 2.0))
                signal[sid] = max(0.15, min(0.80, norm))

            except Exception as e:
                logger.debug(f"Density signal failed for {sid[:8]}: {e}")
                signal[sid] = 0.50

        return signal

    # ── 信号融合 ─────────────────────────────

    def _fuse(
        self,
        h_signal: Dict[str, float],
        u_signal: Dict[str, float],
        d_signal: Dict[str, float],
        sphere_ids: List[str],
    ) -> Dict[str, float]:
        """加权融合三路信号

        Args:
            h_signal: hierarchy 信号 {id: norm}
            u_signal: hubness 信号 {id: norm}
            d_signal: density 信号 {id: norm}

        Returns:
            {id: fused_norm} — 映射到 [_NORM_MIN, _NORM_MAX]
        """
        w_h = self._w.get("hierarchy", 0.25)
        w_u = self._w.get("hubness", 0.25)
        w_d = self._w.get("density", 0.50)

        norms = {}
        for sid in sphere_ids:
            h = h_signal.get(sid, 0.50)
            u = u_signal.get(sid, 0.50)
            d = d_signal.get(sid, 0.50)

            fused = w_h * h + w_u * u + w_d * d
            fused = max(_NORM_MIN, min(_NORM_MAX, fused))
            norms[sid] = float(round(fused, 4))

        return norms


# ──────────────────────────────────────────────
# 快捷函数
# ──────────────────────────────────────────────

def derive_and_write(repo: KnowledgeBaseRepository,
                     sphere_ids: Optional[List[str]] = None):
    """推导范数并写回球体

    Args:
        repo: 存储接口
        sphere_ids: 指定球体（None=全部活跃球体）

    Returns:
        {sphere_id: poincare_norm}
    """
    deriver = RadiusDeriver(repo)

    if sphere_ids is not None:
        norms = deriver.derive_batch(sphere_ids)
    else:
        norms = deriver.derive_all()

    for sid, norm in norms.items():
        repo.set_poincare_norm(sid, norm, "radius_deriver_v3")

    logger.info(
        f"derive_and_write: wrote {len(norms)} norms, "
        f"source=radius_deriver_v3"
    )
    return norms
