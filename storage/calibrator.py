"""
calibrator.py — 球体质量与多样性校准器
=========================================
让 mass 和 diversity 从默认占位值变成有意义的球体属性。

mass（质量）—— 球体在空间中的"重要性"
  - 基础值: 1.0
  - 连接度增益: 连接越多的球体越"重"
  - 高 mass → 在检索中权重越大

diversity（多样性）—— 球体在簇内的"独特性"
  - 基础值: 0.0
  - 离簇中心越远 → diversity 越高
  - 高 diversity → 在多样性排序中获得优势

effective_mass = mass × (1 + diversity_effective_factor × diversity)
  - 高 mass + 高 diversity = 既是重要节点，又提供独特信息
  - 高 mass + 低 diversity = 核心节点（簇内典型代表）
  - 低 mass + 高 diversity = 边缘节点（可能连接不同簇）

使用方式：
  calibrator = SphereCalibrator(sphere_store, vectors_cache)
  calibrator.calibrate_all()
  sphere_store.save()  # 持久化
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from config import calibrator as cfg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# SphereCalibrator
# ──────────────────────────────────────────────

class SphereCalibrator:
    """球体质量与多样性校准器"""

    def __init__(self, sphere_store=None, vectors_cache=None):
        """
        Args:
            sphere_store: SphereStore 实例
            vectors_cache: {sphere_id: np.ndarray} 向量缓存
        """
        self._store = sphere_store
        self._vectors = vectors_cache or {}

    def attach(self, sphere_store, vectors_cache):
        """延迟关联存储"""
        self._store = sphere_store
        self._vectors = vectors_cache

    # ── 全量校准 ─────────────────────────────

    def calibrate_all(self) -> dict:
        """对 store 中所有活跃球体做 mass + diversity 校准

        Returns:
            {"calibrated": int, "mass_range": (float, float),
             "diversity_range": (float, float)}
        """
        if self._store is None:
            raise RuntimeError("SphereCalibrator not attached to a store")

        spheres = self._store.get_active()
        if not spheres:
            logger.warning("No active spheres to calibrate")
            return {"calibrated": 0}

        # 先统计连接度
        degrees = self._compute_degrees(spheres)

        # 按簇分组向量
        cluster_vectors = self._group_by_cluster(spheres)

        # 计算每个簇的质心
        centroids = self._compute_centroids(cluster_vectors)

        # 更新每个球体的 mass 和 diversity
        updated_mass = 0
        updated_div = 0
        avg_degree = np.mean(list(degrees.values())) if degrees else 1.0

        for sphere in spheres:
            # mass = 基础值 + 连接度增益
            deg = degrees.get(sphere.id, 0)
            new_mass = self._calibrate_mass(deg, avg_degree)
            if abs(new_mass - sphere.mass) > 0.01:
                sphere.mass = new_mass
                updated_mass += 1

            # diversity = 到簇质心的语义距离
            vec = self._vectors.get(sphere.id)
            centroid = centroids.get(sphere.cluster_id)
            new_div = self._calibrate_diversity(vec, centroid)
            if abs(new_div - sphere.diversity) > 0.01:
                sphere.diversity = new_div
                updated_div += 1

            # 同步 effective_mass
            sphere._sync_effective_mass()

        masses = [s.mass for s in spheres]
        diversities = [s.diversity for s in spheres]

        result = {
            "calibrated": len(spheres),
            "mass_updated": updated_mass,
            "diversity_updated": updated_div,
            "mass_range": (round(min(masses), 4), round(max(masses), 4)),
            "diversity_range": (round(min(diversities), 4),
                                round(max(diversities), 4)),
            "avg_degree": round(avg_degree, 2),
        }

        logger.info(
            f"Calibrated {len(spheres)} spheres: "
            f"mass [{result['mass_range'][0]:.3f}, {result['mass_range'][1]:.3f}], "
            f"diversity [{result['diversity_range'][0]:.3f}, {result['diversity_range'][1]:.3f}]"
        )
        return result

    def calibrate_single(self, sphere_id: str) -> bool:
        """校准单个球体（增量添加后使用）"""
        sphere = self._store.get(sphere_id)
        if not sphere or not sphere.active:
            return False

        spheres = self._store.get_active()
        degrees = self._compute_degrees(spheres)
        avg_degree = np.mean(list(degrees.values())) if degrees else 1.0

        sphere.mass = self._calibrate_mass(degrees.get(sphere_id, 0), avg_degree)

        cluster_vectors = self._group_by_cluster(spheres)
        centroids = self._compute_centroids(cluster_vectors)
        vec = self._vectors.get(sphere_id)
        centroid = centroids.get(sphere.cluster_id)
        sphere.diversity = self._calibrate_diversity(vec, centroid)

        sphere._sync_effective_mass()
        return True

    # ── mass 校准 ────────────────────────────

    def _calibrate_mass(self, degree: int, avg_degree: float) -> float:
        """质量＝基础值 + 连接度增益

        公式:
          mass = mass_base + mass_connection_factor * clamp(degree/avg_degree, 0, max_mul)
        """
        ratio = degree / max(avg_degree, 1.0)
        ratio = min(ratio, cfg.mass_max_multiplier)
        return round(cfg.mass_base + cfg.mass_connection_factor * ratio, 4)

    # ── diversity 校准 ───────────────────────

    def _calibrate_diversity(self, vector: Optional[np.ndarray],
                              centroid: Optional[np.ndarray]) -> float:
        """多样性＝到簇质心的语义距离

        公式:
          diversity = 1.0 - cosine(vector, centroid)
          值域 [0, 1]

        如果没有 vector 或 centroid，返回 0.0
        """
        if vector is None or centroid is None:
            return 0.0

        vec = vector.flatten() if vector.ndim > 1 else vector
        cent = centroid.flatten() if centroid.ndim > 1 else centroid

        # 余弦相似度
        norm_v = np.linalg.norm(vec)
        norm_c = np.linalg.norm(cent)
        if norm_v == 0 or norm_c == 0:
            return 0.0

        cosine = float(np.dot(vec, cent) / (norm_v * norm_c))
        # 映射到 [0, 1]，clamp 防止数值误差
        raw = 1.0 - max(0.0, min(1.0, (cosine + 1.0) / 2.0))
        return round(raw, 4)

    # ── 辅助 ─────────────────────────────────

    def _compute_degrees(self, spheres) -> Dict[str, int]:
        """计算每个球体的连接度"""
        degrees = {}
        for s in spheres:
            deg = len(s.connections) if s.connections else 0
            degrees[s.id] = deg
        return degrees

    def _group_by_cluster(self, spheres) -> Dict[int, List[np.ndarray]]:
        """按簇分组向量"""
        groups: Dict[int, List[np.ndarray]] = {}
        for s in spheres:
            vec = self._vectors.get(s.id)
            if vec is None:
                continue
            cid = s.cluster_id if s.cluster_id >= 0 else -1
            groups.setdefault(cid, []).append(vec)
        return groups

    def _compute_centroids(self, groups: Dict[int, List[np.ndarray]]) -> Dict[int, np.ndarray]:
        """计算每个簇的质心"""
        centroids = {}
        for cid, vecs in groups.items():
            if not vecs:
                continue
            centroid = np.mean(vecs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[cid] = centroid
        return centroids
