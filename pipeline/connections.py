"""
connections.py — 球体连接检测器
================================
新球体入库时自动发现与已有球体的关系，建立连接网络。

连接类型（5种，按权重降序）：
  1. 同簇语义强连接 (0.6) — 同簇内 Top-3 最相似的球体
  2. 实体重叠连接 (0.4) — 共享 ≥N 个实体的球体
  3. 跨簇近邻连接 (0.3) — 不同簇但 embedding 余弦 > 阈值的球体
  4. 时序相邻连接 (0.25) — 同一源文件相邻切片的球体
  5. 跨簇文献连接 (0.2) — 不同簇但文件名相同（跨簇的同一文档）

连接网络特性：
  - 双向：A→B 和 B→A 同时建立
  - 带权：weight ∈ [0.05, 0.6]
  - 稀疏：每个球体最多 max_connections_per_sphere 个连接
  - 可衰减：decay_per_tick 定期降低权重

使用方式：
  detector = ConnectionDetector(sphere_store, vectors_cache)
  detector.detect_for_new(new_sphere_id, new_vector, entities, source_file, cluster_id)
  detector.detect_batch(all_sphere_ids)  # 首次全量构建
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import connection as cfg

logger = logging.getLogger(__name__)

CONNECTIONS_VERSION = 1


# ──────────────────────────────────────────────
# 连接检测器
# ──────────────────────────────────────────────

class ConnectionDetector:
    """球体连接检测器"""

    def __init__(self, sphere_store=None, vectors_cache=None,
                 store_path: Optional[str] = None):
        """
        Args:
            sphere_store: SphereStore 实例
            vectors_cache: {sphere_id: np.ndarray} 向量缓存
            store_path: 连接表持久化路径
        """
        self._store = sphere_store
        self._vectors = vectors_cache or {}
        self._store_path = Path(store_path or cfg.storage_dir)
        self._connections: Dict[str, Dict[str, float]] = {}
        self._dirty = False

    # ── 关联存储 ─────────────────────────────

    def attach(self, sphere_store, vectors_cache):
        """延迟关联存储"""
        self._store = sphere_store
        self._vectors = vectors_cache

    # ── 为新球体检测连接 ─────────────────────

    def detect_for_new(self, sphere_id: str, vector: np.ndarray,
                        entities: List[str], source_file: str,
                        cluster_id: int) -> Dict[str, float]:
        """新球体入库后，自动发现与已有球体的连接

        不需要全量扫描：用 embedding + 元数据 快速定位候选。

        Args:
            sphere_id: 新球体 ID
            vector: 新球体的向量
            entities: 新球体的实体列表
            source_file: 新球体的源文件
            cluster_id: 新球体的簇 ID

        Returns:
            {target_id: weight, ...} — 新球体连接表
        """
        if self._store is None:
            return {}

        connections: Dict[str, float] = {}
        entities_set = set(entities)
        existing = self._store.get_active()

        # 分批处理已有球体
        for i in range(0, len(existing), cfg.batch_build_size):
            batch = existing[i:i + cfg.batch_build_size]
            for sphere in batch:
                if sphere.id == sphere_id:
                    continue
                weight = self._compute_weight(
                    new_id=sphere_id, new_vec=vector,
                    new_entities=entities_set, new_file=source_file,
                    new_cluster=cluster_id,
                    target=sphere,
                )
                if weight >= cfg.min_weight:
                    connections[sphere.id] = weight

        # 排序取 Top-K
        sorted_conn = sorted(connections.items(), key=lambda x: -x[1])
        sorted_conn = sorted_conn[:cfg.max_connections_per_sphere]
        connections = dict(sorted_conn)

        # 双向保存
        for target_id, weight in connections.items():
            self._add_connection(sphere_id, target_id, weight)

        logger.debug(
            f"New sphere {sphere_id[:8]} got {len(connections)} "
            f"connections (cluster={cluster_id})"
        )
        return connections

    # ── 全量构建 ─────────────────────────────

    def detect_batch(self, sphere_ids: Optional[List[str]] = None) -> int:
        """全量重建所有球体的连接（首次迁移用）

        Args:
            sphere_ids: 要处理的球体 ID 列表，None = 处理所有活跃球体

        Returns:
            创建的连接总数
        """
        if self._store is None:
            raise RuntimeError("ConnectionDetector not attached to a store")

        spheres = self._store.get_active()
        if sphere_ids:
            spheres = [s for s in spheres if s.id in sphere_ids]

        if not spheres:
            return 0

        # 清理旧连接
        self._connections.clear()

        total_connections = 0
        n = len(spheres)

        # 按簇分组
        clusters: Dict[int, List] = {}
        for s in spheres:
            clusters.setdefault(s.cluster_id, []).append(s)

        logger.info(f"Building connections for {n} spheres in {len(clusters)} clusters")

        # 簇内连接（每个球体找同簇 Top-K）
        for cid, group in clusters.items():
            group.sort(key=lambda s: s.id)
            for i, sphere in enumerate(group):
                vec_i = self._vectors.get(sphere.id)
                if vec_i is None:
                    continue
                # 计算与本簇其他球体的相似度
                similarities = []
                for j, other in enumerate(group):
                    if i == j:
                        continue
                    vec_j = self._vectors.get(other.id)
                    if vec_j is None:
                        continue
                    sim = self._cosine(vec_i, vec_j)
                    if sim > 0.5:  # 基础相似度过滤
                        similarities.append((other.id, sim))
                # 取 Top-K
                similarities.sort(key=lambda x: -x[1])
                for target_id, sim in similarities[:cfg.same_cluster_topk]:
                    weight = cfg.same_cluster_weight * (0.8 + 0.2 * sim)
                    weight = max(cfg.min_weight, weight)
                    self._add_connection(sphere.id, target_id, weight)
                    total_connections += 1

        # 跨簇连接（只处理高相似度对）
        # 取每个球体在不同簇中 Top-1
        for sphere in spheres:
            vec = self._vectors.get(sphere.id)
            if vec is None:
                continue
            best_others = []
            for cid, group in clusters.items():
                if cid == sphere.cluster_id:
                    continue
                for other in group:
                    vec_other = self._vectors.get(other.id)
                    if vec_other is None:
                        continue
                    sim = self._cosine(vec, vec_other)
                    if sim > cfg.embedding_threshold:
                        best_others.append((other.id, sim, other.cluster_id))
            # 每个跨簇只取 Top-1
            seen_clusters = set()
            best_others.sort(key=lambda x: -x[1])
            for target_id, sim, tc in best_others:
                if tc not in seen_clusters:
                    seen_clusters.add(tc)
                    weight = cfg.embedding_weight * (0.7 + 0.3 * sim)
                    weight = max(cfg.min_weight, weight)
                    self._add_connection(sphere.id, target_id, weight)
                    total_connections += 1

        # 时序连接（相邻 chunk）
        # 同一源文件中，按 chunk 顺序相邻的球体建连接
        files: Dict[str, List[str]] = {}
        for s in spheres:
            files.setdefault(s.source_file, []).append(s.id)
        for file_name, sids in files.items():
            for i in range(len(sids) - 1):
                weight = cfg.temporal_weight
                self._add_connection(sids[i], sids[i + 1], weight)
                total_connections += 1

        # 同步到 sphere_store
        self._sync_to_store()

        self._dirty = True
        logger.info(
            f"Built {total_connections} connections for {n} spheres"
        )
        return total_connections

    # ── 连接管理 ─────────────────────────────

    def _add_connection(self, a: str, b: str, weight: float):
        """添加双向连接（去重+取最大值）"""
        if a == b:
            return
        weight = max(cfg.min_weight, min(weight, 1.0))

        # a → b
        if a not in self._connections:
            self._connections[a] = {}
        existing = self._connections[a].get(b, 0.0)
        self._connections[a][b] = max(existing, weight)

        # b → a
        if b not in self._connections:
            self._connections[b] = {}
        existing = self._connections[b].get(a, 0.0)
        self._connections[b][a] = max(existing, weight)

    def _compute_weight(self, new_id: str, new_vec: np.ndarray,
                         new_entities: set, new_file: str,
                         new_cluster: int, target) -> float:
        """计算新球体与一个已有球体的连接权重"""
        # 同簇语义
        if new_cluster >= 0 and new_cluster == target.cluster_id:
            vec_t = self._vectors.get(target.id)
            if vec_t is not None:
                sim = self._cosine(new_vec, vec_t)
                if sim > 0.6:
                    return cfg.same_cluster_weight * (0.7 + 0.3 * sim)

        # 实体重叠
        target_entities = set(getattr(target, 'entities', []))
        if target_entities and new_entities:
            overlap = new_entities & target_entities
            if len(overlap) >= cfg.entity_threshold:
                ratio = len(overlap) / max(len(new_entities | target_entities), 1)
                return cfg.entity_weight * (0.6 + 0.4 * ratio)

        # 跨簇语义
        if new_cluster != target.cluster_id:
            vec_t = self._vectors.get(target.id)
            if vec_t is not None:
                sim = self._cosine(new_vec, vec_t)
                if sim > cfg.embedding_threshold:
                    return cfg.embedding_weight * (0.6 + 0.4 * sim)

        # 时序相邻
        if new_file == target.source_file:
            return cfg.temporal_weight * 0.5

        return 0.0

    # ── 衰减与修剪 ───────────────────────────

    def decay_all(self, factor: Optional[float] = None):
        """对所有连接权重乘以衰减因子"""
        factor = factor or cfg.decay_per_tick
        pruned = 0
        for source, targets in list(self._connections.items()):
            for target, weight in list(targets.items()):
                new_weight = weight * factor
                if new_weight < cfg.prune_threshold:
                    del targets[target]
                    # 也清理反向连接
                    if target in self._connections and source in self._connections[target]:
                        del self._connections[target][source]
                    pruned += 1
                else:
                    targets[target] = round(new_weight, 4)
            # 清理空 dict
            if not targets:
                del self._connections[source]

        self._dirty = True
        if pruned > 0:
            logger.info(f"Decayed connections: pruned {pruned} edges")

    def prune(self):
        """修剪低于阈值的连接"""
        self.decay_all(factor=1.0)  # 只修剪不衰减

    # ── 查询接口 ─────────────────────────────

    def get_connections(self, sphere_id: str) -> Dict[str, float]:
        """获取球体的连接表"""
        return self._connections.get(sphere_id, {})

    def get_weight(self, a: str, b: str) -> float:
        """获取两球体间的连接权重"""
        return self._connections.get(a, {}).get(b, 0.0)

    @property
    def total_edges(self) -> int:
        """总连接数（双向计数去重）"""
        seen = set()
        for source, targets in self._connections.items():
            for target in targets:
                edge = tuple(sorted([source, target]))
                seen.add(edge)
        return len(seen)

    @property
    def avg_degree(self) -> float:
        if not self._connections:
            return 0.0
        return sum(len(t) for t in self._connections.values()) / len(self._connections)

    # ── 同步到 SphereStore ───────────────────

    def _sync_to_store(self):
        """将内存连接表同步到 sphere_store 的 connections 字段"""
        if self._store is None:
            return
        for sphere_id, targets in self._connections.items():
            sphere = self._store.get(sphere_id)
            if sphere:
                sphere.connections = targets
        self._store._dirty = True

    # ── 持久化 ───────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """保存连接表到 JSON"""
        save_path = Path(path) if path else (self._store_path / "connections.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 转为可序列化格式
        data = {
            "version": CONNECTIONS_VERSION,
            "total_edges": self.total_edges,
            "connections": self._connections,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._dirty = False
        logger.info(f"Saved {self.total_edges} connections to {save_path}")
        return str(save_path)

    def load(self, path: Optional[str] = None) -> int:
        """从 JSON 加载连接表"""
        load_path = Path(path) if path else (self._store_path / "connections.json")

        if not load_path.exists():
            logger.info(f"No connections file at {load_path}")
            return 0

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._connections = data.get("connections", {})
        self._dirty = False
        logger.info(
            f"Loaded {self.total_edges} connections "
            f"({len(self._connections)} nodes) from {load_path}"
        )
        return self.total_edges

    # ── 工具 ─────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten() if a.ndim > 1 else a
        b = b.flatten() if b.ndim > 1 else b
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
