"""
hierarchy.py — 球体层级生长器（v2 增量版）
===========================================
相较 v1 的改动：
  1. 向量边计算分片批处理，避免 O(N²) OOM
  2. 邻接图持久化到磁盘，支持增量加载
  3. grow(incremental=True) 跳过 _reset_levels，只处理新球体
  4. save_adjacency() / load_adjacency() 接口

生长机制（三步）：
  1. 建图：球体是节点，边来自：
     - 角色表共享实体（A和B都含有"气候变化"）
     - 向量相似度（embedding cosine 高于阈值）
  2. 社区检测（Label Propagation）→ 发现密集社区
  3. 对每个≥阈值的社区：等级划分

增量流程（增量模式）：
  1. 加载磁盘邻接图
  2. 找出新球体（active - graph_sphere_set）
  3. 新球体 → entity edges + batched vector edges
  4. Merge 进既有邻接图
  5. 社区检测：既有标签做 seed，增量传播
  6. 只对新增/变更的球体划等级
"""

import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from config import paths as cfg_paths
from pipeline.role_table import RoleTable
from storage.sphere_store import SphereStore, Sphere, make_concept_id

logger = logging.getLogger(__name__)

# ── 常量 ─────────────────────────────────────
_GRAPH_VERSION = 1
_VECTOR_BATCH_SIZE = 500        # 向量边计算每批大小（减小避免 OOM）
_INCREMENTAL_EDGE_BATCH = 200   # 增量模式：新球体每批查询


# ──────────────────────────────────────────────
# 层级生长配置
# ──────────────────────────────────────────────

@dataclass
class LevelingConfig:
    min_community_size: int = 5
    coverage_ratio: float = 0.33
    max_concepts_per_community: int = 3
    min_internal_cluster: int = 4
    vector_sim_threshold: float = 0.65
    entity_sim_weight: float = 0.6
    vector_sim_weight: float = 0.4
    max_iterations: int = 50
    use_community_detection: bool = True


# ──────────────────────────────────────────────
# 社区检测（Label Propagation）
# ──────────────────────────────────────────────

def detect_communities(
    adjacency: Dict[str, Set[str]],
    max_iter: int = 50,
    seed_labels: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Label Propagation 社区检测

    Args:
        adjacency: {sphere_id: {neighbor_id, ...}}
        max_iter: 最大迭代次数
        seed_labels: 可选，已有标签（增量模式使用，不在 seed 中的节点分配新 ID）

    Returns:
        {sphere_id: community_id}
    """
    next_label = 0
    labels: Dict[str, int] = {}
    if seed_labels:
        # 只取既在邻接图中又有 seed 的节点
        for node in adjacency:
            if node in seed_labels:
                labels[node] = seed_labels[node]
        # 已有标签的最大 ID
        if labels:
            next_label = max(labels.values()) + 1
        # 不在 seed 中的新节点分配新 ID
        for node in adjacency:
            if node not in labels:
                labels[node] = next_label
                next_label += 1
    else:
        labels = {node: i for i, node in enumerate(adjacency)}
        next_label = len(adjacency)

    node_list = list(adjacency.keys())

    for iteration in range(max_iter):
        changed = 0
        np.random.shuffle(node_list)

        for node in node_list:
            neighbors = adjacency.get(node, set())
            if not neighbors:
                continue

            label_counts = Counter()
            for nb in neighbors:
                if nb in labels:
                    label_counts[labels[nb]] += 1
            if not label_counts:
                continue

            max_count = max(label_counts.values())
            best_labels = [
                lbl for lbl, cnt in label_counts.items()
                if cnt == max_count
            ]
            new_label = np.random.choice(best_labels)

            if new_label != labels[node]:
                labels[node] = new_label
                changed += 1

        if changed == 0:
            logger.debug(f"Community detection converged in {iteration + 1} iterations")
            break

    # 紧凑重编号
    unique_labels = sorted(set(labels.values()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = {node: label_map[lbl] for node, lbl in labels.items()}
    return labels


# ──────────────────────────────────────────────
# 邻接图持久化
# ──────────────────────────────────────────────

def save_adjacency(
    adjacency: Dict[str, Set[str]],
    total_spheres: int,
):
    """保存向量邻接图到磁盘

    Args:
        adjacency: {sphere_id: {neighbor_id, ...}}
        total_spheres: 建图时的全库球体数（用于下次增量判断）
    """
    path = os.path.join(cfg_paths.hierarchy_dir, "vector_adjacency.json")
    os.makedirs(cfg_paths.hierarchy_dir, exist_ok=True)

    data = {
        "version": _GRAPH_VERSION,
        "total_spheres_at_build": total_spheres,
        "sphere_ids_in_graph": list(adjacency.keys()),
        "adjacency": {k: list(v) for k, v in adjacency.items()},
        "built_at": time.time(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    logger.info(
        f"Saved adjacency: {len(adjacency)} nodes, "
        f"{sum(len(v) for v in adjacency.values()) // 2} edges"
    )


def load_adjacency() -> Tuple[Optional[Dict[str, Set[str]]], Optional[Set[str]], int]:
    """从磁盘加载向量邻接图

    Returns:
        (adjacency, sphere_ids_set, total_spheres_at_build)
        如果文件不存在则返回 (None, None, 0)
    """
    path = os.path.join(cfg_paths.hierarchy_dir, "vector_adjacency.json")
    if not os.path.exists(path):
        return None, None, 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        adjacency = {k: set(v) for k, v in data.get("adjacency", {}).items()}
        sphere_set = set(data.get("sphere_ids_in_graph", []))
        total = data.get("total_spheres_at_build", 0)

        logger.info(
            f"Loaded adjacency: {len(adjacency)} nodes, "
            f"{sum(len(v) for v in adjacency.values()) // 2} edges "
            f"(built at total={total})"
        )
        return adjacency, sphere_set, total

    except Exception as e:
        logger.warning(f"Failed to load adjacency: {e}")
        return None, None, 0


# ──────────────────────────────────────────────
# 层级生长器
# ──────────────────────────────────────────────

class HierarchyGrower:
    """层级生长器（v2 增量版）"""

    def __init__(
        self,
        sphere_store: SphereStore,
        role_table: RoleTable,
        vector_provider: Optional[Callable] = None,
        config: Optional[LevelingConfig] = None,
    ):
        self._store = sphere_store
        self._table = role_table
        self._vectors = vector_provider
        self._config = config or LevelingConfig()

    # ── 主入口 ───────────────────────────────

    def grow(
        self,
        community_map: Optional[Dict[str, int]] = None,
        incremental: bool = False,
    ) -> Dict:
        """层级生长

        Args:
            community_map: 可选社区映射（KMeans cluster_id）
            incremental: 增量模式。True=保留旧层级，只处理新球体。
                         False=全量重建（v1 行为）。

        Returns:
            统计信息
        """
        t0 = time.time()

        if incremental:
            return self._grow_incremental(community_map, t0)
        else:
            return self._grow_full(community_map, t0)

    def _grow_full(
        self,
        community_map: Optional[Dict[str, int]],
        t0: float,
    ) -> Dict:
        """全量重建

        创建空的节点图（只存节点ID，无边），存到磁盘。
        后续增量重建会逐批加入向量边。

        不做实体边和向量边的原因：
        - 实体边在 26K 球体 + 328K 实体下产生 ~10^8 条边，不可行
        - 向量边在 26K×1024 下 O(N²) 相似度矩阵，不可行
        - 增量模式下只处理新球体，边数可控
        """
        self._reset_levels()

        if community_map is not None:
            communities = community_map
            logger.info(f"Using provided community map: {len(set(communities.values()))} communities")
            adjacency = None
        else:
            # 空邻接图（仅节点列表，无边）
            spheres = self._store.get_active()
            nodes = [s for s in spheres if s.level >= 2]
            adjacency: Dict[str, Set[str]] = {s.id: set() for s in nodes}
            logger.info(f"Created empty graph: {len(adjacency)} nodes (for incremental seeding)")

            if len(adjacency) < 2:
                return {"communities": 0, "level1": 0, "status": "skipped"}
            # 空图下每个节点独自一个社区
            communities = {node: i for i, node in enumerate(adjacency)}

        stats = self._assign_levels(communities, community_map)

        # 保存空邻接图（增量模式的基础）
        if adjacency:
            save_adjacency(adjacency, self._store.count)

        internal = self._cluster_internals()
        stats["level3_clusters"] = internal
        stats["elapsed_s"] = round(time.time() - t0, 2)

        logger.info(
            f"Hierarchy grown (full): {stats.get('level1', 0)} concepts, "
            f"{stats.get('communities', 0)} communities in {stats['elapsed_s']}s"
        )
        return stats

    def _grow_incremental(
        self,
        community_map: Optional[Dict[str, int]],
        t0: float,
    ) -> Dict:
        """增量生长：保留旧层级，只处理新球体"""
        active_spheres = self._store.get_active()
        active_ids = {s.id for s in active_spheres}

        # 加载磁盘邻接图
        saved_adj, saved_sphere_set, saved_total = load_adjacency()

        if saved_adj is None or not saved_adj:
            logger.info("No saved adjacency found, falling back to full build")
            return self._grow_full(community_map, t0)

        # 找出新球体
        new_ids = active_ids - saved_sphere_set
        existing_ids = active_ids & saved_sphere_set

        if not new_ids:
            logger.info("No new spheres since last hierarchy build, skipping")
            return {
                "communities": len(set(community_map.values())) if community_map else 0,
                "level1": len(self._store.get_by_level(1)),
                "status": "no_new_spheres",
                "elapsed_s": round(time.time() - t0, 2),
            }

        logger.info(
            f"Incremental: {len(new_ids)} new spheres, "
            f"{len(existing_ids)} existing, "
            f"loaded {len(saved_adj)} adjacency nodes"
        )

        # 建图：从既存邻接图开始
        adjacency = dict(saved_adj)
        # 补全还在 active 中的既有节点（确保没有遗漏）
        for sid in existing_ids:
            if sid not in adjacency:
                adjacency[sid] = set()

        # 新节点加入邻接图
        for sid in new_ids:
            adjacency[sid] = set()

        # 从 RoleTable 读取新球体的实体边
        new_node_list = [s for s in active_spheres if s.id in new_ids]
        if new_node_list:
            self._build_new_entity_edges(adjacency, new_ids, existing_ids)

        # 向量边：新球体 vs 既有球体 + 新球体之间
        if self._vectors:
            self._build_new_vector_edges(adjacency, new_ids, existing_ids,
                                         active_spheres)

        # 移除孤立节点
        isolated = [nid for nid, nb in adjacency.items() if not nb]
        for nid in isolated:
            del adjacency[nid]
        # 从新节点中也移除
        new_ids = new_ids - set(isolated)

        if len(adjacency) < 2:
            logger.info("Too few nodes after incremental graph, skipping")
            return {"communities": 0, "level1": 0, "status": "skipped"}

        # 社区检测（用既有标签做 seed）
        # 从已保存的社区信息重建 seed_labels
        # 先跑全量 label propagation——邻接图变大了，但节点数没暴增
        # 全量 LP 的复杂度是 O(E × iter)，增量时边数增加有限
        saved_community_path = os.path.join(cfg_paths.hierarchy_dir, "community_labels.json")
        seed_labels = None
        if os.path.exists(saved_community_path):
            try:
                with open(saved_community_path, "r") as f:
                    seed_labels = json.load(f)
            except Exception:
                pass

        communities = detect_communities(
            adjacency, max_iter=self._config.max_iterations,
            seed_labels=seed_labels,
        )

        if community_map:
            communities = community_map  # 用 KMeans 簇覆盖

        # 保存社区标签
        os.makedirs(cfg_paths.hierarchy_dir, exist_ok=True)
        with open(saved_community_path, "w") as f:
            json.dump(communities, f, ensure_ascii=False)

        # 划等级：已有概念的社区跳过，只处理受新球体影响的
        stats = self._assign_levels(communities, community_map, incremental=True)

        # 保存邻接图
        save_adjacency(adjacency, self._store.count)

        internal = self._cluster_internals()
        stats["level3_clusters"] = internal
        stats["elapsed_s"] = round(time.time() - t0, 2)

        logger.info(
            f"Hierarchy grown (incremental): {stats.get('level1', 0)} concepts, "
            f"{stats.get('communities', 0)} communities in {stats['elapsed_s']}s"
        )
        return stats

    # ── 重置 ─────────────────────────────────

    def _reset_levels(self):
        """清空所有球体的层级标记"""
        for sphere in self._store.get_active():
            sphere.level = 2
            sphere.parent_id = ""
        to_remove = [
            sid for sid, s in self._store._spheres.items()
            if s.active and s.level == 1
        ]
        for sid in to_remove:
            del self._store._spheres[sid]
        self._store._dirty = True
        logger.debug(f"Reset levels: cleared {len(to_remove)} Level-1 spheres")

    # ── 建图 ─────────────────────────────────

    def _build_graph(self, incremental: bool = False,
                     skip_vector_edges: bool = False) -> Dict[str, Set[str]]:
        """建球体图

        Args:
            incremental: 如果 True，尝试加载磁盘上的邻接图并 merge
            skip_vector_edges: 如果 True，跳过向量边计算（首次全量重建时）

        Returns:
            {sphere_id: {neighbor_id, ...}}
        """
        if incremental:
            saved_adj, saved_set, _ = load_adjacency()
            if saved_adj and saved_set:
                return saved_adj

        spheres = self._store.get_active()
        nodes = [s for s in spheres if s.level >= 2]
        if not nodes:
            return {}

        node_ids = {s.id for s in nodes}
        adjacency: Dict[str, Set[str]] = {s.id: set() for s in nodes}

        # 边 A：共享实体
        self._build_entity_edges(adjacency, node_ids, nodes)

        # 边 B：向量相似度（分片），首次全量重建时跳过
        if self._vectors and not skip_vector_edges:
            self._build_vector_edges_batched(adjacency, node_ids, nodes)

        # 移除孤立节点
        isolated = [nid for nid, nb in adjacency.items() if not nb]
        for nid in isolated:
            del adjacency[nid]

        logger.info(
            f"Graph built: {len(adjacency)} nodes, "
            f"{sum(len(n) for n in adjacency.values()) // 2} edges, "
            f"{len(isolated)} isolated"
        )
        return adjacency

    def _build_entity_edges(
        self,
        adjacency: Dict[str, Set[str]],
        node_ids: Set[str],
        nodes: List[Sphere],
    ):
        """实体共享边"""
        if not self._table:
            return

        entity_to_spheres: Dict[str, Set[str]] = defaultdict(set)
        for sid in node_ids:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                entity_to_spheres[eid].add(sid)

        for eid, sphere_set in entity_to_spheres.items():
            if len(sphere_set) < 2:
                continue
            sphere_list = list(sphere_set)
            for i in range(len(sphere_list)):
                for j in range(i + 1, len(sphere_list)):
                    a, b = sphere_list[i], sphere_list[j]
                    if a in adjacency and b in adjacency:
                        adjacency[a].add(b)
                        adjacency[b].add(a)

    def _build_vector_edges_batched(
        self,
        adjacency: Dict[str, Set[str]],
        node_ids: Set[str],
        nodes: List[Sphere],
    ):
        """向量相似度边 — 分片批处理（避免 O(N²) OOM）"""
        threshold = self._config.vector_sim_threshold
        if not self._vectors:
            return

        # 收集向量
        vec_map: Dict[str, np.ndarray] = {}
        for s in nodes:
            vec = self._vectors(s.id)
            if vec is not None:
                vec_map[s.id] = vec

        ids = list(vec_map.keys())
        if len(ids) < 2:
            return

        vectors = np.stack([vec_map[nid] for nid in ids], axis=0)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)

        n = len(ids)

        # 分片处理：每批 _VECTOR_BATCH_SIZE 行
        for batch_start in range(0, n, _VECTOR_BATCH_SIZE):
            batch_end = min(batch_start + _VECTOR_BATCH_SIZE, n)

            # 计算当前批与所有向量的相似度
            # shape: (batch_size, n)
            batch_vecs = vectors[batch_start:batch_end]
            batch_norms = norms[batch_start:batch_end]
            sim_batch = (batch_vecs @ vectors.T) / (batch_norms @ norms.T)
            sim_batch = np.clip(sim_batch, 0, 1)

            # 只处理上三角（避免重复建边）
            for local_i in range(sim_batch.shape[0]):
                global_i = batch_start + local_i
                a = ids[global_i]
                # 与当前批之后的列（上三角 + 批内自身上三角）
                col_start = global_i + 1  # 只处理 i < j
                if col_start >= n:
                    continue

                mask = sim_batch[local_i, col_start:] > threshold
                cols = np.where(mask)[0] + col_start
                for col_idx in cols:
                    b = ids[col_idx]
                    if a in adjacency and b in adjacency:
                        adjacency[a].add(b)
                        adjacency[b].add(a)

            # 显式释放大矩阵
            del sim_batch

        logger.debug(
            f"Vector edges (batched): {n} nodes, "
            f"batch={_VECTOR_BATCH_SIZE}"
        )

    # ── 增量建图辅助 ─────────────────────────

    def _build_new_entity_edges(
        self,
        adjacency: Dict[str, Set[str]],
        new_ids: Set[str],
        existing_ids: Set[str],
    ):
        """增量实体边：新球体 vs 所有球体"""
        if not self._table:
            return

        # 新球体的实体→球体分组
        for sid in new_ids:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                # 该实体下所有球体（包括旧的和新的）
                sphere_set = self._table._sphere_entities.get(sid, set())
                # 实际上我们需要知道该实体下 who 有谁
                # 从 RoleTable 反向查
                pass

        # 简化实现：遍历每个新球体的实体，查该实体的所有球体
        for sid in new_ids:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                if eid not in self._table._entities:
                    continue
                # _entities 存储的是 entity_id → EntityInfo，没有反向查 sphere_ids
                # 我们只能通过遍历 adjacency 的已有球体来检查
                # 优化的做法是预先建一个 entity→spheres 的索引
                # 这里用 RoleTable 的方法
                pass

        # RoleTable 没有提供 entity→spheres 的直接反向查询
        # 回退：对每个新球体，检查每个已有球体是否共享实体
        new_list = list(new_ids)
        existing_list = list(existing_ids & set(adjacency.keys()))
        if not existing_list:
            return

        logger.debug(
            f"Checking entity edges: {len(new_list)} new vs {len(existing_list)} existing"
        )

        # 建临时索引：entity_id → [sphere_ids]
        entity_to_spheres: Dict[str, Set[str]] = defaultdict(set)
        for sid in existing_list:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                entity_to_spheres[eid].add(sid)
        # 加入新球体
        for sid in new_list:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                entity_to_spheres[eid].add(sid)

        # 建边：共享实体的新球体 ↔ 该实体下所有球体
        for sid in new_list:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                sphere_set = entity_to_spheres.get(eid, set())
                for other in sphere_set:
                    if other != sid and other in adjacency and sid in adjacency:
                        adjacency[sid].add(other)
                        adjacency[other].add(sid)

    def _build_new_vector_edges(
        self,
        adjacency: Dict[str, Set[str]],
        new_ids: Set[str],
        existing_ids: Set[str],
        all_spheres: List[Sphere],
    ):
        """增量向量边：新球体 vs 既有球体 + 新vs新"""
        threshold = self._config.vector_sim_threshold
        if not self._vectors:
            return

        # 收集既有球体的向量（缓存用）
        existing_vec_map: Dict[str, np.ndarray] = {}
        for s in all_spheres:
            if s.id in existing_ids:
                vec = self._vectors(s.id)
                if vec is not None:
                    existing_vec_map[s.id] = vec

        new_vec_map: Dict[str, np.ndarray] = {}
        for s in all_spheres:
            if s.id in new_ids:
                vec = self._vectors(s.id)
                if vec is not None:
                    new_vec_map[s.id] = vec

        if not new_vec_map:
            return

        existing_ids_list = list(existing_vec_map.keys())
        new_ids_list = list(new_vec_map.keys())

        # 既有向量矩阵
        existing_vecs = np.stack([existing_vec_map[sid] for sid in existing_ids_list], axis=0) if existing_ids_list else np.zeros((0, 1024), dtype=np.float32)

        # 新向量矩阵
        new_vecs = np.stack([new_vec_map[sid] for sid in new_ids_list], axis=0)

        # 分片：每批处理一部分新球体
        for batch_start in range(0, len(new_ids_list), _INCREMENTAL_EDGE_BATCH):
            batch_end = min(batch_start + _INCREMENTAL_EDGE_BATCH, len(new_ids_list))
            batch_ids = new_ids_list[batch_start:batch_end]
            batch_vecs = new_vecs[batch_start:batch_end]

            # 新球体 vs 既有球体
            if len(existing_ids_list) > 0:
                sim_vs_existing = batch_vecs @ existing_vecs.T
                sim_vs_existing = np.clip(sim_vs_existing, 0, 1)

                for local_i, a in enumerate(batch_ids):
                    if a not in adjacency:
                        continue
                    mask = sim_vs_existing[local_i] > threshold
                    cols = np.where(mask)[0]
                    for col_idx in cols:
                        b = existing_ids_list[col_idx]
                        if b in adjacency:
                            adjacency[a].add(b)
                            adjacency[b].add(a)

                del sim_vs_existing

            # 新球体 vs 新球体（同批 + 剩余批）
            remaining_new = new_ids_list[batch_end:]
            if remaining_new:
                remaining_vecs = new_vecs[batch_end:]
                sim_vs_new = batch_vecs @ remaining_vecs.T
                sim_vs_new = np.clip(sim_vs_new, 0, 1)

                for local_i, a in enumerate(batch_ids):
                    if a not in adjacency:
                        continue
                    mask = sim_vs_new[local_i] > threshold
                    cols = np.where(mask)[0]
                    for col_idx in cols:
                        b = remaining_new[col_idx]
                        if b in adjacency:
                            adjacency[a].add(b)
                            adjacency[b].add(a)

                del sim_vs_new

        logger.debug(
            f"New vector edges: {len(new_ids_list)} new vs "
            f"{len(existing_ids_list)} existing (batched)"
        )

    # ── 等级划分 ─────────────────────────────

    def _assign_levels(
        self, communities: Dict[str, int],
        community_map: Optional[Dict[str, int]] = None,
        incremental: bool = False,
    ) -> Dict:
        """按社区检测结果划分等级

        Args:
            communities: {sphere_id: community_id}
            community_map: 原始社区映射（cluster_id）
            incremental: 增量模式——跳过已有概念的社区

        Returns:
            统计信息
        """
        comm_groups: Dict[int, List[str]] = defaultdict(list)
        for sid, cid in communities.items():
            comm_groups[cid].append(sid)

        # 如果增量模式，收集已有 Level-1 球的社区归属
        existing_concept_communities: Set[int] = set()
        if incremental:
            for c in self._store.get_by_level(1):
                if c.child_ids:
                    # 找一个子球的社区
                    for child_id in c.child_ids:
                        if child_id in communities:
                            existing_concept_communities.add(communities[child_id])
                            break

        l1_created = 0
        communities_leveled = 0

        for cid, members in comm_groups.items():
            if len(members) < self._config.min_community_size:
                continue

            # 增量模式：如果该社区已有概念球体，跳过
            if incremental and cid in existing_concept_communities:
                continue

            subject_counts = self._count_subjects_in_community(members)
            if not subject_counts:
                continue

            filtered = self._filter_concepts_for_community(
                subject_counts,
                community_size=len(members),
                coverage_ratio=self._config.coverage_ratio,
                max_concepts=self._config.max_concepts_per_community,
            )

            cluster_id = -1
            if community_map and members:
                cluster_id = community_map.get(members[0], -1)

            for subject, freq in filtered:
                concept_id = make_concept_id(subject)
                if self._store.get(concept_id):
                    continue  # 防止重复创建

                children = self._find_children_for_subject(subject, members)

                concept = Sphere(
                    id=concept_id,
                    text=subject[:80],
                    source_file="__concept__",
                    source_type="",
                    level=1,
                    child_ids=children,
                    embedding_source="subject",
                    mass=1.0 + 0.1 * len(children),
                    effective_mass=1.0 + 0.1 * len(children),
                    cluster_id=cluster_id,
                    active=True,
                )
                self._store.add(concept)

                for child_id in children:
                    child = self._store.get(child_id)
                    if child and not child.parent_id:
                        child.parent_id = concept_id
                        child.level = 2

                l1_created += 1

            communities_leveled += 1

        logger.info(
            f"Level assignment: {communities_leveled} communities → "
            f"{l1_created} concepts{' (incremental)' if incremental else ''}"
        )
        return {
            "communities_total": len(comm_groups),
            "communities_leveled": communities_leveled,
            "level1": l1_created,
        }

    def _count_subjects_in_community(self, member_ids: List[str]) -> List[Tuple[str, int]]:
        if not self._table:
            return []
        subject_counts: Counter = Counter()
        for sid in member_ids:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                info = self._table._entities.get(eid)
                if info and sid in info.as_phrase:
                    subject_counts[info.text] += 1
        return [(text, count) for text, count in subject_counts.most_common()]

    def _find_children_for_subject(self, subject: str, member_ids: List[str]) -> List[str]:
        if not self._table:
            return member_ids[:]
        subject_lower = subject.lower()
        children = []
        for sid in member_ids:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                info = self._table._entities.get(eid)
                if info and info.text.lower() == subject_lower:
                    children.append(sid)
                    break
        return children or member_ids[:]

    # ── 内部聚类（三级标注） ─────────────────

    def _cluster_internals(self) -> int:
        concepts = self._store.get_by_level(1)
        total_clustered = 0
        for c in concepts:
            if len(c.child_ids) < self._config.min_internal_cluster:
                continue
            children = self._store.get_children(c.id)
            if len(children) < self._config.min_internal_cluster:
                continue
            vectors = []
            valid = []
            for child in children:
                vec = self._vectors(child.id) if self._vectors else None
                if vec is not None:
                    vectors.append(vec)
                    valid.append(child)
            if len(valid) < self._config.min_internal_cluster:
                continue
            try:
                from sklearn.cluster import KMeans
                vectors_arr = np.stack(vectors, axis=0)
                k = min(4, max(2, len(valid) // 2))
                km = KMeans(n_clusters=k, random_state=42, n_init=5)
                labels = km.fit_predict(vectors_arr)
                for child, label in zip(valid, labels):
                    child.cluster_id = int(label)
                total_clustered += 1
            except Exception as e:
                logger.debug(f"Internal cluster failed: {e}")
        if total_clustered:
            logger.info(f"Internal clustering: {total_clustered} concepts")
        return total_clustered

    # ── 社区独立筛选 ─────────────────────────

    @staticmethod
    def _filter_concepts_for_community(
        subject_counts: List[Tuple[str, int]],
        community_size: int,
        coverage_ratio: float,
        max_concepts: int,
    ) -> List[Tuple[str, int]]:
        threshold = max(2, int(community_size * coverage_ratio))
        result = []
        for subject, freq in subject_counts:
            if freq < threshold:
                break
            if len(result) >= max_concepts:
                break
            result.append((subject, freq))
        return result

    # ── 一级球体嵌入（写入 FAISS） ───────────

    def embed_concepts(self) -> List[Tuple[str, np.ndarray]]:
        if not self._vectors:
            logger.warning("No vector provider, cannot embed concepts")
            return []
        concepts = self._store.get_by_level(1)
        results = []
        for c in concepts:
            children = self._store.get_children(c.id)
            if not children:
                continue
            vectors = []
            for child in children:
                vec = self._vectors(child.id)
                if vec is not None:
                    vectors.append(vec)
            if not vectors:
                continue
            centroid = np.mean(np.stack(vectors, axis=0), axis=0)
            results.append((c.id, centroid))
        logger.info(f"Computed {len(results)} concept embeddings")
        return results
