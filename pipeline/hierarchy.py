"""
hierarchy.py — 球体层级生长器
================================
不预设任何层级结构，从语料中自然生长。

生长机制（三步）：
  1. 建图：球体是节点，边来自：
     - 角色表共享实体（A和B都含有"气候变化"）
     - 向量相似度（embedding cosine 高于阈值）
  2. 社区检测（Label Propagation）→ 发现密集社区
  3. 对每个≥阈值的社区：
     - 找社区内出现≥2次的主语 → 每个主语建一个一级球体
     - 社区句子都挂到这些一级球体下（共享子球体）
     - 一级球体内部≥4句 → 再聚类 → 三级标注

阈值（每个社区独立判断，不全局）：
  min_community_size: 5  # 社区少于5个句子的不划等级
  coverage_ratio: 0.33   # 主语出现 ≥ 社区大小 × 此比例才建一级
  max_concepts_per_community: 3  # 单个社区最多一级球体数
"""

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from config import paths as cfg_paths
from pipeline.role_table import RoleTable
from storage.sphere_store import SphereStore, Sphere, make_concept_id

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 层级生长配置
# ──────────────────────────────────────────────

@dataclass
class LevelingConfig:
    min_community_size: int = 5        # 社区最少球体数才划等级
    coverage_ratio: float = 0.33       # 主语出现 ≥ 社区大小 × 此比例才建一级
    max_concepts_per_community: int = 3  # 单个社区最多一级球体数
    min_internal_cluster: int = 4      # 一级球体内部≥4句才聚三级
    vector_sim_threshold: float = 0.65 # 向量相似度建边阈值
    entity_sim_weight: float = 0.6     # 实体共享边的基准权重
    vector_sim_weight: float = 0.4     # 向量相似边的基准权重
    max_iterations: int = 50           # 社区检测最大迭代
    use_community_detection: bool = True  # 开关


# ──────────────────────────────────────────────
# 社区检测（Label Propagation）
# ──────────────────────────────────────────────

def detect_communities(
    adjacency: Dict[str, Set[str]],
    max_iter: int = 50,
) -> Dict[str, int]:
    """Label Propagation 社区检测

    Args:
        adjacency: {sphere_id: {neighbor_id, ...}}
        max_iter: 最大迭代次数

    Returns:
        {sphere_id: community_id}
    """
    # 初始化：每个节点一个社区
    labels = {node: i for i, node in enumerate(adjacency)}
    node_list = list(adjacency.keys())

    for iteration in range(max_iter):
        changed = 0
        # 随机顺序（避免初始顺序偏差）
        np.random.shuffle(node_list)

        for node in node_list:
            neighbors = adjacency.get(node, set())
            if not neighbors:
                continue

            # 邻居标签投票
            label_counts = Counter()
            for nb in neighbors:
                if nb in labels:
                    label_counts[labels[nb]] += 1

            if not label_counts:
                continue

            # 选出现次数最多的标签（同票时随机选一个）
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

    # 重新编号（保持紧凑）
    unique_labels = sorted(set(labels.values()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = {node: label_map[lbl] for node, lbl in labels.items()}

    return labels


# ──────────────────────────────────────────────
# 层级生长器
# ──────────────────────────────────────────────

class HierarchyGrower:
    """层级生长器

    每次调用 grow() 时，从当前语料+角色表出发：
      - 建图 → 社区检测 → 等级划分
      - 旧等级被替换（不累积，每次全量刷新）

    这样新语料加入后，社区结构可能重组，等级自然跟随变化。
    """

    def __init__(
        self,
        sphere_store: SphereStore,
        role_table: RoleTable,
        vector_provider: Optional[Callable] = None,
        config: Optional[LevelingConfig] = None,
    ):
        self._store = sphere_store
        self._table = role_table
        self._vectors = vector_provider  # f(sphere_id) → np.ndarray | None
        self._config = config or LevelingConfig()

    # ── 主入口 ───────────────────────────────

    def grow(self) -> Dict:
        """全量生长：社区检测 → 等级划分

        每次全量跑，不增量累积。
        Returns:
            统计信息
        """
        t0 = time.time()

        # Step 1: 重置所有球体的等级（清空旧层级）
        self._reset_levels()

        # Step 2: 建图（球体间的边）
        adjacency = self._build_graph()
        if len(adjacency) < 2:
            logger.info("Too few spheres for hierarchy, skipping")
            return {"communities": 0, "level1": 0, "status": "skipped"}

        # Step 3: 社区检测
        communities = detect_communities(
            adjacency, max_iter=self._config.max_iterations
        )

        # Step 4: 按社区划等级
        stats = self._assign_levels(communities)

        # Step 5: 一级球体内部聚类（三级标注）
        internal = self._cluster_internals()
        stats["level3_clusters"] = internal

        elapsed = time.time() - t0
        stats["elapsed_s"] = round(elapsed, 2)

        logger.info(
            f"Hierarchy grown: {stats.get('level1', 0)} concepts, "
            f"{stats.get('communities', 0)} communities "
            f"in {elapsed:.2f}s"
        )
        return stats

    # ── 重置 ─────────────────────────────────

    def _reset_levels(self):
        """清空所有球体的层级标记"""
        for sphere in self._store.get_active():
            sphere.level = 2      # 默认二级
            sphere.parent_id = ""
        # 一级球体需要彻底移除（ID 冲突会阻止 recreate）
        to_remove = [
            sid for sid, s in self._store._spheres.items()
            if s.active and s.level == 1
        ]
        for sid in to_remove:
            del self._store._spheres[sid]
        self._store._dirty = True

        logger.debug(f"Reset levels: cleared {len(to_remove)} Level-1 spheres")

    # ── 建图 ─────────────────────────────────

    def _build_graph(self) -> Dict[str, Set[str]]:
        """建球体图

        图节点 = 所有活跃的二级球体。
        边的来源有两种：
          a) 共享实体（RoleTable 中共享主语或宾语）
          b) embedding 余弦相似度 > 阈值
        """
        spheres = self._store.get_active()
        # 只对 Level-2（非概念）球体建图
        nodes = [s for s in spheres if s.level >= 2]
        if not nodes:
            return {}

        node_ids = {s.id for s in nodes}
        adjacency: Dict[str, Set[str]] = {s.id: set() for s in nodes}

        # 边 A：共享实体
        self._build_entity_edges(adjacency, node_ids, nodes)

        # 边 B：向量相似度
        if self._vectors:
            self._build_vector_edges(adjacency, node_ids, nodes)

        # 移除孤立节点（无边）
        isolated = [
            nid for nid, neighbors in adjacency.items()
            if not neighbors
        ]
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
        """实体共享边：两个球体包含同一实体"""
        # 从 RoleTable 获取实体→球体映射
        if not self._table:
            return

        # 按实体分组
        entity_to_spheres: Dict[str, Set[str]] = defaultdict(set)
        for sid in node_ids:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                entity_to_spheres[eid].add(sid)

        # 对每个实体，所有包含它的球体两两建边
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

    def _build_vector_edges(
        self,
        adjacency: Dict[str, Set[str]],
        node_ids: Set[str],
        nodes: List[Sphere],
    ):
        """向量相似度边：embedding 余弦 > 阈值"""
        threshold = self._config.vector_sim_threshold
        if not self._vectors:
            return

        # 获取所有向量
        vec_map: Dict[str, np.ndarray] = {}
        for s in nodes:
            vec = self._vectors(s.id)
            if vec is not None:
                vec_map[s.id] = vec

        ids = list(vec_map.keys())
        if len(ids) < 2:
            return

        # 批量计算相似度矩阵
        vectors = np.stack([vec_map[nid] for nid in ids], axis=0)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 防除零
        sim_matrix = vectors @ vectors.T / (norms @ norms.T)
        sim_matrix = np.clip(sim_matrix, 0, 1)

        # 高于阈值则建边
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if sim_matrix[i][j] > threshold:
                    a, b = ids[i], ids[j]
                    if a in adjacency and b in adjacency:
                        adjacency[a].add(b)
                        adjacency[b].add(a)

    # ── 等级划分 ─────────────────────────────

    def _assign_levels(self, communities: Dict[str, int]) -> Dict:
        """按社区检测结果划分等级

        对每个≥阈值的社区：
          1. 从 RoleTable 中统计该社区内各主语出现频率
          2. 出现≥覆盖阈值的主语 → 建一级球体
          3. 一级球体挂载社区句子（child_ids）
          4. 句子标记 parent_id
        """
        # 社区分组
        comm_groups: Dict[int, List[str]] = defaultdict(list)
        for sid, cid in communities.items():
            comm_groups[cid].append(sid)

        l1_created = 0
        communities_leveled = 0

        for cid, members in comm_groups.items():
            if len(members) < self._config.min_community_size:
                continue

            # 统计社区内主语出现频率
            subject_counts = self._count_subjects_in_community(members)
            if not subject_counts:
                continue

            # 社区独立筛选：覆盖阈值 + 上限
            filtered = self._filter_concepts_for_community(
                subject_counts,
                community_size=len(members),
                coverage_ratio=self._config.coverage_ratio,
                max_concepts=self._config.max_concepts_per_community,
            )

            for subject, freq in filtered:
                concept_id = make_concept_id(subject)

                # 找到包含该主语的成员
                children = self._find_children_for_subject(subject, members)

                # 创建一级球体
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
                    active=True,
                )
                self._store.add(concept)

                # 标注二级球体的 parent_id
                for child_id in children:
                    child = self._store.get(child_id)
                    if child and not child.parent_id:
                        child.parent_id = concept_id
                        child.level = 2

                l1_created += 1

            communities_leveled += 1

        logger.info(
            f"Level assignment: {communities_leveled} communities → "
            f"{l1_created} concepts"
        )
        return {
            "communities_total": len(comm_groups),
            "communities_leveled": communities_leveled,
            "level1": l1_created,
        }

    def _count_subjects_in_community(
        self, member_ids: List[str]
    ) -> List[Tuple[str, int]]:
        """统计社区中每个主语的出现次数"""
        if not self._table:
            return []

        subject_counts: Counter = Counter()

        for sid in member_ids:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                info = self._table._entities.get(eid)
                if info and sid in info.as_phrase:
                    subject_counts[info.text] += 1

        # 返回所有非零主语（阈值筛选在 _filter_concepts_for_community 中按社区大小独立处理）
        return [
            (text, count) for text, count in
            subject_counts.most_common()
        ]

    def _find_children_for_subject(
        self, subject: str, member_ids: List[str]
    ) -> List[str]:
        """找到社区中包含该主语（作为主语或宾语）的所有球体"""
        if not self._table:
            return member_ids[:]  # 回退：全部作为子球体

        subject_lower = subject.lower()
        children = []

        for sid in member_ids:
            entities = self._table._sphere_entities.get(sid, set())
            for eid in entities:
                info = self._table._entities.get(eid)
                if info and info.text.lower() == subject_lower:
                    children.append(sid)
                    break

        return children or member_ids[:]  # 回退

    # ── 内部聚类（三级标注） ─────────────────

    def _cluster_internals(self) -> int:
        """对一级球体内部做子空间聚类 → 标注三级"""
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
                    child.cluster_id = int(label)  # 充当三级ID

                total_clustered += 1

            except Exception as e:
                logger.debug(f"Internal cluster failed for '{c.text}': {e}")

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
        """社区独立筛选：覆盖阈值 + 上限

        每个社区独立判断，不用全局参数截断。
        覆盖比例比固定频次更适合实际——100句的社区和10句的社区，
        同一个主语要划一级需要的绝对频次不一样。

        Args:
            subject_counts: [(主语, 频次)] 已按频次降序
            community_size: 社区大小（成员数）
            coverage_ratio: 覆盖比例（默认0.33，即主语出现在1/3以上句子）
            max_concepts: 该社区最多一级球体数

        Returns:
            筛选后的列表，按频次降序
        """
        threshold = max(2, int(community_size * coverage_ratio))
        result = []

        for subject, freq in subject_counts:
            if freq < threshold:
                break  # 降序排列，后面的更不够
            if len(result) >= max_concepts:
                break
            result.append((subject, freq))

        return result

    # ── 一级球体嵌入（写入 FAISS） ───────────

    def embed_concepts(self) -> List[Tuple[str, np.ndarray]]:
        """为所有一级球体生成向量（社区质心）

        在每个社区中，取全部成员向量的算术平均作为概念向量。
        这个向量随后应由外部写入 FAISS。

        Returns:
            [(concept_id, vector), ...]
        """
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
