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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import connection as cfg, axon as axon_cfg

logger = logging.getLogger(__name__)

CONNECTIONS_VERSION = 1

# ── 因果标记词（轴突连接用）──
CAUSE_MARKERS_STRONG = [
    "propose", "demonstrate", "therefore", "enable",
    "we show that", "our results indicate",
    "this demonstrates", "this confirms that",
    "we conclude that", "results in", "leads to",
]

CAUSE_MARKERS_WEAK = [
    "suggesting that", "indicating that", "this implies",
    "these findings suggest", "attributed to", "due to",
    "as a result of", "we report the", "we present",
]

# 合并全部用于匹配
ALL_CAUSE_MARKERS = [
    (m, "strong") for m in CAUSE_MARKERS_STRONG
] + [
    (m, "weak") for m in CAUSE_MARKERS_WEAK
]


# ──────────────────────────────────────────────
# 连接检测器
# ──────────────────────────────────────────────

class ConnectionDetector:
    """球体连接检测器"""

    def __init__(self, sphere_store=None, vectors_cache=None,
                 store_path: Optional[str] = None,
                 role_table=None):
        """
        Args:
            sphere_store: SphereStore 实例
            vectors_cache: {sphere_id: np.ndarray} 向量缓存
            store_path: 连接表持久化路径
            role_table: RoleTable 实例（用于实体重叠+角色桥接连接）
        """
        self._store = sphere_store
        self._vectors = vectors_cache or {}
        self._store_path = Path(store_path or cfg.storage_dir)
        self._connections: Dict[str, Dict[str, float]] = {}
        self._axon_types: Dict[tuple, str] = {}  # {(a,b): direction, ...} 轴突边方向
        self._dirty = False
        self._role_table = role_table

        # 角色桥接缓存（延迟构建）
        self._entity_to_spheres: Dict[str, List[str]] = {}  # entity_id → [sphere_id]
        self._entity_to_cooccur: Dict[str, List[Tuple[str, int]]] = {}  # entity_id → [(other_id, count)]

    # ── 关联角色表 ───────────────────────────

    def attach_role_table(self, role_table):
        """关联角色共现表（用于实体重叠+角色桥接连接）"""
        self._role_table = role_table
        self._build_role_cache()

    def _build_role_cache(self):
        """构建角色表缓存：实体→球体反查索引"""
        if not self._role_table:
            return

        # entity_id → [sphere_ids]
        s2e = getattr(self._role_table, '_sphere_entities', {})
        if not s2e:
            s2e = getattr(self._role_table, 'sphere_entities', {})
        self._entity_to_spheres.clear()
        for sid, eids in s2e.items():
            for eid in eids:
                self._entity_to_spheres.setdefault(eid, []).append(sid)

        # entity_id co-occurrence (from RoleTable._entities co_occurrences)
        entities = getattr(self._role_table, '_entities', {})
        self._entity_to_cooccur.clear()
        for eid, ent in entities.items():
            co = getattr(ent, 'co_occurrences', {}) or {}
            if co:
                sorted_co = sorted(co.items(), key=lambda x: -x[1])[:10]
                self._entity_to_cooccur[eid] = sorted_co

    # ── 关联存储 ─────────────────────────────

    def attach(self, sphere_store, vectors_cache):
        """延迟关联存储"""
        self._store = sphere_store
        self._vectors = vectors_cache

    # ── 轴突连接检测（段落内因果链 → 跨球体连接）──

    def detect_axon_for_sphere(self, sphere_id: str) -> Dict[str, float]:
        """对单球体做因果链检测，在同源文件的其他球体中找果句

        基于实验结论：
          - 因果链均为段落内（barrier=0）
          - 平均跨 2 句，decay_angle < 55°
          - 因果链在语义空间显著区别于随机（p<0.0001）

        场景：球体 A（Methods段）含因句 → 球体 B（Results段）含果句
        """
        if not axon_cfg.enabled:
            return {}
        if self._store is None:
            return {}

        sphere = self._store.get(sphere_id)
        if not sphere or not sphere.text:
            return {}

        # 找到本球的因果标记句
        cause_sentences = self._find_cause_sentences(sphere.text)
        if not cause_sentences:
            return {}

        # 在同源文件的其他球体中找果句
        same_file_spheres = self._store.get_active(source_file=sphere.source_file)
        connections: Dict[str, float] = {}

        for other in same_file_spheres:
            if other.id == sphere_id:
                continue

            for cause in cause_sentences:
                result = self._find_effect_sentence(other.text, cause)
                if result:
                    measure, weight = result
                    connections[other.id] = max(
                        connections.get(other.id, 0.0), weight
                    )
                    break  # 每个球体只保留最强的一条连接

        return connections

    def detect_axon_batch(self, sphere_ids: Optional[List[str]] = None) -> int:
        """全量检测轴突连接"""
        if not axon_cfg.enabled or self._store is None:
            return 0

        spheres = self._store.get_active()
        if sphere_ids:
            spheres = [s for s in spheres if s.id in sphere_ids]

        # 按源文件分组
        by_file: Dict[str, List] = {}
        for s in spheres:
            by_file.setdefault(s.source_file, []).append(s)

        total = 0
        for fname, group in by_file.items():
            if len(group) < 2:
                continue
            # 对每个球体检测因果链
            for sphere in group:
                cause_sentences = self._find_cause_sentences(sphere.text)
                if not cause_sentences:
                    continue

                for other in group:
                    if other.id == sphere.id:
                        continue
                    for cause in cause_sentences:
                        result = self._find_effect_sentence(other.text, cause)
                        if result:
                            measure, weight, effect_sentence = result
                            direction = self._detect_direction(cause, effect_sentence)
                            self._add_connection(
                                sphere.id, other.id, weight,
                                conn_type="axon", direction=direction,
                            )
                            total += 1
                            break

        if total > 0:
            self._sync_to_store()
            logger.info(f"Built {total} axon (causal) connections across "
                        f"{len(by_file)} source files")
        return total

    # ── 轴突检测内部工具 ─────────────────────

    @staticmethod
    def _find_cause_sentences(text: str) -> List[str]:
        """从文本中提取含因果标记的句子"""
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
        causes = []
        seen = set()

        for para in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for s in sentences:
                s = s.strip()
                if len(s) < 15:
                    continue
                lower = s.lower()[:100]  # 只看前 100 字符
                for marker, _ in ALL_CAUSE_MARKERS:
                    if marker in lower:
                        dedup_key = s[:60]
                        if dedup_key not in seen:
                            seen.add(dedup_key)
                            causes.append(s[:200])
                        break

        return causes

    @staticmethod
    def _find_effect_sentence(text: str, cause_sentence: str) -> Optional[Tuple[float, float, str]]:
        """在文本中查找与因句最匹配的果句

        Returns:
            (measure, weight, sentence) 如果找到果句
            measure ∈ [0,1]: 匹配度
            weight ∈ [0.1, 0.6]: 连接权重
            sentence: 匹配到的果句文本
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
        candidates = []

        cause_keywords = set(cause_sentence.lower().split()[:15])

        for para in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for s in sentences:
                s = s.strip()
                if len(s) < 20:
                    continue

                lower_s = s.lower()

                # 果句应含结论性标记或承接性表达
                effect_markers = [
                    "we demonstrate", "we show", "we find", "our results",
                    "this leads", "demonstrates that", "shows that",
                    "confirms", "achieves", "improves", "outperforms",
                    "enables", "provides", "yields", "results in",
                ]
                is_effect = any(m in lower_s for m in effect_markers)

                # 关键词重叠（因句的重要词出现在果句中）
                sent_words = set(lower_s.split()[:20])
                keyword_overlap = len(cause_keywords & sent_words)
                overlap_ratio = keyword_overlap / max(len(cause_keywords), 1)

                candidates.append((s, is_effect, overlap_ratio))

        if not candidates:
            return None

        # 评分
        best_score = 0.0
        best_sentence = None
        for s, is_effect, overlap in candidates:
            score = 0.0
            if is_effect:
                score += 0.4
            score += overlap * 0.3  # 关键词重叠
            # 句长适中加分（太短没信息，太长是段落）
            if 30 < len(s) < 300:
                score += 0.2
            # 含具体数字/指标
            if re.search(r'\d+%|\d+\.\d+', s):
                score += 0.1

            if score > best_score:
                best_score = score
                best_sentence = s

        if best_score < 0.3:
            return None

        # 映射到连接权重
        weight = axon_cfg.axon_weight_weak + best_score * 0.3
        weight = max(cfg.min_weight, min(weight, axon_cfg.axon_weight_strong))

        return (best_score, weight, best_sentence)

    # ── 方向检测 ─────────────────────────────

    @staticmethod
    def _detect_direction(cause_sentence: str, effect_sentence: str) -> str:
        """判断因果连接的方向

        基于因果标记在句中的位置和类型：
          - forward: 因句包含正向因果标记（propose/leads to/enables）
          - reverse: 果句包含逆向标记（due to/attributed to/caused by）
          - bidirectional: 无法判断方向

        规则（非 ML，零成本）：
          因句 mark 在前半句 + 果句 mark 在后半句 → forward
          果句含 "due to" / "attributed to" / "caused by" → reverse
          其他 → bidirectional（保守处理）
        """
        lower_cause = cause_sentence.lower()[:150]
        lower_effect = effect_sentence.lower()[:150]

        # 逆向标记：果句中含“归因于……”
        reverse_markers = ["due to", "attributed to", "caused by",
                          "result from", "stem from", "originate from"]
        if any(m in lower_effect for m in reverse_markers):
            return "reverse"

        # 正向标记出现在因句中
        forward_markers = ["leads to", "results in", "enables",
                          "demonstrates that", "confirmed that",
                          "we propose", "we present", "we introduce"]
        if any(m in lower_cause for m in forward_markers):
            return "forward"

        # 因句前半句含因果标记 → forward（默认）
        cause_first_half = lower_cause[:len(lower_cause)//2]
        generic_cause = ["propose", "demonstrate", "therefore"]
        if any(m in cause_first_half for m in generic_cause):
            return "forward"

        return "bidirectional"

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
            {target_id: (weight, type), ...} — 新球体连接表
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
        self._axon_types.clear()

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

        # ── 实体重叠连接（使用角色表）──
        # 修复：Sphere 对象没有 entities 字段，必须从 role_table 拿
        if self._role_table:
            if not self._entity_to_spheres:
                self._build_role_cache()

            s2e = getattr(self._role_table, '_sphere_entities', {})
            if s2e:
                from collections import Counter
                entity_total = sum(len(eids) for eids in s2e.values())
                logger.info(f"Adding entity overlap connections from {entity_total} entity mappings...")
                entity_overlap = 0

                # 对每个球体，找到与其共享实体的其他球体
                for sphere in spheres:
                    my_entities = s2e.get(sphere.id, [])
                    if not my_entities:
                        continue
                    # 统计与每个其他球体共享的实体数
                    shared_count = Counter()
                    for eid in my_entities:
                        for other_sid in self._entity_to_spheres.get(eid, []):
                            if other_sid != sphere.id:
                                shared_count[other_sid] += 1
                    # 超过阈值的建连接
                    my_set = set(my_entities)
                    for other_sid, count in shared_count.items():
                        if count >= cfg.entity_threshold:
                            other_entities = s2e.get(other_sid, [])
                            union = len(my_set | set(other_entities))
                            ratio = count / max(union, 1)
                            weight = cfg.entity_weight * (0.6 + 0.4 * ratio)
                            weight = max(cfg.min_weight, min(weight, cfg.entity_weight))
                            self._add_connection(sphere.id, other_sid, weight)
                            entity_overlap += 1

                if entity_overlap:
                    logger.info(f"  Entity overlap: {entity_overlap} edges")
                    total_connections += entity_overlap

        # ── 角色桥接连接（v3：降噪版）──
        # 逻辑：实体 A↔B 共现 → 含 A 的球体 ↔ 含 B 的球体
        # 降噪三刀：
        #   1. 只用 as_phrase（完整 AH 对），不用裸名词
        #   2. 共现次数 >= role_bridge_min_cooccur
        #   3. 实体出现在 > max_entity_spread 个球体中 → 跳过（IDF 过滤高频泛化实体）
        if self._role_table and cfg.role_bridge_weight > 0:
            entities = getattr(self._role_table, '_entities', {})
            if entities:
                role_bridge = 0
                entity_pairs_seen = set()

                # 构建 phrase_only 的球体索引（完整 AH 对）
                if cfg.role_bridge_phrase_only:
                    phrase_spheres: Dict[str, List[str]] = {}
                    for eid, ent in entities.items():
                        as_phrase = list(getattr(ent, 'as_phrase', []) or [])
                        if as_phrase:
                            phrase_spheres[eid] = as_phrase
                else:
                    phrase_spheres = self._entity_to_spheres

                for eid, ent in entities.items():
                    co = getattr(ent, 'co_occurrences', {}) or {}
                    if not co:
                        continue
                    spheres_e = phrase_spheres.get(eid, [])
                    n_e = len(spheres_e)
                    # 过滤：太少（不在球体）或太多（高频泛化实体）
                    if n_e < 1 or n_e > cfg.role_bridge_max_entity_spread:
                        continue

                    # 取最强共现，但只取 >= min_cooccur 的
                    strong_co = [
                        (oeid, cnt) for oeid, cnt in co.items()
                        if cnt >= cfg.role_bridge_min_cooccur
                    ]
                    strong_co.sort(key=lambda x: -x[1])

                    for other_eid, co_count in strong_co[:5]:
                        if other_eid <= eid:
                            continue
                        epair = (eid, other_eid)
                        if epair in entity_pairs_seen:
                            continue
                        entity_pairs_seen.add(epair)

                        spheres_other = phrase_spheres.get(other_eid, [])
                        n_other = len(spheres_other)
                        if n_other < 1 or n_other > cfg.role_bridge_max_entity_spread:
                            continue

                        # 桥接：A 含 entity1，B 含 entity2
                        sphere_pairs_seen = set()
                        for sid_a in spheres_e:
                            for sid_b in spheres_other:
                                if sid_a >= sid_b:
                                    continue
                                spair = (sid_a, sid_b)
                                if spair in sphere_pairs_seen:
                                    continue
                                sphere_pairs_seen.add(spair)

                                # 权重：共现越强 → 连接越强，但以 role_bridge_weight 为上限
                                # co_count >= role_bridge_min_cooccur，所以起始比例 >= min/3
                                ratio = min(1.0, co_count / max(cfg.role_bridge_min_cooccur * 2, 1))
                                weight = cfg.role_bridge_weight * ratio
                                weight = max(cfg.min_weight, weight)
                                self._add_connection(sid_a, sid_b, weight)
                                role_bridge += 1

                if role_bridge:
                    logger.info(f"  Role bridge: {role_bridge} edges")
                    total_connections += role_bridge

        # 轴突连接（因果链检测）
        axon_total = self.detect_axon_batch(sphere_ids=sphere_ids)
        total_connections += axon_total

        # 同步到 sphere_store
        self._sync_to_store()

        self._dirty = True
        logger.info(
            f"Built {total_connections} connections for {n} spheres"
        )
        return total_connections

    # ── 连接管理 ─────────────────────────────

    def _add_connection(self, a: str, b: str, weight: float, conn_type: str = "dendrite", direction: str = "bidirectional"):
        """添加双向连接（去重+取最大值）

        Args:
            a: 源球体 ID
            b: 目标球体 ID
            weight: 连接权重 [0, 1]
            conn_type: "dendrite"（联想）或 "axon"（因果）
            direction: "forward"（因→果）| "reverse"（果→因）| "bidirectional"
                       仅 axon 类型有效，dendrite 忽略
        """
        if a == b:
            return
        weight = max(cfg.min_weight, min(weight, 1.0))

        # 主连接表（权重）
        if a not in self._connections:
            self._connections[a] = {}
        existing = self._connections[a].get(b, 0.0)
        self._connections[a][b] = max(existing, weight)

        if b not in self._connections:
            self._connections[b] = {}
        existing = self._connections[b].get(a, 0.0)
        self._connections[b][a] = max(existing, weight)

        # 类型标记（轴突/树突）
        if conn_type == "axon":
            edge = tuple(sorted([a, b]))
            # 保存方向（保留已有方向，不覆盖）
            if edge not in self._axon_types:
                self._axon_types[edge] = direction

    def is_axon(self, a: str, b: str) -> bool:
        """判断连接是否为轴突（因果）类型"""
        edge = tuple(sorted([a, b]))
        return edge in self._axon_types

    def get_axon_direction(self, a: str, b: str) -> str:
        """获取轴突方向："forward" | "reverse" | "bidirectional"
        
        无轴突连接或方向未标注时返回 "bidirectional"（兼容旧数据）
        """
        edge = tuple(sorted([a, b]))
        return self._axon_types.get(edge, "bidirectional")

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

    def get_connection_type(self, a: str, b: str) -> str:
        """获取连接类型详情: \"axon:forward\" | \"axon:reverse\" | \"dendrite\""""
        if self.is_axon(a, b):
            direction = self.get_axon_direction(a, b)
            return f"axon:{direction}"
        return "dendrite"

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
            "axon_types": {str(k): v for k, v in self._axon_types.items()},
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
        # 兼容新旧格式：旧数据是 list，新数据是 dict
        raw_axon = data.get("axon_types", {})
        if isinstance(raw_axon, list):
            # 旧格式 [(a,b), ...] → 默认 bidirectional
            self._axon_types = {tuple(e): "bidirectional" for e in raw_axon}
        else:
            # 新格式 {"(a, b)": direction, ...}
            self._axon_types = {}
            for k, v in raw_axon.items():
                # 反序列化 tuple key
                import ast
                self._axon_types[ast.literal_eval(k)] = v
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
