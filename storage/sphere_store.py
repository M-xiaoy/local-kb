"""
sphere_store.py — 球体库（元数据层）
=====================================
管理球体的元数据、重力空间字段、持久化。
与 FAISS 索引分离——FAISS 只管向量，sphere_store 管文本+场域+连接表。

存储架构：
  FAISS .index  ← 稠密向量，ANN 搜索
        ↓
  registry      ← faiss_id ↔ sphere_id 双向映射
        ↓
  sphere_store  ← Sphere 对象持久化（JSON）

持久化策略（v1）：
  - JSON 格式，人类可读，版本字段可迁移
  - 全量读写，千级规模够用
  - 软删除：active=False（不破坏现有 FAISS ID 对齐）
  - 重建索引时过滤 inactive 的球体

Sphere ID 生成：
  - SHA256(text + source_file)[:12]，保证同一文本不重复入库
  - 支持幂等导入：重复上传同一份文档不会产生重复球体
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import paths as cfg_paths

logger = logging.getLogger(__name__)

SPHERE_VERSION = 3  # v3: 新增 doc_terms 文档级关键术语


# ──────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────

@dataclass
class Sphere:
    """单个球体——文本切片的元数据 + 重力空间字段

    gravity_field: 球体到各场域质心的引力值，由系统自动维护。
                   格式: {"技术笔记": 0.85, "小说创作": 0.32}
                   值域 [0, 1]，越高表示该球体越接近该场域质心。
                   首次入库时预计算，质心变化时增量/全量更新。

    ---- v2 新增层级字段 ----
    level: 1=主语球体（概念级）, 2=句子球体（证据级）, 3=子概念球体
    parent_id: 二级球体指向所属的一级球体
    child_ids: 一级球体跟踪下属的二级球体
    embedding_source: "subject"（一级=主语原文）| "sentence"（二级=完整句子）
    """
    id: str                          # 唯一标识（SHA256[:12]）
    text: str                        # 球体原文
    source_file: str                 # 源文件名
    source_type: str = ""            # 场域标签
    mass: float = 1.0                # 基础质量
    diversity: float = 0.0           # 多样性得分（来源分布广度）
    effective_mass: float = 1.0      # mass × (1 + diversity)
    connections: Dict[str, float] = field(default_factory=dict)
    gravity_field: Dict[str, float] = field(default_factory=dict)
    term_weights: Dict[str, float] = field(default_factory=dict)  # {词: TF权重}
    cluster_id: int = -1             # 所属簇 ID（-1=未分配，聚类后更新）
    active: bool = True              # 软删除标记
    created_at: str = ""             # 入库时间（ISO 格式）
    # ---- v2 层级字段 ----
    level: int = 2                   # 默认二级（向后兼容）
    parent_id: str = ""              # 上级球体 ID（二级→一级）
    child_ids: List[str] = field(default_factory=list)  # 下级球体 ID 列表
    embedding_source: str = "sentence"
    # ---- v3 文档级术语 ----
    doc_terms: List[str] = field(default_factory=list)  # 文档级关键 AH 短语

    def __post_init__(self):
        # 确保 effective_mass 与 mass+diversity 一致
        self._sync_effective_mass()

    def _sync_effective_mass(self):
        self.effective_mass = self.mass * (1.0 + self.diversity)


# ──────────────────────────────────────────────
# 球体库
# ──────────────────────────────────────────────

class SphereStore:
    """球体元数据存储

    管理全部 Sphere 对象的增删查改 + JSON 持久化。
    FAISS 索引与球体库通过 registry 层对齐。
    """

    def __init__(self, storage_path: Optional[str] = None):
        self._path = Path(storage_path or cfg_paths.spheres_data)
        self._spheres: Dict[str, Sphere] = {}  # sphere_id → Sphere
        self._dirty = False  # 是否有未保存的修改

    # ── 属性 ──────────────────────────────────

    @property
    def count(self) -> int:
        """活跃球体数量（排除软删除）"""
        return sum(1 for s in self._spheres.values() if s.active)

    @property
    def total_count(self) -> int:
        """全部球体（含软删除）"""
        return len(self._spheres)

    # ── 增 ────────────────────────────────────

    def add(self, sphere: Sphere) -> bool:
        """添加一个球体。如果 ID 已存在则跳过。

        Returns:
            True 表示新添加，False 表示已存在（幂等）
        """
        if sphere.id in self._spheres:
            logger.debug(f"Sphere {sphere.id[:8]} already exists, skipped")
            return False
        self._spheres[sphere.id] = sphere
        self._dirty = True
        return True

    def add_many(self, spheres: List[Sphere]) -> int:
        """批量添加，返回实际新增数量"""
        added = 0
        for s in spheres:
            if self.add(s):
                added += 1
        return added

    # ── 查 ────────────────────────────────────

    def get(self, sphere_id: str) -> Optional[Sphere]:
        return self._spheres.get(sphere_id)

    def get_many(self, sphere_ids: List[str]) -> List[Sphere]:
        """批量获取，跳过不存在的 ID"""
        return [self._spheres[sid] for sid in sphere_ids if sid in self._spheres]

    def get_active(self) -> List[Sphere]:
        """返回所有活跃球体"""
        return [s for s in self._spheres.values() if s.active]

    def get_by_source(self, source_file: str) -> List[Sphere]:
        """按源文件名查找"""
        return [
            s for s in self._spheres.values()
            if s.source_file == source_file and s.active
        ]

    def get_by_type(self, source_type: str) -> List[Sphere]:
        """按场域标签查找"""
        return [
            s for s in self._spheres.values()
            if s.source_type == source_type and s.active
        ]

    def get_by_level(self, level: int) -> List[Sphere]:
        """按层级查找（1=主语, 2=句子, 3=子概念）"""
        return [
            s for s in self._spheres.values()
            if s.level == level and s.active
        ]

    def get_children(self, parent_id: str) -> List[Sphere]:
        """获取一级球体的所有二级子球体"""
        parent = self._spheres.get(parent_id)
        if not parent:
            return []
        return [
            self._spheres[cid] for cid in parent.child_ids
            if cid in self._spheres and self._spheres[cid].active
        ]

    def get_parent(self, child_id: str) -> Optional[Sphere]:
        """获取二级球体的上级主语球体"""
        child = self._spheres.get(child_id)
        if not child or not child.parent_id:
            return None
        return self._spheres.get(child.parent_id)

    def ids(self) -> List[str]:
        """返回所有活跃球体 ID（按添加顺序）"""
        return [sid for sid, s in self._spheres.items() if s.active]

    # ── 改 ────────────────────────────────────

    def update_mass(self, sphere_id: str, mass: float):
        """更新质量，自动重算 effective_mass"""
        sphere = self._spheres.get(sphere_id)
        if sphere:
            sphere.mass = mass
            sphere._sync_effective_mass()
            self._dirty = True

    def update_diversity(self, sphere_id: str, diversity: float):
        """更新多样性，自动重算 effective_mass"""
        sphere = self._spheres.get(sphere_id)
        if sphere:
            sphere.diversity = diversity
            sphere._sync_effective_mass()
            self._dirty = True

    def update_connection(self, sphere_id: str, target_id: str, weight: float):
        """更新球体之间的连接强度

        连接表在重力空间中用于表示球体之间的引力关系。
        weight > 0 表示吸引，weight < 0 表示抑制。
        """
        sphere = self._spheres.get(sphere_id)
        if sphere:
            sphere.connections[target_id] = weight
            self._dirty = True

    def soft_delete(self, sphere_id: str):
        """软删除球体（active=False，保留元数据）"""
        sphere = self._spheres.get(sphere_id)
        if sphere:
            sphere.active = False
            self._dirty = True

    # ── 持久化 ───────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """保存到 JSON 文件

        Args:
            path: 保存路径，默认 config 中的路径

        Returns:
            保存的文件路径
        """
        save_path = Path(path) if path else self._path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": SPHERE_VERSION,
            "count": len(self._spheres),
            "active_count": self.count,
            "spheres": {
                sid: self._sphere_to_dict(s)
                for sid, s in self._spheres.items()
            },
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._dirty = False
        logger.info(
            f"Saved {self.count} active / {len(self._spheres)} total spheres "
            f"to {save_path}"
        )
        return str(save_path)

    def load(self, path: Optional[str] = None) -> int:
        """从 JSON 文件加载

        Args:
            path: 加载路径，默认 config 中的路径

        Returns:
            加载的球体数量

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 版本不兼容
        """
        load_path = Path(path) if path else self._path

        if not load_path.exists():
            logger.info(f"No sphere store at {load_path}, starting fresh")
            return 0

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", 0)
        if version > SPHERE_VERSION:
            raise ValueError(
                f"Sphere store version {version} > current {SPHERE_VERSION}. "
                "Upgrade required."
            )

        loaded = 0
        for sid, s_dict in data.get("spheres", {}).items():
            sphere = self._dict_to_sphere(s_dict)
            self._spheres[sid] = sphere
            loaded += 1

        self._dirty = False
        logger.info(
            f"Loaded {self.count} active / {loaded} total spheres "
            f"from {load_path}"
        )
        return loaded

    # ── 序列化 ───────────────────────────────

    @staticmethod
    def _sphere_to_dict(sphere: Sphere) -> dict:
        return {
            "id": sphere.id,
            "text": sphere.text,
            "source_file": sphere.source_file,
            "source_type": sphere.source_type,
            "mass": sphere.mass,
            "diversity": sphere.diversity,
            "effective_mass": sphere.effective_mass,
            "connections": sphere.connections,
            "gravity_field": sphere.gravity_field,
            "term_weights": sphere.term_weights,
            "cluster_id": sphere.cluster_id,
            "active": sphere.active,
            "created_at": sphere.created_at,
            "level": sphere.level,
            "parent_id": sphere.parent_id,
            "child_ids": sphere.child_ids,
            "embedding_source": sphere.embedding_source,
            "doc_terms": sphere.doc_terms,
        }

    @staticmethod
    def _dict_to_sphere(d: dict) -> Sphere:
        return Sphere(
            id=d["id"],
            text=d["text"],
            source_file=d.get("source_file", ""),
            source_type=d.get("source_type", ""),
            mass=d.get("mass", 1.0),
            diversity=d.get("diversity", 0.0),
            effective_mass=d.get("effective_mass", 1.0),
            connections=d.get("connections", {}),
            gravity_field=d.get("gravity_field", {}),
            term_weights=d.get("term_weights", {}),
            cluster_id=d.get("cluster_id", -1),
            active=d.get("active", True),
            created_at=d.get("created_at", ""),
            level=d.get("level", 2),
            parent_id=d.get("parent_id", ""),
            child_ids=d.get("child_ids", []),
            embedding_source=d.get("embedding_source", "sentence"),
            doc_terms=d.get("doc_terms", []),
        )


# ──────────────────────────────────────────────
# Sphere ID 生成
# ──────────────────────────────────────────────

def make_sphere_id(text: str, source_file: str = "") -> str:
    """从文本内容生成球体 ID

    二级球体：相同文本 + 相同源文件 = 同一 ID（幂等导入）
    一级球体：使用 make_concept_id（跨源文件一致）
    """
    raw = f"{source_file}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def make_concept_id(concept_text: str) -> str:
    """从概念文本生成一级球体 ID

    不依赖源文件路径，同一概念在不同文档出现时指向同一个 ID。
    前缀添加 "c_" 以与二级球体 ID 区分（便于检索时识别层级）。
    """
    raw = f"concept:{concept_text.strip().lower()}"
    # 使用 MD5[:8] 生成较短的概念 ID
    import hashlib as hl
    return "c_" + hl.md5(raw.encode("utf-8")).hexdigest()[:8]
