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

SPHERE_VERSION = 1  # 用于未来迁移


# ──────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────

@dataclass
class Sphere:
    """单个球体——文本切片的元数据 + 重力空间字段"""
    id: str                          # 唯一标识（SHA256[:12]）
    text: str                        # 原文片段
    source_file: str                 # 源文件名
    source_type: str = ""            # 场域标签
    mass: float = 1.0                # 基础质量
    diversity: float = 0.0           # 多样性得分（来源分布广度）
    effective_mass: float = 1.0      # mass × (1 + diversity)
    connections: Dict[str, float] = field(default_factory=dict)
    active: bool = True              # 软删除标记
    created_at: str = ""             # 入库时间（ISO 格式）

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
            "active": sphere.active,
            "created_at": sphere.created_at,
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
            active=d.get("active", True),
            created_at=d.get("created_at", ""),
        )


# ──────────────────────────────────────────────
# Sphere ID 生成
# ──────────────────────────────────────────────

def make_sphere_id(text: str, source_file: str = "") -> str:
    """从文本内容生成确定的球体 ID

    使用 SHA256 的前 12 位十六进制字符（48 bits），
    冲突概率在千亿级别，足够个人知识库使用。

    同一文本 + 同一源文件 = 同一 ID → 幂等导入。
    """
    raw = f"{source_file}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
