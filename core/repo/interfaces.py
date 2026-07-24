"""repository 抽象接口

定义核心数据访问协议，与具体存储实现解耦。
Phase 1 目标：接口先行，实现向后兼容现有 storage/ 代码。

每条接口方法必须：
  1. 明确返回值类型
  2. 明确异常条件
  3. 不静默吞异常
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# 数据类型
# ──────────────────────────────────────────────

@dataclass
class SphereData:
    """球体核心数据（与序列化解耦）

    用于 repo 接口的统一传输格式。
    所有数值字段都有明确语义，不允许 None。
    """
    id: str
    text: str
    source_file: str
    source_type: str = ""
    mass: float = 1.0
    diversity: float = 0.0
    effective_mass: float = 1.0
    cluster_id: int = -1
    active: bool = True
    created_at: str = ""
    level: int = 2
    parent_id: str = ""
    child_ids: List[str] = field(default_factory=list)
    embedding_source: str = "sentence"
    doc_terms: List[str] = field(default_factory=list)
    poincare_norm: float = 0.5
    poincare_norm_source: str = "default"

    @property
    def has_valid_norm(self) -> bool:
        """范数是否经过推导（不是默认占位值）"""
        return self.poincare_norm_source not in ("default", "")


@dataclass
class SearchResult:
    """向量搜索结果"""
    sphere_ids: List[str]          # 按相关性降序排列
    distances: List[float]         # 余弦/IP 距离
    scores: List[float]            # 最终排序得分
    vectors: Optional[np.ndarray] = None  # shape (n, dim)


# ──────────────────────────────────────────────
# Repository 接口
# ──────────────────────────────────────────────

class SphereRepository(ABC):
    """球体元数据存储接口

    管理 Sphere 数据的 CRUD，包括 Poincaré 半径、质量、连接度。
    连接数据作为球体的属性存储（参照当前 Sphere.connections），
    不单独建边表——千级规模下够用。
    """

    @abstractmethod
    def get(self, sphere_id: str) -> Optional[SphereData]:
        """获取单个球体

        Returns:
            SphereData 或 None（ID 不存在时）
        """
        ...

    @abstractmethod
    def get_many(self, sphere_ids: List[str]) -> List[SphereData]:
        """批量获取，自动跳过不存在的 ID"""
        ...

    @abstractmethod
    def get_active(self) -> List[SphereData]:
        """返回所有活跃球体"""
        ...

    @abstractmethod
    def get_by_source(self, source_file: str) -> List[SphereData]:
        """按源文件名查找"""
        ...

    @abstractmethod
    def get_by_type(self, source_type: str) -> List[SphereData]:
        """按场域类型查找"""
        ...

    @abstractmethod
    def add(self, sphere: SphereData) -> bool:
        """添加球体。

        Returns:
            True 新添加，False 已存在（幂等）

        Raises:
            ValueError: sphere.id 为空
        """
        ...

    @abstractmethod
    def add_many(self, spheres: List[SphereData]) -> int:
        """批量添加，返回实际新增数量"""
        ...

    @abstractmethod
    def remove(self, sphere_id: str):
        """软删除球体（保留元数据）

        Raises:
            KeyError: sphere_id 不存在
        """
        ...

    # ── Poincaré norm（核心差异点） ──────────

    @abstractmethod
    def get_poincare_norm(self, sphere_id: str) -> Optional[float]:
        """获取 Poincaré 范数

        Returns:
            float 或 None（sphere 不存在或没有范数）
        """
        ...

    @abstractmethod
    def set_poincare_norm(self, sphere_id: str, norm: float,
                          source: str = "explicit"):
        """设置 Poincaré 范数

        Args:
            sphere_id: 球体 ID
            norm: 范数值 [0.05, 0.90]
            source: 来源标记（"explicit" / "mass" / "hierarchy" / "hubness"）

        Raises:
            KeyError: sphere_id 不存在
            ValueError: norm 超出 [0.05, 0.90]
        """
        ...

    # ── 质量 / 多样性 ────────────────────────

    @abstractmethod
    def update_mass(self, sphere_id: str, mass: float):
        """更新质量，自动重算 effective_mass"""
        ...

    @abstractmethod
    def update_diversity(self, sphere_id: str, diversity: float):
        """更新多样性，自动重算 effective_mass"""
        ...

    # ── 连接 ──────────────────────────────────

    @abstractmethod
    def get_connections(self, sphere_id: str) -> Dict[str, float]:
        """获取球体的所有连接

        Returns:
            {target_sphere_id: weight}，可能为空字典

        Raises:
            KeyError: sphere_id 不存在
        """
        ...

    @abstractmethod
    def set_connection(self, sphere_id: str, target_id: str,
                       weight: float):
        """设置（或更新）球体之间的连接强度

        weight > 0 表示吸引，weight < 0 表示抑制。
        """
        ...

    @abstractmethod
    def degree(self, sphere_id: str) -> int:
        """获取球体的连接度（边数）

        Raises:
            KeyError: sphere_id 不存在
        """
        ...

    # ── 持久化 ───────────────────────────────

    @abstractmethod
    def save(self):
        """持久化全部数据

        Raises:
            IOError: 写入失败
        """
        ...

    @abstractmethod
    def load(self):
        """从持久化恢复

        Raises:
            IOError: 读取失败或格式不兼容
        """
        ...

    # ── 列表 / 元数据（Phase 4） ────────────

    @abstractmethod
    def list_ids(self, limit: int = 100, offset: int = 0) -> List[str]:
        """分页列出活跃球体 ID"""
        ...

    @abstractmethod
    def get_metadata(self, sphere_id: str) -> Optional[dict]:
        """获取球体元数据（含半径等信息）"""
        ...

    @abstractmethod
    def delete_sphere(self, sphere_id: str) -> bool:
        """删除球体，同步清理注册/向量/边

        Raises:
            KeyError: sphere_id 不存在
        """
        ...

    @abstractmethod
    def delete_poincare_norm(self, sphere_id: str):
        """重置范数为默认值"""
        ...

    @abstractmethod
    def delete_edges(self, sphere_id: str):
        """删除球体的所有连接边"""
        ...

    @abstractmethod
    def add_edge(self, from_id: str, to_id: str,
                 weight: float = 1.0) -> bool:
        """添加连接边"""
        ...

    # ── 统计 ──────────────────────────────────

    @abstractmethod
    def count(self) -> int:
        """活跃球体数量"""
        ...

    @abstractmethod
    def total_count(self) -> int:
        """全部球体（含软删除）"""
        ...


class VectorRepository(ABC):
    """向量索引接口（FAISS 封装）

    输入输出全用 sphere_id（字符串），内部处理 faiss_id 映射。
    对外部调用者屏蔽 FAISS 细节。
    """

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 100) -> SearchResult:
        """向量检索 Top-K

        Args:
            query_vector: shape (dim,) float32，已归一化
            top_k: 返回数量

        Returns:
            SearchResult（按 IP 距离降序）

        Raises:
            RuntimeError: 索引为空
        """
        ...

    @abstractmethod
    def add_vector(self, sphere_id: str, vector: np.ndarray):
        """添加或更新一个向量

        Args:
            sphere_id: 球体 ID
            vector: shape (dim,) float32，已归一化

        Raises:
            ValueError: 维度不匹配或类型错误
        """
        ...

    @abstractmethod
    def add_vectors(self, sphere_ids: List[str],
                    vectors: np.ndarray):
        """批量添加向量

        Args:
            sphere_ids: [n] 球体 ID 列表
            vectors: shape (n, dim) float32

        Raises:
            ValueError: 数量或维度不匹配
        """
        ...

    @abstractmethod
    def get_vector(self, sphere_id: str) -> Optional[np.ndarray]:
        """获取单个向量

        Returns:
            shape (dim,) float32 或 None（不存在）
        """
        ...

    @abstractmethod
    def remove_vector(self, sphere_id: str):
        """移除向量"""
        ...

    @abstractmethod
    def count(self) -> int:
        """索引中的向量数"""
        ...

    @abstractmethod
    def dim(self) -> int:
        """向量维度"""
        ...

    @abstractmethod
    def clear(self):
        """清空索引"""
        ...

    @abstractmethod
    def save(self):
        """持久化索引

        Raises:
            IOError: 写入失败
        """
        ...

    @abstractmethod
    def load(self):
        """加载持久化的索引

        Raises:
            IOError: 读取失败
        """
        ...


# ──────────────────────────────────────────────
# Unified Repository（方便使用时的组合接口）
# ──────────────────────────────────────────────

class KnowledgeBaseRepository(SphereRepository, VectorRepository):
    """完整知识库存储接口

    组合了球体元数据和向量索引，是 core/kb.py 的唯一存储依赖。
    实现类必须同时实现两者。
    """
    pass
