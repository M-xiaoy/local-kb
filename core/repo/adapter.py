"""适配器 — 将现有 storage/ 类适配到 core/repo/interfaces 接口

Phase 1 策略：
  · 不重写 storage/ 内部逻辑
  · Adapter 层做接口转换，让 core/kb.py 只依赖 interfaces
  · 后续 Phase 2 逐步将实现迁移到 repo/ 下，替代旧 storage/

日志规则（"不将就"约束）：
  · 每次写入操作必须记录结构化 INFO 日志
  · 异常必须 propagate（禁止 try/except 静默）
  · poincare_norm 作为一等字段处理
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from core.repo.interfaces import (
    KnowledgeBaseRepository,
    SearchResult,
    SphereData,
)
from storage.sphere_store import SphereStore, Sphere
from storage.faiss_store import FaissStore
from storage.registry import Registry

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 适配器
# ──────────────────────────────────────────────

class AdapterRepository(KnowledgeBaseRepository):
    """将现有 storage/ 类适配到 KnowledgeBaseRepository 接口

    组合 SphereStore + FaissStore + Registry，
    对外暴露统一接口。内部 POJOs 使用现存的 Sphere 数据类。
    """

    def __init__(self, sphere_store: SphereStore,
                 faiss_store: FaissStore, registry: Registry):
        self._sphere = sphere_store
        self._faiss = faiss_store
        self._registry = registry

    # ── Sphere 元数据 ─────────────────────────

    def get(self, sphere_id: str) -> Optional[SphereData]:
        sphere = self._sphere.get(sphere_id)
        return self._to_data(sphere) if sphere else None

    def get_many(self, sphere_ids: List[str]) -> List[SphereData]:
        spheres = self._sphere.get_many(sphere_ids)
        return [self._to_data(s) for s in spheres]

    def get_active(self) -> List[SphereData]:
        return [self._to_data(s) for s in self._sphere.get_active()]

    def get_by_source(self, source_file: str) -> List[SphereData]:
        return [self._to_data(s) for s in self._sphere.get_by_source(source_file)]

    def get_by_type(self, source_type: str) -> List[SphereData]:
        return [self._to_data(s) for s in self._sphere.get_by_type(source_type)]

    def add(self, sphere: SphereData) -> bool:
        if not sphere.id:
            raise ValueError("SphereData.id must not be empty")
        internal = self._from_data(sphere)
        added = self._sphere.add(internal)
        if added:
            logger.info(
                f"SphereRepository.add: id={sphere.id[:8]} "
                f"source={sphere.source_file} type={sphere.source_type}"
            )
        return added

    def add_many(self, spheres: List[SphereData]) -> int:
        internals = [self._from_data(s) for s in spheres]
        added = self._sphere.add_many(internals)
        if added:
            logger.info(f"SphereRepository.add_many: {added} new / {len(spheres)} total")
        return added

    def remove(self, sphere_id: str):
        sphere = self._sphere.get(sphere_id)
        if not sphere:
            raise KeyError(f"Sphere {sphere_id} not found")
        self._sphere.soft_delete(sphere_id)
        logger.info(f"SphereRepository.remove: id={sphere_id[:8]} (soft delete)")

    # ── Poincaré norm ─────────────────────────

    def get_poincare_norm(self, sphere_id: str) -> Optional[float]:
        sphere = self._sphere.get(sphere_id)
        if not sphere:
            return None
        return sphere.poincare_norm

    def set_poincare_norm(self, sphere_id: str, norm: float,
                          source: str = "explicit"):
        if not (0.05 <= norm <= 0.90):
            raise ValueError(
                f"Poincaré norm {norm} outside [0.05, 0.90]"
            )
        sphere = self._sphere.get(sphere_id)
        if not sphere:
            raise KeyError(f"Sphere {sphere_id} not found")
        sphere.poincare_norm = norm
        sphere.poincare_norm_source = source
        self._sphere._dirty = True
        logger.info(
            f"SphereRepository.set_poincare_norm: "
            f"id={sphere_id[:8]} norm={norm:.4f} source={source}"
        )

    # ── 质量 / 多样性 ────────────────────────

    def update_mass(self, sphere_id: str, mass: float):
        self._sphere.update_mass(sphere_id, mass)
        logger.info(
            f"SphereRepository.update_mass: id={sphere_id[:8]} mass={mass:.2f}"
        )

    def update_diversity(self, sphere_id: str, diversity: float):
        self._sphere.update_diversity(sphere_id, diversity)
        logger.info(
            f"SphereRepository.update_diversity: "
            f"id={sphere_id[:8]} diversity={diversity:.4f}"
        )

    # ── 连接 ──────────────────────────────────

    def get_connections(self, sphere_id: str) -> Dict[str, float]:
        sphere = self._sphere.get(sphere_id)
        if not sphere:
            raise KeyError(f"Sphere {sphere_id} not found")
        return dict(sphere.connections)

    def set_connection(self, sphere_id: str, target_id: str,
                       weight: float):
        self._sphere.update_connection(sphere_id, target_id, weight)
        logger.debug(
            f"SphereRepository.set_connection: "
            f"{sphere_id[:8]} → {target_id[:8]} weight={weight:.3f}"
        )

    def degree(self, sphere_id: str) -> int:
        sphere = self._sphere.get(sphere_id)
        if not sphere:
            raise KeyError(f"Sphere {sphere_id} not found")
        return len(sphere.connections)

    # ── 持久化 ───────────────────────────────

    def save(self):
        self._sphere.save()
        self._faiss.save()
        self._registry.save()
        logger.info(
            f"AdapterRepository.save: "
            f"{self._sphere.count} spheres, {self._faiss.count} vectors"
        )

    def load(self):
        self._load_spheres()
        self._load_faiss()
        logger.info(
            f"AdapterRepository.load: "
            f"{self._sphere.count} spheres, {self._faiss.count} vectors"
        )

    def _load_spheres(self):
        self._sphere.load()
        self._registry.load()
        # 校验 registry 与 sphere 一致性
        active_ids = {s.id for s in self._sphere.get_active()}
        orphans = self._registry.verify(active_ids)
        if orphans:
            logger.warning(
                f"Registry has {len(orphans)} orphans not in sphere store "
                f"(first 5: {[o[:8] for o in orphans[:5]]})"
            )

    def _load_faiss(self):
        self._faiss.load()

    # ── 统计 ──────────────────────────────────

    def count(self) -> int:
        return self._sphere.count

    def total_count(self) -> int:
        return self._sphere.total_count

    # ── 向量操作 ─────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> SearchResult:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = np.ascontiguousarray(
            query_vector.astype(np.float32)
        )

        ids, distances, vectors = self._faiss.search(query_vector, top_k)

        sphere_ids = []
        valid_vectors = []
        for idx, fid in enumerate(ids):
            sid = self._registry.sphere_id(int(fid))
            if sid:
                sphere_ids.append(sid)
                if idx < len(vectors):
                    valid_vectors.append(vectors[idx])

        vec_array = None
        if valid_vectors:
            vec_array = np.stack(valid_vectors, axis=0)

        return SearchResult(
            sphere_ids=sphere_ids,
            distances=distances.tolist()[:len(sphere_ids)],
            scores=[float(d) for d in distances[:len(sphere_ids)]],
            vectors=vec_array,
        )

    def add_vector(self, sphere_id: str, vector: np.ndarray):
        fid = self._registry.register(sphere_id)
        vector = np.ascontiguousarray(
            vector.flatten().astype(np.float32).reshape(1, -1)
        )
        ids = np.array([fid], dtype=np.int64)
        self._faiss.add(vector, ids)

    def add_vectors(self, sphere_ids: List[str],
                    vectors: np.ndarray):
        if len(sphere_ids) != vectors.shape[0]:
            raise ValueError(
                f"{len(sphere_ids)} IDs but {vectors.shape[0]} vectors"
            )
        fids = self._registry.register_many(sphere_ids)
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        ids = np.array(fids, dtype=np.int64)
        self._faiss.add(vectors, ids)

    def get_vector(self, sphere_id: str) -> Optional[np.ndarray]:
        fid = self._registry.faiss_id(sphere_id)
        if fid is None:
            return None
        return self._faiss._vectors.get(fid)

    def remove_vector(self, sphere_id: str):
        fid = self._registry.faiss_id(sphere_id)
        if fid is not None:
            self._faiss.remove_ids(np.array([fid], dtype=np.int64))
            self._registry.unregister(sphere_id)

    def dim(self) -> int:
        return self._faiss.dim

    def clear(self):
        self._faiss = FaissStore(self._faiss.dim)
        self._registry.clear()

    # ── 列表/元数据 (Phase 4: 门面模式) ─────

    def list_ids(self, limit: int = 100, offset: int = 0) -> List[str]:
        """分页列出活跃球体 ID

        Args:
            limit: 最大返回数
            offset: 跳过前 N 个

        Returns:
            球体 ID 列表（按添加顺序）
        """
        all_ids = self._sphere.ids()
        return all_ids[offset:offset + limit]

    def get_metadata(self, sphere_id: str) -> Optional[dict]:
        """获取球体元数据（含半径）

        Returns:
            dict 或 None
        """
        sphere = self._sphere.get(sphere_id)
        if not sphere:
            return None
        return {
            "id": sphere.id,
            "source_file": sphere.source_file,
            "source_type": sphere.source_type,
            "mass": sphere.mass,
            "diversity": sphere.diversity,
            "effective_mass": sphere.effective_mass,
            "cluster_id": sphere.cluster_id,
            "active": sphere.active,
            "created_at": sphere.created_at,
            "level": sphere.level,
            "parent_id": sphere.parent_id,
            "poincare_norm": sphere.poincare_norm,
            "poincare_norm_source": sphere.poincare_norm_source,
        }

    def delete_sphere(self, sphere_id: str) -> bool:
        """删除球体（软删除 + 清理注册）

        Raises:
            KeyError: sphere_id 不存在
        """
        sphere = self._sphere.get(sphere_id)
        if not sphere:
            raise KeyError(f"Sphere {sphere_id[:8]} not found")
        self._sphere.soft_delete(sphere_id)
        self._registry.unregister(sphere_id)
        # 清理 FAISS 向量
        fid = self._registry.faiss_id(sphere_id)
        if fid is not None:
            self._faiss.remove_ids(np.array([fid], dtype=np.int64))
        logger.info(f"AdapterRepository.delete_sphere: id={sphere_id[:8]}")
        return True

    def delete_poincare_norm(self, sphere_id: str):
        """重置范数为默认值

        Raises:
            KeyError: sphere_id 不存在
        """
        sphere = self._sphere.get(sphere_id)
        if not sphere:
            raise KeyError(f"Sphere {sphere_id[:8]} not found")
        sphere.poincare_norm = 0.5
        sphere.poincare_norm_source = "default"
        self._sphere._dirty = True
        logger.info(f"AdapterRepository.delete_poincare_norm: id={sphere_id[:8]}")

    def delete_edges(self, sphere_id: str):
        """删除球体的所有连接边

        Raises:
            KeyError: sphere_id 不存在
        """
        sphere = self._sphere.get(sphere_id)
        if not sphere:
            raise KeyError(f"Sphere {sphere_id[:8]} not found")
        sphere.connections.clear()
        self._sphere._dirty = True
        logger.debug(f"AdapterRepository.delete_edges: id={sphere_id[:8]}")

    def add_edge(self, from_id: str, to_id: str,
                 weight: float = 1.0) -> bool:
        """添加连接边

        Raises:
            KeyError: from_id 或 to_id 不存在
        """
        from_s = self._sphere.get(from_id)
        to_s = self._sphere.get(to_id)
        if not from_s or not to_s:
            raise KeyError("Source or target sphere not found")
        from_s.connections[to_id] = weight
        self._sphere._dirty = True
        logger.debug(f"AdapterRepository.add_edge: {from_id[:8]} → {to_id[:8]} w={weight}")
        return True

    # ── 转换辅助 ─────────────────────────────

    @staticmethod
    def _to_data(sphere: Sphere) -> SphereData:
        return SphereData(
            id=sphere.id,
            text=sphere.text,
            source_file=sphere.source_file,
            source_type=sphere.source_type,
            mass=sphere.mass,
            diversity=sphere.diversity,
            effective_mass=sphere.effective_mass,
            cluster_id=sphere.cluster_id,
            active=sphere.active,
            created_at=sphere.created_at,
            level=sphere.level,
            parent_id=sphere.parent_id,
            child_ids=list(sphere.child_ids),
            embedding_source=sphere.embedding_source,
            doc_terms=list(sphere.doc_terms),
            poincare_norm=sphere.poincare_norm,
            poincare_norm_source=sphere.poincare_norm_source,
        )

    @staticmethod
    def _from_data(data: SphereData) -> Sphere:
        return Sphere(
            id=data.id,
            text=data.text,
            source_file=data.source_file,
            source_type=data.source_type,
            mass=data.mass,
            diversity=data.diversity,
            effective_mass=data.effective_mass,
            cluster_id=data.cluster_id,
            active=data.active,
            created_at=data.created_at,
            level=data.level,
            parent_id=data.parent_id,
            child_ids=data.child_ids,
            embedding_source=data.embedding_source,
            doc_terms=data.doc_terms,
            poincare_norm=data.poincare_norm,
            poincare_norm_source=data.poincare_norm_source,
        )
