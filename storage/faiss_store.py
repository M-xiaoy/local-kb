"""
faiss_store.py — FAISS 向量索引
================================
管理稠密向量的 ANN 搜索，返回 Top-K 向量 ID。
与 sphere_store（元数据层）和 registry（ID 映射）配合使用。

FAISS 的职责：
  给定查询向量 → 返回 Top-100 的 (distance, faiss_id)
  FAISS 不知道 metadata、场域、连接表——这些是 sphere_store 的活。

我们的差异点：
  1. FAISS 只是快筛子——智能路由在上层检索层
  2. IndexFlatIP + L2 归一化 = cosine similarity
  3. IndexIDMap 提供稳定 ID（支持删除和重建）
  4. float32 显式转型（不踩 NumPy float64 默认值的坑）

索引类型选择（当前+未来）：
  < 100k  → IndexFlatIP（精确，无参）
  < 1M    → IndexIVFFlat（聚类加速）
  < 100M  → IndexHNSW（图导航）
  我们的个人 KB 在千级，Flat 是最优解。
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu not installed. Run: pip install faiss-cpu"
    )

from config import ollama as cfg_ollama, paths as cfg_paths

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# FAISS 索引包装器
# ──────────────────────────────────────────────

class FaissStore:
    """FAISS 向量索引

    职责边界：
      · 接收已归一化的 float32 向量
      · 接收 int64 类型 ID（与 registry 的 faiss_id 对应）
      · search() 返回 (faiss_ids, distances, vectors) — 不碰元数据

    向量缓存：
      由于 IndexIDMap 不支持 reconstruct，
      FaissStore 维护一份 {faiss_id: vector} 缓存。
      这是众多 FAISS 项目（包括 LangChain）的标准做法。

    与 registry 的关系：
      faiss_id (int64) → registry → sphere_id (str) → Sphere
    """

    def __init__(self, dim: Optional[int] = None):
        self.dim = dim or cfg_ollama.embed_dim
        self._index: Optional[faiss.Index] = None
        self._vectors: Dict[int, np.ndarray] = {}  # faiss_id → vector

    # ── 属性 ──────────────────────────────────

    @property
    def count(self) -> int:
        """索引中的向量数量"""
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def is_built(self) -> bool:
        return self._index is not None and self.count > 0

    # ── 构建 / 添加 ───────────────────────────

    def build(self, vectors: np.ndarray, ids: np.ndarray):
        """从零构建索引

        Args:
            vectors: shape (n, dim), dtype float32, 已归一化
            ids:     shape (n,), dtype int64, FAISS 内部 ID
        """
        self._validate_inputs(vectors, ids)

        base_index = faiss.IndexFlatIP(self.dim)
        self._index = faiss.IndexIDMap(base_index)
        self._index.add_with_ids(vectors, ids)

        # 更新向量缓存
        self._vectors.clear()
        for i in range(len(ids)):
            self._vectors[int(ids[i])] = vectors[i].copy()

        logger.info(f"Built index with {self.count} vectors (dim={self.dim})")

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        """向现有索引追加向量

        Args:
            vectors: shape (n, dim), dtype float32, 已归一化
            ids:     shape (n,), dtype int64
        """
        if self._index is None:
            self.build(vectors, ids)
            return

        self._validate_inputs(vectors, ids)
        self._index.add_with_ids(vectors, ids)

        # 更新向量缓存
        for i in range(len(ids)):
            self._vectors[int(ids[i])] = vectors[i].copy()

        logger.info(f"Added {len(ids)} vectors (total: {self.count})")

    # ── 搜索 ──────────────────────────────────

    def search(
        self, query_vector: np.ndarray, top_k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """搜索 Top-K 最近邻，返回 ID、距离和向量

        Args:
            query_vector: shape (dim,) or (1, dim), float32, 已归一化
            top_k: 返回数量

        Returns:
            (ids, distances, vectors):
              ids:        shape (top_k,), int64 — FAISS 内部 ID
              distances:  shape (top_k,), float32 — IP 距离
              vectors:    shape (top_k, dim), float32 — 对应的向量

        Raises:
            RuntimeError: 索引为空或未构建
        """
        if not self.is_built:
            raise RuntimeError(
                "FAISS index is empty. Add vectors before searching."
            )

        # 确保是 2D 且类型正确
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = np.ascontiguousarray(
            query_vector.astype(np.float32)
        )

        # top_k 不超过索引大小
        k = min(top_k, self.count)

        distances, ids = self._index.search(query_vector, k)

        # 从缓存中获取对应向量（容缺：缓存中没有的向量用零向量填充）
        valid_vecs = []
        valid_ids = []
        valid_dists = []
        for fid, dist in zip(ids[0], distances[0]):
            vec = self._vectors.get(int(fid))
            if vec is not None:
                valid_vecs.append(vec)
                valid_ids.append(fid)
                valid_dists.append(dist)

        if valid_vecs:
            vecs = np.vstack(valid_vecs)
        else:
            vecs = np.zeros((0, self.dim), dtype=np.float32)

        return np.array(valid_ids, dtype=np.int64), np.array(valid_dists, dtype=np.float32), vecs

    # ── 删除 ──────────────────────────────────

    def remove_ids(self, ids_to_remove: np.ndarray):
        """从索引中移除指定 ID 的向量

        Args:
            ids_to_remove: shape (n,), int64 — 要删除的 FAISS ID
        """
        if self._index is None or len(ids_to_remove) == 0:
            return

        id_selector = faiss.IDSelectorArray(ids_to_remove)
        removed = self._index.remove_ids(id_selector)

        # 清理向量缓存
        for fid in ids_to_remove:
            self._vectors.pop(int(fid), None)

        logger.info(f"Removed {removed} vectors (remaining: {self.count})")

    # ── 向量缓存持久化 ───────────────────────
    #
    # FAISS IndexIDMap 不支持 reconstruct（从索引中还原向量），
    # 所以须在索引外单独存一份 {faiss_id → vector} 缓存。
    # 格式：单 .npz 文件，内含 ids(int64) 和 vectors(float32) 两数组。

    def _cache_path(self) -> Path:
        return Path(cfg_paths.faiss_cache)

    def save_cache(self):
        """将 _vectors 缓存保存到 .npz 文件"""
        if not self._vectors:
            return
        cache_path = self._cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        ids = np.array(list(self._vectors.keys()), dtype=np.int64)
        vecs = np.stack(list(self._vectors.values()), axis=0)
        np.savez_compressed(str(cache_path), ids=ids, vectors=vecs)
        logger.info(f"Saved vector cache: {len(ids)} vectors → {cache_path}")

    def load_cache(self):
        """从 .npz 文件恢复 _vectors 缓存"""
        cache_path = self._cache_path()
        if not cache_path.exists():
            logger.info(f"No vector cache at {cache_path}")
            return

        data = np.load(str(cache_path))
        ids = data["ids"]
        vecs = data["vectors"]
        self._vectors = {
            int(fid): vecs[i].copy()
            for i, fid in enumerate(ids)
        }
        logger.info(f"Loaded vector cache: {len(self._vectors)} vectors ← {cache_path}")

    # ── 持久化 ───────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """保存索引到 .index 文件

        Args:
            path: 保存路径，默认 config 中的路径

        Returns:
            保存的文件路径
        """
        if self._index is None:
            raise RuntimeError("No index to save")

        save_path = Path(path) if path else Path(cfg_paths.faiss_index)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(save_path))
        self.save_cache()
        logger.info(f"Saved FAISS index to {save_path} ({self.count} vectors)")
        return str(save_path)

    def load(self, path: Optional[str] = None) -> int:
        """从 .index 文件加载索引

        Args:
            path: 加载路径，默认 config 中的路径

        Returns:
            加载的向量数量

        Raises:
            FileNotFoundError: 文件不存在
        """
        load_path = Path(path) if path else Path(cfg_paths.faiss_index)

        if not load_path.exists():
            logger.info(f"No FAISS index at {load_path}, starting fresh")
            return 0

        self._index = faiss.read_index(str(load_path))
        self.load_cache()
        logger.info(f"Loaded FAISS index from {load_path} ({self.count} vectors)")
        return self.count

    # ── 内部方法 ─────────────────────────────

    def _validate_inputs(self, vectors: np.ndarray, ids: np.ndarray):
        """校验输入向量的维度和类型"""
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(
                f"vectors ({vectors.shape[0]}) and ids ({ids.shape[0]}) "
                "must have same length"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != expected {self.dim}"
            )
        if vectors.dtype != np.float32:
            raise ValueError(
                f"Vectors must be float32, got {vectors.dtype}. "
                "Cast with vectors.astype(np.float32)"
            )
        if ids.dtype != np.int64:
            raise ValueError(
                f"IDs must be int64, got {ids.dtype}. "
                "Cast with ids.astype(np.int64)"
            )
