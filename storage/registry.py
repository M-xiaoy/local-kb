"""
registry.py — 双向 ID 映射
===========================
FAISS 使用 int64 内部 ID，sphere_store 使用 string 类型 sphere_id。
Registry 负责在这两种 ID 格式之间做透明转换。

设计决策（基于对 LangChain / bigRAG / 业界模式的调研）：

  1. 双向字典：faiss_id (int64) ↔ sphere_id (str)
     类似 LangChain 的 index_to_docstore_id + docstore 配对

  2. FAISS ID 直接使用 sphere_id 的整数形式
     sphere_id = SHA256(text+source)[:12] → 12 位 hex → int64
     好处：同一内容每次重建索引 ID 一致，registry 状态可恢复

  3. 持久化用 JSON（与 sphere_store 一致）
     保存时序列化双向映射 + next_id 计数器

  4. 删除时双向同步清理
     软删除在 sphere_store，registry 等 FAISS 真正移除后清理

使用流程：
  # 入库
  fid = registry.resolve(sphere_id)   # 已存在返回旧ID，否则分配新ID
  faiss_store.add(vectors, np.array([fid]))

  # 检索
  faiss_ids, distances = faiss_store.search(query_vec, k=100)
  sphere_ids = [registry.sphere_id(fid) for fid in faiss_ids]
  spheres = sphere_store.get_many(sphere_ids)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from config import paths as cfg_paths

logger = logging.getLogger(__name__)

REGISTRY_VERSION = 1


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

class Registry:
    """FAISS ID ↔ sphere_id 双向映射

    FAISS 端：int64（sphere_id 的整数形式）
    sphere_store 端：str（12位 hex）
    """

    def __init__(self, storage_path: Optional[str] = None):
        self._path = Path(storage_path) if storage_path else Path(cfg_paths.registry_map)
        # 双向映射
        self._f2s: Dict[int, str] = {}   # faiss_id → sphere_id
        self._s2f: Dict[str, int] = {}   # sphere_id → faiss_id
        self._dirty = False

    # ── 核心转换 ──────────────────────────────

    @staticmethod
    def sphere_id_to_faiss_id(sphere_id: str) -> int:
        """将 sphere_id（12位 hex）转为 FAISS int64 ID"""
        return int(sphere_id, 16)

    @staticmethod
    def faiss_id_to_sphere_id(faiss_id: int) -> str:
        """将 FAISS int64 ID 转回 sphere_id（12位 hex）"""
        return format(faiss_id, "012x")

    # ── 注册 / 解析 ──────────────────────────

    def register(self, sphere_id: str) -> int:
        """注册一个 sphere_id，返回对应的 faiss_id

        如果已存在 → 返回现有 faiss_id
        如果不存在 → 从 sphere_id 派生 faiss_id
        """
        if sphere_id in self._s2f:
            return self._s2f[sphere_id]

        fid = self.sphere_id_to_faiss_id(sphere_id)
        self._f2s[fid] = sphere_id
        self._s2f[sphere_id] = fid
        self._dirty = True
        return fid

    def register_many(self, sphere_ids: List[str]) -> List[int]:
        """批量注册，返回对应的 faiss_id 列表"""
        return [self.register(sid) for sid in sphere_ids]

    # ── 查询 ──────────────────────────────────

    def sphere_id(self, faiss_id: int) -> Optional[str]:
        """faiss_id → sphere_id"""
        return self._f2s.get(faiss_id)

    def faiss_id(self, sphere_id: str) -> Optional[int]:
        """sphere_id → faiss_id"""
        return self._s2f.get(sphere_id)

    def get_all_sphere_ids(self) -> List[str]:
        """返回所有已注册的 sphere_id"""
        return list(self._s2f.keys())

    def get_all_faiss_ids(self) -> List[int]:
        """返回所有已注册的 faiss_id"""
        return list(self._f2s.keys())

    @property
    def count(self) -> int:
        return len(self._f2s)

    # ── 撤销注册 ─────────────────────────────

    def unregister(self, sphere_id: str):
        """移除一个 sphere_id 的注册"""
        fid = self._s2f.pop(sphere_id, None)
        if fid is not None:
            self._f2s.pop(fid, None)
            self._dirty = True

    def unregister_many(self, sphere_ids: List[str]):
        """批量移除"""
        for sid in sphere_ids:
            self.unregister(sid)

    def clear(self):
        """清空全部注册"""
        self._f2s.clear()
        self._s2f.clear()
        self._dirty = True

    # ── 持久化 ───────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """保存到 JSON"""
        save_path = Path(path) if path else self._path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": REGISTRY_VERSION,
            "count": self.count,
            "mapping": [
                {"faiss_id": fid, "sphere_id": sid}
                for fid, sid in sorted(self._f2s.items())
            ],
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._dirty = False
        logger.info(f"Saved {self.count} mappings to {save_path}")
        return str(save_path)

    def load(self, path: Optional[str] = None) -> int:
        """从 JSON 加载"""
        load_path = Path(path) if path else self._path

        if not load_path.exists():
            logger.info(f"No registry at {load_path}, starting fresh")
            return 0

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", 0)
        if version > REGISTRY_VERSION:
            raise ValueError(
                f"Registry version {version} > current {REGISTRY_VERSION}. "
                "Upgrade required."
            )

        for entry in data.get("mapping", []):
            fid = entry["faiss_id"]
            sid = entry["sphere_id"]
            self._f2s[fid] = sid
            self._s2f[sid] = fid

        self._dirty = False
        logger.info(f"Loaded {self.count} mappings from {load_path}")
        return self.count

    # ── 校验 ──────────────────────────────────

    def verify(self, expected_sphere_ids: Set[str]) -> List[str]:
        """校验 registry 与 sphere_store 的一致性

        返回不在 sphere_store 中但 registry 里有残留的 sphere_id。
        """
        orphans = []
        for sid in self.get_all_sphere_ids():
            if sid not in expected_sphere_ids:
                orphans.append(sid)
        return orphans
