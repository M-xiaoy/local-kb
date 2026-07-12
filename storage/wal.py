"""
wal.py — Write-Ahead Log 预写日志
==================================
确保上传操作的原子性：要么全部写入磁盘，要么什么都不写。

流程：
  ① 在内存中处理完所有步骤（解析→切片→嵌入→创建球体）
  ② 写 WAL 文件记录「已就绪」（含本次操作的全部 sphere_id）
  ③ 全部持久化到磁盘
  ④ 标记 WAL 为「已完成」→ 删除

启动恢复：
  扫描 WAL 目录 → 找到「已就绪」的条目 → 尝试恢复或清理
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 状态常量
# ──────────────────────────────────────────────

WAL_READY = "ready"        # 已准备好写入磁盘，但还没写
WAL_COMMITTING = "committing"  # 正在写入磁盘中
WAL_DONE = "done"          # 磁盘写入完成
WAL_ROLLED_BACK = "rolled_back"  # 已回滚


# ──────────────────────────────────────────────
# WAL 条目
# ──────────────────────────────────────────────

class WalEntry:
    """一条预写日志"""

    def __init__(self, entry_id: str, data: dict = None):
        self.entry_id = entry_id
        if data:
            self.status = data.get("status", WAL_READY)
            self.file = data.get("file", "")
            self.source_type = data.get("source_type", "")
            self.chunks_total = data.get("chunks_total", 0)
            self.sphere_ids = data.get("sphere_ids", [])
            self.faiss_ids = data.get("faiss_ids", [])
            self.created_at = data.get("created_at", time.time())
            self.updated_at = data.get("updated_at", time.time())
        else:
            self.status = WAL_READY
            self.file = ""
            self.source_type = ""
            self.chunks_total = 0
            self.sphere_ids = []
            self.faiss_ids = []
            self.created_at = time.time()
            self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "status": self.status,
            "file": self.file,
            "source_type": self.source_type,
            "chunks_total": self.chunks_total,
            "sphere_ids": self.sphere_ids,
            "faiss_ids": self.faiss_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ──────────────────────────────────────────────
# WAL 管理器
# ──────────────────────────────────────────────

class WalManager:
    """WAL 管理器，负责创建、提交、回滚、恢复"""

    def __init__(self, wal_dir: str):
        self.wal_dir = Path(wal_dir)
        self.wal_dir.mkdir(parents=True, exist_ok=True)

    # ── 路径 ──────────────────────────────────

    def _path(self, entry_id: str) -> Path:
        return self.wal_dir / f"{entry_id}.wal"

    # ── 原子写入 ─────────────────────────────
    #
    # 先写 .tmp 文件，再 rename 覆盖正式文件。
    # rename 是文件系统的原子操作（同一磁盘内）。
    # 这样 WAL 文件本身不会写半截。

    def _write_atomically(self, path: Path, data: dict):
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        # Windows 上 rename 不能覆盖已有文件
        if path.exists():
            path.unlink()
        tmp_path.rename(path)

    # ── 创建 ──────────────────────────────────

    def create(self, file_name: str, source_type: str,
               sphere_ids: List[str], faiss_ids: List[int],
               chunks_total: int) -> WalEntry:
        """创建一条新的 WAL 条目（状态=ready）"""
        entry_id = uuid.uuid4().hex[:12]
        entry = WalEntry(entry_id)
        entry.file = file_name
        entry.source_type = source_type
        entry.sphere_ids = sphere_ids
        entry.faiss_ids = faiss_ids
        entry.chunks_total = chunks_total
        entry.created_at = time.time()
        entry.updated_at = time.time()
        entry.status = WAL_READY

        self._write_atomically(self._path(entry_id), entry.to_dict())
        logger.info(
            f"WAL[{entry_id[:8]}] created: {file_name} "
            f"({len(sphere_ids)} spheres, status=ready)"
        )
        return entry

    # ── 更新状态 ──────────────────────────────

    def _update(self, entry: WalEntry):
        """更新 WAL 文件（原子写入）"""
        entry.updated_at = time.time()
        self._write_atomically(self._path(entry.entry_id), entry.to_dict())

    def mark_committing(self, entry: WalEntry):
        """标记为正在提交"""
        entry.status = WAL_COMMITTING
        self._update(entry)
        logger.info(f"WAL[{entry.entry_id[:8]}] status→committing")

    def mark_done(self, entry: WalEntry):
        """标记为已完成，然后删除 WAL 文件"""
        entry.status = WAL_DONE
        self._update(entry)
        # 删除 WAL 文件表示操作已安全完成
        wal_path = self._path(entry.entry_id)
        if wal_path.exists():
            wal_path.unlink()
        logger.info(f"WAL[{entry.entry_id[:8]}] done, removed")

    def mark_rolled_back(self, entry: WalEntry):
        """标记为已回滚"""
        entry.status = WAL_ROLLED_BACK
        self._update(entry)
        logger.warning(f"WAL[{entry.entry_id[:8]}] rolled back")

    # ── 加载 ──────────────────────────────────

    def load_entry(self, entry_id: str) -> Optional[WalEntry]:
        """加载一条 WAL 条目"""
        path = self._path(entry_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return WalEntry(entry_id, data)

    def scan(self) -> List[WalEntry]:
        """扫描所有 WAL 条目，按创建时间排序"""
        entries = []
        for f in sorted(self.wal_dir.glob("*.wal")):
            entry_id = f.stem
            entry = self.load_entry(entry_id)
            if entry:
                entries.append(entry)
        return entries

    def find_incomplete(self) -> List[WalEntry]:
        """返回所有未完成的 WAL 条目（ready / committing）"""
        return [e for e in self.scan() if e.status in (WAL_READY, WAL_COMMITTING)]

    # ── 恢复 ──────────────────────────────────

    def recover(self, sphere_store, registry, faiss_store):
        """启动时恢复：处理所有未完成的 WAL 条目

        Args:
            sphere_store: 球体存储（需有 remove 方法）
            registry: 注册表（需有 unregister_by_sphere_id 方法）
            faiss_store: FAISS 存储（需有 remove_ids 方法）

        Returns:
            恢复的记录: {recovered: int, rolled_back: int, errors: int}
        """
        incomplete = self.find_incomplete()
        if not incomplete:
            return {"recovered": 0, "rolled_back": 0, "errors": 0}

        result = {"recovered": 0, "rolled_back": 0, "errors": 0}

        for entry in incomplete:
            try:
                if entry.status == WAL_READY:
                    # 还没开始持久化 → 直接清理
                    logger.info(
                        f"WAL[{entry.entry_id[:8]}] recovery: rolling back "
                        f"ready entry ({len(entry.sphere_ids)} spheres)"
                    )
                    self._rollback(entry, sphere_store, registry, faiss_store)
                    self.mark_rolled_back(entry)
                    result["rolled_back"] += 1

                elif entry.status == WAL_COMMITTING:
                    # 正在持久化时崩溃 → 数据可能写了一半
                    # 安全做法：回滚 + 重建
                    logger.info(
                        f"WAL[{entry.entry_id[:8]}] recovery: rolling back "
                        f"committing entry ({len(entry.sphere_ids)} spheres)"
                    )
                    self._rollback(entry, sphere_store, registry, faiss_store)
                    self.mark_rolled_back(entry)
                    result["rolled_back"] += 1

            except Exception as e:
                logger.error(
                    f"WAL[{entry.entry_id[:8]}] recovery failed: {e}"
                )
                result["errors"] += 1

        return result

    def _rollback(self, entry: WalEntry, sphere_store, registry, faiss_store):
        """回滚一条 WAL 条目：删除对应的球体、注册表条目、FAISS 向量"""
        # 1. 删除球体
        for sphere_id in entry.sphere_ids:
            sphere_store.soft_delete(sphere_id)

        # 2. 删除注册表条目 + 收集 faiss_id
        faiss_ids_to_remove = []
        for idx, sphere_id in enumerate(entry.sphere_ids):
            if idx < len(entry.faiss_ids):
                # 用 WAL 中记录的 faiss_id
                faiss_ids_to_remove.append(entry.faiss_ids[idx])
            # 也尝试从 registry 反查
            fid = registry.faiss_id(sphere_id)
            if fid is not None and fid not in faiss_ids_to_remove:
                faiss_ids_to_remove.append(fid)
            registry.unregister(sphere_id)

        # 3. 删除 FAISS 向量
        if faiss_ids_to_remove:
            import numpy as np
            faiss_store.remove_ids(np.array(faiss_ids_to_remove, dtype=np.int64))

        logger.info(
            f"  Rolled back {len(entry.sphere_ids)} spheres, "
            f"{len(faiss_ids_to_remove)} faiss vectors"
        )

    # ── 清理 ──────────────────────────────────

    def clean_old_entries(self, max_age_hours: int = 24):
        """清理超过指定时间的已完成/已回滚 WAL 文件"""
        now = time.time()
        removed = 0
        for entry in self.scan():
            if entry.status in (WAL_DONE, WAL_ROLLED_BACK):
                age_hours = (now - entry.updated_at) / 3600
                if age_hours > max_age_hours:
                    wal_path = self._path(entry.entry_id)
                    if wal_path.exists():
                        wal_path.unlink()
                        removed += 1
        if removed:
            logger.info(f"Cleaned {removed} old WAL entries (> {max_age_hours}h)")
