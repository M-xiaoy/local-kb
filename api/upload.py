"""
upload.py — 上传端点逻辑
========================
Phase 2：核心入库委托给 core.kb.KnowledgeBase.add_document()，
API 层只负责文件解析、WAL、持久化、重建。
"""

import asyncio
import logging
import time
import os
import shutil

import numpy as np
from fastapi import HTTPException, UploadFile
from pathlib import Path

from pipeline.parser import parse_file, UnsupportedFileError, ParseResult
from pipeline.chunker import chunk_text, chunk_markdown
from pipeline.attr_head_extractor import AttrHeadExtractor
from storage.sphere_store import make_sphere_id
from api.rebuild import rebuild_spaces
from config import paths as cfg_paths

logger = logging.getLogger(__name__)


async def handle_upload_file(
    state, file: UploadFile, source_type: str, auto_rebuild: bool,
    kb=None,
):
    """处理单文件上传

    Phase 2：核心入库（分块/嵌入/球体创建/半径推导）委托给 kb.add_document()。
    API 层保留：文件保存、解析、doc_terms 提取、WAL、原子持久化、重建。
    """
    timings = {}
    t0 = time.time()

    # 1. 保存上传文件
    file_path = state.uploads_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)
    timings["save"] = time.time() - t0

    # 2. 解析
    t1 = time.time()
    try:
        result = parse_file(str(file_path))
    except UnsupportedFileError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse failed: {e}")
    timings["parse"] = time.time() - t1

    # 3. 文档级关键术语提取
    t3 = time.time()
    doc_term_texts = _extract_doc_terms(result.text)
    timings["doc_terms"] = time.time() - t3

    source_type_actual = source_type or (
        "技术笔记" if result.metadata.get("headings") else "其他"
    )

    # 4. 核心入库：委托给 kb.add_document()
    t4 = time.time()
    if kb is not None:
        added = kb.add_document(
            text=result.text,
            source_file=file.filename,
            source_type=source_type_actual,
            chunk=True,
            doc_terms=doc_term_texts,
        )
        logger.debug(f"  kb.add_document returned {added} new spheres")
    else:
        # 降级：走旧的直接写入路径（kb 未初始化时的安全回退）
        added = await _legacy_ingest(
            state, result, file.filename, source_type_actual,
            doc_term_texts, timings,
        )
    timings["kb_ingest"] = time.time() - t4

    if added == 0:
        return {
            "file": file.filename, "file_type": result.file_type,
            "chunks": 0, "new_spheres": 0, "timings": timings,
        }

    # 5. WAL 记录
    #    kb.add_document 已经写了球体和向量，WAL 做双重保障
    wal_entry = _create_wal_entry(state, file, source_type_actual)
    if wal_entry:
        state.wal.mark_committing(wal_entry)

    # 6. 原子持久化
    t7 = time.time()
    try:
        _atomic_save_all_internal(state)
    except Exception as e:
        logger.error(f"Atomic save failed: {e}")
        if wal_entry:
            state.wal._rollback(
                wal_entry, state.sphere_store,
                state.registry, state.faiss_store,
            )
            state.wal.mark_rolled_back(wal_entry)
        raise HTTPException(
            status_code=500, detail=f"Persist failed: {e}"
        )
    if wal_entry:
        state.wal.mark_done(wal_entry)
    timings["persist"] = time.time() - t7

    # 7. 重建
    if added > 0 and auto_rebuild:
        try:
            rebuild_result = await rebuild_spaces(state, mode="cluster")
            rebuild_timings = rebuild_result.get("timings", {})
            if isinstance(rebuild_timings, dict):
                for _rk, _rv in rebuild_timings.items():
                    timings[f"rebuild_{_rk}"] = _rv
            else:
                timings["rebuild"] = rebuild_timings
        except Exception as e:
            logger.warning(f"Post-upload rebuild failed (non-critical): {e}")

    timings["total"] = time.time() - t0

    # 清理临时文件
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass

    return {
        "file": file.filename,
        "file_type": result.file_type,
        "chunks": len(result.text.split("\n")),  # approximate
        "new_spheres": added,
        "timings": timings,
    }


# ──────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────


def _extract_doc_terms(text: str) -> list:
    """从文本中提取文档级关键术语"""
    try:
        _extractor = AttrHeadExtractor()
        _doc_terms = _extractor.extract(text)
        doc_term_texts = list(dict.fromkeys(
            p.full_phrase for p in _doc_terms if p.has_attributive
        ))
        if not doc_term_texts:
            sentences = [
                s.strip() for s in text[:500].split("。")
                if len(s.strip()) > 5
            ]
            doc_term_texts = sentences[:3]
        logger.info(f"  doc_terms: {len(doc_term_texts)} terms")
        return doc_term_texts
    except Exception as e:
        logger.warning(f"  doc_terms failed: {e}")
        return []


def _create_wal_entry(state, file, source_type_actual):
    """创建 WAL 记录（失败回滚用）"""
    try:
        all_sids = [s.id for s in state.sphere_store.get_active()]
        # 只取最新的（按时间排序，取后几个）
        new_sids = all_sids[-100:] if len(all_sids) > 100 else all_sids
        return state.wal.create(
            file_name=file.filename,
            source_type=source_type_actual,
            sphere_ids=new_sids,
            faiss_ids=[state.registry.faiss_id(s) for s in new_sids],
            chunks_total=len(new_sids),
        )
    except Exception as e:
        logger.warning(f"WAL entry creation failed (non-critical): {e}")
        return None


async def _legacy_ingest(
    state, result, filename, source_type_actual,
    doc_term_texts, timings,
):
    """降级路径：kb 未初始化时的安全回退（保持旧行为）"""
    from pipeline.chunker import chunk_text, chunk_markdown
    from pipeline.keywords import extract_keywords
    from storage.sphere_store import Sphere

    # 切片（同步）
    if result.file_type == "md":
        raw_chunks = chunk_markdown(result.text)
    else:
        raw_chunks = chunk_text(result.text, source_type=source_type_actual)
    chunks = [c for c in raw_chunks if c.strip()]

    if not chunks:
        return 0

    # 去重
    faiss_cache_hot = state.faiss_store.is_built
    new_chunks = []
    for chunk in chunks:
        sid = make_sphere_id(chunk, filename)
        existing = state.sphere_store.get(sid)
        if existing and existing.active and faiss_cache_hot:
            continue
        new_chunks.append({"chunk": chunk, "sphere_id": sid})

    if not new_chunks:
        return 0

    # 嵌入
    new_texts = [c["chunk"] for c in new_chunks]
    new_vectors = state.embedder.embed_documents(new_texts)

    # 写入存储
    for entry, vector in zip(new_chunks, new_vectors):
        sid = entry["sphere_id"]
        existing = state.sphere_store.get(sid)
        if existing and existing.active:
            continue
        fid = state.registry.register(sid)
        sphere = Sphere(
            id=sid,
            text=entry["chunk"],
            source_file=filename,
            source_type=source_type_actual,
            doc_terms=doc_term_texts,
        )
        state.sphere_store.add(sphere)
        state.faiss_store.add(
            vector.reshape(1, -1),
            np.array([fid], dtype=np.int64),
        )

    logger.warning(
        f"Legacy ingest used: {len(new_chunks)} chunks written "
        f"(no poincare_norm at insert time — Phase 2 not active)"
    )
    return len(new_chunks)


def _atomic_save_all_internal(state):
    """原子保存所有状态"""
    for name, store, path_attr in [
        ("registry", state.registry, "registry_map"),
        ("spheres", state.sphere_store, "spheres_data"),
    ]:
        path = getattr(cfg_paths, path_attr)
        tmp_path = path + ".tmp"
        try:
            store.save(tmp_path)
            if os.path.exists(path):
                os.remove(path)
            shutil.move(tmp_path, path)
        except Exception:
            raise

    state.faiss_store.save()
