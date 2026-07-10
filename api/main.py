"""
main.py — 重力知识库 FastAPI 应用
==================================
提供三个核心端点：
  POST /upload  — 上传文件解析入库
  POST /query   — 检索问题
  GET  /status  — 知识库状态
  POST /rebuild — 重建索引（从 sphere_store 恢复全部状态）

启动：
  uvicorn api.main:app --reload --port 8765
"""

import logging
import os
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import paths as cfg_paths, web as cfg_web, chunker as cfg_chunker
from pipeline.parser import parse_file, ParseResult, UnsupportedFileError
from pipeline.chunker import chunk_text, chunk_markdown
from pipeline.embedder import Embedder
from storage.faiss_store import FaissStore
from storage.registry import Registry
from storage.sphere_store import SphereStore, Sphere, make_sphere_id
from retrieval.field_detector import FieldDetector
from retrieval.diversity_sorter import DiversitySorter
from retrieval.retriever import Retriever, RetrievalResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# FastAPI 应用
# ──────────────────────────────────────────────

app = FastAPI(
    title="重力知识库",
    description="个人本地知识问答系统 — FAISS + 重力空间路由",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# 全局异常处理
# ──────────────────────────────────────────────


def _make_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _error_response(request_id: str, error_type: str, message: str, detail: str = "", suggestion: str = ""):
    """统一错误响应格式"""
    body = {
        "request_id": request_id,
        "error": error_type,
        "message": message,
    }
    if detail:
        body["detail"] = detail
    if suggestion:
        body["suggestion"] = suggestion
    return body


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """重写 HTTPException 响应为统一格式"""
    req_id = _make_request_id()
    logger.warning(
        f"[req={req_id}] HTTP {exc.status_code} {request.method} {request.url.path}: {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_response(
            request_id=req_id,
            error_type="http_error",
            message=str(exc.detail),
            detail="",
            suggestion={
                400: "请检查请求参数格式是否正确",
                404: "请求的资源不存在",
                422: "请求数据无法处理，请检查文件格式/内容",
                413: "文件过大，请压缩后重试",
            }.get(exc.status_code, ""),
        ),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """参数校验失败 → 指出哪个字段不对"""
    req_id = _make_request_id()
    errors = exc.errors()
    field_errors = [
        f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
        for e in errors
    ]
    logger.warning(
        f"[req={req_id}] 422 {request.method} {request.url.path}: {'; '.join(field_errors)}"
    )
    return JSONResponse(
        status_code=422,
        content=_error_response(
            request_id=req_id,
            error_type="validation_error",
            message="请求参数校验失败",
            detail="; ".join(field_errors),
            suggestion="请检查请求体中的参数类型和值是否符合接口要求",
        ),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底：捕获一切未处理的异常

    返回完整错误信息 + 把 traceback 写进日志。
    不会暴露敏感信息——因为这是本地服务，不对外。
    """
    req_id = _make_request_id()
    tb = traceback.format_exc()

    logger.error(
        f"[req={req_id}] 500 {request.method} {request.url.path}\n{tb}"
    )

    return JSONResponse(
        status_code=500,
        content=_error_response(
            request_id=req_id,
            error_type=type(exc).__name__,
            message="服务器内部错误",
            detail=str(exc),
            suggestion="请检查服务器日志 (kb_server.log) 查看详细 traceback",
        ),
    )


# ──────────────────────────────────────────────
# 全局状态
# ──────────────────────────────────────────────

class AppState:
    """应用生命周期状态（启动时初始化，存活于内存）"""
    def __init__(self):
        self.embedder = Embedder()
        self.faiss_store = FaissStore()
        self.registry = Registry()
        self.sphere_store = SphereStore()
        self.field_detector = FieldDetector()
        self.sorter = DiversitySorter()
        self.retriever = Retriever(
            embedder=self.embedder,
            faiss_store=self.faiss_store,
            registry=self.registry,
            sphere_store=self.sphere_store,
            field_detector=self.field_detector,
            diversity_sorter=self.sorter,
        )
        self.uploads_dir = Path(cfg_paths.uploads_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def is_loaded(self) -> bool:
        return self.faiss_store.is_built


state = AppState()


# ──────────────────────────────────────────────
# 请求/响应模型
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    fetch_k: int = 100


class QueryResponse(BaseModel):
    query: str
    results: List[dict]
    field_affinities: dict
    total_spheres: int
    timing_ms: dict


class UploadResponse(BaseModel):
    file: str
    file_type: str
    chunks: int
    new_spheres: int
    timing_ms: dict


class StatusResponse(BaseModel):
    total_spheres: int
    active_spheres: int
    faiss_vectors: int
    fields: List[str]
    field_counts: dict


# ──────────────────────────────────────────────
# 启动事件
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """启动时从持久化加载已有状态"""
    try:
        # 加载注册表
        registry_count = state.registry.load()
        logger.info(f"Registry loaded: {registry_count} mappings")

        # 加载球体库
        sphere_count = state.sphere_store.load()
        logger.info(f"Sphere store loaded: {sphere_count} spheres")

        # 加载 FAISS 索引
        faiss_count = state.faiss_store.load()
        logger.info(f"FAISS index loaded: {faiss_count} vectors")

        # 从球体库重建场域质心
        if sphere_count > 0:
            field_vectors = _collect_field_vectors(state.sphere_store, state.faiss_store, state.registry)
            state.field_detector.rebuild_centroids(field_vectors)
            logger.info(f"Field centroids rebuilt: {state.field_detector.field_count} fields")

        if faiss_count > 0 and sphere_count > 0:
            logger.info("Knowledge base loaded successfully")
        else:
            logger.info("Empty knowledge base — ready for uploads")
    except Exception as e:
        logger.warning(f"Startup load failed (first run?): {e}")


# ──────────────────────────────────────────────
# 端点：状态
# ──────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """知识库状态概览"""
    active = state.sphere_store.count
    total = state.sphere_store.total_count
    fields = state.field_detector.fields
    field_counts = {f: state.field_detector._field_counts.get(f, 0) for f in fields}
    return StatusResponse(
        total_spheres=total,
        active_spheres=active,
        faiss_vectors=state.faiss_store.count,
        fields=fields,
        field_counts=field_counts,
    )


# ──────────────────────────────────────────────
# 端点：上传
# ──────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    source_type: str = Form(""),
    auto_rebuild: bool = Form(True),
):
    """上传文件并入库

    Args:
        file: PDF/DOCX/MD/TXT 文件
        source_type: 场域标签（如"技术笔记"、"小说创作"）
        auto_rebuild: 是否在入库后保存全部状态
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
        result: ParseResult = parse_file(str(file_path))
    except UnsupportedFileError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse failed: {e}")
    timings["parse"] = time.time() - t1

    # 3. 切片
    t2 = time.time()
    if result.file_type == "md":
        raw_chunks = chunk_markdown(result.text)
    else:
        raw_chunks = chunk_text(result.text)
    chunks = [c for c in raw_chunks if c.strip()]
    timings["chunk"] = time.time() - t2

    if not chunks:
        raise HTTPException(status_code=422, detail="No text chunks extracted from file")

    # ── 3.5 预过滤：只嵌入真正的新切片 ──────────
    t_pre = time.time()
    chunk_sphere_ids = [make_sphere_id(c, file.filename) for c in chunks]

    # 拆分为新/旧两组
    # new_mask[i]=True  → 需要嵌入
    # reused_vectors    → 已缓存的向量，按原序收集
    reused_vectors: dict = {}  # chunk_index → vector
    new_mask = []
    faiss_cache_hot = state.faiss_store.is_built

    for i, sid in enumerate(chunk_sphere_ids):
        existing = state.sphere_store.get(sid)
        if existing and existing.active and faiss_cache_hot:
            fid = state.registry.faiss_id(sid)
            if fid is not None and fid in state.faiss_store._vectors:
                reused_vectors[i] = state.faiss_store._vectors[fid]
                new_mask.append(False)
                continue
        new_mask.append(True)

    new_chunks = [chunks[i] for i, m in enumerate(new_mask) if m]
    timings["prefilter"] = time.time() - t_pre
    logger.info(
        f"  prefilter: {len(new_chunks)} new / {len(chunks) - len(new_chunks)} cached "
        f"(faiss_cache={'hot' if faiss_cache_hot else 'cold'})"
    )

    # ── 4. 只嵌入新切片 ────────────────────────
    t3 = time.time()
    if new_chunks:
        new_vectors = state.embedder.embed_documents(new_chunks)
    else:
        # 全部命中缓存，无需任何嵌入
        embed_dim = state.embedder.embed_dim
        new_vectors = np.zeros((0, embed_dim), dtype=np.float32)
    timings["embed"] = time.time() - t3

    # ── 4.5 按原始顺序重建完整向量列表 ──────────
    chunk_vectors = np.zeros(
        (len(chunks), state.embedder.embed_dim), dtype=np.float32
    )
    new_idx = 0
    skipped_count = 0
    for i in range(len(chunks)):
        if i in reused_vectors:
            chunk_vectors[i] = reused_vectors[i]
            skipped_count += 1
        else:
            chunk_vectors[i] = new_vectors[new_idx]
            new_idx += 1

    # ── 5. 创建新球体 ──────────────────────────
    t4 = time.time()
    new_spheres = 0
    added_vectors: list = []
    added_ids: list = []

    for i, (chunk_text, vec) in enumerate(zip(chunks, chunk_vectors)):
        sphere_id = chunk_sphere_ids[i]

        if new_mask[i]:
            # 新球体：创建 + 注册 + 收集 FAISS
            sphere = Sphere(
                id=sphere_id,
                text=chunk_text,
                source_file=file.filename,
                source_type=source_type or (
                    "技术笔记" if result.metadata.get("headings") else "其他"
                ),
                mass=1.0,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            if state.sphere_store.add(sphere):
                new_spheres += 1
            faiss_id = state.registry.register(sphere_id)
            added_vectors.append(vec)
            added_ids.append(faiss_id)
    timings["create_spheres"] = time.time() - t4

    # ── 6. 新向量 → FAISS ─────────────────────
    t5 = time.time()
    if added_vectors:
        vectors_array = np.stack(added_vectors, axis=0)
        ids_array = np.array(added_ids, dtype=np.int64)
        state.faiss_store.add(vectors_array, ids_array)
    timings["faiss_add"] = time.time() - t5

    # ── 7. 新向量 → 场域质心 ───────────────────
    t6 = time.time()
    if source_type and added_vectors:
        for vec in added_vectors:
            state.field_detector.update_centroid(source_type, vec)
    timings["field_update"] = time.time() - t6

    # ── 8. 持久化 ─────────────────────────────
    t7 = time.time()
    if new_spheres > 0 and auto_rebuild:
        state.registry.save()
        state.sphere_store.save()
        state.faiss_store.save()
    timings["persist"] = time.time() - t7

    timings["total"] = time.time() - t0

    return UploadResponse(
        file=file.filename,
        file_type=result.file_type,
        chunks=len(chunks),
        new_spheres=new_spheres,
        timing_ms={k: round(v * 1000, 1) for k, v in timings.items()},
    )


# ──────────────────────────────────────────────
# 端点：检索
# ──────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """检索问题

    Args:
        query: 用户问题
        top_k: 返回结果数量
        fetch_k: FAISS 粗搜候选池大小

    Returns:
        排序后的切片 + 场域亲和度 + 耗时
    """
    if not state.is_loaded():
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Upload files first.")

    result: RetrievalResult = state.retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        fetch_k=request.fetch_k,
    )

    return QueryResponse(
        query=result.query,
        results=[
            {
                "text": s.text,
                "source_file": s.source_file,
                "source_type": s.source_type,
                "score": float(round(result.scores[i], 4)),
                "sphere_id": s.id,
            }
            for i, s in enumerate(result.spheres)
        ],
        field_affinities={k: float(v) for k, v in result.field_affinities.items()},
        total_spheres=result.total_count,
        timing_ms={k: round(v * 1000, 1) for k, v in result.timing.items()},
    )


# ──────────────────────────────────────────────
# 端点：重建
# ──────────────────────────────────────────────

@app.post("/rebuild")
async def rebuild():
    """从持久化数据重建全部状态

    适用场景：手动修改了 spheres.json 或 registry.json 后重新索引
    """
    try:
        # 清空当前状态
        state.registry.clear()
        state.faiss_store = FaissStore()
        state.field_detector = FieldDetector()

        # 重新加载
        state.registry.load()
        state.sphere_store.load()

        # 从 sphere_store 重新构建 FAISS 索引和场域
        active_spheres = state.sphere_store.get_active()
        if not active_spheres:
            return {"status": "ok", "rebuilt": 0, "message": "No active spheres to rebuild"}

        # 按批次嵌入 + 添加
        all_vectors = []
        all_ids = []
        field_vectors: dict = {}

        for sphere in active_spheres:
            vec = state.embedder.embed_documents([sphere.text])[0]
            faiss_id = state.registry.register(sphere.id)
            all_vectors.append(vec)
            all_ids.append(faiss_id)

            # 收集场域向量
            if sphere.source_type:
                field_vectors.setdefault(sphere.source_type, []).append(vec)

        # 构建 FAISS 索引
        vectors_array = np.stack(all_vectors, axis=0)
        ids_array = np.array(all_ids, dtype=np.int64)
        state.faiss_store.build(vectors_array, ids_array)

        # 重建场域质心
        state.field_detector.rebuild_centroids(field_vectors)

        # 持久化
        state.registry.save()
        state.faiss_store.save()

        return {
            "status": "ok",
            "rebuilt": len(active_spheres),
            "fields": state.field_detector.fields,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {e}")


# ──────────────────────────────────────────────
# 内部函数
# ──────────────────────────────────────────────

def _collect_field_vectors(
    sphere_store: SphereStore,
    faiss_store: FaissStore,
    registry: Registry,
) -> dict:
    """从现有存储中收集场域向量（用于启动时重建质心）"""
    field_vectors: dict = {}
    for sphere in sphere_store.get_active():
        if not sphere.source_type:
            continue
        fid = registry.faiss_id(sphere.id)
        if fid is not None and fid in faiss_store._vectors:
            field_vectors.setdefault(sphere.source_type, []).append(
                faiss_store._vectors[fid]
            )
    return field_vectors
