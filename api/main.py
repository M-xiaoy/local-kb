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
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from config import paths as cfg_paths, web as cfg_web, chunker as cfg_chunker
from pipeline.parser import parse_file, ParseResult, UnsupportedFileError
from pipeline.chunker import chunk_text, chunk_markdown
from pipeline.embedder import Embedder
from pipeline.keywords import extract_keywords
from storage.faiss_store import FaissStore
from storage.registry import Registry
from storage.sphere_store import SphereStore, Sphere, make_sphere_id
from storage.wal import WalManager, WAL_READY, WAL_COMMITTING
from retrieval.field_detector import FieldDetector
from retrieval.diversity_sorter import DiversitySorter
from retrieval.retriever import Retriever, RetrievalResult
from retrieval.session_manager import SessionManager
from retrieval.cluster_engine import ClusterEngine

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 前端 HTML 页面
# ──────────────────────────────────────────────

HTML_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>重力知识库</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; }
.container { max-width: 900px; margin: 0 auto; padding: 24px 16px; }

/* 顶部 */
.header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.header h1 { font-size: 20px; font-weight: 600; }
.header .subtitle { color: #888; font-size: 13px; margin-top: 2px; }
.header-right { text-align: right; font-size: 12px; color: #666; }

/* 搜索 */
.search-area { margin-bottom: 16px; }
.search-row { display: flex; gap: 8px; }
.search-row input { flex: 1; padding: 10px 14px; border: 1px solid #2a2a3a; border-radius: 8px; background: #1a1a2e; color: #e0e0e0; font-size: 14px; outline: none; }
.search-row input:focus { border-color: #4a4a8a; }
.search-row button { padding: 10px 20px; border: none; border-radius: 8px; background: #4a4a8a; color: #fff; font-size: 14px; cursor: pointer; }
.search-row button:hover { background: #5a5a9a; }
.search-row button:disabled { opacity: 0.5; cursor: not-allowed; }

/* 场域聚焦栏 */
.focus-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.focus-bar .label { font-size: 12px; color: #888; margin-right: 4px; }
.focus-btn { padding: 4px 12px; border: 1px solid #2a2a3a; border-radius: 12px; background: transparent; color: #888; font-size: 12px; cursor: pointer; }
.focus-btn:hover { border-color: #4a4a8a; color: #aaa; }
.focus-btn.active { background: #4a4a8a; border-color: #4a4a8a; color: #fff; }
.focus-btn.active:hover { background: #5a5a9a; }
.focus-btn.exit { border-color: #8a3a3a; color: #8a3a3a; }

/* 结果 */
.result-card { background: #1a1a2e; border: 1px solid #2a2a3a; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
.result-card .meta { display: flex; gap: 8px; font-size: 11px; color: #666; margin-bottom: 6px; flex-wrap: wrap; }
.result-card .meta .badge { padding: 1px 6px; border-radius: 4px; font-size: 10px; }
.badge-cluster { background: #2a2a4a; color: #8a8acc; }
.badge-field { background: #2a2a3a; color: #888; }
.badge-score { background: #1a2a1a; color: #6a8; }
.result-card .text { font-size: 13px; line-height: 1.6; color: #ccc; max-height: 150px; overflow-y: auto; }
.result-card .gf { font-size: 10px; color: #555; margin-top: 6px; }

/* 状态 */
.status-bar { font-size: 12px; color: #555; margin-bottom: 16px; }

/* 上传区 */
.upload-area { margin-top: 32px; padding: 16px; border: 1px dashed #2a2a3a; border-radius: 8px; text-align: center; }
.upload-area input { display: none; }
.upload-area label { display: inline-block; padding: 6px 14px; border-radius: 6px; background: #2a2a3a; color: #888; font-size: 12px; cursor: pointer; }
.upload-area label:hover { background: #3a3a4a; }
.upload-status { font-size: 12px; color: #666; margin-top: 8px; }

.loading { opacity: 0.5; }
.empty { color: #666; font-size: 13px; text-align: center; padding: 32px; }
</style>
</head>
<body>
<div class="container" id="app">
  <div class="header">
    <div>
      <h1>✦ 重力知识库</h1>
      <div class="subtitle">FAISS + 引力场路由 + 自动聚类</div>
    </div>
    <div class="header-right" id="status-info">
      <div id="sphere-count">加载中...</div>
    </div>
  </div>

  <div class="search-area">
    <div class="search-row">
      <input id="query-input" type="text" placeholder="搜索知识库..." onkeydown="if(event.key==='Enter') search()">
      <button id="search-btn" onclick="search()">搜索</button>
    </div>
  </div>

  <div class="focus-bar" id="focus-bar">
    <span class="label">聚焦:</span>
    <span id="focus-buttons"></span>
    <button class="focus-btn exit" id="exit-focus" style="display:none" onclick="resetFocus()">× 退出聚焦</button>
  </div>

  <div class="status-bar" id="session-info"></div>

  <div id="results"></div>

  <div class="upload-area">
    <label for="file-upload">📄 上传文件</label>
    <input type="file" id="file-upload" onchange="uploadFile()">
    <div class="upload-status" id="upload-status"></div>
  </div>
</div>

<script>
let sessionId = localStorage.getItem("gravity_session") || null;
let currentFocus = null;

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json()).message || res.statusText);
  return res.json();
}

async function loadStatus() {
  try {
    const s = await fetch("/status").then(r => r.json());
    document.getElementById("sphere-count").textContent = s.active_spheres + " 球体 · " + s.fields.length + " 场域";
    renderFocusButtons(s.fields.map((f, i) => ({ name: f, count: s.field_counts[f] || 0 })));
  } catch(e) {
    document.getElementById("sphere-count").textContent = "连接失败";
  }
}

function renderFocusButtons(fields) {
  let html = "";
  for (const f of fields) {
    const active = currentFocus === f.name ? "active" : "";
    html += `<button class="focus-btn ${active}" onclick="setFocus('${f.name}')">${f.name}</button> `;
  }
  document.getElementById("focus-buttons").innerHTML = html;
  document.getElementById("exit-focus").style.display = currentFocus ? "inline-block" : "none";
}

async function search() {
  const q = document.getElementById("query-input").value.trim();
  if (!q) return;
  
  document.getElementById("search-btn").disabled = true;
  document.getElementById("results").innerHTML = "<div class='loading'>搜索中...</div>";
  
  try {
    const payload = { query: q, top_k: 5 };
    if (sessionId) payload.session_id = sessionId;
    if (currentFocus) payload.field_focus = currentFocus;
    
    const data = await api("/query", payload);
    sessionId = data.session_id;
    if (sessionId) localStorage.setItem("gravity_session", sessionId);
    
    renderResults(data);
    document.getElementById("session-info").textContent = "会话: " + sessionId.slice(0, 8) + "...";
  } catch(e) {
    document.getElementById("results").innerHTML = "<div class='empty'>错误: " + e.message + "</div>";
  }
  document.getElementById("search-btn").disabled = false;
}

function renderResults(data) {
  const r = data.results;
  if (!r || r.length === 0) {
    document.getElementById("results").innerHTML = "<div class='empty'>无结果</div>";
    return;
  }
  let html = "";
  for (const item of r) {
    const text = item.text;
    const gf = item.gravity_field ? Object.entries(item.gravity_field).map(([k,v]) => `${k}:${v.toFixed(2)}`).join(" ") : "";
    html += `<div class="result-card">`;
    html += `<div class="meta">`;
    html += `<span class="badge badge-cluster">簇${item.cluster_id}</span>`;
    html += `<span class="badge badge-field">${item.source_type || "?"}</span>`;
    html += `<span class="badge badge-score">${item.score.toFixed(3)}</span>`;
    html += `</div>`;
    html += `<div class="text">${escapeHtml(text)}</div>`;
    if (gf) html += `<div class="gf">${gf}</div>`;
    html += `</div>`;
  }
  document.getElementById("results").innerHTML = html;
}

function escapeHtml(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function setFocus(field) {
  currentFocus = field;
  renderFocusButtons([]);
  loadStatus();
  if (document.getElementById("query-input").value.trim()) search();
}

function resetFocus() {
  currentFocus = null;
  renderFocusButtons([]);
  loadStatus();
  if (sessionId) {
    api("/query", { query: ".", top_k: 1, session_id: sessionId, reset_focus: true }).then(() => {});
  }
}

async function uploadFile() {
  const fileInput = document.getElementById("file-upload");
  const file = fileInput.files[0];
  if (!file) return;
  
  const status = document.getElementById("upload-status");
  status.textContent = "上传中...";
  
  const form = new FormData();
  form.append("file", file);
  form.append("source_type", "");
  
  try {
    const res = await fetch("/upload", { method: "POST", body: form });
    const data = await res.json();
    status.textContent = "已上传: " + data.new_spheres + " 新球体 (" + data.timing_ms.total.toFixed(0) + "ms)";
    loadStatus();
  } catch(e) {
    status.textContent = "上传失败: " + e.message;
  }
  fileInput.value = "";
}

loadStatus();
</script>
</body>
</html>'''


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
        self.wal = WalManager(str(Path(cfg_paths.wal_dir)))
        self.sessions = SessionManager()
        self.cluster_engine = ClusterEngine()

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
    session_id: Optional[str] = None   # 会话追踪
    field_focus: Optional[str] = None    # 聚焦某场域（null=不聚焦）
    reset_focus: bool = False            # 退出聚焦
    exclude_ids: Optional[List[str]] = None  # 排除已返回的球体


class QueryResponse(BaseModel):
    query: str
    results: List[dict]
    field_affinities: dict
    total_spheres: int
    timing_ms: dict
    session_id: Optional[str] = None   # 会话 ID
    focus_field: Optional[str] = None  # 当前聚焦的场域


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
        # WAL 恢复：处理未完成的写入
        recovery = state.wal.recover(
            sphere_store=state.sphere_store,
            registry=state.registry,
            faiss_store=state.faiss_store,
        )
        if recovery["rolled_back"] > 0 or recovery["errors"] > 0:
            logger.warning(
                f"WAL recovery: {recovery['rolled_back']} rolled back, "
                f"{recovery['errors']} errors"
            )
        else:
            logger.info("WAL recovery: clean (no incomplete entries)")

        # 加载注册表
        registry_count = state.registry.load()
        logger.info(f"Registry loaded: {registry_count} mappings")

        # 加载球体库
        sphere_count = state.sphere_store.load()
        logger.info(f"Sphere store loaded: {sphere_count} spheres")

        # 加载 FAISS 索引
        faiss_count = state.faiss_store.load()
        logger.info(f"FAISS index loaded: {faiss_count} vectors")

        # 加载聚类状态（如果有）
        cluster_loaded = state.cluster_engine.load()
        if cluster_loaded:
            # 从聚类状态恢复每个球体的 cluster_id
            k = state.cluster_engine.n_centroids
            centroids = state.cluster_engine.centroids

            if faiss_count > 0 and centroids is not None:
                try:
                    # 用聚类质心预测每个球体的归属
                    active = state.sphere_store.get_active()
                    vectors_for_cluster = []
                    valid_spheres = []
                    for s in active:
                        fid = state.registry.faiss_id(s.id)
                        if fid is not None and fid in state.faiss_store._vectors:
                            vectors_for_cluster.append(state.faiss_store._vectors[fid])
                            valid_spheres.append(s)

                    if len(vectors_for_cluster) >= 2:
                        labels, _ = state.cluster_engine.predict(
                            np.stack(vectors_for_cluster, axis=0)
                        )
                        for s, label in zip(valid_spheres, labels):
                            s.cluster_id = int(label)

                        # 统计簇大小
                        cluster_counts = {}
                        for s in active:
                            if s.cluster_id >= 0:
                                cluster_counts[s.cluster_id] = cluster_counts.get(s.cluster_id, 0) + 1

                        # 同步到 field_detector 并重算 gravity_field
                        label_map = {i: f"簇{i}" for i in range(k)}
                        state.field_detector.sync_from_clusters(
                            centroids, label_map, cluster_counts
                        )
                        state.field_detector.rebuild_all_gravity_fields(
                            state.sphere_store,
                            {s.id: state.faiss_store._vectors.get(
                                state.registry.faiss_id(s.id)
                            ) for s in active if state.registry.faiss_id(s.id) is not None}
                        )
                        logger.info(f"Restored cluster assignments: {len(valid_spheres)} spheres → {k} clusters")
                except Exception as e:
                    logger.warning(f"Failed to restore cluster assignments: {e}")
            else:
                label_map = {i: f"簇{i}" for i in range(k)}
                state.field_detector.sync_from_clusters(centroids, label_map)
                logger.info(f"Cluster centroids synced to FieldDetector (no spheres for assignment): {k} clusters")

        # 从球体库重建场域质心（仅在无聚类状态时回退）
        if not cluster_loaded and sphere_count > 0:
            field_vectors = _collect_field_vectors(state.sphere_store, state.faiss_store, state.registry)
            state.field_detector.rebuild_centroids(field_vectors)
            logger.info(f"Field centroids rebuilt from labels: {state.field_detector.field_count} fields")

            # gravity_field 迁移：为旧数据补算到各质心的引力值
            migrated = _migrate_gravity_fields(
                state.sphere_store, state.faiss_store,
                state.registry, state.field_detector
            )
            logger.info(f"gravity_field migration: {migrated} spheres updated")

            # 如果有球体但没有保存的聚类状态 → 首次聚类
            if not cluster_loaded and faiss_count >= 2:
                try:
                    vectors = []
                    for s in state.sphere_store.get_active():
                        fid = state.registry.faiss_id(s.id)
                        if fid is not None and fid in state.faiss_store._vectors:
                            vectors.append(state.faiss_store._vectors[fid])

                    if len(vectors) >= 2:
                        vectors_arr = np.stack(vectors, axis=0)
                        centroids, labels, _ = state.cluster_engine.fit_predict(vectors_arr)

                        k = centroids.shape[0]
                        label_map = {i: f"簇{i}" for i in range(k)}

                        for sphere, label in zip(state.sphere_store.get_active(), labels):
                            sphere.cluster_id = int(label)

                        # 统计每个簇的球体数（用于 status 展示）
                        cluster_counts = {}
                        for s in state.sphere_store.get_active():
                            cid = s.cluster_id
                            if cid >= 0:
                                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

                        state.field_detector.sync_from_clusters(
                            centroids, label_map, cluster_counts
                        )

                        # 全量重算 gravity_field
                        state.field_detector.rebuild_all_gravity_fields(
                            state.sphere_store,
                            {s.id: state.faiss_store._vectors.get(
                                state.registry.faiss_id(s.id)
                            ) for s in state.sphere_store.get_active()
                             if state.registry.faiss_id(s.id) is not None}
                        )

                        state.cluster_engine.save()
                        logger.info(f"Initial clustering on startup: {len(vectors)} spheres → {k} clusters")
                except Exception as e:
                    logger.warning(f"Initial clustering failed: {e}")

        if faiss_count > 0 and sphere_count > 0:
            logger.info("Knowledge base loaded successfully")
        else:
            logger.info("Empty knowledge base — ready for uploads")
    except Exception as e:
        logger.warning(f"Startup load failed (first run?): {e}")


# ──────────────────────────────────────────────
# 端点：首页（前端界面）
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """前端首页"""
    return HTMLResponse(content=HTML_PAGE)


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

    new_sphere_ids = [chunk_sphere_ids[i] for i, m in enumerate(new_mask) if m]
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
        embed_dim = state.embedder.embed_dim
        new_vectors = np.zeros((0, embed_dim), dtype=np.float32)
    timings["embed"] = time.time() - t3

    # ── 4.5 按原始顺序重建完整向量列表 ──────────
    chunk_vectors = np.zeros(
        (len(chunks), state.embedder.embed_dim), dtype=np.float32
    )
    new_idx = 0
    for i in range(len(chunks)):
        if i in reused_vectors:
            chunk_vectors[i] = reused_vectors[i]
        else:
            chunk_vectors[i] = new_vectors[new_idx]
            new_idx += 1

    # ── 5. 创建新球体（含 gravity_field 预计算）──
    t4 = time.time()
    new_spheres = 0
    added_vectors: list = []
    added_ids: list = []
    added_sphere_ids: list = []
    new_gravity_fields: list = []  # 收集新球体的 gravity_field，给后续步骤用

    for i, (sphere_id, chunk_txt, vec) in enumerate(zip(chunk_sphere_ids, chunks, chunk_vectors)):
        if not new_mask[i]:
            # 旧球体：检查 gravity_field 是否需要补算
            existing = state.sphere_store.get(sphere_id)
            if existing and not existing.gravity_field:
                existing.gravity_field = state.field_detector.compute_gravity_field(vec)
            continue

        # 新球体：预计算 gravity_field（到各场域质心的引力值）
        gravity_field = state.field_detector.compute_gravity_field(vec)

        # 提取关键词权重（用于术语引力混合检索）
        term_weights = extract_keywords(chunk_txt)

        sphere = Sphere(
            id=sphere_id,
            text=chunk_txt,
            source_file=file.filename,
            source_type=source_type or (
                "技术笔记" if result.metadata.get("headings") else "其他"
            ),
            mass=1.0,
            gravity_field=gravity_field,
            term_weights=term_weights,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        if state.sphere_store.add(sphere):
            new_spheres += 1
        faiss_id = state.registry.register(sphere_id)
        added_vectors.append(vec)
        added_ids.append(faiss_id)
        added_sphere_ids.append(sphere_id)
        new_gravity_fields.append(gravity_field)
    timings["create_spheres"] = time.time() - t4

    # ── 5.5 WAL：记录本次操作 ─────────────────
    wal_entry = None
    if new_spheres > 0 and auto_rebuild:
        wal_entry = state.wal.create(
            file_name=file.filename,
            source_type=source_type or (
                "技术笔记" if result.metadata.get("headings") else "其他"
            ),
            sphere_ids=added_sphere_ids,
            faiss_ids=added_ids,
            chunks_total=len(new_chunks),
        )

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

    # ── 8. 持久化（原子写入）───────────────────
    t7 = time.time()
    if new_spheres > 0 and auto_rebuild:
        # 标记 WAL 为 committing（正在写盘）
        if wal_entry:
            state.wal.mark_committing(wal_entry)

        try:
            _atomic_save_all()
        except Exception as e:
            logger.error(f"Atomic save failed: {e}")
            if wal_entry:
                state.wal._rollback(
                    wal_entry, state.sphere_store,
                    state.registry, state.faiss_store
                )
                state.wal.mark_rolled_back(wal_entry)
            raise HTTPException(
                status_code=500,
                detail=f"Persist failed, changes rolled back: {e}"
            )

        # 标记 WAL 为 done（安全完成）
        if wal_entry:
            state.wal.mark_done(wal_entry)
    timings["persist"] = time.time() - t7

    # ── 9. 上传后聚类 ────────────────────────
    t8 = time.time()
    if new_spheres > 0 and auto_rebuild:
        active_spheres = state.sphere_store.get_active()
        if len(active_spheres) >= 2:  # 至少 2 个才能聚类
            try:
                vectors = []
                for s in active_spheres:
                    fid = state.registry.faiss_id(s.id)
                    if fid is not None and fid in state.faiss_store._vectors:
                        vectors.append(state.faiss_store._vectors[fid])

                if len(vectors) >= 2:
                    vectors_arr = np.stack(vectors, axis=0)
                    centroids, labels, scores = state.cluster_engine.fit_predict(vectors_arr)

                    # 更新每个球体的 cluster_id + gravity_field
                    k = centroids.shape[0]
                    label_map = {i: f"簇{i}" for i in range(k)}

                    for sphere, label, score in zip(active_spheres, labels, scores):
                        sphere.cluster_id = int(label)

                    # 批量重算 gravity_field
                    # 统计每个簇的球体数
                    cluster_counts = {}
                    for s in active_spheres:
                        if s.cluster_id >= 0:
                            cluster_counts[s.cluster_id] = cluster_counts.get(s.cluster_id, 0) + 1

                    state.field_detector.sync_from_clusters(
                        centroids, label_map, cluster_counts
                    )
                    state.field_detector.rebuild_all_gravity_fields(
                        state.sphere_store,
                        {s.id: state.faiss_store._vectors.get(
                            state.registry.faiss_id(s.id)
                        ) for s in active_spheres if state.registry.faiss_id(s.id) is not None}
                    )

                    # 持久化聚类状态
                    state.cluster_engine.save()
                    logger.info(f"Post-upload clustering: {len(active_spheres)} spheres → {k} clusters")
            except Exception as e:
                logger.warning(f"Post-upload clustering failed (non-critical): {e}")
    timings["clustering"] = time.time() - t8

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

    # 会话管理：自动创建（客户端不传时也生成 session_id）
    session = state.sessions.get_or_create(request.session_id)

    # 处理聚焦状态
    if request.reset_focus:
        if session:
            session.reset_focus()
        effective_focus = None
    elif request.field_focus is not None:
        # 用户明确指定聚焦
        if session:
            session.set_focus(request.field_focus)
        effective_focus = request.field_focus
    elif session and session.field_focus:
        # 沿用会话中的聚焦状态
        effective_focus = session.field_focus
    else:
        effective_focus = None

    # 排除已返回的球体
    exclude = set(request.exclude_ids or [])
    if session:
        exclude |= session.exclude_ids

    result: RetrievalResult = state.retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        fetch_k=request.fetch_k,
        field_focus=effective_focus,
        exclude_ids=list(exclude),
    )

    # 记录返回的球体 ID 到会话
    returned_ids = [s.id for s in result.spheres]
    if session:
        session.add_excluded(returned_ids)

    return QueryResponse(
        query=result.query,
        results=[
            {
                "text": s.text,
                "source_file": s.source_file,
                "source_type": s.source_type,
                "score": float(round(result.scores[i], 4)),
                "sphere_id": s.id,
                "gravity_field": s.gravity_field if s.gravity_field else None,
                "cluster_id": s.cluster_id,
            }
            for i, s in enumerate(result.spheres)
        ],
        field_affinities={k: float(v) for k, v in result.field_affinities.items()},
        total_spheres=result.total_count,
        timing_ms={k: round(v * 1000, 1) for k, v in result.timing.items()},
        session_id=session.session_id if session else None,
        focus_field=effective_focus if effective_focus else None,
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

def _atomic_save_all():
    """原子地保存所有状态

    JSON 文件先写 .tmp 再 rename 覆盖（同一磁盘分区内 rename 是原子操作）。
    FAISS 索引文件直接保存（faiss.write_index 是库自己的序列化，相对安全）。
    """
    import shutil

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
            logger.info(f"  Atomic save {name}: {path}")
        except Exception as e:
            logger.error(f"  Atomic save FAILED {name}: {e}")
            raise

    # FAISS 有自己的序列化格式，保存到正式路径
    state.faiss_store.save()


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


def _migrate_gravity_fields(
    sphere_store: SphereStore,
    faiss_store: FaissStore,
    registry: Registry,
    field_detector: FieldDetector,
) -> int:
    """为旧数据补算 gravity_field

    启动时执行，检查所有活跃球体：
      - 没有 gravity_field 的 → 从 FAISS 缓存读向量 → 计算
      - 有但场域列表不完整的 → 补全

    Returns:
        更新的球体数量
    """
    if not hasattr(faiss_store, '_vectors') or not faiss_store._vectors:
        return 0

    updated = 0
    known_fields = set(field_detector.fields)

    for sphere in sphere_store.get_active():
        fid = registry.faiss_id(sphere.id)
        if fid is None or fid not in faiss_store._vectors:
            continue
        vec = faiss_store._vectors[fid]

        current_fields = set(sphere.gravity_field.keys())

        if not sphere.gravity_field or current_fields != known_fields:
            sphere.gravity_field = field_detector.compute_gravity_field(vec)
            updated += 1

    if updated > 0:
        logger.info(f"Migrated gravity_field for {updated} spheres")
    return updated
