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

import asyncio
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
from pipeline.generator import AnswerGenerator, get_generator

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
  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Roboto, sans-serif;
    background: #f8f9fb; color: #1e293b; min-height: 100vh;
  }
  .container { max-width: 720px; margin: 0 auto; padding: 0 20px; }

  /* Header */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 0; border-bottom: 1px solid #e9eef2;
  }
  .header-left { display: flex; align-items: center; gap: 10px; }
  .header-logo { font-size: 18px; font-weight: 700; color: #0f172a; letter-spacing: -0.3px; }
  .header-logo span { color: #6366f1; }
  .header-sub { font-size: 12px; color: #94a3b8; margin-top: 1px; }
  .header-right { font-size: 12px; color: #94a3b8; text-align: right; }

  /* Hero search */
  .hero { padding: 60px 0 32px; text-align: center; }
  .hero h2 { font-size: 24px; font-weight: 600; color: #0f172a; margin-bottom: 6px; }
  .hero p { font-size: 14px; color: #64748b; margin-bottom: 28px; }
  .search-box {
    display: flex; align-items: center;
    max-width: 560px; margin: 0 auto;
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.02);
    transition: box-shadow 0.2s, border-color 0.2s;
  }
  .search-box:focus-within {
    border-color: #a5b4fc;
    box-shadow: 0 1px 3px rgba(99,102,241,0.06), 0 4px 20px rgba(99,102,241,0.08);
  }
  .search-box input {
    flex: 1; border: none; outline: none;
    padding: 12px 16px; font-size: 15px; color: #1e293b;
    background: transparent;
  }
  .search-box input::placeholder { color: #94a3b8; }
  .search-box .actions { display: flex; gap: 6px; padding-right: 4px; }
  .search-box select {
    border: none; outline: none; background: #f1f5f9;
    padding: 6px 10px; border-radius: 8px;
    font-size: 12px; color: #475569; cursor: pointer;
  }
  .search-box select:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn {
    padding: 8px 16px; border: none; border-radius: 8px;
    font-size: 13px; font-weight: 500; cursor: pointer;
    transition: background 0.15s, opacity 0.15s;
  }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: #6366f1; color: #fff; }
  .btn-primary:hover:not(:disabled) { background: #4f46e5; }
  .btn-success { background: #10b981; color: #fff; }
  .btn-success:hover:not(:disabled) { background: #059669; }

  /* Toolbar */
  .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; padding: 0 4px; }
  .toolbar-label { font-size: 12px; color: #94a3b8; margin-right: 4px; }
  .pill {
    padding: 4px 14px; border: 1px solid #e2e8f0; border-radius: 20px;
    background: #fff; color: #64748b; font-size: 12px; cursor: pointer;
    transition: all 0.15s;
  }
  .pill:hover { border-color: #c7d2fe; color: #6366f1; }
  .pill.active { background: #6366f1; border-color: #6366f1; color: #fff; }
  .pill.exit { border-color: #fecaca; color: #ef4444; }
  .pill.exit:hover { background: #fef2f2; }

  /* Answer card */
  .answer-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    border: 1px solid #a7f3d0; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 16px;
  }
  .answer-meta { font-size: 11px; color: #6ee7b7; margin-bottom: 8px; display: flex; gap: 12px; align-items: center; }
  .answer-meta .label { color: #94a3b8; }
  .answer-text { font-size: 14px; line-height: 1.7; color: #065f46; white-space: pre-wrap; }
  .answer-error { color: #ef4444; font-size: 13px; }

  /* Result card */
  .result-card {
    background: #fff; border: 1px solid #e9eef2; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 8px;
    transition: border-color 0.15s;
  }
  .result-card:hover { border-color: #cbd5e1; }
  .result-meta {
    display: flex; gap: 8px; font-size: 11px; color: #94a3b8;
    margin-bottom: 6px; flex-wrap: wrap; align-items: center;
  }
  .badge { padding: 1px 8px; border-radius: 4px; font-size: 10px; font-weight: 500; }
  .badge-cluster { background: #eef2ff; color: #6366f1; }
  .badge-field { background: #f1f5f9; color: #64748b; }
  .badge-score { background: #ecfdf5; color: #059669; }
  .result-text {
    font-size: 13px; line-height: 1.6; color: #475569;
    max-height: 120px; overflow-y: auto;
  }
  .result-gf { font-size: 10px; color: #cbd5e1; margin-top: 6px; }

  /* Status bar */
  .status-bar { font-size: 11px; color: #cbd5e1; margin-bottom: 16px; padding: 0 4px; }

  /* Upload */
  .upload-area {
    margin: 40px 0 32px; padding: 20px;
    border: 1px dashed #d1d5db; border-radius: 10px;
    text-align: center; background: #fafbfc;
  }
  .upload-area label {
    display: inline-block; padding: 6px 18px;
    border-radius: 20px; background: #fff; color: #64748b;
    border: 1px solid #e2e8f0; font-size: 12px; cursor: pointer;
    transition: all 0.15s;
  }
  .upload-area label:hover { border-color: #6366f1; color: #6366f1; }
  .upload-area input { display: none; }
  .upload-status { font-size: 11px; color: #94a3b8; margin-top: 8px; }

  /* Empty state */
  .empty-state { text-align: center; padding: 48px 0; color: #94a3b8; }
  .empty-state .icon { font-size: 32px; margin-bottom: 8px; }
  .empty-state p { font-size: 13px; }

  /* Loading */
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .loading { animation: pulse 1.5s ease-in-out infinite; color: #94a3b8; text-align: center; padding: 32px; font-size: 13px; }

  /* Section title */
  .section-title { font-size: 12px; font-weight: 600; color: #94a3b8; margin: 20px 0 10px; padding: 0 4px; }
</style>
</head>
<body>
<div class="container">
  <!-- Header -->
  <div class="header">
    <div class="header-left">
      <div class="header-logo"><span>✦</span> 重力知识库</div>
      <div class="header-sub">FAISS + 重力场路由 + 自动聚类</div>
    </div>
    <div class="header-right" id="sphere-count">加载中...</div>
  </div>

  <!-- Hero -->
  <div class="hero">
    <h2>搜索你的知识库</h2>
    <p>自动聚类 · 引力场路由 · 多模型问答</p>
    <div class="search-box">
      <input id="query-input" type="text" placeholder="Enter 搜索 · Shift+Enter 问答" autofocus>
      <div class="actions">
        <select id="model-select">
          <option value="ollama">Ollama</option>
          <option value="deepseek">DeepSeek</option>
        </select>
        <button class="btn btn-primary" id="search-btn" onclick="search()">搜索</button>
        <button class="btn btn-success" id="ask-btn" onclick="ask()">问答</button>
      </div>
    </div>
  </div>

  <!-- Toolbar -->
  <div class="toolbar" id="toolbar">
    <span class="toolbar-label">聚焦</span>
    <span id="focus-pills"></span>
    <button class="pill exit" id="exit-focus" style="display:none" onclick="resetFocus()">x 清除</button>
  </div>

  <!-- Status -->
  <div class="status-bar" id="session-info"></div>

  <!-- Answer -->
  <div class="answer-card" id="answer-area" style="display:none">
    <div class="answer-meta">
      <span id="answer-model"></span>
      <span id="answer-timing"></span>
      <span class="answer-error" id="answer-error" style="display:none"></span>
    </div>
    <div class="answer-text" id="answer-text"></div>
  </div>

  <!-- Results -->
  <div class="section-title" id="results-title" style="display:none">检索结果</div>
  <div id="results"></div>

  <!-- Upload -->
  <div class="upload-area">
    <label for="file-upload">+ 上传文件（PDF / DOCX / MD / TXT）</label>
    <input type="file" id="file-upload" onchange="uploadFile()">
    <div class="upload-status" id="upload-status"></div>
  </div>
</div>

<script>
// State
var sessionId = localStorage.getItem("gravity_session") || null;
var currentFocus = null;
var requestId = 0;        // request dedup counter
var busy = false;         // global busy lock

// Keyboard shortcuts
document.getElementById("query-input").addEventListener("keydown", function(e) {
  if (e.key === "Enter") {
    if (e.shiftKey) { ask(); } else { search(); }
  }
});

// API call with error resilience
async function api(path, body) {
  var res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  var text = await res.text();  // always read as text first
  if (!res.ok) {
    throw new Error(path + " (HTTP " + res.status + "): " + text.slice(0, 300));
  }
  try {
    return JSON.parse(text);
  } catch(e) {
    throw new Error(path + ": not JSON: " + text.slice(0, 200));
  }
}

// Status
async function loadStatus() {
  try {
    var res = await fetch("/status");
    var text = await res.text();
    var s = JSON.parse(text);
    document.getElementById("sphere-count").textContent =
      s.active_spheres + " spheres \u00b7 " + s.fields.length + " fields";
    var fields = [];
    for (var key in s.field_counts) {
      fields.push({ name: key, count: s.field_counts[key] });
    }
    renderPills(fields);
  } catch(e) {
    document.getElementById("sphere-count").textContent = "connection failed";
  }
}

function renderPills(fields) {
  var html = "";
  for (var i = 0; i < fields.length; i++) {
    var f = fields[i];
    var active = (currentFocus === f.name) ? "active" : "";
    html += "<button class=\"pill " + active + "\" onclick=\"setFocus('" + f.name + "')\">" + f.name + "</button> ";
  }
  document.getElementById("focus-pills").innerHTML = html;
  document.getElementById("exit-focus").style.display = currentFocus ? "inline-block" : "none";
}

// Search with request dedup
async function search() {
  if (busy) { showError("results", "Please wait for current request"); return; }
  var q = document.getElementById("query-input").value.trim();
  if (!q) return;

  busy = true;
  var myReq = ++requestId;
  document.getElementById("search-btn").disabled = true;
  document.getElementById("results").innerHTML = "<div class=\"loading\">searching...</div>";
  document.getElementById("results-title").style.display = "block";
  document.getElementById("answer-area").style.display = "none";
  document.getElementById("query-input").focus();

  try {
    var payload = { query: q, top_k: 5 };
    if (sessionId) payload.session_id = sessionId;
    if (currentFocus) payload.field_focus = currentFocus;

    var data = await api("/query", payload);
    if (myReq !== requestId) return;  // stale, discard

    sessionId = data.session_id;
    if (sessionId) localStorage.setItem("gravity_session", sessionId);
    renderResults(data);
    var sid = data.session_id || "";
    document.getElementById("session-info").textContent =
      "session " + sid.slice(0, 8) + "...";
  } catch(e) {
    if (myReq === requestId) { showError("results", e.message); }
  }
  document.getElementById("search-btn").disabled = false;
  busy = false;
}

function renderResults(data) {
  var r = data.results;
  if (!r || r.length === 0) {
    document.getElementById("results").innerHTML =
      "<div class=\"empty-state\"><div class=\"icon\">&#x1f50d;</div><p>No results found</p></div>";
    return;
  }
  var html = "";
  if (data.field_affinities) {
    var affs = Object.entries(data.field_affinities);
    if (affs.length > 0) {
      html += "<div class=\"status-bar\">Field affinity: " +
        affs.map(function(a) { return a[0] + " " + a[1].toFixed(2); }).join(" \u00b7 ") + "</div>";
    }
  }
  for (var i = 0; i < r.length; i++) {
    var item = r[i];
    html += "<div class=\"result-card\">";
    html += "<div class=\"result-meta\">";
    html += "<span class=\"badge badge-cluster\">" + (item.source_type || "cluster" + item.cluster_id) + "</span>";
    html += "<span class=\"badge badge-score\">" + (item.score || 0).toFixed(3) + "</span>";
    if (item.source_file) {
      html += "<span style=\"color:#cbd5e1\">" + escapeHtml(item.source_file) + "</span>";
    }
    html += "</div>";
    html += "<div class=\"result-text\">" + escapeHtml(item.text) + "</div>";
    html += "</div>";
  }
  document.getElementById("results").innerHTML = html;
}

function showError(id, msg) {
  document.getElementById(id).innerHTML =
    "<div class=\"empty-state\"><p>&#x2716; " + escapeHtml(msg) + "</p></div>";
}

function escapeHtml(s) {
  if (!s) return "";
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Focus controls
function setFocus(field) {
  currentFocus = field;
  renderPills([]);
  loadStatus();
  var q = document.getElementById("query-input").value.trim();
  if (q) search();
}

function resetFocus() {
  currentFocus = null;
  renderPills([]);
  loadStatus();
}

// Backend availability
async function loadBackends() {
  try {
    var res = await fetch("/backends");
    var text = await res.text();
    var backends = JSON.parse(text);
    var sel = document.getElementById("model-select");
    for (var i = 0; i < sel.options.length; i++) {
      var opt = sel.options[i];
      if (!backends.available[opt.value]) {
        opt.disabled = true;
        opt.textContent += " (offline)";
      }
    }
  } catch(e) {}
}

// Ask
async function ask() {
  if (busy) { showError("results", "Please wait for current request"); return; }
  var q = document.getElementById("query-input").value.trim();
  if (!q) return;

  busy = true;
  var myReq = ++requestId;
  document.getElementById("ask-btn").disabled = true;
  document.getElementById("results-title").style.display = "block";
  document.getElementById("query-input").focus();

  // Clear previous answer
  var area = document.getElementById("answer-area");
  area.style.display = "block";
  document.getElementById("answer-text").textContent = "Generating...";
  document.getElementById("answer-text").className = "answer-text";
  document.getElementById("answer-error").style.display = "none";
  document.getElementById("results").innerHTML = "";

  try {
    var payload = {
      query: q,
      model: document.getElementById("model-select").value,
      top_k: 5,
    };
    if (sessionId) payload.session_id = sessionId;

    var data = await api("/ask", payload);
    if (myReq !== requestId) return;  // stale, discard

    // Track session for follow-ups
    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem("gravity_session", sessionId);
    }

    document.getElementById("answer-model").innerHTML =
      "<span class=\"label\">&#x1f916;</span> " + (data.backend || "?") +
      " <span style=\"color:#94a3b8\">" + (data.model || "") + "</span>";
    document.getElementById("answer-timing").innerHTML =
      "<span class=\"label\">&#x26a1;</span> gen " + (data.generation_ms || 0).toFixed(0) + "ms" +
      " <span style=\"color:#94a3b8\">\u00b7 search " +
      ((data.retrieval_ms && data.retrieval_ms.total) || 0).toFixed(0) + "ms</span>";

    if (data.error) {
      document.getElementById("answer-text").className = "answer-error";
      document.getElementById("answer-text").textContent = "&#x2716; " + data.error;
    } else {
      document.getElementById("answer-text").textContent = data.answer || "(no answer)";
      document.getElementById("session-info").textContent = "session " + (data.session_id || "").slice(0, 8) + "...";
    }

    if (data.results && data.results.length > 0) {
      var html = "<div class=\"section-title\">Reference context</div>";
      for (var i = 0; i < data.results.length; i++) {
        var item = data.results[i];
        html += "<div class=\"result-card\"><div class=\"result-meta\">";
        html += "<span class=\"badge badge-score\">" + (item.score || 0).toFixed(3) + "</span>";
        html += "<span class=\"badge badge-field\">" + escapeHtml(item.source_type || "") + "</span>";
        html += "</div><div class=\"result-text\">" + escapeHtml(item.text) + "</div></div>";
      }
      document.getElementById("results").innerHTML = html;
    }
  } catch(e) {
    if (myReq === requestId) {
      document.getElementById("answer-text").className = "answer-error";
      document.getElementById("answer-text").textContent = "&#x2716; " + e.message;
    }
  }
  document.getElementById("ask-btn").disabled = false;
  busy = false;
}

// Upload
async function uploadFile() {
  if (busy) { showError("results", "Please wait"); return; }
  var fileInput = document.getElementById("file-upload");
  var file = fileInput.files[0];
  if (!file) return;

  busy = true;
  var statusEl = document.getElementById("upload-status");
  statusEl.textContent = "&#x2191; Uploading... (" + (file.size / 1024).toFixed(0) + "KB)";

  var form = new FormData();
  form.append("file", file);
  form.append("source_type", "");

  try {
    var res = await fetch("/upload", { method: "POST", body: form });
    var text = await res.text();
    if (!res.ok) throw new Error(text.slice(0, 200));
    var data = JSON.parse(text);
    statusEl.textContent = "&#x2713; " + data.new_spheres + " new spheres (" + data.timing_ms.total.toFixed(0) + "ms)";
    loadStatus();
  } catch(e) {
    statusEl.textContent = "&#x2716; Upload failed: " + e.message;
  }
  busy = false;
  fileInput.value = "";
}

// Init
loadBackends();
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
        self.generator = AnswerGenerator()

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


class AskRequest(BaseModel):
    query: str
    model: str = ""                         # 空=用配置默认值
    top_k: int = 5
    field_focus: Optional[str] = None
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    model: str
    backend: str
    generation_ms: float
    results: List[dict]
    field_affinities: dict
    retrieval_ms: dict
    session_id: Optional[str] = None
    focus_field: Optional[str] = None
    error: Optional[str] = None


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
                        loop = asyncio.get_event_loop()
                        centroids, labels, _ = await loop.run_in_executor(
                            None, state.cluster_engine.fit_predict, vectors_arr
                        )

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

        # WAL 清理：删除 7 天前的已完成/已回滚条目
        state.wal.clean_old_entries(max_age_hours=24 * 7)

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
                    # run_in_executor 避免阻塞事件循环（sklearn KMeans 是同步的）
                    loop = asyncio.get_event_loop()
                    centroids, labels, scores = await loop.run_in_executor(
                        None, state.cluster_engine.fit_predict, vectors_arr
                    )

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

    # 清理：上传成功后删除原始文件
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning(f"Failed to delete uploaded file {file.filename}: {e}")

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
        timing_ms={k: float(round(v * 1000, 1)) for k, v in result.timing.items()},
        session_id=session.session_id if session else None,
        focus_field=effective_focus if effective_focus else None,
    )


# ──────────────────────────────────────────────
# 端点：问答
# ──────────────────────────────────────────────

@app.post("/ask")
async def ask(request: AskRequest):
    """检索上下文并生成回答

    先走完整检索流程，再根据选择的模型生成回答。
    支持 model=ollama|deepseek|agent（空=配置默认）。
    """
    if not state.is_loaded():
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Upload files first.")

    # 0. 会话管理 + 对话历史
    session = state.sessions.get_or_create(request.session_id)
    history_text = session.history_text

    # 1. 检索上下文
    retrieval_result: RetrievalResult = state.retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        field_focus=request.field_focus,
    )

    # 2. 提取上下文文本
    context_texts = [s.text for s in retrieval_result.spheres]
    context_spheres = [
        {
            "text": s.text[:200] + "..." if len(s.text) > 200 else s.text,
            "source_file": s.source_file,
            "source_type": s.source_type,
            "score": round(retrieval_result.scores[i], 4) if i < len(retrieval_result.scores) else 0.0,
            "cluster_id": s.cluster_id,
        }
        for i, s in enumerate(retrieval_result.spheres)
    ]

    # 3. 生成回答（带上对话历史）
    model_name = request.model or None  # None = config default
    answer_result = state.generator.generate(
        query=request.query,
        context_texts=context_texts,
        model=model_name,
        context_spheres=context_spheres,
        history=history_text,
    )

    # 4. 记录问答到会话历史
    if answer_result.text and not answer_result.error:
        session.add_history(request.query, answer_result.text)

    # 构造响应（手动转换所有值为 Python 原生类型，避免 numpy 序列化问题）
    response_data = {
        "query": str(request.query),
        "answer": str(answer_result.text) if answer_result.text else "",
        "model": str(answer_result.model) if answer_result.model else "",
        "backend": str(answer_result.backend) if answer_result.backend else "",
        "generation_ms": float(round(answer_result.timing_ms, 1)) if answer_result.timing_ms else 0.0,
        "results": [
            {
                "text": str(r.get("text", "")),
                "source_file": str(r.get("source_file", "")),
                "source_type": str(r.get("source_type", "")),
                "score": float(r.get("score", 0.0)),
                "cluster_id": int(r.get("cluster_id", -1)),
            }
            for r in context_spheres
        ],
        "field_affinities": {str(k): float(v) for k, v in retrieval_result.field_affinities.items()},
        "retrieval_ms": {str(k): float(round(v * 1000, 1)) for k, v in retrieval_result.timing.items()},
        "session_id": session.session_id,
        "error": str(answer_result.error) if answer_result.error else None,
    }
    return JSONResponse(content=response_data)


# ──────────────────────────────────────────────
# 端点：后端列表
# ──────────────────────────────────────────────

@app.get("/backends")
async def list_backends():
    """列出可用的生成后端"""
    gen = get_generator()
    backends = gen.list_backends()
    available = {k: gen.is_available(k) for k in backends}
    return {"backends": backends, "available": available, "default": "ollama"}


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
