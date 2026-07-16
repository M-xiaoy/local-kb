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
from pipeline.rewriter import TextRewriter
from pipeline.connections import ConnectionDetector
from pipeline.role_table import RoleTable
from pipeline.hierarchy import HierarchyGrower, LevelingConfig
from storage.faiss_store import FaissStore
from storage.registry import Registry
from storage.sphere_store import SphereStore, Sphere, make_sphere_id
from storage.calibrator import SphereCalibrator
from storage.wal import WalManager, WAL_READY, WAL_COMMITTING
from retrieval.field_detector import FieldDetector
from retrieval.diversity_sorter import DiversitySorter
from retrieval.retriever import Retriever, RetrievalResult
from retrieval.reranker import LocalReranker
from retrieval.activation import ActivationPropagator
from retrieval.session_manager import SessionManager
from retrieval.cluster_engine import ClusterEngine
from retrieval.tools.navigate import navigate_sphere
from retrieval.tools.explore import explore_cluster
from retrieval.tools.trace import trace_conversation
from retrieval.tools.bridge import find_bridge
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
  /* ── Reset & Base ── */
  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Roboto, sans-serif;
    background: #f8f9fb; color: #1e293b; min-height: 100vh;
    display: flex; flex-direction: column;
  }
  body.chat-mode { background: #fff; height: 100vh; overflow: hidden; }

  /* ── Layout ── */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; border-bottom: 1px solid #e9eef2;
    background: #fff; flex-shrink: 0; z-index: 10;
    position: relative;
  }
  .header-left { display: flex; align-items: center; gap: 10px; }
  .header-logo { font-size: 17px; font-weight: 700; color: #0f172a; letter-spacing: -0.3px; }
  .header-logo span { color: #6366f1; }
  .header-sub { font-size: 11px; color: #94a3b8; }
  .header-right { font-size: 12px; color: #94a3b8; display: flex; align-items: center; gap: 12px; }

  /* ── Hero (initial state) ── */
  .hero {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 40px 20px 60px; transition: all 0.4s ease;
  }
  .chat-mode .hero { display: none; }
  .hero-icon { font-size: 40px; margin-bottom: 12px; }
  .hero h2 { font-size: 24px; font-weight: 600; color: #0f172a; margin-bottom: 6px; }
  .hero p { font-size: 14px; color: #64748b; margin-bottom: 28px; }
  .hero .search-box { max-width: 560px; width: 100%; }

  /* ── Chat messages ── */
  .chat-area {
    flex: 1; overflow-y: auto; padding: 4px 0 0;
    display: none; flex-direction: column; min-height: 0;
  }
  .chat-mode .chat-area { display: flex; }
  .chat-inner { max-width: 720px; margin: 0 auto; padding: 0 20px 16px; width: 100%; }

  /* ── Message Cards ── */
  .msg-group { margin-bottom: 16px; }
  .msg-user {
    background: #f1f5f9; border-radius: 12px 12px 4px 12px;
    padding: 10px 16px; margin-bottom: 8px;
    font-size: 14px; line-height: 1.6; color: #1e293b;
    max-width: 80%; margin-left: auto;
  }
  .msg-user .label { font-size: 11px; color: #94a3b8; margin-bottom: 4px; }
  .msg-assistant {
    background: #fff; border: 1px solid #e9eef2; border-radius: 12px;
    padding: 16px 18px; position: relative;
  }

  /* ── Answer text ── */
  .answer-text {
    font-size: 14px; line-height: 1.7; color: #1e293b;
    white-space: pre-wrap; margin-bottom: 12px;
  }
  .answer-text:empty { display: none; }
  .answer-error { color: #ef4444; font-size: 13px; }
  .answer-meta {
    font-size: 11px; color: #94a3b8;
    display: flex; gap: 12px; align-items: center; margin-bottom: 8px;
  }
  .answer-meta .label { color: #94a3b8; }

  /* ── Reference toggles ── */
  .ref-header {
    border-top: 1px solid #f1f5f9; padding-top: 10px; margin-top: 2px;
    display: flex; align-items: center; justify-content: space-between;
    cursor: pointer; user-select: none;
  }
  .ref-header:hover { color: #6366f1; }
  .ref-header span { font-size: 12px; color: #94a3b8; font-weight: 500; }
  .ref-header .arrow { font-size: 10px; transition: transform 0.2s; }
  .ref-header .arrow.open { transform: rotate(180deg); }
  .ref-body { display: none; margin-top: 10px; }
  .ref-body.open { display: block; }

  /* ── Result cards (in references) ── */
  .result-card {
    background: #fafbfc; border: 1px solid #e9eef2; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 6px;
  }
  .result-meta {
    display: flex; gap: 6px; font-size: 11px; color: #94a3b8;
    margin-bottom: 5px; flex-wrap: wrap; align-items: center;
  }
  .badge { padding: 1px 7px; border-radius: 4px; font-size: 10px; font-weight: 500; white-space: nowrap; }
  .badge-cluster { background: #eef2ff; color: #6366f1; }
  .badge-field { background: #f1f5f9; color: #64748b; }
  .badge-score { background: #ecfdf5; color: #059669; }
  .result-text {
    font-size: 13px; line-height: 1.6; color: #475569;
    max-height: 60px; overflow: hidden; transition: max-height 0.25s;
  }
  .result-text.expanded { max-height: none; }
  .result-expand {
    font-size: 11px; color: #a5b4fc; cursor: pointer;
    margin-top: 4px; display: inline-block;
  }
  .result-expand:hover { color: #6366f1; }

  /* ── Search Box (shared) ── */
  .search-box {
    display: flex; align-items: center; width: 100%;
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
    padding: 10px 14px; font-size: 14px; color: #1e293b;
    background: transparent;
  }
  .search-box input::placeholder { color: #94a3b8; }
  .search-box .actions { display: flex; gap: 4px; align-items: center; padding-right: 4px; }
  .search-box select {
    border: none; outline: none; background: #f1f5f9;
    padding: 5px 8px; border-radius: 8px;
    font-size: 11px; color: #475569; cursor: pointer;
  }
  .search-box select:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn {
    padding: 7px 14px; border: none; border-radius: 8px;
    font-size: 12px; font-weight: 500; cursor: pointer;
    transition: background 0.12s, opacity 0.12s;
  }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: #6366f1; color: #fff; }
  .btn-primary:hover:not(:disabled) { background: #4f46e5; }
  .btn-icon {
    padding: 6px 8px; border: none; border-radius: 8px;
    font-size: 16px; cursor: pointer; background: transparent;
    transition: background 0.12s; line-height: 1;
  }
  .btn-icon:hover { background: #f1f5f9; }

  /* ── Bottom input bar (chat mode) ── */
  .input-bar {
    flex-shrink: 0; padding: 10px 20px 16px; border-top: 1px solid #e9eef2;
    background: #fff; display: none; flex-shrink: 0;
  }
  .chat-mode .input-bar { display: block; }
  .input-bar .search-box { max-width: 720px; margin: 0 auto; }
  .focus-row {
    max-width: 720px; margin: 6px auto 0; padding: 0 8px;
    display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
  }
  .focus-label { font-size: 11px; color: #94a3b8; }
  .pill {
    padding: 3px 12px; border: 1px solid #e2e8f0; border-radius: 20px;
    background: #fff; color: #64748b; font-size: 11px; cursor: pointer;
    transition: all 0.12s;
  }
  .pill:hover { border-color: #c7d2fe; color: #6366f1; }
  .pill.active { background: #6366f1; border-color: #6366f1; color: #fff; }

  /* ── Upload (collapsible panel in header) ── */
  .upload-panel {
    position: absolute; top: 100%; right: 20px; z-index: 20;
    background: #fff; border: 1px solid #e9eef2; border-radius: 10px;
    padding: 14px 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    width: 280px; display: none;
  }
  .upload-panel.open { display: block; }
  .upload-panel h4 { font-size: 12px; color: #475569; margin-bottom: 8px; }
  .upload-drop {
    border: 1.5px dashed #d1d5db; border-radius: 8px;
    padding: 16px; text-align: center; cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    font-size: 12px; color: #94a3b8;
  }
  .upload-drop:hover { border-color: #a5b4fc; background: #f8f9ff; }
  .upload-drop.dragover { border-color: #6366f1; background: #eef2ff; }
  .upload-drop input { display: none; }
  .upload-status { font-size: 11px; color: #94a3b8; margin-top: 8px; }
  .upload-btn-header {
    padding: 4px 12px; border: 1px solid #e2e8f0; border-radius: 12px;
    background: #fff; color: #64748b; font-size: 11px; cursor: pointer;
    transition: all 0.12s;
  }
  .upload-btn-header:hover { border-color: #a5b4fc; color: #6366f1; }

  /* ── Loading skeleton ── */
  @keyframes shimmer { 0% { background-position: -200px 0; } 100% { background-position: calc(200px + 100%) 0; } }
  .skeleton {
    background: linear-gradient(90deg, #f1f5f9 25%, #e9eef2 50%, #f1f5f9 75%);
    background-size: 200px 100%; animation: shimmer 1.5s infinite;
    border-radius: 8px; margin-bottom: 8px;
  }
  .skeleton-line { height: 14px; width: 100%; margin-bottom: 6px; }
  .skeleton-line.short { width: 60%; }
  .skeleton-card { height: 80px; border-radius: 8px; margin-bottom: 6px; }

  /* ── Field affinity ── */
  .affinity-bar { font-size: 10px; color: #cbd5e1; margin: 4px 0 8px; }

  /* ── Empty state ── */
  .empty-chat { text-align: center; padding: 60px 0; color: #94a3b8; }
  .empty-chat .icon { font-size: 28px; margin-bottom: 6px; }
  .empty-chat p { font-size: 13px; }

  /* ── Responsive ── */
  @media (max-width: 600px) {
    .header { padding: 10px 14px; }
    .header-sub { display: none; }
    .chat-inner { padding: 0 12px; }
    .msg-user { max-width: 90%; font-size: 13px; }
    .msg-assistant { padding: 12px 14px; }
    .input-bar { padding: 8px 12px 12px; }
    .hero h2 { font-size: 20px; }
    .upload-panel { right: 10px; width: 240px; }
    .search-box select { font-size: 10px; padding: 4px 6px; }
    .search-box input { padding: 8px 10px; font-size: 13px; }
  }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <div class="header-logo"><span>✦</span> 重力知识库</div>
    <div class="header-sub">FAISS + 重力场路由 + 自动聚类</div>
  </div>
  <div class="header-right">
    <span id="sphere-count">加载中...</span>
    <button class="upload-btn-header" id="upload-toggle" onclick="toggleUploadPanel()">\U0001f4ce 上传</button>
  </div>

  <!-- Upload panel (collapsible) -->
  <div class="upload-panel" id="upload-panel">
    <h4>上传文件到知识库</h4>
    <div class="upload-drop" id="upload-drop"
         onclick="document.getElementById('file-input').click()"
         ondragover="event.preventDefault(); this.classList.add('dragover')"
         ondragleave="this.classList.remove('dragover')"
         ondrop="event.preventDefault(); this.classList.remove('dragover'); handleDrop(event)">
      <div>\U0001f4c2 拖拽文件到此处</div>
      <div style="margin-top:6px;font-size:11px;color:#cbd5e1">PDF · DOCX · MD · TXT</div>
      <input type="file" id="file-input" accept=".pdf,.docx,.md,.txt" onchange="handleFileSelect(event)">
    </div>
    <div class="upload-status" id="upload-status"></div>
  </div>
</div>

<!-- Hero (initial) -->
<div class="hero" id="hero">
  <div class="hero-icon">✦</div>
  <h2>搜索你的知识库</h2>
  <p>自动聚类 · 引力场路由 · 多模型问答</p>
  <div class="search-box">
    <input id="query-input-hero" type="text" placeholder="输入问题，Enter 发送" autofocus>
    <div class="actions">
      <select id="model-select-hero">
        <option value="ollama">Ollama</option>
        <option value="deepseek">DeepSeek</option>
      </select>
      <button class="btn btn-primary" onclick="doQuery('hero')">发送</button>
    </div>
  </div>
</div>

<!-- Chat messages -->
<div class="chat-area" id="chat-area">
  <div class="chat-inner" id="chat-inner">
    <div class="empty-chat" id="empty-chat">
      <div class="icon">✦</div>
      <p>搜索或提问，知识库会从已有文档中查找答案</p>
    </div>
  </div>
</div>

<!-- Input bar (chat mode) -->
<div class="input-bar" id="input-bar">
  <div class="search-box">
    <button class="btn-icon" onclick="toggleUploadPanel()" title="上传文件">\U0001f4ce</button>
    <input id="query-input-chat" type="text" placeholder="输入问题，Enter 发送">
    <div class="actions">
      <select id="model-select-chat">
        <option value="ollama">Ollama</option>
        <option value="deepseek">DeepSeek</option>
      </select>
      <button class="btn btn-primary" onclick="doQuery('chat')">发送</button>
    </div>
  </div>
  <div class="focus-row" id="focus-row">
    <span class="focus-label">聚焦</span>
    <span id="focus-pills"></span>
  </div>
</div>

<script>
// State
var sessionId = localStorage.getItem("gravity_session") || null;
var currentFocus = null;
var requestId = 0;
var busy = false;
var initialized = false;

// Keyboard shortcuts
function setupKeyboard() {
  document.getElementById("query-input-hero").addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doQuery("hero"); }
  });
  document.getElementById("query-input-chat").addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doQuery("chat"); }
  });
}

// Safe API call
async function api(path, body) {
  var res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  var text = await res.text();
  if (!res.ok) throw new Error(path + " (HTTP " + res.status + "): " + text.slice(0, 300));
  try { return JSON.parse(text); }
  catch(e) { throw new Error(path + ": not JSON: " + text.slice(0, 200)); }
}

// Status & pills
async function loadStatus() {
  try {
    var res = await fetch("/status");
    var s = JSON.parse(await res.text());
    document.getElementById("sphere-count").textContent =
      s.active_spheres + " spheres · " + s.fields.length + " fields";
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
    html += "<button class=\"pill " + active + "\" onclick=\"setFocus('" + f.name.replace(/'/g, "\\'") + "')\">" + escapeHtml(f.name) + "</button> ";
  }
  document.getElementById("focus-pills").innerHTML = html;
}

function setFocus(field) {
  currentFocus = (currentFocus === field) ? null : field;
  loadStatus();
}

// Backend availability
async function loadBackends() {
  try {
    var res = await fetch("/backends");
    var backends = JSON.parse(await res.text());
    ["model-select-hero", "model-select-chat"].forEach(function(id) {
      var sel = document.getElementById(id);
      for (var i = 0; i < sel.options.length; i++) {
        var opt = sel.options[i];
        if (!backends.available[opt.value]) {
          opt.disabled = true;
          opt.textContent += " (offline)";
        }
      }
    });
  } catch(e) {}
}

// Main query (merged search + Q&A)
async function doQuery(source) {
  if (busy) return;
  var inputId = (source === "hero") ? "query-input-hero" : "query-input-chat";
  var modelId = (source === "hero") ? "model-select-hero" : "model-select-chat";
  var q = document.getElementById(inputId).value.trim();
  if (!q) return;

  busy = true;
  var myReq = ++requestId;

  // Switch to chat mode on first query
  if (!initialized) {
    document.body.classList.add("chat-mode");
    initialized = true;
    setTimeout(function() {
      document.getElementById("query-input-chat").focus();
    }, 100);
  }

  // Add user message bubble
  var inner = document.getElementById("chat-inner");
  var empty = document.getElementById("empty-chat");
  if (empty) empty.style.display = "none";

  var msgGroup = document.createElement("div");
  msgGroup.className = "msg-group";
  msgGroup.id = "msg-" + myReq;
  msgGroup.innerHTML = "<div class=\"msg-user\"><div class=\"label\">You</div>" + escapeHtml(q) + "</div>";
  inner.appendChild(msgGroup);

  // Add skeleton
  var skeleton = document.createElement("div");
  skeleton.className = "msg-assistant";
  skeleton.id = "skeleton-" + myReq;
  skeleton.innerHTML = "<div class=\"skeleton skeleton-line\"></div><div class=\"skeleton skeleton-line short\"></div>";
  msgGroup.appendChild(skeleton);

  // Scroll down
  document.getElementById("chat-area").scrollTop = document.getElementById("chat-area").scrollHeight;

  // Clear input
  document.getElementById(inputId).value = "";
  document.getElementById(inputId).focus();

  // Sync model selector
  var selectedModel = document.getElementById(modelId).value;
  document.getElementById("model-select-hero").value = selectedModel;
  document.getElementById("model-select-chat").value = selectedModel;

  try {
    var payload = {
      query: q,
      model: selectedModel,
      top_k: 5,
    };
    if (sessionId) payload.session_id = sessionId;
    if (currentFocus) payload.field_focus = currentFocus;

    var data = await api("/ask", payload);
    if (myReq !== requestId) return; // stale

    // Track session
    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem("gravity_session", sessionId);
    }

    // Remove skeleton, render answer + refs
    var skel = document.getElementById("skeleton-" + myReq);
    if (skel) skel.remove();
    renderAnswer(msgGroup, data);
  } catch(e) {
    var skel = document.getElementById("skeleton-" + myReq);
    if (skel) {
      skel.innerHTML = "<div class=\"answer-error\">\u2716 " + escapeHtml(e.message) + "</div>";
    }
  }

  // Scroll latest message into view
  if (msgGroup) {
    setTimeout(function() {
      msgGroup.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 50);
  }

  busy = false;
}

function renderAnswer(container, data) {
  var html = "<div class=\"msg-assistant\">";

  // Meta info
  html += "<div class=\"answer-meta\">";
  html += "<span><span class=\"label\">\U0001f916</span> " + escapeHtml(data.backend || "?") + " " + (data.model ? "<span style=\"color:#94a3b8\">" + escapeHtml(data.model) + "</span>" : "") + "</span>";
  if (data.generation_ms) {
    html += "<span><span class=\"label\">\u26a1</span> " + (data.generation_ms).toFixed(0) + "ms</span>";
  }
  html += "</div>";

  // Answer
  if (data.error) {
    html += "<div class=\"answer-error\">\u2716 " + escapeHtml(data.error) + "</div>";
  } else {
    html += "<div class=\"answer-text\">" + escapeHtml(data.answer || "(no answer)") + "</div>";
  }

  // Field affinity
  if (data.field_affinities) {
    var affs = Object.entries(data.field_affinities);
    if (affs.length > 0) {
      html += "<div class=\"affinity-bar\">\u573a\u57df\u4eb2\u548c: " +
        affs.map(function(a) { return a[0] + " " + Number(a[1]).toFixed(2); }).join(" \u00b7 ") + "</div>";
    }
  }

  // Collapsible references
  if (data.results && data.results.length > 0) {
    var refId = "ref-" + (requestId);
    html += "<div class=\"ref-header\" onclick=\"toggleRef('" + refId + "')\">";
    html += "<span>\U0001f4c4 " + data.results.length + " \u6761\u53c2\u8003\u6765\u6e90</span>";
    html += "<span class=\"arrow\" id=\"arrow-" + refId + "\">\u25bc</span>";
    html += "</div>";
    html += "<div class=\"ref-body\" id=\"" + refId + "\">";

    for (var i = 0; i < data.results.length; i++) {
      var item = data.results[i];
      var resultId = refId + "-r" + i;
      html += "<div class=\"result-card\">";
      html += "<div class=\"result-meta\">";
      html += "<span class=\"badge badge-cluster\">" + escapeHtml(item.source_type || "\u7c87" + (item.cluster_id >= 0 ? item.cluster_id : "?")) + "</span>";
      html += "<span class=\"badge badge-score\">" + (item.score || 0).toFixed(3) + "</span>";
      if (item.source_file) {
        html += "<span style=\"color:#cbd5e1\">" + escapeHtml(item.source_file) + "</span>";
      }
      html += "</div>";
      html += "<div class=\"result-text\" id=\"" + resultId + "-text\">" + escapeHtml(item.text) + "</div>";
      html += "<span class=\"result-expand\" id=\"" + resultId + "-toggle\" onclick=\"toggleResultText('" + resultId + "')\">\u5c55\u5f00\u5168\u90e8</span>";
      html += "</div>";
    }

    html += "</div>";
  }

  html += "</div>";
  container.innerHTML += html;
}

function toggleRef(refId) {
  var body = document.getElementById(refId);
  var arrow = document.getElementById("arrow-" + refId);
  if (!body) return;
  body.classList.toggle("open");
  if (arrow) arrow.classList.toggle("open");
}

function toggleResultText(id) {
  var text = document.getElementById(id + "-text");
  var toggle = document.getElementById(id + "-toggle");
  if (!text || !toggle) return;
  var expanded = text.classList.toggle("expanded");
  toggle.textContent = expanded ? "\u6536\u8d77" : "\u5c55\u5f00\u5168\u90e8";
  setTimeout(function() {
    var group = document.getElementById(refId);
    if (group) {
      var container = group.closest('.msg-group');
      if (container) container.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, 50);
}

// Upload
function toggleUploadPanel() {
  var panel = document.getElementById("upload-panel");
  panel.classList.toggle("open");
}

function handleFileSelect(e) {
  var file = e.target.files[0];
  if (file) uploadFile(file);
}

function handleDrop(e) {
  var file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  var statusEl = document.getElementById("upload-status");
  statusEl.textContent = "\u2191 Uploading... (" + (file.size / 1024).toFixed(0) + "KB)";

  var form = new FormData();
  form.append("file", file);
  form.append("source_type", "");

  try {
    var res = await fetch("/upload", { method: "POST", body: form });
    var text = await res.text();
    if (!res.ok) throw new Error(text.slice(0, 200));
    var data = JSON.parse(text);
    statusEl.textContent = "\u2713 " + data.new_spheres + " new spheres (" + data.timing_ms.total.toFixed(0) + "ms)";
    loadStatus();
  } catch(e) {
    statusEl.textContent = "\u2716 " + e.message;
  }
}

// Helpers
function escapeHtml(s) {
  if (!s) return "";
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// Init
setupKeyboard();
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
        self.rewriter = TextRewriter()
        self.calibrator = SphereCalibrator()
        self.conn_detector = None  # 启动时按需创建
        self.role_table = RoleTable()
        self.hierarchy = HierarchyGrower(
            sphere_store=self.sphere_store,
            role_table=self.role_table,
            vector_provider=self._get_vector_cache,
            config=LevelingConfig(),
        )
        self.retriever = Retriever(
            embedder=self.embedder,
            faiss_store=self.faiss_store,
            registry=self.registry,
            sphere_store=self.sphere_store,
            field_detector=self.field_detector,
            diversity_sorter=self.sorter,
            rewriter=self.rewriter,
        )
        self.retriever.attach_role_table(self.role_table)
        self.uploads_dir = Path(cfg_paths.uploads_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.wal = WalManager(str(Path(cfg_paths.wal_dir)))
        self.sessions = SessionManager()
        self.cluster_engine = ClusterEngine()
        self.generator = AnswerGenerator()

    def ensure_connections(self):
        """按需初始化连接检测器并挂载到 retriever"""
        if self.conn_detector is None:
            from pipeline.connections import ConnectionDetector
            self.conn_detector = ConnectionDetector(
                self.sphere_store, self.faiss_store._vectors
            )
            self.conn_detector.load()
            self.retriever.attach_connections(
                self.conn_detector.get_connections,
                type_checker=self.conn_detector.get_connection_type,
            )
            self.calibrator.attach(self.sphere_store, self.faiss_store._vectors)

    def ensure_role_table(self):
        """按需加载角色共现表"""
        if self.role_table.entity_count == 0:
            self.role_table.load()
            if self.role_table.entity_count == 0 and self.sphere_store.count > 0:
                # 首次构建
                self.role_table.build_for_spheres(
                    self.sphere_store.get_active()
                )
                self.role_table.save()
                logger.info(
                    f"Built role table from {self.sphere_store.count} spheres: "
                    f"{self.role_table.entity_count} entities"
                )

    def _get_vector_cache(self, sphere_id: str):
        """向量缓存查询（给 hierarchy grower 用）"""
        fid = self.registry.faiss_id(sphere_id)
        if fid is not None and fid in self.faiss_store._vectors:
            return self.faiss_store._vectors[fid]
        return None

    def build_hierarchy(self):
        """全量构建层级：grow → embed → FAISS add → save"""
        stats = self.hierarchy.grow()

        if stats.get("level1", 0) > 0:
            # 计算概念向量并写入 FAISS
            concepts = self.hierarchy.embed_concepts()
            if concepts:
                cids = []
                cvecs = []
                for cid, vec in concepts:
                    cids.append(cid)
                    cvecs.append(vec)

                vectors_arr = np.stack(cvecs, axis=0)
                ids_arr = np.empty(len(cids), dtype=np.int64)
                self.faiss_store.add(vectors_arr, ids_arr)

                # 注册概念球体到 registry
                for i, cid in enumerate(cids):
                    self.registry.register(cid, int(ids_arr[i]))

                logger.info(
                    f"Hierarchy: {len(concepts)} concepts added to FAISS"
                )

        # 持久化
        self.sphere_store.save()
        self.faiss_store.save()
        self.registry.save()

        logger.info(f"Hierarchy build complete: {stats}")
        return stats

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
    mode: str = "gravity"               # simple | gravity | deep
    use_activation: Optional[bool] = None # 覆盖激活传播开关
    use_reranker: Optional[bool] = None   # 覆盖重排器开关
    max_hops: int = 2                     # 激活传播跳数


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
    top_k: int = 15                          # 给模型更多上下文（排序靠前的最相关）
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
    hierarchy: dict = {}


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

        # ── 构建层级（社区检测→等级划分） ──
        if sphere_count >= 5:
            try:
                state.ensure_role_table()
                stats = state.build_hierarchy()
                logger.info(f"Hierarchy built on startup: {stats}")
            except Exception as e:
                logger.warning(f"Hierarchy build on startup failed: {e}")

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
    hierarchy = state.hierarchy.stats() if hasattr(state, 'hierarchy') else {}
    return StatusResponse(
        total_spheres=total,
        active_spheres=active,
        faiss_vectors=state.faiss_store.count,
        fields=fields,
        field_counts=field_counts,
        hierarchy=hierarchy,
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
        raw_chunks = chunk_text(result.text, source_type=source_type)
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

    # ── 10. 角色共现表增量更新 ───────────────
    t9 = time.time()
    if new_spheres > 0 and auto_rebuild:
        for sphere_id in added_sphere_ids:
            sphere = state.sphere_store.get(sphere_id)
            if sphere and sphere.text:
                state.role_table.register_text(sphere.id, sphere.text)
        state.role_table.save()
        logger.debug(f"Role table: +{new_spheres} spheres, "
                     f"{state.role_table.entity_count} total entities")
    timings["role_table"] = time.time() - t9

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

    # 确保连接层已初始化（如果是 gravity/deep 模式）
    if request.mode in ("gravity", "deep"):
        state.ensure_connections()

    result: RetrievalResult = state.retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        fetch_k=request.fetch_k,
        field_focus=effective_focus,
        exclude_ids=list(exclude),
        mode=request.mode,
        use_activation=request.use_activation,
        use_reranker=request.use_reranker,
        max_hops=request.max_hops,
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

    # 1. 检索上下文（纯 FAISS 相似度）
    # 使用 gravity 模式获取更丰富的上下文
    state.ensure_connections()
    retrieval_result: RetrievalResult = state.retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        field_focus=request.field_focus,
        mode="gravity",
        use_reranker=False,
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

        # 重建层级
        try:
            state.ensure_role_table()
            stats = state.build_hierarchy()
            logger.info(f"Hierarchy rebuilt: {stats}")
        except Exception as e:
            logger.warning(f"Hierarchy rebuild failed: {e}")

        return {
            "status": "ok",
            "rebuilt": len(active_spheres),
            "fields": state.field_detector.fields,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {e}")


# ──────────────────────────────────────────────
# 端点：重写文本
# ──────────────────────────────────────────────

@app.post("/rewrite")
async def api_rewrite(text: str = Form(...), source_type: str = Form(""),
                      source_file: str = Form("")):
    """手动重写一段文本（入库前清洁）"""
    try:
        clean = state.rewriter.rewrite(text, source_type, source_file)
        return {
            "status": "ok",
            "cleaned_text": clean.cleaned_text,
            "entities": clean.entities,
            "title": clean.title,
            "summary": clean.summary,
            "sections": clean.sections,
            "source_type": clean.source_type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rewrite failed: {e}")


# ──────────────────────────────────────────────
# 端点：重建连接
# ──────────────────────────────────────────────

@app.post("/rebuild-connections")
async def api_rebuild_connections():
    """全量重建球体连接网络"""
    state.ensure_connections()
    try:
        total = state.conn_detector.detect_batch()
        state.conn_detector.save()
        return {
            "status": "ok",
            "total_connections": total,
            "avg_degree": round(state.conn_detector.avg_degree, 2),
            "total_nodes": len(state.conn_detector._connections),
            "axon_edges": len(state.conn_detector._axon_types),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild connections failed: {e}")


@app.post("/rebuild-axon")
async def api_rebuild_axon():
    """仅重建轴突（因果链）连接，不重建树突"""
    state.ensure_connections()
    try:
        total = state.conn_detector.detect_axon_batch()
        state.conn_detector.save()
        return {
            "status": "ok",
            "axon_connections": total,
            "total_edges": state.conn_detector.total_edges,
            "axon_edges": len(state.conn_detector._axon_types),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild axon failed: {e}")


# ──────────────────────────────────────────────
# 端点：校准 mass/diversity
# ──────────────────────────────────────────────

@app.post("/rebuild-hierarchy")
async def api_rebuild_hierarchy():
    """重建层级结构（社区检测→等级划分）"""
    try:
        state.ensure_role_table()
        stats = state.build_hierarchy()
        return {"status": "ok", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hierarchy rebuild failed: {e}")


@app.post("/calibrate")
async def api_calibrate():
    """触发球体质量与多样性校准"""
    state.ensure_connections()
    try:
        result = state.calibrator.calibrate_all()
        state.sphere_store.save()
        return {
            "status": "ok",
            "calibrated": result["calibrated"],
            "mass_range": result["mass_range"],
            "diversity_range": result["diversity_range"],
            "avg_degree": result["avg_degree"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibrate failed: {e}")


# ──────────────────────────────────────────────
# 端点：球体导航
# ──────────────────────────────────────────────

@app.get("/navigate/{sphere_id}")
async def api_navigate(sphere_id: str, hops: int = 2):
    """从指定球体出发导航连接图"""
    state.ensure_connections()
    result = navigate_sphere(
        sphere_id=sphere_id,
        sphere_store=state.sphere_store,
        connections_provider=state.conn_detector.get_connections,
        hops=hops,
    )
    return result


# ──────────────────────────────────────────────
# 端点：聚簇展开
# ──────────────────────────────────────────────

@app.get("/explore/{cluster_id}")
async def api_explore(cluster_id: int, sort_by: str = "mass", top_k: int = 30):
    """展开一个聚簇的全部内容"""
    result = explore_cluster(
        cluster_id=cluster_id,
        sphere_store=state.sphere_store,
        field_detector=state.field_detector,
        sort_by=sort_by,
        top_k=top_k,
    )
    return result


# ──────────────────────────────────────────────
# 端点：会话时间线
# ──────────────────────────────────────────────

@app.get("/trace")
async def api_trace(source_file: str):
    """还原一个会话的时间线"""
    state.ensure_connections()
    result = trace_conversation(
        source_file=source_file,
        sphere_store=state.sphere_store,
        connections_provider=state.conn_detector.get_connections if state.conn_detector else None,
    )
    return result


# ──────────────────────────────────────────────
# 端点：球体路径发现
# ──────────────────────────────────────────────

@app.get("/bridge/{sphere_a}/{sphere_b}")
async def api_bridge(sphere_a: str, sphere_b: str):
    """找两个球体之间的最短路径"""
    state.ensure_connections()
    result = find_bridge(
        sphere_a=sphere_a,
        sphere_b=sphere_b,
        sphere_store=state.sphere_store,
        connections_provider=state.conn_detector.get_connections,
        max_hops=4,
    )
    return result


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
