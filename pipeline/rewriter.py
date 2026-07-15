"""
rewriter.py — 入库前文本重写器
================================
将原始文本（尤其是会话记录）重写为结构清晰、自包含的清洁文档。

核心原则：
  - 不改变原意，只改善表达
  - 去除噪音（时间戳、重复词、格式残留）
  - 结构强化（补充标题、分段、标注实体）
  - 指代消解（"他说" → "小刘说"）
  - 实体提取（标注 {人物}、{概念}、{工具}、{决定}）

策略分层：
  REWRITE_FULL  — 会话记录/未知类型：LLM 结构化改写
  REWRITE_LIGHT — 技术笔记/论文：只做实体提取，保留原文

使用流程：
  rewriter = TextRewriter()
  clean = rewriter.rewrite(raw_text, source_type="会话记录", source_file="2026-07-15.md")
  # clean.cleaned_text → 重写后文本
  # clean.entities → 实体列表
  # clean.title → 标题
  # clean.sections → 分段结构
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

from config import rewriter as cfg_rewriter, ollama as cfg_ollama

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────

@dataclass
class CleanDocument:
    """重写后的结构化文档"""
    cleaned_text: str                      # 重写后的纯文本（用于嵌入和存储）
    entities: List[str] = field(default_factory=list)    # 实体列表
    sections: List[Dict] = field(default_factory=list)   # [{"heading": "...", "content": "..."}]
    title: str = ""                        # 自动生成的标题
    summary: str = ""                      # 单句摘要
    original_text: str = ""                # 原始文本备份
    rewrite_model: str = ""                # 使用的模型
    source_type: str = ""                  # 源类型


# ──────────────────────────────────────────────
# 噪音过滤规则（纯代码，不调 LLM）
# ──────────────────────────────────────────────

_NOISE_PATTERNS = [
    # 纯时间戳行
    re.compile(r"^\d{1,2}:\d{2}(:\d{2})?\s*$", re.MULTILINE),
    # /thinking 标记
    re.compile(r"^/*thinking", re.MULTILINE | re.IGNORECASE),
    # 纯标点行 (连续 3+ 标点符号)
    re.compile(r"^[^\w\s]{3,}$", re.MULTILINE),
    # 纯横线装饰行
    re.compile(r"^[-–—=*]{3,}\s*$", re.MULTILINE),
    # 纯数字行（可能只是时间戳）
    re.compile(r"^\d{10,}\s*$", re.MULTILINE),
    # 系统元数据行
    re.compile(r"^\[.*?\]\s*$", re.MULTILINE),
]

_ENTITY_NORMALIZATION = {
    # 名字统一
    "小云": ["小云", "深渊凝视者", "深渊凝视者·小云", "cloud"],
    "小刘": ["小刘", "刘存帅", "用户", "主人"],
    # 术语统一
    "RAG": ["rag", "RAG", "检索增强生成", "检索增强"],
    "Ollama": ["ollama", "Ollama"],
    "ComfyUI": ["comfyui", "ComfyUI", "comfy"],
    "FAISS": ["faiss", "FAISS"],
    "重力知识库": ["重力知识库", "知识库", "local-kb", "kb"],
    "重力空间": ["重力空间", "重力空间架构", "gravity space"],
}


# ──────────────────────────────────────────────
# LLM 重写 Prompt
# ──────────────────────────────────────────────

_REWRITE_SYSTEM_PROMPT = """你是一个文档清洗助手。你的任务是将一段原始文本（可能是对话记录、笔记片段）重写为整洁、自包含的文档。

要求：
1. 保持原意，不要添加新信息
2. 去除无意义的内容（重复、语气词、格式残留）
3. 如果涉及对话，标注说话人（如 "小刘：xxx", "小云：xxx"）
4. 如果原文有指代（"他"、"那个项目"），尽量补充为明确名称
5. 优质标题请根据内容自动生成
6. 输出格式为 JSON

输出 JSON 格式：
{
  "title": "自动生成的标题",
  "summary": "一句话摘要",
  "content": "重写后的清洁文本",
  "entities": ["实体1", "实体2", ...],
  "sections": [
    {"heading": "小节标题", "content": "小节内容"}
  ]
}

注意：content 字段是最终用于检索的文本，需要段落分明、逻辑清晰。"""

_REWRITE_USER_TEMPLATE = """请重写以下{source_type}内容：

{text}"""


# ──────────────────────────────────────────────
# TextRewriter
# ──────────────────────────────────────────────

class TextRewriter:
    """入库前文本重写器"""

    def __init__(self):
        self.model = cfg_rewriter.llm_model
        self.host = cfg_ollama.host
        self.full_strategies = cfg_rewriter.full_strategies
        self.light_strategies = cfg_rewriter.light_strategies
        self.max_input_chars = cfg_rewriter.max_input_chars
        self.batch_delay = cfg_rewriter.batch_delay
        self.timeout = cfg_rewriter.timeout
        self._ollama_url = f"{self.host}/api/chat"

        # 统计
        self.stats = {"full_rewrites": 0, "light_rewrites": 0, "skipped": 0}

    # ── 公开入口 ──────────────────────────────

    def rewrite(self, text: str, source_type: str = "",
                source_file: str = "") -> CleanDocument:
        """重写一条文本

        Args:
            text: 原始文本
            source_type: 源类型（决定重写策略）
            source_file: 源文件名（辅助理解上下文）

        Returns:
            CleanDocument
        """
        strategy = self._resolve_strategy(source_type)

        # Step 1: 噪音过滤（所有策略都跑）
        cleaned = self._filter_noise(text)
        if not cleaned.strip():
            self.stats["skipped"] += 1
            return CleanDocument(
                cleaned_text=text,
                source_type=source_type,
                original_text=text,
            )

        if strategy == "light":
            return self._light_rewrite(cleaned, source_type, source_file)
        else:
            return self._full_rewrite(cleaned, source_type, source_file)

    def rewrite_batch(self, items: List[Tuple[str, str, str]],
                      batch_size: int = 5) -> List[CleanDocument]:
        """批量重写

        Args:
            items: [(text, source_type, source_file), ...]
            batch_size: 每批数量

        Returns:
            [CleanDocument, ...]
        """
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            for text, stype, sfile in batch:
                results.append(self.rewrite(text, stype, sfile))
            if i + batch_size < len(items):
                time.sleep(self.batch_delay)
        return results

    # ── 策略路由 ─────────────────────────────

    def _resolve_strategy(self, source_type: str) -> str:
        if source_type in self.full_strategies:
            return "full"
        if source_type in self.light_strategies:
            return "light"
        return "full"  # 默认全量

    # ── 噪音过滤 ─────────────────────────────

    def _filter_noise(self, text: str) -> str:
        """纯代码过滤噪音，不调 LLM"""
        for pattern in _NOISE_PATTERNS:
            text = pattern.sub("", text)
        # 去除连续空行（3行以上压缩为2行）
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 去除首尾空白
        text = text.strip()
        return text

    # ── 轻量重写 ─────────────────────────────

    def _light_rewrite(self, text: str, source_type: str,
                       source_file: str) -> CleanDocument:
        """轻量重写：只做实体提取，不做 LLM 改写"""
        self.stats["light_rewrites"] += 1

        entities = self._extract_entities(text)
        title = self._guess_title(text, source_file)

        return CleanDocument(
            cleaned_text=text,
            entities=entities,
            title=title,
            source_type=source_type,
            original_text=text,
            rewrite_model="rule",
        )

    # ── 全量重写 ─────────────────────────────

    def _full_rewrite(self, text: str, source_type: str,
                      source_file: str) -> CleanDocument:
        """全量重写：LLM 结构化改写 + 实体提取"""
        self.stats["full_rewrites"] += 1

        # 截断到最大输入长度
        input_text = text[:self.max_input_chars]

        # 调用 LLM
        result = self._call_llm(input_text, source_type)

        if result is None:
            # LLM 失败时降级到轻量重写
            logger.warning("LLM rewrite failed, falling back to light rewrite")
            return self._light_rewrite(text, source_type, source_file)

        # LLM 返回了结构化结果
        cleaned_text = result.get("content", text)
        entities = result.get("entities", [])
        title = result.get("title", self._guess_title(text, source_file))
        summary = result.get("summary", "")
        sections = result.get("sections", [])

        # 实体规范化（去重、统一名称）
        entities = self._normalize_entities(entities)
        # 补充规则提取的实体
        rule_entities = self._extract_entities(text)
        combined = list(set(entities + rule_entities))
        combined.sort()

        return CleanDocument(
            cleaned_text=cleaned_text,
            entities=combined,
            sections=sections,
            title=title,
            summary=summary,
            original_text=text,
            rewrite_model=self.model,
            source_type=source_type,
        )

    # ── LLM 调用 ─────────────────────────────

    def _call_llm(self, text: str, source_type: str) -> Optional[Dict]:
        """调用 Ollama LLM 进行结构化改写"""
        user_prompt = _REWRITE_USER_TEMPLATE.format(
            source_type=source_type or "内容",
            text=text,
        )

        for attempt in range(3):
            try:
                resp = httpx.post(
                    self._ollama_url,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "format": "json",
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 2048,
                        },
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")

                # 解析 JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # 有时候 LLM 返回的 JSON 在代码块里
                    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```",
                                            content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(1))
                    # 尝试从纯文本提取
                    logger.warning(
                        f"LLM response not valid JSON (attempt {attempt+1}): "
                        f"{content[:100]}..."
                    )
                    continue

            except (httpx.ConnectError, httpx.TimeoutException,
                    httpx.HTTPStatusError, json.JSONDecodeError) as e:
                logger.warning(
                    f"LLM rewrite attempt {attempt+1} failed: {e}"
                )
                if attempt < 2:
                    time.sleep(1)
                    continue
                return None

        return None

    # ── 实体提取（规则版） ───────────────────

    def _extract_entities(self, text: str) -> List[str]:
        """基于模式匹配和词汇表的实体提取"""
        found = set()

        # 从规范化表反向匹配
        for canonical, aliases in _ENTITY_NORMALIZATION.items():
            for alias in aliases:
                if alias.lower() in text.lower():
                    found.add(canonical)
                    break

        # 正则匹配特殊格式
        # 文件名: xxx.md
        for match in re.finditer(r"`([^`]+)`", text):
            found.add(match.group(1))

        return sorted(found)

    def _normalize_entities(self, entities: List[str]) -> List[str]:
        """实体名称规范化"""
        normalized = set()
        for entity in entities:
            matched = False
            for canonical, aliases in _ENTITY_NORMALIZATION.items():
                if entity.lower() in [a.lower() for a in aliases]:
                    normalized.add(canonical)
                    matched = True
                    break
            if not matched:
                normalized.add(entity)
        return sorted(normalized)

    # ── 标题猜测 ─────────────────────────────

    def _guess_title(self, text: str, source_file: str = "") -> str:
        """从文本中猜测标题"""
        # 优先使用源文件名
        if source_file:
            return source_file.replace(".md", "").replace("_", " ")

        # 尝试匹配 markdown 标题
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()

        # 取第一行非空行
        for line in text.split("\n"):
            line = line.strip()
            if line and len(line) < 100:
                return line

        return "未命名文档"

    # ── 状态 ─────────────────────────────────

    def stats_report(self) -> dict:
        return dict(self.stats)


# ──────────────────────────────────────────────
# 快捷函数
# ──────────────────────────────────────────────

_global_rewriter: Optional[TextRewriter] = None


def get_rewriter() -> TextRewriter:
    """获取全局 TextRewriter 单例"""
    global _global_rewriter
    if _global_rewriter is None:
        _global_rewriter = TextRewriter()
    return _global_rewriter


def rewrite(text: str, source_type: str = "",
            source_file: str = "") -> CleanDocument:
    """快捷：重写一条文本"""
    return get_rewriter().rewrite(text, source_type, source_file)


def rewrite_batch(items: List[Tuple[str, str, str]],
                  batch_size: int = 5) -> List[CleanDocument]:
    """快捷：批量重写"""
    return get_rewriter().rewrite_batch(items, batch_size)
