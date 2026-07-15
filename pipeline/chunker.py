"""
chunker.py — 切片器（递归 + 结构感知）
=======================================

支持三种切片模式（按 source_type 路由）：
  recursive — 标准递归字符切片（默认）
  markdown  — 按 Markdown 标题结构切分
  section   — 按文档的分段结构切分（用于重写后的会话记录）

三种模式的降级链：
  section → markdown → recursive → fixed

配置由 config.ChunkerConfig.strategy_overrides 控制。
"""

import re
from typing import Dict, List, Optional

from config import chunker as cfg


# ──────────────────────────────────────────────
# 公开接口
# ──────────────────────────────────────────────

def chunk_text(text: str, source_type: str = "",
               sections: Optional[List[Dict]] = None,
               strategy: Optional[str] = None) -> List[str]:
    """切片入口：文本 → 切片列表

    Args:
        text: 要切片的文本
        source_type: 源类型，用于路由切片策略
        sections: 分段结构（section 模式使用）
        strategy: 强制指定策略，覆盖 source_type 路由

    Returns:
        切片后的文本列表
    """
    # 解析策略
    mode = strategy
    if mode is None:
        mode = _resolve_strategy(source_type)

    # 获取该策略的参数
    overrides = cfg.strategy_overrides.get(source_type, {})
    max_chars = overrides.get("max_chars", cfg.max_chunk_chars)
    overlap = overrides.get("overlap", cfg.chunk_overlap)

    # 因果密度模式（v2 新增：基于实验结论的智能切分）
    if mode == "causal_density":
        return _chunk_by_causal_density(text, max_chars, overlap)
    # Section 模式（优先使用，有 sections 才走）
    elif mode == "section" and sections:
        return _chunk_by_sections(text, sections, max_chars, overlap)
    # Markdown 模式
    elif mode == "markdown":
        return _chunk_markdown(text, max_chars, overlap)
    # 默认递归模式
    else:
        return _chunk_recursive(text, max_chars, overlap)


def _resolve_strategy(source_type: str) -> str:
    """根据 source_type 解析切片策略"""
    overrides = cfg.strategy_overrides.get(source_type, {})
    return overrides.get("mode", cfg.mode)


# ──────────────────────────────────────────────
# Section 模式
# ──────────────────────────────────────────────

def _chunk_by_sections(text: str, sections: List[Dict],
                       max_chars: int, overlap: int) -> List[str]:
    """按文档的分段结构切分

    每个 section 作为一个独立 chunk。
    过长的 section 降级到 recursive 策略。
    过短的 section 合并到前一个 chunk。
    """
    if not sections:
        return _chunk_recursive(text, max_chars, overlap)

    chunks = []
    buffer = ""
    current_heading = ""

    for sec in sections:
        heading = sec.get("heading", "")
        content = sec.get("content", "")

        # 构建带标题的文本块
        block = f"## {heading}\n\n{content}" if heading else content

        if not buffer:
            buffer = block
            current_heading = heading
            continue

        # 如果 buffer + block 没超上限 → 合并
        if len(buffer) + len(block) <= max_chars:
            # 不同标题之间加空行分隔
            if heading and heading != current_heading:
                buffer += "\n\n" + block
            else:
                buffer += "\n" + content
            current_heading = heading
            continue

        # 超上限 → buffer 作为一个 chunk 输出
        chunks.append(buffer.strip())

        # 新 buffer = overlap + block
        overlap_text = buffer[-overlap:] if len(buffer) >= overlap else buffer
        buffer = overlap_text + "\n" + block
        current_heading = heading

    if buffer:
        chunks.append(buffer.strip())

    # 对过长的 chunk 降级
    final = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(_chunk_recursive(chunk, max_chars, overlap))

    return final


# ──────────────────────────────────────────────
# Markdown 模式
# ──────────────────────────────────────────────

def _chunk_markdown(text: str, max_chars: int, overlap: int) -> List[str]:
    """按 Markdown 标题结构切分"""
    pattern = r"^(#{1,6}\s+.*)$"
    lines = text.split("\n")

    chunks: List[str] = []
    current: List[str] = []

    for line in lines:
        if re.match(pattern, line.strip()):
            if current:
                chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current))

    # 过长的 chunk 降级到 recursive
    final = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(_chunk_recursive(chunk, max_chars, overlap))

    return final


# ──────────────────────────────────────────────
# Recursive 模式（原始实现）
# ──────────────────────────────────────────────

def _chunk_recursive(text: str, max_chars: int, overlap: int) -> List[str]:
    """递归字符切片（LangChain 标准方案）"""
    if not text or not text.strip():
        return []

    seps = cfg.separators
    splits = _recursive_split(text, seps, max_chars)
    return _merge_with_overlap(splits, max_chars, overlap)


def _recursive_split(text: str, seps: List[str],
                      max_size: int) -> List[str]:
    """沿分隔符优先级递归降级切分"""
    if not text:
        return []

    sep = _pick_separator(text, seps)
    sep_idx = seps.index(sep)
    remaining = seps[sep_idx + 1:]

    parts = _split_by(text, sep)
    result: List[str] = []

    for part in parts:
        if not part or not part.strip():
            continue
        if len(part) <= max_size:
            result.append(part)
        elif remaining:
            result.extend(_recursive_split(part, remaining, max_size))
        else:
            result.extend(_split_by_size(part, max_size))

    return result


def _pick_separator(text: str, seps: List[str]) -> str:
    for s in seps:
        if s == "":
            return s
        if s in text:
            return s
    return ""


def _split_by(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)

    parts = text.split(separator)
    merged: List[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            merged.append(part + separator)
        else:
            merged.append(part)
    return [m for m in merged if m]


def _split_by_size(text: str, max_size: int) -> List[str]:
    return [text[i:i + max_size] for i in range(0, len(text), max_size)]


def _merge_with_overlap(splits: List[str], max_size: int,
                         overlap: int) -> List[str]:
    """合并小片 + 加 overlap"""
    if not splits:
        return []

    chunks: List[str] = []
    buffer = ""

    for split in splits:
        if not buffer:
            buffer = split
            continue

        if len(buffer) + len(split) <= max_size:
            buffer += split
            continue

        chunks.append(buffer)
        overlap_text = buffer[-overlap:] if len(buffer) >= overlap else buffer
        buffer = overlap_text + split

    if buffer:
        chunks.append(buffer)

    return chunks


# ──────────────────────────────────────────────
# 向后兼容
# ──────────────────────────────────────────────

class RecursiveChunker:
    """向后兼容的递归切片器包装"""

    def __init__(self, max_chunk_chars: int, chunk_overlap: int,
                 separators: List[str]):
        self.max_size = max_chunk_chars
        self.overlap = chunk_overlap
        self._seps = separators

    def chunk(self, text: str) -> List[str]:
        return _chunk_recursive(text, self.max_size, self.overlap)

    def chunk_markdown(self, text: str) -> List[str]:
        return _chunk_markdown(text, self.max_size, self.overlap)


# ──────────────────────────────────────────────
# 因果密度模式（v2 新增）
# ──────────────────────────────────────────────

# 因果标记词（与 connections.py 同步）
CAUSE_MARKERS = [
    "propose", "demonstrate", "therefore", "enable",
    "we show that", "our results indicate",
    "this demonstrates", "this confirms that",
    "we conclude that", "results in", "leads to",
    "suggesting that", "indicating that", "this implies",
    "these findings suggest", "we report the",
]

# 果句标记词
EFFECT_MARKERS = [
    "we demonstrate", "we show", "we find", "our results",
    "this leads", "demonstrates that", "shows that",
    "confirms", "achieves", "improves", "outperforms",
    "enables", "provides", "yields", "results in",
]


def _has_causal_chain(paragraph: str) -> bool:
    """检测段落内是否存在因果链（因句→果句，距离 ≤ 2 句）

    基于实验结论：因果链均为段落内，平均跨 2 句。
    """
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    for si, sent in enumerate(sentences):
        lower = sent.lower()[:100]
        # 检查是否含因果标记
        has_cause = any(m in lower for m in CAUSE_MARKERS)
        if not has_cause:
            continue

        # 向后 max_gap=2 句范围内找果句
        for sj in range(si + 1, min(si + 3, len(sentences))):
            target = sentences[sj].lower()[:100]
            has_effect = any(m in target for m in EFFECT_MARKERS)
            if has_effect:
                return True

            # 果句也可能含因果标记（因果对被称为完整）
            has_cause2 = any(m in target for m in CAUSE_MARKERS)
            if has_cause2 and sj < si + 2:
                return True

    return False


def _chunk_by_causal_density(text: str, max_chars: int,
                              overlap: int) -> List[str]:
    """按因果密度动态切分

    算法：
      1. 按段落为单位扫描
      2. 检测每段是否含因果链
      3. 含因果链的段落标记为 causal_block，独立成球体
      4. 无因果链的段落合并到相邻 causal_block（前向优先）
      5. 过长的 causal_block 降级到 recursive

    基于实验：
      - 因果链均为段落内（barrier=0）
      - 平均跨 2 句
      - 含因果链的段落应保留为独立知识单元
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]

    if len(paragraphs) <= 1:
        # 单段全文→无因果密度信息，降级到 recursive
        return _chunk_recursive(text, max_chars, overlap)

    # 第 1 轮：扫描因果链
    para_flags = []  # (text, is_causal)
    for para in paragraphs:
        is_causal = _has_causal_chain(para)
        para_flags.append((para, is_causal))

    # 第 2 轮：合并非因果段到相邻因果段
    merged = []
    buffer = ""
    buffer_has_causal = False

    for para, is_causal in para_flags:
        if not buffer:
            buffer = para
            buffer_has_causal = is_causal
            continue

        # 非因果段 → 合并到 buffer（前向合并）
        if not is_causal and len(buffer) + len(para) < max_chars * 1.5:
            buffer += "\n\n" + para
            continue

        # 因果段 + buffer 已有因果 → 两个因果块
        if buffer_has_causal and is_causal:
            merged.append(buffer.strip())
            buffer = para
            buffer_has_causal = True
            continue

        # 缓冲满 → 输出
        if len(buffer) >= max_chars:
            merged.append(buffer.strip())
            buffer = para
            buffer_has_causal = is_causal
            continue

        # 默认：合并
        buffer += "\n\n" + para
        buffer_has_causal = buffer_has_causal or is_causal

    if buffer:
        merged.append(buffer.strip())

    # 第 3 轮：过长的 chunk 降级
    final = []
    for chunk in merged:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(_chunk_recursive(chunk, max_chars, overlap))

    # 如果因果密度切分出来和原始没区别（全粘一起），降级
    if len(final) <= 1 and len(paragraphs) > 3:
        return _chunk_recursive(text, max_chars, overlap)

    return final


# 快捷函数（向后兼容）
def chunk_markdown(text: str) -> List[str]:
    return _chunk_markdown(text, cfg.max_chunk_chars, cfg.chunk_overlap)
