"""
chunker.py — 递归切片器
=======================
基于 Recursive Character Chunking（LangChain 标准方案）。

核心算法流程：
  1. 按最高优先级分隔符切分（段落 → 行 → 句子 → 词 → 字符）
  2. 块 <= max_size → 保留
  3. 块 > max_size → 用下一级分隔符递归处理
  4. 全部切完后，合并小片 + 加 overlap
  5. 不做 min_size 过滤——短块也是有效信息单元
"""

import re
from typing import List

from config import chunker as cfg


# ──────────────────────────────────────────────
# 公开接口
# ──────────────────────────────────────────────

def chunk_text(text: str) -> List[str]:
    """切片入口：文本 → 切片列表

    这是外部调用的唯一入口。
    内部策略由 config.ChunkerConfig 控制。
    """
    chunker = RecursiveChunker(
        max_chunk_chars=cfg.max_chunk_chars,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
    )
    return chunker.chunk(text)


# ──────────────────────────────────────────────
# RecursiveChunker
# ──────────────────────────────────────────────

class RecursiveChunker:
    """递归降级切片器"""

    def __init__(
        self,
        max_chunk_chars: int,
        chunk_overlap: int,
        separators: List[str],
    ):
        self.max_size = max_chunk_chars
        self.overlap = chunk_overlap
        self._seps = separators  # 完整分隔符优先级列表，内部不修改

    # ── 主入口 ────────────────────────────────

    def chunk(self, text: str) -> List[str]:
        """文本 → 切片列表"""
        if not text or not text.strip():
            return []

        # Step 1: 递归切分到全部 <= max_size
        splits = self._recursive_split(text, self._seps)

        # Step 2: 合并过小的片 + 加 overlap
        return self._merge_with_overlap(splits)

    # ── 递归切分核心 ──────────────────────────

    def _recursive_split(self, text: str, seps: List[str]) -> List[str]:
        """
        沿分隔符优先级递归降级切分。
        返回全部 <= max_size 的碎片列表（未合并，已去空）。
        """
        if not text:
            return []

        # 找到第一个匹配的分隔符
        sep = self._pick_separator(text, seps)
        sep_idx = seps.index(sep)
        remaining = seps[sep_idx + 1:]

        parts = self._split_by(text, sep)
        result: List[str] = []

        for part in parts:
            # 只过滤完全空的内容，不做 .strip()
            # 否则 \n\n 分隔符会被吃掉，导致标题和正文连在一起
            if not part or not part.strip():
                continue

            if len(part) <= self.max_size:
                result.append(part)
            elif remaining:
                result.extend(self._recursive_split(part, remaining))
            else:
                result.extend(self._split_by_size(part))

        return result

    def _pick_separator(self, text: str, seps: List[str]) -> str:
        """
        从 seps 中选出第一个在 text 中匹配的分隔符。
        空字符串 '' 永远作为最终退路（字符级硬切）。
        """
        for s in seps:
            if s == "":
                return s
            if s in text:
                return s
        return ""  # 兜底

    def _split_by(self, text: str, separator: str) -> List[str]:
        """按分隔符分裂，保留分隔符在碎片末尾

        不依赖 re.escape——因为 Python 对 \n 等控制字符的
        re.escape 行为与预期不符（将 \n 转义为 \\n）。
        直接用 str.split 更可靠。
        """
        if separator == "":
            # 字符级：每个字符独立（为硬切准备）
            return list(text)

        parts = text.split(separator)
        merged: List[str] = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                # 分隔符粘回当前碎片末尾
                merged.append(part + separator)
            else:
                merged.append(part)

        return [m for m in merged if m]

    # ── 字符级硬切 ────────────────────────────

    def _split_by_size(self, text: str) -> List[str]:
        """最后退路：按固定长度硬切"""
        return [
            text[i:i + self.max_size]
            for i in range(0, len(text), self.max_size)
        ]

    # ── 合并 + Overlap ────────────────────────

    def _merge_with_overlap(self, splits: List[str]) -> List[str]:
        """
        将过小的碎片合并到前一个 chunk 中。
        相邻 chunk 之间保留 self.overlap 字符的 overlap。
        """
        if not splits:
            return []

        chunks: List[str] = []
        buffer = ""

        for split in splits:
            if not buffer:
                buffer = split
                continue

            # 如果 buffer + split 还没超上限 → 合并
            if len(buffer) + len(split) <= self.max_size:
                buffer += split
                continue

            # 超上限了 → buffer 作为一个 chunk 输出
            chunks.append(buffer)

            # 新 buffer = 前一个 buffer 尾部 overlap 内容 + 当前 split
            overlap_text = buffer[-self.overlap:] if len(buffer) >= self.overlap else buffer
            buffer = overlap_text + split

        # 收尾
        if buffer:
            chunks.append(buffer)

        return chunks

    # ── Markdown 模式（备选） ────────────────

    def chunk_markdown(self, text: str) -> List[str]:
        """按 Markdown 标题结构切分，每个标题级别作为一个独立块"""
        # 匹配 # ~ ###### 标题
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

        # 对过大的块回退到 recursive 策略
        final: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self.max_size:
                final.append(chunk)
            else:
                final.extend(RecursiveChunker(
                    max_chunk_chars=self.max_size,
                    chunk_overlap=self.overlap,
                    separators=self._seps,
                ).chunk(chunk))

        return final


# ──────────────────────────────────────────────
# 快捷调用
# ──────────────────────────────────────────────

def chunk_markdown(text: str) -> List[str]:
    """Markdown 模式切片"""
    chunker = RecursiveChunker(
        max_chunk_chars=cfg.max_chunk_chars,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
    )
    return chunker.chunk_markdown(text)
