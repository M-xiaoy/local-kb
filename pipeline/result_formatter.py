"""
result_formatter.py — 检索结果结构化格式化器
=============================================

核心思路：入库时角色表做了 AH 实体提取，输出时二次利用。

对每个检索到的球体，生成结构化摘要：
  [来源类型] 时间 — 关键实体(3-5个) — 内容摘要(一句)
  ↳ 溯源: 源文件

LLM 侧：上下文文本用结构化模板代替纯文本
前端侧：context_spheres 增加 summary/key_entities 字段
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from storage.sphere_store import Sphere

logger = logging.getLogger(__name__)


# ── 时间提取 ───────────────────────────────

_TIME_PATTERNS = [
    (r"(\d{4}-\d{2}-\d{2})", lambda m: m.group(1)),        # 2026-07-22
    (r"(\d{4})年(\d{1,2})月(\d{1,2})日", lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
    (r"(\d{1,2}:\d{2})", lambda m: m.group(1)),              # 14:35
]


def extract_time(sphere: Sphere) -> str:
    """从球体中提取时间信息"""
    # 1. 从 source_file 提取（"会话_2026-07-22_xxx"）
    sf = sphere.source_file or ""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", sf)
    if m:
        date = m.group(1)
        # 看球体文本里有没有具体时间
        time_m = re.search(r"\((\d{1,2}:\d{2})\)", sphere.text[:80])
        if time_m:
            return f"{date} {time_m.group(1)}"
        return date

    # 2. 从文本开头找
    header = sphere.text[:100]
    for pattern, fmt in _TIME_PATTERNS:
        m = re.search(pattern, header)
        if m:
            return fmt(m)

    # 3. 从 created_at 字段
    if sphere.created_at:
        return sphere.created_at[:10]

    return ""


# ── 实体提取（从角色表）────────────────────

def get_key_entities(
    sphere_id: str,
    sphere_text: str,
    role_table=None,
    max_entities: int = 5,
) -> List[str]:
    """从角色表获取球体的关键实体列表

    Args:
        sphere_id: 球体 ID
        sphere_text: 球体文本（兜底用）
        role_table: RoleTable 实例（可选）
        max_entities: 返回最多实体数

    Returns:
        实体文本列表，按共现频次排序
    """
    if not role_table:
        return _fallback_entities(sphere_text, max_entities)

    # 从 role_table._sphere_entities 找
    s2e = getattr(role_table, "_sphere_entities", {})
    eids = s2e.get(sphere_id, set())
    entities_dict = getattr(role_table, "_entities", {})

    if not eids:
        return _fallback_entities(sphere_text, max_entities)

    # 排序：先按 occurrence（实体全局频次），再按 co_occurrence 丰富度
    scored = []
    for eid in eids:
        ent = entities_dict.get(eid)
        if not ent:
            continue
        text = getattr(ent, "text", "") or ""
        if len(text) < 2:
            continue
        occ = getattr(ent, "occurrences", 0) or 0
        co = getattr(ent, "co_occurrences", {}) or {}
        co_score = len(co)  # 共现实体数反映实体重要性
        scored.append((text, occ * co_score))

    scored.sort(key=lambda x: -x[1])
    # 去重：跳过是其他实体子串的项
    result = []
    for text, _ in scored:
        if any(text in existing or existing in text for existing in result):
            continue
        result.append(text)
        if len(result) >= max_entities:
            break
    return result


def _fallback_entities(text: str, max_entities: int) -> List[str]:
    """无角色表时的兜底实体提取"""
    # 从文本中提取长词/术语作为简单实体
    words = re.findall(r"[\u4e00-\u9fffA-Za-z]{2,}", text[:500])
    # 去掉常见的停用词
    stopwords = {"可以", "这个", "那个", "什么", "一个", "没有", "不是", "我们",
                 "就是", "如果", "因为", "所以", "然后", "之后", "之前", "现在",
                 "已经", "可能", "需要", "使用", "通过", "进行", "实现", "利用"}
    filtered = [w for w in words if w not in stopwords and len(w) >= 2]
    # 按频次排序
    from collections import Counter
    return [w for w, _ in Counter(filtered).most_common(max_entities)]


# ── 摘要生成 ───────────────────────────────

def generate_summary(
    text: str,
    key_entities: Optional[List[str]] = None,
    max_chars: int = 150,
) -> str:
    """生成球体的内容摘要

    策略：
      1. 跳过工具调用行（exec/process/read 等）
      2. 跳过纯代码块
      3. 取第一个有实际内容的句子或片段
      4. 截断到 max_chars

    Args:
        text: 球体原文
        key_entities: 关键实体列表（用于优化摘要）
        max_chars: 最大字符数

    Returns:
        摘要文本
    """
    if not text:
        return ""

    # 按行过滤
    lines = text.split("\n")
    content_lines = []
    for line in lines:
        s = line.strip()
        # 跳过工具调用行
        if re.match(r"^>\s*(工具|Tool|exec|process|read|write|edit):", s, re.IGNORECASE):
            continue
        # 跳过纯代码块标记
        if s in ("```", "```python", "```json", "```bash", "```yaml"):
            continue
        # 跳过空的 assistant/思考前缀
        if re.match(r"^#+\s*(ASSISTANT|USER|TOOL|💭|思考|System)\b", s):
            continue
        # 跳过连字符/分隔线
        if re.match(r"^[-=]{5,}$", s):
            continue
        content_lines.append(line)

    # 取第一个有内容的段落
    full_text = " ".join(content_lines)
    # 清理多余空格
    full_text = re.sub(r"\s+", " ", full_text).strip()

    if not full_text:
        return text[:max_chars].strip()

    if len(full_text) <= max_chars:
        return full_text

    # 在句号处截断
    truncated = full_text[:max_chars]
    last_period = max(
        truncated.rfind("。"),
        truncated.rfind("."),
        truncated.rfind("！"),
        truncated.rfind("？"),
    )
    if last_period > max_chars // 2:
        return truncated[: last_period + 1]
    return truncated + "..."


# ── 结构化格式化 ───────────────────────────

def format_sphere(
    sphere: Sphere,
    role_table=None,
    max_entities: int = 5,
    max_summary_chars: int = 150,
) -> Dict:
    """将单个球体格式化为结构化输出

    Returns:
        {
            "id": str,
            "source_type": str,
            "source_file": str,
            "time": str,
            "key_entities": [str, ...],
            "summary": str,
            "text": str,           # 原始全文（溯源用）
        }
    """
    entities = get_key_entities(sphere.id, sphere.text, role_table, max_entities)
    summary = generate_summary(sphere.text, entities, max_summary_chars)
    time_str = extract_time(sphere)

    return {
        "id": sphere.id,
        "source_type": sphere.source_type or "",
        "source_file": sphere.source_file or "",
        "time": time_str,
        "key_entities": entities,
        "summary": summary,
        "text": sphere.text,  # 保持全文可溯源
    }


def format_context_block(
    sphere: Sphere,
    role_table=None,
    index: int = 0,
) -> str:
    """生成 LLM 上下文块的结构化文本

    示例：
        [第1段] 📚技术笔记 | 2026-07-22
        主题: RAG架构、FAISS索引、ChromaDB
        摘要:
          结合 FAISS 和 bge-m3 嵌入实现多路召回...

    Args:
        sphere: 球体
        role_table: RoleTable 实例
        index: 段落编号

    Returns:
        格式化后的文本块
    """
    entities = get_key_entities(sphere.id, sphere.text, role_table, max_entities=5)
    summary = generate_summary(sphere.text, entities, max_chars=250)
    time_str = extract_time(sphere)

    # 来源标签
    type_tag = sphere.source_type or "通用"
    time_tag = f" | {time_str}" if time_str else ""

    # 实体标签
    entity_tag = ""
    if entities:
        entity_tag = f"\n  主题: {'、'.join(entities)}"

    return (
        f"[第{index + 1}段] {type_tag}{time_tag}"
        f"{entity_tag}\n"
        f"  摘要: {summary}"
    )
