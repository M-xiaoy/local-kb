"""
keywords.py — 关键词权重提取器
================================
为混合检索提供「术语引力」得分。

核心思想：
  每个球体入库时，从文本中提取关键词及其 TF 权重。
  检索时，查询也提取关键词，计算查询词与候选球体关键词的匹配度。
  → 语义余弦 × 0.7 + 术语引力 × 0.3 → 融合分

为什么不用全局 IDF：
  IDF 需要维护全局词表 + 词频统计，引入额外状态。
  这里只用文档内 TF，配合余弦语义分，在个人知识库场景下够用。
  如需提升，后续可加全局 IDF 作为 term_weights 的乘数因子。

依赖：
  零额外依赖，只用 re + collections.Counter
"""

import logging
import re
from collections import Counter
from typing import Dict, List

logger = logging.getLogger(__name__)

# 中文停用词（高频无意义词）
_STOP_WORDS_CN: set = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "们", "那", "什么", "怎么", "为什么", "因为", "所以",
    "但是", "如果", "虽然", "然而", "而且", "或者", "只是",
    "已经", "可以", "这个", "那个", "这些", "那些", "吗", "啊",
    "呢", "吧", "呀", "哦", "嗯", "哈", "嘛", "的", "地", "得",
    "与", "及", "或", "并", "而且", "不仅", "还是", "以及", "以及",
    "通过", "进行", "使用", "利用", "基于", "采用", "具有",
    "我们", "你们", "他们", "它们", "大家", "本", "该", "每",
    "各", "某", "另", "其他", "其中", "之一", "等", "等等",
}

# 英文停用词（常用）
_STOP_WORDS_EN: set = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because",
    "and", "but", "or", "if", "while", "that", "this", "which",
    "who", "whom", "what", "these", "those", "about", "up",
}


def extract_keywords(
    text: str,
    max_features: int = 50,
    min_word_len: int = 2,
) -> Dict[str, float]:
    """从文本中提取关键词及其 TF 权重

    分词策略（不用 jieba，靠模式匹配）：
      - 中文：连续 2+ 个汉字作为一个 token
      - 英文：连续 2+ 个字母作为一个 token（小写）
      - 数字：连续 2+ 位数字作为一个 token
      - 长度限制：至少 2 个字符
      - 停用词过滤

    Args:
        text: 原文
        max_features: 最多保留多少个关键词
        min_word_len: token 最小字符数

    Returns:
        {"重力": 0.12, "空间": 0.08, "clustering": 0.05, ...}
        TF 归一化后 [0, 1]，总和 ≈ 1.0
    """
    # 统一小写
    text_lower = text.lower()

    # 提取 token
    tokens: list = []

    # 中文词：连续 2+ 汉字
    for match in re.finditer(r"[\u4e00-\u9fff]{2,}", text_lower):
        word = match.group()
        if len(word) >= min_word_len and word not in _STOP_WORDS_CN:
            tokens.append(word)

    # 英文词：连续 2+ 字母
    for match in re.finditer(r"[a-z]{2,}", text_lower):
        word = match.group()
        if word not in _STOP_WORDS_EN:
            tokens.append(word)

    # 数字串：连续 2+ 位
    for match in re.finditer(r"\d{2,}", text_lower):
        tokens.append(match.group())

    if not tokens:
        return {}

    # TF 计数 + 归一化
    counter = Counter(tokens)
    total = sum(counter.values())
    total = max(total, 1)

    # 取 Top-K，归一化
    result = {}
    for word, count in counter.most_common(max_features):
        result[word] = round(count / total, 6)

    return result


def match_term_gravity(
    query_keywords: Dict[str, float],
    sphere_weights: Dict[str, float],
) -> float:
    """计算查询与球体的术语引力匹配度

    Args:
        query_keywords:  查询的关键词权重 {"重力": 0.15, "空间": 0.10}
        sphere_weights:  球体的关键词权重 {"重力": 0.12, "聚类": 0.08}

    Returns:
        [0, 1] 的匹配分数
        如果查询没有关键词，返回 0.0
    """
    if not query_keywords or not sphere_weights:
        return 0.0

    matched_sum = 0.0
    for word, q_weight in query_keywords.items():
        if word in sphere_weights:
            # 匹配度 = 查询权重 × 球体权重（调和：两个都高得分才高）
            matched_sum += q_weight * sphere_weights[word]

    # 归一化到 [0, 1]
    # 理论最大值 = sum(q_weight^2) 当所有词都在球体中且权重完全匹配时
    max_possible = sum(w * w for w in query_keywords.values())
    if max_possible == 0:
        return 0.0

    return min(1.0, matched_sum / max_possible)


def extract_from_query(text: str) -> Dict[str, float]:
    """给查询提取关键词（去掉停用词后，权重均分）

    查询通常比文档短，用 TF 归一化会放大短查询中每个词的权重。
    这里用均分权重——假设查询中每个非停用词同等重要。
    """
    keywords = extract_keywords(text, max_features=20)
    if not keywords:
        return {}

    # 均分权重
    n = len(keywords)
    equal_weight = round(1.0 / n, 6)
    return {word: equal_weight for word in keywords}
