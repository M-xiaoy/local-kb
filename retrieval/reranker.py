"""
reranker.py — 轻量候选重排序器
===============================
对 FAISS / 激活传播后的候选球体进行精细化重打分。

方案选择（由 config.reranker.method 控制）：
  - "ollama": 用本地 LLM 对 (query, candidate_text) 对评分
  - "cross-encoder": 用 bge-reranker 等交叉编码器（需另行安装）

当前实现 ollama 方案，避免引入额外依赖。
重排序器是可选的——配置关掉时跳过，不影响主流程。
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np

from config import reranker as cfg, ollama as cfg_ollama

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# LLM 重排 Prompt
# ──────────────────────────────────────────────

_RERANK_SYSTEM_PROMPT = """你是一个检索质量评估器。
给定用户的问题和一段文本，请判断这段文本对回答问题的有用程度。
只输出一个数字 1-5（不要其他文字）：
5 = 直接回答问题的核心内容
4 = 包含回答所需的关键信息
3 = 部分相关，但不够完整
2 = 轻微相关但不直接有用
1 = 完全不相关或无关"""

_RERANK_USER_TEMPLATE = """问题: {query}

文本: {text}

有用程度评分(1-5):"""


# ──────────────────────────────────────────────
# LocalReranker
# ──────────────────────────────────────────────

class LocalReranker:
    """轻量级候选重排序器"""

    def __init__(self):
        self.method = cfg.method
        self.model = cfg.model
        self.host = cfg_ollama.host
        self.batch_size = cfg.batch_size
        self._ollama_url = f"{self.host}/api/chat"
        self.stats = {"calls": 0, "total_seconds": 0.0}

    # ── 主入口 ───────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """对候选进行重排序

        Args:
            query: 用户查询
            candidates: [(sphere_id, text), ...]
            top_k: 返回数量

        Returns:
            [(sphere_id, score), ...] 按得分降序
        """
        if not candidates:
            return []

        candidate_count = min(len(candidates), cfg.candidate_count)
        candidates = candidates[:candidate_count]

        # 评分
        scores = self._score_batch(query, candidates)

        # 排序
        scored = list(zip([c[0] for c in candidates], scores))
        scored.sort(key=lambda x: -x[1])

        return scored[:top_k]

    def rerank_with_objects(
        self,
        query: str,
        spheres: List,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """对 Sphere 对象列表进行重排序"""
        candidates = [(s.id, s.text[:500]) for s in spheres]
        return self.rerank(query, candidates, top_k)

    # ── 批量评分 ─────────────────────────────

    def _score_batch(self, query: str,
                      candidates: List[Tuple[str, str]]) -> List[float]:
        """批量评分候选"""
        scores = []
        t0 = time.time()

        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            for sphere_id, text in batch:
                score = self._score_single(query, text)
                scores.append(score)
            # 批次间延迟
            if i + self.batch_size < len(candidates):
                time.sleep(0.1)

        self.stats["calls"] += len(candidates)
        self.stats["total_seconds"] += time.time() - t0

        # 如果没有有效评分，fallback 到均匀分布
        if all(s == 0.0 for s in scores):
            return [3.0] * len(candidates)

        return scores

    def _score_single(self, query: str, text: str) -> float:
        """单条评分"""
        prompt = _RERANK_USER_TEMPLATE.format(
            query=query[:200], text=text[:500]
        )

        for attempt in range(2):
            try:
                resp = httpx.post(
                    self._ollama_url,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 5,
                        },
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "").strip()

                # 提取数字
                num = self._extract_score(content)
                if num is not None:
                    return num / 5.0  # 归一化到 [0, 1]

            except Exception as e:
                logger.debug(f"Rerank attempt {attempt+1} failed: {e}")
                time.sleep(0.5)
                continue

        return 0.0

    @staticmethod
    def _extract_score(content: str) -> Optional[float]:
        """从 LLM 回复中提取评分"""
        # 直接数字
        try:
            num = float(content.strip())
            if 1.0 <= num <= 5.0:
                return num
        except ValueError:
            pass

        # 从代码块提取
        match = re.search(r"(\d+)", content)
        if match:
            try:
                num = float(match.group(1))
                if 1.0 <= num <= 5.0:
                    return num
            except ValueError:
                pass

        return None

    # ── 统计 ─────────────────────────────────

    def stats_report(self) -> dict:
        avg = self.stats["total_seconds"] / max(self.stats["calls"], 1)
        return {
            "calls": self.stats["calls"],
            "total_seconds": round(self.stats["total_seconds"], 2),
            "avg_seconds_per_call": round(avg, 3),
        }


# ──────────────────────────────────────────────
# 快捷函数
# ──────────────────────────────────────────────

_global_reranker: Optional[LocalReranker] = None


def get_reranker() -> LocalReranker:
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = LocalReranker()
    return _global_reranker


def rerank(query: str, candidates: List[Tuple[str, str]],
           top_k: int = 10) -> List[Tuple[str, float]]:
    return get_reranker().rerank(query, candidates, top_k)
