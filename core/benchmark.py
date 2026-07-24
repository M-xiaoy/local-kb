"""
core/benchmark.py — Cross-Encoder 增强版 Benchmark

替代方案：因 SSL 证书问题无法下载 cross-encoder/ms-marco-MiniLM-L-6-v2，
改用 nomic-embed-text 的余弦相似度作为动态相关性裁判。

原理：
  1. 从知识库中随机采样 N 篇文档作为 query
  2. 用 nomic-embed-text 对 (query_text, doc_text) 分别嵌入
  3. 若余弦相似度 > threshold → 判定为相关
  4. 计算基于动态判定的 MRR / Recall@5

与原固定 Ground Truth 基准测试的区别：
  · 不预设"只有文档 X 才是正确答案"
  · 任何语义接近的文档都被视为相关
  · 更符合真实检索场景
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.kb import KnowledgeBase
from pipeline.embedder import Embedder

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.55  # 余弦相似度阈值，高于此值视为相关


class DynamicBenchmark:
    """基于动态相关性判定的检索评估器

    用法：
        bench = DynamicBenchmark(kb)
        queries = bench.sample_queries(n=50)
        results = bench.evaluate(queries, mode="poincare")
    """

    def __init__(self, kb: KnowledgeBase,
                 threshold: float = _SIMILARITY_THRESHOLD):
        self._kb = kb
        self._threshold = threshold
        self._embedder = Embedder()
        # 预加载所有活跃球体的文本
        self._sphere_texts: Dict[str, str] = {}
        for s in kb.get_active_spheres():
            self._sphere_texts[s.id] = s.text or ""

    def sample_queries(self, n: int = 50) -> List[Tuple[str, str]]:
        """从知识库中采样查询

        对每个采样文档：
          query_text = 文档前 80 字
          query 本身也是 ground truth 的一部分

        Returns:
            [(query_text, expected_sphere_id)]
        """
        all_ids = list(self._sphere_texts.keys())
        if n > len(all_ids):
            n = len(all_ids)

        sampled = random.sample(all_ids, min(n, len(all_ids)))
        queries = []
        for sid in sampled:
            text = self._sphere_texts[sid]
            if len(text) < 10:
                continue
            # 取前 80 字作为查询
            q_text = text[:80].strip()
            if len(q_text) < 5:
                continue
            queries.append((q_text, sid))

        logger.info(
            f"DynamicBenchmark: sampled {len(queries)} queries "
            f"from {len(all_ids)} available spheres"
        )
        return queries

    def evaluate(self, queries: List[Tuple[str, str]],
                 top_k: int = 20, mode: str = "poincare") -> dict:
        """运行评估

        Args:
            queries: [(query_text, source_sphere_id)]
            top_k: 检索深度
            mode: "poincare" | "gravity"

        Returns:
            {
                "mrr": float,
                "recall_at_5": float,
                "recall_at_10": float,
                "total_queries": int,
                "threshold": float,
                "mode": str,
                "per_query": [{"query": str, "mrr": float, "recalled": bool}]
            }
        """
        mrr_total = 0.0
        recall_5_total = 0.0
        recall_10_total = 0.0
        per_query = []

        for q_text, source_id in queries:
            # 检索
            result = self._kb.query(q_text, top_k=top_k, mode=mode)

            # 用 nomic-embed-text 判定相关性
            query_vec = self._embedder.embed_query(q_text)
            query_vec = query_vec.flatten()

            relevant_rank = None
            recall_5_hit = False
            recall_10_hit = False

            for rank, sid in enumerate(result.sphere_ids, start=1):
                doc_text = self._sphere_texts.get(sid, "")
                if not doc_text:
                    continue

                doc_vec = self._embedder.embed_query(doc_text)
                doc_vec = doc_vec.flatten()

                # 余弦相似度
                norm_q = np.linalg.norm(query_vec)
                norm_d = np.linalg.norm(doc_vec)
                if norm_q == 0 or norm_d == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(query_vec, doc_vec) / (norm_q * norm_d))

                if sim > self._threshold:
                    if relevant_rank is None:
                        relevant_rank = rank
                    if rank <= 5:
                        recall_5_hit = True
                    if rank <= 10:
                        recall_10_hit = True
                    break  # 找到第一个相关的就停

            if relevant_rank is not None:
                mrr_total += 1.0 / relevant_rank
            if recall_5_hit:
                recall_5_total += 1.0
            if recall_10_hit:
                recall_10_total += 1.0

            per_query.append({
                "query": q_text[:40],
                "mrr": 1.0 / relevant_rank if relevant_rank else 0.0,
                "recalled_5": recall_5_hit,
                "recalled_10": recall_10_hit,
                "first_relevant_rank": relevant_rank,
            })

        n = len(queries)
        results = {
            "mrr": mrr_total / n if n > 0 else 0.0,
            "recall_at_5": recall_5_total / n if n > 0 else 0.0,
            "recall_at_10": recall_10_total / n if n > 0 else 0.0,
            "total_queries": n,
            "threshold": self._threshold,
            "mode": mode,
            "per_query": per_query,
        }

        logger.info(
            f"DynamicBenchmark [{mode}]: "
            f"MRR={results['mrr']:.4f}, "
            f"Recall@5={results['recall_at_5']:.4f}, "
            f"Recall@10={results['recall_at_10']:.4f} "
            f"(threshold={self._threshold}, n={n})"
        )
        return results

    @staticmethod
    def compare(results: List[dict], labels: List[str]):
        """横向对比多个结果"""
        print("=" * 80)
        print(f"{'Metric':<25s}", end="")
        for label in labels:
            print(f"{label:>15s}", end="")
        print()
        print("-" * 80)

        for metric in ["mrr", "recall_at_5", "recall_at_10", "total_queries"]:
            print(f"{metric:<25s}", end="")
            for r in results:
                v = r.get(metric, 0)
                if isinstance(v, float):
                    print(f"{v:>15.4f}", end="")
                else:
                    print(f"{v:>15}", end="")
            print()

        print("=" * 80)
