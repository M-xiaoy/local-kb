"""
embedder.py — 向量嵌入器
========================
调 Ollama 嵌入模型，将文本转为归一化向量。

最佳实践（2025-2026 行业共识）：
  1. Task Prefix 不可省略——nomic-embed-text 要求文档加 'search_document:'，
     查询加 'search_query:'，不加则检索质量显著下降
  2. L2 归一化——归一化后 cosine similarity = dot product，
     FAISS 可用 IP（内积）代替余弦，速度更快
  3. 批量调用——Ollama /api/embed 支持 batch input，
     比逐条调用快一个数量级
  4. 缓存——embedding 对同一模型+同一输入是确定性的，
     缓存避免重复计算
  5. 同一模型索引全量文档——不同模型的向量空间不可混用

外部调用：
    embedder = Embedder()
    vectors = embedder.embed_documents(["文本1", "文本2", ...])
    q_vec   = embedder.embed_query("用户问题")
    # → np.ndarray shape: (n, 768) — 已 L2 归一化
"""

import hashlib
import logging
from functools import lru_cache
from typing import List, Optional

import httpx
import numpy as np

from config import ollama as cfg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Embedder
# ──────────────────────────────────────────────

class Embedder:
    """Ollama 嵌入模型包装器"""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        batch_size: Optional[int] = None,
        doc_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
        embed_dim: Optional[int] = None,
        timeout: Optional[int] = None,
        cache_size: Optional[int] = None,
    ):
        self.model = model or cfg.embed_model
        self.host = host or cfg.host
        self.batch_size = batch_size or cfg.embed_batch_size
        self.doc_prefix = doc_prefix or cfg.embed_doc_prefix
        self.query_prefix = query_prefix or cfg.embed_query_prefix
        self.embed_dim = embed_dim or cfg.embed_dim
        self.timeout = timeout or cfg.embed_timeout
        self._url = f"{self.host}/api/embed"

        # 缓存：模型名 + 输入文本 hash → 向量
        self._cache: dict = {}
        self._cache_max = cache_size or cfg.embed_cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    # ── 公开接口 ───────────────────────────────

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """文档嵌入：加 'search_document:' 前缀，批量执行"""
        prefixed = [f"{self.doc_prefix}{t}" for t in texts]
        return self._embed_batch(prefixed)

    def embed_query(self, text: str) -> np.ndarray:
        """查询嵌入：加 'search_query:' 前缀"""
        prefixed = f"{self.query_prefix}{text}"
        raw = self._call_api([prefixed])
        vec = np.array(raw[0], dtype=np.float32)
        return self._normalize(vec.reshape(1, -1)).flatten()

    # ── 批量嵌入核心 ──────────────────────────

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入：分片 + 缓存命中检查"""
        all_vectors: List[np.ndarray] = []

        for chunk_start in range(0, len(texts), self.batch_size):
            chunk = texts[chunk_start:chunk_start + self.batch_size]

            # 检查缓存
            uncached: List[tuple] = []  # (index, text)
            cached_vectors: List[tuple] = []  # (index, vector)

            for idx, text in enumerate(chunk):
                key = self._cache_key(text)
                if key in self._cache:
                    self._cache_hits += 1
                    cached_vectors.append((idx, self._cache[key]))
                else:
                    self._cache_misses += 1
                    uncached.append((idx, text))

            # 未命中的部分发 API
            if uncached:
                api_texts = [t for _, t in uncached]
                raw_vectors = self._call_api(api_texts)

                # 更新缓存
                for (idx, text), vec_list in zip(uncached, raw_vectors):
                    vec = np.array(vec_list, dtype=np.float32)
                    key = self._cache_key(text)
                    self._cache[key] = vec
                    cached_vectors.append((idx, vec))

            # 按原始顺序合并
            cached_vectors.sort(key=lambda x: x[0])
            for _, vec in cached_vectors:
                all_vectors.append(vec)

        # 合并为矩阵后归一化
        matrix = np.stack(all_vectors, axis=0)
        return self._normalize(matrix)

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用 Ollama /api/embed"""
        if not texts:
            return []

        for attempt in range(3):  # 最多重试 3 次
            try:
                resp = httpx.post(
                    self._url,
                    json={"model": self.model, "input": texts},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["embeddings"]
            except httpx.ConnectError:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.host}. "
                    "Is Ollama running?"
                )
            except httpx.TimeoutException:
                if attempt < 2:
                    logger.warning(
                        f"Ollama embedding timeout (attempt {attempt + 1}/3)"
                    )
                    continue
                raise TimeoutError(
                    f"Ollama embedding timed out after 3 attempts. "
                    f"Model: {self.model}, batch size: {len(texts)}"
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise LookupError(
                        f"Model '{self.model}' not found in Ollama. "
                        f"Run: ollama pull {self.model}"
                    )
                raise RuntimeError(
                    f"Ollama API error ({e.response.status_code}): {e.response.text}"
                )
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(
                    f"Unexpected Ollama response format: {e}"
                )

    # ── 归一化 ─────────────────────────────────

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 归一化：每个向量除以其 L2 范数

        归一化后 cosine similarity = dot product，
        FAISS 可用 IP 代替余弦，计算更快。
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
    # ── 缓存 ──────────────────────────

    def _cache_key(self, text: str) -> str:
        """生成缓存 key：model + text hash"""
        raw = f"{self.model}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _trim_cache(self):
        """缓存超过上限时淘汰最早的一半"""
        if len(self._cache) > self._cache_max:
            keys = list(self._cache.keys())
            for k in keys[:len(keys) // 2]:
                del self._cache[k]

    def cache_stats(self) -> dict:
        return {
            "size": len(self._cache),
            "max": self._cache_max,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(self._cache_hits + self._cache_misses, 1),
        }

    def clear_cache(self):
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# 


# ──────────────────────────────────────────────
# Poincaré Ball 投影
# ──────────────────────────────────────────────

def poincare_project(vectors: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """将 L2 归一化的单位向量投影到 Poincaré Ball

    每个向量以其对应的范数缩放：v_poincare = v_unit × norm
    投影后的向量在 Poincaré Ball 中，可直接存 FAISS。
    后续用真实 Poincaré 距离重新排序。

    Args:
        vectors: shape (n, dim) float32，L2 归一化的单位向量
        norms: shape (n,) 或 (n, 1) float32，每个向量的 Poincaré 范数

    Returns:
        shape (n, dim) float32，Poincaré Ball 中的向量
    """
    norms = np.asarray(norms, dtype=np.float32).reshape(-1, 1)
    return vectors * norms


def poincare_project_query(query_vec: np.ndarray) -> np.ndarray:
    """查询向量投影到 Poincaré Ball

    查询向量用一个固定 query_norm 缩放（当前 0.3）。
    这个值位于 Poincaré Ball 的偏球心区域，
    保证搜索时对高 norm（具体）和低 norm（抽象）都有合理覆盖。

    Args:
        query_vec: shape (dim,) float32，L2 归一化的查询向量

    Returns:
        shape (dim,) float32，Poincaré Ball 中的查询向量
    """
    from config import poincare_mapping as cfg
    return query_vec * cfg.query_norm


# ──────────────────────────────────────────────
# 快捷函数
# ──────────────────────────────────────────────

_global_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """获取全局 Embedder 单例"""
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = Embedder()
    return _global_embedder


def embed_documents(texts: List[str]) -> np.ndarray:
    """快捷：文档嵌入"""
    return get_embedder().embed_documents(texts)


def embed_query(text: str) -> np.ndarray:
    """快捷：查询嵌入"""
    return get_embedder().embed_query(text)
