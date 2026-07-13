"""
cluster_engine.py — 聚类引擎
=============================
管理球体向量的 k-means 聚类：
  - 接收球体向量 → 聚类 → 输出簇中心和归属
  - 持久化/加载聚类状态
  - 上传完成后触发，全量重聚类（当前数据量 ~1500，全量 <100ms）

与 FieldDetector 的关系：
  FieldDetector 从本引擎读取聚类中心，不自己计算质心。
  本引擎不依赖 FieldDetector。

与 SphereStore 的关系：
  聚类结果（cluster_id）写入每个 Sphere 对象。
  本引擎不直接操作 SphereStore，只返回 labels。
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import clustering as cfg_clustering

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 聚类引擎
# ──────────────────────────────────────────────

class ClusterEngine:
    """球体向量的 k-means 聚类管理器

    用法：
      engine = ClusterEngine()
      centroids, labels, scores = engine.fit_predict(vectors)
      engine.save()  # 持久化
    """

    def __init__(self, n_clusters: Optional[int] = None):
        self.n_clusters = n_clusters or cfg_clustering.n_clusters
        self._model: Optional[MiniBatchKMeans] = None
        self._centroids: Optional[np.ndarray] = None  # shape (k, dim)
        self._inertia: float = 0.0
        self._n_samples: int = 0
        self._n_iter: int = 0
        self._state_path = Path(cfg_clustering.state_file)

    # ── 属性 ──────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._centroids is not None

    @property
    def centroids(self) -> Optional[np.ndarray]:
        return self._centroids

    @property
    def n_centroids(self) -> int:
        return self._centroids.shape[0] if self._centroids is not None else 0

    @property
    def inertia(self) -> float:
        """簇内平方和（越小越好，评估聚类质量）"""
        return self._inertia

    # ── 聚类 ──────────────────────────────────

    def fit_predict(
        self, vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """全量聚类

        Args:
            vectors: shape (n, embed_dim), float32, 已 L2 归一化

        Returns:
            centroids: shape (k, embed_dim), float32, L2 归一化
            labels:    shape (n,), int, 每个向量的簇 ID (0..k-1)
            scores:    shape (n,), float, 到所属簇中心的余弦相似度 [0,1]
        """
        n = vectors.shape[0]
        if n == 0:
            return np.zeros((0, vectors.shape[1])), np.array([], dtype=int), np.array([], dtype=float)

        # 自适应 K
        k = min(self.n_clusters, max(2, n))

        # 自动检测最优 K（silhouette score）
        if cfg_clustering.auto_detect_k and n >= 4:
            detected_k = self._auto_detect_k(vectors)
            if detected_k > 0:
                k = detected_k

        # 全量 KMeans（数据量小，全量比 MiniBatch 更稳定）
        # 使用 k-means++ 初始化 + 余弦距离（向量已 L2 归一化，欧氏最近=余弦最近）
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            max_iter=cfg_clustering.max_iter,
            n_init=cfg_clustering.n_init,
            random_state=cfg_clustering.random_state,
            algorithm="lloyd",
        )

        t0 = time.time()
        model.fit(vectors)
        elapsed = (time.time() - t0) * 1000

        centroids = model.cluster_centers_.astype(np.float32)
        # 确保质心也是 L2 归一化的
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        centroids = centroids / norms

        labels = model.labels_.astype(np.int32)

        # 计算每个点到所属簇中心的余弦相似度
        # 因为向量和质心都已归一化，点积就是余弦相似度
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            score = float(np.dot(vectors[i], centroids[labels[i]]))
            scores[i] = max(0.0, score)

        self._model = model
        self._centroids = centroids
        self._inertia = float(model.inertia_)
        self._n_samples = n
        self._n_iter = int(model.n_iter_)

        logger.info(
            f"Clustering: {n} vectors → {k} clusters "
            f"({elapsed:.1f}ms, inertia={self._inertia:.4f}, "
            f"iter={self._n_iter})"
        )

        return centroids, labels, scores

    # ── 自动检测 K ───────────────────────────

    def _auto_detect_k(self, vectors: np.ndarray) -> int:
        """使用 silhouette score 自动检测最优 K 值

        遍历 k ∈ [n_clusters, max_k]，选 silhouette 最高的。
        对小数据集（<1000 条，k≤20）耗时 <500ms。

        Args:
            vectors: shape (n, dim), float32, 已 L2 归一化

        Returns:
            最优 K 值，失败时返回 -1（调用方保持默认 K）
        """
        n = vectors.shape[0]
        min_k = max(2, self.n_clusters)
        max_k = min(cfg_clustering.max_k, int(np.sqrt(n)))

        if n <= min_k or max_k <= min_k:
            return -1

        best_k = -1
        best_score = -1.0

        for k in range(min_k, max_k + 1):
            try:
                # 快速检测：n_init=3，不跑满
                model = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    max_iter=cfg_clustering.max_iter,
                    n_init=3,
                    random_state=cfg_clustering.random_state,
                    algorithm="lloyd",
                )
                labels = model.fit_predict(vectors)

                unique_labels = set(labels)
                if len(unique_labels) < 2:
                    continue

                score = float(silhouette_score(vectors, labels, random_state=42))

                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                logger.debug(f"Auto-detect k={k} failed: {e}")
                continue

        if best_k > 0:
            logger.info(
                f"Auto-detect K: {best_k} (silhouette={best_score:.4f}, "
                f"range=[{min_k}, {max_k}], n={n})"
            )
        else:
            logger.info(
                f"Auto-detect K: no improvement over default "
                f"(range=[{min_k}, {max_k}], n={n})"
            )

        return best_k

    def predict(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对已有的质心预测新向量的簇归属

        不触发聚类，只做最近质心匹配。
        用于上传后快速分配新球体（不重聚类时）。

        Args:
            vectors: shape (n, dim), float32, 已归一化

        Returns:
            labels: shape (n,), int
            scores: shape (n,), float
        """
        if self._centroids is None or vectors.shape[0] == 0:
            empty = np.array([], dtype=np.int32)
            return empty, np.array([], dtype=float)

        # 最近质心分配
        sim_matrix = vectors @ self._centroids.T  # (n, k)
        labels = np.argmax(sim_matrix, axis=1).astype(np.int32)
        scores = np.maximum(0.0, np.max(sim_matrix, axis=1))

        return labels, scores

    # ── 持久化 ───────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """保存聚类状态"""
        save_path = Path(path) if path else self._state_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self._centroids is None:
            logger.warning("No centroids to save")
            return str(save_path)

        data = {
            "version": 1,
            "n_clusters": self.n_centroids,
            "n_samples": self._n_samples,
            "n_iter": self._n_iter,
            "inertia": self._inertia,
            "centroids": self._centroids.tolist(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Saved cluster state: {self.n_centroids} clusters, "
            f"{self._n_samples} samples → {save_path}"
        )
        return str(save_path)

    def load(self, path: Optional[str] = None) -> bool:
        """加载已保存的聚类状态

        Returns:
            True 表示加载成功，False 表示文件不存在或格式无效
        """
        load_path = Path(path) if path else self._state_path

        if not load_path.exists():
            logger.info(f"No cluster state at {load_path}, starting fresh")
            return False

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            centroids_list = data.get("centroids")
            if not centroids_list:
                logger.warning(f"Cluster state at {load_path} has no centroids")
                return False

            self._centroids = np.array(centroids_list, dtype=np.float32)
            self._n_samples = data.get("n_samples", 0)
            self._inertia = data.get("inertia", 0.0)
            self._n_iter = data.get("n_iter", 0)

            # 重建 KMeans 对象（只存初始化参数）
            self._model = KMeans(
                n_clusters=self.n_centroids,
                random_state=cfg_clustering.random_state,
            )
            self._model.cluster_centers_ = self._centroids

            logger.info(
                f"Loaded cluster state: {self.n_centroids} clusters, "
                f"{self._n_samples} samples, inertia={self._inertia:.4f}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to load cluster state: {e}")
            return False

    # ── 自动检测 K ───────────────────────────

    def _auto_detect_k(self, vectors: np.ndarray) -> int:
        """使用 silhouette score 自动检测最优 K 值

        遍历 k ∈ [self.n_clusters, max_k]，选 silhouette 最高的。
        对小数据集（<1000 条，k≤20）耗时 <500ms。

        Args:
            vectors: shape (n, dim), float32, 已 L2 归一化

        Returns:
            最优 K 值，失败时返回 -1（调用方保持默认 K）
        """
        n = vectors.shape[0]
        min_k = max(2, self.n_clusters)
        max_k = min(cfg_clustering.max_k, int(np.sqrt(n)))

        if n <= min_k or max_k <= min_k:
            return -1

        best_k = -1
        best_score = -1.0

        for k in range(min_k, max_k + 1):
            try:
                # 快速检测：n_init=3，不跑满
                model = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    max_iter=cfg_clustering.max_iter,
                    n_init=3,
                    random_state=cfg_clustering.random_state,
                    algorithm="lloyd",
                )
                labels = model.fit_predict(vectors)

                unique_labels = set(labels)
                if len(unique_labels) < 2:
                    continue

                score = float(silhouette_score(vectors, labels, random_state=42))

                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                logger.debug(f"Auto-detect k={k} failed: {e}")
                continue

        if best_k > 0:
            logger.info(
                f"Auto-detect K: {best_k} (silhouette={best_score:.4f}, "
                f"range=[{min_k}, {max_k}], n={n})"
            )
        else:
            logger.info(
                f"Auto-detect K: no improvement over default "
                f"(range=[{min_k}, {max_k}], n={n})"
            )

        return best_k

    # ── 辅助 ─────────────────────────────────

    def get_cluster_label(
        self, cluster_id: int, label_map: Optional[Dict[int, str]] = None
    ) -> str:
        """获取簇的显示名

        Args:
            cluster_id: 簇 ID
            label_map: {0: "技术笔记", 1: "小说创作", ...}

        Returns:
            "簇0" 或 "技术笔记"（如果有 label_map 映射）
        """
        if label_map and cluster_id in label_map:
            return label_map[cluster_id]
        return f"簇{cluster_id}"

    def get_cluster_sizes(self, labels: np.ndarray) -> Dict[int, int]:
        """统计每个簇的球体数量"""
        unique, counts = np.unique(labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}
