"""
activation.py — 激活传播引擎
==============================
将检索从"找最近向量"升级为"探针注入球体空间，沿连接图扩散信号"。

核心思想：
  - query 作为"探针"进入球体空间
  - FAISS 找到种子球体 = 探针击中目标
  - 沿连接图 BFS 扩散激活值 = 信号在球体之间传导
  - 多路径叠加 = 一个球体可以从多条路径收到信号
  - 最终按总激活值降序输出

与 PageRank 的区别：
  - PageRank: 静态图，随机游走，与 query 无关
  - Activation Propagation: query 驱动的探针注入，不同 query 激活不同子图

使用方式：
  propagator = ActivationPropagator(connections_provider)
  results = propagator.propagate(seed_scores)
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import activation as cfg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ActivationPropagator
# ──────────────────────────────────────────────

class ActivationPropagator:
    """激活传播引擎"""

    def __init__(
        self,
        connections_provider: Optional[Callable[[str], Dict[str, float]]] = None,
        type_checker: Optional[Callable[[str, str], str]] = None,
        max_hops: Optional[int] = None,
        decay: Optional[float] = None,
        activation_threshold: Optional[float] = None,
        min_propagated: Optional[float] = None,
    ):
        """
        Args:
            connections_provider: 函数, 输入 sphere_id, 返回 {neighbor_id: weight}
            type_checker: 函数 (a, b) -> "axon:forward" | "axon:reverse" | "dendrite"
                         轴突连接处理：
                           - forward: 不衰减（因果链完整传导）
                           - reverse: 半衰减（反向传导需谨慎）
                           - dendrite/bidirectional: 按原衰减系数
            max_hops: 最大传播跳数
            decay: 每跳衰减系数
            activation_threshold: 种子激活阈值（低于此值不传播）
            min_propagated: 传播信号保留阈值（低于此值不记录）
        """
        self._conn_provider = connections_provider
        self._type_checker = type_checker or (lambda a, b: "dendrite")
        self.max_hops = max_hops or cfg.max_hops
        self.decay = decay or cfg.decay_factor
        self.activation_threshold = activation_threshold or cfg.seed_activation_threshold
        self.min_propagated = min_propagated or cfg.min_propagated

    def attach(self, connections_provider, type_checker=None):
        """延迟关联连接提供者"""
        self._conn_provider = connections_provider
        if type_checker:
            self._type_checker = type_checker

    # ── 主入口 ───────────────────────────────

    def propagate(
        self,
        seed_scores: Dict[str, float],
        exclude_seeds: bool = False,
    ) -> List[Tuple[str, float]]:
        """执行激活传播

        Args:
            seed_scores: {sphere_id: 初始激活值}
                         通常来自 FAISS 余弦相似度
            exclude_seeds: 是否从结果中排除种子球体

        Returns:
            [(sphere_id, normalized_activation), ...]
            按总激活值降序排列
        """
        if not seed_scores or self._conn_provider is None:
            return list(seed_scores.items())

        # 总激活表: sphere_id → total_activation
        activated: Dict[str, float] = {}
        # 当前跳的输入: sphere_id → activation
        hop_input: Dict[str, float] = {}
        # 已访问的球体（防止环）
        visited: set = set()

        # 初始化种子
        for sid, score in seed_scores.items():
            if score > 0:
                activated[sid] = score
                hop_input[sid] = score
                visited.add(sid)

        # BFS 传播
        for hop in range(self.max_hops):
            hop_output: Dict[str, float] = {}

            for sphere_id, activation in hop_input.items():
                # 低于阈值的种子不传播
                if activation < self.activation_threshold and hop == 0:
                    continue

                # 获取连接
                neighbors = self._conn_provider(sphere_id)
                if not neighbors:
                    continue

                for neighbor_id, conn_weight in neighbors.items():
                    # 已访问的不重复传播（防止环）
                    if neighbor_id in visited:
                        continue

                    # 轴突连接按方向处理
                    conn_type = self._type_checker(sphere_id, neighbor_id)
                    if conn_type.startswith("axon"):
                        if conn_type == "axon:forward":
                            actual_decay = 1.0  # 正向因果链不衰减
                        elif conn_type == "axon:reverse":
                            actual_decay = self.decay * 0.8  # 反向传导衰减更多
                        else:  # axon:bidirectional
                            actual_decay = self.decay * 0.9  # 无方向标记时保守衰减
                    else:
                        actual_decay = self.decay  # 树突正常衰减

                    # 传播信号 = 当前激活 × 连接权重 × 衰减
                    propagated = activation * conn_weight * actual_decay

                    if propagated >= self.min_propagated:
                        # 多路径叠加
                        hop_output[neighbor_id] = \
                            hop_output.get(neighbor_id, 0) + propagated

            # 合并到总激活
            for sid, act in hop_output.items():
                activated[sid] = activated.get(sid, 0) + act
                visited.add(sid)

            hop_input = hop_output

            # 如果本跳没有传播到任何东西，提前结束
            if not hop_output:
                break

        # 排除种子
        if exclude_seeds:
            for sid in seed_scores:
                activated.pop(sid, None)

        # 归一化
        if not activated:
            return []

        max_act = max(activated.values())
        if max_act > 0:
            normalized = {
                sid: round(act / max_act, 4)
                for sid, act in activated.items()
            }
        else:
            normalized = {sid: 0.0 for sid in activated}

        # 按激活值降序排列
        result = sorted(normalized.items(), key=lambda x: -x[1])

        logger.debug(
            f"Propagation: {len(seed_scores)} seeds → {len(result)} activated "
            f"(max_hop={self.max_hops}, decay={self.decay})"
        )
        return result

    # ── 统计 ─────────────────────────────────

    def activation_stats(self, activated: List[Tuple[str, float]]) -> dict:
        """传播结果的统计信息"""
        if not activated:
            return {"count": 0}
        scores = [s for _, s in activated]
        return {
            "count": len(activated),
            "max_score": max(scores),
            "min_score": min(scores),
            "mean_score": round(sum(scores) / len(scores), 4),
        }


# ──────────────────────────────────────────────
# 快捷函数
# ──────────────────────────────────────────────

_global_propagator: Optional[ActivationPropagator] = None


def get_propagator(connections_provider=None) -> ActivationPropagator:
    global _global_propagator
    if _global_propagator is None:
        _global_propagator = ActivationPropagator(connections_provider)
    if connections_provider and _global_propagator._conn_provider is None:
        _global_propagator.attach(connections_provider)
    return _global_propagator
