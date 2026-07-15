"""
bridge.py — 球体间路径发现工具
===============================
找两个球体之间的最短路径（关系中继）。

用途：
  - 发现"技术笔记A"和"会话记录B"的隐式关联
  - 解释"为什么这两个内容相关"
  - 回答时引用路径上的关键球体作为推理链
"""

import logging
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def find_bridge(
    sphere_a: str,
    sphere_b: str,
    sphere_store,
    connections_provider: Callable[[str], Dict[str, float]],
    max_hops: int = 4,
    min_weight: float = 0.1,
) -> dict:
    """找两个球体之间的最短路径

    使用双向 BFS 在连接图中搜索最短路径。

    Args:
        sphere_a: 起点球体 ID
        sphere_b: 终点球体 ID
        sphere_store: SphereStore 实例
        connections_provider: 函数(sphere_id) -> {neighbor_id: weight}
        max_hops: 最大搜索深度
        min_weight: 连接最小权重

    Returns:
        {
            "found": bool,
            "path": [sphere_id, ...],
            "spheres_on_path": [{id, text, ...}, ...],
            "path_length": int,
            "path_types": ["direct", "short", "long"]
        }
    """
    if sphere_a == sphere_b:
        return {
            "found": True,
            "path": [sphere_a],
            "spheres_on_path": [_sphere_info(sphere_a, sphere_store)],
            "path_length": 0,
            "path_types": ["self"],
        }

    if sphere_store is None:
        return {"error": "sphere_store is required"}

    # 双向 BFS
    forward_queue = deque([sphere_a])
    backward_queue = deque([sphere_b])
    forward_visited = {sphere_a: None}  # node -> predecessor
    backward_visited = {sphere_b: None}

    meeting_point = None
    depth = 0

    while forward_queue and backward_queue and depth < max_hops:
        depth += 1

        # 前向搜索（一层）
        for _ in range(len(forward_queue)):
            current = forward_queue.popleft()
            neighbors = connections_provider(current) or {}
            for neighbor_id, weight in neighbors.items():
                if weight < min_weight:
                    continue
                if neighbor_id not in forward_visited:
                    forward_visited[neighbor_id] = current
                    forward_queue.append(neighbor_id)
                    if neighbor_id in backward_visited:
                        meeting_point = neighbor_id
                        break
            if meeting_point:
                break

        if meeting_point:
            break

        # 后向搜索（一层）
        for _ in range(len(backward_queue)):
            current = backward_queue.popleft()
            neighbors = connections_provider(current) or {}
            for neighbor_id, weight in neighbors.items():
                if weight < min_weight:
                    continue
                if neighbor_id not in backward_visited:
                    backward_visited[neighbor_id] = current
                    backward_queue.append(neighbor_id)
                    if neighbor_id in forward_visited:
                        meeting_point = neighbor_id
                        break
            if meeting_point:
                break

        if meeting_point:
            break

    # 重建路径
    if meeting_point:
        path = []
        # 前向路径
        node = meeting_point
        while node is not None:
            path.append(node)
            node = forward_visited[node]
        path.reverse()
        # 后向路径（去掉 meeting_point 重复）
        node = backward_visited[meeting_point]
        while node is not None:
            path.append(node)
            node = backward_visited[node]

        path_length = len(path) - 1

        # 路径类型
        if path_length == 1:
            path_types = ["direct"]
        elif path_length <= 3:
            path_types = ["short"]
        else:
            path_types = ["long"]

        # 路径上的球体信息
        spheres_info = [_sphere_info(sid, sphere_store) for sid in path]

        return {
            "found": True,
            "path": path,
            "spheres_on_path": spheres_info,
            "path_length": path_length,
            "path_types": path_types,
        }

    return {
        "found": False,
        "path": [],
        "spheres_on_path": [],
        "path_length": -1,
        "path_types": ["disconnected"],
    }


def _sphere_info(sphere_id: str, sphere_store) -> dict:
    """获取球体摘要信息"""
    sphere = sphere_store.get(sphere_id) if sphere_store else None
    if sphere:
        return {
            "id": sphere.id,
            "text_preview": sphere.text[:150],
            "source_file": sphere.source_file,
            "source_type": sphere.source_type,
            "cluster_id": sphere.cluster_id,
            "mass": sphere.mass,
        }
    return {"id": sphere_id, "text_preview": "[unknown]"}
