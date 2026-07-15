"""
navigate.py — 球体导航工具
===========================
从指定球体出发，沿连接图行走 n 跳，返回路径上的球体。

用途：
  - 发现"这个概念的上下游是什么"
  - 回答时扩展上下文
  - 知识图谱式探索
"""

import logging
from collections import deque
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def navigate_sphere(
    sphere_id: str,
    sphere_store,
    connections_provider: Callable[[str], Dict[str, float]],
    hops: int = 2,
    min_weight: float = 0.2,
    include_self: bool = True,
) -> dict:
    """从指定球体出发，沿连接图行走 n 跳

    Args:
        sphere_id: 起点球体 ID
        sphere_store: SphereStore 实例
        connections_provider: 函数(sphere_id) -> {neighbor_id: weight}
        hops: 行走跳数
        min_weight: 最小连接权重
        include_self: 结果中是否包含起点

    Returns:
        {
            "start_id": str,
            "hops": int,
            "path": [sphere_id, ...],  # BFS 访问顺序
            "spheres": {id: {text, source, cluster, mass, ...}, ...},
            "edges": [(from, to, weight), ...],
            "total": int
        }
    """
    if not sphere_id:
        return {"error": "sphere_id is required"}

    visited = set()
    queue = deque([(sphere_id, 0)])
    path = []
    edges = []
    spheres_data = {}

    while queue:
        current_id, depth = queue.popleft()
        if current_id in visited or depth > hops:
            continue
        visited.add(current_id)

        # 获取球体元数据
        sphere = sphere_store.get(current_id) if sphere_store else None
        if sphere:
            spheres_data[current_id] = {
                "text": sphere.text[:200],
                "source_file": sphere.source_file,
                "source_type": sphere.source_type,
                "cluster_id": sphere.cluster_id,
                "mass": sphere.mass,
                "diversity": sphere.diversity,
            }

        if depth > 0 or include_self:
            path.append(current_id)

        # 遍历邻居
        neighbors = connections_provider(current_id) if depth < hops else {}
        if neighbors:
            for neighbor_id, weight in sorted(neighbors.items(),
                                               key=lambda x: -x[1]):
                if weight >= min_weight and neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))
                    edges.append({
                        "from": current_id,
                        "to": neighbor_id,
                        "weight": weight,
                    })

    return {
        "start_id": sphere_id,
        "hops": hops,
        "path": path,
        "spheres": spheres_data,
        "edges": edges,
        "total": len(path),
    }
