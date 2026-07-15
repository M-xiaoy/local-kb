"""
trace.py — 会话时间线还原工具
==============================
通过源文件名找到所有相关球体，按时序排列。

用途：
  - 检索会话记录时的上下文组装
  - 回答"之前关于X的讨论中提到过什么"
  - 还原一段对话的完整推理链
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def trace_conversation(
    source_file: str,
    sphere_store,
    connections_provider=None,
) -> dict:
    """还原一个会话的完整时间线

    Args:
        source_file: 源文件名（如 "2026-07-15.md"）
        sphere_store: SphereStore 实例
        connections_provider: 可选，用于获取连接信息

    Returns:
        {
            "source_file": str,
            "total_chunks": int,
            "spheres": [球体简介, ...],  # 按时序（chunk 顺序）
            "connections": [(id1, id2, weight), ...],
            "entity_timeline": {entity: [出现的位置索引]}
        }
    """
    if sphere_store is None:
        return {"error": "sphere_store is required"}

    # 获取该源文件的所有球体
    spheres = sphere_store.get_by_source(source_file)
    if not spheres:
        return {
            "source_file": source_file,
            "total_chunks": 0,
            "spheres": [],
        }

    # 按时序排列（按 ID 近似推断顺序）
    # sphere_id = SHA256[:12]，不能直接反映顺序
    # 用 text 开头的片段推断顺序
    spheres_sorted = _sort_by_temporal_hint(spheres)

    # 球体摘要
    sphere_list = []
    for i, s in enumerate(spheres_sorted):
        sphere_list.append({
            "index": i,
            "id": s.id,
            "text_preview": s.text[:200],
            "cluster_id": s.cluster_id,
            "mass": s.mass,
            "diversity": s.diversity,
            "connections": len(s.connections) if s.connections else 0,
            "entities": getattr(s, 'entities', []),
        })

    # 连接信息
    connections = []
    if connections_provider:
        seen = set()
        for s in spheres_sorted:
            neighbors = connections_provider(s.id)
            if neighbors:
                for nid, weight in sorted(neighbors.items(),
                                           key=lambda x: -x[1])[:5]:
                    edge = tuple(sorted([s.id, nid]))
                    if edge not in seen and weight > 0.15:
                        seen.add(edge)
                        connections.append({
                            "from": s.id,
                            "to": nid,
                            "weight": weight,
                        })

    # 实体时间线
    entity_timeline = {}
    for i, s in enumerate(spheres_sorted):
        entities = getattr(s, 'entities', [])
        if isinstance(entities, list):
            for entity in entities:
                entity_timeline.setdefault(entity, []).append(i)

    return {
        "source_file": source_file,
        "total_chunks": len(spheres_sorted),
        "spheres": sphere_list,
        "connections": connections,
        "entity_timeline": entity_timeline,
    }


def _sort_by_temporal_hint(spheres) -> list:
    """按时序提示排序球体

    目前简单实现：按 ID 的字典序排列（非理想，但作为默认方案）
    理想方案：从 sphere.text 中提取时间戳或序列号
    """
    return sorted(spheres, key=lambda s: s.id)
