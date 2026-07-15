"""
explore.py — 聚簇展开工具
==========================
展开一个聚簇的所有内容，返回簇内球体的层次视图。

用途：
  - 浏览某个主题领域的所有内容
  - 生成器需要该簇的完整背景时调用
  - 回答"这个簇里有什么"类问题
"""

import logging
from typing import Dict, List, Optional

from config import ollama as cfg_ollama

logger = logging.getLogger(__name__)


def explore_cluster(
    cluster_id: int,
    sphere_store,
    field_detector=None,
    sort_by: str = "mass",
    top_k: int = 30,
    include_summary: bool = False,
) -> dict:
    """展开一个聚簇的内容

    Args:
        cluster_id: 簇 ID
        sphere_store: SphereStore 实例
        field_detector: FieldDetector 实例（可选，用于获取场域信息）
        sort_by: 排序方式 (mass | diversity | created_at)
        top_k: 返回数量
        include_summary: 是否生成簇摘要（会调 LLM）

    Returns:
        {
            "cluster_id": int,
            "total_spheres": int,
            "spheres": [Sphere 摘要, ...],
            "top_connections": [(id1, id2, weight), ...],
            "field_mapping": {场域名: 占比},
            "summary": str (可选)
        }
    """
    if sphere_store is None:
        return {"error": "sphere_store is required"}

    # 获取该簇的所有活跃球体
    all_spheres = sphere_store.get_active()
    cluster_spheres = [s for s in all_spheres if s.cluster_id == cluster_id]

    if not cluster_spheres:
        return {
            "cluster_id": cluster_id,
            "total_spheres": 0,
            "spheres": [],
        }

    # 排序
    if sort_by == "mass":
        cluster_spheres.sort(key=lambda s: s.effective_mass, reverse=True)
    elif sort_by == "diversity":
        cluster_spheres.sort(key=lambda s: s.diversity, reverse=True)
    else:
        cluster_spheres.sort(key=lambda s: s.created_at, reverse=True)

    # 截取 Top-K
    top_spheres = cluster_spheres[:top_k]

    # 球体摘要
    spheres_summary = []
    for s in top_spheres:
        spheres_summary.append({
            "id": s.id,
            "text_preview": s.text[:150],
            "source_file": s.source_file,
            "source_type": s.source_type,
            "mass": s.mass,
            "diversity": s.diversity,
            "effective_mass": s.effective_mass,
            "connections": len(s.connections) if s.connections else 0,
            "gravity_field": dict(sorted(
                s.gravity_field.items(), key=lambda x: -x[1]
            )[:3]) if s.gravity_field else {},
        })

    # 簇内连接统计
    connections = []
    for s in top_spheres[:10]:
        if s.connections:
            for target_id, weight in sorted(
                s.connections.items(), key=lambda x: -x[1]
            )[:5]:
                if weight > 0.2:
                    connections.append((s.id, target_id, weight))

    # 场域映射
    field_mapping = {}
    if field_detector:
        for s in cluster_spheres:
            for field, val in s.gravity_field.items():
                if val > 0.3:
                    field_mapping[field] = field_mapping.get(field, 0) + 1
        # 转为占比
        total = len(cluster_spheres)
        if total > 0:
            field_mapping = {
                k: round(v / total, 3)
                for k, v in sorted(field_mapping.items(), key=lambda x: -x[1])
            }

    result = {
        "cluster_id": cluster_id,
        "total_spheres": len(cluster_spheres),
        "spheres": spheres_summary,
        "top_connections": connections,
        "field_mapping": field_mapping,
    }

    return result
