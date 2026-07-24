"""
rebuild.py — 统一重建引擎
=========================
合并了之前散布在 main.py 中的三条 rebuild 路径：

  1. upload_file 后 auto_rebuild=True 的内联聚类
  2. upload_batch 后 auto_rebuild=True 的内联聚类
  3. /rebuild 端点的全量重建

用法：
  from api.rebuild import rebuild_spaces
  stats = await rebuild_spaces(state, mode="cluster")
"""

import asyncio
import logging
import time
import traceback

import numpy as np

from storage.faiss_store import FaissStore
from storage.calibrator import SphereCalibrator
from analysis.deprecated_pipeline.field_detector import FieldDetector  # 离线重建用
from pipeline.norm_deriver import derive_and_write as _old_derive
from core.hyperbolic.radius_deriver import derive_and_write as radius_derive

logger = logging.getLogger(__name__)


async def rebuild_spaces(state, mode: str = "cluster"):
    """统一重建入口

    Args:
        state: AppState 实例（main.py 的全局 state）
        mode: "cluster" | "full"
            "cluster"（默认）— 从 FAISS 缓存读已有向量做聚类 + 场同步 + 连接 + 角色表 + 保存
            "full" — 从持久化恢复状态，必要时重嵌入，重建 FAISS + 聚类 + 全部

    Returns:
        {"status": "ok", ...} dict
    """
    timings = {}
    t0 = time.time()

    if mode == "full":
        return await _rebuild_full(state, timings, t0)

    # ── cluster 模式：只做聚类 + 场同步 + 连接 + 角色表 ──
    return await _rebuild_cluster(state, timings, t0)


async def _rebuild_cluster(state, timings, t0):
    """从 FAISS 缓存读向量，做聚类 + 场同步 + 连接 + 角色表"""

    active_spheres = state.sphere_store.get_active()
    if len(active_spheres) < 2:
        return {"status": "ok", "rebuilt": 0, "message": "Need ≥2 spheres to cluster"}

    # ── 0. 初始化离线分析组件 ─────────────────
    from analysis.deprecated_pipeline.cluster_engine import ClusterEngine  # 离线重建用
    from analysis.deprecated_pipeline.field_detector import FieldDetector  # 离线重建用
    _cluster = ClusterEngine()
    _field_detector = FieldDetector()

    # ── 1. 从 FAISS 缓存收集向量 ────────────────
    vectors = []
    for s in active_spheres:
        fid = state.registry.faiss_id(s.id)
        if fid is not None and fid in state.faiss_store._vectors:
            vectors.append(state.faiss_store._vectors[fid])

    timings["collect"] = time.time() - t0

    if len(vectors) < 2:
        return {"status": "ok", "rebuilt": 0, "message": "Need ≥2 vectors in FAISS cache"}

    # ── 2. KMeans 聚类 ──────────────────────────
    t1 = time.time()
    vectors_arr = np.stack(vectors, axis=0)
    loop = asyncio.get_event_loop()
    centroids, labels, scores = await loop.run_in_executor(
        None, _cluster.fit_predict, vectors_arr
    )
    timings["kmeans"] = time.time() - t1

    # ── 3. 分配 cluster_id ──────────────────────
    t2 = time.time()
    k = centroids.shape[0]
    label_map = {i: f"簇{i}" for i in range(k)}
    cluster_counts = {}

    for sphere, label in zip(active_spheres, labels):
        sphere.cluster_id = int(label)
        cluster_counts[sphere.cluster_id] = cluster_counts.get(sphere.cluster_id, 0) + 1
    timings["assign"] = time.time() - t2

    # ── 4. 场域同步（仅离线分析用，不挂载到检索） ──
    t3 = time.time()
    _field_detector.sync_from_clusters(centroids, label_map, cluster_counts)
    _field_detector.rebuild_all_gravity_fields(
        state.sphere_store,
        {
            s.id: state.faiss_store._vectors.get(state.registry.faiss_id(s.id))
            for s in active_spheres
            if state.registry.faiss_id(s.id) is not None
        },
    )
    timings["field_sync"] = time.time() - t3

    # ── 5. 持久化聚类状态 ──────────────────────
    t4 = time.time()
    _cluster.save()
    timings["save_clusters"] = time.time() - t4

    # ── 6. 角色表（只增量更新活跃球体） ──────
    t5 = time.time()
    for sphere in active_spheres:
        if sphere.text:
            state.role_table.register_text(sphere.id, sphere.text)
    state.role_table.save()
    timings["role_table"] = time.time() - t5

    # ── 7. 连接重建 ────────────────────────────
    t6 = time.time()
    try:
        state.ensure_connections()
        if hasattr(state.conn_detector, 'detect_batch'):
            state.conn_detector.detect_batch()
            state.conn_detector.save()
        timings["connections"] = time.time() - t6
    except Exception as e:
        logger.warning(f"Connection rebuild failed (non-critical): {e}")
        timings["connections"] = time.time() - t6

    # ── 8. 质量与多样性校准 ─────────────────
    t7a = time.time()
    try:
        # 构建向量缓存 {sphere_id: vector}
        vec_cache = {}
        for s in active_spheres:
            fid = state.registry.faiss_id(s.id)
            if fid is not None and fid in state.faiss_store._vectors:
                vec_cache[s.id] = state.faiss_store._vectors[fid]
        calibrator = SphereCalibrator(state.sphere_store, vec_cache)
        cal_result = calibrator.calibrate_all()
        state.sphere_store.save()
        timings["calibrate"] = time.time() - t7a
        logger.info(f"Calibrated: mass {cal_result['mass_range']}, diversity {cal_result['diversity_range']}")
    except Exception as e:
        logger.warning(f"Calibration failed (non-critical): {e}")
        timings["calibrate"] = time.time() - t7a

    # ── 9. Poincaré 范数推导（v3 多信号融合） ──
    t7 = time.time()
    # 从 state 的组件构造临时 repo，避免 from api.main import kb 的循环引用
    try:
        from core.repo.adapter import AdapterRepository
        from core.kb import KnowledgeBase
        repo = AdapterRepository(state.sphere_store, state.faiss_store, state.registry)
        local_kb = KnowledgeBase(repo)
        norms = radius_derive(local_kb.repo)
        logger.info(
            f"RadiusDeriver v3: updated norms for {len(norms)} spheres"
        )
    except Exception:
        # 降级：用旧的 mass 推导
        mass_map = {s.id: s.mass for s in active_spheres}
        _old_derive(state.sphere_store, mass_map)
        logger.info(f"NormDeriver (fallback): updated norms for {len(mass_map)} spheres")
    except Exception as e:
        logger.warning(f"Norm derivation failed (non-critical): {e}")
        logger.warning(traceback.format_exc())
        timings["norm_derive"] = time.time() - t7

    # ── 10. 层级重建 ────────────────────────────
    t8 = time.time()
    try:
        stats = state.build_hierarchy()
        timings["hierarchy"] = time.time() - t8
    except Exception as e:
        logger.warning(f"Hierarchy rebuild failed (non-critical): {e}")
        timings["hierarchy"] = time.time() - t8

    timings["total"] = time.time() - t0

    logger.info(
        f"Rebuild (cluster): {len(active_spheres)} spheres → {k} clusters "
        f"({timings['total']:.1f}s)"
    )

    return {
        "status": "ok",
        "mode": "cluster",
        "rebuilt": len(active_spheres),
        "clusters": k,
        "timings": {k: round(v, 1) for k, v in timings.items()},
    }


async def _rebuild_full(state, timings, t0):
    """从持久化全量重建（类似原 /rebuild 端点，但修复了逐条重嵌入 bug）"""

    # ── 1. 清空 + 重载 ─────────────────────────
    state.registry.clear()
    state.faiss_store = FaissStore()

    state.registry.load()
    state.sphere_store.load()

    active_spheres = state.sphere_store.get_active()
    if not active_spheres:
        return {"status": "ok", "rebuilt": 0, "message": "No active spheres"}
    timings["load"] = time.time() - t0

    # ── 2. 检查 FAISS 缓存 → 只嵌入缺失的 ────
    t1 = time.time()
    all_vectors = []
    all_ids = []
    field_vectors = {}

    registry = state.registry
    faiss_store = state.faiss_store
    embedder = state.embedder

    spheres_to_embed = []
    for sphere in active_spheres:
        fid = registry.faiss_id(sphere.id)
        if fid is not None and fid in faiss_store._vectors:
            vec = faiss_store._vectors[fid]
            all_vectors.append(vec)
            all_ids.append(fid)
        else:
            spheres_to_embed.append(sphere)

    # 只嵌入新/缺失的
    if spheres_to_embed:
        texts = [s.text for s in spheres_to_embed]
        new_vectors = embedder.embed_documents(texts)
        for sphere, vec in zip(spheres_to_embed, new_vectors):
            fid = registry.register(sphere.id)
            all_vectors.append(vec)
            all_ids.append(fid)

    timings["embed"] = time.time() - t1

    if not all_vectors:
        return {"status": "ok", "rebuilt": 0, "message": "No vectors to build"}

    # ── 3. 重建 FAISS 索引 ─────────────────────
    t2 = time.time()
    vectors_array = np.stack(all_vectors, axis=0)
    ids_array = np.array(all_ids, dtype=np.int64)
    faiss_store.build(vectors_array, ids_array)

    # 收集场域向量
    for sphere in active_spheres:
        if sphere.source_type:
            fid = registry.faiss_id(sphere.id)
            if fid is not None and fid in faiss_store._vectors:
                field_vectors.setdefault(sphere.source_type, []).append(
                    faiss_store._vectors[fid]
                )
    _field_detector = FieldDetector()
    _field_detector.rebuild_centroids(field_vectors)
    timings["faiss_build"] = time.time() - t2

    # ── 4. 持久化 ──────────────────────────────
    t3 = time.time()
    registry.save()
    faiss_store.save()
    timings["save_faiss"] = time.time() - t3

    # ── 5. 执行聚类重建 ────────────────────────
    cluster_result = await _rebuild_cluster(state, {}, time.time())
    timings["cluster_phase"] = cluster_result.get("timings", {})

    timings["total"] = time.time() - t0

    logger.info(
        f"Rebuild (full): {len(active_spheres)} spheres "
        f"({len(spheres_to_embed)} re-embedded) → "
        f"{cluster_result.get('clusters', 0)} clusters "
        f"({timings['total']:.1f}s)"
    )

    return {
        "status": "ok",
        "mode": "full",
        "rebuilt": len(active_spheres),
        "re_embedded": len(spheres_to_embed),
        "clusters": cluster_result.get("clusters", 0),
        "timings": {k: _round_timing(v) for k, v in timings.items()},
    }


def _round_timing(val):
    """Safely round a timing value (could be dict or float)"""
    if isinstance(val, dict):
        return {k: round(v, 1) for k, v in val.items() if isinstance(v, (int, float))}
    if isinstance(val, (int, float)):
        return round(val, 1)
    return val
