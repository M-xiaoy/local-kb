"""
migrate_v2.py — 重力知识库 v2 全量迁移脚本
============================================
将 2613 个现有球体迁移到 v2 架构。

执行顺序：
  1. 加载所有持久化数据
  2. 全量连接检测（connections.py）
  3. mass/diversity 校准（calibrator.py）
  4. 持久化新数据
  5. 报告迁移结果

用法：
  python scripts/migrate_v2.py           # 正常迁移
  python scripts/migrate_v2.py --dry-run # 预览，不写文件
  python scripts/migrate_v2.py --quick   # 只做连接+校准，不做重写
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# 确保能找到 local-kb 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 延迟导入（等 sys.path 设好） ──

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("migrate_v2")


def main():
    parser = argparse.ArgumentParser(description="重力知识库 v2 迁移")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不写文件")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：跳过文本重写，只做连接+校准")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("重力知识库 v2 迁移开始")
    logger.info(f"模式: {'DRY RUN' if args.dry_run else '实际执行'}"
                f"{' + QUICK' if args.quick else ''}")
    logger.info("=" * 50)

    # ── 加载现有数据 ──
    logger.info("[1/6] 加载现有数据...")
    from storage.sphere_store import SphereStore
    from storage.registry import Registry
    from storage.faiss_store import FaissStore

    sphere_store = SphereStore()
    count = sphere_store.load()
    logger.info(f"  已加载 {count} 个球体 ({sphere_store.count} 活跃)")

    registry = Registry()
    rcount = registry.load()
    logger.info(f"  已加载 {rcount} 个映射")

    faiss_store = FaissStore()
    fcount = faiss_store.load()
    logger.info(f"  已加载 {fcount} 个 FAISS 向量")

    # ── 检查数据可用性 ──
    if sphere_store.count == 0:
        logger.warning("没有活跃球体，迁移无意义")
        return

    # ── Step 2: 文本重写（可选） ──
    if not args.quick:
        logger.info("[2/6] 文本重写（会话记录）...")
        from pipeline.rewriter import TextRewriter

        rewriter = TextRewriter()
        session_spheres = sphere_store.get_by_type("会话记录")
        logger.info(f"  找到 {len(session_spheres)} 个会话记录球体")

        if not args.dry_run:
            rewritten = 0
            for i, sphere in enumerate(session_spheres):
                if i % 50 == 0 and i > 0:
                    logger.info(f"  进度: {i}/{len(session_spheres)}")
                    sphere_store.save()  # 渐进保存

                try:
                    clean = rewriter.rewrite(
                        sphere.text, source_type="会话记录",
                        source_file=sphere.source_file
                    )
                    if clean.cleaned_text and clean.cleaned_text != sphere.text:
                        sphere.text = clean.cleaned_text
                        # entities 存为 sphere 的扩展属性
                        # 注意 Sphere 目前没有 entities 字段
                        # 用 source_type 携带实体信息
                        sphere.source_type = "会话记录_重写"
                        rewritten += 1
                except Exception as e:
                    logger.warning(f"  球体 {sphere.id[:8]} 重写失败: {e}")

            logger.info(f"  实际重写: {rewritten} 个球体")
        else:
            logger.info(f"  [DRY RUN] 跳过实际重写")
    else:
        logger.info("[2/6] 文本重写: 跳过 (--quick)")

    # ── Step 3: 全量连接检测 ──
    logger.info("[3/6] 全量连接检测...")
    from pipeline.connections import ConnectionDetector

    detector = ConnectionDetector(sphere_store, faiss_store._vectors)

    if not args.dry_run:
        total_conns = detector.detect_batch()
        detector.save()
        logger.info(f"  创建了 {total_conns} 条连接")
        logger.info(f"  平均连接度: {detector.avg_degree:.1f}")
        logger.info(f"  总节点: {len(detector._connections)}")
    else:
        logger.info(f"  [DRY RUN] 跳过实际连接检测")
        total_conns = 0

    # ── Step 4: mass/diversity 校准 ──
    logger.info("[4/6] 质量与多样性校准...")
    from storage.calibrator import SphereCalibrator

    calibrator = SphereCalibrator(sphere_store, faiss_store._vectors)

    if not args.dry_run:
        result = calibrator.calibrate_all()
        logger.info(f"  校准: {result['calibrated']} 个球体")
        logger.info(f"  mass: [{result['mass_range'][0]:.3f}, {result['mass_range'][1]:.3f}]")
        logger.info(f"  diversity: [{result['diversity_range'][0]:.3f}, {result['diversity_range'][1]:.3f}]")
    else:
        logger.info(f"  [DRY RUN] 跳过实际校准")

    # ── Step 5: 场域质心更新 ──
    logger.info("[5/6] 场域质心同步...")
    from analysis.deprecated_pipeline.field_detector import FieldDetector  # 离线重建用

    if not args.dry_run:
        field_detector = FieldDetector()
        # 从聚类引擎同步质心
        from analysis.deprecated_pipeline.cluster_engine import ClusterEngine  # 离线重建用
        engine = ClusterEngine()
        engine.load()

        # 重新构建 gravity_field
        field_detector.sync_from_clusters(
            engine.centroids,
            {i: f"簇{i}" for i in range(engine.n_centroids)},
        )
        field_detector.rebuild_all_gravity_fields(
            sphere_store, faiss_store._vectors
        )
        logger.info(f"  已更新 {len(field_detector.fields)} 个场域质心")
    else:
        logger.info(f"  [DRY RUN] 跳过场域更新")

    # ── Step 6: 持久化 ──
    logger.info("[6/6] 持久化...")
    if not args.dry_run:
        sphere_store.save()
        registry.save()
        faiss_store.save()
        logger.info("  所有数据已持久化")
    else:
        logger.info("  [DRY RUN] 跳过持久化")

    # ── 报告 ──
    logger.info("=" * 50)
    logger.info("迁移报告:")
    logger.info(f"  总球体: {sphere_store.count} 活跃 / {sphere_store.total_count} 总计")
    logger.info(f"  连接数: {total_conns or '(dry run)'}")
    logger.info(f"  场域数: {len(field_detector.fields) if not args.dry_run else '(dry run)'}")
    logger.info(f"  FAISS: {faiss_store.count if not args.dry_run else '(dry run)'}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
