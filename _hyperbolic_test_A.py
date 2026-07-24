"""
_hyperbolic_test_A.py — FAISS 欧氏初召 + Poincaré 重排对比实验

三路测试：
  A: FAISS cosine → top-N (当前 simple 模式)
  B: FAISS cosine → top-N → Poincaré geodesic 重排（建议架构）
  C: Pure Poincaré geodesic 全量扫描（当前 poincare 模式）

不加载 DiversitySorter / FieldDetector / RoleTable 等未验证组件。
"""
import sys, time, json, textwrap
import numpy as np

sys.path.insert(0, '.')
from config import Paths
from storage.sphere_store import SphereStore
from storage.faiss_store import FaissStore
from storage.registry import Registry
from pipeline.embedder import Embedder
from retrieval.poincare_search import to_poincare_ball, batch_poincare_distance

print("Loading...", flush=True)
t0 = time.time()
store = SphereStore(Paths().spheres_data); store.load()
faiss = FaissStore(); faiss.load()
reg = Registry(); reg.load()
embed = Embedder()
print(f"Init: {time.time()-t0:.1f}s ({store.count} spheres)", flush=True)

# ── 查询集 ──────────────────────────────────
QUERIES = [
    ("Transformer 注意力机制的工作原理", ["Transformer 解读.md", "第二章 Transformer架构.md"]),
    ("分布式训练的数据并行和模型并行区别", ["第八章分布式训练.md", "chapter8_第八章分布式训练.md"]),
    ("GPU 显存优化技术 Flash Attention", ["chapter6_第六章GPU和GPU相关的优化.md", "chapter7_第七章GPU高性能编程.md"]),
    ("什么是知识蒸馏", ["notes.md"]),
    ("LoRA 微调的原理和应用", ["notes.md", "readme.md"]),
    ("Agent 工具调用的设计模式", ["notes.md", "第五章 动手搭建大模型.md"]),
    ("RAG 检索增强生成的工作流程", ["notes.md", "readme.md"]),
    ("模型的评估指标 准确率 召回率", ["notes.md"]),
    ("多头注意力 Multi-Head Attention 计算过程", ["Transformer 解读.md", "第二章 Transformer架构.md"]),
    ("强化学习中的奖励模型 Reward Model", ["chapter14_可验证奖励的强化学习.md"]),
    ("PyTorch 的自动求导机制", ["chapter3_pytorch与资源核算.md"]),
    ("分词器类型 BPE WordPiece SentencePiece", ["chapter2_分词器.md"]),
    ("AI 三大流派 符号主义 连接主义 行为主义", ["notes.md"]),
    ("大模型幻觉问题怎么缓解", ["notes.md", "readme.md"]),
    ("向量数据库的检索流程 ANN HNSW", ["notes.md"]),
]


def source_recall(spheres, relevant_sources):
    """命中 top-N 中任一期望来源的比例"""
    if not spheres:
        return 0.0
    hits = sum(1 for s in spheres
               if any(rs.lower() in (s.source_file or '').lower() for rs in relevant_sources))
    return hits / len(spheres)


def source_precision(returned_files, relevant_sources):
    """精确定义：返回的 source_file 中有多少是期望的"""
    if not returned_files:
        return 0.0
    hits = sum(1 for f in returned_files
               if any(rs.lower() in (f or '').lower() for rs in relevant_sources))
    return hits / len(returned_files)


# ── 获取全量向量缓存（Poincaré C 模式需要） ──
all_vectors = {}
if hasattr(faiss, '_vectors'):
    all_vectors = faiss._vectors
elif hasattr(faiss, 'ids') and hasattr(faiss, 'xb'):
    for i, fid in enumerate(faiss.ids):
        all_vectors[int(fid)] = faiss.xb[i]

print(f"FAISS 向量数量: {len(all_vectors)}", flush=True)

# ── 构建 faiss_id → poincare_norm 映射 ──
faiss_to_norm = {}
norm_stats = {"used": 0, "fallback": 0}
for sid, sphere in store._spheres.items():
    if sphere.active:
        fid = reg.faiss_id(sid)
        if fid is not None:
            norm = getattr(sphere, 'poincare_norm', None)
            if norm is not None and isinstance(norm, (int, float)) and 0 < norm < 1:
                faiss_to_norm[fid] = norm
                norm_stats["used"] += 1
            else:
                faiss_to_norm[fid] = 0.5
                norm_stats["fallback"] += 1
print(f"Norm映射: {norm_stats['used']} 使用 / {norm_stats['fallback']} fallback=0.5", flush=True)

# ── 运行 ─────────────────────────────────────
results = []
for idx, (q, relevant_sources) in enumerate(QUERIES):
    print(f"\n{'='*60}", flush=True)
    print(f"[{idx+1}/15] {q}", flush=True)

    qv = embed.embed_query(q)  # (dim,) L2 normalized

    # ── A: FAISS cosine ──
    t_a = time.time()
    faiss_ids, faiss_distances, faiss_vectors = faiss.search(qv, top_k=50)
    t_a_ms = (time.time() - t_a) * 1000

    # Resolve to spheres
    a_spheres = []
    a_files = []
    for i, fid in enumerate(faiss_ids[:10]):
        sid = reg.sphere_id(int(fid))
        sphere = store.get(sid)
        if sphere:
            a_spheres.append(sphere)
            a_files.append(sphere.source_file or "")

    a_recall = source_recall(a_spheres, relevant_sources)
    a_prec = source_precision(a_files, relevant_sources)

    # ── B: FAISS top-50 → Poincaré re-rank ──
    t_b = time.time()
    # Get top-50 vectors from FAISS
    top50_ids = faiss_ids[:50]
    top50_vecs = np.stack([all_vectors[int(fid)] for fid in top50_ids], axis=0)

    # Query norm: word-count heuristic
    word_count = len(q.strip().split())
    query_norm = max(0.1, min(0.9, 0.9 - 0.05 * word_count))

    # Candidate norms
    cand_norms = np.array([faiss_to_norm.get(int(fid), 0.5) for fid in top50_ids], dtype=np.float32)

    # Poincaré distance on top-50 only
    poinc_dists = batch_poincare_distance(qv, top50_vecs, query_norm=query_norm, candidate_norms=cand_norms)

    # Re-rank by Poincaré distance (ascending = closer)
    b_order = np.argsort(poinc_dists)
    b_sorted_ids = top50_ids[b_order]
    b_sorted_dists = poinc_dists[b_order]

    t_b_ms = (time.time() - t_b) * 1000

    b_spheres = []
    b_files = []
    for fid in b_sorted_ids[:10]:
        sid = reg.sphere_id(int(fid))
        sphere = store.get(sid)
        if sphere:
            b_spheres.append(sphere)
            b_files.append(sphere.source_file or "")

    b_recall = source_recall(b_spheres, relevant_sources)
    b_prec = source_precision(b_files, relevant_sources)

    # ── C: Pure Poincaré full scan ──
    t_c = time.time()
    c_ids, c_dists, c_vectors = [], [], []

    if len(all_vectors) > 0:
        all_ids = np.array(list(all_vectors.keys()), dtype=np.int64)
        all_vecs = np.stack([all_vectors[int(fid)] for fid in all_ids], axis=0)
        all_norms = np.array([faiss_to_norm.get(int(fid), 0.5) for fid in all_ids], dtype=np.float32)
        all_poinc = batch_poincare_distance(qv, all_vecs, query_norm=query_norm, candidate_norms=all_norms)
        c_order = np.argsort(all_poinc)
        c_ids = all_ids[c_order][:50]
        c_dists = all_poinc[c_order][:50]
    t_c_ms = (time.time() - t_c) * 1000

    c_spheres = []
    c_files = []
    for fid in c_ids[:10]:
        sid = reg.sphere_id(int(fid))
        sphere = store.get(sid)
        if sphere:
            c_spheres.append(sphere)
            c_files.append(sphere.source_file or "")

    c_recall = source_recall(c_spheres, relevant_sources)
    c_prec = source_precision(c_files, relevant_sources)

    # ── Show ──
    print(f"  A(FAISS余弦): {t_a_ms:.0f}ms  recall@{10}={a_recall:.0%} prec={a_prec:.0%}")
    print(f"    top-5: {[f[-30:] for f in a_files[:5]]}")
    print(f"  B(FAISS→Poinc重排): {t_b_ms:.0f}ms  recall@{10}={b_recall:.0%} prec={b_prec:.0%}")
    print(f"    top-5: {[f[-30:] for f in b_files[:5]]}")
    b_vs_a = ""
    if abs(b_recall - a_recall) > 0.05:
        b_vs_a = " ✅ B胜" if b_recall > a_recall else " ❌ B负"
    print(f"    vs A: {b_vs_a}")
    print(f"  C(PurePoinc): {t_c_ms:.0f}ms  recall@{10}={c_recall:.0%} prec={c_prec:.0%}")
    print(f"    top-5: {[f[-30:] for f in c_files[:5]]}")

    # Show ranking changes between A and B
    a_rank = {int(fid): i for i, fid in enumerate(faiss_ids[:50])}
    b_rank = {int(b_sorted_ids[i]): i for i in range(len(b_sorted_ids))}
    moved = 0
    for fid in a_rank:
        if fid in b_rank:
            if abs(a_rank[fid] - b_rank[fid]) > 5:
                moved += 1
    print(f"  Rank变动>5: {moved}/{len(a_rank)}")

    results.append({
        "query": q,
        "expected": relevant_sources,
        "A_recall@10": round(a_recall, 3),
        "A_prec@10": round(a_prec, 3),
        "A_ms": round(t_a_ms, 1),
        "B_recall@10": round(b_recall, 3),
        "B_prec@10": round(b_prec, 3),
        "B_ms": round(t_b_ms, 1),
        "C_recall@10": round(c_recall, 3),
        "C_prec@10": round(c_prec, 3),
        "C_ms": round(t_c_ms, 1),
        "B_vs_A": "win" if b_recall > a_recall + 0.05 else ("lose" if b_recall < a_recall - 0.05 else "tie"),
        "A_top5_files": [f[-30:] for f in a_files[:5]],
        "B_top5_files": [f[-30:] for f in b_files[:5]],
        "C_top5_files": [f[-30:] for f in c_files[:5]],
    })

# ── Summary ──
print(f"\n{'='*70}", flush=True)
print(f"SUMMARY: 15 queries", flush=True)

avg_a_recall = sum(r['A_recall@10'] for r in results) / len(results)
avg_b_recall = sum(r['B_recall@10'] for r in results) / len(results)
avg_c_recall = sum(r['C_recall@10'] for r in results) / len(results)
avg_a_prec = sum(r['A_prec@10'] for r in results) / len(results)
avg_b_prec = sum(r['B_prec@10'] for r in results) / len(results)
avg_c_prec = sum(r['C_prec@10'] for r in results) / len(results)
avg_a_ms = sum(r['A_ms'] for r in results) / len(results)
avg_b_ms = sum(r['B_ms'] for r in results) / len(results)
avg_c_ms = sum(r['C_ms'] for r in results) / len(results)
wins = sum(1 for r in results if r['B_vs_A'] == 'win')
losses = sum(1 for r in results if r['B_vs_A'] == 'lose')
ties = sum(1 for r in results if r['B_vs_A'] == 'tie')

print(f"{'Mode':<30} {'recall@10':>10} {'prec@10':>10} {'latency':>10}")
print(f"{'─'*60}")
print(f"{'A FAISS cosine':<30} {avg_a_recall:>10.1%} {avg_a_prec:>10.1%} {avg_a_ms:>9.0f}ms")
print(f"{'B FAISS→Poinc re-rank':<30} {avg_b_recall:>10.1%} {avg_b_prec:>10.1%} {avg_b_ms:>9.0f}ms")
print(f"{'C Pure Poinc full scan':<30} {avg_c_recall:>10.1%} {avg_c_prec:>10.1%} {avg_c_ms:>9.0f}ms")

print(f"\nB vs A: 胜 {wins} / 负 {losses} / 平 {ties}", flush=True)
print(f"B同A排名差异>5: 平均 {sum(r.get('rank_moved',0) for r in results)/len(results):.1f}/50", flush=True)

# Save
with open("data/hyperbolic_A_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to data/hyperbolic_A_results.json", flush=True)
