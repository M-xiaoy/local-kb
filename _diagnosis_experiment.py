"""
第四空间诊断实验 — bge-m3 是否有内部分割的几何结构？

测试三类样本对在三种子空间下的距离分布：
  聚类对（Cluster）：同一 cluster_id → 球面子空间应更接近
  语义对（Semantic）：同主题关键词 → 欧氏子空间应更接近  
  层级对（Hierarchy）：含上下位关系 → 双曲子空间应更接近

如果三种距离在不同样本对上呈现出显著的模式分化 → 分割有效
如果三种距离在所有样本对上高度相关 → 分割无效（需要投影层）
"""
import sys; sys.path.insert(0, '.')
import os; os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys; sys.stdout.reconfigure(encoding='utf-8')
import random
import numpy as np
from collections import defaultdict, Counter

from storage.sphere_store import SphereStore
from config import paths
from retrieval.poincare_search import batch_poincare_distance

# ─── 0. Load data ───
store = SphereStore(paths.spheres_data)
store.load()
spheres = store.get_active()
print(f'Total spheres: {len(spheres)}')

# Build vector cache: sphere_id -> vector
from storage.registry import Registry
registry = Registry(paths.registry_map)
registry.load()
from storage.faiss_store import FaissStore
faiss_store = FaissStore()
faiss_store.load(paths.faiss_index)
# Reload vectors from cache
import numpy as np
cache_data = np.load(paths.faiss_cache, allow_pickle=True)
cache_ids = cache_data['ids']
cache_vectors = cache_data['vectors']
# Build: sphere_id -> vector
cache_vec_map = {}
for i in range(len(cache_ids)):
    cache_vec_map[int(cache_ids[i])] = cache_vectors[i]
vectors = {}
for s in spheres:
    fid = registry.faiss_id(s.id)
    if fid is not None and fid in cache_vec_map:
        vectors[s.id] = cache_vec_map[fid]

print(f'Vectors available: {len(vectors)}/{len(spheres)}')
vec_list = list(vectors.values())
vec_all = np.stack(vec_list, axis=0)
print(f'Vector dim: {vec_all.shape[1]}')

# ─── 1. Build ground truth pairs ───

# Hierarchy keywords (上下位关系)
HIERARCHY_PAIRS = [
    ('包含', '子类'), ('分类', '类型'), ('属于', '类别'),
    ('分为', '部分'), ('组成', '组件'), ('层级', '级别'),
    ('上层', '下层'), ('抽象', '具体'), ('概念', '实例'),
    ('大类', '小类'), ('总体', '细分'), ('父类', '子类'),
    ('框架', '模块'), ('系统', '组件'), ('递归', '嵌套'),
]
HIER_KW = set()
for a, b in HIERARCHY_PAIRS:
    HIER_KW.update([a, b])

# Topic keywords for semantic grouping
TOPIC_KW = [
    'gravity', 'poincar', 'hyperbolic', '双曲', '重力',
    'transformer', 'attention', 'embedding', 'token',
    'comfyui', 'stable diffusion', 'lora',
    'rag', 'retrieval', 'faiss', 'vector',
    'llm', 'gpt', 'deepseek', 'ollama',
    'agent', 'tool', 'function calling',
    'remotion', 'video', 'render',
    'stream', '溪流', 'sensor', 'heartbeat',
]

# Build sphere-keyword index
sphere_kws = defaultdict(set)  # sphere_id -> set of matched keywords
kw_spheres = defaultdict(list)  # keyword -> list of sphere_ids

for s in spheres:
    text_lower = s.text.lower()
    for kw in TOPIC_KW:
        if kw in text_lower:
            sphere_kws[s.id].add(kw)
            kw_spheres[kw].append(s.id)

print(f'\nKeyword coverage: {len(sphere_kws)}/{len(spheres)} spheres have at least one keyword')

# Build hierarchy keyword matches
sphere_hier = defaultdict(set)
hier_spheres = defaultdict(list)
for s in spheres:
    text_lower = s.text.lower()
    for hkw in HIER_KW:
        if hkw in text_lower:
            sphere_hier[s.id].add(hkw)
            hier_spheres[hkw].append(s.id)

print(f'Hierarchy keyword coverage: {len(sphere_hier)}/{len(spheres)} spheres')

# Sample pairs for each category
def sample_pairs(ids1, ids2, n=100):
    """Sample n pairs from ids1 x ids2, respecting vectors availability"""
    available = [i for i in ids1 if i in vectors]
    pairs = []
    if not available:
        return pairs
    for _ in range(n * 3):  # oversample
        a = random.choice(available)
        b = random.choice([i for i in ids2 if i in vectors and i != a])
        if not b:
            continue
        pairs.append((a, b))
        if len(pairs) >= n:
            break
    return pairs

random.seed(42)

# Cluster pairs: same cluster_id
cluster_groups = defaultdict(list)
for s in spheres:
    if s.cluster_id is not None and s.id in vectors:
        cluster_groups[s.cluster_id].append(s.id)

cluster_pairs = []
for cid, members in cluster_groups.items():
    if len(members) >= 10:
        cluster_pairs.extend(sample_pairs(members, members, 15))
cluster_pairs = cluster_pairs[:200]
print(f'Cluster pairs: {len(cluster_pairs)}')

# Topic pairs: share at least one keyword
topic_ids = list(sphere_kws.keys())
topic_pairs = []
for _ in range(500):
    a = random.choice(topic_ids)
    kws_a = sphere_kws[a]
    # Find another sphere sharing at least one keyword
    candidates = [i for i in topic_ids if i != a and sphere_kws[i] & kws_a]
    if candidates:
        b = random.choice(candidates)
        topic_pairs.append((a, b))
topic_pairs = list(set(topic_pairs))[:200]
print(f'Semantic (topic) pairs: {len(topic_pairs)}')

# Hierarchy pairs: share hierarchy keywords
hier_ids = list(sphere_hier.keys())
hier_pairs = []
for _ in range(500):
    a = random.choice(hier_ids)
    hkws_a = sphere_hier[a]
    candidates = [i for i in hier_ids if i != a and sphere_hier[i] & hkws_a]
    if candidates:
        b = random.choice(candidates)
        hier_pairs.append((a, b))
hier_pairs = list(set(hier_pairs))[:200]
print(f'Hierarchy pairs: {len(hier_pairs)}')

# Negative pairs: different clusters, no shared keywords
neg_pairs = []
all_ids = list(vectors.keys())
for _ in range(500):
    a = random.choice(all_ids)
    b = random.choice(all_ids)
    if a == b:
        continue
    sa = store.get(a)
    sb = store.get(b)
    if sa and sb:
        # Ensure different clusters and no shared keywords
        diff_cluster = (sa.cluster_id != sb.cluster_id)
        no_shared_kw = not (sphere_kws.get(a, set()) & sphere_kws.get(b, set()))
        if diff_cluster and no_shared_kw:
            neg_pairs.append((a, b))
neg_pairs = list(set(neg_pairs))[:200]
print(f'Negative (unrelated) pairs: {len(neg_pairs)}')

# ─── 2. Define fourth space distance ───
DIM = 1024
P, Q, R = 256, 512, 256  # ball, euclidean, hyperbolic
ALPHA_BALL = 1.0
ALPHA_EUC = 1.0
ALPHA_HYP = 1.0
EPS = 1e-8

def split_vector(v):
    """Split 1024-dim vector into three subspaces"""
    a_raw = v[:P]  # → ball (will be normalized)
    b = v[P:P+Q]   # → euclidean (stay as is)
    c_raw = v[P+Q:] # → hyperbolic (project to Poincare ball)
    return a_raw, b, c_raw

def project_to_ball(a_raw):
    """Normalize to unit sphere"""
    norm = np.linalg.norm(a_raw)
    if norm < EPS:
        return np.zeros_like(a_raw)
    return a_raw / norm

def compute_ball_distance(a1, a2):
    """Chord distance on sphere: 2(1 - cos(θ))"""
    dot = np.dot(a1, a2)
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * (1.0 - dot)

def compute_euclidean_distance(b1, b2):
    """Squared L2"""
    return np.sum((b1 - b2) ** 2)

def project_to_poincare(c_raw):
    """tanh projection to Poincare ball"""
    norm = np.linalg.norm(c_raw)
    if norm < EPS:
        return np.zeros_like(c_raw)
    return np.tanh(norm) * c_raw / (norm + EPS)

def mobius_add(u, v):
    """Mobius addition in Poincare ball"""
    u_norm2 = np.sum(u ** 2)
    v_norm2 = np.sum(v ** 2)
    uv = np.dot(u, v)
    
    denominator = 1.0 + 2.0 * uv + u_norm2 * v_norm2
    if abs(denominator) < EPS:
        denominator = EPS
    
    numerator = (1.0 + 2.0 * uv + v_norm2) * u + (1.0 - u_norm2) * v
    return numerator / denominator

def compute_hyperbolic_distance(c1, c2):
    """Poincare distance via log( (1+norm)/(1-norm) )"""
    diff = mobius_add(-c1, c2)
    norm = np.linalg.norm(diff)
    norm = min(norm, 1.0 - EPS)
    return np.log((1.0 + norm) / (1.0 - norm))

def fourth_space_distance(v1, v2):
    """Combined distance: alpha_ball * ball_dist + alpha_euc * euc_dist + alpha_hyp * hyp_dist"""
    a1, b1, c1 = split_vector(v1)
    a2, b2, c2 = split_vector(v2)
    
    a1_s = project_to_ball(a1)
    a2_s = project_to_ball(a2)
    c1_s = project_to_poincare(c1)
    c2_s = project_to_poincare(c2)
    
    d_ball = compute_ball_distance(a1_s, a2_s)
    d_euc = compute_euclidean_distance(b1, b2)
    d_hyp = compute_hyperbolic_distance(c1_s, c2_s)
    
    return (ALPHA_BALL * d_ball + ALPHA_EUC * d_euc + ALPHA_HYP * d_hyp,
            d_ball, d_euc, d_hyp)

# ─── 3. Run diagnostic ───
def compute_metrics(pairs, label):
    """Compute distances for a list of (id1, id2) pairs"""
    if not pairs:
        print(f'  {label}: No pairs to evaluate')
        return
    
    d_total_list = []
    d_ball_list = []
    d_euc_list = []
    d_hyp_list = []
    
    for a, b in pairs:
        v1 = vectors[a]
        v2 = vectors[b]
        d_total, d_ball, d_euc, d_hyp = fourth_space_distance(v1, v2)
        d_total_list.append(d_total)
        d_ball_list.append(d_ball)
        d_euc_list.append(d_hyp)  # 修正变量名
        d_hyp_list.append(d_hyp)
    
    # 修正
    d_ball_vals = [fourth_space_distance(vectors[a], vectors[b])[1] for a, b in pairs]
    d_euc_vals = [fourth_space_distance(vectors[a], vectors[b])[2] for a, b in pairs]
    d_hyp_vals = [fourth_space_distance(vectors[a], vectors[b])[3] for a, b in pairs]
    
    print(f'\n  {label} ({len(pairs)} pairs):')
    print(f'    Ball dist:    mean={np.mean(d_ball_vals):.4f}  std={np.std(d_ball_vals):.4f}')
    print(f'    Euclidean:    mean={np.mean(d_euc_vals):.4f}  std={np.std(d_euc_vals):.4f}')
    print(f'    Hyperbolic:   mean={np.mean(d_hyp_vals):.4f}  std={np.std(d_hyp_vals):.4f}')
    
    # Correlations between sub-spaces
    corr_be = np.corrcoef(d_ball_vals, d_euc_vals)[0, 1]
    corr_bh = np.corrcoef(d_ball_vals, d_hyp_vals)[0, 1]
    corr_eh = np.corrcoef(d_euc_vals, d_hyp_vals)[0, 1]
    print(f'    Corr B-E: {corr_be:.3f}  B-H: {corr_bh:.3f}  E-H: {corr_eh:.3f}')
    
    return {
        'd_ball': d_ball_vals,
        'd_euc': d_euc_vals,
        'd_hyp': d_hyp_vals,
    }

print('\n' + '='*60)
print('DIAGNOSTIC: bge-m3 1024-dim split into 3 sub-spaces')
print(f'Split: Ball={P}d  Euclid={Q}d  Hyperbolic={R}d')
print('='*60)

results = {}
results['cluster'] = compute_metrics(cluster_pairs, 'Cluster (same cluster_id)')
results['topic'] = compute_metrics(topic_pairs, 'Semantic (same keyword topic)')
results['hierarchy'] = compute_metrics(hier_pairs, 'Hierarchy (same hierarchy kw)')
results['negative'] = compute_metrics(neg_pairs, 'Negative (unrelated)')

# ─── 4. Cross-category comparison ───
print('\n' + '='*60)
print('CROSS-CATEGORY COMPARISON')
print('='*60)
print('If split works:')
print('  Ball dist should be smallest for cluster pairs')
print('  Euclidean should be smallest for topic pairs')
print('  Hyperbolic should be smallest for hierarchy pairs')
print('  All should be largest for negative pairs')
print()

for subspace, name in [('d_ball', 'Ball'), ('d_euc', 'Euclidean'), ('d_hyp', 'Hyperbolic')]:
    print(f'  {name} subspace:')
    for cat in ['cluster', 'topic', 'hierarchy', 'negative']:
        if results[cat]:
            mean_val = np.mean(results[cat][subspace])
            print(f'    {cat:15s}: {mean_val:.4f}')
    print()

# ─── 5. Verdict ───
print('='*60)
print('VERDICT')
print('='*60)
# Check if sub-spaces show different patterns
# If all three correlate highly (>0.9), split is meaningless
all_ball = []
all_euc = []
all_hyp = []
for cat in ['cluster', 'topic', 'hierarchy', 'negative']:
    if results[cat]:
        all_ball.extend(results[cat]['d_ball'])
        all_euc.extend(results[cat]['d_euc'])
        all_hyp.extend(results[cat]['d_hyp'])

overall_corr_be = np.corrcoef(all_ball, all_euc)[0, 1]
overall_corr_bh = np.corrcoef(all_ball, all_hyp)[0, 1]
overall_corr_eh = np.corrcoef(all_euc, all_hyp)[0, 1]

print(f'Overall correlations between sub-spaces:')
print(f'  Ball vs Euclidean:  {overall_corr_be:.3f}')
print(f'  Ball vs Hyperbolic: {overall_corr_bh:.3f}')
print(f'  Euclidean vs Hyperbolic: {overall_corr_eh:.3f}')

if abs(overall_corr_be) > 0.9 and abs(overall_corr_bh) > 0.9 and abs(overall_corr_eh) > 0.9:
    print('\n  CONCLUSION: Sub-spaces are HIGHLY CORRELATED (>0.9).')
    print('  The 1/3 split is meaningless — all three distances measure the same thing.')
    print('  A learned projection layer is needed to disentangle geometries.')
else:
    print('\n  CONCLUSION: Sub-spaces show differentiation (correlations < 0.9).')
    print('  The split has potential — check per-category means to see if')
    print('  sub-spaces specialize as expected.')
