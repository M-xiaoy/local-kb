import sys; sys.path.insert(0, '.')
import re, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

from storage.sphere_store import SphereStore
from config import paths
from collections import Counter, defaultdict

store = SphereStore(paths.spheres_data)
store.load()
spheres = store.get_active()
print(f'Total active spheres: {len(spheres)}')

# Look for heading patterns
h1_topics = Counter()
for s in spheres[:5000]:
    text = s.text[:200]
    m = re.search(r'^##+\s+(.+)$', text, re.MULTILINE)
    if m:
        level = len(m.group(0).split()[0]) - 1
        title = m.group(1).strip()[:50].replace('\ufe0f','').replace('\U0001f6e0','[tool]').replace('\U0001f3a8','[art]')
        h1_topics[(level, title)] += 1

print('\nHeading patterns:')
for (lvl, title), cnt in h1_topics.most_common(25):
    print(f'  h{lvl}: {title} ({cnt})')

# Embedding source
emb_src = Counter(s.embedding_source for s in spheres if s.embedding_source)
print(f'\nEmbedding sources: {dict(emb_src)}')

# Keywords
print('\n--- Keyword frequencies ---')
keywords = ['重力', '双曲', 'poincar', 'gravity', 'hyperbolic',
            'comfyui', 'remotion', '知识库', '溪流', 'stream',
            'transformer', 'attention', 'rag', 'agent',
            'poincare', 'faiss', 'embedding', 'llm', 'gpt']
for kw in keywords:
    count = sum(1 for s in spheres if kw.lower() in s.text.lower())
    print(f'  {kw}: {count} spheres')

# Cluster x Source Type
print('\n--- Cluster x Source Type ---')
cluster_src = defaultdict(Counter)
for s in spheres:
    if s.cluster_id is not None and s.source_type:
        cluster_src[s.cluster_id][s.source_type] += 1
for cid in sorted(cluster_src.keys())[:15]:
    print(f'  Cluster {cid}: {dict(cluster_src[cid])}')
