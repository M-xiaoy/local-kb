# 重力知识库 · 架构设计 v2

> **球体不孤独，检索是一场引力舞蹈。**
> 最后更新：2026-07-15

---

## 目录

1. [设计哲学](#1-设计哲学)
2. [架构总览](#2-架构总览)
3. [核心概念](#3-核心概念)
4. [入库流水线](#4-入库流水线)
5. [球体社交网络](#5-球体社交网络)
6. [检索 = 激活传播](#6-检索--激活传播)
7. [内部路由工具](#7-内部路由工具)
8. [与主流 RAG 的对应关系](#8-与主流-rag-的对应关系)
9. [配置速查](#9-配置速查)

---

## 1. 设计哲学

大多数知识库把知识当作「文档切片」——搜索就是找最像的切片，然后拼起来塞进 LLM 生成答案。

重力知识库不是这样。

它的核心隐喻是**物理空间**：

- 每条知识是一个**球体**（sphere），悬浮在高维语义空间中
- 语义相近的球体自然聚成**星团**（cluster）
- 星团中心的**质心**（centroid）产生引力场，吸引相似的球体靠近
- 每个球体有**质量**（mass）——越重要的球体越重，检索时被优先选中
- 每个球体有**多样性**（diversity）——越独特的球体离星团中心越远，在排序中有优势
- 球体之间通过**连接**（connections）组成社交网络——同主题的、同会话的、共享实体的球体互相认识

检索不再是"找一个最像的向量"，而是：

> **注入一个探针 → 激活种子球体 → 信号沿连接图扩散 → 场域引力聚焦 → 多样性筛选 → 组装回答**

v2 最大的变化：球体不再是孤立的。它们开始说话、交友、传导信号。

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                          入库前 · 文本净化                           │
│                                                                     │
│  原始数据 ──→ TextRewriter ──→ CleanDocument                         │
│  (会话/笔记/论文)    │                                             │
│                      ├─ 噪音过滤 (规则)                              │
│                      ├─ LLM结构化 (Ollama本地)                       │
│                      ├─ 实体提取                                    │
│                      └─ 指代消解                                    │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                          入库 · 球体生成                             │
│                                                                     │
│  CleanDocument                                                      │
│      │                                                              │
│  结构感知切片 (chunker) ──→ 文本块                                  │
│      │                                                              │
│  Ollama 嵌入 (nomic-embed-text) ──→ 768维向量                       │
│      │                                                              │
│  sphere_id = SHA256(text+source)[:12]                               │
│      │                                                              │
│  ┌── Sphere 对象 ────────────────────────────┐                      │
│  │  id, text, vector, source_file,           │                      │
│  │  cluster_id, gravity_field,              │                      │
│  │  mass, diversity, term_weights,           │                      │
│  │  connections ──→ ★ v2 新增               │                      │
│  └──────────────────────────────────────────┘                      │
│      │                                                              │
│  ├─→ 注册 → FAISS 索引 (IndexFlatIP)                               │
│  ├─→ ConnectionDetector → 自动交朋友                                │
│  └─→ 聚类 → 质心更新 → gravity_field 重算                           │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                          检索 · 五层流水线                           │
│                                                                     │
│  Query                                                              │
│    │                                                                │
│  ┌── Query Rewrite (HyDE, 可选) ───────────────────────┐           │
│  │  LLM生成假设答案 → 用假设答案检索 → 弥合语义鸿沟    │           │
│  └──────────────────────────────────────────────────────┘           │
│    │                                                                │
│  ┌── FAISS 粗搜 (Top-100) ────────────────────────────┐            │
│  │  探针注入球体空间，找到初始种子球体                │            │
│  └──────────────────────────────────────────────────────┘           │
│    │                                                                │
│  ┌── Activation Propagation ──────────────────────────┐             │
│  │  信号沿连接图 BFS 扩散 2 跳                       │             │
│  │  多路径叠加 → 发现隐式关联                         │             │
│  └──────────────────────────────────────────────────────┘           │
│    │                                                                │
│  ┌── Gravity Focus ──────────────────────────────────┐              │
│  │  场域质心对匹配球体施加引力增益                   │              │
│  │  query也在场域中 → 匹配球体 mass 提升             │              │
│  └──────────────────────────────────────────────────────┘           │
│    │                                                                │
│  ┌── Cross-encoder Rerank (可选) ────────────────────┐              │
│  │  Ollama逐个评分 (query, candidate) → [1,5]       │              │
│  └──────────────────────────────────────────────────────┘           │
│    │                                                                │
│  ┌── Diversity Sort (五层) ─────────────────────────┐              │
│  │  Layer 1: MMR (相关+多样平衡)                    │              │
│  │  Layer 2: 来源惩罚 (同源扣分)                     │              │
│  │  Layer 3: 场域加权 (匹配场域加分)                 │              │
│  │  Layer 4: 冗余惩罚 (簇内密集扣分)                 │              │
│  │  Layer 5: 连接惩罚 ── ★ v2 新增                  │              │
│  │            与已选球体有强连接的扣分               │              │
│  └──────────────────────────────────────────────────────┘           │
│    │                                                                │
│  ┌── Context Assembly + Generator ──────────────────┐               │
│  │  连续chunk合并 → 注入LLM → 生成回答              │              │
│  └──────────────────────────────────────────────────────┘           │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                          空间维护 · 后台                             │
│                                                                     │
│  ┌─ Calibrator: 按连接度更新 mass，按语义距离更新 diversity          │
│  ├─ Connection Decay: 定期衰减长期未激活的连接权重                   │
│  ├─ Cluster Refit: 新增球体触发全量重聚类                           │
│  └─ Gravity Rebuild: 质心变化时重算所有球体的 gravity_field         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心概念

### 3.1 球体 (Sphere)

一条知识的原子单元。每个球体有：

| 属性 | 类型 | 说明 |
|------|------|------|
| `id` | str | SHA256(text+source)[:12]，幂等 |
| `text` | str | 知识片段原文 |
| `vector` | ndarray | 768 维，nomic-embed-text 嵌入 |
| `mass` | float | **质量**，默认 1.0，按连接度校准 |
| `diversity` | float | **多样性**，默认 0.0，到簇质心的语义距离 |
| `cluster_id` | int | 所属聚簇编号 |
| `gravity_field` | dict | **引力场** `{场域名: 引力值}`，到每个质心的余弦 |
| `connections` | dict | **社交关系** `{球体ID: 连接权重}` |
| `term_weights` | dict | 关键词 TF 权重，用于术语引力检索 |
| `effective_mass` | float | `mass × (1 + 0.5 × diversity)` |

**effective_mass 的物理直觉：**
- 高 mass + 低 diversity = 簇内核心球体（重要且典型）
- 低 mass + 高 diversity = 边缘球体（可能连接不同簇）
- 高 mass + 高 diversity = 重要且稀缺（检索时优先选中）

### 3.2 聚簇 (Cluster)

k-means 聚类将语义相近的球体聚成星团。每个簇有：
- **质心** (centroid)：簇内所有球体向量的均值
- **球体列表**：属于该簇的所有球体
- **场域名**：自动命名 `簇0` ~ `簇N`

聚类在每次上传后自动触发。支持 Auto-K 检测（Silhouette Score）。

### 3.3 引力场 (Gravity Field)

每个球体到所有簇质心的余弦相似度，构成引力场。

引力场决定球体和场域的关系——
球体对某个场域的引力值越高，说明它越属于这个场域。

引力场在检索时被 Gravity Focus 模块使用：查询向量也计算到场域的亲和度，匹配场域的球体获得质量增益。

### 3.4 连接 (Connection)

v2 最核心的新概念。球体之间的显式关系网络。

连接有 5 种类型：

| 类型 | 权重 | 条件 |
|------|------|------|
| 同簇语义 | 0.6 | 同簇内余弦 Top-3 |
| 实体重叠 | 0.4 | 共享 ≥2 个实体 |
| 跨簇近邻 | 0.3 | 不同簇但余弦 > 0.80 |
| 时序相邻 | 0.25 | 同源文件相邻 chunk |
| 跨簇桥接 | 0.2 | 同文件名不同簇 |

连接是**双向的**、**带权的**、**稀疏的**（每个球体最多 50 个朋友）。

---

## 4. 入库流水线

### 4.1 TextRewriter — 文本净化

原始数据（尤其是会话记录）进入知识库前先清洗。

**全量重写（会话记录）：**
```
Step 1: 噪音过滤 (±0ms) — 正则去除时间戳、/thinking标记、纯标点行
Step 2: LLM 结构化 (±3s) — Ollama qwen2.5:7b 生成 JSON
         {title, summary, content, entities, sections}
Step 3: 实体规范化 (±0ms) — 统一名称（"小云"="深渊凝视者"="cloud"）
Step 4: 输出 CleanDocument
```

**轻量重写（技术笔记/论文）：**
只做实体提取+关键词提取，不调 LLM，保留原文。

### 4.2 结构感知切片

按 `source_type` 选择切片策略：

| 类型 | 模式 | 最大字符 | Overlap |
|------|------|---------|---------|
| 会话记录 | section | 1500 | 200 |
| 技术笔记 | markdown | 2500 | 200 |
| 其他 | recursive | 2000 | 300 |

section 模式使用 CleanDocument 的 `sections` 字段作为切分依据，每个 section 一个 chunk。过长的降级到 recursive。

### 4.3 嵌入与存储

```
文本 → nomic-embed-text (768维, L2归一化) → FAISS (IndexFlatIP)
                                              ↘ 
                                              SphereStore (JSON)
                                              ↙
                                            Registry (ID映射)
```

嵌入缓存在 `_vectors` 字典中，同一模型+同一文本的向量不重复计算。

---

## 5. 球体社交网络

### 5.1 自动交朋友

新球体入库时，ConnectionDetector 自动执行 5 条检测规则：

```python
# 伪代码：新球体进库时
def on_new_sphere(new_sphere):
    for existing_sphere in existing_spheres:
        weights = []
        # 同簇语义
        if same_cluster and cosine > 0.6:
            weights.append(0.6 * cosine)
        # 实体重叠
        elif shared_entities >= 2:
            weights.append(0.4 * overlap_ratio)
        # 跨簇近邻
        elif different_cluster and cosine > 0.80:
            weights.append(0.3 * cosine)
        # 时序相邻
        elif same_source_file_adjacent:
            weights.append(0.25)
        # 取最高权重
        weight = max(weights) if weights else 0
        if weight >= 0.1:
            connections[existing_sphere.id] = weight
    
    # 排序取 Top-50
    new_sphere.connections = top_k(connections, 50)
```

### 5.2 连接的生命周期

```
创建：入库时自动建立（首次迁移可对 2923 个球体全量构建）

维护：定期衰减 * 0.98，低于 0.05 的连接被修剪

衰减：长期不检索的连接逐渐淡化

消亡：当连接权重低于 prune_threshold → 自动删除
```

### 5.3 当前数据

当次迁移后（2026-07-15）：
- 2923 个球体，2752 条连接
- 平均每个球体 1.9 个连接
- 2883/2923 个球体有至少 1 个连接

> 连接密度可以通过调整 `config.py` 的 `embedding_threshold`（当前 0.80）或 `min_weight`（当前 0.10）来控制。

---

## 6. 检索 = 激活传播

### 6.1 三种模式

| 模式 | 路径 | 速度 | 适用场景 |
|------|------|------|---------|
| `simple` | FAISS → 直接排序 | 最快 | 过渡期兼容，简单问答 |
| `gravity` | FAISS → 激活传播 → 引力场 → 五层排序 | 默认 | 日常检索 |
| `deep` | 重写 → FAISS → 激活 → rerank → 引力 → 排序 | 最慢但最准 | 困难问题、多跳推理 |

### 6.2 激活传播算法

这是 v2 检索的核心创新。

```
输入: query 向量
输出: 激活的球体列表（按总激活值降序）

1. FAISS 搜索 → Top-100 种子球体
   每个种子获得初始激活值 = FAISS 余弦相似度

2. for hop in range(2):   # 最多传播 2 跳
       for 每个当前激活的球体:
           遍历它的 connections:
               传播信号 = 当前激活值 × 连接权重 × 衰减系数(0.5)
               目标球体收到信号，叠加到总激活值
   
3. 返回按总激活值降序的球体列表
```

**与 PageRank 的区别：**
- PageRank 是静态的，不依赖 query
- Activation Propagation 的探针从 query 注入，不同 query 激活不同子图

### 6.3 五层排序

| 层 | 名称 | 作用 | 数学 |
|----|------|------|------|
| 1 | MMR | 平衡相关性和多样性 | `λ·sim(q,d) - (1-λ)·max sim(d, selected)` |
| 2 | 来源惩罚 | 避免结果全来自同一文档 | 同源文件每多选一个扣 0.15×1.5^(n-1) |
| 3 | 场域加权 | 匹配查询场域的球体加分 | 亲和度 × 0.1 |
| 4 | 冗余惩罚 | 簇内密集区扣分 | 簇内平均相似度 × 0.05 |
| 5 | 连接惩罚 | 与已选球体有强连接的扣分 | 连接权重 × 0.1 |

第 5 层实现了「空间碰撞」——紧密连接的两个球体不会同时被选中，迫使结果分布在球体空间的不同区域。

### 6.4 Gravity Focus — 主动引力

```python
# 检测查询对哪些场域感兴趣
affinities = field_detector.detect(query_vector)

# 对每个候选球体：检查它的 gravity_field 是否与查询匹配
for sphere in candidates:
    match = 0
    for field in top_fields:
        match += sphere.gravity_field[field] * affinities[field]
    sphere.effective_mass *= (1 + 0.2 * match)
```

不做硬路由（不排除任何场域），只做引力增益——匹配场域的球体在排序中自然浮上来。

---

## 7. 内部路由工具

v2 新增四个工具，作为知识库的「空间导航器」。它们注册为 API 端点，生成器在需要时调用。

### 7.1 Navigate — 球体导航

```
GET /navigate/{sphere_id}?hops=2

从指定球体出发，沿连接图行走 n 跳。
返回路径上的所有球体 + 连接边。

用途：发现"这个概念的上下游是什么"
```

### 7.2 Explore — 聚簇展开

```
GET /explore/{cluster_id}?sort_by=mass&top_k=30

展开一个聚簇的所有内容。
返回簇内球体列表 + 排序 + 连接统计 + 场域分布。

用途：浏览"技术笔记"这个域里有什么
```

### 7.3 Trace — 会话时间线

```
GET /trace?source_file=2026-07-15.md

还原一个会话的完整时间线。
返回按时序排列的球体 + 实体出现时间轴。

用途：检索会话记录时组装完整上下文
```

### 7.4 Bridge — 路径发现

```
GET /bridge/{sphere_a}/{sphere_b}

双向 BFS 找两个球体之间的最短路径。
返回路径类型（直达/短链/长链/不连通）。

用途：发现"技术笔记A"和"会话记录B"的隐式关联
```

---

## 8. 与主流 RAG 的对应关系

重力知识库的设计不是对主流 RAG 技术的「替代」，而是「翻译」——将每个 SOTA 思想映射到球体空间的隐喻中。

| 主流技术 | 在重力空间中的翻译 | 差异点 |
|---------|------------------|--------|
| **Hybrid Search** (BM25+密集) | 不是两路并列检索。BM25 信号融入 `term_weights`，在 `keywords.py` 的术语引力匹配中使用 | 词汇匹配作为连接权重的一部分，而非独立检索通道 |
| **Cross-encoder Reranker** | 可选模块 `reranker.py`，Ollama 本地评分 1-5 | 不依赖 GPU，轻量化部署。配置可关 |
| **RAPTOR** (层次树) | k-means 簇质心 = 自然的高层节点。簇摘要将来可 LLM 生成 | 质心由聚类自动产生，不是递归生成 |
| **GraphRAG** (微软) | `connections` 表 + `entities` 标注 = 轻量图。`bridge.py` 做双向 BFS 路径发现 | 图在球体层不在实体层，更细粒度 |
| **Self-RAG** | 不训练反射 tokens。用 `gravity_focus` 做隐式置信度——检索结果场域分散 → 不可信 → 可触发二次检索 | 靠空间拓扑做置信度估计，不需要微调 |
| **HyDE** | `rewriter.py` 的 query rewrite 模式之一：生成假设答案 → 用假设答案检索 | 只在 deep 模式启用，Ollama 本地执行 |
| **CRAG** | 置信度评分机制尚未实现（v2 roadmap 中的 Phase 2） | — |

**独特价值（主流方案没有的）：**
1. **球体社交网络** — 连接图不是实体图，是知识片段之间的关系网
2. **激活传播** — 检索从单点匹配变成信号扩散，发现隐式关联
3. **五层排序** — MMR + 来源 + 场域 + 冗余 + 连接，工业级精细度
4. **内部路由工具** — 知识库不只是检索器，还是可探索的空间

---

## 9. 配置速查

所有配置集中在 `config.py`，新增 5 个配置类：

```python
# 重写配置
rewriter = RewriterConfig(
    enabled=True,
    llm_model="qwen2.5:7b",     # Ollama 本地模型
    full_strategies=["会话记录", "其他", ""],
)

# 连接检测
connection = ConnectionConfig(
    same_cluster_weight=0.6,
    entity_threshold=2,
    embedding_threshold=0.80,
    max_connections_per_sphere=50,
)

# 激活传播
activation = ActivationConfig(
    max_hops=2,
    decay_factor=0.5,
    min_propagated=0.02,
)

# 重排序
reranker = RerankerConfig(
    enabled=True,
    method="ollama",
    candidate_count=50,
)

# 质量校准
calibrator = CalibratorConfig(
    mass_base=1.0,
    mass_connection_factor=0.3,
    diversity_effective_factor=0.5,
)
```

---

## 附录

### A. 文件拓扑

```
local-kb/
├── config.py                        # 配置中心
├── pipeline/
│   ├── chunker.py                   # 结构感知切片
│   ├── rewriter.py    ← ★ v2       # 文本重写器
│   ├── connections.py ← ★ v2       # 连接检测器
│   ├── embedder.py                  # Ollama 嵌入
│   └── keywords.py                  # 关键词提取
├── storage/
│   ├── sphere_store.py              # 球体库 (JSON)
│   ├── faiss_store.py               # 向量索引 (FAISS)
│   ├── registry.py                  # ID 映射
│   ├── calibrator.py  ← ★ v2       # 质量多样性校准
│   └── wal.py                       # WAL 日志
├── retrieval/
│   ├── retriever.py                 # 检索编排器
│   ├── activation.py ← ★ v2        # 激活传播
│   ├── reranker.py   ← ★ v2        # 候选重排
│   ├── field_detector.py            # 场域检测
│   ├── diversity_sorter.py          # 多样性排序
│   ├── cluster_engine.py            # 聚类引擎
│   ├── session_manager.py           # 会话管理
│   └── tools/         ← ★ v2       # 内部路由工具
│       ├── navigate.py
│       ├── explore.py
│       ├── trace.py
│       └── bridge.py
├── api/
│   └── main.py                      # FastAPI 应用
├── pipeline/
│   └── generator.py                 # 答案生成器
└── scripts/
    └── migrate_v2.py  ← ★ v2       # 迁移脚本
```

### B. 数据流

```
入库                 检索
  │                  │
  rewriter           query
  │                  │
  chunker            embedder
  │                  │
  embedder           FAISS search (探针注入)
  │                  │
  sphere id          Activation Propagation (2跳扩散)
  │                  │
  ┌── sphere_store   Gravity Focus (场域引力)
  ├── registry       │
  ├── faiss_index    Reranker (可选)
  └── connections    │
                     Diversity Sort (五层)
                     │
                     Context Assembly
                     │
                     Generator → 回答
```

### C. API 端点

```
GET   /                      前端界面
GET   /status                知识库状态
POST  /upload                上传文件
POST  /query                 检索 {query, mode, top_k}
POST  /ask                   问答（检索+生成）
POST  /rebuild               全量重建索引
POST  /rewrite     ← ★ v2   手动文本重写
POST  /rebuild-connections ★ 全量重建连接
POST  /calibrate   ← ★ v2   触发 mass/diversity 校准
GET   /navigate/*  ← ★ v2   球体导航
GET   /explore/*   ← ★ v2   聚簇展开
GET   /trace       ← ★ v2   会话时间线
GET   /bridge/*/*  ← ★ v2   球体路径发现
GET   /backends              ℹ️ 生成后端列表
```

---

> **重力知识库**不是另一个 RAG 框架。它是一个尝试——把知识管理的核心隐喻从「图书馆」换成**「宇宙」**。
>
> 球体不是书架上挨着的书。球体是悬浮在空间中的星体，有自己的质量、引力场和社交关系。检索不是找书，是感受引力。
