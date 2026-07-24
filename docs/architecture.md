# gravity-local-kb 架构快照

> 最后更新：2026-07-21 17:15  
> 本文件与 `docs/code-guide.md` 配合阅读：快照管架构关系，手册管文件细节

---

## 0. 核心概念

| 术语 | 说明 |
|------|------|
| **球体 (Sphere)** | 知识原子。每个 chunk 对应一个球体，含文本/向量/元数据/连接 |
| **簇 (Cluster)** | K-means 产出的球体分组。按话题方向（余弦相似度）聚合 |
| **场域 (Field)** | FieldDetector 从簇质心组织的上层分组。每个场域 = 一个簇的抽象标签 |
| **重力场 (Gravity Field)** | 球体在所有场域上的亲和度分布。用 `compute_gravity_field()` 算，结果形如 `{"簇0": 0.85, "簇3": 0.12}` |
| **激活传播 (Activation)** | 检索时从 FAISS 种子沿连接网络 BFS 扩散信号，按总激活值排序 |
| **Poincaré 范数 (norm)** | 球体在 Poincaré Ball 中的径向坐标 [0,1)，由社区结构推导。**当前永远=0.5**（norm_deriver 未接入） |
| **有效质量 (Effective Mass)** | 球体的综合影响力，由连接数 + 簇大小 + 层次深度 + 活跃度通过 SphereCalibrator 计算 |

---

## 1. 数据流总图

```
上传 ──→ 解析 ──→ 切片 ──→ 嵌入 ──→ 去重 ──→ 存储 ──→ [重建]
                                  │                      │
                                  │                      ├── 聚类 (K-means)
                                  │                      ├── 场同步 (FieldDetector)
                                  │                      ├── 连接构建 (Connections)
                                  │                      ├── 范数推导 (norm_deriver ⚠️ 未接入)
                                  │                      ├── 角色表 (role_table)
                                  │                      └── 持久化 (save)
                                  │
                                  └── FAISS IndexFlatIP

检索 ──→ [多路召回] (retriever.py)
         ├── FAISS 粗搜 (Top-100)
         ├── 一级球体展开 (概念→子球体)
         ├── 激活传播 (沿连接图扩散)
         ├── 角色共现扩展 (role_expander)
         ├── 场域聚焦 (filter by field)
         ├── 重排序 (reranker, 仅 deep 模式)
         └── 多样性排序 (5 层算法)
```

---

## 2. 模块清单

### 2.1 上传层 — `api/main.py`

**21 个端点，核心 5 个：**

| 端点 | 位置 | 说明 |
|------|------|------|
| `POST /upload` [✅ TRACE] | L1188 | 单文件上传。auto_rebuild=True 时调 KMeans |
| `POST /upload/batch` | L1483 | 批量上传，asyncio 并发处理文件 |
| `POST /query` [✅ TRACE] | L1757 | 检索，四种模式 |
| `GET /status` | L1163 | 系统状态 |
| `GET /` | L1153 | 前端页面 |

**次要端点（14 个）：** `/ask`, `/backends`, `/rebuild`, `/rewrite`, `/rebuild-connections`, `/rebuild-axon`, `/rebuild-hierarchy`, `/calibrate`, `/navigate/{id}`, `/explore/{id}`, `/trace`, `/bridge/{a}/{b}`, 3 个 exception handler。

**上传流程（单文件）：**
```
save tmp → parse_file() → chunk → embed_documents(chunks) → dedup → store
→ (if auto_rebuild) KMeans → field_sync → connections → role_table → save
```

---

### 2.2 嵌入层 — `pipeline/embedder.py`

- **模型：** nomic-embed-text（Ollama, 768 dim）
- **硬件：** GPU（实测 100% GPU, 26ms/向量 batch)）
- **输出：** L2 归一化 float32 向量
- **缓存：** LRU, 10000 条, 文本 hash → 向量
- **批处理：** 一次发最多 16 条给 Ollama

**调用链：**
```
embed_documents(texts)
  → _embed_batch(texts) → _call_api(batch)  # POST /api/embed
  → _normalize(vectors)
```

---

### 2.3 向量存储 — `storage/faiss_store.py`

- **索引类型：** `IndexFlatIP` + `IndexIDMap`（精确暴力搜索）
- **距离：** L2 归一化向量上的内积 = 余弦相似度
- **数据量：** < 100k，Flat 是最优解（精度 100%，速度足够）

**核心方法：**
```
add(vector, id)           → 存入 _vectors 字典 + FAISS
search(query_vec, k)      → 返回 (top-k ids, 距离, 向量本身)
build(vectors, ids)       → 重建整个索引
save() → .faiss 文件     → FAISS 自己的序列化格式
load() → 反序列化
```

---

### 2.4 球体存储 — `storage/sphere_store.py`

**Sphere 数据模型（17 个字段）：**
```python
id: str                  # SHA256(text+filename)[:12]
text: str                # chunk 原文
filename: str
source_type: str         # "技术笔记" / "其他"
cluster_id: int          # -1 = 未分配
poincare_norm: float     # [0,1)，当前永远 0.5
poincare_norm_source: str
level: int               # 1=顶层概念, 2=具体论述, 3=事实细节
parent_id: str / child_ids: list[str]   # 层次关系
effective_mass: float    # 综合影响力
gravity_field: dict      # {场域名: 亲和度}
created_at: float
active: bool
```

**存储：** JSONL，每行一个球体。`spheres.json`。

---

### 2.5 聚类引擎 — `retrieval/cluster_engine.py`

- **算法：** sklearn `KMeans(init='k-means++', algorithm='lloyd')`
- **距离：** **余弦空间**（L2 归一化 → 欧氏 = 余弦）
- **自动 K：** silhouette score，遍历 `[k, max(2, √n)]`
- **持久化：** `save()` → JSON (centroids + labels + n_iter + inertia)

**rebuild 调用链：**
```
vectors = [faiss_store._vectors[fid] for active spheres]
centroids, labels, scores = fit_predict(vectors)
for each sphere: cluster_id = labels[i]
field_detector.sync_from_clusters(centroids, label_map)
connections.detect_batch()  # 建同簇/跨簇连接
role_table.build_for_spheres(spheres)
save all
```

---

### 2.6 场检测器 — `retrieval/field_detector.py`

**核心机制：**
- `sync_from_clusters(centroids, label_map)`：导入 K-means 质心，每个簇 = 一个场域
- `detect(query_vector)`：计算查询向量与每个场域质心的余弦相似度 → `{簇名: 亲和度}`
- `compute_gravity_field(vector)`：一个球体在所有场域上的分布 → 写入 `sphere.gravity_field`

**场域（Field）的定义：** 其实就是簇的命名版本。每个簇被标记为"簇0"、"簇1"等。FieldDetector 不做跨簇合并。

---

### 2.7 范数推导 — `pipeline/norm_deriver.py` ⚠️ 死代码

**设计意图：** 从社区结构推导 Poincaré 范数：

```
簇大小 → 覆盖度（簇越大越靠近边界）
层次深度 → level_factor（顶层概念靠近球心）
连接数 → 连接密度系数
```

**现状：** 未被任何模块 import。所有球体 `poincare_norm = 0.5`。

---

### 2.8 连接构建 — `pipeline/connections.py`

**连接规则：**
- 同簇连接（最强）：`same_cluster_topk=3`，权重 `0.6 × (0.8 + 0.2 × cosine_sim)`
- 跨簇桥接（较弱）：取其他簇最佳匹配，权重 `0.2`

**数据结构：**
```python
_connections: dict[str, dict[str, float]]  
#  {sphere_id: {neighbor_id: weight, ...}, ...}
```

**连接类型（通过_axon_types追踪）：** semantic / structural / temporal，但类型仅用于分析，不参与权重计算。

---

### 2.9 Poincaré 检索 — `retrieval/poincare_search.py`

**距离公式：**
```
d(u,v) = arccosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
```

**流程：**
```
query → to_poincare_ball(query, norm=0.5)
candidates → to_poincare_ball(vectors, norms=poincare_norms)
batch_poincare_distance(q, candidates) → 排序 → Top-K
```

**⚠️ 当前所有 `poincare_norm = 0.5`，径向分量退化为常数。**

---

### 2.10 检索编排 — `retrieval/retriever.py`

**四种模式：**

| 模式 | 路径 | 速度 | 精度 |
|------|------|------|------|
| `simple` | FAISS → 排序 | 最快 | 最低 |
| `gravity` | FAISS → 场域检测 → 多样性排序 | 快 | 中 |
| `deep` | 查询改写 → FAISS → 激活传播 → 重排序 → 多样性排序 | 慢 | 最高 |
| `poincare` | Poincaré 距离 → 场域 → 多样性排序 | 中 | 中高 |

**gravity 模式的完整步骤：**
```
1. 向量化 (embed_query)
2. 场域检测 (detect → field_affinities)
3. FAISS 粗搜 (Top-100)
4. 一级球体展开 (level==1 → expand children)
5. 激活传播 (propagate along connections)
6. 角色共现扩展 (role_expander.expand)
7. 场域聚焦 (filter by field_focus if set)
8. 多样性排序 (5-layer DiversitySorter)
9. 返回 Top-5
```

---

### 2.11 角色表 — `pipeline/role_table.py`

**数据模型：** 每一行 = 一个球体中提取的一个定中短语实体，记录：
- `sphere_id`：出现在哪个球体
- `phrase`：完整定中短语
- `head`：中心语（裸名词）
- `attributive`：定语

**三种桥接（检索时用）：**
| 类型 | 含义 | 置信度 |
|------|------|--------|
| `shared_phrase` | 完整短语跨球体出现 | 0.85 |
| `shared_head` | 相同裸名词作中心语 | 0.60 |
| `shared_attributive` | 相同定语修饰不同名词 | 0.40 |

**注册流程：**
```
upload → chunk → role_table.register_text(sphere_id, chunk_text)
  → AttrHeadExtractor.parse(chunk_text) → 提取定中短语
  → 注册 phrase/head/attributive 到索引
```

---

### 2.12 层次结构 — `pipeline/hierarchy.py`

- 一级：顶层概念（抽象级别最高）
- 二级：具体论述（一级的子节点）
- 三级：事实细节（二级内部用小 KMeans 再分）

**`_cluster_internals()`：** 对二级球体的子句（`child_ids`）做小 KMeans，每个子簇升为三级。阈值 `min_internal_cluster=4`：至少 4 句才拆。

---

### 2.13 其他模块

| 模块 | 大小 | 角色 |
|------|------|------|
| `pipeline/parser.py` | 7 KB | 文件解析器（PDF/DOCX/MD/TXT → 纯文本 + 元数据） |
| `pipeline/chunker.py` | 13 KB | 切片器（递归 / Markdown / 分段三策略，有降级链） |
| `pipeline/keywords.py` | 5 KB | 无依赖关键词提取（TF + 模式匹配，不调 jieba） |
| `pipeline/attr_head_extractor.py` | 21 KB | 定中短语提取（依存句法 + 字典 + 复合词拆解三方案） |
| `pipeline/rewriter.py` | 15 KB | 查询改写（调本地 LLM，仅 deep 模式启用） |
| `pipeline/generator.py` | 10 KB | 答案生成器（ollama / deepseek / agent 三后端） |
| `pipeline/role_extractor_v2.py` | 3 KB | 角色提取备选方案（spaCy 依存句法兜底） |
| `retrieval/activation.py` | 8 KB | 激活传播引擎（BFS 沿连接图扩散信号，跟 query 相关） |
| `retrieval/diversity_sorter.py` | 10 KB | 五层多样性排序（MMR + 来源惩罚 + 场域加权 + 簇冗余惩罚 + 连接密度惩罚） |
| `retrieval/reranker.py` | 7 KB | 轻量重排序（调本地 LLM 逐条打分，仅 deep 模式） |
| `retrieval/role_expander.py` | 5 KB | 角色共现扩展器（FAISS 命中 → role_table 查关联 → 扩展候选池） |
| `retrieval/session_manager.py` | 4 KB | 会话跟踪（场域聚焦 + exclude_ids，重启即丢） |
| `retrieval/tools/navigate.py` | 2 KB | 连接图导航（从球体出发 N 跳） |
| `retrieval/tools/explore.py` | 3 KB | 簇展开（按 mass/diversity/时间排序） |
| `retrieval/tools/trace.py` | 3 KB | 会话时间线还原 |
| `retrieval/tools/bridge.py` | 4 KB | 两点间 BFS 最短路径 |
| `storage/registry.py` | 6 KB | FAISS int64 ↔ sphere_id string 双向映射 |
| `storage/wal.py` | 11 KB | 预写日志（防崩溃：先写日志再写数据，崩溃后恢复） |
| `storage/calibrator.py` | 7 KB | mass/diversity 校准（连接数 + 簇大小 + 层次 + 活跃度 → 综合打分） |
| `config.py` | 12 KB | 全局配置（18 个配置块） |
| `kb_ctl.py` | 7 KB | 运维工具（start/stop/restart/status） |

---

## 3. 距离体系

```
原始嵌入 (nomic-embed-text, 768-dim)
        │
        ▼ L2 归一化
        │
        ├── 余弦距离 (K-means 聚类 + FAISS 检索)
        │
        └── Poincaré Ball 映射
                │
                ▼ arccosh 测地线距离 (Poincaré 检索)
```

**关键理解：** 余弦聚类产社区 → 社区产 radial (norm_deriver) → radial 供 Poincaré 使用。两层解耦但互补。当前 norm_deriver 未接入，radial=常数。

---

## 4. 已知问题

| # | 问题 | 状态 | 附注 |
|---|------|------|------|
| 1 | Upload 后全量重建 ~55s（实测 8343 向量） | ✅ auto_rebuild=False + batch upload 已可用 | |
| 2 | norm_deriver.py 未被 import | ⚠️ 保留代码，待接入 | Poincaré norm 永远 = 0.5 |
| 3 | 三条 rebuild 路径各自独立 | 🔧 计划统一成 `rebuild_spaces(mode)` | |
| 4 | main.py 2260 行 | 🔧 计划拆出 upload.py + rebuild.py | |
| 5 | `/rebuild` 端点逐条重嵌入 | 🔧 统一时修复 | 应从 FAISS 缓存恢复 |
| 6 | Upload timings 返回空 `{}` | ❓ 疑似 debug 代码问题 | |

---

## 5. 如何更新

1. 直接改对应章节，标记 `[2026-07-21]` 和修改人
2. 代码改动后在 `FILE_MAP.md` 更新状态
3. 本文件 + `docs/code-guide.md` 配合阅读
