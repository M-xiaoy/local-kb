# 重力知识库 · 施工指南

> 这不是文档，是地图。
> 回答「代码在哪」「怎么改」「为什么这么写」三个问题。

---

## 一、总览：三条数据流

整个系统只有三条路，记清楚就好：

```
    入库流                      检索流                      启动流
     ──────                      ──────                      ──────
  file                          query                     disk
    │                             │                          │
  parser.py                    embedder.py                 load()
    │                             │                          │
  chunker.py         ┌──────── faiss_store ────── registry ─┤
    │                │            │                    │     │
  embedder.py ───────┤     diversity_sorter          │     │
    │                │            │                    │     │
  sphere_store ◄─────┤       retriever ────► result    │     │
    │                │                                    │
  registry ──────────┤                                    │
    │                │                                    │
  faiss_store ───────┘                                    │
    │                                                     │
  field_detector ◄────────────────────────────────────────┘
```

**每层的知识边界（最重要的设计原则）：**

| 模块 | 知道的事 | 不知道的事 |
|------|----------|------------|
| `faiss_store` | float32 向量、int64 ID | 文本、场域、来源文件 |
| `registry` | faiss_id ↔ sphere_id | 向量内容、元数据 |
| `sphere_store` | 文本、场域、质量、连接表 | 向量、FAISS |
| `field_detector` | 场域质心、向量空间 | 具体文档内容 |
| `diversity_sorter` | 向量、得分、来源、场域 | 谁是最终消费者 |

**修改原则：** 跨层传递数据只能经过模块的公开接口，不直接读内部属性（如 `faiss_store._vectors`），否则哪天重构内部实现时耦合会全断。

---

## 二、文件地图

```
local-kb/
│
├── config.py                   ← 所有配置参数集中管理
├── requirements.txt            ← 依赖声明
│
├── storage/                    ← 持久层 · 存什么 / 怎么存
│   ├── sphere_store.py         → 球体元数据（文本+场域+连接表）
│   ├── faiss_store.py          → 稠密向量索引（FAISS）
│   └── registry.py             → FAISS ID ↔ sphere_id 双向映射
│
├── pipeline/                   ← 加工层 · 文件→向量
│   ├── parser.py               → PDF/DOCX/MD/TXT → 纯文本
│   ├── chunker.py              → 长文本 → 切片列表
│   └── embedder.py             → 文本 → 归一化 float32 向量
│
├── retrieval/                  ← 检索层 · 向量→结果
│   ├── field_detector.py       → 查询向量 → 场域亲和度
│   ├── diversity_sorter.py     → MMR+来源+场域 三层排序
│   └── retriever.py            → 编排以上所有模块的完整流水线
│
└── api/
    └── main.py                 ← FastAPI 入口 · HTTP 端点
```

**三条路径的代码跳跃链：**

- **改入库逻辑：** `api/main.py:upload_file` → `pipeline/parser.py` → `pipeline/chunker.py` → `pipeline/embedder.py` → `storage/`
- **改检索逻辑：** `api/main.py:query` → `retrieval/retriever.py` → `retrieval/field_detector.py` + `retrieval/diversity_sorter.py` → `storage/`
- **改存储格式：** `storage/sphere_store.py` + `storage/registry.py` + `storage/faiss_store.py` → 注意版本号迁移

---

## 三、核心数据类型（数据流通用的形状）

```
           ┌─ string  ←─── query
           │
      embedder.py
           │
           ▼
    np.ndarray, shape=(768,)    ←─── query_vector
    np.ndarray, shape=(n,768)   ←─── document_vectors (L2归一化)
           │
           ▼
    faiss_store.search()
           │
           ├── faiss_ids:  np.ndarray (int64, shape=(k,))
           ├── distances:  np.ndarray (float32, shape=(k,))
           └── vectors:    np.ndarray (float32, shape=(k, dim))
           │
           ▼
    registry.sphere_id(fid) 🡒 str  ("32d3dbba7b63")
           │
           ▼
    sphere_store.get_many()
           │
           └── List[Sphere] ←─── 最终返回给用户
               Sphere {
                   id: str,        // "32d3dbba7b63"
                   text: str,      // 原文片段
                   source_file: str,
                   source_type: str,
                   mass: float,
                   connections: Dict[str, float],
               }
```

**在哪创建新类型：**
- 跨模块传递的结构化数据 → `retriever.py` 的 `RetrievalResult`（参考样式）
- 只在一个模块内部的 helper 类型 → 放在该模块顶部

---

## 四、关键设计决策（改之前必须理解）

### 4.1 FAISS ID 派生自 sphere_id，而非自增

`sphere_id = SHA256(text+source)[:12]` → `faiss_id = int(sphere_id, 16)`

**为什么：** 同一内容每次重建索引 ID 不变，registry 可恢复。LangChain 用自增 id，但在删除+重建场景下会漂移。

**如果要改：** 动 `registry.py` 的 `sphere_id_to_faiss_id` 和 `faiss_id_to_sphere_id`，需要同时更新持久化格式。

### 4.2 向量缓存（faiss_store._vectors）

FAISS IndexIDMap 不支持 reconstruct，所以我们在 `faiss_store.py` 里维护一份 `{faiss_id: vector}` 缓存。这是 LangChain 也用的标准做法。

**但这是耦合点：** `api/main.py` 的 `_collect_field_vectors` 直接读了 `faiss_store._vectors`。如果要改缓存实现，记得更新这个函数。

### 4.3 软删除 vs 真删除

`sphere_store.py` 用 `active=False` 做软删，但 FAISS 索引用 `remove_ids` 真删。

**不同步问题：** 软删后如果 FAISS 没删，检索会返回已删除的球体。当前流程是：`unregister` + `remove_ids` + `soft_delete` 三步走，在 `registry.py` 层面做一致性校验（`verify()` 方法）。

### 4.4 场域不是硬路由

**不是**「查询匹配技术笔记 → 只搜技术笔记」，而是「查询匹配技术笔记 0.85、小说创作 0.32 → 排序时加权」。

如果要改成硬路由：改 `retrieval/retriever.py` 的第 5 步（场域检测），在传给 diversity_sorter 之前过滤掉低亲和度的候选。

### 4.5 持久化三件套

三个独立文件，没有事务保证：

| 文件 | 格式 | 改动风险 |
|------|------|----------|
| `data/spheres/spheres.json` | JSON (version=1) | 低—人类可读，可手动编辑 |
| `data/spheres/registry.json` | JSON (version=1) | 中—与 spheres 严格对齐 |
| `data/index/faiss.index` | FAISS 二进制 | 高—不能手动编辑，坏了就得 rebuild |

**损坏恢复：** `POST /rebuild` 从 spheres.json 重建全部——前提是 spheres.json 没坏。

---

## 五、常见修改场景 · 从哪里切入

| 你想做什么 | 先读 / 改这里 | 注意 |
|-----------|--------------|------|
| 改切片大小/策略 | `config.py` 的 `ChunkerConfig` | mode 支持 recursive/markdown/fixed |
| 换嵌入模型 | `config.py` 的 `OllamaConfig` | 维度变了要改 `embed_dim`，否则 FAISS 炸 |
| 改排序算法 | `retrieval/diversity_sorter.py` | 核心是 `_mmr_score` + 三层叠加 |
| 加新场域选项 | `config.py` 的 `AVAILABLE_FIELDS` | 前端选了才有效，后端不校验 |
| 改返回数量 | `config.py` 的 `RetrievalConfig` 或 API 参数 | `faiss_top_k` 和 `final_top_k` 两处 |
| 加文件格式支持 | `pipeline/parser.py` 的 `DISPATCH` | 加解析函数 + 更新 `_normalize` |
| 改持久化路径 | `config.py` 的 `Paths` | 注意需要迁移旧数据 |
| 加缓存策略 | `pipeline/embedder.py` 的 `_cache` | 当前是 LRU-ish（超上限淘汰一半） |
| 调多样性参数 | `config.py` 的 `RetrievalConfig` | `diversity_weight`, `similarity_weight` |
| 改 API 结构 | `api/main.py` 的 Pydantic 模型 | 改了要同步更新前端 |
| 重建整个索引 | `POST /rebuild` 端点 | 会重跑全部 embedding（耗时） |

---

## 六、数据增长后的瓶颈预估

| 规模 | 瓶颈 | 现在的应对 |
|------|------|-----------|
| < 10k spheres | 无 | 全量 JSON 读写够用 |
| 10k-100k | FAISS 搜索速度 | IndexFlatIP 还行，切换 IndexIVFFlat 需改 `faiss_store.py` |
| 100k+ | JSON 加载速度 | 换 SQLite 存 metadata；FAISS 换 HNSW |
| 百万+ | 嵌入成本 | 加文档级别缓存（不重复嵌入同一文件的不同切片） |

**修改提示：** `faiss_store.py` 第 25 行注释已标注索引类型选择策略。

---

## 七、测试文件速查

| 文件 | 测试什么 | 如何运行 |
|------|---------|---------|
| `_verify_sphere.py` | SphereStore 增删查改+持久化 | `python _verify_sphere.py` |
| `_verify_registry.py` | 双向映射+幂等+孤儿检测 | `python _verify_registry.py` |
| `_verify_faiss.py` | FAISS 构建/搜索/删除/缓存 | 需 FAISS 和 Ollama |
| `_verify_field.py` | 场域检测+增量质心更新 | `python _verify_field.py` |
| `_verify_diversity.py` | 三层排序+MMR+来源+场域 | `python _verify_diversity.py` |
| `_verify_embed.py` | 嵌入+归一化+缓存 | 需 Ollama |
| `_verify.py` | 端到端：文件→入库→检索 | 需 Ollama+完整环境 |

---

## 八、我写这个项目时的思维模型

- **Sphere（球体）** = 一个文本切片 + 它的元数据 + 它在知识空间中的"质量"
- **Registry（注册表）** = 给每个球体一个 FAISS 认识的数字 ID
- **FAISS 索引** = 向量之间的"距离地图"——知道哪些球体内容相似
- **FieldDetector（场域检测）** = 判断一个查询"偏向"哪个领域
- **DiversitySorter（多样性排序）** = 不放回抽样 + 惩罚重复 + 奖励相关领域

**三句话记住全系统：**

> 文件进来，切成片，嵌入成向量，存进 FAISS。
> 查询来了，嵌入成向量，FAISS 捞出 Top-100，场域检测给分，多样性排序重排。
> 用户拿到 Top-5，每个有原文、来源、场域标签、得分。

---

## 九、修改索引（我接下来可以快速定位）

```python
# === 如果我要从文件中搜特定文本或关键词 ===
# 在这几个地方 grep：sphere_id, faiss_id, source_file, source_type

# === 如果我要改检索流程 ===
# retrieval/retriever.py → _retrieve() 的 1-7 步

# === 如果我想知道一个模块被谁调用了 ===
# 用 grep -rn "SphereStore" .   # 或者你手动翻 import
# 核心依赖链：
#   retriever.py → faiss_store, registry, sphere_store, field_detector, diversity_sorter
#   api/main.py  → 以上全部（通过 AppState 持有）

# === 如果修改了字段/结构，需要更新序列化 ===
# sphere_store.py: _sphere_to_dict / _dict_to_sphere
# registry.py:    save / load（mapping 列表格式）
# field_detector.py: get_state / set_state
```

---

## 十、2026-07-10 更新记录

| 改动 | 文件 | 说明 |
|------|------|------|
| 错误响应结构化 | `api/main.py` | 统一 JSON 错误格式 + request_id |
| 向量缓存持久化 | `storage/faiss_store.py` + `config.py` | 新增 `faiss_cache.npz`，启动时加载 |
| 预过滤跳过嵌入 | `api/main.py` upload 函数 | 嵌入前查 sphere_store，缓存热时跳过 |
| 服务控制脚本 | `kb_ctl.py` | start/stop/restart/status + 自动恢复 |
| CLI | `kb.py`（workspace 根目录） | `kb.py query`, `kb.py upload`, `kb.py status` |
| 配置 | `config.py` | 新增 `faiss_cache` 路径 |

*最后更新：2026-07-10 · 小云*
*如果你发现这份指南和代码对不上，先更新代码再更新指南。*
