# local-kb 代码全貌手册

> 用你能听懂的话解释每个文件在做什么
> 最后更新：2026-07-21 17:10
> 总览：47 个 Python 文件，四层架构

---

## ⚡ 一句话认识整个系统

**local-kb = 你的第二大脑。**  
你扔文件进去（日记、笔记、论文），它自动切碎、嵌入成向量、聚类成社区、建连接网络。  
以后你问它问题，它不是关键词匹配，而是**理解语义 + 沿连接网络扩散激活信号 + 多样性排序**后再回答。

---

## 目录

- [第一层：地基（配置 + 启动）](#第一层地基配置--启动)
- [第二层：加工流水线（上传 → 可搜索的知识）](#第二层加工流水线上传--可搜索的知识)
- [第三层：存储系统（知识怎么躺硬盘里）](#第三层存储系统知识怎么躺硬盘里)
- [第四层：检索系统（怎么把知识找回来）](#第四层检索系统怎么把知识找回来)
- [第五层：API 接口（跟外界交互的大门）](#第五层api-接口跟外界交互的大门)
- [附录：所有端点和它们的活路](#附录所有端点和它们的活路)

---

## 第一层：地基（配置 + 启动）

### `config.py`（12 KB）— 全局配置中心

整个系统的**开关面板**。包含 18 个配置块，你用填参数的方式控制所有行为：

```
paths        → 文件存哪
ollama       → 连接哪个模型
embedding    → 嵌入参数（batch size / 缓存大小）
chunker      → 切片大小
clustering   → 聚类设置
retrieval    → 检索参数
activation   → 激活传播开关
reranker     → 重排序配置
poincare     → 双曲空间参数
role         → 角色表配置
```

改行为**第一优先来这里翻开关**，不要改代码逻辑。

---

### `kb_ctl.py`（7 KB）— 服务控制器

四个命令：
```
start    → 启动 uvicorn 服务器
stop     → 杀进程 + 清理 PID 文件
restart  → 先停再起
status   → 查端口 / 活跃球体数 / 场域列表
```

不是代码逻辑的一部分，是运维工具。

---

## 第二层：加工流水线（上传 → 可搜索的知识）

把你扔进来的文件变成系统能理解的知识原子。

### `pipeline/parser.py`（7 KB）— 文件解析器

**输入**：PDF / DOCX / MD / TXT 文件  
**输出**：`ParseResult`（纯文本 + 元数据）  
**做的事**：把各种格式的文档统一吐成纯文本。PDF 用 PyMuPDF 提文字，DOCX 用 python-docx，MD 直接读。PDF 特殊处理：调 dpi 参数控制 OCR 质量。

**数据结构：**
```python
ParseResult:
  text: str          # 纯净文本
  file_type: str     # pdf/docx/md/txt
  metadata: dict     # 文件名、页数、Markdown 标题等
```

---

### `pipeline/chunker.py`（13 KB）— 文本切片器

**输入**：长文本  
**输出**：短文本列表（chunks）  
**做的事**：把一篇长文章切成一条条短文本。支持三种策略：

1. **递归字符切片**（默认）— 按段落切，太长再按句子切，还太长按固定字符数硬切。切完后相邻块有 overlap（重叠），防止在边界断掉语义。
2. **Markdown 标题切片** — 按 `##` `###` 标题结构切，一个标题下的内容是一块。
3. **分段切片** — 按文档已有的分段结构切（重写后的会话记录用）。

**降级链：** section → markdown → recursive → fixed（越往右越暴力）

---

### `pipeline/embedder.py`（8 KB）— 文本向量化引擎

**输入**：文本  
**输出**：768 维 float32 向量（L2 归一化后）  
**做的事**：调 Ollama 的 `nomic-embed-text` 模型，把文字翻译成机器能算的向量。

**关键设计：**
- 批量嵌入：一次最多 16 条文本一起发，Ollama 一次返回（实测 100 条 ~2.6s）
- LRU 缓存：文本 hash → 向量，最近 10000 条命中直接返回，不调 Ollama
- 归一化：L2 normalize，让向量的模长 = 1
- 查询嵌入和文档嵌入走同一模型

**调用链：**
```
embed_documents(["文本1", "文本2", ...])
  → _embed_batch(["文本1", ...])     # 分批次
    → _call_api(batch)               # POST localhost:11434/api/embed
  → _normalize(vectors)              # 每个向量除以其模长
```

**硬件状态**：nomic-embed-text (137M 参数) → 100% GPU → 26ms/向量

---

### `pipeline/keywords.py`（5 KB）— 关键词提取

**输入**：文本  
**输出**：`{"重力": 0.12, "空间": 0.08, ...}`  
**做的事**：从文本里抓关键词并算 TF 权重。不用 jieba（省一个依赖），靠模式匹配——连续 2+ 个汉字算一个中文词，连续 2+ 个字母算英文词。给 `retriever.py` 的 `match_term_gravity()` 提供 keywords，实现术语加权。

---

### `pipeline/attr_head_extractor.py`（21 KB）— 定中短语提取器

**输入**：文本  
**输出**：`[AttrHeadPair(full_phrase, head, attributive, ...), ...]`  
**做的事**：从中文里抠出「定语 + 中心语」结构的短语。比如「基于深度学习的跨模态图像分割方法」→ `full_phrase="跨模态图像分割方法"`, `head="方法"`, `attributive="跨模态"`。

**三种提取策略：**
1. **依存句法**（主方案）— spaCy 做语法分析，找 nsubj/dobj 关系 → 兜底正则
2. **字典匹配** — 常见定中结构模式
3. **复合词拆解** — 纯规则拆长词

**谁用它：** `role_table.py` 注册定中短语时需要，batch upload 时也用来提取文档级术语。

---

### `pipeline/role_table.py`（30 KB）— 定中短语共现表

**整个项目最大的文件**，但概念不复杂。

**做的事：** 记录每个定中短语实体在哪些球体中充当完整短语 / 中心语 / 定语。检索时拿它做**角色共现跳转**——命中一个球体 → 查它包含的实体 → 找共享相同中心语/定语的其他球体 → 扩展候选池。

**三种桥接（强度递减）：**
1. `shared_phrase` — 完整定中短语跨球体出现（最强，0.85）
2. `shared_head` — 相同裸名词作中心语（中等，0.6）
3. `shared_attributive` — 相同定语修饰不同名词（较弱，0.4）

**调用流程：**
```
upload → chunk → role_table.register_text(sphere_id, chunk_text)
                     ↓
                  解析定中短语 → 注册到索引
query  → retriever → role_expander.expand(hit_ids) 
                     ↓
                  查共现 → 扩展候选池
```

---

### `pipeline/role_extractor_v2.py`（3 KB）— 角色提取 v2（备选方案）

`role_table.py` 的辅助模块。提供 `HybridExtractor` 类，用 spaCy 做依存句法 + 正则兜底提取中文定中结构。`role_table.py` 内部调它。

---

### `pipeline/connections.py`（26 KB）— 球体连接网络

**做的事：** 给球体之间建「连接」。连接是带权重的有向边，权重 = 语义相似度 × 位置系数。

**连接规则：**
- **同簇连接**（最强）：同簇内取 Top-3 最相似球体，权重 `0.6 × (0.8 + 0.2 × cosine_sim)`
- **跨簇桥接**（较弱）：跨簇取最佳匹配，权重 `0.2`
- **批处理**：`detect_batch()` 遍历所有球体算连接，`save()` 持久化

**谁用它：** `ActivationPropagator` 在检索时沿连接图扩散信号。连接越多信号传得越远。

---

### `pipeline/hierarchy.py`（19 KB）— 层次结构构建器

**做的事：** 把球体组织成三层结构：
- **一级**（顶层概念）— 抽象级别最高的球体
- **二级**（具体论述）— 一级的子节点
- **三级**（事实细节）— 二级内部的再聚类

**关键逻辑：** `_cluster_internals()` 对二级球体的子句再做一次小规模 KMeans，每个子簇升为三级。

**触发时机：** startup 时自动跑，upload rebuild 时也跑。

---

### `pipeline/norm_deriver.py`（8 KB）— Poincaré 范数推导器 ⚠️

**当前状态：死代码（未被任何模块 import）**

**本来的用途：** 从社区结构（簇大小、角色、连接密度）推导每个球体的 Poincaré 范数。簇越大 → 覆盖度越高 → 范数越大 → 越靠近 Poincaré Ball 边界 → 越具体。

**为什么不工作：** 写好了 `NormDeriver.attach_community()` 和 `NormDeriver.derive_all()`，但 main.py 的 rebuild 路径里没调它。

**后果：** 所有球体的 `poincare_norm = 0.5`（默认值），Poincaré 检索时径向分量退化成常数。如果你不调 Poincaré 模式检索，这个不影响。如果你用 Poincaré 模式，检索精度有损失。

---

### `pipeline/rewriter.py`（15 KB）— 文本改写器

**做的事：** 调 Ollama LLM 对查询文本做清洗——去噪声、纠错、提取实体。只有 `mode="deep"` 的检索才启用。

**三个方法：**
- `rewrite(text)` → 去敏感词、纠错、摘要
- `clean_document(text)` → 从脏文本中提取干净内容
- `_call_llm(prompt)` → 调本地 Ollama

---

### `pipeline/generator.py`（10 KB）— 答案生成器

**做的事：** 调 LLM 根据检索结果生成最终答案。支持三种后端：
- `ollama` — 本地模型
- `deepseek` — DeepSeek API
- `agent` — 子代理模式

**`AnswerGenerator.generate(query, context_spheres)`:**
1. 把检索到的球体文本拼成 context
2. 打 prompt：基于以下知识回答问题
3. 调对应后端的 LLM
4. 返回 AnswerResult（text + sources + backend）

**谁调它：** `main.py` 的 `/ask` 端点。

---

## 第三层：存储系统（知识怎么躺硬盘里）

### `storage/sphere_store.py`（14 KB）— 球体库

Sphere = 知识原子。每个 chunk 对应一个 Sphere。

**Sphere 数据结构：**
```python
Sphere:
  id: str              # SHA256(text+filename)[:12] → 确定性能保证同一内容 id 不变
  text: str            # chunk 原文
  filename: str        # 来源文件
  source_type: str     # "技术笔记" / "其他"
  cluster_id: int      # 属于哪个簇（-1 = 未分配）
  poincare_norm: float # 双曲空间径向坐标（默认 0.5）
  level: int           # 层次等级（1=顶层概念，2=具体论述，3=事实细节）
  parent_id: str       # 父级球体
  child_ids: list[str] # 子级球体列表
  effective_mass: float  # 综合影响力
  gravity_field: dict  # {场域名: 亲和度}
  created_at: float    # 创建时间戳
  active: bool         # 是否活跃（删除 = 设为 False）
```

**存储方式：** JSON 序列化，每行一个球体。`save()` 写 `spheres.json`，`load()` 读回来。

---

### `storage/faiss_store.py`（11 KB）— FAISS 向量索引

**FAISS = Facebook 的向量搜索引擎。** 不关心具体内容，只管「给定查询向量，返回最近的 100 个向量」。

**我们用的索引类型：** `IndexFlatIP`（暴力搜索 × 内积距离）。因为数据量在千级，Flat 就是最优解——不建树不做图，一个一个比，精确不丢。

**内部原理：**
```
add(vector, id)    → 向量存入 _vectors 字典 + FAISS 索引
search(query, k)   → FAISS 返回 (top-k 的 id, 余弦距离, 向量本身)
build(vectors, ids) → 重建整个索引（用于 /rebuild 端点）
save() / load()    → 持久化
```

**为什么用 IndexFlatIP 而不是余弦：** 向量已经是 L2 归一化的了，IndexFlatIP（内积）= 余弦相似度。省一步计算。

---

### `storage/registry.py`（6 KB）— 双向 ID 映射

**解决的问题：** FAISS 用 int64（数字 ID），系统用 string（`sha256[:12]` 字符串 ID）。需要一个翻译官。

**数据结构：** 双向字典 `faiss_id ↔ sphere_id`

**流程：**
```
上传时： sphere_id → 调 resolve() → 拿到 int64 的 faiss_id → 存入 FAISS
检索时： FAISS 返回 int64 → 调 sphere_id(fid) → 拿到字符串 → 查 sphere_store
```

---

### `storage/wal.py`（11 KB）— 预写日志（防崩溃）

**解决的问题：** 上传过程中如果断电/崩溃，怎么保证数据不丢一半。

**流程：**
```
① 内存里处理完所有步骤（解析→切片→嵌入→建球体）
② 写一个 WAL 文件（"已就绪，待写入"）
③ 把全部数据写入磁盘
④ 标记 WAL 为"已完成"
⑤ 删除 WAL 文件
```

启动时扫描 WAL 目录 → 找到"已就绪"的条目 → 要么恢复写入，要么清理残留。

---

### `storage/calibrator.py`（7 KB）— 质量/多样性校准器

**做的事：** 周期性扫描所有球体，重新计算 `effective_mass`（综合影响力）和 `diversity`（多样性分数）。

**mass 的计算因素：**
- 连接数（越多越重要）
- 簇大小（大簇的球体 mass 更高）
- 层次深度（顶层概念 mass 更高）
- 活跃度

**谁调它：** `/calibrate` 端点，或者 startup 时。

---

## 第四层：检索系统（怎么把知识找回来）

### `retrieval/retriever.py`（27 KB）— 检索编排器

**这是整个系统的核心大脑。** 所有检索模块在这里组合成完整流水线。

**四种检索模式：**

| 模式 | 路径 | 速度 | 精度 |
|------|------|------|------|
| `simple` | FAISS → 排序 → 返回 | 最快 | 最低 |
| `gravity` | FAISS → 场域检测 → 多样性排序 → 返回 | 快 | 中 |
| `deep` | 查询改写 → FAISS → 激活传播 → 重排序 → 多样性排序 | 慢 | 最高 |
| `poincare` | Poincaré 双曲距离 → 场域 → 多样性排序 | 中 | 中高 |

**完整检索流程（gravity 模式为例）：**
```
用户提问
  → 向量化（embed_query）
  → 场域检测（这个查询属于哪个知识领域）
  → FAISS 粗搜（Top-100）
  → 一级球体展开（命中顶层概念 → 展开到子球体）
  → 激活传播（沿连接网络扩散信号）
  → 角色共现扩展（命中球体的定中短语 → 找关联球体）
  → 场域聚焦（只保留匹配场域的）
  → 多样性排序（去冗余 + 平衡覆盖度）
  → 输出 Top-5
```

---

### `retrieval/cluster_engine.py`（13 KB）— 聚类引擎

**做的事：** 对所有球体的向量做 K-means 聚类，分成 5~18 个簇（根据数据量自动调 K）。

**关键设计：**
- 对 L2 归一化后的向量做欧氏距离 K-means → 等价于余弦距离聚类
- 自动检测 K：遍历 K 值选 silhouette score 最高的
- k-means++ 初始化（比随机初始化更稳定）
- 质心也 L2 归一化一次（保证余弦一致性）

**持久化：** `save()` 存质心坐标 + labels + n_iter + inertia 到 JSON。启动时 `load()` 恢复。

**谁调它：** rebuild 路径（每次 auto_rebuild=True 的上传后，或手动调 /rebuild）

---

### `retrieval/field_detector.py`（13 KB）— 场域检测器

**做的事：** 把 K-means 的簇质心进一步组织成「场域」。一个场域 = 一个或多个相关簇的组合。

**核心方法：**
- `sync_from_clusters(centroids, label_map)` — 从 K-means 质心同步场域
- `detect(query_vector)` — 判断查询属于哪个场域，返回场域亲和度字典
- `compute_gravity_field(vector)` — 计算一个球体在所有场域上的分布
- `field_count` — 当前场域数量

---

### `retrieval/poincare_search.py`（12 KB）— 双曲空间检索

**做的事：** 在 Poincaré Ball 里算测地线距离，而不是余弦距离。

**Poincaré 距离公式**（你上次看的）：  
```
d(u,v) = arccosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
```

**流程：**
```
原始向量 → to_poincare_ball(query/norms) → 映射到双曲球
           ↓
          batch_poincare_distance(query, candidates)
           ↓
          返回距离最小的 Top-K
```

**关键参数：**
- `query_norm = 0.5`（默认中性抽象度）— 查询的径向位置
- 每个候选球体的 `poincare_norm` 决定它的径向位置
- 当前所有 `poincare_norm = 0.5`（因为 norm_deriver 没接上）

---

### `retrieval/activation.py`（8 KB）— 激活传播引擎

**核心思想：** 检索不是「找最近向量」，而是「探针注入球体空间，沿连接图扩散信号」。

**比喻：** FAISS 找到的种子球体是「点燃的火把」→ 激活传播让火沿着连接网络蔓延 → 离火把越近的球体越亮 → 按亮度（激活值）排序。

**参数控制：**
- `max_hops` — 传播跳数（默认 2，传太远会稀释掉）
- `decay` — 每跳衰减率（默认 0.5，跳一次信号减半）
- `activation_threshold` — 低于这个值的信号被忽略

**与 PageRank 的区别：** PageRank 跟查询无关，激活传播是查询驱动的——不同查询点燃不同种子，激活不同子图。

---

### `retrieval/diversity_sorter.py`（10 KB）— 多样性排序器

**做的事：** 从 FAISS 的 Top-100 里选出 Top-5，但不是简单取最相似的，而是**相关性 + 多样性**平衡。

**五层算法（从下到上）：**

| 层 | 做什么 | 效果 |
|----|--------|------|
| 1. MMR | 行业标准算法：选了 A 之后，相似于 A 的 B 被扣分 | 去重 |
| 2. 来源惩罚 | 同一文件的多块切片先选一块，再选第二块受惩罚 | 不垄断 |
| 3. 场域加权 | 匹配查询场域的球体加分 | 更相关 |
| 4. 簇冗余惩罚 | 同一簇内的球体间相似度高 → 扣分 | 跨簇覆盖 |
| 5. 连接密度惩罚 | 与已选球体有强连接的 → 信息冗余 → 扣分 | 空间碰撞 |

---

### `retrieval/reranker.py`（7 KB）— 轻量重排序

**做的事：** 对候选球体再做一次精细化评分。用本地 LLM (Ollama) 逐条问「这段文本跟查询的相关度打几分？」。慢但准。

仅 `mode="deep"` 时启用，配置可关。

---

### `retrieval/role_expander.py`（5 KB）— 角色共现扩展器

**做的事：** 检索到一批球体后，查 `role_table` 找它们的定中短语共现球体，加到候选池里。

**实例：** 查询命中了一个讲「跨模态图像分割方法」的球体 → role_expander 查到还有另一个球体在讲「基于深度学习的图像分割方法」（共享 "方法" 作中心语）→ 也加进候选池。

---

### `retrieval/session_manager.py`（4 KB）— 会话管理器

**做的事：** 跟踪多轮对话的上下文。记录：
- 当前聚焦的场域
- 已经返回过的球体 ID（防止重复）

重启即丢，不做持久化。

---

### `retrieval/tools/navigate.py`（2 KB）— 球体导航

沿连接图从指定球体出发走 N 跳，返回路径上的所有球体。用于探索知识之间的关联路径。

### `retrieval/tools/explore.py`（3 KB）— 簇展开

展开一个簇的全部内容，按质量/多样性/时间排序。

### `retrieval/tools/trace.py`（3 KB）— 会话时间线

还原一个文件的会话时间线——谁在什么时候说了什么，球体之间的对话链条。

### `retrieval/tools/bridge.py`（4 KB）— 球体路径发现

找两个不相邻的球体之间的最短连接路径（BFS）。类似于知识图谱的「两点之间的最短路径」。

---

## 第五层：API 接口（跟外界交互的大门）

### `api/main.py`（86 KB，2260 行）— 完整的 API 服务

FastAPI 应用。所有端点都在这个文件里，当前太臃肿，计划拆成多个文件。

#### 端点全表（21 个）

见附录。

#### 核心数据流

```
上传 → POST /upload
  解析(pipeline/parser) → 切片(pipeline/chunker) 
  → 嵌入(pipeline/embedder) → 去重 
  → 存 sphere_store + faiss_store + registry 
  → (如果 auto_rebuild) 聚类 + 场同步 + 连接 + 层次 + 持久化

检索 → POST /query
  向量化 → 场域检测 → FAISS 粗搜 
  → 激活传播 → 角色扩展 → 场域聚焦 
  → 多样性排序 → 返回 Top-5

问答 → POST /ask
  /query 的路径 + 把结果喂给 LLM 生成答案
```

---

## 附录：所有端点和它们的活路

### 核心端点（日常用）

| 端点 | 路径 | 做的事 |
|------|------|--------|
| `POST /upload` | `api/main.py:1188` | 单文件上传，auto_rebuild 可选触发聚类 |
| `POST /upload/batch` | `api/main.py:1483` | 批量上传，并发处理+可选一次重建 |
| `POST /query` | `api/main.py:1757` | 检索，四种模式可选 |
| `GET /status` | `api/main.py:1163` | 服务状态+球体数+场域数 |
| `GET /` | `api/main.py:1153` | 前端界面 |

### 次要端点（探索/调试用）

| 端点 | 路径 | 做的事 |
|------|------|--------|
| `POST /ask` | L1844 | /query + LLM 生成答案 |
| `GET /backends` | L1925 | 列出可用 LLM 后端 |
| `POST /rebuild` | L1938 | 从持久化重建索引（⚠️ 超慢，逐条重嵌入） |
| `POST /rebuild-connections` | L2031 | 全量重建连接网络 |
| `POST /rebuild-axon` | L2049 | 仅重建因果链连接 |
| `POST /rebuild-hierarchy` | L2070 | 重建层级结构 |
| `POST /calibrate` | L2081 | 校准 mass/diversity |
| `GET /navigate/{id}` | L2103 | 从球体出发导航连接图 |
| `GET /explore/{id}` | L2120 | 展开一个簇 |
| `GET /trace` | L2137 | 还原会话时间线 |
| `GET /bridge/{a}/{b}` | L2153 | 找两点间路径 |
| `POST /rewrite` | L2008 | 文本清洗 |

---

## 已知问题列表

| # | 问题 | 影响 | 计划 |
|---|------|------|------|
| 1 | norm_deriver.py 未被任何模块 import | Poincaré norm 永远是 0.5 | 接入管道 |
| 2 | `/rebuild` 端点逐条重嵌入（n+1 次 API） | 10000 球体 = 10000 次 Ollama 调用 | 改成从 FAISS 缓存恢复 |
| 3 | 三条 rebuild 路径抄了三遍 | 维护噩梦 | 统一成一个 |
| 4 | main.py 2260 行 | 改不动 | 拆成 api/upload.py + api/rebuild.py |
| 5 | `/upload` 返回的 timings 为空 `{}` | 缺少性能仪表盘 | 调试修复 |
| 6 | 记忆索引损坏 | 我跨会话推理受阻 | 修 memory index |

---

> 问自己：理解完这些，你就知道整个 local-kb 怎么造出来的了。每个文件做什么、谁调谁、数据怎么流——全在这里。不懂的直接问我，不用看代码。
