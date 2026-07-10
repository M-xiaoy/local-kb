# 重力知识库：一个从零搭建的个人本地 RAG 系统

> **定位：** 不是 LangChain 教程，是一个大二学生在暑假用 Python + FAISS + Ollama 手写的知识检索工具。
> 
> **代码：** [M-xiaoy/gravity-space-architecture](https://github.com/M-xiaoy/gravity-space-architecture)
> **作者：** 刘存帅 · 西北大学 大二

---

## 为什么自己写

市面上 RAG 框架很多——LangChain、LlamaIndex、ChromaDB——每一个都能在 10 分钟内搭出一个"能跑的"知识库。

但我的需求不一样：

1. **本地运行**——文档不上云，所有内容存在自己的电脑
2. **场域感知**——同一个知识库里既有技术笔记又有小说创作，检索时希望自动识别查询偏向哪个领域，但不完全排除另一个
3. **多样性排序**——Top-5 结果不要全部来自同一份 PDF，要尽量覆盖不同来源
4. **从零理解**——框架封装了太多细节，出了问题不知道怎么修

所以决定自己写。从切片到嵌入、从 FAISS 索引到 Restful API，每一层自己实现。

---

## 架构

```
文件 → parser → chunker → embedder → FAISS 索引
                                        ↓
查询 → embedder → FAISS 粗搜 → 场域检测 → 多样性排序 → 结果
```

### 存储层（storage/）

三件套，各有各的职责：

| 模块 | 存什么 | 不知道的事 |
|------|--------|-----------|
| `sphere_store` | 文本原文、场域标签、来源文件、连接表 | 向量、FAISS |
| `faiss_store` | float32 向量、ANN 索引 | 文本内容、metadata |
| `registry` | faiss_id ↔ sphere_id 双向映射 | 两者之外的任何事 |

**设计取舍：** FAISS IndexIDMap 不支持从索引还原向量——查到了 ID 但拿不到向量本身。因此单独维护了一份 `{faiss_id: vector}` 缓存，每次持久化时额外保存到 `.npz` 文件，启动时自动加载。这是 LangChain 也用的标准做法。

### 加工层（pipeline/）

- **parser：** PDF（pdfminer）、DOCX（python-docx）、Markdown、纯文本 → 统一文本
- **chunker：** 递归降级切片（段落 → 行 → 句子 → 词 → 字符），800 字符上限 + 100 字符 overlap
- **embedder：** 调 Ollama nomic-embed-text，Task Prefix（search_document/search_query），批量 + 缓存

### 检索层（retrieval/）

**最核心的差异在这层。**

#### 场域检测：不搞硬路由

大多数系统做"查询路由"时用 LLM 判断查询属于哪个域——要么是技术笔记，要么是小说创作，二选一。

我不用。场域检测器维护每个场域的质心（所有球体的均值向量），查询时计算查询向量与每个质心的余弦相似度，**输出亲和度分数**——不是归属判断。

```
查询"如何用小说写法写技术文章"
  → 技术笔记: 0.72
  → 小说创作: 0.58
  → 两个场域都进排序候选，带不同权重
```

质心用 Welford 在线均值算法增量更新——加一个球体不需要重算全部，O(1) 代价。

#### 多样性排序：三层打分

```
最终得分 = MMR相关度 - MMR冗余度 - 来源惩罚 + 场域加分
```

1. **MMR 基础**（Maximum Marginal Relevance）—— λ=0.5 均衡相关性与多样性
2. **来源惩罚**——同一份文档的多段切片，每多选一个惩罚 ×1.5 递增，避免 Top-5 全来自同一篇
3. **场域加分**——field_detector 输出的亲和度加权，不排除低匹配场域的结果，只是让高匹配的排更前

---

## 一些工程决策

### FAISS ID 派生自内容哈希

大多数实现用自增 ID。我用 `SHA256(text + source_file)[:12]` 作为 sphere_id，再转 int64 作为 FAISS ID。好处：**同一内容每次入库 ID 不变**，重建索引不需要重新对齐 registry。

### 预过滤避免重复嵌入

文件解析 + 切片后，先计算每个切片的 sphere_id，检查是否已存在。已存在的跳过嵌入，从向量缓存直接取。在同一 session 内重复上传同一份文档，**嵌入环节耗时从 9 秒降到 0**。

### 错误响应结构化

每个 API 错误返回 `request_id` + 异常类型 + 详细信息 + 修复建议。配合日志里的 `[req=xxx]` 标记，出问题直接 grep。

---

## 命令行

```bash
# 查看状态
python kb.py status

# 检索
python kb.py query "什么是预测编码" --top-k 3

# 灌入文档
python kb.py upload 技术笔记.md --type 技术笔记
```

底层自动管理 FastAPI 服务生命周期——端口被占就杀旧重启，服务不健康就自动恢复。

---

## 当前状态

- **218 个活动球体**，全部技术笔记（AI 速通笔记 + 重力空间架构文档）
- OLLama nomic-embed-text，768 维
- FastAPI + uvicorn 本地 8765 端口
- 灌入 7 份文档耗时 ~40 秒（预过滤后重复上传 0 嵌入成本）

---

## 下一步

这个项目的方向已经定了：**做个人化模型的存储层**。

接下来不是加功能，是让它成为一个每天在用的工具——集成到日常检索链路中，用出痛点再迭代。

---

*2026-07-10 · 小云记*
