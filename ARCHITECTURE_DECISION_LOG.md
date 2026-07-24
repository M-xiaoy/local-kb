# 架构决策日志 Architecture Decision Log

## 2026-07-24：检索架构硬化决策

### 背景
知识库 v0.6 包含 30+ 检索组件，其中超过一半效果未验证。初步评估（15 条查询 × 3 模式）显示：
- **Simple (cosine)**：recall@10 = 29.3%，6ms
- **Poincaré 全管线**：recall@10 = 17.3%，260ms（29x 更慢，recall 更低）
- **FAISS → Poincaré 重排**：recall@10 = **37.3%**，7ms

### 实验证据
_hyperbolic_test_A.py 在 26K 球体上的三路对比：

| 模式 | recall@10 | 延迟 | vs 余弦 |
|------|:-:|:-:|:-:|
| A: FAISS 余弦 | 29.3% | 6ms | — |
| B: FAISS→Poincaré 重排 | **37.3%** | 7ms | **+27%** (+8pp) |
| C: 纯 Poincaré 全量扫描 | 36.0% | 255ms | — |

B 胜 5 / 负 1 / 平 9。**Poincaré 测地线距离在不受管线污染时，稳定优于余弦排序。**

### 结论
1. **Poincaré 距离有价值，但只在最后一层有意义。** 旧管线（DiversitySorter / TermFusion / TractionReranker / GravityFocus）在 Poincaré 之后对排序做了 5 次改写，把信号稀释殆尽。
2. **FAISS 欧氏初召没有漏掉相关文档。** C 列（纯 Poincaré 全量扫描）和 B 列几乎一致，说明 bge-m3 + FAISS 已捕获所有相关候选。
3. **新架构锁定为：** `Embed (BGE-M3) → FAISS ANN (Top-50) → Poincaré Rerank (Top-10) → Output`

### 执行
- DiversitySorter / TractionReranker / ActivationPropagator / LocalReranker / RoleExpander / FieldDetector（gravity_focus）/ ClusterEngine → 物理移出 `retrieval/`，迁至 `analysis/deprecated_pipeline/`
- `retriever.py` 重写为单一检索路径（无多模式路由）
- `poincare_rerank.py` 作为纯函数重排模块（仅依赖 `poincare_search.batch_poincare_distance`）

### 遗留
- `analysis/deprecated_pipeline/` 中的组件代码完整保留，供离线分析和实验使用
- `hierarchy/`、`role_table/`、`connections/` 等离线管道仍可独立运行
- 评估指标的期望来源文件仍偏窄，实际精度可能高于测量的 37.3%

---

## 2026-07-24（续）：存储层与检索层的职责分离基线

### 背景

v0.6 架构中，7 个存储层维护组件（ClusterEngine, FieldDetector, ActivationPropagator 等）躺在 `retrieval/` 目录下并被 `Retriever.retrieve()` 顺序调用，表现得像检索流程的一部分。追溯早期设计意图，这些组件本应归属存储层维护，**不在检索**时实时执行。

### 根源：重力空间的定位错位

2026-07-09 重力空间 Phase 3+4 设计的初衷是：

> 将层级关系、连接图、质量场、场域标签存储为球体元数据 → 通过 Poincaré 映射感知双曲曲率 → 无需训练双曲嵌入模型

但实现过程中，
1. 存储层维护（增量重建、质量传播、场域聚类）写成了检索时执行
2. 检索层多了一个"跑维护组件"的前置阶段（260ms），而真正的排序只需要 1ms
3. 元数据被检索管线改写而非只读引用

### 决策

**存储层和检索层有严格的职责边界：**

| 层 | 做的事 | 时机 | 无状态？ |
|----|--------|------|---------|
| **存储层维护** | HierarchyGrower 建边 / ConnDetector 连边 / MassAssigner 传播质量 / SphereCalibrator 调半径 / FieldDetector 打场域标签 / ClusterEngine 社区检测 | 索引构建或增量维护（离线） | 否 |
| **检索层** | Embed → FAISS ANN → Poincaré rerank | 每次查询（在线） | 是 |

**非零曲率空间是构造出来的，不是训练出来的。**

曲率信息不在 bge-m3 的嵌入向量里（那是欧氏空间），而在球体之间的元数据关系里（层级树、连接图、质量分布）。检索时 Poincaré rerank 只做一件事：**读取存储层已经写好的元数据，翻译为双曲距离，影响排序。**

### 推论（待验证）

Poincaré rerank 的收益与存储层元数据质量正相关：
- ConnDetector 覆盖率 ↑ → Poincaré 的连接感知 ↑
- HierarchyGrower 层级深度 ↑ → Poincaré 的层次感知 ↑
- MassAssigner 质量传播覆盖率 ↑ → 半径缩放的 effective ↑

如果 ConnDetector 只有 50% 覆盖率、层级解构只有 0 个 Level-1 概念，那么 Poincaré 的 rerank 收益会受限于存储层元数据的质量——不是 Poincaré 本身的价值问题。

---

## 2026-07-24（续）：社区分级层数策略从严格限分改为宽松分级

### 背景

`HierarchyGrower` 的层级分配公式为：

```python
num_levels = max(2, int(N_spheres ** 0.33))
# 26K 球体 → 29 个层级 → 平均每个级别只覆盖 ~900 个节点
```

Label Propagation 将 26K 球体分为 ~29 个社区，但到了 Level-1 概念（树根/root-level）数量为 **0**——社区内部无边可连，全部坍缩为独立社区。

### 决策

**双曲空间的层次不需要严格均匀分布。**

Poincaré 球的特征是指数空间膨胀——距离原点越远，容纳的节点越多。这意味着：
- 靠近原点的节点（高层抽象）天然应该稀少
- 远离原点的节点（具体实例）天然应该密集
- 强制 `N_spheres^0.33` 均匀分桶违背了双曲空间的指数几何

**改为更宽松的规则：**
- 取消硬性的 1/3 分桶比例
- Level 阈值由 Label Propagation 的社区检测自然产出
- 大社区（社区直径大）自动获得更高的层级抽象
- 小社区（聚焦概念）自动获得更低的层级位置

### 状态
- 规则已确认，代码未修改（待下一个维护窗口）

---

## 2026-07-24（续）：RoleTable 需降级或替换

### 背景

`pipeline/role_table.py` 的 `RoleTable`：
- **内存占用**：~1.6GB（词频表 + 角色矩阵）
- **初始化耗时**：~8s（spaCy `zh_core_web_sm` 加载 + 角色矩阵重建）
- **实际使用**：检索流程中频繁被降级跳过（实验开关 `use_role_expansion=False`）
- **依赖链**：RoleExpander → RoleTable → spaCy NER + AttrHeadExtractor

### 决策

**1.6GB 只为一个降级走的组件保留是不合理的。**

不删除，但：
1. **不在检索链路中加载**。RoleTable 只在增量重建或角色关联维护时按需加载。
2. **AppState 初始化跳过**。`AppState.__init__` 已改为惰性加载 RoleTable，避免 8s 阻塞启动。
3. **考虑替代方案**：角色信息走嵌入层面的语义关联（bge-m3 本身编码了角色上下文），不一定需要显式的角色矩阵。

### 状态
- AppState 惰性加载 ✅
- 检索链路已移除 RoleTable 依赖 ✅（手术已完成）
- 替代方案评估：未开始

---

## 原则（适用于后续所有架构变更）
1. **实时搜索链路上最多 3 个组件。** 每多一个组件 = 多一层排序噪声 + 多一个认知负载。
2. **未经测量不放行。** 新组件必须先在 `experiments/` 基线对照测试，才能进入搜索链路。
3. **降级不是删除，是物理隔离。** 旧代码保留在 `analysis/` 或 `experiments/` 目录，保证可复现。
