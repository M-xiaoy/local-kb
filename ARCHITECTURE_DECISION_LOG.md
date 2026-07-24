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

## 原则（适用于后续所有架构变更）
1. **实时搜索链路上最多 3 个组件。** 每多一个组件 = 多一层排序噪声 + 多一个认知负载。
2. **未经测量不放行。** 新组件必须先在 `experiments/` 基线对照测试，才能进入搜索链路。
3. **降级不是删除，是物理隔离。** 旧代码保留在 `analysis/` 或 `experiments/` 目录，保证可复现。
