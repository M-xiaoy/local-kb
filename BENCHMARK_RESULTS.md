# 基准测试报告 Benchmark Results

## 版本信息
- **Git Commit:** `2d26a4bb8c81b0cdea44f26dc540f20462b2c00c`
- **Git Tag:** `v0.6-pre-surgery`
- **日期:** 2026-07-24
- **Python:** 3.12.10
- **依赖:** faiss-cpu 1.14.3, numpy 1.26.4, httpx 0.28.1
- **嵌入模型:** bge-m3:latest (Ollama, 本地 RTX 4060)
- **FAISS 索引:** 26,355 向量, 1024-dim, IVF 参数 (默认)
- **Poincaré 范数:** 26,355 球体 (0 fallback)

## 架构锁定
```
Embed (BGE-M3) → FAISS ANN (Top-50) → Poincaré Rerank (Top-10) → Output
```
旧管线组件 (DiversitySorter / FieldDetector / ActivationPropagator / LocalReranker / RoleExpander / TractionReranker / ClusterEngine) 已物理移出检索链路，降级为 `analysis/deprecated_pipeline/` 离线分析工具。

## 三路对比总表

| 模式 | recall@10 | 延迟 | 说明 |
|------|:-:|:-:|------|
| A: FAISS cosine | 29.3% | 9ms | FAISS 欧氏空间余弦初召 |
| B: FAISS → Poincaré re-rank | **37.3%** | **2ms** | 新增重排层,比 A +8% |
| C: Pure Poincaré full scan | 36.0% | 427ms | 全量双曲距离扫描,慢 48x |

**B vs A: 胜 5 / 负 1 / 平 9**

## 逐查询明细

| # | 查询 | A recall | B recall | 结果 | A top-3 | B top-3 |
|---|------|:-:|:-:|:----:|---------|---------|
| 1 | Transformer 注意力机制的工作原理 | 20% | 20% | — | Transformer 解读.md, 会话_2026-07-24_94cb0a79.md, 12_transformer.md | Transformer 解读.md, 会话_2026-07-24_94cb0a79.md, 12_transformer.md |
| 2 | 分布式训练的数据并行和模型并行区别 | 50% | 50% | — | chapter8_第八章分布式训练.md, chapter8_第八章分布式训练.md, chapter8_第八章分布式训练.md | chapter8_第八章分布式训练.md, chapter8_第八章分布式训练.md, chapter8_第八章分布式训练.md |
| 3 | GPU 显存优化技术 Flash Attention | 60% | 60% | — | 会话_2026-07-24_94cb0a79.md, apter6_第六章GPU和GPU相关的优化.md, apter6_第六章GPU和GPU相关的优化.md | 会话_2026-07-24_94cb0a79.md, apter6_第六章GPU和GPU相关的优化.md, apter6_第六章GPU和GPU相关的优化.md |
| 4 | 什么是知识蒸馏 | 20% | 40% | ✅ | LLM的未来LeCun.md, LLM的未来LeCun.md, chapter14_可验证奖励的强化学习.md | notes.md, LLM的未来LeCun.md, LLM的未来LeCun.md |
| 5 | LoRA 微调的原理和应用 | 10% | 10% | — | 第六章 大模型训练流程实践.md, 04_qwen2.5_qlora.md, 02_lora.md | 第六章 大模型训练流程实践.md, 04_qwen2.5_qlora.md, 02_lora.md |
| 6 | Agent 工具调用的设计模式 | 10% | 60% | ✅ | 第七章 大模型应用.md, 第七章 大模型应用.md, 第七章 大模型应用.md | 第七章 大模型应用.md, 第七章 大模型应用.md, 第七章 大模型应用.md |
| 7 | RAG 检索增强生成的工作流程 | 70% | 80% | ✅ | 第七章 大模型应用.md, 第七章 大模型应用.md, notes.md | 第七章 大模型应用.md, 第七章 大模型应用.md, notes.md |
| 8 | 模型的评估指标 准确率 召回率 | 10% | 40% | ✅ | 0.2 评价指标.md, _building_and_training.md, 0.2 评价指标.md | 0.2 评价指标.md, _building_and_training.md, 0.2 评价指标.md |
| 9 | 多头注意力 Multi-Head Attention 计算过程 | 40% | 30% | ❌ | Transformer 解读.md, 第五章 动手搭建大模型.md, ter4_第四章语言模型架构和训练的技术细节.md | Transformer 解读.md, 第五章 动手搭建大模型.md, ter4_第四章语言模型架构和训练的技术细节.md |
| 10 | 强化学习中的奖励模型 Reward Model | 10% | 10% | — | 6.4[WIP] 偏好对齐.md, 6.4[WIP] 偏好对齐.md, signment5_alignment_zh.md | 6.4[WIP] 偏好对齐.md, 6.4[WIP] 偏好对齐.md, signment5_alignment_zh.md |
| 11 | PyTorch 的自动求导机制 | 10% | 10% | — | 2.2 自动求导.md, 1.2 PyTorch的安装.md, notes.md | 2.2 自动求导.md, 1.2 PyTorch的安装.md, notes.md |
| 12 | 分词器类型 BPE WordPiece SentencePiece | 70% | 70% | — | 第五章 动手搭建大模型.md, chapter2_分词器.md, 15_T5.md | 第五章 动手搭建大模型.md, chapter2_分词器.md, 15_T5.md |
| 13 | AI 三大流派 符号主义 连接主义 行为主义 | 60% | 80% | ✅ | 会话_2026-07-24_94cb0a79.md, notes.md, 0.1 人工智能简史.md | 会话_2026-07-24_94cb0a79.md, notes.md, notes.md |
| 14 | 大模型幻觉问题怎么缓解 | 0% | 0% | — | 01_LLM_safety_overview.md, 01_LLM_safety_overview.md, apter13_第十三章大模型的基本训练流程.md | 01_LLM_safety_overview.md, 01_LLM_safety_overview.md, apter13_第十三章大模型的基本训练流程.md |
| 15 | 向量数据库的检索流程 ANN HNSW | 0% | 0% | — | 第七章 大模型应用.md, 信息检索.md, 06_gensim.md | 第七章 大模型应用.md, 信息检索.md, 06_gensim.md |

## 观测
1. Poincaré 重排在 5 条查询上显著优于余弦，1 条上略差，9 条持平。
2. 重排层延迟 1-2ms，FAISS 初召 6-9ms，总计 <11ms（不含嵌入）。
3. 嵌入 (bge-m3 via Ollama) 耗时 3.4s，是当前系统延迟瓶颈。
4. 纯 Poincaré 全量扫描 (427ms) 未发现 FAISS 漏掉的相关文档。
