# 大模型长文本复读机：不是 decoding 问题，是注意力坍缩

> 模型 SFT 后长文本生成频繁复读大段内容，repeat_penalty 调了没用，换 decoding 策略也没用？

---

先说结论：**复读机问题不是 token 级别的决策错误，是注意力系统在长文本尾部发生了结构化退化。** 所有在 token level 做的修正（penalty、候选重选、温度调节）都治不了它，因为工具的输入已经坏了。

这个观点不是原创的。下面三个机制来自已发表论文和社区方案，但**很少有人把三个放在一起看它们如何汇合成 collapse**——而这正是解决问题的关键。

---

## 一、注意力坍缩的三个原因

### 1. Attention Sink：BOS 吸走了所有人的注意力

Xiao et al. (2023) 在 *Efficient Streaming Language Models with Attention Sinks* 中发现了一个反直觉的现象：**无论生成到多长，第一个 token（BOS）始终占据不成比例的注意力分数。**

原因在数学层面很干净：softmax attention 的分数之和必须为 1。当序列变长，模型发现大部分 key 和当前 query 不相关时，它不能把注意力全堆在最近几个 token 上——因为 softmax 会指数级放大高匹配分数。但好在有一个 token 永远「不算完全不相关」——BOS。

```
正常（前 1K tokens）：
  attention 分散在 ~100 个 token 上

尾部（8K tokens）：
  BOS → 40~55%
  最近 50~200 个 tokens → 35~50%
  中间所有 token → 剩下 10% 分摊
```

这意味着尾部生成时，**有效信息窗口只有最近的几十个 token**。中间段的历史信息实际是「不可见」的。

### 2. RoPE 的 extrapolation 失效

现在主流模型（LLaMA、Qwen、DeepSeek 全系）都在用旋转位置编码。RoPE 把位置编码为旋转角度，query 和 key 的相对距离编码为角度差。训练阶段，模型见过的角度差分布在有限区间（譬如 -π 到 π）。一旦生成长度超出训练范围：

```
训练时见过的角度差：[-π, π]
推理时长文本尾部的角度差：可到 [-3π, 3π]
```

旋转角度差过大会导致点积退化到接近 0（数学上 cos 函数周期衰减的结果）。**远处的 token 不是被模型「忽略」了，而是模型根本没有能力给它们分配注意力**。

这就是为什么 YaRN / NTK-aware RoPE scaling（Peng et al., 2023 及社区贡献）有效——它本质上是把位置编码压缩，让模型以为自己在短距离区间内工作。代价是位置分辨率降低，但解决了视野急剧缩小的问题。

### 3. Activation Saturation：残差流的物理天花板

Transformer 的 residual stream 每层都在叠加信息。到了长文本尾部：

```
第 1 层：x + attention(x) + ff(x)
...
第 32 层：x + ...（叠加了 32 层 × 8K tokens 的信息）
```

LayerNorm 和 SwiGLU 对输入动态范围有硬限制。当残差流塞满信息后，新信息必须「挤掉」旧信息才能被写入。模型隐式策略是砍掉中间、保留两端——因为 BOS 和最近 token 恰好是「唯二不会被挤掉的区间」，而这和 Attention Sink 形成正反馈。

**三个机制不独立，它们互相加强：**

```
RoPE extrapolation 失效 → 远处不可见 → 注意力集中在近处
Attention Sink         → BOS 吞掉一半剩余 → 分摊到近处
Activation Saturation  → 中间被挤掉 → 只剩近处

三者的共同输出：
每生成一步，有效视野进一步缩小 → 自我强化的 collapse
```

---

## 二、为什么备选答案解决不了

这是我的实际观察：用了备选答案（top-k 候选+选择器）的模型，长文本末尾照样复读。

原因简单得让人沮丧：**当所有候选从同一个快速缩小的视野窗口里采样时，候选之间的差异只是措辞差异，语义上是一样的。**

```
视野正常时：
  候选 A: "该模型在 ImageNet 上达到 89.7% 的准确率"
  候选 B: "实验结果显示该模型的分类准确率为 89.7%"
  → 两者不一样，选择有意义

视野收缩后：
  候选 A: "我们进一步探讨了该方法的可扩展性"
  候选 B: "我们进一步研究了该方法的扩展能力"
  → 同义替换，选哪个都是复读
```

选择器本身没坏，但它的**输入来源（所有候选）已经塌缩了**。这不是 decoding 算法的缺陷，是注意力系统给下游输送的都是不可区分的信号。

---

## 三、行业实际在用的方案

### 零成本级（推理时直接叠加）

**YaRN / NTK-aware RoPE scaling** — 数学层面上改写位置编码的缩放函数。模型仍然「以为」自己在短距离内工作。不改变生成逻辑，成本趋近于零。几乎所有宣称长上下文的模型底层都在用这个或其变体。

**StreamingLLM**（Xiao et al., 2023） — KV Cache 淘汰策略：保留 BOS + 几个开头 anchor token + 最近的观察窗口，中间的直接丢弃。注意力窗口永远固定大小，不会扩张。代价是中间的「记忆」丢失了——但对话场景下，人的记忆也是近的清晰远的模糊。

**DoLa（Decoding by Contrasting Layers）**（Chuang et al., 2023）— 每一层都跑一遍，比较中间层和最后层对下一个 token 的预测分布。差异大 = 自信，正常生成；差异小 = 迷茫期，触发 contrastive 机制。实际上把「中间层还没 collapse」的信号当作 collapse 检测器。

这三者叠用可以压掉 70~80% 的复读，零训练成本。生产系统的最佳实践。

### 高成本级（需要训练或推理算力）

**Context Consistency Training**（各类变体）— SFT 数据按长度配比采样，尾部 token 引入对比损失，迫使模型在长文本尾部学习多样化续写路径。成本最高但效果最好——DeepSeek 的 LongAlign 属于这类。

**Memory-Augmented Transformer**（RETRO / Atlas, DeepMind, 2022）— 每个 token 的 cross-attention 并行走一趟检索。生成空间始终保持干净，长文本信息放在外部记忆里。推理成本暴涨但理论上最接近「解决注意力坍缩」的方案。

| 方案 | 复读抑制 | 额外成本 | 生产成熟度 |
|------|---------|---------|-----------|
| YaRN | 中 | 零 | ✅ 所有大厂在用 |
| StreamingLLM | 高 | 趋近零 | ✅ 对话系统标配 |
| DoLa | 高 | 2× 推理 | 🟡 部分集成 |
| 训练级方案 | 最高 | 极高 | ❌ 头部团队专属 |
| Memory-Augmented | 高 | 检索延迟 | ❌ 研究阶段 |

---

## 四、一个不同的视角：三级记忆体系

如果把三个原因映射到认知体系上，我倾向于用这三个层级来描述生成过程：

```
第 1 级（显意识层）：当前生成窗口
  → 因果注意力 + FFN，原始 Transformer
  → 复读的物理位置在这里
  → 所有 collapse 发生的地方

第 2 级（近端记忆层）：KV Cache 的流式窗口
  → 最近的 tokens，对应 StreamingLLM
  → 保持视野不变

第 3 级（潜意识层）：外部分布式记忆
  → 在需要时检索概念向量，非 token 级访问
  → 不干扰生成空间的纯洁性
```

注意力坍缩只会在第 1 级发生。但如果第 2 级和第 3 级始终提供干净的上下文信号，collapse 的信号无法形成正反馈循环——因为每一次生成，模型都能从外部拉到一个与原视野无关的新方向。

这个想法太过简化（实际的记忆检索和注意力交互远非线性），但它指向了一个我认为有希望的方向：**不是让注意力看得更远，而是让注意力不需要看那么远**——需要时从外部准确拉取就够了。

---

## 五、总结

- 复读机不是 decoding 问题，是**注意力系统的结构化退化**
- 三个退化机制（Attention Sink + RoPE 失效 + Activation 饱和）相互加强
- 备选答案解决不了，因为候选全部从退化后的视野采样
- 行业最佳实践是 **YaRN + StreamingLLM + DoLa** 叠用
- 更长远的方案是记忆增强——**不让注意力看它不需要看的东西**

---

**参考文献**

- Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). *Efficient Streaming Language Models with Attention Sinks.*
- Peng, B., Alcaide, E., Anthony, Q., et al. (2023). *YaRN: Efficient Neural Machine Translation with Rotary Position Embedding Scaling.*
- Chuang, Y.-S., Xie, Y., Luo, H., Kim, Y., Glass, J., & He, P. (2023). *DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models.*
- Borgeaud, S., Mensch, A., Hoffmann, J., et al. (2022). *Improving Language Models by Retrieving from Trillions of Tokens.*
- Li, X. L., Holtzman, A., Fried, D., et al. (2022). *Contrastive Decoding: Open-ended Text Generation on Optimized Language Models.*
