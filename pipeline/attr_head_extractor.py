"""
attr_head_extractor.py — 定中短语（Attribute-Head Pair）抽取器
==============================================================
核心假设：定语（修饰成分）携带的信息密度远高于中心语（名词）。

设计：
  - 规则优先（无需 spaCy），可移植到任何环境
  - 两层策略：「的」锚定法(高精度) + 复合名词解构(高召回)
  - 三重防噪声：短窗口 / 动词截断 / 后缀白名单

用法：
  from pipeline.attr_head_extractor import AttrHeadExtractor
  ext = AttrHeadExtractor()
  pairs = ext.extract("基于深度学习的跨模态图像分割方法在医疗影像分析中表现优异")
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 常见名词性后缀（按长度降序，优先长匹配） ──
_NOUN_SUFFIXES: tuple = (
    # 4字
    "度量学习方法", "自注意力机制", "特征空间", "神经网络",
    "深度学习", "机器学习", "知识蒸馏", "原型网络", "消息队列",
    "微服务架构", "性能指标",
    # 3字
    "注意力", "编码器", "解码器", "生成器", "判别器",
    "测试集", "验证集", "训练集", "数据集",
    "基准线", "基准测试",
    # 2字
    "模型", "方法", "系统", "算法", "框架", "网络", "架构",
    "数据", "图像", "文本", "音频", "视频", "信号",
    "分析", "检测", "识别", "预测", "分类", "聚类", "回归",
    "评估", "优化", "生成", "提取", "检索", "推理", "训练",
    "引擎", "组件", "模块", "工具", "平台", "协议", "接口",
    "机制", "策略", "方案", "模式", "范式", "流程", "管线",
    "任务", "场景", "应用", "领域", "问题", "挑战",
    "矩阵", "向量", "张量", "参数", "权重", "特征",
    "损失", "梯度", "信息", "误差",
    "性能", "质量", "效率", "精度", "准确率",
    "输出", "输入", "嵌入",
    # 1字
    "法", "器", "机", "仪", "剂", "体", "物",
    "层", "块", "核", "域", "场",
    "型", "识", "码", "图", "表", "卡",
    "性", "率", "度", "量", "值", "集", "流",
    "门", "路", "线", "点", "面",
    "差", "比","态","势",
)

# ── 动词截断词（标记动→名转换点） ──
_CUTOFF_WORDS: tuple = (
    "提出", "设计", "采用", "利用", "使用", "通过",
    "作为", "用于", "称为", "叫做", "包括", "包含",
    "研究", "分析", "讨论", "介绍", "说明", "展示",
    "验证", "证明", "表明", "显示", "发现",
    "是将", "是指", "是", "有",
)

# ── 纯介词/动词（过滤，不接受为中心语） ──
_REJECT_HEADS: set = {
    "进行", "使用", "利用", "基于", "采用", "通过",
    "针对", "对于", "关于", "经过", "作为",
}

# ── 标点/边界字符 ──
_BOUNDARY_CHARS: str = '，,。.;；：:！!？?、()（）【】[]{}「」『』——…\n\t'


@dataclass
class AttrHeadPair:
    """一个定中短语的结构化表示"""
    head: str                           # 中心语（如"方法"）
    attributives: List[str]             # 定语列表（从远到近）
    full_phrase: str                    # 完整定中短语
    rely_on_de: bool = True             # 是否通过「的」识别

    @property
    def has_attributive(self) -> bool:
        return len(self.attributives) > 0 and any(len(a.strip()) > 0 for a in self.attributives)

    @property
    def discriminator(self) -> str:
        """最能代表这个短语的鉴别词（最短的有意义定语或完整短语）"""
        if not self.attributives:
            return self.head
        meaningful = [a for a in self.attributives if len(a) >= 2]
        return min(meaningful, key=len) if meaningful else self.full_phrase


# ── 核心提取器 ──

class AttrHeadExtractor:
    """
    定中短语提取器。
    规则优先，spaCy 作为可选的补充信号。
    """

    def __init__(self, use_spacy: bool = True):
        self._spacy = None
        if use_spacy:
            try:
                import spacy
                self._spacy = spacy.load("zh_core_web_sm")
                logger.info("AttrHeadExtractor: spaCy loaded")
            except Exception:
                logger.info("AttrHeadExtractor: spaCy unavailable, rule-only")
        self.stats: Dict[str, int] = {"de_pairs": 0, "compound_pairs": 0, "spacy_pairs": 0, "sentences": 0}

    def extract(self, text: str) -> List[AttrHeadPair]:
        """从文本中提取所有定中短语"""
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        self.stats["sentences"] = len(sentences)

        all_pairs: List[AttrHeadPair] = []
        for sent in sentences:
            all_pairs.extend(self._extract_from_sentence(sent))

        return self._deduplicate(all_pairs)

    # ═══════════════════════════════════════════
    # 句子分割
    # ═══════════════════════════════════════════

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[。！？.!?])\s*', text)
        return [s.strip() for s in parts if len(s.strip()) > 3]

    # ═══════════════════════════════════════════
    # 单句提取（三条策略）
    # ═══════════════════════════════════════════

    def _extract_from_sentence(self, sentence: str) -> List[AttrHeadPair]:
        pairs: List[AttrHeadPair] = []

        # 1. spaCy（如有）
        if self._spacy:
            spacy_pairs = self._extract_spacy(sentence)
            self.stats["spacy_pairs"] += len(spacy_pairs)
            pairs.extend(spacy_pairs)

        # 2. 「的」锚定法（精度最高）
        de_pairs = self._extract_by_de(sentence)
        self.stats["de_pairs"] += len(de_pairs)
        pairs.extend(de_pairs)

        # 3. 复合名词解构
        compound_pairs = self._extract_compounds(sentence)
        self.stats["compound_pairs"] += len(compound_pairs)
        pairs.extend(compound_pairs)

        return self._deduplicate(pairs)

    # ═══════════════════════════════════════════
    # 策略一：spaCy 依存句法
    # ═══════════════════════════════════════════

    # ── spaCy 中文修饰依赖标签集 ──
    _CN_MODIFIER_DEPS: tuple = (
        "amod",           # 形容词修饰 （优异 → 性能）
        "compound:nn",    # 名词复合   （Transformer → 架构, 服务 → 架构）
        "nmod:assmod",    # 领属修饰   （平台 → 底层, "平台的"）
        "det",            # 限定词     （该 → 方法）
        "nummod",         # 数量修饰   （三个 → 模型）
    )

    def _extract_spacy(self, sentence: str) -> List[AttrHeadPair]:
        if not self._spacy:
            return []
        doc = self._spacy(sentence)
        pairs = []
        for token in doc:
            if token.pos_ not in ("NOUN", "PROPN"):
                continue
            # 找当前名词的所有直接修饰子节点
            modifiers = [c for c in token.children
                         if c.dep_ in self._CN_MODIFIER_DEPS]
            if not modifiers:
                # 特殊：如果 token 本身是复合名词的一部分但有更重要的语法角色
                # 并且它的父节点有 compound:nn 链，向上追
                # 检查子节点是否有 compound:nn 反向传播
                parent = token.head
                if token.dep_ == "compound:nn" and parent.pos_ in ("NOUN", "PROPN"):
                    # 当前 token 是别人定语的一部分，跳过（在父节点处理）
                    pass
                continue

            head = token.text
            attrs = [m.text for m in sorted(modifiers, key=lambda t: t.idx)]

            # 对于 nmod:assmod，把关联的「的」也包含进完整短语
            has_assmod = any(c.dep_ == "nmod:assmod" for c in modifiers)

            min_c = min(m.idx for m in modifiers)
            max_c = max(m.idx + len(m.text) for m in modifiers)
            max_c = max(max_c, token.idx + len(token.text))

            # 如果有关联修饰 + 的，把「的」也包含进来
            if has_assmod:
                # 检查定语后面有没有「的」
                for de_candidate in token.children:
                    if de_candidate.text == "的":
                        max_c = max(max_c, de_candidate.idx + 1)
                    if de_candidate.dep_ == "case" and de_candidate.text == "的":
                        pass

            full_text = sentence[min_c:max_c]

            # 排除纯代词/停用词作定语
            clean_attrs = [a for a in attrs if a not in {"该", "这", "那", "哪", "什么"}]
            if not clean_attrs:
                continue

            pairs.append(AttrHeadPair(head=head, attributives=clean_attrs,
                                      full_phrase=full_text, rely_on_de=False))
        return pairs

    # ═══════════════════════════════════════════
    # 策略二：「的」锚定法（核心）
    # ═══════════════════════════════════════════

    def _extract_by_de(self, sentence: str) -> List[AttrHeadPair]:
        """找「X的Y」模式，Y 以名词性后缀结尾"""
        pairs = []
        for m in re.finditer(r'的', sentence):
            de_pos = m.start()

            head = self._find_head(sentence, de_pos + 1)
            if not head:
                continue

            attr = self._find_attributive(sentence, de_pos)
            if not attr:
                continue

            full = attr + "的" + head
            layers = self._split_attr_layers(attr)
            pairs.append(AttrHeadPair(head=head, attributives=layers,
                                      full_phrase=full, rely_on_de=True))
        return pairs

    _HEAD_SOFT_BOUNDARY: tuple = (
        '在', '中', '上', '下', '里', '内', '外', '前', '后',
        '时', '到', '入', '出',
        '与', '和', '或', '及', '以及', '而', '但', '且',
        '通过', '用于', '来自', '分为', '包括', '属于',
    )

    def _find_head(self, sentence: str, start: int) -> Optional[str]:
        """找「的」后面的名词性中心语"""
        rest = sentence[start:]
        if not rest or not rest[0].strip():
            return None

        # 从左到右读，遇边界、连接词、或超长停
        buf = ''
        for ch in rest:
            if ch in _BOUNDARY_CHARS:
                break
            if ch in self._HEAD_SOFT_BOUNDARY:
                # 连接词是名词短语的结尾标记，但保留已经读到的部分
                break
            if len(buf) >= 12:
                break
            buf += ch

        buf = buf.strip()
        if not buf:
            return None

        # 全范围扫描：从右到左切每一段，找符合后缀或短名词兜底
        for end in range(len(buf), 1, -1):
            sub = buf[:end]
            # 后缀匹配
            for suffix in _NOUN_SUFFIXES:
                if sub.endswith(suffix) and len(sub) > len(suffix):
                    return sub
            # 短名词兜底：2-3 字的名词
            if 2 <= end <= 3:
                known = {'知识','模型','数据','算法','代码','系统','结构','任务',
                         '策略','方法','网络','概念','信息','原理','状态','类型',
                         '框架','特征','参数','权重','损失','梯度','嵌入','编码',
                         '架构','机制','场景','应用','领域','问题','引擎','组件',
                         '序列','冗余','分布','集合','关系','差异','影响','效应',
                         '过程','阶段','步骤','环节','能力','质量','效率','精度',
                         '层次','分支','索引','版本','备份','角色','权限','配置',
                         '通道','连接','映射','转化','融合','对齐','引导','驱动'}
                if sub in known:
                    return sub

        return None

    def _find_attributive(self, sentence: str, de_pos: int) -> Optional[str]:
        """找「的」前面的定语成分（有限窗口 + 动词截断）"""
        before = sentence[:de_pos].rstrip()
        if not before:
            return None

        # 最多取 22 字
        window = min(len(before), 22)
        candidate = before[-window:]

        # 1) 强边界截断：遇到标点，取其后
        for i, ch in enumerate(candidate):
            if ch in _BOUNDARY_CHARS:
                candidate = candidate[i+1:]
                break

        candidate = candidate.lstrip()
        if not candidate:
            return None

        # 2) 动词截断：只在 candidate 开头匹配
        #    如果 cutoff word 在中间，那它大概率是复合名词的一部分（如"验证集"）
        for w in sorted(_CUTOFF_WORDS, key=len, reverse=True):
            if not candidate.startswith(w):
                continue
            after = candidate[len(w):].lstrip()
            if len(after) >= 2:
                candidate = after
                break

        candidate = candidate.lstrip()

        # 3) 最终长度限制（定语本身太长 → 取最后 16 字）
        if len(candidate) > 16:
            candidate = candidate[-16:]
        if len(candidate) < 1:
            return None

        return candidate

    @staticmethod
    def _split_attr_layers(text: str) -> List[str]:
        """分解多层定语：按「的」或「,」拆分"""
        if not text:
            return [""]
        if '的' in text:
            return [p.strip() for p in re.split(r'的', text) if p.strip()]
        parts = re.split(r'[,，、]', text)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]
        return [text.strip()]

    # ═══════════════════════════════════════════
    # 策略三：复合名词解构（无「的」的情况）
    # ═══════════════════════════════════════════

    def _extract_compounds(self, sentence: str) -> List[AttrHeadPair]:
        """
        找短片段（4~20字）以名词后缀结尾的复合名词。
        只在标点/「的」之间的子串中查找，不跨句子。
        """
        pairs = []
        segments = re.split(r'[' + re.escape(_BOUNDARY_CHARS) + r']', sentence)

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            # 按「的」进一步细分（复合词内部不应有「的」）
            for sub in re.split(r'的', seg):
                sub = sub.strip()
                if len(sub) < 4 or len(sub) > 20:
                    continue
                # 去除非中英文数字字符
                clean = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '', sub)
                if len(clean) < 4:
                    continue

                # 从右向左匹配长后缀
                for suffix in _NOUN_SUFFIXES:
                    if not clean.endswith(suffix):
                        continue
                    head_len = len(suffix)
                    attr_len = len(clean) - head_len
                    if attr_len < 2:
                        continue
                    attr = clean[:-head_len]
                    if attr in {'进行', '使用', '利用', '基于', '采用', '通过'}:
                        continue
                    pairs.append(AttrHeadPair(
                        head=clean[-head_len:], attributives=[attr],
                        full_phrase=clean, rely_on_de=False
                    ))
                    break  # 只取最长匹配

        return pairs

    # ═══════════════════════════════════════════
    # 去重
    # ═══════════════════════════════════════════

    @staticmethod
    def _deduplicate(pairs: List[AttrHeadPair]) -> List[AttrHeadPair]:
        seen: set = set()
        result = []
        for p in pairs:
            key = p.full_phrase.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(p)
        return result

    # ═══════════════════════════════════════════
    # 统计
    # ═══════════════════════════════════════════

    def report(self) -> Dict:
        return dict(self.stats)


# ═══════════════════════════════════════════════
# 工具函数：格式化输出 / 对比实验
# ═══════════════════════════════════════════════

def format_pairs(pairs: List[AttrHeadPair], show_all: bool = True) -> str:
    lines = []
    for p in pairs:
        if not show_all and not p.has_attributive:
            continue
        tag = "[DE]" if p.rely_on_de else "[COMP]"
        attr = " | ".join(p.attributives) if p.attributives else "(无)"
        lines.append(f"  {tag} {p.full_phrase}")
        lines.append(f"     定: {attr}  |  中: {p.head}")
    return "\n".join(lines) or "  (无)"


def compare_with_subject_extraction(text: str) -> str:
    """对比定中短语 vs 主谓宾抽取"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipeline.role_extractor_v2 import HybridExtractor

    ah_ext = AttrHeadExtractor()
    sv_ext = HybridExtractor()

    pairs = ah_ext.extract(text)
    triples = sv_ext.extract(text)

    lines = []
    lines.append("=" * 60)
    lines.append("<< 样本文本 >>\n" + text[:300] + ("..." if len(text) > 300 else ""))

    lines.append("\n[当前] 主谓宾抽取")
    lines.append("-" * 30)
    if triples:
        for t in triples:
            s = t.get("subject", "") or "-"
            o = t.get("object", "") or "-"
            lines.append(f"  主语: {s}")
            lines.append(f"  宾语: {o}")
    else:
        lines.append("  (未抽到)")

    lines.append("\n[新] 定中短语抽取")
    lines.append("-" * 30)
    lines.append(format_pairs(pairs, show_all=True))

    # 鉴别力对比
    sv_heads = set()
    for t in triples:
        if t.get("subject"):
            sv_heads.add(t["subject"])
        if t.get("object"):
            sv_heads.add(t["object"])

    ah_heads = set(p.discriminator for p in pairs)

    lines.append("\n[对比] 信息密度")
    lines.append("-" * 30)
    lines.append(f"  SVO 实体词: {len(sv_heads)} 个")
    lines.append(f"    值: {', '.join(sorted(sv_heads)) if sv_heads else '(空)'}")
    lines.append(f"  定中鉴别器: {len(ah_heads)} 个")
    lines.append(f"    值: {', '.join(sorted(ah_heads)) if ah_heads else '(空)'}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    SAMPLES = [
        "基于深度学习的跨模态图像分割方法在医疗影像分析中展现出优异性能。",
        "本研究提出了一种结合注意力机制的双向长短期记忆网络用于情感分类任务。",
        "Transformer架构通过自注意力机制捕获序列中的长距离依赖关系。",
        "针对小样本学习场景，我们设计了一种基于原型网络的特征空间度量学习方法。",
        "通过对比实验验证了该算法在低信噪比条件下的鲁棒性。",
        "知识蒸馏技术将大型教师模型的知识迁移到小型学生模型中。",
        "该平台的底层采用微服务架构，通过消息队列实现异步通信。",
        "该方法在标准测试集上取得了最优结果。",
        "模型在验证集上的准确率达到了97.3%。",
    ]

    print("=" * 60)
    print("定中短语提取器 v0.1 - 初步实验")
    print("=" * 60)

    ext = AttrHeadExtractor()
    all_pairs = []
    for text in SAMPLES:
        pairs = ext.extract(text)
        all_pairs.extend(pairs)

    print(f"\n[STATS] 句子数: {len(SAMPLES)}  定中短语: {len(all_pairs)}  ({ext.report()})")
    print("-" * 60)

    for i, text in enumerate(SAMPLES):
        pairs = ext.extract(text)
        print(f"\n[{i+1}] {text}")
        if pairs:
            for p in pairs:
                tag = "[DE]" if p.rely_on_de else "[COMP]"
                print(f"  {tag} -> {p.full_phrase}  (中: {p.head})")
        else:
            print("  (无)")

    print("\n" + "=" * 60)
    print("对比实验")
    print("=" * 60)
    print()
    print(compare_with_subject_extraction(
        "针对小样本学习场景，我们设计了一种基于原型网络的特征空间度量学习方法，在多个基准数据集上取得了领先结果。"
    ))
    print()
    print(compare_with_subject_extraction(
        "Transformer架构通过自注意力机制捕获序列中的长距离依赖关系，同时位置编码保留了序列的顺序信息。"
    ))
