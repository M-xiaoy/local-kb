"""
role_table.py — 角色共现表（主语-宾语结构）
============================================
核心数据结构，记录每个实体在哪些句子中充当主语、哪些充当宾语。

检索时使用：命中句子 → 查其宾语实体 → 找到该实体做主语的句子 → 扩展候选池
实现因果链的「结构性连接」——不需要依赖文本相似度或LLM推理。

存储架构：
  role_table.json          ← 实体注册表 + 角色共现记录
  role_vectors.npz         ← 实体的embedding向量（选填，多级检索用）

使用方式：
  table = RoleTable()
  table.add_occurrence("冰川融化", "subject", "sphere_abc123")
  table.add_occurrence("海平面上升", "object", "sphere_abc123")
  table.add_occurrence("海平面上升", "subject", "sphere_def456")
  
  # 检索扩展
  jumps = table.expand_from_sphere("sphere_abc123")
  # → [("sphere_def456", "role_bridge", 0.7), ...]
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import paths as cfg_paths

logger = logging.getLogger(__name__)

ROLE_TABLE_VERSION = 1


# ──────────────────────────────────────────────
# 实体信息
# ──────────────────────────────────────────────

@dataclass
class EntityInfo:
    """一个实体在所有句子中的角色分布"""
    text: str                        # 实体原文（如"冰川融化"）
    occurrences: int = 0             # 总出现次数
    as_subject: List[str] = field(default_factory=list)   # 做主语时的句子 sphere_id
    as_object: List[str] = field(default_factory=list)    # 做宾语时的句子 sphere_id
    subject_count: int = 0
    object_count: int = 0
    # 与其他实体的共现记录（可选扩展）
    # 格式: {other_entity_id: 共现次数}
    co_occurrences: Dict[str, int] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """实体是否有足够出现次数，值得作为跳转节点"""
        return self.occurrences >= 2  # ≥2 次才可能同时做主语和宾语


# ──────────────────────────────────────────────
# 角色跳转结果
# ──────────────────────────────────────────────

@dataclass
class JumpCandidate:
    target_sphere_id: str            # 目标句子球体 ID
    bridge_entity: str               # 桥接实体（如"海平面上升"）
    bridge_type: str                 # "subject→object" | "object→subject"
    confidence: float                # 跳转置信度 [0, 1]


# ──────────────────────────────────────────────
# 主语抽取器（规则版）
# ──────────────────────────────────────────────

class SubjectExtractor:
    """基于规则的主语/宾语抽取

    规则设计原则：
      - 不依赖句法树、不依赖词性标注、不依赖外部NLP库
      - 覆盖中文和英文混合文本（技术笔记常见）
      - 优先 precision，recall 可以慢慢扩

    当前规则：
      中文主语: 句首名词性短语（到第一个谓词/标点为止）
      英文主语: 句首名词短语或专有名词
      宾语: 谓语后的名词短语（有限模式）
    """

    # 中文谓词标记词 - 主语到此结束
    PREDICATE_MARKERS_ZH = [
        "是", "不是", "就是", "也是",  # 系动词
        "了", "着", "过",              # 动态助词
        "会", "能", "可以", "可能",    # 情态
        "将", "已经", "正在",           # 时间标记
        "在", "把", "被", "让", "使",   # 介词
        "导致", "引起", "造成",          # 因果动词
        "提出", "发现", "证明",          # 研究动词
        "位于", "属于", "包含",         # 关系动词
        "推进", "实现", "解决",         # 动作动词
        "提高", "降低", "增加",         # 变化动词
        "分为", "构成", "形成",         # 组成
    ]

    # 宾语提取触发词 — 后面跟的名词短语就是宾语
    OBJECT_TRIGGERS = [
        "导致", "引起", "造成", "产生",
        "提出", "发现", "证明", "实现",
        "包含", "包括", "涉及", "分为",
        "推进", "推动", "提高", "降低",
        "需要", "要求", "基于",
    ]

    def extract(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取 (subject, verb, object) 三元组

        Args:
            text: 一段文本（可能包含多句）

        Returns:
            [{"subject": "冰川融化", "verb": "导致", "object": "海平面上升"}, ...]
        """
        sentences = self._split_sentences(text)
        results = []

        for sent in sentences:
            triple = self._extract_one(sent)
            if triple:  # 跳过无法提取的（无主语或无宾语）
                results.append(triple)

        return results

    def _split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子（兼容中英文标点）"""
        # 句号、感叹号、问号、分号+换行
        raw = re.split(r'(?<=[。！？.!?;；])\s*', text)
        return [s.strip() for s in raw if len(s.strip()) > 5]

    def _extract_one(self, sentence: str) -> Optional[Dict[str, str]]:
        """从单个句子中提取主谓宾"""
        if not sentence:
            return None

        # 依次尝试各类规则
        triple = (
            self._try_chinese_svo(sentence)
            or self._try_chinese_ba_bei(sentence)
            or self._try_english_svo(sentence)
        )
        return triple

    # ── 中文常规主谓宾 ────────────────────────

    def _try_chinese_svo(self, sentence: str) -> Optional[Dict[str, str]]:
        """标准中文 SVO：主语 + 谓词 + 宾语"""
        # 找第一个谓词标记
        pivot_idx = -1
        verb = ""
        for marker in self.PREDICATE_MARKERS_ZH:
            idx = sentence.find(marker)
            if idx != -1:
                # 取第一个匹配的
                if pivot_idx == -1 or idx < pivot_idx:
                    pivot_idx = idx
                    verb = marker

        if pivot_idx <= 0:
            return None

        subject = sentence[:pivot_idx].strip()
        remainder = sentence[pivot_idx + len(verb):].strip()

        # 清理主语中的标点和引导词
        subject = self._clean_subject(subject)
        if not subject or len(subject) > 50:
            return None

        # 提取宾语（如果有触发词，走宾语模式；否则取完整的后半句）
        obj = self._extract_object(remainder)

        if obj:
            return {"subject": subject, "verb": verb, "object": obj}
        return None

    # ── 中文把/被/将 结构 ─────────────────────

    def _try_chinese_ba_bei(self, sentence: str) -> Optional[Dict[str, str]]:
        """处理"把/被/将"结构
        
        把：主语 + 把 + 宾语 + 动词 
        被：宾语 + 被 + 主语 + 动词
        将：主语 + 将 + 宾语 + 动词
        """
        for marker in ("把", "将"):
            idx = sentence.find(marker)
            if idx > 0:
                subject = sentence[:idx].strip()
                subject = self._clean_subject(subject)
                if not subject:
                    continue
                # "把/将" 后面到第一个动词之间的名词短语是宾语
                after_marker = sentence[idx + 1:].strip()
                obj_end = self._find_next_verb(after_marker)
                if obj_end > 0:
                    obj = after_marker[:obj_end].strip()
                    return {"subject": subject, "verb": marker, "object": obj}
                elif obj_end == 0:
                    # 标记后直接是动词 → 非标结构
                    return None

        # "被" 结构（被动）
        idx = sentence.find("被")
        if idx > 0:
            subject = sentence[:idx].strip()  # 逻辑宾语
            subject = self._clean_subject(subject)
            if not subject:
                return None
            after_bei = sentence[idx + 1:].strip()
            # "被" 后面可能是施事者 + 动词，或直接是动词
            verb_idx = self._find_next_verb(after_bei)
            if verb_idx > 0:
                agent = after_bei[:verb_idx].strip()
                remainder = after_bei[verb_idx:]
                return {
                    "subject": agent if len(agent) > 1 else subject,
                    "verb": self._extract_verb(remainder) or "被",
                    "object": subject,  # 逻辑主语是真正的宾语
                }

        return None

    # ── 英文 SVO ─────────────────────────────

    def _try_english_svo(self, sentence: str) -> Optional[Dict[str, str]]:
        """简单英文主谓宾"""
        # 只处理含英文特征词的句子（不是纯中文）
        if not any(c.isascii() for c in sentence):
            return None

        # 查找谓词（be/have/do 类或常见动词）
        verb_patterns = [
            r'\b(is|are|was|were|has|have|had|do|does|did)\s',
            r'\b(leads?|causes?|results?|enables?|provides?|achieves?|improves?|reduces?|increases?|contains?|includes?|refers?)\s',
            r'\b(demonstrates?|shows?|finds?|concludes?|proposes?|introduces?|presents?)\s',
            r'\b(is based on|consists of|depends on|focuses on)\s',
        ]

        for pattern in verb_patterns:
            m = re.search(pattern, sentence, re.IGNORECASE)
            if m:
                subject = sentence[:m.start()].strip()
                verb = m.group(0).strip().rstrip()
                remainder = sentence[m.end():].strip()

                subject = self._clean_subject(subject)
                if subject:
                    obj = self._extract_object(remainder)
                    if obj:
                        return {"subject": subject, "verb": verb, "object": obj}
                    # 即使无宾语也返回主语+谓词
                    return {"subject": subject, "verb": verb, "object": ""}
        return None

    # ── 宾语提取 ─────────────────────────────

    def _extract_object(self, remainder: str) -> Optional[str]:
        """从谓词后的剩余文本中提取宾语"""
        if not remainder:
            return None

        # 检查剩余文本是否有宾语触发词（如"导致A→B"，B为跨句时识别）
        # 简单做法：取到句尾的前半部分（自然句号前）
        obj = remainder.rstrip("。.!?！？")
        
        # 如果剩余文本里还有系动词，可能是复合谓词，取后半
        for trigger in self.OBJECT_TRIGGERS:
            idx = obj.find(trigger)
            if idx > 0 and idx < len(obj) - len(trigger):
                # 触发词在中间 → 宾语在触发词之后
                obj_candidate = obj[idx + len(trigger):].strip()
                if len(obj_candidate) >= 4:
                    return obj_candidate[:200]  # 太长截断

        # 默认取整个剩余（视为宾语或描述）
        # 清理过长的宾语（可能是解释性内容）
        if len(obj) > 10 and len(obj) < 200:
            return obj.strip()
        elif len(obj) >= 200:
            # 截断到第一个句号
            end = obj.find("。")
            if end > 10:
                return obj[:end].strip()
            end = obj.find(".")
            if end > 10:
                return obj[:end].strip()

        return obj.strip() if len(obj) >= 4 else None

    def _find_next_verb(self, text: str) -> int:
        """找到文本中第一个谓词标记的位置"""
        if not text:
            return -1
        for marker in self.PREDICATE_MARKERS_ZH:
            idx = text.find(marker)
            if idx != -1:
                return idx
        return -1

    def _extract_verb(self, text: str) -> str:
        """从文本开头提取动词"""
        text = text.strip()
        for marker in self.PREDICATE_MARKERS_ZH:
            if text.startswith(marker):
                return marker
        return ""

    @staticmethod
    def _clean_subject(subject: str) -> str:
        """清理主语：去掉标点、语气词、长度检查"""
        # 去掉句首标点和空白
        subject = re.sub(r'^[「『\"\'【\s,，、]+', '', subject)
        # 去掉尾部标点
        subject = re.sub(r'[「『」」\"\'】\s,，、；]+$', '', subject)
        # 去掉引导性短语（中文）
        subject = re.sub(
            r'^(首先|其次|最后|另外|同时|此外|例如|比如|也就是|换句话说)',
            '', subject
        ).strip()
        # 去掉引导性短语（英文）
        subject = re.sub(
            r'^(First|Second|Third|Finally|Additionally|Meanwhile|For example|In other words)',
            '', subject, flags=re.IGNORECASE
        ).strip()

        # ── 噪声过滤 ──────────────────────────
        # 1. 以条件句中文字开头
        if re.match(r'^(如果|若|当|对于|关于|由于|虽然|尽管|无论|除非|因为)', subject):
            return ""
        # 2. 以条件英文开头
        if re.match(r'^(If|When|While|Although|Unless|Because|For|Since)\s', subject, re.IGNORECASE):
            return ""
        # 3. 包含代码/文件路径特征
        if re.search(r'[\\/].*\.', subject) or '\\' in subject or '/' in subject:
            return ""
        # 4. 包含换行符
        if '\n' in subject or '\r' in subject:
            return ""
        # 5. 纯符号或符号为主
        alpha_ratio = sum(c.isalpha() for c in subject) / max(len(subject), 1)
        if alpha_ratio < 0.4:
            return ""
        # 6. 以命令式标记开头
        if re.match(r'^(Let me|Please |Don'"'"'t|Click|Run|Execute|Set|Create|Add)', subject, re.IGNORECASE):
            return ""
        # 7. 包含markdown格式标记
        if re.search(r'[\[\](){}#*_`>]', subject):
            return ""
        # 8. 编程语言关键词
        code_kw = ['def ', 'class ', 'import ', 'return ', 'function', 'var ', 'let ', 'const ']
        if any(subject.startswith(kw) for kw in code_kw):
            return ""

        # 长度：太短或太长
        if len(subject) < 2 or len(subject) > 50:
            return ""
        return subject


# ──────────────────────────────────────────────
# 角色共现表
# ──────────────────────────────────────────────

class RoleTable:
    """角色共现表

    核心数据结构，管理实体-句子之间的角色映射。
    不修改 SphereStore 或 FAISS，是独立附加层。
    """

    def __init__(self, storage_path: Optional[str] = None):
        self._path = Path(storage_path or cfg_paths.connections_dir) / "role_table.json"
        self._entities: Dict[str, EntityInfo] = {}     # entity_id → EntityInfo
        self._text_to_id: Dict[str, str] = {}          # entity_text → entity_id
        self._sphere_entities: Dict[str, Set[str]] = {}  # sphere_id → {entity_id, ...}
        self._dirty = False
        self._extractor = SubjectExtractor()

    # ── 属性 ──────────────────────────────────

    @property
    def entity_count(self) -> int:
        """注册实体数"""
        return len(self._entities)

    @property
    def active_entity_count(self) -> int:
        """活跃实体数（出现≥2次）"""
        return sum(1 for e in self._entities.values() if e.is_active)

    @property
    def total_occurrences(self) -> int:
        """全部出现次数"""
        return sum(e.occurrences for e in self._entities.values())

    # ── 入库接口 ─────────────────────────────

    def register_text(self, sphere_id: str, text: str) -> int:
        """注册一个句子球体的主谓宾信息

        自动提取三元组，更新角色共现表。

        Args:
            sphere_id: 句子球体 ID（二级ID）
            text: 句子原文

        Returns:
            注册的实体数
        """
        triples = self._extractor.extract(text)
        count = 0

        for triple in triples:
            subject = triple.get("subject", "")
            obj = triple.get("object", "")

            if subject and len(subject) >= 2:
                self._add_role(subject, "subject", sphere_id)
                count += 1

            if obj and len(obj) >= 2:
                self._add_role(obj, "object", sphere_id)
                count += 1

        return count

    def _add_role(self, entity_text: str, role: str, sphere_id: str):
        """注册一个实体在某个句子中的角色"""
        entity_id = self._make_entity_id(entity_text)

        # 获取或创建实体
        if entity_id not in self._entities:
            self._entities[entity_id] = EntityInfo(text=entity_text[:60])
            self._text_to_id[entity_text] = entity_id

        info = self._entities[entity_id]

        # 记录角色
        if role == "subject":
            if sphere_id not in info.as_subject:
                info.as_subject.append(sphere_id)
                info.subject_count += 1
        elif role == "object":
            if sphere_id not in info.as_object:
                info.as_object.append(sphere_id)
                info.object_count += 1

        info.occurrences += 1

        # 记录句子→实体映射（用于反向查询）
        if sphere_id not in self._sphere_entities:
            self._sphere_entities[sphere_id] = set()
        self._sphere_entities[sphere_id].add(entity_id)

        # 记录共现：当前句子的其他实体
        self._update_co_occurrences(sphere_id, entity_id)

        self._dirty = True

    # ── 检索扩展 ─────────────────────────────

    def expand_from_sphere(
        self,
        sphere_id: str,
        max_candidates: int = 10,
        min_confidence: float = 0.3,
    ) -> List[JumpCandidate]:
        """从一个句子球体出发，通过角色模式扩展出相关候选

        扩展逻辑：
          1. 找到本句子中的所有宾语实体
          2. 对每个宾语实体，找到它做主语的其他句子
          3. 同一个实体在句A做宾语、在句B做主语 → A→B 是因果/推理链

        Args:
            sphere_id: 出发句子的球体 ID
            max_candidates: 最大扩展数量
            min_confidence: 最低置信度

        Returns:
            [JumpCandidate, ...] 按置信度降序
        """
        entities = self._sphere_entities.get(sphere_id, set())
        if not entities:
            return []

        candidates: Dict[str, JumpCandidate] = {}

        for eid in entities:
            info = self._entities.get(eid)
            if not info or not info.is_active:
                continue

            # ★ 核心：实体在本句作宾语 → 在目标句作主语
            if sphere_id in info.as_object:
                for target_id in info.as_subject:
                    if target_id == sphere_id:
                        continue
                    confidence = self._jump_confidence(
                        sphere_id, target_id, eid, info
                    )
                    if confidence < min_confidence:
                        continue
                    jc = JumpCandidate(
                        target_sphere_id=target_id,
                        bridge_entity=info.text,
                        bridge_type="object→subject",
                        confidence=confidence,
                    )
                    # 同一目标，保留最高置信度
                    key = (target_id, eid)
                    if key not in candidates or confidence > candidates[key].confidence:
                        candidates[key] = jc

            # 反向：实体在本句作主语 → 在目标句作宾语（可逆的推理链）
            if sphere_id in info.as_subject:
                for target_id in info.as_object:
                    if target_id == sphere_id:
                        continue
                    confidence = self._jump_confidence(
                        sphere_id, target_id, eid, info, reverse=True
                    )
                    if confidence < min_confidence:
                        continue
                    jc = JumpCandidate(
                        target_sphere_id=target_id,
                        bridge_entity=info.text,
                        bridge_type="subject→object",
                        confidence=confidence,
                    )
                    key = (target_id, eid)
                    if key not in candidates or confidence > candidates[key].confidence:
                        candidates[key] = jc

        # 按置信度降序排序
        result = sorted(candidates.values(), key=lambda x: -x.confidence)
        return result[:max_candidates]

    def expand_from_entities(
        self,
        entity_texts: List[str],
        max_candidates: int = 10,
    ) -> List[JumpCandidate]:
        """从实体集合出发，找到穿针引线的相关句子

        不限于主语-宾语模式，任何包含这些实体的句子都算。
        适用于查询关键词已知时的快速扩展。
        """
        candidates: Dict[str, JumpCandidate] = {}

        for et in entity_texts:
            eid = self._text_to_id.get(et)
            if not eid:
                continue
            info = self._entities.get(eid)
            if not info:
                continue

            # 所有包含该实体的句子
            all_sentences = set(info.as_subject) | set(info.as_object)
            for sid in all_sentences:
                if sid not in candidates:
                    candidates[sid] = JumpCandidate(
                        target_sphere_id=sid,
                        bridge_entity=info.text,
                        bridge_type="entity_match",
                        confidence=0.5,
                    )

        result = sorted(candidates.values(), key=lambda x: -x.confidence)
        return result[:max_candidates]

    def get_shared_entities(
        self, sphere_id_a: str, sphere_id_b: str
    ) -> List[str]:
        """获取两个句子共享的实体

        如果两个句子有多个共享实体 → 相关性更强
        """
        ea = self._sphere_entities.get(sphere_id_a, set())
        eb = self._sphere_entities.get(sphere_id_b, set())
        shared = ea & eb
        return [
            self._entities[eid].text
            for eid in shared
            if eid in self._entities
        ]

    def get_role_distribution(self, entity_text: str) -> Optional[Dict]:
        """获取一个实体的角色分布
        
        用于分析：某实体是偏主语型（主动）还是偏宾语型（被动）
        """
        eid = self._text_to_id.get(entity_text)
        if not eid or eid not in self._entities:
            return None
        info = self._entities[eid]
        return {
            "text": info.text,
            "occurrences": info.occurrences,
            "as_subject_count": info.subject_count,
            "as_object_count": info.object_count,
            "subject_ratio": info.subject_count / max(info.occurrences, 1),
        }

    # ── 共现分析 ─────────────────────────────

    def _update_co_occurrences(self, sphere_id: str, new_eid: str):
        """更新实体共现记录"""
        for other_eid in self._sphere_entities.get(sphere_id, set()):
            if other_eid == new_eid:
                continue
            info = self._entities.get(other_eid)
            if info:
                info.co_occurrences[new_eid] = \
                    info.co_occurrences.get(new_eid, 0) + 1

    # ── 置信度计算 ───────────────────────────

    @staticmethod
    def _jump_confidence(
        source_id: str,
        target_id: str,
        bridge_eid: str,
        bridge_info: EntityInfo,
        reverse: bool = False,
    ) -> float:
        """计算跳转置信度

        考虑因素：
          1. 桥接实体出现次数：≥3次 → 高置信
          2. 角色分布均衡度：偏科太严重（总做主语不做宾语）→ 置信略低
          3. 桥接质量：同一实体做的角色转换是强信号

        Returns:
            confidence ∈ [0.3, 0.95]
        """
        base = 0.5

        # 出现次数加分
        if bridge_info.occurrences >= 5:
            base += 0.2
        elif bridge_info.occurrences >= 3:
            base += 0.1

        # 角色均衡度加分
        total = bridge_info.subject_count + bridge_info.object_count
        if total >= 2:
            subject_ratio = bridge_info.subject_count / total
            # 越接近 0.5 越均衡（既能做主又能做宾 → 更好的桥接）
            balance = 1 - 2 * abs(subject_ratio - 0.5)
            base += 0.15 * balance

        # 反转时降 0.05（主语→宾语模式不如宾语→主语模式强）
        if reverse:
            base -= 0.05

        return min(0.95, max(0.3, base))

    # ── 持久化 ───────────────────────────────

    def save(self):
        """保存到 JSON"""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": ROLE_TABLE_VERSION,
            "entities": {
                eid: {
                    "text": info.text,
                    "occurrences": info.occurrences,
                    "as_subject": info.as_subject,
                    "as_object": info.as_object,
                    "subject_count": info.subject_count,
                    "object_count": info.object_count,
                    "co_occurrences": info.co_occurrences,
                }
                for eid, info in self._entities.items()
            },
            "sphere_entities": {
                sid: list(ents)
                for sid, ents in self._sphere_entities.items()
            },
        }

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._dirty = False
        logger.info(
            f"Saved role table: {len(self._entities)} entities, "
            f"{self.total_occurrences} occurrences"
        )

    def load(self) -> int:
        """从 JSON 加载"""
        if not self._path.exists():
            logger.info(f"No role table at {self._path}")
            return 0

        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", 0)
        if version > ROLE_TABLE_VERSION:
            raise ValueError(
                f"Role table version {version} > current {ROLE_TABLE_VERSION}"
            )

        self._entities.clear()
        self._text_to_id.clear()
        self._sphere_entities.clear()

        for eid, edata in data.get("entities", {}).items():
            info = EntityInfo(
                text=edata["text"],
                occurrences=edata.get("occurrences", 0),
                as_subject=edata.get("as_subject", []),
                as_object=edata.get("as_object", []),
                subject_count=edata.get("subject_count", 0),
                object_count=edata.get("object_count", 0),
                co_occurrences=edata.get("co_occurrences", {}),
            )
            self._entities[eid] = info
            self._text_to_id[info.text] = eid

        for sid, ents in data.get("sphere_entities", {}).items():
            self._sphere_entities[sid] = set(ents)

        logger.info(
            f"Loaded role table: {len(self._entities)} entities, "
            f"{self.total_occurrences} occurrences"
        )
        return len(self._entities)

    # ── 与检索系统集成 ───────────────────────

    def build_for_spheres(
        self,
        spheres: List,
        existing_table: Optional["RoleTable"] = None,
    ) -> "RoleTable":
        """从球体列表全量构建角色表

        Args:
            spheres: Sphere 列表（有 .id 和 .text 属性）
            existing_table: 已有角色表（增量更新时传入）

        Returns:
            self（已更新）
        """
        if existing_table:
            # 继承已有的实体数据
            self._entities = existing_table._entities
            self._text_to_id = existing_table._text_to_id
            self._sphere_entities = existing_table._sphere_entities

        for sphere in spheres:
            if not sphere.text or len(sphere.text.strip()) < 5:
                continue
            self.register_text(sphere.id, sphere.text)

        logger.info(
            f"Built role table from {len(spheres)} spheres: "
            f"{len(self._entities)} entities"
        )
        self._dirty = True
        return self

    def query_role_expansion(
        self,
        faiss_hit_ids: List[str],
        top_k: int = 5,
        min_confidence: float = 0.3,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """查询时的角色扩展：对 FAISS 检出的结果进行跳转扩展

        对每个 FAISS 命中的球体，查角色表，找到它能够
        通过宾语-主语桥接的其他球体，合并后去重。

        Args:
            faiss_hit_ids: FAISS 返回的球体 ID 列表
            top_k: 每个命中球体最多扩展几个候选
            min_confidence: 最小置信度

        Returns:
            {original_sphere_id: [(expanded_sphere_id, confidence), ...]}
        """
        expansion_map = {}

        for sid in faiss_hit_ids:
            jumps = self.expand_from_sphere(
                sid,
                max_candidates=top_k,
                min_confidence=min_confidence,
            )
            if jumps:
                expansion_map[sid] = [
                    (jc.target_sphere_id, jc.confidence)
                    for jc in jumps
                ]

        return expansion_map

    # ── 工具 ─────────────────────────────────

    @staticmethod
    def _make_entity_id(text: str) -> str:
        """从实体文本生成唯一 ID"""
        import hashlib
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

    # ── 状态 ─────────────────────────────────

    def stats(self) -> Dict:
        return {
            "entities": self.entity_count,
            "active_entities": self.active_entity_count,
            "total_occurrences": self.total_occurrences,
            "spheres_with_entities": len(self._sphere_entities),
            "dirty": self._dirty,
        }
