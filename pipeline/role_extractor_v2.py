"""
role_extractor_v2.py — 主语抽取 v2：spaCy 优先 + 规则兜底
==========================================================
如果 spaCy 中文模型可用 → 用依存句法（更准）
如果不可用 → 回退规则引擎（不阻塞）
"""
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 尝试加载 spaCy ──
_SPACY_ZH = None
try:
    import spacy
    _SPACY_ZH = spacy.load("zh_core_web_sm")
    logger.info("spaCy zh_core_web_sm loaded")
except Exception:
    logger.info("spaCy zh model not available, using rule fallback")
    from pipeline.role_table import SubjectExtractor


# ── 主语清扫（规则+spaCy 共享） ──
CLEANUP_SUFFIXES = {'通','过','在','把','被','将','从','对','与','和',
                     '研','究','表','明','指','出','显','示','说',
                     '为','了','着','过','基','于'}

REJECT_PATTERNS = [
    r'^(我们|他们|它们|你们|这个|那个|这些|那些|自己|它|他|她|这|那|本文|本工作|本研究|本论文)$',
    r'^.$',
]


def clean_subject(s: str) -> str:
    if not s:
        return ""
    while len(s) > 2 and s[-1] in CLEANUP_SUFFIXES:
        s = s[:-1].strip()
    for pat in REJECT_PATTERNS:
        if re.match(pat, s):
            return ""
    return s


# ── spaCy 抽取 ──

def extract_spacy(sentence: str) -> Optional[Dict[str, str]]:
    """用依存句法提取主谓宾"""
    if _SPACY_ZH is None:
        return None
    doc = _SPACY_ZH(sentence)
    subject = None
    obj = None
    for token in doc:
        if token.dep_ == "nsubj" and not subject:
            # 取包含所有修饰的完整名词短语
            subject = "".join(
                t.text_with_ws for t in token.subtree
                if t.dep_ in ("nsubj", "amod", "compound", "det", "nummod")
                or t == token
            ).strip()
        if token.dep_ in ("dobj", "attr", "pobj") and not obj:
            obj = token.text
    if subject:
        return {"subject": subject, "verb": "", "object": obj or ""}
    return None


# ── 混合抽取器 ──

class HybridExtractor:
    def __init__(self):
        self._rule_ext = None
        if _SPACY_ZH is None:
            from pipeline.role_table import SubjectExtractor
            self._rule_ext = SubjectExtractor()
        self.stats = {"total": 0, "spacy_ok": 0, "rule_ok": 0, "failed": 0}

    def extract(self, text: str) -> List[Dict[str, str]]:
        results = []
        # 分割句子
        import re as _re
        sentences = [s.strip() for s in _re.split(r'(?<=[。！？.!?])\s*', text) if len(s.strip()) > 5]
        for sent in sentences:
            t = self._extract_one(sent)
            if t:
                results.append(t)
        return results

    def _extract_one(self, sentence: str) -> Optional[Dict[str, str]]:
        self.stats["total"] += 1

        # spaCy 优先
        if _SPACY_ZH is not None:
            t = extract_spacy(sentence)
            if t:
                subj = clean_subject(t["subject"])
                if subj:
                    self.stats["spacy_ok"] += 1
                    return {"subject": subj, "verb": t.get("verb", ""), "object": t.get("object", "")}

        # 规则兜底
        if self._rule_ext:
            t = self._rule_ext._extract_one(sentence)
            if t:
                subj = clean_subject(t.get("subject", ""))
                if subj:
                    self.stats["rule_ok"] += 1
                    return {"subject": subj, "verb": t.get("verb", ""), "object": t.get("object", "")}

        self.stats["failed"] += 1
        return None

    def report(self) -> Dict:
        return self.stats
