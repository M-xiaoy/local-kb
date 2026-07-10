"""
重力知识库 — 配置模块
====================
所有可调参数集中管理，不散落在各模块中。
"""

from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────
@dataclass
class Paths:
    faiss_index: str = "data/index/faiss.index"
    faiss_cache: str = "data/index/faiss_cache.npz"
    spheres_data: str = "data/spheres/spheres.json"
    registry_map: str = "data/spheres/registry.json"
    uploads_dir: str = "data/uploads/"


# ──────────────────────────────────────────────
# Ollama 配置
# ──────────────────────────────────────────────
@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"       # 嵌入模型
    embed_dim: int = 768                         # 模型输出维度
    llm_model: str = "qwen2.5:7b"               # 回答生成模型
    llm_temperature: float = 0.3                 # 低温度→稳定回答
    llm_max_tokens: int = 2048
    embed_timeout: int = 30                      # 嵌入请求超时（秒）
    llm_timeout: int = 120                       # 生成请求超时（秒）

    # nomic-embed-text 的 task prefix（不加会影响检索精度）
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "

    # 批处理参数
    embed_batch_size: int = 16                   # 一批最多处理多少条

    # 缓存
    embed_cache_size: int = 10000               # 最大缓存条目数


# ──────────────────────────────────────────────
# 切片器配置
# ──────────────────────────────────────────────
@dataclass
class ChunkerConfig:
    """
    基于 Recursive Character Chunking（LangChain 标准方案）

    递归降级策略：
      \n\n → \n → 句子边界 → 长词 → 字符
    每次按当前最高优先级分隔符切分，块太大则降级到下一级分隔符。
    """
    mode: str = "recursive"                     # recursive | markdown | fixed
    max_chunk_chars: int = 800                  # 目标块上限（字符数）
    chunk_overlap: int = 100                    # 相邻块重叠字符数
    # 递归分隔符优先级（从高到低）
    separators: List[str] = field(default_factory=lambda: [
        "\n\n",   # 段落边界（最高优先级）
        "\n",      # 行边界
        "。\n",    # 中文句号
        ". ",      # 英文句点
        "！", "？", # 中文感叹/疑问
        "!", "?",   # 英文感叹/疑问
        "；", ";",  # 分号
        "，", ",",  # 逗号
        " ",        # 空格（词边界）
        "",         # 字符级硬切（最后退路）
    ])
    # 不做 min_chunk 过滤——短块也是有效信息单元


# ──────────────────────────────────────────────
# 检索配置
# ──────────────────────────────────────────────
@dataclass
class RetrievalConfig:
    faiss_top_k: int = 100                      # FAISS 粗搜返回数量
    final_top_k: int = 5                        # 多样性排序后最终输出数量
    field_match_threshold: float = 0.3          # 低于此值的场域匹配度不计分
    diversity_weight: float = 0.4               # 多样性在最终评分中的权重
    similarity_weight: float = 0.6              # FAISS 相似度在最终评分中的权重
    # 场域匹配度 = 1 - 权重，所以场域权重 = similarity_weight × (1 - 权重偏移)


# ──────────────────────────────────────────────
# Web 服务配置
# ──────────────────────────────────────────────
@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    upload_max_size_mb: int = 50


# ──────────────────────────────────────────────
# 场域配置
# ──────────────────────────────────────────────
# 用户上传文件时选择标签，查询时选择目标场域
# 空列表 = 未启用场域功能（全部统一权重）
AVAILABLE_FIELDS: List[str] = [
    "技术笔记",
    "小说创作",
    "学术论文",
    "工作文档",
    "其他",
]


# ──────────────────────────────────────────────
# 全局单例
# ──────────────────────────────────────────────
paths = Paths()
ollama = OllamaConfig()
chunker = ChunkerConfig()
retrieval = RetrievalConfig()
web = WebConfig()
