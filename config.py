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
    wal_dir: str = "data/wal/"


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
# 聚类配置
# ──────────────────────────────────────────────
@dataclass
class ClusteringConfig:
    """k-means 聚类参数

    场域不再由用户打标签定义，而是系统通过聚类自动发现。
    上传完成后触发全量重聚类，更新每个球体的簇归属。
    """
    n_clusters: int = 3            # 聚簇数量（auto_detect=True 时作为参考/最小值）
    auto_detect_k: bool = True     # 是否自动检测最优 K 值（silhouette score）
    max_k: int = 20                # 自动检测时的最大 K 上限
    max_iter: int = 100            # 最大迭代次数
    random_state: int = 42         # 固定种子，保证可复现
    n_init: int = 10               # 初始化次数（选最佳）
    cluster_threshold: float = 0.3 # 低于此值的球体视为未分配
    # 存储路径
    state_file: str = "data/clusters/cluster_state.json"
    label_map_file: str = "data/clusters/cluster_labels.json"


# ──────────────────────────────────────────────
# 答案生成配置
# ──────────────────────────────────────────────
@dataclass
class GenerationConfig:
    """多后端生成器配置

    model 取值:
      ollama   — 本地 Ollama LLM（默认 qwen2.5:7b）
      deepseek — DeepSeek V4 Pro 云端 API
      agent    — 扩展接口（预留，接入自定义生成服务）
    """
    default_model: str = "ollama"           # ollama | deepseek | agent

    # --- Ollama 本地 ---
    ollama_model: str = "qwen2.5:7b"
    ollama_temperature: float = 0.3
    ollama_max_tokens: int = 2048
    ollama_timeout: int = 120

    # --- DeepSeek 云端 ---
    deepseek_api_key: str = ""               # 从环境变量读取，或手动填
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_temperature: float = 0.3
    deepseek_max_tokens: int = 2048
    deepseek_timeout: int = 60


# ──────────────────────────────────────────────
# 场域配置（已迁移至聚类自动发现）
# ──────────────────────────────────────────────
# 此列表不再作为场域定义，仅用于：
#   1. 历史数据兼容（旧 source_type 保留不删）
#   2. 聚类结果的初始命名参考
# 空列表不影响系统运行，聚类引擎会自行发现簇。
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
clustering = ClusteringConfig()
generation = GenerationConfig()
