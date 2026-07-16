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
    connections_dir: str = "data/connections/"


# ──────────────────────────────────────────────
# Ollama 配置
# ──────────────────────────────────────────────
@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    embed_timeout: int = 30
    llm_timeout: int = 120

    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "

    embed_batch_size: int = 16
    embed_cache_size: int = 10000


# ──────────────────────────────────────────────
# 切片器配置
# ──────────────────────────────────────────────
@dataclass
class ChunkerConfig:
    mode: str = "recursive"
    max_chunk_chars: int = 2000
    chunk_overlap: int = 300
    separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。\n", ". ", "！", "？", "!", "?", "；", ";", "，", ",", " ", "",
    ])

    # 按 source_type 的差异化策略
    strategy_overrides: dict = field(default_factory=lambda: {
        "会话记录": {"mode": "section", "max_chars": 1500, "overlap": 200},
        "技术笔记": {"mode": "markdown", "max_chars": 2500, "overlap": 200},
        "会话记录_重写": {"mode": "section", "max_chars": 1500, "overlap": 200},
        "学术论文": {"mode": "causal_density", "max_chars": 2000, "overlap": 200},
    })


# ──────────────────────────────────────────────
# 检索配置
# ──────────────────────────────────────────────
@dataclass
class RetrievalConfig:
    faiss_top_k: int = 100
    final_top_k: int = 5
    field_match_threshold: float = 0.3
    diversity_weight: float = 0.4
    similarity_weight: float = 0.6


# ──────────────────────────────────────────────
# Web 服务配置
# ──────────────────────────────────────────────
@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8766
    upload_max_size_mb: int = 50


# ──────────────────────────────────────────────
# 聚类配置
# ──────────────────────────────────────────────
@dataclass
class ClusteringConfig:
    n_clusters: int = 5
    auto_detect_k: bool = True
    max_k: int = 20
    max_iter: int = 100
    random_state: int = 42
    n_init: int = 10
    cluster_threshold: float = 0.3
    state_file: str = "data/clusters/cluster_state.json"
    label_map_file: str = "data/clusters/cluster_labels.json"


# ──────────────────────────────────────────────
# 答案生成配置
# ──────────────────────────────────────────────
@dataclass
class GenerationConfig:
    default_model: str = "ollama"
    ollama_model: str = "qwen2.5:7b"
    ollama_temperature: float = 0.3
    ollama_max_tokens: int = 2048
    ollama_timeout: int = 120

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_temperature: float = 0.3
    deepseek_max_tokens: int = 2048
    deepseek_timeout: int = 60


# ──────────────────────────────────────────────
# 文本重写配置（Phase 0.1 新增）
# ──────────────────────────────────────────────
@dataclass
class RewriterConfig:
    """入库前文本重写器参数"""
    enabled: bool = True
    llm_model: str = "qwen2.5:7b"
    # 走全量重写的 source_type（LLM结构化+指代消解）
    full_strategies: List[str] = field(default_factory=lambda: [
        "会话记录", "其他", ""
    ])
    # 走轻量重写的 source_type（只做实体提取）
    light_strategies: List[str] = field(default_factory=lambda: [
        "技术笔记", "学术论文", "工作文档"
    ])
    max_input_chars: int = 4000
    batch_delay: float = 0.5           # Ollama 批次间冷却时间（秒）
    timeout: int = 60


# ──────────────────────────────────────────────
# 球体连接配置（Phase 0.3 新增）
# ──────────────────────────────────────────────
@dataclass
class ConnectionConfig:
    """球体间关系检测参数"""
    enabled: bool = True
    same_cluster_topk: int = 3          # 同簇取 Top-N 建连接
    same_cluster_weight: float = 0.6    # 同簇连接权重
    entity_threshold: int = 2           # 共享实体 ≥ 此数则建连接
    entity_weight: float = 0.4          # 实体重叠连接权重
    embedding_threshold: float = 0.50   # 跨簇语义相似度阈值（v2 实验：0.65 太紧）
    embedding_weight: float = 0.3       # 跨簇语义连接权重
    temporal_weight: float = 0.25       # 时序相邻连接权重
    cross_cluster_weight: float = 0.2   # 跨簇桥接权重
    min_weight: float = 0.1             # 低于此值不建连接
    prune_threshold: float = 0.05       # 季度修剪阈值
    max_connections_per_sphere: int = 50  # 单球体连接数上限
    decay_per_tick: float = 0.98        # 每 tick 连接衰减系数
    storage_dir: str = "data/connections/" # 连接表持久化目录
    batch_build_size: int = 50          # 批量构建时的批次大小


# ──────────────────────────────────────────────
# 轴突连接配置（Phase 1.2 新增 — 因果链检测）
# ──────────────────────────────────────────────
@dataclass
class AxonConfig:
    """段落内因果链 → 轴突连接的参数"""
    enabled: bool = True
    max_sentence_distance: int = 3       # 因句向后找果句的最大句数
    axon_weight_strong: float = 0.6     # 强因果标记（propose/demonstrate）的连接权重
    axon_weight_weak: float = 0.35      # 弱因果标记（suggest/indicate）的连接权重
    decay_angle_threshold: float = 55.0 # 衰减临界角阈值（实验验证值）
    use_embedding_verify: bool = True   # 是否用向量相似度验证因果对


# ──────────────────────────────────────────────
# 激活传播配置（Phase 1.1 新增）
# ──────────────────────────────────────────────
@dataclass
class ActivationConfig:
    """检索时球体激活传播参数"""
    enabled: bool = True
    max_hops: int = 2                   # 最大传播跳数
    decay_factor: float = 0.5           # 每跳信号衰减系数
    seed_activation_threshold: float = 0.05   # 种子激活阈值
    min_propagated: float = 0.02        # 传播信号最低保留值
    max_candidates_before: int = 100    # 传播前的候选数（FAISS返回数）
    max_candidates_after: int = 150     # 传播后的候选数上限


# ──────────────────────────────────────────────
# 重排序配置（Phase 1.4 新增）
# ──────────────────────────────────────────────
@dataclass
class RerankerConfig:
    """检索后候选重排序参数"""
    enabled: bool = True
    method: str = "ollama"              # ollama | cross-encoder
    model: str = "qwen2.5:7b"           # ollama 方案用此模型
    candidate_count: int = 50           # 重排前截断到此数
    top_k_after: int = 20               # 重排后保留数量
    batch_size: int = 5                 # Ollama 每次评分几个候选


# ──────────────────────────────────────────────
# 角色共现表配置（Phase 0.5 新增）
# ──────────────────────────────────────────────
@dataclass
class RoleConfig:
    """角色共现表 + 检索时扩展参数"""
    enabled: bool = True
    min_entity_length: int = 2          # 实体最短字数
    max_entity_length: int = 60         # 实体最长字数
    min_confidence: float = 0.3         # 跳转最低置信度
    decay_factor: float = 0.6           # 扩展得分衰减系数
    max_expansions_per_hit: int = 3     # 每命中最大扩展数
    total_max_expansions: int = 20      # 总扩展上限
    expand_after_faiss: bool = True     # 检索时是否启用角色扩展


# ──────────────────────────────────────────────
# 球体质量校准配置（Phase 0.2 新增）
# ──────────────────────────────────────────────
@dataclass
class CalibratorConfig:
    """mass/diversity 校准参数"""
    mass_base: float = 1.0
    mass_connection_factor: float = 0.3   # 连接度对 mass 的贡献
    mass_max_multiplier: float = 3.0      # mass 最大值倍数
    diversity_effective_factor: float = 0.5  # diversity 对 effective_mass 的贡献系数


# ──────────────────────────────────────────────
# 场域配置（历史兼容）
# ──────────────────────────────────────────────
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
role = RoleConfig()

paths = Paths()
ollama = OllamaConfig()
chunker = ChunkerConfig()
retrieval = RetrievalConfig()
web = WebConfig()
clustering = ClusteringConfig()
generation = GenerationConfig()
rewriter = RewriterConfig()
connection = ConnectionConfig()
axon = AxonConfig()
activation = ActivationConfig()
reranker = RerankerConfig()
calibrator = CalibratorConfig()
