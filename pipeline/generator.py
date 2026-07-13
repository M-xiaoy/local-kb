"""
generator.py — 多后端答案生成器
================================
统一的生成接口，支持：
  - ollama  — 本地 LLM（qwen2.5:7b 等）
  - deepseek — DeepSeek V4 Pro 云端 API
  - agent    — 扩展接口（预留）

架构：
  每种后端实现一个 _generate_xxx 方法，
  generate() 是唯一公开入口，根据 model 参数路由。

RAG Prompt：
  标准指令：用给定的上下文回答问题，
  不知道就说不知道，不捏造信息。
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from config import generation as cfg

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# RAG 提示词模板
# ──────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """你是一个知识库问答助手。请基于以下上下文内容回答用户的问题。

要求：
1. 只使用上下文中提供的信息来回答
2. 如果上下文没有足够信息来回答问题，请明确告知，不要编造
3. 回答简洁准确，不需要额外解释
4. 引用上下文中的具体内容来支持你的回答

===== 上下文开始 =====
{context}
===== 上下文结束 =====

问题：{query}

回答："""

# ──────────────────────────────────────────────
# 回答结果
# ──────────────────────────────────────────────

@dataclass
class AnswerResult:
    text: str              # 生成的回答文本
    model: str             # 实际使用的模型名
    backend: str           # 后端类型（ollama/deepseek/agent）
    timing_ms: float       # 生成耗时
    context_spheres: List[Dict] = field(default_factory=list)  # 参考的球体信息
    tokens: Optional[Dict] = None      # token 统计（后端支持时）
    error: Optional[str] = None        # 错误信息（出错时）


# ──────────────────────────────────────────────
# 生成器
# ──────────────────────────────────────────────

class AnswerGenerator:
    """多后端答案生成器

    用法：
      gen = AnswerGenerator()
      result = gen.generate("什么是重力空间", contexts, model="deepseek")
      # → AnswerResult(text="...", model="deepseek-v4-pro", ...)
    """

    AVAILABLE_BACKENDS = {
        "ollama": "Ollama 本地模型（qwen2.5:7b）",
        "deepseek": "DeepSeek V4 Pro 云端 API",
        "agent": "自定义 Agent 接口（预留）",
    }

    def __init__(self):
        self._deepseek_api_key = (
            cfg.deepseek_api_key
            or os.environ.get("DEEPSEEK_API_KEY", "")
        )

    def list_backends(self) -> Dict[str, str]:
        """列出可用后端"""
        return dict(self.AVAILABLE_BACKENDS)

    def is_available(self, backend: str) -> bool:
        """检查某后端是否可用"""
        if backend == "ollama":
            return self._check_ollama()
        elif backend == "deepseek":
            return bool(self._deepseek_api_key)
        elif backend == "agent":
            return False  # 预留，暂不可用
        return False

    def generate(
        self,
        query: str,
        context_texts: List[str],
        model: Optional[str] = None,
        context_spheres: Optional[List[Dict]] = None,
    ) -> AnswerResult:
        """生成回答

        Args:
            query: 用户问题
            context_texts: 检索到的上下文片段列表
            model: 后端选择（ollama/deepseek/agent），默认从配置读取
            context_spheres: 可选的球体元数据（用于返回给前端展示）

        Returns:
            AnswerResult
        """
        backend = model or cfg.default_model

        # 组装上下文
        context = "\n\n---\n\n".join(
            f"[第{i + 1}段] {t}"
            for i, t in enumerate(context_texts)
        )

        prompt = RAG_PROMPT_TEMPLATE.format(query=query, context=context)

        # 路由到对应后端
        t0 = __import__("time").time()
        try:
            if backend == "ollama":
                text, model_name, tokens = self._generate_ollama(prompt)
            elif backend == "deepseek":
                text, model_name, tokens = self._generate_deepseek(prompt)
            elif backend == "agent":
                text, model_name, tokens = self._generate_agent(prompt)
            else:
                return AnswerResult(
                    text="",
                    model="",
                    backend=backend,
                    timing_ms=0,
                    error=f"未知后端: {backend}，可选: {', '.join(self.AVAILABLE_BACKENDS.keys())}",
                )
        except Exception as e:
            elapsed = (__import__("time").time() - t0) * 1000
            logger.error(f"Generation failed ({backend}): {e}")
            return AnswerResult(
                text="",
                model="",
                backend=backend,
                timing_ms=elapsed,
                error=str(e),
            )

        elapsed = (__import__("time").time() - t0) * 1000

        return AnswerResult(
            text=text,
            model=model_name,
            backend=backend,
            timing_ms=elapsed,
            context_spheres=context_spheres or [],
            tokens=tokens,
        )

    # ── Ollama 本地 ──────────────────────────

    def _generate_ollama(self, prompt: str) -> tuple:
        """调用 Ollama API 生成回答"""
        from config import ollama as cfg_ollama
        url = f"{cfg_ollama.host}/api/generate"

        payload = {
            "model": cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg.ollama_temperature,
                "num_predict": cfg.ollama_max_tokens,
            },
        }

        resp = httpx.post(url, json=payload, timeout=cfg.ollama_timeout)
        resp.raise_for_status()
        data = resp.json()

        return (
            data.get("response", ""),
            data.get("model", cfg.ollama_model),
            {"prompt_tokens": data.get("prompt_eval_count", 0),
             "completion_tokens": data.get("eval_count", 0)},
        )

    # ── DeepSeek 云端 ───────────────────────

    def _generate_deepseek(self, prompt: str) -> tuple:
        """调用 DeepSeek API（OpenAI 兼容格式）生成回答"""
        if not self._deepseek_api_key:
            raise ValueError(
                "DeepSeek API key 未配置。请设置环境变量 DEEPSEEK_API_KEY "
                "或在 config.py 的 generation.deepseek_api_key 中填写。"
            )

        url = f"{cfg.deepseek_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._deepseek_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg.deepseek_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个知识库问答助手，基于给定的上下文回答问题。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": cfg.deepseek_temperature,
            "max_tokens": cfg.deepseek_max_tokens,
        }

        resp = httpx.post(url, json=payload, headers=headers, timeout=cfg.deepseek_timeout)
        resp.raise_for_status()
        data = resp.json()

        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")

        usage = data.get("usage", {})
        return (
            text,
            data.get("model", cfg.deepseek_model),
            {"prompt_tokens": usage.get("prompt_tokens", 0),
             "completion_tokens": usage.get("completion_tokens", 0)},
        )

    # ── Agent 接口（预留） ──────────────────

    def _generate_agent(self, prompt: str) -> tuple:
        """预留的自定义 Agent 接口

        此处可接入 OpenClaw 子代理或其他生成服务。
        当前返回未实现的占位信息。
        """
        raise NotImplementedError(
            "Agent 后端尚未实现。可在此处接入自定义生成服务。"
        )

    # ── 健康检查 ─────────────────────────────

    def _check_ollama(self) -> bool:
        """检查 Ollama 是否正在运行"""
        try:
            from config import ollama as cfg_ollama
            resp = httpx.get(f"{cfg_ollama.host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return any(cfg.ollama_model in m.get("name", "") for m in models)
        except Exception:
            return False


# ──────────────────────────────────────────────
# 单例
# ──────────────────────────────────────────────

_global_generator: Optional[AnswerGenerator] = None


def get_generator() -> AnswerGenerator:
    global _global_generator
    if _global_generator is None:
        _global_generator = AnswerGenerator()
    return _global_generator


def generate(query: str, contexts: List[str], **kwargs) -> AnswerResult:
    return get_generator().generate(query, contexts, **kwargs)
