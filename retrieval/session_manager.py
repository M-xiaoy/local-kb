"""
session_manager.py — 会话管理器
================================
轻量级会话状态容器，管理：
  - field_focus: 当前聚焦的场域（None=未聚焦）
  - exclude_ids: 已返回过的球体 ID

不存对话历史，不做持久化，重启即丢。
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 会话上下文
# ──────────────────────────────────────────────

class SessionContext:
    """单个会话的状态"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.field_focus: Optional[str] = None
        self.exclude_ids: Set[str] = set()
        self.created_at: float = time.time()
        self.last_active: float = time.time()

    def set_focus(self, field: str):
        """进入某场域"""
        self.field_focus = field
        self.last_active = time.time()

    def reset_focus(self):
        """退出场域，回到初始态"""
        self.field_focus = None
        self.last_active = time.time()

    def add_excluded(self, ids: List[str]):
        """记录已返回的球体 ID"""
        self.exclude_ids.update(ids)
        self.last_active = time.time()

    def is_expired(self, timeout_seconds: int = 1800) -> bool:
        """30 分钟无操作自动过期"""
        return (time.time() - self.last_active) > timeout_seconds


# ──────────────────────────────────────────────
# 会话管理器
# ──────────────────────────────────────────────

class SessionManager:
    """管理所有活跃会话"""

    def __init__(self, timeout_seconds: int = 1800):
        self._sessions: Dict[str, SessionContext] = {}
        self._timeout = timeout_seconds

    def get_or_create(self, session_id: Optional[str] = None) -> SessionContext:
        """获取已有会话或创建新会话"""
        if session_id and session_id in self._sessions:
            ctx = self._sessions[session_id]
            if not ctx.is_expired(self._timeout):
                ctx.last_active = time.time()
                return ctx
            # 过期了，清理后重建
            del self._sessions[session_id]

        # 创建新会话
        new_id = session_id or uuid.uuid4().hex[:12]
        ctx = SessionContext(new_id)
        self._sessions[new_id] = ctx
        return ctx

    def cleanup_expired(self):
        """清理过期会话"""
        expired = [
            sid for sid, ctx in self._sessions.items()
            if ctx.is_expired(self._timeout)
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    @property
    def active_count(self) -> int:
        return len(self._sessions)
