"""
## README
Session management for Gemini Multimodal Live Proxy Server.

## CHANGELOG

"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import asyncio


@dataclass
class SessionState:
    """
    跟踪客户端会话状态的类

    属性:
        is_receiving_response: 是否正在接收AI响应
        interrupted: 会话是否被中断
        current_tool_execution: 当前正在执行的任务
        current_audio_stream: 当前的音频流
        genai_session: Gemini AI会话实例
        received_model_response: 在当前轮次中是否已收到模型响应
    """

    is_receiving_response: bool = False
    interrupted: bool = False
    current_tool_execution: Optional[asyncio.Task] = None
    current_audio_stream: Optional[Any] = None
    genai_session: Optional[Any] = None
    received_model_response: bool = False


# 全局会话存储字典，用于保存所有活跃的会话
active_sessions: Dict[str, SessionState] = {}


def create_session(session_id: str) -> SessionState:
    """
    创建并存储一个新的会话

    参数:
        session_id: 会话的唯一标识符

    返回:
        新创建的SessionState实例
    """
    session = SessionState()
    active_sessions[session_id] = session
    return session


def get_session(session_id: str) -> Optional[SessionState]:
    """
    获取已存在的会话

    参数:
        session_id: 要获取的会话ID

    返回:
        如果找到则返回SessionState实例，否则返回None
    """
    return active_sessions.get(session_id)


def remove_session(session_id: str) -> None:
    """
    移除指定的会话

    参数:
        session_id: 要移除的会话ID
    """
    if session_id in active_sessions:
        del active_sessions[session_id]
