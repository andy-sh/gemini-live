"""
## README
WebSocket message handling for Gemini Multimodal Live Proxy Server.

## CHANGELOG
### 20250606
- 去除对工具的依赖.
"""

import logging
import json
import asyncio
import base64
import traceback
from typing import Any, Optional
from google.genai import types

from core.session import create_session, remove_session, SessionState
from core.gemini_client import create_gemini_session

# 日志记录器.
logger = logging.getLogger(__name__)


async def send_error_message(websocket: Any, error_data: dict) -> None:
    """
    向客户端发送格式化的错误消息

    Args:
        websocket: WebSocket连接对象
        error_data: 包含错误信息的字典
    """
    try:
        await websocket.send(json.dumps({"type": "error", "data": error_data}))
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")


async def cleanup_session(session: Optional[SessionState], session_id: str) -> None:
    """
    清理会话资源

    Args:
        session: 会话状态对象
        session_id: 会话ID
    """
    try:
        if session:
            # 取消正在运行的任务
            if session.current_tool_execution:
                session.current_tool_execution.cancel()
                try:
                    await session.current_tool_execution
                except asyncio.CancelledError:
                    pass

            # 关闭Gemini会话
            if session.genai_session:
                try:
                    await session.genai_session.close()
                except Exception as e:
                    logger.error(f"Error closing Gemini session: {e}")

            # 从活动会话中移除
            remove_session(session_id)
            logger.info(f"Session {session_id} cleaned up and ended")
    except Exception as cleanup_error:
        logger.error(f"Error during session cleanup: {cleanup_error}")


async def handle_messages(websocket: Any, session: SessionState) -> None:
    """
    处理客户端和Gemini之间的双向消息流

    Args:
        websocket: WebSocket连接对象
        session: 会话状态对象
    """
    client_task = None
    gemini_task = None

    try:
        async with asyncio.TaskGroup() as tg:
            # 任务1: 处理来自客户端的消息
            client_task = tg.create_task(handle_client_messages(websocket, session))
            # 任务2: 处理来自Gemini的响应
            gemini_task = tg.create_task(handle_gemini_responses(websocket, session))
    except Exception as eg:
        handled = False
        for exc in eg.exceptions:
            if "Quota exceeded" in str(exc):
                logger.info("Quota exceeded error occurred")
                try:
                    # 发送错误消息供UI处理
                    await send_error_message(
                        websocket,
                        {
                            "message": "Quota exceeded.",
                            "action": "Please wait a moment and try again in a few minutes.",
                            "error_type": "quota_exceeded",
                        },
                    )
                    # 发送文本消息显示在聊天中
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "text",
                                "data": "⚠️ Quota exceeded. Please wait a moment and try again in a few minutes.",
                            }
                        )
                    )
                    handled = True
                    break
                except Exception as send_err:
                    logger.error(f"Failed to send quota error message: {send_err}")
            elif "connection closed" in str(exc).lower():
                logger.info("WebSocket connection closed")
                handled = True
                break

        if not handled:
            # 对于其他错误，记录并重新抛出
            logger.error(f"Error in message handling: {eg}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
    finally:
        # 如果任务仍在运行，取消它们
        if client_task and not client_task.done():
            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass

        if gemini_task and not gemini_task.done():
            gemini_task.cancel()
            try:
                await gemini_task
            except asyncio.CancelledError:
                pass


async def handle_client_messages(websocket: Any, session: SessionState) -> None:
    """
    处理来自客户端的消息

    Args:
        websocket: WebSocket连接对象
        session: 会话状态对象
    """
    try:
        async for message in websocket:
            try:
                data = json.loads(message)

                if "type" in data:
                    msg_type = data["type"]
                    if msg_type == "audio":
                        logger.debug("Client -> Gemini: Sending audio data...")
                    elif msg_type == "image":
                        logger.debug("Client -> Gemini: Sending image data...")
                    else:
                        # 在调试输出中用占位符替换音频数据
                        debug_data = data.copy()
                        if "data" in debug_data and debug_data["type"] == "audio":
                            debug_data["data"] = "<audio data>"
                        logger.debug(
                            f"Client -> Gemini: {json.dumps(debug_data, indent=2)}"
                        )

                # 处理不同类型的输入
                if "type" in data:
                    if data["type"] == "audio":
                        logger.debug("Sending audio to Gemini...")
                        await session.genai_session.send(
                            input={"data": data.get("data"), "mime_type": "audio/pcm"},
                            end_of_turn=True,
                        )
                        logger.debug("Audio sent to Gemini")
                    elif data["type"] == "image":
                        logger.info("Sending image to Gemini...")
                        await session.genai_session.send(
                            input={"data": data.get("data"), "mime_type": "image/jpeg"}
                        )
                        logger.info("Image sent to Gemini")
                    elif data["type"] == "text":
                        logger.info("Sending text to Gemini...")
                        await session.genai_session.send(
                            input=data.get("data"), end_of_turn=True
                        )
                        logger.info("Text sent to Gemini")
                    elif data["type"] == "end":
                        logger.info("Received end signal")
                    else:
                        logger.warning(f"Unsupported message type: {data.get('type')}")
            except Exception as e:
                logger.error(f"Error handling client message: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
    except Exception as e:
        if "connection closed" not in str(e).lower():  # 不记录正常的连接关闭
            logger.error(f"WebSocket connection error: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise  # 重新抛出以让父级处理清理


async def handle_gemini_responses(websocket: Any, session: SessionState) -> None:
    """
    处理来自Gemini的响应

    Args:
        websocket: WebSocket连接对象
        session: 会话状态对象
    """
    tool_queue = asyncio.Queue()  # 工具响应的队列

    # 启动后台任务处理工具调用
    tool_processor = asyncio.create_task(
        process_tool_queue(tool_queue, websocket, session)
    )

    try:
        while True:
            async for response in session.genai_session.receive():
                try:
                    # 在调试输出中用占位符替换音频数据
                    debug_response = str(response)
                    if (
                        "data=" in debug_response
                        and "mime_type='audio/pcm" in debug_response
                    ):
                        debug_response = (
                            debug_response.split("data=")[0]
                            + "data=<audio data>"
                            + debug_response.split("mime_type=")[1]
                        )
                    logger.debug(f"Received response from Gemini: {debug_response}")

                    # 如果有工具调用，将其添加到队列并继续
                    if response.tool_call:
                        await tool_queue.put(response.tool_call)
                        continue  # 在工具执行时继续处理其他响应

                    # 立即处理服务器内容（包括音频）
                    await process_server_content(
                        websocket, session, response.server_content
                    )

                except Exception as e:
                    logger.error(f"Error handling Gemini response: {e}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    finally:
        # 取消并清理工具处理器
        if tool_processor and not tool_processor.done():
            tool_processor.cancel()
            try:
                await tool_processor
            except asyncio.CancelledError:
                pass

        # 清空队列中的剩余项
        while not tool_queue.empty():
            try:
                tool_queue.get_nowait()
                tool_queue.task_done()
            except asyncio.QueueEmpty:
                break


async def process_tool_queue(
    queue: asyncio.Queue, websocket: Any, session: SessionState
):
    """
    处理队列中的工具调用

    Args:
        queue: 工具调用队列
        websocket: WebSocket连接对象
        session: 会话状态对象
    """
    while True:
        tool_call = await queue.get()
        try:
            function_responses = []
            for function_call in tool_call.function_calls:
                # 在会话状态中存储工具执行
                session.current_tool_execution = asyncio.current_task()

                # 向客户端发送函数调用（用于UI反馈）
                await websocket.send(
                    json.dumps(
                        {
                            "type": "function_call",
                            "data": {
                                "name": function_call.name,
                                "args": function_call.args,
                            },
                        }
                    )
                )

                tool_result = "This is a test tool result."

                # 向客户端发送函数响应
                await websocket.send(
                    json.dumps({"type": "function_response", "data": tool_result})
                )

                function_responses.append(
                    types.FunctionResponse(
                        name=function_call.name,
                        id=function_call.id,
                        response=tool_result,
                    )
                )

                session.current_tool_execution = None

            if function_responses:
                tool_response = types.LiveClientToolResponse(
                    function_responses=function_responses
                )
                await session.genai_session.send(input=tool_response)
        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
        finally:
            queue.task_done()


async def process_server_content(
    websocket: Any, session: SessionState, server_content: Any
):
    """
    处理服务器内容，包括音频和文本

    Args:
        websocket: WebSocket连接对象
        session: 会话状态对象
        server_content: 服务器内容对象
    """
    # 首先检查中断
    if hasattr(server_content, "interrupted") and server_content.interrupted:
        logger.info("Interruption detected from Gemini")
        await websocket.send(
            json.dumps(
                {
                    "type": "interrupted",
                    "data": {"message": "Response interrupted by user input"},
                }
            )
        )
        session.is_receiving_response = False
        return

    if server_content.model_turn:
        session.received_model_response = True
        session.is_receiving_response = True
        for part in server_content.model_turn.parts:
            if part.inline_data:
                audio_base64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                await websocket.send(
                    json.dumps({"type": "audio", "data": audio_base64})
                )
            elif part.text:
                await websocket.send(json.dumps({"type": "text", "data": part.text}))

    if server_content.turn_complete:
        await websocket.send(json.dumps({"type": "turn_complete"}))
        session.received_model_response = False
        session.is_receiving_response = False


async def handle_client(websocket: Any) -> None:
    """
    处理新的客户端连接

    Args:
        websocket: WebSocket连接对象
    """
    session_id = str(id(websocket))
    session = create_session(session_id)

    try:
        # 创建并初始化Gemini会话
        async with await create_gemini_session() as gemini_session:
            session.genai_session = gemini_session

            # 向客户端发送就绪消息
            await websocket.send(json.dumps({"ready": True}))
            logger.info(f"New session started: {session_id}")

            try:
                # 开始消息处理
                await handle_messages(websocket, session)
            except Exception as e:
                if (
                    "code = 1006" in str(e)
                    or "connection closed abnormally" in str(e).lower()
                ):
                    logger.info(
                        f"Browser disconnected or refreshed for session {session_id}"
                    )
                    await send_error_message(
                        websocket,
                        {
                            "message": "Connection closed unexpectedly",
                            "action": "Reconnecting...",
                            "error_type": "connection_closed",
                        },
                    )
                else:
                    raise

    except asyncio.TimeoutError:
        logger.info(
            f"Session {session_id} timed out - this is normal for long idle periods"
        )
        await send_error_message(
            websocket,
            {
                "message": "Session timed out due to inactivity.",
                "action": "You can start a new conversation.",
                "error_type": "timeout",
            },
        )
    except Exception as e:
        logger.error(f"Error in handle_client: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        if "connection closed" in str(e).lower() or "websocket" in str(e).lower():
            logger.info(f"WebSocket connection closed for session {session_id}")
            # 不需要发送错误消息，因为连接已经关闭
        else:
            await send_error_message(
                websocket,
                {
                    "message": "An unexpected error occurred.",
                    "action": "Please try again.",
                    "error_type": "general",
                },
            )
    finally:
        # 确保始终进行清理
        await cleanup_session(session, session_id)
