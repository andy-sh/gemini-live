"""
## README
Gemini Multimodal Live Proxy Server.

## CHANGELOG
### 20250606
- 去除对工具的依赖.
- 去除对Vertex AI的依赖，只使用开发端点.
- 去除对Google Cloud Secret Manager的依赖.
"""

# 导入必要的库
import logging  # 用于日志记录
import asyncio  # 用于异步IO操作
import os  # 用于环境变量和系统操作
import websockets  # 用于WebSocket服务器实现

from core.websocket_handler import handle_client  # 导入WebSocket客户端处理函数

# 配置日志系统
logging.basicConfig(
    level=getattr(
        logging, os.getenv("LOG_LEVEL", "INFO").upper()
    ),  # 从环境变量获取日志级别，默认为INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式：时间 - 级别 - 消息
    datefmt="%Y-%m-%d %H:%M:%S",  # 设置时间格式
)

# 抑制第三方库的日志输出，保持应用自身的调试信息
for logger_name in [
    "google",  # Google API客户端
    "google.auth",  # Google认证
    "google.auth.transport",  # Google认证传输
    "google.auth.transport.requests",  # Google认证请求
    "urllib3.connectionpool",  # HTTP连接池
    "google.generativeai",  # Google生成式AI
    "websockets.client",  # WebSocket客户端
    "websockets.protocol",  # WebSocket协议
    "httpx",  # HTTP客户端
    "httpcore",  # HTTP核心
]:
    logging.getLogger(logger_name).setLevel(
        logging.ERROR
    )  # 将这些库的日志级别设置为ERROR

# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)


async def main() -> None:
    """启动WebSocket服务器的主函数"""
    port = 8081  # 设置服务器端口

    # 创建并启动WebSocket服务器
    async with websockets.serve(
        handle_client,  # 客户端连接处理函数
        "0.0.0.0",  # 监听所有网络接口
        port,  # 服务器端口
        ping_interval=30,  # 每30秒发送一次ping
        ping_timeout=10,  # ping超时时间为10秒
    ):
        logger.info(f"Running websocket server on 0.0.0.0:{port}...")
        await asyncio.Future()  # 保持服务器运行，直到被手动停止


# 程序入口点
if __name__ == "__main__":
    asyncio.run(main())  # 运行异步主函数
