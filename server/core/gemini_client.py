# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gemini client initialization and connection management
"""

# 导入必要的库
import logging
import os
from google import genai
from config.config import MODEL, CONFIG, api_config, ConfigurationError

# 设置日志记录器
logger = logging.getLogger(__name__)


async def create_gemini_session():
    """
    创建并初始化Gemini客户端和会话

    此函数负责:
    1. 初始化API认证
    2. 配置开发环境端点
    3. 创建Gemini客户端实例
    4. 建立实时连接会话

    Returns:
        session: 返回一个已配置的Gemini实时会话对象

    Raises:
        ConfigurationError: 当配置出现问题时抛出
        Exception: 其他未预期的错误
    """
    try:
        # 初始化API认证配置
        await api_config.initialize()

        # 配置开发环境端点
        logger.info("Initializing development endpoint client")

        # 初始化Gemini客户端
        # vertexai=False 表示不使用Vertex AI服务
        # http_options 设置API版本为v1alpha
        # api_key 使用配置中的API密钥
        client = genai.Client(
            vertexai=False,
            http_options={"api_version": "v1alpha"},
            api_key=api_config.api_key,
        )

        # 创建实时连接会话
        # 使用指定的模型和配置参数
        session = client.aio.live.connect(model=MODEL, config=CONFIG)

        return session

    except ConfigurationError as e:
        # 处理配置错误
        logger.error(f"Configuration error while creating Gemini session: {str(e)}")
        raise
    except Exception as e:
        # 处理其他未预期的错误
        logger.error(f"Unexpected error while creating Gemini session: {str(e)}")
        raise
