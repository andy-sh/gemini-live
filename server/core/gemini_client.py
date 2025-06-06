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
Gemini客户端初始化和连接管理模块
此模块负责创建和管理与Google Gemini AI服务的连接会话
"""

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
    2. 根据配置选择使用Vertex AI或开发端点
    3. 创建并返回Gemini会话
    
    Returns:
        session: Gemini会话对象
        
    Raises:
        ConfigurationError: 当配置出现问题时抛出
        Exception: 其他未预期的错误
    """
    try:
        # 初始化API认证配置
        await api_config.initialize()
        
        if api_config.use_vertex:
            # Vertex AI配置部分
            # 获取Vertex AI的位置和项目ID
            location = os.getenv('VERTEX_LOCATION', 'us-central1')
            project_id = os.environ.get('PROJECT_ID')
            
            # 验证项目ID是否存在
            if not project_id:
                raise ConfigurationError("PROJECT_ID is required for Vertex AI")
            
            logger.info(f"Initializing Vertex AI client with location: {location}, project: {project_id}")
            
            # 初始化Vertex AI客户端
            client = genai.Client(
                vertexai=True,
                location=location,
                project=project_id,
                # http_options={'api_version': 'v1beta'}
            )
            logger.info(f"Vertex AI client initialized with client: {client}")
        else:
            # 开发端点配置部分
            logger.info("Initializing development endpoint client")
            
            # 初始化开发环境客户端
            client = genai.Client(
                vertexai=False,
                http_options={'api_version': 'v1alpha'},
                api_key=api_config.api_key
            )
                
        # 创建Gemini会话
        # 使用指定的模型和配置参数建立连接
        session = client.aio.live.connect(
            model=MODEL,
            config=CONFIG
        )
        
        return session
        
    except ConfigurationError as e:
        # 处理配置错误
        logger.error(f"Configuration error while creating Gemini session: {str(e)}")
        raise
    except Exception as e:
        # 处理其他未预期的错误
        logger.error(f"Unexpected error while creating Gemini session: {str(e)}")
        raise 