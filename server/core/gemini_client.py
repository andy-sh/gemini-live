"""
## README
Gemini client initialization and connection management.

## CHANGELOG
### 20250606
- 去除对Vertex AI的依赖，只使用开发端点.
- 去除对工具的依赖.
"""

import logging
import os
from google import genai
from config.config import MODEL, CONFIG, api_config, ConfigurationError

logger = logging.getLogger(__name__)


async def create_gemini_session():
    """
    创建并初始化Gemini客户端和会话
    """
    try:
        # Initialize authentication
        await api_config.initialize()

        # Development endpoint configuration
        logger.info("Initializing development endpoint client")

        # Initialize development client
        client = genai.Client(api_key=api_config.api_key)

        # Create the session
        session = client.aio.live.connect(model=MODEL, config=CONFIG)

        return session

    except ConfigurationError as e:
        logger.error(f"Configuration error while creating Gemini session: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while creating Gemini session: {str(e)}")
        raise
