"""
## README
配置文件，用于管理与Google Gemini AI服务的连接会话.

## CHANGELOG
### 20250606
- 去除对Vertex AI的依赖，只使用开发端点.
- 去除对工具的依赖.

"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from google.cloud import secretmanager

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""

    pass


class ApiConfig:
    """API configuration handler."""

    def __init__(self):
        self.api_key: Optional[str] = None

        logger.info(f"Initialized API configuration with Gemini API")

    async def initialize(self):
        """Initialize API credentials."""
        try:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        except Exception as e:
            logger.warning(f"Failed to get API key from Secret Manager: {e}")
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ConfigurationError(
                    "No API key available from Secret Manager or environment"
                )


# Initialize API configuration
api_config = ApiConfig()

# Model configuration
MODEL = os.getenv("MODEL_DEV_API", "models/gemini-2.0-flash-exp")
VOICE = os.getenv("VOICE_DEV_API", "Kore")

# Load system instructions
try:
    with open("config/system-instructions.txt", "r") as f:
        SYSTEM_INSTRUCTIONS = f.read()
except Exception as e:
    logger.error(f"Failed to load system instructions: {e}")
    SYSTEM_INSTRUCTIONS = ""

logger.info(f"System instructions: {SYSTEM_INSTRUCTIONS}")

# Gemini Configuration
CONFIG = {
    "generation_config": {"response_modalities": ["AUDIO"], "speech_config": VOICE},
    "tools": [{"function_declarations": []}],
    "system_instruction": SYSTEM_INSTRUCTIONS,
}
