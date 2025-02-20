import os
from dotenv import load_dotenv, find_dotenv
import logging

logger = logging.getLogger(__name__)

# Force reload of environment variables
os.environ.clear()
env_path = find_dotenv()
logger.info(f"Loading .env from: {env_path}")
load_dotenv(env_path, override=True)

# Debug: Print the raw API key value
api_key = os.environ.get("OPENAI_API_KEY")  # Use os.environ.get instead of os.getenv
logger.info(f"Raw API Key value length: {len(api_key) if api_key else 0}")
logger.info(f"API Key starts with: {api_key[:10] if api_key else 'None'}")

# OpenAI Configuration
OPENAI_API_KEY = api_key
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")

# Vector Store Configuration
CHROMA_PERSIST_DIR = "db"

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Model configuration
MODEL_NAME = "gpt-4-turbo-preview"

# Ensure required environment variables are set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
