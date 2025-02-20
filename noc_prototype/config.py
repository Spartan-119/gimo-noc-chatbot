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

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector store configuration
CHROMA_PERSIST_DIR = "chroma_db"

# Model configuration
MODEL_NAME = "gpt-4-turbo-preview"
EMBEDDING_MODEL = "text-embedding-3-small"

# Document processing configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Remove the strict check to allow the app to start
# We'll handle the missing API key in the Streamlit app instead
