from langchain_openai import OpenAIEmbeddings
from .config import OPENAI_API_KEY, EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)

def get_embeddings():
    """Initialize and return the OpenAI embeddings model."""
    logger.info(f"Initializing embeddings with API key (first 10 chars): {OPENAI_API_KEY[:10]}")
    return OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL
    ) 