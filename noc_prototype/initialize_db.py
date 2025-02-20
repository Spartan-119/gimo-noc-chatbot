import os
from .vector_store import process_documents
import logging
from .config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize():
    """Initialize the vector database with documents."""
    try:
        # Debug: Print current working directory and env info
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f".env file exists: {os.path.exists('.env')}")
        
        # Debug: Print API key info
        logger.info(f"API Key from config (first 10 chars): {OPENAI_API_KEY[:10]}")
        logger.info(f"Direct env var (first 10 chars): {os.getenv('OPENAI_API_KEY', 'NOT_FOUND')[:10]}")
        
        with open('.env', 'r') as f:
            logger.info(f"Raw .env content: {f.read()}")
        
        # Process documents and create vector store
        logger.info("Processing documents and creating vector store...")
        vector_store = process_documents()
        
        logger.info("Initialization complete!")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

if __name__ == "__main__":
    initialize()