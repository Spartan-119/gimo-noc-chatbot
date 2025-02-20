from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from .document_loader import load_documents
from .config import CHROMA_PERSIST_DIR
import os
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        """Initialize vector store with OpenAI embeddings."""
        try:
            # Try getting API key from environment first (for local development)
            api_key = os.getenv("OPENAI_API_KEY")
            
            # If not found in environment, try Streamlit secrets (for cloud deployment)
            if not api_key:
                api_key = st.secrets.get("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OpenAI API key not found in environment or Streamlit secrets")
                
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            self.persist_directory = CHROMA_PERSIST_DIR
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
            raise

    def create_vector_store(self, documents):
        """Create and return a Chroma vector store from documents."""
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

    def load_vector_store(self):
        """Load and return the existing Chroma vector store."""
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

def get_vector_store():
    """Get an instance of the vector store."""
    vector_store = VectorStore()
    return vector_store.load_vector_store()

def process_documents():
    """Process documents and create vector store."""
    try:
        # Load and process documents
        documents = load_documents()
        
        # Create vector store
        vector_store = VectorStore()
        return vector_store.create_vector_store(documents)
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise