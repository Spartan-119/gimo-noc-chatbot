from pinecone import Pinecone  # For Pinecone V3 client
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone  # For LangChain integration
from langchain_openai import OpenAIEmbeddings
from .document_loader import load_processed_documents
from langchain.schema import Document
import pinecone
import os
import streamlit as st
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        """Initialize vector store with OpenAI embeddings and Pinecone."""
        try:
            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings()
            
            # Initialize Pinecone (new style)
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index_name = os.getenv("PINECONE_INDEX_NAME")  # Store name directly
            self.index = self.pc.Index(self.index_name)  # Create index instance
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _convert_json_to_documents(self, json_docs):
        """Convert JSON documents to LangChain Document format."""
        documents = []
        for doc in json_docs:
            # Combine all content into a single text
            text = "\n".join(item["text"] for item in doc["content"])
            
            # Create LangChain Document
            document = Document(
                page_content=text,
                metadata={
                    "source": doc["metadata"]["source"],
                    "filename": doc["metadata"]["filename"]
                }
            )
            documents.append(document)
        return documents

    def create_vector_store(self):
        """Create and return a Pinecone vector store from processed documents."""
        try:
            # Load processed JSON documents
            json_docs = load_processed_documents()
            logger.info(f"Loaded {len(json_docs)} JSON documents")
            
            # Convert to LangChain format
            documents = self._convert_json_to_documents(json_docs)
            logger.info(f"Converted {len(documents)} documents")
            
            # Create vector store using LangChain's Pinecone integration
            return LangchainPinecone.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name  # Use stored name directly
            )
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def load_vector_store(self):
        """Load and return the existing Pinecone vector store."""
        return LangchainPinecone.from_existing_index(
            index_name=self.index_name,  # Use stored name directly
            embedding=self.embeddings
        )

def get_vector_store():
    """Get an instance of the vector store."""
    vector_store = VectorStore()
    return vector_store.load_vector_store()

def migrate_to_pinecone():
    """Migrate existing documents to Pinecone."""
    try:
        logger.info("Starting migration to Pinecone...")
        
        # Load documents
        documents = load_processed_documents()
        logger.info(f"Loaded {len(documents)} documents")
        
        # Create new vector store
        vector_store = VectorStore()
        pinecone_store = vector_store.create_vector_store()
        
        logger.info("Migration complete!")
        return pinecone_store
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        raise

def test_pinecone_connection():
    """Test Pinecone connection and API key."""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = pc.list_indexes()
        logger.info(f"Available indexes: {indexes}")
        return True, f"Pinecone connection successful! Found indexes: {indexes}"
    except Exception as e:
        return False, f"Pinecone connection failed: {str(e)}"

def create_pinecone_index():
    """Create the Pinecone index if it doesn't exist."""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Check if index already exists
        existing_indexes = pc.list_indexes()
        if index_name not in existing_indexes:
            # Create the index
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimensions
                metric='cosine'
            )
            logger.info(f"Created new index: {index_name}")
            return True, f"Index {index_name} created successfully"
        else:
            return True, f"Index {index_name} already exists"
            
    except Exception as e:
        return False, f"Failed to create index: {str(e)}"

def test_basic_connection():
    """Test basic HTTP connection to Pinecone."""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        
        # Try a direct HTTP request
        headers = {
            "Api-Key": api_key
        }
        url = f"https://controller.{environment}.pinecone.io/databases"
        response = requests.get(url, headers=headers)
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        return True, "Basic connection test complete"
        
    except Exception as e:
        return False, f"Connection test failed: {str(e)}"

def delete_all_vectors():
    """Delete all vectors from the Pinecone index."""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        
        # Delete all vectors
        index.delete(delete_all=True)
        logger.info("Deleted all vectors from index")
        return True, "All vectors deleted successfully"
    except Exception as e:
        return False, f"Failed to delete vectors: {str(e)}"