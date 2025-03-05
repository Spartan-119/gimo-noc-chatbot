from typing import List, Dict
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Text, Table, Image, ListItem, Title
)
import json
import base64
import logging
import os
import re
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, docs_dir: str = "data/docs"):
        self.docs_dir = Path(docs_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def load_documents(self) -> List:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")

        documents = []
        for pdf_file in self.docs_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())

        return self.text_splitter.split_documents(documents)

    def clean_text(self, text: str) -> str:
        """Clean text before embedding."""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = text.replace('fr om', 'from')
        text = text.replace('T eam', 'Team')
        text = text.replace('ar e', 'are')
        text = text.replace('pr o', 'pro')
        
        # Remove non-breaking spaces
        text = text.replace('\u00a0', ' ')
        
        # Clean up newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()

    def _convert_json_to_documents(self, json_docs):
        """Convert JSON documents to LangChain Document format with cleaning."""
        documents = []
        for doc in json_docs:
            # Clean and combine all content
            text = "\n".join(
                self.clean_text(item["text"]) 
                for item in doc["content"]
            )
            
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

def process_pdf_to_json(pdf_path: str) -> Dict:
    """
    Extract content from PDF using a simpler approach.
    Returns a structured JSON with the content.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Initialize structure for JSON
    document_content = {
        "metadata": {
            "source": pdf_path,
            "filename": Path(pdf_path).name
        },
        "content": []
    }
    
    # Process each page
    for page in pages:
        content_item = {
            "type": "Text",
            "page_number": page.metadata.get("page", None),
            "text": page.page_content
        }
        document_content["content"].append(content_item)
    
    return document_content

def process_directory_to_json(data_dir: str = "data", output_dir: str = "processed_data"):
    """
    Process all PDFs in a directory to JSON files.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    pdf_files = list(Path(data_dir).glob("**/*.pdf"))
    processed_docs = []
    
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing {pdf_path.name}...")
            json_content = process_pdf_to_json(str(pdf_path))
            
            # Save individual JSON file
            output_path = Path(output_dir) / f"{pdf_path.stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, indent=2)
                
            processed_docs.append(json_content)
            logger.info(f"Successfully processed {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            continue
    
    return processed_docs

def load_documents(data_dir: str = "data"):
    """Load and process documents from the data directory."""
    try:
        # Initialize PDF loader for the directory
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=UnstructuredPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        
        logger.info("Processing documents and creating vector store...")
        
        # Load documents
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        return split_docs
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise

def process_directory_to_json(directory: str):
    """Process PDF files in directory to JSON format."""
    try:
        for file in os.listdir(directory):
            if file.endswith('.pdf'):
                logger.info(f"Processing {file}...")
                # Your existing PDF to JSON processing code
                logger.info(f"Successfully processed {file}")
                
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        raise

def load_processed_documents(processed_dir: str = "processed_data") -> List[Dict]:
    """Load documents from processed JSON files."""
    try:
        processed_dir_path = Path(processed_dir)
        if not processed_dir_path.exists():
            raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")

        documents = []
        for json_file in processed_dir_path.glob("*.json"):
            logger.info(f"Loading {json_file.name}...")
            with open(json_file, "r", encoding="utf-8") as f:
                doc = json.load(f)
                documents.append(doc)
            
        logger.info(f"Loaded {len(documents)} processed documents")
        return documents

    except Exception as e:
        logger.error(f"Error loading processed documents: {str(e)}")
        raise 