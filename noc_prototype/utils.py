from pathlib import Path
from typing import List
from langchain.schema import Document

def format_source_documents(source_documents: List[Document]) -> str:
    """Format source documents for display."""
    sources = []
    for doc in source_documents:
        if hasattr(doc.metadata, 'source'):
            source = Path(doc.metadata['source']).name
            page = doc.metadata.get('page', 1)
            sources.append(f"- {source} (Page {page})")
    
    if sources:
        return "\n".join(["Sources:"] + list(set(sources)))
    return "No source documents found." 