import pytest
from pathlib import Path
from noc_prototype.document_loader import DocumentLoader, process_pdf_to_json

def test_document_loader_initialization():
    loader = DocumentLoader()
    assert loader.docs_dir == Path("data/docs")

def test_clean_text():
    loader = DocumentLoader()
    text = "fr om the T eam ar e pr o"
    cleaned = loader.clean_text(text)
    assert cleaned == "from the Team are pro"

def test_json_document_conversion():
    test_json = {
        "metadata": {
            "source": "test.pdf",
            "filename": "test.pdf"
        },
        "content": [
            {"type": "Text", "text": "Test content"}
        ]
    }
    loader = DocumentLoader()
    docs = loader._convert_json_to_documents([test_json])
    assert len(docs) == 1
    assert docs[0].page_content == "Test content" 