import pytest
import json
from pathlib import Path

@pytest.fixture
def sample_document():
    return {
        "metadata": {
            "source": "test.pdf",
            "filename": "test.pdf"
        },
        "content": [
            {
                "type": "Text",
                "text": "Test premium club voucher content"
            }
        ]
    }

@pytest.fixture
def processed_docs_dir():
    return Path("processed_data") 