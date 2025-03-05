import pytest
from noc_prototype.chat_engine import ChatEngine
from noc_prototype.vector_store import VectorStore

@pytest.fixture
def chat_engine():
    vs = VectorStore()
    vector_store = vs.load_vector_store()
    return ChatEngine(vector_store)

def test_premium_club_retrieval(chat_engine):
    query = "what can you tell me about premium club voucher issues?"
    response, docs = chat_engine.get_response(query)
    assert len(docs) > 0
    assert "premium club" in response.lower()
    assert "voucher" in response.lower()

def test_document_retrieval_scores(chat_engine):
    results = chat_engine.test_retrieval("premium club voucher")
    for doc, score in results:
        assert score >= 0.0 and score <= 1.0 