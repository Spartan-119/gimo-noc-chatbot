import pytest
from streamlit.testing.v1 import AppTest
import os

# def test_streamlit_app():
#     # Initialize app
#     at = AppTest.from_file("app_streamlit/app.py")
#     at.run()
    
#     # Test environment setup
#     assert os.getenv("OPENAI_API_KEY") is not None
#     assert os.getenv("PINECONE_API_KEY") is not None
    
#     # Verify chat interface exists
#     assert at.chat_input is not None
    
#     # Get chat input widget
#     chat_input = at.chat_input
#     assert chat_input is not None  # Check if widget exists
    
#     # Simulate chat input
#     chat_input.input("what can you tell me about premium club voucher issues?").run()
    
#     # First ensure the chat input exists
#     assert at.chat_input.exists()
#     # Then simulate input
#     at.chat_input().set_value("what can you tell me about premium club voucher issues?").run()
#     assert "premium club" in at.chat_message[0].markdown 

def test_app_initialization():
    """Test that the app can initialize with required components"""
    # Test environment variables
    assert os.getenv("OPENAI_API_KEY") is not None
    assert os.getenv("PINECONE_API_KEY") is not None
    
    # Test vector store initialization
    from noc_prototype.vector_store import VectorStore
    vs = VectorStore()
    assert vs is not None
    
    # Test chat engine initialization
    from noc_prototype.chat_engine import ChatEngine
    chat_engine = ChatEngine(vs.load_vector_store())
    assert chat_engine is not None 