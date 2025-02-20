from .chat_engine import ChatEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_query(query: str):
    """Test the RAG system with a query."""
    try:
        # Initialize chat engine
        chat_engine = ChatEngine()
        
        # Get response
        response = chat_engine.get_response(query)
        
        print(f"\nQuery: {query}")
        print(f"Response: {response}")
        
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise

if __name__ == "__main__":
    # Example query - replace with your own
    test_query("please tell me about the SMS verification enable/disable") 