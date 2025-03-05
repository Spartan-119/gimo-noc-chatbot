from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from .config import OPENAI_API_KEY, MODEL_NAME
from noc_prototype.vector_store import get_vector_store
from langchain.prompts import PromptTemplate
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

class ChatEngine:
    def __init__(self, vector_store):
        """Initialize the chat engine with vector store and LLM."""
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model_name=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.2
        )
        
        # Create proper prompt template with explicit formatting instructions
        SYSTEM_TEMPLATE = """You are a helpful NOC (Network Operations Center) assistant specializing in technical documentation and procedures. 

When answering questions:
1. If you find ANY relevant information in the context, provide it - even if it's just part of the answer
2. Use direct quotes from the documentation when available
3. If you see test credentials or URLs, include them as they are important for NOC procedures
4. If you find partial information, provide what you know and mention what aspects you're not sure about
5. Only say "I cannot answer" if you find absolutely no relevant information in the context

For Premium Club and Voucher related questions:
- Include any test accounts or verification procedures
- Mention relevant team contacts
- List any shops or exchange points
- Include troubleshooting steps if available

Format your response as:
**Topic**: [Main topic of the question]

**Available Information**:
• [Key points from the context]
• [Relevant procedures]
• [Test accounts if applicable]

**Steps** (if applicable):
1. [Step-by-step procedures]
2. [Additional steps]

**Additional Notes**:
• [Any relevant warnings or important information]
• [Related links or resources]

Context: {context}
Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=SYSTEM_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Create QA chain with custom prompt
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": 4  # Remove score_threshold as it's not supported
                }
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=True
        )

    def get_response(self, query: str) -> tuple[str, list]:
        """Get response with document verification."""
        try:
            # Get documents and scores
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=4
            )
            
            # Filter by score threshold manually
            relevant_docs = [
                doc for doc, score in docs_and_scores 
                if score >= 0.2  # Manual score threshold
            ]
            
            if not relevant_docs:
                return (
                    "I cannot answer this question as it's not covered in the NOC documentation.",
                    []
                )
            
            # Log retrieved documents for verification
            logger.info("Retrieved documents:")
            for i, doc in enumerate(relevant_docs, 1):
                logger.info(f"Doc {i}: {doc.page_content[:200]}...")
            
            # Get response from QA chain using filtered docs
            result = self.qa_chain({
                "question": query,
                "chat_history": []
            })
            
            # Log for verification
            logger.info(f"Response: {result['answer']}")
            logger.info(f"Number of source documents: {len(result['source_documents'])}")
            
            return result["answer"], result["source_documents"]
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise

    def _format_response(self, text: str) -> str:
        """Format the response for better readability."""
        # Add double newlines after headers
        text = re.sub(r'\*\*(.*?)\*\*', r'**\1**\n\n', text)
        
        # Add newlines after bullet points
        text = re.sub(r'(• .*?)(?=(?:• |\*\*|\n|$))', r'\1\n', text)
        
        # Add newlines after numbered points
        text = re.sub(r'(\d+\. .*?)(?=(?:\d+\. |\*\*|\n|$))', r'\1\n', text)
        
        # Add newlines between sections
        text = re.sub(r'(\n)(?!\n)', r'\n\n', text)
        
        # Clean up excessive newlines
        text = re.sub(r'\n{3,}', r'\n\n', text)
        
        # Ensure code blocks are properly formatted
        text = re.sub(r'`(.*?)`', r'`\1`', text)
        
        return text.strip() 

    def test_retrieval(self, query: str):
        """Test document retrieval with different thresholds."""
        try:
            # Get documents and scores
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=10  # Get more docs for testing
            )
            
            logger.info(f"\nQuery: {query}")
            logger.info("Retrieved documents and scores:")
            
            for i, (doc, score) in enumerate(docs_and_scores, 1):
                logger.info(f"\nDocument {i}:")
                logger.info(f"Score: {score}")
                logger.info(f"Content preview: {doc.page_content[:200]}")
                logger.info(f"Metadata: {doc.metadata}")
                
                # Test different thresholds
                thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]
                for threshold in thresholds:
                    would_pass = score >= threshold
                    logger.info(f"Would pass threshold {threshold}: {would_pass}")
                    
            return docs_and_scores
            
        except Exception as e:
            logger.error(f"Error testing retrieval: {str(e)}")
            raise 