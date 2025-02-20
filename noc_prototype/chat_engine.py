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
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 4}
            ),
            return_source_documents=True,
            verbose=True
        )
        
        # Create proper prompt template with explicit formatting instructions
        prompt_template = """Use the following pieces of context to answer the user's question. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Always structure your response in this format:
        1. Start with "**Main Topic**" as a bold header
        2. Break information into sections with "**Section Name**" headers
        3. Use numbered lists for steps, starting each line with "1. ", "2. ", etc.
        4. Use bullet points starting with "• " for lists
        5. Add empty lines between sections
        6. Use `code` for technical items
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

    def get_response(self, query: str) -> tuple[str, list]:
        """
        Get response from the chat engine.
        Returns tuple of (response, source_documents)
        """
        result = self.qa_chain({"question": query, "chat_history": []})
        return result["answer"], result["source_documents"]

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