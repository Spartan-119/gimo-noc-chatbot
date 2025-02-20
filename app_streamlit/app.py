import streamlit as st
from pathlib import Path
import sys
import os
import time

# Get the absolute path to the project root
project_root = Path(__file__).parent.parent.absolute()

# Add the project root to Python path if it's not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Print for debugging
print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")

from noc_prototype.document_loader import DocumentLoader
from noc_prototype.vector_store import VectorStore
from noc_prototype.chat_engine import ChatEngine
from app_streamlit.utils import get_custom_css, format_source_documents

# Page configuration
st.set_page_config(
    page_title="NOC Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Settings ⚙️")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        help="Controls randomness in responses"
    )
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state
if "chat_engine" not in st.session_state:
    # Initialize vector store
    vector_store = VectorStore()
    try:
        with st.spinner("Loading existing knowledge base..."):
            vs = vector_store.load_vector_store()
    except:
        with st.status("Processing documents...", expanded=True) as status:
            st.write("Loading documents...")
            loader = DocumentLoader()
            documents = loader.load_documents()
            
            st.write("Creating vector embeddings...")
            vs = vector_store.create_vector_store(documents)
            
            status.update(label="✅ Document processing complete!", state="complete")
            time.sleep(1)
    
    # Initialize chat engine
    st.session_state.chat_engine = ChatEngine(vs)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Main chat interface
st.title("NOC Team Assistant 🤖")
st.markdown("""
    <div class="main-text">
        Ask me anything about NOC procedures and documentation!
    </div>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        bubble_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
        st.markdown(f"""
            <div class="{bubble_class}">
                <div class="main-text">{message["content"]}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if "sources" in message:
            st.markdown(f"""
                <div class="source-text">
                    {message["sources"]}
                </div>
            """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(f"""
            <div class="user-bubble">
                <div class="main-text">{prompt}</div>
            </div>
        """, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from chat engine
    with st.spinner("Thinking..."):
        response, source_docs = st.session_state.chat_engine.get_response(prompt)
    sources = format_source_documents(source_docs)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(f"""
            <div class="assistant-bubble">
                <div class="main-text">{response}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="source-text">
                {sources}
            </div>
        """, unsafe_allow_html=True)
    
    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    }) 