import app_streamlit as st
from noc_prototype.chat_engine import ChatEngine
import logging
import pyperclip

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def initialize_session_state():
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = ChatEngine()
    if "messages" not in st.session_state:
        st.session_state.messages = []

def copy_to_clipboard(text, button_key):
    if st.button("📋 Copy", key=button_key):
        pyperclip.copy(text)
        st.toast("Copied to clipboard!")

def main():
    st.set_page_config(
        page_title="NOC Documentation Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Minimal CSS
    st.markdown("""
        <style>
        /* Base theme */
        .stApp {
            background: #1E1E1E;
            color: #E0E0E0;
        }
        
        /* Clean layout */
        .main {
            padding: 0;
        }
        
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Hide unnecessary elements */
        #MainMenu, footer, header, .stDeployButton {
            display: none !important;
        }
        
        /* Chat container styling */
        .stChatMessage {
            background: transparent !important;
            border: none !important;
            padding: 0.5rem 1.5rem !important;
        }
        
        /* Message content - iPhone style bubbles */
        [data-testid="stChatMessageContent"] {
            border-radius: 20px !important;
            padding: 1rem !important;
            line-height: 1.6 !important;
            max-width: 85% !important;
            margin: 0.5rem 0 !important;
        }
        
        /* User message styling - right side blue bubble */
        [data-testid="user-message"] [data-testid="stChatMessageContent"] {
            background: #0B93F6 !important;
            color: white !important;
            margin-left: auto !important;
            border-bottom-right-radius: 5px !important;
        }
        
        /* Assistant message styling - left side gray bubble */
        [data-testid="assistant-message"] [data-testid="stChatMessageContent"] {
            background: #2D2D2D !important;
            color: #E0E0E0 !important;
            margin-right: auto !important;
            border-bottom-left-radius: 5px !important;
        }
        
        /* Avatar styling */
        .stChatMessage [data-testid="stImage"] {
            width: 35px !important;
            height: 35px !important;
            margin: 0.5rem !important;
        }
        
        /* Message container spacing */
        .stChatMessageContainer {
            padding: 0.5rem 0 !important;
        }
        
        /* Bullet points and lists */
        [data-testid="stChatMessageContent"] ul {
            margin-left: 1.5rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* Bold headers */
        [data-testid="stChatMessageContent"] strong {
            color: inherit !important;
            display: block !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Code blocks */
        [data-testid="stChatMessageContent"] code {
            background: rgba(0, 0, 0, 0.2) !important;
            padding: 0.2rem 0.4rem !important;
            border-radius: 4px !important;
            color: inherit !important;
        }
        
        /* Center title */
        h1 {
            text-align: center !important;
            padding: 1rem !important;
            margin-bottom: 2rem !important;
        }
        
        /* Copy button styling */
        button[data-testid="baseButton-secondary"] {
            margin-top: 0.5rem !important;
            padding: 0.3rem 1rem !important;
            border-radius: 15px !important;
            background: rgba(255, 255, 255, 0.1) !important;
            border: none !important;
            color: #E0E0E0 !important;
            font-size: 0.8rem !important;
        }
        
        /* Chat input styling */
        .stChatInput {
            border-color: #333 !important;
            background: #1E1E1E !important;
            border-radius: 20px !important;
            padding: 0.5rem 1rem !important;
            margin: 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    st.title("🤖 NOC Documentation Assistant")

    # Chat interface
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                copy_to_clipboard(message["content"], f"copy_{idx}")

    # Input
    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar="🤖"):
            try:
                response = st.session_state.chat_engine.get_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                copy_to_clipboard(response, f"copy_new_{len(st.session_state.messages)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 