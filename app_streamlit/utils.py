import streamlit as st
from pathlib import Path
from typing import Dict, Any

def get_theme_styles() -> Dict[str, Any]:
    """Return theme-specific styles for the UI."""
    is_dark_theme = st.get_option("theme.base") == "dark"
    
    if is_dark_theme:
        return {
            "chat_bubbles": {
                "user": "#2E4F4F",
                "assistant": "#1A3333"
            },
            "text_colors": {
                "main": "#E0E0E0",
                "secondary": "#88CCCA"
            }
        }
    else:
        return {
            "chat_bubbles": {
                "user": "#E8F4F4",
                "assistant": "#F0F7F7"
            },
            "text_colors": {
                "main": "#1A3333",
                "secondary": "#2E4F4F"
            }
        }

def get_custom_css() -> str:
    """Generate custom CSS based on current theme."""
    styles = get_theme_styles()
    
    return f"""
        <style>
            .user-bubble {{
                background-color: {styles["chat_bubbles"]["user"]} !important;
                padding: 15px;
                border-radius: 15px;
                margin-bottom: 10px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            }}
            .assistant-bubble {{
                background-color: {styles["chat_bubbles"]["assistant"]} !important;
                padding: 15px;
                border-radius: 15px;
                margin-bottom: 10px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            }}
            .main-text {{
                color: {styles["text_colors"]["main"]} !important;
                font-size: 1em;
                line-height: 1.5;
            }}
            .source-text {{
                color: {styles["text_colors"]["secondary"]} !important;
                font-size: 0.8em;
                margin-top: 10px;
                padding: 8px;
                border-left: 3px solid {styles["text_colors"]["secondary"]};
            }}
        </style>
    """

def format_source_documents(source_docs) -> str:
    """Format source documents into a readable string."""
    sources = []
    for doc in source_docs:
        if hasattr(doc.metadata, 'source'):
            source = Path(doc.metadata['source']).name
            page = doc.metadata.get('page', 1)
            sources.append(f"📄 {source} (Page {page})")
    
    if sources:
        return "\n".join(["**Sources:**"] + list(set(sources)))
    return "No source documents found." 