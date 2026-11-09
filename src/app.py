

import time
import streamlit as st

from config import DOMAIN_CONFIG
from utils import chunk_text_with_overlap, extract_text_from_file
from rag import ConvoRAG


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables safely."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_uploaded" not in st.session_state:
        st.session_state.document_uploaded = False
    if "active_filename" not in st.session_state:
        st.session_state.active_filename = ""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None


def render_sidebar():
    """Render sidebar configuration and document management tools."""
    with st.sidebar:
        st.title("Configuration")
        model = st.selectbox("Select LLM Model", ["llama3.2"], index=0)
        embed_model = st.selectbox("Select Embedding Model", ["nomic-embed-text"], index=0)

        st.markdown("---")

        if st.session_state.document_uploaded:
            st.success(f"📄 Active File: **{st.session_state.active_filename}**")
            if st.button("🔄 Reset & Upload New PDF"):
                st.session_state.clear()
                st.rerun()
        
        st.subheader("Assistant Info")
        st.markdown(DOMAIN_CONFIG['sidebar_info'])
        
        return model, embed_model


def handle_file_upload(model: str, embed_model: str):
    """Handle the file upload widget and initialization of the RAG pipeline."""
    st.info(DOMAIN_CONFIG['welcome_message'])
    
    uploaded_file = st.file_uploader(
        DOMAIN_CONFIG['file_upload_label'],
        type=["txt", "pdf"]
    )
    
    if uploaded_file is not None:
        with st.spinner(f"Reading and analyzing {uploaded_file.name}..."):
            document_text = extract_text_from_file(uploaded_file)

            if document_text.strip():
                documents = chunk_text_with_overlap(document_text)
                
                # Instantiate isolated RAG container
                rag_instance = ConvoRAG(
                    documents, embedding_model=embed_model, llm_model=model
                )
                st.session_state.rag_system = rag_instance
                st.session_state.document_uploaded = True
                st.session_state.active_filename = uploaded_file.name
                
                # Automatically generate summary of what the document is about
                with st.spinner("Generating document summary..."):
                    doc_summary = rag_instance.get_document_summary()

                welcome_msg = (
                    f"👋 Successfully loaded **{uploaded_file.name}**!\n\n"
                    f"### 📌 Document Summary\n{doc_summary}\n\n"
                    f"*You can now ask me any specific questions about this document!*"
                )
                st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
                st.success("Document analyzed successfully!")
                st.rerun()
            else:
                st.error("Could not extract text from the uploaded file.")


def handle_chat_interface():
    """Render chat messages and handle user queries."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about your uploaded document:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Analyzing document...")
            
            try:
                llm_stream = st.session_state.rag_system.rag(user_input)
                full_response = ""
                
                for chunk in llm_stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        full_response += chunk["message"]["content"]
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)
                        
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception:
                error_str = "I encountered an error querying the document."
                message_placeholder.markdown(error_str)
                st.session_state.messages.append({"role": "assistant", "content": error_str})


def main() -> None:
    st.set_page_config(page_title=DOMAIN_CONFIG['assistant_name'], page_icon="🤖", layout="wide")
    st.title(DOMAIN_CONFIG['assistant_name'])

    initialize_session_state()
    model, embed_model = render_sidebar()

    if not st.session_state.document_uploaded:
        handle_file_upload(model, embed_model)
    else:
        handle_chat_interface()


if __name__ == "__main__":
    main()