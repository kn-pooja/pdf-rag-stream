
DOMAIN_CONFIG = {
    "domain_name": "Multi-Document RAG Assistant",
    "assistant_name": "Document Knowledge Assistant",
    "file_upload_label": "Upload knowledge base document (.txt or .pdf)",
    "welcome_message": "👋 Welcome! Please upload a PDF or text document to begin querying.",
    "fallback_message": "I cannot answer this based on the provided document context. Please check the uploaded file or contact support.",
    "sidebar_info": """
- Upload new documents anytime using the reset option
- Fast embedding via Ollama
- Context-aware retrieval
- Ask questions about the uploaded document
            
*Ask questions specific to your active document.*
    """,
    "categories": {
        "domain-related": "Questions about the uploaded document content, policies, or facts",
        "compliment": "Positive feedback or appreciation",
        "complaint": "Negative feedback or dissatisfaction",
        "chitchat": "General conversation not directly related to inquiries",
        "off-topic": "Questions completely unrelated to the domain"
    }
}