# rag.py

import heapq
import numpy as np
import ollama
import streamlit as st
from typing import List, Tuple, Union, Iterator, Dict, Any, Sequence
from config import DOMAIN_CONFIG

class ConvoRAG:
    documents: List[str]
    embedding_model: str
    llm_model: str
    document_embeddings: List[np.ndarray]
    conversation_history: List[Tuple[str, str]]

    def __init__(
        self,
        documents: List[str],
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
    ) -> None:
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        with st.spinner("Embedding documents (this may take a moment)..."):
            self.document_embeddings = [self.embed_text(doc) for doc in documents]

        self.conversation_history = []

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using Ollama's specified embedding model with strict character truncation safety."""
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            embedding = response["embedding"]
            return np.array(embedding)
        except Exception:
            # Fallback: Aggressively truncate to a safe character window (1500 chars) if an error occurs
            try:
                truncated_text = text[:1500] 
                response = ollama.embeddings(model=self.embedding_model, prompt=truncated_text)
                return np.array(response["embedding"])
            except Exception as inner_e:
                st.warning(f"Skipped an unparsable text block: {str(inner_e)}")
                return np.zeros(768)

    def cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embedding vectors."""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0

    def topk(self, arr: Sequence[float], k: int) -> List[int]:
        """Get the k largest elements and their indexes."""
        if not arr:
            return []
        k = min(k, len(arr))
        try:
            return heapq.nlargest(k, range(len(arr)), key=lambda i: arr[i])
        except Exception:
            return list(range(min(k, len(arr))))

    def search(self, query: str, top_k: int = 5) -> Tuple[str, float]:
        """Search for the most relevant documents based on cosine similarity."""
        if not self.documents or not self.document_embeddings:
            return "No documents available.", 0.0

        try:
            query_embedding = self.embed_text(query)
            similarities = [self.cosine_similarity(query_embedding, doc_emb) for doc_emb in self.document_embeddings]

            if not similarities:
                return "No similarities found.", 0.0

            topk_indices = self.topk(similarities, top_k)
            if not topk_indices:
                return "No relevant documents found.", 0.0
            
            result = "\n".join([self.documents[i] for i in topk_indices])
            return result, similarities[topk_indices[0]]
        except Exception:
            return "Error occurred while searching documents.", 0.0

    def generate_answer(
        self, system_prompt: str, user_prompt: str, is_stream: bool = True
    ) -> Union[Iterator[Dict[str, Any]], str]:
        """Generate a response using the provided system and user prompts."""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=is_stream,
            )
            return response if is_stream else (response["message"]["content"])
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            return iter([{"message": {"content": error_msg}}]) if is_stream else error_msg

    def get_document_summary(self) -> str:
        """Generate a concise summary using the first few chunks of the document."""
        sample_text = "\n".join(self.documents[:3])
        system_prompt = "You are an expert document summarizer. Provide a concise, clear overview paragraph explaining what this document is about based on the provided text."
        user_prompt = f"Here is an excerpt from the document:\n\n{sample_text}\n\nProvide a clear summary of what this document is about:"
        
        summary = self.generate_answer(system_prompt, user_prompt, is_stream=False)
        return str(summary)

    def rag(self, query: str) -> Union[Iterator[Dict[str, Any]], str]:
        """Main function to perform conversational document search and answer generation."""
        try:
            context, relevance_score = self.search(query)
            
            if relevance_score < 0.30 and context != "No documents available.":
                return self.generate_answer(
                    "You are a helpful assistant.", 
                    f"State politely that you cannot find this information in the uploaded document. Question: {query}"
                )

            system_prompt = f"You are an AI assistant. Answer the user query strictly using the provided document context. If the answer is missing, say: '{DOMAIN_CONFIG['fallback_message']}'"
            user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

            return self.generate_answer(system_prompt, user_prompt)
        except Exception:
            return iter([{"message": {"content": "An error occurred during search."}}])