# utils.py

import io
import pypdf
from typing import List

def chunk_text_with_overlap(text: str, max_chars: int = 1000, overlap_chars: int = 200) -> List[str]:
    """Split text into strict character-safe chunks to completely prevent context overflow."""
    if not text:
        return ["No text provided"]
    
    # Normalize whitespace
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]

    chunks = []
    i = 0
    while i < len(text):
        chunk_end = min(i + max_chars, len(text))
        chunks.append(text[i:chunk_end])
        i += max_chars - overlap_chars
    return chunks

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded txt or pdf files."""
    file_extension = uploaded_file.name.split(".")[-1].lower()
    document_text = ""

    if file_extension == "txt":
        document_text = uploaded_file.getvalue().decode("utf-8")
    elif file_extension == "pdf":
        pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                document_text += extracted + "\n"

    return document_text