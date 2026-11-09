
# 📄 Local PDF RAG Streamlit App (Powered by Ollama)

A lightweight, 100% local Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **Python**, **NumPy**, and **Ollama**. 

This app allows you to upload any PDF document, query its contents, and watch the AI generate grounded answers streamed live to your browser—completely offline, with zero API keys or cloud costs required.

## 📂 Modular Project Structure

The codebase is cleanly separated into modular components for maintainability:

```text
src/
│
├── config.py             # Domain configurations, UI text, and prompts
├── utils.py              # File parsing (PDF/TXT) and strict character-safe text chunking
├── rag.py                # Core ConvoRAG engine (embeddings, similarity search, Ollama chat)
└── app.py                # Main Streamlit UI entry point, session state, and chat layout

```

## 🛠️ Step-by-Step Installation Guide (For Absolute Beginners)

If you have never set up a local AI application or Python project before, follow these steps sequentially.

### Step 1: Install Prerequisites on Your Computer

Before touching the code, make sure you have two essential tools installed:
1. **Python (v3.11 or higher)**
2. **Ollama:** Download and install the local AI runner from [ollama.com](https://ollama.com/download). Once installed, make sure the Ollama application is running in your system tray/background.

---

### Step 2: Download the Required AI Models

Open your computer's terminal (or Command Prompt / Terminal on Mac) and run the following commands to download the local embedding model and text generation model:

```bash
# Downloads the embedding model (used to turn text into searchable vectors)
ollama pull nomic-embed-text

# Downloads the language model (used to generate answers)
ollama pull llama3.2
```

### Step 3: Install Dependencies
Open your terminal, navigate to your project's root folder (where requirements.txt is located), create virtual environments and install the dependencies:
```bash
# Create a virtual environment named 'rag-env'
conda create -n rag-env python=3.12 -y
pip install -r requirements.txt
```

### Step 4: Run the Application
```
streamlit run src/app.py
```

Streamlit will automatically launch and open the web application in your default browser at http://localhost:8501

### 🚀 How to Use
**Configure Models**: Use the sidebar to verify or adjust your preferred LLM and embedding models.

**Upload Document**: Upload any PDF or .txt file using the file uploader.

**Review Summary**: The app will automatically chunk the text safely, generate embeddings, and provide a concise summary of your document.

**Ask Questions**: Chat with your document in real-time using the chat interface!

---

## ⚠️ Limitations

- **Hardware Dependent:** Performance relies entirely on your local machine's CPU/GPU and available RAM via Ollama.
- **Session-Based Storage:** Embeddings and chat history live strictly in Streamlit session state—refreshing or resetting the app clears the index.
- **PDF Extraction Limits:** Relies on `pypdf`, meaning scanned documents (images) or complex multi-column tables/charts may not extract cleanly without OCR.
- **Context Window Constraints:** Retrieval is limited to top-$k$ chunks, making it harder to synthesize insights spanning massive, multi-hundred-page books all at once.
