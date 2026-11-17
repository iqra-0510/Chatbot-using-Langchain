# StudyBuddy AI - LangChain RAG Chatbot 📚

A Retrieval-Augmented Generation (RAG) version of StudyBuddy AI, built with Python, Streamlit, LangChain, ChromaDB and the Groq API.
It answers study questions **only from the documents in the `docs/` folder** and shows the sources used.

## How it works (LangChain steps)
1. Load documents: `TextLoader` (TXT) and `PyPDFLoader` (PDF)
2. Split: `RecursiveCharacterTextSplitter`
3. Embed and store: all-MiniLM-L6-v2 embeddings (run through ChromaDB's ONNX runtime and wrapped as a LangChain `Embeddings` class) + `Chroma`
4. Retrieve: `as_retriever(k=4)` with `create_history_aware_retriever`
5. Answer: `ChatGroq` + `ChatPromptTemplate` + `create_stuff_documents_chain` + `create_retrieval_chain`
6. Memory: `RunnableWithMessageHistory` + `InMemoryChatMessageHistory` (one session_id per user)
7. If the answer is not in the documents: "I don't know"

## Features
- Answers based on your own study notes (PDF and TXT)
- Source file and page shown for every answer
- Upload new documents from the sidebar
- Follow-up question support (conversation memory)
- Prompt-injection protection
- Clear conversation button

## Documents (docs/ folder)
- `StudyBuddy_Study_Notes.pdf` - 17 pages: Python, Databases/SQL, Operating Systems, Algorithms, Calculus, Statistics, Study Skills
- `StudyBuddy_Guide.pdf` - purpose, allowed topics and answering rules

To add more subjects later, just put new PDF/TXT files in `docs/` (or upload them from the sidebar). The knowledge base is rebuilt automatically.

## Installation
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your Groq API key
3. Put your study PDF/TXT files in `docs/`
4. Run: `python -m streamlit run app.py`

The first run downloads the embedding model (about 80 MB), so it needs internet and some time. PyTorch is not required.

## Security
Never upload your `.env` file or expose your API key.

<!-- python -m venv venv
venv\Scripts\activate -->

<!-- python -m streamlit run app.py -->