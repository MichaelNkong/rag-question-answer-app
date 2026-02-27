
#Multi-Document RAG Application (Streamlit + LangChain + Chroma)

A production-style Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDFs and ask questions across all documents using an LLM.

Built with modern LLM tooling including LangChain, Chroma, HuggingFace embeddings, and Groq-hosted LLaMA models.

# Features

✅ Upload multiple PDF documents

✅ Automatic document chunking & embedding

✅ Vector similarity search using Chroma

✅ LLM-powered answer generation (LLaMA 3)

✅ Source attribution (shows document chunks used)

✅ OCR support for scanned PDFs (Tesseract)

✅ Session-based vector database (Streamlit session state)

#Architecture Overview
User Uploads PDFs
        ↓
Document Loader (Unstructured / PyPDF)
        ↓
Text Splitting (RecursiveCharacterTextSplitter)
        ↓
Embeddings (HuggingFace Sentence Transformers)
        ↓
Vector Store (ChromaDB)
        ↓
Retriever
        ↓
LLM (Groq - LLaMA 3 70B)
        ↓
Answer + Source Documents


# Tech Stack & Libraries
1️⃣ Streamlit

Purpose: Frontend UI framework

Used for:

File uploads

User input handling

Session state management

Displaying answers and sources

Why Streamlit?

Rapid prototyping

Clean Python-based UI

Great for ML demos and deployment



*2️⃣ LangChain

Purpose: LLM orchestration framework

Used for:

Document loaders

Text splitting

Retrieval pipeline

RetrievalQA chain abstraction

Key Components Used:

UnstructuredPDFLoader

RecursiveCharacterTextSplitter

RetrievalQA

#Why LangChain?

Modular LLM pipeline design

Easy retriever-chain integration

Production-ready abstraction for RAG**

Session state management

Displaying answers and sources

Why Streamlit?

Rapid prototyping

Clean Python-based UI

Great for ML demos and deployment


3️⃣ ChromaDB

Purpose: Vector database

Used for:

Storing document embeddings

Similarity search

Fast semantic retrieval

Why Chroma?

Lightweight

Easy local setup

Perfect for session-based RAG apps


4️⃣ HuggingFace Embeddings

Model Used:
sentence-transformers/all-MiniLM-L6-v2

Purpose: Convert text chunks into dense vector embeddings.

Why this model?

Lightweight

Fast inference

Strong semantic similarity performance

Ideal for local embedding generation


5️⃣ Groq + LLaMA 3 70B

Model: llama-3.3-70b-versatile

Purpose: Answer generation using retrieved context.

Why Groq?

Extremely low latency

High throughput

Ideal for production-scale LLM apps

6️⃣ Unstructured + Tesseract OCR

Purpose: Handle scanned PDFs (image-based documents)

If PDF contains selectable text → parsed directly

If scanned → OCR via Tesseract

This makes the system robust for:

Research papers

Books

Scanned academic material

Corporate documents

Retrieval Strategy

Recursive chunking (1000–2000 characters)

Overlap to preserve context

Similarity-based retrieval (top-k search)

"stuff" chain type (injects retrieved context into LLM prompt)


Example Use Cases

Academic research assistant

Legal document Q&A

Technical documentation search

Internal company knowledge base

AI-powered book reader

#Key Design Decisions
Why Session-State Vector DB?

Fast prototyping

No external DB dependency

Perfect for demo & portfolio

Why Chunking with Overlap?

Prevents context loss at boundaries

Improves retrieval accuracy

Why Separate Embeddings + LLM?

Better modularity

Swap models independently

Scalable architecture

#Performance Considerations

Embedding model optimized for speed

Top-k retrieval tuning

Large LLM model for high-quality responses

Optional persistent Chroma DB for scaling

#Future Improvements

Persistent vector database storage

Streaming LLM responses

Hybrid search (BM25 + vector)

Citation highlighting with page numbers

Authentication for enterprise deployment

#What This Project Demonstrates

Practical implementation of RAG

LLM orchestration using LangChain

Vector database integration

Embedding model selection

Multi-document retrieval system

Handling OCR edge cases

Clean separation of concerns

Production-minded architecture