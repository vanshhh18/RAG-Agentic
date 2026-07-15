Demo: https://huggingface.co/spaces/vanshhh-18/rag-agenticcc


# Agentic RAG System for Technical Documentation

An AI-powered Retrieval-Augmented Generation (RAG) system for querying large-scale technical documentation using **Hybrid Search (BM25 + FAISS)** and **Agentic AI workflows**. The system combines keyword and semantic retrieval to provide accurate, context-grounded answers while minimizing hallucinations.

> **Note:** This project was developed during my AI Internship at DRDO (Defence Research and Development Organisation).

---

## Features

- Hybrid Search (BM25 + FAISS)
- Retrieval-Augmented Generation (RAG)
- Agentic AI workflows using LangGraph
- Semantic document retrieval
- Keyword-based retrieval
- Multi-document reasoning
- Context-grounded answer generation
- Conversation memory
- Support for large-scale technical documentation

---

## Tech Stack

- Python
- LangChain
- LangGraph
- FAISS
- BM25
- HuggingFace Embeddings
- Groq LLM
- FastAPI
- Streamlit

---

## Architecture

```
Technical Documents
        │
        ▼
Document Processing
        │
        ▼
Chunking
        │
        ├──────────────┐
        ▼              ▼
   BM25 Index     Embeddings
                      │
                      ▼
                    FAISS
        └──────────────┘
               │
        Hybrid Retriever
               │
               ▼
        LangGraph Agent
               │
               ▼
             LLM
               │
               ▼
      Grounded Response
```

---

## Workflow

1. Process and chunk technical documents.
2. Build a **BM25 index** for lexical retrieval.
3. Generate embeddings and store them in **FAISS** for semantic search.
4. Retrieve relevant documents using **Hybrid Search (BM25 + FAISS)**.
5. Pass retrieved context into a **LangGraph Agentic workflow**.
6. Generate grounded responses using the LLM.
7. Return detailed answers based only on the retrieved documentation.

---

## Project Highlights

- Built an Agentic RAG system for large-scale technical documentation.
- Indexed **10,000–100,000** technical records.
- Implemented **Hybrid Search (BM25 + FAISS)** to improve retrieval quality.
- Developed LangGraph-based multi-step AI workflows.
- Reduced hallucinations by grounding responses in retrieved context.
- Designed for low-latency, scalable document retrieval.

---

## Future Improvements

- Cross-Encoder Re-ranking
- Multi-modal RAG
- Knowledge Graph Integration
- Query Expansion
- Citation Generation
- Distributed Vector Database

---

## Disclaimer

This repository demonstrates the architecture and implementation approach developed during my internship. Confidential datasets, proprietary documentation, and organization-specific assets are **not included**.
