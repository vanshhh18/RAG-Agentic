

import os
import pickle
import zipfile
from functools import lru_cache

from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


# =========================
# DOWNLOAD FROM HF
# =========================
def download_indexes():
    if not os.path.exists("vector_db"):
        print("Downloading FAISS index...")
        os.makedirs("vector_db", exist_ok=True)
        for filename in ["index.faiss", "index.pkl"]:
            hf_hub_download(
                repo_id="vanshhh-18/rag-agentic",
                filename=filename,
                repo_type="dataset",
                token=os.getenv("HF_TOKEN"),
                local_dir="vector_db"
            )
        print("✅ FAISS ready.")

    if not os.path.exists("bm25.pkl"):
        print("Downloading BM25 index...")
        hf_hub_download(
            repo_id="vanshhh-18/rag-agentic",
            filename="bm25.pkl",
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
            local_dir="."
        )
        print("✅ BM25 ready.")

download_indexes()


# =========================
# EMBEDDINGS
# =========================
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# =========================
# FAISS LOAD
# =========================
@lru_cache(maxsize=1)
def load_faiss():
    return FAISS.load_local(
        "vector_db",
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


# =========================
# BM25 LOAD
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "bm25.pkl"), "rb") as f:
    bm25, docs = pickle.load(f)


# =========================
# RETRIEVERS
# =========================
def faiss_retriever():
    return load_faiss().as_retriever(search_kwargs={"k": 3})


def hybrid_search(query):
    # FAISS
    faiss_docs = faiss_retriever().invoke(query)

    # BM25
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    if isinstance(docs[0], str):
        bm25_docs = [Document(page_content=docs[i]) for i in top_idx]
    else:
        bm25_docs = [docs[i] for i in top_idx]

    # Merge
    seen = set()
    final_docs = []
    for d in faiss_docs + bm25_docs:
        if d.page_content not in seen:
            final_docs.append(d)
            seen.add(d.page_content)

    return final_docs[:3]


# =========================
# LLM
# =========================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)