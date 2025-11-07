# rag.py (LOCAL RAG — No APIs)
import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List

# Load embedder (downloads ~90MB first time)
_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage
_index = None
_chunks = []
_vectors = []

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def upsert_chunks(chunks: List[str], doc_id: str):
    """Store chunks in FAISS index"""
    global _index, _chunks, _vectors
    _chunks = chunks
    _vectors = _embedder.encode(chunks)
    
    # Build index
    dimension = _vectors.shape[1]
    _index = faiss.IndexFlatL2(dimension)
    _index.add(np.array(_vectors).astype('float32'))
    
    st.success(f"Indexed {len(chunks)} chunks from {doc_id}")

def retrieve_chunks(query: str, top_k: int = 3) -> List[str]:
    """Retrieve top-k chunks"""
    if _index is None:
        st.warning("Index PDF first!")
        return []
    
    q_embedding = _embedder.encode([query])
    distances, indices = _index.search(np.array(q_embedding).astype('float32'), top_k)
    return [_chunks[i] for i in indices[0] if i < len(_chunks)]

def rag_generate(query: str, model: str, context: List[str]):
    # ONLY TOP 3 CHUNKS
    context = context[:3]
    context_text = "\n\n".join(context)
    
    # SHORTEN IF TOO LONG
    if len(context_text) > 2000:
        context_text = context_text[:2000] + "..."

    prompt = f"<|system|>You are a helpful assistant. Use ONLY the context.<|end|>\n<|user|>\nContext: {context_text}\n\nQuestion: {query}<|end|>\n<|assistant|>"
    
    from inference import generate
    return generate(prompt, model)
