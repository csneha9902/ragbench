import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

DB_DIR = "vector_db"

# Global caches to avoid reloading model on every search
_embeddings = None
_vector_db = None
_bm25_retriever = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("Loading HuggingFaceEmbeddings in retrieve.py...")
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
            return None
        _vector_db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
    return _vector_db

def load_bm25_retriever():
    global _bm25_retriever
    db = get_vector_db()
    if db is None:
        return None
        
    # Get all documents/chunks from Chroma to build BM25 index
    data = db.get()
    documents = []
    if "documents" in data and data["documents"]:
        for i in range(len(data["documents"])):
            text = data["documents"][i]
            metadata = data["metadatas"][i] if data["metadatas"] else {}
            documents.append(Document(page_content=text, metadata=metadata))
            
    if not documents:
        return None
        
    print(f"Building BM25 index over {len(documents)} chunks...")
    _bm25_retriever = BM25Retriever.from_documents(documents)
    return _bm25_retriever

def rrf(vector_docs, keyword_docs, k=60, top_n=5):
    """
    Reciprocal Rank Fusion (RRF) to blend vector and keyword search.
    """
    scores = {}
    
    # Helper to rank documents
    def score_docs(docs):
        for rank, doc in enumerate(docs):
            # Use page content as unique identifier for the chunk
            doc_id = doc.page_content
            if doc_id not in scores:
                scores[doc_id] = {"doc": doc, "score": 0.0}
            # RRF formula: 1 / (k + rank)
            scores[doc_id]["score"] += 1.0 / (k + (rank + 1))
            
    score_docs(vector_docs)
    score_docs(keyword_docs)
    
    # Sort documents by their RRF score descending
    sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_items[:top_n]]

def retrieve_context(query, top_n=5, alpha=0.5):
    """
    Retrieve documents based on a query.
    alpha = 1.0: Pure Vector Search
    alpha = 0.0: Pure BM25 Keyword Search
    alpha between 0.0 and 1.0: Hybrid search via Reciprocal Rank Fusion (RRF)
    """
    start_time = time.time()
    
    db = get_vector_db()
    if db is None:
        print("Vector database is empty. Please run ingest.py.")
        return [], 0.0
        
    vector_docs = []
    if alpha > 0.0:
        # Search for extra docs to have overlap for RRF
        k_val = top_n * 2 if alpha < 1.0 else top_n
        vector_docs = db.similarity_search(query, k=k_val)
        
    keyword_docs = []
    if alpha < 1.0:
        global _bm25_retriever
        if _bm25_retriever is None:
            load_bm25_retriever()
        if _bm25_retriever:
            _bm25_retriever.k = top_n * 2 if alpha > 0.0 else top_n
            keyword_docs = _bm25_retriever.invoke(query)
            
    # Combine or select
    if alpha == 1.0:
        retrieved_docs = vector_docs[:top_n]
    elif alpha == 0.0:
        retrieved_docs = keyword_docs[:top_n]
    else:
        retrieved_docs = rrf(vector_docs, keyword_docs, top_n=top_n)
        
    latency = time.time() - start_time
    return retrieved_docs, latency

def reset_retrievers():
    """Reset the cached retrievers when re-ingestion occurs."""
    global _vector_db, _bm25_retriever
    _vector_db = None
    _bm25_retriever = None
    print("Cached retrievers have been reset.")
