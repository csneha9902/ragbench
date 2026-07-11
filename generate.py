import os
from dotenv import load_dotenv
from retrieve import retrieve_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Global cache for the LLM client
_llm = None

def get_llm():
    global _llm
    if _llm is not None:
        return _llm
        
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please add it to your .env file.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("Initializing ChatGoogleGenerativeAI (gemini-3.5-flash)...")
        _llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=api_key,
            temperature=0.0
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")
        from langchain_openai import ChatOpenAI
        print("Initializing ChatOpenAI (gpt-4o-mini)...")
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.0
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Choose 'gemini' or 'openai'.")
        
    return _llm

RAG_PROMPT_TEMPLATE = """You are RAGBench, a highly accurate HR Assistant for RAGBench Enterprises.
Answer the user's question using ONLY the provided retrieved context. 
If the answer cannot be determined from the context, state exactly: "I cannot find the answer in the provided documents."
Do not make up facts, use outside knowledge, or hallucinate. Be clear, professional, and direct.

---
RETIRED CONTEXT:
{context}
---

USER QUESTION: {question}

Helpful Answer:"""

def generate_answer(query, top_n=5, alpha=0.5):
    """
    RAG Pipeline:
    1. Retrieve relevant contexts
    2. Format prompt
    3. Generate response using LLM
    """
    # 1. Retrieve context and measure latency
    retrieved_docs, retrieve_latency = retrieve_context(query, top_n=top_n, alpha=alpha)
    
    if not retrieved_docs:
        return {
            "answer": "No documents are ingested in the system yet. Please upload files and run ingestion first.",
            "retrieved_contexts": [],
            "sources": [],
            "retrieve_latency_ms": int(retrieve_latency * 1000)
        }
        
    # 2. Format context for prompt
    context_str = ""
    sources = []
    for idx, doc in enumerate(retrieved_docs):
        src_name = doc.metadata.get("source_name", "Unknown Document")
        page_num = doc.metadata.get("page", 0) + 1  # 0-indexed to 1-indexed
        context_str += f"--- Chunk {idx + 1} (Source: {src_name}, Page: {page_num}) ---\n{doc.page_content}\n\n"
        sources.append({
            "content": doc.page_content,
            "source_name": src_name,
            "page": page_num,
            "rank": idx + 1
        })
        
    # 3. Call LLM
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context_str,
            "question": query
        })
    except Exception as e:
        answer = f"Error generating answer: {e}"
        
    return {
        "answer": answer,
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
        "sources": sources,
        "retrieve_latency_ms": int(retrieve_latency * 1000)
    }
