import os
import glob
import shutil
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
DB_DIR = "vector_db"

def ingest_documents(chunk_size=500, chunk_overlap=50):
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print("No PDF files found in data directory. Please run generate_sample_data.py first.")
        return 0
        
    print(f"Found {len(pdf_files)} PDF files to ingest.")
    documents = []
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        try:
            docs = loader.load()
            for d in docs:
                d.metadata["source_name"] = os.path.basename(pdf_path)
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
        
    print(f"Loaded {len(documents)} pages from PDFs.")
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    
    # Initialize embeddings (MiniLM is local, fast, and free)
    print("Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Remove existing db if exists
    if os.path.exists(DB_DIR):
        print(f"Clearing existing vector database at {DB_DIR}...")
        try:
            # First try client close if open, but in simple script rmtree is fine
            shutil.rmtree(DB_DIR)
        except Exception as e:
            print(f"Warning: Could not clear vector_db directory: {e}")
            
    # Save to Chroma
    print(f"Persisting {len(chunks)} chunks to ChromaDB at {DB_DIR}...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR
    )
    # LangChain Chroma in newer versions automatically persists, but let's call persist() for compatibility
    try:
        db.persist()
    except AttributeError:
        pass
    print("Ingestion completed successfully.")
    return len(chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma DB.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Size of document chunks (characters)")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Overlap between chunks (characters)")
    args = parser.parse_args()
    
    ingest_documents(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
