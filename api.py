import os
import shutil
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
from ingest import ingest_documents
from generate import generate_answer
from retrieve import reset_retrievers
from evaluate import run_evaluation, RESULTS_CSV_PATH

app = FastAPI(
    title="RAGBench: Evaluation-Driven RAG QA API",
    description="Backend API supporting document ingestion, hybrid search, QA, and Ragas automated evaluations.",
    version="1.0"
)

class QueryRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5
    alpha: Optional[float] = 0.5  # 0.0 = Keyword, 1.0 = Vector, 0.5 = Hybrid

class IngestRequest(BaseModel):
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 50

@app.get("/")
def read_root():
    return {
        "project": "RAGBench",
        "description": "Retrieval-Augmented QA System with Automated Evaluation",
        "status": "online"
    }

@app.post("/ingest")
def trigger_ingest(request: IngestRequest):
    try:
        reset_retrievers()
        num_chunks = ingest_documents(chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap)
        return {
            "status": "success",
            "message": f"Successfully ingested documents into {num_chunks} chunks."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), chunk_size: int = 500, chunk_overlap: int = 50):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    upload_dir = "data"
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file to disk: {e}")
        
    try:
        reset_retrievers()
        num_chunks = ingest_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return {
            "status": "success",
            "message": f"Uploaded '{file.filename}' and re-indexed. Total chunks: {num_chunks}."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def run_query(request: QueryRequest):
    try:
        res = generate_answer(request.query, top_n=request.top_n, alpha=request.alpha)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
def trigger_evaluation(top_n: int = 5, alpha: float = 0.5):
    try:
        summary = run_evaluation(top_n=top_n, alpha=alpha)
        return {
            "status": "success",
            "message": "Evaluation completed successfully.",
            "metrics": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_latest_metrics():
    if not os.path.exists(RESULTS_CSV_PATH):
        return {
            "status": "no_eval_run",
            "message": "No evaluation has been run yet."
        }
    try:
        df = pd.read_csv(RESULTS_CSV_PATH)
        
        # Guard against empty dataset
        if df.empty:
            return {
                "status": "empty_results",
                "message": "Evaluation results file is empty."
            }
            
        metrics = {
            "faithfulness": float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
            "context_recall": float(df["context_recall"].mean()),
            "context_precision": float(df["context_precision"].mean()),
            "avg_latency_s": float(df["latency_seconds"].mean()) if "latency_seconds" in df.columns else 0.0,
            "total_questions": len(df)
        }
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
def get_detailed_results():
    if not os.path.exists(RESULTS_CSV_PATH):
        return {
            "status": "no_results",
            "data": []
        }
    try:
        df = pd.read_csv(RESULTS_CSV_PATH)
        df = df.fillna(0)  # Handle any missing evaluation scores gracefully
        data = df.to_dict(orient="records")
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
