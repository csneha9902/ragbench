import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from retrieve import get_embeddings
from generate import get_llm, generate_answer

# Ragas imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

EVAL_DATASET_PATH = "evaluation/eval_dataset.json"
RESULTS_CSV_PATH = "evaluation/results.csv"

def run_evaluation(top_n=5, alpha=0.5):
    if not os.path.exists(EVAL_DATASET_PATH):
        raise FileNotFoundError(f"Evaluation dataset not found at {EVAL_DATASET_PATH}")
        
    print(f"Loading evaluation dataset from {EVAL_DATASET_PATH}...")
    with open(EVAL_DATASET_PATH, "r") as f:
        eval_data = json.load(f)
        
    print(f"Loaded {len(eval_data)} evaluation test cases.")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    latencies = []
    
    print("\nRunning questions through the RAG pipeline...")
    for idx, item in enumerate(eval_data):
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"[{idx+1}/{len(eval_data)}] Q: {q}")
        
        start_time = time.time()
        res = generate_answer(q, top_n=top_n, alpha=alpha)
        latency = time.time() - start_time
        
        questions.append(q)
        answers.append(res["answer"])
        # Ragas expects context as a list of strings
        contexts.append(res["retrieved_contexts"])
        ground_truths.append(gt)
        latencies.append(latency)
        
    # Build dataset
    eval_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    print("\nCreating dataset for Ragas...")
    dataset = Dataset.from_dict(eval_dict)
    
    # Configure custom LLM and Embeddings for Ragas
    print("Configuring LLM and Embeddings wrapper for Ragas...")
    llm = get_llm()
    embeddings = get_embeddings()
    
    ragas_llm = LangchainLLMWrapper(langchain_llm=llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings=embeddings)
    
    # Assign custom LLM and embeddings to metrics
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for m in metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_embeddings
            
    print("Running Ragas evaluation (this may take a few minutes as LLM evaluates answers)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics
    )
    
    # Convert results to dataframe and save
    df_results = results.to_pandas()
    # Add question latency to results
    df_results["latency_seconds"] = latencies
    
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    df_results.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nEvaluation complete. Results saved to {RESULTS_CSV_PATH}")
    
    # Calculate summary metrics
    summary = {
        "Average Faithfulness": df_results["faithfulness"].mean(),
        "Average Answer Relevancy": df_results["answer_relevancy"].mean(),
        "Average Context Recall": df_results["context_recall"].mean(),
        "Average Context Precision": df_results["context_precision"].mean(),
        "Average Latency (s)": df_results["latency_seconds"].mean()
    }
    
    print("\n--- RAG Evaluation Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")
        
    return summary

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"\nError running evaluation: {e}")
