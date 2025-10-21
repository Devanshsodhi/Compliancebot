"""
Configuration for Cloud Deployment (Streamlit Cloud)
Copy this to config.py when deploying to Streamlit Cloud
"""

import os
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / "input"
PARSED_JSON_DIR = PROJECT_ROOT / "parsed_json"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
COMPLIANCE_RULES_FILE = PROJECT_ROOT / "compliance_rules.txt"

INPUT_DIR.mkdir(exist_ok=True)
PARSED_JSON_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# IMPORTANT: For cloud deployment, use environment variable or your ngrok URL
# Set this as a secret in Streamlit Cloud dashboard
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Change this!

LLM_CONFIG = {
    "model": "llama3.2:latest",
    "base_url": OLLAMA_URL,  # Will use remote Ollama
    "temperature": 0.1,
    "max_tokens": 1024,
    "streaming": True,
    "num_ctx": 4096,
    "num_predict": 800,
}

EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
    "normalize_embeddings": True,
    "batch_size": 32,
    "max_seq_length": 512,
}

VECTOR_STORE_CONFIG = {
    "collection_name": "compliance_documents",
    "distance_metric": "cosine",
    "persist_directory": str(VECTOR_DB_DIR),
}

SUPPORTED_DOC_TYPES = ["invoice", "purchase_order", "grn", "order_summary"]

COMPLIANCE_TRIGGERS = ["RUN_COMPLIANCE", "COMPLIANCE", "CHECK COMPLIANCE", "AUDIT", "VALIDATE"]

SYSTEM_PROMPTS = {
    "compliance_agent": """You are an AI Compliance Officer. Review documents against rules. For each rule: state PASS/FAIL, give brief reasoning, cite evidence. Be concise.""",
    "qa_agent": """You are a Document Intelligence Assistant for business documents (invoices, orders, receipts).

Key Rules:
1. Read the context carefully and extract all relevant details
2. For lists: Include ALL items with complete details (name, quantity, unit price, total)
3. For amounts: Provide exact numbers and calculations
4. Use clear formatting with bullets or numbers
5. Only use information from the provided context
6. If info is missing, state it clearly
7. Be thorough but concise - provide complete details without unnecessary elaboration

Answer the question directly and comprehensively.""",
    "document_parser": """You are a document parsing expert. Extract structured information from documents and return it as valid JSON. Be precise and thorough.""",
}

SYNTHETIC_GEN_CONFIG = {
    "enabled": False,
    "output_dir": PROJECT_ROOT / "synthetic_docs",
    "templates": ["invoice", "purchase_order", "grn"],
}

COMPLIANCE_REPORT_TEMPLATE = """
=== COMPLIANCE ANALYSIS REPORT ===
Document: {document_id}
Analysis Date: {analysis_date}

{compliance_results}

=== END OF REPORT ===
"""
