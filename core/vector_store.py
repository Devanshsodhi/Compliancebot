"""
Vector Store and RAG System for Document Intelligence
Embeds structured JSON documents for semantic retrieval
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# ------------------ Optional Dependencies ------------------ #
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
    print(f"✅ ChromaDB available - version: {chromadb.__version__}")
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️ ChromaDB not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    print("✅ Sentence Transformers available")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ Sentence Transformers not available. Install with: pip install sentence-transformers")

from config.config import VECTOR_STORE_CONFIG, EMBEDDING_CONFIG


# ------------------ Vector Store Class ------------------ #
class DocumentVectorStore:
    """Manages vector embeddings and semantic search for documents."""

    def __init__(self):
        self.config = VECTOR_STORE_CONFIG
        self.embedding_config = EMBEDDING_CONFIG

        # Initialize embedding model
        self.embedding_model = None
        if EMBEDDINGS_AVAILABLE:
            print("🔄 Loading embedding model...")
            self.embedding_model = SentenceTransformer(
                self.embedding_config["model_name"],
                device=self.embedding_config.get("device", "cpu")
            )
            print("✅ Embedding model loaded")

        # Initialize ChromaDB
        self.client = None
        self.collection = None
        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=self.config["persist_directory"])
                self.collection = self.client.get_or_create_collection(
                    name=self.config["collection_name"]
                )
                print("✅ ChromaDB client and collection initialized")
            except Exception as e:
                print(f"❌ ChromaDB initialization failed: {e}")
                self.client = None
                self.collection = None

    # -------------------- Document Chunking -------------------- #
    def document_to_text_chunks(self, doc: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert a JSON document into text chunks for embedding."""
        chunks = []
        doc_id = doc.get("document_id", doc.get("source_file", "unknown"))
        doc_type = doc.get("document_type", "unknown")

        # Document overview
        overview_text = f"Document: {doc_id}\nType: {doc_type}\n"
        for key, label in [("order_id","Order ID"), ("customer_name","Customer"), ("total_amount","Total Amount")]:
            if key in doc:
                val = doc[key]
                overview_text += f"{label}: {val}\n"
        chunks.append({"text": overview_text, "chunk_type": "document_overview", "doc_id": doc_id, "doc_type": doc_type})

        # Products
        if doc.get("products"):
            for i, prod in enumerate(doc["products"]):
                prod_text = f"Product {i+1}: {prod.get('product_name','Unknown')}\n"
                prod_text += f"Quantity: {prod.get('quantity',0)} | Unit Price: {prod.get('unit_price',0)} | Line Total: {prod.get('total',0)}\n"
                chunks.append({"text": prod_text, "chunk_type": f"product_{i+1}", "doc_id": doc_id, "doc_type": doc_type})

        return chunks

    # -------------------- Add Documents -------------------- #
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        if not self.collection or not self.embedding_model:
            print("❌ Vector store not initialized")
            return 0

        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.document_to_text_chunks(doc))

        if not all_chunks:
            print("⚠️ No chunks to embed")
            return 0

        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=self.embedding_config.get("normalize_embeddings", True),
            show_progress_bar=True
        )

        metadatas = [{"doc_id": c["doc_id"], "doc_type": c["doc_type"], "chunk_type": c["chunk_type"]} for c in all_chunks]
        ids = [f"{c['doc_id']}_{c['chunk_type']}_{i}" for i, c in enumerate(all_chunks)]

        self.collection.add(embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas, ids=ids)
        print(f"✅ Added {len(all_chunks)} chunks")
        return len(all_chunks)

    # -------------------- Search -------------------- #
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not self.collection or not self.embedding_model:
            print("❌ Vector store not initialized")
            return []

        query_emb = self.embedding_model.encode(query, normalize_embeddings=self.embedding_config.get("normalize_embeddings", True))
        results = self.collection.query(query_embeddings=[query_emb.tolist()], n_results=n_results)

        formatted = []
        if results and results.get("documents"):
            for i in range(len(results["documents"][0])):
                formatted.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        return formatted

    # -------------------- Clear Store -------------------- #
    def clear_store(self):
        if self.collection and self.client:
            try:
                self.client.delete_collection(self.config["collection_name"])
                self.collection = self.client.get_or_create_collection(name=self.config["collection_name"])
                print("✅ Vector store cleared")
            except Exception as e:
                print(f"⚠️ Vector store clear failed: {e}")

