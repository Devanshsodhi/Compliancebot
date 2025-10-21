"""
Vector Store and RAG System for Document Intelligence
Embeds structured JSON documents for semantic retrieval
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

try: 
    import chromadb
    # Try different import patterns for different ChromaDB versions
    try:
        from chromadb.config import Settings
    except ImportError:
        # Newer versions might not have this
        pass
    CHROMA_AVAILABLE = True
    print(f"✅ ChromaDB available - version: {chromadb.__version__}")
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️  ChromaDB not available. Install with: pip install chromadb")
except Exception as e:
    CHROMA_AVAILABLE = False
    print(f"⚠️  ChromaDB error: {e}")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    print("✅ Sentence Transformers available")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  Sentence Transformers not available. Install with: pip install sentence-transformers")
except Exception as e:
    EMBEDDINGS_AVAILABLE = False
    print(f"⚠️  Sentence Transformers error: {e}")

from config.config import VECTOR_STORE_CONFIG, EMBEDDING_CONFIG


class DocumentVectorStore:
    """
    Manages vector embeddings and semantic search for documents
    Enables RAG (Retrieval Augmented Generation) for Q&A
    """
    
    def __init__(self):
        self.config = VECTOR_STORE_CONFIG
        self.embedding_config = EMBEDDING_CONFIG

        # Initialize embedding model
        self.embedding_model = None
        if EMBEDDINGS_AVAILABLE:
            print("🔄 Loading embedding model...")
            self.embedding_model = SentenceTransformer(
                self.embedding_config["model_name"],
                device=self.embedding_config["device"]
            )
            print("✅ Embedding model loaded")

        # Initialize ChromaDB
        self.client = None
        self.collection = None
        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=self.config["persist_directory"])
                # Handle different ChromaDB versions
                try:
                    self.collection = self.client.get_or_create_collection(
                        name=self.config["collection_name"],
                        metadata={"hnsw:space": self.config["distance_metric"]}
                    )
                except Exception:
                    # Fallback for newer versions
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
        """
        Convert structured JSON document into text chunks for embedding
        Enhanced chunking strategy for better semantic retrieval
        """
        chunks = []
        doc_id = doc.get("document_id", doc.get("source_file", "unknown"))
        doc_type = doc.get("document_type", "unknown")

        # 1. Enhanced header/metadata chunk with more context
        header_fields = [
            ("Order ID", "order_id"),
            ("Document ID", "document_id"),
            ("Order Date", "order_date"),
            ("Shipped Date", "shipped_date"),
            ("Customer ID", "customer_id"),
            ("Customer Name", "customer_name"),
            ("Employee Name", "employee_name"),
            ("Ship Name", "ship_name"),
            ("Ship City", "ship_city"),
            ("Ship Country", "ship_country"),
            ("Total Amount", "total_amount"),
            ("Number of Products", "product_count")
        ]
        
        header_text = f"Document: {doc_id}\nType: {doc_type}\nSource: {doc.get('source_file','unknown')}\n\n"
        header_text += "DOCUMENT SUMMARY:\n"
        
        for label, key in header_fields:
            if key in doc and doc[key] is not None:
                val = doc[key]
                if isinstance(val, float):
                    header_text += f"{label}: ${val:.2f}\n"
                else:
                    header_text += f"{label}: {val}\n"
        
        # Add contextual information
        if doc.get("products"):
            header_text += f"\nThis document contains {len(doc['products'])} products with detailed information about quantities, prices, and totals."
        
        chunks.append({"text": header_text, "chunk_type": "document_overview", "doc_id": doc_id, "doc_type": doc_type})

        # 2. Enhanced products chunks with better context
        if doc.get("products"):
            products = doc["products"]
            
            # Products overview chunk
            overview_text = f"PRODUCTS OVERVIEW FOR {doc_id}:\n"
            overview_text += f"Document Type: {doc_type}\n"
            overview_text += f"Total Products: {len(products)}\n"
            if "total_amount" in doc:
                overview_text += f"Total Order Value: ${doc['total_amount']:.2f}\n\n"
            
            overview_text += "COMPLETE PRODUCT LIST:\n"
            for i, prod in enumerate(products, 1):
                name = prod.get("product_name", "Unknown Product")
                qty = prod.get("quantity", 0)
                price = prod.get("unit_price", 0)
                total = prod.get("total", qty * price)
                overview_text += f"{i}. {name}\n   Quantity: {qty} | Unit Price: ${price:.2f} | Line Total: ${total:.2f}\n"
            
            chunks.append({"text": overview_text, "chunk_type": "products_complete_list", "doc_id": doc_id, "doc_type": doc_type})
            
            # Individual product chunks for detailed queries
            for i, prod in enumerate(products):
                if i >= 20:  # Limit to prevent too many chunks
                    break
                    
                prod_text = f"PRODUCT DETAILS FROM {doc_id}:\n"
                prod_text += f"Document Type: {doc_type}\n\n"
                prod_text += f"Product Name: {prod.get('product_name', 'Unknown')}\n"
                
                if prod.get("product_id"):
                    prod_text += f"Product ID: {prod['product_id']}\n"
                
                prod_text += f"Quantity Ordered: {prod.get('quantity', 0)}\n"
                prod_text += f"Unit Price: ${prod.get('unit_price', 0):.2f}\n"
                prod_text += f"Line Total: ${prod.get('total', 0):.2f}\n"
                
                # Add context about the document
                prod_text += f"\nThis product is part of {doc_type} {doc_id}"
                if doc.get("customer_name"):
                    prod_text += f" for customer {doc['customer_name']}"
                if doc.get("order_date"):
                    prod_text += f" dated {doc['order_date']}"
                
                chunks.append({"text": prod_text, "chunk_type": f"product_detail_{i+1}", "doc_id": doc_id, "doc_type": doc_type})

        # 3. Financial summary chunk
        financial_text = f"FINANCIAL SUMMARY FOR {doc_id}:\n"
        financial_text += f"Document Type: {doc_type}\n"
        if doc.get("total_amount"):
            financial_text += f"Total Amount: ${doc['total_amount']:.2f}\n"
        if doc.get("product_count"):
            financial_text += f"Number of Items: {doc['product_count']}\n"
        if doc.get("customer_name"):
            financial_text += f"Customer: {doc['customer_name']}\n"
        if doc.get("order_date"):
            financial_text += f"Date: {doc['order_date']}\n"
            
        # Calculate additional financial metrics if products exist
        if doc.get("products"):
            avg_price = sum(p.get('unit_price', 0) for p in doc['products']) / len(doc['products'])
            total_qty = sum(p.get('quantity', 0) for p in doc['products'])
            financial_text += f"Average Unit Price: ${avg_price:.2f}\n"
            financial_text += f"Total Quantity: {total_qty}\n"
        
        chunks.append({"text": financial_text, "chunk_type": "financial_summary", "doc_id": doc_id, "doc_type": doc_type})

        # 4. Searchable keywords chunk
        keywords_text = f"SEARCHABLE CONTENT FOR {doc_id}:\n"
        keywords_text += f"Document: {doc_id} {doc_type}\n"
        
        # Add all searchable fields
        searchable_fields = ["customer_name", "employee_name", "ship_name", "ship_city", "ship_country", "order_date", "shipped_date"]
        for field in searchable_fields:
            if doc.get(field):
                keywords_text += f"{field.replace('_', ' ').title()}: {doc[field]}\n"
        
        # Add product names for searchability
        if doc.get("products"):
            keywords_text += "\nProducts in this document: "
            product_names = [p.get('product_name', '') for p in doc['products'] if p.get('product_name')]
            keywords_text += ", ".join(product_names[:10])  # Limit to first 10
        
        chunks.append({"text": keywords_text, "chunk_type": "searchable_keywords", "doc_id": doc_id, "doc_type": doc_type})

        return chunks

    # -------------------- Vector Store Operations -------------------- #
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        if not self.collection or not self.embedding_model:
            print("❌ Vector store not initialized")
            return 0

        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.document_to_text_chunks(doc))

        if not all_chunks:
            print("⚠️  No chunks to embed")
            return 0

        # Embed text
        texts = [c["text"] for c in all_chunks]
        try:
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=self.embedding_config.get("normalize_embeddings", True),
                show_progress_bar=True,
                batch_size=self.embedding_config.get("batch_size", 32)
            )
        except Exception as e:
            print(f"⚠️ Embedding failed, trying without batch_size: {e}")
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=self.embedding_config.get("normalize_embeddings", True),
                show_progress_bar=True
            )

        # Metadata and IDs
        metadatas = [{"doc_id": c["doc_id"], "doc_type": c["doc_type"], "chunk_type": c["chunk_type"]} for c in all_chunks]
        ids = [f"{c['doc_id']}_{c['chunk_type']}_{i}" for i, c in enumerate(all_chunks)]

        # Add to Chroma
        self.collection.add(embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas, ids=ids)
        print(f"✅ Added {len(all_chunks)} chunks")
        return len(all_chunks)

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
    
    def enhanced_search(self, query: str, n_results: int = 8) -> List[Dict[str, Any]]:
        """
        Enhanced search with better context selection and reranking
        Optimized for faster performance
        """
        if not self.collection or not self.embedding_model:
            print("❌ Vector store not initialized")
            return []
        
        # Get fewer initial results for faster processing
        initial_results = min(n_results * 2, 25)  # Reduced from *3 and 50
        query_emb = self.embedding_model.encode(query, normalize_embeddings=self.embedding_config.get("normalize_embeddings", True))
        results = self.collection.query(query_embeddings=[query_emb.tolist()], n_results=initial_results)
        
        if not results or not results.get("documents"):
            return []
        
        # Format initial results
        all_results = []
        for i in range(len(results["documents"][0])):
            all_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else 1.0,
                "relevance_score": 1 - results["distances"][0][i] if "distances" in results else 0.0
            })
        
        # Smart reranking based on query type and chunk type
        query_lower = query.lower()
        
        # Determine query intent
        is_product_query = any(keyword in query_lower for keyword in 
                              ['product', 'item', 'list', 'what', 'show', 'order', 'purchase', 'buy'])
        is_financial_query = any(keyword in query_lower for keyword in 
                                ['total', 'amount', 'price', 'cost', 'money', 'pay', 'dollar'])
        is_customer_query = any(keyword in query_lower for keyword in 
                               ['customer', 'client', 'buyer', 'who', 'name'])
        is_date_query = any(keyword in query_lower for keyword in 
                           ['when', 'date', 'time', 'shipped', 'order'])
        
        # Apply query-specific boosting
        for result in all_results:
            chunk_type = result['metadata'].get('chunk_type', '')
            boost = 0.0
            
            # Boost relevant chunk types based on query intent
            if is_product_query:
                if 'product' in chunk_type:
                    boost += 0.3
                elif chunk_type == 'document_overview':
                    boost += 0.1
            
            if is_financial_query:
                if chunk_type == 'financial_summary':
                    boost += 0.3
                elif 'product' in chunk_type:
                    boost += 0.2
            
            if is_customer_query or is_date_query:
                if chunk_type in ['document_overview', 'searchable_keywords']:
                    boost += 0.2
            
            # Always include document overview for context
            if chunk_type == 'document_overview':
                boost += 0.1
            
            result['relevance_score'] += boost
        
        # Sort by enhanced relevance score
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Ensure diversity - don't return too many chunks from same document
        final_results = []
        doc_chunk_count = {}
        
        for result in all_results:
            doc_id = result['metadata']['doc_id']
            chunk_type = result['metadata']['chunk_type']
            
            # Limit chunks per document, but allow more for highly relevant ones
            max_chunks_per_doc = 4 if result['relevance_score'] > 0.7 else 3
            
            if doc_chunk_count.get(doc_id, 0) < max_chunks_per_doc:
                final_results.append(result)
                doc_chunk_count[doc_id] = doc_chunk_count.get(doc_id, 0) + 1
                
                if len(final_results) >= n_results:
                    break
        
        return final_results[:n_results]

    def get_document_context(self, doc_id: str) -> Optional[str]:
        if not self.collection:
            return None
        results = self.collection.get(where={"doc_id": doc_id})
        if results and results.get("documents"):
            return "\n\n".join(results["documents"])
        return None
    
    def get_contextual_chunks(self, doc_ids: List[str], chunk_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get specific chunks from documents for building context
        """
        if not self.collection:
            return []
        
        where_clause = {"doc_id": {"$in": doc_ids}}
        if chunk_types:
            where_clause["chunk_type"] = {"$in": chunk_types}
        
        results = self.collection.get(where=where_clause)
        formatted = []
        
        if results and results.get("documents"):
            for i in range(len(results["documents"])):
                formatted.append({
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                    "relevance_score": 1.0  # Default high relevance for requested chunks
                })
        
        return formatted

    def clear_store(self):
        if self.collection and self.client:
            try:
                self.client.delete_collection(self.config["collection_name"])
                # Handle different ChromaDB versions
                try:
                    self.collection = self.client.create_collection(
                        name=self.config["collection_name"],
                        metadata={"hnsw:space": self.config["distance_metric"]}
                    )
                except Exception:
                    # Fallback for newer versions
                    self.collection = self.client.create_collection(
                        name=self.config["collection_name"]
                    )
                print("✅ Vector store cleared")
            except Exception as e:
                print(f"⚠️ Vector store clear failed: {e}")

    def get_all_document_ids(self) -> List[str]:
        if not self.collection:
            return []
        results = self.collection.get()
        if results and results.get("metadatas"):
            return list({m["doc_id"] for m in results["metadatas"]})
        return []

# -------------------- Quick Test -------------------- #
if __name__ == "__main__":
    store = DocumentVectorStore()
    test_doc = {"document_id": "TEST-001", "document_type": "invoice", "total_amount": 1000, "vendor": "Test Vendor"}
    store.add_documents([test_doc])
    results = store.search("What is the total?")
    print(f"Search results: {len(results)}")
