"""
Document Processing Pipeline
Handles parsing and automatic vector store embedding
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.document_parser import DocumentParser
from core.vector_store import DocumentVectorStore


class DocumentProcessor:
    """
    Unified document processing pipeline that handles:
    1. PDF parsing
    2. JSON storage
    3. Automatic vector store embedding
    """
    
    def __init__(self):
        self.parser = DocumentParser()
        self.vector_store = None
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize vector store if available"""
        try:
            self.vector_store = DocumentVectorStore()
            if self.vector_store.embedding_model and self.vector_store.collection:
                print("✅ Vector store initialized for automatic embedding")
            else:
                print("⚠️ Vector store not fully initialized - embeddings disabled")
                self.vector_store = None
        except Exception as e:
            print(f"⚠️ Vector store initialization failed: {e}")
            self.vector_store = None
    
    def process_documents(self, force_reparse: bool = False, auto_embed: bool = True) -> List[Dict[str, Any]]:
        """
        Process all documents with automatic vector store embedding
        
        Args:
            force_reparse: Re-parse even if JSON exists
            auto_embed: Automatically add to vector store
        
        Returns:
            List of parsed document data
        """
        print("🚀 Starting document processing pipeline...")
        
        # Step 1: Parse documents
        parsed_docs = self.parser.parse_all_documents(force_reparse=force_reparse)
        
        if not parsed_docs:
            print("❌ No documents to process")
            return []
        
        # Step 2: Auto-embed in vector store if enabled
        if auto_embed and self.vector_store:
            print(f"\n🔄 Adding {len(parsed_docs)} documents to vector store...")
            
            try:
                # Clear existing embeddings if force_reparse
                if force_reparse:
                    print("🗑️ Clearing existing vector store...")
                    self.vector_store.clear_store()
                
                # Add documents to vector store
                chunks_added = self.vector_store.add_documents(parsed_docs)
                print(f"✅ Added {chunks_added} chunks to vector store")
                
                # Verify embeddings
                all_doc_ids = self.vector_store.get_all_document_ids()
                print(f"📊 Vector store now contains {len(all_doc_ids)} documents")
                
            except Exception as e:
                print(f"❌ Vector store embedding failed: {e}")
                print("📄 Documents parsed successfully but not embedded")
        
        elif auto_embed and not self.vector_store:
            print("⚠️ Vector store not available - skipping embedding")
        
        print(f"\n🎉 Processing complete! {len(parsed_docs)} documents ready")
        return parsed_docs
    
    def add_single_document(self, pdf_path: Path, auto_embed: bool = True) -> Optional[Dict[str, Any]]:
        """
        Process a single document
        
        Args:
            pdf_path: Path to PDF file
            auto_embed: Automatically add to vector store
        
        Returns:
            Parsed document data or None if failed
        """
        print(f"📄 Processing single document: {pdf_path.name}")
        
        # Parse document
        doc_data = self.parser.parse_document(pdf_path)
        
        if not doc_data:
            print(f"❌ Failed to parse {pdf_path.name}")
            return None
        
        # Save JSON
        json_path = self.parser.parsed_json_dir / f"{pdf_path.stem}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, indent=2, ensure_ascii=False)
        
        # Auto-embed if enabled
        if auto_embed and self.vector_store:
            try:
                chunks_added = self.vector_store.add_documents([doc_data])
                print(f"✅ Added {chunks_added} chunks to vector store")
            except Exception as e:
                print(f"⚠️ Vector embedding failed: {e}")
        
        return doc_data
    
    def refresh_vector_store(self) -> bool:
        """
        Refresh vector store with all parsed documents
        
        Returns:
            True if successful, False otherwise
        """
        if not self.vector_store:
            print("❌ Vector store not available")
            return False
        
        # Load all parsed documents
        parsed_dir = Path(self.parser.parsed_json_dir)
        docs = []
        
        if parsed_dir.exists():
            for json_file in parsed_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        docs.append(json.load(f))
                except Exception as e:
                    print(f"⚠️ Failed to load {json_file.name}: {e}")
        
        if not docs:
            print("❌ No parsed documents found")
            return False
        
        print(f"🔄 Refreshing vector store with {len(docs)} documents...")
        
        try:
            # Clear and rebuild
            self.vector_store.clear_store()
            chunks_added = self.vector_store.add_documents(docs)
            print(f"✅ Vector store refreshed with {chunks_added} chunks")
            return True
        except Exception as e:
            print(f"❌ Vector store refresh failed: {e}")
            return False
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get current processing status
        
        Returns:
            Status information dictionary
        """
        # Count PDFs and JSONs
        pdf_count = len(list(self.parser.input_dir.glob("*.pdf")))
        json_count = len(list(self.parser.parsed_json_dir.glob("*.json")))
        
        # Vector store status
        vector_status = {
            "available": self.vector_store is not None,
            "document_count": 0,
            "chunk_count": 0
        }
        
        if self.vector_store:
            try:
                doc_ids = self.vector_store.get_all_document_ids()
                vector_status["document_count"] = len(doc_ids)
                # Note: chunk_count would require additional method
            except:
                pass
        
        return {
            "pdf_files": pdf_count,
            "parsed_documents": json_count,
            "vector_store": vector_status,
            "parser_available": True,
            "ready_for_queries": vector_status["available"] and json_count > 0
        }


# Convenience function for easy import
def process_all_documents(force_reparse: bool = False, auto_embed: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function to process all documents
    
    Args:
        force_reparse: Re-parse even if JSON exists
        auto_embed: Automatically add to vector store
    
    Returns:
        List of parsed document data
    """
    processor = DocumentProcessor()
    return processor.process_documents(force_reparse=force_reparse, auto_embed=auto_embed)


if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()
    status = processor.get_processing_status()
    print(f"Processing Status: {json.dumps(status, indent=2)}")
    
    # Process documents
    docs = processor.process_documents(force_reparse=False, auto_embed=True)
    print(f"Processed {len(docs)} documents")
