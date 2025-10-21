import json
from typing import Dict, Any, List, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not available. LLM features disabled.")

from config.config import LLM_CONFIG, SYSTEM_PROMPTS


class LLMOrchestrator:
    def __init__(self):
        self.config = LLM_CONFIG
        self.system_prompts = SYSTEM_PROMPTS
        self.available = OLLAMA_AVAILABLE
        if self.available:
            self._test_connection()

    def _test_connection(self):
        try:
            ollama.list()
            print(f"✅ Connected to Ollama (Model: {self.config['model']})")
        except Exception as e:
            print(f"⚠️ Ollama connection issue: {e}")
            print("   Make sure Ollama is running: ollama serve")
            self.available = False

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, context: Optional[str] = None) -> str:
        if not self.available:
            return "❌ LLM not available. Please check Ollama installation."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = prompt if not context else f"Context:\n{context}\n\nQuestion: {prompt}"
        messages.append({"role": "user", "content": user_content})

        try:
            response = ollama.chat(
                model=self.config["model"],
                messages=messages,
                options={
                    "temperature": self.config["temperature"],
                    "num_predict": self.config.get("num_predict", 1024),
                    "num_ctx": self.config.get("num_ctx", 4096),
                    "top_k": 40,
                    "top_p": 0.9,  # Slightly lower for more focused responses
                    "repeat_penalty": 1.1,
                    "num_thread": 8,
                    "num_gpu": 0,  # Force CPU for consistency
                },
                stream=False  # Disable streaming for faster batch processing
            )
            return response["message"]["content"]
        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

    def compliance_analysis(self, document_data: Dict[str, Any], compliance_rules: str) -> str:
        prompt = self._build_compliance_prompt(document_data, compliance_rules)
        system_prompt = self.system_prompts["compliance_agent"]
        print("🧠 LLM analyzing compliance...")
        return self.generate_response(prompt, system_prompt)

    def _build_compliance_prompt(self, document_data: Dict[str, Any], compliance_rules: str) -> str:
        essential_data = {
            "document_id": document_data.get("document_id"),
            "document_type": document_data.get("document_type"),
            "total_amount": document_data.get("total_amount"),
            "product_count": document_data.get("product_count"),
            "order_date": document_data.get("order_date"),
            "customer_name": document_data.get("customer_name"),
        }
        doc_summary = json.dumps(essential_data, indent=2)

        return (
            "Analyze this document against compliance rules. Be concise.\n\n"
            f"DOCUMENT:\n{doc_summary}\n\n"
            f"RULES:\n{compliance_rules[:500]}\n\n"
            "For each rule: State PASS/FAIL, brief reason, evidence. Keep it short."
        )

    def answer_question(self, question: str, retrieved_context: List[Dict[str, Any]]) -> str:
        """
        Enhanced question answering with better context utilization
        """
        if not retrieved_context:
            return "❌ I couldn't find any relevant information in the documents to answer your question. Please make sure documents are loaded and try rephrasing your question."
        
        # Enhanced context formatting
        context_text = self._format_enhanced_rag_context(retrieved_context, question)
        system_prompt = self.system_prompts["qa_agent"]
        
        print(f"🔍 Question: {question}")
        print(f"📚 Context chunks: {len(retrieved_context)}")
        print(f"📄 Documents referenced: {len(set(chunk['metadata']['doc_id'] for chunk in retrieved_context))}")
        
        # Generate response with enhanced prompting
        response = self.generate_response(question, system_prompt, context_text)
        
        # Validate response quality
        if len(response.strip()) < 20:
            fallback_response = self._generate_fallback_response(question, retrieved_context)
            print("⚠️ Short response detected, using fallback")
            return fallback_response
        
        print(f"✅ Response length: {len(response)} characters")
        return response

    def _format_rag_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return "No relevant documents found."
        
        context = "RELEVANT DOCUMENT INFORMATION:\n\n"
        for i, chunk in enumerate(retrieved_chunks, 1):
            doc_id = chunk['metadata'].get('doc_id', 'Unknown')
            doc_type = chunk['metadata'].get('doc_type', 'unknown')
            chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
            
            # Include more content for product-related chunks
            if 'product' in chunk_type.lower():
                content = chunk['text'][:1000]  # More content for product lists
            else:
                content = chunk['text'][:500]
            
            context += f"--- Document {i}: {doc_id} ({doc_type}) ---\n"
            context += f"Section: {chunk_type}\n"
            context += f"{content}\n\n"
        
        return context
    
    def _format_enhanced_rag_context(self, retrieved_chunks: List[Dict[str, Any]], question: str) -> str:
        """
        Optimized context formatting for faster processing
        """
        if not retrieved_chunks:
            return "No relevant documents found."
        
        # Group chunks by document
        docs_chunks = {}
        for chunk in retrieved_chunks:
            doc_id = chunk['metadata'].get('doc_id', 'Unknown')
            if doc_id not in docs_chunks:
                docs_chunks[doc_id] = []
            docs_chunks[doc_id].append(chunk)
        
        # Sort chunks within each document by relevance
        for doc_id in docs_chunks:
            docs_chunks[doc_id].sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        context = "DOCUMENT INFORMATION:\n\n"
        
        # More compact format for faster processing
        for doc_id, chunks in docs_chunks.items():
            if not chunks:
                continue
            
            first_chunk = chunks[0]
            doc_type = first_chunk['metadata'].get('doc_type', 'unknown')
            
            context += f"Document: {doc_id} ({doc_type})\n"
            
            for chunk in chunks:
                chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
                
                # Optimized content length for speed
                if 'product' in chunk_type.lower() or 'complete_list' in chunk_type:
                    content = chunk['text'][:1000]  # Reduced for faster processing
                elif chunk_type == 'financial_summary':
                    content = chunk['text'][:600]
                else:
                    content = chunk['text'][:400]
                
                context += f"\n[{chunk_type}]\n{content}\n"
            
            context += "\n" + "-" * 40 + "\n"
        
        # Simplified instructions
        context += "\nAnswer the question using the above information. Include all relevant details.\n"
        
        return context
    
    def _generate_fallback_response(self, question: str, retrieved_context: List[Dict[str, Any]]) -> str:
        """
        Generate a fallback response when the main response is too short or inadequate
        """
        if not retrieved_context:
            return "I don't have enough information in the loaded documents to answer your question."
        
        # Extract key information from context
        doc_ids = set(chunk['metadata']['doc_id'] for chunk in retrieved_context)
        doc_types = set(chunk['metadata']['doc_type'] for chunk in retrieved_context)
        
        fallback = f"Based on the {len(doc_ids)} document(s) I found ({', '.join(doc_ids)}), "
        
        # Try to provide some relevant information
        product_chunks = [c for c in retrieved_context if 'product' in c['metadata'].get('chunk_type', '')]
        if product_chunks and any(keyword in question.lower() for keyword in ['product', 'item', 'list', 'what']):
            fallback += "here are the key details I can provide:\n\n"
            for chunk in product_chunks[:2]:  # Limit to prevent too long response
                content = chunk['text'][:300]
                fallback += f"From {chunk['metadata']['doc_id']}: {content}\n\n"
        else:
            fallback += "I found relevant information but need you to be more specific about what you're looking for."
        
        return fallback

    def extract_document_fields(self, raw_text: str, doc_type: str) -> Dict[str, Any]:
        prompt = (
            f"Extract structured information from this {doc_type} document.\n\n"
            f"Document Text:\n{raw_text[:2000]}\n\n"
            "Extract the following fields as JSON:\n"
            "- document_id\n- date\n- vendor/supplier\n- customer\n- total_amount\n"
            "- line_items (if present)\n- payment_terms\n- any other relevant fields\n"
            "Return ONLY valid JSON, no additional text."
        )
        system_prompt = self.system_prompts["document_parser"]
        response = self.generate_response(prompt, system_prompt)

        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except:
            pass
        return {"raw_extraction": response}

    def generate_compliance_summary(self, all_reports: List[str]) -> str:
        combined_reports = "\n\n".join(all_reports)
        prompt = (
            "Based on these compliance analysis reports, generate an executive summary.\n\n"
            f"{combined_reports}\n\n"
            "Provide:\n"
            "1. Overall compliance status (% passing)\n"
            "2. Critical issues requiring immediate attention\n"
            "3. Common patterns or systemic issues\n"
            "4. Recommended priority actions\n\n"
            "Keep it concise and actionable."
        )
        system_prompt = self.system_prompts["compliance_agent"]
        return self.generate_response(prompt, system_prompt)


class PromptTemplates:
    @staticmethod
    def compliance_check_template(doc_type: str) -> str:
        templates = {
            "invoice": (
                "Analyze this invoice for compliance with standard invoicing regulations:\n"
                "- Proper invoice number format\n- Valid dates (invoice date, due date)\n"
                "- Complete vendor/customer information\n- Accurate calculations (subtotal + tax = total)\n"
                "- Payment terms clearly stated\n- Line items properly detailed"
            ),
            "purchase_order": (
                "Analyze this purchase order for compliance:\n"
                "- Valid PO number\n- Authorized approver present\n- Delivery dates reasonable\n"
                "- Pricing within budget constraints\n- Complete shipping information\n"
                "- Terms and conditions acknowledged"
            ),
            "grn": (
                "Analyze this Goods Receipt Note for compliance:\n"
                "- Matches corresponding PO\n- Receipt date is valid\n- Quantities match ordered amounts\n"
                "- Quality inspection completed\n- Received by authorized personnel\n- Proper documentation"
            )
        }
        return templates.get(doc_type, "Perform general compliance analysis.")

    @staticmethod
    def few_shot_compliance_example() -> str:
        return (
            "EXAMPLE COMPLIANCE ANALYSIS:\n\n"
            "Document: INV-001\nRule: \"Invoice must have valid date\"\nStatus: PASS\n"
            "Reasoning: Invoice date is 2024-01-15, which is a valid date format\n"
            "Evidence: \"date\": \"01/15/2024\"\nSuggested Action: None required\n\n"
            "Document: INV-001\nRule: \"Payment terms must be specified\"\nStatus: FAIL\n"
            "Reasoning: No payment terms field found in document data\n"
            "Evidence: Field \"payment_terms\" is missing\n"
            "Suggested Action: Contact vendor to clarify payment terms and update invoice\n\n"
            "Now analyze the provided document following this format."
        )


if __name__ == "__main__":
    orchestrator = LLMOrchestrator()
    if orchestrator.available:
        test_response = orchestrator.generate_response(
            "Say hello and confirm you're working.",
            system_prompt="You are a helpful AI assistant."
        )
        print(f"Test response: {test_response}")
