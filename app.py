import streamlit as st
import json
from pathlib import Path
from core.vector_store import DocumentVectorStore
from core.llm_orchestrator import LLMOrchestrator
from core.document_parser import DocumentParser
from core.document_processor import DocumentProcessor
from core.compliance_engine import ComplianceEngine
import time

# Page configuration
st.set_page_config(
    page_title="AI Compliance Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .doc-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
    }
    .source-box {
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: #f8f9fa;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

def load_documents():
    """Load parsed documents"""
    parsed_dir = Path("parsed_json")
    docs = []
    
    if parsed_dir.exists():
        for json_file in parsed_dir.glob("*.json"):
            with open(json_file) as f:
                docs.append(json.load(f))
    
    return docs

def initialize_systems():
    """Initialize vector store and LLM"""
    if not st.session_state.system_ready:
        with st.spinner("🔄 Initializing AI systems..."):
            # Load documents
            st.session_state.documents = load_documents()
            
            if not st.session_state.documents:
                st.error("❌ No documents found. Please add PDFs to the input/ folder and parse them first.")
                return False
            
            # Initialize vector store with enhanced features
            st.session_state.vector_store = DocumentVectorStore()
            st.session_state.vector_store.clear_store()
            chunks_added = st.session_state.vector_store.add_documents(st.session_state.documents)
            print(f"✅ Added {chunks_added} chunks to vector store")
            
            # Initialize LLM
            st.session_state.llm = LLMOrchestrator()
            
            if not st.session_state.llm.available:
                st.warning("⚠️ LLM not available. Make sure Ollama is running.")
                return False
            
            st.session_state.system_ready = True
            return True
    return True

def ask_question(question):
    """Process question and return answer with optimized performance"""
    if not st.session_state.vector_store or not st.session_state.llm:
        return None, []
    
    # Use enhanced search with optimized chunk count for speed
    context_results = st.session_state.vector_store.enhanced_search(question, n_results=6)
    
    if not context_results:
        return "⚠️ No relevant information found in documents. Please make sure your documents are properly loaded and try rephrasing your question.", []
    
    # Get LLM answer with enhanced context
    answer = st.session_state.llm.answer_question(question, context_results)
    
    return answer, context_results

# Header
st.markdown('<div class="main-header">🤖 AI Compliance Bot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your documents • Run compliance checks • Get instant answers</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📚 System Status")
    
    if st.button("🔄 Initialize/Reload System", use_container_width=True):
        st.session_state.system_ready = False
        initialize_systems()
        if st.session_state.system_ready:
            st.success("✅ System initialized!")
    
    st.divider()
    
    # Document summary
    if st.session_state.documents:
        st.subheader("📄 Loaded Documents")
        for doc in st.session_state.documents:
            doc_id = doc.get('document_id', 'Unknown')
            doc_type = doc.get('document_type', 'unknown')
            total = doc.get('total_amount', 0)
            products = doc.get('product_count', 0)
            
            with st.container():
                st.markdown(f"""
                <div class="doc-card">
                    <strong>{doc_id}</strong><br>
                    Type: {doc_type}<br>
                    Products: {products}<br>
                    Total: ${total:.2f}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No documents loaded yet")
    
    st.divider()
    
    # System info
    st.subheader("⚙️ System Info")
    if st.session_state.system_ready:
        st.success("✅ Vector Store Ready")
        st.success("✅ LLM Connected")
    else:
        st.warning("⚠️ System not initialized")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Documents", "💬 Q&A Chat", "⚖️ Compliance Check", "📊 Document Viewer"])

# Tab 1: Upload Documents
with tab1:
    st.header("📤 Upload and Process Documents")
    
    st.info("Upload PDF documents to parse and analyze. The system will extract text, products, and metadata automatically.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents (invoices, purchase orders, etc.)"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        
        # Show uploaded files
        st.markdown("**📄 Selected Files:**")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size / 1024:.1f} KB)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                # Save uploaded files to input directory
                input_dir = Path("input")
                input_dir.mkdir(exist_ok=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save files
                status_text.text("📥 Saving uploaded files...")
                for i, uploaded_file in enumerate(uploaded_files):
                    file_path = input_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    progress_bar.progress((i + 1) / (len(uploaded_files) * 3))
                
                # Process documents with automatic vector store embedding
                status_text.text("📄 Processing documents and building vector database...")
                processor = DocumentProcessor()
                parsed_docs = processor.process_documents(force_reparse=True, auto_embed=True)
                progress_bar.progress(2 / 3)
                
                if parsed_docs:
                    st.success(f"✅ Successfully processed {len(parsed_docs)} document(s)")
                    
                    # Show parsed documents
                    with st.expander("📊 View Processed Documents"):
                        for doc in parsed_docs:
                            st.markdown(f"""
                            **{doc.get('source_file', 'Unknown')}**
                            - Type: {doc.get('document_type', 'unknown')}
                            - Products: {doc.get('product_count', 0)}
                            - Total: ${doc.get('total_amount', 0):.2f}
                            """)
                    
                    # Initialize LLM system (vector store already initialized by processor)
                    status_text.text("🔄 Initializing LLM system...")
                    st.session_state.documents = parsed_docs
                    st.session_state.vector_store = processor.vector_store
                    st.session_state.llm = LLMOrchestrator()
                    
                    if st.session_state.llm.available:
                        st.session_state.system_ready = True
                        progress_bar.progress(1.0)
                        status_text.empty()
                        st.success("🎉 All done! System is ready with enhanced vector search. Go to Q&A Chat to ask questions.")
                        st.balloons()
                    else:
                        st.error("❌ LLM not available. Make sure Ollama is running.")
                else:
                    st.error("❌ Failed to process documents")
        
        with col2:
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                # Clear input and parsed directories
                input_dir = Path("input")
                parsed_dir = Path("parsed_json")
                
                for pdf_file in input_dir.glob("*.pdf"):
                    pdf_file.unlink()
                
                for json_file in parsed_dir.glob("*.json"):
                    json_file.unlink()
                
                st.session_state.documents = []
                st.session_state.system_ready = False
                st.success("✅ All documents cleared")
                st.rerun()
    else:
        st.info("👆 Upload PDF files to get started")
        
        # Show existing documents if any
        existing_docs = load_documents()
        if existing_docs:
            st.divider()
            st.markdown("**📚 Existing Documents:**")
            for doc in existing_docs:
                st.write(f"• {doc.get('source_file', 'Unknown')} ({doc.get('document_type', 'unknown')})")

# Tab 2: Q&A Chat
with tab2:
    st.header("Ask Questions About Your Documents")
    
    # Initialize system if not ready
    if not st.session_state.system_ready:
        if st.button("🚀 Start System", type="primary"):
            if initialize_systems():
                st.success("✅ System ready! You can now ask questions.")
                st.rerun()
    else:
        # Display chat history
        for i, (q, a, sources) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You:** {q}")
                st.markdown(f'<div class="answer-box"><strong>🤖 Bot:</strong><br>{a}</div>', unsafe_allow_html=True)
                
                if sources:
                    with st.expander("📚 View Sources"):
                        for j, source in enumerate(sources, 1):
                            doc_id = source['metadata']['doc_id']
                            chunk_type = source['metadata']['chunk_type']
                            relevance = 1 - source.get('distance', 0)
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {j}:</strong> {doc_id} ({chunk_type}) - {relevance:.1%} relevant
                            </div>
                            """, unsafe_allow_html=True)
                st.divider()
        
        # Question input
        with st.form(key="question_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                question = st.text_input(
                    "Your question:",
                    placeholder="e.g., What is the total amount for order 10250?",
                    label_visibility="collapsed"
                )
            with col2:
                submit = st.form_submit_button("Ask 🚀", use_container_width=True)
        
        if submit and question:
            with st.spinner("🤖 Thinking..."):
                answer, sources = ask_question(question)
                st.session_state.chat_history.append((question, answer, sources))
                st.rerun()

# Tab 3: Compliance Check
with tab3:
    st.header("⚖️ Run Compliance Analysis")
    
    if st.session_state.system_ready and st.session_state.documents:
        st.info("Select a document to run compliance checks")
        
        # Document selector
        doc_options = [f"{doc.get('document_id', 'Unknown')} ({doc.get('document_type', 'unknown')})" 
                      for doc in st.session_state.documents]
        
        selected_doc_idx = st.selectbox("Select Document:", range(len(doc_options)), 
                                        format_func=lambda x: doc_options[x])
        
        if st.button("🔍 Run Compliance Check", type="primary"):
            selected_doc = st.session_state.documents[selected_doc_idx]
            
            with st.spinner("🧠 Analyzing compliance..."):
                # Load compliance rules
                rules_file = Path("config/compliance_rules.txt")
                if rules_file.exists():
                    with open(rules_file) as f:
                        rules = f.read()
                else:
                    rules = "Standard compliance rules"
                
                # Run compliance analysis
                analysis = st.session_state.llm.compliance_analysis(selected_doc, rules)
                
                st.markdown("### 📋 Compliance Report")
                st.markdown(f'<div class="answer-box">{analysis}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please initialize the system first")

# Tab 4: Document Viewer
with tab4:
    st.header("📊 Document Details")
    
    if st.session_state.documents:
        doc_options = [f"{doc.get('document_id', 'Unknown')} - {doc.get('source_file', 'Unknown')}" 
                      for doc in st.session_state.documents]
        
        selected_doc_idx = st.selectbox("Select Document to View:", range(len(doc_options)), 
                                        format_func=lambda x: doc_options[x], key="doc_viewer")
        
        selected_doc = st.session_state.documents[selected_doc_idx]
        
        # Display document details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Document ID", selected_doc.get('document_id', 'N/A'))
            st.metric("Type", selected_doc.get('document_type', 'N/A'))
        
        with col2:
            st.metric("Total Amount", f"${selected_doc.get('total_amount', 0):.2f}")
            st.metric("Products", selected_doc.get('product_count', 0))
        
        with col3:
            st.metric("Order Date", selected_doc.get('order_date', 'N/A'))
            st.metric("Customer", selected_doc.get('customer_name', 'N/A'))
        
        st.divider()
        
        # Products table
        if 'products' in selected_doc and selected_doc['products']:
            st.subheader("🛒 Products")
            
            products_data = []
            for prod in selected_doc['products']:
                products_data.append({
                    'Product Name': prod.get('product_name', 'Unknown'),
                    'Quantity': prod.get('quantity', 0),
                    'Unit Price': f"${prod.get('unit_price', 0):.2f}",
                    'Total': f"${prod.get('total', 0):.2f}"
                })
            
            st.dataframe(products_data, use_container_width=True)
        
        st.divider()
        
        # Raw JSON
        with st.expander("🔍 View Raw JSON"):
            st.json(selected_doc)
    else:
        st.info("No documents loaded")

# Footer
st.divider()
