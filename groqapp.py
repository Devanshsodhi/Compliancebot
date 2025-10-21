import streamlit as st
import requests
import json
from pathlib import Path
from core.vector_store import DocumentVectorStore
from core.document_processor import DocumentProcessor
from config.config import GROQ_API_URL, MODEL_NAME
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  
import time

# -------------------------------
# Streamlit Page Setup
# -------------------------------
print("Using GROQ API Key:", bool(os.environ.get("GROQ_API_KEY")))
st.set_page_config(page_title="AI Compliance Bot (Groq)", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
.main-header {font-size: 3rem; font-weight: bold; text-align: center; color: #1f77b4; margin-bottom: 1rem;}
.sub-header {text-align: center; color: #666; margin-bottom: 2rem;}
.doc-card {padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6; margin: 0.5rem 0;}
.answer-box {padding: 1.5rem; border-radius: 0.5rem; background-color: #e8f4f8; border-left: 4px solid #1f77b4;}
.source-box {padding: 0.5rem; border-radius: 0.3rem; background-color: #f8f9fa; margin: 0.3rem 0; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 AI Compliance Bot (Groq-Powered)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your documents • Run compliance checks • Get instant answers</div>', unsafe_allow_html=True)

# -------------------------------
# Session State Initialization
# -------------------------------
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

# -------------------------------
# Helper Functions
# -------------------------------
def load_documents():
    parsed_dir = Path("parsed_json")
    docs = []
    if parsed_dir.exists():
        for json_file in parsed_dir.glob("*.json"):
            with open(json_file) as f:
                docs.append(json.load(f))
    return docs

def initialize_systems():
    if not st.session_state.system_ready:
        with st.spinner("🔄 Initializing AI systems..."):
            st.session_state.documents = load_documents()
            if not st.session_state.documents:
                st.warning("No parsed documents found. Upload and process PDFs first.")
                return False
            st.session_state.vector_store = DocumentVectorStore()
            st.session_state.vector_store.clear_store()
            chunks_added = st.session_state.vector_store.add_documents(st.session_state.documents)
            st.success(f"✅ Added {chunks_added} chunks to vector store")
            st.session_state.system_ready = True
    return True

def ask_question(question):
    if not st.session_state.vector_store or not st.session_state.system_ready:
        return "⚠️ System not ready.", []

    context_results = st.session_state.vector_store.enhanced_search(question, n_results=6)
    if not context_results:
        return "⚠️ No relevant information found in documents.", []

    context_text = "\n".join([f"{c['metadata']['doc_id']}: {c['text']}" for c in context_results])

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an AI compliance assistant analyzing business documents."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ],
        "temperature": 0.0
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"]
            return answer, context_results
        else:
            return f"⚠️ Groq API did not return choices. Response: {result}", context_results
    except Exception as e:
        return f"❌ Error communicating with Groq API: {e}", context_results

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("📚 System Status")
    if st.button("🔄 Initialize/Reload System"):
        st.session_state.system_ready = False
        initialize_systems()
        if st.session_state.system_ready:
            st.success("✅ System initialized!")

    st.divider()
    if st.session_state.documents:
        st.subheader("📄 Loaded Documents")
        for doc in st.session_state.documents:
            doc_id = doc.get('document_id','Unknown')
            st.markdown(f"<div class='doc-card'><strong>{doc_id}</strong></div>", unsafe_allow_html=True)
    else:
        st.info("No documents loaded yet")
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# -------------------------------
# Main Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Documents", "💬 Q&A Chat", "⚖️ Compliance Check", "📊 Document Viewer"])

# -------------------------------
# Tab 1: Upload Documents
# -------------------------------
with tab1:
    st.header("📤 Upload and Process Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        if st.button("🚀 Process Documents"):
            input_dir = Path("input")
            input_dir.mkdir(exist_ok=True)
            for file in uploaded_files:
                with open(input_dir / file.name, "wb") as f:
                    f.write(file.getbuffer())
            processor = DocumentProcessor()
            parsed_docs = processor.process_documents(force_reparse=True, auto_embed=True)
            if parsed_docs:
                st.success(f"✅ Processed {len(parsed_docs)} documents")
                st.session_state.documents = parsed_docs
                st.session_state.vector_store = processor.vector_store
                st.session_state.system_ready = True
            else:
                st.error("❌ Failed to process documents")
    else:
        st.info("Upload PDFs to process.")

# -------------------------------
# Tab 2: Q&A Chat
# -------------------------------
with tab2:
    st.header("💬 Ask Questions About Your Documents")
    if not st.session_state.system_ready:
        if st.button("🚀 Start System"):
            if initialize_systems():
                st.success("✅ System ready!")
                st.rerun()
    else:
        for q, a, sources in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"<div class='answer-box'><strong>🤖 Bot:</strong><br>{a}</div>", unsafe_allow_html=True)
            if sources:
                with st.expander("📚 View Sources"):
                    for j, source in enumerate(sources, 1):
                        doc_id = source['metadata']['doc_id']
                        st.markdown(f"<div class='source-box'>Source {j}: {doc_id}</div>", unsafe_allow_html=True)
        with st.form("question_form", clear_on_submit=True):
            col1, col2 = st.columns([5,1])
            with col1:
                question = st.text_input("Your question:", placeholder="Ask something about the documents")
            with col2:
                submit = st.form_submit_button("Ask 🚀")
            if submit and question:
                answer, sources = ask_question(question)
                st.session_state.chat_history.append((question, answer, sources))
                st.rerun()

# -------------------------------
# Tab 3: Compliance Check
# -------------------------------
with tab3:
    st.header("⚖️ Run Compliance Analysis")
    if st.session_state.system_ready and st.session_state.documents:
        st.info("Select a document to run compliance checks")
        doc_options = [f"{doc.get('document_id','Unknown')} ({doc.get('document_type','unknown')})" for doc in st.session_state.documents]
        selected_doc_idx = st.selectbox("Select Document:", range(len(doc_options)), format_func=lambda x: doc_options[x])
        selected_doc = st.session_state.documents[selected_doc_idx]

        if st.button("🔍 Run Compliance Check"):
            with st.spinner("🧠 Analyzing compliance..."):
                rules = "Check standard compliance rules for invoices and purchase orders."
                question = f"Analyze this document for compliance:\n{json.dumps(selected_doc, indent=2)}\n\nRules: {rules}"
                answer, _ = ask_question(question)
                st.markdown(f"<div class='answer-box'><strong>Compliance Report:</strong><br>{answer}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please initialize the system first")

# -------------------------------
# Tab 4: Document Viewer
# -------------------------------
with tab4:
    st.header("📊 Document Details")
    if st.session_state.documents:
        doc_options = [f"{doc.get('document_id','Unknown')} - {doc.get('source_file','Unknown')}" for doc in st.session_state.documents]
        selected_doc_idx = st.selectbox("Select Document to View:", range(len(doc_options)), format_func=lambda x: doc_options[x], key="doc_viewer")
        selected_doc = st.session_state.documents[selected_doc_idx]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Document ID", selected_doc.get('document_id','N/A'))
            st.metric("Type", selected_doc.get('document_type','N/A'))
        with col2:
            st.metric("Total Amount", f"${selected_doc.get('total_amount', 0):.2f}")
            st.metric("Products", selected_doc.get('product_count',0))
        with col3:
            st.metric("Order Date", selected_doc.get('order_date','N/A'))
            st.metric("Customer", selected_doc.get('customer_name','N/A'))

        st.divider()
        if 'products' in selected_doc and selected_doc['products']:
            st.subheader("🛒 Products")
            products_data = []
            for prod in selected_doc['products']:
                products_data.append({
                    'Product Name': prod.get('product_name','Unknown'),
                    'Quantity': prod.get('quantity',0),
                    'Unit Price': f"${prod.get('unit_price',0):.2f}",
                    'Total': f"${prod.get('total',0):.2f}"
                })
            st.dataframe(products_data, use_container_width=True)

        st.divider()
        with st.expander("🔍 View Raw JSON"):
            st.json(selected_doc)
    else:
        st.info("No documents loaded")
