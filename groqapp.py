import streamlit as st
import requests
import json
from pathlib import Path
from core.vector_store import DocumentVectorStore
from core.document_processor import DocumentProcessor
from config.config import GROQ_API_URL, MODEL_NAME
import time

st.set_page_config(page_title="AI Compliance Bot (Groq)", layout="wide", initial_sidebar_state="expanded")

# -------------------------------
# GROQ API Key from Streamlit Secrets
# -------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    st.write(st.secrets)
except KeyError:
    st.error("❌ GROQ_API_KEY not found in Streamlit secrets. Add it and redeploy.")
    st.stop()

# -------------------------------
# Streamlit Page Setup
# -------------------------------

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

def initialize_vector_store(documents):
    """
    Initialize vector store and add documents.
    Returns number of chunks added.
    """
    store = DocumentVectorStore()
    store.clear_store()
    chunks_added = store.add_documents(documents)
    return store, chunks_added

def initialize_systems():
    if not st.session_state.system_ready:
        with st.spinner("🔄 Initializing AI systems..."):
            # Load pre-parsed documents
            docs = load_documents()
            if not docs:
                st.warning("No parsed documents found. Upload and process PDFs first.")
                return False
            st.session_state.documents = docs
            st.session_state.vector_store, chunks_added = initialize_vector_store(docs)
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
        if initialize_systems():
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
# Tab 1: Upload Documents
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Documents", "💬 Q&A Chat", "⚖️ Compliance Check", "📊 Document Viewer"])
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
                st.session_state.documents = parsed_docs
                st.session_state.vector_store, chunks_added = initialize_vector_store(parsed_docs)
                st.success(f"✅ Processed {len(parsed_docs)} documents and added {chunks_added} chunks to vector store")
                st.session_state.system_ready = True
            else:
                st.error("❌ Failed to process documents")
    else:
        st.info("Upload PDFs to process.")

# -------------------------------
# The rest of Tabs 2, 3, 4 remain the same
# -------------------------------

