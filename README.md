# 🤖 AI Compliance Bot

**An intelligent document processing and compliance checking system powered by LLMs, RAG, and vector databases.**

## 🎯 Overview

The AI Compliance Bot is a sophisticated document intelligence system that automates document processing, semantic search, and compliance validation for business documents like invoices, purchase orders, and receipts.

### What It Does

- **📄 Document Processing**: Automatically parses PDFs and extracts structured data (products, amounts, dates, customer info)
- **🔍 Semantic Search**: Uses vector embeddings to find relevant information across documents
- **💬 Natural Language Q&A**: Ask questions in plain English and get accurate, context-aware answers
- **⚖️ Compliance Checking**: Validates documents against predefined business rules
- **📊 Analytics Dashboard**: View document details, products, and financial summaries

---

## ✨ Key Features

### 🚀 Core Capabilities

- ✅ **PDF Upload & Processing** - Drag-and-drop PDF documents for instant analysis
- ✅ **Intelligent Data Extraction** - Auto-extract products, prices, quantities, dates, and metadata
- ✅ **Vector Database** - ChromaDB-powered semantic search with sentence transformers
- ✅ **RAG (Retrieval Augmented Generation)** - Context-aware responses from your documents
- ✅ **Dual LLM Support** - Works with both Ollama (local) and Groq Cloud API
- ✅ **Compliance Engine** - Automated validation against 15+ business rules
- ✅ **Real-time Chat Interface** - Interactive Q&A with source citations
- ✅ **Document Viewer** - Detailed view of parsed documents with product tables

### 🎨 Advanced Features

- **Enhanced Chunking Strategy**: Documents split into 4-25 semantic chunks (overview, products, financials, keywords)
- **Smart Reranking**: Query-aware relevance boosting for optimal results
- **Multi-document Search**: Query across all uploaded documents simultaneously
- **Source Attribution**: All answers cite their sources with relevance scores
- **Optimized Performance**: 40-50% faster responses with maintained accuracy

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                         │
│                  (app.py / groqapp.py)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        │             │             │             │
        ▼             ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Document     │ │ Vector       │ │ LLM          │ │ Compliance   │
│ Processor    │ │ Store        │ │ Orchestrator │ │ Engine       │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │                │
       ▼                ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ PDF Parser   │ │ ChromaDB     │ │ Ollama/Groq  │ │ Rules Engine │
│ (Docling)    │ │ + Embeddings │ │ LLM          │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### Data Flow

```
PDF Upload → Parse → Extract → Chunk → Embed → Store → Query → Retrieve → Generate → Display
```

---

## 🛠 Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **LLM (Local)** | Ollama (llama3.2) | Local language model |
| **LLM (Cloud)** | Groq API | Cloud-based inference |
| **Vector DB** | ChromaDB | Semantic search & storage |
| **Embeddings** | Sentence Transformers | Text-to-vector conversion |
| **PDF Parsing** | Docling, PDFPlumber | Document extraction |
| **Data Processing** | Pandas, NumPy | Data manipulation |

### Key Libraries

```
langchain==0.1.0              # LLM framework
chromadb==0.4.22              # Vector database
sentence-transformers==2.2.2   # Embeddings
ollama==0.1.6                 # Local LLM
streamlit==1.29.0             # Web UI
docling==1.0.0                # PDF parsing
pdfplumber==0.10.3            # Fallback parser
```

---

## 📦 Installation

### Prerequisites

- **Python 3.9+** (tested with 3.9-3.11)
- **Ollama** (for local LLM) OR **Groq API Key** (for cloud LLM)
- **4GB+ RAM** recommended
- **Windows/Linux/Mac** supported

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/Compliancebot.git
cd Compliancebot
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Setup LLM

**Option A: Local LLM (Ollama)**

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the model:
```bash
ollama pull llama3.2:latest
```
3. Start Ollama:
```bash
ollama serve
```

**Option B: Cloud LLM (Groq)**

1. Get API key from [console.groq.com](https://console.groq.com)
2. Update `config/config.py`:
```python
GROQ_API_KEY = "your-api-key-here"
```

### Step 5: Verify Installation

```bash
# Check Python packages
pip list | grep -E "chroma|sentence|streamlit"

# Check Ollama (if using local)
curl http://localhost:11434/api/version
```

---

## 🚀 Quick Start

### Start the Application

**Windows:**
```bash
scripts\run_app.bat
```

**Linux/Mac:**
```bash
streamlit run app.py
```

**For Groq Cloud Version:**
```bash
streamlit run groqapp.py
```

The app will open at `http://localhost:8501`

### First Time Setup

1. **Upload Documents** → Go to "📤 Upload Documents" tab
2. **Select PDFs** → Choose invoice/order PDFs from your computer
3. **Process** → Click "🚀 Process Documents" (wait ~10-30 seconds)
4. **Ask Questions** → Switch to "💬 Q&A Chat" tab and start asking!

### Example Questions

```
"What is the total amount for order 10250?"
"List all products in invoice 10251"
"Show me products with quantity greater than 5"
"What is the customer name for order 10250?"
"Calculate the average product price across all orders"
```

---

## 📖 Usage Guide

### 1. Upload & Process Documents

Navigate to **📤 Upload Documents** tab:

1. Click "Choose PDF files"
2. Select one or more PDFs (invoices, purchase orders, receipts)
3. Click **🚀 Process Documents**
4. Wait for parsing and vector embedding to complete
5. See success message with document count

**Supported Document Types:**
- Invoices
- Purchase Orders
- Goods Receipt Notes (GRN)
- Order Summaries

### 2. Ask Questions (Q&A Chat)

Navigate to **💬 Q&A Chat** tab:

1. Click **🚀 Start System** (if not initialized)
2. Type your question in the input box
3. Click **Ask 🚀** or press Enter
4. View the answer with source citations
5. Expand "📚 View Sources" to see where info came from

**Question Types:**
- Product queries: "What products are in order X?"
- Financial queries: "What's the total amount?"
- Customer queries: "Who is the customer for invoice X?"
- Aggregation: "List all products sorted by price"
- Calculations: "What's the average order value?"

### 3. Run Compliance Checks

Navigate to **⚖️ Compliance Check** tab:

1. Select a document from the dropdown
2. Click **🔍 Run Compliance Check**
3. Review the compliance report with PASS/FAIL status
4. Check which rules passed and which failed

**Compliance Rules Include:**
- Valid invoice/PO numbers
- Date consistency checks
- Financial calculation validation
- Required field verification
- Party information completeness

### 4. View Document Details

Navigate to **📊 Document Viewer** tab:

1. Select a document from the dropdown
2. View summary metrics (total, products, dates)
3. See complete product table with prices
4. Expand "🔍 View Raw JSON" for full data

---

## ⚙️ Configuration

### System Configuration

Edit `config/config.py` to customize behavior:

#### LLM Settings

```python
LLM_CONFIG = {
    "model": "llama3.2:latest",       # Model name
    "base_url": "http://localhost:11434",  # Ollama URL
    "temperature": 0.1,                # Creativity (0-1)
    "num_ctx": 4096,                   # Context window
    "num_predict": 800,                # Max response tokens
}
```

#### Embedding Settings

```python
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",                   # or "cuda" for GPU
    "batch_size": 32,
    "max_seq_length": 512,
}
```

#### Vector Store Settings

```python
VECTOR_STORE_CONFIG = {
    "collection_name": "compliance_documents",
    "distance_metric": "cosine",
    "persist_directory": "./vector_db/",
}
```

### Performance Tuning

**For Faster Responses:**
```python
LLM_CONFIG = {
    "num_ctx": 2048,      # Reduce context
    "num_predict": 500,   # Shorter responses
}
# In app.py, reduce search results:
context_results = vector_store.enhanced_search(query, n_results=4)
```

**For Better Quality:**
```python
LLM_CONFIG = {
    "num_ctx": 8192,      # More context
    "num_predict": 1500,  # Longer responses
}
# In app.py, increase search results:
context_results = vector_store.enhanced_search(query, n_results=10)
```

### Model Selection

Choose based on your needs:

| Model | Speed | Accuracy | RAM | Use Case |
|-------|-------|----------|-----|----------|
| llama3.2:1b | Very Fast | Good | 2GB | Quick queries |
| llama3.2:3b | Fast | Better | 4GB | Balanced |
| llama3.2:latest (8b) | Medium | Best | 8GB | Production |

```bash
# Pull different models
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull llama3.2:latest
```

### Compliance Rules

Edit `config/compliance_rules.txt` to customize validation rules:

```
Rule 1: Valid Invoice Number
- Every invoice must have a unique invoice number/ID
- Format should be alphanumeric (e.g., INV-001)

Rule 2: Date Requirements
- Invoice date must be present and valid
- Invoice date should not be in the future
...
```

---

## 📁 Project Structure

```
Compliancebot/
│
├── 📄 app.py                          # Main Streamlit app (Ollama)
├── 📄 groqapp.py                      # Streamlit app (Groq Cloud)
├── 📄 requirements.txt                # Python dependencies
├── 📄 .gitignore                      # Git ignore rules
├── 📄 README.md                       # This file
│
├── 📁 config/                         # Configuration files
│   ├── config.py                      # Main config (Ollama)
│   ├── config_cloud.py                # Cloud config (Groq)
│   └── compliance_rules.txt           # Compliance rules
│
├── 📁 core/                           # Core modules
│   ├── document_parser.py             # PDF parsing logic
│   ├── document_processor.py          # Processing pipeline
│   ├── vector_store.py                # ChromaDB & embeddings
│   ├── llm_orchestrator.py            # Ollama LLM manager
│   ├── LLMOrchestratorGroq.py         # Groq LLM manager
│   └── compliance_engine.py           # Compliance checking
│
├── 📁 docs/                           # Documentation
│   ├── README.md                      # Detailed docs
│   └── PERFORMANCE_OPTIMIZATIONS.md   # Performance guide
│
├── 📁 scripts/                        # Utility scripts
│   └── run_app.bat                    # Windows startup script
│
├── 📁 input/                          # PDF upload folder
├── 📁 parsed_json/                    # Parsed document storage
├── 📁 vector_db/                      # ChromaDB persistence
└── 📁 venv/                           # Virtual environment
```

---

## 📚 API Reference

### DocumentVectorStore

```python
from core.vector_store import DocumentVectorStore

# Initialize
store = DocumentVectorStore()

# Add documents
chunks_added = store.add_documents(documents)

# Enhanced search with reranking
results = store.enhanced_search("query", n_results=6)

# Basic search
results = store.search("query", n_results=10)

# Clear database
store.clear_store()

# Get all document IDs
doc_ids = store.get_all_document_ids()
```

### LLMOrchestrator

```python
from core.llm_orchestrator import LLMOrchestrator

# Initialize
llm = LLMOrchestrator()

# Check if LLM is available
if llm.available:
    # Answer question with context
    answer = llm.answer_question(question, retrieved_context)
    
    # Run compliance analysis
    report = llm.compliance_analysis(doc_data, rules)
    
    # Generate custom response
    response = llm.generate_response(prompt, system_prompt, context)
```

### DocumentParser

```python
from core.document_parser import DocumentParser

# Initialize
parser = DocumentParser()

# Parse single PDF
doc_data = parser.parse_document("path/to/file.pdf")

# Parse all PDFs in input folder
all_docs = parser.parse_all_documents(force_reparse=True)
```

### DocumentProcessor

```python
from core.document_processor import DocumentProcessor

# Initialize
processor = DocumentProcessor()

# Process all documents with auto-embedding
docs = processor.process_documents(force_reparse=True, auto_embed=True)

# Process single document
doc = processor.add_single_document("path/to/file.pdf", auto_embed=True)

# Refresh vector store
processor.refresh_vector_store()

# Get status
status = processor.get_processing_status()
```

---

## 📊 Performance

### Current Metrics

**Response Times:**
- Simple queries: 2-5 seconds
- Complex queries: 5-10 seconds
- Compliance checks: 8-15 seconds
- Document processing: 3-10 seconds per PDF

**Resource Usage:**
- RAM: 2-4 GB (depending on model)
- CPU: Moderate (4+ cores recommended)
- Storage: ~1MB per 100 documents
- Vector DB: ~500KB per document

### Optimizations Applied

✅ Reduced context window: 8192 → 4096 tokens  
✅ Limited response length: 1024 → 800 tokens  
✅ Optimized chunk retrieval: 10 → 6 chunks  
✅ Enhanced search with reranking  
✅ Compact context formatting  
✅ Streaming responses enabled  

**Result: 40-50% faster responses with maintained accuracy**

See `docs/PERFORMANCE_OPTIMIZATIONS.md` for details.

---

## 🐛 Troubleshooting

### Common Issues

#### 1. ChromaDB Not Available

```bash
# Solution
pip install chromadb sentence-transformers
```

#### 2. LLM Not Available (Ollama)

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama
ollama serve

# Check if model is downloaded
ollama list

# Pull model if needed
ollama pull llama3.2:latest
```

#### 3. Port 8501 Already in Use

```bash
# Use different port
streamlit run app.py --server.port 8502
```

#### 4. Slow Responses

**Quick Fixes:**
- Reduce `num_ctx` in `config.py` (try 2048)
- Reduce `n_results` in `app.py` (try 4)
- Use smaller model: `ollama pull llama3.2:1b`

#### 5. Out of Memory

**Solutions:**
- Use smaller model (llama3.2:1b)
- Reduce `batch_size` in `EMBEDDING_CONFIG`
- Process fewer documents at once
- Close other applications

#### 6. PDF Parsing Errors

**Troubleshooting:**
- Ensure PDF is not password-protected
- Check PDF is not corrupted
- Try re-uploading the file
- Check terminal for specific error messages

### Debug Mode

Enable detailed logging:

```python
# Add to config.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

1. Check this troubleshooting section
2. Review `docs/PERFORMANCE_OPTIMIZATIONS.md`
3. Check terminal output for errors
4. Verify Ollama is running: `ollama list`
5. Test with sample documents first

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Improvement

- [ ] Add support for more document types (receipts, contracts)
- [ ] Implement multi-language support
- [ ] Add Excel/CSV export functionality
- [ ] Create custom compliance rule editor UI
- [ ] Build advanced analytics dashboard
- [ ] Add document comparison features
- [ ] Implement email integration for reports
- [ ] Create REST API endpoints
- [ ] Add user authentication
- [ ] Improve error handling and validation

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

