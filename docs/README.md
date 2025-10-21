# 🤖 AI Compliance Bot

**An intelligent document processing and compliance checking system powered by LLMs and vector databases.**

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Features](#features)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The AI Compliance Bot is a sophisticated document intelligence system that combines:
- **PDF Parsing** - Extracts structured data from business documents
- **Vector Database** - Semantic search using embeddings
- **LLM Integration** - Natural language Q&A with Ollama
- **RAG (Retrieval Augmented Generation)** - Context-aware responses
- **Compliance Checking** - Automated rule validation

### Key Capabilities

- ✅ Upload and parse PDF documents (invoices, purchase orders, receipts)
- ✅ Extract structured information (products, amounts, dates, customers)
- ✅ Semantic search across documents using vector embeddings
- ✅ Ask natural language questions and get detailed answers
- ✅ Automated compliance checking against predefined rules
- ✅ Real-time Q&A with context-aware responses

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI                       │
│                        (app.py)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Document     │ │ LLM          │ │ Compliance   │
│ Processor    │ │ Orchestrator │ │ Engine       │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Document     │ │ Vector       │ │ Config       │
│ Parser       │ │ Store        │ │ & Rules      │
└──────────────┘ └──────────────┘ └──────────────┘
       │                │
       ▼                ▼
┌──────────────┐ ┌──────────────┐
│ PDF Files    │ │ ChromaDB     │
│ (input/)     │ │ (vector_db/) │
└──────────────┘ └──────────────┘
```

---

## 🔧 Core Components

### 1. **Streamlit Web Application** (`app.py`)
- **Purpose**: Main user interface for the compliance bot
- **Features**:
  - Document upload and processing
  - Interactive Q&A chat interface
  - Compliance analysis dashboard
  - Document viewer with detailed information
  - Real-time status updates

**Key Functions**:
- `load_documents()` - Loads parsed JSON documents from storage
- `initialize_systems()` - Initializes vector store and LLM components
- `ask_question(question)` - Processes user questions with enhanced vector search (6 chunks, optimized for speed)

**Workflow**:
1. User uploads PDF files
2. Files are saved to `input/` directory
3. DocumentProcessor parses and embeds documents
4. User asks questions via chat interface
5. System retrieves relevant context and generates answers

---

### 2. **Document Parser** (`document_parser.py`)
- **Purpose**: Extracts structured data from PDF documents
- **Technology**: Uses `pdfplumber` for text extraction

**Key Features**:
- Multi-pattern product extraction
- Automatic field detection (dates, amounts, customers)
- Document type classification
- Robust error handling

**Data Extraction**:
```python
{
  "document_id": "10250",
  "document_type": "invoice",
  "source_file": "invoice_10250.pdf",
  "order_date": "2024-01-15",
  "customer_name": "John Doe",
  "total_amount": 1234.56,
  "product_count": 3,
  "products": [
    {
      "product_name": "Laptop Computer",
      "quantity": 2,
      "unit_price": 500.00,
      "total": 1000.00
    }
  ]
}
```

**Methods**:
- `parse_document(pdf_path)` - Parses a single PDF file
- `parse_all_documents(force_reparse)` - Batch processes all PDFs in input folder
- `_extract_products(text)` - Uses regex patterns to extract product information
- `_extract_metadata(text)` - Extracts document metadata (dates, IDs, amounts)

---

### 3. **Document Processor** (`document_processor.py`)
- **Purpose**: Unified pipeline for parsing and embedding documents
- **Integration**: Connects DocumentParser with VectorStore

**Responsibilities**:
1. Parse PDFs into structured JSON
2. Automatically create vector embeddings
3. Store embeddings in ChromaDB
4. Provide processing status and health checks

**Key Methods**:
- `process_documents(force_reparse, auto_embed)` - Main processing pipeline
- `add_single_document(pdf_path, auto_embed)` - Process one document
- `refresh_vector_store()` - Rebuild vector database from parsed documents
- `get_processing_status()` - Returns system status and document counts

**Workflow**:
```
PDF Upload → Parse → Create Chunks → Generate Embeddings → Store in VectorDB
```

---

### 4. **Vector Store** (`vector_store.py`)
- **Purpose**: Semantic search and document retrieval using embeddings
- **Technologies**:
  - **ChromaDB** - Vector database for storage
  - **Sentence Transformers** - `all-MiniLM-L6-v2` model for embeddings

**Enhanced Chunking Strategy**:

Documents are split into multiple semantic chunks for better retrieval:

1. **Document Overview** - Metadata and summary (all header fields)
2. **Products Complete List** - All products with prices and quantities
3. **Individual Product Details** - Each product with context (up to 20 products)
4. **Financial Summary** - Totals, averages, and financial metrics
5. **Searchable Keywords** - All customer info, dates, and product names

**Key Features**:
- **Smart Chunking**: Creates 4-25 chunks per document depending on content
- **Enhanced Search**: Query-aware reranking with relevance boosting
- **Semantic Matching**: Finds relevant information even with different wording

**Methods**:
- `document_to_text_chunks(doc)` - Converts JSON to searchable text chunks
- `add_documents(documents)` - Embeds and stores document chunks
- `enhanced_search(query, n_results=6)` - Retrieves most relevant chunks with smart reranking
- `search(query, n_results)` - Basic semantic search
- `clear_store()` - Resets vector database
- `get_all_document_ids()` - Lists all stored documents

**Search Optimization**:
- Retrieves 12-25 initial results for reranking
- Returns top 6 most relevant chunks (optimized for speed)
- Applies query-specific boosting for product, financial, customer, and date queries
- Ensures diversity across documents

---

### 5. **LLM Orchestrator** (`llm_orchestrator.py`)
- **Purpose**: Manages interactions with Ollama LLM for Q&A and compliance
- **Model**: llama3.2:latest (configurable)

**Core Responsibilities**:
- Generate natural language responses
- Format context for optimal LLM processing
- Handle compliance analysis
- Provide fallback responses when needed

**Key Methods**:

**`generate_response(prompt, system_prompt, context)`**
- Core LLM interaction method
- Sends formatted prompts to Ollama
- Optimized parameters for speed (num_ctx: 4096, num_predict: 800)

**`answer_question(question, retrieved_context)`**
- Enhanced Q&A with quality validation
- Formats context using `_format_enhanced_rag_context()`
- Validates response length and provides fallbacks

**`_format_enhanced_rag_context(retrieved_chunks, question)`**
- Optimized context formatting (reduced from verbose to compact)
- Groups chunks by document
- Sorts by relevance score
- Limits content based on chunk type:
  - Product chunks: 1000 chars
  - Financial: 600 chars
  - Other: 400 chars

**`compliance_analysis(document_data, compliance_rules)`**
- Analyzes documents against compliance rules
- Returns PASS/FAIL with reasoning and evidence

**`_generate_fallback_response(question, retrieved_context)`**
- Backup response when main generation fails
- Extracts key information from context

---

### 6. **Compliance Engine** (`compliance_engine.py`)
- **Purpose**: Automated compliance checking and reporting
- **Rules Source**: `compliance_rules.txt`

**Features**:
- Document-level compliance validation
- Batch compliance analysis
- Detailed compliance reports
- Rule-based validation

**Key Methods**:
- `check_document_compliance(doc)` - Validates single document
- `batch_compliance_check(docs)` - Processes multiple documents
- `generate_compliance_report(results)` - Creates formatted reports

**Compliance Rules**:
- Invoice requirements (number, dates, amounts)
- Purchase order validation
- Customer information verification
- Calculation accuracy checks

---

### 7. **Configuration** (`config.py`)
- **Purpose**: Centralized configuration for all system components

**LLM Configuration**:
```python
LLM_CONFIG = {
    "model": "llama3.2:latest",
    "base_url": "http://localhost:11434",
    "temperature": 0.1,              # Low for consistency
    "num_ctx": 4096,                 # Context window (optimized)
    "num_predict": 800,              # Max response tokens (optimized)
}
```

**Embedding Configuration**:
```python
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
    "normalize_embeddings": True,
    "batch_size": 32,
    "max_seq_length": 512,
}
```

**Vector Store Configuration**:
```python
VECTOR_STORE_CONFIG = {
    "collection_name": "compliance_documents",
    "distance_metric": "cosine",
    "persist_directory": "./vector_db/",
}
```

**System Prompts**:
- **QA Agent**: Optimized for comprehensive, detailed responses
- **Compliance Agent**: Focused on rule validation
- **Document Parser**: Structured data extraction

---

## 📁 File Structure

```
Compliancebot/
├── 📄 app.py                          # Main Streamlit web application
├── 📄 vector_store.py                 # Vector database & semantic search
├── 📄 llm_orchestrator.py             # LLM integration & response generation
├── 📄 document_parser.py              # PDF parsing & data extraction
├── 📄 document_processor.py           # Document processing pipeline
├── 📄 compliance_engine.py            # Compliance checking system
├── 📄 config.py                       # System configuration
│
├── 📄 requirements.txt                # Python dependencies
├── 📄 compliance_rules.txt            # Compliance validation rules
├── 📄 run_app.bat                     # Windows startup script
├── 📄 .gitignore                      # Git ignore rules
├── 📄 README.md                       # This file
├── 📄 PERFORMANCE_OPTIMIZATIONS.md    # Performance tuning guide
│
├── 📁 input/                          # Upload folder for PDF files
├── 📁 parsed_json/                    # Parsed document storage
├── 📁 vector_db/                      # ChromaDB storage
├── 📁 venv/                           # Python virtual environment
├── 📁 __pycache__/                    # Python cache
└── 📁 .streamlit/                     # Streamlit configuration
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.9 or higher
- **Ollama**: For LLM integration
- **Git**: For version control

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Compliancebot
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

**Activate Virtual Environment:**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `streamlit` - Web interface
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `ollama` - LLM integration
- `pdfplumber` - PDF parsing
- `pandas` - Data processing

### Step 4: Install Ollama

Download and install from [ollama.ai](https://ollama.ai)

Pull the LLM model:
```bash
ollama pull llama3.2:latest
```

Start Ollama:
```bash
ollama serve
```

### Step 5: Verify Installation

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Check Python packages
pip list | findstr "chroma sentence"
```

---

## 💻 Usage

### Quick Start

**Windows:**
```bash
run_app.bat
```

**Linux/Mac:**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application

#### 1. **Upload Documents**

- Navigate to "📤 Upload Documents" tab
- Click "Choose PDF files"
- Select one or more PDF files (invoices, orders, receipts)
- Click "🚀 Process Documents"
- Wait for parsing and embedding to complete

#### 2. **Ask Questions**

- Go to "💬 Q&A Chat" tab
- Click "🚀 Start System" if not initialized
- Type your question in the input box
- Example questions:
  ```
  - "What products are in order 10250?"
  - "What's the total amount for all orders?"
  - "Show me products with quantities greater than 5"
  - "What is the customer name for order 10250?"
  - "List all products sorted by price"
  ```

#### 3. **Run Compliance Checks**

- Navigate to "⚖️ Compliance Check" tab
- Select a document from the dropdown
- Click "🔍 Run Compliance Check"
- Review the compliance report with PASS/FAIL status

#### 4. **View Documents**

- Go to "📊 Document Viewer" tab
- Select a document to view details
- See full product list with prices
- View raw JSON data

---

## ⚙️ Configuration

### Adjusting Performance

Edit `config.py` to tune performance:

**For Faster Responses:**
```python
LLM_CONFIG = {
    "num_ctx": 2048,      # Reduce context window
    "num_predict": 500,   # Limit response length
}
```

**For Better Quality:**
```python
LLM_CONFIG = {
    "num_ctx": 8192,      # Increase context window
    "num_predict": 1500,  # Allow longer responses
}
```

### Changing the Model

```python
LLM_CONFIG = {
    "model": "llama3.2:1b",  # Faster, less accurate
    # or
    "model": "llama3.2:3b",  # Balanced
    # or
    "model": "llama3.2:7b",  # Slower, more accurate
}
```

### Adjusting Search Results

In `app.py`, change the number of retrieved chunks:

```python
# For faster responses (less context)
context_results = st.session_state.vector_store.enhanced_search(question, n_results=4)

# For more comprehensive responses (more context)
context_results = st.session_state.vector_store.enhanced_search(question, n_results=10)
```

---

## 🎯 Features

### 1. **Semantic Search**
- Vector-based similarity search
- Understands intent, not just keywords
- Works with different phrasings of the same question

### 2. **Context-Aware Responses**
- RAG (Retrieval Augmented Generation)
- Provides specific information from your documents
- Cites source documents

### 3. **Enhanced Chunking**
- Multiple chunk types per document
- Optimized for different query types
- Preserves context and relationships

### 4. **Smart Reranking**
- Query-aware relevance boosting
- Prioritizes appropriate chunk types
- Ensures document diversity

### 5. **Optimized Performance**
- 40-50% faster than initial version
- Reduced token usage
- Maintained accuracy and detail

---

## 📊 Performance

### Current Optimizations

**Speed Improvements:**
- Context window: 4096 tokens (balanced)
- Response limit: 800 tokens (efficient)
- Chunks retrieved: 6 (optimal)
- Initial search pool: 12-25 (fast reranking)

**Accuracy Maintained:**
- Enhanced search algorithm
- Smart chunk selection
- Query-aware boosting
- Complete detail preservation

### Performance Metrics

**Typical Response Times:**
- Simple queries: 2-5 seconds
- Complex queries: 5-10 seconds
- Compliance checks: 8-15 seconds

**Resource Usage:**
- RAM: 2-4 GB (depending on model)
- CPU: Moderate (4+ cores recommended)
- Storage: Minimal (~1MB per 100 documents)

---

## 🐛 Troubleshooting

### Common Issues

**1. "ChromaDB not available"**
```bash
pip install chromadb sentence-transformers
```

**2. "LLM not available"**
- Ensure Ollama is running: `ollama serve`
- Check if model is pulled: `ollama list`
- Pull model if needed: `ollama pull llama3.2:latest`

**3. "Port 8501 already in use"**
```bash
streamlit run app.py --server.port 8502
```

**4. Slow responses**
- Reduce `num_ctx` and `num_predict` in `config.py`
- Reduce `n_results` in `app.py`
- Use smaller model (llama3.2:1b)

**5. Out of memory**
- Use smaller model
- Reduce batch_size in `EMBEDDING_CONFIG`
- Process fewer documents at once

### Debug Mode

Enable detailed logging:
```python
# In config.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📚 API Reference

### VectorStore Methods

```python
# Initialize
store = DocumentVectorStore()

# Add documents
chunks_added = store.add_documents(documents)

# Search
results = store.enhanced_search("query", n_results=6)

# Get document
doc_ids = store.get_all_document_ids()
```

### LLM Orchestrator Methods

```python
# Initialize
llm = LLMOrchestrator()

# Answer questions
answer = llm.answer_question(question, context)

# Check compliance
report = llm.compliance_analysis(doc_data, rules)
```

### Document Parser Methods

```python
# Initialize
parser = DocumentParser()

# Parse single document
data = parser.parse_document(pdf_path)

# Parse all documents
all_docs = parser.parse_all_documents(force_reparse=True)
```

---

## 🔒 Security Considerations

- Documents are stored locally (not sent to external services)
- Ollama runs locally (no data leaves your machine)
- No API keys or external dependencies required
- Vector database is local (ChromaDB)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional document types support
- More compliance rules
- Performance optimizations
- UI/UX enhancements
- Better error handling

---

## 📝 License

[Add your license here]

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review `PERFORMANCE_OPTIMIZATIONS.md`
3. Examine terminal output for errors
4. Check Ollama is running properly

---

## 🎓 Technical Details

### Document Processing Pipeline

```
1. PDF Upload → input/
2. Parse PDF → Extract text and structure
3. Extract Fields → Products, amounts, dates, customers
4. Save JSON → parsed_json/
5. Create Chunks → 4-25 semantic chunks per document
6. Generate Embeddings → all-MiniLM-L6-v2 model
7. Store Vectors → ChromaDB (vector_db/)
8. Ready for Search → Semantic queries enabled
```

### Q&A Processing Pipeline

```
1. User Question → Natural language query
2. Generate Embedding → Convert to vector
3. Search VectorDB → Find similar chunks (retrieve 12-25)
4. Rerank Results → Query-aware boosting
5. Select Top 6 → Most relevant chunks
6. Format Context → Optimized for LLM (1000-400 chars per chunk)
7. Send to LLM → With system prompt and context
8. Generate Response → Detailed, accurate answer
9. Display to User → With source citations
```

### Chunk Types and Usage

| Chunk Type | Purpose | Size | When Used |
|------------|---------|------|-----------|
| document_overview | Metadata & summary | 400 chars | All queries |
| products_complete_list | All products with prices | 1000 chars | Product queries |
| product_detail_N | Individual product info | 400 chars | Specific product queries |
| financial_summary | Totals & calculations | 600 chars | Financial queries |
| searchable_keywords | Names, dates, IDs | 400 chars | Customer/date queries |

---

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Excel/CSV export
- [ ] Batch document upload via drag-and-drop
- [ ] Custom compliance rules editor
- [ ] Advanced analytics dashboard
- [ ] Document comparison features
- [ ] Email integration for reports
- [ ] API endpoints for programmatic access

---

**Built with ❤️ using Streamlit, Ollama, ChromaDB, and Sentence Transformers**
