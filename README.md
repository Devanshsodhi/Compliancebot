# AI Compliance Bot

An AI-powered document analysis and compliance checking system.

## 📁 Project Structure

```
Compliancebot/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
│
├── core/                       # Core processing modules
│   ├── compliance_engine.py    # Compliance checking logic
│   ├── document_parser.py      # PDF parsing
│   ├── document_processor.py   # Document processing pipeline
│   ├── llm_orchestrator.py     # LLM integration
│   └── vector_store.py         # Vector database management
│
├── config/                     # Configuration files
│   ├── config.py               # Application configuration
│   ├── config_cloud.py         # Cloud deployment config
│   └── compliance_rules.txt    # Compliance rules definition
│
├── docs/                       # Documentation
│   ├── README.md               # Detailed documentation
│   └── PERFORMANCE_OPTIMIZATIONS.md
│
├── scripts/                    # Utility scripts
│   └── run_app.bat             # Windows launcher script
│
├── input/                      # Input PDFs (auto-created)
├── parsed_json/                # Parsed documents (auto-created)
└── vector_db/                  # Vector database (auto-created)
```

## 🚀 Quick Start

### Option 1: Use the launcher script
```bash
cd scripts
run_app.bat
```

### Option 2: Run directly
```bash
streamlit run app.py
```

## 📋 Features

- **Document Upload & Processing**: Upload and parse PDF documents
- **Q&A Chat**: Ask questions about your documents
- **Compliance Checking**: Automated compliance analysis
- **Document Viewer**: Browse and analyze document details

## 🔧 Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Ollama (for LLM):
```bash
ollama serve
ollama pull llama3.2:latest
```

3. Run the application:
```bash
streamlit run app.py
```

## 📚 Documentation

See `docs/README.md` for detailed documentation.
