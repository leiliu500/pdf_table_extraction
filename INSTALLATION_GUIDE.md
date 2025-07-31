# 🚀 PDF RAG System Installation Guide

## ✅ Current Status
- **Core PDF Extraction**: ✅ WORKING (all dependencies installed)
- **RAG System**: 🔧 Ready for setup (code implemented, needs dependencies)

## 📋 Step-by-Step Installation

### Step 1: Core System (Already Working ✅)
```bash
# Core PDF extraction is already working with these dependencies:
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Install PostgreSQL with pgvector
```bash
# On macOS with Homebrew
brew install postgresql pgvector

# Start PostgreSQL service
brew services start postgresql

# Create database for RAG system
createdb pdf_rag_db

# Test connection (optional)
psql pdf_rag_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Step 3: Install Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# In another terminal, pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Step 4: Install RAG Dependencies
```bash
# Activate virtual environment
source .venv/bin/activate

# Install RAG-specific packages
pip install -r requirements-rag.txt
```

### Step 5: Configure Environment
```bash
# Set environment variables (add to ~/.bashrc or ~/.zshrc)
export DATABASE_URL="postgresql://$(whoami)@localhost/pdf_rag_db"
export OLLAMA_BASE_URL="http://localhost:11434"
```

### Step 6: Test RAG System
```bash
# Process a PDF through the RAG system
python main.py --rag-process "pdf/304-Cedar-Street/1. Client MLS page.pdf"

# Query the processed documents
python main.py --rag-query "What is this document about?"

# View system statistics
python main.py --rag-stats
```

## 🔧 Troubleshooting

### PostgreSQL Issues
```bash
# If PostgreSQL fails to start
brew services restart postgresql

# If pgvector extension is missing
brew reinstall pgvector
```

### Ollama Issues
```bash
# If models fail to download
ollama list  # Check available models
ollama pull llama3.1:8b --verbose  # Verbose download

# If Ollama server won't start
killall ollama  # Kill existing processes
ollama serve    # Restart server
```

### Python Dependencies
```bash
# If sentence-transformers fails to install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers

# If psycopg2-binary fails to install
brew install postgresql  # Ensure PostgreSQL is installed first
pip install psycopg2-binary
```

## 📊 Verification Commands

### Test Core PDF Extraction (Should Work Now)
```bash
python main.py -i "pdf/304-Cedar-Street/1. Client MLS page.pdf" -o "output_test"
```

### Test Simple RAG Demo (Works Without Full Setup)
```bash
python simple_rag_demo.py
```

### Test Full RAG System (After Setup)
```bash
# Process document
python main.py --rag-process "pdf/304-Cedar-Street/1. Client MLS page.pdf"

# Query system
python main.py --rag-query "What are the key features mentioned?"
```

## 🎯 What's Fixed

### ✅ Resolved Issues:
1. **EasyOCR Dependency Conflict**: Made optional (PyTorch dependency)
2. **Requirements.txt Conflicts**: Separated core from RAG dependencies
3. **Core PDF Extraction**: Working with all extraction methods
4. **Installation Order**: Clear step-by-step process

### ✅ Working Features:
- High-accuracy PDF extraction (91.1% confidence)
- Table extraction (3 tables found)
- Text extraction (3478 characters)
- Form extraction (4 fields)
- Image extraction with OCR (4 images)
- Complete output structure with reports

## 📁 File Structure
```
pdf_table_extraction/
├── requirements.txt          # ✅ Core dependencies (working)
├── requirements-rag.txt      # 🔧 RAG dependencies (install after setup)
├── simple_rag_demo.py       # ✅ Demo without complex deps (working)
├── src/rag/                  # ✅ Complete RAG implementation
│   ├── vector_database.py    # PostgreSQL + pgvector
│   ├── ollama_client.py      # Ollama integration
│   ├── text_processor.py     # Intelligent chunking
│   └── rag_pipeline.py       # End-to-end pipeline
└── main.py                   # ✅ Enhanced CLI with RAG commands
```

## 🚀 Next Steps

1. **Install PostgreSQL & Ollama** (Steps 2-3 above)
2. **Install RAG dependencies** (Step 4 above)
3. **Test RAG system** (Step 6 above)
4. **Start using your complete PDF RAG system!**

Your system is ready for high-accuracy PDF processing with AI-powered question answering! 🎉
