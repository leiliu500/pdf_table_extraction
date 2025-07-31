#!/usr/bin/env python3
"""
RAG System Demo
Simplified demo of the PDF RAG system functionality
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def demo_rag_system():
    """
    Demo the RAG system components
    """
    print("🚀 PDF RAG System Demo")
    print("=" * 50)
    
    try:
        # Test imports
        print("\n1. Testing core imports...")
        
        try:
            from src.pdf_extractor import PDFExtractor
            print("✓ PDF Extractor imported")
        except ImportError as e:
            print(f"✗ PDF Extractor import failed: {e}")
            return
        
        try:
            # Import directly to avoid init file dependencies
            import importlib.util
            spec = importlib.util.spec_from_file_location("text_processor_demo", "src/rag/text_processor_demo.py")
            text_processor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(text_processor_module)
            ContentChunker = text_processor_module.ContentChunker
            print("✓ Content Chunker (demo) imported")
        except Exception as e:
            print(f"✗ Content Chunker import failed: {e}")
            return
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("vector_database_demo", "src/rag/vector_database_demo.py")
            vector_db_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vector_db_module)
            VectorDatabase = vector_db_module.VectorDatabase
            print("✓ Vector Database (demo) imported")
        except Exception as e:
            print(f"✗ Vector Database import failed: {e}")
            return
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("ollama_client_demo", "src/rag/ollama_client_demo.py")
            ollama_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ollama_module)
            OllamaClient = ollama_module.OllamaClient
            print("✓ Ollama Client (demo) imported")
        except Exception as e:
            print(f"✗ Ollama Client import failed: {e}")
            return
        
        # Test PDF extraction
        print("\n2. Testing PDF extraction...")
        pdf_extractor = PDFExtractor()
        
        # Find a test PDF
        test_pdf = None
        pdf_dirs = [
            Path("pdf"),
            Path("output"),
            Path(".")
        ]
        
        for pdf_dir in pdf_dirs:
            if pdf_dir.exists():
                pdf_files = list(pdf_dir.glob("**/*.pdf"))
                if pdf_files:
                    test_pdf = pdf_files[0]
                    break
        
        if test_pdf and test_pdf.exists():
            print(f"Found test PDF: {test_pdf}")
            
            # Extract content
            print("Extracting content...")
            results = pdf_extractor.extract_all_content(str(test_pdf))
            
            if results:
                print("✓ PDF extraction successful")
                
                # Show extraction summary
                summary = {}
                for content_type in ['texts', 'tables', 'forms', 'images']:
                    if content_type in results and results[content_type]:
                        summary[content_type] = len(results[content_type])
                
                print(f"Extraction summary: {summary}")
                
                # Test content chunking
                print("\n3. Testing content chunking...")
                chunker = ContentChunker()
                chunks = chunker.chunk_extracted_content(results)
                
                if chunks:
                    print(f"✓ Created {len(chunks)} chunks")
                    
                    # Show chunk summary
                    chunk_types = {}
                    for chunk in chunks:
                        content_type = chunk.get('content_type', 'unknown')
                        chunk_types[content_type] = chunk_types.get(content_type, 0) + 1
                    
                    print(f"Chunk types: {chunk_types}")
                    
                    # Show first chunk sample
                    if chunks:
                        first_chunk = chunks[0]
                        print(f"\nSample chunk:")
                        print(f"  Type: {first_chunk.get('content_type')}")
                        print(f"  Length: {first_chunk.get('content_length')} chars")
                        print(f"  Content preview: {first_chunk.get('content', '')[:100]}...")
                        print(f"  Confidence: {first_chunk.get('confidence_score', 0):.2f}")
                
                else:
                    print("✗ No chunks created")
            
            else:
                print("✗ PDF extraction failed")
        
        else:
            print("No test PDF found. Place a PDF in the pdf/ directory to test extraction.")
        
        # Test database connection (without actual PostgreSQL)
        print("\n4. Testing vector database setup...")
        try:
            vector_db = VectorDatabase()
            print("✓ Vector database object created")
            
            # Check configuration
            config = vector_db.postgres_config
            print(f"Database config: {config['host']}:{config['port']}/{config['database']}")
            
        except Exception as e:
            print(f"✗ Vector database setup failed: {e}")
        
        # Test Ollama client setup (without actual Ollama server)
        print("\n5. Testing Ollama client setup...")
        try:
            ollama_client = OllamaClient()
            print("✓ Ollama client object created")
            
            # Check configuration
            print(f"Ollama URL: {ollama_client.base_url}")
            print(f"Embedding model: {ollama_client.embedding_model}")
            print(f"LLM model: {ollama_client.llm_model}")
            
        except Exception as e:
            print(f"✗ Ollama client setup failed: {e}")
        
        print("\n6. RAG System Architecture Overview")
        print("-" * 40)
        print("📄 PDF Processing Pipeline:")
        print("  1. PDF Extraction (texts, tables, forms, images)")
        print("  2. Content Chunking (intelligent splitting)")
        print("  3. Embedding Generation (Ollama + fallback)")
        print("  4. Vector Storage (PostgreSQL + pgvector)")
        print("  5. Similarity Search (cosine similarity)")
        print("  6. Answer Generation (Ollama LLM)")
        
        print("\n🎯 Key Features:")
        print("  • Multi-format extraction (high accuracy OCR)")
        print("  • Intelligent chunking (preserves context)")
        print("  • Quality validation (confidence scoring)")
        print("  • Local inference (Ollama integration)")
        print("  • Scalable storage (PostgreSQL vector DB)")
        print("  • Citation support (source attribution)")
        
        print("\n📊 Accuracy Focus:")
        print("  • Extraction confidence tracking")
        print("  • Embedding quality validation")
        print("  • Query relevance scoring")
        print("  • Answer confidence estimation")
        
        print("\n🚀 Next Steps:")
        print("  1. Install PostgreSQL with pgvector extension")
        print("  2. Install and start Ollama server")
        print("  3. Run: python main.py --rag-process file.pdf")
        print("  4. Query: python main.py --rag-query 'your question'")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

def show_installation_guide():
    """
    Show installation guide for RAG system dependencies
    """
    print("\n📋 RAG System Installation Guide")
    print("=" * 50)
    
    print("\n1. PostgreSQL with pgvector:")
    print("   # macOS (using Homebrew)")
    print("   brew install postgresql")
    print("   brew install pgvector")
    print("   brew services start postgresql")
    print("   createdb pdf_rag_db")
    
    print("\n2. Ollama (Local LLM server):")
    print("   # Download from https://ollama.ai")
    print("   curl -fsSL https://ollama.ai/install.sh | sh")
    print("   ollama serve")
    print("   ollama pull llama3.1:8b")
    print("   ollama pull nomic-embed-text")
    
    print("\n3. Python Dependencies:")
    print("   pip install psycopg2-binary pgvector")
    print("   pip install sentence-transformers")
    print("   pip install aiohttp sqlalchemy asyncpg")
    
    print("\n4. Test Installation:")
    print("   python demo_rag.py")
    print("   python main.py --rag-stats")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--install-guide":
        show_installation_guide()
    else:
        asyncio.run(demo_rag_system())
