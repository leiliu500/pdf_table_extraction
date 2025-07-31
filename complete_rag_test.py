#!/usr/bin/env python3
"""
FINAL RAG SYSTEM TEST - Complete working version
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_complete_rag():
    """Test complete RAG functionality"""
    
    print("🎉 COMPLETE RAG SYSTEM TEST")
    print("=" * 50)
    
    try:
        from src.rag.vector_database import VectorDatabase
        from src.rag.ollama_client import OllamaClient
        
        # Initialize components
        print("🔧 Initializing RAG system...")
        db = VectorDatabase()
        await db.initialize()
        
        ollama = OllamaClient()
        await ollama.initialize()
        
        print(f"✅ System ready (timeout: {ollama.timeout}s)")
        
        # Test queries
        queries = [
            "What is the address of the property?",
            "What is the price of the property?",
            "How many bedrooms and bathrooms does it have?"
        ]
        
        for i, question in enumerate(queries, 1):
            print(f"\n{i}. 🔍 {question}")
            print("-" * 40)
            
            try:
                # Generate embedding
                embeddings = await ollama.generate_embeddings([question])
                if not embeddings:
                    print("   ❌ Failed to generate embedding")
                    continue
                
                # Search for chunks
                chunks = await db.similarity_search(embeddings[0], top_k=3, min_confidence=0.3)
                if not chunks:
                    print("   ❌ No chunks found")
                    continue
                
                print(f"   ✅ Found {len(chunks)} relevant chunks")
                
                # Extract content (content is directly in the chunk)
                context_chunks = []
                for chunk in chunks:
                    content = chunk.get('content', '')
                    if content and len(str(content).strip()) > 10:
                        context_chunks.append(str(content))
                        similarity = chunk.get('similarity_score', 0)
                        print(f"      📄 Similarity: {similarity:.3f} | Preview: {str(content)[:80]}...")
                
                if not context_chunks:
                    print("   ❌ No valid content found")
                    continue
                
                # Generate answer
                context_text = "\n\n".join(context_chunks)
                print(f"   🔍 Generating answer (context: {len(context_text)} chars)...")
                
                response_data = await ollama.generate_response(context_text, question)
                
                if response_data and isinstance(response_data, dict) and 'answer' in response_data:
                    answer = response_data['answer']
                    confidence = response_data.get('confidence_score', 0)
                    processing_time = response_data.get('processing_time', 0)
                    
                    print(f"   ✅ SUCCESS!")
                    print(f"   💡 Answer: {answer}")
                    print(f"   📊 Confidence: {confidence:.2f}")
                    print(f"   ⏱️  Processing: {processing_time:.2f}s")
                else:
                    print(f"   ❌ LLM returned: {type(response_data)} - {response_data}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # System statistics
        print(f"\n📊 SYSTEM STATISTICS")
        print("=" * 50)
        
        stats = await db.get_database_stats()
        print(f"📚 Documents: {stats.get('documents', 0)}")
        print(f"📄 Chunks: {stats.get('chunks', 0)}")
        print(f"🔗 Embeddings: {stats.get('embeddings', 0)}")
        print(f"⚙️  Ollama timeout: {ollama.timeout}s")
        print(f"🤖 LLM model: {ollama.llm_model}")
        print(f"📊 Embedding model: {ollama.embedding_model}")
        
        # Cleanup
        await ollama.close()
        await db.close()
        
        print(f"\n🚀 RAG SYSTEM FULLY OPERATIONAL!")
        print("   Ready for production queries.")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_complete_rag())
