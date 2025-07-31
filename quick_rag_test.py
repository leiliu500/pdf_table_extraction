#!/usr/bin/env python3
"""
Quick RAG test to verify the timeout fix
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def quick_test():
    """Quick test of RAG system"""
    
    print("🔍 QUICK RAG TEST")
    print("=" * 40)
    
    try:
        from src.rag.vector_database import VectorDatabase
        from src.rag.ollama_client import OllamaClient
        
        # Initialize components
        print("✅ Initializing components...")
        db = VectorDatabase()
        await db.initialize()
        
        ollama = OllamaClient()
        await ollama.initialize()
        
        print(f"✅ Ollama timeout: {ollama.timeout}s")
        
        # Test single query
        question = "What is the address of the property?"
        print(f"🔍 Testing: {question}")
        
        # Generate embedding
        embeddings = await ollama.generate_embeddings([question])
        if not embeddings:
            print("❌ Failed to generate embedding")
            return
        
        print("✅ Generated embedding")
        
        # Search for chunks
        chunks = await db.similarity_search(embeddings[0], top_k=3, min_confidence=0.3)
        if not chunks:
            print("❌ No chunks found")
            return
            
        print(f"✅ Found {len(chunks)} chunks")
        
        # Build context
        context_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {list(chunk.keys())}")
            chunk_data = chunk.get('chunk_data', {})
            if chunk_data:
                print(f"      chunk_data keys: {list(chunk_data.keys())}")
            content = chunk_data.get('content', '')
            if not content:
                # Try alternative keys
                content = chunk.get('content', '')
                if not content:
                    content = str(chunk.get('chunk_text', ''))
            
            print(f"      content length: {len(content)}")
            print(f"      content preview: {str(content)[:100]}")
            
            if content and len(str(content).strip()) > 10:
                context_chunks.append(str(content))
        
        if not context_chunks:
            print("❌ No valid context")
            print("   Available chunk data:")
            for i, chunk in enumerate(chunks):
                print(f"   Chunk {i+1}: {chunk}")
            return
            
        context_text = "\n\n".join(context_chunks)
        print(f"✅ Built context: {len(context_text)} chars")
        
        # Test LLM response
        print("🔍 Generating LLM response...")
        response_data = await ollama.generate_response(context_text, question)
        
        if response_data and isinstance(response_data, dict) and 'answer' in response_data:
            answer = response_data['answer']
            confidence = response_data.get('confidence_score', 0)
            print(f"✅ SUCCESS!")
            print(f"   Answer: {answer[:100]}...")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Processing time: {response_data.get('processing_time', 0):.2f}s")
        else:
            print(f"❌ LLM failed: {response_data}")
        
        # Cleanup
        await ollama.close()
        await db.close()
        
        print("\n🎉 QUICK TEST COMPLETE!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())
