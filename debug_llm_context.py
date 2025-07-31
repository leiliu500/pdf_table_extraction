#!/usr/bin/env python3
"""
Debug LLM Context Issue
Test the exact same context that's failing in the RAG pipeline
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag.ollama_client import OllamaClient
from rag.database import DatabaseManager

async def main():
    print("🔍 DEBUGGING LLM CONTEXT ISSUE")
    print("=" * 50)
    
    # Initialize components
    ollama = OllamaClient()
    db = DatabaseManager()
    
    try:
        # Check if Ollama is available
        print("1. Checking Ollama availability...")
        available = await ollama.check_availability()
        if not available:
            print("❌ Ollama is not available")
            return
        print("✅ Ollama is available")
        
        # Initialize database
        print("\n2. Initializing database...")
        await db.initialize()
        print("✅ Database initialized")
        
        # Get the exact context that's failing
        print("\n3. Getting chunks for address query...")
        query = "What is the address of the property?"
        embedding = await ollama.generate_embedding(query)
        
        if not embedding:
            print("❌ Failed to generate embedding")
            return
            
        chunks = await db.search_similar_chunks(embedding, limit=3, similarity_threshold=0.3)
        print(f"✅ Found {len(chunks)} chunks")
        
        if not chunks:
            print("❌ No chunks found")
            return
            
        # Build context exactly like in the RAG pipeline
        context_chunks = []
        for chunk in chunks:
            chunk_data = chunk.get('chunk_data', {})
            content = chunk_data.get('content', '')
            if content and len(content.strip()) > 10:
                context_chunks.append(content)
        
        if not context_chunks:
            print("❌ No valid context chunks")
            return
            
        context_text = "\n\n".join(context_chunks)
        print(f"✅ Built context: {len(context_text)} characters")
        print(f"Context preview: {context_text[:200]}...")
        
        # Test with shorter context first
        print("\n4. Testing with short context...")
        short_context = context_text[:500]
        response = await ollama.generate_response(short_context, query)
        print(f"Short context response: {response}")
        
        # Test with full context
        print("\n5. Testing with full context...")
        response = await ollama.generate_response(context_text, query)
        print(f"Full context response: {response}")
        
        # Test with even longer timeout
        print("\n6. Testing with custom timeout...")
        import aiohttp
        
        # Manually test the API call
        start_time = time.time()
        payload = {
            "model": "llama3.1:8b",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a helpful AI assistant that answers questions based on the provided context."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nPlease provide a detailed answer based on the context above."
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_k": 40,
                "top_p": 0.9,
                "num_predict": 2048
            }
        }
        
        timeout = aiohttp.ClientTimeout(total=120)  # 2 minutes
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    "http://localhost:11434/api/chat",
                    json=payload
                ) as response:
                    print(f"Response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        print(f"Response keys: {list(result.keys())}")
                        if 'message' in result:
                            print(f"Message keys: {list(result['message'].keys())}")
                            if 'content' in result['message']:
                                content = result['message']['content']
                                print(f"✅ Success! Response: {content[:200]}...")
                            else:
                                print(f"❌ No content in message: {result['message']}")
                        else:
                            print(f"❌ No message in result: {result}")
                    else:
                        error_text = await response.text()
                        print(f"❌ Error {response.status}: {error_text}")
            except Exception as e:
                print(f"❌ Request failed: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await ollama.close()
        await db.close()

if __name__ == "__main__":
    import time
    asyncio.run(main())
