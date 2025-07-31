#!/usr/bin/env python3
"""
Test Ollama client functionality
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rag.ollama_client import OllamaClient

async def test_ollama():
    print("Testing Ollama client...")
    
    client = OllamaClient()
    
    # Test initialization
    print("1. Testing initialization...")
    if await client.initialize():
        print("✅ Ollama client initialized successfully")
    else:
        print("❌ Ollama client initialization failed")
        return
    
    # Test embeddings
    print("2. Testing embeddings...")
    try:
        embeddings = await client.generate_embeddings(["This is a test sentence"])
        if embeddings and len(embeddings) > 0:
            print(f"✅ Generated embeddings: {len(embeddings)} vectors of dimension {len(embeddings[0])}")
        else:
            print("❌ Failed to generate embeddings")
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
    
    # Test LLM response
    print("3. Testing LLM response...")
    try:
        response = await client.generate_response("You are a helpful assistant.", "What is the capital of France?")
        if response and isinstance(response, dict) and 'answer' in response:
            answer = response['answer']
            print(f"✅ LLM response: {answer[:100] if len(answer) > 100 else answer}...")
        else:
            print(f"❌ Failed to generate LLM response: {response}")
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
    
    await client.close()
    print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_ollama())
