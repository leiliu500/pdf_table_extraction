#!/usr/bin/env python3
"""
Test Ollama LLM directly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_ollama_llm():
    """Test Ollama LLM directly"""
    
    print("Testing Ollama LLM directly...")
    print("=" * 50)
    
    try:
        from src.rag.ollama_client import OllamaClient
        
        # Initialize Ollama
        ollama = OllamaClient()
        await ollama.initialize()
        
        # Test simple LLM call
        context = "The property is located at 304 Cedar Street, San Carlos 94070. It has 3 bedrooms and 1 bathroom."
        question = "What is the address of the property?"
        
        print(f"Context: {context}")
        print(f"Question: {question}")
        print("-" * 30)
        
        # Test the LLM response
        response_data = await ollama.generate_response(context, question)
        
        if response_data:
            print(f"✅ LLM Response successful!")
            print(f"Answer: {response_data.get('answer', 'No answer')}")
            print(f"Processing time: {response_data.get('processing_time', 0):.2f}s")
            print(f"Confidence: {response_data.get('confidence_score', 0):.2f}")
        else:
            print("❌ LLM Response failed")
        
        # Test if the model is running
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:11434/api/tags') as resp:
                if resp.status == 200:
                    models = await resp.json()
                    print(f"\n📋 Available models:")
                    for model in models.get('models', []):
                        print(f"   • {model.get('name', 'Unknown')}")
                else:
                    print(f"❌ Could not get model list: {resp.status}")
        
        await ollama.close()
        
    except Exception as e:
        print(f"❌ Error testing Ollama LLM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ollama_llm())
