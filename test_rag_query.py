#!/usr/bin/env python3
"""
Test script for RAG query functionality
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.rag_pipeline import RAGPipeline


async def test_rag_query():
    """Test RAG query functionality"""
    
    print("Testing RAG Query System...")
    print("=" * 60)
    
    try:
        # Initialize RAG pipeline
        print("Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        await pipeline.initialize()
        
        # Test query
        question = "What is the property address and price?"
        print(f"\nQuerying: {question}")
        print("-" * 40)
        
        result = await pipeline.query_documents(question)
        
        if result:
            print(f"\n✓ Query successful!")
            print(f"Answer: {result.get('answer', 'No answer generated')}")
            print(f"Confidence: {result.get('confidence_score', 0):.2f}")
            print(f"Sources: {len(result.get('source_chunks', []))} relevant chunks")
            
            # Show source information
            if result.get('source_chunks'):
                print(f"\nRelevant content sources:")
                for i, chunk in enumerate(result['source_chunks'][:3], 1):
                    print(f"  {i}. {chunk.get('content_type', 'unknown')} content")
                    print(f"     Similarity: {chunk.get('similarity_score', 0):.3f}")
                    print(f"     Excerpt: {chunk.get('content', '')[:100]}...")
                    print()
        else:
            print("✗ Query failed or returned no results")
        
        # Close pipeline
        await pipeline.close()
        
    except Exception as e:
        print(f"✗ Error testing RAG query: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_rag_query())
    if success:
        print("✓ RAG query test completed successfully!")
    else:
        print("✗ RAG query test failed!")
        sys.exit(1)
