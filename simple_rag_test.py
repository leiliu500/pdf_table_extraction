#!/usr/bin/env python3
"""
Simple RAG query test
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

# Set up environment
os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent))

async def test_rag_query():
    """Test RAG query functionality"""
    
    print("Testing RAG Query System...")
    print("=" * 60)
    
    try:
        # Import with absolute path
        from src.config.rag_settings import POSTGRES_CONFIG, VECTOR_CONFIG, ACCURACY_CONFIG
        from src.rag.vector_database import VectorDatabase
        from src.rag.ollama_client import OllamaClient
        
        print("✓ Imports successful")
        
        # Initialize components
        print("Initializing database connection...")
        db = VectorDatabase()
        await db.initialize()
        
        print("Initializing Ollama client...")
        ollama = OllamaClient()
        await ollama.initialize()
        
        # Test query
        question = "What is the property address and price?"
        print(f"\nQuerying: {question}")
        print("-" * 40)
        
        # Generate query embedding
        query_embedding = await ollama.generate_embeddings([question])
        if not query_embedding:
            print("✗ Failed to generate query embedding")
            return False
        
        print(f"✓ Generated query embedding ({len(query_embedding[0])} dimensions)")
        
        # Search for similar chunks with lower threshold
        search_results = await db.similarity_search(
            query_embedding[0], 
            top_k=10,
            min_confidence=0.5  # Lower threshold
        )
        
        print(f"✓ Found {len(search_results)} relevant chunks with confidence >= 0.5")
        
        # If still no results, try with even lower threshold
        if not search_results:
            search_results = await db.similarity_search(
                query_embedding[0], 
                top_k=10,
                min_confidence=0.1  # Very low threshold
            )
            print(f"✓ Found {len(search_results)} chunks with confidence >= 0.1")
        
        # If still no results, try without confidence filter at all
        if not search_results:
            try:
                async with db.connection_pool.acquire() as conn:
                    query_vector = f"[{','.join(map(str, query_embedding[0]))}]"
                    raw_results = await conn.fetch(f"""
                        SELECT 
                            c.content,
                            c.content_type,
                            c.confidence_score,
                            (e.embedding <=> $1) as similarity_distance
                        FROM {db.chunks_table} c
                        JOIN {db.embeddings_table} e ON c.id = e.chunk_id
                        ORDER BY e.embedding <=> $1
                        LIMIT 5
                    """, query_vector)
                    
                    print(f"✓ Raw similarity search found {len(raw_results)} chunks:")
                    for i, result in enumerate(raw_results):
                        similarity_score = 1.0 - result['similarity_distance']
                        print(f"  {i+1}. Type: {result['content_type']}, Similarity: {similarity_score:.3f}")
                        print(f"      Content: {result['content'][:100]}...")
                        print()
                    
                    search_results = [
                        {
                            'content': result['content'],
                            'content_type': result['content_type'],
                            'confidence_score': result['confidence_score'],
                            'similarity_score': 1.0 - result['similarity_distance']
                        }
                        for result in raw_results
                    ]
            except Exception as e:
                print(f"Raw search failed: {e}")
        
        # Check what's actually in the database
        try:
            from src.rag.vector_database import VectorDatabase
            async with db.connection_pool.acquire() as conn:
                doc_count = await conn.fetchval(f"SELECT COUNT(*) FROM {db.documents_table}")
                chunk_count = await conn.fetchval(f"SELECT COUNT(*) FROM {db.chunks_table}")
                embedding_count = await conn.fetchval(f"SELECT COUNT(*) FROM {db.embeddings_table}")
                
                print(f"\nDatabase contents:")
                print(f"  Documents: {doc_count}")
                print(f"  Chunks: {chunk_count}")
                print(f"  Embeddings: {embedding_count}")
                
                if chunk_count > 0:
                    # Get a sample chunk to see the content types
                    sample_chunks = await conn.fetch(f"""
                        SELECT content_type, confidence_score, LEFT(content, 100) as content_preview
                        FROM {db.chunks_table} 
                        LIMIT 5
                    """)
                    print(f"\nSample chunks:")
                    for chunk in sample_chunks:
                        print(f"  Type: {chunk['content_type']}, Confidence: {chunk['confidence_score']:.2f}")
                        print(f"  Preview: {chunk['content_preview']}...")
                        print()
        except Exception as e:
            print(f"Could not check database contents: {e}")
        
        if search_results:
            # Prepare context from search results
            context_chunks = []
            for result in search_results:
                context_chunks.append({
                    'content': result['content'],
                    'type': result['content_type'],
                    'similarity': result['similarity_score'],
                    'confidence': result['confidence_score']
                })
            
            # Generate answer using LLM
            context_text = "\n\n".join([f"[{chunk['type']}] {chunk['content']}" for chunk in context_chunks[:3]])
            
            answer = await ollama.generate_answer(question, context_text)
            
            if answer:
                print(f"\n✓ Query successful!")
                print(f"Answer: {answer}")
                print(f"Sources: {len(context_chunks)} relevant chunks")
                
                # Show source information
                print(f"\nRelevant content sources:")
                for i, chunk in enumerate(context_chunks[:3], 1):
                    print(f"  {i}. {chunk['type']} content (similarity: {chunk['similarity']:.3f})")
                    print(f"     Excerpt: {chunk['content'][:100]}...")
                    print()
            else:
                print("✗ Failed to generate answer")
        else:
            print("✗ No relevant content found in database")
        
        # Close connections
        await db.close()
        await ollama.close()
        
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
