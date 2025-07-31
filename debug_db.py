#!/usr/bin/env python3
"""
Debug the database table relationships
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def debug_database():
    """Debug the database table relationships"""
    
    print("Debugging Database Tables...")
    print("=" * 60)
    
    try:
        from src.rag.vector_database import VectorDatabase
        
        # Initialize database
        db = VectorDatabase()
        await db.initialize()
        
        async with db.connection_pool.acquire() as conn:
            # Check documents table
            print("1. Documents table:")
            docs = await conn.fetch(f"SELECT * FROM {db.documents_table}")
            for doc in docs:
                print(f"   ID: {doc['id']}, Filename: {doc['filename']}")
            
            print(f"\n2. Chunks table:")
            chunks = await conn.fetch(f"SELECT id, document_id, chunk_index, content_type, confidence_score, LEFT(content, 50) as preview FROM {db.chunks_table} ORDER BY chunk_index")
            for chunk in chunks:
                print(f"   ID: {chunk['id']}, Doc ID: {chunk['document_id']}, Index: {chunk['chunk_index']}, Type: {chunk['content_type']}")
                print(f"       Preview: {chunk['preview']}...")
            
            print(f"\n3. Embeddings table:")
            embeddings = await conn.fetch(f"SELECT chunk_id, LEFT(embedding::text, 100) as embedding_preview FROM {db.embeddings_table}")
            for emb in embeddings:
                print(f"   Chunk ID: {emb['chunk_id']}, Embedding: {emb['embedding_preview']}...")
            
            print(f"\n4. Testing JOIN:")
            join_test = await conn.fetch(f"""
                SELECT c.id as chunk_id, c.content_type, e.chunk_id as embedding_chunk_id
                FROM {db.chunks_table} c
                LEFT JOIN {db.embeddings_table} e ON c.id = e.chunk_id
                LIMIT 5
            """)
            for result in join_test:
                print(f"   Chunk: {result['chunk_id']}, Type: {result['content_type']}, Embedding Chunk: {result['embedding_chunk_id']}")
            
            print(f"\n5. Testing vector similarity (simple):")
            # Create a simple test vector
            test_vector = "[" + ",".join(["0.1"] * 768) + "]"
            similarity_test = await conn.fetch(f"""
                SELECT c.id, c.content_type, (e.embedding <=> $1) as distance
                FROM {db.chunks_table} c
                JOIN {db.embeddings_table} e ON c.id = e.chunk_id
                ORDER BY distance
                LIMIT 3
            """, test_vector)
            
            for result in similarity_test:
                print(f"   Chunk: {result['id']}, Type: {result['content_type']}, Distance: {result['distance']}")
        
        await db.close()
        
    except Exception as e:
        print(f"Error debugging database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_database())
