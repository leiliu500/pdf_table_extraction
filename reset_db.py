#!/usr/bin/env python3
"""
Reset RAG database tables
"""
import asyncio
import asyncpg
import sys

async def reset_database():
    try:
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='ttoulliu2002',
            password='aA101@math',
            database='pdf_rag'
        )
        
        print("Connected to database pdf_rag")
        
        # Drop existing tables in correct order (reverse dependency order)
        tables = ['accuracy_metrics', 'chunk_embeddings', 'document_chunks', 'pdf_documents']
        for table in tables:
            try:
                await conn.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
                print(f'✓ Dropped table: {table}')
            except Exception as e:
                print(f'⚠️  Error dropping {table}: {e}')
        
        print('✅ Database reset complete')
        await conn.close()
        
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(reset_database())
