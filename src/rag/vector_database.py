"""
Vector Database Module for PostgreSQL with pgvector
Handles document embeddings, similarity search, and vector operations
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import uuid

import asyncpg
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import psycopg2
from pgvector.psycopg2 import register_vector

from ..config.rag_settings import POSTGRES_CONFIG, VECTOR_CONFIG, ACCURACY_CONFIG


class VectorDatabase:
    """
    Vector database for storing and querying document embeddings with PostgreSQL and pgvector
    Focus on accuracy and efficient similarity search
    """
    
    def __init__(self):
        self.postgres_config = POSTGRES_CONFIG
        self.vector_config = VECTOR_CONFIG
        self.accuracy_config = ACCURACY_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Connection pools
        self.connection_pool = None
        self.async_engine = None
        self.sync_engine = None
        
        # Database schema
        self.documents_table = "pdf_documents"
        self.chunks_table = "document_chunks"
        self.embeddings_table = "chunk_embeddings"
        self.accuracy_table = "accuracy_metrics"
        
    async def initialize(self) -> bool:
        """
        Initialize database connection and create required tables
        """
        try:
            await self._create_database_if_not_exists()
            await self._setup_connection_pools()
            await self._create_tables()
            await self._create_indexes()
            
            self.logger.info("Vector database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            return False
    
    async def _create_database_if_not_exists(self):
        """Create the database if it doesn't exist"""
        try:
            # Connect to default postgres database to create our database
            conn = await asyncpg.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                database='postgres'  # Connect to default postgres database first
            )
            
            # Check if database exists
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                self.postgres_config['database']
            )
            
            if not db_exists:
                await conn.execute(f"CREATE DATABASE {self.postgres_config['database']}")
                self.logger.info(f"Created database: {self.postgres_config['database']}")
            
            await conn.close()
            
        except Exception as e:
            self.logger.error(f"Error creating database: {e}")
            raise
    
    async def _setup_connection_pools(self):
        """Setup async and sync connection pools"""
        try:
            # Async connection pool
            self.connection_pool = await asyncpg.create_pool(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                database=self.postgres_config['database'],
                min_size=5,
                max_size=self.postgres_config['pool_size']
            )
            
            # SQLAlchemy engines
            db_url = f"postgresql+asyncpg://{self.postgres_config['user']}:{self.postgres_config['password']}@{self.postgres_config['host']}:{self.postgres_config['port']}/{self.postgres_config['database']}"
            self.async_engine = create_async_engine(db_url, pool_size=10, max_overflow=20)
            
            sync_db_url = f"postgresql://{self.postgres_config['user']}:{self.postgres_config['password']}@{self.postgres_config['host']}:{self.postgres_config['port']}/{self.postgres_config['database']}"
            self.sync_engine = create_engine(sync_db_url, pool_size=10, max_overflow=20)
            
        except Exception as e:
            self.logger.error(f"Error setting up connection pools: {e}")
            raise
    
    async def _create_tables(self):
        """Create all required tables with proper schema"""
        async with self.connection_pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Documents table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.documents_table} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    file_path TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size BIGINT NOT NULL,
                    processing_date TIMESTAMPTZ DEFAULT NOW(),
                    extraction_metadata JSONB,
                    extraction_accuracy FLOAT,
                    total_chunks INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'processed'
                )
            """)
            
            # Document chunks table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.chunks_table} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES {self.documents_table}(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content_type TEXT NOT NULL, -- 'text', 'table', 'form', 'image_ocr'
                    content TEXT NOT NULL,
                    metadata JSONB,
                    confidence_score FLOAT,
                    page_number INTEGER,
                    extraction_method TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Embeddings table with vector column
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chunk_id UUID REFERENCES {self.chunks_table}(id) ON DELETE CASCADE,
                    embedding vector({self.vector_config['dimension']}),
                    embedding_model TEXT NOT NULL,
                    embedding_quality FLOAT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Accuracy metrics table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.accuracy_table} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES {self.documents_table}(id) ON DELETE CASCADE,
                    metric_type TEXT NOT NULL, -- 'extraction', 'embedding', 'retrieval', 'answer'
                    metric_name TEXT NOT NULL, -- 'precision', 'recall', 'f1_score', 'confidence'
                    metric_value FLOAT NOT NULL,
                    metadata JSONB,
                    measured_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
    
    async def _create_indexes(self):
        """Create optimized indexes for similarity search and performance"""
        async with self.connection_pool.acquire() as conn:
            # Vector similarity index
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
                ON {self.embeddings_table} 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = {self.vector_config['index_lists']})
            """)
            
            # Document lookup indexes
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_documents_file_path ON {self.documents_table}(file_path)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_documents_hash ON {self.documents_table}(file_hash)")
            
            # Chunk lookup indexes
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON {self.chunks_table}(document_id)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON {self.chunks_table}(content_type)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_chunks_confidence ON {self.chunks_table}(confidence_score)")
            
            # Embedding indexes
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON {self.embeddings_table}(chunk_id)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_embeddings_model ON {self.embeddings_table}(embedding_model)")
            
            # Accuracy indexes
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_accuracy_document_id ON {self.accuracy_table}(document_id)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_accuracy_type ON {self.accuracy_table}(metric_type)")
    
    async def store_document(self, file_path: str, filename: str, file_hash: str, 
                           file_size: int, extraction_metadata: Dict[str, Any],
                           extraction_accuracy: float) -> str:
        """
        Store document metadata and return document ID
        """
        try:
            async with self.connection_pool.acquire() as conn:
                document_id = await conn.fetchval(f"""
                    INSERT INTO {self.documents_table} 
                    (file_path, filename, file_hash, file_size, extraction_metadata, extraction_accuracy)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """, file_path, filename, file_hash, file_size, json.dumps(extraction_metadata), extraction_accuracy)
                
                self.logger.info(f"Stored document: {filename} with ID: {document_id}")
                return str(document_id)
                
        except Exception as e:
            self.logger.error(f"Error storing document {filename}: {e}")
            raise
    
    async def store_chunks_with_embeddings(self, document_id: str, chunks_data: List[Dict[str, Any]]) -> bool:
        """
        Store document chunks and their embeddings in batch
        """
        try:
            async with self.connection_pool.acquire() as conn:
                async with conn.transaction():
                    # Store chunks
                    chunk_records = []
                    embedding_records = []
                    
                    for chunk_data in chunks_data:
                        # Insert chunk
                        chunk_id = await conn.fetchval(f"""
                            INSERT INTO {self.chunks_table}
                            (document_id, chunk_index, content_type, content, metadata, 
                             confidence_score, page_number, extraction_method)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            RETURNING id
                        """, 
                        document_id, chunk_data['chunk_index'], chunk_data['content_type'],
                        chunk_data['content'], json.dumps(chunk_data.get('metadata', {})),
                        chunk_data.get('confidence_score', 0.0), chunk_data.get('page_number'),
                        chunk_data.get('extraction_method'))
                        
                        # Prepare embedding record
                        if 'embedding' in chunk_data:
                            # Convert embedding list to PostgreSQL vector format
                            embedding_vector = f"[{','.join(map(str, chunk_data['embedding']))}]"
                            embedding_records.append((
                                chunk_id, embedding_vector, chunk_data.get('embedding_model', 'unknown'),
                                chunk_data.get('embedding_quality', 0.0)
                            ))
                    
                    # Batch insert embeddings
                    if embedding_records:
                        await conn.executemany(f"""
                            INSERT INTO {self.embeddings_table}
                            (chunk_id, embedding, embedding_model, embedding_quality)
                            VALUES ($1, $2, $3, $4)
                        """, embedding_records)
                    
                    # Update document chunk count
                    await conn.execute(f"""
                        UPDATE {self.documents_table} 
                        SET total_chunks = $1 
                        WHERE id = $2
                    """, len(chunks_data), document_id)
                    
                    self.logger.info(f"Stored {len(chunks_data)} chunks for document {document_id}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error storing chunks for document {document_id}: {e}")
            return False
    
    async def similarity_search(self, query_embedding: List[float], top_k: int = None,
                              content_types: List[str] = None, 
                              min_confidence: float = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search with optional filters
        """
        if top_k is None:
            top_k = self.vector_config['max_results']
        
        if min_confidence is None:
            min_confidence = self.accuracy_config['extraction_confidence_threshold']
        
        try:
            # Convert query embedding to PostgreSQL vector format
            query_vector = f"[{','.join(map(str, query_embedding))}]"
            
            async with self.connection_pool.acquire() as conn:
                # Build query with filters
                where_conditions = ["c.confidence_score >= $3"]
                params = [query_vector, top_k, min_confidence]
                param_count = 3
                
                if content_types:
                    param_count += 1
                    where_conditions.append(f"c.content_type = ANY(${param_count})")
                    params.append(content_types)
                
                where_clause = " AND ".join(where_conditions)
                
                query = f"""
                    SELECT 
                        c.id as chunk_id,
                        c.content,
                        c.content_type,
                        c.metadata,
                        c.confidence_score,
                        c.page_number,
                        c.extraction_method,
                        d.filename,
                        d.file_path,
                        e.embedding_quality,
                        (e.embedding <=> $1) as similarity_distance
                    FROM {self.chunks_table} c
                    JOIN {self.embeddings_table} e ON c.id = e.chunk_id
                    JOIN {self.documents_table} d ON c.document_id = d.id
                    WHERE {where_clause}
                    ORDER BY e.embedding <=> $1
                    LIMIT $2
                """
                
                results = await conn.fetch(query, *params)
                
                # Convert to list of dictionaries
                search_results = []
                for row in results:
                    similarity_score = 1.0 - row['similarity_distance']  # Convert distance to similarity
                    
                    if similarity_score >= self.vector_config['similarity_threshold']:
                        search_results.append({
                            'chunk_id': str(row['chunk_id']),
                            'content': row['content'],
                            'content_type': row['content_type'],
                            'metadata': row['metadata'] if row['metadata'] else {},
                            'confidence_score': row['confidence_score'],
                            'page_number': row['page_number'],
                            'extraction_method': row['extraction_method'],
                            'filename': row['filename'],
                            'file_path': row['file_path'],
                            'embedding_quality': row['embedding_quality'],
                            'similarity_score': similarity_score
                        })
                
                self.logger.info(f"Found {len(search_results)} similar chunks")
                return search_results
                
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    async def store_accuracy_metric(self, document_id: str, metric_type: str, 
                                  metric_name: str, metric_value: float,
                                  metadata: Dict[str, Any] = None) -> bool:
        """
        Store accuracy metrics for tracking system performance
        """
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute(f"""
                    INSERT INTO {self.accuracy_table}
                    (document_id, metric_type, metric_name, metric_value, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                """, document_id, metric_type, metric_name, metric_value, 
                json.dumps(metadata) if metadata else None)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing accuracy metric: {e}")
            return False
    
    async def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Check if document already exists by hash
        """
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(f"""
                    SELECT id, file_path, filename, processing_date, total_chunks
                    FROM {self.documents_table}
                    WHERE file_hash = $1
                """, file_hash)
                
                if row:
                    return {
                        'id': str(row['id']),
                        'file_path': row['file_path'],
                        'filename': row['filename'],
                        'processing_date': row['processing_date'],
                        'total_chunks': row['total_chunks']
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Error checking document hash: {e}")
            return None
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics for monitoring
        """
        try:
            async with self.connection_pool.acquire() as conn:
                stats = {}
                
                # Document count
                stats['total_documents'] = await conn.fetchval(f"SELECT COUNT(*) FROM {self.documents_table}")
                
                # Chunk count by type
                chunk_stats = await conn.fetch(f"""
                    SELECT content_type, COUNT(*) as count
                    FROM {self.chunks_table}
                    GROUP BY content_type
                """)
                stats['chunks_by_type'] = {row['content_type']: row['count'] for row in chunk_stats}
                
                # Average confidence scores
                avg_confidence = await conn.fetchval(f"""
                    SELECT AVG(confidence_score) FROM {self.chunks_table}
                    WHERE confidence_score > 0
                """)
                stats['average_confidence'] = float(avg_confidence) if avg_confidence else 0.0
                
                # Embedding count
                stats['total_embeddings'] = await conn.fetchval(f"SELECT COUNT(*) FROM {self.embeddings_table}")
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
    
    async def close(self):
        """Close all database connections"""
        if self.connection_pool:
            await self.connection_pool.close()
        if self.async_engine:
            await self.async_engine.dispose()
        if self.sync_engine:
            self.sync_engine.dispose()
