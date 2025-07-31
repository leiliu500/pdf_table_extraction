"""
Simplified Vector Database for Demo
Mock implementation for testing without PostgreSQL
"""

import logging
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import asyncio

class VectorDatabase:
    """
    Simplified vector database for demo purposes
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.postgres_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'pdf_rag_db',
            'user': 'postgres',
            'password': 'postgres'
        }
        
    async def initialize(self) -> bool:
        """Mock initialization"""
        self.logger.info("Vector database (demo mode) initialized")
        return True
    
    async def store_document(self, file_path: str, filename: str, file_hash: str, 
                           file_size: int, extraction_metadata: Dict[str, Any],
                           extraction_accuracy: float) -> str:
        """Mock document storage"""
        doc_id = f"doc_{hash(file_path)}"
        self.logger.info(f"Would store document: {filename}")
        return doc_id
    
    async def store_chunks_with_embeddings(self, document_id: str, chunks_data: List[Dict[str, Any]]) -> bool:
        """Mock chunk storage"""
        self.logger.info(f"Would store {len(chunks_data)} chunks for {document_id}")
        return True
    
    async def similarity_search(self, query_embedding: List[float], top_k: int = 5,
                              content_types: List[str] = None, 
                              min_confidence: float = None) -> List[Dict[str, Any]]:
        """Mock similarity search"""
        self.logger.info("Would perform similarity search")
        return []
    
    async def close(self):
        """Mock close"""
        self.logger.info("Vector database closed")
