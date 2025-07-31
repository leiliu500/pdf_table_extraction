"""
Simplified Ollama Client for Demo
Mock implementation for testing without Ollama server
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

class OllamaClient:
    """
    Simplified Ollama client for demo purposes
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = 'http://localhost:11434'
        self.embedding_model = 'nomic-embed-text'
        self.llm_model = 'llama3.1:8b'
        
    async def initialize(self) -> bool:
        """Mock initialization"""
        self.logger.info("Ollama client (demo mode) initialized")
        return True
    
    async def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Mock embedding generation"""
        self.logger.info(f"Would generate embeddings for {len(texts)} texts")
        # Return mock embeddings (768 dimensions for nomic-embed-text)
        return [[0.1] * 768 for _ in texts]
    
    async def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """Mock single embedding"""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else None
    
    async def generate_response(self, context: str, question: str, 
                              system_prompt: str = None) -> Optional[Dict[str, Any]]:
        """Mock response generation"""
        self.logger.info("Would generate LLM response")
        return {
            'answer': f"Mock answer for: {question}",
            'confidence_score': 0.85,
            'model': self.llm_model
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Mock performance stats"""
        return {
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'total_requests': 0
        }
    
    async def close(self):
        """Mock close"""
        self.logger.info("Ollama client closed")

class EmbeddingManager:
    """Mock embedding manager"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.logger = logging.getLogger(__name__)
    
    async def generate_embeddings_with_validation(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Mock embeddings with validation"""
        results = []
        for text in texts:
            results.append({
                'text': text,
                'embedding': [0.1] * 768,
                'embedding_model': self.ollama_client.embedding_model,
                'embedding_quality': 0.9,
                'passes_quality_threshold': True
            })
        return results
