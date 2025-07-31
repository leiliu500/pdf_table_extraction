"""
Ollama Integration Module
Handles embeddings generation and LLM inference with local Ollama server
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union
import time
from datetime import datetime

import aiohttp
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config.rag_settings import OLLAMA_CONFIG, ACCURACY_CONFIG


class OllamaClient:
    """
    Client for interacting with local Ollama server
    Handles both embeddings and chat completions with accuracy tracking
    """
    
    def __init__(self):
        self.config = OLLAMA_CONFIG
        self.accuracy_config = ACCURACY_CONFIG
        self.logger = logging.getLogger(__name__)
        
        self.base_url = self.config['base_url']
        self.embedding_model = self.config['embedding_model']
        self.llm_model = self.config['llm_model']
        self.timeout = self.config['timeout']
        
        # Session for HTTP requests
        self.session = None
        
        # Fallback embedding model
        self.fallback_embedder = None
        self.use_fallback = False
        
        # Performance tracking
        self.embedding_times = []
        self.llm_response_times = []
        
    async def initialize(self) -> bool:
        """
        Initialize Ollama client and verify server connection
        """
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Test Ollama server connection
            if await self._test_connection():
                self.logger.info("Ollama server connection established")
                
                # Test embedding model
                if await self._test_embedding_model():
                    self.logger.info(f"Embedding model {self.embedding_model} ready")
                else:
                    self.logger.warning("Embedding model not available, initializing fallback")
                    self._initialize_fallback_embedder()
                
                # Test LLM model
                if await self._test_llm_model():
                    self.logger.info(f"LLM model {self.llm_model} ready")
                else:
                    self.logger.error(f"LLM model {self.llm_model} not available")
                    return False
                
                return True
            else:
                self.logger.error("Cannot connect to Ollama server")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama client: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test basic connection to Ollama server"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    models = await response.json()
                    self.logger.info(f"Available models: {[m['name'] for m in models.get('models', [])]}")
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def _test_embedding_model(self) -> bool:
        """Test if embedding model is available"""
        try:
            test_response = await self.generate_embeddings(["test"])
            return test_response is not None and len(test_response) > 0
        except Exception:
            return False
    
    async def _test_llm_model(self) -> bool:
        """Test if LLM model is available"""
        try:
            test_response = await self.generate_response("Test context", "Test question")
            return test_response is not None and isinstance(test_response, dict) and 'answer' in test_response
        except Exception as e:
            self.logger.error(f"LLM model test failed: {e}")
            return False
    
    def _initialize_fallback_embedder(self):
        """Initialize fallback sentence transformer model"""
        try:
            self.fallback_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_fallback = True
            self.logger.info("Fallback embedding model initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize fallback embedder: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for list of texts
        """
        if not texts:
            return []
        
        start_time = time.time()
        
        try:
            if self.use_fallback and self.fallback_embedder:
                # Use fallback sentence transformer
                embeddings = self.fallback_embedder.encode(texts, convert_to_numpy=True)
                embeddings_list = [emb.tolist() for emb in embeddings]
                
                processing_time = time.time() - start_time
                self.embedding_times.append(processing_time)
                
                self.logger.info(f"Generated {len(embeddings_list)} embeddings using fallback model in {processing_time:.2f}s")
                return embeddings_list
            
            else:
                # Use Ollama embedding model
                embeddings = []
                
                for text in texts:
                    payload = {
                        "model": self.embedding_model,
                        "prompt": text
                    }
                    
                    async with self.session.post(
                        f"{self.base_url}/api/embeddings",
                        json=payload
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            if 'embedding' in result:
                                embeddings.append(result['embedding'])
                            else:
                                self.logger.error(f"No embedding in response for text: {text[:50]}...")
                                return None
                        else:
                            error_text = await response.text()
                            self.logger.error(f"Embedding request failed: {response.status} - {error_text}")
                            return None
                
                processing_time = time.time() - start_time
                self.embedding_times.append(processing_time)
                
                self.logger.info(f"Generated {len(embeddings)} embeddings using Ollama in {processing_time:.2f}s")
                return embeddings
                
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return None
    
    async def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else None
    
    async def generate_response(self, context: str, question: str, 
                              system_prompt: str = None) -> Optional[Dict[str, Any]]:
        """
        Generate response using Ollama LLM with context and question
        """
        start_time = time.time()
        
        try:
            # Build system prompt
            if system_prompt is None:
                system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
                Follow these guidelines:
                1. Answer only based on the provided context
                2. If the context doesn't contain enough information, say so clearly
                3. Provide specific citations when possible
                4. Be accurate and concise
                5. If you're uncertain, express that uncertainty"""
            
            # Build user message
            user_message = f"""Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above."""
            
            # Prepare request payload
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "stream": False,
                "options": {
                    "temperature": self.config['temperature'],
                    "top_k": self.config['top_k'],
                    "top_p": self.config['top_p'],
                    "num_predict": self.config['max_tokens']
                }
            }
            
            self.logger.info(f"Sending LLM request - Context: {len(context)} chars, Question: {len(question)} chars")
            
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                
                self.logger.info(f"LLM response status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    self.logger.info(f"LLM response result: {result}")
                    
                    if 'message' in result and 'content' in result['message']:
                        processing_time = time.time() - start_time
                        self.llm_response_times.append(processing_time)
                        
                        response_data = {
                            'answer': result['message']['content'],
                            'model': self.llm_model,
                            'processing_time': processing_time,
                            'timestamp': datetime.utcnow().isoformat(),
                            'context_length': len(context),
                            'question_length': len(question),
                            'response_length': len(result['message']['content']),
                            'confidence_score': self._estimate_confidence(result['message']['content'])
                        }
                        
                        self.logger.info(f"Generated LLM response in {processing_time:.2f}s")
                        return response_data
                    else:
                        self.logger.error(f"No message content in LLM response. Result: {result}")
                        return None
                else:
                    error_text = await response.text()
                    self.logger.error(f"LLM request failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {e}")
            return None
    
    def _estimate_confidence(self, response_text: str) -> float:
        """
        Estimate confidence score based on response characteristics
        """
        try:
            # Simple heuristics for confidence estimation
            confidence_indicators = {
                'uncertainty_phrases': ['not sure', 'unclear', 'might be', 'possibly', 'uncertain'],
                'certainty_phrases': ['clearly', 'definitely', 'specifically', 'exactly', 'precisely'],
                'citation_indicators': ['according to', 'as stated', 'the document shows', 'mentioned in'],
                'hedging_words': ['perhaps', 'maybe', 'likely', 'appears', 'seems']
            }
            
            response_lower = response_text.lower()
            
            # Base confidence
            confidence = 0.7
            
            # Reduce confidence for uncertainty
            uncertainty_count = sum(1 for phrase in confidence_indicators['uncertainty_phrases'] 
                                  if phrase in response_lower)
            confidence -= uncertainty_count * 0.1
            
            # Reduce confidence for hedging
            hedging_count = sum(1 for word in confidence_indicators['hedging_words'] 
                              if word in response_lower)
            confidence -= hedging_count * 0.05
            
            # Increase confidence for certainty and citations
            certainty_count = sum(1 for phrase in confidence_indicators['certainty_phrases'] 
                                if phrase in response_lower)
            confidence += certainty_count * 0.05
            
            citation_count = sum(1 for phrase in confidence_indicators['citation_indicators'] 
                               if phrase in response_lower)
            confidence += citation_count * 0.1
            
            # Length factor (very short or very long responses might be less reliable)
            length_factor = len(response_text)
            if length_factor < 50:
                confidence -= 0.1
            elif length_factor > 1000:
                confidence -= 0.05
            
            # Clamp between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception:
            return 0.5  # Default moderate confidence
    
    async def generate_query_expansion(self, original_query: str) -> List[str]:
        """
        Generate expanded queries for better retrieval
        """
        try:
            system_prompt = """You are a query expansion specialist. Given a user query, generate 2-3 alternative phrasings that would help find relevant information. Focus on:
            1. Synonyms and alternative terms
            2. More specific variations
            3. Related concepts
            
            Return only the alternative queries, one per line, without numbering or explanation."""
            
            response = await self.generate_response("", original_query, system_prompt)
            
            if response and response.get('answer'):
                expanded_queries = [q.strip() for q in response['answer'].split('\n') if q.strip()]
                expanded_queries = [original_query] + expanded_queries  # Include original
                
                self.logger.info(f"Expanded query into {len(expanded_queries)} variations")
                return expanded_queries
            
            return [original_query]  # Fallback to original query
            
        except Exception as e:
            self.logger.error(f"Error in query expansion: {e}")
            return [original_query]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        """
        stats = {
            'embedding_model': self.embedding_model if not self.use_fallback else 'sentence-transformers fallback',
            'llm_model': self.llm_model,
            'using_fallback_embeddings': self.use_fallback
        }
        
        if self.embedding_times:
            stats['embedding_stats'] = {
                'total_requests': len(self.embedding_times),
                'average_time': np.mean(self.embedding_times),
                'min_time': np.min(self.embedding_times),
                'max_time': np.max(self.embedding_times)
            }
        
        if self.llm_response_times:
            stats['llm_stats'] = {
                'total_requests': len(self.llm_response_times),
                'average_time': np.mean(self.llm_response_times),
                'min_time': np.min(self.llm_response_times),
                'max_time': np.max(self.llm_response_times)
            }
        
        return stats
    
    async def close(self):
        """Close client session"""
        if self.session:
            await self.session.close()
        
        # Clear performance tracking
        self.embedding_times.clear()
        self.llm_response_times.clear()


class EmbeddingManager:
    """
    Manager for handling embeddings with quality validation
    """
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.logger = logging.getLogger(__name__)
        self.accuracy_config = ACCURACY_CONFIG
    
    async def generate_embeddings_with_validation(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate embeddings with quality validation
        """
        results = []
        
        try:
            embeddings = await self.ollama_client.generate_embeddings(texts)
            
            if not embeddings:
                self.logger.error("Failed to generate embeddings")
                return []
            
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                # Calculate embedding quality metrics
                quality_score = self._calculate_embedding_quality(embedding, text)
                
                result = {
                    'text': text,
                    'embedding': embedding,
                    'embedding_model': self.ollama_client.embedding_model,
                    'embedding_quality': quality_score,
                    'text_length': len(text),
                    'embedding_dimension': len(embedding),
                    'passes_quality_threshold': quality_score >= self.accuracy_config['embedding_quality_threshold']
                }
                
                results.append(result)
            
            self.logger.info(f"Generated {len(results)} embeddings with validation")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in embedding generation with validation: {e}")
            return []
    
    def _calculate_embedding_quality(self, embedding: List[float], text: str) -> float:
        """
        Calculate embedding quality score based on various metrics
        """
        try:
            # Convert to numpy array for calculations
            emb_array = np.array(embedding)
            
            # Quality factors
            quality_score = 1.0
            
            # Check for zero embeddings (indicates failure)
            if np.allclose(emb_array, 0):
                return 0.0
            
            # Check for NaN or infinite values
            if np.any(np.isnan(emb_array)) or np.any(np.isinf(emb_array)):
                return 0.0
            
            # Magnitude check (embeddings should be normalized or have reasonable magnitude)
            magnitude = np.linalg.norm(emb_array)
            if magnitude < 0.1 or magnitude > 100:
                quality_score *= 0.5
            
            # Variance check (good embeddings should have some variance)
            variance = np.var(emb_array)
            if variance < 1e-6:  # Very low variance indicates poor embedding
                quality_score *= 0.3
            
            # Text length factor
            text_length = len(text.strip())
            if text_length < 10:  # Very short text might have poor embeddings
                quality_score *= 0.7
            elif text_length > 2000:  # Very long text might be truncated
                quality_score *= 0.9
            
            # Dimension check
            expected_dim = 768 if not self.ollama_client.use_fallback else 384
            if len(embedding) != expected_dim:
                quality_score *= 0.8
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating embedding quality: {e}")
            return 0.0
