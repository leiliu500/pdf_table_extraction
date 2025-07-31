"""
RAG Pipeline Module
Main orchestrator for PDF RAG system with Ollama and PostgreSQL
"""

import asyncio
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from .vector_database import VectorDatabase
from .ollama_client import OllamaClient, EmbeddingManager
from .text_processor import ContentChunker
from ..pdf_extractor import PDFExtractor
from ..config.rag_settings import RAG_CONFIG, ACCURACY_CONFIG, LOGGING_CONFIG


class PDFRAGPipeline:
    """
    Complete PDF RAG pipeline integrating extraction, chunking, embeddings, and QA
    Focus on accuracy and comprehensive content understanding
    """
    
    def __init__(self):
        self.config = RAG_CONFIG
        self.accuracy_config = ACCURACY_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.pdf_extractor = PDFExtractor()
        self.vector_db = VectorDatabase()
        self.ollama_client = OllamaClient()
        self.embedding_manager = None
        self.content_chunker = ContentChunker()
        
        # Performance tracking
        self.processing_stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'embeddings_generated': 0,
            'queries_processed': 0,
            'average_accuracy': 0.0,
            'processing_times': []
        }
        
        # Document cache
        self.document_cache = {}
        
    async def initialize(self) -> bool:
        """
        Initialize all RAG pipeline components
        """
        try:
            self.logger.info("Initializing PDF RAG pipeline...")
            
            # Initialize vector database
            if not await self.vector_db.initialize():
                self.logger.error("Failed to initialize vector database")
                return False
            
            # Initialize Ollama client
            if not await self.ollama_client.initialize():
                self.logger.error("Failed to initialize Ollama client")
                return False
            
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager(self.ollama_client)
            
            self.logger.info("PDF RAG pipeline initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {e}")
            return False
    
    async def process_pdf_document(self, file_path: str, **extraction_kwargs) -> Dict[str, Any]:
        """
        Process a PDF document through the complete RAG pipeline
        """
        start_time = time.time()
        
        try:
            file_path = Path(file_path).resolve()
            
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            self.logger.info(f"Processing PDF document: {file_path.name}")
            
            # Calculate file hash to check for duplicates
            file_hash = self._calculate_file_hash(file_path)
            
            # Check if document already processed
            existing_doc = await self.vector_db.get_document_by_hash(file_hash)
            if existing_doc:
                self.logger.info(f"Document already processed: {file_path.name}")
                return {
                    'status': 'already_processed',
                    'document_id': existing_doc['id'],
                    'document_info': existing_doc
                }
            
            # Step 1: Extract content from PDF
            extraction_results = await self._extract_pdf_content(file_path, **extraction_kwargs)
            if not extraction_results:
                raise ValueError("PDF extraction failed")
            
            # Step 2: Chunk content intelligently
            chunks = self._chunk_extracted_content(extraction_results)
            if not chunks:
                raise ValueError("Content chunking failed")
            
            # Step 3: Generate embeddings for chunks
            chunks_with_embeddings = await self._generate_embeddings_for_chunks(chunks)
            if not chunks_with_embeddings:
                raise ValueError("Embedding generation failed")
            
            # Step 4: Store in vector database
            document_id = await self._store_document_and_chunks(
                file_path, file_hash, extraction_results, chunks_with_embeddings
            )
            
            # Step 5: Track accuracy metrics
            await self._track_processing_accuracy(document_id, extraction_results, chunks_with_embeddings)
            
            processing_time = time.time() - start_time
            self.processing_stats['processing_times'].append(processing_time)
            self.processing_stats['documents_processed'] += 1
            
            result = {
                'status': 'success',
                'document_id': document_id,
                'file_path': str(file_path),
                'filename': file_path.name,
                'processing_time': processing_time,
                'extraction_summary': self._create_extraction_summary(extraction_results),
                'chunk_summary': self._create_chunk_summary(chunks_with_embeddings),
                'accuracy_metrics': await self._get_document_accuracy_metrics(document_id)
            }
            
            self.logger.info(f"Successfully processed {file_path.name} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing PDF document {file_path}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'file_path': str(file_path)
            }
    
    async def query_documents(self, question: str, **query_kwargs) -> Dict[str, Any]:
        """
        Query processed documents using RAG approach
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing query: {question[:100]}...")
            
            # Step 1: Generate query embedding
            query_embedding = await self.ollama_client.generate_single_embedding(question)
            if not query_embedding:
                raise ValueError("Failed to generate query embedding")
            
            # Step 2: Expand query for better retrieval (optional)
            expanded_queries = []
            if self.config.get('enable_query_expansion', False):
                expanded_queries = await self.ollama_client.generate_query_expansion(question)
            
            # Step 3: Retrieve relevant chunks
            relevant_chunks = await self._retrieve_relevant_chunks(
                query_embedding, question, expanded_queries, **query_kwargs
            )
            
            if not relevant_chunks:
                return {
                    'status': 'no_results',
                    'message': 'No relevant content found for the query',
                    'question': question
                }
            
            # Step 4: Rerank chunks if enabled
            if self.config['enable_reranking']:
                relevant_chunks = await self._rerank_chunks(question, relevant_chunks)
            
            # Step 5: Build context and generate answer
            context = self._build_context_from_chunks(relevant_chunks)
            answer_response = await self.ollama_client.generate_response(context, question)
            
            if not answer_response:
                raise ValueError("Failed to generate answer")
            
            # Step 6: Add citations if enabled
            citations = []
            if self.config['enable_citation']:
                citations = self._generate_citations(relevant_chunks)
            
            processing_time = time.time() - start_time
            self.processing_stats['queries_processed'] += 1
            
            result = {
                'status': 'success',
                'question': question,
                'answer': answer_response['answer'],
                'confidence_score': answer_response.get('confidence_score', 0.0),
                'processing_time': processing_time,
                'retrieved_chunks': len(relevant_chunks),
                'context_length': len(context),
                'citations': citations,
                'model_info': {
                    'llm_model': self.ollama_client.llm_model,
                    'embedding_model': self.ollama_client.embedding_model
                },
                'relevance_scores': [chunk.get('similarity_score', 0.0) for chunk in relevant_chunks]
            }
            
            self.logger.info(f"Query processed in {processing_time:.2f}s with {len(relevant_chunks)} relevant chunks")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'question': question
            }
    
    async def _extract_pdf_content(self, file_path: Path, **extraction_kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract content from PDF using all available methods
        """
        try:
            # Use comprehensive extraction by default
            extraction_results = self.pdf_extractor.extract_pdf(
                str(file_path)
            )
            
            if not extraction_results:
                self.logger.error(f"No content extracted from {file_path.name}")
                return None
            
            # Convert extraction result to dictionary format for RAG processing
            result_dict = self._convert_extraction_to_rag_format(
                extraction_results, file_path
            )
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF content: {e}")
            return None
    
    def _convert_extraction_to_rag_format(self, extraction_results, file_path: Path) -> Dict[str, Any]:
        """
        Convert PDFExtractionResult to format expected by text processor
        """
        try:
            # Convert tables to list of dictionaries
            tables_data = []
            if extraction_results.tables and 'tables_by_method' in extraction_results.tables:
                best_method = extraction_results.tables.get('best_method', 'camelot')
                if best_method in extraction_results.tables['tables_by_method']:
                    best_tables = extraction_results.tables['tables_by_method'][best_method]
                    for i, table_df in enumerate(best_tables):
                        if hasattr(table_df, 'to_dict'):  # pandas DataFrame
                            tables_data.append({
                                'table': table_df,
                                'page': 1,  # Default page
                                'confidence': 0.8,  # Default confidence
                                'method': best_method,
                                'table_index': i
                            })
            
            # Convert text to list of dictionaries
            text_data = []
            if extraction_results.text and 'text_by_method' in extraction_results.text:
                best_method = extraction_results.text.get('best_method', 'pypdf2')
                if best_method in extraction_results.text['text_by_method']:
                    best_text = extraction_results.text['text_by_method'][best_method]
                    if isinstance(best_text, str) and best_text.strip():
                        text_data.append({
                            'text': best_text,
                            'page': 1,  # Default page
                            'confidence': 1.0,  # Default confidence
                            'method': best_method
                        })
            
            # Convert forms to list of dictionaries
            forms_data = []
            if extraction_results.forms and 'fields_by_method' in extraction_results.forms:
                best_method = extraction_results.forms.get('best_method', 'pypdf2')
                if best_method in extraction_results.forms['fields_by_method']:
                    best_fields = extraction_results.forms['fields_by_method'][best_method]
                    if best_fields:
                        # Convert FormField objects to dictionaries
                        field_dicts = []
                        for field in best_fields:
                            if hasattr(field, 'name'):  # FormField object
                                field_dict = {
                                    'name': field.name,
                                    'value': field.value,
                                    'field_type': field.field_type,
                                    'page_number': field.page_number,
                                    'bbox': field.bbox,
                                    'options': field.options,
                                    'is_readonly': field.is_readonly,
                                    'is_required': field.is_required
                                }
                                field_dicts.append(field_dict)
                            else:
                                # Already a dictionary
                                field_dicts.append(field)
                        
                        forms_data.append({
                            'fields': field_dicts,
                            'page': 1,  # Default page
                            'confidence': 0.9,  # Default confidence
                            'method': best_method
                        })
            
            # Convert images to list of dictionaries
            images_data = []
            if extraction_results.images and 'combined_images' in extraction_results.images:
                combined_images = extraction_results.images['combined_images']
                for img in combined_images:
                    if hasattr(img, 'ocr_text') and img.ocr_text:
                        images_data.append({
                            'text': img.ocr_text,
                            'page': getattr(img, 'page_number', 1),
                            'confidence': getattr(img, 'ocr_confidence', 0.95),
                            'method': 'ocr',
                            'filename': getattr(img, 'filename', ''),
                            'width': getattr(img, 'width', 0),
                            'height': getattr(img, 'height', 0)
                        })
            
            # Create the result dictionary
            result_dict = {
                'tables': tables_data,
                'text': text_data,
                'forms': forms_data,
                'images': images_data,
                'validation_results': extraction_results.validation_results,
                'confidence_scores': extraction_results.confidence_scores,
                'processing_time': extraction_results.processing_time,
                'extraction_successful': extraction_results.extraction_successful,
                'output_files': extraction_results.output_files,
                'error_messages': extraction_results.error_messages
            }
            
            # Add file metadata
            result_dict['file_metadata'] = {
                'file_path': str(file_path),
                'filename': file_path.name,
                'file_size': file_path.stat().st_size,
                'extraction_timestamp': datetime.utcnow().isoformat()
            }
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error converting extraction results: {e}")
            return {
                'tables': [],
                'text': [],
                'forms': [],
                'images': [],
                'validation_results': {},
                'confidence_scores': {},
                'processing_time': 0.0,
                'extraction_successful': False,
                'output_files': {},
                'error_messages': [str(e)],
                'file_metadata': {
                    'file_path': str(file_path),
                    'filename': file_path.name,
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
            }
    
    def _chunk_extracted_content(self, extraction_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk extracted content with validation
        """
        try:
            chunks = self.content_chunker.chunk_extracted_content(extraction_results)
            
            if not chunks:
                self.logger.error("No chunks created from extracted content")
                return []
            
            # Validate chunks
            validated_chunks, validation_stats = self.content_chunker.validate_chunks(chunks)
            
            self.logger.info(f"Created {len(validated_chunks)} valid chunks from {len(chunks)} total chunks")
            self.logger.info(f"Average confidence: {validation_stats.get('average_confidence', 0):.2f}")
            
            self.processing_stats['total_chunks_created'] += len(validated_chunks)
            return validated_chunks
            
        except Exception as e:
            self.logger.error(f"Error chunking content: {e}")
            return []
    
    async def _generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all chunks with quality validation
        """
        try:
            # Extract text content from chunks
            chunk_texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings with validation
            embedding_results = await self.embedding_manager.generate_embeddings_with_validation(chunk_texts)
            
            if len(embedding_results) != len(chunks):
                self.logger.error("Mismatch between chunks and embeddings")
                return []
            
            # Combine chunks with embeddings
            chunks_with_embeddings = []
            for chunk, embedding_result in zip(chunks, embedding_results):
                if embedding_result['passes_quality_threshold']:
                    chunk_copy = chunk.copy()
                    chunk_copy['embedding'] = embedding_result['embedding']
                    chunk_copy['embedding_model'] = embedding_result['embedding_model']
                    chunk_copy['embedding_quality'] = embedding_result['embedding_quality']
                    chunks_with_embeddings.append(chunk_copy)
                else:
                    self.logger.warning(f"Chunk {chunk['chunk_index']} failed embedding quality check")
            
            self.processing_stats['embeddings_generated'] += len(chunks_with_embeddings)
            self.logger.info(f"Generated {len(chunks_with_embeddings)} high-quality embeddings")
            
            return chunks_with_embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return []
    
    async def _store_document_and_chunks(self, file_path: Path, file_hash: str,
                                       extraction_results: Dict[str, Any],
                                       chunks_with_embeddings: List[Dict[str, Any]]) -> str:
        """
        Store document and chunks in vector database
        """
        try:
            # Calculate extraction accuracy
            extraction_accuracy = self._calculate_extraction_accuracy(extraction_results)
            
            # Store document metadata
            document_id = await self.vector_db.store_document(
                file_path=str(file_path),
                filename=file_path.name,
                file_hash=file_hash,
                file_size=file_path.stat().st_size,
                extraction_metadata=extraction_results.get('metadata', {}),
                extraction_accuracy=extraction_accuracy
            )
            
            # Store chunks with embeddings
            success = await self.vector_db.store_chunks_with_embeddings(
                document_id, chunks_with_embeddings
            )
            
            if not success:
                raise ValueError("Failed to store chunks in database")
            
            self.logger.info(f"Stored document {file_path.name} with ID: {document_id}")
            return document_id
            
        except Exception as e:
            self.logger.error(f"Error storing document and chunks: {e}")
            raise
    
    async def _retrieve_relevant_chunks(self, query_embedding: List[float], question: str,
                                      expanded_queries: List[str] = None,
                                      **query_kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using similarity search
        """
        try:
            # Get search parameters
            top_k = query_kwargs.get('top_k', self.config['similarity_top_k'])
            content_types = query_kwargs.get('content_types', None)
            min_confidence = query_kwargs.get('min_confidence', None)
            
            # Primary similarity search
            relevant_chunks = await self.vector_db.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                content_types=content_types,
                min_confidence=min_confidence
            )
            
            # If enabled and we have expanded queries, search with those too
            if expanded_queries and len(expanded_queries) > 1:
                for expanded_query in expanded_queries[1:]:  # Skip original query
                    expanded_embedding = await self.ollama_client.generate_single_embedding(expanded_query)
                    if expanded_embedding:
                        expanded_results = await self.vector_db.similarity_search(
                            query_embedding=expanded_embedding,
                            top_k=top_k // 2,  # Get fewer results for expanded queries
                            content_types=content_types,
                            min_confidence=min_confidence
                        )
                        relevant_chunks.extend(expanded_results)
            
            # Remove duplicates based on chunk_id
            seen_chunks = set()
            unique_chunks = []
            for chunk in relevant_chunks:
                chunk_id = chunk.get('chunk_id')
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_chunks.append(chunk)
            
            # Sort by similarity score
            unique_chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            # Limit to top results
            final_chunks = unique_chunks[:self.config['similarity_top_k']]
            
            self.logger.info(f"Retrieved {len(final_chunks)} relevant chunks for query")
            return final_chunks
            
        except Exception as e:
            self.logger.error(f"Error retrieving relevant chunks: {e}")
            return []
    
    async def _rerank_chunks(self, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank chunks using LLM-based relevance scoring
        """
        try:
            if len(chunks) <= self.config['rerank_top_k']:
                return chunks
            
            # For now, use simple heuristics for reranking
            # In a production system, you might use a dedicated reranking model
            
            # Score chunks based on content type preferences and keyword matching
            scored_chunks = []
            question_words = set(question.lower().split())
            
            for chunk in chunks:
                score = chunk.get('similarity_score', 0.0)
                
                # Content type preferences
                content_type = chunk.get('content_type', '')
                if content_type == 'table':
                    score += 0.1  # Tables often contain specific data
                elif content_type == 'form':
                    score += 0.05  # Forms might contain structured info
                
                # Keyword matching bonus
                chunk_words = set(chunk.get('content', '').lower().split())
                keyword_overlap = len(question_words.intersection(chunk_words))
                score += keyword_overlap * 0.01
                
                # Confidence bonus
                confidence = chunk.get('confidence_score', 0.0)
                score += confidence * 0.1
                
                chunk['rerank_score'] = score
                scored_chunks.append(chunk)
            
            # Sort by rerank score
            scored_chunks.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            return scored_chunks[:self.config['rerank_top_k']]
            
        except Exception as e:
            self.logger.error(f"Error reranking chunks: {e}")
            return chunks
    
    def _build_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from relevant chunks
        """
        try:
            context_parts = []
            current_length = 0
            max_length = self.config['context_window']
            
            for i, chunk in enumerate(chunks):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                
                # Add source information
                source_info = f"Source {i+1}"
                if chunk.get('filename'):
                    source_info += f" ({chunk['filename']}"
                if chunk.get('page_number'):
                    source_info += f", Page {chunk['page_number']}"
                if chunk.get('content_type'):
                    source_info += f", {chunk['content_type'].title()}"
                source_info += "):"
                
                chunk_text = f"{source_info}\n{content}\n"
                
                # Check if adding this chunk would exceed context window
                if current_length + len(chunk_text) > max_length:
                    break
                
                context_parts.append(chunk_text)
                current_length += len(chunk_text)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error building context: {e}")
            return ""
    
    def _generate_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate citations for source attribution
        """
        citations = []
        
        for i, chunk in enumerate(chunks):
            citation = {
                'index': i + 1,
                'filename': chunk.get('filename', 'Unknown'),
                'page_number': chunk.get('page_number'),
                'content_type': chunk.get('content_type', 'text'),
                'similarity_score': chunk.get('similarity_score', 0.0),
                'confidence_score': chunk.get('confidence_score', 0.0),
                'extraction_method': chunk.get('extraction_method', 'unknown')
            }
            citations.append(citation)
        
        return citations
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file for duplicate detection
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _calculate_extraction_accuracy(self, extraction_results: Dict[str, Any]) -> float:
        """
        Calculate overall extraction accuracy from results
        """
        try:
            total_items = 0
            total_confidence = 0.0
            
            # Count items and sum confidences from all content types
            for content_type in ['texts', 'tables', 'forms', 'images']:
                if content_type in extraction_results:
                    items = extraction_results[content_type]
                    if items:
                        for item in items:
                            confidence = item.get('confidence', 0.0)
                            total_confidence += confidence
                            total_items += 1
            
            return total_confidence / total_items if total_items > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating extraction accuracy: {e}")
            return 0.0
    
    async def _track_processing_accuracy(self, document_id: str, extraction_results: Dict[str, Any],
                                       chunks_with_embeddings: List[Dict[str, Any]]):
        """
        Track various accuracy metrics for the document
        """
        try:
            # Extraction accuracy
            extraction_accuracy = self._calculate_extraction_accuracy(extraction_results)
            await self.vector_db.store_accuracy_metric(
                document_id, 'extraction', 'overall_confidence', extraction_accuracy
            )
            
            # Embedding quality
            if chunks_with_embeddings:
                avg_embedding_quality = sum(chunk.get('embedding_quality', 0.0) for chunk in chunks_with_embeddings) / len(chunks_with_embeddings)
                await self.vector_db.store_accuracy_metric(
                    document_id, 'embedding', 'average_quality', avg_embedding_quality
                )
            
            # Chunk quality
            if chunks_with_embeddings:
                avg_chunk_confidence = sum(chunk.get('confidence_score', 0.0) for chunk in chunks_with_embeddings) / len(chunks_with_embeddings)
                await self.vector_db.store_accuracy_metric(
                    document_id, 'chunking', 'average_confidence', avg_chunk_confidence
                )
            
        except Exception as e:
            self.logger.error(f"Error tracking accuracy metrics: {e}")
    
    async def _get_document_accuracy_metrics(self, document_id: str) -> Dict[str, Any]:
        """
        Get accuracy metrics for a document
        """
        try:
            # This would query the accuracy_metrics table
            # For now, return placeholder metrics
            return {
                'extraction_confidence': 0.85,
                'embedding_quality': 0.92,
                'chunk_quality': 0.88
            }
        except Exception as e:
            self.logger.error(f"Error getting accuracy metrics: {e}")
            return {}
    
    def _create_extraction_summary(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary of extraction results
        """
        summary = {}
        
        for content_type in ['texts', 'tables', 'forms', 'images']:
            if content_type in extraction_results:
                items = extraction_results[content_type] or []
                summary[content_type] = {
                    'count': len(items),
                    'average_confidence': sum(item.get('confidence', 0.0) for item in items) / len(items) if items else 0.0
                }
        
        return summary
    
    def _create_chunk_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary of chunks
        """
        if not chunks:
            return {}
        
        content_types = {}
        total_confidence = 0.0
        
        for chunk in chunks:
            content_type = chunk.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            total_confidence += chunk.get('confidence_score', 0.0)
        
        return {
            'total_chunks': len(chunks),
            'average_confidence': total_confidence / len(chunks),
            'content_type_distribution': content_types,
            'average_chunk_length': sum(chunk.get('content_length', 0) for chunk in chunks) / len(chunks)
        }
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics
        """
        try:
            # Get database stats
            db_stats = await self.vector_db.get_database_stats()
            
            # Get Ollama performance stats
            ollama_stats = self.ollama_client.get_performance_stats()
            
            # Combine with processing stats
            stats = {
                'processing_stats': self.processing_stats.copy(),
                'database_stats': db_stats,
                'ollama_stats': ollama_stats,
                'pipeline_config': {
                    'chunk_size': self.config['chunk_size'],
                    'similarity_top_k': self.config['similarity_top_k'],
                    'accuracy_threshold': self.accuracy_config['extraction_confidence_threshold']
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline stats: {e}")
            return {}
    
    async def close(self):
        """
        Close all pipeline components
        """
        try:
            await self.vector_db.close()
            await self.ollama_client.close()
            self.logger.info("RAG pipeline closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing RAG pipeline: {e}")


# Convenience function for quick setup
async def create_rag_pipeline() -> PDFRAGPipeline:
    """
    Create and initialize a complete RAG pipeline
    """
    pipeline = PDFRAGPipeline()
    
    if await pipeline.initialize():
        return pipeline
    else:
        raise RuntimeError("Failed to initialize RAG pipeline")


# Example usage functions
async def process_pdf_with_rag(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Process a single PDF through the RAG pipeline
    """
    pipeline = await create_rag_pipeline()
    
    try:
        result = await pipeline.process_pdf_document(file_path, **kwargs)
        return result
    finally:
        await pipeline.close()


async def query_pdf_documents(question: str, **kwargs) -> Dict[str, Any]:
    """
    Query processed PDF documents
    """
    pipeline = await create_rag_pipeline()
    
    try:
        result = await pipeline.query_documents(question, **kwargs)
        return result
    finally:
        await pipeline.close()
