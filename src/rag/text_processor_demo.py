"""
Text Chunking and Processing Module (Demo Version)
Handles intelligent text chunking for different content types with accuracy tracking
"""

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

from ..config.rag_settings_demo import RAG_CONFIG, ACCURACY_CONFIG


class ContentChunker:
    """
    Intelligent content chunker that handles different content types
    Focus on preserving context and maintaining accuracy
    """
    
    def __init__(self):
        self.config = RAG_CONFIG
        self.accuracy_config = ACCURACY_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Chunking parameters
        self.chunk_size = self.config['chunk_size']
        self.chunk_overlap = self.config['chunk_overlap']
        self.min_chunk_size = self.config['min_chunk_size']
        self.max_chunk_size = self.config['max_chunk_size']
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+')
        self.paragraph_separators = re.compile(r'\n\s*\n')
        
    def chunk_extracted_content(self, extraction_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk all extracted content from PDF with content type awareness
        """
        all_chunks = []
        chunk_index = 0
        
        try:
            # Process text content
            if 'texts' in extraction_results and extraction_results['texts']:
                text_chunks = self._chunk_text_content(
                    extraction_results['texts'], 
                    chunk_index
                )
                all_chunks.extend(text_chunks)
                chunk_index += len(text_chunks)
            
            # Process table content
            if 'tables' in extraction_results and extraction_results['tables']:
                table_chunks = self._chunk_table_content(
                    extraction_results['tables'],
                    chunk_index
                )
                all_chunks.extend(table_chunks)
                chunk_index += len(table_chunks)
            
            # Process form content
            if 'forms' in extraction_results and extraction_results['forms']:
                form_chunks = self._chunk_form_content(
                    extraction_results['forms'],
                    chunk_index
                )
                all_chunks.extend(form_chunks)
                chunk_index += len(form_chunks)
            
            # Process image content (OCR text)
            if 'images' in extraction_results and extraction_results['images']:
                image_chunks = self._chunk_image_content(
                    extraction_results['images'],
                    chunk_index
                )
                all_chunks.extend(image_chunks)
                chunk_index += len(image_chunks)
            
            self.logger.info(f"Created {len(all_chunks)} chunks from extracted content")
            return all_chunks
            
        except Exception as e:
            self.logger.error(f"Error chunking extracted content: {e}")
            return []
    
    def _chunk_text_content(self, text_data: List[Dict[str, Any]], start_index: int) -> List[Dict[str, Any]]:
        """
        Chunk text content with paragraph and sentence awareness
        """
        chunks = []
        chunk_index = start_index
        
        for text_item in text_data:
            try:
                content = text_item.get('text', '')
                page_number = text_item.get('page', 0)
                confidence = text_item.get('confidence', 0.0)
                extraction_method = text_item.get('method', 'unknown')
                
                if not content or len(content.strip()) < self.min_chunk_size:
                    continue
                
                # Split into paragraphs first
                paragraphs = self.paragraph_separators.split(content)
                
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph or len(paragraph) < self.min_chunk_size:
                        continue
                    
                    # If paragraph is small enough, use as single chunk
                    if len(paragraph) <= self.chunk_size:
                        chunk = self._create_chunk(
                            content=paragraph,
                            content_type='text',
                            chunk_index=chunk_index,
                            page_number=page_number,
                            confidence_score=confidence,
                            extraction_method=extraction_method,
                            metadata={
                                'original_text_length': len(content),
                                'paragraph_index': paragraphs.index(paragraph) if paragraph in paragraphs else 0,
                                'chunk_method': 'paragraph_split'
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    else:
                        # Split large paragraphs into sentence-aware chunks
                        paragraph_chunks = self._split_text_by_sentences(
                            paragraph, chunk_index, page_number, confidence, extraction_method
                        )
                        chunks.extend(paragraph_chunks)
                        chunk_index += len(paragraph_chunks)
                
            except Exception as e:
                self.logger.error(f"Error processing text item: {e}")
                continue
        
        return chunks
    
    def _chunk_table_content(self, table_data: List[Dict[str, Any]], start_index: int) -> List[Dict[str, Any]]:
        """
        Chunk table content preserving structure and relationships
        """
        chunks = []
        chunk_index = start_index
        
        for table_item in table_data:
            try:
                # Get table data
                table_df = table_item.get('table')
                page_number = table_item.get('page', 0)
                confidence = table_item.get('confidence', 0.0)
                extraction_method = table_item.get('method', 'unknown')
                
                if table_df is None or table_df.empty:
                    continue
                
                # Convert table to text representations
                table_representations = self._convert_table_to_text(table_df)
                
                for representation_type, content in table_representations.items():
                    if not content or len(content.strip()) < self.min_chunk_size:
                        continue
                    
                    # For tables, we usually want to keep them intact if possible
                    if len(content) <= self.max_chunk_size:
                        chunk = self._create_chunk(
                            content=content,
                            content_type='table',
                            chunk_index=chunk_index,
                            page_number=page_number,
                            confidence_score=confidence,
                            extraction_method=extraction_method,
                            metadata={
                                'table_shape': [len(table_df), len(table_df.columns)],
                                'table_representation': representation_type,
                                'column_names': list(table_df.columns),
                                'row_count': len(table_df),
                                'chunk_method': 'table_intact'
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                
            except Exception as e:
                self.logger.error(f"Error processing table item: {e}")
                continue
        
        return chunks
    
    def _chunk_form_content(self, form_data: List[Dict[str, Any]], start_index: int) -> List[Dict[str, Any]]:
        """
        Chunk form content preserving field relationships
        """
        chunks = []
        chunk_index = start_index
        
        for form_item in form_data:
            try:
                # Get form fields
                fields = form_item.get('fields', [])
                page_number = form_item.get('page', 0)
                confidence = form_item.get('confidence', 0.0)
                extraction_method = form_item.get('method', 'unknown')
                
                if not fields:
                    continue
                
                # Convert field group to text
                content = self._convert_form_fields_to_text(fields)
                
                if not content or len(content.strip()) < self.min_chunk_size:
                    continue
                
                chunk = self._create_chunk(
                    content=content,
                    content_type='form',
                    chunk_index=chunk_index,
                    page_number=page_number,
                    confidence_score=confidence,
                    extraction_method=extraction_method,
                    metadata={
                        'field_count': len(fields),
                        'field_names': [f.get('name', 'unnamed') for f in fields],
                        'chunk_method': 'form_field_group'
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                
            except Exception as e:
                self.logger.error(f"Error processing form item: {e}")
                continue
        
        return chunks
    
    def _chunk_image_content(self, image_data: List[Dict[str, Any]], start_index: int) -> List[Dict[str, Any]]:
        """
        Chunk OCR text from images
        """
        chunks = []
        chunk_index = start_index
        
        for image_item in image_data:
            try:
                # Get OCR results
                ocr_results = image_item.get('ocr_results', {})
                page_number = image_item.get('page', 0)
                image_path = image_item.get('image_path', '')
                
                for method_name, ocr_data in ocr_results.items():
                    if not ocr_data or 'text' not in ocr_data:
                        continue
                    
                    content = ocr_data['text']
                    confidence = ocr_data.get('confidence', 0.0)
                    
                    if not content or len(content.strip()) < self.min_chunk_size:
                        continue
                    
                    chunk = self._create_chunk(
                        content=content,
                        content_type='image_ocr',
                        chunk_index=chunk_index,
                        page_number=page_number,
                        confidence_score=confidence,
                        extraction_method=f"ocr_{method_name}",
                        metadata={
                            'image_path': image_path,
                            'ocr_method': method_name,
                            'image_dimensions': ocr_data.get('image_size', [0, 0]),
                            'word_count': len(content.split()),
                            'chunk_method': 'ocr_intact'
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
            except Exception as e:
                self.logger.error(f"Error processing image item: {e}")
                continue
        
        return chunks
    
    def _split_text_by_sentences(self, text: str, start_index: int, page_number: int, 
                               confidence: float, extraction_method: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks respecting sentence boundaries
        """
        chunks = []
        chunk_index = start_index
        
        # Simple sentence splitting for demo
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Save current chunk
                if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        content=current_chunk,
                        content_type='text',
                        chunk_index=chunk_index,
                        page_number=page_number,
                        confidence_score=confidence,
                        extraction_method=extraction_method,
                        metadata={'chunk_method': 'sentence_split'}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = self._create_chunk(
                content=current_chunk,
                content_type='text',
                chunk_index=chunk_index,
                page_number=page_number,
                confidence_score=confidence,
                extraction_method=extraction_method,
                metadata={'chunk_method': 'sentence_split'}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _convert_table_to_text(self, table_df) -> Dict[str, str]:
        """
        Convert table DataFrame to text representation
        """
        try:
            return {
                'structured': table_df.to_string(index=False),
                'csv': table_df.to_csv(index=False)
            }
        except Exception:
            return {'fallback': str(table_df)}
    
    def _convert_form_fields_to_text(self, fields: List[Dict[str, Any]]) -> str:
        """
        Convert form fields to readable text
        """
        field_texts = []
        
        for field in fields:
            field_name = field.get('name', 'Unknown Field')
            field_value = field.get('value', '')
            field_type = field.get('type', 'text')
            
            if field_value:
                field_texts.append(f"{field_name} ({field_type}): {field_value}")
            else:
                field_texts.append(f"{field_name} ({field_type}): [Empty]")
        
        return "\n".join(field_texts)
    
    def _create_chunk(self, content: str, content_type: str, chunk_index: int,
                     page_number: int, confidence_score: float, extraction_method: str,
                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized chunk object
        """
        # Calculate content hash for deduplication
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        chunk = {
            'chunk_index': chunk_index,
            'content': content,
            'content_type': content_type,
            'content_hash': content_hash,
            'content_length': len(content),
            'word_count': len(content.split()),
            'confidence_score': confidence_score,
            'page_number': page_number,
            'extraction_method': extraction_method,
            'created_at': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        return chunk
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validate chunks and provide quality metrics
        """
        validated_chunks = []
        validation_stats = {
            'total_chunks': len(chunks),
            'valid_chunks': 0,
            'filtered_chunks': 0,
            'average_confidence': 0.0,
            'content_type_distribution': {},
            'quality_issues': []
        }
        
        try:
            confidences = []
            content_types = {}
            
            for chunk in chunks:
                # Basic validation
                if not chunk.get('content') or len(chunk.get('content', '').strip()) < self.min_chunk_size:
                    validation_stats['filtered_chunks'] += 1
                    continue
                
                # Track statistics
                confidence = chunk.get('confidence_score', 0.0)
                confidences.append(confidence)
                content_type = chunk.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                validated_chunks.append(chunk)
                validation_stats['valid_chunks'] += 1
            
            # Calculate statistics
            if confidences:
                validation_stats['average_confidence'] = sum(confidences) / len(confidences)
            validation_stats['content_type_distribution'] = content_types
            
            self.logger.info(f"Validated {validation_stats['valid_chunks']}/{validation_stats['total_chunks']} chunks")
            
        except Exception as e:
            self.logger.error(f"Error validating chunks: {e}")
        
        return validated_chunks, validation_stats
