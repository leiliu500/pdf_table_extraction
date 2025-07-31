"""
Text Chunking and Processing Module
Handles intelligent text chunking for different content types with accuracy tracking
"""

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import pandas as pd

from ..config.rag_settings import RAG_CONFIG, ACCURACY_CONFIG


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
                    
                    else:
                        # Split large tables by rows
                        table_chunks = self._split_large_table(
                            content, table_df, chunk_index, page_number, 
                            confidence, extraction_method, representation_type
                        )
                        chunks.extend(table_chunks)
                        chunk_index += len(table_chunks)
                
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
                
                # Group related fields together
                field_groups = self._group_form_fields(fields)
                
                for group_name, group_fields in field_groups.items():
                    # Convert field group to text
                    content = self._convert_form_fields_to_text(group_fields)
                    
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
                            'field_group': group_name,
                            'field_count': len(group_fields),
                            'field_names': [f.get('name', 'unnamed') for f in group_fields],
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
                    
                    # Split OCR text if it's too long
                    if len(content) <= self.chunk_size:
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
                    
                    else:
                        # Split long OCR text
                        ocr_chunks = self._split_text_by_sentences(
                            content, chunk_index, page_number, confidence, f"ocr_{method_name}"
                        )
                        
                        # Add OCR-specific metadata to each chunk
                        for chunk in ocr_chunks:
                            chunk['metadata'].update({
                                'image_path': image_path,
                                'ocr_method': method_name,
                                'image_dimensions': ocr_data.get('image_size', [0, 0])
                            })
                        
                        chunks.extend(ocr_chunks)
                        chunk_index += len(ocr_chunks)
                
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
        
        # Split by sentences
        sentences = self.sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        sentence_count = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
                sentence_count += 1
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
                        metadata={
                            'sentence_count': sentence_count,
                            'chunk_method': 'sentence_split'
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                sentence_count = 1
        
        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = self._create_chunk(
                content=current_chunk,
                content_type='text',
                chunk_index=chunk_index,
                page_number=page_number,
                confidence_score=confidence,
                extraction_method=extraction_method,
                metadata={
                    'sentence_count': sentence_count,
                    'chunk_method': 'sentence_split'
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _convert_table_to_text(self, table_df) -> Dict[str, str]:
        """
        Convert table DataFrame to multiple text representations
        """
        representations = {}
        
        try:
            # CSV-like representation
            representations['csv'] = table_df.to_csv(index=False)
            
            # Structured text representation
            structured_text = []
            for _, row in table_df.iterrows():
                row_text = []
                for col, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        row_text.append(f"{col}: {value}")
                if row_text:
                    structured_text.append(" | ".join(row_text))
            
            representations['structured'] = "\n".join(structured_text)
            
            # Simple concatenation
            all_values = []
            for col in table_df.columns:
                all_values.append(f"Column {col}:")
                for value in table_df[col].dropna():
                    if str(value).strip():
                        all_values.append(str(value))
            
            representations['concatenated'] = " ".join(all_values)
            
        except Exception as e:
            self.logger.error(f"Error converting table to text: {e}")
            representations['fallback'] = str(table_df.to_string(index=False))
        
        return representations
    
    def _split_large_table(self, content: str, table_df, start_index: int, 
                          page_number: int, confidence: float, extraction_method: str,
                          representation_type: str) -> List[Dict[str, Any]]:
        """
        Split large tables into smaller chunks
        """
        chunks = []
        chunk_index = start_index
        
        try:
            # Split table by rows
            rows_per_chunk = max(1, self.chunk_size // (len(table_df.columns) * 20))  # Estimate
            
            for i in range(0, len(table_df), rows_per_chunk):
                chunk_df = table_df.iloc[i:i + rows_per_chunk]
                chunk_content = chunk_df.to_string(index=False)
                
                if len(chunk_content.strip()) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        content=chunk_content,
                        content_type='table',
                        chunk_index=chunk_index,
                        page_number=page_number,
                        confidence_score=confidence,
                        extraction_method=extraction_method,
                        metadata={
                            'table_representation': representation_type,
                            'chunk_rows': [i, min(i + rows_per_chunk, len(table_df))],
                            'total_rows': len(table_df),
                            'column_names': list(table_df.columns),
                            'chunk_method': 'table_row_split'
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
        except Exception as e:
            self.logger.error(f"Error splitting large table: {e}")
        
        return chunks
    
    def _group_form_fields(self, fields: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group form fields by logical relationships
        """
        groups = {'general': []}
        
        for field in fields:
            field_name = field.get('name', '').lower()
            
            # Simple grouping logic - can be enhanced
            if any(keyword in field_name for keyword in ['name', 'first', 'last', 'full']):
                if 'personal_info' not in groups:
                    groups['personal_info'] = []
                groups['personal_info'].append(field)
            elif any(keyword in field_name for keyword in ['address', 'street', 'city', 'state', 'zip']):
                if 'address' not in groups:
                    groups['address'] = []
                groups['address'].append(field)
            elif any(keyword in field_name for keyword in ['phone', 'email', 'contact']):
                if 'contact' not in groups:
                    groups['contact'] = []
                groups['contact'].append(field)
            else:
                groups['general'].append(field)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
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
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text for maintaining context between chunks
        """
        if not text or len(text) <= self.chunk_overlap:
            return text
        
        # Try to get complete sentences for overlap
        sentences = self.sentence_endings.split(text[-self.chunk_overlap:])
        if len(sentences) > 1:
            return sentences[-1].strip()
        
        # Fallback to character-based overlap
        return text[-self.chunk_overlap:].strip()
    
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
                # Validate chunk structure
                if not self._is_valid_chunk(chunk):
                    validation_stats['filtered_chunks'] += 1
                    validation_stats['quality_issues'].append(f"Invalid chunk structure: {chunk.get('chunk_index', 'unknown')}")
                    continue
                
                # Check confidence threshold
                confidence = chunk.get('confidence_score', 0.0)
                if confidence < self.accuracy_config['extraction_confidence_threshold']:
                    validation_stats['filtered_chunks'] += 1
                    validation_stats['quality_issues'].append(f"Low confidence chunk: {chunk.get('chunk_index', 'unknown')} ({confidence:.2f})")
                    continue
                
                # Track statistics
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
    
    def _is_valid_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Validate chunk structure and content
        """
        required_fields = ['content', 'content_type', 'chunk_index', 'confidence_score']
        
        # Check required fields
        if not all(field in chunk for field in required_fields):
            return False
        
        # Check content length
        content = chunk.get('content', '')
        if not content or len(content.strip()) < self.min_chunk_size:
            return False
        
        # Check content type
        valid_types = ['text', 'table', 'form', 'image_ocr']
        if chunk.get('content_type') not in valid_types:
            return False
        
        return True
