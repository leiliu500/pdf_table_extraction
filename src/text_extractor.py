"""
Text extraction module for PDF files
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# PDF processing libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

from .config.settings import TEXT_EXTRACTION_SETTINGS
from .utils.logger import get_logger

logger = get_logger("text_extractor")


@dataclass
class TextExtractionResult:
    """Result of text extraction from a single method"""
    method: str
    text: str
    metadata: Dict[str, Any]
    processing_time: float
    extraction_successful: bool
    error_message: Optional[str] = None
    page_texts: List[str] = None
    confidence_score: float = 0.0


class PDFPlumberTextExtractor:
    """Text extraction using pdfplumber"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PDFPLUMBER_AVAILABLE
        
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available for text extraction")
    
    def extract_text(self, pdf_path: str) -> TextExtractionResult:
        """Extract text using pdfplumber with layout preservation"""
        if not self.enabled:
            return TextExtractionResult(
                method="pdfplumber",
                text="",
                metadata={},
                processing_time=0.0,
                extraction_successful=False,
                error_message="pdfplumber not available"
            )
        
        start_time = time.time()
        logger.info(f"Starting pdfplumber text extraction", context={'pdf': pdf_path})
        
        try:
            all_text = []
            page_texts = []
            metadata = {'pages': [], 'total_chars': 0}
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text with layout preservation
                    if self.settings.get('preserve_layout', True):
                        page_text = page.extract_text(
                            x_tolerance=self.settings.get('word_margin', 0.1),
                            y_tolerance=self.settings.get('line_margin', 0.5)
                        )
                    else:
                        page_text = page.extract_text()
                    
                    if page_text:
                        page_texts.append(page_text)
                        all_text.append(f"\n--- Page {page_num} ---\n{page_text}")
                        
                        # Collect metadata
                        metadata['pages'].append({
                            'page_num': page_num,
                            'char_count': len(page_text),
                            'bbox': page.bbox,
                            'width': page.width,
                            'height': page.height
                        })
                        
                        logger.debug(f"Extracted {len(page_text)} characters from page {page_num}")
                
                # Extract document metadata if available
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    metadata['document_info'] = pdf.metadata
            
            full_text = '\n'.join(all_text)
            metadata['total_chars'] = len(full_text)
            metadata['total_pages'] = len(page_texts)
            
            processing_time = time.time() - start_time
            confidence_score = self._calculate_confidence(full_text, metadata)
            
            logger.info(f"pdfplumber text extraction completed", 
                       context={
                           'total_chars': len(full_text),
                           'pages': len(page_texts),
                           'processing_time': processing_time,
                           'confidence': confidence_score
                       })
            
            return TextExtractionResult(
                method="pdfplumber",
                text=full_text,
                metadata=metadata,
                processing_time=processing_time,
                extraction_successful=True,
                page_texts=page_texts,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"pdfplumber text extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return TextExtractionResult(
                method="pdfplumber",
                text="",
                metadata={},
                processing_time=processing_time,
                extraction_successful=False,
                error_message=str(e)
            )
    
    def _calculate_confidence(self, text: str, metadata: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted text"""
        if not text.strip():
            return 0.0
        
        factors = []
        
        # Length factor (longer text usually indicates better extraction)
        length_factor = min(len(text) / 1000, 1.0)
        factors.append(length_factor)
        
        # Character diversity factor
        unique_chars = len(set(text))
        diversity_factor = min(unique_chars / 50, 1.0)
        factors.append(diversity_factor)
        
        # Page consistency factor
        if metadata.get('pages'):
            page_char_counts = [p['char_count'] for p in metadata['pages']]
            if page_char_counts:
                avg_chars = sum(page_char_counts) / len(page_char_counts)
                consistency = 1.0 - (max(page_char_counts) - min(page_char_counts)) / max(avg_chars, 1)
                factors.append(max(0, consistency))
        
        return sum(factors) / len(factors) if factors else 0.0


class PyMuPDFTextExtractor:
    """Text extraction using PyMuPDF"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PYMUPDF_AVAILABLE
        
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available for text extraction")
    
    def extract_text(self, pdf_path: str) -> TextExtractionResult:
        """Extract text using PyMuPDF"""
        if not self.enabled:
            return TextExtractionResult(
                method="pymupdf",
                text="",
                metadata={},
                processing_time=0.0,
                extraction_successful=False,
                error_message="PyMuPDF not available"
            )
        
        start_time = time.time()
        logger.info(f"Starting PyMuPDF text extraction", context={'pdf': pdf_path})
        
        try:
            doc = fitz.open(pdf_path)
            all_text = []
            page_texts = []
            metadata = {'pages': [], 'annotations': []}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                page_text = page.get_text()
                if page_text:
                    page_texts.append(page_text)
                    all_text.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                    
                    # Collect page metadata
                    metadata['pages'].append({
                        'page_num': page_num + 1,
                        'char_count': len(page_text),
                        'bbox': page.rect,
                        'rotation': page.rotation
                    })
                
                # Extract annotations if enabled
                if self.settings.get('extract_annotations', True):
                    annotations = page.annots()
                    for annot in annotations:
                        annot_dict = annot.info
                        annot_dict['page'] = page_num + 1
                        metadata['annotations'].append(annot_dict)
                
                logger.debug(f"Processed page {page_num + 1} - {len(page_text)} characters")
            
            # Extract document metadata
            doc_metadata = doc.metadata
            metadata['document_info'] = doc_metadata
            metadata['total_pages'] = len(doc)
            
            doc.close()
            
            full_text = '\n'.join(all_text)
            metadata['total_chars'] = len(full_text)
            
            processing_time = time.time() - start_time
            confidence_score = self._calculate_confidence(full_text, metadata)
            
            logger.info(f"PyMuPDF text extraction completed", 
                       context={
                           'total_chars': len(full_text),
                           'pages': len(page_texts),
                           'annotations': len(metadata['annotations']),
                           'processing_time': processing_time,
                           'confidence': confidence_score
                       })
            
            return TextExtractionResult(
                method="pymupdf",
                text=full_text,
                metadata=metadata,
                processing_time=processing_time,
                extraction_successful=True,
                page_texts=page_texts,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PyMuPDF text extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return TextExtractionResult(
                method="pymupdf",
                text="",
                metadata={},
                processing_time=processing_time,
                extraction_successful=False,
                error_message=str(e)
            )
    
    def _calculate_confidence(self, text: str, metadata: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted text"""
        if not text.strip():
            return 0.0
        
        factors = []
        
        # Basic factors
        length_factor = min(len(text) / 1000, 1.0)
        factors.append(length_factor)
        
        # Structure factor (presence of page breaks)
        page_breaks = text.count('--- Page')
        structure_factor = min(page_breaks / 10, 1.0)
        factors.append(structure_factor)
        
        # Annotation factor (annotations suggest active document)
        if metadata.get('annotations'):
            annotation_factor = min(len(metadata['annotations']) / 5, 1.0)
            factors.append(annotation_factor)
        
        return sum(factors) / len(factors) if factors else 0.0


class PDFMinerTextExtractor:
    """Text extraction using PDFMiner"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PDFMINER_AVAILABLE
        
        if not PDFMINER_AVAILABLE:
            logger.warning("PDFMiner not available for text extraction")
    
    def extract_text(self, pdf_path: str) -> TextExtractionResult:
        """Extract text using PDFMiner with layout analysis"""
        if not self.enabled:
            return TextExtractionResult(
                method="pdfminer",
                text="",
                metadata={},
                processing_time=0.0,
                extraction_successful=False,
                error_message="PDFMiner not available"
            )
        
        start_time = time.time()
        logger.info(f"Starting PDFMiner text extraction", context={'pdf': pdf_path})
        
        try:
            # Configure layout analysis parameters
            laparams = LAParams(
                word_margin=self.settings.get('word_margin', 0.1),
                char_margin=self.settings.get('char_margin', 2.0),
                line_margin=self.settings.get('line_margin', 0.5),
                boxes_flow=self.settings.get('boxes_flow', 0.5)
            )
            
            # Extract text
            text = pdfminer_extract_text(pdf_path, laparams=laparams)
            
            metadata = {
                'total_chars': len(text),
                'extraction_params': {
                    'word_margin': laparams.word_margin,
                    'char_margin': laparams.char_margin,
                    'line_margin': laparams.line_margin,
                    'boxes_flow': laparams.boxes_flow
                }
            }
            
            processing_time = time.time() - start_time
            confidence_score = self._calculate_confidence(text)
            
            logger.info(f"PDFMiner text extraction completed", 
                       context={
                           'total_chars': len(text),
                           'processing_time': processing_time,
                           'confidence': confidence_score
                       })
            
            return TextExtractionResult(
                method="pdfminer",
                text=text,
                metadata=metadata,
                processing_time=processing_time,
                extraction_successful=True,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PDFMiner text extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return TextExtractionResult(
                method="pdfminer",
                text="",
                metadata={},
                processing_time=processing_time,
                extraction_successful=False,
                error_message=str(e)
            )
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for extracted text"""
        if not text.strip():
            return 0.0
        
        # Simple confidence based on text characteristics
        factors = []
        
        # Length factor
        length_factor = min(len(text) / 1000, 1.0)
        factors.append(length_factor)
        
        # Whitespace ratio (good extraction has reasonable whitespace)
        whitespace_ratio = len([c for c in text if c.isspace()]) / len(text)
        whitespace_factor = 1.0 - abs(whitespace_ratio - 0.15)  # Expect ~15% whitespace
        factors.append(max(0, whitespace_factor))
        
        # Character diversity
        unique_chars = len(set(text))
        diversity_factor = min(unique_chars / 50, 1.0)
        factors.append(diversity_factor)
        
        return sum(factors) / len(factors) if factors else 0.0


class PyPDF2TextExtractor:
    """Text extraction using PyPDF2"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PYPDF2_AVAILABLE
        
        if not PYPDF2_AVAILABLE:
            logger.warning("PyPDF2 not available for text extraction")
    
    def extract_text(self, pdf_path: str) -> TextExtractionResult:
        """Extract text using PyPDF2"""
        if not self.enabled:
            return TextExtractionResult(
                method="pypdf2",
                text="",
                metadata={},
                processing_time=0.0,
                extraction_successful=False,
                error_message="PyPDF2 not available"
            )
        
        start_time = time.time()
        logger.info(f"Starting PyPDF2 text extraction", context={'pdf': pdf_path})
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                all_text = []
                page_texts = []
                metadata = {'pages': []}
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    
                    if page_text:
                        page_texts.append(page_text)
                        all_text.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                        
                        metadata['pages'].append({
                            'page_num': page_num + 1,
                            'char_count': len(page_text)
                        })
                
                # Extract document metadata
                if pdf_reader.metadata:
                    metadata['document_info'] = {
                        key: str(value) for key, value in pdf_reader.metadata.items()
                    }
                
                metadata['total_pages'] = len(pdf_reader.pages)
            
            full_text = '\n'.join(all_text)
            metadata['total_chars'] = len(full_text)
            
            processing_time = time.time() - start_time
            confidence_score = self._calculate_confidence(full_text, metadata)
            
            logger.info(f"PyPDF2 text extraction completed", 
                       context={
                           'total_chars': len(full_text),
                           'pages': len(page_texts),
                           'processing_time': processing_time,
                           'confidence': confidence_score
                       })
            
            return TextExtractionResult(
                method="pypdf2",
                text=full_text,
                metadata=metadata,
                processing_time=processing_time,
                extraction_successful=True,
                page_texts=page_texts,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PyPDF2 text extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return TextExtractionResult(
                method="pypdf2",
                text="",
                metadata={},
                processing_time=processing_time,
                extraction_successful=False,
                error_message=str(e)
            )
    
    def _calculate_confidence(self, text: str, metadata: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted text"""
        if not text.strip():
            return 0.0
        
        factors = []
        
        # Basic length factor
        length_factor = min(len(text) / 1000, 1.0)
        factors.append(length_factor)
        
        # Page distribution factor
        if metadata.get('pages'):
            page_counts = [p['char_count'] for p in metadata['pages'] if p['char_count'] > 0]
            if page_counts:
                avg_chars = sum(page_counts) / len(page_counts)
                consistency = 1.0 - (max(page_counts) - min(page_counts)) / max(avg_chars, 1)
                factors.append(max(0, consistency))
        
        return sum(factors) / len(factors) if factors else 0.0


class TextExtractor:
    """Main text extraction class that orchestrates multiple extraction methods"""
    
    def __init__(self, settings: Dict[str, Any] = None):
        self.settings = settings or TEXT_EXTRACTION_SETTINGS
        
        # Initialize extractors
        self.extractors = {}
        
        if PDFPLUMBER_AVAILABLE:
            self.extractors['pdfplumber'] = PDFPlumberTextExtractor(self.settings)
        
        if PYMUPDF_AVAILABLE:
            self.extractors['pymupdf'] = PyMuPDFTextExtractor(self.settings)
        
        if PDFMINER_AVAILABLE:
            self.extractors['pdfminer'] = PDFMinerTextExtractor(self.settings)
        
        if PYPDF2_AVAILABLE:
            self.extractors['pypdf2'] = PyPDF2TextExtractor(self.settings)
        
        logger.info(f"Initialized text extractor", 
                   context={'enabled_methods': list(self.extractors.keys())})
    
    def extract_all_text(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using all available methods and compare results"""
        logger.trace_pdf_processing(pdf_path, "text_extraction_start")
        
        start_time = time.time()
        results = {}
        extraction_results = {}
        
        # Run all extraction methods
        for method_name, extractor in self.extractors.items():
            logger.info(f"Running {method_name} text extraction")
            
            try:
                result = extractor.extract_text(pdf_path)
                extraction_results[method_name] = result
                results[method_name] = result.text
                
                logger.info(f"{method_name} text extraction completed", 
                           context={
                               'text_length': len(result.text),
                               'successful': result.extraction_successful,
                               'processing_time': result.processing_time,
                               'confidence': result.confidence_score
                           })
                
            except Exception as e:
                logger.error(f"{method_name} text extraction failed", exception=e)
                extraction_results[method_name] = TextExtractionResult(
                    method=method_name,
                    text="",
                    metadata={},
                    processing_time=0.0,
                    extraction_successful=False,
                    error_message=str(e)
                )
                results[method_name] = ""
        
        # Determine best method
        best_method = self._determine_best_method(extraction_results)
        
        total_time = time.time() - start_time
        
        final_results = {
            'text_by_method': results,
            'extraction_results': extraction_results,
            'best_method': best_method,
            'total_processing_time': total_time,
            'summary': {
                'successful_methods': len([r for r in extraction_results.values() if r.extraction_successful]),
                'best_text_length': len(results.get(best_method, '')),
                'methods_used': list(self.extractors.keys())
            }
        }
        
        logger.trace_pdf_processing(
            pdf_path, "text_extraction_complete", 
            {'best_method': best_method,
             'text_length': final_results['summary']['best_text_length']}
        )
        
        return final_results
    
    def _determine_best_method(self, extraction_results: Dict[str, TextExtractionResult]) -> str:
        """Determine the best extraction method based on results"""
        method_scores = {}
        
        for method, result in extraction_results.items():
            if not result.extraction_successful:
                method_scores[method] = 0.0
                continue
            
            # Scoring factors
            length_score = min(len(result.text) / 5000, 1.0)  # Normalize to 0-1
            confidence_score = result.confidence_score
            speed_score = max(0, 1.0 - (result.processing_time / 10))  # Penalize slow methods
            
            # Weighted average
            total_score = (
                length_score * 0.4 +
                confidence_score * 0.5 +
                speed_score * 0.1
            )
            
            method_scores[method] = total_score
        
        if not method_scores:
            return "none"
        
        best_method = max(method_scores, key=method_scores.get)
        
        logger.info(f"Best text extraction method determined", 
                   context={'method': best_method, 'scores': method_scores})
        
        return best_method
    
    def get_best_text(self, extraction_results: Dict[str, Any]) -> str:
        """Get the best text from extraction results"""
        best_method = extraction_results.get('best_method', 'none')
        
        if best_method == 'none' or best_method not in extraction_results['text_by_method']:
            # Fallback: return text from the method with most content
            text_by_method = extraction_results['text_by_method']
            if not text_by_method:
                return ""
            
            best_method = max(text_by_method, key=lambda k: len(text_by_method[k]))
        
        return extraction_results['text_by_method'].get(best_method, "")
