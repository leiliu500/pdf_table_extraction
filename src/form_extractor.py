"""
Form extraction module for PDF files
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from .config.settings import FORM_EXTRACTION_SETTINGS
from .utils.logger import get_logger

logger = get_logger("form_extractor")


@dataclass
class FormField:
    """Represents a form field in a PDF"""
    name: str
    value: Any
    field_type: str
    page_number: int
    bbox: Optional[List[float]] = None
    options: Optional[List[str]] = None
    is_readonly: bool = False
    is_required: bool = False


@dataclass
class FormExtractionResult:
    """Result of form extraction from a single method"""
    method: str
    form_fields: List[FormField]
    metadata: Dict[str, Any]
    processing_time: float
    extraction_successful: bool
    error_message: Optional[str] = None
    confidence_score: float = 0.0


class PyMuPDFFormExtractor:
    """Form extraction using PyMuPDF"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PYMUPDF_AVAILABLE and settings.get('extract_form_fields', True)
        
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available for form extraction")
    
    def extract_forms(self, pdf_path: str) -> FormExtractionResult:
        """Extract form fields using PyMuPDF"""
        if not self.enabled:
            return FormExtractionResult(
                method="pymupdf",
                form_fields=[],
                metadata={},
                processing_time=0.0,
                extraction_successful=False,
                error_message="PyMuPDF not available or form extraction disabled"
            )
        
        start_time = time.time()
        logger.info(f"Starting PyMuPDF form extraction", context={'pdf': pdf_path})
        
        try:
            doc = fitz.open(pdf_path)
            form_fields = []
            metadata = {'pages_with_forms': [], 'total_fields': 0, 'field_types': {}}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get form fields on this page
                if hasattr(page, 'widgets'):
                    widgets = page.widgets()
                    page_field_count = 0
                    
                    for widget in widgets:
                        try:
                            field = self._extract_widget_info(widget, page_num + 1)
                            if field:
                                form_fields.append(field)
                                page_field_count += 1
                                
                                # Update metadata
                                field_type = field.field_type
                                metadata['field_types'][field_type] = metadata['field_types'].get(field_type, 0) + 1
                                
                                logger.debug(f"Extracted form field: {field.name} ({field.field_type}) on page {page_num + 1}")
                        
                        except Exception as e:
                            logger.warning(f"Failed to extract widget on page {page_num + 1}", exception=e)
                    
                    if page_field_count > 0:
                        metadata['pages_with_forms'].append({
                            'page': page_num + 1,
                            'field_count': page_field_count
                        })
                
                # Extract annotations if enabled
                if self.settings.get('extract_annotations', True):
                    annotations = page.annots()
                    for annot in annotations:
                        try:
                            if annot.type[1] == 'Widget':  # Form widget annotation
                                field = self._extract_annotation_info(annot, page_num + 1)
                                if field and not any(f.name == field.name for f in form_fields):
                                    form_fields.append(field)
                        except Exception as e:
                            logger.debug(f"Could not extract annotation on page {page_num + 1}: {e}")
            
            doc.close()
            
            metadata['total_fields'] = len(form_fields)
            processing_time = time.time() - start_time
            confidence_score = self._calculate_confidence(form_fields, metadata)
            
            logger.info(f"PyMuPDF form extraction completed", 
                       context={
                           'total_fields': len(form_fields),
                           'pages_with_forms': len(metadata['pages_with_forms']),
                           'processing_time': processing_time,
                           'confidence': confidence_score
                       })
            
            return FormExtractionResult(
                method="pymupdf",
                form_fields=form_fields,
                metadata=metadata,
                processing_time=processing_time,
                extraction_successful=True,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PyMuPDF form extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return FormExtractionResult(
                method="pymupdf",
                form_fields=[],
                metadata={},
                processing_time=processing_time,
                extraction_successful=False,
                error_message=str(e)
            )
    
    def _extract_widget_info(self, widget, page_num: int) -> Optional[FormField]:
        """Extract information from a form widget"""
        try:
            field_info = widget.field_info
            if not field_info:
                return None
            
            field_name = field_info.get('name', f'field_{widget.rect}')
            field_value = field_info.get('value', '')
            field_type = field_info.get('type', 'unknown')
            
            # Map PyMuPDF field types to standard types
            type_mapping = {
                0: 'unknown',
                1: 'button',
                2: 'text',
                3: 'choice',
                4: 'signature'
            }
            
            field_type_name = type_mapping.get(field_type, f'type_{field_type}')
            
            return FormField(
                name=field_name,
                value=field_value,
                field_type=field_type_name,
                page_number=page_num,
                bbox=list(widget.rect),
                options=field_info.get('choices', []),
                is_readonly=field_info.get('readonly', False),
                is_required=field_info.get('required', False)
            )
        
        except Exception as e:
            logger.debug(f"Failed to extract widget info: {e}")
            return None
    
    def _extract_annotation_info(self, annot, page_num: int) -> Optional[FormField]:
        """Extract information from a form annotation"""
        try:
            info = annot.info
            field_name = info.get('name', f'annotation_{annot.rect}')
            field_value = info.get('content', '')
            
            return FormField(
                name=field_name,
                value=field_value,
                field_type='annotation',
                page_number=page_num,
                bbox=list(annot.rect)
            )
        
        except Exception as e:
            logger.debug(f"Failed to extract annotation info: {e}")
            return None
    
    def _calculate_confidence(self, form_fields: List[FormField], metadata: Dict[str, Any]) -> float:
        """Calculate confidence score for form extraction"""
        if not form_fields:
            return 0.0
        
        factors = []
        
        # Number of fields factor
        field_count_factor = min(len(form_fields) / 10, 1.0)
        factors.append(field_count_factor)
        
        # Field completeness factor (fields with values)
        filled_fields = len([f for f in form_fields if f.value and str(f.value).strip()])
        completeness_factor = filled_fields / len(form_fields) if form_fields else 0
        factors.append(completeness_factor)
        
        # Field type diversity factor
        field_types = len(set(f.field_type for f in form_fields))
        diversity_factor = min(field_types / 4, 1.0)  # Expect up to 4 different types
        factors.append(diversity_factor)
        
        return sum(factors) / len(factors) if factors else 0.0


class PyPDF2FormExtractor:
    """Form extraction using PyPDF2"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PYPDF2_AVAILABLE and settings.get('extract_form_fields', True)
        
        if not PYPDF2_AVAILABLE:
            logger.warning("PyPDF2 not available for form extraction")
    
    def extract_forms(self, pdf_path: str) -> FormExtractionResult:
        """Extract form fields using PyPDF2"""
        if not self.enabled:
            return FormExtractionResult(
                method="pypdf2",
                form_fields=[],
                metadata={},
                processing_time=0.0,
                extraction_successful=False,
                error_message="PyPDF2 not available or form extraction disabled"
            )
        
        start_time = time.time()
        logger.info(f"Starting PyPDF2 form extraction", context={'pdf': pdf_path})
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                form_fields = []
                metadata = {'total_fields': 0, 'field_types': {}}
                
                # Check if the PDF has form fields
                if hasattr(pdf_reader, 'get_form_text_fields'):
                    try:
                        # Get text fields
                        text_fields = pdf_reader.get_form_text_fields()
                        if text_fields:
                            for field_name, field_value in text_fields.items():
                                field = FormField(
                                    name=field_name,
                                    value=field_value or '',
                                    field_type='text',
                                    page_number=1  # PyPDF2 doesn't provide page info easily
                                )
                                form_fields.append(field)
                                metadata['field_types']['text'] = metadata['field_types'].get('text', 0) + 1
                    except Exception as e:
                        logger.debug(f"Could not extract text fields: {e}")
                
                # Try to extract fields from document info
                if hasattr(pdf_reader, 'documentInfo') and pdf_reader.documentInfo:
                    doc_info = pdf_reader.documentInfo
                    for key, value in doc_info.items():
                        if value:
                            field = FormField(
                                name=f"doc_info_{key}",
                                value=str(value),
                                field_type='metadata',
                                page_number=0  # Document level
                            )
                            form_fields.append(field)
                            metadata['field_types']['metadata'] = metadata['field_types'].get('metadata', 0) + 1
                
                # Try alternative method for form fields
                try:
                    if hasattr(pdf_reader, 'get_fields'):
                        fields = pdf_reader.get_fields()
                        if fields:
                            for field_name, field_obj in fields.items():
                                field_value = field_obj.get('/V', '') if isinstance(field_obj, dict) else ''
                                field_type = field_obj.get('/FT', 'unknown') if isinstance(field_obj, dict) else 'unknown'
                                
                                # Clean up field type
                                if hasattr(field_type, 'get_object'):
                                    field_type = field_type.get_object()
                                field_type = str(field_type).replace('/', '') if field_type else 'unknown'
                                
                                field = FormField(
                                    name=field_name,
                                    value=str(field_value) if field_value else '',
                                    field_type=field_type,
                                    page_number=1
                                )
                                
                                # Avoid duplicates
                                if not any(f.name == field.name for f in form_fields):
                                    form_fields.append(field)
                                    metadata['field_types'][field_type] = metadata['field_types'].get(field_type, 0) + 1
                
                except Exception as e:
                    logger.debug(f"Alternative form field extraction failed: {e}")
                
                metadata['total_fields'] = len(form_fields)
                processing_time = time.time() - start_time
                confidence_score = self._calculate_confidence(form_fields)
                
                logger.info(f"PyPDF2 form extraction completed", 
                           context={
                               'total_fields': len(form_fields),
                               'processing_time': processing_time,
                               'confidence': confidence_score
                           })
                
                return FormExtractionResult(
                    method="pypdf2",
                    form_fields=form_fields,
                    metadata=metadata,
                    processing_time=processing_time,
                    extraction_successful=True,
                    confidence_score=confidence_score
                )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PyPDF2 form extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return FormExtractionResult(
                method="pypdf2",
                form_fields=[],
                metadata={},
                processing_time=processing_time,
                extraction_successful=False,
                error_message=str(e)
            )
    
    def _calculate_confidence(self, form_fields: List[FormField]) -> float:
        """Calculate confidence score for form extraction"""
        if not form_fields:
            return 0.0
        
        factors = []
        
        # Basic field count factor
        field_count_factor = min(len(form_fields) / 5, 1.0)
        factors.append(field_count_factor)
        
        # Value completeness factor
        filled_fields = len([f for f in form_fields if f.value and str(f.value).strip()])
        completeness_factor = filled_fields / len(form_fields) if form_fields else 0
        factors.append(completeness_factor)
        
        return sum(factors) / len(factors) if factors else 0.0


class FormExtractor:
    """Main form extraction class that orchestrates multiple extraction methods"""
    
    def __init__(self, settings: Dict[str, Any] = None):
        self.settings = settings or FORM_EXTRACTION_SETTINGS
        
        # Initialize extractors
        self.extractors = {}
        
        if PYMUPDF_AVAILABLE and self.settings.get('extract_form_fields', True):
            self.extractors['pymupdf'] = PyMuPDFFormExtractor(self.settings)
        
        if PYPDF2_AVAILABLE and self.settings.get('extract_form_fields', True):
            self.extractors['pypdf2'] = PyPDF2FormExtractor(self.settings)
        
        logger.info(f"Initialized form extractor", 
                   context={'enabled_methods': list(self.extractors.keys())})
    
    def extract_all_forms(self, pdf_path: str) -> Dict[str, Any]:
        """Extract form fields using all available methods"""
        logger.trace_pdf_processing(pdf_path, "form_extraction_start")
        
        start_time = time.time()
        results = {}
        extraction_results = {}
        
        # Run all extraction methods
        for method_name, extractor in self.extractors.items():
            logger.info(f"Running {method_name} form extraction")
            
            try:
                result = extractor.extract_forms(pdf_path)
                extraction_results[method_name] = result
                results[method_name] = result.form_fields
                
                logger.info(f"{method_name} form extraction completed", 
                           context={
                               'fields_found': len(result.form_fields),
                               'successful': result.extraction_successful,
                               'processing_time': result.processing_time,
                               'confidence': result.confidence_score
                           })
                
            except Exception as e:
                logger.error(f"{method_name} form extraction failed", exception=e)
                extraction_results[method_name] = FormExtractionResult(
                    method=method_name,
                    form_fields=[],
                    metadata={},
                    processing_time=0.0,
                    extraction_successful=False,
                    error_message=str(e)
                )
                results[method_name] = []
        
        # Combine and deduplicate form fields
        combined_fields = self._combine_form_fields(results)
        
        # Determine best method
        best_method = self._determine_best_method(extraction_results)
        
        total_time = time.time() - start_time
        
        final_results = {
            'fields_by_method': results,
            'extraction_results': extraction_results,
            'combined_fields': combined_fields,
            'best_method': best_method,
            'total_processing_time': total_time,
            'summary': {
                'total_fields_combined': len(combined_fields),
                'successful_methods': len([r for r in extraction_results.values() if r.extraction_successful]),
                'methods_used': list(self.extractors.keys()),
                'field_types_found': list(set(f.field_type for f in combined_fields))
            }
        }
        
        logger.trace_pdf_processing(
            pdf_path, "form_extraction_complete", 
            {'total_fields': len(combined_fields),
             'best_method': best_method}
        )
        
        return final_results
    
    def _combine_form_fields(self, results: Dict[str, List[FormField]]) -> List[FormField]:
        """Combine form fields from multiple methods, removing duplicates"""
        combined = []
        seen_fields = set()
        
        # Prioritize methods (PyMuPDF usually more accurate)
        method_priority = ['pymupdf', 'pypdf2']
        
        for method in method_priority:
            if method in results:
                for field in results[method]:
                    field_key = (field.name, field.page_number, field.field_type)
                    if field_key not in seen_fields:
                        combined.append(field)
                        seen_fields.add(field_key)
        
        # Add any remaining fields from other methods
        for method, fields in results.items():
            if method not in method_priority:
                for field in fields:
                    field_key = (field.name, field.page_number, field.field_type)
                    if field_key not in seen_fields:
                        combined.append(field)
                        seen_fields.add(field_key)
        
        logger.info(f"Combined form fields from multiple methods", 
                   context={'total_combined': len(combined), 'methods': list(results.keys())})
        
        return combined
    
    def _determine_best_method(self, extraction_results: Dict[str, FormExtractionResult]) -> str:
        """Determine the best extraction method based on results"""
        method_scores = {}
        
        for method, result in extraction_results.items():
            if not result.extraction_successful:
                method_scores[method] = 0.0
                continue
            
            # Scoring factors
            field_count_score = min(len(result.form_fields) / 10, 1.0)
            confidence_score = result.confidence_score
            speed_score = max(0, 1.0 - (result.processing_time / 5))
            
            # Weighted average
            total_score = (
                field_count_score * 0.5 +
                confidence_score * 0.4 +
                speed_score * 0.1
            )
            
            method_scores[method] = total_score
        
        if not method_scores:
            return "none"
        
        best_method = max(method_scores, key=method_scores.get)
        
        logger.info(f"Best form extraction method determined", 
                   context={'method': best_method, 'scores': method_scores})
        
        return best_method
    
    def get_best_fields(self, extraction_results: Dict[str, Any]) -> List[FormField]:
        """Get the best form fields from extraction results"""
        # Return combined fields if available (usually best)
        if 'combined_fields' in extraction_results:
            return extraction_results['combined_fields']
        
        best_method = extraction_results.get('best_method', 'none')
        
        if best_method == 'none' or best_method not in extraction_results['fields_by_method']:
            # Fallback: return fields from the method with most fields
            fields_by_method = extraction_results['fields_by_method']
            if not fields_by_method:
                return []
            
            best_method = max(fields_by_method, key=lambda k: len(fields_by_method[k]))
        
        return extraction_results['fields_by_method'].get(best_method, [])
