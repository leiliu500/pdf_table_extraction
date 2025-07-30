"""
Main PDF extraction class that orchestrates all extraction methods
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

from .table_extractor import TableExtractor
from .text_extractor import TextExtractor
from .form_extractor import FormExtractor
from .image_extractor import ImageExtractor
from .utils.logger import get_logger
from .utils.file_utils import get_file_handler
from .utils.validation import get_validator
from .config.settings import (
    TABLE_EXTRACTION_METHODS, TEXT_EXTRACTION_SETTINGS,
    FORM_EXTRACTION_SETTINGS, IMAGE_EXTRACTION_SETTINGS,
    OUTPUT_FORMATS, VALIDATION_SETTINGS
)

logger = get_logger("pdf_extractor")


@dataclass
class PDFExtractionResult:
    """Complete result of PDF extraction"""
    pdf_path: str
    tables: Dict[str, Any]
    text: Dict[str, Any]
    forms: Dict[str, Any]
    images: Dict[str, Any]
    validation_results: Dict[str, Any]
    processing_time: float
    extraction_successful: bool
    output_files: Dict[str, List[str]]
    error_messages: List[str]
    confidence_scores: Dict[str, float]


class PDFExtractor:
    """
    Main PDF extraction class that coordinates all extraction methods
    and provides comprehensive PDF content extraction with accuracy validation
    """
    
    def __init__(self, 
                 input_dir: Union[str, Path] = None,
                 output_dir: Union[str, Path] = None,
                 log_level: str = "INFO",
                 enable_validation: bool = True,
                 custom_settings: Dict[str, Any] = None):
        """
        Initialize PDF extractor
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for extraction results
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_validation: Whether to enable accuracy validation
            custom_settings: Custom extraction settings
        """
        self.input_dir = Path(input_dir) if input_dir else Path.cwd()
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "output"
        self.log_level = log_level
        self.enable_validation = enable_validation
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize settings
        self.settings = {
            'tables': custom_settings.get('tables', TABLE_EXTRACTION_METHODS) if custom_settings else TABLE_EXTRACTION_METHODS,
            'text': custom_settings.get('text', TEXT_EXTRACTION_SETTINGS) if custom_settings else TEXT_EXTRACTION_SETTINGS,
            'forms': custom_settings.get('forms', FORM_EXTRACTION_SETTINGS) if custom_settings else FORM_EXTRACTION_SETTINGS,
            'images': custom_settings.get('images', IMAGE_EXTRACTION_SETTINGS) if custom_settings else IMAGE_EXTRACTION_SETTINGS,
            'output': custom_settings.get('output', OUTPUT_FORMATS) if custom_settings else OUTPUT_FORMATS,
            'validation': custom_settings.get('validation', VALIDATION_SETTINGS) if custom_settings else VALIDATION_SETTINGS
        }
        
        # Initialize extractors
        self.table_extractor = TableExtractor(self.settings['tables'])
        self.text_extractor = TextExtractor(self.settings['text'])
        self.form_extractor = FormExtractor(self.settings['forms'])
        self.image_extractor = ImageExtractor(self.settings['images'])
        
        # Initialize utilities
        self.file_handler = get_file_handler(self.output_dir)
        if self.enable_validation:
            self.validator = get_validator(
                confidence_threshold=self.settings['validation'].get('confidence_threshold', 0.8)
            )
        
        logger.info(f"Initialized PDF extractor", 
                   context={
                       'input_dir': str(self.input_dir),
                       'output_dir': str(self.output_dir),
                       'validation_enabled': self.enable_validation
                   })
    
    def extract_pdf(self, pdf_path: Union[str, Path]) -> PDFExtractionResult:
        """
        Extract all content from a single PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFExtractionResult with all extracted content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.trace_pdf_processing(str(pdf_path), "extraction_start")
        start_time = time.time()
        
        # Create output structure for this PDF
        output_structure = self.file_handler.create_output_structure(pdf_path.name)
        
        error_messages = []
        extraction_results = {}
        
        # Extract tables
        logger.info(f"Starting table extraction for {pdf_path.name}")
        try:
            table_results = self.table_extractor.extract_all_tables(str(pdf_path))
            extraction_results['tables'] = table_results
            
            # Save table results
            if table_results['summary']['total_tables_found'] > 0:
                self._save_table_results(table_results, output_structure['tables'], pdf_path.name)
            
            logger.info(f"Table extraction completed", 
                       context={'tables_found': table_results['summary']['total_tables_found']})
        
        except Exception as e:
            error_messages.append(f"Table extraction failed: {str(e)}")
            logger.error(f"Table extraction failed for {pdf_path.name}", exception=e)
            extraction_results['tables'] = {'summary': {'total_tables_found': 0}}
        
        # Extract text
        logger.info(f"Starting text extraction for {pdf_path.name}")
        try:
            text_results = self.text_extractor.extract_all_text(str(pdf_path))
            extraction_results['text'] = text_results
            
            # Save text results
            if text_results['summary']['best_text_length'] > 0:
                self._save_text_results(text_results, output_structure['text'], pdf_path.name)
            
            logger.info(f"Text extraction completed", 
                       context={'text_length': text_results['summary']['best_text_length']})
        
        except Exception as e:
            error_messages.append(f"Text extraction failed: {str(e)}")
            logger.error(f"Text extraction failed for {pdf_path.name}", exception=e)
            extraction_results['text'] = {'summary': {'best_text_length': 0}}
        
        # Extract forms
        logger.info(f"Starting form extraction for {pdf_path.name}")
        try:
            form_results = self.form_extractor.extract_all_forms(str(pdf_path))
            extraction_results['forms'] = form_results
            
            # Save form results
            if form_results['summary']['total_fields_combined'] > 0:
                self._save_form_results(form_results, output_structure['forms'], pdf_path.name)
            
            logger.info(f"Form extraction completed", 
                       context={'fields_found': form_results['summary']['total_fields_combined']})
        
        except Exception as e:
            error_messages.append(f"Form extraction failed: {str(e)}")
            logger.error(f"Form extraction failed for {pdf_path.name}", exception=e)
            extraction_results['forms'] = {'summary': {'total_fields_combined': 0}}
        
        # Extract images
        logger.info(f"Starting image extraction for {pdf_path.name}")
        try:
            image_results = self.image_extractor.extract_all_images(str(pdf_path), output_structure['images'])
            extraction_results['images'] = image_results
            
            logger.info(f"Image extraction completed", 
                       context={'images_found': image_results['summary']['total_images_combined']})
        
        except Exception as e:
            error_messages.append(f"Image extraction failed: {str(e)}")
            logger.error(f"Image extraction failed for {pdf_path.name}", exception=e)
            extraction_results['images'] = {'summary': {'total_images_combined': 0}}
        
        # Perform validation if enabled
        validation_results = {}
        if self.enable_validation and self.settings['validation'].get('enable_comparison', True):
            logger.info(f"Starting validation for {pdf_path.name}")
            try:
                validation_results = self._perform_validation(extraction_results)
                logger.info(f"Validation completed", 
                           context={'validations_performed': len(validation_results)})
            except Exception as e:
                error_messages.append(f"Validation failed: {str(e)}")
                logger.error(f"Validation failed for {pdf_path.name}", exception=e)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(extraction_results, validation_results)
        
        # Create comprehensive report
        processing_time = time.time() - start_time
        
        # Prepare output files list
        output_files = self._get_output_files(output_structure)
        
        # Save comprehensive report
        report_data = {
            'extraction_results': extraction_results,
            'validation_results': validation_results,
            'confidence_scores': confidence_scores,
            'processing_time': processing_time,
            'error_messages': error_messages,
            'output_files': output_files
        }
        
        report_path = self.file_handler.create_extraction_report(pdf_path.name, report_data, output_structure['reports'])
        
        extraction_successful = len(error_messages) == 0 or any([
            extraction_results.get('tables', {}).get('summary', {}).get('total_tables_found', 0) > 0,
            extraction_results.get('text', {}).get('summary', {}).get('best_text_length', 0) > 0,
            extraction_results.get('forms', {}).get('summary', {}).get('total_fields_combined', 0) > 0,
            extraction_results.get('images', {}).get('summary', {}).get('total_images_combined', 0) > 0
        ])
        
        result = PDFExtractionResult(
            pdf_path=str(pdf_path),
            tables=extraction_results.get('tables', {}),
            text=extraction_results.get('text', {}),
            forms=extraction_results.get('forms', {}),
            images=extraction_results.get('images', {}),
            validation_results=validation_results,
            processing_time=processing_time,
            extraction_successful=extraction_successful,
            output_files=output_files,
            error_messages=error_messages,
            confidence_scores=confidence_scores
        )
        
        logger.trace_pdf_processing(
            str(pdf_path), "extraction_complete",
            {
                'successful': extraction_successful,
                'processing_time': processing_time,
                'tables': extraction_results.get('tables', {}).get('summary', {}).get('total_tables_found', 0),
                'confidence': confidence_scores.get('overall', 0.0)
            }
        )
        
        return result
    
    def extract_all(self, pattern: str = "*.pdf") -> List[PDFExtractionResult]:
        """
        Extract content from all PDF files in the input directory
        
        Args:
            pattern: File pattern to match (default: "*.pdf")
            
        Returns:
            List of PDFExtractionResult for each processed file
        """
        pdf_files = list(self.input_dir.glob(pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir} matching pattern '{pattern}'")
            return []
        
        logger.info(f"Starting batch extraction", 
                   context={'total_files': len(pdf_files), 'pattern': pattern})
        
        results = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}")
            try:
                result = self.extract_pdf(pdf_file)
                results.append(result)
                
                if result.extraction_successful:
                    logger.info(f"Successfully processed {pdf_file.name}")
                else:
                    logger.warning(f"Partial success for {pdf_file.name}", 
                                 context={'errors': result.error_messages})
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}", exception=e)
                # Create error result
                error_result = PDFExtractionResult(
                    pdf_path=str(pdf_file),
                    tables={}, text={}, forms={}, images={},
                    validation_results={},
                    processing_time=0.0,
                    extraction_successful=False,
                    output_files={},
                    error_messages=[str(e)],
                    confidence_scores={}
                )
                results.append(error_result)
        
        logger.info(f"Batch extraction completed", 
                   context={
                       'total_processed': len(results),
                       'successful': len([r for r in results if r.extraction_successful]),
                       'failed': len([r for r in results if not r.extraction_successful])
                   })
        
        return results
    
    def extract_tables_only(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract only tables from a PDF file (optimized for table extraction)
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Table extraction results
        """
        pdf_path = Path(pdf_path)
        logger.trace_pdf_processing(str(pdf_path), "table_only_extraction_start")
        
        try:
            table_results = self.table_extractor.extract_all_tables(str(pdf_path))
            
            # Create output structure and save results
            output_structure = self.file_handler.create_output_structure(pdf_path.name)
            if table_results['summary']['total_tables_found'] > 0:
                self._save_table_results(table_results, output_structure['tables'], pdf_path.name)
            
            logger.trace_pdf_processing(str(pdf_path), "table_only_extraction_complete",
                                      {'tables_found': table_results['summary']['total_tables_found']})
            
            return table_results
        
        except Exception as e:
            logger.error(f"Table-only extraction failed for {pdf_path.name}", exception=e)
            raise
    
    def _save_table_results(self, table_results: Dict[str, Any], output_dir: Path, pdf_name: str):
        """Save table extraction results to files"""
        best_tables = self.table_extractor.get_best_tables(table_results)
        
        if not best_tables:
            logger.warning("No tables to save")
            return
        
        # Save to Excel if enabled
        if self.settings['output'].get('excel', True):
            excel_path = output_dir / f"{pdf_name.replace('.pdf', '')}_tables.xlsx"
            sheet_names = [f"Table_{i+1}" for i in range(len(best_tables))]
            self.file_handler.save_tables_to_excel(best_tables, excel_path, sheet_names)
        
        # Save to CSV if enabled
        if self.settings['output'].get('csv', True):
            self.file_handler.save_tables_to_csv(best_tables, output_dir, "table")
        
        # Save raw data as JSON
        if self.settings['output'].get('json', True):
            json_data = {
                'tables': [table.to_dict() if hasattr(table, 'to_dict') else str(table) for table in best_tables],
                'metadata': table_results.get('summary', {}),
                'extraction_methods': table_results.get('tables_by_method', {}).keys()
            }
            json_path = output_dir / f"{pdf_name.replace('.pdf', '')}_tables.json"
            self.file_handler.save_json(json_data, json_path)
    
    def _save_text_results(self, text_results: Dict[str, Any], output_dir: Path, pdf_name: str):
        """Save text extraction results to files"""
        best_text = self.text_extractor.get_best_text(text_results)
        
        if not best_text.strip():
            logger.warning("No text to save")
            return
        
        # Save as plain text
        text_path = output_dir / f"{pdf_name.replace('.pdf', '')}_text.txt"
        self.file_handler.save_text(best_text, text_path)
        
        # Save as JSON with metadata
        if self.settings['output'].get('json', True):
            json_data = {
                'text': best_text,
                'metadata': text_results.get('summary', {}),
                'extraction_methods': list(text_results.get('text_by_method', {}).keys())
            }
            json_path = output_dir / f"{pdf_name.replace('.pdf', '')}_text.json"
            self.file_handler.save_json(json_data, json_path)
    
    def _save_form_results(self, form_results: Dict[str, Any], output_dir: Path, pdf_name: str):
        """Save form extraction results to files"""
        best_fields = self.form_extractor.get_best_fields(form_results)
        
        if not best_fields:
            logger.warning("No form fields to save")
            return
        
        # Convert form fields to serializable format
        fields_data = []
        for field in best_fields:
            fields_data.append({
                'name': field.name,
                'value': field.value,
                'type': field.field_type,
                'page': field.page_number,
                'bbox': field.bbox,
                'options': field.options,
                'readonly': field.is_readonly,
                'required': field.is_required
            })
        
        # Save as JSON
        json_data = {
            'form_fields': fields_data,
            'metadata': form_results.get('summary', {}),
            'extraction_methods': list(form_results.get('fields_by_method', {}).keys())
        }
        json_path = output_dir / f"{pdf_name.replace('.pdf', '')}_forms.json"
        self.file_handler.save_json(json_data, json_path)
        
        # Save as CSV for easy viewing
        if self.settings['output'].get('csv', True) and fields_data:
            import pandas as pd
            df = pd.DataFrame(fields_data)
            csv_path = output_dir / f"{pdf_name.replace('.pdf', '')}_forms.csv"
            df.to_csv(csv_path, index=False)
    
    def _perform_validation(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform accuracy validation on extraction results"""
        validation_results = {}
        
        # Validate tables
        table_results = extraction_results.get('tables', {})
        if table_results:
            table_validation = self.validator.validate_extraction_results(table_results)
            validation_results['tables'] = table_validation
        
        # Validate text
        text_results = extraction_results.get('text', {})
        if text_results:
            text_validation = self.validator.validate_extraction_results(text_results)
            validation_results['text'] = text_validation
        
        return validation_results
    
    def _calculate_confidence_scores(self, extraction_results: Dict[str, Any], 
                                   validation_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall confidence scores"""
        confidence_scores = {}
        
        # Table confidence
        table_results = extraction_results.get('tables', {})
        if table_results and 'extraction_results' in table_results:
            table_confidences = []
            for method_result in table_results['extraction_results'].values():
                if hasattr(method_result, 'confidence_scores') and method_result.confidence_scores:
                    table_confidences.extend(method_result.confidence_scores)
            
            if table_confidences:
                confidence_scores['tables'] = sum(table_confidences) / len(table_confidences)
            else:
                confidence_scores['tables'] = 0.0
        
        # Text confidence
        text_results = extraction_results.get('text', {})
        if text_results and 'extraction_results' in text_results:
            text_confidences = []
            for method_result in text_results['extraction_results'].values():
                if hasattr(method_result, 'confidence_score'):
                    text_confidences.append(method_result.confidence_score)
            
            if text_confidences:
                confidence_scores['text'] = sum(text_confidences) / len(text_confidences)
            else:
                confidence_scores['text'] = 0.0
        
        # Validation-based confidence
        if validation_results:
            validation_scores = []
            for validation_category in validation_results.values():
                for validation_result in validation_category.values():
                    if hasattr(validation_result, 'score'):
                        validation_scores.append(validation_result.score)
            
            if validation_scores:
                confidence_scores['validation'] = sum(validation_scores) / len(validation_scores)
        
        # Overall confidence
        if confidence_scores:
            confidence_scores['overall'] = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            confidence_scores['overall'] = 0.0
        
        return confidence_scores
    
    def _get_output_files(self, output_structure: Dict[str, Path]) -> Dict[str, List[str]]:
        """Get list of generated output files"""
        output_files = {}
        
        for category, path in output_structure.items():
            if path.exists():
                files = [str(f.relative_to(self.output_dir)) for f in path.rglob('*') if f.is_file()]
                if files:
                    output_files[category] = files
        
        return output_files
