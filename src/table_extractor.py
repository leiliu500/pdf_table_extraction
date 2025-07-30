"""
Advanced table extraction module with multiple methods and accuracy validation
"""
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass

# PDF processing libraries
import os
try:
    # Set Ghostscript paths for camelot
    gs_bin_path = "/usr/local/Cellar/ghostscript/10.05.1/bin"
    gs_lib_path = "/usr/local/Cellar/ghostscript/10.05.1/lib"
    
    # Add Ghostscript binary to PATH
    if os.path.exists(gs_bin_path):
        os.environ['PATH'] = f"{gs_bin_path}:{os.environ.get('PATH', '')}"
    
    # Add Ghostscript library to library path (critical for camelot)
    if os.path.exists(gs_lib_path):
        current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
        os.environ['DYLD_LIBRARY_PATH'] = f"{gs_lib_path}:{current_dyld}" if current_dyld else gs_lib_path
    
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

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

from .config.settings import TABLE_EXTRACTION_METHODS
from .utils.logger import get_logger
from .utils.validation import get_validator

logger = get_logger("table_extractor")


@dataclass
class TableExtractionResult:
    """Result of table extraction from a single method"""
    method: str
    tables: List[pd.DataFrame]
    confidence_scores: List[float]
    processing_time: float
    page_numbers: List[int]
    extraction_successful: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class CamelotTableExtractor:
    """Table extraction using Camelot library"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = CAMELOT_AVAILABLE and settings.get('enabled', False)
        
        if not CAMELOT_AVAILABLE and settings.get('enabled', False):
            logger.warning("Camelot library not available, disabling Camelot extraction")
    
    def extract_tables(self, pdf_path: str, pages: str = 'all') -> TableExtractionResult:
        """Extract tables using Camelot"""
        if not self.enabled:
            return TableExtractionResult(
                method="camelot",
                tables=[],
                confidence_scores=[],
                processing_time=0.0,
                page_numbers=[],
                extraction_successful=False,
                error_message="Camelot not available or disabled"
            )
        
        start_time = time.time()
        logger.info(f"Starting Camelot table extraction", 
                   context={'pdf': pdf_path, 'pages': pages})
        
        all_tables = []
        confidence_scores = []
        page_numbers = []
        
        try:
            # Try lattice method first (for tables with clear borders)
            if 'lattice' in self.settings.get('flavors', []):
                logger.debug("Attempting Camelot lattice extraction")
                tables_lattice = camelot.read_pdf(
                    pdf_path,
                    pages=pages,
                    flavor='lattice'
                )
                
                for table in tables_lattice:
                    if not table.df.empty:
                        all_tables.append(table.df)
                        # Normalize Camelot accuracy (0-100) to confidence score (0-1)
                        confidence_scores.append(table.accuracy / 100.0)
                        page_numbers.append(table.page)
                        
                        logger.trace_table_extraction(
                            "camelot_lattice", table.page, 1, 
                            table.accuracy, {'shape': table.df.shape}
                        )
            
            # Try stream method for tables without clear borders
            if 'stream' in self.settings.get('flavors', []):
                logger.debug("Attempting Camelot stream extraction")
                
                # Build stream parameters, only include custom tolerances if specified
                stream_params = {
                    'pages': pages,
                    'flavor': 'stream'
                }
                
                if 'edge_tol' in self.settings:
                    stream_params['edge_tol'] = self.settings['edge_tol']
                if 'row_tol' in self.settings:
                    stream_params['row_tol'] = self.settings['row_tol']
                if 'column_tol' in self.settings:
                    stream_params['column_tol'] = self.settings['column_tol']
                
                tables_stream = camelot.read_pdf(pdf_path, **stream_params)
                
                for table in tables_stream:
                    if not table.df.empty:
                        # Avoid duplicates by checking if similar table already exists
                        is_duplicate = False
                        for existing_table in all_tables:
                            if self._tables_similar(table.df, existing_table):
                                is_duplicate = True
                                logger.debug(f"Detected duplicate table on page {table.page}")
                                break
                        
                        if not is_duplicate:
                            all_tables.append(table.df)
                            # Normalize Camelot accuracy (0-100) to confidence score (0-1)
                            confidence_scores.append(table.accuracy / 100.0)
                            page_numbers.append(table.page)
                            
                            logger.trace_table_extraction(
                                "camelot_stream", table.page, 1,
                                table.accuracy, {'shape': table.df.shape}
                            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Camelot extraction completed", 
                       context={
                           'tables_found': len(all_tables),
                           'processing_time': processing_time,
                           'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0
                       })
            
            return TableExtractionResult(
                method="camelot",
                tables=all_tables,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                page_numbers=page_numbers,
                extraction_successful=True,
                metadata={
                    'flavors_used': self.settings.get('flavors', []),
                    'total_tables': len(all_tables)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Camelot extraction failed", exception=e, 
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return TableExtractionResult(
                method="camelot",
                tables=[],
                confidence_scores=[],
                processing_time=processing_time,
                page_numbers=[],
                extraction_successful=False,
                error_message=str(e)
            )
    
    def _tables_similar(self, table1: pd.DataFrame, table2: pd.DataFrame, 
                       threshold: float = 0.95) -> bool:
        """Check if two tables are similar (to avoid duplicates)"""
        # First check: different shapes are definitely different tables
        if table1.shape != table2.shape:
            return False
        
        # Second check: if shapes are identical but very small, be more lenient
        if table1.size <= 4:  # 2x2 or smaller tables
            threshold = 0.9
        
        try:
            # Convert to string and compare
            str1 = table1.fillna('').astype(str)
            str2 = table2.fillna('').astype(str)
            similarity = (str1 == str2).sum().sum() / str1.size
            return similarity >= threshold
        except:
            return False


class TabulaTableExtractor:
    """Table extraction using Tabula library"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = TABULA_AVAILABLE and settings.get('enabled', False)
        
        if not TABULA_AVAILABLE and settings.get('enabled', False):
            logger.warning("Tabula library not available, disabling Tabula extraction")
    
    def extract_tables(self, pdf_path: str, pages: Union[str, List[int]] = 'all') -> TableExtractionResult:
        """Extract tables using Tabula"""
        if not self.enabled:
            return TableExtractionResult(
                method="tabula",
                tables=[],
                confidence_scores=[],
                processing_time=0.0,
                page_numbers=[],
                extraction_successful=False,
                error_message="Tabula not available or disabled"
            )
        
        start_time = time.time()
        logger.info(f"Starting Tabula table extraction", 
                   context={'pdf': pdf_path, 'pages': pages})
        
        all_tables = []
        page_numbers = []
        
        try:
            # Extract with lattice method (for tables with lines)
            if self.settings.get('lattice', True):
                logger.debug("Attempting Tabula lattice extraction")
                tables_lattice = tabula.read_pdf(
                    pdf_path,
                    pages=pages,
                    lattice=True,
                    multiple_tables=self.settings.get('multiple_tables', True),
                    pandas_options=self.settings.get('pandas_options', {})
                )
                
                if isinstance(tables_lattice, list):
                    for i, table in enumerate(tables_lattice):
                        if not table.empty:
                            all_tables.append(table)
                            # Estimate page number (Tabula doesn't provide this directly)
                            page_numbers.append(i + 1)
                            
                            logger.trace_table_extraction(
                                "tabula_lattice", i + 1, 1, None,
                                {'shape': table.shape}
                            )
            
            # Extract with stream method (for tables without lines)
            if self.settings.get('stream', True):
                logger.debug("Attempting Tabula stream extraction")
                tables_stream = tabula.read_pdf(
                    pdf_path,
                    pages=pages,
                    stream=True,
                    multiple_tables=self.settings.get('multiple_tables', True),
                    pandas_options=self.settings.get('pandas_options', {})
                )
                
                if isinstance(tables_stream, list):
                    for i, table in enumerate(tables_stream):
                        if not table.empty:
                            # Check for duplicates
                            is_duplicate = False
                            for existing_table in all_tables:
                                if self._tables_similar(table, existing_table):
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                all_tables.append(table)
                                page_numbers.append(i + 1)
                                
                                logger.trace_table_extraction(
                                    "tabula_stream", i + 1, 1, None,
                                    {'shape': table.shape}
                                )
            
            # Generate confidence scores (Tabula doesn't provide them)
            confidence_scores = [self._estimate_confidence(table) for table in all_tables]
            
            processing_time = time.time() - start_time
            
            logger.info(f"Tabula extraction completed", 
                       context={
                           'tables_found': len(all_tables),
                           'processing_time': processing_time,
                           'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0
                       })
            
            return TableExtractionResult(
                method="tabula",
                tables=all_tables,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                page_numbers=page_numbers,
                extraction_successful=True,
                metadata={
                    'methods_used': ['lattice', 'stream'],
                    'total_tables': len(all_tables)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Tabula extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return TableExtractionResult(
                method="tabula",
                tables=[],
                confidence_scores=[],
                processing_time=processing_time,
                page_numbers=[],
                extraction_successful=False,
                error_message=str(e)
            )
    
    def _tables_similar(self, table1: pd.DataFrame, table2: pd.DataFrame, 
                       threshold: float = 0.8) -> bool:
        """Check if two tables are similar"""
        if table1.shape != table2.shape:
            return False
        
        try:
            str1 = table1.fillna('').astype(str)
            str2 = table2.fillna('').astype(str)
            similarity = (str1 == str2).sum().sum() / str1.size
            return similarity >= threshold
        except:
            return False
    
    def _estimate_confidence(self, table: pd.DataFrame) -> float:
        """Estimate confidence score for extracted table"""
        if table.empty:
            return 0.0
        
        # Factors affecting confidence
        factors = []
        
        # Size factor (larger tables generally more reliable)
        size_factor = min(table.size / 50, 1.0)  # Normalize to 0-1
        factors.append(size_factor)
        
        # Data completeness factor
        null_ratio = table.isnull().sum().sum() / table.size
        completeness_factor = 1.0 - null_ratio
        factors.append(completeness_factor)
        
        # Structure factor (uniform column types suggest better extraction)
        try:
            if hasattr(table, 'dtypes'):
                type_consistency = len(set(table.dtypes)) / len(table.columns)
                structure_factor = 1.0 - min(type_consistency, 1.0)
                factors.append(structure_factor)
            else:
                factors.append(0.5)
        except:
            factors.append(0.5)
        
        return np.mean(factors)


class PDFPlumberTableExtractor:
    """Table extraction using pdfplumber library"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PDFPLUMBER_AVAILABLE and settings.get('enabled', False)
        
        if not PDFPLUMBER_AVAILABLE and settings.get('enabled', False):
            logger.warning("pdfplumber library not available, disabling pdfplumber extraction")
    
    def extract_tables(self, pdf_path: str) -> TableExtractionResult:
        """Extract tables using pdfplumber"""
        if not self.enabled:
            return TableExtractionResult(
                method="pdfplumber",
                tables=[],
                confidence_scores=[],
                processing_time=0.0,
                page_numbers=[],
                extraction_successful=False,
                error_message="pdfplumber not available or disabled"
            )
        
        start_time = time.time()
        logger.info(f"Starting pdfplumber table extraction", 
                   context={'pdf': pdf_path})
        
        all_tables = []
        confidence_scores = []
        page_numbers = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.debug(f"Processing page {page_num} with pdfplumber")
                    
                    # Extract tables with custom settings
                    table_settings = self.settings.get('table_settings', {})
                    tables = page.extract_tables(table_settings)
                    
                    for table_data in tables:
                        if table_data and len(table_data) > 1:  # At least header + 1 row
                            # Convert to DataFrame
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            
                            # Clean empty rows and columns
                            df = self._clean_table(df)
                            
                            if not df.empty:
                                all_tables.append(df)
                                confidence_scores.append(self._estimate_confidence(df, page))
                                page_numbers.append(page_num)
                                
                                logger.trace_table_extraction(
                                    "pdfplumber", page_num, 1,
                                    confidence_scores[-1],
                                    {'shape': df.shape}
                                )
            
            processing_time = time.time() - start_time
            
            logger.info(f"pdfplumber extraction completed", 
                       context={
                           'tables_found': len(all_tables),
                           'processing_time': processing_time,
                           'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0
                       })
            
            return TableExtractionResult(
                method="pdfplumber",
                tables=all_tables,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                page_numbers=page_numbers,
                extraction_successful=True,
                metadata={
                    'table_settings': self.settings.get('table_settings', {}),
                    'total_tables': len(all_tables)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"pdfplumber extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return TableExtractionResult(
                method="pdfplumber",
                tables=[],
                confidence_scores=[],
                processing_time=processing_time,
                page_numbers=[],
                extraction_successful=False,
                error_message=str(e)
            )
    
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean extracted table data"""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Strip whitespace from string columns
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
                    # Replace 'None' strings with actual None
                    df[col] = df[col].replace('None', None)
            except (AttributeError, KeyError):
                # Skip columns that cause issues
                continue
        
        return df
    
    def _estimate_confidence(self, df: pd.DataFrame, page) -> float:
        """Estimate confidence score"""
        if df.empty:
            return 0.0
        
        factors = []
        
        # Size and structure factors
        size_factor = min(df.size / 20, 1.0)
        factors.append(size_factor)
        
        # Data quality factor
        null_ratio = df.isnull().sum().sum() / df.size
        factors.append(1.0 - null_ratio)
        
        # Consistency factor (uniform row lengths)
        try:
            row_lengths = [len(str(row).split()) for _, row in df.iterrows()]
            if row_lengths:
                consistency = 1.0 - (np.std(row_lengths) / np.mean(row_lengths))
                factors.append(max(0, consistency))
        except:
            factors.append(0.5)
        
        return np.mean(factors)


class PyMuPDFTableExtractor:
    """Table extraction using PyMuPDF library"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PYMUPDF_AVAILABLE and settings.get('enabled', False)
        
        if not PYMUPDF_AVAILABLE and settings.get('enabled', False):
            logger.warning("PyMuPDF library not available, disabling PyMuPDF extraction")
    
    def extract_tables(self, pdf_path: str) -> TableExtractionResult:
        """Extract tables using PyMuPDF"""
        if not self.enabled:
            return TableExtractionResult(
                method="pymupdf",
                tables=[],
                confidence_scores=[],
                processing_time=0.0,
                page_numbers=[],
                extraction_successful=False,
                error_message="PyMuPDF not available or disabled"
            )
        
        start_time = time.time()
        logger.info(f"Starting PyMuPDF table extraction", 
                   context={'pdf': pdf_path})
        
        all_tables = []
        confidence_scores = []
        page_numbers = []
        
        try:
            doc = None
            try:
                doc = fitz.open(pdf_path)
                total_pages = len(doc)  # Capture page count before processing
                
                for page_num in range(total_pages):
                    page = doc[page_num]
                    logger.debug(f"Processing page {page_num + 1} with PyMuPDF")
                    
                    # Find tables on the page
                    if self.settings.get('find_tables', True):
                        tables = page.find_tables()
                        
                        for table in tables:
                            try:
                                # Extract table data
                                table_data = table.extract()
                                
                                if table_data and len(table_data) > 1:
                                    # Convert to DataFrame
                                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                    df = self._clean_table(df)
                                    
                                    if not df.empty:
                                        all_tables.append(df)
                                        confidence_scores.append(self._estimate_confidence(df, table))
                                        page_numbers.append(page_num + 1)
                                        
                                        logger.trace_table_extraction(
                                            "pymupdf", page_num + 1, 1,
                                            confidence_scores[-1],
                                            {'shape': df.shape, 'bbox': table.bbox}
                                        )
                            except Exception as e:
                                logger.warning(f"Failed to extract table from page {page_num + 1}: {str(e)}")
                
                processing_time = time.time() - start_time
                
                logger.info(f"PyMuPDF extraction completed", 
                           context={
                               'tables_found': len(all_tables),
                               'processing_time': processing_time,
                               'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0
                           })
                
                return TableExtractionResult(
                    method="pymupdf",
                    tables=all_tables,
                    confidence_scores=confidence_scores,
                    processing_time=processing_time,
                    page_numbers=page_numbers,
                    extraction_successful=True,
                    metadata={
                        'total_tables': len(all_tables),
                        'pages_processed': total_pages  # Use captured value instead of len(doc)
                    }
                )
                
            finally:
                # Ensure document is always closed
                if doc is not None:
                    doc.close()
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PyMuPDF extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return TableExtractionResult(
                method="pymupdf",
                tables=[],
                confidence_scores=[],
                processing_time=processing_time,
                page_numbers=[],
                extraction_successful=False,
                error_message=str(e)
            )
    
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean extracted table data"""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean string data
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace(['None', 'nan', ''], None)
            except (AttributeError, KeyError):
                # Skip columns that cause issues
                continue
        
        return df
    
    def _estimate_confidence(self, df: pd.DataFrame, table) -> float:
        """Estimate confidence score"""
        if df.empty:
            return 0.0
        
        factors = []
        
        # Table area factor (larger bounding box suggests more reliable detection)
        bbox = table.bbox
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_factor = min(area / 50000, 1.0)  # Normalize
        factors.append(area_factor)
        
        # Data completeness
        null_ratio = df.isnull().sum().sum() / df.size
        factors.append(1.0 - null_ratio)
        
        # Size factor
        size_factor = min(df.size / 15, 1.0)
        factors.append(size_factor)
        
        return np.mean(factors)


class TableExtractor:
    """Main table extraction class that orchestrates multiple extraction methods"""
    
    def __init__(self, settings: Dict[str, Any] = None):
        self.settings = settings or TABLE_EXTRACTION_METHODS
        self.validator = get_validator()
        
        # Initialize extractors
        self.extractors = {}
        
        if self.settings.get('camelot', {}).get('enabled', False):
            self.extractors['camelot'] = CamelotTableExtractor(self.settings['camelot'])
        
        if self.settings.get('tabula', {}).get('enabled', False):
            self.extractors['tabula'] = TabulaTableExtractor(self.settings['tabula'])
        
        if self.settings.get('pdfplumber', {}).get('enabled', False):
            self.extractors['pdfplumber'] = PDFPlumberTableExtractor(self.settings['pdfplumber'])
        
        if self.settings.get('pymupdf', {}).get('enabled', False):
            self.extractors['pymupdf'] = PyMuPDFTableExtractor(self.settings['pymupdf'])
        
        logger.info(f"Initialized table extractor", 
                   context={'enabled_methods': list(self.extractors.keys())})
    
    def extract_all_tables(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables using all enabled methods and compare results"""
        logger.trace_pdf_processing(pdf_path, "table_extraction_start")
        
        start_time = time.time()
        results = {}
        extraction_results = {}
        
        # Run all extraction methods
        for method_name, extractor in self.extractors.items():
            logger.info(f"Running {method_name} extraction")
            
            try:
                if method_name in ['camelot', 'tabula']:
                    # These methods support page specification
                    result = extractor.extract_tables(pdf_path, pages='all')
                else:
                    result = extractor.extract_tables(pdf_path)
                
                extraction_results[method_name] = result
                results[method_name] = result.tables
                
                logger.info(f"{method_name} extraction completed", 
                           context={
                               'tables_found': len(result.tables),
                               'successful': result.extraction_successful,
                               'processing_time': result.processing_time
                           })
                
            except Exception as e:
                logger.error(f"{method_name} extraction failed", exception=e)
                extraction_results[method_name] = TableExtractionResult(
                    method=method_name,
                    tables=[],
                    confidence_scores=[],
                    processing_time=0.0,
                    page_numbers=[],
                    extraction_successful=False,
                    error_message=str(e)
                )
                results[method_name] = []
        
        # Validate and compare results
        validation_results = self.validator.validate_extraction_results({'tables': results})
        
        # Calculate overall confidence
        successful_extractions = [r for r in extraction_results.values() if r.extraction_successful]
        overall_confidence = self.validator.table_validator.calculate_extraction_confidence([
            {
                'table_count': len(r.tables),
                'has_clear_structure': any(score > 0.7 for score in r.confidence_scores),
                'extraction_successful': r.extraction_successful
            }
            for r in successful_extractions
        ])
        
        total_time = time.time() - start_time
        
        # Determine best method based on confidence and table count
        best_method = self._determine_best_method(extraction_results)
        
        final_results = {
            'tables_by_method': results,
            'extraction_results': extraction_results,
            'validation_results': validation_results,
            'best_method': best_method,
            'overall_confidence': overall_confidence,
            'total_processing_time': total_time,
            'summary': {
                'total_tables_found': sum(len(tables) for tables in results.values()),
                'successful_methods': len(successful_extractions),
                'methods_used': list(self.extractors.keys())
            }
        }
        
        logger.trace_pdf_processing(
            pdf_path, "table_extraction_complete", 
            {'total_tables': final_results['summary']['total_tables_found'],
             'best_method': best_method,
             'confidence': overall_confidence}
        )
        
        return final_results
    
    def _determine_best_method(self, extraction_results: Dict[str, TableExtractionResult]) -> str:
        """Determine the best extraction method based on results"""
        method_scores = {}
        
        for method, result in extraction_results.items():
            if not result.extraction_successful:
                method_scores[method] = 0.0
                continue
            
            # Scoring factors
            table_count_score = min(len(result.tables) / 5, 1.0)  # Normalize to 0-1
            confidence_score = np.mean(result.confidence_scores) if result.confidence_scores else 0.0
            speed_score = max(0, 1.0 - (result.processing_time / 30))  # Penalize slow methods
            
            # Weighted average
            total_score = (
                table_count_score * 0.4 +
                confidence_score * 0.5 +
                speed_score * 0.1
            )
            
            method_scores[method] = total_score
        
        if not method_scores:
            return "none"
        
        best_method = max(method_scores, key=method_scores.get)
        
        logger.info(f"Best extraction method determined", 
                   context={'method': best_method, 'scores': method_scores})
        
        return best_method
    
    def get_best_tables(self, extraction_results: Dict[str, Any]) -> List[pd.DataFrame]:
        """Get the best tables from extraction results"""
        best_method = extraction_results.get('best_method', 'none')
        
        if best_method == 'none' or best_method not in extraction_results['tables_by_method']:
            # Fallback: return tables from the method with most tables
            tables_by_method = extraction_results['tables_by_method']
            if not tables_by_method:
                return []
            
            best_method = max(tables_by_method, key=lambda k: len(tables_by_method[k]))
        
        return extraction_results['tables_by_method'].get(best_method, [])
