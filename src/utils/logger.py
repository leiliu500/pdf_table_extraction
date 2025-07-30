"""
Advanced logging configuration for PDF extraction with detailed tracing
"""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from datetime import datetime

from ..config.settings import LOGS_DIR, LOG_SETTINGS


class PDFExtractionLogger:
    """
    Custom logger for PDF extraction with structured logging and tracing
    """
    
    def __init__(self, name: str = "pdf_extractor", log_level: str = "DEBUG"):
        self.name = name
        self.log_level = log_level
        self.logger = logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup loguru logger with file and console handlers"""
        # Remove default handler
        self.logger.remove()
        
        # Console handler with colors
        self.logger.add(
            sys.stdout,
            format=LOG_SETTINGS['format'],
            level=self.log_level,
            colorize=True
        )
        
        # File handler for all logs
        log_file = LOGS_DIR / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        self.logger.add(
            log_file,
            format=LOG_SETTINGS['format'],
            level="DEBUG",
            rotation=LOG_SETTINGS['rotation'],
            retention=LOG_SETTINGS['retention'],
            compression=LOG_SETTINGS['compression']
        )
        
        # Separate file for errors
        error_log_file = LOGS_DIR / f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        self.logger.add(
            error_log_file,
            format=LOG_SETTINGS['format'],
            level="ERROR",
            rotation=LOG_SETTINGS['rotation'],
            retention=LOG_SETTINGS['retention']
        )
    
    def trace_pdf_processing(self, pdf_path: str, operation: str, details: dict = None):
        """Log PDF processing operations with detailed tracing"""
        details = details or {}
        self.logger.info(
            f"PDF_TRACE | File: {pdf_path} | Operation: {operation} | Details: {details}"
        )
    
    def trace_table_extraction(self, method: str, page_num: int, table_count: int, 
                              confidence: float = None, details: dict = None):
        """Log table extraction operations"""
        details = details or {}
        msg = f"TABLE_TRACE | Method: {method} | Page: {page_num} | Tables: {table_count}"
        if confidence is not None:
            msg += f" | Confidence: {confidence:.2f}"
        msg += f" | Details: {details}"
        self.logger.info(msg)
    
    def trace_extraction_comparison(self, method1: str, method2: str, 
                                  similarity: float, differences: dict):
        """Log comparison between extraction methods"""
        self.logger.info(
            f"COMPARISON_TRACE | Methods: {method1} vs {method2} | "
            f"Similarity: {similarity:.2f} | Differences: {differences}"
        )
    
    def trace_accuracy_validation(self, validation_type: str, score: float, 
                                 passed: bool, details: dict = None):
        """Log accuracy validation results"""
        details = details or {}
        status = "PASS" if passed else "FAIL"
        self.logger.info(
            f"VALIDATION_TRACE | Type: {validation_type} | Score: {score:.2f} | "
            f"Status: {status} | Details: {details}"
        )
    
    def log_extraction_summary(self, pdf_path: str, results: dict):
        """Log comprehensive extraction summary"""
        self.logger.info(f"EXTRACTION_SUMMARY | File: {pdf_path}")
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.logger.info(f"  {key}.{sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def error(self, message: str, exception: Exception = None, context: dict = None):
        """Log errors with context"""
        context = context or {}
        if exception:
            self.logger.error(f"{message} | Exception: {str(exception)} | Context: {context}")
        else:
            self.logger.error(f"{message} | Context: {context}")
    
    def warning(self, message: str, context: dict = None):
        """Log warnings with context"""
        context = context or {}
        self.logger.warning(f"{message} | Context: {context}")
    
    def info(self, message: str, context: dict = None):
        """Log info with context"""
        context = context or {}
        self.logger.info(f"{message} | Context: {context}")
    
    def debug(self, message: str, context: dict = None):
        """Log debug with context"""
        context = context or {}
        self.logger.debug(f"{message} | Context: {context}")


# Global logger instance
pdf_logger = PDFExtractionLogger()


def get_logger(name: str = "pdf_extractor", log_level: str = "DEBUG") -> PDFExtractionLogger:
    """Get a configured logger instance"""
    return PDFExtractionLogger(name, log_level)
