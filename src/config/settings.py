"""
Configuration settings for PDF extraction
"""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "pdf"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Extraction settings
TABLE_EXTRACTION_METHODS = {
    'camelot': {
        'enabled': True,  # Re-enabled after fixing Ghostscript dependency
        'flavors': ['lattice', 'stream'],
        # Use default tolerance settings for maximum table detection accuracy
        # 'edge_tol': 500,    # Commented out to use defaults
        # 'row_tol': 10,      # Commented out to use defaults  
        # 'column_tol': 0     # Commented out to use defaults
    },
    'tabula': {
        'enabled': True,
        'lattice': True,
        'stream': True,
        'multiple_tables': True,
        'pandas_options': {'header': 0}
    },
    'pdfplumber': {
        'enabled': True,
        'table_settings': {
            'vertical_strategy': 'lines',
            'horizontal_strategy': 'lines',
            'intersection_tolerance': 3
        }
    },
    'pymupdf': {
        'enabled': True,
        'find_tables': True
    }
}

# Text extraction settings
TEXT_EXTRACTION_SETTINGS = {
    'preserve_layout': True,
    'extract_metadata': True,
    'extract_annotations': True,
    'word_margin': 0.1,
    'char_margin': 2.0,
    'line_margin': 0.5,
    'boxes_flow': 0.5
}

# Image extraction settings
IMAGE_EXTRACTION_SETTINGS = {
    'extract_images': True,
    'min_width': 50,
    'min_height': 50,
    'supported_formats': ['PNG', 'JPEG', 'JPG', 'TIFF', 'BMP'],
    # OCR settings
    'enable_ocr': True,
    'tesseract_enabled': True,
    'easyocr_enabled': True,
    'ocr_min_confidence': 0.3,
    'ocr_min_text_length': 3,
    'image_preprocessing': True
}

# Form extraction settings
FORM_EXTRACTION_SETTINGS = {
    'extract_form_fields': True,
    'extract_annotations': True,
    'extract_widgets': True
}

# Output settings
OUTPUT_FORMATS = {
    'excel': True,
    'csv': True,
    'json': True,
    'html': True
}

# Logging settings
LOG_SETTINGS = {
    'level': 'DEBUG',
    'format': '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
    'rotation': '10 MB',
    'retention': '1 week',
    'compression': 'zip'
}

# Accuracy validation settings
VALIDATION_SETTINGS = {
    'enable_comparison': True,
    'confidence_threshold': 0.8,
    'generate_diff_reports': True,
    'visual_validation': True
}
