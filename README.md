# PDF Table Extraction Tool

A comprehensive Python application for extracting tables, text, forms, and images from PDF files with high accuracy and detailed logging.

## Features

- **Table Extraction**: Multiple extraction methods (Camelot, Tabula, pdfplumber, PyMuPDF)
- **Text Extraction**: Structured text extraction with formatting preservation
- **Form Field Detection**: Extract form fields and their values
- **Image Extraction**: Extract embedded images from PDFs
- **Comprehensive Logging**: Detailed logging and tracing for accuracy verification
- **Multiple Output Formats**: Export to Excel, CSV, JSON
- **Validation**: Built-in accuracy checks and comparison tools

## Quick Start

1. **Setup (one-time):**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Activate environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Test installation:**
   ```bash
   python test_installation.py
   ```

4. **Extract tables from your PDFs:**
   ```bash
   # Single PDF
   python main.py -i pdf/304-Cedar-Street/1.pdf -o output/
   
   # All PDFs in directory
   python main.py -i pdf/304-Cedar-Street/ -o output/
   
   # Tables only (faster)
   python main.py -i pdf/304-Cedar-Street/1.pdf -o output/ --tables-only
   ```

## Command Line Usage

```bash
# Basic extraction
python main.py -i INPUT_PDF_OR_DIR -o OUTPUT_DIR

# Advanced options
python main.py -i pdf/file.pdf -o output/ --log-level DEBUG --no-validation

# Custom configuration
python main.py -i pdf/file.pdf -o output/ --config config_sample.json

# Help
python main.py --help
```

## Python API Usage

```python
from src.pdf_extractor import PDFExtractor

# Initialize extractor
extractor = PDFExtractor(
    input_dir="pdf/304-Cedar-Street",
    output_dir="output",
    log_level="DEBUG"
)

# Extract all content
result = extractor.extract_pdf("pdf/304-Cedar-Street/1.pdf")

# Extract from all PDFs
results = extractor.extract_all()

# Extract only tables
tables = extractor.extract_tables_only("pdf/file.pdf")
```

## Project Structure

```
pdf_table_extraction/
├── src/
│   ├── __init__.py
│   ├── pdf_extractor.py          # Main orchestrator class
│   ├── table_extractor.py        # Table extraction (4 methods)
│   ├── text_extractor.py         # Text extraction (4 methods)
│   ├── form_extractor.py         # Form field extraction
│   ├── image_extractor.py        # Image extraction
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py             # Advanced logging with tracing
│   │   ├── file_utils.py         # File handling utilities
│   │   └── validation.py         # Accuracy validation tools
│   └── config/
│       ├── __init__.py
│       └── settings.py           # Configuration settings
├── output/                       # Extraction results
├── logs/                        # Log files
├── pdf/                         # Input PDF files
│   └── 304-Cedar-Street/        # Your real estate PDFs
│       ├── 0. Coversheet.pdf
│       ├── 1. Client MLS page.pdf
│       └── 2. Property Details.pdf
├── requirements.txt             # Python dependencies
├── setup.sh                    # Automated setup script
├── test_installation.py        # Installation verification
├── main.py                     # Command line entry point
├── config_sample.json          # Sample configuration
├── USAGE.md                    # Detailed usage guide
└── README.md                   # This file
```

## Output Structure

For each PDF, the tool creates:

```
output/
├── PDF_NAME/
│   ├── tables/
│   │   ├── PDF_NAME_tables.xlsx      # All tables in Excel
│   │   ├── table_1.csv, table_2.csv  # Individual CSVs
│   │   └── PDF_NAME_tables.json      # Raw data + metadata
│   ├── text/
│   │   ├── PDF_NAME_text.txt         # Extracted text
│   │   └── PDF_NAME_text.json        # Text + metadata
│   ├── forms/
│   │   ├── PDF_NAME_forms.json       # Form fields data
│   │   └── PDF_NAME_forms.csv        # Form fields as CSV
│   ├── images/
│   │   └── page_X_img_Y.png          # Extracted images
│   └── reports/
│       ├── extraction_report.json    # Comprehensive report
│       └── extraction_report.txt     # Human-readable report
```

## Accuracy Features

### Multiple Extraction Methods
- **Tables**: Camelot, Tabula, pdfplumber, PyMuPDF
- **Text**: pdfplumber, PyMuPDF, PDFMiner, PyPDF2
- **Forms**: PyMuPDF, PyPDF2
- **Images**: PyMuPDF, pdfplumber

### Validation & Comparison
- Compares results between methods
- Calculates confidence scores
- Generates accuracy reports
- Identifies extraction differences

### Comprehensive Logging
```bash
# Enable detailed tracing
python main.py -i file.pdf -o output/ --log-level DEBUG
```

Logs include:
- Method-by-method extraction results
- Table detection confidence scores
- Processing time for each component
- Validation and comparison results
- Error details with context

## Real Estate PDF Examples

Your PDFs in `pdf/304-Cedar-Street/` are ready to process:

```bash
# Extract property details tables
python main.py -i "pdf/304-Cedar-Street/2. Property Details.pdf" -o output/

# Process MLS page
python main.py -i "pdf/304-Cedar-Street/1. Client MLS page.pdf" -o output/ --tables-only

# Batch process all documents
python main.py -i pdf/304-Cedar-Street/ -o output/
```
