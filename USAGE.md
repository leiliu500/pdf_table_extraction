# PDF Table Extraction Tool - Usage Guide

This comprehensive guide explains how to use the PDF table extraction tool for accurate extraction of tables, text, forms, and images from PDF files.

## Quick Start

1. **Setup the environment:**
   ```bash
   # Make setup script executable and run it
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Test the installation:**
   ```bash
   python test_installation.py
   ```

4. **Extract tables from your PDFs:**
   ```bash
   # Extract from a single PDF
   python main.py -i pdf/304-Cedar-Street/1.pdf -o output/

   # Extract from all PDFs in directory
   python main.py -i pdf/304-Cedar-Street/ -o output/
   ```

## Command Line Usage

### Basic Commands

```bash
# Extract all content from a single PDF
python main.py -i path/to/file.pdf -o output_directory/

# Extract from all PDFs in a directory
python main.py -i path/to/pdf_directory/ -o output_directory/

# Extract only tables (faster processing)
python main.py -i path/to/file.pdf -o output/ --tables-only

# Use custom file pattern
python main.py -i pdf_directory/ -o output/ --pattern "*.PDF"
```

### Advanced Options

```bash
# Enable debug logging for detailed tracing
python main.py -i file.pdf -o output/ --log-level DEBUG

# Disable validation for faster processing
python main.py -i file.pdf -o output/ --no-validation

# Use custom configuration
python main.py -i file.pdf -o output/ --config config_sample.json
```

### Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `-i, --input` | Input PDF file or directory | `-i pdf/file.pdf` |
| `-o, --output` | Output directory | `-o results/` |
| `--tables-only` | Extract only tables | `--tables-only` |
| `--log-level` | Set logging level | `--log-level DEBUG` |
| `--no-validation` | Disable accuracy validation | `--no-validation` |
| `--pattern` | File pattern for directory processing | `--pattern "*.PDF"` |
| `--config` | Custom configuration file | `--config my_config.json` |

## Python API Usage

### Basic Usage

```python
from src.pdf_extractor import PDFExtractor

# Initialize extractor
extractor = PDFExtractor(
    input_dir="pdf/304-Cedar-Street",
    output_dir="output",
    log_level="INFO"
)

# Extract from single PDF
result = extractor.extract_pdf("pdf/304-Cedar-Street/1.pdf")

# Extract from all PDFs
results = extractor.extract_all()

# Extract only tables
table_results = extractor.extract_tables_only("pdf/304-Cedar-Street/1.pdf")
```

### Advanced Configuration

```python
custom_settings = {
    'tables': {
        'camelot': {'enabled': True, 'flavors': ['lattice']},
        'tabula': {'enabled': True, 'lattice': True},
        'pdfplumber': {'enabled': False},  # Disable pdfplumber
        'pymupdf': {'enabled': True}
    },
    'validation': {
        'confidence_threshold': 0.9,  # Higher accuracy requirement
        'enable_comparison': True
    }
}

extractor = PDFExtractor(
    input_dir="pdf/304-Cedar-Street",
    output_dir="output",
    custom_settings=custom_settings
)
```

### Individual Extractors

```python
from src.table_extractor import TableExtractor
from src.text_extractor import TextExtractor

# Use individual extractors
table_extractor = TableExtractor()
table_results = table_extractor.extract_all_tables("file.pdf")

text_extractor = TextExtractor()
text_results = text_extractor.extract_all_text("file.pdf")
```

## Output Structure

The tool creates an organized output structure for each PDF:

```
output/
├── PDF_NAME/
│   ├── tables/
│   │   ├── PDF_NAME_tables.xlsx      # All tables in Excel
│   │   ├── table_1.csv               # Individual table CSVs
│   │   ├── table_2.csv
│   │   └── PDF_NAME_tables.json      # Table data + metadata
│   ├── text/
│   │   ├── PDF_NAME_text.txt         # Extracted text
│   │   └── PDF_NAME_text.json        # Text + metadata
│   ├── forms/
│   │   ├── PDF_NAME_forms.json       # Form fields data
│   │   └── PDF_NAME_forms.csv        # Form fields as CSV
│   ├── images/
│   │   ├── page_1_img_1.png          # Extracted images
│   │   └── page_2_img_1.png
│   └── reports/
│       ├── extraction_report.json    # Comprehensive report
│       └── extraction_report.txt     # Human-readable report
```

## Extraction Methods

### Table Extraction

The tool uses multiple methods and compares results for maximum accuracy:

1. **Camelot** - Best for tables with clear borders
   - Lattice method: Tables with visible lines
   - Stream method: Tables without visible lines

2. **Tabula** - Java-based extraction
   - Lattice mode: Line-based tables
   - Stream mode: Space-separated tables

3. **pdfplumber** - Python-native extraction
   - Good for complex layouts
   - Preserves table structure

4. **PyMuPDF** - Fast C-based extraction
   - Good for standard PDF tables
   - Handles rotated tables

### Text Extraction

Multiple text extraction approaches:

1. **pdfplumber** - Layout-preserving extraction
2. **PyMuPDF** - Fast extraction with annotations
3. **PDFMiner** - Advanced layout analysis
4. **PyPDF2** - Simple text extraction

### Form Extraction

Extracts form fields and annotations:

1. **PyMuPDF** - Widget and annotation extraction
2. **PyPDF2** - Basic form field extraction

### Image Extraction

Extracts embedded images:

1. **PyMuPDF** - Full image data extraction
2. **pdfplumber** - Image metadata detection

## Accuracy Features

### Validation and Comparison

The tool includes several accuracy features:

1. **Multi-method comparison** - Compares results from different extraction methods
2. **Confidence scoring** - Calculates confidence scores for each extraction
3. **Validation reports** - Detailed accuracy analysis
4. **Diff generation** - Shows differences between methods

### Logging and Tracing

Comprehensive logging for accuracy verification:

```bash
# Enable debug logging to see detailed extraction steps
python main.py -i file.pdf -o output/ --log-level DEBUG
```

Log messages include:
- Extraction method performance
- Table detection confidence
- Processing time for each method
- Validation results
- Error details

## Configuration

### Custom Configuration File

Create a JSON configuration file to customize extraction behavior:

```json
{
  "tables": {
    "camelot": {
      "enabled": true,
      "flavors": ["lattice", "stream"],
      "edge_tol": 500
    },
    "tabula": {
      "enabled": true,
      "lattice": true,
      "stream": false
    }
  },
  "validation": {
    "confidence_threshold": 0.8,
    "enable_comparison": true
  }
}
```

Use with: `python main.py -i file.pdf -o output/ --config my_config.json`

### Environment Variables

Set environment variables for default behavior:

```bash
export PDF_EXTRACTOR_LOG_LEVEL=DEBUG
export PDF_EXTRACTOR_OUTPUT_DIR=./output
```

## Real Estate PDF Examples

For your real estate PDFs in `pdf/304-Cedar-Street/`:

### Extract Property Details Tables

```bash
# Extract tables from property details with high accuracy
python main.py -i pdf/304-Cedar-Street/2.\ Property\ Details.pdf -o output/ --log-level DEBUG
```

### Batch Process All Documents

```bash
# Process all real estate documents
python main.py -i pdf/304-Cedar-Street/ -o output/
```

### Extract MLS Data

```bash
# Focus on MLS page tables
python main.py -i pdf/304-Cedar-Street/1.\ Client\ MLS\ page.pdf -o output/ --tables-only
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

2. **Camelot Installation Issues**
   ```bash
   # Install system dependencies
   brew install ghostscript  # macOS
   sudo apt-get install ghostscript  # Ubuntu
   ```

3. **Java Not Found (Tabula)**
   ```bash
   # Install Java
   brew install openjdk  # macOS
   ```

4. **Low Table Detection**
   - Try different extraction methods in config
   - Enable debug logging to see detection details
   - Check if PDF has image-based tables (requires OCR)

### Performance Optimization

```bash
# For faster processing, disable validation
python main.py -i file.pdf -o output/ --no-validation

# Extract only tables
python main.py -i file.pdf -o output/ --tables-only

# Use only fast methods in config
{
  "tables": {
    "camelot": {"enabled": false},
    "tabula": {"enabled": false},
    "pdfplumber": {"enabled": true},
    "pymupdf": {"enabled": true}
  }
}
```

## Best Practices

1. **Start with table-only extraction** to quickly assess what can be extracted
2. **Use debug logging** to understand extraction behavior
3. **Compare multiple methods** for critical documents
4. **Validate results** against original PDFs
5. **Customize configuration** based on your PDF types
6. **Process in batches** for large document sets

## Support

- Check logs in `logs/` directory for detailed error information
- Run `python test_installation.py` to verify setup
- Use `--log-level DEBUG` for maximum detail
- Compare extraction results between methods in the validation reports
