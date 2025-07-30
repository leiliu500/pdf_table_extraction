#!/bin/bash

# Setup script for PDF Table Extraction Tool
# This script installs all required dependencies

echo "=========================================="
echo "PDF Table Extraction Tool Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install system dependencies for macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS. Checking for required system dependencies..."
    
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install ghostscript (required for Camelot)
    echo "Installing ghostscript..."
    brew install ghostscript
    
    # Install tkinter if needed
    echo "Installing python-tk..."
    brew install python-tk
    
    # Install other dependencies
    echo "Installing additional dependencies..."
    brew install poppler  # For better PDF processing
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."

# Check critical imports
python3 -c "
import sys
missing_packages = []

try:
    import pandas
    print('✓ pandas installed')
except ImportError:
    missing_packages.append('pandas')

try:
    import pdfplumber
    print('✓ pdfplumber installed')
except ImportError:
    missing_packages.append('pdfplumber')

try:
    import camelot
    print('✓ camelot-py installed')
except ImportError:
    missing_packages.append('camelot-py')

try:
    import tabula
    print('✓ tabula-py installed')
except ImportError:
    missing_packages.append('tabula-py')

try:
    import fitz
    print('✓ PyMuPDF installed')
except ImportError:
    missing_packages.append('PyMuPDF')

try:
    import loguru
    print('✓ loguru installed')
except ImportError:
    missing_packages.append('loguru')

if missing_packages:
    print('')
    print('❌ Missing packages:', ', '.join(missing_packages))
    print('Please run: pip install', ' '.join(missing_packages))
    sys.exit(1)
else:
    print('')
    print('✅ All critical packages installed successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "To use the tool:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run the extraction tool: python main.py -i pdf/304-Cedar-Street/ -o output/"
    echo ""
    echo "For help: python main.py --help"
    echo ""
    echo "Example commands:"
    echo "  # Extract all content from a single PDF"
    echo "  python main.py -i pdf/304-Cedar-Street/1.pdf -o output/"
    echo ""
    echo "  # Extract from all PDFs in a directory"
    echo "  python main.py -i pdf/304-Cedar-Street/ -o output/"
    echo ""
    echo "  # Extract only tables (faster)"
    echo "  python main.py -i pdf/304-Cedar-Street/1.pdf -o output/ --tables-only"
    echo ""
    echo "  # Enable debug logging"
    echo "  python main.py -i pdf/304-Cedar-Street/1.pdf -o output/ --log-level DEBUG"
else
    echo ""
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi
