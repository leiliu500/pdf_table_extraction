#!/usr/bin/env python3
"""
Quick demonstration of the PDF table extraction tool
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_table_extraction():
    """Demonstrate table extraction from the real estate PDFs"""
    print("PDF Table Extraction Tool - Quick Demo")
    print("=" * 50)
    
    # Check if PDFs exist
    pdf_dir = Path("pdf/304-Cedar-Street")
    if not pdf_dir.exists():
        print("❌ PDF directory not found: pdf/304-Cedar-Street/")
        print("Please ensure your PDF files are in the correct location.")
        return False
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in pdf/304-Cedar-Street/")
        return False
    
    print(f"✓ Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  • {pdf_file.name}")
    
    print("\nTo extract tables and content from your PDFs:")
    print("\n1. QUICK TABLE EXTRACTION (recommended to start):")
    print("   python main.py -i 'pdf/304-Cedar-Street/2. Property Details.pdf' -o output/ --tables-only")
    
    print("\n2. FULL EXTRACTION (all content types):")
    print("   python main.py -i 'pdf/304-Cedar-Street/2. Property Details.pdf' -o output/")
    
    print("\n3. BATCH PROCESS ALL FILES:")
    print("   python main.py -i pdf/304-Cedar-Street/ -o output/")
    
    print("\n4. DEBUG MODE (detailed logging):")
    print("   python main.py -i 'pdf/304-Cedar-Street/1. Client MLS page.pdf' -o output/ --log-level DEBUG")
    
    print("\n5. FAST MODE (no validation):")
    print("   python main.py -i pdf/304-Cedar-Street/ -o output/ --no-validation")
    
    print("\nOUTPUT STRUCTURE:")
    print("output/")
    print("├── PDF_NAME/")
    print("│   ├── tables/")
    print("│   │   ├── PDF_NAME_tables.xlsx    # All tables in Excel")
    print("│   │   ├── table_1.csv, table_2.csv # Individual tables")
    print("│   │   └── PDF_NAME_tables.json    # Raw data + metadata")
    print("│   ├── text/")
    print("│   │   └── PDF_NAME_text.txt       # Extracted text")
    print("│   ├── forms/")
    print("│   │   └── PDF_NAME_forms.json     # Form fields")
    print("│   ├── images/")
    print("│   │   └── page_X_img_Y.png        # Extracted images")
    print("│   └── reports/")
    print("│       └── extraction_report.json  # Comprehensive report")
    
    print("\nACCURACY FEATURES:")
    print("• Multiple extraction methods (Camelot, Tabula, pdfplumber, PyMuPDF)")
    print("• Automatic method comparison and validation")
    print("• Confidence scoring for each extraction")
    print("• Detailed logging for verification")
    
    print("\nNEXT STEPS:")
    print("1. Run: python test_installation.py  (to verify setup)")
    print("2. Start with: python main.py -i 'pdf/304-Cedar-Street/2. Property Details.pdf' -o output/ --tables-only")
    print("3. Check results in output/ directory")
    print("4. Review logs in logs/ directory")
    
    return True

if __name__ == "__main__":
    demo_table_extraction()
