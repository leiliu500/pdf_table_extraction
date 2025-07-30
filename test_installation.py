#!/usr/bin/env python3
"""
Test script to verify PDF extraction installation and basic functionality
"""
import sys
import os
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test core modules
        from src.pdf_extractor import PDFExtractor
        from src.table_extractor import TableExtractor
        from src.text_extractor import TextExtractor
        from src.form_extractor import FormExtractor
        from src.image_extractor import ImageExtractor
        print("✓ Core modules imported successfully")
        
        # Test utility modules
        from src.utils.logger import get_logger
        from src.utils.file_utils import get_file_handler
        from src.utils.validation import get_validator
        print("✓ Utility modules imported successfully")
        
        # Test configuration
        from src.config.settings import TABLE_EXTRACTION_METHODS
        print("✓ Configuration modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are available"""
    print("\nTesting dependencies...")
    
    missing_deps = []
    
    # Test pandas
    try:
        import pandas as pd
        print("✓ pandas available")
    except ImportError:
        missing_deps.append("pandas")
    
    # Test logging
    try:
        from loguru import logger
        print("✓ loguru available")
    except ImportError:
        missing_deps.append("loguru")
    
    # Test PDF libraries (these may not all be available)
    pdf_libs = {
        'pdfplumber': 'pdfplumber',
        'camelot': 'camelot-py',
        'tabula': 'tabula-py', 
        'PyMuPDF': 'fitz',
        'PyPDF2': 'PyPDF2',
        'pdfminer': 'pdfminer.six'
    }
    
    available_pdf_libs = []
    for lib_name, import_name in pdf_libs.items():
        try:
            if import_name == 'fitz':
                import fitz
            elif import_name == 'pdfminer.six':
                from pdfminer.high_level import extract_text
            else:
                __import__(import_name)
            available_pdf_libs.append(lib_name)
            print(f"✓ {lib_name} available")
        except ImportError:
            print(f"⚠ {lib_name} not available")
    
    if len(available_pdf_libs) == 0:
        missing_deps.append("At least one PDF library (pdfplumber, camelot, tabula, PyMuPDF)")
    
    # Test image processing
    try:
        from PIL import Image
        print("✓ PIL/Pillow available")
    except ImportError:
        print("⚠ PIL/Pillow not available (image processing will be limited)")
    
    if missing_deps:
        print(f"\n❌ Missing critical dependencies: {', '.join(missing_deps)}")
        return False
    else:
        print(f"\n✅ Found {len(available_pdf_libs)} PDF processing libraries")
        return True


def test_basic_functionality():
    """Test basic functionality without requiring PDF files"""
    print("\nTesting basic functionality...")
    
    try:
        # Test logger
        from src.utils.logger import get_logger
        logger = get_logger("test")
        logger.info("Test log message")
        print("✓ Logger working")
        
        # Test file handler
        from src.utils.file_utils import get_file_handler
        with tempfile.TemporaryDirectory() as temp_dir:
            file_handler = get_file_handler(temp_dir)
            test_data = {"test": "data"}
            test_file = Path(temp_dir) / "test.json"
            file_handler.save_json(test_data, test_file)
            loaded_data = file_handler.load_json(test_file)
            assert loaded_data == test_data
        print("✓ File handler working")
        
        # Test extractor initialization
        from src.pdf_extractor import PDFExtractor
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = PDFExtractor(
                input_dir=temp_dir,
                output_dir=temp_dir,
                log_level="INFO"
            )
        print("✓ PDF extractor initialization working")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def test_with_sample_pdf():
    """Test with actual PDF files if available"""
    print("\nTesting with sample PDFs...")
    
    # Look for PDF files
    pdf_dir = Path("pdf/304-Cedar-Street")
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if pdf_files:
            print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
            
            # Test with first PDF file
            test_pdf = pdf_files[0]
            print(f"Testing with: {test_pdf.name}")
            
            try:
                from src.pdf_extractor import PDFExtractor
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    extractor = PDFExtractor(
                        input_dir=pdf_dir,
                        output_dir=temp_dir,
                        log_level="WARNING"  # Reduce noise for testing
                    )
                    
                    # Test table extraction only (faster)
                    result = extractor.extract_tables_only(test_pdf)
                    
                    tables_found = result['summary']['total_tables_found']
                    methods_used = result['summary']['methods_used']
                    
                    print(f"✓ Table extraction completed")
                    print(f"  Tables found: {tables_found}")
                    print(f"  Methods used: {', '.join(methods_used)}")
                    
                    if tables_found > 0:
                        print(f"✓ Successfully extracted tables from {test_pdf.name}")
                    else:
                        print(f"⚠ No tables found in {test_pdf.name} (this may be normal)")
                
                return True
                
            except Exception as e:
                print(f"❌ PDF processing failed: {e}")
                return False
        else:
            print("⚠ No PDF files found for testing")
            return True
    else:
        print("⚠ No PDF directory found for testing")
        return True


def main():
    """Run all tests"""
    print("="*60)
    print("PDF Table Extraction Tool - Installation Test")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Dependencies Test", test_dependencies),
        ("Basic Functionality Test", test_basic_functionality),
        ("Sample PDF Test", test_with_sample_pdf)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✅ All tests passed! The PDF extraction tool is ready to use.")
        print("\nTry running:")
        print("  python main.py -i pdf/304-Cedar-Street/ -o output/ --tables-only")
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed. Please check the installation.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
