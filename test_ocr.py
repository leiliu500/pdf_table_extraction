#!/usr/bin/env python3
"""
Test OCR functionality standalone
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import with proper module path
from src.image_extractor import OCRImageProcessor

def test_ocr():
    # Test individual imports first
    print("Testing OCR library imports...")
    
    try:
        import pytesseract
        print("✓ pytesseract imported successfully")
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract version: {version}")
        except Exception as e:
            print(f"✗ Tesseract not available: {e}")
    except ImportError as e:
        print(f"✗ pytesseract import failed: {e}")
    
    try:
        import easyocr
        print("✓ easyocr imported successfully")
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            print("✓ EasyOCR reader initialized successfully")
        except Exception as e:
            print(f"✗ EasyOCR initialization failed: {e}")
    except ImportError as e:
        print(f"✗ easyocr import failed: {e}")
    
    # Initialize OCR processor
    settings = {
        'enable_ocr': True,
        'tesseract_enabled': True,
        'easyocr_enabled': True,
        'ocr_min_confidence': 0.3
    }
    
    try:
        ocr_processor = OCRImageProcessor(settings)
        print("\nOCR Processor initialized successfully!")
        print(f"Tesseract enabled: {ocr_processor.tesseract_enabled}")
        print(f"EasyOCR enabled: {ocr_processor.easyocr_enabled}")
        
        # Test with a real image file
        image_path = Path("output/0. Coversheet/images/page_1_img_1.png")
        if image_path.exists():
            print(f"\nTesting OCR on: {image_path}")
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            text, confidence, method = ocr_processor.extract_text_from_image(image_data)
            print(f"OCR Text: '{text}'")
            print(f"Confidence: {confidence}")
            print(f"Method: {method}")
        else:
            print(f"Test image not found: {image_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr()
