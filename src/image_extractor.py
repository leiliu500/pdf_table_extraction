"""
Image extraction module for PDF files with OCR capabilities
"""
import time
import io
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# OCR libraries
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np

from .config.settings import IMAGE_EXTRACTION_SETTINGS
from .utils.logger import get_logger

logger = get_logger("image_extractor")


@dataclass
class ExtractedImage:
    """Represents an extracted image from a PDF"""
    filename: str
    format: str
    width: int
    height: int
    page_number: int
    image_data: bytes
    bbox: Optional[List[float]] = None
    size_bytes: int = 0
    dpi: Optional[Tuple[int, int]] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    ocr_method: Optional[str] = None


@dataclass
class ImageExtractionResult:
    """Result of image extraction from a single method"""
    method: str
    images: List[ExtractedImage]
    metadata: Dict[str, Any]
    processing_time: float
    extraction_successful: bool
    error_message: Optional[str] = None


class PyMuPDFImageExtractor:
    """Image extraction using PyMuPDF"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PYMUPDF_AVAILABLE and settings.get('extract_images', True)
        
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available for image extraction")
    
    def extract_images(self, pdf_path: str, output_dir: Path) -> ImageExtractionResult:
        """Extract images using PyMuPDF"""
        if not self.enabled:
            return ImageExtractionResult(
                method="pymupdf",
                images=[],
                metadata={},
                processing_time=0.0,
                extraction_successful=False,
                error_message="PyMuPDF not available or image extraction disabled"
            )
        
        start_time = time.time()
        logger.info(f"Starting PyMuPDF image extraction", context={'pdf': pdf_path})
        
        try:
            doc = fitz.open(pdf_path)
            extracted_images = []
            metadata = {'pages_with_images': [], 'total_images': 0, 'formats': {}}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                if image_list:
                    page_images = []
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image data
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            # Check image dimensions
                            if (pix.width < self.settings.get('min_width', 50) or 
                                pix.height < self.settings.get('min_height', 50)):
                                pix = None
                                continue
                            
                            # Convert to appropriate format
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                img_format = "PNG"
                                img_data = pix.tobytes("png")
                            else:  # CMYK
                                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                                img_format = "PNG"
                                img_data = pix1.tobytes("png")
                                pix1 = None
                            
                            # Create filename
                            filename = f"page_{page_num + 1}_img_{img_index + 1}.{img_format.lower()}"
                            
                            # Save image file
                            image_path = output_dir / filename
                            with open(image_path, 'wb') as f:
                                f.write(img_data)
                            
                            # Create ExtractedImage object
                            extracted_img = ExtractedImage(
                                filename=filename,
                                format=img_format,
                                width=pix.width,
                                height=pix.height,
                                page_number=page_num + 1,
                                image_data=img_data,
                                size_bytes=len(img_data),
                                dpi=pix.xres if hasattr(pix, 'xres') else None
                            )
                            
                            extracted_images.append(extracted_img)
                            page_images.append(filename)
                            
                            # Update metadata
                            metadata['formats'][img_format] = metadata['formats'].get(img_format, 0) + 1
                            
                            logger.debug(f"Extracted image {filename} from page {page_num + 1}")
                            
                            pix = None
                        
                        except Exception as e:
                            logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}", 
                                         exception=e)
                    
                    if page_images:
                        metadata['pages_with_images'].append({
                            'page': page_num + 1,
                            'image_count': len(page_images),
                            'images': page_images
                        })
            
            doc.close()
            
            metadata['total_images'] = len(extracted_images)
            processing_time = time.time() - start_time
            
            logger.info(f"PyMuPDF image extraction completed", 
                       context={
                           'total_images': len(extracted_images),
                           'pages_with_images': len(metadata['pages_with_images']),
                           'processing_time': processing_time,
                           'formats': list(metadata['formats'].keys())
                       })
            
            return ImageExtractionResult(
                method="pymupdf",
                images=extracted_images,
                metadata=metadata,
                processing_time=processing_time,
                extraction_successful=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PyMuPDF image extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return ImageExtractionResult(
                method="pymupdf",
                images=[],
                metadata={},
                processing_time=processing_time,
                extraction_successful=False,
                error_message=str(e)
            )


class PDFPlumberImageExtractor:
    """Image extraction using pdfplumber"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PDFPLUMBER_AVAILABLE and settings.get('extract_images', True)
        
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available for image extraction")
    
    def extract_images(self, pdf_path: str, output_dir: Path) -> ImageExtractionResult:
        """Extract images using pdfplumber"""
        if not self.enabled:
            return ImageExtractionResult(
                method="pdfplumber",
                images=[],
                metadata={},
                processing_time=0.0,
                extraction_successful=False,
                error_message="pdfplumber not available or image extraction disabled"
            )
        
        start_time = time.time()
        logger.info(f"Starting pdfplumber image extraction", context={'pdf': pdf_path})
        
        try:
            extracted_images = []
            metadata = {'pages_with_images': [], 'total_images': 0}
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Get images on this page
                    if hasattr(page, 'images'):
                        images = page.images
                        page_images = []
                        
                        for img_index, img in enumerate(images):
                            try:
                                # Extract image properties
                                bbox = [img.get('x0', 0), img.get('y0', 0), 
                                       img.get('x1', 0), img.get('y1', 0)]
                                width = int(img.get('width', 0))
                                height = int(img.get('height', 0))
                                
                                # Check minimum size requirements
                                if (width < self.settings.get('min_width', 50) or 
                                    height < self.settings.get('min_height', 50)):
                                    continue
                                
                                # Create filename
                                filename = f"page_{page_num}_img_{img_index + 1}_pdfplumber.png"
                                
                                # Note: pdfplumber doesn't directly provide image data
                                # We can only get metadata about images
                                extracted_img = ExtractedImage(
                                    filename=filename,
                                    format="PNG",
                                    width=width,
                                    height=height,
                                    page_number=page_num,
                                    image_data=b'',  # No actual image data available
                                    bbox=bbox,
                                    size_bytes=0
                                )
                                
                                extracted_images.append(extracted_img)
                                page_images.append(filename)
                                
                                logger.debug(f"Identified image {filename} on page {page_num}")
                            
                            except Exception as e:
                                logger.warning(f"Failed to process image {img_index} from page {page_num}", 
                                             exception=e)
                        
                        if page_images:
                            metadata['pages_with_images'].append({
                                'page': page_num,
                                'image_count': len(page_images),
                                'images': page_images
                            })
            
            metadata['total_images'] = len(extracted_images)
            processing_time = time.time() - start_time
            
            logger.info(f"pdfplumber image extraction completed", 
                       context={
                           'total_images': len(extracted_images),
                           'pages_with_images': len(metadata['pages_with_images']),
                           'processing_time': processing_time
                       })
            
            # Note: This method only identifies images, doesn't extract actual data
            logger.warning("pdfplumber can only identify images, not extract image data")
            
            return ImageExtractionResult(
                method="pdfplumber",
                images=extracted_images,
                metadata=metadata,
                processing_time=processing_time,
                extraction_successful=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"pdfplumber image extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return ImageExtractionResult(
                method="pdfplumber",
                images=[],
                metadata={},
                processing_time=processing_time,
                extraction_successful=False,
                error_message=str(e)
            )


class OCRImageProcessor:
    """OCR processing for extracted images"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.tesseract_enabled = PYTESSERACT_AVAILABLE and settings.get('tesseract_enabled', True)
        self.easyocr_enabled = EASYOCR_AVAILABLE and settings.get('easyocr_enabled', True)
        self.cv2_available = CV2_AVAILABLE
        self.easyocr_reader = None
        
        # Initialize EasyOCR reader if available
        if self.easyocr_enabled and EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR reader initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {str(e)}")
                self.easyocr_enabled = False
        elif not EASYOCR_AVAILABLE and settings.get('easyocr_enabled', True):
            logger.info("EasyOCR not available (requires PyTorch), using Tesseract only")
            self.easyocr_enabled = False
        
        # Check Tesseract availability
        if self.tesseract_enabled and PYTESSERACT_AVAILABLE:
            try:
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR available")
            except Exception as e:
                logger.warning(f"Tesseract not available: {str(e)}")
                self.tesseract_enabled = False
        elif not PYTESSERACT_AVAILABLE:
            logger.warning("pytesseract not available")
            self.tesseract_enabled = False
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # If OpenCV is available, use advanced preprocessing
            if self.cv2_available and CV2_AVAILABLE:
                # Convert to numpy array for OpenCV processing
                image_array = np.array(pil_image)
                
                # Convert RGB to BGR for OpenCV
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Convert to grayscale for better OCR
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                
                # Apply image enhancement techniques
                # 1. Noise reduction
                denoised = cv2.fastNlMeansDenoising(gray)
                
                # 2. Contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
                
                # 3. Adaptive thresholding for better text extraction
                thresh = cv2.adaptiveThreshold(
                    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                
                return thresh
            else:
                # Fallback to basic PIL processing if OpenCV not available
                logger.debug("Using basic PIL preprocessing (OpenCV not available)")
                grayscale = pil_image.convert('L')
                return np.array(grayscale)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            # Fallback to original image
            try:
                pil_image = Image.open(io.BytesIO(image_data))
                return np.array(pil_image.convert('L'))  # Convert to grayscale
            except:
                return None
    
    def extract_text_tesseract(self, image_data: bytes) -> Tuple[Optional[str], Optional[float]]:
        """Extract text using Tesseract OCR"""
        if not self.tesseract_enabled:
            return None, None
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            if processed_image is None:
                return None, None
            
            # Configure Tesseract for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?@#$%^&*()_+-=[]{}|;:,.<>?/~`'
            
            # Extract text
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Get confidence scores
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Clean and validate text
            cleaned_text = self._clean_ocr_text(text)
            
            return cleaned_text, avg_confidence / 100.0  # Convert to 0-1 range
            
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {str(e)}")
            return None, None
    
    def extract_text_easyocr(self, image_data: bytes) -> Tuple[Optional[str], Optional[float]]:
        """Extract text using EasyOCR"""
        if not self.easyocr_enabled or self.easyocr_reader is None or not EASYOCR_AVAILABLE:
            return None, None
        
        try:
            # Convert image data to numpy array
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            
            # Use OpenCV if available, otherwise use PIL
            if self.cv2_available and CV2_AVAILABLE:
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                # Fallback to PIL
                pil_image = Image.open(io.BytesIO(image_data))
                image = np.array(pil_image)
            
            if image is None:
                return None, None
            
            # Extract text with EasyOCR
            results = self.easyocr_reader.readtext(image)
            
            # Combine text and calculate average confidence
            extracted_texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.1:  # Filter out very low confidence results
                    extracted_texts.append(text)
                    confidences.append(confidence)
            
            if not extracted_texts:
                return None, None
            
            # Combine texts and calculate average confidence
            combined_text = ' '.join(extracted_texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            cleaned_text = self._clean_ocr_text(combined_text)
            
            return cleaned_text, avg_confidence
            
        except Exception as e:
            logger.warning(f"EasyOCR failed: {str(e)}")
            return None, None
    
    def extract_text_from_image(self, image_data: bytes) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """Extract text from image using the best available OCR method"""
        tesseract_text, tesseract_conf = None, None
        easyocr_text, easyocr_conf = None, None
        
        # Try both methods
        if self.tesseract_enabled:
            tesseract_text, tesseract_conf = self.extract_text_tesseract(image_data)
        
        if self.easyocr_enabled:
            easyocr_text, easyocr_conf = self.extract_text_easyocr(image_data)
        
        # Choose the best result based on confidence and text length
        best_text, best_conf, best_method = None, 0, None
        
        if tesseract_text and tesseract_conf:
            if len(tesseract_text.strip()) > 5 and tesseract_conf > 0.3:
                best_text, best_conf, best_method = tesseract_text, tesseract_conf, "tesseract"
        
        if easyocr_text and easyocr_conf:
            # Prefer EasyOCR if it has higher confidence or much longer text
            if (easyocr_conf > best_conf + 0.1) or (len(easyocr_text.strip()) > len(best_text or '') * 1.5):
                best_text, best_conf, best_method = easyocr_text, easyocr_conf, "easyocr"
        
        # Fallback: use any available result if no high-confidence result found
        if not best_text:
            if tesseract_text and len(tesseract_text.strip()) > 2:
                best_text, best_conf, best_method = tesseract_text, tesseract_conf or 0, "tesseract"
            elif easyocr_text and len(easyocr_text.strip()) > 2:
                best_text, best_conf, best_method = easyocr_text, easyocr_conf or 0, "easyocr"
        
        return best_text, best_conf, best_method
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I')  # Common misread
        text = text.replace('0', 'O') if text.isalpha() else text  # Context-dependent replacement
        
        # Remove very short "words" that are likely artifacts
        words = text.split()
        filtered_words = [word for word in words if len(word) > 1 or word.isalnum()]
        
        return ' '.join(filtered_words).strip()


class ImageExtractor:
    """Main image extraction class that orchestrates multiple extraction methods with OCR capabilities"""
    
    def __init__(self, settings: Dict[str, Any] = None):
        self.settings = settings or IMAGE_EXTRACTION_SETTINGS
        
        # Initialize extractors
        self.extractors = {}
        
        if PYMUPDF_AVAILABLE and self.settings.get('extract_images', True):
            self.extractors['pymupdf'] = PyMuPDFImageExtractor(self.settings)
        
        if PDFPLUMBER_AVAILABLE and self.settings.get('extract_images', True):
            self.extractors['pdfplumber'] = PDFPlumberImageExtractor(self.settings)
        
        # Initialize OCR processor
        self.ocr_processor = None
        if self.settings.get('enable_ocr', True):
            try:
                self.ocr_processor = OCRImageProcessor(self.settings)
                logger.info("OCR processor initialized for image text extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize OCR processor: {str(e)}")
        
        logger.info(f"Initialized image extractor", 
                   context={'enabled_methods': list(self.extractors.keys()),
                           'ocr_enabled': self.ocr_processor is not None})
    
    def extract_all_images(self, pdf_path: str, output_dir: Path) -> Dict[str, Any]:
        """Extract images using all available methods"""
        logger.trace_pdf_processing(pdf_path, "image_extraction_start")
        
        start_time = time.time()
        results = {}
        extraction_results = {}
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run all extraction methods
        for method_name, extractor in self.extractors.items():
            logger.info(f"Running {method_name} image extraction")
            
            try:
                result = extractor.extract_images(pdf_path, output_dir)
                extraction_results[method_name] = result
                results[method_name] = result.images
                
                logger.info(f"{method_name} image extraction completed", 
                           context={
                               'images_found': len(result.images),
                               'successful': result.extraction_successful,
                               'processing_time': result.processing_time
                           })
                
            except Exception as e:
                logger.error(f"{method_name} image extraction failed", exception=e)
                extraction_results[method_name] = ImageExtractionResult(
                    method=method_name,
                    images=[],
                    metadata={},
                    processing_time=0.0,
                    extraction_successful=False,
                    error_message=str(e)
                )
                results[method_name] = []
        
        # Determine best method
        best_method = self._determine_best_method(extraction_results)
        
        # Combine unique images
        combined_images = self._combine_images(results)
        
        # Add OCR processing to extracted images
        if self.ocr_processor and combined_images:
            logger.info("Starting OCR processing for extracted images")
            ocr_start_time = time.time()
            ocr_processed_count = 0
            
            for image in combined_images:
                if image.image_data:  # Only process images with actual data
                    try:
                        ocr_text, ocr_confidence, ocr_method = self.ocr_processor.extract_text_from_image(image.image_data)
                        if ocr_text and len(ocr_text.strip()) > 3:  # Only keep meaningful text
                            image.ocr_text = ocr_text
                            image.ocr_confidence = ocr_confidence
                            image.ocr_method = ocr_method
                            ocr_processed_count += 1
                            logger.debug(f"OCR extracted from {image.filename}: {len(ocr_text)} chars, confidence: {ocr_confidence:.3f}")
                        else:
                            logger.debug(f"No meaningful text found in {image.filename}")
                    except Exception as e:
                        logger.warning(f"OCR failed for {image.filename}: {str(e)}")
            
            ocr_time = time.time() - ocr_start_time
            logger.info(f"OCR processing completed", 
                       context={'images_processed': ocr_processed_count, 
                               'total_images': len(combined_images),
                               'ocr_time': ocr_time})
        
        total_time = time.time() - start_time
        
        final_results = {
            'images_by_method': results,
            'extraction_results': extraction_results,
            'combined_images': combined_images,
            'best_method': best_method,
            'total_processing_time': total_time,
            'summary': {
                'total_images_combined': len(combined_images),
                'successful_methods': len([r for r in extraction_results.values() if r.extraction_successful]),
                'methods_used': list(self.extractors.keys()),
                'output_directory': str(output_dir)
            }
        }
        
        logger.trace_pdf_processing(
            pdf_path, "image_extraction_complete", 
            {'total_images': len(combined_images),
             'best_method': best_method}
        )
        
        return final_results
    
    def _combine_images(self, results: Dict[str, List[ExtractedImage]]) -> List[ExtractedImage]:
        """Combine images from multiple methods, removing duplicates"""
        combined = []
        seen_images = set()
        
        # Prioritize PyMuPDF as it extracts actual image data
        method_priority = ['pymupdf', 'pdfplumber']
        
        for method in method_priority:
            if method in results:
                for image in results[method]:
                    # Use page number and approximate size to identify duplicates
                    image_key = (image.page_number, image.width, image.height)
                    if image_key not in seen_images:
                        combined.append(image)
                        seen_images.add(image_key)
        
        # Add any remaining images from other methods
        for method, images in results.items():
            if method not in method_priority:
                for image in images:
                    image_key = (image.page_number, image.width, image.height)
                    if image_key not in seen_images:
                        combined.append(image)
                        seen_images.add(image_key)
        
        logger.info(f"Combined images from multiple methods", 
                   context={'total_combined': len(combined), 'methods': list(results.keys())})
        
        return combined
    
    def _determine_best_method(self, extraction_results: Dict[str, ImageExtractionResult]) -> str:
        """Determine the best extraction method based on results"""
        method_scores = {}
        
        for method, result in extraction_results.items():
            if not result.extraction_successful:
                method_scores[method] = 0.0
                continue
            
            # Scoring factors
            image_count_score = min(len(result.images) / 10, 1.0)
            
            # Bonus for methods that extract actual image data
            data_quality_score = 1.0 if any(img.size_bytes > 0 for img in result.images) else 0.5
            
            speed_score = max(0, 1.0 - (result.processing_time / 10))
            
            # Weighted average
            total_score = (
                image_count_score * 0.4 +
                data_quality_score * 0.5 +
                speed_score * 0.1
            )
            
            method_scores[method] = total_score
        
        if not method_scores:
            return "none"
        
        best_method = max(method_scores, key=method_scores.get)
        
        logger.info(f"Best image extraction method determined", 
                   context={'method': best_method, 'scores': method_scores})
        
        return best_method
    
    def get_best_images(self, extraction_results: Dict[str, Any]) -> List[ExtractedImage]:
        """Get the best images from extraction results"""
        # Return combined images if available (usually best)
        if 'combined_images' in extraction_results:
            return extraction_results['combined_images']
        
        best_method = extraction_results.get('best_method', 'none')
        
        if best_method == 'none' or best_method not in extraction_results['images_by_method']:
            # Fallback: return images from the method with most images
            images_by_method = extraction_results['images_by_method']
            if not images_by_method:
                return []
            
            best_method = max(images_by_method, key=lambda k: len(images_by_method[k]))
        
        return extraction_results['images_by_method'].get(best_method, [])
